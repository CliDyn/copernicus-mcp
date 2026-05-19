from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def coordinator(tmp_path: Path):
    from copernicus_mcp.data_model.coordinator import DataModelCoordinator
    from copernicus_mcp.persistence import SqliteBackend

    backend = SqliteBackend(tmp_path / "state.db")
    await backend.initialise()
    try:
        yield DataModelCoordinator(persistence=backend)
    finally:
        await backend.close()


def _subset_kwargs() -> dict:
    return dict(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        dataset_version="202411",
        variables=["thetao"],
        minimum_longitude=-10.0,
        maximum_longitude=5.0,
        minimum_latitude=35.0,
        maximum_latitude=45.0,
        minimum_depth=0.0,
        maximum_depth=100.0,
        start_datetime="2024-01-01T00:00:00Z",
        end_datetime="2024-01-02T00:00:00Z",
    )


@pytest.mark.asyncio
async def test_subset_key_deterministic(coordinator) -> None:
    from copernicus_mcp.data_model.schemas_cmems import CmemsSubsetRequest

    req = CmemsSubsetRequest(**_subset_kwargs())
    a = coordinator.cache_key_for_subset(req)
    b = coordinator.cache_key_for_subset(req)
    assert a == b
    assert a.startswith("cmems:submit:cmems_mod_glo_phy_anfc_0.083deg_P1D-m:")


@pytest.mark.asyncio
async def test_subset_key_independent_of_variable_order(coordinator) -> None:
    from copernicus_mcp.data_model.schemas_cmems import CmemsSubsetRequest

    kw = _subset_kwargs()
    kw["variables"] = ["thetao", "so"]
    a = coordinator.cache_key_for_subset(CmemsSubsetRequest(**kw))
    kw["variables"] = ["so", "thetao"]
    b = coordinator.cache_key_for_subset(CmemsSubsetRequest(**kw))
    assert a == b


@pytest.mark.asyncio
async def test_subset_key_changes_with_dataset_version(coordinator) -> None:
    from copernicus_mcp.data_model.schemas_cmems import CmemsSubsetRequest

    kw = _subset_kwargs()
    a = coordinator.cache_key_for_subset(CmemsSubsetRequest(**kw))
    kw["dataset_version"] = "202412"
    b = coordinator.cache_key_for_subset(CmemsSubsetRequest(**kw))
    assert a != b


@pytest.mark.asyncio
async def test_search_key_basic(coordinator) -> None:
    from copernicus_mcp.data_model.schemas_cmems import CmemsSearchRequest

    req = CmemsSearchRequest(keyword="temperature", limit=5)
    key = coordinator.cache_key_for_search(req)
    assert key.startswith("cmems:search:")
    # search has no dataset_id — implementation must use a sentinel
    same = coordinator.cache_key_for_search(
        CmemsSearchRequest(keyword="temperature", limit=5)
    )
    assert key == same


# ---------------------------------------------------------------------------
# T-CMEMS-GET-002: cache_key_for_get
# ---------------------------------------------------------------------------


def _get_kwargs() -> dict:
    return dict(
        dataset_id="cmems_obs-ins_glo_phy-temp-sal_my_easycora_irr",
    )


@pytest.mark.asyncio
async def test_get_key_deterministic(coordinator) -> None:
    """Same request → same cache key. Prefix uses ``cmems:get:<dataset_id>:``
    so a future ``copernicus://files/`` resource resolver can route on
    operation."""
    from copernicus_mcp.data_model.schemas_cmems import CmemsGetRequest

    req = CmemsGetRequest(**_get_kwargs())
    a = coordinator.cache_key_for_get(req)
    b = coordinator.cache_key_for_get(req)
    assert a == b
    assert a.startswith(
        "cmems:get:cmems_obs-ins_glo_phy-temp-sal_my_easycora_irr:"
    )


@pytest.mark.asyncio
async def test_get_key_includes_dataset_version(coordinator) -> None:
    from copernicus_mcp.data_model.schemas_cmems import CmemsGetRequest

    a = coordinator.cache_key_for_get(CmemsGetRequest(**_get_kwargs()))
    b = coordinator.cache_key_for_get(
        CmemsGetRequest(**_get_kwargs(), dataset_version="202411")
    )
    assert a != b


@pytest.mark.asyncio
async def test_get_key_includes_filter(coordinator) -> None:
    from copernicus_mcp.data_model.schemas_cmems import CmemsGetRequest

    a = coordinator.cache_key_for_get(CmemsGetRequest(**_get_kwargs()))
    b = coordinator.cache_key_for_get(
        CmemsGetRequest(**_get_kwargs(), filter="*1990*")
    )
    assert a != b


@pytest.mark.asyncio
async def test_get_key_includes_regex(coordinator) -> None:
    from copernicus_mcp.data_model.schemas_cmems import CmemsGetRequest

    a = coordinator.cache_key_for_get(CmemsGetRequest(**_get_kwargs()))
    b = coordinator.cache_key_for_get(
        CmemsGetRequest(**_get_kwargs(), regex=".*1990.*")
    )
    assert a != b


@pytest.mark.asyncio
async def test_get_key_includes_dataset_part(coordinator) -> None:
    from copernicus_mcp.data_model.schemas_cmems import CmemsGetRequest

    a = coordinator.cache_key_for_get(CmemsGetRequest(**_get_kwargs()))
    b = coordinator.cache_key_for_get(
        CmemsGetRequest(**_get_kwargs(), dataset_part="default")
    )
    assert a != b


@pytest.mark.asyncio
async def test_get_key_file_list_order_independent(coordinator) -> None:
    """``file_list`` is set-semantically. Same files in different order
    must yield the same cache key — otherwise a caller who sorts their
    list differently on the second call would miss the cache."""
    from copernicus_mcp.data_model.schemas_cmems import CmemsGetRequest

    a = coordinator.cache_key_for_get(
        CmemsGetRequest(**_get_kwargs(), file_list=["a.nc", "b.nc"])
    )
    b = coordinator.cache_key_for_get(
        CmemsGetRequest(**_get_kwargs(), file_list=["b.nc", "a.nc"])
    )
    assert a == b


@pytest.mark.asyncio
async def test_get_key_distinct_namespace_from_subset(coordinator) -> None:
    """``operation`` differs, so a subset and a get against the same
    dataset cannot collide. Cache invariant 6 (purity)."""
    from copernicus_mcp.data_model.schemas_cmems import (
        CmemsGetRequest,
        CmemsSubsetRequest,
    )

    subset = CmemsSubsetRequest(**_subset_kwargs())
    get = CmemsGetRequest(dataset_id=subset.dataset_id)
    assert coordinator.cache_key_for_subset(
        subset
    ) != coordinator.cache_key_for_get(get)


@pytest.mark.asyncio
async def test_get_key_independent_of_sync_overwrite_skip(coordinator) -> None:
    """``sync``, ``skip_existing``, ``overwrite`` are SDK-forwarded
    operational flags — they don't change the data delivered, so they
    must NOT affect the cache key. Otherwise two semantically-identical
    downloads would compete for cache slots."""
    from copernicus_mcp.data_model.schemas_cmems import CmemsGetRequest

    a = coordinator.cache_key_for_get(CmemsGetRequest(**_get_kwargs()))
    b = coordinator.cache_key_for_get(
        CmemsGetRequest(
            **_get_kwargs(),
            sync=True,
            skip_existing=False,
            overwrite=True,
        )
    )
    assert a == b
