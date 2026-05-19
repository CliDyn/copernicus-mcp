"""``DataModelCoordinator.cache_key_for_cds_retrieve`` tests (T-CDS-002).

Cache keys for CDS retrieve requests must:
- be deterministic (same inputs → same key);
- depend on dataset_id AND the full ``inputs`` dict (semantically distinct
  requests must NOT collide);
- be insensitive to dict ordering (Python dict insertion order);
- strip credential-shaped keys recursively (the credential-isolation invariant).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def coordinator(tmp_path: Path):
    from copernicus_mcp.data_model.coordinator import DataModelCoordinator
    from copernicus_mcp.persistence import SqliteBackend

    persistence = SqliteBackend(tmp_path / "state.db")
    await persistence.initialise()
    try:
        yield DataModelCoordinator(persistence=persistence)
    finally:
        await persistence.close()


def _req(**overrides):
    from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest

    base = {
        "dataset_id": "reanalysis-era5-single-levels",
        "inputs": {
            "variable": ["2m_temperature"],
            "year": ["2024"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00"],
            "data_format": "grib",
        },
    }
    base.update(overrides)
    return CdsRetrieveRequest(**base)


@pytest.mark.asyncio
async def test_cache_key_is_deterministic(coordinator) -> None:
    a = coordinator.cache_key_for_cds_retrieve(_req())
    b = coordinator.cache_key_for_cds_retrieve(_req())
    assert a == b
    assert a.startswith("cds:submit:reanalysis-era5-single-levels:")


@pytest.mark.asyncio
async def test_cache_key_differs_on_dataset_id(coordinator) -> None:
    a = coordinator.cache_key_for_cds_retrieve(_req())
    b = coordinator.cache_key_for_cds_retrieve(
        _req(dataset_id="reanalysis-era5-pressure-levels")
    )
    assert a != b


@pytest.mark.asyncio
async def test_cache_key_differs_on_inputs(coordinator) -> None:
    a = coordinator.cache_key_for_cds_retrieve(_req())
    b = coordinator.cache_key_for_cds_retrieve(
        _req(
            inputs={
                "variable": ["2m_temperature"],
                "year": ["2025"],  # different year
                "month": ["01"],
                "day": ["01"],
                "time": ["00:00"],
                "data_format": "grib",
            }
        )
    )
    assert a != b


@pytest.mark.asyncio
async def test_cache_key_insensitive_to_dict_ordering(coordinator) -> None:
    """Same content in different insertion order → same key."""
    a = coordinator.cache_key_for_cds_retrieve(
        _req(
            inputs={
                "variable": ["t"],
                "year": ["2024"],
                "month": ["01"],
                "day": ["01"],
                "time": ["00:00"],
            }
        )
    )
    b = coordinator.cache_key_for_cds_retrieve(
        _req(
            inputs={
                "time": ["00:00"],
                "day": ["01"],
                "month": ["01"],
                "year": ["2024"],
                "variable": ["t"],
            }
        )
    )
    assert a == b


@pytest.mark.asyncio
async def test_cache_key_sensitive_to_list_order(coordinator) -> None:
    """``area`` is ordered ``[N, W, S, E]`` per research §6.9.2 — list
    order is meaningful. Reordering must change the key."""
    a = coordinator.cache_key_for_cds_retrieve(
        _req(
            inputs={
                "variable": ["t"],
                "year": ["2024"],
                "month": ["01"],
                "day": ["01"],
                "time": ["00:00"],
                "area": [60.0, -10.0, 50.0, 5.0],
            }
        )
    )
    b = coordinator.cache_key_for_cds_retrieve(
        _req(
            inputs={
                "variable": ["t"],
                "year": ["2024"],
                "month": ["01"],
                "day": ["01"],
                "time": ["00:00"],
                "area": [50.0, -10.0, 60.0, 5.0],  # N/S swapped
            }
        )
    )
    assert a != b


@pytest.mark.asyncio
async def test_cache_key_strips_credential_shaped_input(
    coordinator,
) -> None:
    """the credential-isolation invariant + cache-key purity (the project error-class convention
    invariant #6): a credential-shaped key in ``inputs`` (placed there
    by a buggy caller) must NOT influence the cache key."""
    a = coordinator.cache_key_for_cds_retrieve(_req())
    # Same request but with an extra ``api_key`` smuggled in by a buggy
    # caller. The cache key must collapse to the same digest.
    b = coordinator.cache_key_for_cds_retrieve(
        _req(
            inputs={
                "variable": ["2m_temperature"],
                "year": ["2024"],
                "month": ["01"],
                "day": ["01"],
                "time": ["00:00"],
                "data_format": "grib",
                "api_key": "TOPSECRET-DO-NOT-INFLUENCE-KEY",
            }
        )
    )
    assert a == b
