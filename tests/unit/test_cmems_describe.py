from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio


def _make_foundation(tmp_path: Path):
    from copernicus_mcp.auth import CredentialResolver
    from copernicus_mcp.backends.abstract import FoundationServices
    from copernicus_mcp.cache import CacheManager
    from copernicus_mcp.config import ConfigLoader
    from copernicus_mcp.data_model.coordinator import DataModelCoordinator
    from copernicus_mcp.data_model.provenance import ProvenanceRecorder
    from copernicus_mcp.errors.sanitiser import Sanitiser
    from copernicus_mcp.http import HttpClientFactory
    from copernicus_mcp.persistence import SqliteBackend

    config = ConfigLoader().load()
    persistence = SqliteBackend(tmp_path / "state.db")
    cache = CacheManager(
        cache_directory=tmp_path / "cache",
        persistence=persistence,
        size_limit_bytes=10 * 1024 * 1024,
    )
    return (
        FoundationServices(
            config=config,
            credential_resolver=CredentialResolver(),
            http_client_factory=HttpClientFactory(http_config=config.http),
            persistence=persistence,
            cache=cache,
            sanitiser=Sanitiser(),
            data_model=DataModelCoordinator(persistence=persistence),
            provenance=ProvenanceRecorder(
                persistence=persistence,
                software_versions={"copernicus-mcp": "0.0.1"},
            ),
        ),
        persistence,
    )


@pytest_asyncio.fixture
async def foundation(tmp_path: Path):
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        yield found
    finally:
        await persistence.close()


def _creds():
    from copernicus_mcp.auth.resolver import ResolvedCredentials

    return ResolvedCredentials(
        backend="cmems",
        source="explicit",
        source_detail="test",
        fields={"username": "u", "password": "p"},
    )


def _fake_full_response() -> dict[str, Any]:
    """Mimics ``copernicusmarine.describe(dataset_id=...)`` for one dataset."""
    return {
        "products": [
            {
                "product_id": "GLOBAL_ANALYSISFORECAST_PHY_001_024",
                "title": "Global Ocean Physics",
                "doi": "https://doi.org/10.48670/moi-00016",
                "license": "Mercator Ocean Licence",
                "citation": "EU Copernicus Marine Service Information",
                "datasets": [
                    {
                        "dataset_id": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
                        "title": "Daily mean physics",
                        "versions": [
                            {
                                "label": "202411",
                                "parts": [
                                    {
                                        "name": "default",
                                        "services": [
                                            {
                                                "service_type": "geoseries",
                                                "service_name": "arco-geo-series",
                                                "uri": "https://my.cmems-du.eu/x",
                                            }
                                        ],
                                        "variables": [
                                            {
                                                "short_name": "thetao",
                                                "long_name": "Sea water temperature",
                                                "units": "degree_Celsius",
                                                "valid_min": -2.0,
                                                "valid_max": 40.0,
                                                "coordinates": [
                                                    {
                                                        "coordinate_id": "longitude",
                                                        "minimum_value": -180.0,
                                                        "maximum_value": 180.0,
                                                        "step": 0.0833,
                                                        "values": list(range(50)),
                                                    },
                                                    {
                                                        "coordinate_id": "latitude",
                                                        "minimum_value": -90.0,
                                                        "maximum_value": 90.0,
                                                        "step": 0.0833,
                                                        "values": list(range(40)),
                                                    },
                                                    {
                                                        "coordinate_id": "depth",
                                                        "minimum_value": 0.0,
                                                        "maximum_value": 5727.9,
                                                        "values": [0.49, 1.54, 2.65],
                                                    },
                                                    {
                                                        "coordinate_id": "time",
                                                        "minimum_value": "2024-01-01",
                                                        "maximum_value": "2024-12-31",
                                                        "step": 86400,
                                                        "values": [
                                                            f"2024-01-{d:02d}"
                                                            for d in range(1, 32)
                                                        ],
                                                    },
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                        "spatial_extent": {
                            "min_lon": -180,
                            "max_lon": 180,
                            "min_lat": -90,
                            "max_lat": 90,
                        },
                        "temporal_extent": {
                            "start": "2022-01-01",
                            "end": "2025-04-01",
                        },
                    }
                ],
            }
        ]
    }


def _install_fake_module(monkeypatch, describe_fn) -> types.ModuleType:
    mod = types.ModuleType("copernicusmarine")
    mod.describe = describe_fn  # type: ignore[attr-defined]

    class LoginError(Exception):
        pass

    class DatasetNotFound(Exception):
        pass

    mod.LoginError = LoginError  # type: ignore[attr-defined]
    mod.DatasetNotFound = DatasetNotFound  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "copernicusmarine", mod)
    return mod


@pytest.mark.asyncio
async def test_describe_returns_full_metadata(foundation, monkeypatch) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    _install_fake_module(monkeypatch, lambda **k: _fake_full_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    result = await backend.describe("cmems_mod_glo_phy_anfc_0.083deg_P1D-m")

    assert result["dataset_id"] == "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
    assert result["dataset_version"] == "202411"
    assert "default" in result["dataset_parts"]
    assert any(s["service_type"] == "geoseries" for s in result["services"])
    assert any(v["name"] == "thetao" for v in result["variables"])
    assert result["doi"]
    assert result["license"]


def _fake_response_real_sdk_shape() -> dict[str, Any]:
    """Mimics the real ``copernicusmarine`` SDK: extent on variable, not dataset.

    Empirically verified via ``cm.describe(dataset_id='cmems_mod_med_phy-sal_anfc_4.2km_P1M-m')``:
    ``dataset.spatial_extent`` is absent; each variable has ``bbox = [min_lon,
    min_lat, max_lon, max_lat]``. Agent that promised "all of Mediterranean"
    silently clipped the Atlantic buffer west of Gibraltar because describe
    returned ``spatial_extent: null`` and the agent guessed from geography.
    Backend must aggregate variable bboxes into dataset spatial_extent so
    LLM agents see the ground truth.
    """
    return {
        "products": [
            {
                "product_id": "MEDSEA_ANALYSISFORECAST_PHY_006_013",
                "title": "Mediterranean Sea physics",
                "datasets": [
                    {
                        "dataset_id": "cmems_mod_med_phy-sal_anfc_4.2km_P1M-m",
                        "title": "Salinity monthly mean",
                        "versions": [
                            {
                                "label": "202511",
                                "parts": [
                                    {
                                        "name": "default",
                                        "services": [
                                            {
                                                "service_name": "arco-geo-series",
                                                "uri": "https://example/x",
                                                "variables": [
                                                    {
                                                        "short_name": "so",
                                                        "bbox": [
                                                            -17.29,
                                                            30.19,
                                                            36.29,
                                                            45.98,
                                                        ],
                                                    },
                                                    {
                                                        "short_name": "so_extra",
                                                        "bbox": [
                                                            -18.0,
                                                            30.0,
                                                            36.5,
                                                            46.0,
                                                        ],
                                                    },
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                        # NOTE: no spatial_extent / temporal_extent here —
                        # mirrors real SDK output.
                    }
                ],
            }
        ]
    }


@pytest.mark.asyncio
async def test_describe_aggregates_spatial_extent_from_variable_bbox(
    foundation, monkeypatch
) -> None:
    """When SDK omits dataset-level extent but variables carry bbox, the
    backend must aggregate (union) variable bboxes into ``spatial_extent``."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    _install_fake_module(
        monkeypatch, lambda **k: _fake_response_real_sdk_shape()
    )
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    result = await backend.describe(
        "cmems_mod_med_phy-sal_anfc_4.2km_P1M-m"
    )

    extent = result["spatial_extent"]
    assert extent is not None, "spatial_extent must be derived from variables"
    # Union of the two variable bboxes
    assert extent["min_lon"] == -18.0
    assert extent["max_lon"] == 36.5
    assert extent["min_lat"] == 30.0
    assert extent["max_lat"] == 46.0


@pytest.mark.asyncio
async def test_describe_empty_identifier_rejected(foundation, monkeypatch) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import ValidationError

    _install_fake_module(monkeypatch, lambda **k: _fake_full_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(ValidationError):
        await backend.describe("")


@pytest.mark.asyncio
async def test_describe_unknown_dataset_raises_not_found(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import NotFoundError

    def fake(**kwargs):
        from copernicusmarine import DatasetNotFound  # type: ignore[attr-defined]

        raise DatasetNotFound("nope")

    _install_fake_module(monkeypatch, fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(NotFoundError):
        await backend.describe("missing")


@pytest.mark.asyncio
async def test_describe_caches(foundation, monkeypatch) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    n = {"calls": 0}

    def fake(**kwargs):
        n["calls"] += 1
        return _fake_full_response()

    _install_fake_module(monkeypatch, fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    a = await backend.describe("cmems_mod_glo_phy_anfc_0.083deg_P1D-m")
    b = await backend.describe("cmems_mod_glo_phy_anfc_0.083deg_P1D-m")
    assert a == b
    assert n["calls"] == 1


@pytest.mark.asyncio
async def test_get_coordinates_small_returns_full_list(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    _install_fake_module(monkeypatch, lambda **k: _fake_full_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    coords = await backend._get_coordinates(
        "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
    )
    # depth has 3 entries → full list
    assert isinstance(coords["depth"], list)
    assert len(coords["depth"]) == 3
    # time has 31 entries → still small, full list
    assert isinstance(coords["time"], list)


@pytest.mark.asyncio
async def test_get_coordinates_large_time_summarised(
    foundation, monkeypatch
) -> None:
    """Time axis with >5000 entries is summarised, not enumerated."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    big = _fake_full_response()
    coords = (
        big["products"][0]["datasets"][0]["versions"][0]["parts"][0][
            "variables"
        ][0]["coordinates"]
    )
    time_coord = next(c for c in coords if c["coordinate_id"] == "time")
    time_coord["values"] = [f"2024-{i:05d}" for i in range(6000)]

    _install_fake_module(monkeypatch, lambda **k: big)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    out = await backend._get_coordinates(
        "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
    )
    assert isinstance(out["time"], dict)
    assert out["time"]["count"] == 6000
    assert "start" in out["time"]
    assert "end" in out["time"]
    # Time axis uses ``stride_seconds`` (the step is in seconds).
    assert "stride_seconds" in out["time"]


@pytest.mark.asyncio
async def test_get_coordinates_large_longitude_summarised(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    big = _fake_full_response()
    coords = (
        big["products"][0]["datasets"][0]["versions"][0]["parts"][0][
            "variables"
        ][0]["coordinates"]
    )
    lon = next(c for c in coords if c["coordinate_id"] == "longitude")
    lon["values"] = list(range(15000))

    _install_fake_module(monkeypatch, lambda **k: big)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    out = await backend._get_coordinates(
        "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"
    )
    assert isinstance(out["longitude"], dict)
    assert out["longitude"]["count"] == 15000
    # Spatial axis uses plain ``stride`` (degrees), not ``stride_seconds``.
    assert "stride" in out["longitude"]
    assert "stride_seconds" not in out["longitude"]


@pytest.mark.asyncio
async def test_get_coordinates_unknown_service_raises_not_found(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import NotFoundError

    _install_fake_module(monkeypatch, lambda **k: _fake_full_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(NotFoundError, match="service"):
        await backend._get_coordinates(
            "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            service="does-not-exist",
        )


@pytest.mark.asyncio
async def test_get_coordinates_unknown_version_raises_not_found(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import NotFoundError

    _install_fake_module(monkeypatch, lambda **k: _fake_full_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(NotFoundError, match="version"):
        await backend._get_coordinates(
            "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            dataset_version="999999",
        )


def test_latest_version_picks_highest_label() -> None:
    """_latest_version uses lex sort, not list order, so newer can be anywhere."""
    from copernicus_mcp.backends.cmems.backend import _latest_version

    dataset = {
        "versions": [
            {"label": "202405"},
            {"label": "202411"},  # newest
            {"label": "202408"},
        ]
    }
    assert _latest_version(dataset)["label"] == "202411"


# ---------------------------------------------------------------------------
# Public ``get_coordinates(params)`` — T-022 second half, MCP-facing surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_public_get_coordinates_validates_via_schema(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import ValidationError

    _install_fake_module(monkeypatch, lambda **k: _fake_full_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(ValidationError):
        await backend.get_coordinates({"unknown": "boom"})


@pytest.mark.asyncio
async def test_public_get_coordinates_returns_coords(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    _install_fake_module(monkeypatch, lambda **k: _fake_full_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    coords = await backend.get_coordinates(
        {"dataset_id": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"}
    )
    assert "depth" in coords
    assert "time" in coords


@pytest.mark.asyncio
async def test_public_get_coordinates_requires_credentials(foundation) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import AuthError

    backend = CmemsBackend(foundation=foundation, credentials=None)
    with pytest.raises(AuthError):
        await backend.get_coordinates({"dataset_id": "x"})


@pytest.mark.asyncio
async def test_public_get_coordinates_blank_dataset_id_rejected(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import ValidationError

    _install_fake_module(monkeypatch, lambda **k: _fake_full_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(ValidationError):
        await backend.get_coordinates({"dataset_id": "   "})


@pytest.mark.asyncio
async def test_public_get_coordinates_unknown_field_does_not_leak_value(
    foundation, monkeypatch
) -> None:
    # T-CMEMS-GET-INDEX-004 H1 pattern: pydantic ValidationError must not
    # echo raw input values via input_value=... — our _validate helper
    # projects to loc/msg/type only.
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import ValidationError

    _install_fake_module(monkeypatch, lambda **k: _fake_full_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(ValidationError) as exc_info:
        await backend.get_coordinates({"dataset_id": "x", "password": "hunter2"})
    assert "hunter2" not in str(exc_info.value)
