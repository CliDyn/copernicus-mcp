"""Delivered-content check at finalisation (T-CDS-MODEL-002).

Last line of defence for the single-model-execution datasets: even with the
fan-out in place, a service-side regression could deliver the wrong or a
missing model — and a delivery that silently lacks what was requested must
never be stored under the request's cache key, where it would satisfy every
future dedupe looking for the real thing. Verified rule from the live CORDEX
probe (spike T-CDS-MODEL-000): requested tokens are lowercase snake
(``knmi_racmo22e``), delivered member names are mixed case with hyphens
(``KNMI-RACMO22E``) — compare with case and ``-``/``_`` folded; check ``.nc``
members only (Rook adds ``provenance.json``/``provenance.png`` to every zip).
"""

from __future__ import annotations

import dataclasses
import io
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

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


def _fake_creds():
    from copernicus_mcp.auth.resolver import ResolvedCredentials

    return ResolvedCredentials(
        backend="cds",
        source="explicit",
        source_detail="test",
        fields={"key": "abcdef01-2345-6789-abcd-ef0123456789"},
    )


def _with_budget(foundation, **updates):
    budget = foundation.config.budget.model_copy(update=updates)
    config = foundation.config.model_copy(update={"budget": budget})
    return dataclasses.replace(foundation, config=config)


def _backend(foundation):
    from copernicus_mcp.backends.cds.backend import CdsBackend

    return CdsBackend(foundation=foundation, credentials=_fake_creds())


def _zip_bytes(*names: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in names:
            zf.writestr(name, b"netcdf-payload")
    return buf.getvalue()


def _patch_costing_flat(monkeypatch, *, units: float = 10.0, limit: float = 400.0):
    from copernicus_mcp.backends.cds.costing import CostingResult

    async def _fake(dataset_id, inputs, **_kwargs):
        return CostingResult(units=units, limit=limit)

    monkeypatch.setattr("copernicus_mcp.backends.cds.backend.fetch_costing", _fake)


def _fake_remote(request_id: str) -> MagicMock:
    remote = MagicMock()
    remote.request_id = request_id
    return remote


def _patch_cdsapi_delivering(monkeypatch, status_by_request, payload_for):
    """The chunk-lifecycle fake, with ``payload_for(request_id) -> bytes``
    controlling exactly what each download delivers."""
    import sys
    import types

    fake_module = types.ModuleType("cdsapi")
    instance = MagicMock()
    counter = {"n": 0}

    def _retrieve(name, request, target):
        counter["n"] += 1
        return _fake_remote(f"child-{counter['n']}")

    instance.retrieve = MagicMock(side_effect=_retrieve)
    inner = MagicMock()

    def _get_remote(request_id):
        rem = MagicMock()
        rem.json = {
            "status": status_by_request.get(request_id, "running"),
            "jobID": request_id,
        }
        return rem

    inner.get_remote = MagicMock(side_effect=_get_remote)
    inner.delete = MagicMock(return_value={"deleted": True})

    def _download_results(request_id, target):
        Path(target).write_bytes(payload_for(request_id))
        return target

    inner.download_results = MagicMock(side_effect=_download_results)
    instance.client = inner
    fake_class = MagicMock(return_value=instance)
    fake_module.Client = fake_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cdsapi", fake_module)
    return instance


def _cmip6_single(model: str) -> dict[str, Any]:
    return {
        "dataset_id": "projections-cmip6",
        "inputs": {
            "temporal_resolution": "monthly",
            "experiment": "historical",
            "variable": "near_surface_air_temperature",
            "model": [model],
            "year": ["2000"],
            "month": ["01"],
        },
    }


async def _submit_and_poll(backend, params, status):
    out = await backend.submit(params)
    status[out["request_id"]] = "successful"
    # A direct submit's request_id is the SDK-assigned child id.
    for _ in range(3):
        st = await backend.check_status(out["request_id"])
        if st["status"] in ("successful", "failed"):
            return st
    return st


# ---------------------------------------------------------------------------
# pure normalisation
# ---------------------------------------------------------------------------


def test_missing_model_tokens_folds_case_and_separators(tmp_path: Path) -> None:
    from copernicus_mcp.backends.cds.backend import _missing_model_tokens

    path = tmp_path / "d.zip"
    path.write_bytes(
        _zip_bytes(
            "tas_EUR-11_MPI-M-MPI-ESM-LR_historical_r1i1p1_KNMI-RACMO22E_v1_mon.nc",
            "provenance.json",
            "provenance.png",
        )
    )
    inputs = {"gcm_model": "mpi_m_mpi_esm_lr", "rcm_model": ["knmi_racmo22e"]}
    assert (
        _missing_model_tokens(path, inputs, ("gcm_model", "rcm_model")) == []
    )
    # And the failure leg: an RCM that is NOT in the delivery is reported.
    inputs_bad = {"gcm_model": "mpi_m_mpi_esm_lr", "rcm_model": ["smhi_rca4"]}
    assert _missing_model_tokens(path, inputs_bad, ("gcm_model", "rcm_model")) == [
        "smhi_rca4"
    ]


def test_missing_model_tokens_inapplicable_shapes(tmp_path: Path) -> None:
    """Non-zip payloads and zips without .nc members cannot be judged — the
    check declines (None) rather than failing valid data."""
    from copernicus_mcp.backends.cds.backend import _missing_model_tokens

    bare = tmp_path / "d.nc"
    bare.write_bytes(b"CDF\x01netcdf-payload")
    assert _missing_model_tokens(bare, {"model": ["access_cm2"]}, ("model",)) is None

    empty = tmp_path / "e.zip"
    empty.write_bytes(_zip_bytes("provenance.json"))
    assert _missing_model_tokens(empty, {"model": ["access_cm2"]}, ("model",)) is None


# ---------------------------------------------------------------------------
# end-to-end through finalisation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_correct_delivery_stores_and_succeeds(foundation, monkeypatch) -> None:
    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_delivering(
        monkeypatch,
        status,
        lambda rid: _zip_bytes(
            "tas_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_200001.nc",
            "provenance.json",
        ),
    )
    backend = _backend(foundation)

    st = await _submit_and_poll(backend, _cmip6_single("access_cm2"), status)

    assert st["status"] == "successful"
    assert st["result"]["filepath"]


@pytest.mark.asyncio
async def test_wrong_model_delivery_fails_and_never_reaches_the_cache(
    foundation, monkeypatch
) -> None:
    """Requested access_cm2, delivered MIROC6: the row fails with
    delivered_content_mismatch naming the missing token, and NOTHING is stored
    under the cache key — a poisoned entry would satisfy every future dedupe."""
    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_delivering(
        monkeypatch,
        status,
        lambda rid: _zip_bytes("tas_Amon_MIROC6_historical_r1i1p1f1_gn_200001.nc"),
    )
    backend = _backend(foundation)

    st = await _submit_and_poll(backend, _cmip6_single("access_cm2"), status)

    assert st["status"] == "failed"
    err = st["error_details"]
    assert err["error_subclass"] == "delivered_content_mismatch"
    assert "access_cm2" in err["message"]

    row = await foundation.persistence.fetch_workflow(st["request_id"])
    from copernicus_mcp.backends.cds.backend import _cache_storage_key

    cached = await foundation.cache.lookup_file(_cache_storage_key(row["cache_key"]))
    assert cached is None


@pytest.mark.asyncio
async def test_check_disabled_by_config_restores_old_behaviour(
    foundation, monkeypatch
) -> None:
    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_delivering(
        monkeypatch,
        status,
        lambda rid: _zip_bytes("tas_Amon_MIROC6_historical_r1i1p1f1_gn_200001.nc"),
    )
    backend = _backend(
        _with_budget(foundation, cds_delivery_check_enabled=False)
    )

    st = await _submit_and_poll(backend, _cmip6_single("access_cm2"), status)

    assert st["status"] == "successful"


@pytest.mark.asyncio
async def test_non_registry_dataset_is_not_checked(foundation, monkeypatch) -> None:
    """ERA5 zips carry no model naming convention — no check applies."""
    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_delivering(
        monkeypatch, status, lambda rid: _zip_bytes("era5_data_stream.nc")
    )
    backend = _backend(foundation)

    st = await _submit_and_poll(
        backend,
        {
            "dataset_id": "reanalysis-era5-single-levels",
            "inputs": {
                "product_type": ["reanalysis"],
                "variable": ["2m_temperature"],
                "year": ["2024"],
                "month": ["01"],
                "day": ["01"],
                "time": ["00:00"],
            },
        },
        status,
    )

    assert st["status"] == "successful"


@pytest.mark.asyncio
async def test_fanout_child_with_wrong_model_fails_its_parent(
    foundation, monkeypatch
) -> None:
    """The full protection: a 2-model fan-out where the service delivers
    the FIRST model twice. Child 2's delivery lacks its model → child fails →
    parent fails with partial_result carrying only the good chunk. A silently
    partial ensemble can never look complete."""
    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_delivering(
        monkeypatch,
        status,
        # Rook regression simulated: every child gets ACCESS-CM2's file.
        lambda rid: _zip_bytes("tas_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_200001.nc"),
    )
    backend = _backend(foundation)

    params = {
        "dataset_id": "projections-cmip6",
        "inputs": {
            "temporal_resolution": "monthly",
            "experiment": "historical",
            "variable": "near_surface_air_temperature",
            "model": ["access_cm2", "miroc6"],
            "year": ["2000"],
            "month": ["01"],
        },
    }
    out = await backend.submit(params)
    assert out["chunked"] is True
    status["child-1"] = "successful"
    status["child-2"] = "successful"

    final = None
    for _ in range(4):
        final = await backend.check_status(out["request_id"])
        if final["status"] in ("successful", "failed"):
            break

    assert final is not None and final["status"] == "failed"
    partial = final["partial_result"]
    assert partial["chunk_indices"] == [0]  # only the genuine ACCESS-CM2 chunk
