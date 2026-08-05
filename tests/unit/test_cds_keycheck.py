"""Reject unknown input keys at submit time (T-CDS-KEYCHECK-001).

field run 28: `insitu-gridded-observations-europe` was sent an `area` key. The
dataset is delivered as whole-domain files and its own input set is
`[grid_resolution, period, product_type, variable, version]` — no `area`. The
submit was accepted, the job ran two minutes, and came back
`remote_job_failed` with an empty log; the reporter isolated the cause only by diffing
four runs' state databases. The live CORDEX probe (spike T-CDS-MODEL-000)
reproduced the same class: obsolete `start_year`/`end_year` keys were
silently ignored and the ENTIRE 1950-2005 series was delivered instead of the
requested block.

The information to reject this up front ships in our own constraints
snapshot, and the snapshot includes adaptor keys where genuinely supported —
verified sweep 2026-08-04: `area`+`data_format` on ERA5/ERA5-Land, `date` on
CAMS (ADS), `hyear` on GloFAS (EWDS), `leadtime_hour` on seasonal all pass
with zero unknowns, while E-OBS+`area` is the single flag. Fail-open when a
dataset has no snapshot entry; `__options.skip_input_validation=true` is the
per-request escape hatch and `budget.cds_input_key_validation=false` the
global kill-switch.
"""

from __future__ import annotations

import dataclasses
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


def _patch_costing_flat(monkeypatch, *, units: float = 1.0, limit: float = 400.0):
    from copernicus_mcp.backends.cds.costing import CostingResult

    async def _fake(dataset_id, inputs, **_kwargs):
        return CostingResult(units=units, limit=limit)

    monkeypatch.setattr("copernicus_mcp.backends.cds.backend.fetch_costing", _fake)


def _patch_cdsapi(monkeypatch):
    import sys
    import types

    fake_module = types.ModuleType("cdsapi")
    instance = MagicMock()
    counter = {"n": 0}

    def _retrieve(name, request, target):
        counter["n"] += 1
        remote = MagicMock()
        remote.request_id = f"req-{counter['n']}"
        return remote

    instance.retrieve = MagicMock(side_effect=_retrieve)
    inner = MagicMock()
    instance.client = inner
    fake_class = MagicMock(return_value=instance)
    fake_module.Client = fake_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cdsapi", fake_module)
    return instance


def _eobs_with_area(**options: Any) -> dict[str, Any]:
    return {
        "dataset_id": "insitu-gridded-observations-europe",
        "inputs": {
            "product_type": "ensemble_mean",
            "variable": "mean_temperature",
            "grid_resolution": "0_25deg",
            "period": "full_period",
            "version": "30_0e",
            "area": [72, -25, 34, 45],
        },
        "__options": {"confirmed": True, **options},
    }


_EOBS_INPUTS: dict[str, Any] = _eobs_with_area()["inputs"]


def _era5_with_area(**extra: Any) -> dict[str, Any]:
    return {
        "dataset_id": "reanalysis-era5-single-levels",
        "inputs": {
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature"],
            "year": ["2024"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00"],
            "area": [46, -6, 30, 36.5],
            "data_format": "grib",
            **extra,
        },
        "__options": {"confirmed": True},
    }


# ---------------------------------------------------------------------------
# pure helper
# ---------------------------------------------------------------------------


def test_unknown_keys_flags_eobs_area_and_only_that() -> None:
    from copernicus_mcp.backends.cds.backend import _unknown_input_keys

    assert _unknown_input_keys(
        "insitu-gridded-observations-europe", _EOBS_INPUTS
    ) == ["area"]
    assert (
        _unknown_input_keys(
            "reanalysis-era5-single-levels", _era5_with_area()["inputs"]
        )
        == []
    )


def test_unknown_keys_fails_open_without_a_snapshot_entry() -> None:
    from copernicus_mcp.backends.cds.backend import _unknown_input_keys

    assert (
        _unknown_input_keys("some-brand-new-dataset", {"whatever": 1}) is None
    )


def test_known_good_request_shapes_sweep_clean() -> None:
    """Pin the snapshot property the whole feature depends on: adaptor keys
    (area, data_format, date, hyear, leadtime_hour) are PRESENT in the
    snapshots for the datasets that genuinely accept them. If a snapshot
    refresh ever drops them, this fails before users see false rejections."""
    from copernicus_mcp.backends.cds.backend import _unknown_input_keys

    sweep: dict[str, list[str]] = {
        "reanalysis-era5-single-levels": ["product_type", "variable", "year", "month", "day", "time", "data_format", "area"],
        "reanalysis-era5-land": ["variable", "year", "month", "day", "time", "area", "data_format"],
        "derived-era5-single-levels-daily-statistics": ["variable", "year", "month", "day", "daily_statistic", "time_zone", "frequency", "area"],
        "cams-global-reanalysis-eac4": ["variable", "date", "time", "data_format"],
        "cems-glofas-historical": ["system_version", "hydrological_model", "product_type", "variable", "hyear", "hmonth", "hday", "data_format"],
        "seasonal-original-single-levels": ["originating_centre", "system", "variable", "year", "month", "day", "leadtime_hour", "data_format"],
        "projections-cmip6": ["temporal_resolution", "experiment", "variable", "model", "year", "month", "area"],
    }
    for dataset_id, keys in sweep.items():
        unknown = _unknown_input_keys(dataset_id, dict.fromkeys(keys, "x"))
        assert unknown == [], f"{dataset_id}: false positives {unknown}"


# ---------------------------------------------------------------------------
# submit integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_eobs_area_is_rejected_at_submit_naming_the_key(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.errors import ValidationError

    _patch_costing_flat(monkeypatch)
    sdk = _patch_cdsapi(monkeypatch)
    backend = _backend(foundation)

    with pytest.raises(ValidationError) as exc:
        await backend.submit(_eobs_with_area())

    record = exc.value.error_record
    assert "'area'" in record.message
    assert "insitu-gridded-observations-europe" in record.message
    assert "grid_resolution" in (record.next_action_hint or "")  # accepted keys listed
    sdk.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_era5_with_area_and_data_format_passes(foundation, monkeypatch) -> None:
    _patch_costing_flat(monkeypatch)
    sdk = _patch_cdsapi(monkeypatch)
    backend = _backend(foundation)

    out = await backend.submit(_era5_with_area())

    assert out["status"] == "queued"
    sdk.retrieve.assert_called_once()


@pytest.mark.asyncio
async def test_legacy_format_key_gets_a_targeted_hint(foundation, monkeypatch) -> None:
    from copernicus_mcp.errors import ValidationError

    _patch_costing_flat(monkeypatch)
    _patch_cdsapi(monkeypatch)
    backend = _backend(foundation)

    params = _era5_with_area()
    del params["inputs"]["data_format"]
    params["inputs"]["format"] = "grib"
    with pytest.raises(ValidationError) as exc:
        await backend.submit(params)

    assert "data_format" in (exc.value.error_record.next_action_hint or "")


@pytest.mark.asyncio
async def test_unknown_dataset_fails_open(foundation, monkeypatch) -> None:
    """A dataset missing from the snapshot must never be blocked by the
    checker — a stale snapshot is our problem, not the caller's."""
    _patch_costing_flat(monkeypatch)
    sdk = _patch_cdsapi(monkeypatch)
    backend = _backend(foundation)

    out = await backend.submit(
        {
            "dataset_id": "some-brand-new-dataset",
            "inputs": {"exotic_key": "value", "variable": ["x"]},
            "__options": {"confirmed": True},
        }
    )

    assert out["status"] == "queued"
    sdk.retrieve.assert_called_once()


@pytest.mark.asyncio
async def test_escape_hatch_and_kill_switch(foundation, monkeypatch) -> None:
    _patch_costing_flat(monkeypatch)
    sdk = _patch_cdsapi(monkeypatch)

    out = await _backend(foundation).submit(
        _eobs_with_area(skip_input_validation=True)
    )
    assert out["status"] == "queued"

    out = await _backend(
        _with_budget(foundation, cds_input_key_validation=False)
    ).submit(_eobs_with_area(force_refresh=True))
    assert out["status"] == "queued"
    assert sdk.retrieve.call_count >= 2


def test_timeseries_location_key_is_accepted() -> None:
    """The ARCO ``*-timeseries`` products take a nested ``location`` point that
    the upstream machine-readable form omits; ``describe`` injects it into
    ``available_inputs`` (T-TS-007) and the key check must use the SAME
    augmented set — otherwise a documented core request shape is falsely
    rejected."""
    from copernicus_mcp.backends.cds.backend import _unknown_input_keys

    for dataset_id in (
        "reanalysis-era5-land-timeseries",
        "reanalysis-era5-single-levels-timeseries",
    ):
        unknown = _unknown_input_keys(
            dataset_id,
            {
                "variable": ["2m_temperature"],
                "location": {"latitude": 54.0, "longitude": 13.0},
                "date": "2024-01-01/2024-01-31",
                "data_format": "csv",
            },
        )
        assert unknown == [], f"{dataset_id}: {unknown}"
    # And a form WITHOUT `variable` keeps its own vocabulary intact: only
    # `location` is augmented, nothing else is blanket-allowed.
    unknown = _unknown_input_keys(
        "cams-solar-radiation-timeseries",
        {
            "location": {"latitude": 54.0, "longitude": 13.0},
            "date": "2024-01-01/2024-01-31",
            "sky_type": "clear",
            "variable": ["ghi"],
        },
    )
    assert unknown == ["variable"]


@pytest.mark.asyncio
async def test_unknown_key_names_are_sanitised_in_the_error(
    foundation, monkeypatch
) -> None:
    """local LOW: a credential-shaped string pasted as a KEY would otherwise
    echo verbatim into the persisted/wire error record."""
    from copernicus_mcp.errors import ValidationError

    _patch_costing_flat(monkeypatch)
    _patch_cdsapi(monkeypatch)
    backend = _backend(foundation)

    secret_key = "deadbeef-cafe-1234-5678-aabbccddeeff"
    params = _era5_with_area()
    params["inputs"][secret_key] = "x"
    with pytest.raises(ValidationError) as exc:
        await backend.submit(params)

    assert secret_key not in exc.value.error_record.message


@pytest.mark.asyncio
async def test_cordex_stale_period_keys_are_rejected_live_form_accepted(
    foundation, monkeypatch
) -> None:
    """Round-1 HIGH: the bundled CORDEX entry advertised `start_year`/`end_year`
    — keys the live server now silently IGNORES (spike probe 1 asked for
    1971-1975 and received the entire 1950-2005 run), while the live
    constraints engine serves `year`/`month`. The snapshot entry is refreshed
    from the live engine (2026-08-04): the proven-ignored keys must be
    REJECTED, and the live form must pass."""
    from copernicus_mcp.errors import ValidationError

    _patch_costing_flat(monkeypatch)
    sdk = _patch_cdsapi(monkeypatch)
    backend = _backend(foundation)

    base = {
        "domain": "europe",
        "horizontal_resolution": "0_11_degree_x_0_11_degree",
        "experiment": "historical",
        "temporal_resolution": "monthly_mean",
        "variable": ["2m_air_temperature"],
        "gcm_model": "mpi_m_mpi_esm_lr",
        "rcm_model": ["knmi_racmo22e"],
        "ensemble_member": "r1i1p1",
    }
    with pytest.raises(ValidationError) as exc:
        await backend.submit(
            {
                "dataset_id": "projections-cordex-domains-single-levels",
                "inputs": {**base, "start_year": ["1971"], "end_year": ["1975"]},
                "__options": {"confirmed": True},
            }
        )
    assert "start_year" in exc.value.error_record.message
    sdk.retrieve.assert_not_called()

    out = await backend.submit(
        {
            "dataset_id": "projections-cordex-domains-single-levels",
            "inputs": {**base, "year": ["1971"], "month": ["01"]},
            "__options": {"confirmed": True},
        }
    )
    assert out["status"] == "queued"
