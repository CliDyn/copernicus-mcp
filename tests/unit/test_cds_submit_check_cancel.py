"""``CdsBackend.submit`` / ``check_status`` / ``fetch_result`` / ``cancel``
tests (T-CDS-005).

CDS is async-by-design (research §6.5): every retrieve goes through the
queue. The backend ``submit`` only registers a job; ``check_status`` is
the finaliser that polls the remote job and downloads on ``successful``.
Mocks target ``cdsapi.Client`` (which routes to
``ecmwf.datastores.legacy_client.LegacyClient`` for UUID PATs) and the
underlying ``datastores.Client`` / ``datastores.Remote`` surface.

No network calls in these tests — integration coverage is T-CDS-008.
"""

from __future__ import annotations

import asyncio
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


def _good_params() -> dict[str, Any]:
    return {
        "dataset_id": "reanalysis-era5-single-levels",
        "inputs": {
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature"],
            "year": ["2024"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00"],
            "data_format": "grib",
        },
    }


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_raises_validation_error_on_invalid_params(
    foundation,
) -> None:
    """Empty ``dataset_id`` must be rejected before any SDK call."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import ValidationError

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    bad = {"dataset_id": "", "inputs": {"variable": ["x"]}}
    with pytest.raises(ValidationError):
        await backend.submit(bad)


@pytest.mark.asyncio
async def test_submit_without_credentials_raises_auth_error(foundation) -> None:
    """Valid params but no credentials must surface ``AuthError`` —
    don't even attempt the SDK call."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import AuthError

    backend = CdsBackend(foundation=foundation, credentials=None)
    with pytest.raises(AuthError):
        await backend.submit(_good_params())


# ---------------------------------------------------------------------------
# product_type required-field pre-flight (T-CDS-PT — field report)
# ---------------------------------------------------------------------------


def test_missing_product_type_flags_dataset_that_requires_it() -> None:
    from copernicus_mcp.backends.cds.backend import _missing_product_type

    inputs = {
        "variable": ["10m_u_component_of_wind"],
        "year": ["1993"], "month": ["01"], "day": ["01"], "time": ["00:00"],
        "data_format": "netcdf",
    }
    vals = _missing_product_type("reanalysis-era5-single-levels", inputs)
    assert vals is not None
    assert "reanalysis" in vals


def test_missing_product_type_none_when_present() -> None:
    from copernicus_mcp.backends.cds.backend import _missing_product_type

    inputs = {"product_type": ["reanalysis"], "variable": ["x"], "year": ["1993"]}
    assert _missing_product_type("reanalysis-era5-single-levels", inputs) is None


def test_missing_product_type_none_for_dataset_without_it() -> None:
    from copernicus_mcp.backends.cds.backend import _missing_product_type

    # satellite-sea-surface-temperature has bundled constraints but no
    # product_type field → the rule must not fire.
    assert (
        _missing_product_type(
            "satellite-sea-surface-temperature", {"variable": ["x"]}
        )
        is None
    )


def test_missing_product_type_treats_empty_as_missing() -> None:
    from copernicus_mcp.backends.cds.backend import _missing_product_type

    assert (
        _missing_product_type(
            "reanalysis-era5-single-levels", {"product_type": [], "variable": ["x"]}
        )
        is not None
    )


def test_missing_product_type_skips_non_reanalysis_families() -> None:
    """The rule is scoped to the ECMWF reanalysis families, where the
    MARS-split 'Duplicate value for month' failure is evidenced. A
    non-reanalysis dataset that merely LISTS product_type in the empty-
    inputs constraints snapshot (satellite/insitu/sis/derived/...) is NOT
    blocked — the snapshot can't tell required from optional, so we don't
    guess (v2 review MEDIUM)."""
    from copernicus_mcp.backends.cds.backend import _missing_product_type

    # satellite-fire-radiative-power defines product_type in constraints
    # but is not a reanalysis dataset → must not fire.
    assert (
        _missing_product_type("satellite-fire-radiative-power", {"variable": ["x"]})
        is None
    )


def test_missing_product_type_scalar_string_counts_as_present() -> None:
    """A scalar-string product_type (``"reanalysis"``, not a list) is a
    legal cdsapi shape and must count as present (reviewer LOW)."""
    from copernicus_mcp.backends.cds.backend import _missing_product_type

    assert (
        _missing_product_type(
            "reanalysis-era5-single-levels", {"product_type": "reanalysis"}
        )
        is None
    )


@pytest.mark.asyncio
async def test_submit_requires_product_type_before_network(foundation) -> None:
    """A dataset that defines product_type (ERA5) must reject a submit that
    omits it BEFORE any network/SDK call — turning the 40-min queue→MARS
    "Duplicate value for month" failure into an instant, actionable error.
    credentials=None proves the check fires pre-credential (so no network)."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import ValidationError

    backend = CdsBackend(foundation=foundation, credentials=None)
    params = {
        "dataset_id": "reanalysis-era5-single-levels",
        "inputs": {
            "variable": ["10m_u_component_of_wind"],
            "year": ["1993"], "month": ["01"], "day": ["01"], "time": ["00:00"],
            "data_format": "netcdf",
        },
    }
    with pytest.raises(ValidationError) as exc:
        await backend.submit(params)
    assert "product_type" in str(exc.value).lower()


def _fake_remote(request_id: str = "abc-123") -> MagicMock:
    """Mimic the surface of ``ecmwf.datastores.processing.Remote`` we
    use: ``request_id`` plus the ``json`` shape returned by the GET."""
    remote = MagicMock()
    remote.request_id = request_id
    remote.json = {"status": "accepted", "jobID": request_id}
    return remote


_CDS_DEFAULT_DOWNLOAD_BYTES = b"GRIB-content-from-cds"


def _patch_cdsapi(
    monkeypatch,
    *,
    retrieve_returns=None,
    get_remote_json=None,
    download_bytes: bytes = _CDS_DEFAULT_DOWNLOAD_BYTES,
):
    """Stub ``cdsapi.Client`` so the backend doesn't hit the network.

    Returns ``(fake_class, instance)`` so tests can assert constructor
    kwargs and call counts. The instance has:
    - ``retrieve(name, request, target)`` returning ``retrieve_returns``
      (typically a fake Remote);
    - ``client`` (the underlying ``datastores.Client``) with
      ``get_remote(request_id)`` returning a Remote whose ``json`` is
      ``get_remote_json`` (default ``{"status": "running"}``);
    - ``client.delete(*request_ids)`` MagicMock for cancel tests;
    - ``client.download_results(request_id, target)`` MagicMock that
      writes a tiny file at ``target`` and returns its path.
    """
    import sys
    import types

    fake_module = types.ModuleType("cdsapi")
    instance = MagicMock()
    instance.retrieve = MagicMock(return_value=retrieve_returns)

    inner = MagicMock()
    poll_remote = MagicMock()
    poll_remote.json = get_remote_json or {"status": "running"}
    inner.get_remote = MagicMock(return_value=poll_remote)
    inner.delete = MagicMock(return_value={"deleted": True})

    def _download_results(request_id: str, target: str) -> str:
        Path(target).write_bytes(download_bytes)
        return target

    inner.download_results = MagicMock(side_effect=_download_results)
    instance.client = inner

    fake_class = MagicMock(return_value=instance)
    fake_module.Client = fake_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cdsapi", fake_module)
    return fake_class, instance


async def _seed_workflow_row(
    foundation,
    *,
    request_id: str,
    cache_key: str = "ck-test",
    status: str = "queued",
    error_record_json: str | None = None,
    age_seconds: float = 0,
) -> None:
    """Insert a workflow row mirroring what ``submit`` would create."""
    from datetime import UTC, datetime, timedelta

    base = datetime.now(UTC) - timedelta(seconds=age_seconds)
    iso = base.strftime("%Y-%m-%dT%H:%M:%SZ")
    await foundation.persistence.record_workflow(
        {
            "request_id": request_id,
            "backend_id": "cds",
            "operation": "submit",
            "status": status,
            "cache_key": cache_key,
            "request_json": "{}",
            "response_json": None,
            "error_record_json": error_record_json,
            "created_at": iso,
            "updated_at": iso,
        }
    )


@pytest.mark.asyncio
async def test_submit_returns_pending_envelope_with_request_id(foundation, monkeypatch) -> None:
    """Happy path: SDK returns a Remote, backend persists a workflow
    row and returns the canonical pending envelope with the cdsapi
    request_id surfaced."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    remote = _fake_remote("req-happy")
    _, sdk = _patch_cdsapi(monkeypatch, retrieve_returns=remote)

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.submit(_good_params())

    assert out["status"] == "queued"
    assert out["request_id"] == "req-happy"
    assert out["cache_hit"] is False
    assert "cache_key" in out
    assert sdk.retrieve.call_count == 1


@pytest.mark.asyncio
async def test_submit_idempotent_on_cache_hit(foundation, monkeypatch, tmp_path: Path) -> None:
    """If the cache_key already maps to a stored file, ``submit`` returns
    a ``cache_hit=True`` envelope without invoking the SDK at all.
    Mirrors the CMEMS short-circuit at backends/cmems/backend.py:333."""
    from copernicus_mcp.backends.cds.backend import (
        CdsBackend,
        _cache_storage_key,
    )
    from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    req = CdsRetrieveRequest.model_validate(_good_params())
    cache_key = foundation.data_model.cache_key_for_cds_retrieve(req)

    # Pre-seed cache
    canned = tmp_path / "canned.grib"
    canned.write_bytes(b"GRIB-content")
    await foundation.cache.store_file(
        _cache_storage_key(cache_key),
        canned,
        backend_id="cds",
        content_type="application/x-grib",
    )

    fake_class, sdk = _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote())

    out = await backend.submit(_good_params())
    assert out["cache_hit"] is True
    assert out["status"] == "successful"
    fake_class.assert_not_called()  # SDK was never instantiated
    assert sdk.retrieve.call_count == 0


@pytest.mark.asyncio
async def test_submit_dedupes_inflight_request(foundation, monkeypatch) -> None:
    """A workflow row with the same ``cache_key`` and a non-terminal
    status must reuse the existing ``request_id``; we do NOT enqueue a
    duplicate job on the CDS server."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    req = CdsRetrieveRequest.model_validate(_good_params())
    cache_key = foundation.data_model.cache_key_for_cds_retrieve(req)

    # Pre-seed an in-flight workflow row.
    now = "2026-05-08T00:00:00Z"
    await foundation.persistence.record_workflow(
        {
            "request_id": "existing-req-id",
            "backend_id": "cds",
            "operation": "submit",
            "status": "running",
            "cache_key": cache_key,
            "request_json": "{}",
            "response_json": None,
            "error_record_json": None,
            "created_at": now,
            "updated_at": now,
        }
    )

    fake_class, sdk = _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote())
    out = await backend.submit(_good_params())
    assert out["request_id"] == "existing-req-id"
    fake_class.assert_not_called()
    assert sdk.retrieve.call_count == 0


@pytest.mark.asyncio
async def test_submit_gate_above_size_threshold_raises_confirmation(
    foundation, monkeypatch
) -> None:
    """Estimate above ``cds_per_request_size_warning_gb`` without
    ``confirmed=True`` raises ``ConfirmationRequired`` and never calls
    the SDK."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.workflow.confirmation import ConfirmationRequired

    # 30y of monthly hourly ERA5-PL across many pressure-levels — heavy.
    big_params = {
        "dataset_id": "reanalysis-era5-pressure-levels",
        "inputs": {
            "product_type": ["reanalysis"],
            "variable": ["temperature"],
            "year": [str(y) for y in range(1990, 2024)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "pressure_level": ["500", "850", "1000"],
            "data_format": "grib",
        },
    }

    fake_class, _ = _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote())
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    with pytest.raises(ConfirmationRequired):
        await backend.submit(big_params)
    fake_class.assert_not_called()


@pytest.mark.asyncio
async def test_submit_gate_on_medium_tier_raises_confirmation(foundation, monkeypatch) -> None:
    """Codex spec review HIGH-3: queue latency in research §6.5.4 is
    field-count-driven, not bytes-driven. A tiny-area / many-field
    request stays sub-threshold on bytes but still queues medium/heavy
    on the server. Tier-based gate must trigger here."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.workflow.confirmation import ConfirmationRequired

    # 1 var × 1y × 12mo × 31d = 372 fields → tier=medium per estimator
    # ``_TIER_LIGHT_MAX_FIELDS=100``, well below 1 GB at default 2 MB/field
    # (372 × 2_000_000 = ~744 MB, under 1 GB). With area restriction the
    # bytes drop further but the tier stays medium.
    medium_params = {
        "dataset_id": "unknown-dataset-id",  # falls back to default bytes/field
        "inputs": {
            "variable": ["t"],
            "year": ["2024"],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "area": [50.0, 0.0, 49.0, 1.0],  # tiny 1°x1° patch
        },
    }

    fake_class, _ = _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote())
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    with pytest.raises(ConfirmationRequired):
        await backend.submit(medium_params)
    fake_class.assert_not_called()


@pytest.mark.asyncio
async def test_submit_gate_with_confirmed_bypasses(foundation, monkeypatch) -> None:
    """``__options.confirmed=true`` bypasses both the bytes and the tier
    gate."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    big_params = {
        "dataset_id": "reanalysis-era5-pressure-levels",
        "inputs": {
            "product_type": ["reanalysis"],
            "variable": ["temperature"],
            "year": [str(y) for y in range(1990, 2024)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "pressure_level": ["500"],
            "data_format": "grib",
        },
        "__options": {"confirmed": True},
    }

    _, sdk = _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote("req-big"))
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    out = await backend.submit(big_params)
    assert out["request_id"] == "req-big"
    assert sdk.retrieve.call_count == 1


@pytest.mark.asyncio
async def test_submit_gate_below_threshold_no_confirmation_needed(foundation, monkeypatch) -> None:
    """A small (light tier, sub-GB bytes) request flows through without
    a confirmation prompt, even though estimate is always approximate."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    # 1 field, default bytes/field => 2 MB. Light tier.
    tiny_params = {
        "dataset_id": "reanalysis-era5-single-levels",
        "inputs": {
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature"],
            "year": ["2024"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00"],
            "data_format": "grib",
        },
    }

    _, sdk = _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote("req-tiny"))
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    out = await backend.submit(tiny_params)
    assert out["request_id"] == "req-tiny"
    assert sdk.retrieve.call_count == 1


# ---------------------------------------------------------------------------
# T-CDS-EST2-002: pre-flight cost-limit rejection + nullable-size gate
# ---------------------------------------------------------------------------


def _patch_costing(monkeypatch, result) -> None:
    async def _fake(*_a, **_k):
        return result

    monkeypatch.setattr("copernicus_mcp.backends.cds.backend.fetch_costing", _fake)


def _whole_file_params() -> dict[str, object]:
    """A whole-file product request — count_fields == 1."""
    return {
        "dataset_id": "satellite-carbon-dioxide",
        "inputs": {
            "processing_level": "level_3",
            "variable": "xco2",
            "sensor_and_algorithm": "merged_obs4mips",
            "version": "4_5",
        },
    }


@pytest.mark.asyncio
async def test_submit_cost_over_limit_raises_validation_error_pre_sdk(
    foundation, monkeypatch
) -> None:
    """Field case D: cost 1827 > limit 400 → structured ValidationError with a
    year-split hint, raised before any cdsapi call."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.backends.cds.costing import CostingResult
    from copernicus_mcp.errors.classes import ValidationError as CmcpValidationError

    _patch_costing(monkeypatch, CostingResult(units=1827.0, limit=400.0))
    fake_class, _ = _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote())
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    with pytest.raises(CmcpValidationError) as exc:
        await backend.submit(_good_params())

    fake_class.assert_not_called()
    ctx = exc.value.error_record.context
    assert ctx["cost_units"] == 1827.0
    assert ctx["cost_limit"] == 400.0
    assert ctx["suggested_split"]["chunks"] == 5  # ceil(1827/400)
    assert ctx["suggested_split"]["dimension"] == "year"


@pytest.mark.asyncio
async def test_submit_over_limit_proposal_precedes_size_confirmation(
    foundation, monkeypatch
) -> None:
    """T-CDS-CHUNK-002 (model B): an over-limit *splittable* request that would
    also trip the size gate yields the chunk PROPOSAL, not the generic size
    ConfirmationRequired — the cost-limit branch still runs first (one
    actionable round-trip)."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.backends.cds.costing import CostingResult
    from copernicus_mcp.workflow.confirmation import ConfirmationRequired

    _patch_costing(monkeypatch, CostingResult(units=8000.0, limit=400.0))
    _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote())
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    big_params = {
        "dataset_id": "reanalysis-era5-pressure-levels",
        "inputs": {
            "product_type": ["reanalysis"],
            "variable": ["temperature"],
            "year": [str(y) for y in range(1990, 2024)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "pressure_level": ["500", "850", "1000"],
        },
    }
    with pytest.raises(ConfirmationRequired) as exc:
        await backend.submit(big_params)
    assert exc.value.payload["reason"] == "cost_limit_requires_chunking"


@pytest.mark.asyncio
async def test_submit_unknown_size_raises_confirmation_when_flag_on(
    foundation, monkeypatch
) -> None:
    """With cds_confirm_on_unknown_size=True (opt-in; v2 default is False), an
    unknown-size request raises confirmation 'estimated_size_unknown', no SDK call."""
    import dataclasses

    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.backends.cds.costing import CostingResult
    from copernicus_mcp.workflow.confirmation import ConfirmationRequired

    budget_on = foundation.config.budget.model_copy(
        update={"cds_confirm_on_unknown_size": True}
    )
    config_on = foundation.config.model_copy(update={"budget": budget_on})
    found_on = dataclasses.replace(foundation, config=config_on)

    _patch_costing(monkeypatch, CostingResult(units=1.0, limit=10000.0))
    fake_class, _ = _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote())
    backend = CdsBackend(foundation=found_on, credentials=_fake_creds())

    with pytest.raises(ConfirmationRequired) as exc:
        await backend.submit(_whole_file_params())
    fake_class.assert_not_called()
    assert exc.value.payload["reason"] == "estimated_size_unknown"


@pytest.mark.asyncio
async def test_submit_unknown_size_flag_off_proceeds(foundation, monkeypatch) -> None:
    """With cds_confirm_on_unknown_size=False, an unknown-size request submits
    without a confirmation prompt."""
    import dataclasses

    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.backends.cds.costing import CostingResult

    budget_off = foundation.config.budget.model_copy(
        update={"cds_confirm_on_unknown_size": False}
    )
    config_off = foundation.config.model_copy(update={"budget": budget_off})
    found_off = dataclasses.replace(foundation, config=config_off)

    _patch_costing(monkeypatch, CostingResult(units=1.0, limit=10000.0))
    _, sdk = _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote("req-unknown"))
    backend = CdsBackend(foundation=found_off, credentials=_fake_creds())

    out = await backend.submit(_whole_file_params())
    assert out["request_id"] == "req-unknown"
    assert sdk.retrieve.call_count == 1


@pytest.mark.asyncio
async def test_submit_costing_none_never_rejects_on_cost_limit(
    foundation, monkeypatch
) -> None:
    """Costing unreachable (None) → no cost-limit rejection; submit proceeds
    (the autouse default already returns None)."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    _, sdk = _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote("req-nocost"))
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    out = await backend.submit(_good_params())
    assert out["request_id"] == "req-nocost"
    assert sdk.retrieve.call_count == 1


# ---------------------------------------------------------------------------
# T-CDS-CHUNK-002: model-B auto-chunking (MCP proposes, agent disposes)
# ---------------------------------------------------------------------------


def _splittable_params() -> dict[str, Any]:
    """A 5-year daily request (field case D shape) — splittable along year."""
    return {
        "dataset_id": "derived-era5-single-levels-daily-statistics",
        "inputs": {
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature"],
            "year": ["2020", "2021", "2022", "2023", "2024"],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
        },
    }


def _patch_costing_by_shape(monkeypatch, *, per_year: float, limit: float) -> None:
    """``fetch_costing`` stub whose cost scales with the year count, so a
    multi-year request exceeds ``limit`` while each single-year child fits
    under it — the realistic chunk-then-validate path."""
    from copernicus_mcp.backends.cds.costing import CostingResult

    async def _fake(dataset_id, inputs, **_kwargs):
        years = inputs.get("year")
        n_years = len(years) if isinstance(years, list) else 1
        return CostingResult(units=per_year * n_years, limit=limit)

    monkeypatch.setattr("copernicus_mcp.backends.cds.backend.fetch_costing", _fake)


def _patch_cdsapi_seq(monkeypatch):
    """Like ``_patch_cdsapi`` but ``retrieve`` returns a fresh Remote with a
    unique request_id each call — chunk children must not collide on the PK."""
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
    poll_remote = MagicMock()
    poll_remote.json = {"status": "running"}
    inner.get_remote = MagicMock(return_value=poll_remote)
    inner.delete = MagicMock(return_value={"deleted": True})
    instance.client = inner
    fake_class = MagicMock(return_value=instance)
    fake_module.Client = fake_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cdsapi", fake_module)
    return fake_class, instance


@pytest.mark.asyncio
async def test_submit_over_limit_no_chunk_by_raises_proposal(
    foundation, monkeypatch
) -> None:
    """Over-limit splittable request with no ``chunk_by`` → a chunk PROPOSAL
    (ConfirmationRequired) carrying per-granularity viability, no cdsapi job."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.workflow.confirmation import ConfirmationRequired

    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    fake_class, _ = _patch_cdsapi_seq(monkeypatch)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    with pytest.raises(ConfirmationRequired) as exc:
        await backend.submit(_splittable_params())

    payload = exc.value.payload
    assert payload["reason"] == "cost_limit_requires_chunking"
    assert payload["chunked"] is True
    ctx = payload["context"]
    assert round(ctx["cost_units"]) == 1827
    assert ctx["cost_limit"] == 400.0
    gran = payload["chunking"]["granularities"]
    assert gran["year"]["chunks"] == 5  # 5 whole-year chunks
    assert gran["year"]["fits"] is True  # ~365/yr ≤ 400
    assert payload["chunking"]["suggested_granularity"] == "year"
    fake_class.assert_not_called()


@pytest.mark.asyncio
async def test_submit_over_limit_chunk_by_year_submits_all_children(
    foundation, monkeypatch
) -> None:
    """``chunk_by=year`` → parent row + first wave submitted; all 5 chunks fit
    inside the default ``cds_chunk_max_inflight=5``, so every child id lands
    in the persisted plan on submit (T-CDS-RESIL-002 pacing is a no-op here)."""
    import json as _json

    from copernicus_mcp.backends.cds.backend import CdsBackend

    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, {})
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    out = await backend.submit(params)

    assert out["chunked"] is True
    assert out["chunk_count"] == 5
    assert out["status"] == "queued"
    parent_id = out["request_id"]
    assert sdk.retrieve.call_count == 5  # all chunks submitted at once

    parent = await foundation.persistence.fetch_workflow(parent_id)
    assert parent is not None
    assert parent["parent_request_id"] is None
    plan = _json.loads(parent["chunk_plan_json"])
    assert plan["granularity"] == "year"
    assert plan["stopped"] is False
    assert "max_inflight" not in plan
    assert len(plan["chunks"]) == 5
    submitted = [c for c in plan["chunks"] if c["child_request_id"] is not None]
    assert len(submitted) == 5  # all
    assert all(round(c["units"]) == 365 for c in plan["chunks"])

    children = await foundation.persistence.list_child_workflows(parent_id)
    assert len(children) == 5
    assert all(c["parent_request_id"] == parent_id for c in children)


@pytest.mark.asyncio
async def test_submit_resubmit_inflight_parent_dedupes_no_refanout(
    foundation, monkeypatch
) -> None:
    """An identical re-submit while the chunked parent is still queued dedupes
    to the SAME parent id and does NOT fan out a second set of children."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, {})
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    first = await backend.submit(params)
    assert sdk.retrieve.call_count == 5

    again = await backend.submit(params)
    assert again["request_id"] == first["request_id"]
    assert again["chunked"] is True
    assert again["chunk_count"] == 5
    assert again["status"] == "queued"
    assert sdk.retrieve.call_count == 5  # no re-fan-out
    children = await foundation.persistence.list_child_workflows(first["request_id"])
    assert len(children) == 5


@pytest.mark.asyncio
async def test_submit_over_limit_confirmed_defaults_to_year(
    foundation, monkeypatch
) -> None:
    """``confirmed=true`` with no ``chunk_by`` accepts the suggested granularity,
    so a naive agent that just confirms still succeeds."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, {})
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    params = _splittable_params()
    params["__options"] = {"confirmed": True}
    out = await backend.submit(params)
    assert out["chunked"] is True
    assert out["chunk_count"] == 5
    assert sdk.retrieve.call_count == 5


@pytest.mark.asyncio
async def test_submit_over_limit_auto_chunk_disabled_keeps_est2_error(
    foundation, monkeypatch
) -> None:
    """Config ``cds_auto_chunk_enabled=False`` → unchanged EST2 manual-split
    ValidationError, even when the agent supplies a ``chunk_by``."""
    import dataclasses

    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors.classes import ValidationError as CmcpValidationError

    budget_off = foundation.config.budget.model_copy(
        update={"cds_auto_chunk_enabled": False}
    )
    config_off = foundation.config.model_copy(update={"budget": budget_off})
    found_off = dataclasses.replace(foundation, config=config_off)

    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    fake_class, _ = _patch_cdsapi_seq(monkeypatch)
    backend = CdsBackend(foundation=found_off, credentials=_fake_creds())

    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    with pytest.raises(CmcpValidationError) as exc:
        await backend.submit(params)
    assert exc.value.error_record.context["cost_limit"] == 400.0
    fake_class.assert_not_called()


@pytest.mark.asyncio
async def test_submit_over_limit_opt_out_keeps_est2_error(
    foundation, monkeypatch
) -> None:
    """Per-request ``__options.auto_chunk=false`` → unchanged EST2 error."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors.classes import ValidationError as CmcpValidationError

    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_seq(monkeypatch)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    params = _splittable_params()
    params["__options"] = {"auto_chunk": False}
    with pytest.raises(CmcpValidationError):
        await backend.submit(params)


@pytest.mark.asyncio
async def test_submit_chunk_plan_too_many_chunks_maps_to_validation_error(
    foundation, monkeypatch
) -> None:
    """A plan exceeding ``cds_auto_chunk_max_chunks`` → ChunkPlanError mapped to
    a ValidationError advising a narrower request, no cdsapi job."""
    import dataclasses

    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors.classes import ValidationError as CmcpValidationError

    budget_cap = foundation.config.budget.model_copy(
        update={"cds_auto_chunk_max_chunks": 3}
    )
    config_cap = foundation.config.model_copy(update={"budget": budget_cap})
    found_cap = dataclasses.replace(foundation, config=config_cap)

    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    fake_class, _ = _patch_cdsapi_seq(monkeypatch)
    backend = CdsBackend(foundation=found_cap, credentials=_fake_creds())

    params = _splittable_params()  # year split → 5 chunks > cap 3
    params["__options"] = {"chunk_by": "year"}
    with pytest.raises(CmcpValidationError) as exc:
        await backend.submit(params)
    assert exc.value.error_record.context.get("chunk_plan_reason") == "too_many_chunks"
    fake_class.assert_not_called()


def _patch_cdsapi_retrieve_fails_first(monkeypatch, *, fail_times: int = 1):
    """``retrieve`` raises for the first ``fail_times`` calls, then returns a
    fresh unique Remote — drives the first-wave-failure cleanup path."""
    import sys
    import types

    fake_module = types.ModuleType("cdsapi")
    instance = MagicMock()
    counter = {"n": 0}

    def _retrieve(name, request, target):
        counter["n"] += 1
        if counter["n"] <= fail_times:
            raise RuntimeError("CDS retrieve blew up")
        return _fake_remote(f"child-{counter['n']}")

    instance.retrieve = MagicMock(side_effect=_retrieve)
    inner = MagicMock()
    poll_remote = MagicMock()
    poll_remote.json = {"status": "running"}
    inner.get_remote = MagicMock(return_value=poll_remote)
    inner.delete = MagicMock(return_value={"deleted": True})
    instance.client = inner
    fake_class = MagicMock(return_value=instance)
    fake_module.Client = fake_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cdsapi", fake_module)
    return fake_class, instance


def _parent_cache_key(foundation, params: dict[str, Any]) -> str:
    from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest

    clean = {k: v for k, v in params.items() if k != "__options"}
    req = CdsRetrieveRequest.model_validate(clean)
    return foundation.data_model.cache_key_for_cds_retrieve(req)


@pytest.mark.asyncio
async def test_submit_chunk_first_wave_failure_marks_parent_failed_no_poison(
    foundation, monkeypatch
) -> None:
    """Codex Tier-A HIGH: a first-wave child failure must NOT leave the parent
    stuck ``queued`` (which would poison the cache-key dedupe forever). The
    parent goes ``failed`` with the plan stopped; a later retry (SDK healthy)
    creates a fresh parent rather than re-using the dead one."""
    import json as _json

    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import CopernicusMcpError

    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_retrieve_fails_first(monkeypatch, fail_times=1)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    with pytest.raises(CopernicusMcpError):
        await backend.submit(params)

    cache_key = _parent_cache_key(foundation, params)
    parent = await foundation.persistence.lookup_workflow_by_cache_key(cache_key)
    assert parent is not None
    assert parent["status"] == "failed"  # NOT queued — dedupe can't return it
    plan = _json.loads(parent["chunk_plan_json"])
    assert plan["stopped"] is True

    # Retry now that the SDK is healthy → a brand-new parent, not the dead one.
    again = await backend.submit(params)
    assert again["chunked"] is True
    assert again["request_id"] != parent["request_id"]


@pytest.mark.asyncio
async def test_abort_marks_parent_failed_even_if_child_cancel_client_build_fails(
    foundation, monkeypatch
) -> None:
    """Round-2 HIGH (both reviewers): the parent-failed write must NOT be gated
    behind best-effort remote child cleanup. Child[0] submits, child[1] fails →
    abort; the abort's child-cancel client build then raises — the parent must
    STILL end ``failed`` (poison cured), not stuck ``queued``."""
    import sys
    import types

    from copernicus_mcp.backends.cds import backend as backend_mod
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import CopernicusMcpError

    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)

    # retrieve: child[0] OK, child[1] raises → abort with submitted=[child0].
    fake_module = types.ModuleType("cdsapi")
    instance = MagicMock()
    rcounter = {"n": 0}

    def _retrieve(name, request, target):
        rcounter["n"] += 1
        if rcounter["n"] == 2:
            raise RuntimeError("second child retrieve failed")
        return _fake_remote(f"child-{rcounter['n']}")

    instance.retrieve = MagicMock(side_effect=_retrieve)
    inner = MagicMock()
    inner.delete = MagicMock(return_value={"deleted": True})
    poll_remote = MagicMock()
    poll_remote.json = {"status": "running"}
    inner.get_remote = MagicMock(return_value=poll_remote)
    instance.client = inner
    fake_module.Client = MagicMock(return_value=instance)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cdsapi", fake_module)

    # _make_cdsapi_client succeeds for the two child submits, raises on the 3rd
    # construction (the abort's child-cancel client).
    real_make = backend_mod._make_cdsapi_client
    mcounter = {"n": 0}

    def _make_or_fail(adapter, *, dataset_id=None):
        mcounter["n"] += 1
        if mcounter["n"] >= 3:
            raise RuntimeError("client build failed during abort")
        return real_make(adapter, dataset_id=dataset_id)

    monkeypatch.setattr(backend_mod, "_make_cdsapi_client", _make_or_fail)

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    with pytest.raises(CopernicusMcpError):
        await backend.submit(params)

    cache_key = _parent_cache_key(foundation, params)
    parent = await foundation.persistence.lookup_workflow_by_cache_key(cache_key)
    assert parent is not None
    assert parent["status"] == "failed"  # cured despite the abort client-build failure


def _patch_cdsapi_children(
    monkeypatch, status_by_request, *, download_bytes=b"GRIB-chunk", bytes_for=None
):
    """cdsapi mock for the chunked-parent lifecycle: ``retrieve`` assigns
    sequential child ids; ``get_remote(id).json`` reflects ``status_by_request``
    (a mutable dict the test mutates between polls); ``download_results`` writes a
    file per child (``bytes_for(request_id)`` overrides ``download_bytes`` for the
    heterogeneous-format case) so ``_finalise_successful`` succeeds."""
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
        rem.json = {"status": status_by_request.get(request_id, "running"), "jobID": request_id}
        return rem

    inner.get_remote = MagicMock(side_effect=_get_remote)
    inner.delete = MagicMock(return_value={"deleted": True})

    def _download_results(request_id, target):
        payload = bytes_for(request_id) if bytes_for is not None else download_bytes
        Path(target).write_bytes(payload)
        return target

    inner.download_results = MagicMock(side_effect=_download_results)
    instance.client = inner
    fake_class = MagicMock(return_value=instance)
    fake_module.Client = fake_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cdsapi", fake_module)
    return fake_class, instance


async def _drive_parent_to_successful(backend, parent_id, status, n=5):
    """Mark all ``n`` children successful and poll until the parent completes."""
    for i in range(1, n + 1):
        status[f"child-{i}"] = "successful"
    st = None
    for _ in range(n + 2):
        st = await backend.check_status(parent_id)
        if st["status"] == "successful":
            break
    return st


@pytest.mark.asyncio
async def test_fetch_result_parent_multifile_ordered(
    foundation, monkeypatch, tmp_path: Path
) -> None:
    """A successful chunked parent returns an ordered descriptor SET (one file
    per chunk), with span + a merge hint — the MCP never stitches."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    out = await backend.submit(params)
    parent_id = out["request_id"]
    await _drive_parent_to_successful(backend, parent_id, status)

    res = await backend.fetch_result(parent_id, target=tmp_path / "ignored")
    assert res["status"] == "successful"
    assert res["chunked"] is True
    assert res["chunk_count"] == 5
    files = res["result"]["files"]
    assert [f["chunk_index"] for f in files] == [0, 1, 2, 3, 4]  # ordered
    assert all(f["filepath"] for f in files)
    assert all("span" in f for f in files)
    assert res["result"]["heterogeneous_formats"] is False
    assert "merge_hint" in res["result"]


@pytest.mark.asyncio
async def test_fetch_result_parent_flags_heterogeneous_formats(
    foundation, monkeypatch, tmp_path: Path
) -> None:
    """Per the user requirement: when chunks come back in different formats (e.g.
    CDS zips one), the MCP FLAGS it in metadata — it never normalizes."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    # child-1 comes back as a ZIP, the rest as GRIB.
    def _bytes_for(rid: str) -> bytes:
        return b"PK\x03\x04zip" if rid == "child-1" else b"GRIB-chunk"

    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(monkeypatch, status, bytes_for=_bytes_for)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    out = await backend.submit(params)
    parent_id = out["request_id"]
    await _drive_parent_to_successful(backend, parent_id, status)

    res = await backend.fetch_result(parent_id, target=tmp_path / "ignored")
    assert res["result"]["heterogeneous_formats"] is True
    assert len(res["result"]["formats"]) >= 2


@pytest.mark.asyncio
async def test_fetch_result_parent_evicted_child_raises_cache_error(
    foundation, monkeypatch, tmp_path: Path
) -> None:
    """An evicted child file → CacheError(cache_eviction) naming the chunk index,
    not a partial/garbled set (decision 11b)."""
    from copernicus_mcp.backends.cds.backend import CdsBackend, _cache_storage_key
    from copernicus_mcp.errors import CacheError

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    out = await backend.submit(params)
    parent_id = out["request_id"]
    await _drive_parent_to_successful(backend, parent_id, status)

    # Evict child-1's file by making lookup_file return None for its key.
    children = await foundation.persistence.list_child_workflows(parent_id)
    evicted_key = _cache_storage_key(children[0]["cache_key"])
    real_lookup = foundation.cache.lookup_file

    async def _evicting_lookup(key):
        return None if key == evicted_key else await real_lookup(key)

    monkeypatch.setattr(foundation.cache, "lookup_file", _evicting_lookup)
    with pytest.raises(CacheError) as exc:
        await backend.fetch_result(parent_id, target=tmp_path / "ignored")
    assert exc.value.error_record.error_subclass == "cache_eviction"


@pytest.mark.asyncio
async def test_fetch_result_parent_not_ready_lists_partial_files(
    foundation, monkeypatch, tmp_path: Path
) -> None:
    """A non-successful parent → result_not_ready error carrying partial_files
    (descriptors of any chunk already done, so work is never thrown away)."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import BackendError

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    out = await backend.submit(params)
    parent_id = out["request_id"]

    # Finish only child-1, leave the parent running.
    status["child-1"] = "successful"
    st = await backend.check_status(parent_id)
    assert st["status"] == "running"

    with pytest.raises(BackendError) as exc:
        await backend.fetch_result(parent_id, target=tmp_path / "ignored")
    assert exc.value.error_record.error_subclass == "result_not_ready"
    partial = exc.value.error_record.context.get("partial_files")
    assert partial and len(partial) >= 1


@pytest.mark.asyncio
async def test_check_status_parent_polls_children_not_parent_id(
    foundation, monkeypatch
) -> None:
    """check_status on a chunked parent polls its SUBMITTED children but never
    pokes CDS with the synthetic parent id; the response carries aggregate counts."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    out = await backend.submit(params)
    parent_id = out["request_id"]

    sdk.client.get_remote.reset_mock()
    st = await backend.check_status(parent_id)
    assert st["chunked"] is True
    assert st["status"] == "running"
    assert st["chunks"]["total"] == 5
    assert st["chunks"]["running"] == 5  # v2: all submitted at once
    assert st["chunks"]["queued"] == 0
    polled = {c.args[0] for c in sdk.client.get_remote.call_args_list}
    assert polled == {f"child-{i}" for i in range(1, 6)}
    assert parent_id not in polled


@pytest.mark.asyncio
async def test_check_status_parent_advances_to_successful(
    foundation, monkeypatch
) -> None:
    """All chunks submit at once; once every child finalises, the parent reaches
    successful (read-repaired)."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    out = await backend.submit(params)
    parent_id = out["request_id"]
    assert sdk.retrieve.call_count == 5  # all chunks submitted at submit time

    for i in range(1, 6):
        status[f"child-{i}"] = "successful"
    final = await backend.check_status(parent_id)
    assert final["status"] == "successful"
    assert final["chunks"]["successful"] == 5
    assert sdk.retrieve.call_count == 5  # no extra submits
    row = await foundation.persistence.fetch_workflow(parent_id)
    assert row["status"] == "successful"  # read-repaired
    children = await foundation.persistence.list_child_workflows(parent_id)
    assert len(children) == 5
    assert all(c["status"] == "successful" for c in children)


@pytest.mark.asyncio
async def test_check_status_successful_parent_returns_files_directly(
    foundation, monkeypatch
) -> None:
    """Field feedback, part 2: a successful chunked parent's check_status returns the full
    multi-file descriptor set (paths + merge_hint) directly — the agent does NOT
    need a second download call, since the files are already cached."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    out = await backend.submit(params)
    parent_id = out["request_id"]

    st = await _drive_parent_to_successful(backend, parent_id, status)
    assert st["status"] == "successful"
    files = st["result"]["files"]
    assert len(files) == 5
    assert [f["chunk_index"] for f in files] == [0, 1, 2, 3, 4]
    assert all(f["filepath"] for f in files)
    assert "merge_hint" in st["result"]
    # A re-check of the terminal parent also returns the files (no re-poll).
    again = await backend.check_status(parent_id)
    assert len(again["result"]["files"]) == 5


@pytest.mark.asyncio
async def test_check_status_parent_child_failure_fails_parent(
    foundation, monkeypatch
) -> None:
    """A child that fails terminally fails the parent (decision 4)."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    out = await backend.submit(params)
    parent_id = out["request_id"]

    status["child-1"] = "failed"
    st = await backend.check_status(parent_id)
    assert st["status"] == "failed"
    assert st["chunks"]["failed"] >= 1
    row = await foundation.persistence.fetch_workflow(parent_id)
    assert row["status"] == "failed"


@pytest.mark.asyncio
async def test_check_status_parent_failed_child_cancels_siblings(
    foundation, monkeypatch
) -> None:
    """Codex CHUNK-003 HIGH: when a child fails, the in-flight siblings are
    best-effort cancelled before the parent goes terminal (decision 4) — no
    orphaned CDS job under a dead parent."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    out = await backend.submit(params)
    parent_id = out["request_id"]

    status["child-1"] = "failed"  # child-2 still running
    st = await backend.check_status(parent_id)
    assert st["status"] == "failed"
    children = await foundation.persistence.list_child_workflows(parent_id)
    by_id = {c["request_id"]: c["status"] for c in children}
    assert by_id["child-1"] == "failed"
    assert by_id["child-2"] == "cancelled"  # in-flight sibling cancelled
    assert sdk.client.delete.call_count >= 1


@pytest.mark.asyncio
async def test_failed_parent_marks_siblings_cancelled_even_if_remote_delete_fails(
    foundation, monkeypatch
) -> None:
    """Round-2: the LOCAL sibling-cancel runs first and unconditionally, so a
    failing remote delete (or unbuildable client) still leaves the aggregate
    consistent — the sibling row is cancelled regardless."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)
    sdk.client.delete.side_effect = RuntimeError("remote delete down")
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    out = await backend.submit(params)
    parent_id = out["request_id"]

    status["child-1"] = "failed"
    st = await backend.check_status(parent_id)
    assert st["status"] == "failed"
    children = await foundation.persistence.list_child_workflows(parent_id)
    by_id = {c["request_id"]: c["status"] for c in children}
    assert by_id["child-2"] == "cancelled"  # local-marked despite remote failure


@pytest.mark.asyncio
async def test_check_status_parent_dismissed_child_fails(
    foundation, monkeypatch
) -> None:
    """Codex CHUNK-003 HIGH: a child CDS dismisses (→ canonical cancelled) without
    a parent cancel fails the parent rather than running forever."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    out = await backend.submit(params)
    parent_id = out["request_id"]

    status["child-1"] = "dismissed"  # CDS dismissed → canonical cancelled
    st = await backend.check_status(parent_id)
    assert st["status"] == "failed"


@pytest.mark.asyncio
async def test_submit_chunk_child_adopts_existing_no_duplicate(
    foundation, monkeypatch
) -> None:
    """Codex CHUNK-003 HIGH: a chunk child durably submitted but lost from the
    plan (interrupted persist) is ADOPTED by cache_key on the next attempt — no
    duplicate CDS job — but only for the SAME parent."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    _, sdk = _patch_cdsapi_children(monkeypatch, {})
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    base = _splittable_params()["inputs"]
    kwargs = dict(
        dataset_id="derived-era5-single-levels-daily-statistics",
        parent_inputs=base,
        overrides={"year": ["2020"]},
        units=365.0,
        cost_limit=400.0,
        parent_id="parent-x",
    )
    id1 = await backend._submit_chunk_child(**kwargs)
    id2 = await backend._submit_chunk_child(**kwargs)  # same chunk + parent
    assert id2 == id1  # adopted, not duplicated
    assert sdk.retrieve.call_count == 1

    # A different parent with the same chunk does NOT adopt — separate request.
    id3 = await backend._submit_chunk_child(**{**kwargs, "parent_id": "parent-y"})
    assert id3 != id1
    assert sdk.retrieve.call_count == 2


@pytest.mark.asyncio
async def test_submit_chunk_child_adopts_own_orphan_not_shadowed(
    foundation, monkeypatch
) -> None:
    """Codex CHUNK-003 r5 HIGH: a same-chunk child of ANOTHER parent (a newer row
    for the same cache_key) must NOT shadow this parent's own orphan — the adopt
    lookup is scoped to the parent's own children, not the global newest."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    _, sdk = _patch_cdsapi_children(monkeypatch, {})
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    base = _splittable_params()["inputs"]
    kwargs = dict(
        dataset_id="derived-era5-single-levels-daily-statistics",
        parent_inputs=base,
        overrides={"year": ["2020"]},
        units=365.0,
        cost_limit=400.0,
    )
    id_x = await backend._submit_chunk_child(**kwargs, parent_id="parent-x")
    # parent-y then submits the SAME chunk — a NEWER row for the same cache_key.
    id_y = await backend._submit_chunk_child(**kwargs, parent_id="parent-y")
    assert id_y != id_x
    assert sdk.retrieve.call_count == 2

    # parent-x re-attempts (its plan lost the id) → adopts ITS OWN child, not the
    # newer parent-y child that shares the cache_key.
    id_x2 = await backend._submit_chunk_child(**kwargs, parent_id="parent-x")
    assert id_x2 == id_x
    assert sdk.retrieve.call_count == 2  # no duplicate


@pytest.mark.asyncio
async def test_check_status_terminal_failed_parent_recovers_orphan_children(
    foundation, monkeypatch
) -> None:
    """Round-3 (codex): if a previous advance was interrupted mid-cleanup leaving
    a terminal parent with stray in-flight children, the next poll RE-CLEANS them
    (poll-driven recovery), rather than leaking them forever behind the terminal
    short-circuit."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    out = await backend.submit(params)
    parent_id = out["request_id"]

    # Simulate an interrupted cleanup: parent terminal failed, children still running.
    await foundation.persistence.update_workflow_status(parent_id, "failed")
    st = await backend.check_status(parent_id)
    assert st["status"] == "failed"
    children = await foundation.persistence.list_child_workflows(parent_id)
    assert all(c["status"] == "cancelled" for c in children)  # orphans recovered


@pytest.mark.asyncio
async def test_submit_over_limit_dirty_calendar_token_not_chunked(
    foundation, monkeypatch
) -> None:
    """Codex Tier-A MEDIUM (invariant 2): a credential-shaped value on a split
    axis must not engage the chunk path — it falls back to the EST2 manual-split
    ValidationError, so the value never reaches a persisted chunk plan."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors.classes import ValidationError as CmcpValidationError

    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    fake_class, _ = _patch_cdsapi_seq(monkeypatch)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    params = _splittable_params()
    params["inputs"]["year"] = [
        "2020",
        "abcdef01-2345-6789-abcd-ef0123456789",  # credential-shaped
        "2022",
        "2023",
        "2024",
    ]
    params["__options"] = {"chunk_by": "year"}
    with pytest.raises(CmcpValidationError) as exc:
        await backend.submit(params)
    assert exc.value.error_record.context.get("cost_limit") == 400.0
    fake_class.assert_not_called()


@pytest.mark.asyncio
async def test_submit_terminal_successful_parent_rebuilds_no_refanout(
    foundation, monkeypatch, tmp_path: Path
) -> None:
    """Decision 11a: re-submitting an already-complete chunked parent rebuilds
    the multi-file result from the child cache — same parent id, zero new CDS
    jobs (no duplicate fan-out)."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    out = await backend.submit(params)
    parent_id = out["request_id"]
    await _drive_parent_to_successful(backend, parent_id, status)
    retrieve_before = sdk.retrieve.call_count  # 5 children

    again = await backend.submit(_splittable_params())
    assert again["status"] == "successful"
    assert again["chunked"] is True
    assert again["request_id"] == parent_id
    assert len(again["result"]["files"]) == 5
    assert sdk.retrieve.call_count == retrieve_before  # no new fan-out


@pytest.mark.asyncio
async def test_cancel_parent_cascades(foundation, monkeypatch) -> None:
    """cancel(parent) stops the plan, best-effort cancels in-flight children, and
    marks the parent cancelled; a later poll never submits more (decision 8)."""
    import json as _json

    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    out = await backend.submit(params)
    parent_id = out["request_id"]

    res = await backend.cancel(parent_id)
    assert res["cancelled"] is True
    assert res["status"] == "cancelled"

    row = await foundation.persistence.fetch_workflow(parent_id)
    assert row["status"] == "cancelled"
    plan = _json.loads(row["chunk_plan_json"])
    assert plan["stopped"] is True
    children = await foundation.persistence.list_child_workflows(parent_id)
    assert len(children) == 5  # all submitted at once (v2)
    assert all(c["status"] == "cancelled" for c in children)
    assert sdk.client.delete.call_count >= 5  # remote delete attempted per child

    submits_after_cancel = sdk.retrieve.call_count
    st = await backend.check_status(parent_id)
    assert st["status"] == "cancelled"
    assert sdk.retrieve.call_count == submits_after_cancel  # no submits after cancel


@pytest.mark.asyncio
async def test_cancel_parent_preserves_successful_child(foundation, monkeypatch) -> None:
    """A child already successful when the parent is cancelled keeps its terminal
    state + cached file; only the in-flight children are cancelled (decision 8)."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    out = await backend.submit(params)
    parent_id = out["request_id"]

    status["child-1"] = "successful"
    await backend.check_status(parent_id)  # finalise child-1, refill child-3

    res = await backend.cancel(parent_id)
    assert res["status"] == "cancelled"
    children = await foundation.persistence.list_child_workflows(parent_id)
    by_id = {c["request_id"]: c["status"] for c in children}
    assert by_id["child-1"] == "successful"  # preserved
    assert all(s == "cancelled" for cid, s in by_id.items() if cid != "child-1")


@pytest.mark.asyncio
async def test_cancel_pops_inflight_cost_when_it_wins(foundation, monkeypatch) -> None:
    """A cancel that actually transitions the row drops the pre-flight cost
    (no download will follow, so no observation needs it)."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote())
    await _seed_workflow_row(foundation, request_id="rid-cancel", status="running")
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    backend._inflight_costing["rid-cancel"] = {"units": 24.0, "limit": 400.0}

    out = await backend.cancel("rid-cancel")
    assert out["status"] == "cancelled"
    assert "rid-cancel" not in backend._inflight_costing


@pytest.mark.asyncio
async def test_cancel_preserves_inflight_cost_when_it_loses_race(
    foundation, monkeypatch
) -> None:
    """If cancel loses the terminal-state race to a successful finalize, it must
    NOT strip the cost — the in-flight finalizer still needs it to record the
    calibration observation (codex Tier-A MEDIUM)."""
    from unittest.mock import AsyncMock

    from copernicus_mcp.backends.cds.backend import CdsBackend

    _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote())
    await _seed_workflow_row(foundation, request_id="rid-lose", status="running")
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    backend._inflight_costing["rid-lose"] = {"units": 24.0, "limit": 400.0}

    # Simulate the finalizer winning: the conditional update does not commit.
    monkeypatch.setattr(
        foundation.persistence,
        "update_workflow_status_if_pending",
        AsyncMock(return_value=False),
    )
    await backend.cancel("rid-lose")
    assert backend._inflight_costing["rid-lose"] == {"units": 24.0, "limit": 400.0}


@pytest.mark.asyncio
async def test_successful_download_writes_size_observation(foundation, monkeypatch) -> None:
    """T-CDS-EST2-003: submit→poll→success captures one size_observations
    row with the downloaded byte size and the pre-flight cost units."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.backends.cds.costing import CostingResult

    _patch_costing(monkeypatch, CostingResult(units=24.0, limit=121000.0))
    _patch_cdsapi(
        monkeypatch,
        retrieve_returns=_fake_remote("req-obs"),
        get_remote_json={"status": "successful"},
    )
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await backend.submit(_good_params())
    out = await backend.check_status("req-obs")

    assert out["status"] == "successful"
    rows = await foundation.persistence.list_size_observations(
        "cds", "reanalysis-era5-single-levels", None
    )
    assert len(rows) == 1
    assert rows[0]["cost_units"] == 24.0
    assert rows[0]["size_bytes"] == len(_CDS_DEFAULT_DOWNLOAD_BYTES)
    assert rows[0]["signature"]  # non-empty signature string
    assert rows[0]["request_id"] == "req-obs"


@pytest.mark.asyncio
async def test_restart_shaped_success_writes_null_cost_observation(
    foundation, monkeypatch
) -> None:
    """If the in-memory costing is gone (process restart / FIFO eviction) when
    the download finalises, the observation is still written with NULL cost."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.backends.cds.costing import CostingResult

    _patch_costing(monkeypatch, CostingResult(units=24.0, limit=121000.0))
    _patch_cdsapi(
        monkeypatch,
        retrieve_returns=_fake_remote("req-restart"),
        get_remote_json={"status": "successful"},
    )
    submit_backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await submit_backend.submit(_good_params())

    # Fresh backend instance = empty _inflight_costing (simulates restart).
    poll_backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await poll_backend.check_status("req-restart")

    assert out["status"] == "successful"
    rows = await foundation.persistence.list_size_observations(
        "cds", "reanalysis-era5-single-levels", None
    )
    assert len(rows) == 1
    assert rows[0]["cost_units"] is None
    assert rows[0]["size_bytes"] == len(_CDS_DEFAULT_DOWNLOAD_BYTES)


@pytest.mark.asyncio
async def test_submit_sanitises_workflow_request_json(foundation, monkeypatch) -> None:
    """the project conventions invariant 2 — no credentials in persisted records.
    A param named like a credential field must be redacted in the
    workflow row's ``request_json``."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    sneaky = {
        "dataset_id": "reanalysis-era5-single-levels",
        "inputs": {
            "variable": ["2m_temperature"],
            "year": ["2024"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00"],
        },
        "api_key": "abcdef01-2345-6789-abcd-ef0123456789",  # ~credential-shaped
    }

    _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote("req-sanit"))
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    # Pydantic ``extra=forbid`` will reject the top-level ``api_key`` —
    # the test asserts the validator catches it BEFORE persistence.
    from copernicus_mcp.errors import ValidationError

    with pytest.raises(ValidationError):
        await backend.submit(sneaky)


@pytest.mark.asyncio
async def test_submit_force_refresh_skips_cache(foundation, monkeypatch, tmp_path: Path) -> None:
    """``__options.force_refresh=true`` bypasses cache hit and dedupe,
    triggering a fresh SDK call."""
    from copernicus_mcp.backends.cds.backend import (
        CdsBackend,
        _cache_storage_key,
    )
    from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    req = CdsRetrieveRequest.model_validate(_good_params())
    cache_key = foundation.data_model.cache_key_for_cds_retrieve(req)
    canned = tmp_path / "canned.grib"
    canned.write_bytes(b"GRIB-content")
    await foundation.cache.store_file(
        _cache_storage_key(cache_key),
        canned,
        backend_id="cds",
        content_type="application/x-grib",
    )

    _, sdk = _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote("forced"))

    params = dict(_good_params())
    params["__options"] = {"force_refresh": True}
    out = await backend.submit(params)
    assert out["status"] == "queued"
    assert out["request_id"] == "forced"
    assert sdk.retrieve.call_count == 1


@pytest.mark.asyncio
async def test_submit_raises_when_cdsapi_extra_missing(foundation, monkeypatch) -> None:
    """``CdsBackend`` must keep importing without the ``cds`` extra
    (mirrors CMEMS pattern), but ``submit`` must surface a clean
    ``BackendError(missing_dependency)`` if it tries to actually use
    the SDK and the import fails."""
    import sys

    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import BackendError

    monkeypatch.setitem(sys.modules, "cdsapi", None)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    with pytest.raises(BackendError) as exc:
        await backend.submit(_good_params())
    assert exc.value.error_record.error_subclass == "missing_dependency"


@pytest.mark.asyncio
async def test_submit_constructs_cdsapi_with_locked_kwargs(foundation, monkeypatch) -> None:
    """Codex spec review: ``LegacyClient`` debug-logs ``key=...`` if
    ``debug=True``; ``progress=True`` writes a tqdm bar to stderr in
    daemon mode; ``wait_until_complete=True`` would block the event
    loop. Pin the constructor kwargs to prevent regressions."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    remote = _fake_remote("req-locked")
    fake_class, _ = _patch_cdsapi(monkeypatch, retrieve_returns=remote)

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await backend.submit(_good_params())

    fake_class.assert_called_once()
    kwargs = fake_class.call_args.kwargs
    assert kwargs["wait_until_complete"] is False
    assert kwargs["quiet"] is True
    assert kwargs["progress"] is False
    assert kwargs["debug"] is False


@pytest.mark.asyncio
async def test_submit_caps_cdsapi_retry_kwargs(foundation, monkeypatch) -> None:
    """T-CDS-013: ``cdsapi.Client`` defaults to ``retry_max=500`` and
    ``sleep_max=120`` — a hard-failing server can keep us retrying for
    ~17 hours. The 2026-05-13 EWDS ``efas-historical`` smoke surfaced
    this: an HTTP 500 went through hundreds of attempts before manual
    kill. Cap to a small bounded retry so MCP fails fast and the
    orchestrator/user decides when to retry, instead of the SDK
    silently busy-waiting."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    remote = _fake_remote("req-retrycap")
    fake_class, _ = _patch_cdsapi(monkeypatch, retrieve_returns=remote)

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await backend.submit(_good_params())

    kwargs = fake_class.call_args.kwargs
    # Concrete values pinned. ``retry_max`` is forwarded by cdsapi to
    # ``multiurl.robust(maximum_tries=...)`` — i.e. total attempts,
    # not retries on top of one initial call (verified by codex round-1
    # against ``cdsapi==0.7.7``). 3 attempts with a 10s back-off ceiling
    # => ~30s worst case per HTTP request, vs ~17h with cdsapi defaults
    # (500 × 120s).
    assert kwargs["retry_max"] == 3
    assert kwargs["sleep_max"] == 10


# ---------------------------------------------------------------------------
# check_status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_status_unknown_request_raises_not_found(foundation) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import NotFoundError

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    with pytest.raises(NotFoundError):
        await backend.check_status("missing-id")


def test_cds_target_filename_picks_zip_when_download_format_is_zip() -> None:
    """T-CDS-018: ``download_format: zip`` outer wrapper wins regardless
    of inner data_format — the file on disk IS a zip."""
    from copernicus_mcp.backends.cds.backend import _cds_target_filename

    assert _cds_target_filename(
        "abc123", {"data_format": "netcdf", "download_format": "zip"}
    ) == "cds_abc123.zip"


def test_cds_target_filename_picks_nc_for_unarchived_netcdf() -> None:
    from copernicus_mcp.backends.cds.backend import _cds_target_filename

    assert _cds_target_filename(
        "abc123",
        {"data_format": "netcdf", "download_format": "unarchived"},
    ) == "cds_abc123.nc"


def test_cds_target_filename_picks_grib_for_unarchived_grib() -> None:
    from copernicus_mcp.backends.cds.backend import _cds_target_filename

    assert _cds_target_filename(
        "abc123",
        {"data_format": "grib", "download_format": "unarchived"},
    ) == "cds_abc123.grib"


def test_cds_target_filename_falls_back_to_bin_when_unknown() -> None:
    from copernicus_mcp.backends.cds.backend import _cds_target_filename

    assert _cds_target_filename("abc123", {}) == "cds_abc123.bin"
    assert _cds_target_filename(
        "abc123", {"data_format": "exotic", "download_format": "unarchived"}
    ) == "cds_abc123.bin"


def test_cds_target_filename_handles_legacy_format_key() -> None:
    """Legacy ``format`` key still used in old reproducers — derive from
    it as a best-effort fallback so the cached file isn't a useless .bin."""
    from copernicus_mcp.backends.cds.backend import _cds_target_filename

    assert _cds_target_filename(
        "abc123", {"format": "netcdf"}
    ) == "cds_abc123.nc"
    assert _cds_target_filename(
        "abc123", {"format": "grib"}
    ) == "cds_abc123.grib"
    # grib2 → .grib (server still ships GRIB1/2 in .grib containers).
    assert _cds_target_filename(
        "abc123", {"format": "grib2"}
    ) == "cds_abc123.grib"


@pytest.mark.asyncio
async def test_finalise_downloads_with_smart_extension_for_netcdf(
    foundation, monkeypatch
) -> None:
    """End-to-end: seed a row whose request_json says data_format=netcdf,
    let check_status finalise, assert the cached file ends in .nc."""
    import json

    from copernicus_mcp.backends.cds.backend import CdsBackend

    cache_key = "ck-nc"
    request_payload = {
        "dataset_id": "reanalysis-era5-single-levels",
        "inputs": {
            "data_format": "netcdf",
            "download_format": "unarchived",
            "variable": ["t2m"],
        },
    }
    from datetime import UTC
    from datetime import datetime as _dt

    now = _dt.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    await foundation.persistence.record_workflow(
        {
            "request_id": "rid-nc",
            "backend_id": "cds",
            "operation": "submit",
            "status": "running",
            "cache_key": cache_key,
            "request_json": json.dumps(request_payload, sort_keys=True),
            "response_json": None,
            "error_record_json": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    _, sdk = _patch_cdsapi(
        monkeypatch,
        get_remote_json={"status": "successful"},
        download_bytes=b"CDF\x01\x00\x00\x00",
    )

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.check_status("rid-nc")
    assert out["status"] == "successful"
    filepath = out["result"]["filepath"]
    assert filepath.endswith(".nc"), filepath


@pytest.mark.asyncio
async def test_finalise_envelope_carries_content_type_for_netcdf(
    foundation, monkeypatch
) -> None:
    """T-CDS-018: envelope metadata gets an explicit content_type field
    so the agent can surface 'this is a NetCDF file' to the user."""
    import json

    from copernicus_mcp.backends.cds.backend import CdsBackend

    cache_key = "ck-meta"
    request_payload = {
        "dataset_id": "reanalysis-era5-single-levels",
        "inputs": {
            "data_format": "netcdf",
            "download_format": "unarchived",
            "variable": ["t2m"],
        },
    }
    from datetime import UTC
    from datetime import datetime as _dt

    now = _dt.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    await foundation.persistence.record_workflow(
        {
            "request_id": "rid-meta",
            "backend_id": "cds",
            "operation": "submit",
            "status": "running",
            "cache_key": cache_key,
            "request_json": json.dumps(request_payload, sort_keys=True),
            "response_json": None,
            "error_record_json": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    _patch_cdsapi(
        monkeypatch,
        get_remote_json={"status": "successful"},
        download_bytes=b"CDF\x01\x00\x00\x00",
    )

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.check_status("rid-meta")
    metadata = out["result"].get("metadata") or {}
    assert metadata.get("content_type") == "application/x-netcdf", out["result"]


# --- T-CDS-018 round-2 fixes (cr findings) -------------------------------


def test_cds_target_filename_handles_netcdf4_literal() -> None:
    """Round-2 (local MED-1): bundled constraints list ``netcdf4`` as a
    valid ``data_format``; agents that copy from ``apply_constraints``
    pass that literal verbatim. Must map to ``.nc``."""
    from copernicus_mcp.backends.cds.backend import _cds_target_filename

    assert _cds_target_filename(
        "abc123", {"data_format": "netcdf4"}
    ) == "cds_abc123.nc"


def test_cds_target_filename_handles_netcdf_legacy_literal() -> None:
    """Round-2 (codex LOW): ``netcdf_legacy`` is a documented CDS/ADS
    data_format producing NetCDF3 — same on-disk extension."""
    from copernicus_mcp.backends.cds.backend import _cds_target_filename

    assert _cds_target_filename(
        "abc123", {"data_format": "netcdf_legacy"}
    ) == "cds_abc123.nc"


def test_cds_target_filename_accepts_single_element_list_format() -> None:
    """Round-2 (local MED-2): the schema allows list-valued inputs (e.g.
    ``{"data_format": ["netcdf"]}``); cdsapi normalises them. Match the
    scalar so the user-visible filename agrees with content."""
    from copernicus_mcp.backends.cds.backend import _cds_target_filename

    assert _cds_target_filename(
        "abc123", {"data_format": ["netcdf"]}
    ) == "cds_abc123.nc"
    assert _cds_target_filename(
        "abc123", {"download_format": ["zip"]}
    ) == "cds_abc123.zip"


def test_cds_target_filename_falls_back_when_list_is_multi_element() -> None:
    """Multi-element list is genuinely ambiguous (an agent error in
    practice — these are scalars in the upstream API). Don't guess —
    bin fallback so the magic-byte sniff has the final say."""
    from copernicus_mcp.backends.cds.backend import _cds_target_filename

    assert _cds_target_filename(
        "abc123", {"data_format": ["netcdf", "grib"]}
    ) == "cds_abc123.bin"


def test_cds_sniff_extension_recognises_zip_magic() -> None:
    """Round-2 (codex MED-1): ECMWF can return ZIP for
    netcdf+unarchived multi-variable requests. Magic-byte sniff is the
    truth oracle — extension follows content, not the request."""
    import tempfile
    from pathlib import Path as _P

    from copernicus_mcp.backends.cds.backend import _cds_sniff_extension

    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(b"PK\x03\x04rest-of-zip-header")
        path = _P(fh.name)
    try:
        assert _cds_sniff_extension(path) == ".zip"
    finally:
        path.unlink()


def test_cds_sniff_extension_recognises_netcdf3_magic() -> None:
    import tempfile
    from pathlib import Path as _P

    from copernicus_mcp.backends.cds.backend import _cds_sniff_extension

    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(b"CDF\x01\x00\x00\x00\x00more")
        path = _P(fh.name)
    try:
        assert _cds_sniff_extension(path) == ".nc"
    finally:
        path.unlink()


def test_cds_sniff_extension_recognises_hdf5_magic() -> None:
    """NetCDF4 is HDF5 under the hood — recognise the HDF signature."""
    import tempfile
    from pathlib import Path as _P

    from copernicus_mcp.backends.cds.backend import _cds_sniff_extension

    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(b"\x89HDF\r\n\x1a\n_more_hdf_payload")
        path = _P(fh.name)
    try:
        assert _cds_sniff_extension(path) == ".nc"
    finally:
        path.unlink()


def test_cds_sniff_extension_recognises_grib_magic() -> None:
    import tempfile
    from pathlib import Path as _P

    from copernicus_mcp.backends.cds.backend import _cds_sniff_extension

    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(b"GRIB\x00\x00\x00")
        path = _P(fh.name)
    try:
        assert _cds_sniff_extension(path) == ".grib"
    finally:
        path.unlink()


def test_cds_sniff_extension_returns_none_for_unknown_bytes() -> None:
    import tempfile
    from pathlib import Path as _P

    from copernicus_mcp.backends.cds.backend import _cds_sniff_extension

    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(b"\x00\x01\x02\x03random-bytes")
        path = _P(fh.name)
    try:
        assert _cds_sniff_extension(path) is None
    finally:
        path.unlink()


@pytest.mark.asyncio
async def test_finalise_renames_to_zip_when_cds_returns_zip_for_netcdf(
    foundation, monkeypatch
) -> None:
    """Round-2 integration (codex MED-1): user asked for
    data_format=netcdf+unarchived, CDS handed back a ZIP (ERA5
    multi-variable). The cached file MUST end in ``.zip`` and
    ``content_type`` MUST be ``application/zip`` — don't lie about
    the bytes on disk."""
    import json

    from copernicus_mcp.backends.cds.backend import CdsBackend

    cache_key = "ck-zip-override"
    request_payload = {
        "dataset_id": "reanalysis-era5-single-levels",
        "inputs": {
            "data_format": "netcdf",
            "download_format": "unarchived",
            "variable": ["t2m", "u10"],
        },
    }
    from datetime import UTC
    from datetime import datetime as _dt

    now = _dt.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    await foundation.persistence.record_workflow(
        {
            "request_id": "rid-zip",
            "backend_id": "cds",
            "operation": "submit",
            "status": "running",
            "cache_key": cache_key,
            "request_json": json.dumps(request_payload, sort_keys=True),
            "response_json": None,
            "error_record_json": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    _patch_cdsapi(
        monkeypatch,
        get_remote_json={"status": "successful"},
        download_bytes=b"PK\x03\x04zip-header-bytes",
    )

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.check_status("rid-zip")
    assert out["status"] == "successful"
    filepath = out["result"]["filepath"]
    assert filepath.endswith(".zip"), filepath
    metadata = out["result"].get("metadata") or {}
    assert metadata.get("content_type") == "application/zip", metadata


@pytest.mark.asyncio
async def test_cache_hit_response_carries_content_type(
    foundation, monkeypatch, tmp_path: Path
) -> None:
    """Round-2 (codex MED-2): the submit cache-hit envelope must carry
    ``content_type`` so the agent gets the same shape from cache-hit as
    from finalise."""
    from copernicus_mcp.backends.cds.backend import _cache_storage_key

    canned = tmp_path / "canned.nc"
    canned.write_bytes(b"CDF\x01\x00\x00\x00stub")
    cache_key = "ck-cachehit-ct"
    await foundation.cache.store_file(
        _cache_storage_key(cache_key),
        canned,
        backend_id="cds",
        content_type="application/x-netcdf",
    )
    # Build the envelope via the dedicated helper.
    from copernicus_mcp.backends.cds.backend import (
        _success_response_from_cache,
    )

    looked_up = await foundation.cache.lookup_file(_cache_storage_key(cache_key))
    assert looked_up is not None
    env = _success_response_from_cache(
        request_id="rid-x", cache_key=cache_key, filepath=looked_up
    )
    assert env["result"]["metadata"].get("content_type") == "application/x-netcdf"


# Round-3 cr fixes (local-HIGH + codex-LOW) -------------------------------


def test_cds_sniff_extension_rejects_truncated_cdf() -> None:
    """Round-3 codex-LOW: bare ``CDF`` without a version byte is not a
    valid NetCDF3 file. Tightened signature requires CDF\\x01/\\x02/\\x05."""
    import tempfile
    from pathlib import Path as _P

    from copernicus_mcp.backends.cds.backend import _cds_sniff_extension

    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(b"CDF")  # 3 bytes only — no version byte
        path = _P(fh.name)
    try:
        assert _cds_sniff_extension(path) is None
    finally:
        path.unlink()


def test_cds_sniff_extension_recognises_empty_zip_magic() -> None:
    """Round-3 codex-LOW: empty-archive ZIP signature (``PK\\x05\\x06``)
    is still a valid ZIP and must classify as .zip."""
    import tempfile
    from pathlib import Path as _P

    from copernicus_mcp.backends.cds.backend import _cds_sniff_extension

    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(b"PK\x05\x06\x00\x00\x00\x00")
        path = _P(fh.name)
    try:
        assert _cds_sniff_extension(path) == ".zip"
    finally:
        path.unlink()


def test_cds_sniff_extension_rejects_short_hdf5_prefix() -> None:
    """Round-3 codex-LOW: HDF5 requires the full 8-byte signature so
    a stray ``\\x89HDF`` byte sequence in another format isn't
    misclassified."""
    import tempfile
    from pathlib import Path as _P

    from copernicus_mcp.backends.cds.backend import _cds_sniff_extension

    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(b"\x89HDFXXXX")  # right prefix, wrong tail
        path = _P(fh.name)
    try:
        assert _cds_sniff_extension(path) is None
    finally:
        path.unlink()


@pytest.mark.asyncio
async def test_finalise_marks_failed_when_sniff_rename_raises(
    foundation, monkeypatch
) -> None:
    """Round-3 local-HIGH: an ``OSError`` from ``target_path.replace``
    must NOT leave the workflow row stuck ``running`` with a leaked
    staging file. Pattern mirrors download/store_file error handling."""
    import json

    from copernicus_mcp.backends.cds.backend import CdsBackend

    cache_key = "ck-rename-fail"
    request_payload = {
        "dataset_id": "reanalysis-era5-single-levels",
        "inputs": {
            "data_format": "netcdf",
            "download_format": "unarchived",
            "variable": ["t2m", "u10"],
        },
    }
    from datetime import UTC
    from datetime import datetime as _dt

    now = _dt.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    await foundation.persistence.record_workflow(
        {
            "request_id": "rid-rename-fail",
            "backend_id": "cds",
            "operation": "submit",
            "status": "running",
            "cache_key": cache_key,
            "request_json": json.dumps(request_payload, sort_keys=True),
            "response_json": None,
            "error_record_json": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    # Force sniff to fire (zip magic vs .nc input) and the rename to fail.
    _patch_cdsapi(
        monkeypatch,
        get_remote_json={"status": "successful"},
        download_bytes=b"PK\x03\x04zip-header-bytes",
    )

    def _boom(self, target):
        raise OSError("ENOSPC: no space left on device")

    monkeypatch.setattr(Path, "replace", _boom)

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    # T-CDS-ASYNC-DOWNLOAD: the backgrounded download/sniff failure settles the row
    # to ``failed`` and surfaces as a failed envelope (within the inline grace), not
    # raised from check_status.
    out = await backend.check_status("rid-rename-fail")
    assert out["status"] == "failed"

    row = await foundation.persistence.fetch_workflow("rid-rename-fail")
    assert row is not None
    assert row["status"] == "failed", row
    assert row["error_record_json"] is not None
    err = json.loads(row["error_record_json"])
    # Round-3 cr LOW (codex + local): pin the classification.
    # ``_wrap_sdk_error(op="sniff_rename")`` always returns BackendError
    # with error_subclass="sdk_sniff_rename_failure". A future accidental
    # re-categorisation should break this test.
    assert err.get("error_class") == "BackendError", err
    assert err.get("error_subclass") == "sdk_sniff_rename_failure", err

    # Round-3 cr LOW (codex): also pin staging cleanup. After the
    # except block, the original target file and the staging dir should
    # both be gone — otherwise long-running servers leak ERA5-sized
    # files on every transient rename failure.
    zone = foundation.cache.cache_zone_for("cds")
    staging_root = zone / ".staging"
    leftover_files = (
        list(staging_root.rglob("cds_*")) if staging_root.exists() else []
    )
    assert leftover_files == [], leftover_files


@pytest.mark.asyncio
async def test_fetch_result_carries_content_type(
    foundation, tmp_path: Path
) -> None:
    """Round-2 (codex MED-2): ``fetch_result`` must mirror
    ``check_status`` — the agent shouldn't see two different shapes for
    the same underlying file."""
    from copernicus_mcp.backends.cds.backend import (
        CdsBackend,
        _cache_storage_key,
    )

    canned = tmp_path / "canned.nc"
    canned.write_bytes(b"CDF\x01\x00\x00\x00stub")
    cache_key = "ck-fetchresult"
    await foundation.cache.store_file(
        _cache_storage_key(cache_key),
        canned,
        backend_id="cds",
        content_type="application/x-netcdf",
    )
    from datetime import UTC
    from datetime import datetime as _dt

    now = _dt.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    await foundation.persistence.record_workflow(
        {
            "request_id": "rid-fr",
            "backend_id": "cds",
            "operation": "submit",
            "status": "successful",
            "cache_key": cache_key,
            "request_json": "{}",
            "response_json": None,
            "error_record_json": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.fetch_result("rid-fr", target=tmp_path / "ignored")
    assert (
        out["result"]["metadata"].get("content_type") == "application/x-netcdf"
    ), out["result"]


@pytest.mark.asyncio
async def test_check_status_terminal_successful_returns_cached_state(
    foundation, monkeypatch, tmp_path: Path
) -> None:
    """A workflow row already marked ``successful`` returns its
    descriptor from the cache without invoking the SDK."""
    from copernicus_mcp.backends.cds.backend import (
        CdsBackend,
        _cache_storage_key,
    )

    canned = tmp_path / "canned.grib"
    canned.write_bytes(b"GRIB")
    cache_key = "ck-success"
    await foundation.cache.store_file(
        _cache_storage_key(cache_key),
        canned,
        backend_id="cds",
        content_type="application/x-grib",
    )
    await _seed_workflow_row(
        foundation, request_id="rid-1", cache_key=cache_key, status="successful"
    )

    fake_class, _ = _patch_cdsapi(monkeypatch)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.check_status("rid-1")
    assert out["status"] == "successful"
    assert out["request_id"] == "rid-1"
    assert "filepath" in out["result"]
    fake_class.assert_not_called()


@pytest.mark.asyncio
async def test_check_status_polls_remote_running(foundation, monkeypatch) -> None:
    """A queued row whose remote is ``running`` returns canonical
    ``running`` (and persists the transition)."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_workflow_row(foundation, request_id="rid-r", status="queued")
    _patch_cdsapi(monkeypatch, get_remote_json={"status": "running"})

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.check_status("rid-r")
    assert out["status"] == "running"

    row = await foundation.persistence.fetch_workflow("rid-r")
    assert row is not None and row["status"] == "running"


@pytest.mark.asyncio
async def test_check_status_accepted_maps_to_queued(foundation, monkeypatch) -> None:
    """cdsapi status ``accepted`` is the same logical state as
    ``queued`` per research §6.5.2 — both must surface as canonical
    ``queued``."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_workflow_row(foundation, request_id="rid-a", status="queued")
    _patch_cdsapi(monkeypatch, get_remote_json={"status": "accepted"})

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.check_status("rid-a")
    assert out["status"] == "queued"


@pytest.mark.asyncio
async def test_check_status_finalises_successful_downloads_to_cache(
    foundation, monkeypatch
) -> None:
    """Remote transitioned to ``successful``: backend downloads via
    ``client.download_results``, stores in cache, persists row, returns
    descriptor."""
    from copernicus_mcp.backends.cds.backend import (
        CdsBackend,
        _cache_storage_key,
    )

    cache_key = "ck-final"
    await _seed_workflow_row(foundation, request_id="rid-f", cache_key=cache_key, status="running")
    _, sdk = _patch_cdsapi(monkeypatch, get_remote_json={"status": "successful"})

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.check_status("rid-f")
    assert out["status"] == "successful"
    assert "filepath" in out["result"]

    row = await foundation.persistence.fetch_workflow("rid-f")
    assert row is not None and row["status"] == "successful"

    # Cache resolves the descriptor key
    cached = await foundation.cache.lookup_file(_cache_storage_key(cache_key))
    assert cached is not None
    assert cached.read_bytes() == b"GRIB-content-from-cds"
    sdk.client.download_results.assert_called_once()


@pytest.mark.asyncio
async def test_check_status_successful_writes_provenance_sidecar(
    foundation, monkeypatch
) -> None:
    """T-CDS-011.5: every successful CDS download writes a
    ``<file>.provenance.json`` sidecar alongside the cached file (and
    the corresponding row to the provenance SQLite table). Mirrors
    the CMEMS behaviour (T-024) so a user holding a bare ``.bin``
    file can reconstruct the request without re-querying persistence.
    """
    import json

    # T-CDS-011.5: provenance needs the original dataset_id from the
    # persisted request_json — seed it directly here rather than via
    # the generic _seed_workflow_row helper (which seeds "{}").
    import json as _json
    from datetime import UTC, datetime

    from copernicus_mcp.backends.cds.backend import (
        CdsBackend,
        _cache_storage_key,
    )

    cache_key = "ck-prov"
    realistic_payload = {
        "dataset_id": "reanalysis-era5-single-levels",
        "inputs": {
            "variable": ["2m_temperature"],
            "year": ["2024"], "month": ["01"], "day": ["01"],
            "time": ["00:00"],
        },
    }
    iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    await foundation.persistence.record_workflow(
        {
            "request_id": "rid-prov",
            "backend_id": "cds",
            "operation": "submit",
            "status": "running",
            "cache_key": cache_key,
            "request_json": _json.dumps(realistic_payload),
            "response_json": None,
            "error_record_json": None,
            "created_at": iso,
            "updated_at": iso,
        }
    )

    _patch_cdsapi(monkeypatch, get_remote_json={"status": "successful"})

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.check_status("rid-prov")
    assert out["status"] == "successful"

    cached = await foundation.cache.lookup_file(_cache_storage_key(cache_key))
    assert cached is not None
    sidecar = cached.with_suffix(cached.suffix + ".provenance.json")
    assert sidecar.is_file(), f"sidecar missing at {sidecar}"
    record = json.loads(sidecar.read_text())
    # Minimal sanity: schema metadata + backend + dataset block exist.
    assert record["backend"]["id"] == "cds"
    assert record["dataset"]["dataset_id"]
    # The file block carries an MD5 of the cached bytes.
    files = record.get("files") or []
    assert files and files[0].get("md5")


@pytest.mark.asyncio
async def test_check_status_failed_status_persists_failed(foundation, monkeypatch) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_workflow_row(foundation, request_id="rid-x", status="running")
    _patch_cdsapi(monkeypatch, get_remote_json={"status": "failed"})

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.check_status("rid-x")
    assert out["status"] == "failed"
    row = await foundation.persistence.fetch_workflow("rid-x")
    assert row is not None and row["status"] == "failed"


@pytest.mark.asyncio
async def test_remote_job_failed_carries_backend_diagnostics(
    foundation, monkeypatch
) -> None:
    """T-CDS-011.3: when the CDS server returns a structured failed
    job-state JSON, the persisted error record must carry its salient
    fields under ``context.backend_diagnostics``. Previously only a
    one-line ``message`` survived, which forced users to guess whether
    the failure was 'too many concurrent jobs', 'request too large',
    or an internal server error."""
    import json

    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_workflow_row(foundation, request_id="rid-diag", status="running")
    _patch_cdsapi(
        monkeypatch,
        get_remote_json={
            "status": "failed",
            "error": {
                "type": "system",
                "code": 500,
                "message": "internal-error: backend pipeline timeout",
                "trace_id": "trace-deadbeef",
            },
            "attempt": 3,
            "jobID": "rid-diag",
            "metadata": {"queue": "heavy"},
        },
    )

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.check_status("rid-diag")
    assert out["status"] == "failed"

    row = await foundation.persistence.fetch_workflow("rid-diag")
    assert row is not None and row["error_record_json"]
    rec = json.loads(row["error_record_json"])
    diag = rec.get("context", {}).get("backend_diagnostics") or {}
    # Structured error survives at the top of the diagnostics.
    assert diag.get("status") == "failed"
    assert diag.get("error", {}).get("code") == 500
    assert diag.get("error", {}).get("trace_id") == "trace-deadbeef"
    # Non-error contextual fields preserved.
    assert diag.get("attempt") == 3


@pytest.mark.asyncio
async def test_user_supplied_uuid_under_inputs_jobid_is_redacted(
    foundation, monkeypatch
) -> None:
    """T-CDS-011 round-2 codex HIGH: ``CdsRetrieveRequest.inputs``
    accepts arbitrary keys (cdsapi-shaped). A malicious or careless
    caller can stick a UUID-shape secret under ``inputs.jobID`` and,
    if the sanitiser globally trusts that key, the secret survives
    into ``request_json`` / provenance.

    The fix is narrow: keep the GLOBAL sanitiser allowlist to keys
    that are universally safe (``request_id``); preserve
    server-side jobIDs locally at the diagnostics call site only.
    This test pins the user-input path stays redacted regardless
    of whether the diagnostics path got a UUID jobID at the same
    time.
    """
    import json as _json

    from copernicus_mcp.backends.cds.backend import CdsBackend

    user_uuid = "deadbeef-cafe-1234-5678-aabbccddeeff"
    _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote("req-userjid"))
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    params = {
        "dataset_id": "reanalysis-era5-single-levels",
        "inputs": {
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature"],
            "year": ["2024"], "month": ["01"], "day": ["01"],
            "time": ["00:00"],
            # User-supplied — must be sanitised at the boundary.
            "jobID": user_uuid,
            "job_id": user_uuid,
        },
        # T-CDS-KEYCHECK-001 now rejects unknown keys (incl. jobID) up front
        # on snapshot-covered datasets, which is an even stronger guarantee —
        # but the sanitiser redaction is the defence-in-depth layer for the
        # FAIL-OPEN path (no snapshot entry / stale snapshot), so this test
        # bypasses the key check to keep that layer pinned.
        "__options": {"skip_input_validation": True},
    }
    await backend.submit(params)

    row = await foundation.persistence.fetch_workflow("req-userjid")
    assert row is not None
    persisted_inputs = (
        _json.loads(row["request_json"]).get("inputs") or {}
    )
    assert persisted_inputs.get("jobID") == "[REDACTED]", persisted_inputs
    assert persisted_inputs.get("job_id") == "[REDACTED]", persisted_inputs


@pytest.mark.asyncio
async def test_remote_job_failed_preserves_real_uuid_job_id(
    foundation, monkeypatch
) -> None:
    """T-CDS-011 cr round-1 H1 + round-2 codex HIGH: real CDS jobIDs
    are canonical UUIDs and must survive into ``backend_diagnostics``
    so an operator filing an ECMWF support ticket has the actual
    identifier, not ``[REDACTED]``. The fix is NOT in the global
    sanitiser allowlist (that opened a user-input exfil — see
    ``test_user_supplied_uuid_under_inputs_jobid_is_redacted``) but
    in a local per-key restore right after the diagnostics sanitise
    pass in ``CdsBackend._record_terminal``."""
    import json as _json

    from copernicus_mcp.backends.cds.backend import CdsBackend

    real_uuid_job = "4f3d2c1b-9e8a-4c7d-8e9f-0a1b2c3d4e5f"
    await _seed_workflow_row(foundation, request_id="rid-uuid", status="running")
    _patch_cdsapi(
        monkeypatch,
        get_remote_json={
            "status": "failed",
            "jobID": real_uuid_job,
            "error": {"message": "internal error"},
        },
    )

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await backend.check_status("rid-uuid")

    row = await foundation.persistence.fetch_workflow("rid-uuid")
    assert row is not None
    rec = _json.loads(row["error_record_json"])
    diag = rec.get("context", {}).get("backend_diagnostics") or {}
    assert diag.get("jobID") == real_uuid_job, diag


@pytest.mark.asyncio
async def test_remote_job_failed_includes_next_action_hint_about_quota(
    foundation, monkeypatch
) -> None:
    """T-CDS-011.4: CDS empirically rate-limits to ~5-6 concurrent
    jobs/user; excess submits accept into the queue but reap with
    ``remote_job_failed`` after a few minutes. Surface the empirical
    workaround (serialise submits / wait before retry-storm) as a
    structured ``next_action_hint`` on every remote_job_failed error
    so LLM agents and human readers see it without needing to read
    the project decision log."""
    import json

    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_workflow_row(foundation, request_id="rid-q", status="running")
    _patch_cdsapi(monkeypatch, get_remote_json={"status": "failed"})

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await backend.check_status("rid-q")

    row = await foundation.persistence.fetch_workflow("rid-q")
    assert row is not None and row["error_record_json"]
    rec = json.loads(row["error_record_json"])
    hint = rec.get("next_action_hint") or ""
    lo = hint.lower()
    # Hint must mention the empirical pattern in a recoverable form.
    assert "concurrent" in lo or "serialise" in lo or "serialize" in lo, hint


@pytest.mark.asyncio
async def test_remote_job_failed_hint_mentions_data_format_pitfall(
    foundation, monkeypatch
) -> None:
    """T-CDS-014 session 2026-05-16: real-world investigation against
    EWDS efas-historical surfaced that the most common cause of a
    ``remote_job_failed`` with empty server-side log is the legacy
    ``format: ...`` field (CDS migrated to ``data_format`` + matching
    ``download_format``). The structured hint must mention this so
    agents don't repeatedly submit the same malformed request."""
    import json

    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_workflow_row(foundation, request_id="rid-df", status="running")
    _patch_cdsapi(monkeypatch, get_remote_json={"status": "failed"})

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await backend.check_status("rid-df")

    row = await foundation.persistence.fetch_workflow("rid-df")
    assert row is not None and row["error_record_json"]
    rec = json.loads(row["error_record_json"])
    hint = rec.get("next_action_hint") or ""
    lo = hint.lower()
    assert "data_format" in lo, hint
    assert "download_format" in lo, hint


@pytest.mark.asyncio
async def test_remote_job_failed_hint_mentions_apply_constraints_for_live_truth(
    foundation, monkeypatch
) -> None:
    """T-CDS-017 session 2026-05-16: when a submit fails after the agent
    used a stale ``cds_describe_dataset → available_inputs`` snapshot,
    the hint must point at ``cds_apply_constraints`` as the LIVE
    server-side truth so the agent stops retrying with stale field
    names."""
    import json

    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_workflow_row(foundation, request_id="rid-live", status="running")
    _patch_cdsapi(monkeypatch, get_remote_json={"status": "failed"})

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await backend.check_status("rid-live")

    row = await foundation.persistence.fetch_workflow("rid-live")
    assert row is not None and row["error_record_json"]
    rec = json.loads(row["error_record_json"])
    hint = rec.get("next_action_hint") or ""
    assert "cds_apply_constraints" in hint, hint


@pytest.mark.asyncio
async def test_remote_job_failed_hint_mentions_time_invariant_aux_vars(
    foundation, monkeypatch
) -> None:
    """Auxiliary time-invariant variables (e.g. EFAS elevation,
    soil_depth, upstream_area) refuse hyear/hmonth/hday/time params.
    Surface this as a structured hint alongside the quota / data_format
    nudges."""
    import json

    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_workflow_row(foundation, request_id="rid-aux", status="running")
    _patch_cdsapi(monkeypatch, get_remote_json={"status": "failed"})

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await backend.check_status("rid-aux")

    row = await foundation.persistence.fetch_workflow("rid-aux")
    assert row is not None and row["error_record_json"]
    rec = json.loads(row["error_record_json"])
    hint = rec.get("next_action_hint") or ""
    lo = hint.lower()
    assert "time-invariant" in lo or "auxiliary" in lo, hint


@pytest.mark.asyncio
async def test_check_status_rejected_maps_to_failed(foundation, monkeypatch) -> None:
    """``rejected`` is a server-side terminal failure (e.g., constraint
    violation) — collapses to canonical ``failed``."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_workflow_row(foundation, request_id="rid-rej", status="running")
    _patch_cdsapi(monkeypatch, get_remote_json={"status": "rejected"})

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.check_status("rid-rej")
    assert out["status"] == "failed"


@pytest.mark.asyncio
async def test_check_status_dismissed_maps_to_cancelled(foundation, monkeypatch) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_workflow_row(foundation, request_id="rid-d", status="running")
    _patch_cdsapi(monkeypatch, get_remote_json={"status": "dismissed"})

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.check_status("rid-d")
    assert out["status"] == "cancelled"


@pytest.mark.asyncio
async def test_check_status_unknown_status_raises_backend_error(foundation, monkeypatch) -> None:
    """An unrecognised SDK status surfaces as ``BackendError`` rather
    than a DB CHECK violation (the project conventions invariant 5)."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import BackendError

    await _seed_workflow_row(foundation, request_id="rid-u", status="running")
    _patch_cdsapi(monkeypatch, get_remote_json={"status": "transmuted"})

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    with pytest.raises(BackendError) as exc:
        await backend.check_status("rid-u")
    assert exc.value.error_record.error_subclass == "unknown_remote_status"


@pytest.mark.asyncio
async def test_check_status_does_not_mark_old_running_as_failed(foundation, monkeypatch) -> None:
    """Codex spec review HIGH-1: CDS jobs can validly stay queued/running
    for hours or days. ``check_status`` MUST poll remote for truth and
    not flip a stale row to ``failed`` based on local age."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_workflow_row(foundation, request_id="rid-old", status="running", age_seconds=3600)
    _patch_cdsapi(monkeypatch, get_remote_json={"status": "running"})

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.check_status("rid-old")
    assert out["status"] == "running"


@pytest.mark.asyncio
async def test_check_status_concurrent_finalise_downloads_once(foundation, monkeypatch) -> None:
    """Two simultaneous ``check_status`` calls on the same successful job must
    download EXACTLY once (review HIGH: the registry mutex serialises the spawn
    decision across the terminal-row recheck). With backgrounding, one caller may
    observe the download still in flight (running / phase downloading) while the
    other sees it finished — but only one download fires, then both settle."""
    import contextlib as _ctx

    from copernicus_mcp.backends.cds.backend import CdsBackend

    cache_key = "ck-concurrent"
    await _seed_workflow_row(foundation, request_id="rid-c", cache_key=cache_key, status="running")
    _, sdk = _patch_cdsapi(monkeypatch, get_remote_json={"status": "successful"})

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    a, b = await asyncio.gather(backend.check_status("rid-c"), backend.check_status("rid-c"))
    # Each caller is either already-successful or still-downloading — never an error.
    assert {a["status"], b["status"]} <= {"successful", "running"}
    # The critical invariant: the file downloaded exactly once (no double-spawn).
    assert sdk.client.download_results.call_count == 1
    # Drain any in-flight background download, then a poll settles to successful.
    for t in list(backend._downloads.values()):
        with _ctx.suppress(Exception):
            await t
    final = await backend.check_status("rid-c")
    assert final["status"] == "successful"
    assert sdk.client.download_results.call_count == 1


@pytest.mark.asyncio
async def test_check_status_runs_sdk_in_thread_pool(foundation, monkeypatch) -> None:
    """SDK calls must go through ``asyncio.to_thread`` — a synchronous
    call on the event loop would block all other backend operations."""
    import threading

    from copernicus_mcp.backends.cds.backend import CdsBackend

    main_thread_id = threading.get_ident()
    seen_threads: list[int] = []

    def _record_thread(*_args, **_kwargs):
        seen_threads.append(threading.get_ident())
        m = MagicMock()
        m.json = {"status": "running"}
        return m

    await _seed_workflow_row(foundation, request_id="rid-t", status="running")
    _, sdk = _patch_cdsapi(monkeypatch)
    sdk.client.get_remote = MagicMock(side_effect=_record_thread)

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await backend.check_status("rid-t")
    assert seen_threads, "get_remote was never called"
    assert seen_threads[0] != main_thread_id


@pytest.mark.asyncio
async def test_check_status_cache_evicted_after_successful_synthesises_error(
    foundation, monkeypatch
) -> None:
    """If the row is ``successful`` but the cache file was LRU-evicted,
    surface a synthetic ``cache_eviction`` error instead of returning
    ``status=successful`` with an empty ``result``. Mirrors CMEMS at
    backends/cmems/backend.py:849."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_workflow_row(
        foundation, request_id="rid-ev", cache_key="ck-ev", status="successful"
    )
    _patch_cdsapi(monkeypatch)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.check_status("rid-ev")
    assert out["status"] == "failed"
    err = out["error_details"]
    assert isinstance(err, dict)
    assert err.get("error_subclass") == "cache_eviction"


@pytest.mark.asyncio
async def test_check_status_download_failure_persists_failed(foundation, monkeypatch) -> None:
    """If ``download_results`` raises after remote ``successful``, the
    row settles to ``failed`` with sanitised error details — never
    leaves a half-finalised row in ``running``."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_workflow_row(foundation, request_id="rid-dl", cache_key="ck-dl", status="running")
    _, sdk = _patch_cdsapi(monkeypatch, get_remote_json={"status": "successful"})
    sdk.client.download_results = MagicMock(side_effect=RuntimeError("transient socket reset"))

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    # T-CDS-ASYNC-DOWNLOAD: a backgrounded download failure settles the row to
    # ``failed`` and surfaces as a failed envelope (within the inline grace).
    out = await backend.check_status("rid-dl")
    assert out["status"] == "failed"
    row = await foundation.persistence.fetch_workflow("rid-dl")
    assert row is not None and row["status"] == "failed"


# ---------------------------------------------------------------------------
# fetch_result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_result_unknown_raises_not_found(foundation) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import NotFoundError

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    with pytest.raises(NotFoundError):
        await backend.fetch_result("missing", Path("/tmp/whatever"))


@pytest.mark.asyncio
async def test_fetch_result_running_row_raises_not_ready(foundation) -> None:
    """A row not yet ``successful`` must surface a structured
    ``BackendError(error_subclass="result_not_ready")`` — never an empty
    descriptor."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import BackendError

    await _seed_workflow_row(foundation, request_id="rid-w", status="running")
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    with pytest.raises(BackendError) as exc:
        await backend.fetch_result("rid-w", Path("/tmp/x"))
    assert exc.value.error_record.error_subclass == "result_not_ready"


@pytest.mark.asyncio
async def test_fetch_result_successful_returns_descriptor(foundation, tmp_path: Path) -> None:
    """Successful row resolves to the canonical large-data descriptor —
    file already in the cache zone (CMEMS pattern)."""
    from copernicus_mcp.backends.cds.backend import (
        CdsBackend,
        _cache_storage_key,
    )

    cache_key = "ck-fetch"
    canned = tmp_path / "fetch.grib"
    canned.write_bytes(b"GRIB-fetch")
    await foundation.cache.store_file(
        _cache_storage_key(cache_key),
        canned,
        backend_id="cds",
        content_type="application/x-grib",
    )
    await _seed_workflow_row(
        foundation, request_id="rid-ok", cache_key=cache_key, status="successful"
    )

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.fetch_result("rid-ok", Path("/tmp/ignored"))
    assert out["status"] == "successful"
    assert out["request_id"] == "rid-ok"
    assert out["cache_hit"] is True
    assert "filepath" in out["result"]


@pytest.mark.asyncio
async def test_fetch_result_cache_evicted_raises_cache_error(
    foundation,
) -> None:
    """Successful row whose cache file was evicted must raise
    ``CacheError(cache_eviction)`` — caller decides whether to re-run."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import CacheError

    await _seed_workflow_row(
        foundation, request_id="rid-gone", cache_key="ck-gone", status="successful"
    )
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    with pytest.raises(CacheError) as exc:
        await backend.fetch_result("rid-gone", Path("/tmp/y"))
    assert exc.value.error_record.error_subclass == "cache_eviction"


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_unknown_raises_not_found(foundation) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import NotFoundError

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    with pytest.raises(NotFoundError):
        await backend.cancel("missing")


@pytest.mark.asyncio
async def test_cancel_terminal_returns_cancelled_false(foundation, monkeypatch) -> None:
    """A row already in ``successful`` / ``failed`` / ``cancelled`` is
    not re-cancelled and we never call the SDK delete."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_workflow_row(foundation, request_id="rid-done", status="successful")
    fake_class, sdk = _patch_cdsapi(monkeypatch)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.cancel("rid-done")
    assert out["cancelled"] is False
    fake_class.assert_not_called()
    assert sdk.client.delete.call_count == 0


@pytest.mark.asyncio
async def test_cancel_running_calls_sdk_delete_and_persists_cancelled(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_workflow_row(foundation, request_id="rid-run", status="running")
    _, sdk = _patch_cdsapi(monkeypatch)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.cancel("rid-run")
    assert out["cancelled"] is True
    sdk.client.delete.assert_called_once_with("rid-run")
    row = await foundation.persistence.fetch_workflow("rid-run")
    assert row is not None and row["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_against_already_terminal_does_not_overwrite(
    foundation, monkeypatch
) -> None:
    """A cancel call on a row that already settled to ``successful``
    must not flip it to ``cancelled``. The atomic ``_if_pending`` UPDATE
    is the guard. The complementary race-with-finalise scenario lives
    in ``test_check_status_finalisation_does_not_overwrite_cancelled``."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_workflow_row(foundation, request_id="rid-race", status="running")
    _patch_cdsapi(monkeypatch)
    await foundation.persistence.update_workflow_status("rid-race", "successful")

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.cancel("rid-race")
    assert out["cancelled"] is False
    row = await foundation.persistence.fetch_workflow("rid-race")
    assert row is not None and row["status"] == "successful"


@pytest.mark.asyncio
async def test_cancel_swallows_sdk_failure(foundation, monkeypatch) -> None:
    """If the remote DELETE returns an error (job already finished
    server-side, network blip), ``cancel`` still settles the local row
    so the caller can stop polling."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_workflow_row(foundation, request_id="rid-err", status="queued")
    _, sdk = _patch_cdsapi(monkeypatch)
    sdk.client.delete = MagicMock(side_effect=RuntimeError("404 not found"))

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.cancel("rid-err")
    # Even though SDK call raised, the local row settles to cancelled.
    row = await foundation.persistence.fetch_workflow("rid-err")
    assert row is not None and row["status"] == "cancelled"
    assert out["cancelled"] is True


# ---------------------------------------------------------------------------
# Round-1 review fixes (codex + code-reviewer)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_concurrent_calls_enqueue_only_once(
    foundation, monkeypatch
) -> None:
    """Codex/code-reviewer round-1 HIGH: two concurrent ``submit`` calls
    with the same ``cache_key`` must produce exactly one ``client.retrieve``
    invocation. The dedupe-→-record_workflow critical section needs a
    per-backend ``asyncio.Lock`` (mirroring CMEMS's ``_async_submit_lock``).
    """
    from copernicus_mcp.backends.cds.backend import CdsBackend

    # Synchronisation: park the first retrieve until both submits have
    # entered. If the lock is correctly placed, the second submit
    # observes the in-flight workflow row and short-circuits — so
    # retrieve fires only once.
    gate = asyncio.Event()
    call_count = {"n": 0}

    def _retrieve_with_gate(*_args, **_kwargs):
        call_count["n"] += 1
        # First call parks; second call should never reach here.
        return _fake_remote(f"req-{call_count['n']}")

    fake_class, sdk = _patch_cdsapi(monkeypatch, retrieve_returns=None)
    sdk.retrieve = MagicMock(side_effect=_retrieve_with_gate)

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    async def _submit_one():
        return await backend.submit(_good_params())

    # Run two concurrent submits with identical params.
    a, b = await asyncio.gather(_submit_one(), _submit_one())
    gate.set()  # noop — for documentation

    # Both responses share the same request_id (one was deduped).
    assert a["request_id"] == b["request_id"]
    # SDK retrieve called exactly once.
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_check_status_finalisation_does_not_overwrite_cancelled(
    foundation, monkeypatch
) -> None:
    """Codex/code-reviewer round-1 HIGH: a row that flipped to ``cancelled``
    between ``check_status``'s remote poll and its commit must NOT be
    overwritten by ``successful``. Atomic ``update_workflow_status_if_pending``
    closes the race."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    cache_key = "ck-race-finalise"
    await _seed_workflow_row(
        foundation, request_id="rid-race", cache_key=cache_key, status="running"
    )
    _, sdk = _patch_cdsapi(
        monkeypatch, get_remote_json={"status": "successful"}
    )

    # Race injection: while the finaliser is downloading, a concurrent
    # cancel commits to the database. We piggy-back on the
    # ``download_results`` side_effect to flip the row mid-finalisation.
    # If the success-commit is unconditional, the row ends up
    # ``successful``; with ``update_workflow_status_if_pending`` it
    # stays ``cancelled``.
    persistence = foundation.persistence

    def _download_then_cancel(request_id: str, target: str) -> str:
        Path(target).write_bytes(b"GRIB-from-cds")
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                persistence.update_workflow_status_if_pending(
                    request_id, "cancelled"
                )
            )
        finally:
            loop.close()
        return target

    sdk.client.download_results = MagicMock(side_effect=_download_then_cancel)

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await backend.check_status("rid-race")
    row = await foundation.persistence.fetch_workflow("rid-race")
    assert row is not None and row["status"] == "cancelled", (
        f"finaliser overwrote cancelled row (status={row['status']!r})"
    )


@pytest.mark.asyncio
async def test_check_status_failed_status_sanitises_remote_message(
    foundation, monkeypatch
) -> None:
    """Codex round-1 HIGH: server-supplied error messages must be passed
    through ``Sanitiser`` before persistence. A credential-shaped UUID
    in the remote error message must be redacted in the workflow row's
    ``error_record_json`` and in the response."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    leak = "abcdef01-2345-6789-abcd-ef0123456789"
    bad_remote = {
        "status": "failed",
        "error": {"message": f"server hiccup: token {leak} rejected"},
    }
    await _seed_workflow_row(foundation, request_id="rid-leak", status="running")
    _patch_cdsapi(monkeypatch, get_remote_json=bad_remote)

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.check_status("rid-leak")
    assert out["status"] == "failed"

    # Persisted error must not contain the raw UUID.
    row = await foundation.persistence.fetch_workflow("rid-leak")
    assert row is not None
    persisted = row["error_record_json"] or ""
    assert leak not in persisted, (
        "credential-shaped UUID leaked into workflow row"
    )
    # Response payload also must not contain it.
    assert leak not in repr(out)


@pytest.mark.asyncio
async def test_submit_validates_dataset_id_against_credential_shapes(
    foundation, monkeypatch
) -> None:
    """Codex round-1 HIGH (defense-in-depth, invariant 6): ``dataset_id``
    flows into the cache_key and the file URI. Reject obviously
    secret-bearing strings (whitespace, embedded ``=``, leading or
    trailing punctuation) before they reach the cache index."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import ValidationError

    sneaky = {
        "dataset_id": "reanalysis-era5 password=hunter2",
        "inputs": {"variable": ["t"], "year": ["2024"]},
    }
    _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote())
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    with pytest.raises(ValidationError):
        await backend.submit(sneaky)


@pytest.mark.asyncio
async def test_finalise_locks_does_not_grow_unboundedly(
    foundation, monkeypatch
) -> None:
    """Code-reviewer round-1 HIGH: each distinct ``request_id`` allocates
    a Lock that must be released after the row is terminal — otherwise
    the dict grows forever in a long-running server."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    _patch_cdsapi(monkeypatch, get_remote_json={"status": "successful"})

    for idx in range(5):
        rid = f"rid-leak-{idx}"
        await _seed_workflow_row(
            foundation, request_id=rid, cache_key=f"ck-leak-{idx}", status="running"
        )
        out = await backend.check_status(rid)
        assert out["status"] == "successful"

    # After 5 successful finalisations, no leftover locks.
    assert len(backend._finalise_locks) == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_submit_record_workflow_failure_deletes_orphan_remote_job(
    foundation, monkeypatch
) -> None:
    """Codex round-1 MEDIUM: if ``record_workflow`` raises after the
    SDK ``retrieve`` call (e.g., disk full, CHECK constraint violation),
    the CDS-server job is orphaned. Best-effort delete keeps the queue
    slot from accumulating."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    _, sdk = _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote("orphan-1"))

    boom = RuntimeError("disk full")

    async def _raise(*_args, **_kwargs):
        raise boom

    monkeypatch.setattr(foundation.persistence, "record_workflow", _raise)

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    with pytest.raises(RuntimeError):
        await backend.submit(_good_params())

    sdk.client.delete.assert_called_once_with("orphan-1")


# ---------------------------------------------------------------------------
# Round-2 review fixes (codex)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_status_non_terminal_poll_does_not_overwrite_cancelled(
    foundation, monkeypatch
) -> None:
    """Codex round-2 HIGH-A: the queued→running transition in
    ``_poll_and_maybe_finalise`` was unconditional. A row that flipped
    to ``cancelled`` between the row fetch and the status commit could
    be silently re-promoted to ``running``.

    Race injection: ``get_remote.json`` flips the row to cancelled
    before the backend writes the new transition.
    """
    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_workflow_row(foundation, request_id="rid-poll-race", status="queued")
    _, sdk = _patch_cdsapi(monkeypatch)

    persistence = foundation.persistence

    class _RaceyJson:
        """A property-like ``.json`` that flips the row to cancelled
        the first time it's read, simulating a concurrent cancel that
        committed between our row fetch and our status update."""

        def __get__(self, instance, owner=None):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    persistence.update_workflow_status_if_pending(
                        "rid-poll-race", "cancelled"
                    )
                )
            finally:
                loop.close()
            return {"status": "running"}

    fake_remote = MagicMock()
    type(fake_remote).json = _RaceyJson()
    sdk.client.get_remote = MagicMock(return_value=fake_remote)

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await backend.check_status("rid-poll-race")

    row = await foundation.persistence.fetch_workflow("rid-poll-race")
    assert row is not None and row["status"] == "cancelled", (
        f"non-terminal poll overwrote cancelled (status={row['status']!r})"
    )


@pytest.mark.asyncio
async def test_submit_concurrent_different_cache_keys_do_not_serialise(
    foundation, monkeypatch
) -> None:
    """Codex round-2 / code-reviewer round-2 HIGH-B: ``_submit_lock`` was
    held across the SDK retrieve HTTP call. Two submits with **different**
    cache_keys must run concurrently — the lock must protect only the
    dedupe→placeholder window, not the network round-trip.

    We pin the property by parking the first retrieve on an event and
    asserting that the second retrieve fires before the first
    finishes.
    """
    from copernicus_mcp.backends.cds.backend import CdsBackend

    started_a = asyncio.Event()

    a_was_running_when_b_called = {"flag": False}
    main_loop = asyncio.get_running_loop()

    def _retrieve_factory(rid: str):
        def _impl(name: str, request: dict, target: Any = None):
            # Park A; let B observe that A is in flight.
            if rid == "req-A":
                # Signal we're inside the SDK call. ``to_thread`` runs us
                # in a worker thread with no event loop, so use the
                # captured main loop's threadsafe API.
                main_loop.call_soon_threadsafe(started_a.set)
                # Wait until B has fired.
                import time

                deadline = time.monotonic() + 2.0
                while not a_was_running_when_b_called["flag"]:
                    if time.monotonic() > deadline:
                        break
                    time.sleep(0.01)
            else:
                # B fires while A is still in retrieve.
                a_was_running_when_b_called["flag"] = True
            return _fake_remote(rid)

        return _impl

    fake_class, sdk = _patch_cdsapi(monkeypatch)
    sdk.retrieve = MagicMock(side_effect=_retrieve_factory("req-A"))

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    params_a = dict(_good_params())
    params_a["dataset_id"] = "reanalysis-era5-single-levels"
    params_b = dict(_good_params())
    params_b["dataset_id"] = "reanalysis-era5-pressure-levels"

    async def _submit_a():
        return await backend.submit(params_a)

    async def _submit_b():
        await started_a.wait()
        # Re-bind the side_effect for B so the second call returns a
        # different request_id.
        sdk.retrieve.side_effect = _retrieve_factory("req-B")
        return await backend.submit(params_b)

    a, b = await asyncio.gather(_submit_a(), _submit_b())
    assert a["request_id"] == "req-A"
    assert b["request_id"] == "req-B"
    assert a_was_running_when_b_called["flag"], (
        "B was blocked on A's lock; submits with different cache_keys "
        "must not serialise on _submit_lock"
    )


@pytest.mark.asyncio
async def test_serialise_error_record_preserves_json_validity_with_sensitive_keys(
    foundation,
) -> None:
    """Code-reviewer / codex round-2: ``_serialise_error_record`` did
    ``model_dump_json() → sanitiser`` which corrupts JSON when the
    sanitiser substitutes a value inside a quoted JSON-string context.
    Order must be: sanitise dict, then ``json.dumps``."""
    import json

    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import BackendError
    from copernicus_mcp.errors.records import build_error_record

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    exc = BackendError(
        "boom",
        record=build_error_record(
            "BackendError",
            message="boom",
            error_subclass="generic",
            recovery_action="retry_with_modification",
            context={"password": "hunter2", "ok_field": "hello"},
        ),
    )

    raw = backend._serialise_error_record(exc)  # type: ignore[attr-defined]
    decoded = json.loads(raw)  # MUST parse — i.e., sanitisation didn't break JSON
    assert "password" in decoded.get("context", {})
    assert decoded["context"]["password"] != "hunter2", "password not redacted"


@pytest.mark.asyncio
async def test_check_status_finalisation_loses_to_cancel_clears_cache(
    foundation, monkeypatch
) -> None:
    """Codex round-2 MEDIUM: when the success-commit loses the race to
    cancel, the cache file was already stored. A later submit for the
    same params returns a cache hit tied to a cancelled workflow.
    The race-loser must invalidate the cache entry."""
    from copernicus_mcp.backends.cds.backend import (
        CdsBackend,
        _cache_storage_key,
    )

    cache_key = "ck-finalise-loser"
    await _seed_workflow_row(
        foundation, request_id="rid-loser", cache_key=cache_key, status="running"
    )
    _, sdk = _patch_cdsapi(
        monkeypatch, get_remote_json={"status": "successful"}
    )

    persistence = foundation.persistence

    def _download_and_concurrent_cancel(request_id: str, target: str) -> str:
        Path(target).write_bytes(b"GRIB-from-cds")
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                persistence.update_workflow_status_if_pending(
                    request_id, "cancelled"
                )
            )
        finally:
            loop.close()
        return target

    sdk.client.download_results = MagicMock(side_effect=_download_and_concurrent_cancel)

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await backend.check_status("rid-loser")

    # The cache entry must have been invalidated; otherwise a subsequent
    # submit would return a cache_hit for a cancelled workflow.
    cached = await foundation.cache.lookup_file(_cache_storage_key(cache_key))
    assert cached is None, (
        "cache entry leaked from a finalisation that lost to cancel"
    )


@pytest.mark.asyncio
async def test_record_terminal_envelope_reflects_actual_row_state(
    foundation, monkeypatch
) -> None:
    """Codex round-2 MEDIUM: ``_record_terminal`` ignored the boolean
    return from ``update_workflow_error_if_pending``. If the conditional
    UPDATE failed (row already cancelled), the envelope still claimed
    ``failed``. The response must reflect the persisted row, not the
    intent."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    cache_key = "ck-terminal-env"
    await _seed_workflow_row(
        foundation, request_id="rid-env", cache_key=cache_key, status="running"
    )

    persistence = foundation.persistence

    class _RaceyJson:
        def __get__(self, instance, owner=None):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    persistence.update_workflow_status_if_pending(
                        "rid-env", "cancelled"
                    )
                )
            finally:
                loop.close()
            return {"status": "failed", "error": {"message": "remote oops"}}

    _, sdk = _patch_cdsapi(monkeypatch)
    fake_remote = MagicMock()
    type(fake_remote).json = _RaceyJson()
    sdk.client.get_remote = MagicMock(return_value=fake_remote)

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.check_status("rid-env")
    # Row should be cancelled (cancel won), envelope should also say cancelled.
    row = await foundation.persistence.fetch_workflow("rid-env")
    assert row is not None and row["status"] == "cancelled"
    assert out["status"] == "cancelled", (
        "envelope claims failed but row is cancelled — response lied"
    )


# ---------------------------------------------------------------------------
# Round-3 review fixes (codex)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_lock_split_when_first_caller_fails_before_recording(
    foundation, monkeypatch
) -> None:
    """Codex round-3 HIGH: round-2's per-key lock pattern pops the dict
    entry on the way out. If task A fails (SDK exception) BEFORE
    inserting the workflow row, task B (already holding a reference to
    lock_v1) and task C (arriving after the pop, getting a fresh
    lock_v2 from setdefault) can both pass the dedupe and both call
    retrieve — duplicate CDS-server jobs.

    Pin the property: A enters lock and fails before recording, then
    B and C both try with the same cache_key. SDK retrieve must be
    called at most ONCE successfully (one for A's failure attempt,
    then exactly one more across {B, C}).
    """
    from copernicus_mcp.backends.cds.backend import CdsBackend

    a_started = asyncio.Event()
    a_can_fail = asyncio.Event()
    main_loop = asyncio.get_running_loop()
    successful_calls: list[str] = []

    def _retrieve_factory(rid: str, *, fail: bool):
        def _impl(name: str, request: dict, target: Any = None):
            if rid == "A":
                main_loop.call_soon_threadsafe(a_started.set)
                # Block until B and C have both arrived in dedupe.
                import time

                deadline = time.monotonic() + 1.0
                while not a_can_fail.is_set() and time.monotonic() < deadline:
                    time.sleep(0.01)
                raise RuntimeError("SDK transient failure simulating A")
            successful_calls.append(rid)
            return _fake_remote(rid)

        return _impl

    _, sdk = _patch_cdsapi(monkeypatch)

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    params = _good_params()

    async def _run_a():
        sdk.retrieve = MagicMock(side_effect=_retrieve_factory("A", fail=True))
        from copernicus_mcp.errors import BackendError

        with pytest.raises(BackendError):
            await backend.submit(params)

    async def _run_b_and_c():
        await a_started.wait()
        # Re-bind to a non-failing impl. Both B and C will attempt
        # retrieve if the lock is split.
        sdk.retrieve = MagicMock(side_effect=_retrieve_factory("BC", fail=False))
        a_can_fail.set()
        results = await asyncio.gather(
            backend.submit(params),
            backend.submit(params),
            return_exceptions=True,
        )
        return results

    a_task = asyncio.create_task(_run_a())
    b_c_task = asyncio.create_task(_run_b_and_c())
    await a_task
    await b_c_task

    # Across A (failed before retrieve return), B, C: at most ONE
    # successful retrieve call. If the lock split, we'd see TWO.
    assert len(successful_calls) <= 1, (
        f"split-lock race: {len(successful_calls)} successful retrieves, "
        "expected at most 1"
    )


@pytest.mark.asyncio
async def test_finalise_race_loss_does_not_invalidate_other_workflow_cache(
    foundation, monkeypatch, tmp_path: Path
) -> None:
    """Codex round-3 MEDIUM: round-2 fix-D invalidates the cache entry
    on race-loss, but ``cache.store_file`` had already overwritten the
    previous content for the same key. If an older successful workflow
    had a valid file at ``cache_key=K``, a force_refresh workflow that
    loses to cancel will (a) overwrite #1's file with #2's, then (b)
    invalidate the slot — destroying #1's cached result.

    Right behaviour: defer ``store_file`` until the conditional UPDATE
    wins. On race-loss, only delete staging.

    Test: pre-seed cache with #1's file, run #2 with force_refresh that
    loses to cancel; assert #1's cache file survives.
    """
    from copernicus_mcp.backends.cds.backend import (
        CdsBackend,
        _cache_storage_key,
    )

    cache_key = "ck-shared"
    # #1: an older successful workflow with a real cached file.
    canned = tmp_path / "old.grib"
    canned.write_bytes(b"PRESERVED-from-workflow-1")
    await foundation.cache.store_file(
        _cache_storage_key(cache_key),
        canned,
        backend_id="cds",
        content_type="application/x-grib",
    )

    # #2: in-flight workflow that will finalise but lose to cancel.
    await _seed_workflow_row(
        foundation, request_id="rid-2", cache_key=cache_key, status="running"
    )
    _, sdk = _patch_cdsapi(monkeypatch, get_remote_json={"status": "successful"})

    persistence = foundation.persistence

    def _download_and_concurrent_cancel(request_id: str, target: str) -> str:
        Path(target).write_bytes(b"NEW-content-from-2")
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                persistence.update_workflow_status_if_pending(
                    request_id, "cancelled"
                )
            )
        finally:
            loop.close()
        return target

    sdk.client.download_results = MagicMock(side_effect=_download_and_concurrent_cancel)

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await backend.check_status("rid-2")

    # #1's cache MUST still be valid and contain the original content.
    cached = await foundation.cache.lookup_file(_cache_storage_key(cache_key))
    assert cached is not None, (
        "#2's race-loss invalidated #1's cache entry — wrong scope"
    )
    assert cached.read_bytes() == b"PRESERVED-from-workflow-1", (
        "#2 overwrote #1's cache content even though it lost the race"
    )


@pytest.mark.asyncio
async def test_submit_locks_dict_is_empty_after_distinct_submits(
    foundation, monkeypatch
) -> None:
    """Code-reviewer round-3 MEDIUM: parallel to the existing
    ``_finalise_locks`` cleanup test — assert ``_submit_locks`` does
    not grow without bound after distinct successful submits."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    for idx in range(5):
        rid = f"req-locks-{idx}"
        _, sdk = _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote(rid))
        params = dict(_good_params())
        # Different inputs ⇒ different cache_keys.
        params["inputs"] = dict(params["inputs"])
        params["inputs"]["year"] = [str(2020 + idx)]
        out = await backend.submit(params)
        assert out["request_id"] == rid

    assert len(backend._submit_locks) == 0, (  # type: ignore[attr-defined]
        f"_submit_locks not bounded: still has {list(backend._submit_locks)}"  # type: ignore[attr-defined]
    )


# ---------------------------------------------------------------------------
# Smoke-test regression: real CDS request_id is UUID-shape (8-4-4-4-12)
# and was getting redacted by the response-envelope sanitiser pass.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_response_preserves_uuid_shape_request_id(
    foundation, monkeypatch
) -> None:
    """T-CDS-000 smoke regression: the real CDS server returns
    UUID-shape request_ids (e.g. ``4f3d2c1b-9e8a-4c7d-8e9f-0a1b2c3d4e5f``).
    The sanitiser pattern redacts UUIDs because PATs are UUID-shape;
    the response envelope must NOT redact the request_id otherwise the
    caller has no handle for poll/download/cancel."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    real_uuid = "4f3d2c1b-9e8a-4c7d-8e9f-0a1b2c3d4e5f"
    _, sdk = _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote(real_uuid))

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.submit(_good_params())
    assert out["request_id"] == real_uuid, (
        f"sanitiser redacted UUID-shape request_id: got {out['request_id']!r}"
    )
    # The URI also embeds it; must not be redacted.
    assert real_uuid in out["result"]["uri"]


@pytest.mark.asyncio
async def test_check_status_response_preserves_uuid_shape_request_id(
    foundation, monkeypatch
) -> None:
    """Same property for ``check_status`` — the running envelope must
    surface the original UUID request_id verbatim."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    real_uuid = "4f3d2c1b-9e8a-4c7d-8e9f-0a1b2c3d4e5f"
    await _seed_workflow_row(foundation, request_id=real_uuid, status="queued")
    _patch_cdsapi(monkeypatch, get_remote_json={"status": "running"})

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.check_status(real_uuid)
    assert out["request_id"] == real_uuid, (
        f"sanitiser redacted UUID-shape request_id in check_status response: "
        f"got {out['request_id']!r}"
    )


# ---------------------------------------------------------------------------
# Round-4 codex async re-review fixes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_status_store_file_failure_reverts_row_to_failed(
    foundation, monkeypatch
) -> None:
    """Round-4 HIGH (codex async): if ``cache.store_file`` raises after
    ``update_workflow_status_if_pending(_, "successful")`` has already
    committed, the row was stuck at ``successful`` with no cache file.
    Subsequent ``check_status`` would short-circuit at the terminal
    branch and return a synthetic ``cache_eviction`` forever — user
    has no path to recovery.

    Fix: on ``store_file`` failure, unconditionally revert the row to
    ``failed`` with the error record so a follow-up ``submit`` can
    re-fetch.
    """
    from copernicus_mcp.backends.cds.backend import CdsBackend

    cache_key = "ck-store-fail"
    await _seed_workflow_row(
        foundation, request_id="rid-sf", cache_key=cache_key, status="running"
    )
    _, sdk = _patch_cdsapi(monkeypatch, get_remote_json={"status": "successful"})

    # Force ``cache.store_file`` to blow up (e.g. disk full).
    async def _broken_store_file(*args, **kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(foundation.cache, "store_file", _broken_store_file)

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    # T-CDS-ASYNC-DOWNLOAD: surfaced as a failed envelope (within the inline grace),
    # not raised — the row must still settle to ``failed``.
    out = await backend.check_status("rid-sf")
    assert out["status"] == "failed"

    # Row must NOT be stuck at successful.
    row = await foundation.persistence.fetch_workflow("rid-sf")
    assert row is not None and row["status"] == "failed", (
        f"row stuck at {row['status']!r} after store_file failure — "
        "user cannot recover"
    )
    assert row["error_record_json"] is not None


@pytest.mark.asyncio
async def test_submit_orphan_cleanup_runs_when_record_raises_cancelled(
    foundation, monkeypatch
) -> None:
    """Round-4 MEDIUM (codex async) — case 1: ``record_workflow``
    itself raises ``CancelledError``. The previous ``except Exception``
    form would have skipped cleanup; the new ``try/finally`` runs it.
    """
    from copernicus_mcp.backends.cds.backend import CdsBackend

    _, sdk = _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote("orphan-x"))

    async def _cancel_at_record(*_args, **_kwargs):
        raise asyncio.CancelledError("simulated cancel mid-record")

    monkeypatch.setattr(foundation.persistence, "record_workflow", _cancel_at_record)

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    with pytest.raises(asyncio.CancelledError):
        await backend.submit(_good_params())

    sdk.client.delete.assert_called_once_with("orphan-x")


@pytest.mark.asyncio
async def test_submit_orphan_cleanup_survives_outer_task_cancellation(
    foundation, monkeypatch
) -> None:
    """Round-4 MEDIUM (codex async) — case 2: the outer asyncio Task
    is cancelled while the coroutine is awaiting inside
    ``record_workflow``. The ``asyncio.shield`` around the cleanup
    must keep ``client.client.delete`` running to completion even
    though the outer ``await`` raises ``CancelledError``.

    Code-reviewer round-4 follow-up: test_orphan_cleanup_runs_when_record_raises_cancelled
    above does not actually exercise the ``asyncio.shield`` — the
    cleanup ``await`` completes normally because no outer cancel is
    active at finally-time. This test pins the shield's value:
    without it, ``delete`` is also cancelled and never invokes the
    Mock.
    """
    from copernicus_mcp.backends.cds.backend import CdsBackend

    _, sdk = _patch_cdsapi(monkeypatch, retrieve_returns=_fake_remote("orphan-y"))

    record_started = asyncio.Event()
    record_can_finish = asyncio.Event()

    async def _blocking_record(*_args, **_kwargs):
        record_started.set()
        # Park here until externally cancelled.
        await record_can_finish.wait()

    monkeypatch.setattr(foundation.persistence, "record_workflow", _blocking_record)

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    submit_task = asyncio.create_task(backend.submit(_good_params()))

    await record_started.wait()
    submit_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await submit_task

    # Give the shielded cleanup a moment to schedule + run on the loop.
    for _ in range(20):
        if sdk.client.delete.called:
            break
        await asyncio.sleep(0.01)

    sdk.client.delete.assert_called_once_with("orphan-y")


# ---------------------------------------------------------------------------
# T-CDS-CHUNK fan-out confirmation tiers: confirm > N, repeat-confirm > M
# ---------------------------------------------------------------------------


def _make_foundation_low_fanout(tmp_path: Path):
    """Foundation with low auto-chunk fan-out thresholds (confirm>2, reconfirm>4)
    so a 5-chunk split exercises both confirmation tiers without 100+ children."""
    from copernicus_mcp.auth import CredentialResolver
    from copernicus_mcp.backends.abstract import FoundationServices
    from copernicus_mcp.cache import CacheManager
    from copernicus_mcp.config import ConfigLoader
    from copernicus_mcp.data_model.coordinator import DataModelCoordinator
    from copernicus_mcp.data_model.provenance import ProvenanceRecorder
    from copernicus_mcp.errors.sanitiser import Sanitiser
    from copernicus_mcp.http import HttpClientFactory
    from copernicus_mcp.persistence import SqliteBackend

    config = ConfigLoader().load(
        cli_overrides={
            "budget": {
                "cds_auto_chunk_confirm_above": 2,
                "cds_auto_chunk_reconfirm_above": 4,
            }
        }
    )
    persistence = SqliteBackend(tmp_path / "state.db")
    cache = CacheManager(
        cache_directory=tmp_path / "cache",
        persistence=persistence,
        size_limit_bytes=10 * 1024 * 1024,
    )
    foundation = FoundationServices(
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
    )
    return foundation, persistence


@pytest_asyncio.fixture
async def low_fanout_backend(tmp_path: Path):
    from copernicus_mcp.backends.cds.backend import CdsBackend

    foundation, persistence = _make_foundation_low_fanout(tmp_path)
    await persistence.initialise()
    try:
        yield CdsBackend(foundation=foundation, credentials=_fake_creds())
    finally:
        await persistence.close()


@pytest.mark.asyncio
async def test_fanout_over_confirm_threshold_requires_confirmation(
    low_fanout_backend, monkeypatch
) -> None:
    """A 5-chunk plan (> confirm_above=2), not confirmed → ConfirmationRequired
    carrying the job count; no cdsapi job is created."""
    from copernicus_mcp.workflow.confirmation import ConfirmationRequired

    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    fake_class, _ = _patch_cdsapi_seq(monkeypatch)

    params = _splittable_params()
    params["__options"] = {"chunk_by": "year"}
    with pytest.raises(ConfirmationRequired) as exc:
        await low_fanout_backend.submit(params)

    payload = exc.value.payload
    assert payload["reason"] == "auto_chunk_job_count"
    assert payload["chunk_count"] == 5
    assert payload["context"]["confirm_threshold"] == 2
    fake_class.assert_not_called()


@pytest.mark.asyncio
async def test_fanout_confirmed_below_reconfirm_still_reconfirms(
    low_fanout_backend, monkeypatch
) -> None:
    """confirmed=true clears the first tier, but 5 chunks (> reconfirm_above=4)
    demands the SECOND, deliberate ack — confirmed alone is not enough."""
    from copernicus_mcp.workflow.confirmation import ConfirmationRequired

    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    fake_class, _ = _patch_cdsapi_seq(monkeypatch)

    params = _splittable_params()
    params["__options"] = {"chunk_by": "year", "confirmed": True}
    with pytest.raises(ConfirmationRequired) as exc:
        await low_fanout_backend.submit(params)

    payload = exc.value.payload
    assert payload["reason"] == "auto_chunk_job_count_large"
    assert payload["chunk_count"] == 5
    assert payload["context"]["confirm_threshold"] == 4
    fake_class.assert_not_called()


@pytest.mark.asyncio
async def test_fanout_both_acks_submits_all_children(
    low_fanout_backend, monkeypatch
) -> None:
    """confirmed=true AND confirm_large_fanout=true → both tiers cleared, every
    child submitted."""
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, {})

    params = _splittable_params()
    params["__options"] = {
        "chunk_by": "year",
        "confirmed": True,
        "confirm_large_fanout": True,
    }
    out = await low_fanout_backend.submit(params)
    assert out["chunked"] is True
    assert out["chunk_count"] == 5
    assert sdk.retrieve.call_count == 5


@pytest.mark.asyncio
async def test_fanout_at_threshold_does_not_gate(low_fanout_backend, monkeypatch) -> None:
    """Boundary: the gates are strict ``>``, so n == confirm_above (=2) does not
    gate — a 2-chunk split submits directly."""
    _patch_costing_by_shape(monkeypatch, per_year=300.0, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, {})

    params = {
        "dataset_id": "derived-era5-single-levels-daily-statistics",
        "inputs": {
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature"],
            "year": ["2020", "2021"],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
        },
        "__options": {"chunk_by": "year"},
    }
    out = await low_fanout_backend.submit(params)
    assert out["chunked"] is True
    assert out["chunk_count"] == 2
    assert sdk.retrieve.call_count == 2


@pytest.mark.asyncio
async def test_fanout_large_ack_alone_does_not_skip_tier1(
    low_fanout_backend, monkeypatch
) -> None:
    """confirm_large_fanout WITHOUT confirmed cannot skip tier 1: a 5-chunk plan
    still raises the first-tier confirmation (tier 1 keys only on confirmed)."""
    from copernicus_mcp.workflow.confirmation import ConfirmationRequired

    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    fake_class, _ = _patch_cdsapi_seq(monkeypatch)

    params = _splittable_params()
    params["__options"] = {"chunk_by": "year", "confirm_large_fanout": True}
    with pytest.raises(ConfirmationRequired) as exc:
        await low_fanout_backend.submit(params)
    assert exc.value.payload["reason"] == "auto_chunk_job_count"
    fake_class.assert_not_called()


def test_chunk_parent_response_includes_progress() -> None:
    """T-DOWNLOAD-PROGRESS: the chunked-parent status carries an explicit
    completed/total progress block (download position) over the per-state counts."""
    from copernicus_mcp.backends.cds.backend import _chunk_parent_response

    plan = {"chunks": [{"index": i, "child_request_id": f"c{i}"} for i in range(5)]}
    child_status = {"c0": "successful", "c1": "successful", "c2": "running", "c3": "running"}
    resp = _chunk_parent_response(
        parent_id="p",
        cache_key="k",
        plan=plan,
        child_status=child_status,
        status="running",
    )
    assert resp["progress"] == {"completed": 2, "total": 5}
    assert resp["chunks"]["successful"] == 2


# ---------------------------------------------------------------------------
# T-CDS-ASYNC-DOWNLOAD: the file fetch is backgrounded so check_status does not
# block the agent on the transfer.
# ---------------------------------------------------------------------------


def _make_foundation_grace(tmp_path: Path, grace: float):
    """Foundation with a tiny download inline-grace so a gated download reliably
    exceeds it and backgrounds."""
    from copernicus_mcp.auth import CredentialResolver
    from copernicus_mcp.backends.abstract import FoundationServices
    from copernicus_mcp.cache import CacheManager
    from copernicus_mcp.config import ConfigLoader
    from copernicus_mcp.data_model.coordinator import DataModelCoordinator
    from copernicus_mcp.data_model.provenance import ProvenanceRecorder
    from copernicus_mcp.errors.sanitiser import Sanitiser
    from copernicus_mcp.http import HttpClientFactory
    from copernicus_mcp.persistence import SqliteBackend

    config = ConfigLoader().load(
        cli_overrides={"budget": {"cds_download_inline_grace_seconds": grace}}
    )
    persistence = SqliteBackend(tmp_path / "state.db")
    cache = CacheManager(
        cache_directory=tmp_path / "cache",
        persistence=persistence,
        size_limit_bytes=10 * 1024 * 1024,
    )
    foundation = FoundationServices(
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
    )
    return foundation, persistence


@pytest_asyncio.fixture
async def grace_backend(tmp_path: Path):
    from copernicus_mcp.backends.cds.backend import CdsBackend

    foundation, persistence = _make_foundation_grace(tmp_path, grace=0.05)
    await persistence.initialise()
    try:
        yield CdsBackend(foundation=foundation, credentials=_fake_creds())
    finally:
        await persistence.close()


def _gated_download(gate, calls=None):
    """A ``download_results`` stub that blocks on ``gate`` until the test releases
    it (so the download outlasts the inline grace), then writes the file."""

    def _dl(request_id, target):
        if calls is not None:
            calls["n"] += 1
        gate.wait(timeout=5)
        Path(target).write_bytes(b"grib-bytes")
        return target

    return _dl


@pytest.mark.asyncio
async def test_check_status_backgrounds_slow_download(grace_backend, monkeypatch) -> None:
    """A download slower than the inline grace is backgrounded: check_status returns
    immediately (status running / phase downloading) instead of blocking on the
    transfer; a later poll returns successful. No double-spawn."""
    import threading

    backend = grace_backend
    await _seed_workflow_row(
        backend.foundation, request_id="rid-bg", cache_key="ck-bg", status="running"
    )
    gate = threading.Event()
    calls = {"n": 0}
    _, sdk = _patch_cdsapi(monkeypatch, get_remote_json={"status": "successful"})
    sdk.client.download_results = MagicMock(side_effect=_gated_download(gate, calls))

    out1 = await backend.check_status("rid-bg")
    assert out1["status"] == "running"
    assert out1["phase"] == "downloading"
    task = backend._downloads.get("rid-bg")
    assert task is not None and not task.done()

    # A second poll while still downloading → still downloading, not re-spawned.
    out_mid = await backend.check_status("rid-bg")
    assert out_mid.get("phase") == "downloading"
    assert calls["n"] == 1

    gate.set()
    await task

    out2 = await backend.check_status("rid-bg")
    assert out2["status"] == "successful"
    assert "filepath" in out2["result"]
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_cancel_cancels_inflight_download(grace_backend, monkeypatch) -> None:
    """A cancel during a backgrounded download removes the in-flight task and
    settles the row to cancelled (best-effort, gotcha #8)."""
    import contextlib as _ctx
    import threading

    backend = grace_backend
    await _seed_workflow_row(
        backend.foundation, request_id="rid-cx", cache_key="ck-cx", status="running"
    )
    gate = threading.Event()
    _, sdk = _patch_cdsapi(monkeypatch, get_remote_json={"status": "successful"})
    sdk.client.download_results = MagicMock(side_effect=_gated_download(gate))

    out1 = await backend.check_status("rid-cx")
    assert out1["phase"] == "downloading"
    task = backend._downloads.get("rid-cx")
    assert task is not None and not task.done()

    await backend.cancel("rid-cx")
    assert "rid-cx" not in backend._downloads
    row = await backend.foundation.persistence.fetch_workflow("rid-cx")
    assert row is not None and row["status"] == "cancelled"

    # Regression (review HIGH): a poll after cancel must NOT re-spawn the download —
    # cancel commits ``cancelled`` first, so check_status short-circuits the terminal
    # row instead of starting a second concurrent transfer.
    out2 = await backend.check_status("rid-cx")
    assert out2["status"] == "cancelled"
    assert sdk.client.download_results.call_count == 1

    # Release the worker thread + drain the cancelled task (best-effort cleanup).
    gate.set()
    with _ctx.suppress(asyncio.CancelledError, Exception):
        await task
    # Review MEDIUM: the cancelled download's staging dir is cleaned — no leaked file.
    staging_root = backend.foundation.cache.cache_zone_for("cds") / ".staging"
    leftover_files = (
        [p for p in staging_root.rglob("*") if p.is_file()]
        if staging_root.exists()
        else []
    )
    assert leftover_files == [], leftover_files
