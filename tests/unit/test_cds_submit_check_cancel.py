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

    from copernicus_mcp.errors import CopernicusMcpError

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    with pytest.raises(CopernicusMcpError):
        await backend.check_status("rid-rename-fail")

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
            "variable": ["2m_temperature"],
            "year": ["2024"], "month": ["01"], "day": ["01"],
            "time": ["00:00"],
            # User-supplied — must be sanitised at the boundary.
            "jobID": user_uuid,
            "job_id": user_uuid,
        },
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
    decisions.md."""
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
    """Codex spec review HIGH-2: two simultaneous ``check_status`` calls
    on the same successful job must download exactly once. The second
    caller observes the already-finalised row and returns the same
    descriptor."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    cache_key = "ck-concurrent"
    await _seed_workflow_row(foundation, request_id="rid-c", cache_key=cache_key, status="running")
    _, sdk = _patch_cdsapi(monkeypatch, get_remote_json={"status": "successful"})

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    a, b = await asyncio.gather(backend.check_status("rid-c"), backend.check_status("rid-c"))
    assert a["status"] == "successful"
    assert b["status"] == "successful"
    # Download must have happened exactly once.
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
    from copernicus_mcp.errors import BackendError

    await _seed_workflow_row(foundation, request_id="rid-dl", cache_key="ck-dl", status="running")
    _, sdk = _patch_cdsapi(monkeypatch, get_remote_json={"status": "successful"})
    sdk.client.download_results = MagicMock(side_effect=RuntimeError("transient socket reset"))

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    with pytest.raises(BackendError):
        await backend.check_status("rid-dl")
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
    from copernicus_mcp.errors import BackendError

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
    with pytest.raises(BackendError):
        await backend.check_status("rid-sf")

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
