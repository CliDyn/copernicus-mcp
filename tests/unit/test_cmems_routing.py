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


def _params() -> dict[str, Any]:
    return dict(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        variables=["thetao"],
        minimum_longitude=-1.0,
        maximum_longitude=1.0,
        minimum_latitude=0.0,
        maximum_latitude=1.0,
        minimum_depth=0.0,
        maximum_depth=10.0,
        start_datetime="2024-01-01T00:00:00Z",
        end_datetime="2024-01-02T00:00:00Z",
    )


def _install_fake(monkeypatch, *, subset_fn):
    mod = types.ModuleType("copernicusmarine")

    def _wrapper(**kwargs):
        if not kwargs.get("dry_run"):
            outdir = Path(kwargs["output_directory"])
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / kwargs["output_filename"]).write_bytes(b"nc")
        return subset_fn(**kwargs)

    mod.subset = _wrapper  # type: ignore[attr-defined]
    mod.describe = lambda **kw: {"products": []}  # type: ignore[attr-defined]

    class LoginError(Exception):
        pass

    class DatasetNotFound(Exception):
        pass

    class WrongFormatRequested(Exception):
        pass

    mod.LoginError = LoginError  # type: ignore[attr-defined]
    mod.DatasetNotFound = DatasetNotFound  # type: ignore[attr-defined]
    mod.WrongFormatRequested = WrongFormatRequested  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "copernicusmarine", mod)
    return mod


# --- validate ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_valid_params(foundation) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    out = await backend.validate(_params())
    assert out["valid"] is True
    assert "errors" not in out or not out["errors"]


@pytest.mark.asyncio
async def test_validate_invalid_params(foundation) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    bad = _params()
    bad["minimum_latitude"] = 99.0  # > 90
    bad["maximum_latitude"] = 1.0  # min > max
    out = await backend.validate(bad)
    assert out["valid"] is False
    assert out["errors"]


@pytest.mark.asyncio
async def test_validate_does_not_echo_credentials_in_errors(foundation) -> None:
    """codex T-026 HIGH: validate() must not surface a password-shaped value
    that snuck into a string field via raw user input."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    bad = _params()
    # Credential-shaped junk in a datetime string (the worst case).
    bad["start_datetime"] = "password=hunter2"
    out = await backend.validate(bad)
    assert out["valid"] is False
    serialised = str(out)
    assert "hunter2" not in serialised


@pytest.mark.asyncio
async def test_validate_recognises_get_shape(foundation) -> None:
    """T-CMEMS-GET-001: a request shaped for ``get_files`` (no
    subset-mandatory fields, but valid ``dataset_id`` + optional
    selection filter) validates as ``CmemsGetRequest`` and
    returns ``{valid: True}``."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    out = await backend.validate(
        {
            "dataset_id": "cmems_obs-ins_glo_phy-temp-sal_my_easycora_irr",
            "filter": "*1990*",
        }
    )
    assert out["valid"] is True


@pytest.mark.asyncio
async def test_validate_rejects_get_shape_with_two_selection_filters(
    foundation,
) -> None:
    """T-CMEMS-GET-001: get-shape with both ``filter`` and
    ``regex`` set → not valid; canonical ValidationError mapped
    to ``{valid: False, errors: [...]}``."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    out = await backend.validate(
        {
            "dataset_id": "x",
            "filter": "*1990*",
            "regex": ".*1991.*",
        }
    )
    assert out["valid"] is False
    assert out["errors"]


@pytest.mark.asyncio
async def test_validate_rejects_get_shape_with_blank_dataset_id(
    foundation,
) -> None:
    """T-CMEMS-GET-001: get-shape with blank ``dataset_id`` is
    invalid even with selection filters set."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    out = await backend.validate({"dataset_id": "", "filter": "*1990*"})
    assert out["valid"] is False
    assert out["errors"]


@pytest.mark.asyncio
async def test_validate_routes_subset_payload_missing_only_variables(
    foundation,
) -> None:
    """cr+codex round-1 MEDIUM: a subset request that's missing
    ``variables`` (typo, dropped field) should NOT be treated as
    a get request. Any subset-only field present in the payload
    (bbox / time / depth) flags it as subset-shaped — so the
    error message points at the actual problem ("variables
    required") rather than at 8 ``extra_forbidden`` noise lines."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    bad = _params()
    bad.pop("variables")
    out = await backend.validate(bad)
    assert out["valid"] is False
    # Exactly one missing-field error pointing at 'variables', not
    # 8 extra-forbidden noise lines.
    serialised = str(out).lower()
    assert "variables" in serialised
    assert "extra_forbidden" not in serialised


@pytest.mark.asyncio
async def test_validate_get_shape_does_not_leak_credential_shaped_input(
    foundation,
) -> None:
    """cr+codex round-1 LOW: the existing subset path proved
    Pydantic errors echo raw input. Pin the same defence for the
    new get-shape path."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    out = await backend.validate(
        {
            "dataset_id": "x",
            "filter": "password=hunter2-leak-canary",
            "regex": ".*",
        }
    )
    assert out["valid"] is False
    assert "hunter2-leak-canary" not in str(out)


@pytest.mark.asyncio
async def test_validate_routes_subset_payload_with_only_file_format(
    foundation,
) -> None:
    """cr round-2 MEDIUM: round-1 widened the discriminator to
    subset-mandatory fields (bbox/time/depth) but missed
    subset-only optional fields (``file_format``, ``service``,
    ``coordinates_selection_method``, ``netcdf_compression_level``,
    ``dataset_part``). A payload like
    ``{"dataset_id": "x", "file_format": "netcdf"}`` still landed
    on the get path and emitted noisy ``extra_forbidden`` instead
    of a clean subset error."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    out = await backend.validate({"dataset_id": "x", "file_format": "netcdf"})
    assert out["valid"] is False
    # Should NOT emit `extra_forbidden` on `file_format`: the request
    # is recognised as subset-shape and the relevant subset errors fire.
    assert "extra_forbidden" not in str(out).lower()


def test_subset_discriminator_covers_every_subset_only_field() -> None:
    """codex round-2 LOW: pin set parity so a future required
    subset-only field cannot be added to ``CmemsSubsetRequest``
    without updating ``_SUBSET_DISCRIMINATOR_FIELDS``."""
    from copernicus_mcp.backends.cmems.backend import _SUBSET_DISCRIMINATOR_FIELDS
    from copernicus_mcp.data_model.schemas_cmems import (
        CmemsGetRequest,
        CmemsSubsetRequest,
    )

    subset_only = set(CmemsSubsetRequest.model_fields) - set(CmemsGetRequest.model_fields)
    # cr+codex round-3 LOW: assert exact equality, not just
    # ``subset_only ⊆ discriminator``. A future refactor that
    # hand-curates the frozenset back instead of deriving it
    # could otherwise add a spurious field without tripping
    # this guard.
    assert subset_only == _SUBSET_DISCRIMINATOR_FIELDS, (
        f"discriminator drift: subset_only={sorted(subset_only)} vs "
        f"discriminator={sorted(_SUBSET_DISCRIMINATOR_FIELDS)}"
    )


@pytest.mark.asyncio
async def test_validate_does_not_call_toolbox(foundation, monkeypatch) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    n = {"calls": 0}

    def fake(**kwargs):
        n["calls"] += 1
        return None

    _install_fake(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    await backend.validate(_params())
    assert n["calls"] == 0


# --- check_status -----------------------------------------------------------


@pytest.mark.asyncio
async def test_check_status_known_request(foundation, tmp_path) -> None:
    """T-039 round 3: check_status now verifies the cache file actually
    exists when ``status=successful``; if the file was evicted, the call
    surfaces a synthetic ``cache_eviction`` failure. Plant a real cache
    entry + file so the row reflects a usable success."""
    from datetime import UTC, datetime

    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    cache_key = "cmems:submit:x:abc"
    cache_file = tmp_path / "fake.nc"
    cache_file.write_bytes(b"fake-netcdf")
    await foundation.persistence.record_cache_entry(
        {
            "namespace": "file",
            "key": f"file:{cache_key}",
            "value_json": "{}",
            "file_path": str(cache_file),
            "size_bytes": cache_file.stat().st_size,
            "content_type": "application/x-netcdf",
            "created_at": now,
            "last_accessed_at": now,
        }
    )
    await foundation.persistence.record_workflow(
        {
            "request_id": "req-test-1",
            "backend_id": "cmems",
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
    out = await backend.check_status("req-test-1")
    assert out["status"] == "successful"
    assert out["request_id"] == "req-test-1"
    assert out["cache_key"] == "cmems:submit:x:abc"
    assert "submitted_at" in out
    assert "updated_at" in out


@pytest.mark.asyncio
async def test_check_status_unknown_raises_not_found(foundation) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import NotFoundError

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(NotFoundError):
        await backend.check_status("does-not-exist")


# --- cancel -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_running_workflow(foundation) -> None:
    from datetime import UTC, datetime

    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    await foundation.persistence.record_workflow(
        {
            "request_id": "req-running",
            "backend_id": "cmems",
            "operation": "submit",
            "status": "running",
            "cache_key": "k",
            "request_json": "{}",
            "response_json": None,
            "error_record_json": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    out = await backend.cancel("req-running")
    assert out["cancelled"] is True
    row = await foundation.persistence.fetch_workflow("req-running")
    assert row is not None and row["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_terminal_no_op(foundation) -> None:
    from datetime import UTC, datetime

    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    await foundation.persistence.record_workflow(
        {
            "request_id": "req-done",
            "backend_id": "cmems",
            "operation": "submit",
            "status": "successful",
            "cache_key": "k",
            "request_json": "{}",
            "response_json": None,
            "error_record_json": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    out = await backend.cancel("req-done")
    assert out["cancelled"] is False
    assert "already terminal" in out["reason"].lower()
    row = await foundation.persistence.fetch_workflow("req-done")
    assert row is not None and row["status"] == "successful"


@pytest.mark.asyncio
async def test_cancel_unknown_raises_not_found(foundation) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import NotFoundError

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(NotFoundError):
        await backend.cancel("does-not-exist")


# --- fetch_result -----------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_result_re_emits_large_data_result(
    foundation, monkeypatch, tmp_path: Path
) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    _install_fake(
        monkeypatch,
        subset_fn=lambda **k: types.SimpleNamespace(
            file_size=0.5,
            data_transfer_size=0.5,
            status="DRY_RUN",
            variables=["thetao"],
            service="arco-geo-series",
        ),
    )
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    res = await backend.submit(_params())

    fetched = await backend.fetch_result(res["request_id"], tmp_path / "ignored")
    assert fetched["status"] == "successful"
    assert fetched["request_id"] == res["request_id"]
    assert fetched["result"]["filepath"] == res["result"]["filepath"]


@pytest.mark.asyncio
async def test_fetch_result_unknown_raises(foundation, tmp_path: Path) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import NotFoundError

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(NotFoundError):
        await backend.fetch_result("nope", tmp_path / "x")


@pytest.mark.asyncio
async def test_fetch_result_non_successful_raises(foundation) -> None:
    from datetime import UTC, datetime

    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import BackendError

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    await foundation.persistence.record_workflow(
        {
            "request_id": "req-fail",
            "backend_id": "cmems",
            "operation": "submit",
            "status": "failed",
            "cache_key": "k",
            "request_json": "{}",
            "response_json": None,
            "error_record_json": None,
            "created_at": now,
            "updated_at": now,
        }
    )
    with pytest.raises(BackendError):
        await backend.fetch_result("req-fail", Path("/tmp/x"))
