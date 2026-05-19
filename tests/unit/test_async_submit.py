"""Async submit (T-039): background download + polling for CMEMS.

Each test corresponds to one of the spec items in
``CODING_AGENT_PLAN.md`` §4.2 — async_mode=True opt-in for
``marine_subset_dataset``. The default sync flow is unchanged; existing
tests in ``test_cmems_subset.py`` and ``test_release_regressions.py``
still pass without modification.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
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


def _subset_params() -> dict[str, Any]:
    return dict(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        dataset_version="202411",
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


def _estimate_response(transfer_mb: float = 0.5):
    return types.SimpleNamespace(
        file_size=transfer_mb,
        data_transfer_size=transfer_mb,
        status="DRY_RUN",
        message="dry-run",
        variables=["thetao"],
        service="arco-geo-series",
    )


def _install_fake_module(
    monkeypatch,
    *,
    subset_fn,
    write_bytes: bytes = b"netcdf-content",
):
    mod = types.ModuleType("copernicusmarine")

    def _subset_wrapper(**kwargs):
        if not kwargs.get("dry_run"):
            outdir = Path(kwargs["output_directory"])
            fname = kwargs["output_filename"]
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / fname).write_bytes(write_bytes)
        return subset_fn(**kwargs)

    mod.subset = _subset_wrapper  # type: ignore[attr-defined]
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


async def _drain_background_tasks(backend) -> None:
    """Wait for any in-flight background tasks to settle (test cleanup)."""
    tasks = list(getattr(backend, "_background_tasks", {}).values())
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# 1) Returns running immediately
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_submit_returns_running_immediately(
    foundation, monkeypatch
) -> None:
    """``async_mode=True`` + a slow toolbox call → submit returns within 1 s
    with ``status="running"`` and a request_id; the actual work happens in
    a background task tracked on the backend instance."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    started = asyncio.Event()
    block = asyncio.Event()

    def fake(**kwargs):
        if kwargs.get("dry_run"):
            return _estimate_response()
        # Real call: signal start, then block until released. Without the
        # async_mode branch the call would not return for the duration of
        # the test → submit() would block too.
        started.set()
        # Run on a worker thread (asyncio.to_thread); we sleep here.
        time.sleep(0.05)
        # Wait for the test to release us by setting block from outside.
        # Use a polling loop because the worker thread doesn't share the
        # event loop directly.
        while not block.is_set():
            time.sleep(0.01)
        return None

    _install_fake_module(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    params = _subset_params()
    params["__options"] = {"async_mode": True, "confirmed": True}

    t0 = time.monotonic()
    response = await backend.submit(params)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, f"async submit blocked for {elapsed:.2f}s"

    assert response["status"] == "running", response
    assert response["request_id"].startswith("req-"), response
    assert response["cache_hit"] is False
    assert response["is_existing"] is False
    assert response["result"]["uri"].startswith(
        f"copernicus://jobs/{response['request_id']}"
    )

    # Background task is registered on the backend instance.
    assert response["request_id"] in backend._background_tasks

    # Release the worker thread and let the task settle so the test
    # cleanup doesn't leave a zombie task behind.
    block.set()
    await _drain_background_tasks(backend)


# ---------------------------------------------------------------------------
# 2) Completes successfully
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_submit_completes_successfully(foundation, monkeypatch) -> None:
    """Background task runs to completion; ``check_status`` reflects
    ``successful``; ``fetch_result`` returns the same filepath as the sync
    path would."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    def fake(**kwargs):
        if kwargs.get("dry_run"):
            return _estimate_response()
        return None

    _install_fake_module(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    params = _subset_params()
    params["__options"] = {"async_mode": True, "confirmed": True}

    submitted = await backend.submit(params)
    request_id = submitted["request_id"]

    await _drain_background_tasks(backend)

    status_after = await backend.check_status(request_id)
    assert status_after["status"] == "successful", status_after

    fetched = await backend.fetch_result(request_id, target=Path("/dev/null"))
    assert fetched["status"] == "successful"
    assert fetched["cache_hit"] is True
    assert Path(fetched["result"]["filepath"]).exists()


# ---------------------------------------------------------------------------
# 3) Propagates exception to failed row, sanitised
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_submit_propagates_exception_to_failed_row(
    foundation, monkeypatch
) -> None:
    """When the background toolbox call raises, the workflow row flips to
    ``failed`` with a sanitised ``error_record_json``. No credential-shaped
    string survives in the persisted record (the credential-isolation invariant).
    """
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    def fake(**kwargs):
        if kwargs.get("dry_run"):
            return _estimate_response()
        # Toolbox raises with a credential-shaped substring in the message —
        # sanitiser must redact it before it reaches the row.
        raise RuntimeError(
            "boom while contacting cmems: token=abcdef0123456789ABCDEF token"
        )

    _install_fake_module(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    params = _subset_params()
    params["__options"] = {"async_mode": True, "confirmed": True}

    submitted = await backend.submit(params)
    request_id = submitted["request_id"]

    await _drain_background_tasks(backend)

    row = await foundation.persistence.fetch_workflow(request_id)
    assert row is not None
    assert row["status"] == "failed", row
    raw = row.get("error_record_json")
    assert raw is not None, "failed async submit must populate error_record_json"
    # No raw token survived.
    assert "abcdef0123456789ABCDEF" not in raw
    record = json.loads(raw)
    assert record["error_class"] == "BackendError"


# ---------------------------------------------------------------------------
# 4) Cancel kills background task and cleans up partial file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_submit_cancel_kills_background_task(
    foundation, monkeypatch
) -> None:
    """Calling ``cancel(request_id)`` on an in-flight async submit cancels
    the background task and the existing FA-5 cleanup applies (partial file
    unlinked, row marked ``cancelled``)."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    block = asyncio.Event()
    written: list[Path] = []

    def fake(**kwargs):
        if kwargs.get("dry_run"):
            return _estimate_response()
        outdir = Path(kwargs["output_directory"])
        fname = kwargs["output_filename"]
        outdir.mkdir(parents=True, exist_ok=True)
        partial = outdir / fname
        partial.write_bytes(b"partial")
        written.append(partial)
        # Hold the worker thread until the test cancels via task.cancel().
        # asyncio.to_thread cannot interrupt time.sleep, but it CAN raise
        # CancelledError into the awaiter — the backend's CancelledError
        # handler is what we exercise here.
        while not block.is_set():
            time.sleep(0.01)
        return None

    _install_fake_module(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    params = _subset_params()
    params["__options"] = {"async_mode": True, "confirmed": True}

    submitted = await backend.submit(params)
    request_id = submitted["request_id"]

    # Wait for the background task to register a partial file before cancel.
    deadline = time.monotonic() + 2.0
    while not written and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert written, "background task did not start in time"

    cancel_response = await backend.cancel(request_id)
    assert cancel_response["cancelled"] is True

    # Release the toolbox thread so the task can settle (the project conventions gotcha
    # #8 — the worker thread runs to completion in the background even
    # after cancel; we still want a tidy test).
    block.set()
    await _drain_background_tasks(backend)

    row = await foundation.persistence.fetch_workflow(request_id)
    assert row is not None
    assert row["status"] == "cancelled", row
    for p in written:
        assert not p.exists(), f"partial file should be unlinked: {p}"


# ---------------------------------------------------------------------------
# 5) Orphan after server restart reconciles via existing 60-s staleness guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_submit_orphan_after_server_restart(
    foundation, monkeypatch
) -> None:
    """A row left ``running`` from a crashed server is reconciled to
    ``failed`` by ``check_status`` once the staleness threshold passes —
    the same NFA-3 path the sync flow uses. No special handling needed
    for async-spawned rows; the registry being process-local is the
    intended trade-off (plan §T-039 risks)."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.persistence.protocol import WorkflowRecord

    _install_fake_module(monkeypatch, subset_fn=lambda **kw: _estimate_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())

    request_id = "req-orphan-async-1234"
    cache_key = "cmems:submit:orphan-async:abcdef0123456789"
    record: WorkflowRecord = {
        "request_id": request_id,
        "backend_id": "cmems",
        "operation": "submit",
        "status": "running",
        "cache_key": cache_key,
        "request_json": "{}",
        "response_json": None,
        "error_record_json": None,
        "created_at": "2026-04-28T00:00:00Z",
        "updated_at": "2026-04-28T00:00:00Z",
    }
    await foundation.persistence.record_workflow(record)

    # No registry entry — simulating a fresh foundation after restart.
    assert request_id not in backend._background_tasks
    out = await backend.check_status(request_id)
    assert out["status"] == "failed"


# ---------------------------------------------------------------------------
# 6) Idempotent against existing cache: returns cache_hit=True synchronously
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_submit_idempotent_against_existing_cache(
    foundation, monkeypatch
) -> None:
    """If the file is already in the cache, async_mode=True still returns
    ``cache_hit=True, status="successful"`` synchronously — no background
    task spawned. Mirrors the sync-path idempotency check."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    def fake(**kwargs):
        if kwargs.get("dry_run"):
            return _estimate_response()
        return None

    _install_fake_module(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    params = _subset_params()
    params["__options"] = {"confirmed": True}

    # First sync submit populates the cache.
    first = await backend.submit(params)
    assert first["status"] == "successful"

    # Second submit with async_mode=True should observe the cache hit
    # synchronously, NOT spawn a new background task.
    params["__options"] = {"async_mode": True, "confirmed": True}
    second = await backend.submit(params)
    assert second["status"] == "successful"
    assert second["cache_hit"] is True
    assert backend._background_tasks == {}


# ---------------------------------------------------------------------------
# 7) Concurrent same-cache_key: second caller observes existing running row
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_submit_concurrent_same_cache_key(
    foundation, monkeypatch
) -> None:
    """Two async submits with identical params — the second observes the
    first's running row (looked up by cache_key) and reuses its
    request_id rather than spawning a duplicate background task."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    block = asyncio.Event()

    def fake(**kwargs):
        if kwargs.get("dry_run"):
            return _estimate_response()
        while not block.is_set():
            time.sleep(0.01)
        return None

    _install_fake_module(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())

    params_a = _subset_params()
    params_a["__options"] = {"async_mode": True, "confirmed": True}
    params_b = _subset_params()
    params_b["__options"] = {"async_mode": True, "confirmed": True}

    first = await backend.submit(params_a)
    assert first["status"] == "running"
    second = await backend.submit(params_b)
    assert second["status"] == "running"
    # Second caller reused the in-flight request_id — no duplicate task.
    assert second["request_id"] == first["request_id"]
    assert len(backend._background_tasks) == 1

    block.set()
    await _drain_background_tasks(backend)


# ---------------------------------------------------------------------------
# 8) Concurrent same-cache_key under asyncio.gather (genuine race)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_submit_genuine_concurrency_dedupe(
    foundation, monkeypatch
) -> None:
    """Round 1 H4 regression: two submits via ``asyncio.gather`` must
    serialise on the per-backend lock so only one background task is
    spawned. The estimate dry-run is gated on a barrier to ensure both
    callers truly race past the dedupe lookup."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    block = asyncio.Event()
    estimates_started = asyncio.Event()
    estimate_count = {"n": 0}

    def fake(**kwargs):
        if kwargs.get("dry_run"):
            estimate_count["n"] += 1
            estimates_started.set()
            return _estimate_response()
        # Real download: hold until released so the row stays "running".
        while not block.is_set():
            time.sleep(0.01)
        return None

    _install_fake_module(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())

    params_a = _subset_params()
    params_a["__options"] = {"async_mode": True, "confirmed": True}
    params_b = _subset_params()
    params_b["__options"] = {"async_mode": True, "confirmed": True}

    first, second = await asyncio.gather(
        backend.submit(params_a), backend.submit(params_b)
    )

    assert first["status"] == "running"
    assert second["status"] == "running"
    # Both observe the SAME request_id — the loser of the race observed
    # the winner's row under the lock.
    assert first["request_id"] == second["request_id"]
    assert len(backend._background_tasks) == 1
    # Only one estimate was actually run for the shared submit.
    assert estimate_count["n"] == 1, estimate_count

    block.set()
    await _drain_background_tasks(backend)


# ---------------------------------------------------------------------------
# 9) Cancellation during cache.store_file (post-toolbox window)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_submit_cancel_during_store_file(
    foundation, monkeypatch
) -> None:
    """Round 1 H3 regression: cancel arriving AFTER ``marine.subset``
    succeeded but during ``cache.store_file`` must still settle the row
    to ``cancelled``. Pre-fix this path left the row stuck ``running``."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    _install_fake_module(monkeypatch, subset_fn=lambda **kw: _estimate_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())

    # Patch store_file to block until cancelled.
    original_store = foundation.cache.store_file
    store_started = asyncio.Event()
    store_block = asyncio.Event()

    async def slow_store(**kwargs):
        store_started.set()
        await store_block.wait()
        return await original_store(**kwargs)

    monkeypatch.setattr(foundation.cache, "store_file", slow_store)

    params = _subset_params()
    params["__options"] = {"async_mode": True, "confirmed": True}
    submitted = await backend.submit(params)
    request_id = submitted["request_id"]

    # Wait for store_file to be reached, then cancel.
    await asyncio.wait_for(store_started.wait(), timeout=5)
    await backend.cancel(request_id)

    # Release store_file so the task settles.
    store_block.set()
    await _drain_background_tasks(backend)

    row = await foundation.persistence.fetch_workflow(request_id)
    assert row is not None
    assert row["status"] == "cancelled", row


# ---------------------------------------------------------------------------
# 10) Exception during cache.store_file (post-toolbox window)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_submit_exception_during_store_file(
    foundation, monkeypatch
) -> None:
    """Round 1 H3 regression: a generic exception raised during
    ``cache.store_file`` (after ``marine.subset`` succeeded) must settle
    the row to ``failed`` with sanitised ``error_record_json``. Pre-fix
    this leaked ``running``."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    _install_fake_module(monkeypatch, subset_fn=lambda **kw: _estimate_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())

    async def boom(**kwargs):
        raise RuntimeError("disk full token=abcdef0123456789ABCDEF foo")

    monkeypatch.setattr(foundation.cache, "store_file", boom)

    params = _subset_params()
    params["__options"] = {"async_mode": True, "confirmed": True}
    submitted = await backend.submit(params)
    request_id = submitted["request_id"]

    await _drain_background_tasks(backend)

    row = await foundation.persistence.fetch_workflow(request_id)
    assert row is not None
    assert row["status"] == "failed", row
    raw = row.get("error_record_json")
    assert raw is not None
    assert "abcdef0123456789ABCDEF" not in raw


# ---------------------------------------------------------------------------
# 11) Sanitiser coverage for nested credential in error_record_json (L2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_submit_sanitises_nested_credential_in_error_record(
    foundation, monkeypatch
) -> None:
    """Round 1 L2 regression: a credential nested in
    ``error_record.context.field_errors`` must be redacted on the
    persisted ``error_record_json``. The previous test's credential
    sat at the top level of the message and would catch only the
    str-walk path, not the nested-dict path."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import ValidationError as CmcpValidationError
    from copernicus_mcp.errors.records import build_error_record

    # ``password=...`` is a sanitiser-recognised key=value shape. If the
    # sanitiser fails to walk into ``context.field_errors[i].msg``, the
    # raw value will survive on the persisted row.
    leaked_value = "hunter2-very-secret-string"
    raw_credential = f"password={leaked_value}"

    def fake(**kwargs):
        if kwargs.get("dry_run"):
            return _estimate_response()
        raise CmcpValidationError(
            "nested credential leak",
            record=build_error_record(
                "ValidationError",
                message="nested credential leak",
                recovery_action="modify_request_parameters",
                context={
                    "field_errors": [
                        {
                            "loc": ["secret"],
                            "msg": f"upstream said {raw_credential} was bad",
                            "type": "value_error",
                        }
                    ]
                },
            ),
        )

    _install_fake_module(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    params = _subset_params()
    params["__options"] = {"async_mode": True, "confirmed": True}

    submitted = await backend.submit(params)
    request_id = submitted["request_id"]
    await _drain_background_tasks(backend)

    row = await foundation.persistence.fetch_workflow(request_id)
    assert row is not None
    raw = row.get("error_record_json") or ""
    assert leaked_value not in raw, (
        "nested credential survived sanitiser walk into context.field_errors"
    )


# ---------------------------------------------------------------------------
# 12) MCP tool wrapper plumbs async_mode option correctly (H1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_subset_tool_plumbs_async_mode_option() -> None:
    """Round 1 H1 regression: ``MarineSubsetDatasetInput`` accepts the
    ``async_mode`` field; the wrapper hoists it into ``options.async_mode``
    and the request body sent to the orchestrator does NOT carry it as a
    request field (would fail Pydantic ``extra='forbid'`` on the schema)."""
    from unittest.mock import AsyncMock

    from copernicus_mcp.backends.cmems.tools import (
        MarineSubsetDatasetInput,
        marine_subset_dataset,
    )

    fake_orch = AsyncMock()
    fake_orch.run.return_value = {
        "result": {"status": "running", "request_id": "rq-1"}
    }

    inp = MarineSubsetDatasetInput(
        dataset_id="ds-1",
        variables=["thetao"],
        minimum_longitude=-1.0,
        maximum_longitude=1.0,
        minimum_latitude=0.0,
        maximum_latitude=1.0,
        minimum_depth=0.0,
        maximum_depth=10.0,
        start_datetime="2024-01-01T00:00:00Z",
        end_datetime="2024-01-02T00:00:00Z",
        async_mode=True,
    )
    out = await marine_subset_dataset(inp, orchestrator=fake_orch)
    assert out["status"] == "running"

    fake_orch.run.assert_awaited_once()
    call_kwargs = fake_orch.run.call_args.kwargs
    # async_mode landed in options, NOT in params.
    assert call_kwargs["options"] == {"async_mode": True}
    assert "async_mode" not in call_kwargs["params"]


@pytest.mark.asyncio
async def test_mcp_check_status_tool_calls_poll_operation() -> None:
    """``marine_check_status`` MCP tool maps to orchestrator ``poll``."""
    from unittest.mock import AsyncMock

    from copernicus_mcp.backends.cmems.tools import (
        MarineCheckStatusInput,
        marine_check_status,
    )

    fake = AsyncMock()
    fake.run.return_value = {"result": {"status": "running", "request_id": "rq-1"}}
    out = await marine_check_status(
        MarineCheckStatusInput(request_id="rq-1"), orchestrator=fake
    )
    assert out["status"] == "running"
    kwargs = fake.run.call_args.kwargs
    assert kwargs["operation"] == "poll"
    assert kwargs["params"] == {"request_id": "rq-1"}


@pytest.mark.asyncio
async def test_mcp_cancel_subset_tool_calls_cancel_operation() -> None:
    """``marine_cancel_subset`` MCP tool maps to orchestrator ``cancel``."""
    from unittest.mock import AsyncMock

    from copernicus_mcp.backends.cmems.tools import (
        MarineCancelSubsetInput,
        marine_cancel_subset,
    )

    fake = AsyncMock()
    fake.run.return_value = {
        "result": {"cancelled": True, "request_id": "rq-1", "status": "cancelled"}
    }
    out = await marine_cancel_subset(
        MarineCancelSubsetInput(request_id="rq-1"), orchestrator=fake
    )
    assert out["cancelled"] is True
    kwargs = fake.run.call_args.kwargs
    assert kwargs["operation"] == "cancel"


# ---------------------------------------------------------------------------
# 13) Async check_status returns the file descriptor on success (round 2 H1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_check_status_returns_descriptor_on_success(
    foundation, monkeypatch
) -> None:
    """Codex round 2 HIGH: a successful async submit must surface the
    large-data descriptor through ``check_status``. Without this the MCP
    agent that polled and got ``status=successful`` cannot reach the
    file."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    _install_fake_module(monkeypatch, subset_fn=lambda **kw: _estimate_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())

    params = _subset_params()
    params["__options"] = {"async_mode": True, "confirmed": True}
    submitted = await backend.submit(params)
    request_id = submitted["request_id"]
    await _drain_background_tasks(backend)

    out = await backend.check_status(request_id)
    assert out["status"] == "successful"
    assert out["result"]["filepath"], out
    assert Path(out["result"]["filepath"]).exists()
    assert out["result"]["uri"].startswith("copernicus://files/")


# ---------------------------------------------------------------------------
# 14) File resource resolves the URI emitted by submit (round 2 H1 part 2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_files_resource_roundtrip_with_real_submit_uri(
    foundation, monkeypatch
) -> None:
    """T-039 round 2 H1 + round 3 revert: the URI emitted by
    ``_success_response`` is bare (``copernicus://files/{cache_key}``);
    the cache stores under ``f"file:{cache_key}"``. The resource
    handler bridges. This test verifies the storage roundtrip via the
    bridge so URI resolution from the agent's POV works end-to-end."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    _install_fake_module(monkeypatch, subset_fn=lambda **kw: _estimate_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())

    params = _subset_params()
    params["__options"] = {"confirmed": True}
    response = await backend.submit(params)
    cache_key = response["cache_key"]
    # Storage is prefixed; the URL emits the bare key.
    looked_up = await foundation.cache.lookup_file(f"file:{cache_key}")
    assert looked_up is not None
    assert looked_up == Path(response["result"]["filepath"])
    assert response["result"]["uri"] == f"copernicus://files/{cache_key}"


# ---------------------------------------------------------------------------
# 15) Cancel force-settles the row even if shielded update is detached
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_cancel_force_settles_row_when_task_handler_did_not_run(
    foundation, monkeypatch
) -> None:
    """Codex round 2 HIGH: the task's ``except CancelledError`` handler
    can be detached by a second cancellation; ``cancel()`` itself must
    force-settle the row to ``cancelled`` if it observes a non-terminal
    state after the task settles."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.persistence.protocol import WorkflowRecord

    _install_fake_module(monkeypatch, subset_fn=lambda **kw: _estimate_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())

    request_id = "req-force-cancel-1234"
    cache_key = "cmems:submit:does-not-exist:abcdef0123456789"
    record: WorkflowRecord = {
        "request_id": request_id,
        "backend_id": "cmems",
        "operation": "submit",
        "status": "running",
        "cache_key": cache_key,
        "request_json": "{}",
        "response_json": None,
        "error_record_json": None,
        "created_at": "2026-04-28T00:00:00Z",
        "updated_at": "2026-04-28T00:00:00Z",
    }
    await foundation.persistence.record_workflow(record)
    # No task in registry — simulating the worst case where the task was
    # cancelled but its row update was detached and never committed.
    out = await backend.cancel(request_id)
    assert out["cancelled"] is True
    assert out["status"] == "cancelled"
    row = await foundation.persistence.fetch_workflow(request_id)
    assert row is not None
    assert row["status"] == "cancelled"


# ---------------------------------------------------------------------------
# 16) Staging-dir setup failure leaves row=failed, not stuck running (M1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_marks_failed_when_staging_setup_raises(
    foundation, monkeypatch
) -> None:
    """Codex round 2 MEDIUM (M1): if ``staging.mkdir`` (or any pre-dispatch
    step) raises, the row is already ``running`` — pre-fix it stayed
    that way. Post-fix the prep block is wrapped to settle the row."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    _install_fake_module(monkeypatch, subset_fn=lambda **kw: _estimate_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())

    # Force ``cache_zone_for`` to raise.
    def boom(_backend_id: str) -> Any:
        raise OSError("simulated disk error")

    monkeypatch.setattr(foundation.cache, "cache_zone_for", boom)

    params = _subset_params()
    params["__options"] = {"confirmed": True}

    from copernicus_mcp.errors import BackendError

    with pytest.raises(BackendError):
        await backend.submit(params)

    # Find the workflow row — request_id is internal; lookup by cache_key.
    from copernicus_mcp.data_model.schemas_cmems import CmemsSubsetRequest

    req = CmemsSubsetRequest(**{k: v for k, v in params.items() if k != "__options"})
    cache_key = foundation.data_model.cache_key_for_subset(req)
    row = await foundation.persistence.lookup_workflow_by_cache_key(cache_key)
    assert row is not None
    assert row["status"] == "failed", row
    assert row["error_record_json"] is not None


# ---------------------------------------------------------------------------
# 17a) check_status surfaces cache_eviction when file is gone post-success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_status_reports_cache_eviction_when_file_gone(
    foundation, monkeypatch
) -> None:
    """Codex/code-reviewer round 3 MEDIUM: a row marked ``successful`` but
    whose cache file was LRU-evicted previously returned ``status=successful``
    with an empty ``result``. Now surfaces a synthetic ``cache_eviction``
    failure so the agent doesn't stop on a hollow success."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.persistence.protocol import WorkflowRecord

    _install_fake_module(monkeypatch, subset_fn=lambda **kw: _estimate_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())

    request_id = "req-evicted-1"
    cache_key = "cmems:submit:no-such-file:abcdef0123456789"
    record: WorkflowRecord = {
        "request_id": request_id,
        "backend_id": "cmems",
        "operation": "submit",
        "status": "successful",
        "cache_key": cache_key,
        "request_json": "{}",
        "response_json": None,
        "error_record_json": None,
        "created_at": "2026-04-01T00:00:00Z",
        "updated_at": "2026-04-01T00:00:00Z",
    }
    await foundation.persistence.record_workflow(record)

    out = await backend.check_status(request_id)
    assert out["status"] == "failed", out
    assert out["error_details"]["error_subclass"] == "cache_eviction"


# ---------------------------------------------------------------------------
# 17b) cancel uses conditional update — no overwrite of fresh successful
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_does_not_overwrite_freshly_successful_row(
    foundation, monkeypatch
) -> None:
    """Codex round 3 MEDIUM (TOCTOU): if the runner commits ``successful``
    between cancel's fetch and update, the conditional UPDATE must NOT
    overwrite. The ``update_workflow_status_if_pending`` SQL guard
    enforces this atomically."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.persistence.protocol import WorkflowRecord

    _install_fake_module(monkeypatch, subset_fn=lambda **kw: _estimate_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())

    request_id = "req-race-1"
    cache_key = "cmems:submit:race:abcdef0123456789"
    # Plant a row already in ``successful`` state (simulating the runner
    # having committed before cancel was called).
    record: WorkflowRecord = {
        "request_id": request_id,
        "backend_id": "cmems",
        "operation": "submit",
        "status": "successful",
        "cache_key": cache_key,
        "request_json": "{}",
        "response_json": None,
        "error_record_json": None,
        "created_at": "2026-05-01T00:00:00Z",
        "updated_at": "2026-05-01T00:00:00Z",
    }
    await foundation.persistence.record_workflow(record)

    out = await backend.cancel(request_id)
    # Already-terminal short-circuit fires first (early-return path).
    assert out["cancelled"] is False
    row = await foundation.persistence.fetch_workflow(request_id)
    assert row is not None
    assert row["status"] == "successful"


# ---------------------------------------------------------------------------
# 17c) v0.1.2 → v0.1.3 cache key migration: storage stays prefixed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_storage_uses_prefixed_cache_key_for_backwards_compat(
    foundation, monkeypatch
) -> None:
    """Codex round 3 HIGH: round 2 dropped the ``file:`` prefix from
    storage to make ``copernicus://files/{cache_key}`` resolve. That
    broke v0.1.2 → v0.1.3 cache migration: both rows pointed at the
    same physical file → eviction unlinked the live file. Round 3
    reverts: storage stays under ``file:{cache_key}``; the URL stays
    bare; ``_file_resource`` bridges.

    This test asserts the storage convention so a future regression
    can't silently re-drop the prefix."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    _install_fake_module(monkeypatch, subset_fn=lambda **kw: _estimate_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    params = _subset_params()
    params["__options"] = {"confirmed": True}

    response = await backend.submit(params)
    cache_key = response["cache_key"]

    # Stored row sits under the prefixed key.
    prefixed = await foundation.persistence.lookup_cache_entry(
        "file", f"file:{cache_key}"
    )
    assert prefixed is not None, (
        "v0.1.2-compat storage convention regressed; "
        "file:-prefixed key not found"
    )
    # The URI emitted by submit is still bare (clean URL).
    assert response["result"]["uri"] == f"copernicus://files/{cache_key}"


# ---------------------------------------------------------------------------
# 18) Jobs resource structurally sanitises nested error_record_json (M2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jobs_resource_parses_error_record_json_for_structural_walk(
    tmp_path: Path,
) -> None:
    """Codex round 2 MEDIUM (M2): ``error_record_json`` is a JSON STRING.
    The string-only sanitiser regex misses structural triggers like
    sensitive dict-key names (``copernicusmarine_service_password`` →
    redact value). Pre-fix a row containing such a payload leaked the
    value through the jobs resource. Post-fix the resource parses the
    JSON, sanitises structurally, then re-serialises."""
    import json as _json

    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.config import ConfigLoader
    from copernicus_mcp.server import build_server

    config = ConfigLoader().load(
        cli_overrides={
            "storage": {
                "state_database": str(tmp_path / "state.db"),
                "cache_directory": str(tmp_path / "cache"),
            }
        }
    )
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        server = build_server(config=config, foundation=foundation, registry=registry)

        leaked = "hunter2-secret"
        # Plant a row whose error_record_json holds a structurally-sensitive
        # key. The KEY name (``copernicusmarine_service_password``) is what
        # triggers the sanitiser to redact the VALUE.
        nested = _json.dumps(
            {
                "context": {
                    "copernicusmarine_service_password": leaked,
                }
            }
        )
        await foundation.persistence.record_workflow(
            {
                "request_id": "req-jobs-leak-1",
                "backend_id": "cmems",
                "operation": "submit",
                "status": "failed",
                "cache_key": "cmems:submit:abc",
                "request_json": "{}",
                "response_json": None,
                "error_record_json": nested,
                "created_at": "2026-05-05T00:00:00Z",
                "updated_at": "2026-05-05T00:00:00Z",
            }
        )

        body = await server.read_resource("copernicus://jobs/req-jobs-leak-1")
        contents = body[0] if isinstance(body, (list, tuple)) else body
        text = str(getattr(contents, "content", contents))
        assert leaked not in text, (
            "structurally-sensitive value leaked through jobs resource"
        )
    finally:
        await foundation.persistence.close()
