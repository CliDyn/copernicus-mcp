"""T-JOBS-RECOVERY: cross-session job discovery.

Layer 1 here: ``SqliteBackend.list_workflows`` — the read path that lets a fresh
agent enumerate persisted jobs after a restart (no ``request_id`` needed).
Orchestrator / tool / CLI layers are exercised in sibling test modules.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
import pytest_asyncio


def _ts(day: int = 1) -> str:
    """Deterministic, distinct ISO-8601 UTC timestamps (one per ``day``)."""
    return datetime(2026, 1, day, 0, 0, 0, tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _wf(
    request_id: str,
    *,
    backend_id: str = "cds",
    status: str = "successful",
    request_json: str = "{}",
    response_json: str | None = None,
    error_record_json: str | None = None,
    created_at: str | None = None,
) -> dict:
    ts = created_at or _ts()
    return {
        "request_id": request_id,
        "backend_id": backend_id,
        "operation": "submit",
        "status": status,
        "cache_key": f"ck-{request_id}",
        "request_json": request_json,
        "response_json": response_json,
        "error_record_json": error_record_json,
        "created_at": ts,
        "updated_at": ts,
        "parent_request_id": None,
        "chunk_plan_json": None,
    }


# --- Layer 1: persistence.list_workflows ---------------------------------------


@pytest.mark.asyncio
async def test_list_workflows_newest_first(sqlite_backend) -> None:
    await sqlite_backend.record_workflow(_wf("a", created_at=_ts(1)))
    await sqlite_backend.record_workflow(_wf("b", created_at=_ts(3)))
    await sqlite_backend.record_workflow(_wf("c", created_at=_ts(2)))
    rows = await sqlite_backend.list_workflows()
    assert [r["request_id"] for r in rows] == ["b", "c", "a"]


@pytest.mark.asyncio
async def test_list_workflows_respects_limit(sqlite_backend) -> None:
    for i in range(5):
        await sqlite_backend.record_workflow(_wf(f"r{i}", created_at=_ts(i + 1)))
    rows = await sqlite_backend.list_workflows(limit=2)
    assert [r["request_id"] for r in rows] == ["r4", "r3"]


@pytest.mark.asyncio
async def test_list_workflows_status_filter_single(sqlite_backend) -> None:
    await sqlite_backend.record_workflow(_wf("ok1", status="successful", created_at=_ts(1)))
    await sqlite_backend.record_workflow(_wf("run1", status="running", created_at=_ts(2)))
    await sqlite_backend.record_workflow(_wf("ok2", status="successful", created_at=_ts(3)))
    rows = await sqlite_backend.list_workflows(status=["successful"])
    assert {r["request_id"] for r in rows} == {"ok1", "ok2"}


@pytest.mark.asyncio
async def test_list_workflows_status_filter_multi(sqlite_backend) -> None:
    await sqlite_backend.record_workflow(_wf("q", status="queued", created_at=_ts(1)))
    await sqlite_backend.record_workflow(_wf("r", status="running", created_at=_ts(2)))
    await sqlite_backend.record_workflow(_wf("s", status="successful", created_at=_ts(3)))
    rows = await sqlite_backend.list_workflows(status=["queued", "running"])
    assert {r["request_id"] for r in rows} == {"q", "r"}


@pytest.mark.asyncio
async def test_list_workflows_created_after_is_strict(sqlite_backend) -> None:
    await sqlite_backend.record_workflow(_wf("a", created_at=_ts(1)))
    await sqlite_backend.record_workflow(_wf("b", created_at=_ts(2)))
    await sqlite_backend.record_workflow(_wf("c", created_at=_ts(3)))
    rows = await sqlite_backend.list_workflows(created_after=_ts(2))
    assert [r["request_id"] for r in rows] == ["c"]


@pytest.mark.asyncio
async def test_list_workflows_empty(sqlite_backend) -> None:
    assert await sqlite_backend.list_workflows() == []


@pytest.mark.asyncio
async def test_list_workflows_limit_clamped_low(sqlite_backend) -> None:
    # SQLite treats LIMIT -1/0 as "no limit"; clamp so a bad input can't dump all.
    for i in range(3):
        await sqlite_backend.record_workflow(_wf(f"r{i}", created_at=_ts(i + 1)))
    assert len(await sqlite_backend.list_workflows(limit=0)) == 1
    assert len(await sqlite_backend.list_workflows(limit=-5)) == 1


@pytest.mark.asyncio
async def test_list_workflows_large_limit_ok(sqlite_backend) -> None:
    for i in range(3):
        await sqlite_backend.record_workflow(_wf(f"r{i}", created_at=_ts(i + 1)))
    rows = await sqlite_backend.list_workflows(limit=10_000)
    assert len(rows) == 3


@pytest.mark.asyncio
async def test_list_workflows_empty_status_means_all(sqlite_backend) -> None:
    # Guard (review LOW): empty list == no filter, NOT "IN ()" which would raise.
    # Pins the `if status:` semantics so a refactor to `if status is not None:`
    # (which would emit invalid SQL) is caught.
    await sqlite_backend.record_workflow(_wf("q", status="queued", created_at=_ts(1)))
    await sqlite_backend.record_workflow(_wf("s", status="successful", created_at=_ts(2)))
    rows = await sqlite_backend.list_workflows(status=[])
    assert {r["request_id"] for r in rows} == {"q", "s"}


# --- Layer 2: orchestrator.list_jobs (shaping helpers + envelope) ---------------


def test_job_summary_core_fields_and_dataset() -> None:
    from copernicus_mcp.workflow.orchestrator import _job_summary

    row = _wf(
        "j1",
        backend_id="cmems",
        status="running",
        request_json=json.dumps({"dataset_id": "glo_phy", "variables": ["thetao"]}),
        created_at=_ts(4),
    )
    s = _job_summary(row)
    assert s["request_id"] == "j1"
    assert s["backend"] == "cmems"
    assert s["operation"] == "submit"
    assert s["status"] == "running"
    assert s["dataset"] == "glo_phy"
    assert s["created_at"] == _ts(4)
    # response_json is structurally always NULL → no file fields on the summary.
    assert "filepath" not in s and "size_bytes" not in s


def test_job_summary_failed_surfaces_error_class() -> None:
    from copernicus_mcp.workflow.orchestrator import _job_summary

    row = _wf(
        "j2",
        status="failed",
        error_record_json=json.dumps({"error_class": "BackendError", "message": "boom"}),
    )
    assert _job_summary(row)["error_class"] == "BackendError"


def test_job_summary_does_not_leak_extra_request_params() -> None:
    from copernicus_mcp.workflow.orchestrator import _job_summary

    row = _wf(
        "j3",
        request_json=json.dumps(
            {"dataset_id": "d", "copernicusmarine_service_password": "hunter2"}
        ),
    )
    s = _job_summary(row)
    assert s["dataset"] == "d"
    assert "hunter2" not in json.dumps(s)


def test_job_summary_missing_dataset_is_none() -> None:
    from copernicus_mcp.workflow.orchestrator import _job_summary

    assert _job_summary(_wf("j4", request_json="{}"))["dataset"] is None


def test_validate_status_filter_rejects_noncanonical() -> None:
    from copernicus_mcp.errors.classes import ValidationError
    from copernicus_mcp.workflow.orchestrator import _validate_status_filter

    with pytest.raises(ValidationError):
        _validate_status_filter(["successful", "bogus"])


def test_validate_status_filter_accepts_canonical_and_none() -> None:
    from copernicus_mcp.workflow.orchestrator import _validate_status_filter

    _validate_status_filter(None)
    _validate_status_filter(["queued", "running", "successful", "failed", "cancelled"])


@pytest_asyncio.fixture
async def orchestrator(tmp_path):
    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.config import ConfigLoader
    from copernicus_mcp.workflow.orchestrator import WorkflowOrchestrator

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
        yield (
            WorkflowOrchestrator(registry=registry, foundation=foundation),
            foundation.persistence,
        )
    finally:
        await foundation.persistence.close()


@pytest.mark.asyncio
async def test_list_jobs_envelope_newest_first(orchestrator) -> None:
    orch, persistence = orchestrator
    await persistence.record_workflow(
        _wf("a", created_at=_ts(1), request_json=json.dumps({"dataset_id": "da"}))
    )
    await persistence.record_workflow(
        _wf("b", created_at=_ts(2), request_json=json.dumps({"dataset_id": "db"}))
    )
    out = await orch.list_jobs()
    assert out["count"] == 2
    assert [r["request_id"] for r in out["results"]] == ["b", "a"]
    assert out["results"][0]["dataset"] == "db"


@pytest.mark.asyncio
async def test_list_jobs_bad_status_returns_error_envelope(orchestrator) -> None:
    orch, _persistence = orchestrator
    out = await orch.list_jobs(status=["bogus"])
    assert "error" in out
    assert out["error"]["error_class"] == "ValidationError"


@pytest.mark.asyncio
async def test_list_jobs_does_not_leak_request_params(orchestrator) -> None:
    orch, persistence = orchestrator
    await persistence.record_workflow(
        _wf(
            "sekret",
            request_json=json.dumps(
                {"dataset_id": "d", "copernicusmarine_service_password": "hunter2"}
            ),
        )
    )
    out = await orch.list_jobs()
    assert "hunter2" not in json.dumps(out)


@pytest.mark.asyncio
async def test_list_jobs_created_after_normalizes_equivalent_utc(orchestrator) -> None:
    # Review M1: created_after must be a temporal bound, not a lexical one.
    # r0 and the two bounds are the SAME instant → r0 is excluded (strict >).
    orch, persistence = orchestrator
    await persistence.record_workflow(_wf("r0", created_at="2026-06-01T00:00:00Z"))
    await persistence.record_workflow(_wf("r1", created_at="2026-06-01T00:00:05Z"))
    for bound in ("2026-06-01T00:00:00.000Z", "2026-06-01T00:00:00+00:00"):
        out = await orch.list_jobs(created_after=bound)
        assert [r["request_id"] for r in out["results"]] == ["r1"], bound


@pytest.mark.asyncio
async def test_list_jobs_created_after_rejects_malformed(orchestrator) -> None:
    # Review M1: the documented ISO-8601 contract is enforced, not silently
    # mis-compared. A non-timestamp is a ValidationError, not a wrong result.
    orch, _persistence = orchestrator
    out = await orch.list_jobs(created_after="yesterday")
    assert "error" in out
    assert out["error"]["error_class"] == "ValidationError"


# --- Layer 3: copernicus_mcp_list_jobs MCP tool --------------------------------


@pytest_asyncio.fixture
async def server_and_persistence(tmp_path):
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
        yield server, foundation.persistence
    finally:
        await foundation.persistence.close()


@pytest.mark.asyncio
async def test_list_jobs_tool_is_registered(server_and_persistence) -> None:
    server, _persistence = server_and_persistence
    tools = await server.list_tools()
    assert "copernicus_mcp_list_jobs" in {t.name for t in tools}


@pytest.mark.asyncio
async def test_list_jobs_tool_returns_recorded_jobs(server_and_persistence) -> None:
    server, persistence = server_and_persistence
    await persistence.record_workflow(
        _wf("a", created_at=_ts(1), request_json=json.dumps({"dataset_id": "da"}))
    )
    await persistence.record_workflow(
        _wf("b", created_at=_ts(2), request_json=json.dumps({"dataset_id": "db"}))
    )
    result = await server.call_tool("copernicus_mcp_list_jobs", {})
    structured = result[1] if isinstance(result, tuple) else result
    assert structured["count"] == 2
    assert [r["request_id"] for r in structured["results"]] == ["b", "a"]


@pytest.mark.asyncio
async def test_list_jobs_tool_status_filter(server_and_persistence) -> None:
    server, persistence = server_and_persistence
    await persistence.record_workflow(_wf("ok", status="successful", created_at=_ts(1)))
    await persistence.record_workflow(_wf("run", status="running", created_at=_ts(2)))
    result = await server.call_tool("copernicus_mcp_list_jobs", {"status": ["running"]})
    structured = result[1] if isinstance(result, tuple) else result
    assert [r["request_id"] for r in structured["results"]] == ["run"]
