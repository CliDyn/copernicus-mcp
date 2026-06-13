"""T-CDS-CHUNK-001: parent/child workflow columns + additive migration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
import pytest

from copernicus_mcp.persistence import SqliteBackend


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _wf(
    request_id: str,
    *,
    status: str = "queued",
    parent_request_id: str | None = None,
    chunk_plan_json: str | None = None,
) -> dict:
    return {
        "request_id": request_id,
        "backend_id": "cds",
        "operation": "submit",
        "status": status,
        "cache_key": f"ck-{request_id}",
        "request_json": "{}",
        "response_json": None,
        "error_record_json": None,
        "created_at": _ts(),
        "updated_at": _ts(),
        "parent_request_id": parent_request_id,
        "chunk_plan_json": chunk_plan_json,
    }


# Pre-change workflows DDL (no parent_request_id / chunk_plan_json) — used to
# build an "old" database for the upgrade-path test.
_OLD_WORKFLOWS_DDL = """
CREATE TABLE IF NOT EXISTS workflows (
    request_id TEXT PRIMARY KEY,
    backend_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('queued','running','successful','failed','cancelled')
    ),
    cache_key TEXT,
    request_json TEXT NOT NULL,
    response_json TEXT,
    error_record_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


@pytest.mark.asyncio
async def test_fresh_db_has_chunk_columns(tmp_path: Path) -> None:
    backend = SqliteBackend(tmp_path / "fresh.db")
    await backend.initialise()
    try:
        conn = await aiosqlite.connect(str(tmp_path / "fresh.db"))
        cur = await conn.execute("PRAGMA table_info(workflows)")
        cols = {row[1] for row in await cur.fetchall()}
        await conn.close()
        assert "parent_request_id" in cols
        assert "chunk_plan_json" in cols
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_upgrade_adds_chunk_columns_and_preserves_rows(tmp_path: Path) -> None:
    db = tmp_path / "old.db"
    # Build an OLD-schema db with one pre-existing row.
    conn = await aiosqlite.connect(str(db))
    await conn.execute(_OLD_WORKFLOWS_DDL)
    await conn.execute(
        "INSERT INTO workflows (request_id, backend_id, operation, status, "
        "cache_key, request_json, response_json, error_record_json, created_at, "
        "updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("old-1", "cds", "submit", "successful", "ck", "{}", None, None, _ts(), _ts()),
    )
    await conn.commit()
    await conn.close()

    # Initialise with the new code → migration adds the columns.
    backend = SqliteBackend(db)
    await backend.initialise()
    try:
        old = await backend.fetch_workflow("old-1")
        assert old is not None
        assert old["status"] == "successful"
        assert old["parent_request_id"] is None
        assert old["chunk_plan_json"] is None
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_check_constraint_still_rejects_sixth_status(sqlite_backend) -> None:
    from copernicus_mcp.errors.classes import ValidationError

    with pytest.raises(ValidationError):
        await sqlite_backend.record_workflow(_wf("bad", status="existing_success"))


@pytest.mark.asyncio
async def test_record_and_fetch_parent_and_plan(sqlite_backend) -> None:
    plan = json.dumps({"chunks": [{"index": 0}], "stopped": False})
    await sqlite_backend.record_workflow(_wf("parent-1", chunk_plan_json=plan))
    row = await sqlite_backend.fetch_workflow("parent-1")
    assert row["chunk_plan_json"] == plan
    assert row["parent_request_id"] is None


@pytest.mark.asyncio
async def test_list_child_workflows(sqlite_backend) -> None:
    await sqlite_backend.record_workflow(_wf("parent-2"))
    await sqlite_backend.record_workflow(_wf("child-a", parent_request_id="parent-2"))
    await sqlite_backend.record_workflow(_wf("child-b", parent_request_id="parent-2"))
    await sqlite_backend.record_workflow(_wf("unrelated"))
    children = await sqlite_backend.list_child_workflows("parent-2")
    assert {c["request_id"] for c in children} == {"child-a", "child-b"}


@pytest.mark.asyncio
async def test_update_chunk_plan(sqlite_backend) -> None:
    await sqlite_backend.record_workflow(_wf("parent-3"))
    await sqlite_backend.update_chunk_plan("parent-3", '{"stopped": true}')
    row = await sqlite_backend.fetch_workflow("parent-3")
    assert row["chunk_plan_json"] == '{"stopped": true}'
