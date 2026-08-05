"""Compare-and-swap chunk-plan writes (T-CDS-RESIL-006, persistence layer).

The per-parent lock is process-LOCAL, but a typical deployment polls one
chunked parent from two processes (MCP server + `cds wait` / an ephemeral
poller). Every plan write used to be an unconditional whole-JSON overwrite —
two writers could both read ``attempt: 0``, both submit a remote job, and one
live CDS job vanished from the plan. The write is now a CAS on a monotonic
``chunk_plan_version`` column: a losing writer re-reads and re-decides
instead of clobbering.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def backend(tmp_path: Path):
    from copernicus_mcp.persistence import SqliteBackend

    b = SqliteBackend(tmp_path / "state.db")
    await b.initialise()
    try:
        yield b
    finally:
        await b.close()


def _record(request_id: str = "parent-1", plan: str | None = '{"chunks": []}'):
    return {
        "request_id": request_id,
        "backend_id": "cds",
        "operation": "submit",
        "status": "queued",
        "cache_key": f"ck-{request_id}",
        "request_json": "{}",
        "response_json": None,
        "error_record_json": None,
        "created_at": "2026-08-05T00:00:00Z",
        "updated_at": "2026-08-05T00:00:00Z",
        "chunk_plan_json": plan,
    }


@pytest.mark.asyncio
async def test_fresh_row_has_version_zero(backend) -> None:
    await backend.record_workflow(_record())
    row = await backend.fetch_workflow("parent-1")
    assert row is not None
    assert row["chunk_plan_version"] == 0


@pytest.mark.asyncio
async def test_cas_write_wins_once_and_bumps_the_version(backend) -> None:
    await backend.record_workflow(_record())

    won = await backend.update_chunk_plan(
        "parent-1", '{"chunks": ["a"]}', expected_version=0
    )
    assert won is True
    row = await backend.fetch_workflow("parent-1")
    assert row["chunk_plan_json"] == '{"chunks": ["a"]}'
    assert row["chunk_plan_version"] == 1

    # The same expected_version again is a STALE write — it must lose and
    # must not touch the stored plan.
    lost = await backend.update_chunk_plan(
        "parent-1", '{"chunks": ["CLOBBER"]}', expected_version=0
    )
    assert lost is False
    row = await backend.fetch_workflow("parent-1")
    assert row["chunk_plan_json"] == '{"chunks": ["a"]}'
    assert row["chunk_plan_version"] == 1


@pytest.mark.asyncio
async def test_unconditional_write_still_bumps_the_version(backend) -> None:
    """The escape hatch (no expected_version) keeps working for callers that
    genuinely want last-writer-wins — but it must STILL bump the version so a
    concurrent CAS writer observes the movement and re-reads."""
    await backend.record_workflow(_record())
    ok = await backend.update_chunk_plan("parent-1", '{"x": 1}')
    assert ok is True
    row = await backend.fetch_workflow("parent-1")
    assert row["chunk_plan_version"] == 1
    # A CAS against the pre-write version now loses.
    assert (
        await backend.update_chunk_plan("parent-1", "{}", expected_version=0)
        is False
    )


@pytest.mark.asyncio
async def test_cas_on_missing_row_reports_a_loss(backend) -> None:
    assert (
        await backend.update_chunk_plan("ghost", "{}", expected_version=0)
        is False
    )


@pytest.mark.asyncio
async def test_two_connections_to_one_file_serialise_via_cas(
    tmp_path: Path,
) -> None:
    """The two-process shape: two independent SqliteBackend connections to the
    SAME database file. Both read version 0; only one CAS write commits, and
    the loser sees the winner's content on re-read."""
    from copernicus_mcp.persistence import SqliteBackend

    a = SqliteBackend(tmp_path / "shared.db")
    b = SqliteBackend(tmp_path / "shared.db")
    await a.initialise()
    await b.initialise()
    try:
        await a.record_workflow(_record("p"))

        row_a = await a.fetch_workflow("p")
        row_b = await b.fetch_workflow("p")
        assert row_a["chunk_plan_version"] == row_b["chunk_plan_version"] == 0

        assert await a.update_chunk_plan("p", '{"winner": "a"}', expected_version=0)
        assert (
            await b.update_chunk_plan("p", '{"winner": "b"}', expected_version=0)
            is False
        )

        fresh = await b.fetch_workflow("p")
        assert fresh["chunk_plan_json"] == '{"winner": "a"}'
        assert fresh["chunk_plan_version"] == 1
        # The loser retries against the fresh version and now wins.
        assert await b.update_chunk_plan(
            "p", '{"winner": "b2"}', expected_version=1
        )
        assert (await a.fetch_workflow("p"))["chunk_plan_json"] == '{"winner": "b2"}'
    finally:
        await a.close()
        await b.close()


@pytest.mark.asyncio
async def test_migration_adds_the_version_column_to_an_old_db(
    tmp_path: Path,
) -> None:
    """An existing pre-RESIL-006 database (no chunk_plan_version column) is
    upgraded additively on initialise; old rows read back as version 0."""
    from copernicus_mcp.persistence import SqliteBackend

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE workflows (
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
            updated_at TEXT NOT NULL,
            parent_request_id TEXT,
            chunk_plan_json TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO workflows (request_id, backend_id, operation, status, "
        "request_json, created_at, updated_at, chunk_plan_json) "
        "VALUES ('old-p','cds','submit','running','{}','t','t','{\"chunks\":[]}')"
    )
    conn.commit()
    conn.close()

    backend = SqliteBackend(db)
    await backend.initialise()
    try:
        row = await backend.fetch_workflow("old-p")
        assert row is not None
        assert row["chunk_plan_version"] == 0
        assert await backend.update_chunk_plan(
            "old-p", '{"chunks": [1]}', expected_version=0
        )
        assert (await backend.fetch_workflow("old-p"))["chunk_plan_version"] == 1
    finally:
        await backend.close()
