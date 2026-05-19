from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _wf(
    request_id: str = "req-1",
    cache_key: str = "ck-1",
    status: str = "queued",
) -> dict:
    return {
        "request_id": request_id,
        "backend_id": "cmems",
        "operation": "subset",
        "status": status,
        "cache_key": cache_key,
        "request_json": json.dumps({"a": 1}),
        "response_json": None,
        "error_record_json": None,
        "created_at": _ts(),
        "updated_at": _ts(),
    }


@pytest.mark.asyncio
async def test_initialise_is_idempotent(sqlite_backend) -> None:
    # The fixture already called initialise — call again.
    await sqlite_backend.initialise()
    await sqlite_backend.initialise()


@pytest.mark.asyncio
async def test_close_is_idempotent(sqlite_backend) -> None:
    await sqlite_backend.close()
    await sqlite_backend.close()


@pytest.mark.asyncio
async def test_parent_directory_is_auto_created(tmp_path: Path) -> None:
    from copernicus_mcp.persistence import SqliteBackend

    nested = tmp_path / "deep" / "nested" / "state.db"
    backend = SqliteBackend(nested)
    await backend.initialise()
    assert nested.exists()
    await backend.close()


@pytest.mark.asyncio
async def test_sqlite_master_lists_four_tables(sqlite_backend) -> None:
    rows = await sqlite_backend.list_tables()
    assert {"workflows", "provenance_records", "acceptance_events", "cache_entries"} <= set(rows)


@pytest.mark.asyncio
async def test_workflow_round_trip(sqlite_backend) -> None:
    rec = _wf()
    await sqlite_backend.record_workflow(rec)
    fetched = await sqlite_backend.fetch_workflow("req-1")
    assert fetched is not None
    assert fetched["request_id"] == "req-1"
    assert fetched["cache_key"] == "ck-1"
    assert fetched["status"] == "queued"


@pytest.mark.asyncio
async def test_workflow_lookup_by_cache_key(sqlite_backend) -> None:
    await sqlite_backend.record_workflow(_wf("req-A", "shared-key"))
    found = await sqlite_backend.lookup_workflow_by_cache_key("shared-key")
    assert found is not None and found["request_id"] == "req-A"
    assert await sqlite_backend.lookup_workflow_by_cache_key("missing") is None


@pytest.mark.asyncio
async def test_update_workflow_status_overwrites(sqlite_backend) -> None:
    await sqlite_backend.record_workflow(_wf("req-U"))
    await sqlite_backend.update_workflow_status("req-U", "running")
    rec = await sqlite_backend.fetch_workflow("req-U")
    assert rec["status"] == "running"
    await sqlite_backend.update_workflow_status("req-U", "successful")
    rec2 = await sqlite_backend.fetch_workflow("req-U")
    assert rec2["status"] == "successful"
    assert rec2["updated_at"] >= rec["updated_at"]


@pytest.mark.asyncio
async def test_update_workflow_status_if_pending_guards_terminals(
    sqlite_backend,
) -> None:
    """Codex T-039 round 4 LOW: pin the SQL guard directly. The conditional
    UPDATE must succeed for queued/running rows and skip for terminals,
    independent of any caller-side short-circuit. This locks the
    regression that was missed by ``test_cancel_does_not_overwrite_freshly_successful_row``
    (which short-circuits via ``cancel()``'s early-terminal-return)."""
    # queued -> cancelled: succeeds, returns True.
    await sqlite_backend.record_workflow(_wf("req-q", "ck-q", status="queued"))
    changed = await sqlite_backend.update_workflow_status_if_pending(
        "req-q", "cancelled"
    )
    assert changed is True
    assert (await sqlite_backend.fetch_workflow("req-q"))["status"] == "cancelled"

    # running -> cancelled: succeeds, returns True.
    await sqlite_backend.record_workflow(_wf("req-r", "ck-r", status="running"))
    changed = await sqlite_backend.update_workflow_status_if_pending(
        "req-r", "cancelled"
    )
    assert changed is True

    # successful -> cancelled: SKIPPED, returns False, status preserved.
    await sqlite_backend.record_workflow(_wf("req-s", "ck-s", status="successful"))
    changed = await sqlite_backend.update_workflow_status_if_pending(
        "req-s", "cancelled"
    )
    assert changed is False
    assert (await sqlite_backend.fetch_workflow("req-s"))["status"] == "successful"

    # failed -> cancelled: SKIPPED.
    await sqlite_backend.record_workflow(_wf("req-f", "ck-f", status="failed"))
    changed = await sqlite_backend.update_workflow_status_if_pending(
        "req-f", "cancelled"
    )
    assert changed is False

    # cancelled -> cancelled: SKIPPED (already terminal).
    await sqlite_backend.record_workflow(
        _wf("req-c", "ck-c", status="cancelled")
    )
    changed = await sqlite_backend.update_workflow_status_if_pending(
        "req-c", "cancelled"
    )
    assert changed is False

    # Nonexistent request_id: returns False (rowcount=0), no exception.
    changed = await sqlite_backend.update_workflow_status_if_pending(
        "req-does-not-exist", "cancelled"
    )
    assert changed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_status", ["existing_success", "completed", "done", ""])
async def test_invalid_status_rejected_on_insert(sqlite_backend, bad_status: str) -> None:
    from copernicus_mcp.errors import ValidationError

    bad = _wf("req-bad", "ck-bad", bad_status)
    with pytest.raises(ValidationError):
        await sqlite_backend.record_workflow(bad)


@pytest.mark.asyncio
async def test_invalid_status_rejected_on_update(sqlite_backend) -> None:
    """Codex T-009 spec review: CHECK must fire on UPDATE too."""
    from copernicus_mcp.errors import ValidationError

    await sqlite_backend.record_workflow(_wf("req-up"))
    with pytest.raises(ValidationError):
        await sqlite_backend.update_workflow_status("req-up", "existing_success")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_existing_success_can_never_reach_db_via_raw_sql(sqlite_backend) -> None:
    """Defense-in-depth: even a direct write attempt is rejected by CHECK."""
    import aiosqlite

    conn = sqlite_backend._conn  # type: ignore[attr-defined]
    with pytest.raises(aiosqlite.IntegrityError):
        await conn.execute(
            "INSERT INTO workflows (request_id, backend_id, operation, status, "
            "cache_key, request_json, response_json, error_record_json, "
            "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "x", "cmems", "subset", "existing_success", "ck",
                "{}", None, None, _ts(), _ts(),
            ),
        )
    await conn.rollback()


@pytest.mark.asyncio
async def test_provenance_round_trip(sqlite_backend) -> None:
    await sqlite_backend.record_workflow(_wf("req-prov"))
    rec = {
        "record_id": "prov-1",
        "workflow_request_id": "req-prov",
        "provenance_json": json.dumps({"backend": "cmems", "url": "https://x"}),
        "created_at": _ts(),
    }
    await sqlite_backend.record_provenance(rec)
    fetched = await sqlite_backend.fetch_provenance("prov-1")
    assert fetched is not None
    assert json.loads(fetched["provenance_json"])["backend"] == "cmems"


@pytest.mark.asyncio
async def test_provenance_with_null_workflow_id_allowed(sqlite_backend) -> None:
    rec = {
        "record_id": "prov-orphan",
        "workflow_request_id": None,
        "provenance_json": json.dumps({"orphan": True}),
        "created_at": _ts(),
    }
    await sqlite_backend.record_provenance(rec)
    fetched = await sqlite_backend.fetch_provenance("prov-orphan")
    assert fetched is not None
    assert fetched["workflow_request_id"] is None


@pytest.mark.asyncio
async def test_provenance_with_bad_workflow_id_raises(sqlite_backend) -> None:
    """FK enforcement: non-NULL but non-existent workflow_request_id rejected."""
    from copernicus_mcp.errors import ValidationError

    rec = {
        "record_id": "prov-bad",
        "workflow_request_id": "does-not-exist",
        "provenance_json": "{}",
        "created_at": _ts(),
    }
    with pytest.raises(ValidationError):
        await sqlite_backend.record_provenance(rec)


@pytest.mark.asyncio
async def test_acceptance_round_trip(sqlite_backend) -> None:
    await sqlite_backend.record_workflow(_wf("req-acc"))
    ev = {
        "event_id": "ev-1",
        "workflow_request_id": "req-acc",
        "accepted_at": _ts(),
        "accepted_by": "alice",
        "request_hash": "abcdef",
    }
    await sqlite_backend.record_acceptance(ev)
    fetched = await sqlite_backend.lookup_acceptance("req-acc")
    assert fetched is not None
    assert fetched["accepted_by"] == "alice"


@pytest.mark.asyncio
async def test_cache_entry_round_trip_and_namespace_isolation(
    sqlite_backend,
) -> None:
    file_entry = {
        "namespace": "file",
        "key": "ck-1",
        "value_json": "{}",
        "file_path": "/tmp/x.nc",
        "size_bytes": 1024,
        "content_type": "application/x-netcdf",
        "created_at": _ts(),
        "last_accessed_at": _ts(),
    }
    meta_entry = {
        **file_entry,
        "namespace": "metadata",
        "key": "ck-1",
        "file_path": None,
        "size_bytes": 0,
        "content_type": "application/json",
    }
    await sqlite_backend.record_cache_entry(file_entry)
    await sqlite_backend.record_cache_entry(meta_entry)

    f = await sqlite_backend.lookup_cache_entry("file", "ck-1")
    m = await sqlite_backend.lookup_cache_entry("metadata", "ck-1")
    assert f is not None and m is not None
    assert f["file_path"] == "/tmp/x.nc"
    assert m["file_path"] is None

    files = [e async for e in sqlite_backend.iter_cache_entries_by_namespace("file")]
    assert len(files) == 1 and files[0]["key"] == "ck-1"

    assert await sqlite_backend.delete_cache_entry("file", "ck-1") is True
    assert await sqlite_backend.lookup_cache_entry("file", "ck-1") is None
    assert await sqlite_backend.lookup_cache_entry("metadata", "ck-1") is not None


@pytest.mark.asyncio
async def test_cache_entry_upsert_preserves_created_at(sqlite_backend) -> None:
    """Codex spec review: ON CONFLICT DO UPDATE preserves created_at,
    bumps last_accessed_at."""
    initial_created = _ts()
    await sqlite_backend.record_cache_entry(
        {
            "namespace": "file",
            "key": "ck-up",
            "value_json": "{}",
            "file_path": "/tmp/v1.nc",
            "size_bytes": 100,
            "content_type": "application/x-netcdf",
            "created_at": initial_created,
            "last_accessed_at": initial_created,
        }
    )
    later = "2099-01-01T00:00:00Z"
    await sqlite_backend.record_cache_entry(
        {
            "namespace": "file",
            "key": "ck-up",
            "value_json": "{}",
            "file_path": "/tmp/v2.nc",
            "size_bytes": 200,
            "content_type": "application/x-netcdf",
            "created_at": later,  # caller supplies later — must be ignored
            "last_accessed_at": later,
        }
    )
    out = await sqlite_backend.lookup_cache_entry("file", "ck-up")
    assert out["created_at"] == initial_created
    assert out["last_accessed_at"] == later
    assert out["file_path"] == "/tmp/v2.nc"
    assert out["size_bytes"] == 200


@pytest.mark.asyncio
async def test_concurrent_writes_do_not_deadlock(sqlite_backend) -> None:
    async def _write(i: int) -> None:
        await sqlite_backend.record_workflow(_wf(f"req-{i}", f"ck-{i}"))

    await asyncio.gather(*[_write(i) for i in range(10)])
    for i in range(10):
        rec = await sqlite_backend.fetch_workflow(f"req-{i}")
        assert rec is not None


@pytest.mark.asyncio
async def test_json_unicode_survives(sqlite_backend) -> None:
    payload = {"label": "Привет 🌊", "unit": "°C"}
    rec = _wf("req-unicode", "ck-unicode")
    rec["request_json"] = json.dumps(payload, ensure_ascii=False)
    await sqlite_backend.record_workflow(rec)
    out = await sqlite_backend.fetch_workflow("req-unicode")
    assert json.loads(out["request_json"]) == payload


@pytest.mark.asyncio
async def test_workflow_cache_key_index_exists(sqlite_backend) -> None:
    """Codex T-009 spec review: assert the cache_key index is present."""
    conn = sqlite_backend._conn  # type: ignore[attr-defined]
    cur = await conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    )
    names = [row[0] async for row in cur]
    assert any("cache_key" in n for n in names)
