"""T-CDS-FINALISE-ATOMIC Task 004: rows 0.7.0 left ``successful`` without a
cache entry are reported honestly (read-only) and repaired or re-fetched by
the foreground download path; a CDS-done pending row downloads in the
foreground.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_cds_finalise_atomic import (
    _backend_for,
    _db_state,
    _drive_to_successful,
    _submit,
)
from tests.unit.test_cds_resume_download import (
    _PAYLOAD,
    _FakeMultiurl,
    _make_foundation,
    _patch_cdsapi_resumable,
    _patch_costing_flat,
)


def _break_registration(tmp_path: Path, rid: str, *, legacy_name: bool, truncate: bool = False) -> Path:
    """Shape the store like the field incident: row ``successful``, file on
    disk, no ``cache_entries`` row. ``legacy_name`` renames the file to 0.7.0's
    canonical ``cds_<hash>.<ext>``."""
    db = tmp_path / "state.db"
    _, paths = _db_state(db, rid)
    path = Path(paths[0])
    con = sqlite3.connect(str(db))
    try:
        con.execute("DELETE FROM cache_entries WHERE file_path = ?", (str(path),))
        con.commit()
    finally:
        con.close()
    if legacy_name:
        stem = path.name.split(".")[0]  # cds_<hash>
        legacy = path.with_name(f"{stem}{path.suffix}")
        path.rename(legacy)
        path = legacy
    if truncate:
        path.write_bytes(_PAYLOAD[:100])
    return path


def _downloads(inner: Any) -> int:
    """Re-downloads through either path: the SDK call, or the resumable
    transfer (``multiurl``) when the result handle allows it."""
    from copernicus_mcp.backends.cds import backend as backend_mod

    via_multiurl = len(getattr(backend_mod.multiurl, "calls", []))
    return int(inner.download_results.call_count) + via_multiurl


async def _successful_then_broken(tmp_path: Path, monkeypatch, **kw: Any) -> tuple[str, Path, Any]:
    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    inner = _patch_cdsapi_resumable(monkeypatch, status_map)
    rid = await _submit(tmp_path, status_map)
    final = await _drive_to_successful(tmp_path, rid, status_map, monkeypatch)
    assert final["status"] == "successful"
    from copernicus_mcp.backends.cds import backend as backend_mod

    monkeypatch.setattr(backend_mod, "multiurl", _FakeMultiurl())  # fresh counter
    return rid, _break_registration(tmp_path, rid, **kw), inner


@pytest.mark.asyncio
async def test_check_status_reports_a_lost_entry_honestly_and_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    rid, legacy, _ = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        out = await _backend_for(found).check_status(rid)
    finally:
        await persistence.close()
    err = out["error_details"]
    assert out["status"] == "failed"
    assert err["error_class"] == "CacheError"
    assert err["error_subclass"] == "cache_entry_missing"
    assert err["recovery_action"] == "use_alternative_tool"
    assert err["automatic_retry_recommended"] is False
    assert err["retryable"] is True
    assert err["context"] == {"remote_job_status": "successful", "local_only": True, "request_id": rid}
    assert "cds download <request_id>" in err["next_action_hint"]
    assert _db_state(tmp_path / "state.db", rid)[1] == [], "check_status must not write entries"


@pytest.mark.asyncio
async def test_lru_tombstone_still_reads_cache_eviction(tmp_path: Path, monkeypatch) -> None:
    rid, path, _ = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=False)
    con = sqlite3.connect(str(tmp_path / "state.db"))
    try:
        key = con.execute("SELECT cache_key FROM workflows WHERE request_id=?", (rid,)).fetchone()[0]
        from copernicus_mcp.backends.cds.backend import _cache_storage_key

        con.execute(
            "INSERT INTO cache_evictions (namespace, key, file_path, reason, evicted_at) VALUES ('file', ?, ?, 'lru', '2026-10-19T00:00:00Z')",
            (_cache_storage_key(key), str(path)),
        )
        con.commit()
    finally:
        con.close()
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        out = await _backend_for(found).check_status(rid)
    finally:
        await persistence.close()
    assert out["error_details"]["error_subclass"] == "cache_eviction"
    assert out["error_details"]["recovery_action"] == "use_alternative_tool"


@pytest.mark.asyncio
async def test_download_adopts_a_size_matching_legacy_file_without_downloading(tmp_path: Path, monkeypatch) -> None:
    rid, legacy, inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True)
    inner.download_results.reset_mock()
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        out = await _backend_for(found).fetch_result(rid, Path("ignored"))
    finally:
        await persistence.close()
    assert out["status"] == "successful"
    assert out["result"]["filepath"] == str(legacy)
    assert _downloads(inner) == 0
    assert _db_state(tmp_path / "state.db", rid)[1] == [str(legacy)]


@pytest.mark.asyncio
async def test_download_refetches_when_the_legacy_file_size_differs(tmp_path: Path, monkeypatch) -> None:
    rid, legacy, inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True, truncate=True)
    inner.download_results.reset_mock()
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        out = await _backend_for(found).fetch_result(rid, Path("ignored"))
    finally:
        await persistence.close()
    assert out["status"] == "successful"
    path = Path(out["result"]["filepath"])
    assert path != legacy
    assert path.read_bytes() == _PAYLOAD
    assert _downloads(inner) == 1
    assert _db_state(tmp_path / "state.db", rid)[1] == [str(path)]


@pytest.mark.asyncio
async def test_download_does_not_adopt_a_legacy_file_shared_by_two_successful_rows(tmp_path: Path, monkeypatch) -> None:
    """Force-refresh history: equal size is not identity when two successful
    rows share the cache key — re-fetch instead."""
    rid, legacy, inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True)
    con = sqlite3.connect(str(tmp_path / "state.db"))
    try:
        row = con.execute("SELECT * FROM workflows WHERE request_id=?", (rid,)).fetchone()
        cols = [d[1] for d in con.execute("PRAGMA table_info(workflows)")]
        twin = dict(zip(cols, row, strict=True))
        twin["request_id"] = "twin-" + rid
        con.execute(
            f"INSERT INTO workflows ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})",
            [twin[c] for c in cols],
        )
        con.commit()
    finally:
        con.close()
    inner.download_results.reset_mock()
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        out = await _backend_for(found).fetch_result(rid, Path("ignored"))
    finally:
        await persistence.close()
    assert out["status"] == "successful"
    assert Path(out["result"]["filepath"]) != legacy
    assert _downloads(inner) == 1


@pytest.mark.asyncio
async def test_download_adopts_the_rows_own_request_scoped_file(tmp_path: Path, monkeypatch) -> None:
    rid, own, inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=False)
    inner.download_results.reset_mock()
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        out = await _backend_for(found).fetch_result(rid, Path("ignored"))
    finally:
        await persistence.close()
    assert out["result"]["filepath"] == str(own)
    assert _downloads(inner) == 0


@pytest.mark.asyncio
async def test_download_reports_remote_result_expired_on_404(tmp_path: Path, monkeypatch) -> None:
    import requests

    from copernicus_mcp.errors import BackendError

    rid, legacy, inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True, truncate=True)

    def gone(request_id):  # type: ignore[no-untyped-def]
        resp = requests.Response()
        resp.status_code = 404
        raise requests.HTTPError("404 Client Error: Not Found", response=resp)

    inner.get_remote.side_effect = gone
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        with pytest.raises(BackendError) as ei:
            await _backend_for(found).fetch_result(rid, Path("ignored"))
    finally:
        await persistence.close()
    rec = ei.value.error_record
    assert rec.error_subclass == "remote_result_expired"
    assert rec.recovery_action == "modify_request_parameters"
    assert "retention" in rec.message


@pytest.mark.asyncio
async def test_download_runs_a_pending_cds_done_row_in_the_foreground(tmp_path: Path, monkeypatch) -> None:
    from copernicus_mcp.backends.cds import backend as backend_mod

    class _Slow(_FakeMultiurl):
        def download(self, url, *, target, **kwargs) -> None:
            assert self.block is not None
            self.block.wait(timeout=30)
            with open(target, "ab") as fh:
                fh.write(_PAYLOAD)

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    rid = await _submit(tmp_path, status_map)
    gate = threading.Event()
    monkeypatch.setattr(backend_mod, "multiurl", _Slow(block=gate))
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found, cds_download_inline_grace_seconds=0.0)
        backend.set_fetch_wait(10.0)
        fetch = asyncio.create_task(backend.fetch_result(rid, Path("ignored")))
        await asyncio.sleep(0.3)
        assert not fetch.done(), "the foreground download waits for the transfer"
        gate.set()
        out = await fetch
    finally:
        gate.set()
        await persistence.close()
    assert out["status"] == "successful"
    assert Path(out["result"]["filepath"]).read_bytes() == _PAYLOAD


@pytest.mark.asyncio
async def test_download_of_an_unfinished_cds_job_says_not_ready(tmp_path: Path, monkeypatch) -> None:
    from copernicus_mcp.errors import BackendError

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    rid = await _submit(tmp_path, status_map)
    status_map[rid] = "running"
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        with pytest.raises(BackendError) as ei:
            await _backend_for(found).fetch_result(rid, Path("ignored"))
    finally:
        await persistence.close()
    assert ei.value.error_record.error_subclass == "result_not_ready"
    assert "cds wait <request_id>" in (ei.value.error_record.next_action_hint or "")


@pytest.mark.asyncio
async def test_chunk_parent_download_repairs_a_child_with_a_lost_entry(tmp_path: Path, monkeypatch) -> None:
    from tests.unit.test_cds_chunk_partial import (
        _drive_all_chunks_successful,
        _patch_cdsapi_children,
        _patch_costing_by_shape,
        _splittable_params,
    )

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, instance = _patch_cdsapi_children(monkeypatch, status)
    real_get_remote = instance.client.get_remote.side_effect

    class _Res:
        location = "https://objectstore.example/results/chunk.grib"
        content_length = len(b"GRIB-chunk")

    def get_remote(request_id):  # type: ignore[no-untyped-def]
        rem = real_get_remote(request_id)
        rem.get_results.return_value = _Res()
        return rem

    instance.client.get_remote.side_effect = get_remote
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found, cds_resume_downloads=False)
        out = await backend.submit(_splittable_params())
        parent_id = out["request_id"]
        await _drive_all_chunks_successful(backend, status, parent_id)
        con = sqlite3.connect(str(tmp_path / "state.db"))
        try:
            key = con.execute("SELECT cache_key FROM workflows WHERE request_id='child-3'").fetchone()[0]
            con.execute("DELETE FROM cache_entries WHERE key LIKE ?", (f"%{key.rsplit(':', 1)[-1]}%",))
            con.commit()
        finally:
            con.close()
        st = await backend.check_status(parent_id)
        assert st["result"]["complete"] is False
        res = await backend.fetch_result(parent_id, Path("ignored"))
    finally:
        await persistence.close()
    assert res["status"] == "successful"
    assert res["result"]["complete"] is True
    assert len(res["result"]["files"]) == 5


def test_cds_wait_refetches_once_when_the_entry_was_lost(monkeypatch) -> None:
    import json

    from typer.testing import CliRunner

    from copernicus_mcp import cli

    calls: list[str] = []

    class _Orch:
        async def run(self, *, backend, operation, params):  # type: ignore[no-untyped-def]
            calls.append(operation)
            if operation == "poll" and calls.count("poll") == 1:
                return {"result": {"status": "failed", "request_id": "rid", "error_details": {
                    "error_class": "CacheError", "error_subclass": "cache_entry_missing",
                    "context": {"remote_job_status": "successful", "local_only": True}}}}
            if operation == "poll":
                return {"result": {"status": "successful", "request_id": "rid", "submitted_at": "t",
                                   "updated_at": "t", "cache_key": "k", "error_details": None,
                                   "result": {"filepath": "/x.nc"}}}
            return {"result": {"status": "successful", "request_id": "rid", "cache_hit": True,
                               "result": {"filepath": "/x.nc"}}}

    class _Builder:
        def __call__(self, **kwargs):
            return self

        async def __aenter__(self):
            return _Orch()

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _Builder())
    res = CliRunner().invoke(cli.app, ["cds", "wait", "rid", "--json", "--interval", "0"])
    assert res.exit_code == 0, res.stdout + str(res.exception)
    assert calls == ["poll", "fetch", "poll"], "re-poll so wait prints its usual JSON shape"
    out = json.loads(res.stdout)
    assert out["status"] == "successful" and "submitted_at" in out



# --- Final review fixes -------------------------------------------------------


def _uuid_request_ids(monkeypatch) -> None:
    """Make the fake CDS hand out dashed-UUID job ids, like production."""
    import sys
    import uuid

    instance = sys.modules["cdsapi"].Client.return_value

    def _retrieve(name, request, target):  # type: ignore[no-untyped-def]
        from unittest.mock import MagicMock

        remote = MagicMock()
        remote.request_id = str(uuid.uuid4())
        return remote

    instance.retrieve.side_effect = _retrieve


@pytest.mark.asyncio
async def test_lost_entry_hint_survives_the_sanitiser_for_uuid_ids(tmp_path: Path, monkeypatch) -> None:
    """Fable final M1: a dashed UUID inside a hint string was redacted. The
    hint uses a placeholder; the id itself sits under ``context.request_id``."""
    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    _uuid_request_ids(monkeypatch)
    rid = await _submit(tmp_path, status_map)
    assert rid.count("-") == 4
    await _drive_to_successful(tmp_path, rid, status_map, monkeypatch)
    _break_registration(tmp_path, rid, legacy_name=False)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        out = await _backend_for(found).check_status(rid)
    finally:
        await persistence.close()
    err = out["error_details"]
    assert "[REDACTED]" not in err["next_action_hint"]
    assert "[REDACTED]" not in err["message"]
    assert err["context"]["request_id"] == rid


@pytest.mark.asyncio
async def test_refetch_replaces_a_truncated_own_file_instead_of_reusing_it(tmp_path: Path, monkeypatch) -> None:
    """Codex final HIGH / Fable final M2: the rejected (wrong-size) own file
    sat at the re-fetch's publish name; EEXIST then registered it."""
    rid, own, inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=False, truncate=True)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        out = await _backend_for(found).fetch_result(rid, Path("ignored"))
    finally:
        await persistence.close()
    path = Path(out["result"]["filepath"])
    assert path.read_bytes() == _PAYLOAD
    assert _db_state(tmp_path / "state.db", rid)[1] == [str(path)]


@pytest.mark.asyncio
async def test_refetch_is_bounded_by_the_fetch_wait_and_resumes(tmp_path: Path, monkeypatch) -> None:
    """Codex final M2: a stalled re-download must not hold the call past its
    budget; the next call resumes it."""
    import time

    from copernicus_mcp.backends.cds import backend as backend_mod
    from copernicus_mcp.errors import BackendError

    rid, legacy, inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True, truncate=True)
    gate = threading.Event()

    class _Stall(_FakeMultiurl):
        def download(self, url, *, target, **kwargs) -> None:
            self.calls.append(0)
            if len(self.calls) == 1:
                gate.wait(timeout=30)
            with open(target, "ab") as fh:
                fh.write(_PAYLOAD)

    monkeypatch.setattr(backend_mod, "multiurl", _Stall())
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found)
        backend.set_fetch_wait(0.2)
        t0 = time.monotonic()
        with pytest.raises(BackendError) as ei:
            await backend.fetch_result(rid, Path("ignored"))
        assert time.monotonic() - t0 < 3.0
        assert ei.value.error_record.error_subclass == "result_not_ready"
        gate.set()
        backend.set_fetch_wait(10.0)
        out = await backend.fetch_result(rid, Path("ignored"))
    finally:
        gate.set()
        await persistence.close()
    assert Path(out["result"]["filepath"]).read_bytes() == _PAYLOAD


@pytest.mark.asyncio
async def test_repair_never_adopts_a_file_an_eviction_is_removing(tmp_path: Path, monkeypatch) -> None:
    """Codex final M3: another process evicted the entry (tombstone written,
    file not yet unlinked). Adopting that path lets the evictor delete a newly
    registered file — re-fetch under a fresh name instead."""
    from copernicus_mcp.backends.cds.backend import _cache_storage_key

    rid, own, inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=False)
    con = sqlite3.connect(str(tmp_path / "state.db"))
    try:
        key = con.execute("SELECT cache_key FROM workflows WHERE request_id=?", (rid,)).fetchone()[0]
        con.execute(
            "INSERT INTO cache_evictions (namespace, key, file_path, reason, evicted_at) VALUES ('file', ?, ?, 'lru', '2026-10-19T00:00:00Z')",
            (_cache_storage_key(key), str(own)),
        )
        con.commit()
    finally:
        con.close()
    inner.download_results.reset_mock()
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        out = await _backend_for(found).fetch_result(rid, Path("ignored"))
    finally:
        await persistence.close()
    path = Path(out["result"]["filepath"])
    assert path != own
    assert path.read_bytes() == _PAYLOAD


def test_cds_wait_honours_its_deadline_before_recovering(monkeypatch) -> None:
    from typer.testing import CliRunner

    from copernicus_mcp import cli

    calls: list[str] = []

    class _Orch:
        async def run(self, *, backend, operation, params):  # type: ignore[no-untyped-def]
            calls.append(operation)
            return {"result": {"status": "failed", "request_id": "rid", "error_details": {
                "error_class": "CacheError", "error_subclass": "cache_entry_missing",
                "context": {"remote_job_status": "successful", "local_only": True}}}}

    class _Builder:
        def __call__(self, **kwargs):
            return self

        async def __aenter__(self):
            return _Orch()

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _Builder())
    res = CliRunner().invoke(cli.app, ["cds", "wait", "rid", "--json", "--timeout", "0"])
    assert calls == ["poll"], "no recovery once the deadline has passed"
    assert res.exit_code == 1


@pytest.mark.asyncio
async def test_two_concurrent_downloads_of_a_lost_entry_both_succeed_once(tmp_path: Path, monkeypatch) -> None:
    """Fable confirmation M1/M2: two fetch calls on one lost-entry row in one
    process (an agent calling the tool twice) — the second used to join the
    first's transfer and then fail on the vanished part."""
    import time

    from copernicus_mcp.backends.cds import backend as backend_mod

    rid, legacy, inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True, truncate=True)

    class _Slow(_FakeMultiurl):
        def download(self, url, *, target, **kwargs) -> None:
            self.calls.append(0)
            time.sleep(0.3)
            with open(target, "ab") as fh:
                fh.write(_PAYLOAD)

    slow = _Slow()
    monkeypatch.setattr(backend_mod, "multiurl", slow)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found)
        backend.set_fetch_wait(10.0)
        a, b = await asyncio.gather(
            backend.fetch_result(rid, Path("ignored")),
            backend.fetch_result(rid, Path("ignored")),
        )
    finally:
        await persistence.close()
    assert a["status"] == b["status"] == "successful"
    assert a["result"]["filepath"] == b["result"]["filepath"]
    assert len(slow.calls) == 1, "one re-download for both calls"
