"""T-CDS-FINALISE-ATOMIC Task 001: a successful CDS row always has its cache
entry.

Field incident: a one-shot CLI exited while the background finalise ran; the
row was already committed ``successful`` but the cache entry was never
recorded, so every later read synthesised ``cache_eviction`` and only a fresh
CDS job recovered. The fault matrix below interrupts the finalise at every
persistence call (cancel, or a store closed under it) and checks that the row
is either still re-drivable or fully registered — and that a fresh backend
then finishes it.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_cds_resume_download import (
    _HALF,
    _PAYLOAD,
    _era5,
    _fake_creds,
    _FakeMultiurl,
    _make_foundation,
    _patch_cdsapi_resumable,
    _patch_costing_flat,
)

_FILE_NS = "file"


class _FaultProxy:
    """Wraps the persistence backend; counts the calls made from the finalise
    task and injects one fault at call ``fault_at``."""

    def __init__(self, real: Any, *, fault_at: int | None, mode: str) -> None:
        self._real = real
        self._fault_at = fault_at
        self._mode = mode
        self.finalise_task: asyncio.Task[Any] | None = None
        self.calls = 0
        self.fired = False

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._real, name)
        if name == "iter_cache_entries_by_namespace" or not asyncio.iscoroutinefunction(attr):
            return attr

        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            task = asyncio.current_task()
            if task is not None and task is self.finalise_task:
                n = self.calls
                self.calls += 1
                if n == self._fault_at and not self.fired:
                    self.fired = True
                    if self._mode == "cancel":
                        task.cancel()
                        await asyncio.sleep(0)
                    else:
                        await self._real.close()
            return await attr(*args, **kwargs)

        return wrapped


def _proxied_foundation(tmp_path: Path, *, fault_at: int | None, mode: str):
    from copernicus_mcp.cache import CacheManager

    found, persistence = _make_foundation(tmp_path)
    proxy = _FaultProxy(persistence, fault_at=fault_at, mode=mode)
    cache = CacheManager(
        cache_directory=found.config.storage.cache_directory,
        persistence=proxy,  # type: ignore[arg-type]
        size_limit_bytes=10 * 1024 * 1024,
    )
    return dataclasses.replace(found, persistence=proxy, cache=cache), persistence, proxy


def _backend_for(found, **budget):
    from copernicus_mcp.backends.cds.backend import CdsBackend

    if budget:
        b = found.config.budget.model_copy(update=budget)
        found = dataclasses.replace(found, config=found.config.model_copy(update={"budget": b}))
    return CdsBackend(foundation=found, credentials=_fake_creds())


def _db_state(db: Path, rid: str) -> tuple[str, list[str]]:
    """Read the row + the cache entries straight from SQLite (a fresh reader,
    independent of any backend object)."""
    con = sqlite3.connect(str(db))
    try:
        status = con.execute("SELECT status, cache_key FROM workflows WHERE request_id=?", (rid,)).fetchone()
        assert status is not None
        key_suffix = status[1]
        paths = [
            r[0]
            for r in con.execute(
                "SELECT file_path FROM cache_entries WHERE namespace=? AND key LIKE ?",
                (_FILE_NS, f"%{key_suffix}%"),
            )
        ]
        return status[0], paths
    finally:
        con.close()


async def _drive_to_successful(tmp_path: Path, rid: str, status_map: dict[str, str], monkeypatch) -> dict[str, Any]:
    """A fresh process: new store connection, new backend, poll until done."""
    from copernicus_mcp.backends.cds import backend as backend_mod

    monkeypatch.setattr(backend_mod, "multiurl", _FakeMultiurl())
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found, cds_download_inline_grace_seconds=5.0)
        final: dict[str, Any] = {}
        for _ in range(10):
            final = await backend.check_status(rid)
            if final["status"] == "successful":
                break
            await asyncio.sleep(0.02)
        return final
    finally:
        await persistence.close()


async def _submit(tmp_path: Path, status_map: dict[str, str]) -> str:
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        out = await _backend_for(found).submit(_era5())
    finally:
        await persistence.close()
    rid = out["request_id"]
    status_map[rid] = "successful"
    return rid


async def _run_faulted_finalise(tmp_path: Path, rid: str, *, fault_at: int | None, mode: str, monkeypatch) -> _FaultProxy:
    from copernicus_mcp.backends.cds import backend as backend_mod

    monkeypatch.setattr(backend_mod, "multiurl", _FakeMultiurl())
    found, persistence, proxy = _proxied_foundation(tmp_path, fault_at=fault_at, mode=mode)
    await persistence.initialise()
    backend = _backend_for(found, cds_download_inline_grace_seconds=0.0)
    real_create_task = asyncio.create_task

    def spy_create_task(coro, **kw):  # type: ignore[no-untyped-def]
        task = real_create_task(coro, **kw)
        if getattr(coro, "__name__", "") == "_finalise_successful":
            proxy.finalise_task = task
        return task

    monkeypatch.setattr(asyncio, "create_task", spy_create_task)
    try:
        with contextlib.suppress(Exception):
            await backend.check_status(rid)
        task = proxy.finalise_task
        assert task is not None, "check_status did not spawn a finalise"
        with contextlib.suppress(BaseException):
            await task
    finally:
        monkeypatch.setattr(asyncio, "create_task", real_create_task)
        with contextlib.suppress(Exception):
            await persistence.close()
    return proxy


def _assert_invariant(db: Path, rid: str) -> None:
    status, paths = _db_state(db, rid)
    if status == "successful":
        assert len(paths) == 1, f"successful row must have exactly one entry, got {paths}"
        assert Path(paths[0]).exists(), "the registered file must exist"
    else:
        assert status in ("queued", "running"), f"interrupted finalise must leave a re-drivable row, got {status}"
        assert paths == [], f"a non-successful row must not be registered, got {paths}"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["cancel", "close"])
async def test_fault_matrix_never_leaves_successful_without_entry(tmp_path: Path, monkeypatch, mode: str) -> None:
    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)

    # Clean run: how many persistence calls does one finalise make?
    clean_dir = tmp_path / "clean"
    rid = await _submit(clean_dir, status_map)
    clean = await _run_faulted_finalise(clean_dir, rid, fault_at=None, mode=mode, monkeypatch=monkeypatch)
    assert clean.calls > 0
    _assert_invariant(clean_dir / "state.db", rid)

    for n in range(clean.calls):
        run_dir = tmp_path / f"{mode}-{n}"
        rid = await _submit(run_dir, status_map)
        proxy = await _run_faulted_finalise(run_dir, rid, fault_at=n, mode=mode, monkeypatch=monkeypatch)
        assert proxy.fired
        _assert_invariant(run_dir / "state.db", rid)
        final = await _drive_to_successful(run_dir, rid, status_map, monkeypatch)
        assert final["status"] == "successful", f"fault at call {n} left a dead end: {final}"
        assert Path(final["result"]["filepath"]).read_bytes() == _PAYLOAD
        _assert_invariant(run_dir / "state.db", rid)


@pytest.mark.asyncio
async def test_lost_commit_ack_does_not_fail_a_registered_row(tmp_path: Path, monkeypatch) -> None:
    """COMMIT reached the database but the caller saw an error (NFS): the
    finalise must not then write ``failed`` over a registered row."""
    from copernicus_mcp.backends.cds import backend as backend_mod
    from copernicus_mcp.persistence import SqliteBackend

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    monkeypatch.setattr(backend_mod, "multiurl", _FakeMultiurl())
    rid = await _submit(tmp_path, status_map)

    real = SqliteBackend.commit_success_with_cache_entry

    async def commit_then_raise(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        await real(self, *args, **kwargs)
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(SqliteBackend, "commit_success_with_cache_entry", commit_then_raise)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found, cds_download_inline_grace_seconds=5.0)
        with contextlib.suppress(Exception):
            await backend.check_status(rid)
        for task in list(backend._downloads.values()):
            with contextlib.suppress(BaseException):
                await task
    finally:
        await persistence.close()
    status, paths = _db_state(tmp_path / "state.db", rid)
    assert status == "successful"
    assert len(paths) == 1 and Path(paths[0]).exists()


@pytest.mark.asyncio
async def test_kill_after_publish_retry_completes_without_second_download(tmp_path: Path, monkeypatch) -> None:
    from copernicus_mcp.backends.cds import backend as backend_mod
    from copernicus_mcp.persistence import SqliteBackend

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    inner = _patch_cdsapi_resumable(monkeypatch, status_map)
    monkeypatch.setattr(backend_mod, "multiurl", _FakeMultiurl())
    rid = await _submit(tmp_path, status_map)

    real = SqliteBackend.commit_success_with_cache_entry

    async def die_before_commit(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise asyncio.CancelledError

    monkeypatch.setattr(SqliteBackend, "commit_success_with_cache_entry", die_before_commit)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found, cds_download_inline_grace_seconds=5.0)
        with contextlib.suppress(BaseException):
            await backend.check_status(rid)
        for task in list(backend._downloads.values()):
            with contextlib.suppress(BaseException):
                await task
    finally:
        await persistence.close()
    status, paths = _db_state(tmp_path / "state.db", rid)
    assert status in ("queued", "running") and paths == []

    monkeypatch.setattr(SqliteBackend, "commit_success_with_cache_entry", real)
    second = _FakeMultiurl()
    monkeypatch.setattr(backend_mod, "multiurl", second)
    inner.download_results.reset_mock()
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found, cds_download_inline_grace_seconds=5.0)
        final = await backend.check_status(rid)
    finally:
        await persistence.close()
    assert final["status"] == "successful"
    assert second.calls == [], "the published file must be reused, not downloaded again"
    assert inner.download_results.call_count == 0
    assert Path(final["result"]["filepath"]).read_bytes() == _PAYLOAD


@pytest.mark.asyncio
async def test_published_name_is_request_scoped(tmp_path: Path, monkeypatch) -> None:
    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    rid = await _submit(tmp_path, status_map)
    final = await _drive_to_successful(tmp_path, rid, status_map, monkeypatch)
    from copernicus_mcp.backends.cds.backend import _request_file_tag

    name = Path(final["result"]["filepath"]).name
    assert f".{_request_file_tag(rid)}." in name
    assert name.endswith(".nc")


@pytest.mark.asyncio
async def test_two_backends_finalising_one_row_register_exactly_once(tmp_path: Path, monkeypatch) -> None:
    """Two processes (two store connections), resume off: both download, one
    commits, the other must neither fail the row nor delete the registered file."""
    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    rid = await _submit(tmp_path, status_map)

    fa, pa = _make_foundation(tmp_path)
    fb, pb = _make_foundation(tmp_path)
    await pa.initialise()
    await pb.initialise()
    try:
        ba = _backend_for(fa, cds_resume_downloads=False, cds_download_inline_grace_seconds=5.0)
        bb = _backend_for(fb, cds_resume_downloads=False, cds_download_inline_grace_seconds=5.0)
        results = await asyncio.gather(ba.check_status(rid), bb.check_status(rid), return_exceptions=True)
        for b in (ba, bb):
            for task in list(b._downloads.values()):
                with contextlib.suppress(BaseException):
                    await task
    finally:
        await pa.close()
        await pb.close()
    assert not any(isinstance(r, BaseException) for r in results), results
    status, paths = _db_state(tmp_path / "state.db", rid)
    assert status == "successful"
    assert len(paths) == 1 and Path(paths[0]).exists()


@pytest.mark.asyncio
async def test_force_refresh_supersedes_the_old_file_after_commit(tmp_path: Path, monkeypatch) -> None:
    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    first = await _submit(tmp_path, status_map)
    one = await _drive_to_successful(tmp_path, first, status_map, monkeypatch)
    old_path = Path(one["result"]["filepath"])

    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        out = await _backend_for(found).submit(_era5(force_refresh=True))
    finally:
        await persistence.close()
    second = out["request_id"]
    assert second != first
    status_map[second] = "successful"
    two = await _drive_to_successful(tmp_path, second, status_map, monkeypatch)
    new_path = Path(two["result"]["filepath"])
    assert new_path != old_path
    assert new_path.exists()
    assert not old_path.exists(), "the superseded file is removed after the commit"
    _, paths = _db_state(tmp_path / "state.db", second)
    assert paths == [str(new_path)]


@pytest.mark.asyncio
async def test_cancel_removes_an_uncommitted_publication(tmp_path: Path, monkeypatch) -> None:
    from copernicus_mcp.backends.cds import backend as backend_mod
    from copernicus_mcp.backends.cds.backend import _request_file_tag
    from copernicus_mcp.persistence import SqliteBackend

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    monkeypatch.setattr(backend_mod, "multiurl", _FakeMultiurl())
    rid = await _submit(tmp_path, status_map)

    async def die_before_commit(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise asyncio.CancelledError

    monkeypatch.setattr(SqliteBackend, "commit_success_with_cache_entry", die_before_commit)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found, cds_download_inline_grace_seconds=5.0)
        with contextlib.suppress(BaseException):
            await backend.check_status(rid)
        for task in list(backend._downloads.values()):
            with contextlib.suppress(BaseException):
                await task
        published = list((tmp_path / "cache").rglob(f"cds_*.{_request_file_tag(rid)}.*"))
        assert published, "precondition: the file was published before the kill"
        out = await backend.cancel(rid)
        assert out["status"] == "cancelled"
    finally:
        await persistence.close()
    assert not list((tmp_path / "cache").rglob(f"cds_*.{_request_file_tag(rid)}.*"))


# --- Review round 1 (code): one writer per part inode -----------------------


class _SlowThenComplete(_FakeMultiurl):
    """Writes half, waits on ``gate``, then writes the rest — a live writer
    that is slow, not dead."""

    def download(self, url, *, target, **kwargs) -> None:
        existing = os.path.getsize(target) if os.path.exists(target) else 0
        self.calls.append(existing)
        with open(target, "ab") as fh:
            if existing < _HALF:
                fh.write(_PAYLOAD[existing:_HALF])
                fh.flush()
                assert self.block is not None
                self.block.wait(timeout=30)
                fh.write(_PAYLOAD[_HALF:])
            else:
                fh.write(_PAYLOAD[existing:])


def _plant_foreign_part(found, rid: str, *, name: str, size: int, age_s: float) -> Path:
    from tests.unit.test_cds_resume_download import _FakeResults

    staging = found.cache.staging_root_for("cds")
    staging.mkdir(parents=True, exist_ok=True)
    part = staging / name
    part.write_bytes(_PAYLOAD[:size])
    meta = Path(str(part) + ".meta.json")
    meta.write_text(json.dumps({"expected_size": len(_PAYLOAD), "location": _FakeResults.location}))
    old = time.time() - age_s
    os.utime(part, (old, old))
    os.utime(meta, (old, old))
    return part


@pytest.mark.asyncio
async def test_two_resume_pollers_never_fail_or_corrupt(tmp_path: Path, monkeypatch) -> None:
    """Fable review HIGH-1: two processes, resume ON (default), polling one
    row together — 0.7.0's shared part made the loser write ``failed`` and the
    winner delete its own file. Now each round must end successful or pending,
    never failed, and the result is byte-exact."""
    from copernicus_mcp.backends.cds import backend as backend_mod

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    monkeypatch.setattr(backend_mod, "multiurl", _FakeMultiurl())
    for attempt in range(5):
        run = tmp_path / f"r{attempt}"
        rid = await _submit(run, status_map)
        fa, pa = _make_foundation(run)
        fb, pb = _make_foundation(run)
        await pa.initialise()
        await pb.initialise()
        try:
            ba = _backend_for(fa, cds_download_inline_grace_seconds=5.0)
            bb = _backend_for(fb, cds_download_inline_grace_seconds=5.0)
            final = None
            for _ in range(6):
                outs = await asyncio.gather(ba.check_status(rid), bb.check_status(rid))
                for b in (ba, bb):
                    for task in list(b._downloads.values()):
                        with contextlib.suppress(BaseException):
                            await task
                assert all(o["status"] != "failed" for o in outs), outs
                row = await pa.fetch_workflow(rid)
                assert row is not None and row["status"] != "failed"
                if row["status"] == "successful":
                    final = await ba.check_status(rid)
                    break
            assert final is not None and final["status"] == "successful"
            assert Path(final["result"]["filepath"]).read_bytes() == _PAYLOAD
        finally:
            await pa.close()
            await pb.close()


@pytest.mark.asyncio
async def test_live_foreign_part_makes_the_second_process_back_off(tmp_path: Path, monkeypatch) -> None:
    import threading

    from copernicus_mcp.backends.cds import backend as backend_mod

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    gate = threading.Event()
    slow = _SlowThenComplete(block=gate)
    monkeypatch.setattr(backend_mod, "multiurl", slow)
    rid = await _submit(tmp_path, status_map)
    fa, pa = _make_foundation(tmp_path)
    fb, pb = _make_foundation(tmp_path)
    await pa.initialise()
    await pb.initialise()
    try:
        bb = _backend_for(fb, cds_download_inline_grace_seconds=0.0)
        await bb.check_status(rid)  # B starts and stalls at half
        staging = fb.cache.staging_root_for("cds")
        for _ in range(500):
            if any(p.stat().st_size >= 480 for p in staging.glob("*.part")):
                break
            await asyncio.sleep(0.01)
        ba = _backend_for(fa, cds_download_inline_grace_seconds=1.0)
        out = await ba.check_status(rid)
        assert out["status"] == "running" and out.get("phase") == "downloading"
        assert slow.calls == [0], "A must not start a second writer beside B's live part"
        gate.set()
        for task in list(bb._downloads.values()):
            with contextlib.suppress(BaseException):
                await task
        final = await ba.check_status(rid)
    finally:
        gate.set()
        await pa.close()
        await pb.close()
    assert final["status"] == "successful"
    assert Path(final["result"]["filepath"]).read_bytes() == _PAYLOAD


@pytest.mark.asyncio
@pytest.mark.parametrize("name_kind", ["foreign", "legacy"])
async def test_stale_foreign_part_is_adopted_by_copy(tmp_path: Path, monkeypatch, name_kind: str) -> None:
    """A part left by a dead process (or by 0.7.0's ``<rid>.part`` naming) is
    resumed from — copied into this process's own part, never written in
    place — and then removed."""
    from copernicus_mcp.backends.cds import backend as backend_mod

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    fake = _FakeMultiurl()
    monkeypatch.setattr(backend_mod, "multiurl", fake)
    rid = await _submit(tmp_path, status_map)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        name = f"{rid}.deadbeef0000.part" if name_kind == "foreign" else f"{rid}.part"
        foreign = _plant_foreign_part(found, rid, name=name, size=300, age_s=3600)
        backend = _backend_for(found, cds_download_inline_grace_seconds=5.0)
        final = await backend.check_status(rid)
    finally:
        await persistence.close()
    assert final["status"] == "successful"
    assert fake.calls == [300], "resumed from the copied prefix"
    assert Path(final["result"]["filepath"]).read_bytes() == _PAYLOAD
    assert not foreign.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("name_kind", ["foreign", "legacy"])
async def test_surviving_foreign_writer_cannot_change_the_registered_file(
    tmp_path: Path, monkeypatch, name_kind: str
) -> None:
    """Codex review HIGH-1: a stalled writer of the stale part wakes up after
    the adopter committed. Its bytes must not reach the registered file."""
    from copernicus_mcp.backends.cds import backend as backend_mod

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    monkeypatch.setattr(backend_mod, "multiurl", _FakeMultiurl())
    rid = await _submit(tmp_path, status_map)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        name = f"{rid}.deadbeef0000.part" if name_kind == "foreign" else f"{rid}.part"
        foreign = _plant_foreign_part(found, rid, name=name, size=300, age_s=3600)
        stalled_writer = open(foreign, "ab")  # noqa: SIM115 — the writer outlives the test body
        backend = _backend_for(found, cds_download_inline_grace_seconds=5.0)
        final = await backend.check_status(rid)
        stalled_writer.write(_PAYLOAD[300:])
        stalled_writer.close()
    finally:
        await persistence.close()
    path = Path(final["result"]["filepath"])
    assert path.read_bytes() == _PAYLOAD
    assert path.stat().st_size == len(_PAYLOAD)


@pytest.mark.asyncio
async def test_own_part_vanishing_mid_transfer_backs_off(tmp_path: Path, monkeypatch) -> None:
    """Our stalled part was adopted (and removed) by another process: our
    transfer ends with no file at the part path. That is not a failed job."""
    from copernicus_mcp.backends.cds import backend as backend_mod

    class _VanishAtEnd(_FakeMultiurl):
        def download(self, url, *, target, **kwargs) -> None:
            with open(target, "ab") as fh:
                fh.write(_PAYLOAD)
            os.unlink(target)

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    monkeypatch.setattr(backend_mod, "multiurl", _VanishAtEnd())
    rid = await _submit(tmp_path, status_map)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found, cds_download_inline_grace_seconds=5.0)
        await backend.check_status(rid)
        for task in list(backend._downloads.values()):
            with contextlib.suppress(BaseException):
                await task
        row = await persistence.fetch_workflow(rid)
        assert row is not None and row["status"] in ("queued", "running")
    finally:
        await persistence.close()
    final = await _drive_to_successful(tmp_path, rid, status_map, monkeypatch)
    assert final["status"] == "successful"


@pytest.mark.asyncio
async def test_own_part_size_mismatch_fails_the_row(tmp_path: Path, monkeypatch) -> None:
    """With one writer per part, a wrong final size is a broken transfer
    again — fail, as before (no contention budget)."""
    from copernicus_mcp.backends.cds import backend as backend_mod

    class _Overshoot(_FakeMultiurl):
        def download(self, url, *, target, **kwargs) -> None:
            with open(target, "ab") as fh:
                fh.write(_PAYLOAD + b"extra")

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    monkeypatch.setattr(backend_mod, "multiurl", _Overshoot())
    rid = await _submit(tmp_path, status_map)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found, cds_download_inline_grace_seconds=5.0)
        await backend.check_status(rid)
        for task in list(backend._downloads.values()):
            with contextlib.suppress(BaseException):
                await task
        row = await persistence.fetch_workflow(rid)
    finally:
        await persistence.close()
    assert row is not None and row["status"] == "failed"


def test_publish_without_hard_links_uses_an_atomic_rename(tmp_path: Path, monkeypatch) -> None:
    """Codex review MEDIUM-3: on a mount that refuses hard links the fallback
    must not call ``link`` again."""
    import errno

    from copernicus_mcp.cache import CacheManager

    def no_link(src, dst, *a, **k):  # type: ignore[no-untyped-def]
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", no_link)
    mgr = CacheManager(cache_directory=tmp_path / "cache", persistence=None, size_limit_bytes=1)  # type: ignore[arg-type]
    src = tmp_path / "staged.nc"
    src.write_bytes(_PAYLOAD)
    final = tmp_path / "cache" / "zone" / "cds_h.tag.nc"
    assert mgr.publish_request_file(src, final) == final
    assert final.read_bytes() == _PAYLOAD
    assert not src.exists()
    assert not [p for p in final.parent.iterdir() if p.name.startswith(".")]


@pytest.mark.asyncio
@pytest.mark.parametrize("when", ["before", "after"])
async def test_commit_interrupted_inside_the_transaction(tmp_path: Path, monkeypatch, when: str) -> None:
    """Both reviewers: the matrix above interrupts BETWEEN persistence calls.
    This one cancels the awaiter at every statement INSIDE the transaction,
    before the statement runs and after it ran in the worker (COMMIT done,
    awaiter cancelled), then lets the same connection commit an unrelated
    write: no half of the transaction may ever become durable."""
    from copernicus_mcp.persistence import SqliteBackend

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)

    class _ConnProxy:
        def __init__(self, real: Any, fault_at: int) -> None:
            self._real = real
            self._fault_at = fault_at
            self.n = 0

        @property
        def in_transaction(self) -> bool:
            return bool(self._real.in_transaction)

        def __getattr__(self, name: str) -> Any:
            attr = getattr(self._real, name)
            if name not in ("execute", "commit", "rollback"):
                return attr

            async def wrapped(*args: Any, **kwargs: Any) -> Any:
                k = self.n
                self.n += 1
                hit = k == self._fault_at and name != "rollback"
                task = asyncio.current_task()
                if hit and when == "before":
                    assert task is not None
                    task.cancel()
                    await asyncio.sleep(0)
                result = await attr(*args, **kwargs)
                if hit and when == "after":
                    assert task is not None
                    task.cancel()
                    await asyncio.sleep(0)
                return result

            return wrapped

    for fault_at in range(6):
        run = tmp_path / f"{when}-{fault_at}"
        rid = await _submit(run, status_map)
        store = SqliteBackend(run / "state.db")
        await store.initialise()
        real_conn = store._conn
        proxy = _ConnProxy(real_conn, fault_at)
        store._conn = proxy  # type: ignore[assignment]
        entry = {
            "namespace": _FILE_NS, "key": "cds:file:k", "value_json": "{}", "file_path": str(run / "f.nc"),
            "size_bytes": 1, "content_type": "application/x-netcdf",
            "created_at": "2026-10-19T00:00:00Z", "last_accessed_at": "2026-10-19T00:00:00Z",
        }
        with contextlib.suppress(asyncio.CancelledError):
            await store.commit_success_with_cache_entry(rid, entry)  # type: ignore[arg-type]
        store._conn = real_conn
        assert not real_conn.in_transaction
        other = dict(entry, key="cds:file:other", file_path=str(run / "o.nc"))
        await store.record_cache_entry(other)  # type: ignore[arg-type]
        await store.close()
        con = sqlite3.connect(str(run / "state.db"))
        try:
            st = con.execute("SELECT status FROM workflows WHERE request_id=?", (rid,)).fetchone()[0]
            n = con.execute("SELECT count(*) FROM cache_entries WHERE key='cds:file:k'").fetchone()[0]
        finally:
            con.close()
        assert (st, n) in ((st, 0), ("successful", 1)) and (st == "successful") == (n == 1), (fault_at, st, n)


@pytest.mark.asyncio
async def test_lock_dropped_when_a_step_after_the_commit_is_cancelled(tmp_path: Path, monkeypatch) -> None:
    """Fable review MEDIUM-2: the row is terminal once the commit returned,
    even if a later step (provenance) is cancelled."""
    from copernicus_mcp.backends.cds import backend as backend_mod
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    monkeypatch.setattr(backend_mod, "multiurl", _FakeMultiurl())
    rid = await _submit(tmp_path, status_map)

    async def cancelled_provenance(self, **kwargs):  # type: ignore[no-untyped-def]
        raise asyncio.CancelledError

    monkeypatch.setattr(CdsBackend, "_record_cds_provenance", cancelled_provenance)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found, cds_download_inline_grace_seconds=5.0)
        with contextlib.suppress(BaseException):
            await backend.check_status(rid)
        for task in list(backend._downloads.values()):
            with contextlib.suppress(BaseException):
                await task
        assert (await persistence.fetch_workflow(rid))["status"] == "successful"
        assert rid not in backend._finalise_locks
    finally:
        await persistence.close()


# --- Code review round 2: no file vanishes under a pending row's writer -----


class _RealTailMultiurl(_FakeMultiurl):
    """Mirrors multiurl 0.3.9's trust-size tail: after the stream it asserts
    ``os.path.getsize(target) == size`` by PATH — a part unlinked under the
    writer raises ``FileNotFoundError`` inside the library."""

    def __init__(self, block=None, *, stall_first: bool = False) -> None:
        super().__init__(block=block)
        self.stall_first = stall_first

    def download(self, url, *, target, **kwargs) -> None:
        existing = os.path.getsize(target) if os.path.exists(target) else 0
        self.calls.append(existing)
        with open(target, "ab") as fh:
            if self.stall_first and len(self.calls) == 1 and existing < _HALF:
                fh.write(_PAYLOAD[existing:_HALF])
                fh.flush()
                assert self.block is not None
                self.block.wait(timeout=30)
                fh.write(_PAYLOAD[_HALF:])
            else:
                fh.write(_PAYLOAD[existing:])
        assert os.path.getsize(target) == len(_PAYLOAD)


class _TwoWriters(_FakeMultiurl):
    """Call 1 (B) writes half, then waits on ``gate_b``; call 2 (A, the
    adopter) waits on ``gate_a`` before streaming. Both end with multiurl's
    by-path size assertion."""

    def __init__(self, gate_b, gate_a) -> None:
        super().__init__()
        self.gate_b = gate_b
        self.gate_a = gate_a

    def download(self, url, *, target, **kwargs) -> None:
        existing = os.path.getsize(target) if os.path.exists(target) else 0
        self.calls.append(existing)
        call = len(self.calls)
        with open(target, "ab") as fh:
            if call == 1:
                fh.write(_PAYLOAD[existing:_HALF])
                fh.flush()
                self.gate_b.wait(timeout=30)
                fh.write(_PAYLOAD[_HALF:])
            else:
                self.gate_a.wait(timeout=30)
                fh.write(_PAYLOAD[existing:])
        assert os.path.getsize(target) == len(_PAYLOAD)


@pytest.mark.asyncio
async def test_woken_stalled_writer_after_takeover_neither_fails_nor_deletes(tmp_path: Path, monkeypatch) -> None:
    """Both reviewers, round 2 HIGH-1: B stalls past the live window, A takes
    over by copy, B's stream then finishes BEFORE A commits. B must not fail
    the row, and the result must end registered and byte-exact."""
    import threading

    from copernicus_mcp.backends.cds import backend as backend_mod

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    gate_b, gate_a = threading.Event(), threading.Event()
    monkeypatch.setattr(backend_mod, "multiurl", _TwoWriters(gate_b, gate_a))
    rid = await _submit(tmp_path, status_map)
    fa, pa = _make_foundation(tmp_path)
    fb, pb = _make_foundation(tmp_path)
    await pa.initialise()
    await pb.initialise()
    try:
        bb = _backend_for(fb, cds_download_inline_grace_seconds=0.0)
        await bb.check_status(rid)
        staging = fb.cache.staging_root_for("cds")
        b_part = None
        for _ in range(500):
            parts = [p for p in staging.glob("*.part") if p.stat().st_size >= _HALF]
            if parts:
                b_part = parts[0]
                break
            await asyncio.sleep(0.01)
        assert b_part is not None
        old = time.time() - 3600
        os.utime(b_part, (old, old))  # B looks stalled
        ba = _backend_for(fa, cds_download_inline_grace_seconds=0.0)
        await ba.check_status(rid)  # A adopts, then waits on gate_a
        for _ in range(500):
            if len(backend_mod.multiurl.calls) >= 2:
                break
            await asyncio.sleep(0.01)
        gate_b.set()  # B finishes first
        for task in list(bb._downloads.values()):
            with contextlib.suppress(BaseException):
                await task
        mid = await pa.fetch_workflow(rid)
        gate_a.set()
        for task in list(ba._downloads.values()):
            with contextlib.suppress(BaseException):
                await task
        final = await ba.check_status(rid)
    finally:
        gate_a.set()
        gate_b.set()
        await pa.close()
        await pb.close()
    assert mid is not None and mid["status"] != "failed", "the woken writer failed a fine job"
    assert final["status"] == "successful"
    status, paths = _db_state(tmp_path / "state.db", rid)
    assert status == "successful" and len(paths) == 1
    assert Path(paths[0]).read_bytes() == _PAYLOAD


@pytest.mark.asyncio
async def test_two_adopters_of_one_stale_part_never_fail(tmp_path: Path, monkeypatch) -> None:
    from copernicus_mcp.backends.cds import backend as backend_mod

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    monkeypatch.setattr(backend_mod, "multiurl", _RealTailMultiurl())
    rid = await _submit(tmp_path, status_map)
    fa, pa = _make_foundation(tmp_path)
    fb, pb = _make_foundation(tmp_path)
    await pa.initialise()
    await pb.initialise()
    try:
        _plant_foreign_part(fa, rid, name=f"{rid}.deadbeef0000.part", size=300, age_s=3600)
        ba = _backend_for(fa, cds_download_inline_grace_seconds=5.0)
        bb = _backend_for(fb, cds_download_inline_grace_seconds=5.0)
        outs = await asyncio.gather(ba.check_status(rid), bb.check_status(rid))
        for b in (ba, bb):
            for task in list(b._downloads.values()):
                with contextlib.suppress(BaseException):
                    await task
        assert all(o["status"] != "failed" for o in outs), outs
        final = await ba.check_status(rid)
    finally:
        await pa.close()
        await pb.close()
    assert final["status"] == "successful"
    assert Path(final["result"]["filepath"]).read_bytes() == _PAYLOAD


@pytest.mark.asyncio
async def test_foreign_part_vanishing_during_adoption_backs_off(tmp_path: Path, monkeypatch) -> None:
    """The foreign part disappears between listing and copying (its owner
    promoted it, or another adopter's cleanup): back off, never fail."""
    from copernicus_mcp.backends.cds import backend as backend_mod

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    monkeypatch.setattr(backend_mod, "multiurl", _RealTailMultiurl())
    rid = await _submit(tmp_path, status_map)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    real_copy = backend_mod.shutil.copyfile
    calls = {"n": 0}

    def vanishing_copy(src, dst, *a, **k):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            raise FileNotFoundError(2, "No such file or directory", str(src))
        return real_copy(src, dst, *a, **k)

    monkeypatch.setattr(backend_mod.shutil, "copyfile", vanishing_copy)
    try:
        _plant_foreign_part(found, rid, name=f"{rid}.deadbeef0000.part", size=300, age_s=3600)
        backend = _backend_for(found, cds_download_inline_grace_seconds=5.0)
        first = await backend.check_status(rid)
        assert first["status"] in ("queued", "running")
        assert (await persistence.fetch_workflow(rid))["status"] in ("queued", "running")
        final = await backend.check_status(rid)
    finally:
        await persistence.close()
    assert final["status"] == "successful"


def test_abandoned_adoption_copy_never_replaces_an_active_part(tmp_path: Path) -> None:
    """Codex round 2 HIGH-1: a cancelled adoption's copy thread finishes after
    a retry already owns the part — it must not replace it."""
    from copernicus_mcp.backends.cds.backend import _adopt_foreign_part

    foreign = tmp_path / "rid.deadbeef0000.part"
    foreign.write_bytes(b"F" * 10)
    own = tmp_path / "rid.ownowner0000.part"
    own.write_bytes(b"ACTIVE")
    assert _adopt_foreign_part(foreign, own) is False
    assert own.read_bytes() == b"ACTIVE"
    assert foreign.exists(), "adoption never removes the foreign part"
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


@pytest.mark.asyncio
async def test_stale_foreign_parts_are_removed_after_the_commit_live_ones_kept(tmp_path: Path, monkeypatch) -> None:
    from copernicus_mcp.backends.cds import backend as backend_mod

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    monkeypatch.setattr(backend_mod, "multiurl", _RealTailMultiurl())
    rid = await _submit(tmp_path, status_map)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        stale = _plant_foreign_part(found, rid, name=f"{rid}.deadbeef0000.part", size=300, age_s=3600)
        backend = _backend_for(found, cds_download_inline_grace_seconds=5.0)
        final = await backend.check_status(rid)
    finally:
        await persistence.close()
    assert final["status"] == "successful"
    assert not stale.exists()
    assert not Path(str(stale) + ".meta.json").exists()


@pytest.mark.asyncio
async def test_race_loser_keeps_its_file_unless_the_row_was_cancelled(tmp_path: Path, monkeypatch) -> None:
    """Fable round 2 MEDIUM-1: a lost flip on a FAILED row must not destroy
    the only complete copy of the result (repair can still use it)."""
    from copernicus_mcp.backends.cds import backend as backend_mod
    from copernicus_mcp.backends.cds.backend import _request_file_tag
    from copernicus_mcp.cache import CacheManager

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    monkeypatch.setattr(backend_mod, "multiurl", _RealTailMultiurl())
    rid = await _submit(tmp_path, status_map)
    real_register = CacheManager.register_success

    async def someone_failed_it_first(self, **kwargs):  # type: ignore[no-untyped-def]
        await self._persistence.update_workflow_error(kwargs["request_id"], "failed", '{"error_class": "BackendError"}')
        return await real_register(self, **kwargs)

    monkeypatch.setattr(CacheManager, "register_success", someone_failed_it_first)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found, cds_download_inline_grace_seconds=5.0)
        await backend.check_status(rid)
        for task in list(backend._downloads.values()):
            with contextlib.suppress(BaseException):
                await task
    finally:
        await persistence.close()
    kept = [p for p in (tmp_path / "cache").rglob(f"cds_*.{_request_file_tag(rid)}.*") if ".staging" not in p.parts]
    assert kept, "the published result must survive a lost flip on a failed row"


# --- Confirmation round (codex): a complete part at transfer start ----------


class _RealPreconditionMultiurl(_FakeMultiurl):
    """Mirrors multiurl 0.3.9 on both ends: resume asserts ``bytes < size``
    (http.py:282) before streaming, and the tail asserts the size by path."""

    def download(self, url, *, target, **kwargs) -> None:
        existing = os.path.getsize(target) if os.path.exists(target) else 0
        self.calls.append(existing)
        if existing:
            assert existing < len(_PAYLOAD), (existing, len(_PAYLOAD))
        with open(target, "ab") as fh:
            fh.write(_PAYLOAD[existing:])
        assert os.path.getsize(target) == len(_PAYLOAD)


def test_resumable_download_accepts_a_part_that_is_already_complete(tmp_path: Path, monkeypatch) -> None:
    """A part that became complete before the transfer started (an adoption
    copy landed, a previous writer finished) is done — real multiurl would
    otherwise fail it on its ``bytes < size`` precondition."""
    from copernicus_mcp.backends.cds import backend as backend_mod

    fake = _RealPreconditionMultiurl()
    monkeypatch.setattr(backend_mod, "multiurl", fake)
    part = tmp_path / "rid.owner.part"
    part.write_bytes(_PAYLOAD)
    backend_mod._resumable_download("https://x/result.nc", part, len(_PAYLOAD))
    assert fake.calls == []
    assert part.read_bytes() == _PAYLOAD


@pytest.mark.asyncio
async def test_cancelled_adoption_is_adopted_by_the_next_poll_not_repeated(tmp_path: Path, monkeypatch) -> None:
    """Codex confirmation HIGH: an adoption copy that outlives its cancelled
    finalise must be the SAME tracked transfer the next poll waits on — not a
    stray copy racing a second one."""
    import threading

    from copernicus_mcp.backends.cds import backend as backend_mod

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    monkeypatch.setattr(backend_mod, "multiurl", _RealPreconditionMultiurl())
    rid = await _submit(tmp_path, status_map)
    gate = threading.Event()
    copies: list[str] = []
    real_copy = backend_mod.shutil.copyfile

    def paused_copy(src, dst, *a, **k):  # type: ignore[no-untyped-def]
        if str(src).endswith(".part"):
            copies.append(str(src))
            gate.wait(timeout=30)
        return real_copy(src, dst, *a, **k)

    monkeypatch.setattr(backend_mod.shutil, "copyfile", paused_copy)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        _plant_foreign_part(found, rid, name=f"{rid}.deadbeef0000.part", size=300, age_s=3600)
        backend = _backend_for(found, cds_download_inline_grace_seconds=0.0)
        await backend.check_status(rid)
        for _ in range(500):
            if copies:
                break
            await asyncio.sleep(0.01)
        task = backend._downloads[rid]
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        retry = asyncio.create_task(backend.check_status(rid))
        await asyncio.sleep(0.2)
        gate.set()
        await retry
        final = None
        for _ in range(20):
            final = await backend.check_status(rid)
            if final["status"] in ("successful", "failed"):
                break
            await asyncio.sleep(0.05)
    finally:
        gate.set()
        await persistence.close()
    assert final is not None and final["status"] == "successful", final
    assert len(copies) == 1, f"one adoption, adopted by the retry; got {copies}"
    assert Path(final["result"]["filepath"]).read_bytes() == _PAYLOAD


@pytest.mark.asyncio
@pytest.mark.parametrize("err", ["ENOSPC", "EACCES"])
async def test_adoption_error_other_than_vanished_downloads_from_scratch(tmp_path: Path, monkeypatch, caplog, err: str) -> None:
    """Fable confirmation MEDIUM-1: adoption is an optimisation. A copy that
    fails for any reason but "the part vanished" (full disk, another user's
    0600 part) must not loop forever as ``downloading`` without a trace: log
    it and download from scratch."""
    import errno
    import logging

    from copernicus_mcp.backends.cds import backend as backend_mod

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    fake = _RealPreconditionMultiurl()
    monkeypatch.setattr(backend_mod, "multiurl", fake)
    rid = await _submit(tmp_path, status_map)
    real_copy = backend_mod.shutil.copyfile
    code = getattr(errno, err)

    def failing_copy(src, dst, *a, **k):  # type: ignore[no-untyped-def]
        if str(src).endswith(".part"):
            raise OSError(code, os.strerror(code), str(src))
        return real_copy(src, dst, *a, **k)

    monkeypatch.setattr(backend_mod.shutil, "copyfile", failing_copy)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        _plant_foreign_part(found, rid, name=f"{rid}.deadbeef0000.part", size=300, age_s=3600)
        backend = _backend_for(found, cds_download_inline_grace_seconds=5.0)
        with caplog.at_level(logging.WARNING):
            final = None
            for _ in range(4):
                final = await backend.check_status(rid)
                if final["status"] in ("successful", "failed"):
                    break
    finally:
        await persistence.close()
    assert final is not None and final["status"] == "successful"
    assert fake.calls == [0], "downloaded from scratch, without the foreign prefix"
    assert any("adoption" in r.getMessage() for r in caplog.records)
