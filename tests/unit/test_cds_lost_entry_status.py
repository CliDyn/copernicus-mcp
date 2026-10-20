"""T-CDS-FINALISE-ATOMIC follow-up (decisions 13-19): a lost cache entry reads
as a pending download, and the re-fetch path's retro-review fixes.
"""

from __future__ import annotations

import asyncio
import gc
import threading
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_cds_finalise_atomic import _backend_for
from tests.unit.test_cds_refetch import _successful_then_broken
from tests.unit.test_cds_resume_download import _PAYLOAD, _FakeMultiurl, _make_foundation


@pytest.mark.asyncio
async def test_a_failed_transfer_never_registers_its_full_size_part(tmp_path: Path, monkeypatch) -> None:
    """R7 HIGH-1: the library wrote the full length of WRONG bytes, then
    raised. The next fetch must download again, not promote that part."""
    from copernicus_mcp.backends.cds import backend as backend_mod
    from copernicus_mcp.errors import CopernicusMcpError

    rid, _legacy, _inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True, truncate=True)

    class _FailAfterLastByte(_FakeMultiurl):
        def download(self, url, *, target, **kwargs) -> None:
            self.calls.append(0)
            with open(target, "ab") as fh:
                fh.write(b"x" * len(_PAYLOAD) if len(self.calls) == 1 else _PAYLOAD)
            if len(self.calls) == 1:
                raise OSError("connection reset after the last byte")

    fake = _FailAfterLastByte()
    monkeypatch.setattr(backend_mod, "multiurl", fake)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found)
        backend.set_fetch_wait(10.0)
        with pytest.raises(CopernicusMcpError):
            await backend.fetch_result(rid, Path("ignored"))
        out = await backend.fetch_result(rid, Path("ignored"))
    finally:
        await persistence.close()
    assert len(fake.calls) == 2, "the failed transfer's bytes must be re-downloaded"
    assert Path(out["result"]["filepath"]).read_bytes() == _PAYLOAD


@pytest.mark.asyncio
async def test_a_late_sdk_error_never_reaches_the_loop_exception_handler(tmp_path: Path, monkeypatch) -> None:
    """R7 HIGH-2: the SDK fallback's error arrives after its awaiter timed out.
    Nobody retrieved it, so asyncio printed it raw — secrets included."""
    from copernicus_mcp.backends.cds import backend as backend_mod
    from copernicus_mcp.errors import BackendError

    rid, _legacy, inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True, truncate=True)
    monkeypatch.setattr(backend_mod, "multiurl", None)  # forces the SDK fallback
    gate = threading.Event()
    finished = threading.Event()

    def _late_failure(request_id: str, target: str) -> Any:
        try:
            gate.wait(timeout=30)
            raise RuntimeError("upstream said: Authorization: Bearer sk-live-SECRET")
        finally:
            finished.set()

    inner.download_results.side_effect = _late_failure
    seen: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: seen.append(ctx))
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found)
        backend.set_fetch_wait(0.1)
        subclass = None
        try:
            await backend.fetch_result(rid, Path("ignored"))
        except BackendError as exc:  # not kept: its traceback pins the future
            subclass = exc.error_record.error_subclass
        assert subclass == "result_not_ready"
        gate.set()
        assert finished.wait(timeout=10)
        for _ in range(5):
            await asyncio.sleep(0.02)
        gc.collect()
        await asyncio.sleep(0)
    finally:
        gate.set()
        loop.set_exception_handler(previous)
        await persistence.close()
    leaked = [ctx for ctx in seen if "SECRET" in repr(ctx)]
    assert leaked == [], "an abandoned future's error reached the default handler"


@pytest.mark.asyncio
async def test_repair_registration_rechecks_eviction_inside_its_transaction(tmp_path: Path, monkeypatch) -> None:
    """R7 MEDIUM-3: the repair read "no tombstone", then another process
    committed an invalidation of that same path (unlink still to come). The
    registration itself must refuse the doomed path; a fresh copy is
    registered instead."""
    import sqlite3

    from copernicus_mcp.backends.cds.backend import _cache_storage_key

    rid, own, _inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=False)
    con = sqlite3.connect(str(tmp_path / "state.db"))
    try:
        key = con.execute("SELECT cache_key FROM workflows WHERE request_id=?", (rid,)).fetchone()[0]
        con.execute(
            "INSERT INTO cache_evictions (namespace, key, file_path, reason, evicted_at) "
            "VALUES ('file', ?, ?, 'invalidated', '2026-10-20T00:00:00Z')",
            (_cache_storage_key(key), str(own)),
        )
        con.commit()
    finally:
        con.close()
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()

    async def stale_snapshot(namespace: str, key: str) -> None:
        return None  # read before the other process's tombstone landed

    monkeypatch.setattr(persistence, "latest_cache_eviction", stale_snapshot)
    try:
        backend = _backend_for(found)
        backend.set_fetch_wait(10.0)
        out = await backend.fetch_result(rid, Path("ignored"))
    finally:
        await persistence.close()
    path = Path(out["result"]["filepath"])
    assert path != own, "the path an eviction is removing must never be registered"
    assert path.read_bytes() == _PAYLOAD


def _foreign_part(found: Any, rid: str, *, data: bytes, age_seconds: float) -> Path:
    """A part another process left: ``<rid>.<owner>.part`` + its sidecar."""
    import json
    import os
    import time

    from copernicus_mcp.backends.cds.backend import _part_basename
    from tests.unit.test_cds_resume_download import _FakeResults

    root = found.cache.staging_root_for("cds")
    root.mkdir(parents=True, exist_ok=True)
    part = root / _part_basename(rid, "f0f0f0f0f0f0")
    part.write_bytes(data)
    Path(str(part) + ".meta.json").write_text(
        json.dumps({"expected_size": len(_PAYLOAD), "location": _FakeResults.location})
    )
    then = time.time() - age_seconds
    os.utime(part, (then, then))
    return part


@pytest.mark.asyncio
async def test_refetch_resumes_a_stale_foreign_part(tmp_path: Path, monkeypatch) -> None:
    """R7 MEDIUM-5: a one-shot recovery that timed out left a prefix; the next
    process (a new owner token) must resume it, not restart from zero."""
    from copernicus_mcp.backends.cds import backend as backend_mod
    from tests.unit.test_cds_resume_download import _HALF

    rid, _legacy, _inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True, truncate=True)
    fake = _FakeMultiurl()
    monkeypatch.setattr(backend_mod, "multiurl", fake)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        _foreign_part(found, rid, data=_PAYLOAD[:_HALF], age_seconds=600)
        backend = _backend_for(found)
        backend.set_fetch_wait(10.0)
        out = await backend.fetch_result(rid, Path("ignored"))
    finally:
        await persistence.close()
    assert fake.calls == [_HALF], "resumed from the foreign prefix"
    assert Path(out["result"]["filepath"]).read_bytes() == _PAYLOAD


@pytest.mark.asyncio
async def test_refetch_backs_off_beside_a_live_foreign_writer(tmp_path: Path, monkeypatch) -> None:
    from copernicus_mcp.backends.cds import backend as backend_mod
    from copernicus_mcp.errors import BackendError
    from tests.unit.test_cds_resume_download import _HALF

    rid, _legacy, _inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True, truncate=True)
    fake = _FakeMultiurl()
    monkeypatch.setattr(backend_mod, "multiurl", fake)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        _foreign_part(found, rid, data=_PAYLOAD[:_HALF], age_seconds=1)
        backend = _backend_for(found)
        backend.set_fetch_wait(10.0)
        with pytest.raises(BackendError) as ei:
            await backend.fetch_result(rid, Path("ignored"))
    finally:
        await persistence.close()
    assert ei.value.error_record.error_subclass == "result_not_ready"
    assert fake.calls == [], "no second writer beside a live one"


@pytest.mark.asyncio
async def test_a_held_repair_lock_is_bounded_by_the_deadline(tmp_path: Path, monkeypatch) -> None:
    """R7 MEDIUM-4: waiting for another repair of the same row must not
    outlive the caller's budget."""
    import time

    from copernicus_mcp.errors import BackendError

    rid, _legacy, _inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True, truncate=True)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found)
        backend.set_fetch_wait(0.2)
        lock = backend._repair_locks.setdefault(rid, asyncio.Lock())
        await lock.acquire()
        t0 = time.monotonic()
        try:
            with pytest.raises(BackendError) as ei:
                await asyncio.wait_for(backend.fetch_result(rid, Path("ignored")), timeout=5)
        finally:
            lock.release()
    finally:
        await persistence.close()
    assert ei.value.error_record.error_subclass == "result_not_ready"
    assert time.monotonic() - t0 < 2.0


@pytest.mark.asyncio
async def test_a_stalled_results_lookup_is_bounded_by_the_deadline(tmp_path: Path, monkeypatch) -> None:
    import time

    from copernicus_mcp.errors import BackendError
    from tests.unit.test_cds_resume_download import _FakeResults

    rid, _legacy, inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True, truncate=True)
    gate = threading.Event()
    real_get_remote = inner.get_remote.side_effect

    def _stalled_remote(request_id: str) -> Any:
        rem = real_get_remote(request_id)

        def _slow_results() -> Any:
            gate.wait(timeout=30)
            return _FakeResults()

        rem.get_results.side_effect = _slow_results
        return rem

    inner.get_remote.side_effect = _stalled_remote
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found)
        backend.set_fetch_wait(0.2)
        t0 = time.monotonic()
        with pytest.raises(BackendError) as ei:
            await asyncio.wait_for(backend.fetch_result(rid, Path("ignored")), timeout=5)
        elapsed = time.monotonic() - t0
    finally:
        gate.set()
        await persistence.close()
    assert ei.value.error_record.error_subclass == "result_not_ready"
    assert elapsed < 2.0


@pytest.mark.asyncio
async def test_a_full_size_foreign_part_without_the_complete_flag_is_downloaded_again(tmp_path: Path, monkeypatch) -> None:
    """R8 HIGH-1: a stalled writer's full-length part, copied before that
    writer failed, says nothing about its bytes."""
    from copernicus_mcp.backends.cds import backend as backend_mod

    rid, _legacy, _inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True, truncate=True)
    fake = _FakeMultiurl()
    monkeypatch.setattr(backend_mod, "multiurl", fake)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        _foreign_part(found, rid, data=b"x" * len(_PAYLOAD), age_seconds=600)
        backend = _backend_for(found)
        backend.set_fetch_wait(10.0)
        out = await backend.fetch_result(rid, Path("ignored"))
    finally:
        await persistence.close()
    assert fake.calls == [0], "downloaded from zero"
    assert Path(out["result"]["filepath"]).read_bytes() == _PAYLOAD


@pytest.mark.asyncio
async def test_a_verified_complete_foreign_part_is_promoted_without_downloading(tmp_path: Path, monkeypatch) -> None:
    from copernicus_mcp.backends.cds import backend as backend_mod

    rid, _legacy, _inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True, truncate=True)
    fake = _FakeMultiurl()
    monkeypatch.setattr(backend_mod, "multiurl", fake)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        part = _foreign_part(found, rid, data=_PAYLOAD, age_seconds=600)
        backend_mod._mark_part_complete(part)
        backend = _backend_for(found)
        backend.set_fetch_wait(10.0)
        out = await backend.fetch_result(rid, Path("ignored"))
    finally:
        await persistence.close()
    assert fake.calls == []
    assert Path(out["result"]["filepath"]).read_bytes() == _PAYLOAD


def test_result_extension_reads_every_published_name_shape() -> None:
    from copernicus_mcp.backends.cds.backend import _request_file_tag, _result_extension

    rid = "4f1c-aa77-request-id"
    tag = _request_file_tag(rid)
    assert _result_extension("cds_abc123.nc", "abc123", rid) == ".nc"
    assert _result_extension(f"cds_abc123.{tag}.nc", "abc123", rid) == ".nc"
    assert _result_extension(f"cds_abc123.{tag}.r0a1b2c3d.zip", "abc123", rid) == ".zip"
    assert _result_extension(f"cds_abc123.{tag}.tar.gz", "abc123", rid) == ".tar.gz"


class _HalfThenGate(_FakeMultiurl):
    """Writes half the result, then holds the transfer open until ``gate``."""

    def __init__(self, gate: threading.Event) -> None:
        super().__init__()
        self.gate = gate

    def download(self, url, *, target, **kwargs) -> None:
        import os

        existing = os.path.getsize(target) if os.path.exists(target) else 0
        self.calls.append(existing)
        with open(target, "ab") as fh:
            fh.write(_PAYLOAD[existing : len(_PAYLOAD) // 2])
            fh.flush()
            self.gate.wait(timeout=30)
            fh.write(_PAYLOAD[max(existing, len(_PAYLOAD) // 2) :])


async def _poll_until(backend: Any, rid: str, status: str, *, tries: int = 100) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for _ in range(tries):
        out = await backend.check_status(rid)
        if out["status"] == status:
            return out
        await asyncio.sleep(0.05)
    return out


@pytest.mark.asyncio
async def test_check_status_reads_downloading_while_a_long_refetch_runs(tmp_path: Path, monkeypatch) -> None:
    """Decision 13: no usable file and a transfer longer than the grace →
    ``running`` / ``downloading`` (never ``failed``), then ``successful``."""
    from copernicus_mcp.backends.cds import backend as backend_mod
    from tests.unit.test_cds_finalise_atomic import _db_state

    rid, _legacy, _inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True, truncate=True)
    gate = threading.Event()
    fake = _HalfThenGate(gate)
    monkeypatch.setattr(backend_mod, "multiurl", fake)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found, cds_download_inline_grace_seconds=0.1)
        first = await backend.check_status(rid)
        second = await backend.check_status(rid)  # joins the same repair
        stored = _db_state(tmp_path / "state.db", rid)[0]
        gate.set()
        final = await _poll_until(backend, rid, "successful")
        await backend.close()
    finally:
        gate.set()
        await persistence.close()
    assert (first["status"], first.get("phase")) == ("running", "downloading")
    assert (second["status"], second.get("phase")) == ("running", "downloading")
    assert stored == "successful", "the read never changes the stored row"
    assert final["status"] == "successful"
    assert Path(final["result"]["filepath"]).read_bytes() == _PAYLOAD
    assert fake.calls == [0], "one transfer for all polls"


@pytest.mark.asyncio
async def test_a_second_process_polling_meanwhile_does_not_download_twice(tmp_path: Path, monkeypatch) -> None:
    from copernicus_mcp.backends.cds import backend as backend_mod

    rid, _legacy, _inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True, truncate=True)
    gate = threading.Event()
    fake = _HalfThenGate(gate)
    monkeypatch.setattr(backend_mod, "multiurl", fake)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        a = _backend_for(found, cds_download_inline_grace_seconds=0.1)
        b = _backend_for(found, cds_download_inline_grace_seconds=0.1)
        assert a._part_owner != b._part_owner
        first = await a.check_status(rid)
        other = await b.check_status(rid)
        gate.set()
        final = await _poll_until(a, rid, "successful")
        await a.close()
        await b.close()
    finally:
        gate.set()
        await persistence.close()
    assert first.get("phase") == "downloading"
    assert (other["status"], other.get("phase")) == ("running", "downloading")
    assert final["status"] == "successful"
    assert fake.calls == [0], "the second process backs off beside the live writer"


@pytest.mark.asyncio
async def test_no_download_reads_download_pending_for_a_lost_entry(tmp_path: Path, monkeypatch) -> None:
    from tests.unit.test_cds_refetch import _downloads

    rid, _own, inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=False)
    inner.download_results.reset_mock()
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found)
        backend.set_download_spawning(False)
        out = await backend.check_status(rid)
    finally:
        await persistence.close()
    assert (out["status"], out["phase"]) == ("running", "download_pending")
    assert backend._downloads == {}
    assert _downloads(inner) == 0


class _NotFound(Exception):
    def __init__(self) -> None:
        super().__init__("404 Client Error: Not Found")
        self.response = type("R", (), {"status_code": 404})()


def _results_raise(inner: Any, exc_factory: Any, *, delay: float = 0.0) -> None:
    import time

    real_get_remote = inner.get_remote.side_effect

    def _remote(request_id: str) -> Any:
        rem = real_get_remote(request_id)

        def _results() -> Any:
            time.sleep(delay)
            raise exc_factory()

        rem.get_results.side_effect = _results
        return rem

    inner.get_remote.side_effect = _remote


@pytest.mark.asyncio
async def test_a_purged_result_reads_failed_inside_the_grace(tmp_path: Path, monkeypatch) -> None:
    """The one honest ``failed``: CDS no longer holds the result."""
    from tests.unit.test_cds_finalise_atomic import _db_state

    rid, _legacy, inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True, truncate=True)
    _results_raise(inner, _NotFound)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        out = await _backend_for(found, cds_download_inline_grace_seconds=2.0).check_status(rid)
    finally:
        await persistence.close()
    assert out["status"] == "failed"
    assert out["error_details"]["error_subclass"] == "remote_result_expired"
    assert out["error_details"]["context"]["request_id"] == rid
    assert _db_state(tmp_path / "state.db", rid)[0] == "successful"


@pytest.mark.asyncio
async def test_a_purged_result_found_after_the_grace_is_reported_by_the_next_poll(tmp_path: Path, monkeypatch) -> None:
    """R8 MEDIUM-4: the outcome must not depend on whether it beat the grace."""
    rid, _legacy, inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True, truncate=True)
    _results_raise(inner, _NotFound, delay=0.3)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found, cds_download_inline_grace_seconds=0.05)
        first = await backend.check_status(rid)
        await asyncio.sleep(0.6)
        second = await backend.check_status(rid)
    finally:
        await persistence.close()
    assert first.get("phase") == "downloading"
    assert second["status"] == "failed"
    assert second["error_details"]["error_subclass"] == "remote_result_expired"


@pytest.mark.asyncio
async def test_another_repair_error_is_raised_like_a_poll_error(tmp_path: Path, monkeypatch) -> None:
    from copernicus_mcp.errors import BackendError

    rid, _legacy, inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True, truncate=True)
    _results_raise(inner, lambda: RuntimeError("service unavailable"))
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        with pytest.raises(BackendError) as ei:
            await _backend_for(found, cds_download_inline_grace_seconds=2.0).check_status(rid)
    finally:
        await persistence.close()
    assert ei.value.error_record.error_subclass == "sdk_refetch_failure"


def test_adoption_never_carries_a_later_generations_complete_flag(tmp_path: Path, monkeypatch) -> None:
    """R9 codex HIGH-1: the adopter copied a stalled writer's unverified
    bytes; meanwhile that writer failed, a retry rewrote the same path and
    flagged it complete. The flag of the NEW bytes must not vouch for the
    copied old ones."""
    import json

    from copernicus_mcp.backends.cds import backend as backend_mod

    foreign = tmp_path / "rid.aaaaaaaaaaaa.part"
    own = tmp_path / "rid.bbbbbbbbbbbb.part"
    record = {"expected_size": len(_PAYLOAD), "location": "https://x/result.nc"}
    foreign.write_bytes(b"x" * len(_PAYLOAD))
    Path(f"{foreign}.meta.json").write_text(json.dumps(record))
    real_copy = backend_mod.shutil.copyfile

    def copy_then_regenerate(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
        out = real_copy(src, dst, *args, **kwargs)
        if str(src) == str(foreign):
            foreign.unlink()
            foreign.write_bytes(_PAYLOAD)
            Path(f"{foreign}.meta.json").write_text(json.dumps(record))
            backend_mod._mark_part_complete(foreign)
        return out

    monkeypatch.setattr(backend_mod.shutil, "copyfile", copy_then_regenerate)
    assert backend_mod._adopt_foreign_part(foreign, own) is True
    assert own.read_bytes() == b"x" * len(_PAYLOAD)
    assert not backend_mod._part_verified(own)


def test_adopting_an_unchanged_verified_part_keeps_it_verified(tmp_path: Path) -> None:
    import json

    from copernicus_mcp.backends.cds import backend as backend_mod

    foreign = tmp_path / "rid.aaaaaaaaaaaa.part"
    own = tmp_path / "rid.bbbbbbbbbbbb.part"
    foreign.write_bytes(_PAYLOAD)
    Path(f"{foreign}.meta.json").write_text(
        json.dumps({"expected_size": len(_PAYLOAD), "location": "https://x/result.nc"})
    )
    backend_mod._mark_part_complete(foreign)
    assert backend_mod._adopt_foreign_part(foreign, own) is True
    assert backend_mod._part_verified(own)
    assert not backend_mod._part_verified(tmp_path / "missing.part")


@pytest.mark.asyncio
async def test_a_losing_repair_never_removes_the_winners_registered_file(tmp_path: Path, monkeypatch) -> None:
    """R9 codex HIGH-2: repair A registered Q; repair B, with a stale view,
    adopts Q. B must leave Q (and A's entry) intact when it loses."""
    from copernicus_mcp.backends.cds.backend import _cache_storage_key
    from tests.unit.test_cds_finalise_atomic import _db_state

    rid, own, _inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=False)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found)
        row = await persistence.fetch_workflow(rid)
        assert row is not None
        won = await found.cache.register_repaired(
            request_id=rid,
            cache_key=_cache_storage_key(row["cache_key"]),
            file_path=own,
            content_type="application/x-netcdf",
        )
        assert won, "repair A registered the file"
        loop = asyncio.get_running_loop()
        out = await backend._repair_or_refetch_locked(row, deadline=loop.time() + 10)
    finally:
        await persistence.close()
    assert out == own
    assert own.exists(), "the loser removed the winner's registered file"
    assert _db_state(tmp_path / "state.db", rid)[1] == [str(own)]


def test_sweep_removes_abandoned_adoption_copies(tmp_path: Path) -> None:
    """Fable R1 M2: an adoption copy killed with its process left a hidden
    ``.<part>.<hex>.tmp`` the sweep never matched."""
    import os
    import time

    from copernicus_mcp.backends.cds import backend as backend_mod

    leftover = tmp_path / ".rid.aaaaaaaaaaaa.part.0123456789abcdef.tmp"
    leftover.write_bytes(b"prefix")
    old = time.time() - backend_mod._RESUME_PART_STALE_SECONDS - 60
    os.utime(leftover, (old, old))
    backend_mod._sweep_stale_parts(tmp_path)
    assert not leftover.exists()


def _run_wait(monkeypatch, polls: list[dict[str, Any]], fetches: list[dict[str, Any]], *args: str) -> tuple[Any, list[str]]:
    from typer.testing import CliRunner

    from copernicus_mcp import cli

    calls: list[str] = []

    class _Orch:
        async def run(self, *, backend, operation, params):  # type: ignore[no-untyped-def]
            calls.append(operation)
            queue = polls if operation == "poll" else fetches
            return queue.pop(0) if len(queue) > 1 else queue[0]

    class _Builder:
        def __call__(self, **kwargs):
            return self

        async def __aenter__(self):
            return _Orch()

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _Builder())
    res = CliRunner().invoke(cli.app, ["cds", "wait", "rid", "--json", "--interval", "0", *args])
    return res, calls


_INCOMPLETE_PARENT = {"result": {"status": "successful", "request_id": "rid", "error_details": None,
                                 "result": {"files": [], "complete": False, "evicted_chunk_indices": [3]}}}
_PENDING = {"result": {"status": "running", "phase": "download_pending", "request_id": "rid", "error_details": None}}
_DONE = {"result": {"status": "successful", "request_id": "rid", "submitted_at": "t", "updated_at": "t",
                    "cache_key": "k", "error_details": None, "result": {"filepath": "/x.nc"}}}
_FETCHED = {"result": {"status": "successful", "request_id": "rid", "cache_hit": True, "result": {"filepath": "/x.nc"}}}


def test_cds_wait_never_exits_0_on_a_parent_still_incomplete_after_its_fetch(monkeypatch) -> None:
    """R9 codex MEDIUM: a part left the cache again between the fetch and the
    re-poll; ``wait`` must not print an incomplete set as its success."""
    res, calls = _run_wait(monkeypatch, [_INCOMPLETE_PARENT], [_FETCHED])
    assert calls == ["poll", "fetch", "poll"]
    assert res.exit_code == 1


def test_cds_wait_keeps_polling_while_another_process_writes_the_file(monkeypatch) -> None:
    """Fable R1 M3: ``result_not_ready`` from the one fetch (a live writer in
    another process) is not a failure; ``wait`` polls on toward its deadline."""
    not_ready = {"error": {"error_class": "BackendError", "error_subclass": "result_not_ready",
                           "message": "still running", "recovery_action": "retry_automatic"}}
    res, calls = _run_wait(monkeypatch, [_PENDING, _PENDING, _DONE], [not_ready, _FETCHED])
    assert res.exit_code == 0, res.stdout + str(res.exception)
    assert calls == ["poll", "fetch", "poll", "fetch", "poll"]


def _refetch_link(original: Path, zone: Path, rid: str) -> Path:
    """Another repair's in-flight fresh name: a hard link ``.r<rand>`` of the
    file it adopted, not registered yet."""
    import os

    from copernicus_mcp.backends.cds.backend import _request_file_tag

    stem = original.name.split(".")[0]  # cds_<hash>
    link = zone / f"{stem}.{_request_file_tag(rid)}.r0123abcd{original.suffix}"
    os.link(original, link)
    return link


@pytest.mark.asyncio
async def test_repair_prefers_the_original_publication_over_a_refetch_name(tmp_path: Path, monkeypatch) -> None:
    """R10 codex M1 / Fable M2: adopting another repair's unregistered
    ``.r<rand>`` name made the winner retire THAT name, leaking the original
    (and its provenance sidecar) outside cache accounting."""
    rid, own, _inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=False)
    sidecar = own.with_suffix(own.suffix + ".provenance.json")
    sidecar.write_text("{}")
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        other = _refetch_link(own, own.parent, rid)
        out = await _backend_for(found).fetch_result(rid, Path("ignored"))
    finally:
        await persistence.close()
    path = Path(out["result"]["filepath"])
    assert path not in (own, other)
    assert not own.exists(), "the adopted original is retired"
    assert path.with_suffix(path.suffix + ".provenance.json").exists()


@pytest.mark.asyncio
async def test_repair_prefers_the_legacy_file_over_a_refetch_name(tmp_path: Path, monkeypatch) -> None:
    rid, legacy, _inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True)
    sidecar = legacy.with_suffix(legacy.suffix + ".provenance.json")
    sidecar.write_text("{}")
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        other = _refetch_link(legacy, legacy.parent, rid)
        out = await _backend_for(found).fetch_result(rid, Path("ignored"))
    finally:
        await persistence.close()
    path = Path(out["result"]["filepath"])
    assert path not in (legacy, other)
    assert not legacy.exists()
    assert path.with_suffix(path.suffix + ".provenance.json").exists()


@pytest.mark.asyncio
async def test_register_repaired_refuses_a_vanished_file(tmp_path: Path, monkeypatch) -> None:
    rid, own, _inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=False)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        ok = await found.cache.register_repaired(
            request_id=rid, cache_key="k", file_path=own.with_name("gone.nc"), content_type="application/x-netcdf"
        )
    finally:
        await persistence.close()
    assert ok is False


@pytest.mark.asyncio
async def test_a_repair_whose_source_vanished_returns_the_winners_entry(tmp_path: Path, monkeypatch) -> None:
    """R10 Fable M2: the winner retired the name this repair was adopting;
    the entry is already there — no re-download."""
    import shutil

    from copernicus_mcp.backends.cds.backend import _cache_storage_key
    from tests.unit.test_cds_refetch import _downloads

    rid, own, inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=False)
    inner.download_results.reset_mock()
    winner = own.with_name(own.name.replace(".nc", ".r0f0f0f0f.nc"))
    shutil.copyfile(own, winner)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found)
        row = await persistence.fetch_workflow(rid)
        assert row is not None
        assert await found.cache.register_repaired(
            request_id=rid, cache_key=_cache_storage_key(row["cache_key"]), file_path=winner,
            content_type="application/x-netcdf",
        )

        def vanished(*_a: Any, **_k: Any) -> Path:
            raise FileNotFoundError("retired by the winner")

        monkeypatch.setattr(found.cache, "publish_request_file", vanished)
        loop = asyncio.get_running_loop()
        out = await backend._repair_or_refetch_locked(row, deadline=loop.time() + 10)
    finally:
        await persistence.close()
    assert out == winner
    assert _downloads(inner) == 0


def test_cds_wait_never_sleeps_past_its_deadline_after_result_not_ready(monkeypatch) -> None:
    """R10 codex M2: the keep-polling branch slept a whole interval, then
    polled and exited 0 well after --timeout."""
    import time

    not_ready = {"error": {"error_class": "BackendError", "error_subclass": "result_not_ready",
                           "message": "still running", "recovery_action": "retry_automatic"}}
    t0 = time.monotonic()
    res, calls = _run_wait(monkeypatch, [_PENDING, _DONE], [not_ready], "--timeout", "0.05", "--interval", "0.3")
    elapsed = time.monotonic() - t0
    assert res.exit_code == 1
    assert calls == ["poll", "fetch"]
    assert elapsed < 0.25


def test_republication_copy_never_holds_a_one_shot_exit(tmp_path: Path, monkeypatch) -> None:
    """R10 codex M3: on a mount without hard links the adoption copy ran in
    the default executor, which ``asyncio.run`` joins at shutdown."""
    import errno
    import os
    import shutil
    import time

    from copernicus_mcp.backends.cds import backend as backend_mod
    from copernicus_mcp.cache import manager as manager_mod

    async def _prepare() -> str:
        rid, _own, _inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=False)
        return rid

    rid = asyncio.run(_prepare())
    real_link = os.link
    real_copy = shutil.copyfile

    def no_link(src: Any, dst: Any, *a: Any, **k: Any) -> None:
        if ".r" in str(dst):
            raise OSError(errno.EPERM, "hard links refused")
        real_link(src, dst, *a, **k)

    def slow_copy(src: Any, dst: Any, *a: Any, **k: Any) -> Any:
        time.sleep(0.8)
        return real_copy(src, dst, *a, **k)

    monkeypatch.setattr(manager_mod.os, "link", no_link)
    monkeypatch.setattr(manager_mod.shutil, "copyfile", slow_copy)

    async def _one_shot_poll() -> None:
        found, persistence = _make_foundation(tmp_path)
        await persistence.initialise()
        try:
            backend = _backend_for(found, cds_download_inline_grace_seconds=0.05)
            await backend.check_status(rid)
            await backend.close()
        finally:
            await persistence.close()

    t0 = time.monotonic()
    asyncio.run(_one_shot_poll())
    elapsed = time.monotonic() - t0
    time.sleep(1.0)  # let the daemon copy finish before tmp_path goes away
    assert elapsed < 0.6, f"asyncio.run waited {elapsed:.2f}s for the copy"
    _ = backend_mod


@pytest.mark.asyncio
async def test_a_stale_repair_failure_is_forgotten_once_the_file_is_back(tmp_path: Path, monkeypatch) -> None:
    """R11 Fable M1: a repair failed after the grace, then another process
    registered the file. A later LRU eviction must read ``download_pending``,
    not the old repair error."""
    import shutil
    import sqlite3

    from copernicus_mcp.backends.cds.backend import _cache_storage_key

    rid, legacy, inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True, truncate=True)
    _results_raise(inner, lambda: RuntimeError("service unavailable"), delay=0.3)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found, cds_download_inline_grace_seconds=0.05)
        first = await backend.check_status(rid)
        await asyncio.sleep(0.6)  # the repair failed in the background
        row = await persistence.fetch_workflow(rid)
        assert row is not None
        key = _cache_storage_key(row["cache_key"])
        winner = legacy.with_name("elsewhere.nc")
        shutil.copyfile(Path(__file__), winner)
        assert await found.cache.register_repaired(
            request_id=rid, cache_key=key, file_path=winner, content_type="application/x-netcdf"
        )
        back = await backend.check_status(rid)
        winner.unlink()
        con = sqlite3.connect(str(tmp_path / "state.db"))
        try:
            con.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            con.execute(
                "INSERT INTO cache_evictions (namespace, key, file_path, reason, evicted_at) "
                "VALUES ('file', ?, ?, 'lru', '2026-10-20T00:00:00Z')",
                (key, str(winner)),
            )
            con.commit()
        finally:
            con.close()
        evicted = await backend.check_status(rid)
    finally:
        await persistence.close()
    assert first.get("phase") == "downloading"
    assert back["status"] == "successful"
    assert (evicted["status"], evicted.get("phase")) == ("running", "download_pending")


@pytest.mark.asyncio
async def test_a_fresh_name_retired_by_a_competing_repair_yields_the_winners_entry(tmp_path: Path, monkeypatch) -> None:
    """R11 Fable M2: our just-published ``.r<rand>`` copy was adopted and
    retired by a competing repair before our size check; no raw
    ``FileNotFoundError`` — the winner's entry is the answer."""
    import shutil

    from copernicus_mcp.backends.cds.backend import _cache_storage_key

    rid, legacy, _inner = await _successful_then_broken(tmp_path, monkeypatch, legacy_name=True, truncate=True)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found)
        row = await persistence.fetch_workflow(rid)
        assert row is not None
        winner = legacy.with_name("elsewhere.nc")
        shutil.copyfile(Path(__file__), winner)
        assert await found.cache.register_repaired(
            request_id=rid, cache_key=_cache_storage_key(row["cache_key"]), file_path=winner,
            content_type="application/x-netcdf",
        )
        real_publish = found.cache.publish_request_file

        def publish_then_retired(source: Path, final: Path, **kw: Any) -> Path:
            out = real_publish(source, final, **kw)
            final.unlink()
            return out

        monkeypatch.setattr(found.cache, "publish_request_file", publish_then_retired)
        loop = asyncio.get_running_loop()
        out = await backend._repair_or_refetch_locked(row, deadline=loop.time() + 10)
    finally:
        await persistence.close()
    assert out == winner


def test_cds_wait_never_sleeps_past_its_deadline_while_downloading(monkeypatch) -> None:
    """R11 codex M: a background repair reads ``downloading``; the ordinary
    poll sleep must stop at the deadline too."""
    import time

    downloading = {"result": {"status": "running", "phase": "downloading", "request_id": "rid", "error_details": None}}
    t0 = time.monotonic()
    res, calls = _run_wait(monkeypatch, [downloading, _DONE], [_FETCHED], "--timeout", "0.05", "--interval", "2")
    elapsed = time.monotonic() - t0
    assert res.exit_code == 1
    assert calls == ["poll"]
    assert elapsed < 1.0
