"""Resumable result transfers (T-CDS-DL-001).

Field report 06-25: an 800 MB E-OBS result sat at ``phase=downloading`` for four
hours under an ephemeral poller — every poll re-spawned the transfer into a
fresh ``.staging/<uuid>`` from byte zero and abandoned it on exit, so a file
larger than one grace-window of bytes could NEVER land. The SDK already
enables multiurl's Range resume (spike T-CDS-DL-000); what broke it was the
per-attempt uuid staging plus the SDK's own delete-before-download. Staging
is now a STABLE part keyed by the cache hash: an interrupted transfer keeps
its bytes, and the next poll's re-spawned finalise appends from where it
stopped. Gated by ``budget.cds_resume_downloads`` (default true).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import time
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

    config = ConfigLoader().load(
        cli_overrides={
            "storage": {
                "state_database": str(tmp_path / "state.db"),
                "cache_directory": str(tmp_path / "cache"),
            }
        }
    )
    persistence = SqliteBackend(config.storage.state_database)
    cache = CacheManager(
        cache_directory=config.storage.cache_directory,
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


def _with_budget(foundation, **updates):
    budget = foundation.config.budget.model_copy(update=updates)
    config = foundation.config.model_copy(update={"budget": budget})
    return dataclasses.replace(foundation, config=config)


def _backend(foundation, **budget):
    from copernicus_mcp.backends.cds.backend import CdsBackend

    found = _with_budget(foundation, **budget) if budget else foundation
    return CdsBackend(foundation=found, credentials=_fake_creds())


def _patch_costing_flat(monkeypatch, *, units: float = 1.0, limit: float = 400.0):
    from copernicus_mcp.backends.cds.costing import CostingResult

    async def _fake(dataset_id, inputs, **_kwargs):
        return CostingResult(units=units, limit=limit)

    monkeypatch.setattr("copernicus_mcp.backends.cds.backend.fetch_costing", _fake)


_PAYLOAD = b"NETCDF-payload-" * 64  # 960 bytes
_HALF = len(_PAYLOAD) // 2


class _FakeResults:
    location = "https://objectstore.example/results/result.nc"
    content_length = len(_PAYLOAD)
    content_type = "application/netcdf"


def _patch_cdsapi_resumable(monkeypatch, status_by_request):
    """cdsapi fake whose Remote serves a Results handle; the transfer itself
    goes through the (also faked) multiurl seam."""
    import sys
    import types

    fake_module = types.ModuleType("cdsapi")
    instance = MagicMock()
    counter = {"n": 0}

    def _retrieve(name, request, target):
        counter["n"] += 1
        remote = MagicMock()
        remote.request_id = f"child-{counter['n']}"
        return remote

    instance.retrieve = MagicMock(side_effect=_retrieve)
    inner = MagicMock()

    def _get_remote(request_id):
        rem = MagicMock()
        rem.json = {
            "status": status_by_request.get(request_id, "running"),
            "jobID": request_id,
        }
        rem.get_results = MagicMock(return_value=_FakeResults())
        return rem

    inner.get_remote = MagicMock(side_effect=_get_remote)
    inner.delete = MagicMock(return_value={"deleted": True})

    def _download_results(request_id, target):
        Path(target).write_bytes(_PAYLOAD)
        return target

    inner.download_results = MagicMock(side_effect=_download_results)
    instance.client = inner
    fake_class = MagicMock(return_value=instance)
    fake_module.Client = fake_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cdsapi", fake_module)
    return inner


class _FakeMultiurl:
    """Simulates multiurl's Range resume: appends from the target's current
    size. ``block`` (an asyncio-unaware threading Event) freezes the transfer
    mid-file to model a client dying mid-download."""

    def __init__(self, block=None) -> None:
        self.block = block
        self.calls: list[int] = []

    def download(self, url, *, target, **kwargs) -> None:
        assert kwargs.get("resume_transfers") is True
        assert kwargs.get("stream") is True
        existing = os.path.getsize(target) if os.path.exists(target) else 0
        self.calls.append(existing)
        if self.block is not None and existing == 0:
            with open(target, "ab") as fh:
                fh.write(_PAYLOAD[:_HALF])
            self.block.wait(timeout=30)
            raise OSError("connection torn down mid-transfer")
        with open(target, "ab") as fh:
            fh.write(_PAYLOAD[existing:])


def _era5(**options: Any) -> dict[str, Any]:
    return {
        "dataset_id": "reanalysis-era5-single-levels",
        "inputs": {
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature"],
            "year": ["2024"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00"],
            "data_format": "netcdf",
        },
        "__options": {"confirmed": True, **options},
    }


def _staging(foundation) -> Path:
    # Day-INDEPENDENT root (local review HIGH): parts must survive a
    # UTC-midnight boundary, so they do NOT live under the day-partitioned
    # cache zone.
    return foundation.cache.staging_root_for("cds")


def _parts_in(foundation) -> list[Path]:
    return sorted(_staging(foundation).glob("*.part"))


@pytest.mark.asyncio
async def test_lost_download_task_keeps_the_part_and_the_next_poll_resumes(
    foundation, monkeypatch
) -> None:
    """The field scenario end-to-end: the transfer's task dies mid-file (client
    gone), the partial bytes SURVIVE, and the next poll's re-spawned finalise
    appends the remainder instead of restarting from zero."""
    import threading

    from copernicus_mcp.backends.cds import backend as backend_mod

    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status)
    gate = threading.Event()
    first = _FakeMultiurl(block=gate)
    monkeypatch.setattr(backend_mod, "multiurl", first)
    backend = _backend(foundation, cds_download_inline_grace_seconds=0.0)

    out = await backend.submit(_era5())
    rid = out["request_id"]
    status[rid] = "successful"

    st = await backend.check_status(rid)
    assert st["phase"] == "downloading"
    # Wait until the transfer has demonstrably written its first half…
    for _ in range(500):
        parts = _parts_in(foundation)
        if parts and parts[0].stat().st_size >= _HALF:
            break
        await asyncio.sleep(0.01)
    # …then the client dies: the in-flight download task is torn down.
    task = backend._downloads[rid]
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    gate.set()

    parts = _parts_in(foundation)
    assert len(parts) == 1, "the partial transfer must survive the lost task"
    assert parts[0].stat().st_size == _HALF
    # Review round 1 (codex HIGH-1): the part is keyed by REQUEST id, not by
    # cache key — two rows sharing a cache_key (force_refresh) are different
    # remote jobs with different result objects and must never share a part.
    assert parts[0].name == f"{rid}.part"
    row = await foundation.persistence.fetch_workflow(rid)
    # Not terminal — nothing is wrong yet; the request is still resumable.
    assert row["status"] in ("queued", "running")

    # Next poll: fresh finalise resumes FROM the part, not from zero.
    second = _FakeMultiurl()
    monkeypatch.setattr(backend_mod, "multiurl", second)
    final = None
    for _ in range(6):
        final = await backend.check_status(rid)
        if final["status"] == "successful":
            break
        await asyncio.sleep(0.05)
    assert final is not None and final["status"] == "successful"
    assert second.calls == [_HALF]  # resumed exactly at the interruption point
    assert Path(final["result"]["filepath"]).read_bytes() == _PAYLOAD
    assert not _parts_in(foundation)  # part promoted, nothing left behind


@pytest.mark.asyncio
async def test_size_change_invalidates_the_stale_part(
    foundation, monkeypatch
) -> None:
    """A part from an older, differently-sized result must not be appended
    to — the meta sidecar pins the expected size and a mismatch restarts."""
    from copernicus_mcp.backends.cds import backend as backend_mod

    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status)
    fake = _FakeMultiurl()
    monkeypatch.setattr(backend_mod, "multiurl", fake)
    backend = _backend(foundation)

    out = await backend.submit(_era5())
    rid = out["request_id"]
    status[rid] = "successful"

    # Plant a stale part + meta claiming a DIFFERENT expected size.
    staging = _staging(foundation)
    staging.mkdir(parents=True, exist_ok=True)
    part = staging / f"{rid}.part"
    part.write_bytes(b"old-bytes")
    (staging / f"{rid}.part.meta.json").write_text(
        json.dumps({"expected_size": 123, "location": _FakeResults.location})
    )

    final = None
    for _ in range(4):
        final = await backend.check_status(rid)
        if final["status"] == "successful":
            break
    assert final is not None and final["status"] == "successful"
    assert fake.calls == [0]  # restarted clean, never appended to old bytes
    assert Path(final["result"]["filepath"]).read_bytes() == _PAYLOAD


@pytest.mark.asyncio
async def test_changed_result_location_resets_the_part(
    foundation, monkeypatch
) -> None:
    """Review round 1 (codex HIGH-2): equal size is NOT identity. A part whose
    meta records a different result URL belongs to a different object — a
    force_refresh regeneration can produce a same-sized result whose bytes
    differ; appending to the old prefix would store a corrupt hybrid."""
    from copernicus_mcp.backends.cds import backend as backend_mod

    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status)
    fake = _FakeMultiurl()
    monkeypatch.setattr(backend_mod, "multiurl", fake)
    backend = _backend(foundation)

    out = await backend.submit(_era5())
    rid = out["request_id"]
    status[rid] = "successful"

    staging = _staging(foundation)
    staging.mkdir(parents=True, exist_ok=True)
    part = staging / f"{rid}.part"
    part.write_bytes(_PAYLOAD[:_HALF])  # right size prefix, WRONG object
    (staging / f"{rid}.part.meta.json").write_text(
        json.dumps(
            {
                "expected_size": len(_PAYLOAD),
                "location": "https://objectstore.example/results/OLD.nc",
            }
        )
    )

    final = None
    for _ in range(4):
        final = await backend.check_status(rid)
        if final["status"] == "successful":
            break
    assert final is not None and final["status"] == "successful"
    assert fake.calls == [0]  # restarted clean — never appended to old bytes
    assert Path(final["result"]["filepath"]).read_bytes() == _PAYLOAD


@pytest.mark.asyncio
async def test_live_detached_writer_is_adopted_not_duplicated(
    foundation, monkeypatch
) -> None:
    """Review round 1 (codex HIGH-3): cancelling the download task abandons
    the worker thread, which keeps writing the part. A later poll must ADOPT
    that live transfer — never start a second multiurl writer on the same
    file (two appenders interleave bytes)."""
    import threading

    from copernicus_mcp.backends.cds import backend as backend_mod

    gate = threading.Event()

    class _BlockThenFinish(_FakeMultiurl):
        def download(self, url, *, target, **kwargs) -> None:
            existing = os.path.getsize(target) if os.path.exists(target) else 0
            self.calls.append(existing)
            with open(target, "ab") as fh:
                fh.write(_PAYLOAD[:_HALF])
            gate.wait(timeout=30)
            with open(target, "ab") as fh:
                fh.write(_PAYLOAD[_HALF:])

    fake = _BlockThenFinish()
    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status)
    monkeypatch.setattr(backend_mod, "multiurl", fake)
    backend = _backend(foundation, cds_download_inline_grace_seconds=0.0)

    out = await backend.submit(_era5())
    rid = out["request_id"]
    status[rid] = "successful"

    st = await backend.check_status(rid)
    assert st["phase"] == "downloading"
    for _ in range(500):
        parts = _parts_in(foundation)
        if parts and parts[0].stat().st_size >= _HALF:
            break
        await asyncio.sleep(0.01)
    task = backend._downloads[rid]
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    # Writer thread is STILL alive (blocked on the gate). The next poll must
    # not open a second writer on the part.
    st2 = await backend.check_status(rid)
    assert st2["status"] in ("queued", "running")
    await asyncio.sleep(0.05)
    assert fake.calls == [0], "a second concurrent writer was spawned"

    gate.set()  # the original writer finishes the file
    final = None
    for _ in range(100):
        final = await backend.check_status(rid)
        if final["status"] == "successful":
            break
        await asyncio.sleep(0.02)
    assert final is not None and final["status"] == "successful"
    assert fake.calls == [0]  # exactly one transfer, ever
    assert Path(final["result"]["filepath"]).read_bytes() == _PAYLOAD
    assert not _parts_in(foundation)


@pytest.mark.asyncio
async def test_transfer_runs_on_a_daemon_thread(foundation, monkeypatch) -> None:
    """Review round 2 (codex HIGH): the default executor is joined by
    ``asyncio.run`` at shutdown, so a one-shot CLI poll would BLOCK until the
    whole transfer finished — the ephemeral-poller contract is "exit promptly,
    keep the prefix, resume next poll". The transfer must therefore run on a
    daemon thread (dies with the process; the flushed prefix stays valid for
    Range resume)."""
    import threading

    from copernicus_mcp.backends.cds import backend as backend_mod

    gate = threading.Event()

    class _BlockThenFinish(_FakeMultiurl):
        def download(self, url, *, target, **kwargs) -> None:
            existing = os.path.getsize(target) if os.path.exists(target) else 0
            self.calls.append(existing)
            with open(target, "ab") as fh:
                fh.write(_PAYLOAD[:_HALF])
            gate.wait(timeout=30)
            with open(target, "ab") as fh:
                fh.write(_PAYLOAD[_HALF:])

    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status)
    monkeypatch.setattr(backend_mod, "multiurl", _BlockThenFinish())
    backend = _backend(foundation, cds_download_inline_grace_seconds=0.0)

    out = await backend.submit(_era5())
    rid = out["request_id"]
    status[rid] = "successful"
    await backend.check_status(rid)
    for _ in range(500):
        parts = _parts_in(foundation)
        if parts and parts[0].stat().st_size >= _HALF:
            break
        await asyncio.sleep(0.01)

    transfer_threads = [
        t for t in threading.enumerate() if t.name.startswith("cds-transfer")
    ]
    assert transfer_threads, "transfer must run on a named non-executor thread"
    assert all(t.daemon for t in transfer_threads)

    gate.set()
    for _ in range(100):
        final = await backend.check_status(rid)
        if final["status"] == "successful":
            break
        await asyncio.sleep(0.02)
    assert final["status"] == "successful"


@pytest.mark.asyncio
async def test_cancel_reaps_the_transfer_registry(foundation, monkeypatch) -> None:
    """Review round 2 (codex MEDIUM): a cancelled row is terminal — no later
    finalise ever adopts its transfer, so ``cancel`` must arrange the registry
    entry's removal (and exception retrieval) itself or the dict grows
    forever in a long-running server."""
    import threading

    from copernicus_mcp.backends.cds import backend as backend_mod

    gate = threading.Event()

    class _BlockThenFail(_FakeMultiurl):
        def download(self, url, *, target, **kwargs) -> None:
            existing = os.path.getsize(target) if os.path.exists(target) else 0
            self.calls.append(existing)
            with open(target, "ab") as fh:
                fh.write(_PAYLOAD[:_HALF])
            gate.wait(timeout=30)
            raise OSError("torn down")

    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status)
    monkeypatch.setattr(backend_mod, "multiurl", _BlockThenFail())
    backend = _backend(foundation, cds_download_inline_grace_seconds=0.0)

    out = await backend.submit(_era5())
    rid = out["request_id"]
    status[rid] = "successful"
    await backend.check_status(rid)
    for _ in range(500):
        parts = _parts_in(foundation)
        if parts and parts[0].stat().st_size >= _HALF:
            break
        await asyncio.sleep(0.01)
    assert backend._part_transfers  # the transfer is registered

    cancelled = await backend.cancel(rid)
    assert cancelled["status"] == "cancelled"
    gate.set()  # the writer errors out; the registry entry must be reaped
    for _ in range(300):
        if not backend._part_transfers:
            break
        await asyncio.sleep(0.01)
    assert backend._part_transfers == {}


@pytest.mark.asyncio
async def test_complete_part_is_promoted_without_redownload(
    foundation, monkeypatch
) -> None:
    """Review round 3 (codex MEDIUM): a daemon writer can FINISH the file
    after its awaiter died (cancelled poll, closed loop). Re-downloading a
    complete part trips multiurl's existing<remote assertion and would turn a
    perfectly good result into a failed row — a complete part is promoted
    directly, no transfer at all."""
    from copernicus_mcp.backends.cds import backend as backend_mod

    class _Refuses(_FakeMultiurl):
        def download(self, url, *, target, **kwargs) -> None:
            raise AssertionError("a complete part must never be re-downloaded")

    fake = _Refuses()
    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status)
    monkeypatch.setattr(backend_mod, "multiurl", fake)
    backend = _backend(foundation)

    out = await backend.submit(_era5())
    rid = out["request_id"]
    status[rid] = "successful"

    staging = _staging(foundation)
    staging.mkdir(parents=True, exist_ok=True)
    (staging / f"{rid}.part").write_bytes(_PAYLOAD)  # complete
    (staging / f"{rid}.part.meta.json").write_text(
        json.dumps(
            {"expected_size": len(_PAYLOAD), "location": _FakeResults.location}
        )
    )

    final = None
    for _ in range(4):
        final = await backend.check_status(rid)
        if final["status"] in ("successful", "failed"):
            break
    assert final is not None and final["status"] == "successful"
    assert Path(final["result"]["filepath"]).read_bytes() == _PAYLOAD
    assert not _parts_in(foundation)


@pytest.mark.asyncio
async def test_failed_transfer_disowns_its_complete_looking_part(
    foundation, monkeypatch
) -> None:
    """Review round 4 (codex MEDIUM): a transfer that RAISED may still have
    left a part at the expected size (the library can fail after the last
    byte). When the in-memory future records a failure, the bytes are suspect
    — reset and re-download rather than promote a known-failed transfer."""
    from copernicus_mcp.backends.cds import backend as backend_mod

    fake = _FakeMultiurl()
    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status)
    monkeypatch.setattr(backend_mod, "multiurl", fake)
    backend = _backend(foundation)

    out = await backend.submit(_era5())
    rid = out["request_id"]
    status[rid] = "successful"

    staging = _staging(foundation)
    staging.mkdir(parents=True, exist_ok=True)
    part = staging / f"{rid}.part"
    part.write_bytes(b"?" * len(_PAYLOAD))  # full length, suspect bytes
    (staging / f"{rid}.part.meta.json").write_text(
        json.dumps(
            {"expected_size": len(_PAYLOAD), "location": _FakeResults.location}
        )
    )
    failed = asyncio.get_running_loop().create_future()
    failed.set_exception(OSError("failed after the last byte"))
    backend._part_transfers[str(part)] = failed

    final = None
    for _ in range(4):
        final = await backend.check_status(rid)
        if final["status"] in ("successful", "failed"):
            break
    assert final is not None and final["status"] == "successful"
    assert fake.calls == [0]  # re-downloaded from scratch, no blind promote
    assert Path(final["result"]["filepath"]).read_bytes() == _PAYLOAD


@pytest.mark.asyncio
async def test_concurrent_transfers_are_capped(foundation, monkeypatch) -> None:
    """Review round 3 (codex MEDIUM): daemonising dropped the default
    executor's worker cap — a batch poll over many ready results must not
    open unbounded concurrent sockets/files. At most 4 transfers run at
    once; the rest queue."""
    import threading

    from copernicus_mcp.backends.cds import backend as backend_mod

    class _Gauge(_FakeMultiurl):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.peak = 0
            self.lock = threading.Lock()
            self.release = threading.Event()

        def download(self, url, *, target, **kwargs) -> None:
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                self.release.wait(timeout=30)
                existing = (
                    os.path.getsize(target) if os.path.exists(target) else 0
                )
                with open(target, "ab") as fh:
                    fh.write(_PAYLOAD[existing:])
            finally:
                with self.lock:
                    self.active -= 1

    fake = _Gauge()
    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status)
    monkeypatch.setattr(backend_mod, "multiurl", fake)
    backend = _backend(foundation, cds_download_inline_grace_seconds=0.0)

    rids = []
    for month in ("01", "02", "03", "04", "05", "06"):
        req = _era5()
        req["inputs"]["month"] = [month]  # six distinct requests
        out = await backend.submit(req)
        rids.append(out["request_id"])
        status[out["request_id"]] = "successful"
    for rid in rids:
        await backend.check_status(rid)

    for _ in range(500):  # all six spawned; only four may be transferring
        if fake.active == 4:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)
    assert fake.peak == 4, f"transfer concurrency must be capped (peak={fake.peak})"

    fake.release.set()
    done = 0
    for rid in rids:
        for _ in range(200):
            final = await backend.check_status(rid)
            if final["status"] == "successful":
                done += 1
                break
            await asyncio.sleep(0.02)
    assert done == 6
    assert fake.peak == 4


@pytest.mark.asyncio
async def test_resume_survives_a_utc_midnight_boundary(
    foundation, monkeypatch
) -> None:
    """Local review round (HIGH): the cache zone partitions by calendar day,
    so a part parked under it would become invisible (and unsweepable) after
    UTC midnight — the resume would silently restart from zero. Interrupt a
    transfer 'today', roll the calendar, and the next poll must still append
    from the interruption point."""
    import datetime as _dt
    import threading

    from copernicus_mcp.backends.cds import backend as backend_mod
    from copernicus_mcp.cache import manager as manager_mod

    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status)
    gate = threading.Event()
    first = _FakeMultiurl(block=gate)
    monkeypatch.setattr(backend_mod, "multiurl", first)
    backend = _backend(foundation, cds_download_inline_grace_seconds=0.0)

    out = await backend.submit(_era5())
    rid = out["request_id"]
    status[rid] = "successful"
    await backend.check_status(rid)
    for _ in range(500):
        parts = _parts_in(foundation)
        if parts and parts[0].stat().st_size >= _HALF:
            break
        await asyncio.sleep(0.01)
    task = backend._downloads[rid]
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    gate.set()

    class _Tomorrow(_dt.datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ANN001
            return _dt.datetime.now(tz) + _dt.timedelta(days=1)

    monkeypatch.setattr(manager_mod, "datetime", _Tomorrow)

    second = _FakeMultiurl()
    monkeypatch.setattr(backend_mod, "multiurl", second)
    final = None
    for _ in range(6):
        final = await backend.check_status(rid)
        if final["status"] == "successful":
            break
        await asyncio.sleep(0.05)
    assert final is not None and final["status"] == "successful"
    assert second.calls == [_HALF], "midnight must not reset the resume point"
    assert Path(final["result"]["filepath"]).read_bytes() == _PAYLOAD


def test_sweep_spares_live_transfers(tmp_path) -> None:
    """Review round 2 (codex LOW): the stale sweep must never unlink a part a
    live in-process writer is appending to, however old its mtime."""
    from copernicus_mcp.backends.cds.backend import _sweep_stale_parts

    staging = tmp_path / ".staging"
    staging.mkdir()
    live = staging / "live.part"
    dead = staging / "dead.part"
    for p in (live, dead):
        p.write_bytes(b"x")
        old = time.time() - 9 * 86400
        os.utime(p, (old, old))

    _sweep_stale_parts(staging, keep=frozenset({str(live)}))

    assert live.exists()
    assert not dead.exists()


@pytest.mark.asyncio
async def test_stale_parts_are_swept(foundation, monkeypatch) -> None:
    from copernicus_mcp.backends.cds import backend as backend_mod

    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status)
    monkeypatch.setattr(backend_mod, "multiurl", _FakeMultiurl())
    backend = _backend(foundation)

    staging = _staging(foundation)
    staging.mkdir(parents=True, exist_ok=True)
    ancient = staging / "deadbeef.part"
    ancient.write_bytes(b"forgotten")
    old = time.time() - 9 * 86400
    os.utime(ancient, (old, old))

    out = await backend.submit(_era5())
    status[out["request_id"]] = "successful"
    for _ in range(4):
        st = await backend.check_status(out["request_id"])
        if st["status"] == "successful":
            break

    assert not ancient.exists()


@pytest.mark.asyncio
async def test_resume_disabled_uses_the_legacy_download(
    foundation, monkeypatch
) -> None:
    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    inner = _patch_cdsapi_resumable(monkeypatch, status)
    backend = _backend(foundation, cds_resume_downloads=False)

    out = await backend.submit(_era5())
    status[out["request_id"]] = "successful"
    final = None
    for _ in range(4):
        final = await backend.check_status(out["request_id"])
        if final["status"] == "successful":
            break

    assert final is not None and final["status"] == "successful"
    inner.download_results.assert_called()
    assert not _parts_in(foundation)


@pytest.mark.asyncio
async def test_completed_transfer_with_wrong_size_fails_and_discards(
    foundation, monkeypatch
) -> None:
    """A transfer that ENDS at the wrong byte count is corrupt — fail the row
    and discard the part rather than promote garbage into the cache."""
    from copernicus_mcp.backends.cds import backend as backend_mod

    class _Short(_FakeMultiurl):
        def download(self, url, *, target, **kwargs) -> None:
            with open(target, "ab") as fh:
                fh.write(_PAYLOAD[: _HALF])  # claims done, is not

    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status)
    monkeypatch.setattr(backend_mod, "multiurl", _Short())
    backend = _backend(foundation)

    out = await backend.submit(_era5())
    rid = out["request_id"]
    status[rid] = "successful"
    final = None
    for _ in range(4):
        final = await backend.check_status(rid)
        if final["status"] in ("successful", "failed"):
            break

    assert final is not None and final["status"] == "failed"
    assert not _parts_in(foundation)
