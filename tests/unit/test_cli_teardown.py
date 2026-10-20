"""T-CDS-FINALISE-ATOMIC Task 003: a one-shot CLI tears down its own work
before closing the store, can read status without downloading, and turns
SIGTERM into an orderly exit.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from tests.unit.test_cds_finalise_atomic import _backend_for, _submit
from tests.unit.test_cds_resume_download import (
    _FakeMultiurl,
    _make_foundation,
    _patch_cdsapi_resumable,
    _patch_costing_flat,
)


class _Blocking(_FakeMultiurl):
    def download(self, url, *, target, **kwargs) -> None:
        self.calls.append(0)
        assert self.block is not None
        self.block.wait(timeout=30)
        raise OSError("connection torn down")


@pytest.mark.asyncio
async def test_close_cancels_and_awaits_finalise_tasks_with_the_store_open(tmp_path: Path, monkeypatch) -> None:
    from copernicus_mcp.backends.cds import backend as backend_mod

    status_map: dict[str, str] = {}
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi_resumable(monkeypatch, status_map)
    gate = threading.Event()
    monkeypatch.setattr(backend_mod, "multiurl", _Blocking(block=gate))
    rid = await _submit(tmp_path, status_map)
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        backend = _backend_for(found, cds_download_inline_grace_seconds=0.0)
        out = await backend.check_status(rid)
        assert out.get("phase") == "downloading"
        tasks = list(backend._downloads.values())
        assert tasks
        await backend.close()
        assert all(t.done() for t in tasks)
        # Still usable: close ran BEFORE the store closed.
        row = await persistence.fetch_workflow(rid)
        assert row is not None and row["status"] in ("queued", "running")
        # No new finalise after close.
        await backend.check_status(rid)
        assert not [t for t in backend._downloads.values() if not t.done()]
    finally:
        gate.set()
        await persistence.close()


@pytest.mark.asyncio
async def test_no_download_reads_status_without_spawning(tmp_path: Path, monkeypatch) -> None:
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
        backend = _backend_for(found, cds_download_inline_grace_seconds=5.0)
        backend.set_download_spawning(False)
        out = await backend.check_status(rid)
        normal_keys = set((await backend._downloading_envelope(await persistence.fetch_workflow(rid))).keys())
    finally:
        await persistence.close()
    assert out["status"] == "running"
    assert out["phase"] == "download_pending"
    assert set(out.keys()) == normal_keys, "same JSON shape as a normal check-status"
    assert backend._downloads == {}
    assert fake.calls == []


@pytest.mark.asyncio
async def test_cli_orchestrator_closes_backends_before_closing_the_store(monkeypatch) -> None:
    from copernicus_mcp import cli
    from copernicus_mcp.backends.registry import BackendRegistry

    seen: list[bool] = []

    async def spy_close_all(self):  # type: ignore[no-untyped-def]
        seen.append(True)

    monkeypatch.setattr(BackendRegistry, "close_all", spy_close_all)
    async with cli._build_orchestrator_for_cli() as orch:
        foundation = orch._foundation
    assert seen == [True]
    assert foundation.persistence._conn is None  # closed after close_all


def test_check_status_no_download_flag_disables_spawning(monkeypatch) -> None:
    from typer.testing import CliRunner

    from copernicus_mcp import cli

    captured: dict[str, object] = {}

    class _Builder:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return self

        async def __aenter__(self):
            class _Orch:
                async def run(self, **kw):
                    return {"result": {"status": "running", "phase": "download_pending", "request_id": "id1"}}

            return _Orch()

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _Builder())
    res = CliRunner().invoke(cli.app, ["cds", "check-status", "id1", "--no-download", "--json"])
    assert res.exit_code == 0, res.stdout + str(res.exception)
    assert captured == {"spawn_downloads": False}
    assert json.loads(res.stdout)["phase"] == "download_pending"


def test_inline_grace_env_override(monkeypatch) -> None:
    from copernicus_mcp.config import ConfigLoader

    monkeypatch.setenv("COPERNICUS_MCP_CDS_DOWNLOAD_INLINE_GRACE_SECONDS", "7.5")
    assert ConfigLoader().load().budget.cds_download_inline_grace_seconds == 7.5


def test_sigterm_runs_teardown_and_exits_143(tmp_path: Path) -> None:
    """The one-shot CLI runner turns SIGTERM into cancellation of its main
    task, so ``finally`` blocks (aclose, store close) run, then exits 143."""
    marker = tmp_path / "teardown-ran"
    ready = tmp_path / "ready"
    script = f"""
import asyncio, pathlib, typer
from copernicus_mcp import cli

async def main():
    try:
        pathlib.Path({str(ready)!r}).write_text("1")
        await asyncio.sleep(60)
    finally:
        pathlib.Path({str(marker)!r}).write_text("1")

try:
    cli._run(main())
except typer.Exit as e:
    raise SystemExit(e.exit_code)
"""
    env = dict(os.environ)
    proc = subprocess.Popen([sys.executable, "-c", script], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for _ in range(200):
            if ready.exists():
                break
            if proc.poll() is not None:
                break
            import time

            time.sleep(0.05)
        assert ready.exists(), proc.stderr.read().decode() if proc.stderr else ""
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert rc == 143
    assert marker.exists(), "teardown must run on SIGTERM"
