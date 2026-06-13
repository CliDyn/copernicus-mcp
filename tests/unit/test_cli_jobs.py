"""CLI tests for the ``jobs`` subcommand group (T-JOBS-RECOVERY)."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def env(monkeypatch: Any, tmp_path: Path) -> Path:
    monkeypatch.setenv("COPERNICUS_MCP_STORAGE__STATE_DATABASE", str(tmp_path / "state.db"))
    monkeypatch.setenv("COPERNICUS_MCP_STORAGE__CACHE_DIRECTORY", str(tmp_path / "cache"))
    return tmp_path


def _fake_builder(orch: AsyncMock) -> Any:
    @contextlib.asynccontextmanager
    async def _builder() -> Any:
        yield orch

    return _builder


def test_jobs_list_dispatches_with_defaults(runner: CliRunner, env: Path, monkeypatch: Any) -> None:
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.list_jobs.return_value = {
        "results": [{"request_id": "r1", "backend": "cds", "dataset": "d", "status": "running"}],
        "count": 1,
    }
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(cli.app, ["jobs", "list", "--json"])
    assert res.exit_code == 0, res.stdout + res.stderr
    payload = json.loads(res.stdout)
    assert payload["count"] == 1
    assert fake.list_jobs.call_args.kwargs == {
        "status": None,
        "limit": 50,
        "created_after": None,
    }


def test_jobs_list_passes_filters(runner: CliRunner, env: Path, monkeypatch: Any) -> None:
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.list_jobs.return_value = {"results": [], "count": 0}
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(
        cli.app,
        [
            "jobs", "list",
            "--status", "running,queued",
            "--limit", "10",
            "--created-after", "2026-01-01T00:00:00Z",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    kwargs = fake.list_jobs.call_args.kwargs
    assert kwargs["status"] == ["running", "queued"]
    assert kwargs["limit"] == 10
    assert kwargs["created_after"] == "2026-01-01T00:00:00Z"


def test_jobs_list_error_envelope_exits_nonzero(
    runner: CliRunner, env: Path, monkeypatch: Any
) -> None:
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.list_jobs.return_value = {
        "error": {"error_class": "ValidationError", "message": "invalid status filter"}
    }
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(cli.app, ["jobs", "list", "--status", "bogus", "--json"])
    assert res.exit_code != 0
    assert "ValidationError" in res.stdout


def test_jobs_help_lists_list_subcommand(runner: CliRunner) -> None:
    from copernicus_mcp.cli import app

    res = runner.invoke(app, ["jobs", "--help"])
    assert res.exit_code == 0
    assert "list" in res.stdout
