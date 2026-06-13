"""CLI tests for the ``cds`` subcommand group (T-CDS-007)."""

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
    monkeypatch.setenv(
        "COPERNICUS_MCP_STORAGE__STATE_DATABASE", str(tmp_path / "state.db")
    )
    monkeypatch.setenv(
        "COPERNICUS_MCP_STORAGE__CACHE_DIRECTORY", str(tmp_path / "cache")
    )
    return tmp_path


def _fake_builder(orch: AsyncMock) -> Any:
    @contextlib.asynccontextmanager
    async def _builder() -> Any:
        yield orch

    return _builder


def _good_inputs() -> dict[str, Any]:
    return {
        "variable": ["2m_temperature"],
        "year": ["2024"],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],
    }


def test_cds_apply_constraints_dispatches(
    runner: CliRunner, env: Path, monkeypatch: Any, tmp_path: Path
) -> None:
    from copernicus_mcp import cli

    inputs_file = tmp_path / "inputs.json"
    inputs_file.write_text(json.dumps({"variable": ["x"]}))
    fake = AsyncMock()
    fake.run.return_value = {"result": {"valid_remaining": {"variable": ["x", "y"]}}}
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(
        cli.app,
        ["cds", "apply-constraints", "--dataset-id", "ds-1", "--inputs-file", str(inputs_file), "--json"],
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    kwargs = fake.run.call_args.kwargs
    assert kwargs["operation"] == "apply_constraints"
    assert kwargs["params"]["dataset_id"] == "ds-1"
    assert kwargs["params"]["inputs"] == {"variable": ["x"]}


def test_cds_apply_constraints_empty_inputs_default(
    runner: CliRunner, env: Path, monkeypatch: Any
) -> None:
    """No --inputs-file → empty inputs (the canonical-vocabulary call the
    calibration campaign's P0.1 uses)."""
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.return_value = {"result": {"valid_remaining": {}}}
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(cli.app, ["cds", "apply-constraints", "--dataset-id", "ds-1", "--json"])
    assert res.exit_code == 0, res.stdout + res.stderr
    assert fake.run.call_args.kwargs["params"]["inputs"] == {}


def test_cds_help_lists_subcommands(runner: CliRunner) -> None:
    from copernicus_mcp.cli import app

    res = runner.invoke(app, ["cds", "--help"])
    assert res.exit_code == 0, res.stdout + res.stderr
    for cmd in (
        "search",
        "describe",
        "estimate",
        "submit",
        "check-status",
        "wait",
        "download",
        "cancel",
    ):
        assert cmd in res.stdout


def test_cds_search_dispatches(
    runner: CliRunner, env: Path, monkeypatch: Any
) -> None:
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.return_value = {
        "result": {"datasets": [{"id": "reanalysis-era5-single-levels"}], "total_count": 1}
    }
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(
        cli.app, ["cds", "search", "--keyword", "temp", "--limit", "5", "--json"]
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    parsed = json.loads(res.stdout)
    assert parsed["datasets"][0]["id"] == "reanalysis-era5-single-levels"
    kwargs = fake.run.call_args.kwargs
    assert kwargs["backend"] == "cds"
    assert kwargs["operation"] == "search"
    assert kwargs["params"] == {"keyword": "temp", "limit": 5}


def test_cds_describe_passes_identifier(
    runner: CliRunner, env: Path, monkeypatch: Any
) -> None:
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.return_value = {"result": {"id": "ds-1"}}
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(cli.app, ["cds", "describe", "ds-1", "--json"])
    assert res.exit_code == 0, res.stdout + res.stderr
    kwargs = fake.run.call_args.kwargs
    assert kwargs["operation"] == "describe"
    assert kwargs["params"] == {"identifier": "ds-1"}


def test_cds_estimate_reads_inputs_from_file(
    runner: CliRunner, env: Path, monkeypatch: Any, tmp_path: Path
) -> None:
    from copernicus_mcp import cli

    inputs_file = tmp_path / "inputs.json"
    inputs_file.write_text(json.dumps(_good_inputs()))

    fake = AsyncMock()
    fake.run.return_value = {"result": {"estimated_size_bytes": 12345}}
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(
        cli.app,
        [
            "cds",
            "estimate",
            "--dataset-id",
            "reanalysis-era5-single-levels",
            "--inputs-file",
            str(inputs_file),
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    kwargs = fake.run.call_args.kwargs
    assert kwargs["operation"] == "estimate"
    assert kwargs["params"]["dataset_id"] == "reanalysis-era5-single-levels"
    assert kwargs["params"]["inputs"]["variable"] == ["2m_temperature"]


def test_cds_estimate_rejects_missing_inputs_file(
    runner: CliRunner, env: Path, monkeypatch: Any
) -> None:
    """Without ``--inputs-file`` the CLI exits with EXIT_USER_INPUT (2)
    rather than calling the orchestrator with an empty inputs dict."""
    from copernicus_mcp import cli

    fake = AsyncMock()
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))
    res = runner.invoke(
        cli.app,
        ["cds", "estimate", "--dataset-id", "reanalysis-era5-single-levels"],
    )
    assert res.exit_code == 2, res.stdout + res.stderr
    fake.run.assert_not_awaited()


def test_cds_submit_with_yes_skips_confirmation(
    runner: CliRunner, env: Path, monkeypatch: Any, tmp_path: Path
) -> None:
    """When the orchestrator returns a confirmation envelope and
    ``--yes`` is set, the CLI re-submits with ``options={confirmed:
    true}`` and emits the final payload."""
    from copernicus_mcp import cli

    inputs_file = tmp_path / "inputs.json"
    inputs_file.write_text(json.dumps(_good_inputs()))

    fake = AsyncMock()
    fake.run.side_effect = [
        # First call returns confirmation prompt.
        {
            "confirmation_required": True,
            "estimated_size_bytes": 5_000_000_000,
            "advisory_message": "big request",
        },
        # Second call (after --yes) returns the queued envelope.
        {"result": {"status": "queued", "request_id": "abc-123"}},
    ]
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(
        cli.app,
        [
            "cds",
            "submit",
            "--dataset-id",
            "reanalysis-era5-single-levels",
            "--inputs-file",
            str(inputs_file),
            "--yes",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    parsed = json.loads(res.stdout)
    assert parsed["status"] == "queued"
    assert fake.run.await_count == 2
    second_kwargs = fake.run.call_args_list[1].kwargs
    # A human confirming at the CLI grants BOTH the size/tier confirm and the
    # large-fan-out reconfirm — the two-tier "repeat" is an agent-escalation gate;
    # an interactive human is already the deliberate authority.
    assert second_kwargs["options"] == {
        "confirmed": True,
        "confirm_large_fanout": True,
    }


def test_show_confirmation_surfaces_chunk_count(capsys: Any) -> None:
    """The fan-out reconfirm envelope must show the human the job count before
    they approve — the CLI's single confirm grants both acks, so without this a
    human would authorize a 100+-job batch seeing only a generic panel."""
    from copernicus_mcp.cli import _show_confirmation

    _show_confirmation(
        {
            "confirmation_required": True,
            "reason": "auto_chunk_job_count_large",
            "chunk_count": 150,
            "next_action": "re-submit with confirmed=true AND confirm_large_fanout=true",
        }
    )
    err = capsys.readouterr().err
    assert "150" in err


def test_cds_check_status_passes_request_id(
    runner: CliRunner, env: Path, monkeypatch: Any
) -> None:
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.return_value = {"result": {"status": "running", "request_id": "rid"}}
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))
    res = runner.invoke(
        cli.app, ["cds", "check-status", "rid", "--json"]
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    kwargs = fake.run.call_args.kwargs
    assert kwargs["operation"] == "poll"
    assert kwargs["params"] == {"request_id": "rid"}


def test_cds_download_passes_request_id_and_target(
    runner: CliRunner, env: Path, monkeypatch: Any
) -> None:
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.return_value = {
        "result": {"status": "successful", "filepath": "/tmp/x.bin"}
    }
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))
    res = runner.invoke(
        cli.app,
        ["cds", "download", "rid", "--target", "/tmp/x.bin", "--json"],
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    kwargs = fake.run.call_args.kwargs
    assert kwargs["operation"] == "fetch"
    assert kwargs["params"] == {"request_id": "rid", "target": "/tmp/x.bin"}


def test_cds_cancel_dispatches(
    runner: CliRunner, env: Path, monkeypatch: Any
) -> None:
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.return_value = {
        "result": {"cancelled": True, "request_id": "rid", "status": "cancelled"}
    }
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))
    res = runner.invoke(cli.app, ["cds", "cancel", "rid", "--json"])
    assert res.exit_code == 0, res.stdout + res.stderr
    kwargs = fake.run.call_args.kwargs
    assert kwargs["operation"] == "cancel"


def test_cds_wait_returns_terminal_status(
    runner: CliRunner, env: Path, monkeypatch: Any
) -> None:
    """``cds wait`` polls until terminal; mocked orchestrator returns
    ``running`` then ``successful`` and the CLI exits 0 with the
    final payload."""
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.side_effect = [
        {"result": {"status": "running", "request_id": "rid"}},
        {"result": {"status": "successful", "request_id": "rid"}},
    ]
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))
    res = runner.invoke(
        cli.app, ["cds", "wait", "rid", "--interval", "0", "--timeout", "5", "--json"]
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    parsed = json.loads(res.stdout)
    assert parsed["status"] == "successful"
    assert fake.run.await_count == 2


def test_progress_line_formats_chunk_progress() -> None:
    from copernicus_mcp.cli import _progress_line

    line = _progress_line(
        {"progress": {"completed": 3, "total": 12}, "chunks": {"running": 2, "queued": 7, "failed": 0}}
    )
    assert line == "parts: 3/12 done (running 2, queued 7)"
    # a single (non-chunked) request carries no progress block → no line
    assert _progress_line({"status": "running"}) is None


def test_cds_wait_shows_chunk_progress(runner: CliRunner, env: Path, monkeypatch: Any) -> None:
    """Non-json wait prints a live 'parts done' line to stderr for a chunked parent,
    so the agent/human sees the download position instead of a silent hang."""
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.side_effect = [
        {
            "result": {
                "status": "running",
                "request_id": "p",
                "progress": {"completed": 1, "total": 3},
                "chunks": {"running": 2, "queued": 0, "failed": 0},
            }
        },
        {
            "result": {
                "status": "successful",
                "request_id": "p",
                "progress": {"completed": 3, "total": 3},
                "chunks": {"running": 0, "queued": 0, "failed": 0},
            }
        },
    ]
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))
    res = runner.invoke(cli.app, ["cds", "wait", "p", "--interval", "0", "--timeout", "5"])
    assert res.exit_code == 0, res.stdout + res.stderr
    assert "1/3" in res.stderr
    assert "3/3" in res.stderr
