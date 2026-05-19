from __future__ import annotations

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


def test_help_lists_must_have_subcommands(runner: CliRunner) -> None:
    from copernicus_mcp.cli import app

    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for cmd in ("serve", "version", "status", "marine"):
        assert cmd in res.stdout


def test_marine_help_lists_subcommands(runner: CliRunner) -> None:
    from copernicus_mcp.cli import app

    res = runner.invoke(app, ["marine", "--help"])
    assert res.exit_code == 0
    for cmd in (
        "search-datasets",
        # T-CMEMS-HIER-007: the two new hierarchical-search subcommands.
        "search-groups",
        "search-products",
        "describe",
        "estimate",
        "subset",
        "check-status",
    ):
        assert cmd in res.stdout


def test_search_groups_dispatches_search_groups_operation(
    runner: CliRunner, env: Path, monkeypatch: Any
) -> None:
    """T-CMEMS-HIER-007: ``marine search-groups`` forwards the query
    to the orchestrator's ``search_groups`` operation."""
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.return_value = {
        "result": {
            "selected": [{"group_id": "physics-arctic-state", "score": 6.0}],
            "rejected": [],
            "reason": "ok",
            "confidence": "high",
            "fallback_available": False,
        }
    }
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(
        cli.app,
        ["marine", "search-groups", "--query", "arctic sea ice", "--json"],
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    parsed = json.loads(res.stdout)
    assert parsed["selected"][0]["group_id"] == "physics-arctic-state"
    kwargs = fake.run.call_args.kwargs
    assert kwargs["operation"] == "search_groups"
    assert kwargs["params"]["query"] == "arctic sea ice"


def test_search_products_dispatches_search_products_operation(
    runner: CliRunner, env: Path, monkeypatch: Any
) -> None:
    """T-CMEMS-HIER-007: ``marine search-products`` forwards the
    group ids to the orchestrator's ``search_products`` operation."""
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.return_value = {
        "result": {
            "selected": [{"product_id": "ARCTIC_ANALYSISFORECAST_PHY_002_001", "score": 0.0}],
            "rejected": [],
            "reason": "ok",
            "confidence": "medium",
            "fallback_available": False,
        }
    }
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(
        cli.app,
        [
            "marine",
            "search-products",
            "--groups",
            "physics-arctic-state",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    parsed = json.loads(res.stdout)
    assert parsed["selected"][0]["product_id"] == "ARCTIC_ANALYSISFORECAST_PHY_002_001"
    kwargs = fake.run.call_args.kwargs
    assert kwargs["operation"] == "search_products"
    assert kwargs["params"]["group_ids"] == ["physics-arctic-state"]


def test_search_datasets_accepts_product_ids(
    runner: CliRunner, env: Path, monkeypatch: Any
) -> None:
    """T-CMEMS-HIER-007: ``marine search-datasets --product-ids A,B``
    routes through the hierarchical cards path."""
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.return_value = {
        "result": {
            "selected": [{"dataset_id": "ds-1", "product_id": "A"}],
            "rejected": [],
            "reason": "ok",
            "confidence": "high",
            "fallback_available": False,
        }
    }
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(
        cli.app,
        [
            "marine",
            "search-datasets",
            "--product-ids",
            "GLOBAL_ANALYSISFORECAST_PHY_001_024,GLOBAL_MULTIYEAR_PHY_001_030",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    kwargs = fake.run.call_args.kwargs
    assert kwargs["params"]["product_ids"] == [
        "GLOBAL_ANALYSISFORECAST_PHY_001_024",
        "GLOBAL_MULTIYEAR_PHY_001_030",
    ]


def test_version_prints_version(runner: CliRunner) -> None:
    from copernicus_mcp.cli import app
    from copernicus_mcp.version import __version__

    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert __version__ in res.stdout


def test_status_prints_block(runner: CliRunner, env: Path) -> None:
    from copernicus_mcp.cli import app

    res = runner.invoke(app, ["status"])
    assert res.exit_code == 0, res.stdout + res.stderr
    assert "version" in res.stdout.lower() or "backend" in res.stdout.lower()


def test_status_json(runner: CliRunner, env: Path) -> None:
    from copernicus_mcp.cli import app

    res = runner.invoke(app, ["status", "--json"])
    assert res.exit_code == 0
    parsed = json.loads(res.stdout)
    assert "version" in parsed
    assert "backends" in parsed


def test_search_datasets_json_uses_orchestrator(
    runner: CliRunner, env: Path, monkeypatch: Any
) -> None:
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.return_value = {"result": {"results": [{"dataset_id": "x", "title": "Sea temp"}]}}
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(cli.app, ["marine", "search-datasets", "--keyword", "temp", "--json"])
    assert res.exit_code == 0, res.stdout + res.stderr
    parsed = json.loads(res.stdout)
    assert parsed["results"][0]["dataset_id"] == "x"
    fake.run.assert_awaited_once()
    kwargs = fake.run.call_args.kwargs
    assert kwargs["operation"] == "search"
    assert kwargs["params"]["keyword"] == "temp"


def test_describe_passes_dataset_id(runner: CliRunner, env: Path, monkeypatch: Any) -> None:
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.return_value = {"result": {"dataset_id": "ds-1", "title": "T"}}
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(cli.app, ["marine", "describe", "ds-1", "--json"])
    assert res.exit_code == 0, res.stdout + res.stderr
    kwargs = fake.run.call_args.kwargs
    assert kwargs["operation"] == "describe"
    assert kwargs["params"]["identifier"] == "ds-1"


def test_subset_with_yes_proceeds_through_confirmation(
    runner: CliRunner, env: Path, monkeypatch: Any
) -> None:
    """ConfirmationRequired payload + --yes → re-invoke with confirmed=True."""
    from copernicus_mcp import cli

    fake = AsyncMock()
    confirmation = {
        "confirmation_required": True,
        "advisory_message": "approx 2.5 GB",
        "estimated_size_bytes": 2_500_000_000,
    }
    success = {"result": {"filepath": "/tmp/x.nc", "size_bytes": 1234}}
    fake.run.side_effect = [confirmation, success]
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(
        cli.app,
        [
            "marine",
            "subset",
            "--dataset",
            "ds-1",
            "--bbox",
            "-10,30,10,50",
            "--time",
            "2024-01-01T00:00:00Z,2024-01-02T00:00:00Z",
            "--variables",
            "thetao",
            "--depth",
            "0,100",
            "--yes",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    assert fake.run.await_count == 2
    second = fake.run.call_args_list[1].kwargs
    assert second["options"]["confirmed"] is True


def test_subset_without_yes_non_tty_aborts_with_exit_code(
    runner: CliRunner, env: Path, monkeypatch: Any
) -> None:
    """Non-interactive stdin + ConfirmationRequired without --yes → abort code 3."""
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.return_value = {
        "confirmation_required": True,
        "advisory_message": "approx 2.5 GB",
        "estimated_size_bytes": 2_500_000_000,
    }
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(
        cli.app,
        [
            "marine",
            "subset",
            "--dataset",
            "ds-1",
            "--bbox",
            "-10,30,10,50",
            "--time",
            "2024-01-01T00:00:00Z,2024-01-02T00:00:00Z",
            "--variables",
            "thetao",
            "--depth",
            "0,100",
        ],
        # CliRunner stdin is non-TTY by default.
    )
    assert res.exit_code == 3, res.stdout + (res.stderr or "")
    # The confirmation panel + abort message go to stderr to keep --json
    # stdout pipe-safe. Just assert exit code 3 here.


def test_check_status_passes_request_id(runner: CliRunner, env: Path, monkeypatch: Any) -> None:
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.return_value = {"result": {"status": "successful"}}
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(cli.app, ["marine", "check-status", "req-1", "--json"])
    assert res.exit_code == 0, res.stdout + res.stderr
    kwargs = fake.run.call_args.kwargs
    assert kwargs["operation"] == "poll"
    assert kwargs["params"]["request_id"] == "req-1"


def test_estimate_returns_size(runner: CliRunner, env: Path, monkeypatch: Any) -> None:
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.return_value = {"result": {"estimated_size_bytes": 12345}}
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(
        cli.app,
        [
            "marine",
            "estimate",
            "--dataset",
            "ds-1",
            "--bbox",
            "-10,30,10,50",
            "--time",
            "2024-01-01T00:00:00Z,2024-01-02T00:00:00Z",
            "--variables",
            "thetao",
            "--depth",
            "0,100",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    parsed = json.loads(res.stdout)
    assert parsed["estimated_size_bytes"] == 12345


def test_json_mode_emits_pure_json_on_error(runner: CliRunner, env: Path, monkeypatch: Any) -> None:
    """code-reviewer round 1 HIGH: --json must produce valid JSON on stdout
    even when the orchestrator returns an error envelope. Rich panels go to
    stderr."""
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.return_value = {
        "error": {
            "error_class": "ValidationError",
            "message": "bad bbox",
            "recovery_action": "modify_request_parameters",
        }
    }
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(
        cli.app,
        ["marine", "search-datasets", "--keyword", "x", "--json"],
        # Mix stdout/stderr so we can see what landed where.
    )
    assert res.exit_code != 0
    # stdout MUST be parseable JSON containing the error envelope.
    parsed = json.loads(res.stdout)
    assert parsed["error"]["error_class"] == "ValidationError"


def test_subset_empty_variables_exits_with_user_input_code(
    runner: CliRunner, env: Path, monkeypatch: Any
) -> None:
    """code-reviewer round 1 MEDIUM: empty --variables → CLI exit 2 with
    a clear message, not a downstream Pydantic surprise."""
    from copernicus_mcp import cli

    fake = AsyncMock()
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(
        cli.app,
        [
            "marine",
            "subset",
            "--dataset",
            "ds-1",
            "--bbox",
            "-10,30,10,50",
            "--time",
            "2024-01-01T00:00:00Z,2024-01-02T00:00:00Z",
            "--variables",
            "",
            "--depth",
            "0,100",
        ],
    )
    assert res.exit_code == 2
    fake.run.assert_not_awaited()


def test_cli_subset_does_not_expose_async_flag(runner: CliRunner, env: Path) -> None:
    """Round 1 H2: ``--async`` was removed from the CLI because the
    one-shot process model cancels the task on shutdown. The intended
    use is the MCP server. ``--async`` invocation must now exit with
    a typer-level user-input error code rather than silently spawning
    a doomed task."""
    from copernicus_mcp.cli import app

    res = runner.invoke(
        app,
        [
            "marine",
            "subset",
            "--dataset",
            "ds-1",
            "--bbox",
            "-10,30,10,50",
            "--time",
            "2024-01-01T00:00:00Z,2024-01-02T00:00:00Z",
            "--variables",
            "thetao",
            "--depth",
            "0,100",
            "--async",
            "--json",
        ],
    )
    assert res.exit_code == 2, res.stdout + (res.stderr or "")


def test_marine_wait_polls_until_terminal(runner: CliRunner, env: Path, monkeypatch: Any) -> None:
    """T-039: ``marine wait REQUEST_ID`` polls ``check_status`` until the
    status reaches a terminal value (successful/failed/cancelled). Final
    payload is emitted on stdout."""
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.side_effect = [
        {"result": {"status": "running", "request_id": "req-poll-1"}},
        {"result": {"status": "running", "request_id": "req-poll-1"}},
        {
            "result": {
                "status": "successful",
                "request_id": "req-poll-1",
                "cache_key": "cmems:submit:xyz",
            }
        },
    ]
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(
        cli.app,
        [
            "marine",
            "wait",
            "req-poll-1",
            "--interval",
            "0",
            "--timeout",
            "5",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout + (res.stderr or "")
    parsed = json.loads(res.stdout)
    assert parsed["status"] == "successful"
    assert fake.run.await_count == 3


def test_marine_wait_times_out(runner: CliRunner, env: Path, monkeypatch: Any) -> None:
    """``marine wait`` exits 1 with a CANONICAL TimeoutError envelope when
    the timeout expires before terminal. Codex round 2 LOW: the prior
    assertion of ``exit_code == 1`` would pass if the path regressed to
    any nonzero error — strengthen by parsing the JSON envelope and
    asserting the canonical record fields."""
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.return_value = {"result": {"status": "running", "request_id": "rq"}}
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = runner.invoke(
        cli.app,
        [
            "marine",
            "wait",
            "rq",
            "--interval",
            "0",
            "--timeout",
            "0",
            "--json",
        ],
    )
    assert res.exit_code == 1
    parsed = json.loads(res.stdout)
    assert parsed["error"]["error_class"] == "TimeoutError"
    assert parsed["error"]["recovery_action"] == "retry_with_modification"
    # Canonical record fields populated by ``build_error_record``.
    assert "error_id" in parsed["error"]
    assert "timestamp_utc" in parsed["error"]


def test_serve_calls_server_main(runner: CliRunner, env: Path, monkeypatch: Any) -> None:
    """code-reviewer round 1 MEDIUM: ``serve`` wiring smoke test."""
    import copernicus_mcp.server as server_mod

    called: dict[str, bool] = {}

    def _fake_main(cli_overrides: dict[str, object] | None = None) -> None:
        called["yes"] = True

    monkeypatch.setattr(server_mod, "main", _fake_main)
    from copernicus_mcp.cli import app

    res = runner.invoke(app, ["serve"])
    assert res.exit_code == 0
    assert called.get("yes") is True


def _fake_builder(orch: AsyncMock) -> Any:
    """Return a context-manager-style builder yielding the mocked orchestrator."""
    import contextlib

    @contextlib.asynccontextmanager
    async def _builder() -> Any:
        yield orch

    return _builder
