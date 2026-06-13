from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError


def test_defaults_load_and_validate() -> None:
    from copernicus_mcp.config import ConfigLoader

    cfg = ConfigLoader().load()
    assert cfg.server.name == "copernicus-mcp"
    assert cfg.server.log_level == "INFO"
    assert cfg.server.transport == "stdio"
    # Both backends enabled out-of-box. Registration auto-gates on
    # credentials (T-CDS-007 dual-gate), so users without a CDS PAT
    # see no extra tools and pay no extra context cost.
    assert cfg.enabled_backends == ["cmems", "cds"]
    assert cfg.storage.cache_size_limit_gb == 50.0
    assert cfg.storage.cache_eviction_policy == "lru"


def test_budget_policy_auto_chunk_defaults() -> None:
    """T-CDS-CHUNK v2: auto-chunking on by default, no inflight throttle (submit
    all), with a generous max_chunks guard against pathological fan-out."""
    from copernicus_mcp.config.schema import BudgetPolicy

    budget = BudgetPolicy()
    assert budget.cds_auto_chunk_enabled is True
    assert budget.cds_auto_chunk_max_chunks == 366
    assert not hasattr(budget, "cds_auto_chunk_max_inflight")


def test_yaml_overrides_defaults(tmp_path: Path) -> None:
    from copernicus_mcp.config import ConfigLoader

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("server:\n  log_level: DEBUG\n")
    cfg = ConfigLoader().load(explicit_config_path=cfg_path)
    assert cfg.server.log_level == "DEBUG"


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from copernicus_mcp.config import ConfigLoader

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("server:\n  log_level: DEBUG\n")
    monkeypatch.setenv("COPERNICUS_MCP_LOG_LEVEL", "WARNING")
    cfg = ConfigLoader().load(explicit_config_path=cfg_path)
    assert cfg.server.log_level == "WARNING"


def test_cli_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from copernicus_mcp.config import ConfigLoader

    monkeypatch.setenv("COPERNICUS_MCP_LOG_LEVEL", "WARNING")
    cfg = ConfigLoader().load(cli_overrides={"server": {"log_level": "ERROR"}})
    assert cfg.server.log_level == "ERROR"


def test_invalid_log_level_raises(tmp_path: Path) -> None:
    from copernicus_mcp.config import ConfigLoader

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("server:\n  log_level: TRACE\n")
    with pytest.raises(ValidationError) as exc_info:
        ConfigLoader().load(explicit_config_path=cfg_path)
    assert "log_level" in str(exc_info.value)


def test_tilde_expansion_in_path_fields() -> None:
    from copernicus_mcp.config import ConfigLoader

    cfg = ConfigLoader().load()
    assert "~" not in str(cfg.storage.cache_directory)
    assert "~" not in str(cfg.storage.state_database)
    assert cfg.storage.cache_directory.is_absolute()
    assert cfg.storage.state_database.is_absolute()


def test_env_overrides_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from copernicus_mcp.config import ConfigLoader

    custom = tmp_path / "custom-cache"
    monkeypatch.setenv("COPERNICUS_MCP_CACHE_DIR", str(custom))
    cfg = ConfigLoader().load()
    assert cfg.storage.cache_directory == custom


# T-CDS-019: cross-platform default paths via platformdirs


def test_default_cache_directory_uses_platformdirs() -> None:
    """The bare default (no env, no yaml override, no CLI override) is
    derived from ``platformdirs.user_cache_dir`` so Windows users get
    ``%LOCALAPPDATA%\\copernicus-mcp\\Cache``, macOS users get
    ``~/Library/Caches/copernicus-mcp``, and Linux keeps
    ``~/.cache/copernicus-mcp``. Drops the hard-coded ``~/.cache/...``
    that was wrong on macOS/Windows.

    ``appauthor=False`` is required to avoid the doubled
    ``copernicus-mcp\\copernicus-mcp`` segment on Windows."""
    from platformdirs import user_cache_dir

    from copernicus_mcp.config import ConfigLoader

    cfg = ConfigLoader().load()
    expected = Path(user_cache_dir("copernicus-mcp", appauthor=False))
    assert cfg.storage.cache_directory == expected


def test_default_state_database_uses_platformdirs() -> None:
    """``state.db`` uses ``user_state_dir`` so Linux backcompat is
    preserved (existing users keep their
    ``~/.local/state/copernicus-mcp/state.db``).

    Resolves to:
      - Linux/XDG: ``~/.local/state/copernicus-mcp/state.db``
      - macOS:     ``~/Library/Application Support/copernicus-mcp/state.db``
      - Windows:   ``%LOCALAPPDATA%\\copernicus-mcp\\state.db``"""
    from platformdirs import user_state_dir

    from copernicus_mcp.config import ConfigLoader

    cfg = ConfigLoader().load()
    expected = Path(user_state_dir("copernicus-mcp", appauthor=False)) / "state.db"
    assert cfg.storage.state_database == expected


def test_cli_override_beats_env_for_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI > env > yaml > defaults. A ``--cache-dir`` passed at the CLI
    layer wins over ``COPERNICUS_MCP_CACHE_DIR``."""
    from copernicus_mcp.config import ConfigLoader

    env_path = tmp_path / "env-cache"
    cli_path = tmp_path / "cli-cache"
    monkeypatch.setenv("COPERNICUS_MCP_CACHE_DIR", str(env_path))
    cfg = ConfigLoader().load(
        cli_overrides={"storage": {"cache_directory": str(cli_path)}}
    )
    assert cfg.storage.cache_directory == cli_path


def test_cli_cache_dir_flag_overrides_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-CDS-019: the top-level ``--cache-dir`` flag injects the value
    into the ConfigLoader's ``cli_overrides`` and wins over env. End-to-
    end through the Typer app so the option wiring is exercised."""
    from typer.testing import CliRunner

    from copernicus_mcp import cli

    custom = tmp_path / "cli-flag-cache"
    captured: dict[str, object] = {}

    def _fake_load(self, cli_overrides=None, explicit_config_path=None):
        captured["cli_overrides"] = cli_overrides
        from copernicus_mcp.config.schema import CopernicusMcpConfig

        return CopernicusMcpConfig()

    monkeypatch.setattr(
        "copernicus_mcp.config.loader.ConfigLoader.load", _fake_load
    )

    # Pre-seed an env override so we can assert the CLI wins.
    monkeypatch.setenv(
        "COPERNICUS_MCP_CACHE_DIR", str(tmp_path / "env-cache")
    )

    import contextlib
    from unittest.mock import AsyncMock

    fake = AsyncMock()
    fake.status.return_value = {"ok": True}

    @contextlib.asynccontextmanager
    async def _builder():
        # Force the override-capturing load() to fire.
        cli._cli_config_overrides()
        from copernicus_mcp.config import ConfigLoader

        ConfigLoader().load(cli_overrides=cli._cli_config_overrides() or None)
        yield fake

    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _builder)

    runner = CliRunner()
    res = runner.invoke(
        cli.app, ["--cache-dir", str(custom), "status", "--json"]
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    assert captured["cli_overrides"] == {
        "storage": {"cache_directory": str(custom)}
    }


def test_deep_merge_nested_dicts() -> None:
    from copernicus_mcp.config.loader import _deep_merge

    base = {"a": {"x": 1, "y": 2}, "b": [1, 2], "c": 3}
    overlay = {"a": {"y": 99, "z": 3}, "b": [9], "d": 4}
    merged = _deep_merge(base, overlay)
    assert merged == {
        "a": {"x": 1, "y": 99, "z": 3},
        "b": [9],
        "c": 3,
        "d": 4,
    }
    # base must not be mutated
    assert base == {"a": {"x": 1, "y": 2}, "b": [1, 2], "c": 3}


def test_budget_policy_validates_fanout_thresholds() -> None:
    """BudgetPolicy enforces the two real constraints — non-negative thresholds
    and ordered tiers (confirm <= reconfirm) — but NOT reconfirm < max_chunks: a
    hard cap set below the soft thresholds is a legitimate conservative config
    (plans over it are hard-rejected, stricter than tier 2, not bypassed)."""
    from copernicus_mcp.config.schema import BudgetPolicy

    BudgetPolicy()  # default 30 <= 100, max 366
    # A low hard cap below the soft thresholds is allowed (conservative, safe).
    BudgetPolicy(cds_auto_chunk_max_chunks=50)
    BudgetPolicy(cds_auto_chunk_reconfirm_above=400, cds_auto_chunk_max_chunks=366)
    with pytest.raises(ValidationError):  # tiers out of order
        BudgetPolicy(cds_auto_chunk_confirm_above=50, cds_auto_chunk_reconfirm_above=30)
    with pytest.raises(ValidationError):  # negative threshold
        BudgetPolicy(cds_auto_chunk_confirm_above=-1)
    with pytest.raises(ValidationError):  # nonsensical zero hard cap
        BudgetPolicy(cds_auto_chunk_max_chunks=0)
