from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from copernicus_mcp.config.schema import CopernicusMcpConfig

_DEFAULTS_PATH = Path(__file__).parent / "defaults.yaml"

_USER_CONFIG_PATHS: tuple[Path, ...] = (
    Path("~/.config/copernicus-mcp/config.yaml").expanduser(),
    Path("~/.copernicus-mcp.yaml").expanduser(),
)

# env var -> dotted path inside the config tree
_ENV_VAR_MAP: dict[str, tuple[str, ...]] = {
    "COPERNICUS_MCP_LOG_LEVEL": ("server", "log_level"),
    "COPERNICUS_MCP_CACHE_DIR": ("storage", "cache_directory"),
    "COPERNICUS_MCP_STATE_DB": ("storage", "state_database"),
    "COPERNICUS_MCP_CDS_CHUNK_MAX_INFLIGHT": ("budget", "cds_chunk_max_inflight"),
    "COPERNICUS_MCP_CDS_CHUNK_RETRY_LIMIT": ("budget", "cds_chunk_retry_limit"),
    "COPERNICUS_MCP_CDS_CHUNK_RETRY_BACKOFF_SECONDS": (
        "budget",
        "cds_chunk_retry_backoff_seconds",
    ),
    "COPERNICUS_MCP_CDS_RESUME_DOWNLOADS": ("budget", "cds_resume_downloads"),
}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge overlay into a deep copy of base. Dicts merge; everything else replaces."""
    result = copy.deepcopy(base)
    for key, ov in overlay.items():
        cur = result.get(key)
        if isinstance(cur, dict) and isinstance(ov, dict):
            result[key] = _deep_merge(cur, ov)
        else:
            result[key] = copy.deepcopy(ov)
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must contain a mapping at top level")
    return data


def _env_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for env_name, dotted in _ENV_VAR_MAP.items():
        value = os.environ.get(env_name)
        if not value:
            continue
        cursor = overrides
        for key in dotted[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[dotted[-1]] = value
    # Per-backend toggle via env: ``COPERNICUS_MCP_ENABLED_BACKENDS`` accepts
    # a comma-separated list (e.g. ``cmems,cds``) and overrides the
    # top-level ``enabled_backends``. Added for T-CDS-008 so integration
    # tests / CLI subprocesses can enable CDS without mutating the user's
    # ``~/.config/copernicus-mcp/config.yaml``.
    raw = os.environ.get("COPERNICUS_MCP_ENABLED_BACKENDS")
    if raw:
        items = [s.strip() for s in raw.split(",") if s.strip()]
        if items:
            overrides["enabled_backends"] = items
    return overrides


class ConfigLoader:
    """Layered config loader: defaults -> user files -> explicit -> env -> CLI."""

    def load(
        self,
        cli_overrides: dict[str, Any] | None = None,
        explicit_config_path: Path | None = None,
    ) -> CopernicusMcpConfig:
        merged = _load_yaml(_DEFAULTS_PATH)
        for user_path in _USER_CONFIG_PATHS:
            if user_path.exists():
                merged = _deep_merge(merged, _load_yaml(user_path))
        if explicit_config_path is not None and explicit_config_path.exists():
            merged = _deep_merge(merged, _load_yaml(explicit_config_path))
        merged = _deep_merge(merged, _env_overrides())
        if cli_overrides:
            merged = _deep_merge(merged, cli_overrides)
        return CopernicusMcpConfig.model_validate(merged)
