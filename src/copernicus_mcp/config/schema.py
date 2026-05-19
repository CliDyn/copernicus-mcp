from __future__ import annotations

from pathlib import Path
from typing import Literal

from platformdirs import user_cache_dir, user_state_dir
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ServerConfig(_Base):
    name: str = "copernicus-mcp"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    transport: Literal["stdio"] = "stdio"


def _default_cache_directory() -> Path:
    """Per-OS cache directory via ``platformdirs``.

    T-CDS-019: replaces the hard-coded ``~/.cache/copernicus-mcp`` which
    was Linux-only. ``appauthor=False`` is critical on Windows —
    without it platformdirs defaults the author segment to the appname,
    producing ``%LOCALAPPDATA%\\copernicus-mcp\\copernicus-mcp\\Cache``
    (codex round-1 MEDIUM).

    Resolves to:
      - Linux/XDG: ``~/.cache/copernicus-mcp`` (respects ``XDG_CACHE_HOME``).
      - macOS:    ``~/Library/Caches/copernicus-mcp``.
      - Windows:  ``%LOCALAPPDATA%\\copernicus-mcp\\Cache``.
    """
    return Path(user_cache_dir("copernicus-mcp", appauthor=False))


def _default_state_database() -> Path:
    """Per-OS state-DB location via ``platformdirs``.

    Uses ``user_state_dir`` (not ``user_data_dir``) so Linux users
    keep their existing ``~/.local/state/copernicus-mcp/state.db`` —
    swapping to ``user_data_dir`` would orphan it under
    ``~/.local/share/...`` (codex round-1 MEDIUM). On macOS / Windows
    platformdirs treats state == data so the result matches their
    persistent-data conventions.

    Resolves to:
      - Linux/XDG: ``~/.local/state/copernicus-mcp/state.db``
                   (respects ``XDG_STATE_HOME``).
      - macOS:    ``~/Library/Application Support/copernicus-mcp/state.db``.
      - Windows:  ``%LOCALAPPDATA%\\copernicus-mcp\\state.db``.
    """
    return Path(user_state_dir("copernicus-mcp", appauthor=False)) / "state.db"


class StorageConfig(_Base):
    cache_directory: Path = Field(default_factory=_default_cache_directory)
    state_database: Path = Field(default_factory=_default_state_database)
    cache_size_limit_gb: float = 50.0
    cache_eviction_policy: Literal["lru"] = "lru"

    @field_validator("cache_directory", "state_database", mode="after")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser()


class HttpConfig(_Base):
    default_timeout_seconds: int = 60
    default_retry_max_attempts: int = 5
    default_retry_base_delay_seconds: float = 1.0
    default_retry_max_delay_seconds: float = 60.0


class CacheConfig(_Base):
    search_results_ttl_seconds: int = 3600
    metadata_ttl_seconds: int = 86400


class BudgetPolicy(_Base):
    cmems_max_concurrent_subset_operations: int = 2
    cmems_per_request_size_warning_gb: float = 1.0
    # Toolbox dry-run estimate timeout (seconds). Kept distinct from
    # http.default_timeout_seconds because copernicusmarine does not use httpx.
    cmems_estimate_timeout_seconds: float = 30.0
    # CDS confirmation thresholds. Per T-CDS-005 / codex spec review:
    # CDS estimate is always heuristic (epistemic_status="approximate") so
    # we cannot mirror CMEMS's "approximate ⇒ ask" rule without forcing
    # confirmation on every submit. We gate on bytes OR queue tier
    # (research §6.5.4: queue latency is field-count-driven, independent
    # of bytes). ``cds_confirm_on_queue_tier`` is the second guard rail.
    cds_per_request_size_warning_gb: float = 1.0
    cds_confirm_on_queue_tier: tuple[str, ...] = ("medium", "heavy")


class ObservabilityConfig(_Base):
    structured_logging: bool = True
    log_format: Literal["json", "console"] = "json"


class CopernicusMcpConfig(_Base):
    server: ServerConfig = Field(default_factory=ServerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    http: HttpConfig = Field(default_factory=HttpConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    budget: BudgetPolicy = Field(default_factory=BudgetPolicy)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    # T-022/008 session 2026-05-16: default flipped from ["cmems"] to
    # ["cmems", "cds"]. CDS tool registration is double-gated (in
    # enabled_backends AND credentials resolve), so users without a
    # CDS PAT see no extra tools and pay no extra context cost.
    enabled_backends: list[Literal["cmems", "cds"]] = Field(
        default_factory=lambda: ["cmems", "cds"]
    )
