from __future__ import annotations

from pathlib import Path
from typing import Literal

from platformdirs import user_cache_dir, user_state_dir
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    # T-STATEDB-001: upper bound on each journal-mode open attempt of the
    # state database. On a network home a stale WAL can make the open HANG
    # (not error) until the MCP client's connect times out and the agent
    # proceeds tool-less; this converts the hang into the DELETE-journal
    # fallback chain and, at worst, a loud canonical error.
    state_db_pragma_timeout_seconds: float = Field(default=15.0, gt=0.0)

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
    # T-CDS v2: byte size is now honestly "unknown" for any uncalibrated request
    # (not just whole-file products), so confirming on every unknown size would
    # prompt on most first-time submits. The byte-heavy / dangerous requests are
    # already gated by the queue-tier (medium/heavy) and cost-limit checks, so we
    # do NOT additionally block on unknown size by default. Set True to re-add a
    # confirmation whenever the byte size cannot be estimated.
    cds_confirm_on_unknown_size: bool = False
    # T-CDS-CHUNK (model B): when a CDS request exceeds the dataset's
    # server-side cost limit, the backend splits it into whole calendar units
    # (year/month/day) as one logical multi-file workflow. The agent picks the
    # granularity (``__options.chunk_by``); the MCP proposes and validates.
    # Submission is PACED (``cds_chunk_max_inflight`` below — the v2 "CDS
    # queues any excess rather than rejecting it" premise was disproved by a
    # live run: a 21-child burst tripped the per-user concurrency throttle and
    # failed 39/42 children). On top of pacing, three fan-out guards (a large
    # split is many jobs even when paced):
    #   - ``confirm_above``: a plan with > N chunks raises a ConfirmationRequired
    #     the agent satisfies with ``confirmed=true`` — putting a human in the loop.
    #   - ``reconfirm_above``: > N chunks demands a SECOND, deliberate confirmation
    #     — ``confirmed`` alone is not enough; the agent must ALSO pass
    #     ``__options.confirm_large_fanout=true``. This is the "repeat" gate that a
    #     glitched agent blanket-setting ``confirmed`` cannot slip a runaway through.
    #   - ``max_chunks``: hard ceiling — over it the request is rejected outright
    #     (no confirm bypasses it), with a "use a coarser granularity" hint.
    # Constraints: 0 ≤ confirm_above ≤ reconfirm_above, and max_chunks ≥ 1.
    # ``max_chunks`` is INDEPENDENT — a low hard cap below the soft thresholds is a
    # valid, more conservative config (it hard-rejects a big split rather than
    # re-confirming it).
    cds_auto_chunk_enabled: bool = True
    cds_auto_chunk_confirm_above: int = 30
    cds_auto_chunk_reconfirm_above: int = 100
    cds_auto_chunk_max_chunks: int = 366
    # T-CDS-RESIL-002: bound on concurrently active (queued/running) children
    # of a chunked parent. The first wave submits at most this many; the
    # poll-driven refill tops the level back up as children reach a terminal
    # state. 0 = unlimited (the old fan-out-everything behaviour). ~5 matches
    # what the service tolerates empirically (field run 31); the real ceiling is
    # undocumented and may differ per account, hence configurable.
    cds_chunk_max_inflight: int = Field(default=5, ge=0)
    # T-CDS-RESIL-003: bounded re-submission of a chunk whose child failed
    # with the capacity signature (RESIL-001 classification, or an empty-log
    # "unknown" later corroborated by a successful sibling). Same overrides,
    # new child id, at most ``retry_limit`` times per chunk, spaced by
    # ``retry_backoff_seconds`` (poll-driven — the wait is observed on the
    # next poll after the window passes). Content failures are never retried:
    # a malformed request that is retried is a slower way to fail. 0 disables
    # retry (restores fail-fast).
    cds_chunk_retry_limit: int = Field(default=3, ge=0)
    cds_chunk_retry_backoff_seconds: float = Field(default=120.0, ge=0.0)
    # T-CDS-MODEL-002: after downloading a result for a single-model-execution
    # dataset (projections-cmip6 / CORDEX), verify the delivered archive
    # actually contains the requested model before storing it under the cache
    # key. The Rook backend has been observed delivering the FIRST model of a
    # list silently; a poisoned cache entry would satisfy every future dedupe.
    # Mismatch → the workflow fails with ``delivered_content_mismatch``.
    cds_delivery_check_enabled: bool = True
    # T-CDS-KEYCHECK-001: reject input keys the dataset's constraints snapshot
    # does not list, at submit time. The server accepts unknown keys and
    # silently ignores them — delivering the wrong selection, or failing
    # minutes later with an empty log. Fail-open when a dataset has no
    # snapshot entry; per-request escape hatch:
    # ``__options.skip_input_validation=true``.
    cds_input_key_validation: bool = True
    # T-CDS-LICENCE-001: register the MCP-facing ``cds_accept_licence`` tool.
    # Accepting a dataset licence legally binds the ACCOUNT owner, so the
    # agent-visible surface is operator-opt-in (the operator "standing authorisation"
    # model). The CLI ``cds accept-licence`` works regardless — the CLI is the
    # operator. Listing licences is harmless and always registered.
    cds_licence_accept_enabled: bool = False
    # T-CDS-ASYNC-DOWNLOAD: check_status spawns the result-file download and waits
    # at most this long for it inline. A fast / small file completes in one poll;
    # a large one exceeds the grace and finishes in the BACKGROUND (the poll
    # returns status "running", phase "downloading") so the agent is not blocked on
    # the transfer. ``asyncio.wait`` returns the instant the download finishes, so
    # this adds no delay to a quick download — it is only an upper bound. 0 =
    # always background.
    cds_download_inline_grace_seconds: float = Field(default=2.0, ge=0.0)
    # T-CDS-DL-001: download results into a STABLE staging part keyed by the
    # cache hash and resume it via HTTP Range (multiurl ``resume_transfers``)
    # instead of restarting from byte zero on every poll. An interrupted
    # transfer (client died, poll process exited) keeps its bytes; the next
    # poll appends from where it stopped — without this a file larger than one
    # grace-window of bytes can never land for an ephemeral poller. Off =
    # per-attempt throwaway staging (pre-DL-001 behaviour).
    cds_resume_downloads: bool = True

    @model_validator(mode="after")
    def _check_fanout_thresholds(self) -> BudgetPolicy:
        """Sanity-check the auto-chunk fan-out thresholds: non-negative and ordered
        (confirm <= reconfirm). ``max_chunks`` is an INDEPENDENT hard cap and may
        sit below the soft thresholds — a low ceiling just hard-rejects a big split
        (stricter than the confirms, never a bypass), a valid conservative config;
        only require it >= 1."""
        if not (
            0 <= self.cds_auto_chunk_confirm_above <= self.cds_auto_chunk_reconfirm_above
            and self.cds_auto_chunk_max_chunks >= 1
        ):
            raise ValueError(
                "auto-chunk fan-out thresholds must satisfy 0 <= confirm_above <= "
                "reconfirm_above and max_chunks >= 1; got "
                f"confirm_above={self.cds_auto_chunk_confirm_above}, "
                f"reconfirm_above={self.cds_auto_chunk_reconfirm_above}, "
                f"max_chunks={self.cds_auto_chunk_max_chunks}"
            )
        return self



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
