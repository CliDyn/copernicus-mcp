"""Common four-field request envelope and cross-backend ``RequestOptions``.

Iter 1 narrows ``BackendId`` to ``"cmems"`` (research §12.1.2 lists all seven;
plan T-011 step 1 explicitly restricts to CMEMS for this iteration). Widening
is purely additive in later iterations — extend the ``Literal`` and append the
new backend's request schema in ``schemas_<backend>.py``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

BackendId = Literal["cmems"]
OperationType = Literal[
    "search",
    "describe",
    "validate",
    "estimate",
    "submit",
    "get",  # T-CMEMS-GET-006: native-file retrieval via ``backend.get_files``.
    "list_files",  # T-CMEMS-GET-INDEX-004: Layer 2 index-driven file listing.
    "get_coordinates",  # T-022 second half: dataset coordinate axes.
    "apply_constraints",  # T-CDS-016: progressive valid-values narrowing.
    "poll",
    "fetch",
    "cancel",
]


class RequestOptions(BaseModel):
    """Cross-backend operational options (research §12.1.3).

    Every field is optional; absence activates server defaults. ``extra="forbid"``
    catches typos at the boundary so a misspelt ``cache_polciy`` cannot silently
    bypass a real cache directive.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Pre-flight controls
    dry_run: bool | None = None

    # Cost-aware controls
    confirmed: bool | None = None
    bypass_threshold_bytes: int | None = None
    bypass_threshold_pu: float | None = None

    # Lifecycle controls
    timeout_seconds: int | None = None
    priority: Literal["low", "normal", "high"] | None = None

    # Caching controls
    cache_policy: Literal["use", "skip", "refresh"] | None = None
    force_refresh: bool | None = None

    # Provenance controls
    suppress_provenance: bool | None = None
    citation_format: Literal["formal", "bibtex", "ris"] | None = None

    # Output controls
    output_format: Literal["filepath", "uri", "cache_key"] | None = None
    target_directory: str | None = None


class RequestEnvelope(BaseModel):
    """Four-field common envelope. ``request`` is opaque at this layer.

    Backend-specific parsing happens in ``schemas_<backend>.py`` after the
    orchestrator dispatches by ``backend``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: BackendId
    operation: OperationType
    request: dict[str, Any]
    options: RequestOptions = Field(default_factory=RequestOptions)
