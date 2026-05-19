"""User-facing MCP tool surface for the CMEMS backend.

Each tool has:
  - a Pydantic ``<ToolName>Input`` model (trimmed to user-facing fields),
  - an async wrapper function that calls the orchestrator,
  - a docstring that becomes the MCP tool description (T-019 finding).

``register_marine_tools(server, orchestrator)`` wires the must-have tools
to an MCP server instance. Optional tools (``marine_get_coordinates`` per
T-022 second half, ``marine_get_files`` per T-025 STRETCH) are registered
only when their backing backend method is implemented; both are absent in
Iteration 1.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from copernicus_mcp.data_model.schemas_cmems import (
    CmemsGetCoordinatesRequest,
    CmemsGetRequest,
    CmemsListFilesRequest,
    CmemsSubsetRequest,
)
from copernicus_mcp.workflow.orchestrator import WorkflowOrchestrator

_FORBID_FROZEN = ConfigDict(extra="forbid", frozen=True)


class MarineSearchDatasetsInput(BaseModel):
    """User-facing inputs for the search tool."""

    model_config = _FORBID_FROZEN

    keyword: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    time_range: tuple[str, str] | None = None
    # T-CMEMS-HIER-005: shortlist of product_ids — when present,
    # search routes through the hierarchical cards path so the
    # response carries enriched fields (domain/region/data_type/
    # variables_normalized/...). Usually obtained from a prior
    # ``marine_search_groups`` + ``marine_search_products`` round.
    product_ids: list[str] | None = None
    service_types: (
        list[
            Literal[
                "geoseries",
                "timeseries",
                "omi-arco",
                "static-arco",
                "platformseries",
            ]
        ]
        | None
    ) = None
    limit: int | None = Field(default=None, ge=1)
    # T-CMEMS-CAT-003a: opt-in hybrid search.
    # ``live=False`` (default) reads the bundled catalogue snapshot —
    # fast, no credentials, may be days/weeks stale. ``live=True``
    # calls ``copernicusmarine.describe()`` against the live Marine
    # Service — fresh, requires credentials, ~10 s round-trip. Pass
    # ``live=True`` when you need post-snapshot products or to
    # verify freshness.
    live: bool = Field(default=False)


class MarineSearchGroupsInput(BaseModel):
    """User-facing inputs for the hierarchical group-search tool
    (T-CMEMS-HIER-005)."""

    model_config = _FORBID_FROZEN

    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def _reject_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank or whitespace-only")
        return v


class MarineSearchProductsInput(BaseModel):
    """User-facing inputs for the hierarchical product-search tool
    (T-CMEMS-HIER-005)."""

    model_config = _FORBID_FROZEN

    group_ids: list[str] = Field(min_length=1)
    query: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


class MarineDescribeDatasetInput(BaseModel):
    """User-facing inputs for the describe tool."""

    model_config = _FORBID_FROZEN

    dataset_id: str = Field(min_length=1)

    @field_validator("dataset_id")
    @classmethod
    def _reject_blank(cls, v: str) -> str:
        # ``min_length=1`` accepts whitespace-only strings; the orchestrator
        # only rejects empty/None. Strip and require non-empty.
        if not v.strip():
            raise ValueError("dataset_id must not be blank or whitespace-only")
        return v


class ToolReturnedError(Exception):
    """Raised by a tool wrapper when the orchestrator returned ``{"error": ...}``.

    FastMCP catches this and re-raises as ``ToolError(f"Error executing
    tool {name}: {e}")`` — the wire response is ``isError=true`` with
    ``content[0].text`` carrying the prefixed message and
    ``structuredContent=None``. Plan T-031 step 3 explicitly permits this
    fallback: the SDK does not yet expose a structured ``data`` field for
    tool errors, so the canonical record is serialised to the message.

    **Client recovery** of ``error_class`` / ``recovery_action`` /
    ``next_action_hint``: strip the ``"Error executing tool <name>: "``
    prefix, then ``json.loads`` the remainder. The original record is
    also available on the exception instance as ``.record`` for
    in-process callers (tests, CLI). Revisit when the SDK adds
    structured tool-error data — at that point the wrapper can return a
    ``CallToolResult(isError=True, structuredContent=<record>)`` directly.
    """

    def __init__(self, record: dict[str, Any]) -> None:
        import json as _json

        self.record = record
        super().__init__(_json.dumps(record, default=str))


def _unwrap(envelope: dict[str, Any]) -> dict[str, Any]:
    """Translate orchestrator envelope to user-facing tool output.

    - ``{"result": <dict>}``           → return ``<dict>`` flat.
    - ``{"error": <record>}``          → raise ``ToolReturnedError`` so the
      MCP wire response is ``isError=true`` (T-031 step 3).
    - ``{"confirmation_required": ...}`` (and any other non-error/non-result
      envelope) → return as-is so the LLM client can prompt the user.
    """
    if "error" in envelope and isinstance(envelope["error"], dict):
        raise ToolReturnedError(envelope["error"])
    if "result" in envelope and isinstance(envelope["result"], dict):
        inner: dict[str, Any] = envelope["result"]
        return inner
    return envelope


class MarineEstimateSubsetInput(CmemsSubsetRequest):
    """User-facing inputs for the estimate tool — same shape as subset."""


class MarineSubsetDatasetInput(CmemsSubsetRequest):
    """User-facing inputs for the subset tool.

    T-039: ``async_mode=True`` opts the request into background download.
    The tool then returns ``{status: "running", request_id, ...}``
    immediately; poll completion via ``marine_check_status``.
    """

    async_mode: bool = Field(
        default=False,
        description=(
            "Submit as background task and return request_id immediately. "
            "For downloads expected to take more than ~5 minutes, prefer "
            "async_mode=True and poll via marine_check_status."
        ),
    )
    confirmed: bool = Field(
        default=False,
        description=(
            "Set to true to bypass the size confirmation gate. When false "
            "(default), submits estimated above the configured size "
            "threshold raise ``ConfirmationRequired`` so the caller (LLM "
            "agent or CLI) can prompt the user before paying the download "
            "cost. Mirrors the CDS submit ``confirmed`` flag."
        ),
    )


class MarineGetCoordinatesInput(CmemsGetCoordinatesRequest):
    """User-facing inputs for ``marine_get_coordinates`` (T-022 second half).

    Identical surface to ``CmemsGetCoordinatesRequest``; pinned as a
    distinct class for FastMCP's schema name.
    """


class MarineListFilesInput(CmemsListFilesRequest):
    """User-facing inputs for ``marine_list_files`` (T-CMEMS-GET-INDEX-004).

    Identical surface to ``CmemsListFilesRequest`` — pinned as a
    distinct class only so FastMCP's auto-derived JSON schema reflects
    the tool, not the backend schema.
    """


class MarineGetFilesInput(CmemsGetRequest):
    """User-facing inputs for ``marine_get_files`` (T-CMEMS-GET-006).

    Extends ``CmemsGetRequest`` with the top-level ``confirmed`` flag
    matching ``MarineSubsetDatasetInput``. Sparse-format datasets (the
    primary motivation for native-file retrieval) almost never expose
    a precise dry-run size, so the gate fires by default and callers
    must opt in via ``confirmed=true`` to download.
    """

    confirmed: bool = Field(
        default=False,
        description=(
            "Set to true to bypass the size confirmation gate. When false "
            "(default), get-files calls whose dry-run estimate is over the "
            "configured threshold OR whose epistemic status is "
            "``approximate`` raise ``ConfirmationRequired`` so the caller "
            "can prompt the user before paying the download cost. "
            "Mirrors marine_subset_dataset's ``confirmed`` flag."
        ),
    )


class MarineCheckStatusInput(BaseModel):
    """User-facing inputs for the check-status tool (T-039)."""

    model_config = _FORBID_FROZEN

    request_id: str = Field(min_length=1)


class MarineCancelSubsetInput(BaseModel):
    """User-facing inputs for the cancel tool (T-039)."""

    model_config = _FORBID_FROZEN

    request_id: str = Field(min_length=1)


async def marine_search_datasets(
    input: MarineSearchDatasetsInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """Search the CMEMS catalogue by keyword, bbox, or time range.

    Use this tool to discover dataset ids before describing or subsetting.

    **Workflow hints based on the returned ``service_types``:**

    - ``geoseries`` / ``timeseries`` (grid datasets) → use
      ``marine_subset_dataset`` for spatio-temporal subsets.
    - ``arco-platform-series`` / ``original-files`` (sparse / in-situ)
      → use ``marine_list_files`` to filter the dataset's file index by
      bbox/time/variables, THEN ``marine_get_files(file_list=[...])`` to
      download only the matching files. Going straight to
      ``marine_get_files`` without ``marine_list_files`` downloads the
      whole bundle (often multi-GB), which is rarely what you want.

    Inputs:
      - keyword: free-text match against dataset titles / descriptions.
      - bbox: (min_lon, min_lat, max_lon, max_lat); restricts to datasets
        whose coverage intersects this box.
      - time_range: (start, end) ISO 8601 UTC; restricts to datasets whose
        temporal coverage overlaps.
      - service_types: filter by service kind (geoseries, timeseries, …).
      - limit: max dataset records to return.
      - live (default false): when true, hit the Copernicus Marine
        Service live (requires CMEMS credentials, ~10 s) instead of
        reading the bundled catalogue snapshot. Use live=true to see
        products added after the last snapshot refresh, otherwise
        leave at the default for fast credless discovery.

    Outputs (envelope):
      - datasets: list of slim records (dataset_id, title, product_id,
        variables, spatial_extent, ...).
      - total_count: PRE-slice match count (may exceed len(datasets)
        when limit is applied).
      - mode: "offline" (snapshot) or "live" (live SDK call).
      - catalogue_fetched_at: ISO timestamp when mode="offline";
        null when mode="live".
    """
    out: dict[str, Any] = await orchestrator.run(
        backend="cmems",
        operation="search",
        params=input.model_dump(exclude_none=True),
    )
    return _unwrap(out)


async def marine_search_groups(
    input: MarineSearchGroupsInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """Shortlist CMEMS routing groups for a free-text query.

    First step in the hierarchical search pipeline (groups → products
    → datasets). Each group bundles related products by region,
    domain, and intent (e.g. ``physics-mediterranean-state``,
    ``ocean-acidification-monitoring``).

    Inputs:
      - query: free-text — what the user is looking for.
      - top_k: max groups to return (default 5).

    Outputs (envelope):
      - selected: ranked list of {group_id, group_title, summary,
        product_ids, score}.
      - rejected: groups deselected by exclude phrases (with reason).
      - reason: one-line human explanation of the top match.
      - confidence: "high" | "medium" | "low".
      - fallback_available: true when confidence is low; caller
        should consider ``marine_search_datasets`` (flat) instead.
    """
    out: dict[str, Any] = await orchestrator.run(
        backend="cmems",
        operation="search_groups",
        params=input.model_dump(exclude_none=True),
    )
    return _unwrap(out)


async def marine_search_products(
    input: MarineSearchProductsInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """Filter CMEMS products by group membership, optionally refining
    by keyword.

    Second step in the hierarchical search pipeline. Takes the
    ``group_ids`` from ``marine_search_groups`` and returns the
    candidate products with their summaries — feed the resulting
    ``product_ids`` to ``marine_search_datasets`` for the final
    dataset shortlist.

    Inputs:
      - group_ids: list of group_id strings (from search_groups).
      - query: optional keyword to re-rank within the group set.
      - top_k: max products to return (default 20).

    Outputs (envelope): same shape as ``marine_search_groups``.
    """
    out: dict[str, Any] = await orchestrator.run(
        backend="cmems",
        operation="search_products",
        params=input.model_dump(exclude_none=True),
    )
    return _unwrap(out)


async def marine_describe_dataset(
    input: MarineDescribeDatasetInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """Return full metadata for a single CMEMS dataset.

    Use this tool to inspect variables, axes, and services before estimating
    or submitting a subset.

    Inputs:
      - dataset_id: id obtained from marine_search_datasets.

    Outputs:
      - {dataset_id, variables, axes, services, ...}.
    """
    # Orchestrator dispatch for ``describe`` reads ``params["identifier"]``
    # (positional-arg shape — see orchestrator._dispatch). Tool input field
    # is named ``dataset_id`` for user clarity; map it here.
    out: dict[str, Any] = await orchestrator.run(
        backend="cmems",
        operation="describe",
        params={"identifier": input.dataset_id},
    )
    return _unwrap(out)


async def marine_estimate_subset(
    input: MarineEstimateSubsetInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """Estimate the size and cost of a subset request before submitting it.

    Use this tool to preview byte-size, variable list, and confirmation
    requirements without actually downloading data.

    Inputs: same shape as marine_subset_dataset.

    Outputs:
      - estimated_size_bytes, variables, advisory_message, ...
    """
    out: dict[str, Any] = await orchestrator.run(
        backend="cmems",
        operation="estimate",
        params=input.model_dump(exclude_none=True),
    )
    return _unwrap(out)


async def marine_subset_dataset(
    input: MarineSubsetDatasetInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """Subset a CMEMS dataset and return a file descriptor.

    This tool returns a descriptor (filepath + metadata + provenance), not
    the data itself. Read the file via your filesystem; the path is stable
    until cache eviction.

    Inputs:
      - dataset_id, variables, bbox, depth range, time range, file_format.
      - async_mode (bool, default false): submit as background task and
        return request_id immediately. Poll via marine_check_status.
      - confirmed (bool, default false): bypass the size confirmation gate.
        First call returns ``confirmation_required=True`` with the estimate;
        call again with confirmed=true to proceed.

    Outputs (sync mode):
      - {filepath, uri, metadata, provenance, cache_hit?}.

    Outputs (async_mode=True):
      - {status: "running", request_id, cache_key,
        result: {uri: "copernicus://jobs/<request_id>"}}.
    """
    payload = input.model_dump(exclude_none=True)
    options: dict[str, Any] = {}
    # ``async_mode`` and ``confirmed`` are options the orchestrator forwards
    # under ``__options``, not Pydantic-validated fields on the wire request —
    # keep the request envelope clean. ``confirmed`` mirrors the CDS submit
    # flow so the LLM client can satisfy the gate's
    # ``next_action: "call ... with options.confirmed=true"`` instruction.
    async_mode = bool(payload.pop("async_mode", False))
    confirmed = bool(payload.pop("confirmed", False))
    if async_mode:
        options["async_mode"] = True
    if confirmed:
        options["confirmed"] = True
    out: dict[str, Any] = await orchestrator.run(
        backend="cmems",
        operation="submit",
        params=payload,
        options=options or None,
    )
    return _unwrap(out)


async def marine_get_coordinates(
    input: MarineGetCoordinatesInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """Return the dataset's coordinate axes (lon / lat / depth / time).

    Use this BEFORE ``marine_subset_dataset`` or ``marine_estimate_subset``
    when you need to know the dataset's real extent — e.g. its actual
    depth levels, its time stride, or whether your bbox is even inside
    the dataset's coverage. The returned axes mirror what
    ``copernicusmarine.describe(...)`` exposes, but with **large axes
    summarised** so the response stays compact:

      - ``time`` returns as a full list when ≤5000 entries, else as a
        summary ``{start, end, count, stride_seconds}``;
      - ``longitude`` / ``latitude`` / ``depth`` return as a full list
        when ≤10000 entries, else as a summary ``{start, end, count,
        stride}``.

    Inputs:
      - dataset_id: required CMEMS dataset id (e.g. from
        ``marine_search_datasets``).
      - dataset_version: optional version label (e.g. ``"202411"``);
        defaults to the latest.
      - service: optional service id (``geoseries``, ``timeseries``, …)
        to disambiguate when a dataset version exposes multiple services.

    Output: a dict keyed by axis name. Always emits the four canonical
    spatio-temporal axes: ``"longitude"``, ``"latitude"``, ``"depth"``,
    ``"time"`` (depth comes back as an empty list ``[]`` for surface-only
    datasets). Each value is either a list of axis values (short axis)
    or a summary dict (long axis) per the limits above.
    """
    out: dict[str, Any] = await orchestrator.run(
        backend="cmems",
        operation="get_coordinates",
        params=input.model_dump(exclude_none=True),
    )
    return _unwrap(out)


async def marine_list_files(
    input: MarineListFilesInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """List CMEMS native files for a dataset, filtered by bbox / time / variables / platform.

    Use this tool BEFORE ``marine_get_files`` when you have a sparse
    dataset (CORA, EasyCORA, INSITU-BGC) and want to download a precise
    subset rather than the full multi-GB bundle. The first call per
    dataset fetches the index (one SDK round-trip); subsequent calls
    read a local Parquet cache (offline). The returned ``files`` list
    feeds directly into ``marine_get_files(file_list=[...])``.

    Inputs:
      - dataset_id: required. Must be a CMEMS dataset id.
      - bbox: (min_lon, min_lat, max_lon, max_lat). Antimeridian-crossing
        bboxes (min_lon > max_lon) are accepted — they wrap east of
        ``min_lon`` AND west of ``max_lon``. The resulting file_list is
        usable with ``marine_get_files`` but NOT with
        ``marine_subset_dataset``.
      - time_range: (start, end) ISO 8601 UTC. Inverted ranges rejected.
      - variables: optional whitelist of variable names; rows with
        unknown variables (``None`` in the index) are kept.
      - platform_types: optional whitelist (PF, CT, MO, DB, ...); rows
        with unknown platform are dropped.
      - limit: optional cap on the returned files (sorted by file_path
        ASC for determinism). The envelope surfaces
        ``matched_count_uncapped`` so callers see the true match count.

    Outputs (envelope):
      - files: list of per-file records (file_path, bbox, time range,
        platform_type, variables, size_bytes) ready for marine_get_files.
      - matched_count, matched_count_uncapped, truncated: caller-visible
        slicing metadata.
      - total_count_in_index: pre-filter row count for the dataset.
      - total_size_bytes_known, rows_with_unknown_size: size aggregation
        distinguishing "we know N bytes" from "M rows had no size info".
      - filters_applied: echoes the user's filter axes.
      - index_fetched_at: when the local Parquet cache was last written.
      - mode: "offline" (cache hit) or "fresh" (SDK round-trip).
    """
    out: dict[str, Any] = await orchestrator.run(
        backend="cmems",
        operation="list_files",
        params=input.model_dump(exclude_none=True),
    )
    return _unwrap(out)


async def marine_get_files(
    input: MarineGetFilesInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """Download native CMEMS files (no Zarr slicing).

    Use this tool for datasets whose service is ``original-files`` or
    ``arco-platform-series`` (sparse / in-situ observations) — these
    don't support ``marine_subset_dataset``. The result is a list of
    per-file descriptors (filepath + metadata + provenance), one per
    file the toolbox produced. the project conventions invariant 1: never serves
    bytes inline.

    **Tip for filtered subsets:** if you want only the files matching
    a bbox / time range / variable / platform filter, call
    ``marine_list_files`` FIRST with those filters and pass the
    returned ``file_path`` values as this tool's ``file_list``. This
    avoids downloading a multi-GB bundle when you only need a few
    files (e.g. "CORA Mediterranean 2010-2015").

    Inputs:
      - dataset_id: required.
      - dataset_version, dataset_part: optional, pin a non-default
        version / part.
      - filter / regex / file_list: at most one selection mechanism.
        Omit to download whatever the toolbox defaults to. ``filter``
        is a glob (e.g. ``*1990*``), ``regex`` is a Python regex,
        ``file_list`` is an explicit list of paths.
      - sync / skip_existing / overwrite: forwarded to the SDK
        unchanged.
      - confirmed (default false): bypass the size confirmation gate.

    Outputs:
      - {status: "successful", cache_hit, request_id, cache_key,
        mode: "offline", result: {files: [{filepath, uri, metadata,
        provenance}, ...], provenance: {reference?}}}.
      - First call with an unknown / approximate estimate returns
        ``confirmation_required=True``; re-call with ``confirmed=true``
        to download.
    """
    payload = input.model_dump(exclude_none=True)
    options: dict[str, Any] = {}
    confirmed = bool(payload.pop("confirmed", False))
    if confirmed:
        options["confirmed"] = True
    out: dict[str, Any] = await orchestrator.run(
        backend="cmems",
        operation="get",
        params=payload,
        options=options or None,
    )
    return _unwrap(out)


async def marine_check_status(
    input: MarineCheckStatusInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """Look up the status of an in-flight or completed CMEMS subset.

    Use this after a ``marine_subset_dataset`` call with ``async_mode=True``.

    **Polling guidance — do NOT sleep between calls.** The Claude Code
    harness blocks standalone ``sleep N`` precisely to discourage
    naive sleep-then-retry loops; workarounds (``until [...]; do
    sleep 5; done``) burn context budget on every wake. Instead:

    - **User-paced re-call** (recommended): return ``status=queued|
      running`` to the user, let them decide when to ask "ready yet?"
      — each call is a fresh tool invocation, no agent-side waiting.
    - **CLI ``copernicus-mcp marine wait <request_id>``** for offline
      polling — runs server-side in the CLI process, doesn't tie up
      agent context.

    Typical CMEMS async-subset latency: tens of seconds to a few
    minutes; large multi-GB requests can take 10+ minutes.

    Inputs:
      - request_id: returned by the original async submit.

    Outputs:
      - {status, request_id, submitted_at, updated_at, cache_key,
        error_details}. Status is one of queued/running/successful/
        failed/cancelled.
    """
    out: dict[str, Any] = await orchestrator.run(
        backend="cmems",
        operation="poll",
        params={"request_id": input.request_id},
    )
    return _unwrap(out)


async def marine_cancel_subset(
    input: MarineCancelSubsetInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """Cancel an in-flight CMEMS subset request.

    If a background task is running for the request_id, it is interrupted
    and the workflow row marked ``cancelled``. If the request has already
    settled (successful/failed/cancelled), the call is a no-op and reports
    the current status.

    Inputs:
      - request_id: returned by the original async submit.

    Outputs:
      - {cancelled: bool, request_id, status}.
    """
    out: dict[str, Any] = await orchestrator.run(
        backend="cmems",
        operation="cancel",
        params={"request_id": input.request_id},
    )
    return _unwrap(out)


def register_marine_tools(
    server: Any,
    *,
    orchestrator: WorkflowOrchestrator,
) -> None:
    """Register CMEMS tools on a FastMCP-compatible server.

    Each registered wrapper has a concrete Pydantic-model input annotation
    so FastMCP auto-derives a rich JSON schema for the LLM client (T-019
    finding: ``input: Any`` would yield an empty schema and the tool would
    receive a raw dict at call time, breaking ``model_dump()``).

    Tool name and description are set explicitly via the decorator's
    ``name=``/``description=`` kwargs so neither depends on the local
    function name (which has to be unique per ``def``).

    T-CMEMS-GET-006: ``marine_get_files`` is registered alongside the
    must-have tools. ``marine_get_coordinates`` remains out of scope
    for Iteration 1.
    """

    @server.tool(  # type: ignore[untyped-decorator]
        name=marine_search_datasets.__name__,
        description=marine_search_datasets.__doc__,
    )
    async def _search(input: MarineSearchDatasetsInput) -> dict[str, Any]:
        return await marine_search_datasets(input, orchestrator=orchestrator)

    @server.tool(  # type: ignore[untyped-decorator]
        name=marine_search_groups.__name__,
        description=marine_search_groups.__doc__,
    )
    async def _search_groups(
        input: MarineSearchGroupsInput,
    ) -> dict[str, Any]:
        return await marine_search_groups(input, orchestrator=orchestrator)

    @server.tool(  # type: ignore[untyped-decorator]
        name=marine_search_products.__name__,
        description=marine_search_products.__doc__,
    )
    async def _search_products(
        input: MarineSearchProductsInput,
    ) -> dict[str, Any]:
        return await marine_search_products(input, orchestrator=orchestrator)

    @server.tool(  # type: ignore[untyped-decorator]
        name=marine_describe_dataset.__name__,
        description=marine_describe_dataset.__doc__,
    )
    async def _describe(input: MarineDescribeDatasetInput) -> dict[str, Any]:
        return await marine_describe_dataset(input, orchestrator=orchestrator)

    @server.tool(  # type: ignore[untyped-decorator]
        name=marine_estimate_subset.__name__,
        description=marine_estimate_subset.__doc__,
    )
    async def _estimate(input: MarineEstimateSubsetInput) -> dict[str, Any]:
        return await marine_estimate_subset(input, orchestrator=orchestrator)

    @server.tool(  # type: ignore[untyped-decorator]
        name=marine_subset_dataset.__name__,
        description=marine_subset_dataset.__doc__,
    )
    async def _subset(input: MarineSubsetDatasetInput) -> dict[str, Any]:
        return await marine_subset_dataset(input, orchestrator=orchestrator)

    @server.tool(  # type: ignore[untyped-decorator]
        name=marine_get_coordinates.__name__,
        description=marine_get_coordinates.__doc__,
    )
    async def _get_coordinates(input: MarineGetCoordinatesInput) -> dict[str, Any]:
        return await marine_get_coordinates(input, orchestrator=orchestrator)

    @server.tool(  # type: ignore[untyped-decorator]
        name=marine_list_files.__name__,
        description=marine_list_files.__doc__,
    )
    async def _list_files(input: MarineListFilesInput) -> dict[str, Any]:
        return await marine_list_files(input, orchestrator=orchestrator)

    @server.tool(  # type: ignore[untyped-decorator]
        name=marine_get_files.__name__,
        description=marine_get_files.__doc__,
    )
    async def _get_files(input: MarineGetFilesInput) -> dict[str, Any]:
        return await marine_get_files(input, orchestrator=orchestrator)

    @server.tool(  # type: ignore[untyped-decorator]
        name=marine_check_status.__name__,
        description=marine_check_status.__doc__,
    )
    async def _check_status(input: MarineCheckStatusInput) -> dict[str, Any]:
        return await marine_check_status(input, orchestrator=orchestrator)

    @server.tool(  # type: ignore[untyped-decorator]
        name=marine_cancel_subset.__name__,
        description=marine_cancel_subset.__doc__,
    )
    async def _cancel(input: MarineCancelSubsetInput) -> dict[str, Any]:
        return await marine_cancel_subset(input, orchestrator=orchestrator)
