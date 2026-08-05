"""User-facing MCP tool surface for the CDS backend (T-CDS-007).

Mirrors the CMEMS ``marine_*`` tool surface (see ``backends/cmems/tools.py``)
but with seven tools — search, describe, estimate, submit, check_status,
download (fetch_result), cancel — to match the inherently async CDS
workflow (research §6.5; T-CDS-005). The CMEMS surface has six because
its sync mode merges submit + download into one tool; CDS is queue-backed
so download is always a separate step that resolves the cached file
descriptor for an already-successful workflow row.

Each tool returns a flat user-facing dict; ``ToolReturnedError`` carries
canonical ``ErrorRecord`` content for ``isError=true`` MCP wire responses.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from copernicus_mcp.data_model.schemas_cds import (
    CdsApplyConstraintsRequest,
    CdsRetrieveRequest,
    CdsSearchGroupsRequest,
    CdsSearchRequest,
)
from copernicus_mcp.workflow.orchestrator import WorkflowOrchestrator

_FORBID_FROZEN = ConfigDict(extra="forbid", frozen=True)

# Sentinel passed to the orchestrator's ``fetch`` dispatch (which
# requires a non-empty ``target``) when the caller didn't supply one.
# The CDS backend ignores ``target`` entirely. Round-2 cr LOW (L2):
# kept as a single source of truth so the CLI helper and the tool
# wrapper cannot drift.
_TARGET_UNUSED_SENTINEL = "__cds_unused__"


class CdsSearchDatasetsInput(CdsSearchRequest):
    """User-facing inputs for the CDS catalogue search tool."""


class CdsSearchGroupsInput(CdsSearchGroupsRequest):
    """User-facing inputs for the hierarchical group-search tool
    (T-CDS-021 PR-2)."""


class CdsDescribeDatasetInput(BaseModel):
    """User-facing inputs for the describe tool."""

    model_config = _FORBID_FROZEN

    dataset_id: str = Field(min_length=1)

    @field_validator("dataset_id")
    @classmethod
    def _reject_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("dataset_id must not be blank or whitespace-only")
        return v


class CdsApplyConstraintsInput(CdsApplyConstraintsRequest):
    """User-facing inputs for ``cds_apply_constraints`` (T-CDS-016)."""


class CdsEstimateRequestInput(CdsRetrieveRequest):
    """User-facing inputs for the estimate tool — same shape as submit."""


class CdsSubmitRequestInput(CdsRetrieveRequest):
    """User-facing inputs for the CDS submit tool.

    CDS is async-by-design (research §6.5): every submit goes through the
    server queue. The tool returns ``{status: "queued", request_id, ...}``
    immediately; poll completion via ``cds_check_request_status`` and
    download via ``cds_download_request_result``.
    """

    confirmed: bool = Field(
        default=False,
        description=(
            "Set to true to bypass the size/queue-tier confirmation gate. "
            "When false (default), submits estimated above the configured "
            "size threshold or with queue tier ``medium``/``heavy`` raise "
            "``ConfirmationRequired`` so the caller (LLM agent or CLI) can "
            "prompt the user before paying the queue / download cost. Also "
            "accepts the auto-chunking proposal at the default ``year`` "
            "granularity when no ``chunk_by`` is given."
        ),
    )
    chunk_by: Literal["year", "month", "day"] | None = Field(
        default=None,
        description=(
            "Calendar granularity for splitting a request that exceeds the "
            "dataset's server-side cost limit. When the first submit returns a "
            "``cost_limit_requires_chunking`` proposal, re-submit with chunk_by "
            "set to year, month, or day (or confirmed=true to accept the "
            "suggested year). The parts run as one logical workflow under a "
            "single parent request_id; the result is a multi-file set, one file "
            "per chunk (the MCP never stitches them)."
        ),
    )
    auto_chunk: bool = Field(
        default=True,
        description=(
            "Set false to disable auto-chunking for this request: an over-limit "
            "request then raises the manual-split ValidationError instead of "
            "proposing a split."
        ),
    )
    force_refresh: bool = Field(
        default=False,
        description=(
            "Set true to bypass the cache/idempotency check and re-run from "
            "scratch — including re-running every chunk of a previously "
            "completed chunked parent (e.g. to repopulate evicted chunk files)."
        ),
    )
    confirm_large_fanout: bool = Field(
        default=False,
        description=(
            "Second, deliberate ack for a very large auto-chunk split. When a "
            "split would launch more CDS jobs at once than the reconfirm "
            "threshold, confirmed=true alone is NOT enough — also set this true. "
            "Surfaced by the 'auto_chunk_job_count_large' confirmation."
        ),
    )


class CdsCheckRequestStatusInput(BaseModel):
    """User-facing inputs for the check-status tool. Exactly ONE of
    ``request_id`` (single) or ``request_ids`` (batch, T-CDS-OPS-002)."""

    model_config = _FORBID_FROZEN

    request_id: str | None = Field(default=None, min_length=1)
    # Local review round (MEDIUM): ``min_length`` on the list constrains the
    # LIST, so each element needs its own floor — ``[""]`` must fail exactly
    # like ``request_id=""`` does in single mode.
    request_ids: list[Annotated[str, StringConstraints(min_length=1)]] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Batch mode: poll several requests in one call (bounded "
            "concurrency). Returns {results: [...], count} preserving input "
            "order, with per-id errors inline — one bad id does not fail the "
            "batch."
        ),
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> CdsCheckRequestStatusInput:
        if (self.request_id is None) == (self.request_ids is None):
            raise ValueError(
                "provide exactly one of request_id or request_ids"
            )
        return self


class CdsDownloadRequestResultInput(BaseModel):
    """User-facing inputs for the download tool.

    Round-1 cr M1: ``target`` was previously required, but the backend
    ignores it (the file is already at the canonical cache location per
    the project conventions invariant 1). Requiring an ignored field bait the LLM
    into inventing plausible-looking paths it would never use; making
    it optional removes the trap.
    """

    model_config = _FORBID_FROZEN

    request_id: str = Field(min_length=1)
    target: str | None = Field(
        default=None,
        description=(
            "Optional. Currently IGNORED — the backend always returns "
            "a descriptor pointing at the canonical cache location "
            "(the project conventions invariant 1). Reserved for future use."
        ),
    )


class CdsCancelRequestInput(BaseModel):
    """User-facing inputs for the cancel tool."""

    model_config = _FORBID_FROZEN

    request_id: str = Field(min_length=1)


class CdsListLicencesInput(BaseModel):
    """User-facing inputs for ``cds_list_licences`` (T-CDS-LICENCE-001)."""

    model_config = _FORBID_FROZEN

    dataset_id: str | None = Field(
        default=None,
        description=(
            "Optional dataset id to route the call to the right store "
            "(CDS / ADS / EWDS) — licence ids are per-store."
        ),
    )


class CdsAcceptLicenceInput(BaseModel):
    """User-facing inputs for ``cds_accept_licence`` (T-CDS-LICENCE-001)."""

    model_config = _FORBID_FROZEN

    licence_id: str = Field(min_length=1, description="Licence id from cds_list_licences.")
    # Local review round (MEDIUM): strict — plain ``int`` would coerce
    # ``revision=True`` to 1 before the backend's bool guard could see it;
    # negative revisions fail fast here instead of on the live API.
    revision: int = Field(
        strict=True, ge=0, description="Licence revision from cds_list_licences."
    )
    dataset_id: str | None = Field(
        default=None,
        description="Optional dataset id for store routing (CDS / ADS / EWDS).",
    )


class ToolReturnedError(Exception):
    """Tool wrapper raises this when the orchestrator returns
    ``{"error": <record>}``. Mirrors the CMEMS pattern at
    ``backends/cmems/tools.py:ToolReturnedError``: FastMCP catches this
    and surfaces the canonical record as the wire ``isError=true``
    payload with the JSON-serialised record in ``content[0].text``.
    """

    def __init__(self, record: dict[str, Any]) -> None:
        import json as _json

        self.record = record
        super().__init__(_json.dumps(record, default=str))


def _unwrap(envelope: dict[str, Any]) -> dict[str, Any]:
    """Translate orchestrator envelope to user-facing tool output.
    Same semantics as the CMEMS tool's ``_unwrap``."""
    if "error" in envelope and isinstance(envelope["error"], dict):
        raise ToolReturnedError(envelope["error"])
    if "result" in envelope and isinstance(envelope["result"], dict):
        inner: dict[str, Any] = envelope["result"]
        return inner
    return envelope


async def cds_search_datasets(
    input: CdsSearchDatasetsInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """Search the bundled CDS / ADS / EWDS catalogue snapshot.

    Use this tool to discover candidate dataset ids before describing
    or submitting a request. Returns a slim record per dataset; call
    ``cds_describe_dataset`` for the full STAC metadata.

    Inputs (all optional, AND-combined):
      - keyword: free-text match. The query is split on whitespace and a
        dataset is kept only if EVERY word matches (word order is
        irrelevant) — each word either hits id / title / description /
        keywords / summaries at a word boundary OR names a variable in the
        dataset's constraints, so ``"2m temperature"`` finds ERA5 even
        though that phrase appears nowhere in the prose. Results are
        relevance-ranked (exact-phrase hits first), so a small ``limit``
        keeps the most on-topic datasets. Prefer 1-3 content words;
        for a specific variable, ``variable=`` is more precise.
      - store: filter (cds / ads / ewds).
      - bbox: ``(west, south, east, north)`` WGS84 degrees. Records
        whose spatial extent intersects this bbox are kept. Antimeridian-
        crossing queries (west > east) are rejected — split into two.
      - time_range: ``(start_iso, end_iso)``. Records whose temporal
        extent overlaps the window are kept; bare dates / naive ISO
        strings are interpreted as UTC.
      - variable: substring (case-insensitive) match across keywords,
        summaries, title/description, AND the bundled constraints'
        ``variable`` enum (so ``"temperature"`` finds ERA5 even though
        STAC keywords don't name specific variables).
      - domain: exact match against the dataset's
        ``"Variable domain: X"`` keyword (e.g. ``"Ocean (physics)"``).
        Use ``cds_search_groups`` first to discover valid domains.
      - category: exact match against the dataset's ID family prefix
        (e.g. ``"reanalysis"``, ``"satellite"``, ``"cams"``).
      - limit: max records to return.

    Outputs:
      - {datasets: [{id, title, description, keywords, store}, ...],
         total_count}.
    """
    out: dict[str, Any] = await orchestrator.run(
        backend="cds",
        operation="search",
        params=input.model_dump(exclude_none=True),
    )
    return _unwrap(out)


async def cds_search_groups(
    input: CdsSearchGroupsInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """T-CDS-021 PR-2: hierarchical discovery over CDS / ADS / EWDS.

    **Use this BEFORE ``cds_search_datasets`` when you have a vague
    free-text intent.** Returns a ranked list of ``(domain, category)``
    groups (e.g. ``Atmosphere (surface) / reanalysis``) with sample
    dataset titles + counts. Pick a group, then call
    ``cds_search_datasets(domain=..., category=...)`` for the narrowed
    candidate list. Mirrors CMEMS's ``marine_search_groups`` UX.

    Without ``query``: every populated group sorted by ``dataset_count``
    descending (~40 groups across 164 datasets in the current snapshot).

    With ``query``: scored by substring matches against domain, category,
    and member-dataset metadata. ``top_k`` caps the ranked list.

    Bundled-snapshot only — NEVER hits the network.

    Inputs:
      - query (optional): free-text intent like
        ``"atmosphere temperature reanalysis"``.
      - top_k (optional): cap returned group count after ranking.

    Outputs:
      - ``{groups: [{id, domain, category, dataset_count, sample_titles,
        score?}, ...], total_count}``. ``id`` is a stable
        ``"<domain-slug>|<category>"`` chaining key for scripting.
    """
    out: dict[str, Any] = await orchestrator.run(
        backend="cds",
        operation="search_groups",
        params=input.model_dump(exclude_none=True),
    )
    return _unwrap(out)


async def cds_describe_dataset(
    input: CdsDescribeDatasetInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """Return the full STAC metadata for a single CDS / ADS / EWDS dataset.

    **Use this BEFORE composing a `cds_submit_request`.** The output
    includes an ``available_inputs`` field listing every parameter the
    dataset accepts and the valid values for each (e.g.
    ``data_format: [netcdf, grib]``, ``download_format: [zip,
    unarchived]``, the canonical ``variable`` enum). Compose your
    submit request using only these field names and values; the legacy
    ``format: ...`` key was deprecated by the new CDS processes
    engine and silently rejected by the server.

    **Caveat — ``available_inputs`` is a SNAPSHOT.** It ships in the
    bundled catalogue refreshed manually (typically every few weeks).
    CDS occasionally rotates field names / adds new constraints. If a
    submit composed from this snapshot fails with `remote_job_failed`,
    call ``cds_apply_constraints(dataset_id, inputs={})`` for the LIVE
    server-side valid values verbatim — they are the canonical source
    of truth. Use ``cds_apply_constraints`` with a PARTIAL request to
    progressively narrow the valid combinations — but note its caveats:
    for some datasets the constraints response is a flat, non-narrowing
    union, so the real submit is the decisive validator.

    Use this after ``cds_search_datasets`` to inspect required fields,
    licence, and constraints before submitting a request. The lookup is
    cross-store: the dataset_id alone determines which bundled snapshot
    is consulted.

    Inputs:
      - dataset_id: id obtained from ``cds_search_datasets``.

    Outputs:
      - the STAC record augmented with a ``store`` field and (when the
        bundled constraints snapshot has data for this dataset) an
        ``available_inputs`` dict.
    """
    out: dict[str, Any] = await orchestrator.run(
        backend="cds",
        operation="describe",
        params={"identifier": input.dataset_id},
    )
    return _unwrap(out)


async def cds_apply_constraints(
    input: CdsApplyConstraintsInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """Server-side narrowing: given a PARTIAL CDS / ADS / EWDS request,
    return the remaining valid values for unfilled fields.

    **Use this to compose a submit request step-by-step** instead of
    guessing field names / enum values. The bundled
    ``cds_describe_dataset → available_inputs`` shows the static
    top-level choices; this tool returns the LIVE narrowing as you
    fill the partial request. Particularly useful when:

    - You don't know whether a field accepts the legacy ``format`` or
      modern ``data_format`` / ``download_format`` keys (this endpoint
      uses the modern keys exclusively).
    - You picked a ``variable`` and want to know whether it accepts a
      time range (auxiliary time-invariant variables like EFAS
      ``elevation`` will NOT return ``hyear/hmonth/hday/time`` in the
      response — that's the signal to drop those fields).
    - You want to confirm a dataset's required additional fields (e.g.
      EFAS v5.0 needs ``hydrological_model: [lisflood]``).

    **Caveats (observed on live datasets — read before trusting the
    response):**

    - Some datasets return a NON-NARROWING flat union: ``valid_remaining``
      echoes the full vocabulary no matter what you pin, and values
      coupled to your selection may be silently omitted. Per-field
      membership does NOT guarantee the combination is retrievable.
    - Cross-field couplings (e.g. a spatial grid that only exists for
      certain temporal aggregations) surface only when ALL related axes
      are pinned TOGETHER. When probing whether a combination exists, pin
      variable + product type + spatial + temporal + version in the same
      call — narrowing one field at a time can pass on every step and
      still fail at submit.
    - This live endpoint and the bundled ``cds_describe_dataset``
      snapshot can be different vintages; for vocabulary this endpoint
      wins. Either way the REAL submit is the decisive validator — treat
      a passing constraints probe as advisory, not as proof.

    Inputs:
      - dataset_id: CDS / ADS / EWDS dataset id.
      - inputs: partial request dict. Empty dict ``{}`` returns top-level
        valid values for every field. Each subsequent narrowing call
        appends to inputs.

    Outputs:
      - ``dataset_id``, ``store``, ``inputs_provided`` (echoed for
        traceability), and ``valid_remaining`` — a dict mapping each
        remaining field to its valid values given the partial selection.
    """
    out: dict[str, Any] = await orchestrator.run(
        backend="cds",
        operation="apply_constraints",
        params=input.model_dump(),
    )
    return _unwrap(out)


async def cds_estimate_request(
    input: CdsEstimateRequestInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """Size + queue estimate for a CDS retrieve request.

    Combines the CDS ``/costing`` pre-flight (the server's own cost units
    and the dataset's per-request cost limit) with a calibrated/curated
    bytes-per-unit factor. Use this to gauge submission cost before paying
    the queue.

    **The byte-size estimate is APPROXIMATE and can be wrong by a large
    factor — do NOT rely on it for hard limits, quotas, billing, or disk
    planning.** Treat it as a rough order-of-magnitude only. The ``cost``
    units are exact; only the byte size is uncertain. Every response repeats
    this disclaimer in the ``size_estimate_caveat`` field.

    Inputs: same shape as ``cds_submit_request``.

    Outputs:
      - ``estimated_size_bytes`` / ``estimated_size_human``: **may be ``null``**
        — for whole-file products whose size cannot be known from request
        shape, the estimate is honestly reported as unknown rather than an
        invented number.
      - ``epistemic_status``: one of ``calibrated`` (from your own prior
        downloads of this shape — most trustworthy), ``approximate`` (a
        bundled-seed estimate, signature-matched — rough, can be off by a large
        factor), or ``unknown`` (no calibration yet, or a whole-file product
        whose size is unknowable until the first retrieval — no size is shown).
      - ``cost``: ``{units, limit, exceeds_limit, source}`` from the costing
        endpoint, or ``null`` if it was unreachable. ``exceeds_limit=true``
        means the server will reject the request — narrow it (or, once
        auto-chunking ships, it will be split automatically).
      - ``fields_count``, ``queue_latency_tier``, ``advisory_message``.
      - ``size_estimate_caveat``: always present — the plain-language warning
        that the byte size is approximate and must not be relied on.
    """
    out: dict[str, Any] = await orchestrator.run(
        backend="cds",
        operation="estimate",
        params=input.model_dump(exclude_none=True),
    )
    return _unwrap(out)


async def cds_submit_request(
    input: CdsSubmitRequestInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """Submit a CDS / ADS / EWDS retrieve request.

    CDS is queue-backed: the call returns immediately after the server
    acknowledges the request. Track progress via
    ``cds_check_request_status`` and download via
    ``cds_download_request_result`` once status reaches ``successful``.

    A T&C-not-accepted server response is surfaced as the canonical
    ``TermsNotAcceptedError`` with ``recovery_url`` pointing at the
    licence page (T-CDS-006). Open the URL, accept the licence, and
    re-submit.

    **Unknown input keys are rejected up front** (the server would
    silently ignore them — wrong selection, or an empty-log failure
    minutes later). Compose requests from ``cds_describe_dataset``'s
    ``available_inputs``; ``__options.skip_input_validation=true``
    bypasses the check if the bundled snapshot is stale.

    **Multi-model requests on ``projections-cmip6`` and
    ``projections-cordex-domains-single-levels`` always fan out** into
    one part per model (those datasets execute ONE model per request
    and silently deliver only the first of a list) — the response is
    then a chunked parent, and each downloaded part is verified to
    contain its requested model before caching.

    Inputs:
      - dataset_id: id from the catalogue (e.g.
        ``reanalysis-era5-pressure-levels``).
      - inputs: cdsapi-shaped request dict (variable, year, month,
        day, time, area, pressure_level, ...).

        For the ``reanalysis-*`` families (ERA5, ERA5-Land, CERRA,
        CARRA, ...) ``product_type`` is REQUIRED (e.g. ``"reanalysis"``;
        see ``cds_describe_dataset`` ``available_inputs`` for the valid
        values). Omitting it is rejected before submit — and on the live
        server a multi-month request without it fails server-side with
        ``Duplicate value for month`` after a long queue.

        WARNING — ``area`` ordering: CDS uses
        ``[north, west, south, east]`` (NWSE), the OPPOSITE of the
        common GIS ``[west, south, east, north]`` (WSEN). Sending
        ``[w, s, e, n]`` does not error — it silently retrieves the
        wrong region. Example for Mediterranean basin (lon -6..36.5,
        lat 30..46) the correct value is ``[46, -6, 30, 36.5]``.
      - confirmed (bool, default false): bypass the size + queue-tier
        confirmation gate; also accepts an auto-chunking proposal at the
        default ``year`` granularity when no ``chunk_by`` is given.
      - chunk_by (year|month|day, optional): granularity for splitting a
        request that exceeds the dataset's server-side cost limit. The
        first submit of an over-limit request returns a
        ``cost_limit_requires_chunking`` proposal; re-submit with chunk_by
        set (or confirmed=true for the suggested year) to run the parts as
        one logical multi-file workflow under a single parent request_id.
      - auto_chunk (bool, default true): set false to reject the split and
        get the manual-split ValidationError instead.
      - force_refresh (bool, default false): re-run from scratch, bypassing
        the cache (repopulates evicted chunk files of a completed parent).
      - confirm_large_fanout (bool, default false): the SECOND ack for a very
        large split. A split over the reconfirm threshold needs BOTH
        confirmed=true and this; confirmed alone will not launch it.

    Outputs (queued):
      - {status: "queued", request_id, cache_key, result: {uri:
        "copernicus://jobs/<request_id>"}}.

    Outputs (queued — chunked parent):
      - {status: "queued", request_id, cache_key, chunked: true,
        chunk_count: N, ...}. Poll / download / cancel the parent
        request_id; the parts advance on each poll.

    Outputs (cache hit):
      - {status: "successful", cache_hit: true, request_id, cache_key,
        result: {filepath, uri, metadata, provenance}}.
    """
    payload = input.model_dump(exclude_none=True)
    options: dict[str, Any] = {}
    if payload.pop("confirmed", False):
        options["confirmed"] = True
    chunk_by = payload.pop("chunk_by", None)
    if chunk_by is not None:
        options["chunk_by"] = chunk_by
    if payload.pop("auto_chunk", True) is False:
        options["auto_chunk"] = False
    if payload.pop("force_refresh", False):
        options["force_refresh"] = True
    if payload.pop("confirm_large_fanout", False):
        options["confirm_large_fanout"] = True
    out: dict[str, Any] = await orchestrator.run(
        backend="cds",
        operation="submit",
        params=payload,
        options=options or None,
    )
    return _unwrap(out)


async def cds_check_request_status(
    input: CdsCheckRequestStatusInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """Poll the CDS server for the status of a submitted request.

    The first call after the server settles a job to ``successful``
    triggers the download into the local cache; subsequent calls
    return the cached descriptor. Status is one of ``queued`` /
    ``running`` / ``successful`` / ``failed`` / ``cancelled``.

    **Polling guidance — do NOT sleep between calls.** The Claude Code
    harness blocks standalone ``sleep N`` precisely to discourage
    naive sleep-then-retry loops; agents who work around it with
    ``until [...]; do sleep 5; done`` burn context budget on every
    wake. Instead, pick one of:

    - **User-paced re-call** (recommended for chat agents): return
      ``status=queued|running`` to the user, let them decide when
      to ask "ready yet?" — each call is a fresh tool invocation,
      no agent-side waiting.
    - **CLI ``copernicus-mcp cds wait <request_id>``** for offline
      polling — runs server-side in the CLI process, doesn't tie up
      agent context.

    Typical CDS queue latency: seconds to minutes for small ERA5
    requests; up to ~30 min for heavy multi-field hindcasts. EFAS /
    GloFAS reanalysis queries can sit in queue for ~5-10 min during
    busy periods.

    Inputs:
      - request_id: returned by ``cds_submit_request``.

    Outputs:
      - {status, request_id, submitted_at, updated_at, cache_key,
         error_details, result: {filepath?, uri, metadata, provenance}}.

    **Chunked (auto-split) requests** answer with extra fields, because one
    request id stands for many CDS jobs:

    - ``progress: {completed, total}`` and ``chunks: {...}`` — per-state
      counts that PARTITION the parts (``successful``, ``running``,
      ``downloading``, ``retrying``, ``queued``, ``failed``, ``cancelled``),
      so they sum to ``total``.
    - ``per_chunk: [{index, request_id, status, phase?, attempt?}]`` — every
      part that has been ORDERED is a first-class request id: poll it, or
      resolve its file directly with ``cds_download_request_result(<that
      id>)``. That works even when the parent as a whole failed. Parts not yet
      ordered have ``request_id: null`` and status ``queued`` — there is
      nothing to poll for those yet.
    - ``phase: "downloading"`` on the parent — every remaining part has
      finished server-side and only the local file transfers are left.
      Transfers RESUME across polls and processes (an interrupted partial
      file is continued, not restarted), so repeated polling does finish
      them — but one long-running ``copernicus-mcp cds wait`` completes
      the remaining transfers fastest, in a single uninterrupted pass.
    - ``retrying`` parts are ones the server refused for capacity; they are
      re-submitted automatically a bounded number of times. A part that failed
      because the REQUEST is wrong is never retried.
    - ``partial_result: {files, chunk_indices, missing_chunk_indices}`` on a
      FAILED parent — the parts that did land, so paid-for data is never lost.
      It is deliberately outside ``result``: a failed parent never presents
      itself as a complete delivery.
    - ``result.complete`` on a SUCCESSFUL parent — ``false`` means a part's
      file has since been evicted from the cache, so the ``files`` list is
      short; ``evicted_chunk_indices`` names the gaps and ``recovery_hint``
      says how to refill them.
    """
    if input.request_ids is not None:
        # T-CDS-OPS-002: a 21-part window polled every 30 s over 10 h was
        # ~18k separate invocations. Bounded concurrency — each poll can
        # trigger a download grace, so an unbounded gather would thunder.
        semaphore = asyncio.Semaphore(4)

        async def _one(rid: str) -> dict[str, Any]:
            async with semaphore:
                envelope = await orchestrator.run(
                    backend="cds",
                    operation="poll",
                    params={"request_id": rid},
                )
            if "error" in envelope and isinstance(envelope["error"], dict):
                # Inline, never raising: one bad id must not fail the batch.
                # Local review round (MEDIUM): the failed entry carries ITS
                # id — correlating by list order alone forced callers to
                # zip against their input, and ``request_id: null`` already
                # means "not yet submitted" elsewhere in this API.
                return {"request_id": rid, "error": envelope["error"]}
            if "result" in envelope and isinstance(envelope["result"], dict):
                return envelope["result"]
            return envelope

        results = await asyncio.gather(
            *(_one(rid) for rid in input.request_ids)
        )
        return {"results": list(results), "count": len(results)}

    out: dict[str, Any] = await orchestrator.run(
        backend="cds",
        operation="poll",
        params={"request_id": input.request_id},
    )
    return _unwrap(out)


async def cds_download_request_result(
    input: CdsDownloadRequestResultInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """Resolve the cached file descriptor for a successful CDS request.

    No bytes are returned; per the project conventions invariant 1 the response
    is ``{filepath, uri, cache_key, metadata, provenance}``. Read the
    file from disk via the returned path. The path is stable until
    cache eviction.

    Use this after ``cds_check_request_status`` reports
    ``status=successful``.

    Inputs:
      - request_id: returned by ``cds_submit_request``.

    Outputs:
      - {status: "successful", request_id, cache_key, cache_hit: true,
         result: {filepath, uri, metadata, provenance}}.
    """
    # Round-1 cr M1: ``target`` is informational; if the caller didn't
    # supply one, plug a placeholder so the orchestrator's non-empty
    # ``fetch`` validation doesn't reject. The backend ignores it.
    out: dict[str, Any] = await orchestrator.run(
        backend="cds",
        operation="fetch",
        params={
            "request_id": input.request_id,
            "target": input.target or _TARGET_UNUSED_SENTINEL,
        },
    )
    return _unwrap(out)


async def cds_cancel_request(
    input: CdsCancelRequestInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """Cancel an in-flight CDS request.

    Best-effort — the SDK's delete is fire-and-forget; the server
    may still complete the job if cancellation arrives too late.
    Local row is settled to ``cancelled`` in any case so the polling
    caller knows to stop. Calling on an already-terminal row is a
    no-op and reports the current status.

    Inputs:
      - request_id: returned by ``cds_submit_request``.

    Outputs:
      - {cancelled: bool, request_id, status, reason?}.
    """
    out: dict[str, Any] = await orchestrator.run(
        backend="cds",
        operation="cancel",
        params={"request_id": input.request_id},
    )
    return _unwrap(out)


async def cds_list_licences(
    input: CdsListLicencesInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """List the store's dataset licences and which ones this account has
    already accepted.

    Use when a submit fails with ``TermsNotAcceptedError``: compare the
    ``available`` and ``accepted`` sets to find the missing licence and its
    ``revision``. Accepting happens on the licence web page (the error's
    ``recovery_url``), or — when the operator has enabled it — in-band via
    ``cds_accept_licence``.

    Inputs:
      - dataset_id (optional): routes to the dataset's store; licence ids
        are per-store (CDS / ADS / EWDS).

    Outputs:
      - {store, available: [{id, revision, label, scope, contents_url}],
         accepted: [...same shape...]}.
    """
    out: dict[str, Any] = await orchestrator.run(
        backend="cds",
        operation="list_licences",
        params=input.model_dump(),
    )
    return _unwrap(out)


async def cds_accept_licence(
    input: CdsAcceptLicenceInput,
    *,
    orchestrator: WorkflowOrchestrator,
) -> dict[str, Any]:
    """Accept a dataset licence in-band, under the operator's standing
    authority.

    **Acceptance legally binds the account owner.** This tool is registered
    only when the operator set ``budget.cds_licence_accept_enabled`` — its
    presence IS the authorisation. Get the ``licence_id`` and ``revision``
    from ``cds_list_licences`` (or from the ``TermsNotAcceptedError``
    context), accept, then re-submit the original request.

    Inputs:
      - licence_id, revision: from ``cds_list_licences``.
      - dataset_id (optional): store routing.

    Outputs:
      - {accepted: true, licence_id, revision, store, note}.
    """
    out: dict[str, Any] = await orchestrator.run(
        backend="cds",
        operation="accept_licence",
        params=input.model_dump(),
    )
    return _unwrap(out)


def register_cds_tools(
    server: Any,
    *,
    orchestrator: WorkflowOrchestrator,
    licence_accept_enabled: bool = False,
) -> None:
    """Register all seven CDS tools on a FastMCP-compatible server.

    Same wiring discipline as ``register_marine_tools``: each registered
    wrapper has a concrete Pydantic-model input annotation so FastMCP
    auto-derives a rich JSON schema for the LLM client. ``name=`` and
    ``description=`` are set explicitly so neither depends on the local
    function name.
    """

    @server.tool(  # type: ignore[untyped-decorator]
        name=cds_search_groups.__name__,
        description=cds_search_groups.__doc__,
    )
    async def _search_groups(input: CdsSearchGroupsInput) -> dict[str, Any]:
        return await cds_search_groups(input, orchestrator=orchestrator)

    @server.tool(  # type: ignore[untyped-decorator]
        name=cds_search_datasets.__name__,
        description=cds_search_datasets.__doc__,
    )
    async def _search(input: CdsSearchDatasetsInput) -> dict[str, Any]:
        return await cds_search_datasets(input, orchestrator=orchestrator)

    @server.tool(  # type: ignore[untyped-decorator]
        name=cds_describe_dataset.__name__,
        description=cds_describe_dataset.__doc__,
    )
    async def _describe(input: CdsDescribeDatasetInput) -> dict[str, Any]:
        return await cds_describe_dataset(input, orchestrator=orchestrator)

    @server.tool(  # type: ignore[untyped-decorator]
        name=cds_apply_constraints.__name__,
        description=cds_apply_constraints.__doc__,
    )
    async def _apply_constraints(
        input: CdsApplyConstraintsInput,
    ) -> dict[str, Any]:
        return await cds_apply_constraints(input, orchestrator=orchestrator)

    @server.tool(  # type: ignore[untyped-decorator]
        name=cds_estimate_request.__name__,
        description=cds_estimate_request.__doc__,
    )
    async def _estimate(input: CdsEstimateRequestInput) -> dict[str, Any]:
        return await cds_estimate_request(input, orchestrator=orchestrator)

    @server.tool(  # type: ignore[untyped-decorator]
        name=cds_submit_request.__name__,
        description=cds_submit_request.__doc__,
    )
    async def _submit(input: CdsSubmitRequestInput) -> dict[str, Any]:
        return await cds_submit_request(input, orchestrator=orchestrator)

    @server.tool(  # type: ignore[untyped-decorator]
        name=cds_check_request_status.__name__,
        description=cds_check_request_status.__doc__,
    )
    async def _check_status(
        input: CdsCheckRequestStatusInput,
    ) -> dict[str, Any]:
        return await cds_check_request_status(input, orchestrator=orchestrator)

    @server.tool(  # type: ignore[untyped-decorator]
        name=cds_download_request_result.__name__,
        description=cds_download_request_result.__doc__,
    )
    async def _download(
        input: CdsDownloadRequestResultInput,
    ) -> dict[str, Any]:
        return await cds_download_request_result(
            input, orchestrator=orchestrator
        )

    @server.tool(  # type: ignore[untyped-decorator]
        name=cds_list_licences.__name__,
        description=cds_list_licences.__doc__,
    )
    async def _list_licences(input: CdsListLicencesInput) -> dict[str, Any]:
        return await cds_list_licences(input, orchestrator=orchestrator)

    # T-CDS-LICENCE-001: the accept tool is registered ONLY under the
    # operator's explicit opt-in — its presence on the tool surface is the
    # standing authorisation. The CLI equivalent is always available.
    if licence_accept_enabled:

        @server.tool(  # type: ignore[untyped-decorator]
            name=cds_accept_licence.__name__,
            description=cds_accept_licence.__doc__,
        )
        async def _accept_licence(input: CdsAcceptLicenceInput) -> dict[str, Any]:
            return await cds_accept_licence(input, orchestrator=orchestrator)

    @server.tool(  # type: ignore[untyped-decorator]
        name=cds_cancel_request.__name__,
        description=cds_cancel_request.__doc__,
    )
    async def _cancel(input: CdsCancelRequestInput) -> dict[str, Any]:
        return await cds_cancel_request(input, orchestrator=orchestrator)
