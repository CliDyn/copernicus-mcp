"""Pure request splitter for CDS auto-chunking.

When a request's cost exceeds the dataset's per-request limit, split it along the
calendar axis into **whole** calendar units — one chunk per year, per month, or
per day (no odd partial-period groupings; v2 simplification). ``build_chunk_plan``
picks the coarsest granularity at which every chunk's live cost fits under the
limit. ``propose_chunks`` is pure; the async per-chunk validation lives in
``build_chunk_plan``.
"""

from __future__ import annotations

import calendar
import itertools
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

_GRANULARITY_ORDER: tuple[str, ...] = ("year", "month", "day")

# T-CDS-MODEL-001: dataset families whose backend executes ONE model per
# request while the constraints endpoint accepts a list. Requesting several
# silently delivers only the FIRST — no error, no per-model status (field run 19
# for CMIP6, confirmed by a controlled cdsapi probe; CORDEX confirmed live in
# spike T-CDS-MODEL-000: 2 RCMs requested, 1 delivered, status successful).
# A multi-model request on these datasets therefore ALWAYS fans out into one
# child per combination of the listed axes — the loss is semantic, not size.
# CORDEX carries TWO model axes (driving GCM × RCM); the proposal is the
# cartesian product of whichever registered axes are multi-valued, and the
# GCM↔RCM coupling is validated per child by live costing/submit.
# Conservative: SIS derived products (sis-*-cmip6) also carry a model axis but
# have no evidence of this behaviour — gate them only on evidence; the
# delivered-content check (T-CDS-MODEL-002) is the safety net meanwhile.
SINGLE_MODEL_EXECUTION_AXES: dict[str, tuple[str, ...]] = {
    "projections-cmip6": ("model",),
    "projections-cordex-domains-single-levels": ("gcm_model", "rcm_model"),
}


def _axis_list(inputs: dict[str, Any], axis: str) -> list[Any] | None:
    values = inputs.get(axis)
    return values if isinstance(values, list) and len(values) > 0 else None


def _to_int(token: Any) -> int | None:
    """Parse a calendar token to ``int``, or ``None`` if it is not a plain
    integer. Tokens reaching the splitter are calendar-clean (see
    ``calendar_axes_clean``), but we parse defensively for direct callers."""
    try:
        return int(str(token).strip())
    except (TypeError, ValueError):
        return None


def _valid_year_month(year: Any, month: Any) -> bool:
    """True iff ``month`` is a real month (1..12). The calendar-token shape
    check accepts any ≤2-digit run, so ``"13"`` would otherwise slip through."""
    m = _to_int(month)
    return _to_int(year) is not None and m is not None and 1 <= m <= 12


def _valid_year_month_day(year: Any, month: Any, day: Any) -> bool:
    """True iff (year, month, day) is a real calendar date — guards the day
    split from emitting impossible dates (Feb 29 in a non-leap year, Apr 31,
    day 32). A single such chunk fails at CDS and, via
    ``compute_parent_status``, fails the whole parent (v2 review HIGH)."""
    y, m, d = _to_int(year), _to_int(month), _to_int(day)
    if y is None or m is None or d is None or not 1 <= m <= 12:
        return False
    return 1 <= d <= calendar.monthrange(y, m)[1]


def propose_chunks(
    inputs: dict[str, Any], granularity: str
) -> list[dict[str, Any]] | None:
    """Clean per-calendar-unit chunk overrides: one chunk per whole year /
    (year, month) / (year, month, day). Returns the override list, or ``None`` if
    the request lacks the list-shaped axes this granularity needs."""
    years = _axis_list(inputs, "year")
    if years is None:
        return None
    if granularity == "year":
        return [{"year": [y]} for y in years]
    months = _axis_list(inputs, "month")
    if months is None:
        return None
    if granularity == "month":
        chunks = [
            {"year": [y], "month": [m]}
            for y in years
            for m in months
            if _valid_year_month(y, m)
        ]
        return chunks or None
    if granularity == "day":
        days = _axis_list(inputs, "day")
        if days is None:
            return None
        chunks = [
            {"year": [y], "month": [m], "day": [d]}
            for y in years
            for m in months
            for d in days
            if _valid_year_month_day(y, m, d)
        ]
        return chunks or None
    raise ValueError(f"unknown granularity {granularity!r}")


def apply_chunk(inputs: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``inputs`` with the chunk's ``year``/``month``/``day``
    overrides applied; all other keys are untouched."""
    child = deepcopy(inputs)
    child.update(deepcopy(chunk))
    return child


_CALENDAR_AXES: tuple[str, ...] = ("year", "month", "day")


def sibling_corroborates(
    entry: dict[str, Any],
    chunks: list[dict[str, Any]],
    statuses: Mapping[str, str],
) -> bool:
    """Does a SUCCESSFUL sibling chunk corroborate a capacity suspicion for
    ``entry``? Only when it shares every NON-CALENDAR override: a successful
    model-A chunk proves the shared shape works for model A's calendar slices,
    not that model B (or another GCM×RCM pair) is valid at all — so
    cross-model success must never promote an empty-log failure to retryable
    (review M3; content failures are never retried). Calendar-only plans have
    empty non-calendar overrides everywhere, so every sibling still
    corroborates — the original Phase-1 rule is preserved exactly there."""

    def _non_calendar(overrides: Any) -> dict[str, Any]:
        if not isinstance(overrides, dict):
            return {}
        return {k: v for k, v in overrides.items() if k not in _CALENDAR_AXES}

    mine = _non_calendar(entry.get("overrides"))
    entry_cid = entry.get("child_request_id")
    for chunk in chunks:
        cid = chunk.get("child_request_id")
        # Identity AND id equality: the same-parse identity contract is
        # fragile (round-2 LOW) — an entry COPY must not let a chunk
        # corroborate itself through its own child id.
        if chunk is entry or (cid and cid == entry_cid):
            continue
        if not cid or statuses.get(cid) != "successful":
            continue
        if _non_calendar(chunk.get("overrides")) == mine:
            return True
    return False


def compute_parent_status(
    plan: dict[str, Any], child_status_by_id: dict[str, str]
) -> str:
    """Aggregate a chunked parent's status from its plan + child statuses
    (decision 4). Pure.

    Precedence ``failed > cancelled(stopped) > cancelled-child > successful >
    running > queued``:
      - any child terminally ``failed`` → ``failed`` (a real failure outranks a
        concurrent cancel; remaining chunks stop) — EXCEPT a chunk flagged
        ``retry_pending`` (T-CDS-RESIL-003): its failure is capacity-classified
        and a bounded re-submission is still owed, so the chunk counts as
        active (``running``) — a parent is not terminal while any of its
        chunks is still retryable;
      - else the ``stopped`` flag (user cancel cascaded) → ``cancelled``;
      - else any child terminally ``cancelled`` *without* a parent cancel (e.g.
        CDS dismissed it, or the child id was cancelled directly) → ``failed``:
        that chunk's file is permanently gone, so the aggregate can never deliver
        the full set, and the parent must not run forever (codex CHUNK-003 HIGH);
      - else every chunk submitted AND every child ``successful`` → ``successful``;
      - else at least one child submitted → ``running`` (more to do / in flight);
      - else nothing submitted → ``queued``.

    A submitted child whose id has no known status (row missing) is treated as
    non-terminal, so it never promotes the parent to ``successful``."""
    chunks = plan.get("chunks", [])
    submitted = [c for c in chunks if c.get("child_request_id")]
    statuses = []
    for c in submitted:
        s = child_status_by_id.get(c["child_request_id"])
        if s == "failed" and c.get("retry_pending"):
            s = "running"  # failed-but-retryable masks as active (RESIL-003)
        statuses.append(s)
    if any(s == "failed" for s in statuses):
        return "failed"
    if plan.get("stopped"):
        return "cancelled"
    if any(s == "cancelled" for s in statuses):
        return "failed"
    all_submitted = len(chunks) > 0 and len(submitted) == len(chunks)
    if all_submitted and all(s == "successful" for s in statuses):
        return "successful"
    if submitted:
        return "running"
    return "queued"


# Max digits per calendar axis. Bounding the length is what makes this a
# credential-isolation guard: a ≤4-char ASCII-digit token cannot carry a secret,
# so even a purely-numeric credential is rejected by year's 4-digit ceiling.
_AXIS_MAX_DIGITS: dict[str, int] = {"year": 4, "month": 2, "day": 2}


def _is_calendar_token(value: Any, max_digits: int) -> bool:
    """A plausible calendar token: a short run of **ASCII** digits.

    Plain ``str.isdigit()`` accepts Unicode digits (``'१२'``, ``'²'``) and
    arbitrarily long numeric strings — so a long all-digit secret could slip
    through. We require ASCII digits and a per-axis length ceiling. ``bool`` is
    excluded explicitly (it is an ``int`` subclass)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        value = str(value)
    if not isinstance(value, str):
        return False
    token = value.strip()
    return token.isascii() and token.isdigit() and 1 <= len(token) <= max_digits


def calendar_axes_clean(inputs: dict[str, Any]) -> bool:
    """True iff every list-shaped year/month/day axis holds only calendar tokens.

    Invariant-2 guard: the split overrides are persisted verbatim in
    ``chunk_plan_json`` and reused by later waves, so they cannot be
    sanitiser-scrubbed without breaking the plan. Instead we refuse to chunk when
    a split axis carries a non-calendar (e.g. credential-shaped) value — such a
    value then never reaches a persisted workflow record via this path."""
    for axis, max_digits in _AXIS_MAX_DIGITS.items():
        values = inputs.get(axis)
        if isinstance(values, list) and not all(
            _is_calendar_token(v, max_digits) for v in values
        ):
            return False
    return True


def proposable_chunk_counts(inputs: dict[str, Any]) -> dict[str, int]:
    """Per-granularity clean chunk count for each available calendar axis."""
    counts: dict[str, int] = {}
    for granularity in _GRANULARITY_ORDER:
        proposed = propose_chunks(inputs, granularity)
        if proposed is not None:
            counts[granularity] = len(proposed)
    return counts


def granularity_estimates(
    inputs: dict[str, Any], cost_units: float, cost_limit: float
) -> dict[str, dict[str, Any]]:
    """Per-granularity proposal info for the agent: chunk count, the *estimated*
    cost per chunk (total cost spread evenly across whole units), and whether that
    estimate fits the limit. The real per-chunk cost is validated later by live
    costing — this only lets the agent pick a granularity that is likely to fit
    (e.g. it surfaces 'a single year is over the limit; use months')."""
    out: dict[str, dict[str, Any]] = {}
    for granularity, n in proposable_chunk_counts(inputs).items():
        per_chunk = cost_units / n if n else cost_units
        out[granularity] = {
            "chunks": n,
            "est_units_per_chunk": round(per_chunk, 1),
            "fits": per_chunk <= cost_limit,
        }
    return out


def is_splittable(inputs: dict[str, Any]) -> bool:
    """True iff some calendar granularity yields **≥2** whole-unit chunks and the
    split axes hold only calendar tokens (invariant-2 guard). A single calendar
    cell is not splittable. The fit-under-limit check is done later by live
    costing in ``build_chunk_plan``."""
    if not calendar_axes_clean(inputs):
        return False
    return max(proposable_chunk_counts(inputs).values(), default=0) >= 2


def propose_model_chunks(
    inputs: dict[str, Any], axes: tuple[str, ...]
) -> list[dict[str, Any]] | None:
    """One override per combination of the MULTI-VALUED registered model axes
    (T-CDS-MODEL-001). Axes that are scalar or single-element lists stay in
    the parent request untouched. ``None`` when nothing fans out."""
    multi = [
        (axis, list(dict.fromkeys(inputs[axis])))  # dedupe, order-preserving
        for axis in axes
        if isinstance(inputs.get(axis), list) and len(inputs[axis]) > 1
    ]
    multi = [(axis, values) for axis, values in multi if len(values) > 1]
    if not multi:
        return None
    axis_names = [axis for axis, _ in multi]
    return [
        {axis: [value] for axis, value in zip(axis_names, combo, strict=True)}
        for combo in itertools.product(*(values for _, values in multi))
    ]


def model_tokens_outside_vocabulary(
    overrides: list[dict[str, Any]], vocabulary: dict[str, set[str]]
) -> list[str]:
    """Model tokens in the proposed overrides that the dataset's known
    vocabulary does not contain (invariant-2 parity with the calendar-token
    guard: overrides are persisted verbatim in ``chunk_plan_json``, so a
    non-vocabulary — e.g. credential-shaped — value must refuse the split,
    never be persisted). The check is fail-CLOSED: an axis with no vocabulary
    marks all its tokens bad."""
    bad: list[str] = []
    for override in overrides:
        for axis, values in override.items():
            known = vocabulary.get(axis)
            for value in values:
                if not isinstance(value, str) or known is None or value not in known:
                    bad.append(str(value))
    return bad


@dataclass(frozen=True)
class ChunkSpec:
    """One validated chunk: the request-input overrides plus the live cost units
    the ``/costing`` endpoint returned for it (≤ the dataset limit)."""

    overrides: dict[str, Any]
    units: float


@dataclass(frozen=True)
class ChunkPlan:
    """A validated split: the granularity used and the per-chunk specs (each
    ≤ the cost limit, per-chunk verified by live costing)."""

    granularity: str
    chunks: list[ChunkSpec]


class ChunkPlanError(Exception):
    """No usable chunk plan. ``reason`` is one of: ``not_splittable``,
    ``too_many_chunks``, ``costing_unavailable``, ``exceeds_at_finest``,
    ``invalid_model_combination``. ``detail`` optionally carries a
    caller-presentable elaboration (e.g. which model combination failed
    costing) — built from vocabulary-validated tokens only."""

    def __init__(self, reason: str, detail: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


async def build_chunk_plan(
    inputs: dict[str, Any],
    cost_limit: float,
    chunk_by: str,
    *,
    costing_fn: Callable[[dict[str, Any]], Awaitable[float | None]],
    max_chunks: int = 366,
) -> ChunkPlan:
    """Build a validated plan starting at ``chunk_by`` granularity, escalating
    year→month→day until every (whole-calendar-unit) chunk's **live** cost is
    under the limit — the coarsest granularity that fits wins.

    ``costing_fn(child_inputs)`` returns a chunk's cost units (or ``None`` if
    costing is unavailable). ``ChunkPlanError`` reasons: ``not_splittable`` (no
    calendar axis to split along), ``too_many_chunks`` (the clean split exceeds
    ``max_chunks``), ``costing_unavailable``, ``exceeds_at_finest`` (even single
    days are over the limit)."""
    start = _GRANULARITY_ORDER.index(chunk_by) if chunk_by in _GRANULARITY_ORDER else 0
    proposed_any = False
    for granularity in _GRANULARITY_ORDER[start:]:
        proposal = propose_chunks(inputs, granularity)
        if proposal is None:
            continue  # this axis is not present; try a finer one
        proposed_any = True
        if len(proposal) > max_chunks:
            raise ChunkPlanError("too_many_chunks")
        units_per_chunk: list[float] = []
        all_under = True
        for chunk in proposal:
            units = await costing_fn(apply_chunk(inputs, chunk))
            if units is None:
                raise ChunkPlanError("costing_unavailable")
            if units > cost_limit:
                all_under = False
                break
            units_per_chunk.append(units)
        if all_under:
            return ChunkPlan(
                granularity=granularity,
                chunks=[
                    ChunkSpec(overrides=chunk, units=units)
                    for chunk, units in zip(proposal, units_per_chunk, strict=True)
                ],
            )
        # Some whole unit is still over the limit → escalate to the finer axis.
    raise ChunkPlanError("exceeds_at_finest" if proposed_any else "not_splittable")


async def build_model_chunk_plan(
    inputs: dict[str, Any],
    model_overrides: list[dict[str, Any]],
    *,
    cost_limit: float | None,
    chunk_by: str,
    costing_fn: Callable[[dict[str, Any]], Awaitable[float | None]],
    max_chunks: int = 366,
) -> ChunkPlan:
    """Compose the model fan-out with the calendar escalation
    (T-CDS-MODEL-001): one child per model combination; a per-model child
    whose live cost exceeds ``cost_limit`` is further calendar-split and its
    sub-chunks carry BOTH the model and the calendar override.

    The model split is SEMANTIC — it must happen even when cost information is
    missing — so unlike ``build_chunk_plan`` an unavailable costing does not
    fail the plan: the child is kept whole with ``units: 0.0`` and an
    over-limit child falls back to the submit-time 403 (which also covers
    CORDEX, whose paired ``start_year``/``end_year`` blocks are not calendar
    axes the splitter knows). ``granularity`` is ``"model"`` or
    ``"model+<axis>"`` when any child was calendar-split."""
    chunks: list[ChunkSpec] = []
    sub_granularities: set[str] = set()
    costed: list[tuple[dict[str, Any], float | None]] = []
    for override in model_overrides:
        child_inputs = apply_chunk(inputs, override)
        units = await costing_fn(child_inputs)
        costed.append((override, units))
        if (
            units is not None
            and cost_limit is not None
            and cost_limit > 0
            and units > cost_limit
        ):
            try:
                # Invariant-2 parity with the cost branch (review, local M2):
                # calendar overrides are persisted verbatim, so a sub-split is
                # allowed only over clean calendar tokens — a garbage (e.g.
                # credential-shaped) year value refuses the plan instead of
                # riding a per-chunk costing that happens to tolerate it.
                if not calendar_axes_clean(child_inputs):
                    raise ChunkPlanError("not_splittable")
                subplan = await build_chunk_plan(
                    child_inputs,
                    cost_limit,
                    chunk_by,
                    costing_fn=costing_fn,
                    max_chunks=max_chunks,
                )
            except ChunkPlanError as exc:
                if exc.reason == "not_splittable":
                    # No calendar axis to sub-split (the CORDEX shape) and the
                    # child is KNOWN over the limit — submitting it would burn
                    # a job slot on a guaranteed 403 (review M2: the normal
                    # cost branch refuses this pre-flight; the model path must
                    # not be laxer). Only a child with UNKNOWN cost rides the
                    # 403 fallback.
                    raise ChunkPlanError("exceeds_at_finest") from exc
                raise
            sub_granularities.add(subplan.granularity)
            for spec in subplan.chunks:
                chunks.append(
                    ChunkSpec(
                        overrides={**override, **spec.overrides},
                        units=spec.units,
                    )
                )
        elif units is not None:
            chunks.append(ChunkSpec(overrides=dict(override), units=units))
        # A ``None`` costing is deferred: judged after the loop (retry, then
        # invalid-combination vs systemic-outage decision).
        if len(chunks) > max_chunks:
            raise ChunkPlanError("too_many_chunks")
    # Review, local M5: costing answering for SOME combos but not others is
    # the signature of an invalid model combination (a GCM×RCM pair that never
    # ran) — submitting it whole would 400 at submit time and the first-wave
    # abort would cancel every valid sibling. Refuse pre-flight, naming the
    # combination (tokens are vocabulary-validated upstream, safe to echo).
    # When costing failed for EVERYTHING the outage is systemic, not
    # combination-specific: the split still happens (the loss is semantic) and
    # the submit-time 403 stays the backstop.
    failed = [ov for ov, u in costed if u is None]
    if failed and len(failed) < len(costed):
        # Round-2 review (local): ``fetch_costing`` returns ``None`` for ANY
        # failure — timeout, 5xx, burst throttling — not only "combination
        # does not exist", and the model fan-out fires N costings in a burst.
        # Re-cost the failures once before judging, or a single blip would
        # produce a refusal whose guidance makes the agent drop a VALID model
        # — the exact silent loss this feature prevents, laundered through a
        # validation error.
        still_failed: list[dict[str, Any]] = []
        for override in failed:
            retry_units = await costing_fn(apply_chunk(inputs, override))
            if retry_units is None:
                still_failed.append(override)
            else:
                chunks.append(
                    ChunkSpec(overrides=dict(override), units=retry_units)
                )
        if still_failed:
            names = ", ".join(
                "×".join(str(v[0]) for v in ov.values()) for ov in still_failed
            )
            raise ChunkPlanError(
                "invalid_model_combination",
                detail=(
                    f"The costing pre-flight repeatedly failed for {names} "
                    "while other combinations costed normally. Either a "
                    "transient costing outage — resubmit the same request to "
                    "rule that out — or a combination that does not exist; "
                    "submit it alone to see the server's own error. Do not "
                    "silently drop a model you need."
                ),
            )
    elif failed:
        # EVERY combination failed to cost: a systemic outage, not a bad
        # pair. The split still happens (the loss is semantic; spike §5) with
        # unknown units — the submit-time 403 is the backstop.
        for override in failed:
            chunks.append(ChunkSpec(overrides=dict(override), units=0.0))
    if len(chunks) > max_chunks:
        raise ChunkPlanError("too_many_chunks")
    granularity = "model"
    if sub_granularities:
        ordered = [g for g in _GRANULARITY_ORDER if g in sub_granularities]
        granularity = "model+" + "/".join(ordered or sorted(sub_granularities))
    return ChunkPlan(granularity=granularity, chunks=chunks)
