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
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

_GRANULARITY_ORDER: tuple[str, ...] = ("year", "month", "day")


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


def compute_parent_status(
    plan: dict[str, Any], child_status_by_id: dict[str, str]
) -> str:
    """Aggregate a chunked parent's status from its plan + child statuses
    (decision 4). Pure.

    Precedence ``failed > cancelled(stopped) > cancelled-child > successful >
    running > queued``:
      - any child terminally ``failed`` → ``failed`` (a real failure outranks a
        concurrent cancel; remaining chunks stop);
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
    statuses = [child_status_by_id.get(c["child_request_id"]) for c in submitted]
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
    ``too_many_chunks``, ``costing_unavailable``, ``exceeds_at_finest``."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


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
