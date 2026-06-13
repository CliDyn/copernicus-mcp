"""CDS auto-chunking splitter — clean per-calendar-unit chunks (v2)."""

from __future__ import annotations

import asyncio

import pytest

from copernicus_mcp.backends.cds.chunking import (
    ChunkPlanError,
    apply_chunk,
    build_chunk_plan,
    compute_parent_status,
    granularity_estimates,
    is_splittable,
    proposable_chunk_counts,
    propose_chunks,
)

_YEARS_5 = ["2020", "2021", "2022", "2023", "2024"]
_MONTHS_12 = [f"{m:02d}" for m in range(1, 13)]
_DAYS_31 = [f"{d:02d}" for d in range(1, 32)]


# --- propose_chunks: one chunk per whole calendar unit ------------------


def test_year_granularity_one_chunk_per_year() -> None:
    inputs = {"variable": ["2m_temperature"], "year": _YEARS_5, "month": _MONTHS_12}
    assert propose_chunks(inputs, "year") == [{"year": [y]} for y in _YEARS_5]


def test_month_granularity_one_chunk_per_year_month() -> None:
    inputs = {"year": ["2020", "2021"], "month": ["01", "02", "03"]}
    chunks = propose_chunks(inputs, "month")
    assert len(chunks) == 2 * 3
    assert {"year": ["2020"], "month": ["01"]} in chunks
    assert {"year": ["2021"], "month": ["03"]} in chunks
    # every chunk is a single (year, month)
    assert all(len(c["year"]) == 1 and len(c["month"]) == 1 for c in chunks)


def test_day_granularity_one_chunk_per_year_month_day() -> None:
    inputs = {"year": ["2020"], "month": ["01"], "day": _DAYS_31}
    chunks = propose_chunks(inputs, "day")
    assert len(chunks) == 31
    assert all(
        len(c["year"]) == 1 and len(c["month"]) == 1 and len(c["day"]) == 1
        for c in chunks
    )
    # days partition the original, order preserved
    assert [c["day"][0] for c in chunks] == _DAYS_31


def test_day_granularity_skips_impossible_dates_non_leap() -> None:
    """A day-split must enumerate REAL calendar dates, not the full
    year×month×day cross-product. 2023 is non-leap: February stops at 28,
    April at 30. Impossible dates (Feb 29/30/31, Apr 31) would otherwise
    submit child requests CDS rejects, and one failed chunk fails the whole
    parent (codex/cr v2 HIGH)."""
    inputs = {"year": ["2023"], "month": ["02", "04"], "day": _DAYS_31}
    chunks = propose_chunks(inputs, "day")
    feb_days = [int(c["day"][0]) for c in chunks if c["month"] == ["02"]]
    apr_days = [int(c["day"][0]) for c in chunks if c["month"] == ["04"]]
    assert feb_days == list(range(1, 29))  # 28 days, no 29/30/31
    assert apr_days == list(range(1, 31))  # 30 days, no 31
    assert len(chunks) == 28 + 30


def test_day_granularity_includes_feb_29_in_leap_year() -> None:
    """2024 is a leap year — Feb 29 is a real date and must be emitted."""
    inputs = {"year": ["2024"], "month": ["02"], "day": _DAYS_31}
    feb_days = [int(c["day"][0]) for c in propose_chunks(inputs, "day")]
    assert feb_days == list(range(1, 30))  # 1..29


def test_proposable_counts_use_real_calendar_days() -> None:
    """The per-granularity day count must reflect real days so the
    ``max_chunks`` ceiling and per-chunk cost estimate are correct (a full
    year of daily data is ≤366 chunks, not 12×31=372)."""
    non_leap = proposable_chunk_counts(
        {"year": ["2023"], "month": _MONTHS_12, "day": _DAYS_31}
    )
    assert non_leap["day"] == 365
    leap = proposable_chunk_counts(
        {"year": ["2024"], "month": _MONTHS_12, "day": _DAYS_31}
    )
    assert leap["day"] == 366


def test_month_granularity_skips_out_of_range_month_token() -> None:
    """A 2-digit-but-out-of-range month (``"13"``) passes the calendar-token
    shape check but is not a real month → never emitted as a chunk."""
    inputs = {"year": ["2023"], "month": ["12", "13"]}
    months = [c["month"][0] for c in propose_chunks(inputs, "month")]
    assert months == ["12"]


def test_month_none_without_month_list() -> None:
    assert propose_chunks({"year": _YEARS_5}, "month") is None


def test_day_none_without_day_list() -> None:
    assert propose_chunks({"year": ["2020"], "month": ["01"]}, "day") is None


@pytest.mark.parametrize(
    "inputs",
    [
        {"year": "2020", "month": _MONTHS_12},  # scalar year
        {"variable": ["x"]},  # no year
        {"date": "2020-01-01/2020-12-31"},  # MARS date string
        {"year": []},  # empty year list
    ],
)
def test_not_splittable_shapes_return_none(inputs: dict) -> None:
    assert propose_chunks(inputs, "year") is None


def test_year_chunks_partition_original_years() -> None:
    chunks = propose_chunks({"year": _YEARS_5}, "year")
    assert [y for c in chunks for y in c["year"]] == _YEARS_5  # union, order, no overlap


# --- apply_chunk --------------------------------------------------------


def test_apply_chunk_replaces_year_preserves_rest() -> None:
    inputs = {"variable": ["x"], "year": _YEARS_5, "month": _MONTHS_12, "data_format": "grib"}
    out = apply_chunk(inputs, {"year": ["2020"]})
    assert out == {"variable": ["x"], "year": ["2020"], "month": _MONTHS_12, "data_format": "grib"}
    assert inputs["year"] == _YEARS_5  # original untouched (deep copy)


def test_apply_chunk_replaces_year_and_month() -> None:
    inputs = {"variable": ["x"], "year": _YEARS_5, "month": _MONTHS_12}
    out = apply_chunk(inputs, {"year": ["2021"], "month": ["02"]})
    assert out["year"] == ["2021"]
    assert out["month"] == ["02"]
    assert out["variable"] == ["x"]


# --- proposable_chunk_counts / is_splittable ----------------------------


def test_proposable_chunk_counts_all_granularities() -> None:
    inputs = {"year": _YEARS_5, "month": _MONTHS_12, "day": _DAYS_31}
    # 5 years; 5*12 = 60 (year,month); day = REAL calendar days, not 5*12*31:
    # 2020 (366) + 2021 (365) + 2022 (365) + 2023 (365) + 2024 (366) = 1827.
    assert proposable_chunk_counts(inputs) == {"year": 5, "month": 60, "day": 1827}


def test_proposable_chunk_counts_year_only() -> None:
    assert proposable_chunk_counts({"year": _YEARS_5}) == {"year": 5}


def test_is_splittable_true_for_multi_year() -> None:
    assert is_splittable({"year": _YEARS_5, "month": _MONTHS_12}) is True


def test_is_splittable_false_for_single_cell() -> None:
    assert is_splittable({"year": ["2024"], "month": ["01"], "day": ["01"]}) is False


def test_is_splittable_false_without_year_list() -> None:
    assert is_splittable({"date": "2020-01-01/2020-12-31"}) is False


def test_is_splittable_false_for_non_calendar_token() -> None:
    dirty = {"year": ["2020", "abcdef01-2345-6789-abcd-ef0123456789"], "month": _MONTHS_12}
    assert is_splittable(dirty) is False


def test_is_splittable_rejects_unicode_and_overlong_numeric_tokens() -> None:
    assert is_splittable({"year": ["2020", "१२", "2022"]}) is False
    assert is_splittable({"year": ["2020", "1234567890123456"]}) is False
    assert is_splittable({"year": _YEARS_5, "month": ["01", "123"]}) is False
    assert is_splittable({"year": _YEARS_5, "month": _MONTHS_12}) is True


# --- granularity_estimates (agent-facing viability) ---------------------


def test_granularity_estimates_flags_year_too_big() -> None:
    """ERA5-Land-style: a single year is over the limit, months fit — the agent
    sees that up front instead of guessing."""
    inputs = {"year": ["2020", "2021", "2022"], "month": _MONTHS_12}
    est = granularity_estimates(inputs, cost_units=53568.0, cost_limit=12000.0)
    assert est["year"]["chunks"] == 3
    assert est["year"]["est_units_per_chunk"] == 17856.0
    assert est["year"]["fits"] is False
    assert est["month"]["chunks"] == 36
    assert est["month"]["est_units_per_chunk"] == 1488.0
    assert est["month"]["fits"] is True


# --- build_chunk_plan (live-cost escalation, whole units) ---------------


def _fake_cost_fn(scale: float = 1.0):
    """Cost ≈ count of (year × month × day) cells; missing axes assumed full."""
    async def cost(inputs: dict) -> float | None:
        y = len(inputs["year"]) if isinstance(inputs.get("year"), list) else 5
        m = len(inputs["month"]) if isinstance(inputs.get("month"), list) else 12
        d = len(inputs["day"]) if isinstance(inputs.get("day"), list) else 31
        return y * m * d * scale
    return cost


def _run(coro):
    return asyncio.run(coro)


def test_build_plan_year_fits() -> None:
    inputs = {"year": _YEARS_5, "month": _MONTHS_12, "day": _DAYS_31}
    # per single-year chunk = 1*12*31 = 372 ≤ 400 → whole years.
    plan = _run(build_chunk_plan(inputs, 400.0, "year", costing_fn=_fake_cost_fn()))
    assert plan.granularity == "year"
    assert len(plan.chunks) == 5
    assert all(len(c.overrides["year"]) == 1 for c in plan.chunks)
    assert [round(c.units) for c in plan.chunks] == [372] * 5


def test_build_plan_escalates_year_to_clean_months() -> None:
    inputs = {"year": _YEARS_5, "month": _MONTHS_12, "day": _DAYS_31}
    # limit 300: a single year (372) is over → escalate to WHOLE months (31 each).
    plan = _run(build_chunk_plan(inputs, 300.0, "year", costing_fn=_fake_cost_fn()))
    assert plan.granularity == "month"
    assert len(plan.chunks) == 60  # 5 years * 12 months, clean
    assert all(
        len(c.overrides["year"]) == 1 and len(c.overrides["month"]) == 1
        for c in plan.chunks
    )


def test_build_plan_respects_agent_chosen_month() -> None:
    inputs = {"year": _YEARS_5, "month": _MONTHS_12}
    plan = _run(build_chunk_plan(inputs, 400.0, "month", costing_fn=_fake_cost_fn()))
    assert plan.granularity == "month"


def test_build_plan_not_splittable_raises() -> None:
    with pytest.raises(ChunkPlanError) as exc:
        _run(build_chunk_plan({"variable": ["x"]}, 400.0, "year", costing_fn=_fake_cost_fn()))
    assert exc.value.reason == "not_splittable"


def test_build_plan_too_many_chunks_raises() -> None:
    inputs = {"year": ["2020"], "month": ["01"], "day": _DAYS_31}
    with pytest.raises(ChunkPlanError) as exc:
        _run(
            build_chunk_plan(
                inputs, 400.0, "day", costing_fn=_fake_cost_fn(), max_chunks=10
            )
        )
    assert exc.value.reason == "too_many_chunks"


def test_build_plan_full_year_daily_within_default_cap() -> None:
    """A single year of daily data is ≤366 real days, so it must NOT trip the
    default ``max_chunks=366`` ceiling — the old 12×31=372 cross-product
    wrongly rejected a valid one-year-daily request as ``too_many_chunks``."""
    inputs = {"year": ["2023"], "month": _MONTHS_12, "day": _DAYS_31}
    plan = _run(build_chunk_plan(inputs, 10.0, "day", costing_fn=_fake_cost_fn()))
    assert plan.granularity == "day"
    assert len(plan.chunks) == 365  # real calendar days, not 12×31=372


def test_build_plan_costing_unavailable_raises() -> None:
    async def none_cost(inputs: dict) -> float | None:
        return None

    inputs = {"year": _YEARS_5, "month": _MONTHS_12}
    with pytest.raises(ChunkPlanError) as exc:
        _run(build_chunk_plan(inputs, 400.0, "year", costing_fn=none_cost))
    assert exc.value.reason == "costing_unavailable"


def test_build_plan_exceeds_at_finest_raises() -> None:
    # A single day already over the limit → even day granularity can't help.
    inputs = {"year": ["2020"], "month": ["01"], "day": ["01", "02", "03"]}
    with pytest.raises(ChunkPlanError) as exc:
        _run(build_chunk_plan(inputs, 400.0, "day", costing_fn=_fake_cost_fn(1000.0)))
    assert exc.value.reason == "exceeds_at_finest"


# --- compute_parent_status (decision-4 truth table) ---------------------


def _plan(n: int, *, submitted: int = 0, stopped: bool = False) -> dict:
    chunks = [
        {
            "index": i,
            "overrides": {"year": [str(2020 + i)]},
            "child_request_id": f"child-{i}" if i < submitted else None,
            "units": 100.0,
        }
        for i in range(n)
    ]
    return {"granularity": "year", "stopped": stopped, "chunks": chunks}


def test_parent_status_queued_when_no_child_submitted() -> None:
    assert compute_parent_status(_plan(5, submitted=0), {}) == "queued"


def test_parent_status_running_with_unsubmitted_chunks_remaining() -> None:
    plan = _plan(5, submitted=2)
    assert compute_parent_status(plan, {"child-0": "successful", "child-1": "running"}) == "running"


def test_parent_status_running_when_all_submitted_but_one_in_flight() -> None:
    plan = _plan(3, submitted=3)
    statuses = {"child-0": "successful", "child-1": "successful", "child-2": "running"}
    assert compute_parent_status(plan, statuses) == "running"


def test_parent_status_successful_when_all_submitted_and_successful() -> None:
    plan = _plan(3, submitted=3)
    statuses = {"child-0": "successful", "child-1": "successful", "child-2": "successful"}
    assert compute_parent_status(plan, statuses) == "successful"


def test_parent_status_failed_when_any_child_failed() -> None:
    plan = _plan(5, submitted=2)
    assert compute_parent_status(plan, {"child-0": "successful", "child-1": "failed"}) == "failed"


def test_parent_status_cancelled_when_stopped() -> None:
    plan = _plan(5, submitted=2, stopped=True)
    assert compute_parent_status(plan, {"child-0": "successful", "child-1": "cancelled"}) == "cancelled"


def test_parent_status_failed_beats_cancelled() -> None:
    plan = _plan(5, submitted=2, stopped=True)
    assert compute_parent_status(plan, {"child-0": "failed", "child-1": "cancelled"}) == "failed"


def test_parent_status_independent_cancelled_child_fails_parent() -> None:
    plan = _plan(5, submitted=2, stopped=False)
    assert compute_parent_status(plan, {"child-0": "successful", "child-1": "cancelled"}) == "failed"
