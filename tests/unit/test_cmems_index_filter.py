"""Unit tests for ``copernicus_mcp.backends.cmems._index_filter``.

Table-driven over synthetic DataFrames matching the canonical IndexRow
schema. No fixture files needed — filters are pure and the inputs are
small enough to hardcode.
"""

from __future__ import annotations

import pandas as pd
import pytest

from copernicus_mcp.backends.cmems._index_filter import (
    apply_filters,
    filter_by_bbox,
    filter_by_platform,
    filter_by_time,
    filter_by_variables,
)
from copernicus_mcp.errors import ValidationError

_CANONICAL_COLUMNS = [
    "file_path",
    "lon_min",
    "lon_max",
    "lat_min",
    "lat_max",
    "time_start",
    "time_end",
    "platform_type",
    "variables",
    "size_bytes",
]


def _row(
    file_path: str,
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    time_start: str = "2010-01-01T00:00:00Z",
    time_end: str = "2010-12-31T23:59:59Z",
    platform_type: str | None = None,
    variables: tuple[str, ...] | None = None,
    size_bytes: int | None = None,
) -> dict[str, object]:
    return {
        "file_path": file_path,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "time_start": pd.Timestamp(time_start),
        "time_end": pd.Timestamp(time_end),
        "platform_type": platform_type,
        "variables": variables,
        "size_bytes": size_bytes,
    }


def _df(*rows: dict[str, object]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            {
                "file_path": pd.Series(dtype=str),
                "lon_min": pd.Series(dtype=float),
                "lon_max": pd.Series(dtype=float),
                "lat_min": pd.Series(dtype=float),
                "lat_max": pd.Series(dtype=float),
                "time_start": pd.Series(dtype="datetime64[us, UTC]"),
                "time_end": pd.Series(dtype="datetime64[us, UTC]"),
                "platform_type": pd.Series(dtype=object),
                "variables": pd.Series(dtype=object),
                "size_bytes": pd.Series(dtype=object),
            }
        )[_CANONICAL_COLUMNS]
    df = pd.DataFrame(list(rows))
    df["time_start"] = pd.to_datetime(df["time_start"], utc=True)
    df["time_end"] = pd.to_datetime(df["time_end"], utc=True)
    return df[_CANONICAL_COLUMNS]


class TestFilterByBbox:
    """Spatial intersection — each row's bbox vs user bbox."""

    def test_exact_match_keeps_row(self) -> None:
        df = _df(_row("a.nc", lon_min=-5, lon_max=10, lat_min=30, lat_max=46))
        out = filter_by_bbox(df, (-5.0, 30.0, 10.0, 46.0))
        assert list(out["file_path"]) == ["a.nc"]

    def test_row_fully_inside_user_bbox_kept(self) -> None:
        df = _df(_row("a.nc", lon_min=-2, lon_max=5, lat_min=33, lat_max=40))
        out = filter_by_bbox(df, (-10.0, 30.0, 10.0, 46.0))
        assert list(out["file_path"]) == ["a.nc"]

    def test_row_fully_contains_user_bbox_kept(self) -> None:
        # Row's bbox covers a wider area than the user — still intersects.
        df = _df(_row("a.nc", lon_min=-180, lon_max=180, lat_min=-90, lat_max=90))
        out = filter_by_bbox(df, (-5.0, 30.0, 10.0, 46.0))
        assert list(out["file_path"]) == ["a.nc"]

    def test_outside_row_dropped(self) -> None:
        df = _df(_row("a.nc", lon_min=100, lon_max=120, lat_min=0, lat_max=10))
        out = filter_by_bbox(df, (-5.0, 30.0, 10.0, 46.0))
        assert len(out) == 0

    def test_partial_overlap_kept(self) -> None:
        df = _df(_row("a.nc", lon_min=8, lon_max=20, lat_min=40, lat_max=50))
        out = filter_by_bbox(df, (-5.0, 30.0, 10.0, 46.0))
        assert list(out["file_path"]) == ["a.nc"]

    def test_edge_touching_intersection_kept(self) -> None:
        # Row's lon_max exactly equals user's lon_min — counts as touching.
        df = _df(_row("a.nc", lon_min=-15, lon_max=-5, lat_min=33, lat_max=40))
        out = filter_by_bbox(df, (-5.0, 30.0, 10.0, 46.0))
        assert list(out["file_path"]) == ["a.nc"]

    def test_disjoint_lat_dropped(self) -> None:
        # lon overlaps, lat doesn't.
        df = _df(_row("a.nc", lon_min=-5, lon_max=10, lat_min=50, lat_max=60))
        out = filter_by_bbox(df, (-5.0, 30.0, 10.0, 46.0))
        assert len(out) == 0

    def test_disjoint_lon_dropped(self) -> None:
        df = _df(_row("a.nc", lon_min=20, lon_max=30, lat_min=33, lat_max=40))
        out = filter_by_bbox(df, (-5.0, 30.0, 10.0, 46.0))
        assert len(out) == 0

    def test_empty_df_returns_empty(self) -> None:
        df = _df()
        out = filter_by_bbox(df, (-5.0, 30.0, 10.0, 46.0))
        assert len(out) == 0
        assert list(out.columns) == _CANONICAL_COLUMNS

    def test_mixed_rows_only_intersecting_kept(self) -> None:
        df = _df(
            _row("hit.nc", lon_min=0, lon_max=5, lat_min=35, lat_max=40),
            _row("miss.nc", lon_min=50, lon_max=60, lat_min=0, lat_max=10),
            _row("edge.nc", lon_min=-5, lon_max=-5, lat_min=46, lat_max=46),
        )
        out = filter_by_bbox(df, (-5.0, 30.0, 10.0, 46.0))
        assert set(out["file_path"]) == {"hit.nc", "edge.nc"}

    # Antimeridian — user bbox where lon_min > lon_max wraps the dateline.

    def test_antimeridian_user_bbox_keeps_row_in_eastern_half(self) -> None:
        # User wants 170E .. 170W → [170,180] ∪ [-180,-170].
        # Row sits in the eastern half (175 to 179).
        df = _df(_row("east.nc", lon_min=175, lon_max=179, lat_min=0, lat_max=10))
        out = filter_by_bbox(df, (170.0, -10.0, -170.0, 10.0))
        assert list(out["file_path"]) == ["east.nc"]

    def test_antimeridian_user_bbox_keeps_row_in_western_half(self) -> None:
        df = _df(_row("west.nc", lon_min=-179, lon_max=-175, lat_min=0, lat_max=10))
        out = filter_by_bbox(df, (170.0, -10.0, -170.0, 10.0))
        assert list(out["file_path"]) == ["west.nc"]

    def test_antimeridian_user_bbox_drops_rows_outside(self) -> None:
        # Row in the "middle" of the planet — Pacific just across the antimeridian
        # but not in either user-wedge.
        df = _df(_row("middle.nc", lon_min=-150, lon_max=-100, lat_min=0, lat_max=10))
        out = filter_by_bbox(df, (170.0, -10.0, -170.0, 10.0))
        assert len(out) == 0

    def test_antimeridian_row_straddling_dateline_via_two_halves(self) -> None:
        # User bbox is antimeridian. A row that has its own bbox like
        # [-180, 180] (global) intersects both halves — keep.
        df = _df(_row("global.nc", lon_min=-180, lon_max=180, lat_min=-10, lat_max=10))
        out = filter_by_bbox(df, (170.0, -10.0, -170.0, 10.0))
        assert list(out["file_path"]) == ["global.nc"]


class TestFilterByTime:
    """Temporal intersection — row's [time_start, time_end] vs user range."""

    def test_row_fully_inside_user_range_kept(self) -> None:
        df = _df(_row("a.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                       time_start="2010-06-01T00:00:00Z", time_end="2010-06-30T23:59:59Z"))
        out = filter_by_time(df, ("2010-01-01T00:00:00Z", "2010-12-31T23:59:59Z"))
        assert list(out["file_path"]) == ["a.nc"]

    def test_row_fully_contains_user_range_kept(self) -> None:
        df = _df(_row("a.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                       time_start="2000-01-01T00:00:00Z", time_end="2030-12-31T23:59:59Z"))
        out = filter_by_time(df, ("2010-01-01T00:00:00Z", "2010-12-31T23:59:59Z"))
        assert list(out["file_path"]) == ["a.nc"]

    def test_disjoint_before_user_range_dropped(self) -> None:
        df = _df(_row("a.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                       time_start="2005-01-01T00:00:00Z", time_end="2005-12-31T23:59:59Z"))
        out = filter_by_time(df, ("2010-01-01T00:00:00Z", "2010-12-31T23:59:59Z"))
        assert len(out) == 0

    def test_disjoint_after_user_range_dropped(self) -> None:
        df = _df(_row("a.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                       time_start="2020-01-01T00:00:00Z", time_end="2020-12-31T23:59:59Z"))
        out = filter_by_time(df, ("2010-01-01T00:00:00Z", "2010-12-31T23:59:59Z"))
        assert len(out) == 0

    def test_partial_overlap_left_kept(self) -> None:
        # Row starts before user range, ends inside it.
        df = _df(_row("a.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                       time_start="2009-06-01T00:00:00Z", time_end="2010-03-01T00:00:00Z"))
        out = filter_by_time(df, ("2010-01-01T00:00:00Z", "2010-12-31T23:59:59Z"))
        assert list(out["file_path"]) == ["a.nc"]

    def test_partial_overlap_right_kept(self) -> None:
        df = _df(_row("a.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                       time_start="2010-10-01T00:00:00Z", time_end="2011-03-01T00:00:00Z"))
        out = filter_by_time(df, ("2010-01-01T00:00:00Z", "2010-12-31T23:59:59Z"))
        assert list(out["file_path"]) == ["a.nc"]

    def test_edge_touching_endpoint_kept(self) -> None:
        # Row's time_end exactly equals user range's start.
        df = _df(_row("a.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                       time_start="2009-06-01T00:00:00Z", time_end="2010-01-01T00:00:00Z"))
        out = filter_by_time(df, ("2010-01-01T00:00:00Z", "2010-12-31T23:59:59Z"))
        assert list(out["file_path"]) == ["a.nc"]

    def test_empty_df_returns_empty(self) -> None:
        df = _df()
        out = filter_by_time(df, ("2010-01-01T00:00:00Z", "2010-12-31T23:59:59Z"))
        assert len(out) == 0
        assert list(out.columns) == _CANONICAL_COLUMNS

    def test_user_range_with_offset_normalises_to_utc(self) -> None:
        # User passes a +02:00 offset — should be UTC-normalised before
        # comparison. 2010-01-01T02:00:00+02:00 == 2010-01-01T00:00:00Z.
        df = _df(_row("a.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                       time_start="2010-01-01T00:30:00Z", time_end="2010-12-31T00:00:00Z"))
        out = filter_by_time(df, ("2010-01-01T02:00:00+02:00", "2010-12-31T02:00:00+02:00"))
        assert list(out["file_path"]) == ["a.nc"]

    def test_inverted_user_range_raises_validation_error(self) -> None:
        # The contract is user.start <= user.end; reject inversion at the
        # boundary rather than producing an empty result.
        df = _df()
        with pytest.raises(ValidationError):
            filter_by_time(df, ("2011-01-01T00:00:00Z", "2010-01-01T00:00:00Z"))

    def test_naive_user_datetime_raises_validation_error(self) -> None:
        # iso8601_utc rejects naive datetimes; surface that contract.
        df = _df()
        with pytest.raises(ValidationError):
            filter_by_time(df, ("2010-01-01T00:00:00", "2010-12-31T23:59:59"))


class TestFilterByVariables:
    """Variable-set membership — permissive on None (unknown)."""

    def test_row_with_intersecting_variable_kept(self) -> None:
        df = _df(_row("a.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                       variables=("TEMP", "PSAL")))
        out = filter_by_variables(df, ["TEMP"])
        assert list(out["file_path"]) == ["a.nc"]

    def test_row_with_disjoint_variables_dropped(self) -> None:
        df = _df(_row("a.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                       variables=("PSAL",)))
        out = filter_by_variables(df, ["TEMP"])
        assert len(out) == 0

    def test_row_with_none_variables_is_kept_permissively(self) -> None:
        # Per sub-plan: ``None`` means "unknown, can't reject" → keep.
        df = _df(_row("a.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                       variables=None))
        out = filter_by_variables(df, ["TEMP"])
        assert list(out["file_path"]) == ["a.nc"]

    def test_multiple_requested_any_match_kept(self) -> None:
        df = _df(_row("a.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                       variables=("PSAL", "CHLA")))
        out = filter_by_variables(df, ["TEMP", "PSAL"])
        assert list(out["file_path"]) == ["a.nc"]

    def test_empty_requested_keeps_everything(self) -> None:
        # An empty filter is a no-op; the caller can pass [] to mean "all".
        df = _df(_row("a.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                       variables=("TEMP",)))
        out = filter_by_variables(df, [])
        assert list(out["file_path"]) == ["a.nc"]

    def test_empty_tuple_variables_dropped(self) -> None:
        # A row with a NON-None but empty tuple variables: explicitly says
        # "no variables here". Strict — drop on any non-empty requested.
        df = _df(_row("a.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                       variables=()))
        out = filter_by_variables(df, ["TEMP"])
        assert len(out) == 0

    def test_empty_df_returns_empty(self) -> None:
        out = filter_by_variables(_df(), ["TEMP"])
        assert len(out) == 0
        assert list(out.columns) == _CANONICAL_COLUMNS

    def test_mixed_rows_partition_correctly(self) -> None:
        df = _df(
            _row("hit.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                  variables=("TEMP",)),
            _row("miss.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                  variables=("PSAL",)),
            _row("unknown.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                  variables=None),
        )
        out = filter_by_variables(df, ["TEMP"])
        assert set(out["file_path"]) == {"hit.nc", "unknown.nc"}

    def test_nan_variables_treated_as_unknown_kept(self) -> None:
        # cr round-1 I2: NaN/pd.NA in the variables column would crash the
        # iterable check. Today's parser path only ever emits None | tuple,
        # but defence-in-depth: any nullish sentinel must behave like None.
        import numpy as np
        df = pd.DataFrame(
            [
                {"file_path": "nan.nc", "variables": np.nan, "platform_type": None},
                {"file_path": "na.nc", "variables": pd.NA, "platform_type": None},
            ]
        )
        out = filter_by_variables(df, ["TEMP"])
        assert set(out["file_path"]) == {"nan.nc", "na.nc"}

    def test_list_and_frozenset_variables_accepted(self) -> None:
        # cr round-2 contract-widening: the IndexRow contract pins variables
        # as tuple, but a defensive filter must not silently drop rows just
        # because a future caller used a different iterable shape.
        df = pd.DataFrame(
            [
                {"file_path": "tup.nc", "variables": ("TEMP",)},
                {"file_path": "lst.nc", "variables": ["TEMP"]},
                {"file_path": "fro.nc", "variables": frozenset({"TEMP"})},
                {"file_path": "set.nc", "variables": {"TEMP"}},
            ]
        )
        out = filter_by_variables(df, ["TEMP"])
        assert set(out["file_path"]) == {"tup.nc", "lst.nc", "fro.nc", "set.nc"}

    def test_string_variable_value_is_treated_as_malformed_and_dropped(self) -> None:
        # A plain string is a scalar in the contract — not a variable list.
        # Treat as malformed and drop on any non-empty request.
        df = pd.DataFrame([{"file_path": "str.nc", "variables": "TEMP"}])
        out = filter_by_variables(df, ["TEMP"])
        assert len(out) == 0


class TestFilterByPlatform:
    """Strict platform-type membership."""

    def test_row_with_matching_platform_kept(self) -> None:
        df = _df(_row("a.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                       platform_type="PF"))
        out = filter_by_platform(df, ["PF"])
        assert list(out["file_path"]) == ["a.nc"]

    def test_row_with_non_matching_platform_dropped(self) -> None:
        df = _df(_row("a.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                       platform_type="CT"))
        out = filter_by_platform(df, ["PF"])
        assert len(out) == 0

    def test_row_with_none_platform_dropped(self) -> None:
        # Strict membership — None is not in any list of platform types.
        df = _df(_row("a.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                       platform_type=None))
        out = filter_by_platform(df, ["PF"])
        assert len(out) == 0

    def test_multiple_types_any_match_kept(self) -> None:
        df = _df(
            _row("a.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1, platform_type="PF"),
            _row("b.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1, platform_type="CT"),
            _row("c.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1, platform_type="DB"),
        )
        out = filter_by_platform(df, ["PF", "DB"])
        assert set(out["file_path"]) == {"a.nc", "c.nc"}

    def test_empty_requested_keeps_everything(self) -> None:
        df = _df(_row("a.nc", lon_min=0, lon_max=1, lat_min=0, lat_max=1,
                       platform_type="PF"))
        out = filter_by_platform(df, [])
        assert list(out["file_path"]) == ["a.nc"]

    def test_empty_df_returns_empty(self) -> None:
        out = filter_by_platform(_df(), ["PF"])
        assert len(out) == 0
        assert list(out.columns) == _CANONICAL_COLUMNS


class TestApplyFilters:
    """AND-combine over the four axes."""

    def _all_axes_df(self) -> pd.DataFrame:
        return _df(
            _row("med_2010_pf.nc",
                  lon_min=10, lon_max=20, lat_min=35, lat_max=40,
                  time_start="2010-06-01T00:00:00Z", time_end="2010-06-30T23:59:59Z",
                  platform_type="PF", variables=("TEMP", "PSAL")),
            _row("med_2010_ct.nc",
                  lon_min=10, lon_max=20, lat_min=35, lat_max=40,
                  time_start="2010-06-01T00:00:00Z", time_end="2010-06-30T23:59:59Z",
                  platform_type="CT", variables=("TEMP",)),
            _row("nws_2010_pf.nc",
                  lon_min=-10, lon_max=0, lat_min=50, lat_max=60,
                  time_start="2010-06-01T00:00:00Z", time_end="2010-06-30T23:59:59Z",
                  platform_type="PF", variables=("TEMP",)),
            _row("med_2005_pf.nc",
                  lon_min=10, lon_max=20, lat_min=35, lat_max=40,
                  time_start="2005-06-01T00:00:00Z", time_end="2005-06-30T23:59:59Z",
                  platform_type="PF", variables=("TEMP",)),
            _row("med_2010_pf_chla.nc",
                  lon_min=10, lon_max=20, lat_min=35, lat_max=40,
                  time_start="2010-06-01T00:00:00Z", time_end="2010-06-30T23:59:59Z",
                  platform_type="PF", variables=("CHLA",)),
        )

    def test_all_none_passes_through(self) -> None:
        df = self._all_axes_df()
        out = apply_filters(df)
        pd.testing.assert_frame_equal(out, df)

    def test_bbox_only(self) -> None:
        df = self._all_axes_df()
        out = apply_filters(df, bbox=(5.0, 30.0, 25.0, 46.0))  # Mediterranean
        assert set(out["file_path"]) == {
            "med_2010_pf.nc", "med_2010_ct.nc", "med_2005_pf.nc", "med_2010_pf_chla.nc",
        }

    def test_combined_bbox_time_variables_platform(self) -> None:
        df = self._all_axes_df()
        out = apply_filters(
            df,
            bbox=(5.0, 30.0, 25.0, 46.0),
            time_range=("2010-01-01T00:00:00Z", "2010-12-31T23:59:59Z"),
            variables=["TEMP"],
            platform_types=["PF"],
        )
        # Med + 2010 + TEMP variable + PF platform = med_2010_pf.nc only
        assert list(out["file_path"]) == ["med_2010_pf.nc"]

    def test_empty_result_no_error(self) -> None:
        df = self._all_axes_df()
        out = apply_filters(df, bbox=(150.0, -10.0, 160.0, 10.0))  # Pacific
        assert len(out) == 0

    def test_empty_df_passes_through_all_filters(self) -> None:
        out = apply_filters(
            _df(),
            bbox=(5.0, 30.0, 25.0, 46.0),
            time_range=("2010-01-01T00:00:00Z", "2010-12-31T23:59:59Z"),
            variables=["TEMP"],
            platform_types=["PF"],
        )
        assert len(out) == 0
        assert list(out.columns) == _CANONICAL_COLUMNS

    def test_filter_order_does_not_affect_result(self) -> None:
        # The AND-combine is commutative; sanity test.
        df = self._all_axes_df()
        a = apply_filters(df, bbox=(5.0, 30.0, 25.0, 46.0), platform_types=["PF"])
        b = apply_filters(df, platform_types=["PF"], bbox=(5.0, 30.0, 25.0, 46.0))
        pd.testing.assert_frame_equal(a, b)


class TestSchemaConsistency:
    """cr round-1 I1: the empty-DataFrame branches in parser + filter share
    the canonical IndexRow dtypes so downstream `assert_frame_equal` doesn't
    flag spurious schema drift between parser-produced and filter-produced
    empty frames."""

    def test_filter_empty_df_has_same_time_dtype_as_parser_output(self) -> None:
        from copernicus_mcp.backends.cmems._index_parser import parse_cora_paths

        # Real parser output dtype (post-pd.to_datetime, utc=True).
        sample = parse_cora_paths(["mediterrane/2010/CO_DMQCGL01_20100101_PR_CT.nc"])
        parser_dtype = sample["time_start"].dtype

        # Empty filter output dtype.
        empty = filter_by_bbox(_df(), (-180.0, -90.0, 180.0, 90.0))
        assert empty["time_start"].dtype == parser_dtype
