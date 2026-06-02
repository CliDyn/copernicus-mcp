"""Unit tests for ``copernicus_mcp.backends.cmems._index_parser``.

Tests follow the TDD discipline pinned in the project conventions Tier A: one
behaviour per test, golden-file tests against the committed fixtures
in ``tests/fixtures/cmems_indices/``, plus targeted edge-case tests
for malformed input.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from copernicus_mcp.backends.cmems._index_parser import (
    REGION_BBOX,
    IndexRow,
    detect_format,
    parse_canyon_index_file,
    parse_cora_paths,
    parse_easycora_paths,
    parse_index,
    parse_insitu_index_file,
)
from copernicus_mcp.errors import ValidationError

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "cmems_indices"


class TestIndexRow:
    """Canonical row schema produced by every parser."""

    def _valid_kwargs(self) -> dict:
        return dict(
            file_path="path/to/file.nc",
            lon_min=-10.0,
            lon_max=10.0,
            lat_min=30.0,
            lat_max=40.0,
            time_start=datetime(2010, 1, 1, tzinfo=UTC),
            time_end=datetime(2010, 12, 31, tzinfo=UTC),
            platform_type=None,
            variables=None,
            size_bytes=None,
        )

    def test_accepts_minimal_valid_input(self) -> None:
        row = IndexRow(**self._valid_kwargs())
        assert row.file_path == "path/to/file.nc"
        assert row.lon_min == -10.0
        assert row.time_start == datetime(2010, 1, 1, tzinfo=UTC)

    def test_is_frozen(self) -> None:
        row = IndexRow(**self._valid_kwargs())
        with pytest.raises(PydanticValidationError):
            row.file_path = "different/path.nc"  # type: ignore[misc]

    def test_rejects_extra_fields(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["unknown_field"] = "boom"
        with pytest.raises(PydanticValidationError):
            IndexRow(**kwargs)

    def test_rejects_latitude_out_of_range(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["lat_min"] = -91.0
        with pytest.raises(PydanticValidationError):
            IndexRow(**kwargs)

    def test_rejects_longitude_out_of_range(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["lon_max"] = 181.0
        with pytest.raises(PydanticValidationError):
            IndexRow(**kwargs)

    def test_rejects_time_start_after_time_end(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["time_start"] = datetime(2011, 1, 1, tzinfo=UTC)
        kwargs["time_end"] = datetime(2010, 1, 1, tzinfo=UTC)
        with pytest.raises(PydanticValidationError):
            IndexRow(**kwargs)

    def test_rejects_lon_min_greater_than_lon_max(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["lon_min"] = 10.0
        kwargs["lon_max"] = -10.0
        with pytest.raises(PydanticValidationError):
            IndexRow(**kwargs)

    def test_rejects_lat_min_greater_than_lat_max(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["lat_min"] = 40.0
        kwargs["lat_max"] = 30.0
        with pytest.raises(PydanticValidationError):
            IndexRow(**kwargs)

    def test_rejects_naive_datetimes(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["time_start"] = datetime(2010, 1, 1)  # no tz
        with pytest.raises(PydanticValidationError):
            IndexRow(**kwargs)

    def test_accepts_optional_fields(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["platform_type"] = "MO"
        kwargs["variables"] = ("FCO2", "PSAL", "TEMP")
        kwargs["size_bytes"] = 12345
        row = IndexRow(**kwargs)
        assert row.platform_type == "MO"
        assert row.variables == ("FCO2", "PSAL", "TEMP")
        assert row.size_bytes == 12345


class TestRegionBbox:
    """7-entry CMEMS region → bbox lookup (PUM-013-001)."""

    EXPECTED_REGIONS = {
        "arctic",
        "baltic",
        "blacksea",
        "global",
        "mediterrane",
        "northwestshelf",
        "southwestshelf",
    }

    def test_has_all_seven_regions(self) -> None:
        assert set(REGION_BBOX.keys()) == self.EXPECTED_REGIONS

    def test_global_is_full_earth(self) -> None:
        assert REGION_BBOX["global"] == (-180.0, -90.0, 180.0, 90.0)

    def test_mediterrane_matches_spike_findings(self) -> None:
        # FINDINGS.md §2 — Mediterranean folder bbox.
        assert REGION_BBOX["mediterrane"] == (-6.0, 30.0, 42.0, 46.0)

    def test_every_entry_is_a_4_tuple_of_floats(self) -> None:
        for region, bbox in REGION_BBOX.items():
            assert isinstance(bbox, tuple), f"{region}: not a tuple"
            assert len(bbox) == 4, f"{region}: expected 4-tuple"
            for value in bbox:
                assert isinstance(value, float), f"{region}: non-float value"

    def test_bounds_are_well_ordered_and_in_range(self) -> None:
        for region, (lon_min, lat_min, lon_max, lat_max) in REGION_BBOX.items():
            assert -180.0 <= lon_min <= lon_max <= 180.0, region
            assert -90.0 <= lat_min <= lat_max <= 90.0, region


CANONICAL_COLUMNS = [
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


class TestParseInsituIndexFile:
    """``insitu_index_file_v3`` — CSV with ``#``-comment header."""

    def test_golden_fixture_parses_to_canonical_columns(self) -> None:
        raw = (_FIXTURES / "insitu_index_file_v3.txt").read_bytes()
        df = parse_insitu_index_file(raw)
        assert list(df.columns) == CANONICAL_COLUMNS

    def test_golden_fixture_has_expected_row_count(self) -> None:
        # Fixture: header + 25 first rows + ~25 trailing rows.
        # Rows 57-60 have blank lat/lon fields — those must be DROPPED
        # (not silently coerced to 0), so the parser returns fewer rows
        # than the raw line count.
        raw = (_FIXTURES / "insitu_index_file_v3.txt").read_bytes()
        df = parse_insitu_index_file(raw)
        # We don't pin an exact number — robust against fixture edits.
        # Drop must remove the 4 malformed trailing rows.
        assert len(df) > 0
        assert len(df) < 55  # 55 = total data lines including malformed

    def test_first_row_round_trips_into_index_row(self) -> None:
        raw = (_FIXTURES / "insitu_index_file_v3.txt").read_bytes()
        df = parse_insitu_index_file(raw)
        first = df.iloc[0]
        # Validate the first row through IndexRow — schema must accept it.
        row = IndexRow(
            file_path=first["file_path"],
            lon_min=float(first["lon_min"]),
            lon_max=float(first["lon_max"]),
            lat_min=float(first["lat_min"]),
            lat_max=float(first["lat_max"]),
            time_start=first["time_start"].to_pydatetime(),
            time_end=first["time_end"].to_pydatetime(),
            platform_type=first["platform_type"],
            variables=first["variables"],
            size_bytes=None,
        )
        assert row.file_path.endswith(".nc")

    def test_variables_column_is_tuple_of_strings(self) -> None:
        raw = (_FIXTURES / "insitu_index_file_v3.txt").read_bytes()
        df = parse_insitu_index_file(raw)
        for variables in df["variables"]:
            assert isinstance(variables, tuple)
            assert all(isinstance(v, str) for v in variables)
            # First-row fixture has "FCO2 PSAL TEMP" → 3 variables.
            assert len(variables) >= 1

    def test_time_columns_are_utc_datetimes(self) -> None:
        raw = (_FIXTURES / "insitu_index_file_v3.txt").read_bytes()
        df = parse_insitu_index_file(raw)
        # pandas datetime64[us, UTC] OR datetime objects — both should
        # represent UTC.
        assert pd.api.types.is_datetime64_any_dtype(df["time_start"])
        assert pd.api.types.is_datetime64_any_dtype(df["time_end"])
        # tz must be UTC.
        assert str(df["time_start"].dt.tz) in ("UTC", "tzutc()")

    def test_lat_lon_within_canonical_ranges(self) -> None:
        raw = (_FIXTURES / "insitu_index_file_v3.txt").read_bytes()
        df = parse_insitu_index_file(raw)
        assert (df["lat_min"] >= -90.0).all()
        assert (df["lat_max"] <= 90.0).all()
        assert (df["lon_min"] >= -180.0).all()
        assert (df["lon_max"] <= 180.0).all()
        assert (df["lat_min"] <= df["lat_max"]).all()
        assert (df["lon_min"] <= df["lon_max"]).all()

    def test_time_start_le_time_end(self) -> None:
        raw = (_FIXTURES / "insitu_index_file_v3.txt").read_bytes()
        df = parse_insitu_index_file(raw)
        assert (df["time_start"] <= df["time_end"]).all()

    def test_size_bytes_is_none_when_absent(self) -> None:
        raw = (_FIXTURES / "insitu_index_file_v3.txt").read_bytes()
        df = parse_insitu_index_file(raw)
        # Source CSV has no size column, so every row's size is None.
        assert df["size_bytes"].isna().all()

    def test_platform_type_is_none_when_absent(self) -> None:
        # insitu_index_file_v3 doesn't carry a platform column; the
        # platform_type field of every row must be None.
        raw = (_FIXTURES / "insitu_index_file_v3.txt").read_bytes()
        df = parse_insitu_index_file(raw)
        assert df["platform_type"].isna().all()

    def test_empty_input_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            parse_insitu_index_file(b"")

    def test_truncated_header_raises_validation_error(self) -> None:
        # Comment header but no CSV header line.
        with pytest.raises(ValidationError):
            parse_insitu_index_file(b"# Title : in-situ files catalog\n")

    def test_missing_required_columns_raises_validation_error(self) -> None:
        # CSV header missing the mandatory geospatial_lat_min column.
        raw = (
            b"# Title : in-situ files catalog\n"
            b"product_id,file_name,time_coverage_start,time_coverage_end\n"
            b"P,f.nc,2020-01-01T00:00:00Z,2020-12-31T23:59:59Z\n"
        )
        with pytest.raises(ValidationError):
            parse_insitu_index_file(raw)

    def test_unknown_format_raises_validation_error(self) -> None:
        # Not an insitu header at all.
        with pytest.raises(ValidationError):
            parse_insitu_index_file(b"random gibberish without insitu header")

    def test_non_utf8_bytes_raise_validation_error(self) -> None:
        # Random binary blob.
        with pytest.raises(ValidationError):
            parse_insitu_index_file(b"\xff\xfe\x00\x00\x01\x02")


def _load_cora_fixture_paths() -> list[str]:
    """Read the CORA fixture, return non-comment, non-empty lines."""
    raw = (_FIXTURES / "cora_path_listing.txt").read_text()
    return [line for line in raw.splitlines() if line and not line.startswith("#")]


class TestParseCoraPaths:
    """``cora_path_v1`` — paths of form ``<region>/<year>/CO_DMQCGL01_YYYYMMDD_TYPE_PLATFORM.nc``."""

    def test_single_path_produces_one_row(self) -> None:
        df = parse_cora_paths(["mediterrane/2010/CO_DMQCGL01_20100101_PR_CT.nc"])
        assert len(df) == 1

    def test_canonical_columns(self) -> None:
        df = parse_cora_paths(["mediterrane/2010/CO_DMQCGL01_20100101_PR_CT.nc"])
        assert list(df.columns) == CANONICAL_COLUMNS

    def test_mediterrane_path_maps_to_med_bbox(self) -> None:
        df = parse_cora_paths(["mediterrane/2010/CO_DMQCGL01_20100101_PR_CT.nc"])
        med_lon_min, med_lat_min, med_lon_max, med_lat_max = REGION_BBOX["mediterrane"]
        row = df.iloc[0]
        assert row["lon_min"] == med_lon_min
        assert row["lat_min"] == med_lat_min
        assert row["lon_max"] == med_lon_max
        assert row["lat_max"] == med_lat_max

    def test_filename_date_maps_to_time_range(self) -> None:
        df = parse_cora_paths(["mediterrane/2010/CO_DMQCGL01_20100315_PR_CT.nc"])
        row = df.iloc[0]
        assert row["time_start"] == pd.Timestamp("2010-03-15", tz="UTC")
        # End-of-day boundary so per-day filtering captures the full date.
        assert row["time_end"] == pd.Timestamp("2010-03-15 23:59:59", tz="UTC")

    def test_platform_code_extracted(self) -> None:
        df = parse_cora_paths(["mediterrane/2010/CO_DMQCGL01_20100101_PR_CT.nc"])
        assert df.iloc[0]["platform_type"] == "CT"

    def test_variables_none_for_path_based(self) -> None:
        df = parse_cora_paths(["mediterrane/2010/CO_DMQCGL01_20100101_PR_CT.nc"])
        assert df.iloc[0]["variables"] is None

    def test_size_bytes_none_for_path_based(self) -> None:
        df = parse_cora_paths(["mediterrane/2010/CO_DMQCGL01_20100101_PR_CT.nc"])
        assert pd.isna(df.iloc[0]["size_bytes"])

    def test_file_path_preserved_verbatim(self) -> None:
        path = "mediterrane/2010/CO_DMQCGL01_20100101_PR_CT.nc"
        df = parse_cora_paths([path])
        assert df.iloc[0]["file_path"] == path

    def test_all_seven_regions_resolve(self) -> None:
        paths = [
            f"{region}/2020/CO_DMQCGL01_20200101_PR_CT.nc"
            for region in REGION_BBOX
        ]
        df = parse_cora_paths(paths)
        assert len(df) == len(REGION_BBOX)

    def test_empty_input_returns_empty_dataframe(self) -> None:
        df = parse_cora_paths([])
        assert len(df) == 0
        assert list(df.columns) == CANONICAL_COLUMNS

    def test_unknown_region_drops_row(self) -> None:
        # Unknown region prefix → cannot map to bbox → row is dropped.
        # (Not raised — we tolerate stray paths in a 1M-file listing.)
        df = parse_cora_paths(
            [
                "atlantis/2010/CO_DMQCGL01_20100101_PR_CT.nc",
                "mediterrane/2010/CO_DMQCGL01_20100101_PR_CT.nc",
            ]
        )
        assert len(df) == 1
        assert df.iloc[0]["file_path"].startswith("mediterrane/")

    def test_malformed_filename_drops_row(self) -> None:
        df = parse_cora_paths(
            [
                "mediterrane/2010/random_blob.nc",
                "mediterrane/2010/CO_DMQCGL01_20100101_PR_CT.nc",
            ]
        )
        assert len(df) == 1

    def test_invalid_date_drops_row(self) -> None:
        df = parse_cora_paths(
            [
                "mediterrane/2010/CO_DMQCGL01_20100231_PR_CT.nc",  # Feb 31 invalid
                "mediterrane/2010/CO_DMQCGL01_20100101_PR_CT.nc",
            ]
        )
        assert len(df) == 1

    def test_rejects_ecocora_prefix(self) -> None:
        # ECO_ paths must be parsed by parse_easycora_paths, not this function.
        df = parse_cora_paths(["mediterrane/2010/ECO_DMQCGL01_20100101_PR_CT.nc"])
        assert len(df) == 0

    def test_golden_fixture_parses(self) -> None:
        # The mixed fixture has both CO_ and ECO_ paths; parse_cora_paths
        # only accepts CO_ rows.
        paths = _load_cora_fixture_paths()
        cora_paths = [p for p in paths if "/CO_" in p]
        df = parse_cora_paths(paths)
        assert len(df) == len(cora_paths)
        # All produced rows must have valid CMEMS regions.
        for region_dir in df["file_path"].str.split("/").str[0]:
            assert region_dir in REGION_BBOX

    def test_all_rows_satisfy_index_row_contract(self) -> None:
        paths = _load_cora_fixture_paths()
        df = parse_cora_paths(paths)
        for _, row in df.iterrows():
            IndexRow(
                file_path=row["file_path"],
                lon_min=float(row["lon_min"]),
                lon_max=float(row["lon_max"]),
                lat_min=float(row["lat_min"]),
                lat_max=float(row["lat_max"]),
                time_start=row["time_start"].to_pydatetime(),
                time_end=row["time_end"].to_pydatetime(),
                platform_type=row["platform_type"],
                variables=row["variables"],
                size_bytes=None,
            )


class TestParseEasyCoraPaths:
    """``easycora_path_v1`` — same structure as CORA with ``ECO_`` prefix."""

    def test_canonical_columns(self) -> None:
        df = parse_easycora_paths(["mediterrane/2015/ECO_DMQCGL01_20150601_PR_CT.nc"])
        assert list(df.columns) == CANONICAL_COLUMNS

    def test_eco_prefix_path_parses(self) -> None:
        df = parse_easycora_paths(["mediterrane/2015/ECO_DMQCGL01_20150601_PR_CT.nc"])
        assert len(df) == 1
        assert df.iloc[0]["file_path"] == "mediterrane/2015/ECO_DMQCGL01_20150601_PR_CT.nc"

    def test_rejects_co_prefix(self) -> None:
        # CO_ paths belong to parse_cora_paths.
        df = parse_easycora_paths(["mediterrane/2015/CO_DMQCGL01_20150601_PR_CT.nc"])
        assert len(df) == 0

    def test_region_bbox_mapping(self) -> None:
        df = parse_easycora_paths(["arctic/2020/ECO_DMQCGL01_20200201_PR_CT.nc"])
        lon_min, lat_min, lon_max, lat_max = REGION_BBOX["arctic"]
        assert df.iloc[0]["lon_min"] == lon_min
        assert df.iloc[0]["lat_max"] == lat_max

    def test_date_extraction(self) -> None:
        df = parse_easycora_paths(["global/2024/ECO_DMQCGL01_20241231_PR_PF.nc"])
        assert df.iloc[0]["time_start"] == pd.Timestamp("2024-12-31", tz="UTC")

    def test_platform_extraction(self) -> None:
        df = parse_easycora_paths(["global/2010/ECO_DMQCGL01_20100715_PR_PF.nc"])
        assert df.iloc[0]["platform_type"] == "PF"

    def test_empty_input_returns_empty_dataframe(self) -> None:
        df = parse_easycora_paths([])
        assert len(df) == 0
        assert list(df.columns) == CANONICAL_COLUMNS

    def test_golden_fixture_eco_rows_only(self) -> None:
        paths = _load_cora_fixture_paths()
        eco_count = sum(1 for p in paths if "/ECO_" in p)
        df = parse_easycora_paths(paths)
        assert len(df) == eco_count


class TestDetectFormat:
    """``detect_format`` discriminates only by-bytes formats."""

    def test_insitu_header_detected(self) -> None:
        raw = (_FIXTURES / "insitu_index_file_v3.txt").read_bytes()
        assert detect_format(raw) == "insitu_index_file_v3"

    def test_insitu_header_detected_with_crlf(self) -> None:
        # CRLF endings show up on Windows-produced uploads.
        raw = b"# Title : in-situ files catalog\r\n# other\r\nproduct_id,file_name\r\n"
        assert detect_format(raw) == "insitu_index_file_v3"

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValidationError):
            detect_format(b"")

    def test_unknown_header_raises(self) -> None:
        with pytest.raises(ValidationError):
            detect_format(b"random text\nthat has no header\n")

    def test_non_utf8_bytes_raise(self) -> None:
        with pytest.raises(ValidationError):
            detect_format(b"\xff\xfe\x00\x00")

    def test_canyon_header_detected(self) -> None:
        raw = (_FIXTURES / "multiobs_canyon_v1.txt").read_bytes()
        assert detect_format(raw) == "multiobs_canyon_v1"


class TestParseCanyonIndexFile:
    """``multiobs_canyon_v1`` — point observations aggregated by file_name."""

    def test_golden_fixture_parses_to_canonical_columns(self) -> None:
        raw = (_FIXTURES / "multiobs_canyon_v1.txt").read_bytes()
        df = parse_canyon_index_file(raw)
        assert list(df.columns) == CANONICAL_COLUMNS

    def test_multiple_rows_per_file_aggregate_to_single_row(self) -> None:
        # Fixture has 50 data rows but only a few unique file_names —
        # rows sharing the same file_name aggregate into one IndexRow
        # (bbox = min/max of all observations in that file).
        raw = (_FIXTURES / "multiobs_canyon_v1.txt").read_bytes()
        df = parse_canyon_index_file(raw)
        assert len(df) > 0
        assert df["file_path"].is_unique

    def test_lon_lat_aggregation_uses_min_max(self) -> None:
        # Synthetic: 2 rows in same file with different points → row's
        # bbox spans both.
        raw = (
            b"# title : Nutrient and Carbon profiles vertical distribution\n"
            b"file_name,date,wmo,n_cycle,longitude,latitude,"
            b"temp_dt_mode,psal_dt_mode,doxy_dt_mode,date_update\n"
            b"x.nc,2010-01-01 00:00:00,1,1,-10.0,30.0,D,D,D,20240101\n"
            b"x.nc,2010-01-02 00:00:00,1,2,5.0,40.0,D,D,D,20240101\n"
        )
        df = parse_canyon_index_file(raw)
        assert len(df) == 1
        assert df.iloc[0]["lon_min"] == -10.0
        assert df.iloc[0]["lon_max"] == 5.0
        assert df.iloc[0]["lat_min"] == 30.0
        assert df.iloc[0]["lat_max"] == 40.0

    def test_time_aggregation_uses_min_max(self) -> None:
        raw = (
            b"# title : Nutrient and Carbon profiles vertical distribution\n"
            b"file_name,date,wmo,n_cycle,longitude,latitude,"
            b"temp_dt_mode,psal_dt_mode,doxy_dt_mode,date_update\n"
            b"x.nc,2010-01-01 12:00:00,1,1,0.0,0.0,D,D,D,20240101\n"
            b"x.nc,2010-12-31 23:00:00,1,2,0.0,0.0,D,D,D,20240101\n"
        )
        df = parse_canyon_index_file(raw)
        assert len(df) == 1
        assert df.iloc[0]["time_start"] == pd.Timestamp("2010-01-01 12:00:00", tz="UTC")
        assert df.iloc[0]["time_end"] == pd.Timestamp("2010-12-31 23:00:00", tz="UTC")

    def test_variables_derived_from_dt_mode_columns(self) -> None:
        # *_dt_mode columns indicate which variables are present.
        # Non-empty mode → variable name in the canonical tuple
        # (TEMP, PSAL, DOXY).
        raw = (
            b"# title : Nutrient and Carbon profiles vertical distribution\n"
            b"file_name,date,wmo,n_cycle,longitude,latitude,"
            b"temp_dt_mode,psal_dt_mode,doxy_dt_mode,date_update\n"
            b"x.nc,2010-01-01 00:00:00,1,1,0.0,0.0,D,D,D,20240101\n"
            b"y.nc,2010-01-01 00:00:00,1,1,0.0,0.0,D,D,,20240101\n"
            b"z.nc,2010-01-01 00:00:00,1,1,0.0,0.0,,,A,20240101\n"
        )
        df = parse_canyon_index_file(raw).sort_values("file_path").reset_index(drop=True)
        # x.nc has all three vars; y has TEMP+PSAL; z has only DOXY.
        x_vars = df.iloc[df.index[df["file_path"] == "x.nc"][0]]["variables"]
        y_vars = df.iloc[df.index[df["file_path"] == "y.nc"][0]]["variables"]
        z_vars = df.iloc[df.index[df["file_path"] == "z.nc"][0]]["variables"]
        assert set(x_vars) == {"TEMP", "PSAL", "DOXY"}
        assert set(y_vars) == {"TEMP", "PSAL"}
        assert set(z_vars) == {"DOXY"}

    def test_whitespace_only_dt_mode_is_absent(self) -> None:
        """Codex retro PR #113 LOW: ``na_values=[""]`` only nulls EXACT
        empty fields; a single space ``" "`` passes through and
        ``.notna()`` returns True, so the variable would be falsely
        flagged as present. Whitespace-only values must be treated as
        absent (no QC flag → variable not measured)."""
        raw = (
            b"# title : Nutrient and Carbon profiles vertical distribution\n"
            b"file_name,date,wmo,n_cycle,longitude,latitude,"
            b"temp_dt_mode,psal_dt_mode,doxy_dt_mode,date_update\n"
            b"a.nc,2010-01-01 00:00:00,1,1,0.0,0.0, , , ,20240101\n"
            b"b.nc,2010-01-01 00:00:00,1,1,0.0,0.0,D,\t,  ,20240101\n"
        )
        df = parse_canyon_index_file(raw).sort_values("file_path").reset_index(drop=True)
        a_vars = df.iloc[df.index[df["file_path"] == "a.nc"][0]]["variables"]
        b_vars = df.iloc[df.index[df["file_path"] == "b.nc"][0]]["variables"]
        # a: all three columns whitespace-only → no variables. The
        # parser's _variables helper returns None when the tuple is
        # empty (matches the schema's ``variables: tuple | None``).
        assert a_vars is None
        # b: only temp has a real value; psal (tab) and doxy (spaces)
        # are whitespace-only and must be absent.
        assert set(b_vars) == {"TEMP"}

    def test_platform_and_size_are_none(self) -> None:
        # canyon format has WMO platform codes but no canonical
        # platform_type mapping; per-file aggregation can't pick one.
        # size_bytes is not in the format.
        raw = (_FIXTURES / "multiobs_canyon_v1.txt").read_bytes()
        df = parse_canyon_index_file(raw)
        for value in df["platform_type"]:
            assert value is None
        for value in df["size_bytes"]:
            assert pd.isna(value) or value is None

    def test_all_rows_satisfy_index_row_contract(self) -> None:
        raw = (_FIXTURES / "multiobs_canyon_v1.txt").read_bytes()
        df = parse_canyon_index_file(raw)
        for _, row in df.iterrows():
            IndexRow(
                file_path=row["file_path"],
                lon_min=float(row["lon_min"]),
                lon_max=float(row["lon_max"]),
                lat_min=float(row["lat_min"]),
                lat_max=float(row["lat_max"]),
                time_start=row["time_start"].to_pydatetime(),
                time_end=row["time_end"].to_pydatetime(),
                platform_type=row["platform_type"],
                variables=row["variables"],
                size_bytes=None,
            )

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValidationError):
            parse_canyon_index_file(b"")

    def test_missing_required_columns_raises(self) -> None:
        raw = (
            b"# title : Nutrient and Carbon profiles vertical distribution\n"
            b"file_name,date\n"
            b"x.nc,2010-01-01 00:00:00\n"
        )
        with pytest.raises(ValidationError):
            parse_canyon_index_file(raw)

    def test_parse_index_dispatches_via_detect(self) -> None:
        # Auto-detection through parse_index without explicit format.
        raw = (_FIXTURES / "multiobs_canyon_v1.txt").read_bytes()
        df = parse_index(raw)
        assert list(df.columns) == CANONICAL_COLUMNS
        assert len(df) > 0

    def test_parse_index_dispatches_with_explicit_format(self) -> None:
        raw = (_FIXTURES / "multiobs_canyon_v1.txt").read_bytes()
        df = parse_index(raw, expected_format="multiobs_canyon_v1")
        assert list(df.columns) == CANONICAL_COLUMNS


class TestParseIndex:
    """``parse_index`` entry point — dispatches on expected_format or detects."""

    def test_dispatches_to_insitu_with_explicit_format(self) -> None:
        raw = (_FIXTURES / "insitu_index_file_v3.txt").read_bytes()
        df = parse_index(raw, expected_format="insitu_index_file_v3")
        # Same shape as calling parse_insitu_index_file directly.
        direct = parse_insitu_index_file(raw)
        pd.testing.assert_frame_equal(df.reset_index(drop=True), direct.reset_index(drop=True))

    def test_auto_detects_insitu_format(self) -> None:
        raw = (_FIXTURES / "insitu_index_file_v3.txt").read_bytes()
        df = parse_index(raw)
        assert list(df.columns) == CANONICAL_COLUMNS
        assert len(df) > 0

    def test_path_based_format_not_supported_via_bytes_path(self) -> None:
        # cora_path_v1 / easycora_path_v1 are produced from a path listing,
        # not from raw bytes. Asking parse_index for them must reject —
        # the SDK call sites use parse_cora_paths/parse_easycora_paths
        # directly with the list[str] from dry_run output.
        with pytest.raises(ValidationError):
            parse_index(b"# Title : in-situ files catalog\n", expected_format="cora_path_v1")

    def test_path_based_easycora_rejected_via_bytes_path(self) -> None:
        with pytest.raises(ValidationError):
            parse_index(b"# Title : in-situ files catalog\n", expected_format="easycora_path_v1")

    def test_explicit_format_skips_detection(self) -> None:
        # Raw bytes WITHOUT the insitu header but tagged as insitu_index_file_v3.
        # parse_index must dispatch directly to the INSITU parser, which then
        # rejects the malformed payload via missing-column ValidationError.
        # The point: detection is skipped, so detect_format's header check
        # would not run.
        raw = b"product_id,file_name,geospatial_lat_min\nP,f.nc,10\n"
        with pytest.raises(ValidationError):
            # Will fail validation (missing other required cols), not detection.
            parse_index(raw, expected_format="insitu_index_file_v3")

    def test_unknown_format_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            parse_index(b"random gibberish")

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValidationError):
            parse_index(b"")

    def test_unknown_expected_format_string_raises(self) -> None:
        # Defensive: callers should pass only valid FormatId values, but
        # parse_index should fail loudly on garbage rather than silently
        # passing through.
        with pytest.raises(ValidationError):
            parse_index(b"# Title : in-situ files catalog\nx\n", expected_format="not_a_format")  # type: ignore[arg-type]


class TestRoundOneFixes:
    """Regression coverage for cr round-1 HIGH + MEDIUM findings."""

    # C1 — per-row IndexRow contract enforced on the INSITU path.

    def _insitu_csv(self, *data_rows: str) -> bytes:
        header = (
            "# Title : in-situ files catalog\n"
            "product_id,file_name,geospatial_lat_min,geospatial_lat_max,"
            "geospatial_lon_min,geospatial_lon_max,time_coverage_start,"
            "time_coverage_end,institution,date_update,data_mode,parameters\n"
        )
        return (header + "\n".join(data_rows) + "\n").encode("utf-8")

    # T-CMEMS-GET-INDEX-006 policy reversal: contract-violating rows are
    # DROPPED with a structured log warning rather than raising the entire
    # parse. Real CMEMS indices contain a small fraction of malformed rows
    # (e.g. ship tracks with lon_max > 180); rejecting whole datasets is
    # too brittle. The C1 cr-round-1 concern ("malformed rows must not
    # propagate into filter results") is preserved by dropping them.

    def test_C1_out_of_range_latitude_is_dropped(self) -> None:
        raw = self._insitu_csv(
            "P,f.nc,200,300,0,10,2020-01-01T00:00:00Z,2020-12-31T23:59:59Z,,,,FCO2",
            "P,g.nc,0,10,0,10,2020-01-01T00:00:00Z,2020-12-31T23:59:59Z,,,,FCO2",
        )
        df = parse_insitu_index_file(raw)
        assert list(df["file_path"]) == ["g.nc"]

    def test_C1_out_of_range_longitude_is_dropped(self) -> None:
        raw = self._insitu_csv(
            "P,bad.nc,0,10,-200,200,2020-01-01T00:00:00Z,2020-12-31T23:59:59Z,,,,FCO2",
            "P,good.nc,0,10,-10,10,2020-01-01T00:00:00Z,2020-12-31T23:59:59Z,,,,FCO2",
        )
        df = parse_insitu_index_file(raw)
        assert list(df["file_path"]) == ["good.nc"]

    def test_C1_inverted_time_range_is_dropped(self) -> None:
        raw = self._insitu_csv(
            "P,bad.nc,0,10,0,10,2021-01-01T00:00:00Z,2020-01-01T00:00:00Z,,,,FCO2",
            "P,good.nc,0,10,0,10,2020-01-01T00:00:00Z,2020-12-31T23:59:59Z,,,,FCO2",
        )
        df = parse_insitu_index_file(raw)
        assert list(df["file_path"]) == ["good.nc"]

    def test_C1_inverted_lon_range_is_dropped(self) -> None:
        raw = self._insitu_csv(
            "P,bad.nc,0,10,10,-10,2020-01-01T00:00:00Z,2020-12-31T23:59:59Z,,,,FCO2",
            "P,good.nc,0,10,-10,10,2020-01-01T00:00:00Z,2020-12-31T23:59:59Z,,,,FCO2",
        )
        df = parse_insitu_index_file(raw)
        assert list(df["file_path"]) == ["good.nc"]

    def test_C1_inverted_lat_range_is_dropped(self) -> None:
        raw = self._insitu_csv(
            "P,bad.nc,40,30,0,10,2020-01-01T00:00:00Z,2020-12-31T23:59:59Z,,,,FCO2",
            "P,good.nc,30,40,0,10,2020-01-01T00:00:00Z,2020-12-31T23:59:59Z,,,,FCO2",
        )
        df = parse_insitu_index_file(raw)
        assert list(df["file_path"]) == ["good.nc"]

    def test_C1_logs_warning_when_dropping_violations(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging
        raw = self._insitu_csv(
            "P,bad.nc,0,10,-200,200,2020-01-01T00:00:00Z,2020-12-31T23:59:59Z,,,,FCO2",
            "P,good.nc,0,10,-10,10,2020-01-01T00:00:00Z,2020-12-31T23:59:59Z,,,,FCO2",
        )
        with caplog.at_level(logging.WARNING):
            parse_insitu_index_file(raw)
        assert any(
            "violating IndexRow contract" in record.getMessage()
            for record in caplog.records
        )

    def test_C1_all_rows_violating_raises(self) -> None:
        # cr round-1 I2: 100% violation signals a misclassified format
        # (we shouldn't silently cache an empty DataFrame).
        raw = self._insitu_csv(
            "P,bad1.nc,0,10,-200,200,2020-01-01T00:00:00Z,2020-12-31T23:59:59Z,,,,FCO2",
            "P,bad2.nc,200,300,0,10,2020-01-01T00:00:00Z,2020-12-31T23:59:59Z,,,,FCO2",
        )
        with pytest.raises(ValidationError, match="every row"):
            parse_insitu_index_file(raw)

    # C2 — _CORA_PATH_RE must not accept a trailing newline.

    def test_C2_cora_path_with_trailing_newline_is_dropped(self) -> None:
        # A real listing from splitlines() won't include trailing newlines,
        # but ad-hoc inputs that pass "raw.split('\\n')" would. Either way,
        # the file_path must never carry stray whitespace.
        df = parse_cora_paths(["mediterrane/2010/CO_DMQCGL01_20100101_PR_CT.nc\n"])
        assert len(df) == 0

    def test_C2_easycora_path_with_trailing_newline_is_dropped(self) -> None:
        df = parse_easycora_paths(["mediterrane/2010/ECO_DMQCGL01_20100101_PR_CT.nc\n"])
        assert len(df) == 0

    # M1 — naive datetimes must not be silently treated as UTC.

    def test_M1_naive_datetime_row_is_dropped(self) -> None:
        # An INSITU row without Z (or explicit offset) is treated as missing
        # data and dropped — not silently coerced to UTC.
        raw = self._insitu_csv(
            "P,f.nc,0,10,0,10,2020-01-01T00:00:00,2020-12-31T23:59:59,,,,FCO2"
        )
        df = parse_insitu_index_file(raw)
        assert len(df) == 0

    def test_M1_explicit_offset_accepted_and_normalised(self) -> None:
        # Non-Z timezones must work too.
        raw = self._insitu_csv(
            "P,f.nc,0,10,0,10,2020-01-01T02:00:00+02:00,2020-12-31T02:00:00+02:00,,,,FCO2"
        )
        df = parse_insitu_index_file(raw)
        assert len(df) == 1
        assert df.iloc[0]["time_start"] == pd.Timestamp("2020-01-01T00:00:00Z")

    # M2 — detect_format must not misclassify the platforms catalog.

    def test_M2_platforms_catalog_not_detected_as_file_catalog(self) -> None:
        raw = b"# Title : in-situ platforms catalog\n# columns\nplatform_code\nP\n"
        with pytest.raises(ValidationError):
            detect_format(raw)

    def test_M2_files_catalog_still_detected(self) -> None:
        # Tightened marker must still accept the canonical INSITU header.
        raw = b"# Title : in-situ files catalog\nproduct_id\n"
        assert detect_format(raw) == "insitu_index_file_v3"

    # M3 — pin the expected INSITU row count.

    def test_M3_insitu_fixture_drops_exactly_four_malformed_rows(self) -> None:
        # Fixture: 54 data rows, 4 trailing rows with empty bbox values.
        # The parser must drop exactly those 4 and keep the other 50.
        raw = (_FIXTURES / "insitu_index_file_v3.txt").read_bytes()
        df = parse_insitu_index_file(raw)
        assert len(df) == 50


class TestRoundTwoFixes:
    """Regression coverage for cr round-2 HIGH findings."""

    def _insitu_csv(self, *data_rows: str) -> bytes:
        header = (
            "# Title : in-situ files catalog\n"
            "product_id,file_name,geospatial_lat_min,geospatial_lat_max,"
            "geospatial_lon_min,geospatial_lon_max,time_coverage_start,"
            "time_coverage_end,institution,date_update,data_mode,parameters\n"
        )
        return (header + "\n".join(data_rows) + "\n").encode("utf-8")

    def test_H1_mixed_precision_timestamps_both_survive(self) -> None:
        # Without ``format="mixed"`` pandas infers a single format from the
        # first row and silently NaTs every other shape — losing real data.
        # The parser must accept whole-second AND sub-second rows in the
        # same series.
        raw = self._insitu_csv(
            "P,a.nc,0,10,0,10,2020-01-01T00:00:00Z,2020-12-31T23:59:59Z,,,,FCO2",
            "P,b.nc,0,10,0,10,2020-01-01T00:00:00.123Z,2020-12-31T23:59:59.456Z,,,,FCO2",
        )
        df = parse_insitu_index_file(raw)
        assert len(df) == 2
        files = set(df["file_path"])
        assert files == {"a.nc", "b.nc"}

    def test_H1_mixed_offset_and_z_both_survive(self) -> None:
        # Three distinct ISO 8601 shapes in one series — Z, ±HH:MM, ±HHMM.
        # All three must produce surviving rows.
        raw = self._insitu_csv(
            "P,a.nc,0,10,0,10,2020-01-01T00:00:00Z,2020-12-31T00:00:00Z,,,,FCO2",
            "P,b.nc,0,10,0,10,2020-01-01T02:00:00+02:00,2020-12-31T02:00:00+02:00,,,,FCO2",
            "P,c.nc,0,10,0,10,2020-01-01T02:00:00+0200,2020-12-31T02:00:00+0200,,,,FCO2",
        )
        df = parse_insitu_index_file(raw)
        assert len(df) == 3

    def test_R3_non_iso_date_format_is_rejected(self) -> None:
        # ``format="mixed"`` would have silently accepted ``01/02/2020T00:00:00Z``
        # as Jan 2 2020. ``format="ISO8601"`` rejects it. Real CMEMS files
        # emit ISO 8601 exclusively; this guard catches future format drift.
        raw = self._insitu_csv(
            "P,a.nc,0,10,0,10,01/02/2020T00:00:00Z,01/02/2020T23:59:59Z,,,,FCO2",
        )
        df = parse_insitu_index_file(raw)
        assert len(df) == 0
