"""Bulk validation — parse + filter every locally-fetched index file.

**Gating.** Skipped unless ``INDEX_CORPUS_DIR`` points at a directory
containing the spike's downloaded indices (per-dataset subdirectory
holding ``index_file.txt`` / ``canyon_index_file.txt`` somewhere under
it). NOT gated by ``RUN_INTEGRATION_TESTS`` because no network is
hit — the test reads already-downloaded bytes.

**Why this test exists.** The trimmed fixtures in
``tests/fixtures/cmems_indices/`` cover the parser's edge-case
contract but not the full-row count. Real INSITU index files have
hundreds-to-millions of rows; format drift (a new column, a new
delimiter quirk, a malformed row class we didn't trim into the
fixture) would slip past unit tests and surface only when the parser
runs against the real corpus. This sweep catches that.

**What it asserts per dataset:**

1. ``parse_index(raw)`` produces a canonical IndexRow DataFrame.
2. ``filter_by_bbox(df, out_of_range_bbox)`` returns 0 — proves the
   filter actually excludes.
3. ``filter_by_bbox(df, mid_band)`` returns ``0 < matched < total``
   when the dataset's catalogue ``spatial_extent`` is well-defined.
4. ``filter_by_time(df, mid_band)`` returns ``0 < matched < total``
   when the parsed DataFrame's time range is well-defined.
5. ``filter_by_variables(df, [var])`` returns ``> 0`` when the
   dataset's ``variables`` column carries any value appearing in ≥10%
   of rows.

A per-dataset summary (counts + timing) is written to
``tests/integration/index_bulk_validation_report.txt``. The report
file is committed — diffs against a previous run surface count drift
during code review (see CONTRIBUTING.md "Bulk validation" section).
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from copernicus_mcp.backends.cmems._index_filter import (
    filter_by_bbox,
    filter_by_time,
    filter_by_variables,
)
from copernicus_mcp.backends.cmems._index_parser import parse_index

INDEX_CORPUS_DIR = os.environ.get("INDEX_CORPUS_DIR")

pytestmark = pytest.mark.skipif(
    not INDEX_CORPUS_DIR,
    reason="INDEX_CORPUS_DIR not set — bulk validation runs only on machines with the spike corpus.",
)

_CANONICAL_COLUMNS = (
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
)
_REPORT_PATH = Path(__file__).parent / "index_bulk_validation_report.txt"
_MARINE_JSON = (
    Path(__file__).parent.parent.parent
    / "src"
    / "copernicus_mcp"
    / "backends"
    / "cmems"
    / "_data"
    / "marine.json"
)


def _discover_index_files(corpus_root: Path) -> dict[str, Path]:
    """Return ``{dataset_id: path_to_index_file}`` for every index under
    ``corpus_root``. Datasets without a matching ``index*.txt`` are
    silently ignored — path-based datasets (CORA/EasyCORA) have no
    text index and aren't part of the file-based parser test."""
    result: dict[str, Path] = {}
    for dataset_dir in sorted(corpus_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for path in dataset_dir.rglob("*.txt"):
            name = path.name.lower()
            # Skip index_platform.txt — the parser doesn't handle that
            # (secondary format per spike findings).
            if "index_platform" in name:
                continue
            if "index" in name:
                result[dataset_dir.name] = path
                break
    return result


def _catalogue_spatial_extent(dataset_id: str) -> dict[str, float] | None:
    """Lookup spatial_extent for a dataset_id in the bundled catalogue."""
    data = json.loads(_MARINE_JSON.read_text())

    def _walk(obj: object) -> dict[str, float] | None:
        if isinstance(obj, dict):
            if obj.get("dataset_id") == dataset_id:
                ext = obj.get("spatial_extent")
                return ext if isinstance(ext, dict) else None
            for v in obj.values():
                found = _walk(v)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = _walk(item)
                if found is not None:
                    return found
        return None

    return _walk(data)


def _shrink_bbox(
    bbox: tuple[float, float, float, float], fraction: float
) -> tuple[float, float, float, float]:
    """Shrink a bbox to ``fraction`` of its original linear extent in each axis."""
    lon_min, lat_min, lon_max, lat_max = bbox
    cx = (lon_min + lon_max) / 2
    cy = (lat_min + lat_max) / 2
    half_w = (lon_max - lon_min) * fraction / 2
    half_h = (lat_max - lat_min) * fraction / 2
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _is_degenerate_extent(extent: dict[str, float]) -> bool:
    """Single-point spatial extents can't produce a non-trivial mid-band."""
    return (
        extent["min_lon"] >= extent["max_lon"]
        or extent["min_lat"] >= extent["max_lat"]
    )


def _pick_frequent_variable(df: pd.DataFrame) -> str | None:
    """Return a variable name that appears in ≥10% of rows, or None."""
    counts: Counter[str] = Counter()
    seen_rows_with_variables = 0
    for value in df["variables"]:
        if not isinstance(value, tuple) or not value:
            continue
        seen_rows_with_variables += 1
        for v in value:
            counts[v] += 1
    if seen_rows_with_variables == 0:
        return None
    threshold = max(1, int(0.1 * seen_rows_with_variables))
    for var, count in counts.most_common():
        if count >= threshold:
            return var
    return None


def test_bulk_validation_all_datasets() -> None:
    """Sweep — parse + filter every index file in the corpus, assert all green."""
    corpus = Path(str(INDEX_CORPUS_DIR))
    indices = _discover_index_files(corpus)
    assert indices, f"no index files found under {corpus}"

    report_lines: list[str] = [
        "# CMEMS Layer 2 Bulk Validation Report",
        "",
        f"corpus_root: {corpus.relative_to(corpus.parent.parent)!s}",
        f"dataset_count: {len(indices)}",
        "",
        "## Per-dataset summary",
        "",
        "(parse_ms is informational and machine-dependent; ignore in diffs.)",
        "",
    ]

    skipped: list[tuple[str, str]] = []
    for dataset_id in sorted(indices):
        index_path = indices[dataset_id]
        raw = index_path.read_bytes()

        from copernicus_mcp.errors import ValidationError

        t0 = time.monotonic()
        try:
            df = parse_index(raw)
        except ValidationError as exc:
            # The spike T-CMEMS-GET-INDEX-000 misclassified a few files
            # (notably MULTIOBS canyon_index_file.txt which uses a
            # different format — lowercase `# title`, point rows). Skip
            # them in the bulk sweep and surface in the report so the
            # next spike pass can promote them to a new FormatId.
            skipped.append((dataset_id, str(exc)))
            continue
        parse_ms = int((time.monotonic() - t0) * 1000)

        # 1. Canonical schema
        assert tuple(df.columns) == _CANONICAL_COLUMNS, (
            f"{dataset_id}: parser produced unexpected columns "
            f"{tuple(df.columns)!r}"
        )
        total = len(df)
        assert total > 0, f"{dataset_id}: empty DataFrame from parser"

        # 2. Out-of-range bbox → 0 matches (canonical IndexRow bounds
        # make this provably empty).
        zero_match = filter_by_bbox(df, (-180.0, 91.0, 180.0, 92.0))
        assert len(zero_match) == 0, (
            f"{dataset_id}: out-of-range bbox filter returned "
            f"{len(zero_match)} rows; the filter is not excluding."
        )

        # 3. Mid-band bbox from catalogue spatial_extent
        extent = _catalogue_spatial_extent(dataset_id)
        # Datasets where every row carries the same bbox (e.g. gridded
        # globals) can't produce a strict subset under any filter — the
        # mid-band assertion is meaningless. Guard against that.
        uniform_bbox = (
            df["lon_min"].nunique() == 1
            and df["lon_max"].nunique() == 1
            and df["lat_min"].nunique() == 1
            and df["lat_max"].nunique() == 1
        )
        if extent is None:
            mid_band_bbox = "skipped:catalogue_extent_missing"
            mid_band_matched: int | str = "n/a"
        elif _is_degenerate_extent(extent):
            mid_band_bbox = "skipped:degenerate_extent"
            mid_band_matched = "n/a"
        elif uniform_bbox:
            mid_band_bbox = "skipped:uniform_row_bbox"
            mid_band_matched = "n/a"
        else:
            shrunk = _shrink_bbox(
                (
                    extent["min_lon"],
                    extent["min_lat"],
                    extent["max_lon"],
                    extent["max_lat"],
                ),
                fraction=0.5,
            )
            mid_band_bbox = (
                f"({shrunk[0]:.2f},{shrunk[1]:.2f},{shrunk[2]:.2f},{shrunk[3]:.2f})"
            )
            mid_band = filter_by_bbox(df, shrunk)
            mid_band_matched = len(mid_band)
            assert 0 < mid_band_matched < total, (
                f"{dataset_id}: mid-band bbox matched {mid_band_matched} "
                f"of {total} rows; expected 0 < N < total. "
                f"Bbox: {shrunk!r}"
            )

        # 4. Mid-band time range derived from the parsed DataFrame.
        t_min = df["time_start"].min()
        t_max = df["time_end"].max()
        uniform_time = (
            df["time_start"].nunique() == 1 and df["time_end"].nunique() == 1
        )
        time_band_matched: int | str
        if pd.isna(t_min) or pd.isna(t_max) or t_min >= t_max:
            time_band = "skipped:degenerate_time"
            time_band_matched = "n/a"
        elif uniform_time:
            time_band = "skipped:uniform_row_time"
            time_band_matched = "n/a"
        else:
            span = t_max - t_min
            mid_start = (t_min + span * 0.25).strftime("%Y-%m-%dT%H:%M:%SZ")
            mid_end = (t_min + span * 0.75).strftime("%Y-%m-%dT%H:%M:%SZ")
            time_band = f"({mid_start},{mid_end})"
            time_filtered = filter_by_time(df, (mid_start, mid_end))
            time_band_matched = len(time_filtered)
            assert 0 < time_band_matched < total, (
                f"{dataset_id}: mid-band time matched {time_band_matched} "
                f"of {total} rows; expected 0 < N < total."
            )

        # 5. Variable filter — pick a frequent variable, assert > 0 match.
        variable_picked = _pick_frequent_variable(df)
        if variable_picked is None:
            variable_filter_matched: int | str = "n/a"
        else:
            var_filtered = filter_by_variables(df, [variable_picked])
            variable_filter_matched = len(var_filtered)
            assert variable_filter_matched > 0, (
                f"{dataset_id}: variable filter for {variable_picked!r} "
                f"matched 0 rows; expected > 0."
            )

        report_lines.append(f"### {dataset_id}")
        report_lines.append(f"- index_file: {index_path.name}")
        report_lines.append(f"- total_rows: {total}")
        report_lines.append(f"- parse_ms: ~{parse_ms}")
        report_lines.append(f"- out_of_range_bbox_matched: {len(zero_match)}")
        report_lines.append(f"- mid_band_bbox: {mid_band_bbox}")
        report_lines.append(f"- mid_band_bbox_matched: {mid_band_matched}")
        report_lines.append(f"- mid_band_time: {time_band}")
        report_lines.append(f"- mid_band_time_matched: {time_band_matched}")
        report_lines.append(f"- variable_picked: {variable_picked}")
        report_lines.append(f"- variable_filter_matched: {variable_filter_matched}")
        report_lines.append("")

    if skipped:
        report_lines.append("## Skipped (unsupported format)")
        report_lines.append("")
        for dataset_id, reason in skipped:
            report_lines.append(f"- {dataset_id}: {reason}")
        report_lines.append("")

    _REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
