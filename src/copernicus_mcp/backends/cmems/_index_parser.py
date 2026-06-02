"""Layer 2 index parsers — canonical IndexRow DataFrame across CMEMS formats.

Three formats discovered by the T-CMEMS-GET-INDEX-000 spike (2026-05-16):

- ``insitu_index_file_v3`` — CSV with ``#``-comment header. Per-file
  ``geospatial_{lat,lon}_{min,max}`` and ``time_coverage_{start,end}``.
- ``cora_path_v1`` — no index file. Built by parsing the dataset's full
  file listing (~1M paths) from ``marine.get(dry_run=True)``. Path
  structure ``<region>/<year>/CO_DMQCGL01_<YYYYMMDD>_<OBS_TYPE>_<PLATFORM>.nc``.
- ``easycora_path_v1`` — identical structure to CORA but ``ECO_`` prefix.

Each parser returns a canonical ``IndexRow`` DataFrame so downstream
filters (``_index_filter``) operate on one shape regardless of source.
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from copernicus_mcp.errors import ValidationError
from copernicus_mcp.observability.logger import get_logger

_logger = get_logger(__name__)

FormatId = Literal[
    "insitu_index_file_v3",
    "cora_path_v1",
    "easycora_path_v1",
    "multiobs_canyon_v1",
]

# Marker tightened in cr round 1 (M2): the platforms catalog also begins with
# "# Title : in-situ" — match the specific files-catalog suffix instead.
_INSITU_HEADER_MARKER = "# Title : in-situ files"
_CANYON_HEADER_MARKER = "# title : Nutrient and Carbon"

_CANYON_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "file_name",
        "date",
        "longitude",
        "latitude",
        "temp_dt_mode",
        "psal_dt_mode",
        "doxy_dt_mode",
    }
)

_CANONICAL_COLUMNS: tuple[str, ...] = (
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

_INSITU_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "file_name",
        "geospatial_lat_min",
        "geospatial_lat_max",
        "geospatial_lon_min",
        "geospatial_lon_max",
        "time_coverage_start",
        "time_coverage_end",
        "parameters",
    }
)

# CMEMS folder-level bbox lookup for the path-based CORA / EasyCORA
# strategies. Boundaries reflect CMEMS folder definitions (PUM-013-001),
# not geographic sea boundaries. Bbox tuples are ``(lon_min, lat_min,
# lon_max, lat_max)`` to match the project's bbox convention.
REGION_BBOX: dict[str, tuple[float, float, float, float]] = {
    "arctic": (-180.0, 62.0, 180.0, 90.0),
    "baltic": (9.0, 53.0, 30.0, 66.0),
    "blacksea": (27.0, 40.0, 42.0, 47.0),
    "global": (-180.0, -90.0, 180.0, 90.0),
    "mediterrane": (-6.0, 30.0, 42.0, 46.0),
    "northwestshelf": (-16.0, 48.0, 13.0, 62.0),
    "southwestshelf": (-20.0, 26.0, 0.0, 48.0),
}


def parse_insitu_index_file(raw: bytes) -> pd.DataFrame:
    """Parse ``insitu_index_file_v3`` bytes into a canonical IndexRow DataFrame.

    Format: ``#``-comment lines, then a CSV header line, then data rows.
    Required CSV columns: ``file_name``, ``geospatial_{lat,lon}_{min,max}``,
    ``time_coverage_{start,end}``, ``parameters``.

    Rows with NaN in any required numeric / datetime field are dropped —
    they cannot satisfy the canonical IndexRow contract. ``platform_type``
    and ``size_bytes`` are absent in this format and emitted as ``None``.

    Raises ``ValidationError`` on: non-UTF8 bytes, empty input, missing
    CSV header, missing required columns, no parseable data rows.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(
            "insitu_index_file: bytes are not valid UTF-8"
        ) from exc

    if not text.strip():
        raise ValidationError("insitu_index_file: empty input")

    try:
        df_raw = pd.read_csv(
            io.StringIO(text),
            comment="#",
            dtype=str,
            keep_default_na=False,
            na_values=[""],
        )
    except pd.errors.EmptyDataError as exc:
        raise ValidationError(
            "insitu_index_file: no CSV header or data rows found"
        ) from exc
    except pd.errors.ParserError as exc:
        raise ValidationError(
            "insitu_index_file: CSV parse failure"
        ) from exc

    missing = _INSITU_REQUIRED_COLUMNS - set(df_raw.columns)
    if missing:
        raise ValidationError(
            f"insitu_index_file: missing required columns {sorted(missing)}"
        )

    df = _normalise_insitu_rows(df_raw)
    return _drop_contract_violations(df, format_id="insitu_index_file_v3")


# cr round-1 M1: require an explicit timezone (Z or ±HH:MM offset). Without
# this guard ``pd.to_datetime(..., utc=True)`` silently coerces naive strings
# into UTC, mis-stamping data that may actually be in local time.
_TZ_SUFFIX_RE = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


def _normalise_insitu_rows(df_raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "file_path": df_raw["file_name"].astype(str),
            "lon_min": pd.to_numeric(df_raw["geospatial_lon_min"], errors="coerce"),
            "lon_max": pd.to_numeric(df_raw["geospatial_lon_max"], errors="coerce"),
            "lat_min": pd.to_numeric(df_raw["geospatial_lat_min"], errors="coerce"),
            "lat_max": pd.to_numeric(df_raw["geospatial_lat_max"], errors="coerce"),
            "time_start": _parse_tz_aware_datetimes(df_raw["time_coverage_start"]),
            "time_end": _parse_tz_aware_datetimes(df_raw["time_coverage_end"]),
            "platform_type": pd.Series([None] * len(df_raw), dtype=object),
            "variables": df_raw["parameters"].map(_split_parameters),
            "size_bytes": pd.Series([None] * len(df_raw), dtype=object),
        }
    )
    out = out.dropna(
        subset=[
            "file_path",
            "lon_min",
            "lon_max",
            "lat_min",
            "lat_max",
            "time_start",
            "time_end",
        ]
    ).reset_index(drop=True)
    return out[list(_CANONICAL_COLUMNS)]


def _parse_tz_aware_datetimes(series: pd.Series) -> pd.Series:
    """Parse datetimes, but reject (NaT) any value lacking an explicit tz.

    pandas' ``utc=True`` would silently attach UTC to naive strings; this
    helper preserves the distinction so naive rows fall out via ``dropna``.

    ``format="ISO8601"`` (cr round-3 tightening of round-2 H1 fix): pandas
    infers per-row ISO 8601 shape so mixed whole/sub-second ``Z`` rows in
    the same series both parse, while non-ISO inputs like ``01/02/2020``
    correctly NaT — ``format="mixed"`` would have silently accepted them.
    """
    has_tz = series.fillna("").astype(str).str.contains(_TZ_SUFFIX_RE)
    cleaned = series.where(has_tz, other=pd.NA)
    return pd.to_datetime(cleaned, errors="coerce", utc=True, format="ISO8601")


def _drop_contract_violations(
    df: pd.DataFrame, *, format_id: str
) -> pd.DataFrame:
    """Drop rows that violate the ``IndexRow`` contract; log the count.

    Real CMEMS indices DO contain malformed rows — Task 006 (bulk
    validation) found 3/134 GLODAP-obs rows with ``lon_max > 180``
    (antimeridian-crossing ship tracks incorrectly encoded). Raising on
    any violation made entire datasets unusable; dropping the bad rows
    preserves the rest while keeping the cr round-1 H1 concern
    addressed (malformed rows never propagate into filter results).

    Round-1 H1 raised; this is a deliberate policy reversal logged in
    ``the project decision log`` under T-CMEMS-GET-INDEX-006.
    """
    if df.empty:
        return df
    violations = (
        (df["lat_min"] < -90.0)
        | (df["lat_min"] > 90.0)
        | (df["lat_max"] < -90.0)
        | (df["lat_max"] > 90.0)
        | (df["lon_min"] < -180.0)
        | (df["lon_min"] > 180.0)
        | (df["lon_max"] < -180.0)
        | (df["lon_max"] > 180.0)
        | (df["lat_min"] > df["lat_max"])
        | (df["lon_min"] > df["lon_max"])
        | (df["time_start"] > df["time_end"])
    )
    bad = int(violations.sum())
    if bad == len(df):
        # cr round-1 I2: 100% violations strongly signals a misclassified
        # format (we'd silently cache emptiness otherwise, locking users
        # out until the upstream index changes). Raise so the failure is
        # visible.
        raise ValidationError(
            f"{format_id}: every row ({bad}) violates the IndexRow contract; "
            "this likely indicates the index file is in a different format "
            "than expected. Verify the registry entry's format_id."
        )
    if bad:
        _logger.warning(
            "%s: dropped %d row(s) violating IndexRow contract "
            "(lat/lon out of range, inverted min/max, or inverted time range)",
            format_id,
            bad,
        )
        return df.loc[~violations].reset_index(drop=True)
    return df


def _split_parameters(value: object) -> tuple[str, ...] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    tokens = str(value).split()
    return tuple(tokens) if tokens else None


# CORA / EasyCORA paths: ``<region>/<year>/<PREFIX>_DMQCGL01_YYYYMMDD_OBS_PLATFORM.nc``
# where PREFIX is ``CO`` for CORA and ``ECO`` for EasyCORA.
# cr round-1 C2: ``\Z`` (not ``$``) so a trailing newline / whitespace cannot
# slip through into ``file_path``.
_CORA_PATH_RE = re.compile(
    r"^(?P<region>[a-z]+)/"
    r"(?P<year>\d{4})/"
    r"(?P<prefix>CO|ECO)_DMQCGL01_"
    r"(?P<date>\d{8})_"
    r"(?P<obs>[A-Z]{2})_"
    r"(?P<platform>[A-Z]{2})\.nc\Z"
)


def parse_cora_paths(file_paths: list[str]) -> pd.DataFrame:
    """Parse CORA path listing (``CO_`` prefix) into a canonical DataFrame.

    Path structure: ``<region>/<year>/CO_DMQCGL01_<YYYYMMDD>_<OBS>_<PLATFORM>.nc``.
    Region → bbox via ``REGION_BBOX``. Date → both ``time_start`` (00:00:00Z)
    and ``time_end`` (23:59:59Z) so a single-day filter captures the file.

    Paths that don't match the expected shape, carry an unknown region, or
    have an invalid date are SILENTLY DROPPED — a real CORA listing has
    over a million entries and a handful of strays must not fail the whole
    listing. ``ECO_`` paths belong to ``parse_easycora_paths`` and are
    likewise dropped here.

    Returns a DataFrame with the canonical IndexRow columns. Empty input
    returns an empty DataFrame with the canonical column layout.
    """
    return _parse_path_listing(file_paths, expected_prefix="CO")


def parse_easycora_paths(file_paths: list[str]) -> pd.DataFrame:
    """Parse EasyCORA path listing — identical structure to CORA but ``ECO_`` prefix."""
    return _parse_path_listing(file_paths, expected_prefix="ECO")


def parse_canyon_index_file(raw: bytes) -> pd.DataFrame:
    """Parse the MULTIOBS canyon (``multiobs_canyon_v1``) index file.

    Format characteristics (discovered T-CMEMS-GET-INDEX-008):
    - ``#``-comment header starting with ``# title : Nutrient and Carbon``
      (note lowercase ``title``).
    - CSV columns ``file_name, date, wmo, n_cycle, longitude, latitude,
      temp_dt_mode, psal_dt_mode, doxy_dt_mode, date_update``.
    - **Point observations, NOT per-file rows.** Many rows share the same
      ``file_name`` (one .nc file holds multiple profile cycles).
    - ``date`` is naive (`YYYY-MM-DD HH:MM:SS`); the catalogue documents
      it as UTC — we attach UTC explicitly here.

    Aggregation: rows with the same ``file_name`` collapse to one
    IndexRow. ``lon_min/max`` and ``lat_min/max`` are the per-file bbox
    spanning all observations; ``time_start/end`` are the per-file
    extent. ``variables`` is the union of present ``*_dt_mode`` columns
    across the file's observations (TEMP / PSAL / DOXY).

    Raises ``ValidationError`` on non-UTF8 / empty / missing-column input.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError(
            "canyon_index_file: bytes are not valid UTF-8"
        ) from exc

    if not text.strip():
        raise ValidationError("canyon_index_file: empty input")

    try:
        df_raw = pd.read_csv(
            io.StringIO(text),
            comment="#",
            dtype=str,
            keep_default_na=False,
            na_values=[""],
        )
    except pd.errors.EmptyDataError as exc:
        raise ValidationError(
            "canyon_index_file: no CSV header or data rows found"
        ) from exc
    except pd.errors.ParserError as exc:
        raise ValidationError("canyon_index_file: CSV parse failure") from exc

    missing = _CANYON_REQUIRED_COLUMNS - set(df_raw.columns)
    if missing:
        raise ValidationError(
            f"canyon_index_file: missing required columns {sorted(missing)}"
        )

    df = _normalise_canyon_rows(df_raw)
    return _drop_contract_violations(df, format_id="multiobs_canyon_v1")


_CANYON_DT_VARIABLE_MAP: dict[str, str] = {
    "temp_dt_mode": "TEMP",
    "psal_dt_mode": "PSAL",
    "doxy_dt_mode": "DOXY",
}


def _normalise_canyon_rows(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-observation rows into per-file rows."""
    work = pd.DataFrame(
        {
            "file_path": df_raw["file_name"].astype(str),
            "lon": pd.to_numeric(df_raw["longitude"], errors="coerce"),
            "lat": pd.to_numeric(df_raw["latitude"], errors="coerce"),
            # canyon dates are naive UTC; attach explicitly.
            "time": pd.to_datetime(
                df_raw["date"], errors="coerce", utc=True, format="ISO8601"
            ),
        }
    )
    # Drop rows that fail coercion before grouping.
    work = work.dropna(subset=["file_path", "lon", "lat", "time"])

    # Variable presence per source row. Codex retro LOW: ``na_values=[""]``
    # only catches EXACT empty fields; whitespace-only values
    # (single space, tab) survive and ``.notna()`` returns True,
    # falsely flagging the variable as present. Strip before
    # presence detection so any pure-whitespace cell counts as absent.
    for column, var_name in _CANYON_DT_VARIABLE_MAP.items():
        present = (
            df_raw[column].fillna("").astype(str).str.strip().ne("")
        )
        work[f"has_{var_name}"] = present.reindex(
            work.index, fill_value=False
        )

    if work.empty:
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
        )[list(_CANONICAL_COLUMNS)]

    grouped = work.groupby("file_path", sort=False).agg(
        lon_min=("lon", "min"),
        lon_max=("lon", "max"),
        lat_min=("lat", "min"),
        lat_max=("lat", "max"),
        time_start=("time", "min"),
        time_end=("time", "max"),
        has_TEMP=("has_TEMP", "any"),
        has_PSAL=("has_PSAL", "any"),
        has_DOXY=("has_DOXY", "any"),
    )
    grouped = grouped.reset_index()

    def _variables(row: pd.Series) -> tuple[str, ...] | None:
        present = tuple(
            var for var in ("TEMP", "PSAL", "DOXY") if row[f"has_{var}"]
        )
        return present if present else None

    grouped["variables"] = grouped.apply(_variables, axis=1)
    grouped["platform_type"] = pd.Series([None] * len(grouped), dtype=object)
    grouped["size_bytes"] = pd.Series([None] * len(grouped), dtype=object)
    return grouped[list(_CANONICAL_COLUMNS)]


def detect_format(raw: bytes) -> FormatId:
    """Identify the format of an index file by its leading bytes.

    Detects by-bytes formats: ``insitu_index_file_v3`` and
    ``multiobs_canyon_v1``. Path-based formats (CORA / EasyCORA) don't
    have raw bytes — they're built from a ``marine.get(dry_run=True)``
    path listing, so the caller knows the format from the registry
    entry's ``format_id``.

    Raises ``ValidationError`` on empty / non-UTF8 / unrecognised input.
    """
    if not raw:
        raise ValidationError("detect_format: empty input")
    try:
        # Read only the first KB — enough to spot the header without
        # decoding a 40 MB index just to identify it.
        head = raw[:1024].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError("detect_format: bytes are not valid UTF-8") from exc
    stripped = head.lstrip()
    if stripped.startswith(_INSITU_HEADER_MARKER):
        return "insitu_index_file_v3"
    if stripped.startswith(_CANYON_HEADER_MARKER):
        return "multiobs_canyon_v1"
    raise ValidationError(
        "detect_format: unrecognised index format — no known header prefix"
    )


def parse_index(
    raw: bytes, expected_format: FormatId | None = None
) -> pd.DataFrame:
    """Top-level dispatch from raw bytes to a canonical IndexRow DataFrame.

    ``expected_format`` short-circuits detection — used on the hot path
    when the registry already pins the format for a known dataset.
    ``None`` runs ``detect_format`` first; only by-bytes formats are
    auto-detectable.

    Path-based formats (``cora_path_v1`` / ``easycora_path_v1``) are NOT
    reachable via this function — they come from a path listing, not
    raw bytes; call ``parse_cora_paths`` / ``parse_easycora_paths``
    directly with the SDK output.
    """
    fmt = expected_format if expected_format is not None else detect_format(raw)
    if fmt == "insitu_index_file_v3":
        return parse_insitu_index_file(raw)
    if fmt == "multiobs_canyon_v1":
        return parse_canyon_index_file(raw)
    if fmt in ("cora_path_v1", "easycora_path_v1"):
        raise ValidationError(
            f"parse_index: format {fmt!r} is path-based — "
            "call parse_cora_paths / parse_easycora_paths with the path listing"
        )
    raise ValidationError(
        f"parse_index: unknown format identifier {fmt!r}"
    )


def _parse_path_listing(
    file_paths: list[str], *, expected_prefix: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in file_paths:
        row = _extract_path_row(path, expected_prefix=expected_prefix)
        if row is not None:
            rows.append(row)

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
        )[list(_CANONICAL_COLUMNS)]

    df = pd.DataFrame(rows)
    df["time_start"] = pd.to_datetime(df["time_start"], utc=True)
    df["time_end"] = pd.to_datetime(df["time_end"], utc=True)
    return df[list(_CANONICAL_COLUMNS)]


def _extract_path_row(path: str, *, expected_prefix: str) -> dict[str, object] | None:
    match = _CORA_PATH_RE.match(path)
    if match is None:
        return None
    if match.group("prefix") != expected_prefix:
        return None
    region = match.group("region")
    bbox = REGION_BBOX.get(region)
    if bbox is None:
        return None
    try:
        date = datetime.strptime(match.group("date"), "%Y%m%d")
    except ValueError:
        return None
    lon_min, lat_min, lon_max, lat_max = bbox
    return {
        "file_path": path,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "time_start": date.strftime("%Y-%m-%dT00:00:00Z"),
        "time_end": date.strftime("%Y-%m-%dT23:59:59Z"),
        "platform_type": match.group("platform"),
        "variables": None,
        "size_bytes": None,
    }


class IndexRow(BaseModel):
    """Canonical per-file row produced by every parser.

    Lat/lon constrained to standard ranges via Pydantic Field. Cross-field
    invariants (min <= max, time_start <= time_end, tz-aware datetimes)
    enforced by a single model validator.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    file_path: str = Field(min_length=1)
    lon_min: float = Field(ge=-180.0, le=180.0)
    lon_max: float = Field(ge=-180.0, le=180.0)
    lat_min: float = Field(ge=-90.0, le=90.0)
    lat_max: float = Field(ge=-90.0, le=90.0)
    time_start: datetime
    time_end: datetime
    platform_type: str | None = None
    variables: tuple[str, ...] | None = None
    size_bytes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_invariants(self) -> IndexRow:
        if self.lon_min > self.lon_max:
            raise ValueError("lon_min must be <= lon_max")
        if self.lat_min > self.lat_max:
            raise ValueError("lat_min must be <= lat_max")
        if self.time_start.tzinfo is None or self.time_end.tzinfo is None:
            raise ValueError("time_start and time_end must be timezone-aware")
        if self.time_start > self.time_end:
            raise ValueError("time_start must be <= time_end")
        return self
