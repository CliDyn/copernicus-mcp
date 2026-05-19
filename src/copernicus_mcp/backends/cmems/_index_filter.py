"""Pure filter primitives over a canonical IndexRow DataFrame.

Each filter is a pure function: same input → same output, no I/O, no
hidden state. The canonical row schema is fixed by ``_index_parser.IndexRow``
(lat/lon stored as separate columns, ``time_*`` tz-aware, ``variables``
nullable tuple, ``platform_type`` nullable string).

Antimeridian carve-out (the project conventions invariant 7, sub-plan decisions #10
and #18): SDK-bound operations (``CmemsSubsetRequest``) reject
antimeridian-crossing user bboxes because the SDK cannot split the
request. Offline index filtering operates on pre-parsed per-row
bboxes in pure Python and accepts antimeridian-crossing user input —
each row carries its own bbox so the intersection is unambiguous.
The resulting ``file_list`` is downloadable via ``marine_get_files``;
it is NOT a valid input for ``marine_subset_dataset``.
"""

from __future__ import annotations

import pandas as pd

from copernicus_mcp.common.time import iso8601_utc
from copernicus_mcp.errors import ValidationError


def filter_by_bbox(
    df: pd.DataFrame, bbox: tuple[float, float, float, float]
) -> pd.DataFrame:
    """Keep rows whose per-file bbox intersects the user bbox.

    ``bbox`` is ``(lon_min, lat_min, lon_max, lat_max)``. When
    ``lon_min > lon_max`` the user range is treated as
    ``[lon_min, 180] ∪ [-180, lon_max]`` (antimeridian wrap). Latitude
    intersection is always the standard interval test.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_mask = (df["lat_min"] <= lat_max) & (df["lat_max"] >= lat_min)
    if lon_min <= lon_max:
        lon_mask = (df["lon_min"] <= lon_max) & (df["lon_max"] >= lon_min)
    else:
        east = (df["lon_min"] <= 180.0) & (df["lon_max"] >= lon_min)
        west = (df["lon_min"] <= lon_max) & (df["lon_max"] >= -180.0)
        lon_mask = east | west
    return df[lat_mask & lon_mask]


def filter_by_time(df: pd.DataFrame, time_range: tuple[str, str]) -> pd.DataFrame:
    """Keep rows whose ``[time_start, time_end]`` intersects the user range.

    Both endpoints are normalised via ``iso8601_utc`` (so naive datetimes
    are rejected and offsets fold to UTC). Inverted user ranges
    (``start > end``) raise ``ValidationError``.
    """
    start_norm = iso8601_utc(time_range[0])
    end_norm = iso8601_utc(time_range[1])
    if start_norm > end_norm:
        raise ValidationError("filter_by_time: time_range start must be <= end")
    start = pd.Timestamp(start_norm)
    end = pd.Timestamp(end_norm)
    mask = (df["time_start"] <= end) & (df["time_end"] >= start)
    return df[mask]


def filter_by_variables(df: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    """Keep rows whose ``variables`` is ``None`` OR shares any token with ``requested``.

    ``None`` means "unknown, can't reject" — kept. An explicitly empty
    tuple means "no variables here" — dropped on any non-empty request.
    Empty ``requested`` is a no-op pass-through.
    """
    if not requested or df.empty:
        return df
    requested_set = set(requested)

    def _keep(value: object) -> bool:
        # ``None`` / ``NaN`` / ``pd.NA`` all mean "no variable info on this
        # row" — treat as unknown, keep (cr round-1 I2). Any concrete
        # variable-set iterable (tuple / list / set / frozenset) goes through
        # set intersection (cr round-2: parser today emits tuple, but the
        # filter accepts the broader contract for forward compatibility).
        # Strings are scalars, not variable lists — explicit drop.
        if value is None:
            return True
        if isinstance(value, (tuple, list, set, frozenset)):
            return bool(requested_set.intersection(value))
        try:
            if pd.isna(value):
                return True
        except (TypeError, ValueError):
            pass
        return False

    return df[df["variables"].map(_keep)]


def filter_by_platform(df: pd.DataFrame, types: list[str]) -> pd.DataFrame:
    """Keep rows whose ``platform_type`` is in ``types``.

    Strict membership: rows with ``platform_type=None`` are dropped on
    any non-empty ``types`` (an empty ``types`` is a no-op pass-through).
    """
    if not types:
        return df
    return df[df["platform_type"].isin(types)]


def apply_filters(
    df: pd.DataFrame,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    time_range: tuple[str, str] | None = None,
    variables: list[str] | None = None,
    platform_types: list[str] | None = None,
) -> pd.DataFrame:
    """AND-combine the four filter primitives.

    Each argument is optional; ``None`` skips that axis. All four ``None``
    returns ``df`` unchanged.
    """
    if bbox is not None:
        df = filter_by_bbox(df, bbox)
    if time_range is not None:
        df = filter_by_time(df, time_range)
    if variables is not None:
        df = filter_by_variables(df, variables)
    if platform_types is not None:
        df = filter_by_platform(df, platform_types)
    return df
