"""Heuristic byte-size estimator for CDS retrieve requests (T-CDS-004).

Per ``the project research notes`` §6.7.4 option 1: the legacy
``cdsapi`` 0.7.7 client has no estimation API, and the
``ecmwf-datastores-client.apply_constraints`` (option 2) is not yet on the
runtime dependency list. Using a real submit as a probe (option 3) abuses
the queue.

So we estimate from request shape alone:

    estimated_bytes = N_fields × bytes_per_field(dataset_id) × area_fraction

where:

- ``N_fields`` is the product of list-cardinalities of cdsapi-canonical
  multiplicative fields: ``variable``, ``year``, ``month``, ``day``,
  ``time``, ``pressure_level``, ``model_level``, ``leadtime_hour``,
  ``leadtime_month``, ``level`` (research §6.9.2). Single-value scalar
  strings (e.g. MARS-shape ``date: "2013-01-01"``) count as cardinality
  1 — we accept the underestimate for legacy MARS datasets rather than
  ship a parser.
- ``bytes_per_field`` is a curated dataset-specific constant (e.g. ERA5
  pressure-levels at 0.25° global GRIB ≈ 4 MB/field). Unknown datasets
  fall back to a conservative default ``_DEFAULT_BYTES_PER_FIELD``.
- ``area_fraction`` is ``(lat_span × lon_span) / (180 × 360)`` when the
  request has an ``area: [N, W, S, E]`` field; ``1.0`` (global) otherwise.

The estimate is *always* ``epistemic_status="approximate"``; expected
accuracy is roughly ±50%. Confirmation gate decisions (T-CDS-005) treat
the rounded number as a hint to the user, not a contract.
"""

from __future__ import annotations

import math
from typing import Any

# Multiplicative list-cardinality fields per research §6.9.2. Order does
# not matter for ``count_fields`` (product is commutative); listed here
# in canonical CDS request order for readability.
_CARDINALITY_FIELDS: tuple[str, ...] = (
    "variable",
    "year",
    "month",
    "day",
    "time",
    "date",  # research §6.9.2: ``str | List[str]`` — list form is multiplicative
    "pressure_level",
    "model_level",
    "level",
    "leadtime_hour",
    "leadtime_month",
    "leadtime_day",
    "step",  # MARS-style forecast step list
    "product_type",  # ensemble_members vs reanalysis vs forecast — multiplies
    "forecast_type",  # ADS: instantaneous vs time-averaged
    "system_version",  # EWDS: EFAS / GloFAS forecasts
    "originating_centre",  # EWDS multi-model
    "hydrological_year",  # EWDS historical
    "sky_type",  # ADS radiation
    "band",  # ADS spectral
    "forcing_type",  # ADS radiative forcings
)

# Empirical bytes-per-field for top datasets, drawn from research §6.7.4
# (option 1) and CDS user-facing docs (e.g. "1 hourly snapshot at 0.25°
# global GRIB ≈ 4 MB"). These are global-coverage values; ``area_fraction``
# scales them down for area-restricted requests.
#
# Values are conservative *over*-estimates by ~10-20% — the confirmation
# gate downstream prefers false positives (asking the user) to false
# negatives (silent multi-GB download).
_BYTES_PER_FIELD: dict[str, int] = {
    # ERA5 family — primary CDS workhorse. Values per global 0.25° field
    # in GRIB. NetCDF is ~30% larger but the heuristic ignores format.
    "reanalysis-era5-pressure-levels": 4_200_000,
    "reanalysis-era5-pressure-levels-monthly-means": 4_200_000,
    "reanalysis-era5-single-levels": 2_100_000,
    "reanalysis-era5-single-levels-monthly-means": 2_100_000,
    # ERA5-Land: 0.1° native (16x finer cells per area than ERA5-PL 0.25°)
    # over land mask (~30% of globe). Code-reviewer T-CDS-004 MEDIUM:
    # earlier estimate of 800 KB/field was tied to a small-area assumption,
    # but the heuristic also multiplies by ``area_fraction`` so a global
    # request would underestimate. CDS docs / forum threads cite ~5 MB
    # per global-hourly GRIB field for ERA5-Land.
    "reanalysis-era5-land": 5_000_000,
    "reanalysis-era5-land-monthly-means": 5_000_000,
    "reanalysis-era5-land-timeseries": 50_000,  # time-series form is sparse
    "reanalysis-era5-complete": 5_000_000,  # MARS internal; coarser estimate
    # Seasonal forecasts — coarser native resolution.
    "seasonal-original-pressure-levels": 1_500_000,
    "seasonal-original-single-levels": 750_000,
    "seasonal-monthly-pressure-levels": 1_500_000,
    "seasonal-monthly-single-levels": 750_000,
    # CMIP / CORDEX climate projections — variable, conservative default.
    "projections-cmip6": 8_000_000,
    "projections-cordex-domains-single-levels": 4_000_000,
}

# Conservative default for datasets not in the curated map. ~2 MB/field
# matches global single-level ERA5; ERA5 PL is 2x, so the overestimate
# stays bounded at ~2x for unknown datasets — acceptable for the
# confirmation gate's "ask if in doubt" stance.
_DEFAULT_BYTES_PER_FIELD: int = 2_000_000

# Queue-latency tier thresholds in FIELD COUNT. Codex T-CDS-004 MEDIUM:
# research §6.5.4 ties queue latency to field cardinality (the server
# pulls each field independently from tape/disk), not to download
# bytes. A tiny-area / many-field request still queues long even
# though bytes are small. Tiering on N_fields is the documented
# proxy:
#   <100 fields              → light (seconds-minutes)
#   100 - 10_000 fields      → medium (minutes-tens of minutes)
#   >10_000 fields           → heavy (tens of minutes - hours)
_TIER_LIGHT_MAX_FIELDS: int = 100
_TIER_MEDIUM_MAX_FIELDS: int = 10_000

_GLOBAL_AREA_DEG_SQ: float = 180.0 * 360.0


def count_fields(inputs: dict[str, Any]) -> int:
    """Multiplicative cardinality of a cdsapi request inputs dict.

    Lists count as their length; non-list values count as 1 (a scalar
    field still represents one chosen value). MARS-style slash-delimited
    strings like ``"00/06/12/18"`` cannot be expanded without a parser
    — they count as 1. Result is at minimum 1.

    Only fields in ``_CARDINALITY_FIELDS`` participate; format /
    archive / area / grid keys are ignored (they are scalars or
    transforms, not multipliers).
    """
    total = 1
    for field in _CARDINALITY_FIELDS:
        value = inputs.get(field)
        if value is None:
            continue
        if isinstance(value, list):
            cardinality = len(value) or 1
        else:
            cardinality = 1
        total *= cardinality
    return total


def area_fraction(inputs: dict[str, Any]) -> float:
    """Return the fraction of the global domain covered by ``area``.

    ``area: [N, W, S, E]`` per research §6.9.2 (non-standard order).

    Wrap-around longitudes (W > E, antimeridian-crossing) are NOT
    explicitly handled — we compute ``|N-S| × |E-W|`` which OVERestimates
    for antimeridian bboxes (e.g. ``[70, 170, 60, -170]`` reads
    ``lon_span=340°`` while true coverage is 20°). The overestimate is
    SAFE for the confirmation gate (errs toward asking the user) and
    avoids a parser dependency. Antimeridian-crossing bboxes are also
    rejected upstream by ``CmemsSubsetRequest`` — for CDS, the engine
    handles them server-side.

    Missing or malformed area returns 1.0 (global). Codex/code-reviewer
    T-CDS-004 MEDIUM: NaN / ±inf coordinates would crash the downstream
    ``int(... × area_fraction)`` cast — gate explicitly via
    ``math.isfinite``.
    """
    area = inputs.get("area")
    if not isinstance(area, list) or len(area) != 4:
        return 1.0
    try:
        north, west, south, east = (float(v) for v in area)
    except (TypeError, ValueError):
        return 1.0
    if not all(math.isfinite(v) for v in (north, west, south, east)):
        # NaN / ±inf — treat as malformed and fall back to global. Better
        # to slightly overestimate than crash with an uncaught ValueError.
        return 1.0
    lat_span = abs(north - south)
    lon_span = abs(east - west)
    fraction = (lat_span * lon_span) / _GLOBAL_AREA_DEG_SQ
    if fraction <= 0.0:
        # Degenerate (zero-area or inverted) → underestimate is fine; the
        # request will fail server-side for other reasons.
        return 0.0
    return min(fraction, 1.0)


def _human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    size_f = float(num_bytes) / 1024
    for unit in ("KB", "MB", "GB", "TB"):
        if size_f < 1024:
            return f"{size_f:.1f} {unit}"
        size_f /= 1024
    return f"{size_f:.1f} PB"


def _tier_for_fields(n_fields: int) -> str:
    """Codex T-CDS-004 MEDIUM: queue latency in research §6.5.4 is tied
    to field cardinality (per-field tape/disk fetch on the server),
    not to downloaded bytes. A tiny-area request can still pull tens
    of thousands of fields and sit in the queue for hours."""
    if n_fields < _TIER_LIGHT_MAX_FIELDS:
        return "light"
    if n_fields <= _TIER_MEDIUM_MAX_FIELDS:
        return "medium"
    return "heavy"


def estimate(dataset_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical estimate envelope for a CDS retrieve request.

    Shape mirrors ``CmemsBackend.estimate`` plus CDS-specific
    ``fields_count`` (debugging), ``queue_latency_tier`` (research
    §6.7.4 tail), ``epistemic_status`` (T-CDS-011.6 split), and
    ``runtime_compatible`` (T-CDS-011.2 — true iff the dataset is in
    the bundled catalogue and therefore handled by the runtime).
    """
    # Lazy import to keep the estimator independent of the catalogue
    # at module load time (the catalogue reads bundled JSON files).
    # ``runtime_supports`` consults both the catalogue and the routing
    # table, so it's the right source of truth — pre-cr-round-1-M1
    # we used catalogue presence alone, which silently mis-reported
    # true for catalogue-only stores that the runtime can't route.
    from copernicus_mcp.backends.cds.backend import runtime_supports

    n_fields = count_fields(inputs)
    bytes_per_field = _BYTES_PER_FIELD.get(dataset_id, _DEFAULT_BYTES_PER_FIELD)
    fraction = area_fraction(inputs)
    bytes_estimate = max(int(n_fields * bytes_per_field * fraction), 0)

    tier = _tier_for_fields(n_fields)

    in_curated_map = dataset_id in _BYTES_PER_FIELD
    is_runtime_supported = runtime_supports(dataset_id)

    if in_curated_map:
        epistemic_status = "curated_approximate"
        advisory = (
            f"Heuristic estimate from N_fields × bytes_per_field × "
            f"area_fraction; expected accuracy ±50%. Dataset "
            f"{dataset_id!r} is in the curated bytes-per-field map."
        )
    else:
        epistemic_status = "default_heuristic"
        advisory = (
            f"Heuristic estimate using default bytes_per_field "
            f"({_DEFAULT_BYTES_PER_FIELD // 1_000_000} MB) — dataset "
            f"{dataset_id!r} is not in the curated map and the actual "
            "size could differ by a larger factor (±10×). Run a tiny "
            "probe request first if the estimate is over the "
            "confirmation threshold."
        )

    return {
        "type": "free",
        "estimated_size_bytes": bytes_estimate,
        "estimated_size_human": _human_size(bytes_estimate),
        "epistemic_status": epistemic_status,
        "runtime_compatible": is_runtime_supported,
        "advisory_message": advisory,
        "fields_count": n_fields,
        "queue_latency_tier": tier,
    }
