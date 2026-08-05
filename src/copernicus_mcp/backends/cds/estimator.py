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

from copernicus_mcp.backends.cds.costing import CostingResult

# Surfaced to the agent alongside every byte-size estimate. The estimate is a
# heuristic / seed / calibration figure, NOT a contract — make that impossible to
# miss so an agent never treats it as exact for limits, quotas, or provisioning.
_SIZE_ESTIMATE_CAVEAT = (
    "APPROXIMATE — this byte-size estimate is heuristic and can be wrong by a "
    "large factor (often several times, occasionally far more). Do NOT rely on it "
    "for hard limits, quotas, billing, or disk provisioning; treat it as a rough "
    "order-of-magnitude only. The server cost units are exact; only the byte size "
    "is uncertain."
)

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


def estimate(
    dataset_id: str,
    inputs: dict[str, Any],
    *,
    costing: CostingResult | None = None,
    calibration: Any = None,
) -> dict[str, Any]:
    """Return the canonical estimate envelope (v2) for a CDS retrieve request.

    ``costing`` is the optional ``/costing`` pre-flight result (T-CDS-EST2-001);
    ``None`` ⇒ the endpoint was unreachable and the legacy
    ``N_fields × bytes_per_field × area_fraction`` heuristic is used, byte-for-byte
    as before. ``calibration`` (T-CDS-EST2-004) is accepted here but treated as
    "no entry" until that task wires it.

    Envelope v2 additions over v1:

    - ``estimated_size_bytes`` / ``estimated_size_human`` may be ``None`` —
      honest "size unknown" for whole-file products (decision 4) rather than an
      invented number.
    - ``epistemic_status`` gains ``"unknown"`` (demoted whole-file product).
    - ``cost`` block (``{units, limit, exceeds_limit, source}`` or ``None``).
    - ``queue_latency_tier`` derives from ``cost.units`` when available, else the
      local field count.
    """
    # Lazy import to keep the estimator independent of the catalogue at module
    # load time; ``runtime_supports`` consults catalogue + routing table.
    from copernicus_mcp.backends.cds.backend import runtime_supports

    n_fields = count_fields(inputs)
    fraction = area_fraction(inputs)
    is_runtime_supported = runtime_supports(dataset_id)

    cost_block: dict[str, Any] | None = None
    if costing is not None:
        cost_block = {
            "units": costing.units,
            "limit": costing.limit,
            "exceeds_limit": costing.exceeds_limit,
            "source": "costing_api",
        }

    # Queue latency is per-field tape/disk fetch (research §6.5.4): cost.units is
    # the server's own field count when available, else our local product.
    tier_basis = int(costing.units) if costing is not None else n_fields
    tier = _tier_for_fields(tier_basis)

    # Decision 9 — resolve a calibrated bytes-per-unit for this request shape.
    cal_result = None
    if calibration is not None:
        from copernicus_mcp.backends.cds.calibration import signature

        cal_result = calibration.resolve(dataset_id, signature(inputs))

    # v2 honesty (live-test decision): show a byte size ONLY when it is
    # backed by your own prior downloads of this request shape (a calibration
    # result with ≥1 LOCAL observation). The bundled seed / curated map /
    # cross-signature median proved unreliable — a 15×-off number that was falsely
    # labelled "calibrated (0 observations)". Without local data we say "unknown"
    # rather than guess. The server cost units stay exact; only the byte size is
    # uncertain, and the user can probe a small slice to calibrate.
    bytes_estimate: int | None
    calibration_observations: int | None = None
    if cal_result is not None and (cal_result.n_obs >= 1 or cal_result.source == "seed"):
        units_basis = costing.units if costing is not None else n_fields
        calibration_observations = cal_result.n_obs
        bytes_estimate = max(int(units_basis * cal_result.bytes_per_unit * fraction), 0)
        if cal_result.n_obs >= 1:
            epistemic_status = "calibrated"
            advisory = (
                f"Calibrated from {cal_result.n_obs} of your prior download(s) for "
                "this request shape (cost_units × measured bytes_per_unit × "
                "area_fraction) — still an ESTIMATE, not exact: actual size can "
                "differ. See size_estimate_caveat."
            )
        else:
            # Seed-backed (signature-matched bundled defaults, no local downloads):
            # show a rough number but flag it loudly as approximate.
            epistemic_status = "approximate"
            advisory = (
                "APPROXIMATE size from the bundled seed (measured on reference "
                "requests, NOT your own downloads): cost_units × seed bytes_per_unit "
                "× area_fraction. Can be off by a large factor — see "
                "size_estimate_caveat. Download a slice of this shape to calibrate it."
            )
    else:
        bytes_estimate = None
        epistemic_status = "unknown"
        calibration_observations = 0
        if costing is not None and costing.units == 1.0 and n_fields == 1:
            advisory = (
                f"Size unknown: {dataset_id!r} serves a whole-period/whole-file "
                "product (cost==1 file), so the per-field heuristic does not "
                "apply. The first successful retrieval calibrates future estimates."
            )
        else:
            advisory = (
                "Size unknown — no calibration from your own downloads for this "
                f"request shape on {dataset_id!r}. For a real estimate, download a "
                "small slice first (e.g. one day and/or a small area); its actual "
                "size calibrates the full request. The server cost units are "
                "exact; only the byte size is uncertain."
            )

    human = _human_size(bytes_estimate) if bytes_estimate is not None else None

    envelope: dict[str, Any] = {
        "type": "free",
        "estimated_size_bytes": bytes_estimate,
        "estimated_size_human": human,
        "epistemic_status": epistemic_status,
        "runtime_compatible": is_runtime_supported,
        "advisory_message": advisory,
        "size_estimate_caveat": _SIZE_ESTIMATE_CAVEAT,
        "fields_count": n_fields,
        "queue_latency_tier": tier,
        "cost": cost_block,
    }
    if calibration_observations is not None:
        envelope["calibration_observations"] = calibration_observations
    return envelope
