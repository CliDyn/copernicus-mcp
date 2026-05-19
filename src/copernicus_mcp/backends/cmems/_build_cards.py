"""Dataset-card derivation helpers for T-CMEMS-HIER.

Takes a slim record (output of ``_catalogue_build.slim_marine_record``)
and adds the level-1 card fields: domain, region, data_type,
``variables_normalized``, ``spatial_label``, ``temporal_label``,
``quality_flags``. Pure-Python, no I/O at call time. The variables
lookup table is loaded once from
``_data/variables_lookup.json`` and cached.

Sub-plan: internal design notes. Schema lock: same
file, level-1 cards table.

NOT imported by ``CmemsBackend`` at runtime. Build-side only.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from copernicus_mcp.backends.cmems._annotate_cards import annotate_card

_DATA_DIR: Path = Path(__file__).resolve().parent / "_data"

# Domain map: SDK product-id tokens → canonical domain string.
# Tokenise product_id on ``_`` and look up each token; first hit wins.
# Domain captures WHAT the data is about (physics, biogeochemistry,
# sea ice, ...). Product-class markers (OMI, INSITU, MULTIOBS) belong
# to ``data_type``, not here — for ``ANTARCTIC_OMI_SI_extent`` the
# subject is sea ice, the class is an indicator.
#
# Coverage tuned against the bundled 1251-row snapshot — the long-form
# tokens (``OCEANCOLOUR``, ``SEALEVEL``, ``BATHYMETRY``, ``WAVE``) are
# what real CMEMS products actually use, not just the short codes.
_DOMAIN_TOKEN_MAP: dict[str, str] = {
    "OCEANCOLOUR": "ocean_color",
    "SST": "sst",
    "SEAICE": "sea_ice",
    "BATHYMETRY": "bathymetry",
    "BATHY": "bathymetry",
    "SEALEVEL": "sea_level",
    "SL": "sea_level",  # OMI_CLIMATE_SL_*, OMI_EXTREME_SL_*
    "WIND": "wind",
    "BGC": "biogeochemistry",
    "WAVE": "waves",
    "WAV": "waves",
    "SI": "sea_ice",
    "PHY": "physics",
    # OMI sub-tokens (round-1b coverage): OMI is a product class
    # (handled by data_type=indicator). The token AFTER OMI encodes
    # the actual subject domain.
    "TEMPSAL": "physics",
    "CIRCULATION": "physics",
    "HEALTH": "biogeochemistry",
    "SEASTATE": "waves",
    "WMHE": "physics",  # Water Mass Heat Exchange
    "NATLANTIC": "physics",
    "CLIMVAR": "indicator",  # composite climate-variability indicator
    # Composite multi-discipline marker on INSITU_*_PHYBGCWAV_* —
    # primarily physics with BGC and waves overlay.
    "PHYBGCWAV": "physics",
    # Round-1c: tail of OMI sub-tokens (all physics).
    "CURRENTS": "physics",
    "OHC": "physics",  # Ocean Heat Content
    "OFC": "physics",  # Ocean Fluid Content / coupling
    "OSC": "physics",  # Ocean Surface / State Currents
    "THSL": "physics",  # Thermohaline
    "MHW": "physics",  # Marine Heat Wave
    # codex round-1 PR #87: the only ``STATE`` token in the bundled
    # snapshot is ``OMI_EXTREME_STATE_GLOBAL_trend`` — a Significant
    # Wave Height indicator backed by ``WAVE_GLO_PHY_SWH_L4_MY_*``.
    # Sits in the waves family, not physics.
    "STATE": "waves",
    "WMF": "physics",  # Water Mass Formation
}

# Region map: SDK product-id tokens → list of region tags.
# Coverage tuned against the bundled snapshot — both short codes
# (``GLO``, ``ARC``, ``BLK``, ``BS``, ``ANT``, ``ATL``) and long-form
# names (``GLOBAL``, ``BALTICSEA``, ``NWSHELF``) appear in real
# CMEMS product_ids.
_REGION_TOKEN_MAP: dict[str, list[str]] = {
    "GLOBAL": ["global"],
    "GLO": ["global"],
    "ARCTIC": ["arctic"],
    "ARC": ["arctic"],
    "ANTARCTIC": ["antarctic", "southern_hemisphere"],
    "ANT": ["antarctic", "southern_hemisphere"],
    "MEDSEA": ["mediterranean"],
    "MED": ["mediterranean"],
    "BLKSEA": ["black_sea"],
    "BLK": ["black_sea"],
    "BS": ["black_sea"],
    "BAL": ["baltic"],
    "BALTIC": ["baltic"],
    "BALTICSEA": ["baltic"],
    "NWS": ["north_west_shelf"],
    "NWSHELF": ["north_west_shelf"],
    "NORTHWESTSHELF": ["north_west_shelf"],
    "IBI": ["iberia_biscay_ireland"],
    "ATL": ["atlantic"],
    "ATLANTIC": ["atlantic"],
    "PACIFIC": ["pacific"],
    "EUR": ["europe"],
    "EUROPE": ["europe"],
}

# data_type tiers: priority-ordered, first matching tier wins, no
# combining across tiers. Observation markers are checked before
# reanalysis because a multi-year observation series (which carries
# both ``INSITU`` and ``_my_``) is still an observation product —
# the ``_my_`` "multi-year" tag in that combination means "long
# time-series", not "reanalysis derived".
#
# **Case handling** (codex+cr round-1 HIGH): the haystack is
# lower-cased before matching, so both ``_MY_`` (uppercase, common
# in product_ids like ``OCEANCOLOUR_ARC_BGC_L4_MY_009_124``) and
# ``_my_`` (lowercase, common in dataset_ids) trigger the
# reanalysis tier. Markers below MUST be written lowercase.
_DATA_TYPE_TIERS: list[tuple[list[str], list[str]]] = [
    # CMEMS uses three forms for observation prefixes in dataset_ids:
    # ``_obs_`` (both underscores, rare), ``_obs-`` (underscore +
    # hyphen, e.g. ``cmems_obs-wind_*``), and ``-obs-`` (both
    # hyphens, e.g. ``cmems-obs-wave_*``). Plus the upper-case
    # tokens in product_id. Round-1b: ``obs-`` covers most real
    # CMEMS observation products including multi-year obs which
    # used to mis-classify as reanalysis via the ``_my_`` marker.
    #
    # cr round-1 HIGH on PR #87: ``observations`` must be DELIMITED
    # (``_observations_``). Bare ``observations`` matched product
    # title prose ("from Satellite Observations") and silently
    # mis-classified 79 OMI products as observation instead of
    # indicator. Same shape as the ``reanalysis`` bug PR #86
    # round-2 fixed.
    (
        ["insitu", "multiobs", "_observations_", "_obs_", "obs-"],
        ["observation"],
    ),
    # ``omi_`` matches both ``_omi_`` (middle) and ``omi_`` (leading
    # token of OMI_CLIMATE_* products). cr round-2 MEDIUM: leading-
    # OMI products used to fall through to the reanalysis tier when
    # their titles mentioned "reanalysis".
    (["omi_"], ["indicator"]),
    # Three forms in real CMEMS product_ids:
    #   ``_anfc_`` (short, e.g. GLOBAL_ANALYSISFORECAST_PHY_001_024
    #   in dataset_id form: cmems_mod_glo_phy_anfc_*).
    #   ``analysisforecast`` (one word, e.g.
    #   ARCTIC_ANALYSISFORECAST_PHY_TIDE_002_015).
    #   ``_analysis_forecast_`` (two words, e.g.
    #   ARCTIC_ANALYSIS_FORECAST_WAV_002_014).
    (
        ["_anfc_", "analysisforecast", "_analysis_forecast_"],
        ["analysis", "forecast"],
    ),
    # ``_reanalysis_`` is delimited (cr round-2 MEDIUM): the bare
    # ``reanalysis`` literal that round-1 added caused 4 OMI products
    # with that word in their title to be mis-classified.
    (["_my_", "_multiyear_", "_reanalysis_"], ["reanalysis"]),
    # Pure ``_nrt_`` (without an obs- prefix) is a near-real-time
    # model analysis product. Lowest priority so it only fires when
    # nothing else matched — keeps the existing classifications
    # stable while picking up the remaining unknowns. Round-1b.
    (["_nrt_"], ["analysis"]),
]


def _tokens(product_id: str) -> list[str]:
    """Split product_id on ``_`` for token-based inference."""
    return [t for t in (product_id or "").split("_") if t]


def infer_domain(product_id: str, title: str) -> str:
    """Return a single canonical domain string.

    Tokenises ``product_id`` and matches against ``_DOMAIN_TOKEN_MAP``.
    First matching token wins. Title is currently unused for domain
    (token-based inference is strong enough); kept in the signature
    for future title-keyword fallback. Returns ``"unknown"`` when no
    token matches.
    """
    for tok in _tokens(product_id):
        upper = tok.upper()
        if upper in _DOMAIN_TOKEN_MAP:
            return _DOMAIN_TOKEN_MAP[upper]
    return "unknown"


def infer_region(product_id: str, title: str) -> list[str]:
    """Return a list of canonical region tags. Multi-valued because a
    product can span more than one region tag (e.g. ANTARCTIC carries
    both ``antarctic`` and ``southern_hemisphere``).

    Returns ``["unknown"]`` when no token matches.
    """
    regions: list[str] = []
    for tok in _tokens(product_id):
        upper = tok.upper()
        if upper in _REGION_TOKEN_MAP:
            for r in _REGION_TOKEN_MAP[upper]:
                if r not in regions:
                    regions.append(r)
            # First-match-wins: don't keep scanning after a region hit,
            # to avoid double-tagging when a thematic token (e.g.
            # OCEANCOLOUR) is followed by a region token (ARC).
            return regions
    return ["unknown"]


def infer_data_type(dataset_id: str, product_id: str, title: str) -> list[str]:
    """Return a list of data-type tags. Multi-valued because some
    single tiers (analysis+forecast) carry combinations. Returns
    ``["unknown"]`` when no marker matches. Priority-ordered:
    observation > indicator > analysis/forecast > reanalysis.

    Case-insensitive: the haystack is lower-cased before scanning
    so uppercase product_id tokens like ``_MY_`` match the
    ``_my_`` marker (codex+cr round-1 HIGH).
    """
    haystack = f"{dataset_id or ''} {product_id or ''} {title or ''}".lower()
    for markers, tags in _DATA_TYPE_TIERS:
        if any(m in haystack for m in markers):
            return list(tags)
    return ["unknown"]


def _load_variables_lookup() -> dict[str, str]:
    """Load and cache the curated SDK-short-name → canonical-name map."""
    global _VARS_CACHE  # noqa: PLW0603 — module-level cache is intentional
    if _VARS_CACHE is None:
        raw = json.loads((_DATA_DIR / "variables_lookup.json").read_text(encoding="utf-8"))
        _VARS_CACHE = {k: v for k, v in raw.items() if not k.startswith("_")}
    return _VARS_CACHE


_VARS_CACHE: dict[str, str] | None = None


def normalize_variables(short_names: list[str]) -> list[str]:
    """Map SDK short names to canonical names via the curated lookup.

    Unknown short names appear verbatim in the output (never silently
    dropped — codex spec-review MEDIUM-3 fallback rule). The result
    is deduped while preserving insertion order, so several short
    names that collapse to the same canonical only produce one entry.
    """
    lookup = _load_variables_lookup()
    out: list[str] = []
    for sn in short_names:
        canonical = lookup.get(sn, sn)
        if canonical not in out:
            out.append(canonical)
    return out


def bbox_to_spatial_label(bbox: dict[str, float] | None) -> str | None:
    """Render a slim record's ``spatial_extent`` dict as a short
    human-readable string.

    A handful of rule-based ranges cover the most common CMEMS
    extents (global, antarctic, arctic, regional seas). Falls back
    to a coordinate dump when no range matches.
    """
    if not isinstance(bbox, dict):
        return None
    try:
        min_lon = float(bbox["min_lon"])
        min_lat = float(bbox["min_lat"])
        max_lon = float(bbox["max_lon"])
        max_lat = float(bbox["max_lat"])
    except (KeyError, TypeError, ValueError):
        return None

    # Global: full longitude span (or close enough) + most of latitude.
    if min_lon <= -179.0 and max_lon >= 179.0 and min_lat <= -89.0 and max_lat >= 89.0:
        return "global"
    # Antarctic: full lon span, latitude below ~-50.
    if min_lon <= -179.0 and max_lon >= 179.0 and max_lat <= -40.0:
        return f"antarctic, {min_lat:.0f} to {max_lat:.0f} latitude"
    # Arctic: full lon span, latitude above ~50.
    if min_lon <= -179.0 and max_lon >= 179.0 and min_lat >= 40.0:
        return f"arctic, {min_lat:.0f} to {max_lat:.0f} latitude"
    # Fallback: explicit numeric range.
    return f"lon {min_lon:.2f} to {max_lon:.2f}, lat {min_lat:.2f} to {max_lat:.2f}"


_PRESENT_THRESHOLD_DAYS = 30
# Forecast horizons longer than this become explicit "(forecast)"
# labels rather than the bare "to present" string. Anything within
# a month either way of "now" is collapsed to "to present".
_FORECAST_HORIZON_DAYS = 30


def temporal_extent_to_label(
    extent: dict[str, str] | None,
) -> str | None:
    """Render a slim record's ``temporal_extent`` dict as a short
    human-readable label.

    Cases:
      - ``end`` within ±30 days of "now" → ``"<start> to present"``.
      - ``end`` more than 30 days in the FUTURE (long forecast
        horizon, e.g. a 1-year prediction) → ``"<start> to <end>
        (forecast)"``. cr round-1 HIGH: bare ``"to present"`` for
        arbitrarily far-future ends silently overclaimed currency.
      - Otherwise (historical) → ``"<start> to <end>"``.
    """
    if not isinstance(extent, dict):
        return None
    start_iso = extent.get("start")
    end_iso = extent.get("end")
    if not isinstance(start_iso, str) or not isinstance(end_iso, str):
        return None
    try:
        start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
    except ValueError:
        return None

    start_label = start_dt.strftime("%Y-%m")
    end_label = end_dt.strftime("%Y-%m")
    now = datetime.now(UTC)
    delta = now - end_dt
    # Recent past OR near future (within ±30 days): "to present".
    if abs(delta) <= timedelta(days=_PRESENT_THRESHOLD_DAYS):
        return f"{start_label} to present"
    # Far future: explicit forecast horizon label.
    if delta < timedelta(0):
        return f"{start_label} to {end_label} (forecast)"
    # Historical.
    return f"{start_label} to {end_label}"


def quality_flags(card: dict[str, Any]) -> list[str]:
    """Compute the ``quality_flags`` list for a card.

    Flags fire when a card has missing extents or low-confidence
    inferences. Cards with full coverage + confident domain/region
    return ``[]``.
    """
    flags: list[str] = []
    if card.get("spatial_extent") is None:
        flags.append("no_spatial_extent")
    if card.get("temporal_extent") is None:
        flags.append("no_temporal_extent")
    if card.get("domain") == "unknown":
        flags.append("low_confidence_domain")
    region = card.get("region")
    if isinstance(region, list) and region == ["unknown"]:
        flags.append("low_confidence_region")
    return flags


def build_dataset_card(slim: dict[str, Any]) -> dict[str, Any]:
    """Project a slim record into a level-1 card with derived fields.

    All slim record fields pass through verbatim; new fields are
    appended: ``domain``, ``region``, ``data_type``,
    ``variables_normalized``, ``spatial_label``, ``temporal_label``,
    ``quality_flags``, plus ``best_for`` / ``not_good_for`` placeholders
    (LLM-enrichable in T-CMEMS-HIER-004; ``None`` until then).
    """
    product_id = slim.get("product_id") or ""
    title = slim.get("product_title") or ""
    dataset_id = slim.get("dataset_id") or ""

    domain = infer_domain(product_id, title)
    region = infer_region(product_id, title)
    data_type = infer_data_type(dataset_id, product_id, title)
    variables_normalized = normalize_variables(slim.get("variables") or [])
    spatial_label = bbox_to_spatial_label(slim.get("spatial_extent"))
    temporal_label = temporal_extent_to_label(slim.get("temporal_extent"))

    card = {
        "dataset_id": slim.get("dataset_id"),
        "dataset_name": slim.get("dataset_name"),
        "title": slim.get("title"),
        "product_id": slim.get("product_id"),
        "product_title": slim.get("product_title"),
        "description": slim.get("description"),
        "doi": slim.get("doi"),
        "service_types": slim.get("service_types") or [],
        "variables": slim.get("variables") or [],
        "variables_normalized": variables_normalized,
        "versions": slim.get("versions") or [],
        "spatial_extent": slim.get("spatial_extent"),
        "spatial_label": spatial_label,
        "temporal_extent": slim.get("temporal_extent"),
        "temporal_label": temporal_label,
        "domain": domain,
        "region": region,
        "data_type": data_type,
    }
    # T-CMEMS-HIER-004: rule-based best_for / not_good_for derived
    # from data_type + domain. The annotator reads only the fields
    # already in ``card`` so the call is pure and deterministic.
    # An external review system (Phase 2) refines these with LLM
    # judgment; until then, the rule-based baseline ships in the
    # bundled cards.
    best, not_good = annotate_card(card)
    card["best_for"] = best
    card["not_good_for"] = not_good
    # quality_flags reads from the assembled dict.
    card["quality_flags"] = quality_flags(card)
    return card
