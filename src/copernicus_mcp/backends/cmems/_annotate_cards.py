"""Rule-based ``best_for`` / ``not_good_for`` annotator.

T-CMEMS-HIER-004 Phase 1. The runtime router consults
``dataset_cards.json`` after a product has been short-listed; the
two annotation lists tell the router *what each dataset is for* and
*what it is not*. They are the per-card analogue of the per-group
``include_when_query_mentions`` / ``exclude_when_query_mentions``
hints in ``groups.json``.

This is a v1 baseline. The annotations are derived from
``data_type`` (forecast / reanalysis / observation / analysis /
indicator) layered with a domain hint (physics / biogeochemistry /
sea_ice / ocean_color / ...). An external review system (Phase 2)
refines the wording and adds dataset-specific overrides; until
then, the rule-based baseline gives the router useful signal on
every card.

Determinism is a hard requirement: the bundled
``dataset_cards.json`` must diff cleanly across refreshes, so
``annotate_card`` is stateless and the output lists are
sort-stable.
"""

from __future__ import annotations

from typing import Any

# data_type-driven templates. The first axis of every annotation —
# "what TIME REGIME does this dataset belong to" — answers the
# router's main relevance question.
_DATA_TYPE_BEST: dict[str, list[str]] = {
    "forecast": [
        "near-real-time monitoring",
        "operational decision support",
        "short-range planning (days to weeks ahead)",
    ],
    "analysis": [
        "near-real-time monitoring",
        "best-estimate present-day ocean state",
    ],
    "reanalysis": [
        "climate baselines and historical reconstruction",
        "multi-decadal trend analysis",
        "model validation against a long record",
    ],
    "observation": [
        "validation against in-situ or satellite truth",
        "ground-truth for model assessment",
        "data assimilation inputs",
    ],
    "indicator": [
        "trend and anomaly monitoring",
        "policy and reporting summaries",
        "concise ocean-state diagnostics",
    ],
}

_DATA_TYPE_NOT_GOOD: dict[str, list[str]] = {
    "forecast": [
        "climate-baseline reconstruction (use a reanalysis instead)",
        "long historical record requiring stable methodology",
    ],
    "analysis": [
        "climate-baseline reconstruction (use a reanalysis instead)",
    ],
    "reanalysis": [
        "near-real-time or operational use (data are produced with delay)",
        "the most recent days/weeks (reanalyses stop earlier)",
    ],
    "observation": [
        "spatially-complete fields (observations are sparse in space and time)",
        "uniform global coverage at every time step",
    ],
    "indicator": [
        "high-resolution spatial or temporal analysis",
        "primary data ingestion (indicators are summaries)",
    ],
}

# Per-dataset overrides surfaced by the round-2 external review.
# Keyed by ``dataset_id``; value is a dict with ``best_for`` and
# ``not_good_for`` lists used verbatim, bypassing the rule-based
# templates. Use sparingly — every entry here is hand-curated and
# carries a maintenance cost. Most cards should ride on the
# data_type/domain templates.
_DATASET_OVERRIDES: dict[str, dict[str, list[str]]] = {
    "cmems_mod_blk_phy-cur_anfc_2.5km_P1D-m": {
        "best_for": [
            "Black Sea 3D current velocities (eastward, northward, vertical)",
            "circulation patterns, eddies, and transport pathways at daily scales",
            "near-real-time monitoring",
            "operational decision support",
        ],
        "not_good_for": [
            "temperature or salinity studies (variables absent from this product)",
            "climate baselines or multi-decadal trend analysis",
        ],
    },
    "cmems_mod_blk_bgc-car_anfc_2.5km_P1D-m": {
        "best_for": [
            "Black Sea carbonate-system studies (dissolved inorganic carbon, alkalinity, pH)",
            "ocean-acidification monitoring",
            "air-sea CO2 flux calculations",
            "near-real-time monitoring",
        ],
        "not_good_for": [
            "primary productivity, oxygen, or nutrient budgets",
            "chlorophyll or nutrient analyses",
        ],
    },
    "cmems_mod_blk_bgc-pp-o2_anfc_2.5km_P1D-m": {
        "best_for": [
            "Black Sea net primary production and dissolved oxygen (3D and bottom)",
            "ecosystem productivity assessments",
            "hypoxia monitoring",
            "fisheries habitat studies",
        ],
        "not_good_for": [
            "carbonate-system variables (pH, CO2 flux)",
            "long-term climate reconstruction",
        ],
    },
    "cmems_mod_arc_bgc_anfc_ecosmo_P1D-m": {
        "best_for": [
            "Arctic biogeochemistry (chlorophyll, nutrients, primary production, oxygen, carbonate system) at 6.25 km",
            "monitoring Arctic productivity and nutrient cycling",
            "near-real-time monitoring",
        ],
        "not_good_for": [
            "physical circulation diagnostics (currents, sea level)",
            "long-term climatology — use the multi-year reanalysis instead",
        ],
    },
    "antarctic_omi_si_extent": {
        "best_for": [
            "Antarctic sea-ice extent indicator from reanalysis and observations",
            "policy reporting and large-scale variability monitoring",
            "trend monitoring",
        ],
        "not_good_for": [
            "high-resolution spatial analysis",
            "primary data ingestion (this is an aggregated indicator)",
        ],
    },
}

# Domain-driven hints layered on top. Each entry contributes a
# domain-specific variable/topic hint to ``best_for`` so the router
# can match a query like "sea ice extent" without re-reading the
# full card.
_DOMAIN_HINTS: dict[str, str] = {
    "physics": "ocean state variables (temperature, salinity, currents, sea surface height)",
    "biogeochemistry": "ocean carbon, nutrients, oxygen, and chlorophyll budgets",
    "ocean_color": "ocean colour and chlorophyll-a / productivity proxies from satellite",
    "sea_ice": "sea-ice extent, concentration, thickness, and drift",
    "sea_level": "sea-surface height anomaly and sea-level rise diagnostics",
    "wind": "surface wind stress and atmosphere-ocean coupling",
    "waves": "significant wave height, period, direction, and extreme sea state",
    "sst": "sea-surface temperature fields and SST gradients",
    "bathymetry": "ocean bathymetry and seafloor topography",
    "indicator": "synthesised ocean monitoring indicators (OMIs)",
}


def annotate_card(card: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return ``(best_for, not_good_for)`` lists for a dataset card.

    Resolution order:
    0. If ``card["dataset_id"]`` matches an entry in
       ``_DATASET_OVERRIDES``, use those lists verbatim and skip
       the rule-based templates. (Round-2 external review.)
    1. Otherwise each entry in ``card["data_type"]`` contributes its
       template to both lists. Combined types (e.g. ``["analysis",
       "forecast"]``) union both contributions and dedupe.
    2. The ``card["domain"]`` contributes one hint string to
       ``best_for``.

    Output lists are sorted to keep the bundled JSON diff stable
    across refreshes."""
    dataset_id = card.get("dataset_id")
    if isinstance(dataset_id, str) and dataset_id in _DATASET_OVERRIDES:
        override = _DATASET_OVERRIDES[dataset_id]
        return list(override["best_for"]), list(override["not_good_for"])

    data_types = card.get("data_type") or []
    if not isinstance(data_types, list):
        data_types = []
    domain = card.get("domain")
    if not isinstance(domain, str):
        domain = ""

    best: set[str] = set()
    not_good: set[str] = set()
    for dt in data_types:
        if not isinstance(dt, str):
            continue
        for hint in _DATA_TYPE_BEST.get(dt, ()):
            best.add(hint)
        for hint in _DATA_TYPE_NOT_GOOD.get(dt, ()):
            not_good.add(hint)

    if domain in _DOMAIN_HINTS:
        best.add(_DOMAIN_HINTS[domain])

    return sorted(best), sorted(not_good)
