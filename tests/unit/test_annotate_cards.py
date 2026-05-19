"""T-CMEMS-HIER-004 rule-based card annotator tests.

``annotate_card`` fills the ``best_for`` and ``not_good_for`` fields
of a dataset card based on its ``data_type`` and ``domain``. Output
is deterministic so the bundle diff stays minimal across refreshes.

The annotations are a *v1 baseline* — a downstream review system
(Phase 2 of T-CMEMS-HIER-004) refines them with LLM judgment. The
rule-based baseline gives the router useful signal even before that
refinement lands.
"""

from __future__ import annotations

from typing import Any


def _card(
    *,
    data_type: list[str] | None = None,
    domain: str = "physics",
) -> dict[str, Any]:
    return {
        "dataset_id": "test",
        "product_id": "TEST",
        "domain": domain,
        "data_type": data_type if data_type is not None else ["analysis"],
    }


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_annotate_card_returns_lists_of_strings() -> None:
    from copernicus_mcp.backends.cmems._annotate_cards import annotate_card

    best, not_good = annotate_card(_card())
    assert isinstance(best, list) and all(isinstance(x, str) for x in best)
    assert isinstance(not_good, list) and all(isinstance(x, str) for x in not_good)


def test_annotate_card_returns_non_empty_lists_for_known_combos() -> None:
    """Every realistic data_type × domain combo gets at least one
    ``best_for`` and one ``not_good_for`` hint."""
    from copernicus_mcp.backends.cmems._annotate_cards import annotate_card

    best, not_good = annotate_card(_card(data_type=["forecast"], domain="physics"))
    assert best, "forecast/physics must surface at least one best_for"
    assert not_good, "forecast/physics must surface at least one not_good_for"


# ---------------------------------------------------------------------------
# data_type-driven semantics
# ---------------------------------------------------------------------------


def test_annotate_card_forecast_is_for_near_real_time() -> None:
    from copernicus_mcp.backends.cmems._annotate_cards import annotate_card

    best, _ = annotate_card(_card(data_type=["analysis", "forecast"]))
    text = " ".join(best).lower()
    assert "near-real-time" in text or "operational" in text


def test_annotate_card_forecast_excludes_climate_baseline() -> None:
    """A forecast product is intentionally not the right tool for
    climate-baseline reconstruction — say so."""
    from copernicus_mcp.backends.cmems._annotate_cards import annotate_card

    _, not_good = annotate_card(_card(data_type=["analysis", "forecast"]))
    text = " ".join(not_good).lower()
    assert "climate" in text or "baseline" in text or "historical" in text


def test_annotate_card_reanalysis_is_for_historical_baselines() -> None:
    from copernicus_mcp.backends.cmems._annotate_cards import annotate_card

    best, _ = annotate_card(_card(data_type=["reanalysis"]))
    text = " ".join(best).lower()
    assert "climate" in text or "historical" in text or "baseline" in text


def test_annotate_card_reanalysis_excludes_near_real_time() -> None:
    from copernicus_mcp.backends.cmems._annotate_cards import annotate_card

    _, not_good = annotate_card(_card(data_type=["reanalysis"]))
    text = " ".join(not_good).lower()
    assert "near-real-time" in text or "operational" in text


def test_annotate_card_observation_is_for_validation_and_ground_truth() -> None:
    from copernicus_mcp.backends.cmems._annotate_cards import annotate_card

    best, _ = annotate_card(_card(data_type=["observation"]))
    text = " ".join(best).lower()
    assert "validation" in text or "ground" in text or "observation" in text


def test_annotate_card_indicator_is_for_trend_and_policy() -> None:
    from copernicus_mcp.backends.cmems._annotate_cards import annotate_card

    best, _ = annotate_card(_card(data_type=["indicator"], domain="indicator"))
    text = " ".join(best).lower()
    assert "trend" in text or "policy" in text or "indicator" in text


# ---------------------------------------------------------------------------
# domain hints layered on top
# ---------------------------------------------------------------------------


def test_annotate_card_physics_domain_surfaces_ocean_state_keywords() -> None:
    from copernicus_mcp.backends.cmems._annotate_cards import annotate_card

    best, _ = annotate_card(_card(data_type=["analysis", "forecast"], domain="physics"))
    text = " ".join(best).lower()
    assert "temperature" in text or "salinity" in text or "current" in text


def test_annotate_card_sea_ice_domain_surfaces_ice_keywords() -> None:
    from copernicus_mcp.backends.cmems._annotate_cards import annotate_card

    best, _ = annotate_card(_card(data_type=["analysis", "forecast"], domain="sea_ice"))
    text = " ".join(best).lower()
    assert "ice" in text


def test_annotate_card_biogeochemistry_surfaces_carbon_or_nutrient_keywords() -> None:
    from copernicus_mcp.backends.cmems._annotate_cards import annotate_card

    best, _ = annotate_card(_card(data_type=["reanalysis"], domain="biogeochemistry"))
    text = " ".join(best).lower()
    assert "carbon" in text or "nutrient" in text or "oxygen" in text or "chlorophyll" in text


def test_annotate_card_ocean_color_surfaces_chlorophyll_keywords() -> None:
    from copernicus_mcp.backends.cmems._annotate_cards import annotate_card

    best, _ = annotate_card(_card(data_type=["observation"], domain="ocean_color"))
    text = " ".join(best).lower()
    assert "chlorophyll" in text or "ocean color" in text or "productivity" in text


# ---------------------------------------------------------------------------
# Determinism and idempotence
# ---------------------------------------------------------------------------


def test_annotate_card_is_deterministic() -> None:
    """Same input → same output, byte-identical. Required for stable
    bundle diffs."""
    from copernicus_mcp.backends.cmems._annotate_cards import annotate_card

    card = _card(data_type=["analysis", "forecast"], domain="physics")
    a, b = annotate_card(card)
    c, d = annotate_card(card)
    assert (a, b) == (c, d)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_annotate_card_tolerates_missing_data_type() -> None:
    from copernicus_mcp.backends.cmems._annotate_cards import annotate_card

    best, not_good = annotate_card({"product_id": "X", "domain": "physics"})
    # No crash, lists possibly empty (no data_type → no template fires).
    assert isinstance(best, list) and isinstance(not_good, list)


def test_annotate_card_tolerates_unknown_domain() -> None:
    from copernicus_mcp.backends.cmems._annotate_cards import annotate_card

    best, not_good = annotate_card(_card(data_type=["analysis"], domain="totally_made_up_domain"))
    # Domain-specific hints don't fire, but data_type-driven ones do.
    assert isinstance(best, list)
    assert isinstance(not_good, list)


# ---------------------------------------------------------------------------
# Bulk operation against the bundled cards
# ---------------------------------------------------------------------------


def test_annotate_card_applies_dataset_specific_override() -> None:
    """T-CMEMS-HIER-004 round-2: an external review surfaced
    5 datasets whose generic rule-based annotation is misleading.
    The annotator now checks a per-dataset_id override table first
    and uses those lists verbatim when present, falling back to the
    rule-based templates otherwise."""
    from copernicus_mcp.backends.cmems._annotate_cards import (
        _DATASET_OVERRIDES,
        annotate_card,
    )

    # Pick any override entry from the table.
    sample_id = next(iter(_DATASET_OVERRIDES))
    override = _DATASET_OVERRIDES[sample_id]
    card = _card(data_type=["analysis", "forecast"], domain="physics")
    card["dataset_id"] = sample_id

    best, not_good = annotate_card(card)
    assert best == override["best_for"]
    assert not_good == override["not_good_for"]


def test_annotate_card_override_takes_precedence_over_rule_based() -> None:
    """The override is verbatim — the rule-based template's
    contributions for this card's data_type / domain do NOT bleed
    into the override output."""
    from copernicus_mcp.backends.cmems._annotate_cards import (
        _DATASET_OVERRIDES,
        annotate_card,
    )

    sample_id = next(iter(_DATASET_OVERRIDES))
    card = _card(data_type=["analysis", "forecast"], domain="physics")
    card["dataset_id"] = sample_id

    best, _ = annotate_card(card)
    # Rule-based template contributes "near-real-time monitoring";
    # if it's present, the override wasn't applied verbatim.
    assert best == _DATASET_OVERRIDES[sample_id]["best_for"]


def test_annotate_bundled_cards_yields_non_empty_lists_for_most_cards() -> None:
    """Smoke against the bundled snapshot: every card with a known
    data_type lands with non-empty annotations."""
    import json
    from pathlib import Path

    from copernicus_mcp.backends.cmems._annotate_cards import annotate_card

    data_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "copernicus_mcp"
        / "backends"
        / "cmems"
        / "_data"
    )
    cards = json.loads((data_dir / "dataset_cards.json").read_text())
    empty_best = 0
    empty_not = 0
    for c in cards:
        best, not_good = annotate_card(c)
        if not best:
            empty_best += 1
        if not not_good:
            empty_not += 1
    # All bundled cards have a valid data_type (0% unknown) so the
    # annotator should fire for every one.
    assert empty_best == 0, f"{empty_best} bundled cards have empty best_for"
    assert empty_not == 0, f"{empty_not} bundled cards have empty not_good_for"
