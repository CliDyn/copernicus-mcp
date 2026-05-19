"""T-CMEMS-HIER-003: product manifest aggregator tests.

``build_products`` collapses N dataset cards into one product entry,
unioning categorical axes (domain, region, data_type, variables) and
generating a rule-based summary string. Runtime routing
(T-CMEMS-HIER-005) reads ``products.json`` first to short-list a
small set of products, then drills into ``dataset_cards.json`` —
keeping the per-query token footprint bounded.

These tests pin the aggregation contract: union semantics, sort
stability, summary non-emptiness, and a few invariants that protect
the runtime from surprises (e.g. a product with no datasets is
filtered out rather than emitted as a stub).
"""

from __future__ import annotations

from typing import Any

import pytest


def _card(
    product_id: str,
    dataset_id: str,
    *,
    domain: str = "physics",
    region: list[str] | None = None,
    data_type: list[str] | None = None,
    variables: list[str] | None = None,
    variables_normalized: list[str] | None = None,
    product_title: str = "Test Product",
    description: str = "A test product.",
    doi: str = "10.0/test",
) -> dict[str, Any]:
    """Build a minimal dataset card shape — only the fields
    ``build_products`` reads."""
    return {
        "dataset_id": dataset_id,
        "product_id": product_id,
        "product_title": product_title,
        "description": description,
        "doi": doi,
        "domain": domain,
        "region": region if region is not None else ["global"],
        "data_type": data_type if data_type is not None else ["analysis"],
        "variables": variables if variables is not None else ["thetao"],
        "variables_normalized": (
            variables_normalized
            if variables_normalized is not None
            else ["sea_water_potential_temperature"]
        ),
    }


# ---------------------------------------------------------------------------
# Shape and grouping
# ---------------------------------------------------------------------------


def test_build_products_groups_cards_by_product_id() -> None:
    """Two cards with the same ``product_id`` become one product
    entry; one with a different ``product_id`` is its own entry."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    cards = [
        _card("A", "a1"),
        _card("A", "a2"),
        _card("B", "b1"),
    ]
    products = build_products(cards)
    by_id = {p["product_id"]: p for p in products}
    assert set(by_id) == {"A", "B"}
    assert by_id["A"]["dataset_count"] == 2
    assert sorted(by_id["A"]["dataset_ids"]) == ["a1", "a2"]
    assert by_id["B"]["dataset_count"] == 1


def test_build_products_returns_empty_list_for_empty_input() -> None:
    from copernicus_mcp.backends.cmems._build_products import build_products

    assert build_products([]) == []


def test_build_products_entries_have_locked_schema() -> None:
    """Pin the full set of keys an entry exposes — runtime routing
    relies on this stability."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    [product] = build_products([_card("X", "x1")])
    assert sorted(product.keys()) == sorted(
        [
            "product_id",
            "product_title",
            "description",
            "doi",
            "dataset_ids",
            "dataset_count",
            "domains",
            "regions",
            "data_types",
            "variables",
            "variables_normalized",
            "summary",
        ]
    )


# ---------------------------------------------------------------------------
# Union semantics — domain / region / data_type / variables
# ---------------------------------------------------------------------------


def test_build_products_unions_domains_across_member_cards() -> None:
    """Multi-dataset products usually share one domain, but cross-domain
    products exist (e.g. PHYBGCWAV products combining physics +
    biogeochemistry + waves)."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    cards = [
        _card("Z", "z1", domain="physics"),
        _card("Z", "z2", domain="biogeochemistry"),
        _card("Z", "z3", domain="waves"),
    ]
    [product] = build_products(cards)
    assert product["domains"] == ["biogeochemistry", "physics", "waves"]


def test_build_products_unions_regions_flattening_lists() -> None:
    """Card ``region`` is a list of region tokens. Product unions
    those lists into a flat sorted set."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    cards = [
        _card("Z", "z1", region=["antarctic", "southern_hemisphere"]),
        _card("Z", "z2", region=["global"]),
    ]
    [product] = build_products(cards)
    assert product["regions"] == ["antarctic", "global", "southern_hemisphere"]


def test_build_products_unions_data_types() -> None:
    """A product can carry both analysis+forecast datasets and reanalysis
    datasets (e.g. GLOBAL_ANALYSISFORECAST_PHY products usually have
    forecast variants of reanalysis siblings under a different product)."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    cards = [
        _card("Z", "z1", data_type=["analysis", "forecast"]),
        _card("Z", "z2", data_type=["reanalysis"]),
    ]
    [product] = build_products(cards)
    assert product["data_types"] == ["analysis", "forecast", "reanalysis"]


def test_build_products_unions_variables_dropping_duplicates() -> None:
    """Variables are sometimes repeated across datasets of one product
    (different spatial/temporal slices of the same variable). The union
    must dedupe."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    cards = [
        _card("Z", "z1", variables=["thetao", "so"]),
        _card("Z", "z2", variables=["so", "uo", "vo"]),
    ]
    [product] = build_products(cards)
    assert product["variables"] == ["so", "thetao", "uo", "vo"]


def test_build_products_unions_normalized_variables() -> None:
    """variables_normalized maps short-names to CF canonical names.
    The union may collapse to fewer entries than ``variables`` does
    (two SDK shortnames mapping to one CF name)."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    cards = [
        _card(
            "Z",
            "z1",
            variables_normalized=["sea_water_temperature", "sea_water_salinity"],
        ),
        _card(
            "Z",
            "z2",
            variables_normalized=["sea_water_salinity", "sea_water_density"],
        ),
    ]
    [product] = build_products(cards)
    assert product["variables_normalized"] == [
        "sea_water_density",
        "sea_water_salinity",
        "sea_water_temperature",
    ]


# ---------------------------------------------------------------------------
# Sort stability
# ---------------------------------------------------------------------------


def test_build_products_output_is_sorted_by_product_id() -> None:
    """Stable ordering keeps git diffs of the bundled
    ``products.json`` minimal across refreshes."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    cards = [
        _card("C", "c1"),
        _card("A", "a1"),
        _card("B", "b1"),
    ]
    products = build_products(cards)
    assert [p["product_id"] for p in products] == ["A", "B", "C"]


def test_build_products_dataset_ids_are_sorted() -> None:
    """Same reason — stable diffs."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    cards = [
        _card("Z", "z3"),
        _card("Z", "z1"),
        _card("Z", "z2"),
    ]
    [product] = build_products(cards)
    assert product["dataset_ids"] == ["z1", "z2", "z3"]


# ---------------------------------------------------------------------------
# Product-level metadata from member cards
# ---------------------------------------------------------------------------


def test_build_products_takes_product_title_from_first_card() -> None:
    """All cards in a product share product_title at the SDK level.
    Pin that we lift it onto the product."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    [product] = build_products(
        [
            _card("Z", "z1", product_title="Global Physics Analysis and Forecast"),
            _card("Z", "z2", product_title="Global Physics Analysis and Forecast"),
        ]
    )
    assert product["product_title"] == "Global Physics Analysis and Forecast"


def test_build_products_lifts_description_and_doi() -> None:
    from copernicus_mcp.backends.cmems._build_products import build_products

    [product] = build_products(
        [
            _card("Z", "z1", description="A product description.", doi="10.0/xyz"),
        ]
    )
    assert product["description"] == "A product description."
    assert product["doi"] == "10.0/xyz"


# ---------------------------------------------------------------------------
# Summary template
# ---------------------------------------------------------------------------


def test_build_products_summary_is_non_empty_for_every_product() -> None:
    """Acceptance criterion: every product gets a non-empty summary."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    cards = [
        _card("A", "a1"),
        _card("B", "b1", domain="biogeochemistry", region=["mediterranean"]),
        _card("C", "c1", data_type=["observation"]),
    ]
    products = build_products(cards)
    for p in products:
        assert isinstance(p["summary"], str) and p["summary"].strip()


def test_build_products_summary_mentions_dataset_count_and_categorical_axes() -> None:
    """The summary is the runtime's one-line product hint. It must
    surface enough axes that a router can confirm relevance from the
    summary alone."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    [product] = build_products(
        [
            _card(
                "Z",
                "z1",
                product_title="Global Ocean Physics Analysis and Forecast",
                domain="physics",
                region=["global"],
                data_type=["analysis", "forecast"],
                variables=["thetao", "so", "uo", "vo", "zos"],
            ),
        ]
    )
    summary = product["summary"]
    assert "Global Ocean Physics Analysis and Forecast" in summary
    assert "physics" in summary
    assert "global" in summary
    assert "analysis" in summary or "forecast" in summary
    # Variable hints in the summary — at least one canonical short name.
    assert "thetao" in summary or "so" in summary


def test_build_products_summary_prefers_canonical_variables() -> None:
    """cr round-1 PR #89 MEDIUM: alphabetical sort biased the summary
    of large physics products toward grid bookkeeping (``e1t``,
    ``deptho``) instead of the science variables that actually carry
    routing signal. The picker now drops bookkeeping shortnames and
    derived statistics, and prefers CF-canonical names from
    ``variables_normalized`` over short shortnames."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    [product] = build_products(
        [
            _card(
                "Z",
                "z1",
                variables=[
                    "bottomT",
                    "bottomT_mean",
                    "bottomT_std",
                    "deptho",
                    "e1t",
                    "e2t",
                    "e3t",
                    "mask",
                    "thetao",
                    "so",
                ],
                variables_normalized=[
                    "bottomT_mean",
                    "bottomT_std",
                    "deptho",
                    "e1t",
                    "e2t",
                    "e3t",
                    "sea_water_potential_temperature",
                    "sea_water_salinity",
                ],
            ),
        ]
    )
    summary = product["summary"]
    assert "sea_water_potential_temperature" in summary
    assert "sea_water_salinity" in summary
    # Grid metrics and derived stats must NOT appear.
    assert "e1t" not in summary
    assert "deptho" not in summary
    assert "bottomT_mean" not in summary


def test_build_products_summary_keeps_derived_when_base_absent() -> None:
    """If only the ``_mean`` variant exists (no plain base), keep it —
    losing the only available form would drop the variable from the
    router's view entirely."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    [product] = build_products(
        [
            _card(
                "Z",
                "z1",
                variables=["mlotst_mean"],
                variables_normalized=["mlotst_mean"],
            ),
        ]
    )
    assert "mlotst_mean" in product["summary"]


def test_build_products_strips_whitespace_from_product_id() -> None:
    """cr round-1 PR #89 LOW: a card with ``product_id=" FOO"`` must
    land in the same bucket as ``"FOO"`` — silent duplication on
    SDK shape drift would otherwise double-count the product."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    products = build_products(
        [
            _card("FOO", "a"),
            _card(" FOO ", "b"),
            _card("FOO\t", "c"),
        ]
    )
    assert len(products) == 1
    assert products[0]["product_id"] == "FOO"
    assert products[0]["dataset_count"] == 3


def test_build_products_unions_domains_split_on_comma() -> None:
    """cr round-1 PR #89 MEDIUM: a latent SDK shape carrying a
    comma-delimited domain string must contribute multiple tokens,
    not one composite. ``_union_strings`` splits on comma + strips."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    [product] = build_products(
        [
            _card("Z", "z1", domain="physics, biogeochemistry"),
            _card("Z", "z2", domain="waves"),
        ]
    )
    assert product["domains"] == ["biogeochemistry", "physics", "waves"]


def test_build_products_summary_omits_variables_when_all_bookkeeping() -> None:
    """cr round-2 PR #89 LOW: the picker drops bookkeeping shortnames.
    If a card has nothing else, the summary must not synthesise an
    empty "Variables include ." sentence."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    [product] = build_products(
        [
            _card(
                "Z",
                "z1",
                variables=["e1t", "e2t", "mask"],
                variables_normalized=["e1t", "e2t", "mask"],
            ),
        ]
    )
    assert "Variables include" not in product["summary"]


def test_build_products_summary_uses_comma_separator_for_data_types() -> None:
    """cr round-1 PR #89 LOW: ``+`` was inconsistent with the other
    axes. Comma-space everywhere."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    [product] = build_products([_card("Z", "z1", data_type=["analysis", "forecast"])])
    assert "analysis, forecast" in product["summary"]
    assert "analysis+forecast" not in product["summary"]


def test_build_products_handles_multi_axis_products() -> None:
    """A product spanning multiple domains/regions must surface every
    axis (e.g. "physics, biogeochemistry"). Empirically rare on real
    CMEMS but must not silently truncate."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    [product] = build_products(
        [
            _card("Z", "z1", domain="physics", region=["global"]),
            _card("Z", "z2", domain="biogeochemistry", region=["arctic"]),
        ]
    )
    summary = product["summary"]
    assert "physics" in summary
    assert "biogeochemistry" in summary
    assert "global" in summary
    assert "arctic" in summary


# ---------------------------------------------------------------------------
# Robustness — odd inputs that real CMEMS data could produce
# ---------------------------------------------------------------------------


def test_build_products_tolerates_card_with_empty_variables() -> None:
    """Some bathymetry / sentinel-only products have no listed
    variables. The aggregator must not crash."""
    from copernicus_mcp.backends.cmems._build_products import build_products

    [product] = build_products(
        [
            _card("Z", "z1", variables=[], variables_normalized=[]),
        ]
    )
    assert product["variables"] == []
    assert product["variables_normalized"] == []
    assert product["summary"].strip()


def test_build_products_dataset_count_matches_dataset_ids_length() -> None:
    from copernicus_mcp.backends.cmems._build_products import build_products

    cards = [_card("Z", f"z{i}") for i in range(7)]
    [product] = build_products(cards)
    assert product["dataset_count"] == len(product["dataset_ids"]) == 7


# ---------------------------------------------------------------------------
# Acceptance against the bundled cards
# ---------------------------------------------------------------------------


def test_build_products_against_bundled_cards_yields_expected_count() -> None:
    """End-to-end smoke: 1251 bundled cards → ~306 products."""
    import json
    from pathlib import Path

    from copernicus_mcp.backends.cmems._build_products import build_products

    data_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "copernicus_mcp"
        / "backends"
        / "cmems"
        / "_data"
    )
    cards = json.loads((data_dir / "dataset_cards.json").read_text())
    products = build_products(cards)
    # CMEMS catalogue snapshot 2026-05-14 = 306 products.
    assert 300 <= len(products) <= 315, f"unexpected product count: {len(products)}"
    # Every product has a non-empty summary.
    assert all(p["summary"].strip() for p in products)
    # No product has zero datasets.
    assert all(p["dataset_count"] >= 1 for p in products)


@pytest.mark.parametrize("axis", ["domains", "regions", "data_types"])
def test_build_products_categorical_axes_are_non_empty_lists(axis) -> None:
    """Every product must surface every categorical axis as a
    non-empty list (no implicit ``[]`` masquerading as "all" / "any")."""
    import json
    from pathlib import Path

    from copernicus_mcp.backends.cmems._build_products import build_products

    data_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "copernicus_mcp"
        / "backends"
        / "cmems"
        / "_data"
    )
    cards = json.loads((data_dir / "dataset_cards.json").read_text())
    products = build_products(cards)
    bad = [p["product_id"] for p in products if not p[axis]]
    assert not bad, f"products with empty {axis}: {bad[:5]}"
