"""T-CMEMS-HIER-004 lint helper tests.

``validate_groups`` checks the v1 groups manifest against the
product manifest. It returns a list of human-readable problem
strings — empty when the manifest is consistent — so a refresh-pipeline
caller can fail-fast before bundling a broken file. The validator is
a *lint*, not a schema: it pins the cross-reference invariants the
runtime router needs, not every JSON-shape constraint.

Invariants:
1. Every ``product_id`` cited in a group exists in ``products``.
2. No group is empty (``product_ids == []``).
3. Every group declares non-empty ``include_when_query_mentions``.
4. Every group declares non-empty ``exclude_when_query_mentions``
   (can be a deliberately empty list, but the field must exist).
5. ``group_id``, ``group_title``, ``summary`` are non-empty strings.
6. ``group_id`` values are unique across the manifest.
7. Orphan products (those not cited in any group) are reported but
   not treated as a hard failure — multi-membership is allowed and a
   product may legitimately need a future group.
"""

from __future__ import annotations

from typing import Any

import pytest


def _group(
    group_id: str = "physics-global-forecast",
    *,
    group_title: str = "Global Ocean Physics Forecast",
    summary: str = "Daily forecast of ocean temperature, salinity, and currents at global scale.",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    product_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "group_title": group_title,
        "summary": summary,
        "include_when_query_mentions": include
        if include is not None
        else ["forecast", "operational", "temperature", "global"],
        "exclude_when_query_mentions": exclude
        if exclude is not None
        else ["climate", "historical"],
        "product_ids": product_ids
        if product_ids is not None
        else ["GLOBAL_ANALYSISFORECAST_PHY_001_024"],
    }


def _product(product_id: str = "GLOBAL_ANALYSISFORECAST_PHY_001_024") -> dict[str, Any]:
    return {"product_id": product_id, "product_title": "Test"}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_validate_groups_empty_problems_for_well_formed_manifest() -> None:
    """A manifest that obeys every invariant returns ``[]``."""
    from copernicus_mcp.backends.cmems._build_groups import validate_groups

    groups = [_group()]
    products = [_product()]
    assert validate_groups(groups, products) == []


# ---------------------------------------------------------------------------
# Invariant 1: every cited product_id exists
# ---------------------------------------------------------------------------


def test_validate_groups_reports_unknown_product_ids() -> None:
    from copernicus_mcp.backends.cmems._build_groups import validate_groups

    groups = [_group(product_ids=["KNOWN", "GHOST_PRODUCT"])]
    products = [_product("KNOWN")]
    problems = validate_groups(groups, products)
    assert any("GHOST_PRODUCT" in p for p in problems)


# ---------------------------------------------------------------------------
# Invariant 2: no empty groups
# ---------------------------------------------------------------------------


def test_validate_groups_reports_empty_group() -> None:
    from copernicus_mcp.backends.cmems._build_groups import validate_groups

    groups = [_group(group_id="empty-group", product_ids=[])]
    problems = validate_groups(groups, [_product()])
    assert any("empty-group" in p and "empty" in p.lower() for p in problems)


# ---------------------------------------------------------------------------
# Invariant 3 + 4: include/exclude fields
# ---------------------------------------------------------------------------


def test_validate_groups_reports_missing_include_when_mentions() -> None:
    from copernicus_mcp.backends.cmems._build_groups import validate_groups

    groups = [_group(include=[])]
    problems = validate_groups(groups, [_product()])
    assert any("include_when_query_mentions" in p for p in problems)


def test_validate_groups_accepts_empty_exclude_list() -> None:
    """``exclude_when_query_mentions`` may legitimately be empty for a
    group that doesn't need a negative-match filter (e.g. a generic
    intent group like "global ocean state"). The field must still
    be present — distinct from being missing."""
    from copernicus_mcp.backends.cmems._build_groups import validate_groups

    groups = [_group(exclude=[])]
    problems = validate_groups(groups, [_product()])
    # No problem reported for empty-but-present exclude.
    assert not any("exclude_when_query_mentions" in p for p in problems)


def test_validate_groups_reports_missing_exclude_field_entirely() -> None:
    from copernicus_mcp.backends.cmems._build_groups import validate_groups

    bad = _group()
    del bad["exclude_when_query_mentions"]
    problems = validate_groups([bad], [_product()])
    assert any("exclude_when_query_mentions" in p for p in problems)


# ---------------------------------------------------------------------------
# Invariant 5: required string fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["group_id", "group_title", "summary"])
def test_validate_groups_reports_empty_required_string_field(field) -> None:
    from copernicus_mcp.backends.cmems._build_groups import validate_groups

    bad = _group()
    bad[field] = ""
    problems = validate_groups([bad], [_product()])
    assert any(field in p for p in problems)


@pytest.mark.parametrize("field", ["group_id", "group_title", "summary"])
def test_validate_groups_reports_missing_required_string_field(field) -> None:
    from copernicus_mcp.backends.cmems._build_groups import validate_groups

    bad = _group()
    del bad[field]
    problems = validate_groups([bad], [_product()])
    assert any(field in p for p in problems)


# ---------------------------------------------------------------------------
# Invariant 6: group_id uniqueness
# ---------------------------------------------------------------------------


def test_validate_groups_reports_duplicate_group_ids() -> None:
    from copernicus_mcp.backends.cmems._build_groups import validate_groups

    groups = [_group(group_id="dup"), _group(group_id="dup")]
    problems = validate_groups(groups, [_product()])
    assert any("dup" in p and "duplicate" in p.lower() for p in problems)


# ---------------------------------------------------------------------------
# Invariant 7: orphan products are reported, not failed
# ---------------------------------------------------------------------------


def test_validate_groups_reports_orphan_products_as_informational() -> None:
    """A product not cited in any group is reported with the word
    ``orphan`` so the curator can decide whether to add a group or
    accept the gap. The check still returns problems — just so the
    curator notices."""
    from copernicus_mcp.backends.cmems._build_groups import validate_groups

    products = [_product("KNOWN"), _product("ORPHAN_ONE"), _product("ORPHAN_TWO")]
    groups = [_group(product_ids=["KNOWN"])]
    problems = validate_groups(groups, products)
    assert any("ORPHAN_ONE" in p for p in problems)
    assert any("ORPHAN_TWO" in p for p in problems)


def test_validate_groups_allows_multi_membership() -> None:
    """A product cited by two groups is not an error — multi-membership
    is the explicit design choice."""
    from copernicus_mcp.backends.cmems._build_groups import validate_groups

    products = [_product("SHARED")]
    groups = [
        _group(group_id="g1", product_ids=["SHARED"]),
        _group(group_id="g2", product_ids=["SHARED"]),
    ]
    problems = validate_groups(groups, products)
    # No problem about multi-membership; no orphans either.
    assert not any("SHARED" in p for p in problems)


# ---------------------------------------------------------------------------
# Bundled manifest sanity check
# ---------------------------------------------------------------------------


def test_validate_groups_bundled_manifest_is_consistent() -> None:
    """End-to-end smoke against the bundled ``groups.json`` +
    ``products.json``."""
    import json
    from pathlib import Path

    from copernicus_mcp.backends.cmems._build_groups import validate_groups

    data_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "copernicus_mcp"
        / "backends"
        / "cmems"
        / "_data"
    )
    groups = json.loads((data_dir / "groups.json").read_text())
    products = json.loads((data_dir / "products.json").read_text())
    problems = validate_groups(groups, products)
    # Orphans are allowed in v1; the curator inspects them in the PR
    # before deciding on additions. Hard failures (missing fields,
    # bad cross-refs) are NOT allowed.
    hard_failures = [p for p in problems if "orphan" not in p.lower()]
    assert not hard_failures, f"v1 manifest hard failures: {hard_failures[:5]}"


def test_validate_groups_bundled_manifest_has_expected_group_count() -> None:
    """Phase-1+2 acceptance: 35-50 groups in v1 + r2 additions."""
    import json
    from pathlib import Path

    data_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "copernicus_mcp"
        / "backends"
        / "cmems"
        / "_data"
    )
    groups = json.loads((data_dir / "groups.json").read_text())
    assert 35 <= len(groups) <= 50, f"unexpected group count: {len(groups)}"


# ---------------------------------------------------------------------------
# Predicate smoke tests (round-2 Q5c)
# ---------------------------------------------------------------------------
#
# Pin the cross-reference invariants the predicates promise: a
# comprehensive bundle should contain ALL products of the matching
# region; the new round-2 groups should contain at least the
# minimum expected entries (so a future product-catalogue change
# that breaks the predicate surfaces here).


def _load_bundle() -> tuple[list[dict], list[dict]]:
    import json
    from pathlib import Path

    data_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "copernicus_mcp"
        / "backends"
        / "cmems"
        / "_data"
    )
    groups = json.loads((data_dir / "groups.json").read_text())
    products = json.loads((data_dir / "products.json").read_text())
    return groups, products


def _find_group(groups: list[dict], group_id: str) -> dict | None:
    return next((g for g in groups if g["group_id"] == group_id), None)


@pytest.mark.parametrize(
    "group_id,region",
    [
        ("arctic-comprehensive", "arctic"),
        ("mediterranean-comprehensive", "mediterranean"),
        ("baltic-comprehensive", "baltic"),
        ("black-sea-comprehensive", "black_sea"),
        ("ibi-comprehensive", "iberia_biscay_ireland"),
        ("nwshelf-comprehensive", "north_west_shelf"),
    ],
)
def test_comprehensive_bundle_contains_every_product_in_its_region(group_id, region) -> None:
    """Each ``<region>-comprehensive`` group must contain every
    product whose ``regions`` list includes that region. A missing
    member here means the predicate fell out of sync with the
    region-token map."""
    groups, products = _load_bundle()
    group = _find_group(groups, group_id)
    assert group is not None, f"group {group_id} not in bundle"

    expected = {p["product_id"] for p in products if region in p["regions"]}
    actual = set(group["product_ids"])
    assert expected == actual, (
        f"{group_id}: expected {len(expected)} products in region "
        f"'{region}', got {len(actual)}. Missing: "
        f"{sorted(expected - actual)[:5]}, extra: "
        f"{sorted(actual - expected)[:5]}"
    )


def test_wind_regional_contains_only_non_global_wind() -> None:
    """``wind-regional`` (round-2) is the complement of
    ``wind-global`` for the wind domain."""
    groups, products = _load_bundle()
    group = _find_group(groups, "wind-regional")
    assert group is not None
    expected = {
        p["product_id"] for p in products if "wind" in p["domains"] and "global" not in p["regions"]
    }
    assert expected == set(group["product_ids"])
    # Sanity: there should be a non-trivial number of regional wind
    # products (10 in the current catalogue).
    assert len(expected) >= 5


def test_mdt_multi_membership_in_sea_level_and_bathymetry() -> None:
    """Round-2 Q3 decision: MDT products live in both a sea-level
    group (``sea-level-global`` for global MDT, ``sea-level-european``
    for regional MDT) and in ``bathymetry-and-static-fields`` as
    static reference fields. Every MDT product must appear in at
    least one sea-level group AND in bathymetry-and-static-fields."""
    groups, products = _load_bundle()
    mdt_products = {p["product_id"] for p in products if "_MDT_" in p["product_id"]}
    assert mdt_products, "no MDT products in bundle — predicate untestable"

    sea_level_global = _find_group(groups, "sea-level-global")
    sea_level_european = _find_group(groups, "sea-level-european")
    bathymetry = _find_group(groups, "bathymetry-and-static-fields")
    assert sea_level_global and sea_level_european and bathymetry

    sea_level_union = set(sea_level_global["product_ids"]) | set(sea_level_european["product_ids"])
    bathymetry_ids = set(bathymetry["product_ids"])
    for pid in mdt_products:
        assert pid in sea_level_union, (
            f"{pid} missing from both sea-level-global and sea-level-european"
        )
        assert pid in bathymetry_ids, f"{pid} missing from bathymetry-and-static-fields"


def test_ocean_acidification_group_contains_carbon_omis() -> None:
    """Round-2 new group: every ``OMI_HEALTH_*`` (carbon / pH /
    CO2-flux) product should be in ``ocean-acidification-monitoring``."""
    groups, products = _load_bundle()
    group = _find_group(groups, "ocean-acidification-monitoring")
    assert group is not None
    health_omis = {p["product_id"] for p in products if "OMI_HEALTH" in p["product_id"]}
    member_ids = set(group["product_ids"])
    missing = health_omis - member_ids
    assert not missing, f"OMI_HEALTH products missing: {missing}"


def test_ocean_circulation_indices_group_contains_amoc_and_circulation_omis() -> None:
    """Round-2 new group: AMOC + general circulation OMIs must
    appear together so a query about "AMOC strength" sees them all."""
    groups, products = _load_bundle()
    group = _find_group(groups, "ocean-circulation-indices")
    assert group is not None
    circ_omis = {
        p["product_id"]
        for p in products
        if "OMI_NATLANTIC" in p["product_id"] or "OMI_CIRCULATION" in p["product_id"]
    }
    member_ids = set(group["product_ids"])
    missing = circ_omis - member_ids
    assert not missing, f"AMOC / circulation OMIs missing: {missing}"
