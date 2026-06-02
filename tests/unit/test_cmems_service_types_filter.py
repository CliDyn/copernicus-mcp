"""T-TS-004: ``marine_search_datasets(service_types=[...])`` filter.

The public schema advertises short names (``timeseries``/``geoseries``/...);
the bundled catalogue records use the long SDK names
(``arco-time-series``/...). The filter maps short -> long and keeps records
whose ``service_types`` intersect the request. Previously rejected as
"not yet implemented".
"""

from __future__ import annotations


def test_canonical_service_types_alias() -> None:
    from copernicus_mcp.backends.cmems.catalogue import canonical_service_types

    assert canonical_service_types(["timeseries"]) == {"arco-time-series"}
    assert canonical_service_types(["geoseries", "platformseries"]) == {
        "arco-geo-series",
        "arco-platform-series",
    }
    # already-canonical / passthrough values are preserved
    assert canonical_service_types(["omi-arco", "static-arco"]) == {
        "omi-arco",
        "static-arco",
    }


def test_catalogue_search_filters_by_service_type() -> None:
    from copernicus_mcp.backends.cmems.catalogue import search

    rows = search(service_types={"arco-time-series"}, limit=25)
    assert rows
    assert all("arco-time-series" in (r.get("service_types") or []) for r in rows)


def test_catalogue_count_matches_service_types_is_strict_subset() -> None:
    from copernicus_mcp.backends.cmems.catalogue import count_matches

    ts = count_matches(service_types={"arco-time-series"})
    total = count_matches()
    # 977 of 1251 expose arco-time-series (survey 2026-06-01) — a strict subset.
    assert 0 < ts < total


def test_validate_search_accepts_service_types() -> None:
    """T-TS-004: service_types is no longer rejected at validation (flat path)."""
    from copernicus_mcp.backends.cmems.backend import _validate_search

    req = _validate_search({"keyword": "temperature", "service_types": ["timeseries"]})
    assert req.service_types == ["timeseries"]
