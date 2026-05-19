"""T-CMEMS-CAT-002: runtime read module tests.

``backends/cmems/catalogue.py`` is the runtime counterpart to the
build-side ``_catalogue_build.py``. It loads the bundled snapshot
(written by ``scripts/refresh_marine_catalogue.py``) once at module
import time and serves ``search()`` queries offline — no credentials,
no network.

Schema lock + slim-record fields:
``spikes/T-CMEMS-CAT-000-describe-shape/FINDINGS.md`` Stage 3.

Mirrors ``backends/cds/catalogue.py`` (T-CDS-003) wherever the
patterns transfer; differences are CMEMS-specific (single-store,
`product_id` filter).
"""

from __future__ import annotations


def _reset_module_cache() -> None:
    """Clear the module-level cache between tests so each test sees a
    fresh load. The cache is intentional in production (loaded once
    per process), but tests should not bleed state across cases."""
    from copernicus_mcp.backends.cmems import catalogue

    catalogue._catalogue_cache = None  # type: ignore[attr-defined]
    catalogue._fetched_at_cache = None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# load_catalogue + fetched_at — bundled snapshot smoke tests
# ---------------------------------------------------------------------------


def test_load_catalogue_returns_list_from_bundled_snapshot() -> None:
    """The bundled ``_data/marine.json`` ships with the wheel and is
    read at first call. We assert >0 records to catch a packaging
    misconfiguration (snapshot stripped from the wheel)."""
    _reset_module_cache()
    from copernicus_mcp.backends.cmems.catalogue import load_catalogue

    catalogue = load_catalogue()
    assert isinstance(catalogue, list)
    assert len(catalogue) > 100, (
        "snapshot suspiciously small — packaging misconfiguration?"
    )


def test_load_catalogue_is_cached_across_calls() -> None:
    """Production code reads the snapshot once per process. Verify the
    second call returns the same in-memory list (identity), not a
    re-read from disk."""
    _reset_module_cache()
    from copernicus_mcp.backends.cmems.catalogue import load_catalogue

    first = load_catalogue()
    second = load_catalogue()
    assert first is second


def test_fetched_at_returns_iso_timestamp() -> None:
    """Snapshot timestamp matches the format the refresh script
    writes: ``YYYY-MM-DDTHH:MM:SSZ``."""
    import re

    _reset_module_cache()
    from copernicus_mcp.backends.cmems.catalogue import fetched_at

    ts = fetched_at()
    assert isinstance(ts, str)
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", ts
    ), f"unexpected timestamp shape: {ts!r}"


# ---------------------------------------------------------------------------
# search — keyword + product_id filtering
# ---------------------------------------------------------------------------


def test_search_with_no_filters_returns_full_catalogue() -> None:
    """An empty query returns every record (subject to limit)."""
    _reset_module_cache()
    from copernicus_mcp.backends.cmems.catalogue import (
        load_catalogue,
        search,
    )

    catalogue = load_catalogue()
    results = search()
    assert results == catalogue


def test_search_keyword_matches_substring_case_insensitive() -> None:
    """Keyword filter is case-insensitive substring match. Verify by
    searching for a known marker that won't accidentally match
    everything."""
    _reset_module_cache()
    from copernicus_mcp.backends.cmems.catalogue import search

    # "temperature" appears in many CMEMS dataset titles / descriptions
    # but not all 1251.
    results = search(keyword="TEMPERATURE")
    assert 0 < len(results) < 1251
    # Every returned record should plausibly mention "temperature"
    # somewhere in the searchable fields.
    for r in results[:5]:
        haystack = " ".join(
            str(r.get(f) or "")
            for f in (
                "dataset_id",
                "dataset_name",
                "title",
                "product_title",
                "description",
            )
        ).lower()
        # Or it could be in variables list.
        vars_blob = " ".join(str(v) for v in (r.get("variables") or [])).lower()
        assert "temperature" in haystack or "temperature" in vars_blob


def test_search_returns_empty_list_when_keyword_matches_nothing() -> None:
    _reset_module_cache()
    from copernicus_mcp.backends.cmems.catalogue import search

    results = search(keyword="xyzzy-no-such-marker-anywhere")
    assert results == []


def test_search_product_id_filter_exact_match() -> None:
    """``product_id`` is an exact-match filter (codex spec-review
    MEDIUM-2 on T-CMEMS-CATALOGUE sub-plan). Live search forwarded
    this filter via ``copernicusmarine.describe(product_id=...)``;
    the snapshot path must preserve the same semantics."""
    _reset_module_cache()
    from copernicus_mcp.backends.cmems.catalogue import search

    # ARCTIC_ANALYSISFORECAST_BGC_002_004 has multiple datasets (per
    # T-CMEMS-CAT-000 smoke).
    results = search(product_id="ARCTIC_ANALYSISFORECAST_BGC_002_004")
    assert len(results) >= 1
    for r in results:
        assert r["product_id"] == "ARCTIC_ANALYSISFORECAST_BGC_002_004"


def test_search_product_id_no_partial_match() -> None:
    """``product_id`` is exact — a prefix like ``ARCTIC`` does NOT
    match any dataset of a product with that prefix."""
    _reset_module_cache()
    from copernicus_mcp.backends.cmems.catalogue import search

    results = search(product_id="ARCTIC")  # no full product id matches.
    assert results == []


def test_search_combines_product_id_and_keyword() -> None:
    """Both filters AND together — a record must match both."""
    _reset_module_cache()
    from copernicus_mcp.backends.cmems.catalogue import search

    # All datasets under the GLOBAL physics product.
    all_global = search(product_id="GLOBAL_ANALYSISFORECAST_PHY_001_024")
    # Sub-filter by a keyword likely to match some but not all.
    subset = search(
        product_id="GLOBAL_ANALYSISFORECAST_PHY_001_024", keyword="temperature"
    )
    assert 0 < len(subset) <= len(all_global)
    for r in subset:
        assert r["product_id"] == "GLOBAL_ANALYSISFORECAST_PHY_001_024"


def test_search_respects_limit() -> None:
    """``limit`` slices the result list. Returns fewer when fewer
    match; never returns more than ``limit``."""
    _reset_module_cache()
    from copernicus_mcp.backends.cmems.catalogue import search

    five = search(limit=5)
    assert len(five) == 5
    full = search()
    assert five == full[:5]


def test_search_limit_treated_as_no_limit_when_none_or_zero() -> None:
    """``limit=None`` or ``limit<=0`` returns the unbounded match list.
    Defence-in-depth — the MCP schema rejects ``limit < 1``, but
    direct callers (backend.search() invariants, tests) bypass that."""
    _reset_module_cache()
    from copernicus_mcp.backends.cmems.catalogue import (
        load_catalogue,
        search,
    )

    full = load_catalogue()
    assert search(limit=None) == full
    assert search(limit=0) == full


def test_search_returns_independent_records_caller_can_mutate() -> None:
    """``search`` must not return references into the module cache,
    otherwise a caller mutating ``results[0]["title"]`` would silently
    corrupt the in-process catalogue for every other call.

    Mirrors the CDS pattern of returning deep copies / new dicts."""
    _reset_module_cache()
    from copernicus_mcp.backends.cmems.catalogue import (
        load_catalogue,
        search,
    )

    pre_load = load_catalogue()
    pre_title = pre_load[0]["title"]

    results = search(limit=1)
    results[0]["title"] = "MUTATED"

    # Re-loading must produce the original title — caller mutation
    # did not bleed into the module cache.
    re_load = load_catalogue()
    assert re_load[0]["title"] == pre_title


# ---------------------------------------------------------------------------
# Round-1 findings
# ---------------------------------------------------------------------------


def test_search_product_id_is_case_sensitive() -> None:
    """cr round-1 M1: ``product_id`` is an exact case-sensitive
    identifier match, not free text. Pin the case-sensitivity so a
    future PR can't quietly case-fold and silently widen results.

    Real-world driver: ~10% of CMEMS product ids are mixed-case
    (e.g. ``ANTARCTIC_OMI_SI_extent``) — a user passing the
    all-caps version gets zero results."""
    _reset_module_cache()
    # Discover a mixed-case product_id from the live snapshot so the
    # test doesn't break when CMEMS renames products.
    from copernicus_mcp.backends.cmems.catalogue import load_catalogue, search

    mixed_case = None
    for r in load_catalogue():
        pid = r.get("product_id")
        if pid and pid != pid.upper() and pid != pid.lower():
            mixed_case = pid
            break
    assert mixed_case is not None, "expected at least one mixed-case product_id"

    # Exact case matches.
    assert len(search(product_id=mixed_case)) >= 1
    # All-caps variant does NOT match.
    assert search(product_id=mixed_case.upper()) == []


def test_record_matches_does_not_bleed_across_fields() -> None:
    """cr round-1 LOW-2: ``_record_matches`` used to join fields with
    ``"\\n"``, so a needle containing a newline could span two
    fields. Pin per-field independence — a substring must be fully
    contained in ONE field to match."""
    _reset_module_cache()
    from copernicus_mcp.backends.cmems.catalogue import _record_matches

    record = {
        "dataset_id": "abc-id",
        "dataset_name": "Name xyz",
        "title": None,
        "product_id": None,
        "product_title": None,
        "description": None,
        "variables": [],
    }
    # "id\nName" used to match because fields were joined with \n.
    # Per-field independence: no single field contains that whole
    # substring.
    assert _record_matches(record, "id\nName") is False
    # Sanity: in-field substring still matches.
    assert _record_matches(record, "abc") is True
    assert _record_matches(record, "Name") is True


def test_load_catalogue_fails_fast_on_non_dict_record(
    tmp_path, monkeypatch
) -> None:
    """codex round-1 MEDIUM: silently dropping non-dict entries with
    ``[r for r in loaded if isinstance(r, dict)]`` lets a partially-
    corrupt snapshot serve truncated results with no signal. Fail
    fast instead, with the offending index in the error message."""
    import json

    import pytest

    from copernicus_mcp.backends.cmems import catalogue as cat

    # Build a partially-corrupt snapshot in a tmp data dir.
    fake_data = tmp_path / "_data"
    fake_data.mkdir()
    (fake_data / "marine.json").write_text(
        json.dumps([{"dataset_id": "ok"}, "not-a-dict", {"dataset_id": "ok2"}])
    )
    (fake_data / "fetched_at.json").write_text('{"marine": "2026-05-13T18:00:00Z"}')

    monkeypatch.setattr(cat, "_DATA_DIR", fake_data)
    _reset_module_cache()

    with pytest.raises(ValueError, match=r"record 1"):
        cat.load_catalogue()


def test_load_catalogue_rejects_non_list_top_level(
    tmp_path, monkeypatch
) -> None:
    """cr round-1 LOW-4 / codex round-1 MEDIUM: a malformed top-level
    (dict, str, etc.) raises ValueError with a clear message."""
    import json

    import pytest

    from copernicus_mcp.backends.cmems import catalogue as cat

    fake_data = tmp_path / "_data"
    fake_data.mkdir()
    (fake_data / "marine.json").write_text(json.dumps({"products": []}))

    monkeypatch.setattr(cat, "_DATA_DIR", fake_data)
    _reset_module_cache()

    with pytest.raises(ValueError, match="not a list"):
        cat.load_catalogue()


def test_fetched_at_rejects_null_value(tmp_path, monkeypatch) -> None:
    """codex round-1 LOW: ``{"marine": null}`` previously stringified
    to ``"None"`` which would leak into the search envelope's
    ``catalogue_fetched_at`` field. Validate non-empty string."""
    import pytest

    from copernicus_mcp.backends.cmems import catalogue as cat

    fake_data = tmp_path / "_data"
    fake_data.mkdir()
    (fake_data / "marine.json").write_text("[]")
    (fake_data / "fetched_at.json").write_text('{"marine": null}')

    monkeypatch.setattr(cat, "_DATA_DIR", fake_data)
    _reset_module_cache()

    with pytest.raises(ValueError, match="marine"):
        cat.fetched_at()


def test_fetched_at_rejects_empty_string(tmp_path, monkeypatch) -> None:
    """codex round-1 LOW: ``{"marine": ""}`` also fails the
    non-empty-string check."""
    import pytest

    from copernicus_mcp.backends.cmems import catalogue as cat

    fake_data = tmp_path / "_data"
    fake_data.mkdir()
    (fake_data / "marine.json").write_text("[]")
    (fake_data / "fetched_at.json").write_text('{"marine": ""}')

    monkeypatch.setattr(cat, "_DATA_DIR", fake_data)
    _reset_module_cache()

    with pytest.raises(ValueError, match="marine"):
        cat.fetched_at()


# ---------------------------------------------------------------------------
# count_matches helper (cr round-1 M2 / codex round-1 LOW-1)
# ---------------------------------------------------------------------------


def test_count_matches_returns_unsliced_match_count() -> None:
    """``count_matches`` returns the number of records matching the
    filter, without slicing or copying. CAT-003 uses it to populate
    ``total_count`` in the search envelope without paying for a
    deep-copy of the full match list."""
    _reset_module_cache()
    from copernicus_mcp.backends.cmems.catalogue import (
        count_matches,
        search,
    )

    n = count_matches(keyword="temperature")
    matches = search(keyword="temperature")
    assert n == len(matches)
    assert n > 1  # sanity: snapshot has many temperature datasets


def test_count_matches_unfiltered_equals_full_catalogue_length() -> None:
    _reset_module_cache()
    from copernicus_mcp.backends.cmems.catalogue import (
        count_matches,
        load_catalogue,
    )

    assert count_matches() == len(load_catalogue())


def test_count_matches_respects_product_id_filter() -> None:
    _reset_module_cache()
    # Pick a product_id dynamically.
    from copernicus_mcp.backends.cmems.catalogue import (
        count_matches,
        load_catalogue,
        search,
    )

    pid = load_catalogue()[0]["product_id"]
    assert count_matches(product_id=pid) == len(search(product_id=pid))


def test_count_matches_consistent_with_search_for_combined_filters() -> None:
    """codex round-2 LOW: pin the invariant for the combined
    keyword + product_id case so a future refactor that diverges the
    filter logic between ``count_matches`` and ``search`` fails
    loudly. Today both functions share ``_iter_matches`` so the
    invariant is structural, but a refactor could split it."""
    _reset_module_cache()
    from copernicus_mcp.backends.cmems.catalogue import (
        count_matches,
        search,
    )

    pid = "GLOBAL_ANALYSISFORECAST_PHY_001_024"
    assert count_matches(
        keyword="temperature", product_id=pid
    ) == len(search(keyword="temperature", product_id=pid))
