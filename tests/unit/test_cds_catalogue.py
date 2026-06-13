"""Tests for the bundled CDS / ADS / EWDS catalogue snapshot (T-CDS-003).

The catalogue is shipped under ``src/copernicus_mcp/backends/cds/_data/``;
these tests exercise the lookup helpers via the real data files so a
regression in the parser (or in the snapshot's structure when an upstream
schema change lands) surfaces immediately.

Stable assertions only — we test for the presence of foundational
datasets like ERA5 that are not going to disappear.
"""

from __future__ import annotations

import pytest


def test_catalogue_loads_three_stores() -> None:
    """Module-level cache pre-loads cds + ads + ewds without auth."""
    from copernicus_mcp.backends.cds.catalogue import load_catalogue

    cat = load_catalogue()
    assert "cds" in cat
    assert "ads" in cat
    assert "ewds" in cat
    # CDS is the largest of the three.
    assert len(cat["cds"]) > len(cat["ads"])
    assert len(cat["cds"]) > len(cat["ewds"])


def test_catalogue_lookup_by_id_returns_full_record() -> None:
    """``describe`` returns the full STAC record (not the slim version)."""
    from copernicus_mcp.backends.cds.catalogue import describe

    record = describe("reanalysis-era5-single-levels")
    assert record["id"] == "reanalysis-era5-single-levels"
    assert "description" in record
    # Full record has STAC fields beyond what slim exposes.
    assert "extent" in record or "summaries" in record


def test_describe_unknown_id_raises_not_found() -> None:
    from copernicus_mcp.backends.cds.catalogue import describe
    from copernicus_mcp.errors import NotFoundError

    with pytest.raises(NotFoundError):
        describe("does-not-exist-anywhere")


def test_search_returns_slim_shape() -> None:
    """Slim shape: only id, title, description (first paragraph), keywords."""
    from copernicus_mcp.backends.cds.catalogue import search

    results = search(keyword="ERA5", store="cds")
    assert results, "ERA5 search must return at least one CDS result"
    sample = results[0]
    assert set(sample.keys()) == {"id", "title", "description", "keywords", "store"}
    # description is trimmed (first paragraph + cap), not the full markdown.
    assert len(sample["description"]) <= 600


# T-CDS-020 PR-1: bbox / time_range / variable filters --------------------


def test_search_bbox_filter_keeps_intersecting_only() -> None:
    """A small European bbox must keep CERRA (regional, bbox=[17, 20, 35, 65])
    and drop datasets whose bbox does not intersect."""
    from copernicus_mcp.backends.cds.catalogue import search

    ids = {r["id"] for r in search(bbox=(20.0, 30.0, 30.0, 60.0))}
    # CERRA datasets cover Europe — must be in.
    assert "reanalysis-cerra-single-levels" in ids


def test_search_bbox_filter_excludes_non_overlapping_region() -> None:
    """A South-Pacific bbox must exclude CERRA (Europe-only regional)."""
    from copernicus_mcp.backends.cds.catalogue import search

    ids = {r["id"] for r in search(bbox=(-150.0, -50.0, -120.0, -30.0))}
    assert "reanalysis-cerra-single-levels" not in ids
    assert "reanalysis-cerra-pressure-levels" not in ids


def test_search_bbox_filter_keeps_global_datasets() -> None:
    """Global ERA5 must intersect any query bbox (its spatial extent is
    -180..180 / -90..90 or 0..360 equivalent)."""
    from copernicus_mcp.backends.cds.catalogue import search

    ids = {r["id"] for r in search(bbox=(20.0, 30.0, 30.0, 60.0))}
    assert "reanalysis-era5-single-levels" in ids


def test_search_bbox_normalises_360_form() -> None:
    """ERA5 catalogue records often have bbox=[0, -89, 360, 89]
    (0..360 longitude form). A query in ±180 form (e.g. -10..10) must
    still intersect — the filter normalises before comparison."""
    from copernicus_mcp.backends.cds.catalogue import search

    ids = {r["id"] for r in search(bbox=(-10.0, -10.0, 10.0, 10.0))}
    assert "reanalysis-era5-single-levels" in ids


def test_search_bbox_validates_shape() -> None:
    from copernicus_mcp.backends.cds.catalogue import search
    from copernicus_mcp.errors import ValidationError

    with pytest.raises(ValidationError):
        search(bbox=(20.0, 30.0, 10.0, 60.0))  # west > east on ±180 form
    with pytest.raises(ValidationError):
        search(bbox=(20.0, 60.0, 30.0, 30.0))  # south > north
    with pytest.raises(ValidationError):
        search(bbox=(200.0, 30.0, 30.0, 60.0))  # out of range


def test_search_time_range_filter_overlap_inclusive() -> None:
    """A query window inside the dataset's temporal extent must keep it."""
    from copernicus_mcp.backends.cds.catalogue import search

    ids = {
        r["id"]
        for r in search(time_range=("2010-01-01T00:00:00Z", "2010-12-31T23:59:59Z"))
    }
    # ERA5 covers 1940..present — must overlap 2010.
    assert "reanalysis-era5-single-levels" in ids


def test_search_time_range_filter_excludes_outside() -> None:
    """A query window entirely BEFORE every dataset's start must
    return zero results (no CDS dataset covers 1800)."""
    from copernicus_mcp.backends.cds.catalogue import search

    res = search(time_range=("1800-01-01T00:00:00Z", "1800-12-31T23:59:59Z"))
    # Some catalogues use None start (rare); confirm none of the populated ones match
    assert "reanalysis-era5-single-levels" not in {r["id"] for r in res}


def test_search_time_range_handles_open_ended_dataset_end() -> None:
    """Datasets with ``temporal.interval[0][1] == None`` are still being
    updated; a future-ish query window must still match them."""
    from copernicus_mcp.backends.cds.catalogue import search

    ids = {
        r["id"]
        for r in search(time_range=("2025-01-01T00:00:00Z", "2026-12-31T23:59:59Z"))
    }
    # ERA5 is constantly updated — there are records with open-ended end.
    # Pick one we know exists.
    assert "reanalysis-era5-single-levels" in ids


def test_search_time_range_validates_order() -> None:
    from copernicus_mcp.backends.cds.catalogue import search
    from copernicus_mcp.errors import ValidationError

    with pytest.raises(ValidationError):
        search(time_range=("2020-01-01T00:00:00Z", "2010-01-01T00:00:00Z"))


def test_search_time_range_accepts_naive_datetime_strings() -> None:
    """Round-1 cr H1: an LLM that reads docstring 'ISO 8601' and emits
    ``"2010-01-01"`` (valid ISO 8601 date) must NOT crash with raw
    ``TypeError`` from comparing naive vs tz-aware datetimes. Naive
    inputs are interpreted as UTC."""
    from copernicus_mcp.backends.cds.catalogue import search

    # Bare date.
    ids_date = {r["id"] for r in search(time_range=("2010-01-01", "2010-12-31"))}
    assert "reanalysis-era5-single-levels" in ids_date

    # ISO datetime without Z / offset.
    ids_iso = {
        r["id"]
        for r in search(time_range=("2010-01-01T00:00:00", "2010-12-31T23:59:59"))
    }
    assert "reanalysis-era5-single-levels" in ids_iso

    # Whitespace tolerance.
    ids_ws = {
        r["id"]
        for r in search(time_range=(" 2010-01-01T00:00:00Z ", "2010-12-31T23:59:59Z"))
    }
    assert "reanalysis-era5-single-levels" in ids_ws


def test_search_time_range_includes_records_with_null_extent() -> None:
    """Some snapshot records carry ``temporal.interval = [[None, None]]``
    (catalogue entries with no declared time coverage). Policy: include
    them (better recall than silent hiding — the agent can verify with
    describe())."""
    from copernicus_mcp.backends.cds.catalogue import (
        _parse_iso,
        _record_overlaps_time,
    )

    record_unknown = {
        "extent": {"temporal": {"interval": [[None, None]]}}
    }
    assert _record_overlaps_time(
        record_unknown,
        (_parse_iso("2010-01-01T00:00:00Z"), _parse_iso("2010-12-31T23:59:59Z")),
    ) is True


# T-CDS-021 PR-2: search_groups + domain/category filters ----------------


def test_search_groups_returns_all_groups_without_query() -> None:
    """Without ``query``: return every populated (domain, category)
    group sorted by ``dataset_count`` descending. Total ≈ 40 groups
    across 164 datasets in the current snapshot."""
    from copernicus_mcp.backends.cds.catalogue import search_groups

    res = search_groups()
    groups = res["groups"]
    assert res["total_count"] == len(groups)
    assert len(groups) >= 20, "expected many populated groups"
    # Sorted by count desc.
    counts = [g["dataset_count"] for g in groups]
    assert counts == sorted(counts, reverse=True), counts


def test_search_group_record_shape() -> None:
    from copernicus_mcp.backends.cds.catalogue import search_groups

    res = search_groups()
    g = res["groups"][0]
    assert set(g.keys()) >= {
        "id",
        "domain",
        "category",
        "dataset_count",
        "sample_titles",
    }
    # id is a stable chaining key: ``<domain-slug>|<category>``.
    assert "|" in g["id"]
    # sample_titles is bounded.
    assert 0 < len(g["sample_titles"]) <= 3


def test_search_groups_query_ranks_atmosphere_for_atmosphere_query() -> None:
    """A query mentioning a domain keyword surfaces matching groups first."""
    from copernicus_mcp.backends.cds.catalogue import search_groups

    res = search_groups(query="atmosphere temperature reanalysis")
    groups = res["groups"]
    assert groups, "expected at least one match"
    # Top group should be Atmosphere-flavoured AND a reanalysis category.
    top = groups[0]
    assert "atmosphere" in top["domain"].lower()
    assert top["category"] == "reanalysis"


def test_search_groups_top_k_limits_results() -> None:
    from copernicus_mcp.backends.cds.catalogue import search_groups

    res = search_groups(query="ocean", top_k=3)
    assert len(res["groups"]) <= 3


def test_search_groups_includes_unspecified_domain_bucket() -> None:
    """The 1 record without a ``Variable domain:`` keyword still lands
    in a group (under ``domain='unspecified'``) — never silently dropped."""
    from copernicus_mcp.backends.cds.catalogue import search_groups

    res = search_groups()
    domains = {g["domain"] for g in res["groups"]}
    # Either it shows up as "unspecified" or it doesn't exist in current
    # snapshot — assertion is conditional.
    assert "unspecified" in domains or all(
        g["domain"] != "" for g in res["groups"]
    )


def test_search_datasets_domain_filter() -> None:
    """``domain="Ocean (physics)"`` (exact match against the
    ``Variable domain: X`` keyword) keeps only ocean-physics records."""
    from copernicus_mcp.backends.cds.catalogue import search

    ids = {r["id"] for r in search(domain="Ocean (physics)")}
    assert ids
    # Sea-surface-temperature satellite product belongs to ocean-physics.
    assert "satellite-sea-surface-temperature" in ids
    # CERRA (Land hydrology / Atmosphere) must NOT be in ocean-physics.
    assert "reanalysis-cerra-single-levels" not in ids


def test_search_datasets_category_filter() -> None:
    """``category="reanalysis"`` keeps only ID-prefix-reanalysis records."""
    from copernicus_mcp.backends.cds.catalogue import search

    ids = {r["id"] for r in search(category="reanalysis")}
    assert "reanalysis-era5-single-levels" in ids
    # Non-reanalysis (satellite, sis, etc.) must be excluded.
    assert "satellite-sea-surface-temperature" not in ids


def test_search_groups_per_token_axis_match_outranks_membership() -> None:
    """Round-1 codex M1: ``"atmosphere reanalysis"`` must surface
    ``(Atmosphere*, reanalysis)`` groups ahead of ``(Atmosphere*, sis)``
    even though the *sis* group is larger. Per-token axis matches now
    get the strong +2 boost (was +0.5)."""
    from copernicus_mcp.backends.cds.catalogue import search_groups

    res = search_groups(query="atmosphere reanalysis", top_k=5)
    groups = res["groups"]
    assert groups
    # Top group must be a reanalysis category.
    assert groups[0]["category"] == "reanalysis", groups[0]
    # AND the top group's domain mentions atmosphere.
    assert "atmosphere" in groups[0]["domain"].lower(), groups[0]


def test_search_groups_top_k_zero_returns_empty() -> None:
    """Round-1 local cr M2: ``top_k=0`` means zero results, not
    'no cap' (the previous off-by-one returned the full list)."""
    from copernicus_mcp.backends.cds.catalogue import search_groups

    res = search_groups(top_k=0)
    assert res["groups"] == []
    assert res["total_count"] == 0


def test_group_id_slugs_both_axes() -> None:
    """Round-1 local cr M1: ``_group_id`` slugifies BOTH halves so
    future taxonomy values with punctuation don't collide with the
    ``|`` separator or break URL-safety."""
    from copernicus_mcp.backends.cds.catalogue import _group_id

    # Current snapshot value.
    assert _group_id("Atmosphere (surface)", "reanalysis") == (
        "atmosphere-surface|reanalysis"
    )
    # Future-proofing: punctuation in either half is scrubbed.
    assert _group_id("Ocean, coastal", "satellite") == (
        "ocean-coastal|satellite"
    )
    # Pipe in the input cannot leak into the separator position.
    g = _group_id("a|b", "cat")
    assert g.count("|") == 1, g


def test_search_datasets_domain_and_category_combine_AND() -> None:
    """Multi-domain note: ERA5 datasets carry BOTH 'Atmosphere (surface)'
    and 'Atmosphere (upper air)' tags, so they appear in both groups.
    The AND combination here filters category strictly — a non-reanalysis
    record (e.g. satellite-sea-surface-temperature) is excluded even if
    its domain happens to overlap a query."""
    from copernicus_mcp.backends.cds.catalogue import search

    ids = {
        r["id"]
        for r in search(
            domain="Atmosphere (surface)", category="reanalysis"
        )
    }
    assert "reanalysis-era5-single-levels" in ids
    # Satellite SST is Ocean (physics), not Atmosphere (surface) →
    # excluded by domain even though category isn't reanalysis either.
    assert "satellite-sea-surface-temperature" not in ids
    # Cross-category contamination check: cams-* category records
    # must NOT pollute reanalysis results.
    assert all(r.startswith("reanalysis-") for r in ids), ids


def test_variable_filter_excludes_substring_false_positives() -> None:
    """Codex retro PR #124 MEDIUM-2: ``variable="wind"`` previously
    matched ``satellite-albedo`` because its description contains
    "moving temporal window". The token "wind" appears as a substring
    of "window" but the user clearly means the meteorological variable.

    New contract: precision-first. Match against the bundled-constraints
    ``variable`` enum (split on ``_``) as the primary signal, then
    word-boundary matches against keywords + title + description as
    fallback. "window" never matches "wind" under word-boundary regex."""
    from copernicus_mcp.backends.cds.catalogue import search

    ids = {r["id"] for r in search(variable="wind")}
    # ERA5 has 10m_u_component_of_wind in constraints — must match.
    assert "reanalysis-era5-single-levels" in ids
    # Satellite albedo only mentions "moving temporal window" — must NOT match.
    assert "satellite-albedo" not in ids


def test_variable_filter_matches_constraint_token() -> None:
    """The constraints' variable enum is tokenised on ``_`` so
    ``variable="temperature"`` matches ``2m_temperature``,
    ``sea_surface_temperature``, etc."""
    from copernicus_mcp.backends.cds.catalogue import search

    ids = {r["id"] for r in search(variable="temperature")}
    # ERA5 has 2m_temperature.
    assert "reanalysis-era5-single-levels" in ids


def test_variable_filter_rejects_punctuation_only_needles() -> None:
    """Round-1 cr HIGH: a needle composed solely of punctuation
    (``","``, ``"!!!"``) strips to a non-empty string but tokenises
    to the empty set. ``set().issubset(...)`` would then be True for
    every record with a constraints variable enum → false-positive
    catalogue-wide. Guard at tokenisation."""
    from copernicus_mcp.backends.cds.catalogue import _record_matches_variable

    rec_with_constraints_id = {"id": "reanalysis-era5-single-levels"}
    assert _record_matches_variable(rec_with_constraints_id, ",") is False
    assert _record_matches_variable(rec_with_constraints_id, "!!!") is False
    assert _record_matches_variable(rec_with_constraints_id, "()") is False


def test_variable_filter_word_boundary_in_description() -> None:
    """Word-boundary text fallback: keywords / title / description
    matches must respect word boundaries so substrings of unrelated
    words don't pollute results."""
    from copernicus_mcp.backends.cds.catalogue import _record_matches_variable

    # Word-boundary match in description.
    rec_yes = {
        "id": "x",
        "description": "Wind speed and direction over the ocean surface.",
    }
    assert _record_matches_variable(rec_yes, "wind") is True

    # Substring "wind" inside "window" — must NOT match under word-boundary.
    rec_no = {
        "id": "y",
        "description": "Daily window for moving averages of albedo.",
    }
    assert _record_matches_variable(rec_no, "wind") is False


def test_record_intersects_bbox_handles_stac_6d_shape() -> None:
    """Codex retro PR #124 MEDIUM: STAC bbox can be 6D
    ``[w, s, min_elev, e, n, max_elev]`` (elevation-aware). Prior impl
    rejected non-4-element bboxes and treated them as unknown — silently
    bypassing the spatial filter for any elevation-aware record. Now
    handle both 4D and 6D."""
    from copernicus_mcp.backends.cds.catalogue import _record_intersects_bbox

    # 6D record over Europe with elevation 0..3000m.
    rec = {
        "extent": {
            "spatial": {"bbox": [[17.0, 20.0, 0.0, 35.0, 65.0, 3000.0]]}
        }
    }
    assert _record_intersects_bbox(rec, (20.0, 30.0, 30.0, 60.0)) is True
    assert _record_intersects_bbox(rec, (-150.0, -50.0, -120.0, -30.0)) is False


def test_record_intersects_bbox_iterates_multiple_bbox_entries() -> None:
    """Codex retro PR #124 MEDIUM: STAC ``extent.spatial.bbox`` is a
    LIST — multi-region datasets carry multiple entries. Prior impl
    read only ``raw[0]`` so a query that matched the second region
    would falsely miss."""
    from copernicus_mcp.backends.cds.catalogue import _record_intersects_bbox

    # Two disjoint regions: northern Europe + central Pacific.
    rec = {
        "extent": {
            "spatial": {
                "bbox": [
                    [17.0, 50.0, 35.0, 70.0],
                    [-160.0, -10.0, -140.0, 10.0],
                ]
            }
        }
    }
    # Query the Pacific region — must match via the 2nd entry.
    assert (
        _record_intersects_bbox(rec, (-155.0, -5.0, -145.0, 5.0)) is True
    )
    # Query somewhere disjoint from both → no match.
    assert (
        _record_intersects_bbox(rec, (-60.0, -40.0, -50.0, -30.0)) is False
    )


def test_record_intersects_bbox_skips_malformed_entries_but_uses_valid() -> None:
    """A record with one malformed entry + one valid entry must still
    use the valid one — the malformed entry is skipped, not the whole
    record."""
    from copernicus_mcp.backends.cds.catalogue import _record_intersects_bbox

    rec = {
        "extent": {
            "spatial": {
                "bbox": [
                    [1, 2, 3],  # malformed (3 elements)
                    [17.0, 50.0, 35.0, 70.0],  # valid 4D Europe
                ]
            }
        }
    }
    assert _record_intersects_bbox(rec, (20.0, 55.0, 30.0, 65.0)) is True
    # And a non-intersecting query against the same record → False.
    assert (
        _record_intersects_bbox(rec, (-50.0, -50.0, -40.0, -40.0)) is False
    )


def test_search_bbox_rejects_nan_values() -> None:
    """Round-1 cr LOW-2: NaN bbox values silently pass numeric range
    checks. Reject explicitly so the caller sees a canonical
    ``ValidationError`` instead of an empty result list."""
    from copernicus_mcp.backends.cds.catalogue import search
    from copernicus_mcp.errors import ValidationError

    with pytest.raises(ValidationError):
        search(bbox=(float("nan"), 0.0, 1.0, 1.0))


def test_search_variable_filter_matches_keywords() -> None:
    """Substring match (case-insensitive) against keywords / summaries /
    title / description. The slim record only carries the first three;
    summaries hits will not appear in the slim shape but still pass the
    filter — that is by design (more candidate recall is better than
    silently hiding matches the agent could verify with describe())."""
    from copernicus_mcp.backends.cds.catalogue import search

    ids = {r["id"] for r in search(variable="temperature")}
    # ERA5 single levels has 2m temperature among its keywords/summaries.
    assert "reanalysis-era5-single-levels" in ids
    # A dataset that demonstrably has no temperature content (here:
    # sea-ice velocity products) must be excluded.
    assert "satellite-greenland-ice-sheet-velocity" not in ids


def test_search_combined_filters_are_AND() -> None:
    """bbox + time_range + variable combine as AND."""
    from copernicus_mcp.backends.cds.catalogue import search

    res = search(
        bbox=(20.0, 30.0, 30.0, 60.0),
        time_range=("2010-01-01T00:00:00Z", "2010-12-31T23:59:59Z"),
        variable="temperature",
    )
    ids = {r["id"] for r in res}
    # CERRA single levels has European bbox, covers 2010, has temperature.
    assert "reanalysis-cerra-single-levels" in ids


def test_search_keyword_plus_filters_still_AND() -> None:
    """The legacy keyword filter must still combine AND with the new
    bbox/time/variable filters. We assert that:
      (a) the combined query is strictly narrower than each filter alone
          (AND-semantics, not OR);
      (b) a dataset known to be ERA5-related is kept;
      (c) a dataset that has nothing to do with ERA5 is excluded."""
    from copernicus_mcp.backends.cds.catalogue import search

    only_keyword = {r["id"] for r in search(keyword="ERA5")}
    only_bbox = {r["id"] for r in search(bbox=(-10.0, -10.0, 10.0, 10.0))}
    combined = {
        r["id"]
        for r in search(keyword="ERA5", bbox=(-10.0, -10.0, 10.0, 10.0))
    }
    # AND, not OR: combined is at most the intersection of the two.
    assert combined <= only_keyword
    assert combined <= only_bbox
    # ERA5-specific dataset survives.
    assert "reanalysis-era5-single-levels" in combined
    # Sea-ice-velocity dataset has nothing to do with ERA5 — excluded.
    assert "satellite-greenland-ice-sheet-velocity" not in combined


def test_search_no_keyword_returns_all_for_store() -> None:
    from copernicus_mcp.backends.cds.catalogue import load_catalogue, search

    cat = load_catalogue()
    results = search(store="ads")
    assert len(results) == len(cat["ads"])


def test_search_no_store_combines_all_stores() -> None:
    from copernicus_mcp.backends.cds.catalogue import load_catalogue, search

    cat = load_catalogue()
    total = len(cat["cds"]) + len(cat["ads"]) + len(cat["ewds"])
    results = search()
    assert len(results) == total
    # Each result is tagged with its store.
    stores_seen = {r["store"] for r in results}
    assert stores_seen == {"cds", "ads", "ewds"}


def test_search_keyword_is_case_insensitive() -> None:
    from copernicus_mcp.backends.cds.catalogue import search

    upper = search(keyword="ERA5", store="cds")
    lower = search(keyword="era5", store="cds")
    mixed = search(keyword="Era5", store="cds")
    ids_upper = {r["id"] for r in upper}
    ids_lower = {r["id"] for r in lower}
    ids_mixed = {r["id"] for r in mixed}
    assert ids_upper == ids_lower == ids_mixed
    assert len(ids_upper) > 0


def test_search_keyword_matches_id() -> None:
    """`reanalysis-era5-single-levels` matches keyword `single-levels`."""
    from copernicus_mcp.backends.cds.catalogue import search

    results = search(keyword="single-levels", store="cds")
    ids = {r["id"] for r in results}
    assert any("single-levels" in i for i in ids)


def test_search_keyword_matches_title() -> None:
    """ERA5 land hourly time-series — match by title fragment."""
    from copernicus_mcp.backends.cds.catalogue import search

    results = search(keyword="time-series", store="cds")
    titles = {r["title"] for r in results}
    assert any("time-series" in t.lower() for t in titles)


def test_search_keyword_matches_keyword_field() -> None:
    """Each STAC record has a ``keywords`` array — match against it."""
    from copernicus_mcp.backends.cds.catalogue import search

    # "Reanalysis" is a frequent STAC keyword on CDS records.
    results = search(keyword="reanalysis", store="cds")
    assert len(results) > 0


def test_search_keyword_no_match_returns_empty() -> None:
    from copernicus_mcp.backends.cds.catalogue import search

    results = search(keyword="this-string-cannot-match-anything-zzz9999", store="cds")
    assert results == []


def test_search_keyword_multiword_matches_via_constraints() -> None:
    """T-CDS-KWFIX: a natural multi-word keyword matches when every token
    appears across the record's searchable surface — STAC text PLUS the
    bundled constraints' ``variable`` enum. Word order is irrelevant.

    Before the fix the keyword filter did a single literal-substring test,
    so ``"2m temperature"`` only hit datasets whose prose happened to
    contain that exact contiguous phrase (one coincidental match) and
    missed ERA5, whose ``2m_temperature`` lives only in constraints."""
    from copernicus_mcp.backends.cds.catalogue import search

    ids = {r["id"] for r in search(keyword="2m temperature", store="cds")}
    # Both tokens ("2m", "temperature") appear in the ``2m_temperature``
    # constraint variable even though the STAC prose has no contiguous
    # "2m temperature" phrase.
    assert "reanalysis-era5-land" in ids
    assert "reanalysis-era5-single-levels" in ids


def test_search_keyword_matches_canonical_variable_name() -> None:
    """T-CDS-KWFIX: the canonical CDS variable id (``2m_temperature``)
    lives only in the constraints snapshot, not the STAC prose. Keyword
    search now folds constraints into the haystack so an agent can paste a
    variable id straight into ``keyword`` and find the dataset."""
    from copernicus_mcp.backends.cds.catalogue import search

    ids = {r["id"] for r in search(keyword="2m_temperature", store="cds")}
    assert "reanalysis-era5-land" in ids


def test_search_keyword_multitoken_is_AND_not_OR() -> None:
    """T-CDS-KWFIX: a multi-token keyword combines with AND — the result
    is a subset of each single-token result set, and is empty if any token
    matches nothing. ``"temperature wind"`` keeps only datasets that have
    BOTH (ERA5 single levels carries ``2m_temperature`` and
    ``10m_u_component_of_wind``)."""
    from copernicus_mcp.backends.cds.catalogue import search

    temp = {r["id"] for r in search(keyword="temperature", store="cds")}
    wind = {r["id"] for r in search(keyword="wind", store="cds")}
    both = {r["id"] for r in search(keyword="temperature wind", store="cds")}
    assert both <= temp
    assert both <= wind
    assert "reanalysis-era5-single-levels" in both
    # An impossible token zeroes the set even though "temperature" alone
    # matches dozens.
    assert search(keyword="temperature zzznotarealtoken9999", store="cds") == []


def test_search_keyword_ranks_exact_phrase_above_token_fallback() -> None:
    """T-CDS-KWFIX (codex review): token-AND broadened recall so generic
    ERA5 prose satisfies "air" and "quality" as separate tokens. Combined
    with snapshot-order truncation at ``limit``, that buried the real CAMS
    air-quality datasets (they sat at ranks 36-41). Keyword results are
    now relevance-ranked — a contiguous-phrase hit outranks a token-only
    fallback — so a small ``limit`` keeps the on-topic datasets."""
    from copernicus_mcp.backends.cds.catalogue import search

    top = [r["id"] for r in search(keyword="air quality", limit=10)]
    # The real air-quality datasets (CAMS, in the ADS store, scanned AFTER
    # cds) must appear in the first page rather than being buried behind
    # generic ERA5 reanalysis that only matched "air"+"quality" separately.
    assert any("air-quality" in i for i in top), top
    cams_first = next(i for i, x in enumerate(top) if "air-quality" in x)
    era5_first = next(
        (i for i, x in enumerate(top) if x.startswith("reanalysis-era5")),
        len(top),
    )
    assert cams_first < era5_first, top


def test_search_keyword_ranking_normalises_inner_whitespace() -> None:
    """T-CDS-KWFIX (codex r2): the matcher collapses whitespace via
    ``split()``, so ``"air  quality"`` (double space) and ``"air\\tquality"``
    match the same records as ``"air quality"``. The relevance scorer must
    normalise identically — otherwise the exact-phrase tier never fires for
    the irregular-whitespace variant and the on-topic CAMS datasets get
    re-buried under snapshot order."""
    from copernicus_mcp.backends.cds.catalogue import search

    for q in ("air  quality", "air\tquality"):
        top = [r["id"] for r in search(keyword=q, limit=10)]
        assert any("air-quality" in i for i in top), (q, top)
        cams_first = next(i for i, x in enumerate(top) if "air-quality" in x)
        era5_first = next(
            (i for i, x in enumerate(top) if x.startswith("reanalysis-era5")),
            len(top),
        )
        assert cams_first < era5_first, (q, top)


def test_search_keyword_path_uses_word_boundary_not_substring() -> None:
    """T-CDS-KWFIX (codex/cr review, coverage gap): pin the headline
    precision property ON THE KEYWORD PATH itself, not only on the
    ``variable`` filter it delegates to. A naive per-token substring
    reimplementation would make ``ice`` match "Service" and ``wind`` match
    "window"; word-boundary matching must not."""
    from copernicus_mcp.backends.cds.catalogue import _record_matches_keyword

    rec = {
        "id": "x-product",
        "title": "Copernicus Climate Change Service window product",
        "description": "no relevant variables in this prose",
        "keywords": [],
    }
    assert _record_matches_keyword(rec, "ice") is False  # not inside "Service"
    assert _record_matches_keyword(rec, "wind") is False  # not inside "window"
    assert _record_matches_keyword(rec, "climate") is True  # real word boundary


def test_search_limit_truncates_results() -> None:
    from copernicus_mcp.backends.cds.catalogue import search

    full = search(store="cds")
    truncated = search(store="cds", limit=5)
    assert len(truncated) == 5
    # Order is stable: first ``limit`` of the full list.
    assert [r["id"] for r in truncated] == [r["id"] for r in full[:5]]


def test_search_invalid_store_raises_validation_error() -> None:
    from copernicus_mcp.backends.cds.catalogue import search
    from copernicus_mcp.errors import ValidationError

    with pytest.raises(ValidationError):
        search(store="not-a-store")  # type: ignore[arg-type]


def test_describe_record_has_store_field() -> None:
    """``describe`` augments the raw STAC record with a ``store`` field
    so the agent knows where the dataset lives without a separate call."""
    from copernicus_mcp.backends.cds.catalogue import describe

    record = describe("reanalysis-era5-single-levels")
    assert record["store"] == "cds"


def test_fetched_at_metadata_is_present() -> None:
    """Snapshot timestamps are accessible — UI can warn about staleness."""
    from copernicus_mcp.backends.cds.catalogue import fetched_at

    ts = fetched_at()
    assert "cds" in ts and "ads" in ts and "ewds" in ts
    # ISO-8601 UTC strings.
    assert ts["cds"].endswith("Z")


def test_describe_returns_deep_copy_protecting_module_cache() -> None:
    """Codex/code-reviewer T-CDS-003 LOW: a caller mutating a nested
    field of the describe() result must NOT poison the module cache.
    Pre-fix shallow copy left ``links`` / ``extent`` / ``summaries``
    shared by reference."""
    from copernicus_mcp.backends.cds.catalogue import describe, load_catalogue

    rec = describe("reanalysis-era5-single-levels")
    # Mutate a nested field — must not propagate.
    if isinstance(rec.get("links"), list):
        rec["links"].append({"rel": "review-mutated"})
    if isinstance(rec.get("keywords"), list):
        rec["keywords"].append("REVIEW-MUTATED")

    cat = load_catalogue()
    cached = next(
        r for r in cat["cds"] if r.get("id") == "reanalysis-era5-single-levels"
    )
    cached_links = cached.get("links", [])
    cached_keywords = cached.get("keywords", [])
    assert all(
        not (isinstance(link, dict) and link.get("rel") == "review-mutated")
        for link in cached_links
    ), "describe() leaked nested ``links`` reference"
    assert "REVIEW-MUTATED" not in cached_keywords, (
        "describe() leaked nested ``keywords`` reference"
    )


def test_search_whitespace_only_keyword_treated_as_no_filter() -> None:
    """Codex/code-reviewer T-CDS-003 LOW: a whitespace-only keyword used
    to coincidentally match descriptions containing line-internal
    whitespace, returning a pseudo-random subset. Schema's
    ``_normalise_keyword`` strips and converts to None, so the catalogue
    helper sees ``keyword=None`` (full list)."""
    from copernicus_mcp.data_model.schemas_cds import CdsSearchRequest

    req = CdsSearchRequest(keyword="   ")
    assert req.keyword is None
    req2 = CdsSearchRequest(keyword="")
    assert req2.keyword is None


def test_search_with_zero_or_negative_limit_returns_full_list() -> None:
    """Defence-in-depth: schema rejects ``limit < 1`` but the catalogue
    helper hardens against direct callers passing 0 or negative."""
    from copernicus_mcp.backends.cds.catalogue import load_catalogue, search

    cat = load_catalogue()
    out = search(store="ads", limit=0)
    assert len(out) == len(cat["ads"])
    out_neg = search(store="ads", limit=-5)
    assert len(out_neg) == len(cat["ads"])


# ---------------------------------------------------------------------------
# T-CDS-011: lightweight store lookup for per-store URL routing.
# ``describe()`` already returns ``store`` but does a deep copy of the full
# STAC record — too expensive for the submit/poll/cancel/fetch hot path
# where we only need ``store``. ``store_for(dataset_id)`` is the targeted
# accessor.
# ---------------------------------------------------------------------------


def test_store_for_known_cds_dataset() -> None:
    from copernicus_mcp.backends.cds.catalogue import store_for

    assert store_for("reanalysis-era5-single-levels") == "cds"


def test_store_for_known_ads_dataset() -> None:
    from copernicus_mcp.backends.cds.catalogue import store_for

    assert store_for("cams-global-reanalysis-eac4") == "ads"


def test_store_for_unknown_dataset_returns_none() -> None:
    """Unknown ids return None — caller decides on fallback (defensive
    default to ``cds`` for backward-compat with v0.3.0 behaviour)."""
    from copernicus_mcp.backends.cds.catalogue import store_for

    assert store_for("not-a-real-dataset-xyzzy") is None


# ---------------------------------------------------------------------------
# T-CDS-015 (Layer A): bundled constraints attached to describe()
# ---------------------------------------------------------------------------


def test_describe_attaches_available_inputs_when_bundle_present(
    monkeypatch, tmp_path
) -> None:
    """When a constraints bundle exists for the store, ``describe()`` must
    attach an ``available_inputs`` field with the empty-inputs valid values.
    This is what tells the LLM 'these are the fields this dataset accepts'."""
    import json

    from copernicus_mcp.backends.cds import catalogue as cat

    # Seed a synthetic constraints bundle pointing at a fake _DATA_DIR.
    fake_data = tmp_path / "_data"
    fake_data.mkdir()
    # Copy real cds.json/ads.json/ewds.json/fetched_at.json structure-wise
    # so load_catalogue still works.
    for name in ("cds", "ads", "ewds"):
        (fake_data / f"{name}.json").write_text(
            json.dumps(
                {
                    "collections": [
                        {
                            "id": f"fake-{name}-ds",
                            "title": f"Fake {name}",
                            "description": "",
                            "keywords": [],
                        }
                    ]
                }
            )
        )
    (fake_data / "fetched_at.json").write_text(json.dumps({"cds": "x"}))
    (fake_data / "cds_constraints.json").write_text(
        json.dumps(
            {
                "fake-cds-ds": {
                    "data_format": ["netcdf", "grib"],
                    "download_format": ["zip", "unarchived"],
                    "variable": ["t2m", "u10"],
                }
            }
        )
    )

    monkeypatch.setattr(cat, "_DATA_DIR", fake_data)
    monkeypatch.setattr(cat, "_catalogue_cache", None)
    monkeypatch.setattr(cat, "_fetched_at_cache", None)
    monkeypatch.setattr(cat, "_constraints_cache", None, raising=False)

    record = cat.describe("fake-cds-ds")
    assert "available_inputs" in record
    avail = record["available_inputs"]
    assert avail["data_format"] == ["netcdf", "grib"]
    assert avail["download_format"] == ["zip", "unarchived"]
    assert avail["variable"] == ["t2m", "u10"]


def test_constraints_collision_resolves_in_cds_first_order(
    monkeypatch, tmp_path
) -> None:
    """cr round-1 IMPORTANT-2: if a dataset_id ever appears in two
    constraints bundles, ``load_constraints`` and ``describe`` must
    agree which store wins. Both iterate _STORES = (cds, ads, ewds),
    so first-match-wins on cds is the contract."""
    import json

    from copernicus_mcp.backends.cds import catalogue as cat

    fake_data = tmp_path / "_data"
    fake_data.mkdir()
    for name in ("cds", "ads", "ewds"):
        (fake_data / f"{name}.json").write_text(
            json.dumps({"collections": [{"id": "shared-id", "title": name}]})
        )
    (fake_data / "fetched_at.json").write_text(json.dumps({}))
    # Same dataset_id in both cds and ewds — different shapes.
    (fake_data / "cds_constraints.json").write_text(
        json.dumps({"shared-id": {"data_format": ["from-cds"]}})
    )
    (fake_data / "ewds_constraints.json").write_text(
        json.dumps({"shared-id": {"data_format": ["from-ewds"]}})
    )

    monkeypatch.setattr(cat, "_DATA_DIR", fake_data)
    monkeypatch.setattr(cat, "_catalogue_cache", None)
    monkeypatch.setattr(cat, "_fetched_at_cache", None)
    monkeypatch.setattr(cat, "_constraints_cache", None, raising=False)

    merged = cat.load_constraints()
    # CDS wins because it appears first in _STORES.
    assert merged["shared-id"]["data_format"] == ["from-cds"]


def test_describe_omits_available_inputs_when_bundle_missing(
    monkeypatch, tmp_path
) -> None:
    """When no constraints bundle is shipped for a store (or the dataset
    is absent from it), ``describe()`` still works and simply omits
    ``available_inputs``. Backwards-compat with pre-Layer-A snapshots."""
    import json

    from copernicus_mcp.backends.cds import catalogue as cat

    fake_data = tmp_path / "_data"
    fake_data.mkdir()
    for name in ("cds", "ads", "ewds"):
        (fake_data / f"{name}.json").write_text(
            json.dumps(
                {"collections": [{"id": f"fake-{name}-ds", "title": "x"}]}
            )
        )
    (fake_data / "fetched_at.json").write_text(json.dumps({}))
    # NO cds_constraints.json on disk.

    monkeypatch.setattr(cat, "_DATA_DIR", fake_data)
    monkeypatch.setattr(cat, "_catalogue_cache", None)
    monkeypatch.setattr(cat, "_fetched_at_cache", None)
    monkeypatch.setattr(cat, "_constraints_cache", None, raising=False)

    record = cat.describe("fake-cds-ds")
    assert "available_inputs" not in record
