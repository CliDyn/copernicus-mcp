"""T-CMEMS-HIER-005 runtime routing module tests.

``routing.py`` is the runtime path: ``search_groups`` →
``search_products`` → ``search_datasets_in_products``. Tests pin the
envelope shape, scoring contract, and bbox/time-range filtering
behaviour, plus the lazy-loader pattern shared with
``catalogue.load_catalogue``.

Each search level returns the same envelope::

    {
        "selected": [...],
        "rejected": [...],
        "reason": "<str>",
        "confidence": "high" | "medium" | "low",
        "fallback_available": <bool>,
    }
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Lazy loaders
# ---------------------------------------------------------------------------


def test_load_groups_returns_bundled_groups_list() -> None:
    """``load_groups`` reads ``_data/groups.json`` and returns the
    list of group entries. 47 groups (T-CMEMS-HIER-004 round 2)."""
    from copernicus_mcp.backends.cmems.routing import load_groups

    groups = load_groups()
    assert isinstance(groups, list)
    assert 40 <= len(groups) <= 60, f"unexpected group count: {len(groups)}"
    assert all(isinstance(g, dict) for g in groups)
    assert all("group_id" in g for g in groups)


def test_load_products_returns_bundled_products_list() -> None:
    """``load_products`` returns the 306-entry product manifest."""
    from copernicus_mcp.backends.cmems.routing import load_products

    products = load_products()
    assert isinstance(products, list)
    assert 300 <= len(products) <= 315
    assert all("product_id" in p for p in products)


def test_load_cards_returns_bundled_cards_list() -> None:
    """``load_cards`` returns the 1251-entry dataset card manifest."""
    from copernicus_mcp.backends.cmems.routing import load_cards

    cards = load_cards()
    assert isinstance(cards, list)
    assert 1200 <= len(cards) <= 1300
    assert all("dataset_id" in c for c in cards)


def test_loaders_are_module_cached() -> None:
    """Lazy loaders cache the result — multiple calls return the
    SAME object (not just an equal one). Matches the
    ``catalogue.load_catalogue`` pattern."""
    from copernicus_mcp.backends.cmems.routing import (
        load_cards,
        load_groups,
        load_products,
    )

    assert load_groups() is load_groups()
    assert load_products() is load_products()
    assert load_cards() is load_cards()


# ---------------------------------------------------------------------------
# search_groups — envelope, scoring, confidence
# ---------------------------------------------------------------------------


def test_search_groups_returns_envelope_with_required_keys() -> None:
    """Standard envelope: selected, rejected, reason, confidence,
    fallback_available."""
    from copernicus_mcp.backends.cmems.routing import search_groups

    result = search_groups("arctic sea ice extent")
    assert set(result.keys()) >= {
        "selected",
        "rejected",
        "reason",
        "confidence",
        "fallback_available",
    }
    assert isinstance(result["selected"], list)
    assert isinstance(result["rejected"], list)
    assert isinstance(result["reason"], str)
    assert result["confidence"] in ("high", "medium", "low")
    assert isinstance(result["fallback_available"], bool)


def test_search_groups_selected_entries_have_group_metadata() -> None:
    """Each ``selected`` entry carries enough context for the agent
    to pass it forward — group_id + score at minimum."""
    from copernicus_mcp.backends.cmems.routing import search_groups

    result = search_groups("arctic sea ice extent", top_k=3)
    assert result["selected"], "expected at least one match for clear query"
    first = result["selected"][0]
    assert "group_id" in first
    assert "score" in first
    assert isinstance(first["score"], (int, float))
    assert first["score"] > 0


def test_search_groups_arctic_query_picks_arctic_groups() -> None:
    """End-to-end sanity: an Arctic-flavoured query should surface
    Arctic-prefixed groups in the top selections."""
    from copernicus_mcp.backends.cmems.routing import search_groups

    result = search_groups("arctic sea ice extent", top_k=5)
    selected_ids = [g["group_id"] for g in result["selected"]]
    # At least one Arctic group must be in the top-5.
    assert any("arctic" in gid for gid in selected_ids), selected_ids


def test_search_groups_respects_exclude_phrases() -> None:
    """A query with an exclude phrase (``antarctic``) must not match
    the ``physics-arctic-state`` group which has ``antarctic`` in its
    exclude list."""
    from copernicus_mcp.backends.cmems.routing import search_groups

    result = search_groups("antarctic ocean temperature", top_k=5)
    selected_ids = [g["group_id"] for g in result["selected"]]
    # Arctic physics must NOT win an antarctic query.
    assert "physics-arctic-state" not in selected_ids


def test_search_groups_top_k_caps_results() -> None:
    """``top_k`` caps the size of ``selected``."""
    from copernicus_mcp.backends.cmems.routing import search_groups

    result = search_groups("ocean", top_k=3)
    assert len(result["selected"]) <= 3


def test_search_groups_low_confidence_when_no_strong_match() -> None:
    """An empty / nonsense query should produce low confidence and
    signal that flat search is available as fallback."""
    from copernicus_mcp.backends.cmems.routing import search_groups

    result = search_groups("xyzzyzzy nonsense query 999")
    assert result["confidence"] == "low"
    assert result["fallback_available"] is True


def test_search_groups_high_confidence_when_clear_winner() -> None:
    """A query that clearly maps to one group should produce high
    confidence."""
    from copernicus_mcp.backends.cmems.routing import search_groups

    # "AMOC" is a unique phrase in ocean-circulation-indices.
    result = search_groups("AMOC strength trend", top_k=3)
    assert result["confidence"] == "high"
    selected_ids = [g["group_id"] for g in result["selected"]]
    assert "ocean-circulation-indices" in selected_ids


# ---------------------------------------------------------------------------
# search_products — group_ids filter
# ---------------------------------------------------------------------------


def test_search_products_filters_by_group_membership() -> None:
    """Given group_ids, return only products that appear in at least
    one of those groups' product_ids."""
    from copernicus_mcp.backends.cmems.routing import (
        load_groups,
        search_products,
    )

    result = search_products(group_ids=["physics-arctic-state"])
    assert result["selected"]
    groups = load_groups()
    arctic = next(g for g in groups if g["group_id"] == "physics-arctic-state")
    selected_ids = {p["product_id"] for p in result["selected"]}
    assert selected_ids.issubset(set(arctic["product_ids"]))


def test_search_products_envelope_has_required_keys() -> None:
    from copernicus_mcp.backends.cmems.routing import search_products

    result = search_products(group_ids=["physics-arctic-state"])
    assert set(result.keys()) >= {
        "selected",
        "rejected",
        "reason",
        "confidence",
        "fallback_available",
    }


def test_search_products_unknown_group_id_yields_empty_selection() -> None:
    """An unknown group_id is not an error — just zero matches with
    low confidence + fallback signalled."""
    from copernicus_mcp.backends.cmems.routing import search_products

    result = search_products(group_ids=["does-not-exist"])
    assert result["selected"] == []
    assert result["confidence"] == "low"
    assert result["fallback_available"] is True


def test_search_products_top_k_caps_results() -> None:
    """``top_k`` caps the size of ``selected``."""
    from copernicus_mcp.backends.cmems.routing import search_products

    result = search_products(group_ids=["arctic-comprehensive"], top_k=5)
    assert len(result["selected"]) <= 5


def test_search_products_query_refines_within_group() -> None:
    """When ``query`` is set, products in the group are ranked by
    keyword match against summary / title / variables."""
    from copernicus_mcp.backends.cmems.routing import search_products

    result = search_products(group_ids=["arctic-comprehensive"], query="sea ice", top_k=5)
    selected_titles = " ".join(p.get("product_title", "") for p in result["selected"][:3]).lower()
    assert "ice" in selected_titles or "sea" in selected_titles


def test_search_products_multi_group_unions_membership() -> None:
    """Two group_ids → products from EITHER group are eligible."""
    from copernicus_mcp.backends.cmems.routing import (
        load_groups,
        search_products,
    )

    result = search_products(group_ids=["physics-arctic-state", "bgc-arctic"], top_k=20)
    groups = {g["group_id"]: g for g in load_groups()}
    union = set(groups["physics-arctic-state"]["product_ids"]) | set(
        groups["bgc-arctic"]["product_ids"]
    )
    selected_ids = {p["product_id"] for p in result["selected"]}
    assert selected_ids.issubset(union)


# ---------------------------------------------------------------------------
# search_datasets_in_products — product_ids filter + bbox + time_range
# ---------------------------------------------------------------------------


def test_search_datasets_in_products_filters_by_product_membership() -> None:
    """Given product_ids, return only cards whose ``product_id`` is in
    the list."""
    from copernicus_mcp.backends.cmems.routing import (
        search_datasets_in_products,
    )

    result = search_datasets_in_products(
        product_ids=["GLOBAL_ANALYSISFORECAST_PHY_001_024"], limit=10
    )
    assert result["selected"]
    assert all(c["product_id"] == "GLOBAL_ANALYSISFORECAST_PHY_001_024" for c in result["selected"])


def test_search_datasets_in_products_envelope_has_required_keys() -> None:
    from copernicus_mcp.backends.cmems.routing import (
        search_datasets_in_products,
    )

    result = search_datasets_in_products(
        product_ids=["GLOBAL_ANALYSISFORECAST_PHY_001_024"], limit=10
    )
    assert set(result.keys()) >= {
        "selected",
        "rejected",
        "reason",
        "confidence",
        "fallback_available",
    }


def test_search_datasets_in_products_limit_caps_results() -> None:
    """``limit`` caps the response — ≤10 per spec acceptance."""
    from copernicus_mcp.backends.cmems.routing import (
        search_datasets_in_products,
    )

    result = search_datasets_in_products(
        product_ids=["IBI_MULTIYEAR_PHY_005_002"],  # 39 datasets
        limit=5,
    )
    assert len(result["selected"]) <= 5


def test_search_datasets_in_products_bbox_excludes_null_spatial_extent() -> None:
    """Acceptance: when bbox is set, cards with
    ``spatial_extent is None`` are excluded."""
    from copernicus_mcp.backends.cmems.routing import (
        search_datasets_in_products,
    )

    # Antarctic OMI products often have null spatial extents.
    result = search_datasets_in_products(
        product_ids=["ANTARCTIC_OMI_SI_extent"],
        limit=10,
        bbox=(-180.0, -90.0, 180.0, -50.0),
    )
    for c in result["selected"]:
        assert c["spatial_extent"] is not None


def test_search_datasets_in_products_bbox_filters_by_overlap() -> None:
    """A Black Sea bbox (20-30°E, 41-47°N) should select only
    datasets whose spatial_extent overlaps."""
    from copernicus_mcp.backends.cmems.routing import (
        search_datasets_in_products,
    )

    result = search_datasets_in_products(
        product_ids=["BLKSEA_ANALYSISFORECAST_PHY_007_001"],
        limit=20,
        bbox=(20.0, 41.0, 30.0, 47.0),
    )
    for c in result["selected"]:
        ext = c["spatial_extent"]
        assert ext is not None
        assert ext["min_lon"] <= 30.0 and ext["max_lon"] >= 20.0
        assert ext["min_lat"] <= 47.0 and ext["max_lat"] >= 41.0


def test_search_datasets_in_products_time_range_excludes_null_temporal() -> None:
    """When time_range is set, cards with ``temporal_extent is None``
    are excluded."""
    from copernicus_mcp.backends.cmems.routing import (
        search_datasets_in_products,
    )

    result = search_datasets_in_products(
        product_ids=["ANTARCTIC_OMI_SI_extent"],
        limit=10,
        time_range=("2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"),
    )
    for c in result["selected"]:
        assert c["temporal_extent"] is not None


def test_search_datasets_in_products_no_filter_returns_all_cards_in_product() -> None:
    """Without bbox or time_range, every card belonging to the
    product is eligible (capped by limit)."""
    from copernicus_mcp.backends.cmems.routing import (
        search_datasets_in_products,
    )

    result = search_datasets_in_products(product_ids=["ANTARCTIC_OMI_SI_extent"], limit=20)
    assert result["selected"]


# ---------------------------------------------------------------------------
# Round-trip integration: agent pipeline (query → groups → products → datasets)
# ---------------------------------------------------------------------------


def test_round_trip_antarctic_sea_ice_picks_correct_dataset() -> None:
    """Acceptance: simulate the 3-hop pipeline for ``Antarctic sea
    ice extent``. Must surface an Antarctic OMI SI dataset within
    3 hops."""
    from copernicus_mcp.backends.cmems.routing import (
        search_datasets_in_products,
        search_groups,
        search_products,
    )

    g = search_groups("antarctic sea ice extent", top_k=3)
    p = search_products(
        group_ids=[entry["group_id"] for entry in g["selected"]],
        top_k=10,
    )
    d = search_datasets_in_products(
        product_ids=[entry["product_id"] for entry in p["selected"]],
        limit=10,
    )
    dataset_ids = [c["dataset_id"] for c in d["selected"]]
    assert any("antarctic" in did.lower() and "si" in did.lower() for did in dataset_ids), (
        f"no Antarctic SI dataset in top-{len(dataset_ids)}: {dataset_ids[:5]}"
    )


def test_round_trip_global_temperature_reanalysis() -> None:
    """3-hop pipeline must surface a global physics reanalysis
    dataset for ``global temperature reanalysis``."""
    from copernicus_mcp.backends.cmems.routing import (
        search_datasets_in_products,
        search_groups,
        search_products,
    )

    g = search_groups("global temperature reanalysis", top_k=3)
    p = search_products(
        group_ids=[entry["group_id"] for entry in g["selected"]],
        top_k=10,
    )
    d = search_datasets_in_products(
        product_ids=[entry["product_id"] for entry in p["selected"]],
        limit=10,
    )
    selected = d["selected"]
    assert selected
    # At least one global physics reanalysis card.
    assert any(
        "global" in (c.get("regions") or c.get("region") or [])
        or c.get("product_id", "").startswith("GLOBAL_MULTIYEAR_PHY_")
        for c in selected
    )


# ---------------------------------------------------------------------------
# CmemsBackend protocol wrappers
# ---------------------------------------------------------------------------


def _make_foundation_for_backend(tmp_path):
    """Build a FoundationServices with the minimal moving parts the
    backend needs for the offline routing methods. No SDK calls,
    no network — search_groups / search_products / search(product_ids=...)
    are pure offline reads of the bundled manifests."""
    from copernicus_mcp.auth import CredentialResolver
    from copernicus_mcp.backends.abstract import FoundationServices
    from copernicus_mcp.cache import CacheManager
    from copernicus_mcp.config import ConfigLoader
    from copernicus_mcp.data_model.coordinator import DataModelCoordinator
    from copernicus_mcp.data_model.provenance import ProvenanceRecorder
    from copernicus_mcp.errors.sanitiser import Sanitiser
    from copernicus_mcp.http import HttpClientFactory
    from copernicus_mcp.persistence import SqliteBackend

    config = ConfigLoader().load()
    persistence = SqliteBackend(tmp_path / "state.db")
    cache = CacheManager(
        cache_directory=tmp_path / "cache",
        persistence=persistence,
        size_limit_bytes=10 * 1024 * 1024,
    )
    return FoundationServices(
        config=config,
        credential_resolver=CredentialResolver(),
        http_client_factory=HttpClientFactory(http_config=config.http),
        persistence=persistence,
        cache=cache,
        sanitiser=Sanitiser(),
        data_model=DataModelCoordinator(persistence=persistence),
        provenance=ProvenanceRecorder(
            persistence=persistence,
            software_versions={"copernicus-mcp": "0.0.1"},
        ),
    )


def test_backend_search_groups_returns_routing_envelope(tmp_path) -> None:
    """``CmemsBackend.search_groups({"query": "..."})`` returns the
    same envelope as ``routing.search_groups``."""
    import asyncio

    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    foundation = _make_foundation_for_backend(tmp_path)
    backend = CmemsBackend(foundation=foundation, credentials=None)

    result = asyncio.run(backend.search_groups({"query": "arctic sea ice extent"}))
    assert "selected" in result and result["selected"]
    assert result["confidence"] in ("high", "medium", "low")


def test_backend_search_products_returns_routing_envelope(tmp_path) -> None:
    import asyncio

    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    foundation = _make_foundation_for_backend(tmp_path)
    backend = CmemsBackend(foundation=foundation, credentials=None)

    result = asyncio.run(backend.search_products({"group_ids": ["physics-arctic-state"]}))
    assert "selected" in result and result["selected"]


def test_backend_search_with_product_ids_routes_via_cards(tmp_path) -> None:
    """``CmemsBackend.search`` with ``product_ids`` in params routes
    through the hierarchical cards path instead of the flat slim
    search."""
    import asyncio

    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    foundation = _make_foundation_for_backend(tmp_path)
    backend = CmemsBackend(foundation=foundation, credentials=None)

    result = asyncio.run(
        backend.search(
            {
                "product_ids": ["GLOBAL_ANALYSISFORECAST_PHY_001_024"],
                "limit": 5,
            }
        )
    )
    assert result["selected"]
    # Enriched card fields are present.
    first = result["selected"][0]
    assert "domain" in first
    assert "region" in first
    assert "data_type" in first


def test_backend_search_without_product_ids_keeps_flat_behaviour(tmp_path) -> None:
    """Backward compat: ``CmemsBackend.search({"keyword": ...})`` still
    returns the slim flat envelope (``datasets``, ``total_count``,
    ``mode``)."""
    import asyncio

    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    foundation = _make_foundation_for_backend(tmp_path)
    backend = CmemsBackend(foundation=foundation, credentials=None)

    result = asyncio.run(backend.search({"keyword": "temperature", "limit": 3}))
    assert "datasets" in result
    assert "total_count" in result
    assert "mode" in result


# ---------------------------------------------------------------------------
# Round-1 cr+codex HIGH/MEDIUM regressions
# ---------------------------------------------------------------------------


def test_search_datasets_low_confidence_when_query_yields_zero_score() -> None:
    """cr+codex round-1 HIGH: ``confidence`` was always ``"high"`` even when
    the query had zero substring match on any card — because the pool was
    sliced to ``limit`` BEFORE the ``len > limit`` check. A query that doesn't
    match any card must surface as low confidence + fallback_available=True
    so the agent knows to widen the search."""
    from copernicus_mcp.backends.cmems.routing import (
        search_datasets_in_products,
    )

    result = search_datasets_in_products(
        product_ids=["GLOBAL_ANALYSISFORECAST_PHY_001_024"],
        query="definitelynotindataset",
        limit=10,
    )
    assert result["confidence"] == "low"
    assert result["fallback_available"] is True


def test_search_datasets_medium_confidence_when_pool_exceeds_limit() -> None:
    """When the candidate pool exceeds ``limit`` (more results available
    than returned), confidence is medium — the caller should paginate or
    refine."""
    from copernicus_mcp.backends.cmems.routing import (
        search_datasets_in_products,
    )

    # IBI_MULTIYEAR_PHY_005_002 has 39 cards; limit=5 returns 5 of 39.
    result = search_datasets_in_products(product_ids=["IBI_MULTIYEAR_PHY_005_002"], limit=5)
    assert result["confidence"] == "medium"


def test_search_with_unsupported_service_types_raises_validation_error(
    tmp_path,
) -> None:
    """codex round-1 HIGH: when bbox/time_range/product_ids routed
    through the cards path, ``service_types`` was silently dropped
    instead of raising ValidationError (the flat path does reject)."""
    import asyncio

    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import ValidationError as CmcpValidationError

    foundation = _make_foundation_for_backend(tmp_path)
    backend = CmemsBackend(foundation=foundation, credentials=None)

    with pytest.raises(CmcpValidationError):
        asyncio.run(
            backend.search(
                {
                    "product_ids": ["GLOBAL_ANALYSISFORECAST_PHY_001_024"],
                    "service_types": ["geoseries"],
                }
            )
        )


def test_search_with_malformed_bbox_raises_validation_error(tmp_path) -> None:
    """cr round-1 HIGH: a 3-element bbox was silently dropped instead
    of raising ValidationError with a recovery hint."""
    import asyncio

    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import ValidationError as CmcpValidationError

    foundation = _make_foundation_for_backend(tmp_path)
    backend = CmemsBackend(foundation=foundation, credentials=None)

    with pytest.raises(CmcpValidationError):
        asyncio.run(
            backend.search(
                {
                    "product_ids": ["GLOBAL_ANALYSISFORECAST_PHY_001_024"],
                    "bbox": [20.0, 41.0, 30.0],  # 3 elements
                }
            )
        )


def test_search_with_antimeridian_bbox_raises_validation_error(tmp_path) -> None:
    """cr round-1 MEDIUM: an antimeridian-crossing bbox
    (min_lon > max_lon) silently produced wrong results. the project conventions
    inv-7 forbids antimeridian bboxes — reject with a recovery hint."""
    import asyncio

    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import ValidationError as CmcpValidationError

    foundation = _make_foundation_for_backend(tmp_path)
    backend = CmemsBackend(foundation=foundation, credentials=None)

    with pytest.raises(CmcpValidationError):
        asyncio.run(
            backend.search(
                {
                    "product_ids": ["GLOBAL_ANALYSISFORECAST_PHY_001_024"],
                    "bbox": [170.0, -10.0, -170.0, 10.0],  # crosses dateline
                }
            )
        )


def test_phrase_in_normalises_hyphen_to_space() -> None:
    """codex round-1 MEDIUM: manifest phrases like ``sea-surface
    height`` should match user queries spelled ``sea surface height``
    and vice versa. Hyphens are visual artefacts, not semantic."""
    from copernicus_mcp.backends.cmems.routing import _phrase_in

    assert _phrase_in("sea-surface height", "global sea surface height anomaly")
    assert _phrase_in("sea surface height", "global sea-surface height anomaly")


def test_phrase_in_word_boundary_rejects_substring_match() -> None:
    """Regression pin for the ``arctic`` ⊂ ``antarctic`` false
    positive that round-1 reported on PR #91 spec discussion."""
    from copernicus_mcp.backends.cmems.routing import _phrase_in

    assert _phrase_in("arctic", "the arctic ocean")
    assert not _phrase_in("arctic", "the antarctic ocean")


def test_temporal_overlap_handles_mixed_iso_precision() -> None:
    """cr round-1 MEDIUM: lexicographic comparison broke on mixed
    precision (``2020-06-01`` vs ``2020-06-01T00:00:00Z``). Parse as
    datetime and compare proper time values."""
    from copernicus_mcp.backends.cmems.routing import _temporal_overlaps

    # Extent end equals query start (touching edge → overlap).
    extent = {
        "start_datetime": "2019-01-01",
        "end_datetime": "2020-06-01",
    }
    assert _temporal_overlaps(extent, ("2020-06-01T00:00:00Z", "2020-12-31T00:00:00Z"))
    # Extent ends BEFORE query starts → no overlap.
    extent_before = {
        "start_datetime": "2019-01-01",
        "end_datetime": "2020-05-31",
    }
    assert not _temporal_overlaps(extent_before, ("2020-06-01T00:00:00Z", "2020-12-31T00:00:00Z"))


# ---------------------------------------------------------------------------
# Round-2 codex HIGH + MEDIUMs
# ---------------------------------------------------------------------------


def test_search_rejects_product_id_and_product_ids_set_together(tmp_path) -> None:
    """codex round-2 HIGH: ``product_id=A, product_ids=[A,B]`` used
    to slip through (singular ``in`` list), then routing silently
    included B too. Always reject when both are set."""
    import asyncio

    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import ValidationError as CmcpValidationError

    foundation = _make_foundation_for_backend(tmp_path)
    backend = CmemsBackend(foundation=foundation, credentials=None)

    with pytest.raises(CmcpValidationError):
        asyncio.run(
            backend.search(
                {
                    "product_id": "GLOBAL_ANALYSISFORECAST_PHY_001_024",
                    "product_ids": [
                        "GLOBAL_ANALYSISFORECAST_PHY_001_024",
                        "GLOBAL_MULTIYEAR_PHY_001_030",
                    ],
                    "bbox": [-10.0, 0.0, 10.0, 10.0],
                }
            )
        )


def test_search_rejects_non_iso_time_range(tmp_path) -> None:
    """codex round-2 MEDIUM: ``_validate_time_range`` parsed only
    shape, not ISO format. Non-ISO strings now raise."""
    import asyncio

    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import ValidationError as CmcpValidationError

    foundation = _make_foundation_for_backend(tmp_path)
    backend = CmemsBackend(foundation=foundation, credentials=None)

    with pytest.raises(CmcpValidationError):
        asyncio.run(
            backend.search(
                {
                    "product_ids": ["GLOBAL_ANALYSISFORECAST_PHY_001_024"],
                    "time_range": ["bad-start", "bad-end"],
                }
            )
        )


def test_search_rejects_reversed_time_range(tmp_path) -> None:
    """codex round-2 MEDIUM: reversed time_range (start >= end)
    used to pass shape validation and degrade to silent no-match."""
    import asyncio

    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import ValidationError as CmcpValidationError

    foundation = _make_foundation_for_backend(tmp_path)
    backend = CmemsBackend(foundation=foundation, credentials=None)

    with pytest.raises(CmcpValidationError):
        asyncio.run(
            backend.search(
                {
                    "product_ids": ["GLOBAL_ANALYSISFORECAST_PHY_001_024"],
                    "time_range": [
                        "2024-12-31T00:00:00Z",
                        "2024-01-01T00:00:00Z",
                    ],
                }
            )
        )


def test_search_accepts_zero_width_bbox(tmp_path) -> None:
    """codex round-2 MEDIUM: ``min_lon == max_lon`` (zero-width
    meridian bbox) was rejected as antimeridian — inconsistent with
    the subset invariant which allows ``min <= max``. Now accepted."""
    import asyncio

    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    foundation = _make_foundation_for_backend(tmp_path)
    backend = CmemsBackend(foundation=foundation, credentials=None)

    # A zero-width bbox should not raise. Result content is not
    # the point — only that the validator accepts the shape.
    result = asyncio.run(
        backend.search(
            {
                "product_ids": ["GLOBAL_ANALYSISFORECAST_PHY_001_024"],
                "bbox": [0.0, -10.0, 0.0, 10.0],
            }
        )
    )
    assert "selected" in result


def test_temporal_overlaps_returns_false_for_corrupt_extent() -> None:
    """codex round-2 MEDIUM: a card with ``start > end`` could
    spuriously match broad queries straddling both endpoints. Treat
    as no extent (under-select)."""
    from copernicus_mcp.backends.cmems.routing import _temporal_overlaps

    corrupt = {
        "start_datetime": "2020-12-31T00:00:00Z",
        "end_datetime": "2020-01-01T00:00:00Z",
    }
    # Broad query that would straddle both endpoints of a corrupt
    # extent under the naive (start <= q_end and end >= q_start)
    # check.
    assert not _temporal_overlaps(corrupt, ("2019-01-01T00:00:00Z", "2021-12-31T00:00:00Z"))


def test_search_datasets_limit_capped_at_upper_bound() -> None:
    """LOW round-1: ``limit`` had no upper bound; a caller could
    request 1251 cards inline. Cap at a sensible upper bound (50)."""
    from copernicus_mcp.backends.cmems.routing import (
        search_datasets_in_products,
    )

    result = search_datasets_in_products(product_ids=["IBI_MULTIYEAR_PHY_005_002"], limit=10_000)
    assert len(result["selected"]) <= 50


def test_round_trip_mediterranean_salinity() -> None:
    """3-hop pipeline for ``Mediterranean salinity`` must surface a
    Med physics dataset within 3 hops."""
    from copernicus_mcp.backends.cmems.routing import (
        search_datasets_in_products,
        search_groups,
        search_products,
    )

    g = search_groups("mediterranean salinity", top_k=3)
    p = search_products(
        group_ids=[entry["group_id"] for entry in g["selected"]],
        top_k=10,
    )
    d = search_datasets_in_products(
        product_ids=[entry["product_id"] for entry in p["selected"]],
        limit=10,
    )
    dataset_ids = [c["product_id"] for c in d["selected"]]
    assert any("MEDSEA" in pid and "PHY" in pid for pid in dataset_ids), (
        f"no Mediterranean physics product in top-{len(dataset_ids)}"
    )
