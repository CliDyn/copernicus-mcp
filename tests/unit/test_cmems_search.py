"""T-CMEMS-CAT-003: ``CmemsBackend.search`` now reads the bundled
catalogue snapshot offline instead of calling
``copernicusmarine.describe()``. This file replaces the live-SDK
search tests (which were retired alongside the live path); SDK
error-mapping is still exercised by ``test_cmems_describe.py``
since ``describe()`` retains the live call.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio


def _make_foundation(tmp_path: Path):
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
    return (
        FoundationServices(
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
        ),
        persistence,
    )


@pytest_asyncio.fixture
async def foundation(tmp_path: Path):
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        yield found
    finally:
        await persistence.close()


def _creds():
    from copernicus_mcp.auth.resolver import ResolvedCredentials

    return ResolvedCredentials(
        backend="cmems",
        source="explicit",
        source_detail="test",
        fields={"username": "u", "password": "p"},
    )


# ---------------------------------------------------------------------------
# Schema validation (retained from the live-SDK era — still applies)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_invalid_params_raise_validation_error(foundation) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import ValidationError

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(ValidationError):
        await backend.search({"limit": -1})  # ge=1 violated


# ---------------------------------------------------------------------------
# Offline-search envelope (new in CAT-003)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_envelope_from_offline_snapshot(
    foundation,
) -> None:
    """``search`` no longer calls the SDK — it reads the bundled
    ``_data/marine.json`` snapshot and returns the same canonical
    envelope (``datasets`` + ``total_count``) plus the new
    ``catalogue_fetched_at`` field.

    Pin against the real bundled snapshot (1251 records, 306 products)
    so a packaging misconfiguration that strips ``_data/`` from the
    wheel fails this test rather than silently regressing offline
    search."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    result = await backend.search({"keyword": "temperature", "limit": 5})

    assert isinstance(result, dict)
    assert "datasets" in result
    assert "total_count" in result
    assert "catalogue_fetched_at" in result
    assert 0 < len(result["datasets"]) <= 5
    assert result["total_count"] > 5  # many more than 5 match
    # Each row preserves the slim-record schema fields.
    for ds in result["datasets"]:
        assert {"dataset_id", "title", "product_id"} <= set(ds.keys())


@pytest.mark.asyncio
async def test_search_works_without_credentials(foundation) -> None:
    """T-CMEMS-CAT-003 flips the v0.3.1 behaviour: search no longer
    requires CMEMS credentials. Discovery is offline, so a credless
    setup can still browse the catalogue.

    Replaces the v0.3.1 ``test_search_no_credentials_raises_auth_error``
    test (deleted with this PR). The remaining live-SDK operations
    (``describe``, ``estimate``, ``subset``) still raise ``AuthError``
    without credentials — pinned in their respective tests."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=None)
    result = await backend.search({"keyword": "temperature"})

    assert "datasets" in result
    assert result["total_count"] > 0


@pytest.mark.asyncio
async def test_search_does_not_import_copernicusmarine(
    foundation, monkeypatch
) -> None:
    """The strongest possible pin on "search is fully offline".

    codex round-1 LOW-2: if ``CmemsBackend`` is imported BEFORE the
    blocker is installed and a previous test in the same process
    already loaded ``copernicusmarine``, the import-machinery
    interception silently misses. Fix order:

    1. Purge every ``copernicusmarine`` / ``copernicusmarine.<sub>``
       key from ``sys.modules`` so any subsequent import has to go
       through ``sys.meta_path``.
    2. Install the meta_path blocker.
    3. THEN import ``CmemsBackend`` and run search.

    This way, if a future regression adds an unconditional
    ``import copernicusmarine`` at backend-module load time the
    blocker fires at backend import, the test fails loudly, and
    the regression is caught.
    """
    import sys

    # 1. Purge ALL cached copernicusmarine entries (top-level + subpkgs).
    for mod_name in list(sys.modules):
        if mod_name == "copernicusmarine" or mod_name.startswith(
            "copernicusmarine."
        ):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
    # Also drop any cached CmemsBackend so re-importing it now runs
    # the module body again under the blocker.
    for mod_name in list(sys.modules):
        if mod_name.startswith("copernicus_mcp.backends.cmems"):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    # 2. Install the blocker BEFORE re-importing the backend.
    real_meta_path = sys.meta_path

    class _BlockMarineImport:
        def find_spec(self, name, *args, **kwargs):
            if name == "copernicusmarine" or name.startswith(
                "copernicusmarine."
            ):
                raise ImportError(
                    "search must not import copernicusmarine; "
                    f"attempted to import {name!r}"
                )
            return None

    monkeypatch.setattr(
        sys, "meta_path", [_BlockMarineImport(), *real_meta_path]
    )

    # 3. Re-import backend under the active blocker.
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    result = await backend.search({"keyword": "temperature", "limit": 1})
    assert isinstance(result, dict)
    assert len(result["datasets"]) == 1


@pytest.mark.asyncio
async def test_search_total_count_is_unfiltered_match_count(
    foundation,
) -> None:
    """Live-path semantics preserved: ``total_count`` reflects the
    PRE-slice match count, not the returned page length. Live
    ``_map_describe_response`` did this; the snapshot path must too."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    page = await backend.search({"keyword": "temperature", "limit": 3})
    full = await backend.search({"keyword": "temperature"})

    assert page["total_count"] == full["total_count"] == len(full["datasets"])
    assert len(page["datasets"]) == 3
    assert page["total_count"] > 3


@pytest.mark.asyncio
async def test_search_catalogue_fetched_at_matches_bundled_snapshot(
    foundation,
) -> None:
    """``catalogue_fetched_at`` is the timestamp the refresh script
    wrote at slim-build time. The envelope exposes it so agents and
    users can see how stale the offline catalogue is.

    Compared to the read-side module directly so a future refresh
    that rotates the timestamp does not break the test."""
    from copernicus_mcp.backends.cmems import catalogue as cat
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    # Force a fresh module-cache read so the test value matches the
    # actual bundled file at test time.
    cat._catalogue_cache = None  # type: ignore[attr-defined]
    cat._fetched_at_cache = None  # type: ignore[attr-defined]

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    result = await backend.search({"keyword": "temperature", "limit": 1})

    assert result["catalogue_fetched_at"] == cat.fetched_at()


@pytest.mark.asyncio
async def test_search_product_id_filter_end_to_end(foundation) -> None:
    """The ``product_id`` schema field flows through to
    ``catalogue.search(product_id=...)``. Live search forwarded this
    to ``copernicusmarine.describe(product_id=...)``; the snapshot
    path must preserve the exact-match semantics."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    result = await backend.search(
        {"product_id": "GLOBAL_ANALYSISFORECAST_PHY_001_024"}
    )

    assert len(result["datasets"]) >= 1
    for ds in result["datasets"]:
        assert ds["product_id"] == "GLOBAL_ANALYSISFORECAST_PHY_001_024"


@pytest.mark.asyncio
async def test_search_unknown_keyword_returns_empty_list_with_zero_count(
    foundation,
) -> None:
    """No matches → ``{datasets: [], total_count: 0,
    catalogue_fetched_at: <iso>}``."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    result = await backend.search(
        {"keyword": "xyzzy-no-such-marker-anywhere-in-the-catalogue"}
    )

    assert result["datasets"] == []
    assert result["total_count"] == 0
    assert "catalogue_fetched_at" in result


# ---------------------------------------------------------------------------
# Hybrid live / offline mode (T-CMEMS-CAT-003a)
# ---------------------------------------------------------------------------


def _install_fake_cmem_sdk(monkeypatch, describe_fn):
    """Install a fake ``copernicusmarine`` module exposing ``describe``,
    used by the ``live=True`` branch of search."""
    import sys
    import types

    mod = types.ModuleType("copernicusmarine")
    mod.describe = describe_fn  # type: ignore[attr-defined]

    class LoginError(Exception):
        pass

    class DatasetNotFound(Exception):
        pass

    mod.LoginError = LoginError  # type: ignore[attr-defined]
    mod.DatasetNotFound = DatasetNotFound  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "copernicusmarine", mod)
    return mod


def _fake_live_describe_response():
    """Real SDK shape: ``versions[].parts[].services[].variables[]``
    nested 4 levels deep. Matches the empirical shape from
    T-CMEMS-CAT-000 smoke."""
    return {
        "products": [
            {
                "product_id": "LIVE_FRESH_PRODUCT",
                "title": "Live-fetched fresh product",
                "description": "A product published after the snapshot was refreshed.",
                "digital_object_identifier": "10.0/live-doi",
                "sources": [],
                "processing_level": None,
                "production_center": "Mercator Ocean International",
                "keywords": [],
                "thumbnail_url": "",
                "datasets": [
                    {
                        "dataset_id": "live_fresh_ds_001",
                        "dataset_name": "Live fresh dataset",
                        "product_id": "LIVE_FRESH_PRODUCT",
                        "digital_object_identifier": "10.0/live-doi",
                        "versions": [
                            {
                                "label": "202506",
                                "parts": [
                                    {
                                        "name": "default",
                                        "services": [
                                            {
                                                "service_name": "original-files",
                                                "service_short_name": "files",
                                                "variables": [
                                                    {
                                                        "short_name": "thetao",
                                                        "bbox": [-180.0, -90.0, 180.0, 90.0],
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }


@pytest.mark.asyncio
async def test_search_offline_mode_is_default(foundation) -> None:
    """``live`` defaults to False. An unspecified-mode search reads
    the bundled snapshot — the v0.3.2 behavior."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=None)
    result = await backend.search({"keyword": "temperature", "limit": 1})
    # No-creds path proves it took the offline branch.
    assert result["mode"] == "offline"
    assert "catalogue_fetched_at" in result
    assert result["catalogue_fetched_at"]  # non-empty


@pytest.mark.asyncio
async def test_search_live_mode_requires_credentials(foundation) -> None:
    """Opt-in ``live=True`` re-introduces the live-SDK path, which
    requires CMEMS credentials. With ``credentials=None`` the
    backend raises ``AuthError`` before any SDK call."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import AuthError

    backend = CmemsBackend(foundation=foundation, credentials=None)
    with pytest.raises(AuthError):
        await backend.search({"keyword": "temperature", "live": True})


@pytest.mark.asyncio
async def test_search_live_mode_calls_sdk_and_returns_slim_records(
    foundation, monkeypatch
) -> None:
    """``live=True`` with valid credentials calls ``copernicusmarine.
    describe(contains=[keyword], ...)`` and returns the result mapped
    into the SAME slim-record envelope shape offline mode produces."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    captured_kwargs: dict = {}

    def fake_describe(**kwargs):
        captured_kwargs.update(kwargs)
        return _fake_live_describe_response()

    _install_fake_cmem_sdk(monkeypatch, fake_describe)

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    result = await backend.search({"keyword": "fresh", "live": True})

    assert captured_kwargs.get("contains") == ["fresh"]
    assert result["mode"] == "live"
    # Live results are not from the snapshot; the timestamp field is
    # explicitly None so consumers can distinguish freshness sources.
    assert result["catalogue_fetched_at"] is None
    assert result["total_count"] == 1
    ds = result["datasets"][0]
    assert ds["dataset_id"] == "live_fresh_ds_001"
    assert ds["product_id"] == "LIVE_FRESH_PRODUCT"
    # Slim schema preserved: title mirrors dataset_name.
    assert ds["title"] == "Live fresh dataset"
    # Variables walked from the deeply-nested SDK shape.
    assert ds["variables"] == ["thetao"]


@pytest.mark.asyncio
async def test_search_live_mode_forwards_product_id_to_sdk(
    foundation, monkeypatch
) -> None:
    """When ``product_id`` is set, the live path forwards it as an
    SDK kwarg (server-side exact-match) rather than client-side
    filtering."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    captured_kwargs: dict = {}

    def fake_describe(**kwargs):
        captured_kwargs.update(kwargs)
        return _fake_live_describe_response()

    _install_fake_cmem_sdk(monkeypatch, fake_describe)

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    await backend.search(
        {"product_id": "GLOBAL_ANALYSISFORECAST_PHY_001_024", "live": True}
    )

    assert captured_kwargs.get("product_id") == "GLOBAL_ANALYSISFORECAST_PHY_001_024"


@pytest.mark.asyncio
async def test_search_live_mode_respects_limit(foundation, monkeypatch) -> None:
    """``limit`` slices live results client-side. ``total_count`` is
    the unsliced match count (matches offline semantics)."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    def fake_describe(**kwargs):
        # 3 datasets in the response so we can slice to 2.
        resp = _fake_live_describe_response()
        resp["products"][0]["datasets"] = [
            {**resp["products"][0]["datasets"][0], "dataset_id": f"live_ds_{i}"}
            for i in range(3)
        ]
        return resp

    _install_fake_cmem_sdk(monkeypatch, fake_describe)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    result = await backend.search({"live": True, "limit": 2})

    assert len(result["datasets"]) == 2
    assert result["total_count"] == 3


@pytest.mark.asyncio
async def test_search_offline_mode_envelope_includes_mode_field(
    foundation,
) -> None:
    """Both modes ship a ``mode`` field so consumers can tell snapshot
    vs live results apart."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    result = await backend.search({"keyword": "temperature", "limit": 1})
    assert result["mode"] == "offline"


@pytest.mark.asyncio
async def test_search_live_mode_passes_disable_progress_bar(
    foundation, monkeypatch
) -> None:
    """cr round-1 M1: the live path must always pass
    ``disable_progress_bar=True`` so the SDK doesn't write a tqdm
    progress bar to stderr/stdout under MCP stdio (the project conventions inv-4
    stdout discipline). Pin the kwarg explicitly."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    captured_kwargs: dict = {}

    def fake_describe(**kwargs):
        captured_kwargs.update(kwargs)
        return _fake_live_describe_response()

    _install_fake_cmem_sdk(monkeypatch, fake_describe)

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    await backend.search({"keyword": "fresh", "live": True})

    assert captured_kwargs.get("disable_progress_bar") is True


@pytest.mark.asyncio
async def test_search_live_mode_works_with_no_filters(
    foundation, monkeypatch
) -> None:
    """cr round-1 M3: ``live=True`` with neither ``keyword`` nor
    ``product_id`` is the "full live catalogue" path (~10 s SDK
    call). Pin that it works: no ``contains`` / no ``product_id``
    kwargs forwarded, response still mapped into the slim envelope.

    A future "always require a filter" guard would silently narrow
    the live contract; this test catches it."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    captured_kwargs: dict = {}

    def fake_describe(**kwargs):
        captured_kwargs.update(kwargs)
        return _fake_live_describe_response()

    _install_fake_cmem_sdk(monkeypatch, fake_describe)

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    result = await backend.search({"live": True})

    # codex round-2 LOW: stronger pin — assert the exact kwargs set
    # so an accidental future kwarg leak (e.g. someone adds a default
    # ``include_versions=True``) shows up here.
    assert captured_kwargs == {"disable_progress_bar": True}
    # Envelope still well-formed.
    assert result["mode"] == "live"
    assert result["total_count"] == 1


@pytest.mark.asyncio
async def test_search_live_mode_maps_login_error_to_auth_error(
    foundation, monkeypatch
) -> None:
    """cr round-1 M2: ``_wrap_marine_exception`` is shared with
    describe/estimate/subset, but the live-search path is new and
    the ``op="search"`` interpolation hadn't been covered. Pin that
    a fake ``LoginError`` from the live SDK becomes ``AuthError``
    with the right recovery action."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import AuthError

    def fake_describe(**kwargs):
        # Raise the SDK's LoginError exception class as published by
        # the fake module (matches the real shape).
        from copernicusmarine import LoginError  # type: ignore[attr-defined]

        raise LoginError("bad creds")

    _install_fake_cmem_sdk(monkeypatch, fake_describe)

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(AuthError):
        await backend.search({"keyword": "x", "live": True})


@pytest.mark.asyncio
async def test_search_live_mode_whitespace_keyword_normalised_like_offline(
    foundation, monkeypatch
) -> None:
    """codex round-1 MEDIUM: offline strips whitespace keyword and
    treats ``"   "`` as no filter (``catalogue._iter_matches`` at
    ``catalogue.py``). The live path must match: a whitespace-only
    keyword should NOT forward ``contains=["   "]`` to the SDK
    (which would behave as OR / no-op per the empirical SDK note
    historically in ``_describe_kwargs``)."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    captured_kwargs: dict = {}

    def fake_describe(**kwargs):
        captured_kwargs.update(kwargs)
        return _fake_live_describe_response()

    _install_fake_cmem_sdk(monkeypatch, fake_describe)

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    await backend.search({"keyword": "   ", "live": True})

    # contains must not be forwarded for whitespace-only keyword.
    assert "contains" not in captured_kwargs
