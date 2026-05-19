"""Regression tests added during the T-038 release-checklist Pass C.

Each test corresponds to one or more checklist items:

- ``test_describe_with_missing_credentials_returns_auth_error`` — FA-3 part. (Was a search-based test pre-T-CMEMS-CAT-003; search is now offline.)
- ``test_subset_coverage_unavailable_mapping`` — FA-3 part.
- ``test_subset_cancel_cleans_up_partial_file_and_marks_workflow`` — FA-5.
- ``test_concurrent_searches_against_one_orchestrator`` — NFA-2.
- ``test_check_status_reconciles_stale_running_row`` — NFA-3.
- ``test_lru_eviction_removes_file_sidecar_and_row_in_lockstep`` — NFA-6.

These are pure unit tests: a fake ``copernicusmarine`` is installed in
``sys.modules`` so no network is involved.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

# --- shared fixtures (mirrored from tests/unit/test_cmems_subset.py) ---------


def _make_foundation(tmp_path: Path, *, cache_size_bytes: int = 10 * 1024 * 1024):
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
        size_limit_bytes=cache_size_bytes,
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


@pytest_asyncio.fixture
async def tiny_cache_foundation(tmp_path: Path):
    """Cache sized for two ~600-byte entries — second store evicts the first."""
    found, persistence = _make_foundation(tmp_path, cache_size_bytes=1000)
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


def _subset_params() -> dict[str, Any]:
    return dict(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        dataset_version="202411",
        variables=["thetao"],
        minimum_longitude=-1.0,
        maximum_longitude=1.0,
        minimum_latitude=0.0,
        maximum_latitude=1.0,
        minimum_depth=0.0,
        maximum_depth=10.0,
        start_datetime="2024-01-01T00:00:00Z",
        end_datetime="2024-01-02T00:00:00Z",
    )


def _estimate_response(transfer_mb: float = 0.5):
    return types.SimpleNamespace(
        file_size=transfer_mb,
        data_transfer_size=transfer_mb,
        status="DRY_RUN",
        message="dry-run",
        variables=["thetao"],
        service="arco-geo-series",
    )


def _install_fake_module(
    monkeypatch,
    *,
    subset_fn,
    describe_fn=None,
    write_bytes: bytes = b"netcdf-content",
):
    mod = types.ModuleType("copernicusmarine")

    def _subset_wrapper(**kwargs):
        if not kwargs.get("dry_run"):
            outdir = Path(kwargs["output_directory"])
            fname = kwargs["output_filename"]
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / fname).write_bytes(write_bytes)
        return subset_fn(**kwargs)

    mod.subset = _subset_wrapper  # type: ignore[attr-defined]
    mod.describe = describe_fn or (lambda **kw: {"products": []})  # type: ignore[attr-defined]

    class LoginError(Exception):
        pass

    class DatasetNotFound(Exception):
        pass

    class WrongFormatRequested(Exception):
        pass

    mod.LoginError = LoginError  # type: ignore[attr-defined]
    mod.DatasetNotFound = DatasetNotFound  # type: ignore[attr-defined]
    mod.WrongFormatRequested = WrongFormatRequested  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "copernicusmarine", mod)
    return mod


# --- FA-3: AuthError when credentials missing --------------------------------


@pytest.mark.asyncio
async def test_describe_with_missing_credentials_returns_auth_error(
    foundation, monkeypatch
) -> None:
    """FA-3 (1/3 error classes): missing credentials → AuthError envelope.

    The orchestrator must surface this as ``{"error": <record>}`` with
    ``error_class="AuthError"`` and ``recovery_action="configure_credentials"``,
    not as an unhandled exception.

    T-CMEMS-CAT-003 moved this assertion from ``search`` to
    ``describe``: search is now offline (reads the bundled catalogue
    snapshot) and no longer requires credentials. ``describe``,
    ``estimate``, and ``subset`` still hit the live SDK and still
    surface ``AuthError`` when credentials are missing; ``describe``
    is the simplest of the three to drive end-to-end from the
    orchestrator.
    """
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.backends.registry import BackendRegistry
    from copernicus_mcp.workflow.orchestrator import WorkflowOrchestrator

    _install_fake_module(monkeypatch, subset_fn=lambda **kw: _estimate_response())
    registry = BackendRegistry()
    registry.register(CmemsBackend(foundation=foundation, credentials=None))
    orch = WorkflowOrchestrator(registry=registry, foundation=foundation)

    envelope = await orch.run(
        backend="cmems",
        operation="describe",
        params={"identifier": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m"},
    )
    assert "error" in envelope, envelope
    record = envelope["error"]
    assert record["error_class"] == "AuthError"
    assert record["recovery_action"] == "configure_credentials"


# --- FA-3: CoverageUnavailableError mapping ----------------------------------


@pytest.mark.asyncio
async def test_subset_coverage_unavailable_mapping(foundation, monkeypatch) -> None:
    """FA-3 (2/3 error classes): SDK error containing coverage-shaped wording
    is mapped to ``CoverageUnavailableError`` with a recovery action."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    def fake(**kwargs):
        if kwargs.get("dry_run"):
            # Estimate raises with a coverage-shaped message — backend's
            # _wrap_subset_exception should detect and map.
            raise ValueError(
                "Some of your subset selection [0.0, 1.0] for the latitude "
                "dimension exceed the dataset coordinates [-89.9, -89.0]"
            )
        return _estimate_response()

    _install_fake_module(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())

    from copernicus_mcp.errors import CoverageUnavailableError

    with pytest.raises(CoverageUnavailableError) as exc_info:
        await backend.estimate(_subset_params())
    record = exc_info.value.error_record
    assert record.recovery_action == "modify_request_parameters"


# --- FA-5: cancellation cleanup ----------------------------------------------


@pytest.mark.asyncio
async def test_subset_cancel_cleans_up_partial_file_and_marks_workflow(
    foundation, monkeypatch
) -> None:
    """FA-5: a cancelled subset leaves the workflow row as ``cancelled`` AND
    removes the partial file from the staging directory.

    Strategy: the fake SDK writes a partial file before raising
    ``CancelledError`` from inside the toolbox call. The backend's
    ``except asyncio.CancelledError`` handler must mark the row cancelled
    and unlink the partial file before re-raising.
    """
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    written_paths: list[Path] = []

    def fake(**kwargs):
        if kwargs.get("dry_run"):
            return _estimate_response()
        # Real call: write a partial file, then raise CancelledError.
        outdir = Path(kwargs["output_directory"])
        fname = kwargs["output_filename"]
        outdir.mkdir(parents=True, exist_ok=True)
        partial = outdir / fname
        partial.write_bytes(b"partial-bytes")
        written_paths.append(partial)
        raise asyncio.CancelledError("simulated cancel mid-download")

    _install_fake_module(monkeypatch, subset_fn=fake, write_bytes=b"never-reached")
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    params = _subset_params()
    params["__options"] = {"confirmed": True}

    with pytest.raises(asyncio.CancelledError):
        await backend.submit(params)

    # The partial file must be gone.
    for p in written_paths:
        assert not p.exists(), f"partial file should be unlinked: {p}"

    # Find the workflow row by cache_key (request_id is internal).
    from copernicus_mcp.data_model.schemas_cmems import CmemsSubsetRequest

    req = CmemsSubsetRequest(**{k: v for k, v in params.items() if k != "__options"})
    cache_key = foundation.data_model.cache_key_for_subset(req)
    row = await foundation.persistence.lookup_workflow_by_cache_key(cache_key)
    assert row is not None
    assert row["status"] == "cancelled", row


# --- NFA-2: concurrent correctness -------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_searches_against_one_orchestrator(
    foundation,
) -> None:
    """NFA-2: 5 concurrent searches share one orchestrator; all 5
    succeed with no SQLite-locked errors and no result-mixing.

    T-CMEMS-CAT-003: search is now offline (bundled catalogue), so
    the test no longer needs a fake SDK with per-keyword synthesised
    datasets — it pins concurrency against the real snapshot.

    What this test asserts (round-1 cr M1 wording fix — the
    keywords below are NOT disjoint: e.g. ``arctic`` is a substring
    of ``antarctic`` and they overlap on 21 records in the current
    snapshot, plus other small overlaps; that's fine and expected):

    1. Every concurrent call returns a successful envelope (no
       SQLite-locked / no exception leaks).
    2. ``total_count > 0`` for each keyword — sanity-check the
       snapshot still contains relevant data.
    3. **No result-mixing**: every record returned for a given
       keyword actually contains that keyword in its searchable
       fields. A bug where one concurrent call's results bled into
       another's would surface here as a record where the asked
       keyword is absent.
    """
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.backends.registry import BackendRegistry
    from copernicus_mcp.workflow.orchestrator import WorkflowOrchestrator

    registry = BackendRegistry()
    registry.register(CmemsBackend(foundation=foundation, credentials=_creds()))
    orch = WorkflowOrchestrator(registry=registry, foundation=foundation)

    keywords = ["arctic", "baltic", "mediterranean", "antarctic", "global"]
    results = await asyncio.gather(
        *[orch.run(backend="cmems", operation="search", params={"keyword": kw}) for kw in keywords]
    )
    for kw, env in zip(keywords, results, strict=True):
        assert "error" not in env, (kw, env)
        payload = env["result"]
        assert payload["total_count"] > 0, (kw, payload)
        # No result-mixing: every returned record actually matches
        # the keyword the request asked for.
        for ds in payload["datasets"]:
            hay = " ".join(
                str(ds.get(f) or "")
                for f in (
                    "dataset_id",
                    "dataset_name",
                    "title",
                    "product_id",
                    "product_title",
                    "description",
                )
            ).lower()
            assert kw in hay, (kw, ds)


@pytest.mark.asyncio
async def test_search_without_credentials_succeeds_through_orchestrator(
    foundation,
) -> None:
    """Round-1 cr LOW-3 (symmetric counterpart to FA-3): search is
    now offline. With ``credentials=None`` the orchestrator should
    surface a successful envelope, NOT an AuthError. Pins the
    behavior flip end-to-end so a future accidental restore of
    ``_check_credentials_or_raise()`` into ``search`` fails at the
    orchestrator boundary too."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.backends.registry import BackendRegistry
    from copernicus_mcp.workflow.orchestrator import WorkflowOrchestrator

    registry = BackendRegistry()
    registry.register(CmemsBackend(foundation=foundation, credentials=None))
    orch = WorkflowOrchestrator(registry=registry, foundation=foundation)

    envelope = await orch.run(
        backend="cmems", operation="search", params={"keyword": "temperature"}
    )
    assert "error" not in envelope, envelope
    assert envelope["result"]["total_count"] > 0
    assert "catalogue_fetched_at" in envelope["result"]


def test_marine_snapshot_ships_in_package_data() -> None:
    """Round-1 cr LOW-2: pin that the bundled ``_data/marine.json``
    is on the importable package tree, so a future Hatch config
    change (or build-backend migration) that silently strips
    non-``.py`` files fails this test rather than ship a broken
    wheel."""
    from importlib.resources import files

    snapshot = files("copernicus_mcp.backends.cmems").joinpath("_data").joinpath("marine.json")
    fetched_at = (
        files("copernicus_mcp.backends.cmems").joinpath("_data").joinpath("fetched_at.json")
    )
    # codex round-1 PR #88 LOW: also pin ``dataset_cards.json``,
    # bundled in T-CMEMS-HIER-002 alongside ``marine.json``.
    dataset_cards = (
        files("copernicus_mcp.backends.cmems").joinpath("_data").joinpath("dataset_cards.json")
    )
    # T-CMEMS-HIER-003: bundled product manifest.
    products = files("copernicus_mcp.backends.cmems").joinpath("_data").joinpath("products.json")
    # T-CMEMS-HIER-004: bundled groups manifest.
    groups = files("copernicus_mcp.backends.cmems").joinpath("_data").joinpath("groups.json")
    assert snapshot.is_file(), f"missing bundled snapshot: {snapshot}"
    assert fetched_at.is_file(), f"missing fetched_at: {fetched_at}"
    assert dataset_cards.is_file(), f"missing dataset_cards: {dataset_cards}"
    assert products.is_file(), f"missing products: {products}"
    assert groups.is_file(), f"missing groups: {groups}"


# --- NFA-3: server-restart resilience ----------------------------------------


@pytest.mark.asyncio
async def test_check_status_reconciles_stale_running_row(foundation, monkeypatch) -> None:
    """NFA-3: a workflow row stuck in ``running`` (e.g. server crashed
    mid-subset) is reconciled to ``failed`` on the next ``check_status``
    call when the corresponding cache file is absent.
    """
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.persistence.protocol import WorkflowRecord

    _install_fake_module(monkeypatch, subset_fn=lambda **kw: _estimate_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())

    request_id = "req-stale-running-12345"
    cache_key = "cmems:submit:does-not-exist:abcdef0123456789"
    record: WorkflowRecord = {
        "request_id": request_id,
        "backend_id": "cmems",
        "operation": "submit",
        "status": "running",
        "cache_key": cache_key,
        "request_json": "{}",
        "response_json": None,
        "error_record_json": None,
        "created_at": "2026-04-28T00:00:00Z",
        "updated_at": "2026-04-28T00:00:00Z",
    }
    await foundation.persistence.record_workflow(record)

    out = await backend.check_status(request_id)
    assert out["status"] == "failed", out

    # Persisted state also flipped, not just the in-memory return.
    row = await foundation.persistence.fetch_workflow(request_id)
    assert row is not None
    assert row["status"] == "failed"


@pytest.mark.asyncio
async def test_check_status_does_not_reconcile_fresh_running_row(foundation, monkeypatch) -> None:
    """NFA-3 staleness guard: a ``running`` row updated < 60 s ago is in-flight,
    not crashed — it must NOT be reconciled to ``failed``. Otherwise an
    external poller hitting the window between ``marine.subset`` returning
    and ``store_file`` completing would see a wrong status."""
    from datetime import UTC, datetime

    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.persistence.protocol import WorkflowRecord

    _install_fake_module(monkeypatch, subset_fn=lambda **kw: _estimate_response())
    backend = CmemsBackend(foundation=foundation, credentials=_creds())

    request_id = "req-fresh-running-67890"
    cache_key = "cmems:submit:does-not-exist:fedcba9876543210"
    now_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    record: WorkflowRecord = {
        "request_id": request_id,
        "backend_id": "cmems",
        "operation": "submit",
        "status": "running",
        "cache_key": cache_key,
        "request_json": "{}",
        "response_json": None,
        "error_record_json": None,
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    await foundation.persistence.record_workflow(record)

    out = await backend.check_status(request_id)
    # Row is fresh (just-now updated); reconcile must skip even though the
    # cache file is absent — submit may still be writing it.
    assert out["status"] == "running", out


# --- NFA-6: LRU eviction lockstep --------------------------------------------


@pytest.mark.asyncio
async def test_lru_eviction_removes_file_sidecar_and_row_in_lockstep(
    tiny_cache_foundation,
) -> None:
    """NFA-6: when ``size_limit_bytes`` is exceeded, the LRU entry's file,
    its sidecar (if present), and its SQLite cache row are all removed —
    no orphan rows, no orphan files, no orphan sidecars.
    """
    foundation = tiny_cache_foundation
    cache = foundation.cache

    # Two ~600-byte payloads; cache limit is 1000 bytes — second store
    # forces eviction of the first.
    payload_a = b"a" * 600
    payload_b = b"b" * 600

    staging = cache.cache_zone_for("cmems") / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    src_a = staging / "src-a.bin"
    src_b = staging / "src-b.bin"
    src_a.write_bytes(payload_a)
    src_b.write_bytes(payload_b)

    path_a = await cache.store_file(
        cache_key="file:ck-a",
        source_path=src_a,
        backend_id="cmems",
        content_type="application/octet-stream",
    )
    # Emulate the provenance sidecar that ProvenanceRecorder would write
    # next to a real subset download.
    sidecar_a = path_a.with_suffix(path_a.suffix + ".provenance.json")
    sidecar_a.write_text('{"placeholder": "A"}')

    path_b = await cache.store_file(
        cache_key="file:ck-b",
        source_path=src_b,
        backend_id="cmems",
        content_type="application/octet-stream",
    )

    # Entry B still present.
    assert path_b.exists()
    assert (await foundation.persistence.lookup_cache_entry("file", "file:ck-b")) is not None

    # Entry A: file + sidecar + SQLite row all removed in lockstep.
    assert not path_a.exists(), "evicted file should be gone"
    assert (await foundation.persistence.lookup_cache_entry("file", "file:ck-a")) is None, (
        "evicted SQLite cache row should be gone"
    )
    assert not sidecar_a.exists(), (
        "evicted entry's provenance sidecar must also be removed; "
        "leftover sidecars misrepresent provenance for absent files"
    )
