"""T-CMEMS-GET-003: ``CmemsBackend.get_files`` happy path.

Mocked-SDK tests that cover:
- N data files → N-descriptor envelope + manifest + one provenance sidecar.
- second call → cache hit (no SDK call, ``is_existing=True``).
- empty match → ``NotFoundError`` with a recovery hint.
- two concurrent calls for the same params → SDK called once (the
  ``_async_submit_lock`` from the existing ``submit`` flow serialises
  both, and the second observes the cache).

Cancellation, confirmation gate, and async-mode are out of scope for
this task (sub-plan T-CMEMS-GET-003 acceptance: happy path only).
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

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
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = CacheManager(
        cache_directory=cache_dir,
        persistence=persistence,
        size_limit_bytes=50 * 1024 * 1024,
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


def _params() -> dict[str, Any]:
    """Minimal valid CmemsGetRequest payload for a sparse dataset."""
    return dict(
        dataset_id="cmems_obs-ins_glo_phy-temp-sal_my_easycora_irr",
        filter="*1990*",
    )


def _params_confirmed() -> dict[str, Any]:
    """``_params`` plus ``__options.confirmed=True``. T-CMEMS-GET-004
    test fakes don't surface a precise dry-run size, so the gate
    (sub-plan D3) treats unknown-size as always-gate; tests that
    don't exercise the gate itself pass ``confirmed=True`` so the
    download path runs."""
    return dict(_params(), __options={"confirmed": True})


def _install_fake(
    monkeypatch,
    *,
    files_to_write: dict[str, bytes] | None = None,
) -> types.ModuleType:
    """Install a fake ``copernicusmarine`` whose ``get`` writes the
    given files into ``output_directory``. ``None``/empty → write
    nothing (simulates an empty-match)."""
    mod = types.ModuleType("copernicusmarine")
    files_to_write = files_to_write or {}

    def _get_wrapper(**kwargs):
        if kwargs.get("dry_run"):
            # T-CMEMS-GET-004: tests that don't exercise the gate use
            # this fake. Surface no size fields so the estimate falls
            # back to ``epistemic_status="approximate"`` — sub-plan D3
            # always-gates that case, which is why these tests pass
            # ``confirmed=True`` via ``_params_confirmed`` to bypass.
            return types.SimpleNamespace(status="DRY_RUN")
        outdir = Path(kwargs["output_directory"])
        outdir.mkdir(parents=True, exist_ok=True)
        for relpath, content in files_to_write.items():
            target = outdir / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return types.SimpleNamespace(status="OK")

    mod.get = _get_wrapper  # type: ignore[attr-defined]
    mod.subset = lambda **kw: types.SimpleNamespace(status="DRY_RUN")  # type: ignore[attr-defined]
    mod.describe = lambda **kw: {"products": []}  # type: ignore[attr-defined]

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


@pytest.mark.asyncio
async def test_get_files_happy_path(foundation, monkeypatch) -> None:
    """Two fake .nc files → envelope with two descriptors, one
    manifest cache entry, workflow row ``successful``."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    _install_fake(
        monkeypatch,
        files_to_write={"a_1990.nc": b"data-a", "b_1990.nc": b"data-b"},
    )

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    result = await backend.get_files(_params_confirmed())

    assert result["status"] == "successful"
    assert result["cache_hit"] is False
    assert result["is_existing"] is False
    assert result["request_id"]
    assert result["cache_key"].startswith("cmems:get:")
    assert result["mode"] == "offline"

    files = result["result"]["files"]
    assert len(files) == 2
    rels = {Path(f["filepath"]).name for f in files}
    assert rels == {"a_1990.nc", "b_1990.nc"}
    for f in files:
        assert Path(f["filepath"]).exists()
        assert f["uri"].startswith("copernicus://files/")
        assert "size_bytes" in f["metadata"]

    # Bundle-level provenance reference (one provenance record per
    # ``get`` call, parallel to how ``submit`` populates ``provenance``
    # inside ``result`` — see ``_success_response`` for the subset
    # shape).
    assert "reference" in result["result"]["provenance"]
    assert result["result"]["provenance"]["reference"].startswith(
        "copernicus://provenance/"
    )

    # Workflow row settled cleanly.
    wf = await foundation.persistence.fetch_workflow(result["request_id"])
    assert wf is not None
    assert wf["status"] == "successful"
    assert wf["operation"] == "get"


@pytest.mark.asyncio
async def test_get_files_cache_hit_second_call(foundation, monkeypatch) -> None:
    """Second call with identical params → no second SDK call; envelope
    returns from the manifest with ``cache_hit=True``."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    sdk_calls: list[dict[str, Any]] = []

    def _capture_get(**kwargs):
        sdk_calls.append(kwargs)
        if kwargs.get("dry_run"):
            return types.SimpleNamespace(status="DRY_RUN")
        outdir = Path(kwargs["output_directory"])
        (outdir / "f1.nc").write_bytes(b"x")
        return types.SimpleNamespace(status="OK")

    mod = _install_fake(monkeypatch, files_to_write={"f1.nc": b"x"})
    mod.get = _capture_get  # type: ignore[attr-defined]

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    first = await backend.get_files(_params_confirmed())
    # Two SDK calls: dry_run (estimate) + actual get.
    assert len([c for c in sdk_calls if not c.get("dry_run")]) == 1

    second = await backend.get_files(_params_confirmed())
    assert second["cache_hit"] is True
    assert second["is_existing"] is True
    assert second["cache_key"] == first["cache_key"]
    # No further real SDK call on the cached path (estimate is also
    # skipped — cache hit short-circuits before the gate).
    assert len([c for c in sdk_calls if not c.get("dry_run")]) == 1
    assert len(second["result"]["files"]) == 1


@pytest.mark.asyncio
async def test_get_files_empty_match_raises_not_found(
    foundation, monkeypatch
) -> None:
    """SDK produces zero files (empty match) → ``NotFoundError`` with a
    ``modify_request_parameters`` recovery hint. No cache entry, no
    successful workflow row."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import NotFoundError

    _install_fake(monkeypatch, files_to_write={})  # SDK writes nothing

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(NotFoundError) as exc_info:
        await backend.get_files(_params_confirmed())

    record = exc_info.value.error_record
    assert record.recovery_action == "modify_request_parameters"
    assert record.next_action_hint  # non-empty


@pytest.mark.asyncio
async def test_get_files_force_refresh_does_not_lose_data(
    foundation, monkeypatch
) -> None:
    """cr round-1 HIGH: when ``force_refresh=True`` is passed AND the
    bundle's deterministic on-disk path (sha256 of cache_key) already
    exists, the pre-move ``rmtree`` would race with
    ``store_manifest``'s own idempotent-overwrite ``rmtree`` and leave
    the cache entry pointing at an empty directory. Files must remain
    readable after the second call."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    sdk_calls: list[dict[str, Any]] = []

    def _twice_writer(**kwargs):
        sdk_calls.append(kwargs)
        if kwargs.get("dry_run"):
            return types.SimpleNamespace(status="DRY_RUN")
        outdir = Path(kwargs["output_directory"])
        real_count = len([c for c in sdk_calls if not c.get("dry_run")])
        (outdir / "v.nc").write_bytes(b"version-" + str(real_count).encode())
        return types.SimpleNamespace(status="OK")

    mod = _install_fake(monkeypatch, files_to_write={})
    mod.get = _twice_writer  # type: ignore[attr-defined]

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    first = await backend.get_files(_params_confirmed())

    p2 = dict(_params(), __options={"force_refresh": True, "confirmed": True})
    second = await backend.get_files(p2)

    # Two real (non-dry_run) SDK calls — force_refresh re-runs the
    # download even though the cache had the bundle.
    real_calls = [c for c in sdk_calls if not c.get("dry_run")]
    assert len(real_calls) == 2
    assert second["cache_hit"] is False
    # The post-replace file must exist on disk.
    for f in second["result"]["files"]:
        path = Path(f["filepath"])
        assert path.exists(), f"file {path} disappeared after force_refresh"
    # ... and cache_key is unchanged across the two calls (same params).
    assert first["cache_key"] == second["cache_key"]


@pytest.mark.asyncio
async def test_get_files_cache_hit_recovers_from_corrupt_manifest(
    foundation, monkeypatch
) -> None:
    """cr+codex round-1 MEDIUM: a corrupt ``manifest.json`` at the
    cache-hit path used to propagate ``json.JSONDecodeError`` and wedge
    the cache key. We must instead invalidate the bundle and let the
    download path repopulate."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    sdk_calls: list[dict[str, Any]] = []

    def _capture(**kwargs):
        sdk_calls.append(kwargs)
        if kwargs.get("dry_run"):
            return types.SimpleNamespace(status="DRY_RUN")
        outdir = Path(kwargs["output_directory"])
        (outdir / "x.nc").write_bytes(b"x")
        return types.SimpleNamespace(status="OK")

    mod = _install_fake(monkeypatch, files_to_write={})
    mod.get = _capture  # type: ignore[attr-defined]

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    first = await backend.get_files(_params_confirmed())

    def real_calls() -> list[dict[str, Any]]:
        return [c for c in sdk_calls if not c.get("dry_run")]

    assert len(real_calls()) == 1

    # Tamper with the manifest to be invalid JSON.
    manifest = Path(first["result"]["files"][0]["filepath"]).parent / "manifest.json"
    manifest.write_text("{ not valid json")

    # Second call: cache-hit path reads the corrupt manifest, must
    # invalidate the entry and fall through to a fresh download.
    second = await backend.get_files(_params_confirmed())
    assert len(real_calls()) == 2, "corrupt manifest should trigger re-download"
    assert second["cache_hit"] is False
    assert len(second["result"]["files"]) == 1


@pytest.mark.asyncio
async def test_get_files_empty_match_ignores_sdk_dotfiles(
    foundation, monkeypatch
) -> None:
    """cr round-1 MEDIUM: ``staging.rglob('*')`` includes dot-prefixed
    files. If the SDK drops a session-state dotfile but no data
    files, the empty-match guard would silently slip through and
    ``build_manifest`` would raise a generic ``ValueError``. The
    empty-match check must skip the same files that
    ``_iter_data_files`` skips."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import NotFoundError

    def _dotfile_only(**kwargs):
        if kwargs.get("dry_run"):
            return types.SimpleNamespace(status="DRY_RUN")
        outdir = Path(kwargs["output_directory"])
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / ".session_state").write_bytes(b"sdk-internal")
        return types.SimpleNamespace(status="OK")

    mod = _install_fake(monkeypatch, files_to_write={})
    mod.get = _dotfile_only  # type: ignore[attr-defined]

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(NotFoundError):
        await backend.get_files(_params_confirmed())


@pytest.mark.asyncio
async def test_get_files_missing_sdk_does_not_leave_running_row(
    foundation, monkeypatch
) -> None:
    """cr round-2 MEDIUM: a missing ``copernicusmarine`` extra used
    to leave the workflow row stuck in ``running`` because the SDK
    loader ran AFTER ``record_workflow``. The fix loads the SDK
    BEFORE the row is recorded so a missing extra never records a
    row."""
    from copernicus_mcp.backends.cmems import backend as backend_mod
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import BackendError
    from copernicus_mcp.errors.records import build_error_record

    _install_fake(monkeypatch, files_to_write={})  # creds check passes

    def _no_sdk():
        raise BackendError(
            "copernicusmarine extra not installed",
            record=build_error_record(
                "BackendError",
                message="copernicusmarine extra not installed",
                error_subclass="missing_extra",
                recovery_action="report_to_administrator",
            ),
        )

    monkeypatch.setattr(backend_mod, "_load_copernicusmarine", _no_sdk)

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(BackendError):
        await backend.get_files(_params())

    # The cache_key for this request would have been derived; if a
    # workflow row got recorded it'd be findable by cache_key. With
    # the fix the loader fails before record_workflow, so no row
    # exists.
    from copernicus_mcp.data_model.coordinator import DataModelCoordinator
    from copernicus_mcp.data_model.schemas_cmems import CmemsGetRequest

    coord = DataModelCoordinator(persistence=foundation.persistence)
    expected_key = coord.cache_key_for_get(CmemsGetRequest(**_params()))
    row = await foundation.persistence.lookup_workflow_by_cache_key(expected_key)
    assert row is None, f"workflow row should not exist; got {row}"


# ---------------------------------------------------------------------------
# T-CMEMS-GET-005: cancellation parity + commit-point shielding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_files_cancel_before_commit_rolls_back(
    foundation, monkeypatch
) -> None:
    """Cancellation BEFORE the commit point (``store_manifest``) must
    roll back cleanly: workflow row → ``cancelled``, no cache entry,
    no orphan files on disk. Mirrors subset's T-039 contract."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    started = {"path": None}

    def slow_get(**kwargs):
        if kwargs.get("dry_run"):
            return types.SimpleNamespace(
                status="DRY_RUN", file_size=0.001, total_size=0.001
            )
        outdir = Path(kwargs["output_directory"])
        outdir.mkdir(parents=True, exist_ok=True)
        partial = outdir / "partial.nc"
        partial.write_bytes(b"partial-bytes")
        started["path"] = partial
        # Block long enough for the test to cancel — the thread will
        # finish in the background (the project conventions gotcha #8), but the
        # awaiter sees CancelledError immediately.
        import time

        time.sleep(2.0)
        return types.SimpleNamespace(status="OK")

    mod = _install_fake(monkeypatch, files_to_write={})
    mod.get = slow_get  # type: ignore[attr-defined]

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    task = asyncio.create_task(backend.get_files(_params_confirmed()))
    # Wait until the thread has at least started — that confirms the
    # cancel lands after ``record_workflow`` but before
    # ``store_manifest``.
    for _ in range(50):
        if started["path"] is not None and started["path"].exists():
            break
        await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # No cache entry for the bundle.
    entries = [
        e async for e in foundation.persistence.iter_cache_entries_by_namespace("file")
    ]
    assert not any("cmems:get:" in e["key"] for e in entries)

    # Workflow row settled to ``cancelled``.
    from copernicus_mcp.data_model.coordinator import DataModelCoordinator
    from copernicus_mcp.data_model.schemas_cmems import CmemsGetRequest

    coord = DataModelCoordinator(persistence=foundation.persistence)
    cache_key = coord.cache_key_for_get(CmemsGetRequest(**_params()))
    wf = await foundation.persistence.lookup_workflow_by_cache_key(cache_key)
    assert wf is not None
    assert wf["status"] == "cancelled"


@pytest.mark.asyncio
async def test_get_files_cancel_after_commit_still_succeeds(
    foundation, monkeypatch
) -> None:
    """Cancellation AFTER the commit point must NOT undo the commit.
    The shielded section (``store_manifest`` + provenance + workflow
    finalize) completes atomically. The awaiter sees
    ``CancelledError`` but the cache entry and ``successful`` row
    survive."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    _install_fake_with_get_estimate(
        monkeypatch,
        estimate_bytes=1024,  # under threshold → no gate
        files_to_write={"data.nc": b"committed-bytes"},
    )

    # codex round-1 LOW: replace sleep-based timing with an explicit
    # event. ``post_commit_started`` fires the moment we know the
    # cache row has been published, so the test cancel lands
    # deterministically on the post-commit side.
    post_commit_started = asyncio.Event()
    original = foundation.provenance.record_successful_retrieve

    async def signalling_provenance(**kwargs):
        # store_manifest already returned by the time provenance
        # runs — the commit point has passed.
        post_commit_started.set()
        await asyncio.sleep(0.2)  # window for the cancel to land
        return await original(**kwargs)

    monkeypatch.setattr(
        foundation.provenance, "record_successful_retrieve", signalling_provenance
    )

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    task = asyncio.create_task(backend.get_files(_params_confirmed()))
    await post_commit_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Cache entry IS committed (commit point passed).
    entries = [
        e async for e in foundation.persistence.iter_cache_entries_by_namespace("file")
    ]
    bundle_entries = [e for e in entries if "cmems:get:" in e["key"]]
    assert len(bundle_entries) == 1

    # Workflow row IS settled to ``successful`` (shielded finalize).
    from copernicus_mcp.data_model.coordinator import DataModelCoordinator
    from copernicus_mcp.data_model.schemas_cmems import CmemsGetRequest

    coord = DataModelCoordinator(persistence=foundation.persistence)
    cache_key = coord.cache_key_for_get(CmemsGetRequest(**_params()))
    wf = await foundation.persistence.lookup_workflow_by_cache_key(cache_key)
    assert wf is not None
    assert wf["status"] == "successful"


@pytest.mark.asyncio
async def test_get_files_wrong_format_hint_points_at_subset(
    foundation, monkeypatch
) -> None:
    """T-CMEMS-GET-007: when ``marine.get`` raises
    ``WrongFormatRequested`` (grid dataset that only supports the
    Zarr subset path), the recovery hint must point at
    ``marine_subset_dataset`` — NOT ``marine_get_files`` (which is
    the tool that just failed)."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import ValidationError

    mod = _install_fake(monkeypatch, files_to_write={})

    def wrong_format(**kwargs):
        if kwargs.get("dry_run"):
            return types.SimpleNamespace(
                status="DRY_RUN", file_size=0.001, total_size=0.001
            )
        raise mod.WrongFormatRequested("not a native-file dataset")

    mod.get = wrong_format  # type: ignore[attr-defined]

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(ValidationError) as exc_info:
        await backend.get_files(_params_confirmed())
    record = exc_info.value.error_record
    assert "marine_subset_dataset" in (record.next_action_hint or "")
    # And explicitly NOT the circular suggestion.
    assert "marine_get_files" not in (record.next_action_hint or "")


@pytest.mark.asyncio
async def test_get_files_post_commit_failure_preserves_bundle(
    foundation, monkeypatch
) -> None:
    """cr+codex round-1 MEDIUM: a failure AFTER ``store_manifest``
    (e.g. ``read_manifest`` race, finalize SQL error) must NOT delete
    the published bundle. Prior code keyed the rollback decision off
    ``commit_task.exception() is None`` which misclassified any
    post-commit failure as pre-commit and tore down committed data.

    Inject a failure in the workflow-finalize SQL call (a no-op
    that raises). The cache entry must still be present after the
    error propagates."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    _install_fake_with_get_estimate(
        monkeypatch,
        estimate_bytes=1024,
        files_to_write={"data.nc": b"committed-bytes"},
    )

    # Wrap ``update_workflow_status`` so the finalize step raises
    # AFTER ``store_manifest`` has already committed.
    original = foundation.persistence.update_workflow_status

    async def failing_finalize(request_id: str, status: str):
        if status == "successful":
            raise RuntimeError("simulated finalize failure")
        return await original(request_id, status)

    monkeypatch.setattr(
        foundation.persistence, "update_workflow_status", failing_finalize
    )

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    # cr round-2 MEDIUM: the raw RuntimeError from the shielded
    # coro must NOT escape ``get_files``. It is wrapped into the
    # canonical ``BackendError`` (the project conventions).
    from copernicus_mcp.errors import BackendError

    with pytest.raises(BackendError, match="post-commit finalize failed") as exc_info:
        await backend.get_files(_params_confirmed())
    # codex round-3 LOW: pin the taxonomy so a future refactor can't
    # silently drift the error_subclass / recovery_action.
    assert exc_info.value.error_record.error_subclass == "post_commit_finalize_failure"
    assert exc_info.value.error_record.recovery_action == "report_to_administrator"

    entries = [
        e async for e in foundation.persistence.iter_cache_entries_by_namespace("file")
    ]
    bundle_entries = [e for e in entries if "cmems:get:" in e["key"]]
    assert len(bundle_entries) == 1, (
        "post-commit failure must not delete the committed bundle"
    )
    bundle_path = Path(bundle_entries[0]["file_path"]).parent
    assert (bundle_path / "data.nc").exists()

    # cr+codex round-2 MEDIUM: the workflow row must NOT stay
    # ``running``. The bundle is committed → ``successful`` is the
    # honest terminal state.
    from copernicus_mcp.data_model.coordinator import DataModelCoordinator
    from copernicus_mcp.data_model.schemas_cmems import CmemsGetRequest

    coord = DataModelCoordinator(persistence=foundation.persistence)
    cache_key = coord.cache_key_for_get(CmemsGetRequest(**_params()))
    wf = await foundation.persistence.lookup_workflow_by_cache_key(cache_key)
    assert wf is not None
    assert wf["status"] == "successful"


@pytest.mark.asyncio
async def test_get_files_retry_after_pre_commit_cancel_succeeds(
    foundation, monkeypatch
) -> None:
    """A pre-commit cancellation leaves no stale subdirectory. A
    retry of the same cache_key must therefore produce a fresh
    bundle without picking up partials from the cancelled attempt."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    call_count = {"n": 0}

    def behaviour(**kwargs):
        if kwargs.get("dry_run"):
            return types.SimpleNamespace(
                status="DRY_RUN", file_size=0.001, total_size=0.001
            )
        call_count["n"] += 1
        outdir = Path(kwargs["output_directory"])
        outdir.mkdir(parents=True, exist_ok=True)
        if call_count["n"] == 1:
            # First call: write partial then block for cancel.
            (outdir / "partial.nc").write_bytes(b"partial")
            import time

            time.sleep(2.0)
        else:
            (outdir / "fresh.nc").write_bytes(b"fresh")
        return types.SimpleNamespace(status="OK")

    mod = _install_fake(monkeypatch, files_to_write={})
    mod.get = behaviour  # type: ignore[attr-defined]

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    task = asyncio.create_task(backend.get_files(_params_confirmed()))
    await asyncio.sleep(0.3)  # let partial appear
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Second call: must succeed with the fresh bundle.
    result = await backend.get_files(_params_confirmed())
    assert result["status"] == "successful"
    assert result["cache_hit"] is False
    files = result["result"]["files"]
    assert len(files) == 1
    assert "fresh.nc" in Path(files[0]["filepath"]).name
    # No leaked "partial.nc" from the cancelled attempt.
    bundle_dir = Path(files[0]["filepath"]).parent
    assert not (bundle_dir / "partial.nc").exists()


# ---------------------------------------------------------------------------
# T-CMEMS-GET-004: estimate + confirmation gate for `get`
# ---------------------------------------------------------------------------


def _install_fake_with_get_estimate(
    monkeypatch,
    *,
    estimate_bytes: int | None,
    files_to_write: dict[str, bytes] | None = None,
) -> types.ModuleType:
    """Install a fake ``copernicusmarine`` whose ``get(dry_run=True)``
    returns a precise size when ``estimate_bytes`` is given, or a
    response without size fields when ``None`` (simulates an SDK
    version that doesn't surface ``dry_run`` totals)."""
    mod = types.ModuleType("copernicusmarine")
    files_to_write = files_to_write or {}

    def _get_wrapper(**kwargs):
        if kwargs.get("dry_run"):
            if estimate_bytes is None:
                # No size fields — gate must fire on epistemic
                # "approximate" / "unknown".
                return types.SimpleNamespace(status="DRY_RUN")
            # Report bytes; ``_map_get_estimate_response`` converts.
            return types.SimpleNamespace(
                status="DRY_RUN",
                file_size=estimate_bytes / (1024 * 1024),
                total_size=estimate_bytes / (1024 * 1024),
            )
        outdir = Path(kwargs["output_directory"])
        outdir.mkdir(parents=True, exist_ok=True)
        for relpath, content in files_to_write.items():
            target = outdir / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return types.SimpleNamespace(status="OK")

    mod.get = _get_wrapper  # type: ignore[attr-defined]
    mod.subset = lambda **kw: types.SimpleNamespace(status="DRY_RUN")  # type: ignore[attr-defined]
    mod.describe = lambda **kw: {"products": []}  # type: ignore[attr-defined]

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


@pytest.mark.asyncio
async def test_get_estimate_precise_small(foundation, monkeypatch) -> None:
    """``estimate`` dispatches on params shape — a get-shape request
    runs ``marine.get(dry_run=True)`` and returns a precise size."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    _install_fake_with_get_estimate(
        monkeypatch, estimate_bytes=10 * 1024 * 1024  # 10 MB
    )
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    estimate = await backend.estimate(_params())
    assert estimate["epistemic_status"] == "precise"
    assert estimate["estimated_size_bytes"] == 10 * 1024 * 1024


@pytest.mark.asyncio
async def test_get_estimate_approximate_when_no_size(
    foundation, monkeypatch
) -> None:
    """SDK dry_run that doesn't surface size → ``epistemic_status``
    falls back to ``approximate`` (sub-plan D3) so the gate fires
    even on a zero-byte estimate."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    _install_fake_with_get_estimate(monkeypatch, estimate_bytes=None)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    estimate = await backend.estimate(_params())
    assert estimate["epistemic_status"] == "approximate"


@pytest.mark.asyncio
async def test_get_files_gate_fires_when_over_threshold(
    foundation, monkeypatch
) -> None:
    """Precise estimate larger than
    ``cmems_per_request_size_warning_gb`` raises
    ``ConfirmationRequired`` rather than downloading."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.workflow.confirmation import ConfirmationRequired

    # 5 GB precise (default threshold = 1 GB).
    huge_bytes = 5 * 1_000_000_000
    _install_fake_with_get_estimate(
        monkeypatch,
        estimate_bytes=huge_bytes,
        files_to_write={"would_have.nc": b"never written"},
    )
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(ConfirmationRequired) as exc_info:
        await backend.get_files(_params())
    payload = exc_info.value.payload
    assert payload["confirmation_required"] is True
    assert payload["estimated_cost"]["estimated_size_bytes"] == huge_bytes
    assert payload["context"]["tool_name"] == "marine_get_files"


@pytest.mark.asyncio
async def test_get_files_gate_fires_when_size_unknown(
    foundation, monkeypatch
) -> None:
    """Approximate estimate (SDK didn't report size) always gates,
    regardless of bytes — sub-plan D3 explicitly drops the
    'small-get-skips-gate' rule that v1 had."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.workflow.confirmation import ConfirmationRequired

    _install_fake_with_get_estimate(
        monkeypatch,
        estimate_bytes=None,
        files_to_write={"would_have.nc": b"x"},
    )
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(ConfirmationRequired):
        await backend.get_files(_params())


@pytest.mark.asyncio
async def test_get_files_gate_bypassed_when_confirmed(
    foundation, monkeypatch
) -> None:
    """Passing ``options.confirmed=True`` skips the gate. Mirrors the
    subset flow."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    _install_fake_with_get_estimate(
        monkeypatch,
        estimate_bytes=5 * 1_000_000_000,  # over threshold
        files_to_write={"data.nc": b"actual-bytes"},
    )
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    p = dict(_params(), __options={"confirmed": True})
    result = await backend.get_files(p)
    assert result["status"] == "successful"
    assert result["cache_hit"] is False


@pytest.mark.asyncio
async def test_get_files_gate_passes_when_under_threshold(
    foundation, monkeypatch
) -> None:
    """Small precise estimate (well under threshold) — no gate,
    download proceeds."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    _install_fake_with_get_estimate(
        monkeypatch,
        estimate_bytes=1024,  # 1 KB
        files_to_write={"tiny.nc": b"small"},
    )
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    result = await backend.get_files(_params())
    assert result["status"] == "successful"


@pytest.mark.asyncio
async def test_get_files_concurrent_calls_serialised(
    foundation, monkeypatch
) -> None:
    """Two concurrent ``get_files`` calls with identical params: only
    one SDK invocation should happen; the second observes the manifest
    cache after the lock is released."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    sdk_calls: list[dict[str, Any]] = []

    async def _slow_get(**kwargs):
        # Give the asyncio scheduler a chance to start the second call
        # so the serialisation test is meaningful.
        await asyncio.sleep(0.05)
        sdk_calls.append(kwargs)
        outdir = Path(kwargs["output_directory"])
        (outdir / "only.nc").write_bytes(b"y")

    def _sync_wrapper(**kwargs):
        # ``copernicusmarine.get`` is sync; the backend wraps it in
        # ``asyncio.to_thread``. We just block briefly in the thread.
        import time

        if kwargs.get("dry_run"):
            sdk_calls.append(kwargs)
            return types.SimpleNamespace(status="DRY_RUN")
        time.sleep(0.05)
        sdk_calls.append(kwargs)
        outdir = Path(kwargs["output_directory"])
        (outdir / "only.nc").write_bytes(b"y")
        return types.SimpleNamespace(status="OK")

    mod = _install_fake(monkeypatch, files_to_write={})
    mod.get = _sync_wrapper  # type: ignore[attr-defined]

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    a, b = await asyncio.gather(
        backend.get_files(_params_confirmed()),
        backend.get_files(_params_confirmed()),
    )
    real_calls = [c for c in sdk_calls if not c.get("dry_run")]
    assert len(real_calls) == 1
    # Exactly one of the calls is a fresh download, the other is the
    # cached read.
    cache_hits = sorted([a["cache_hit"], b["cache_hit"]])
    assert cache_hits == [False, True]
    assert a["cache_key"] == b["cache_key"]
