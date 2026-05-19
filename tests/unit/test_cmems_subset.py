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


def _params() -> dict[str, Any]:
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


def _make_estimate_response(transfer_mb: float = 0.5):
    return types.SimpleNamespace(
        file_size=transfer_mb,
        data_transfer_size=transfer_mb,
        status="DRY_RUN",
        message="dry-run",
        variables=["thetao"],
        service="arco-geo-series",
    )


def _install_fake(monkeypatch, *, subset_fn, write_bytes: bytes = b"netcdf-content") -> types.ModuleType:
    """Install a fake copernicusmarine that creates the file when subset is called."""
    mod = types.ModuleType("copernicusmarine")

    def _subset_wrapper(**kwargs):
        if not kwargs.get("dry_run"):
            outdir = Path(kwargs["output_directory"])
            fname = kwargs["output_filename"]
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / fname).write_bytes(write_bytes)
        return subset_fn(**kwargs)

    mod.subset = _subset_wrapper  # type: ignore[attr-defined]
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
async def test_submit_happy_path(foundation, monkeypatch) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return _make_estimate_response(0.5)

    _install_fake(monkeypatch, subset_fn=fake, write_bytes=b"hello-netcdf")
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    result = await backend.submit(_params())

    assert result["status"] == "successful"
    assert result["cache_hit"] is False
    assert result["is_existing"] is False
    assert result["request_id"]
    assert result["cache_key"].startswith("cmems:submit:")
    assert "filepath" in result["result"]
    assert Path(result["result"]["filepath"]).exists()
    assert result["result"]["uri"].startswith("copernicus://files/")
    assert "provenance" in result["result"]

    # Workflow row → successful (never "existing_success")
    workflow = await foundation.persistence.fetch_workflow(result["request_id"])
    assert workflow is not None
    assert workflow["status"] == "successful"

    # Toolbox called twice: estimate (dry_run=True) + actual subset.
    dry_run_calls = [c for c in calls if c.get("dry_run") is True]
    real_calls = [c for c in calls if c.get("dry_run") is False]
    assert len(dry_run_calls) == 1
    assert len(real_calls) == 1


@pytest.mark.asyncio
async def test_submit_idempotent_cache_hit(foundation, monkeypatch) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    n = {"calls": 0}

    def fake(**kwargs):
        n["calls"] += 1
        return _make_estimate_response(0.5)

    _install_fake(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    first = await backend.submit(_params())
    second = await backend.submit(_params())

    assert first["status"] == "successful"
    assert second["status"] == "successful"
    assert second["cache_hit"] is True
    assert second["is_existing"] is True
    # Second submit must NOT call the toolbox again.
    real_calls = n["calls"]
    third = await backend.submit(_params())
    assert third["cache_hit"] is True
    assert n["calls"] == real_calls


@pytest.mark.asyncio
async def test_submit_force_refresh_re_downloads(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    n = {"calls": 0}

    def fake(**kwargs):
        n["calls"] += 1
        return _make_estimate_response(0.5)

    _install_fake(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    p = _params()
    await backend.submit(p)
    p2 = dict(p)
    p2["__options"] = {"force_refresh": True}
    out = await backend.submit(p2)
    assert out["cache_hit"] is False, "force_refresh must bypass cache"


@pytest.mark.asyncio
async def test_submit_confirmation_gate(foundation, monkeypatch) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.workflow.confirmation import ConfirmationRequired

    # Estimate returns a HUGE size (10 GB) > threshold (1 GB default).
    n = {"real_calls": 0}

    def fake(**kwargs):
        if not kwargs.get("dry_run"):
            n["real_calls"] += 1
        return _make_estimate_response(10 * 1024)  # 10 GB

    _install_fake(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(ConfirmationRequired) as exc_info:
        await backend.submit(_params())
    assert exc_info.value.payload["confirmation_required"] is True
    assert n["real_calls"] == 0, "must not download before confirmation"


@pytest.mark.asyncio
async def test_submit_confirmed_bypasses_gate(foundation, monkeypatch) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    def fake(**kwargs):
        return _make_estimate_response(10 * 1024)  # 10 GB

    _install_fake(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    p = _params()
    p["__options"] = {"confirmed": True}
    result = await backend.submit(p)
    assert result["status"] == "successful"


@pytest.mark.asyncio
async def test_submit_workflow_row_never_writes_existing_success(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    def fake(**kwargs):
        return _make_estimate_response(0.5)

    _install_fake(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    res1 = await backend.submit(_params())
    res2 = await backend.submit(_params())  # idempotent

    # Both rows readable; status column is "successful" for both
    w1 = await foundation.persistence.fetch_workflow(res1["request_id"])
    w2 = await foundation.persistence.fetch_workflow(res2["request_id"])
    assert w1 is not None and w1["status"] == "successful"
    if w2 is not None:  # cache hit may reuse the original row
        assert w2["status"] == "successful"


@pytest.mark.asyncio
async def test_submit_toolbox_failure_marks_workflow_failed(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import BackendError

    def fake(**kwargs):
        if not kwargs.get("dry_run"):
            raise RuntimeError("kaboom")
        return _make_estimate_response(0.5)

    _install_fake(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(BackendError):
        await backend.submit(_params())

    # The workflow row should reflect failure
    # (we don't know the request_id, but we can iterate by listing all)
    # For now: assert the cache is empty (no spurious entry) by re-submitting
    # — should hit the toolbox again, not return a cache hit.
    n = {"calls": 0}

    def good(**kwargs):
        n["calls"] += 1
        return _make_estimate_response(0.5)

    _install_fake(monkeypatch, subset_fn=good)
    out = await backend.submit(_params())
    assert out["status"] == "successful"
    assert n["calls"] >= 1


@pytest.mark.asyncio
async def test_submit_workflow_failed_on_toolbox_error(
    foundation, monkeypatch
) -> None:
    """L5: assert workflow row status='failed' when toolbox raises."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import BackendError

    captured = {"request_ids": []}
    orig_record = foundation.persistence.record_workflow

    async def spy_record(rec):
        captured["request_ids"].append(rec["request_id"])
        return await orig_record(rec)

    monkeypatch.setattr(foundation.persistence, "record_workflow", spy_record)

    def fake(**kwargs):
        if not kwargs.get("dry_run"):
            raise RuntimeError("kaboom")
        return _make_estimate_response(0.5)

    _install_fake(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(BackendError):
        await backend.submit(_params())

    assert captured["request_ids"], "no workflow row was recorded"
    row = await foundation.persistence.fetch_workflow(captured["request_ids"][-1])
    assert row is not None
    assert row["status"] == "failed"


@pytest.mark.asyncio
async def test_submit_status_transitions_queued_running_successful(
    foundation, monkeypatch
) -> None:
    """L4: queued -> running -> successful sequence verified via spy."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    transitions: list[tuple[str, str]] = []
    orig_record = foundation.persistence.record_workflow
    orig_update = foundation.persistence.update_workflow_status

    async def spy_record(rec):
        transitions.append((rec["request_id"], rec["status"]))
        return await orig_record(rec)

    async def spy_update(req_id, status):
        transitions.append((req_id, status))
        return await orig_update(req_id, status)

    monkeypatch.setattr(foundation.persistence, "record_workflow", spy_record)
    monkeypatch.setattr(foundation.persistence, "update_workflow_status", spy_update)

    def fake(**kwargs):
        return _make_estimate_response(0.5)

    _install_fake(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    res = await backend.submit(_params())

    statuses = [s for (rid, s) in transitions if rid == res["request_id"]]
    assert statuses[0] == "queued"
    assert "running" in statuses
    assert statuses[-1] == "successful"
    # Must never write "existing_success".
    assert "existing_success" not in statuses


@pytest.mark.asyncio
async def test_submit_dataset_id_path_traversal_neutralised(
    foundation, monkeypatch, tmp_path: Path
) -> None:
    """codex T-024 HIGH H1: malicious dataset_id must not escape staging."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    captured: dict[str, Any] = {}

    def fake(**kwargs):
        if not kwargs.get("dry_run"):
            captured["output_directory"] = Path(kwargs["output_directory"])
            captured["output_filename"] = kwargs["output_filename"]
        return _make_estimate_response(0.5)

    _install_fake(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    p = _params()
    # First: ensure the schema rejects characters the slug would also reject.
    p["dataset_id"] = "../../../etc/passwd"
    # Pydantic doesn't reject this string per se (just str), but our slug must.
    res = await backend.submit(p)
    # Filename must not contain path separators.
    assert "/" not in captured["output_filename"]
    assert ".." not in captured["output_filename"].split("_")[0]
    # Filepath must stay inside the cache zone.
    assert "downloads/cmems" in res["result"]["filepath"]


@pytest.mark.asyncio
async def test_submit_passes_credentials_to_toolbox(
    foundation, monkeypatch
) -> None:
    """codex T-024 HIGH H2: explicit credentials must flow into the SDK call."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    received: list[dict[str, Any]] = []

    def fake(**kwargs):
        received.append(dict(kwargs))
        return _make_estimate_response(0.5)

    _install_fake(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    await backend.submit(_params())
    real_call = next(c for c in received if c.get("dry_run") is False)
    assert real_call.get("username") == "u"
    assert real_call.get("password") == "p"


@pytest.mark.asyncio
async def test_submit_credentials_in_options_redacted_everywhere(
    foundation, monkeypatch
) -> None:
    """H1: defence-in-depth — even if a credential sneaks past Pydantic
    extra='forbid' (e.g. via the orchestrator-shaped __options bypass),
    Sanitiser must scrub it before it reaches SQLite or provenance."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    def fake(**kwargs):
        return _make_estimate_response(0.5)

    _install_fake(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    p = _params()
    # __options bypasses Pydantic validation by design (popped before validate).
    p["__options"] = {"password": "leak-me-please"}
    res = await backend.submit(p)
    row = await foundation.persistence.fetch_workflow(res["request_id"])
    assert row is not None
    assert "leak-me-please" not in row["request_json"]


@pytest.mark.asyncio
async def test_submit_cancellation_cleans_up(foundation, monkeypatch) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    captured_outdir = {"path": None}

    def fake(**kwargs):
        if not kwargs.get("dry_run"):
            captured_outdir["path"] = (
                Path(kwargs["output_directory"]) / kwargs["output_filename"]
            )
            # Write a partial file then sleep to be cancellable.
            captured_outdir["path"].parent.mkdir(parents=True, exist_ok=True)
            captured_outdir["path"].write_bytes(b"partial")
            import time
            time.sleep(2.0)
        return _make_estimate_response(0.5)

    _install_fake(monkeypatch, subset_fn=fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())

    task = asyncio.create_task(backend.submit(_params()))
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Partial file should be removed.
    if captured_outdir["path"] is not None:
        assert not captured_outdir["path"].exists()
