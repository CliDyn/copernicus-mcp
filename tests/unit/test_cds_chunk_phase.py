"""Parent-level ``downloading`` phase + per-chunk states (T-CDS-RESIL-005).

A single request exposes ``phase="downloading"`` when its CDS job is done
and only the local transfer remains; a chunked parent aggregated statuses
only, so an ephemeral poller could not see "every remaining step is a local
transfer — hand it to one long-lived ``cds wait``". The parent envelope now
carries ``chunks.downloading`` / ``chunks.retrying`` counts, per-chunk
``phase``/``attempt``, and a parent ``phase="downloading"`` exactly when no
remote-side work is left.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

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


def _fake_creds():
    from copernicus_mcp.auth.resolver import ResolvedCredentials

    return ResolvedCredentials(
        backend="cds",
        source="explicit",
        source_detail="test",
        fields={"key": "abcdef01-2345-6789-abcd-ef0123456789"},
    )


def _with_budget(foundation, **updates):
    budget = foundation.config.budget.model_copy(update=updates)
    config = foundation.config.model_copy(update={"budget": budget})
    return dataclasses.replace(foundation, config=config)


def _splittable_params() -> dict[str, Any]:
    return {
        "dataset_id": "derived-era5-single-levels-daily-statistics",
        "inputs": {
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature"],
            "year": ["2020", "2021", "2022", "2023", "2024"],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
        },
        "__options": {"chunk_by": "year"},
    }


def _patch_costing_by_shape(monkeypatch, *, per_year: float, limit: float) -> None:
    from copernicus_mcp.backends.cds.costing import CostingResult

    async def _fake(dataset_id, inputs, **_kwargs):
        years = inputs.get("year")
        n_years = len(years) if isinstance(years, list) else 1
        return CostingResult(units=per_year * n_years, limit=limit)

    monkeypatch.setattr("copernicus_mcp.backends.cds.backend.fetch_costing", _fake)


def _fake_remote(request_id: str) -> MagicMock:
    remote = MagicMock()
    remote.request_id = request_id
    return remote


def _patch_cdsapi_children(monkeypatch, status_by_request, *, download_gate=None):
    import sys
    import types

    fake_module = types.ModuleType("cdsapi")
    instance = MagicMock()
    counter = {"n": 0}

    def _retrieve(name, request, target):
        counter["n"] += 1
        return _fake_remote(f"child-{counter['n']}")

    instance.retrieve = MagicMock(side_effect=_retrieve)
    inner = MagicMock()

    def _get_remote(request_id):
        rem = MagicMock()
        rem.json = {
            "status": status_by_request.get(request_id, "running"),
            "jobID": request_id,
        }
        return rem

    inner.get_remote = MagicMock(side_effect=_get_remote)
    inner.delete = MagicMock(return_value={"deleted": True})

    def _download_results(request_id, target):
        if download_gate is not None:
            download_gate.wait(timeout=10)
        Path(target).write_bytes(b"GRIB-chunk")
        return target

    inner.download_results = MagicMock(side_effect=_download_results)
    instance.client = inner
    fake_class = MagicMock(return_value=instance)
    fake_module.Client = fake_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cdsapi", fake_module)
    return fake_class, instance


# ---------------------------------------------------------------------------
# pure response-builder behaviour
# ---------------------------------------------------------------------------


def _plan_of_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    return {"granularity": "year", "stopped": False, "chunks": chunks}


def test_response_counts_downloading_and_flags_per_chunk_phase() -> None:
    from copernicus_mcp.backends.cds.backend import _chunk_parent_response

    plan = _plan_of_chunks(
        [
            {"index": 0, "child_request_id": "a"},
            {"index": 1, "child_request_id": "b"},
            {"index": 2, "child_request_id": "c"},
        ]
    )
    out = _chunk_parent_response(
        parent_id="p",
        cache_key="k",
        plan=plan,
        child_status={"a": "successful", "b": "running", "c": "running"},
        status="running",
        phase_by_child={"b": "downloading"},
    )
    assert out["chunks"]["downloading"] == 1
    assert out["per_chunk"][1]["phase"] == "downloading"
    assert "phase" not in out["per_chunk"][2]
    assert "phase" not in out  # chunk c is still remote-side active


def test_response_promotes_parent_phase_when_only_transfers_remain() -> None:
    from copernicus_mcp.backends.cds.backend import _chunk_parent_response

    plan = _plan_of_chunks(
        [
            {"index": 0, "child_request_id": "a"},
            {"index": 1, "child_request_id": "b"},
            {"index": 2, "child_request_id": "c"},
        ]
    )
    out = _chunk_parent_response(
        parent_id="p",
        cache_key="k",
        plan=plan,
        child_status={"a": "successful", "b": "running", "c": "running"},
        status="running",
        phase_by_child={"b": "downloading", "c": "downloading"},
    )
    assert out["phase"] == "downloading"
    assert out["chunks"]["downloading"] == 2


def test_response_phase_blocked_by_unsubmitted_or_retrying_chunk() -> None:
    from copernicus_mcp.backends.cds.backend import _chunk_parent_response

    plan = _plan_of_chunks(
        [
            {"index": 0, "child_request_id": "a"},
            {"index": 1, "child_request_id": None},
        ]
    )
    out = _chunk_parent_response(
        parent_id="p",
        cache_key="k",
        plan=plan,
        child_status={"a": "running"},
        status="running",
        phase_by_child={"a": "downloading"},
    )
    assert "phase" not in out  # a chunk is not yet ordered

    plan = _plan_of_chunks(
        [
            {"index": 0, "child_request_id": "a"},
            {
                "index": 1,
                "child_request_id": "b",
                "retry_pending": True,
                "attempt": 1,
            },
        ]
    )
    out = _chunk_parent_response(
        parent_id="p",
        cache_key="k",
        plan=plan,
        child_status={"a": "running", "b": "failed"},
        status="running",
        phase_by_child={"a": "downloading"},
    )
    assert "phase" not in out  # a retry is still owed
    assert out["chunks"]["retrying"] == 1
    assert out["chunks"]["failed"] == 0
    assert out["per_chunk"][1]["status"] == "retrying"
    assert out["per_chunk"][1]["attempt"] == 1


def test_response_terminal_parent_never_carries_phase() -> None:
    from copernicus_mcp.backends.cds.backend import _chunk_parent_response

    plan = _plan_of_chunks([{"index": 0, "child_request_id": "a"}])
    out = _chunk_parent_response(
        parent_id="p",
        cache_key="k",
        plan=plan,
        child_status={"a": "running"},
        status="failed",
        phase_by_child={"a": "downloading"},
    )
    assert "phase" not in out


# ---------------------------------------------------------------------------
# end-to-end through the real poll path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parent_reports_downloading_phase_end_to_end(
    foundation, monkeypatch
) -> None:
    """All CDS jobs done server-side, transfers gated → the parent poll says
    ``phase="downloading"`` with every chunk counted; releasing the gate lets
    a later poll reach ``successful``."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    gate = threading.Event()
    status: dict[str, str] = {f"child-{i}": "successful" for i in range(1, 6)}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(monkeypatch, status, download_gate=gate)
    backend = CdsBackend(
        foundation=_with_budget(foundation, cds_download_inline_grace_seconds=0.0),
        credentials=_fake_creds(),
    )
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]

    st = await backend.check_status(parent_id)
    assert st["status"] == "running"
    assert st["phase"] == "downloading"
    assert st["chunks"]["downloading"] == 5
    assert all(p["phase"] == "downloading" for p in st["per_chunk"])

    gate.set()
    final = None
    for _ in range(10):
        await asyncio.sleep(0.05)
        final = await backend.check_status(parent_id)
        if final["status"] == "successful":
            break
    assert final is not None and final["status"] == "successful"
    assert "phase" not in final


def test_progress_line_includes_downloading_and_retrying() -> None:
    from copernicus_mcp.cli import _progress_line

    line = _progress_line(
        {
            "progress": {"completed": 3, "total": 12},
            "chunks": {
                "running": 2,
                "queued": 5,
                "failed": 0,
                "downloading": 1,
                "retrying": 1,
            },
        }
    )
    assert line == (
        "parts: 3/12 done (running 2, downloading 1, retrying 1, queued 5)"
    )


# ---------------------------------------------------------------------------
# counts must be disjoint, and a terminal parent must not advertise a retry
# ---------------------------------------------------------------------------


def test_downloading_is_disjoint_from_running() -> None:
    """The per-state counts partition the chunks: a child that is transferring
    is counted under ``downloading`` and NOT also under ``running``, so the
    states sum to ``total``. Otherwise a caller adding them up double-counts."""
    from copernicus_mcp.backends.cds.backend import _chunk_parent_response

    plan = _plan_of_chunks(
        [
            {"index": 0, "child_request_id": "a"},
            {"index": 1, "child_request_id": "b"},
            {"index": 2, "child_request_id": "c"},
        ]
    )
    out = _chunk_parent_response(
        parent_id="p",
        cache_key="k",
        plan=plan,
        child_status={"a": "successful", "b": "running", "c": "running"},
        status="running",
        phase_by_child={"b": "downloading"},
    )
    counts = out["chunks"]
    assert counts["downloading"] == 1
    assert counts["running"] == 1  # only chunk c, which is NOT downloading
    partition = ("successful", "running", "downloading", "retrying",
                 "queued", "failed", "cancelled")
    assert sum(counts[k] for k in partition) == counts["total"]


def test_terminal_parent_does_not_advertise_a_pending_retry() -> None:
    """Once the parent is terminal nothing will ever run the retry, so a chunk
    must not keep reporting ``retrying`` — an agent would wait for a retry that
    is never coming, and ``cds wait`` would print it on a dead job."""
    from copernicus_mcp.backends.cds.backend import _chunk_parent_response

    plan = _plan_of_chunks(
        [
            {"index": 0, "child_request_id": "a", "retry_pending": True, "attempt": 1},
            {"index": 1, "child_request_id": "b"},
        ]
    )
    out = _chunk_parent_response(
        parent_id="p",
        cache_key="k",
        plan=plan,
        child_status={"a": "failed", "b": "failed"},
        status="failed",
    )
    assert out["chunks"]["retrying"] == 0
    assert out["chunks"]["failed"] == 2
    assert out["per_chunk"][0]["status"] == "failed"


def test_progress_line_renders_a_real_producer_payload() -> None:
    """Drive the CLI line from the actual envelope builder rather than a
    hand-written dict, so the test cannot assert on a shape the producer is
    incapable of emitting."""
    from copernicus_mcp.backends.cds.backend import _chunk_parent_response
    from copernicus_mcp.cli import _progress_line

    plan = _plan_of_chunks(
        [
            {"index": 0, "child_request_id": "a"},
            {"index": 1, "child_request_id": "b"},
            {"index": 2, "child_request_id": "c"},
            {"index": 3, "child_request_id": "d", "retry_pending": True},
            {"index": 4, "child_request_id": None},
        ]
    )
    payload = _chunk_parent_response(
        parent_id="p",
        cache_key="k",
        plan=plan,
        child_status={"a": "successful", "b": "running", "c": "running", "d": "failed"},
        status="running",
        phase_by_child={"c": "downloading"},
    )

    assert _progress_line(payload) == (
        "parts: 1/5 done (running 1, downloading 1, retrying 1, queued 1)"
    )
