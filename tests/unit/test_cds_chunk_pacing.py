"""Paced chunk submission — ``cds_chunk_max_inflight`` (T-CDS-RESIL-002).

Field run 31: two 21-child parents fired all children within 24–42 s and hit
CDS's per-user concurrency throttle — 39/42 children failed with the
empty-log signature, disproving the v2 premise "CDS queues any excess
rather than rejecting it". A chunked parent must submit a bounded first
wave and let the poll-driven refill submit the rest as earlier children
reach a terminal state. ``0`` restores the old fan-out-everything
behaviour.
"""

from __future__ import annotations

import dataclasses
import json
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


def _with_max_inflight(foundation, n: int):
    budget = foundation.config.budget.model_copy(
        update={"cds_chunk_max_inflight": n}
    )
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


def _patch_cdsapi_children(monkeypatch, status_by_request):
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
        Path(target).write_bytes(b"GRIB-chunk")
        return target

    inner.download_results = MagicMock(side_effect=_download_results)
    instance.client = inner
    fake_class = MagicMock(return_value=instance)
    fake_module.Client = fake_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cdsapi", fake_module)
    return fake_class, instance


async def _active_children(persistence, parent_id: str) -> int:
    children = await persistence.list_child_workflows(parent_id)
    return sum(1 for c in children if c["status"] in ("queued", "running"))


def test_pacing_knob_default_is_five() -> None:
    from copernicus_mcp.config import ConfigLoader

    config = ConfigLoader().load()
    assert config.budget.cds_chunk_max_inflight == 5


@pytest.mark.asyncio
async def test_first_wave_bounded_by_max_inflight(foundation, monkeypatch) -> None:
    """max_inflight=2, 5-chunk plan → submit fires exactly 2 children; the
    plan still records all 5 chunks (3 not yet ordered)."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(
        foundation=_with_max_inflight(foundation, 2), credentials=_fake_creds()
    )

    out = await backend.submit(_splittable_params())

    assert out["chunked"] is True
    assert out["chunk_count"] == 5
    assert sdk.retrieve.call_count == 2
    parent = await foundation.persistence.fetch_workflow(out["request_id"])
    plan = json.loads(parent["chunk_plan_json"])
    assert len(plan["chunks"]) == 5
    submitted = [c for c in plan["chunks"] if c["child_request_id"]]
    assert len(submitted) == 2


@pytest.mark.asyncio
async def test_refill_keeps_inflight_at_the_bound_until_done(
    foundation, monkeypatch
) -> None:
    """Polls refill freed slots without ever exceeding the bound, and the
    parent completes over successive waves (field acceptance check #1)."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(
        foundation=_with_max_inflight(foundation, 2), credentials=_fake_creds()
    )
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]
    assert await _active_children(foundation.persistence, parent_id) == 2

    # Every child succeeds the moment it is polled (ids are assigned in
    # submission order, so pre-marking all five is safe).
    for i in range(1, 6):
        status[f"child-{i}"] = "successful"

    final = None
    for _ in range(8):
        final = await backend.check_status(parent_id)
        assert await _active_children(foundation.persistence, parent_id) <= 2
        if final["status"] == "successful":
            break

    assert final is not None and final["status"] == "successful"
    assert sdk.retrieve.call_count == 5  # every chunk was eventually ordered


@pytest.mark.asyncio
async def test_refill_submits_only_freed_slots(foundation, monkeypatch) -> None:
    """One completed child frees exactly one slot: the next poll submits one
    more chunk, not the whole remainder."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(
        foundation=_with_max_inflight(foundation, 2), credentials=_fake_creds()
    )
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]
    assert sdk.retrieve.call_count == 2

    status["child-1"] = "successful"  # child-2 keeps running
    st = await backend.check_status(parent_id)

    assert st["status"] == "running"
    assert sdk.retrieve.call_count == 3  # exactly one refill
    assert await _active_children(foundation.persistence, parent_id) == 2


@pytest.mark.asyncio
async def test_max_inflight_zero_restores_full_fanout(foundation, monkeypatch) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(
        foundation=_with_max_inflight(foundation, 0), credentials=_fake_creds()
    )

    out = await backend.submit(_splittable_params())

    assert out["chunk_count"] == 5
    assert sdk.retrieve.call_count == 5


@pytest.mark.asyncio
async def test_twenty_one_part_window_completes_against_a_five_job_account(
    foundation, monkeypatch
) -> None:
    """The headline acceptance criterion, at the size that motivated the work:
    a window too large for one request, split into 21 parts, against an account
    that tolerates about five at a time, COMPLETES — slowly, over as many waves
    as it takes. Run 31 fired all 21 at once and lost the whole retrieval.

    The fake refuses any submit that would put more than five jobs in flight,
    the way the service does, so pacing is what makes this pass rather than an
    assertion about our own bookkeeping.
    """
    from copernicus_mcp.backends.cds.backend import CdsBackend

    years = [str(y) for y in range(2000, 2021)]
    assert len(years) == 21
    params = _splittable_params()
    params["inputs"]["year"] = years

    status: dict[str, str] = {}
    live: set[str] = set()
    refusals: list[str] = []

    def _on_submit(request_id: str) -> None:
        if len(live) >= 5:
            refusals.append(request_id)
            raise AssertionError(
                f"submitted {request_id} with {len(live)} jobs already in flight"
            )
        live.add(request_id)

    def _on_terminal(request_id: str) -> None:
        live.discard(request_id)

    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)

    original_retrieve = sdk.retrieve.side_effect

    def _retrieve(name, request, target):
        remote = original_retrieve(name, request, target)
        _on_submit(remote.request_id)
        return remote

    sdk.retrieve.side_effect = _retrieve

    backend = CdsBackend(
        foundation=_with_max_inflight(foundation, 5), credentials=_fake_creds()
    )
    out = await backend.submit(params)
    parent_id = out["request_id"]
    assert out["chunk_count"] == 21
    assert len(live) == 5  # first wave only

    # Each poll: finish whatever is in flight, then let the backend refill.
    final = None
    for _ in range(30):
        for rid in list(live):
            status[rid] = "successful"
            _on_terminal(rid)
        final = await backend.check_status(parent_id)
        assert len(live) <= 5
        if final["status"] == "successful":
            break

    assert not refusals
    assert final is not None and final["status"] == "successful"
    assert final["progress"] == {"completed": 21, "total": 21}
    assert final["result"]["complete"] is True
    assert len(final["result"]["files"]) == 21
    assert sdk.retrieve.call_count == 21  # every part ordered exactly once
