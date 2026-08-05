"""Cross-process chunk-plan safety in the CDS backend (T-CDS-RESIL-006).

The per-parent lock is process-local; the plan lives in SQLite shared by every
process polling the same parent (MCP server + `cds wait` + ephemeral pollers).
These tests pin the write protocol that makes a duplicate remote submit
impossible rather than merely unlikely:

- a chunk slot is RESERVED by a winning CAS write *before* the remote call;
- a fresh foreign reservation is skipped by other pollers (and counts toward
  the inflight budget); a stale one is reclaimed (the adoption scan dedupes
  if the dead reserver did reach the server);
- a CAS loser re-reads the fresh plan and re-applies its decision instead of
  clobbering a concurrent writer's committed state.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sqlite3
from datetime import UTC, datetime, timedelta
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


def _patch_cdsapi_children(monkeypatch, status_by_request, *, db_path=None):
    """Fake cdsapi shared by every backend instance in the test. Records the
    ``year`` values of each retrieve; when ``db_path`` is given, snapshots the
    PERSISTED plan at the moment of each remote call (runs in a worker thread,
    so plain sqlite3 — this pins 'reserve BEFORE the remote submit')."""
    import sys
    import types

    fake_module = types.ModuleType("cdsapi")
    instance = MagicMock()
    counter = {"n": 0}
    submitted_years: list[str] = []
    plans_at_submit: list[dict[str, Any]] = []

    def _retrieve(name, request, target):
        counter["n"] += 1
        years = request.get("year")
        submitted_years.append(
            years[0] if isinstance(years, list) and years else "?"
        )
        if db_path is not None:
            conn = sqlite3.connect(db_path)
            try:
                row = conn.execute(
                    "SELECT chunk_plan_json FROM workflows "
                    "WHERE chunk_plan_json IS NOT NULL "
                    "ORDER BY created_at ASC LIMIT 1"
                ).fetchone()
            finally:
                conn.close()
            plans_at_submit.append(json.loads(row[0]) if row and row[0] else {})
        remote = MagicMock()
        remote.request_id = f"child-{counter['n']}"
        return remote

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
    fake_module.Client = MagicMock(return_value=instance)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cdsapi", fake_module)
    return instance, submitted_years, plans_at_submit


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


async def _edit_plan(persistence, parent_id: str, edit) -> None:
    """Simulate ANOTHER process's committed plan write (version bump)."""
    row = await persistence.fetch_workflow(parent_id)
    plan = json.loads(row["chunk_plan_json"])
    edit(plan)
    assert await persistence.update_chunk_plan(
        parent_id,
        json.dumps(plan, sort_keys=True, default=str),
        expected_version=row["chunk_plan_version"],
    )


@pytest.mark.asyncio
async def test_slot_is_reserved_in_the_persisted_plan_before_the_remote_call(
    tmp_path, foundation, monkeypatch
) -> None:
    """The heart of RESIL-006: at the moment the remote submit fires, the
    PERSISTED plan already carries this chunk's reservation — a concurrent
    poller reading the plan can no longer pick the same slot."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, _, plans_at_submit = _patch_cdsapi_children(
        monkeypatch, status, db_path=tmp_path / "state.db"
    )
    backend = CdsBackend(
        foundation=_with_max_inflight(foundation, 2), credentials=_fake_creds()
    )

    await backend.submit(_splittable_params())

    assert len(plans_at_submit) == 2
    for snapshot in plans_at_submit:
        reserved = [
            c
            for c in snapshot.get("chunks", [])
            if c.get("reserved_at") and not c.get("child_request_id")
        ]
        assert reserved, (
            "the persisted plan must carry a reservation before the remote "
            f"call; snapshot chunks: {snapshot.get('chunks')}"
        )


@pytest.mark.asyncio
async def test_fresh_foreign_reservation_is_skipped_and_counts_toward_budget(
    foundation, monkeypatch
) -> None:
    """A fresh reservation by another process holds its slot AND occupies one
    inflight budget slot: this poller (bound 2, one reservation outstanding)
    orders exactly one chunk, and not the reserved one."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, submitted_years, _ = _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(
        foundation=_with_max_inflight(foundation, 1), credentials=_fake_creds()
    )
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]
    # Wave 1 ordered exactly chunk 0 (year 2020). Let it succeed.
    first_child = submitted_years and None  # noqa: F841 - readability
    children = await foundation.persistence.list_child_workflows(parent_id)
    for c in children:
        status[c["request_id"]] = "successful"

    # Another process holds a FRESH reservation on chunk 1 (year 2021).
    now = datetime.now(UTC)
    await _edit_plan(
        foundation.persistence,
        parent_id,
        lambda p: p["chunks"][1].__setitem__("reserved_at", _iso(now)),
    )

    poller = CdsBackend(
        foundation=_with_max_inflight(foundation, 2), credentials=_fake_creds()
    )
    await poller.check_status(parent_id)

    assert submitted_years.count("2021") == 0, (
        "a freshly reserved slot must not be double-submitted"
    )
    # Budget 2 = 1 foreign reservation + 1 slot for us → exactly one new
    # chunk ordered, the first unreserved one (2022).
    assert submitted_years.count("2022") == 1


@pytest.mark.asyncio
async def test_stale_reservation_is_reclaimed(foundation, monkeypatch) -> None:
    """A reservation whose holder died (stale timestamp) must not strand the
    chunk: the next poller reclaims and orders it. The adoption scan protects
    against the dead holder having actually reached the server."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, submitted_years, _ = _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(
        foundation=_with_max_inflight(foundation, 1), credentials=_fake_creds()
    )
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]
    for c in await foundation.persistence.list_child_workflows(parent_id):
        status[c["request_id"]] = "successful"

    stale = datetime.now(UTC) - timedelta(hours=2)
    await _edit_plan(
        foundation.persistence,
        parent_id,
        lambda p: p["chunks"][1].__setitem__("reserved_at", _iso(stale)),
    )

    await backend.check_status(parent_id)

    assert submitted_years.count("2021") == 1, (
        "a stale reservation must be reclaimed, not strand its chunk"
    )


@pytest.mark.asyncio
async def test_cancel_survives_concurrent_plan_writes_without_clobbering(
    foundation, monkeypatch
) -> None:
    """The stopped flag must land even when another process committed plan
    writes since this process read the plan — AND the concurrent writer's
    state must survive (re-read + re-apply, not blind overwrite)."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(
        foundation=_with_max_inflight(foundation, 1), credentials=_fake_creds()
    )
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]

    # Another process commits an assignment AFTER our backend last read the
    # plan: chunk 3 got a child and a bumped version.
    await _edit_plan(
        foundation.persistence,
        parent_id,
        lambda p: p["chunks"][3].__setitem__(
            "child_request_id", "foreign-child-99"
        ),
    )

    await backend.cancel(parent_id)

    row = await foundation.persistence.fetch_workflow(parent_id)
    plan = json.loads(row["chunk_plan_json"])
    assert plan.get("stopped") is True
    assert plan["chunks"][3]["child_request_id"] == "foreign-child-99", (
        "cancel must not clobber a concurrently committed assignment"
    )


def test_reservation_freshness_boundary() -> None:
    """Review LOW: pin the TTL boundary — strictly-less-than 900 s is fresh,
    at/after 900 s is reclaimable."""
    from copernicus_mcp.backends.cds.backend import (
        _RESERVATION_TTL_SECONDS,
        _reservation_fresh,
    )

    now = datetime(2026, 8, 5, 12, 0, 0, tzinfo=UTC)
    now_iso = _iso(now)
    just_fresh = _iso(now - timedelta(seconds=_RESERVATION_TTL_SECONDS - 1))
    exactly_ttl = _iso(now - timedelta(seconds=_RESERVATION_TTL_SECONDS))
    assert _reservation_fresh({"reserved_at": just_fresh}, now_iso) is True
    assert _reservation_fresh({"reserved_at": exactly_ttl}, now_iso) is False
    assert _reservation_fresh({}, now_iso) is False
    assert _reservation_fresh({"reserved_at": "garbage"}, now_iso) is False


@pytest.mark.asyncio
async def test_release_requires_the_reservation_token(
    foundation, monkeypatch
) -> None:
    """Review round 1 (codex HIGH): a reservation carries an owner token. A
    process whose reservation was reclaimed (>TTL stall) must not release —
    or overwrite — the RECLAIMER's fresh reservation."""
    from copernicus_mcp.backends.cds.backend import CdsBackend, _load_chunk_plan

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(
        foundation=_with_max_inflight(foundation, 1), credentials=_fake_creds()
    )
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]

    # Slot 1 is reserved by ANOTHER process (its token, fresh stamp).
    now = datetime.now(UTC)
    await _edit_plan(
        foundation.persistence,
        parent_id,
        lambda p: p["chunks"][1].update(
            {"reserved_at": _iso(now), "reserved_by": "foreign-token"}
        ),
    )
    row = await foundation.persistence.fetch_workflow(parent_id)
    plan = _load_chunk_plan(row)

    # Our (stale) release with OUR token must abort, not drop theirs.
    await backend._release_reservation(parent_id, plan, 1, "our-token")

    row = await foundation.persistence.fetch_workflow(parent_id)
    entry = json.loads(row["chunk_plan_json"])["chunks"][1]
    assert entry.get("reserved_at"), "a foreign reservation was released"
    assert entry.get("reserved_by") == "foreign-token"


@pytest.mark.asyncio
async def test_stale_reserver_waking_up_cannot_clobber_the_reclaimer(
    tmp_path, monkeypatch
) -> None:
    """Review round 1 (codex HIGH), the full scenario: A reserves and stalls
    past the TTL; B reclaims the slot and commits its own child. When A wakes
    and finishes its submit, its assignment must ABORT — B's committed child
    stays in the plan."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    found_a, persistence_a = _make_foundation(tmp_path)
    await persistence_a.initialise()
    try:
        status: dict[str, str] = {}
        _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
        _patch_cdsapi_children(monkeypatch, status)
        a = CdsBackend(
            foundation=_with_max_inflight(found_a, 2), credentials=_fake_creds()
        )
        seed = CdsBackend(
            foundation=_with_max_inflight(found_a, 1), credentials=_fake_creds()
        )
        out = await seed.submit(_splittable_params())
        parent_id = out["request_id"]
        for c in await persistence_a.list_child_workflows(parent_id):
            status[c["request_id"]] = "successful"

        gate = asyncio.Event()
        started = asyncio.Event()
        original = a._submit_chunk_child

        async def _stalled(**kwargs):
            started.set()
            await gate.wait()
            return await original(**kwargs)

        a._submit_chunk_child = _stalled  # type: ignore[method-assign]
        a_poll = asyncio.ensure_future(a.check_status(parent_id))
        await asyncio.wait_for(started.wait(), timeout=10)

        # While A is stalled, "B" reclaims the slot (as it would after the
        # TTL) and commits its own child.
        def _reclaim(p):
            c = p["chunks"][1]
            c["reserved_at"] = None
            c["reserved_by"] = None
            c["child_request_id"] = "b-child-77"

        await _edit_plan(persistence_a, parent_id, _reclaim)
        status["b-child-77"] = "successful"

        gate.set()
        await a_poll

        row = await persistence_a.fetch_workflow(parent_id)
        plan = json.loads(row["chunk_plan_json"])
        assert plan["chunks"][1]["child_request_id"] == "b-child-77", (
            "the woken stale reserver overwrote the reclaimer's committed child"
        )
    finally:
        await persistence_a.close()


@pytest.mark.asyncio
async def test_terminal_read_repair_uses_the_freshest_plan(
    foundation, monkeypatch
) -> None:
    """Review round 1 (codex MEDIUM): another process's plan write landing
    during our child-poll must be visible to the terminal read-repair — a
    poll that makes no writes of its own otherwise computes the parent status
    from a stale plan and leaves the row non-terminal."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(monkeypatch, status)
    budget = foundation.config.budget.model_copy(
        update={
            "cds_chunk_max_inflight": 2,
            "cds_chunk_retry_limit": 3,
            "cds_chunk_retry_backoff_seconds": 3600.0,
        }
    )
    config = foundation.config.model_copy(update={"budget": budget})
    found = dataclasses.replace(foundation, config=config)
    backend = CdsBackend(foundation=found, credentials=_fake_creds())

    params = _splittable_params()
    params["inputs"]["year"] = ["2020", "2021"]  # exactly two chunks
    out = await backend.submit(params)
    parent_id = out["request_id"]
    children = await foundation.persistence.list_child_workflows(parent_id)
    assert len(children) == 2
    # Chunk 0 delivered; chunk 1 failed with the capacity signature and is
    # masked as awaiting a retry (backoff far in the future → our poll makes
    # no writes at all for it).
    status[children[0]["request_id"]] = "successful"
    failed_id = children[1]["request_id"]
    status[failed_id] = "failed"
    from copernicus_mcp.errors.records import build_error_record

    # Retryable (capacity signature): our poll takes the "backoff not yet
    # elapsed" continue path and writes NOTHING — the stale-plan window.
    record = build_error_record(
        "BackendError",
        message="refused",
        error_subclass="remote_job_failed",
        retryable=True,
    )
    await foundation.persistence.update_workflow_error_if_pending(
        failed_id, "failed", json.dumps(record.model_dump(mode="json"), default=str)
    )
    await _edit_plan(
        foundation.persistence,
        parent_id,
        lambda p: p["chunks"][1].update(
            {"retry_pending": True, "last_failed_at": _iso(datetime.now(UTC))}
        ),
    )
    # Let one poll observe the masked state (so its in-memory plan is warm),
    # then have "another process" clear the mask DURING the next poll's
    # child sweep.
    original_sweep = backend._poll_chunk_children

    async def _sweep_then_foreign_write(plan):
        result = await original_sweep(plan)
        await _edit_plan(
            foundation.persistence,
            parent_id,
            lambda p: p["chunks"][1].update(
                {"retry_pending": False, "last_failed_at": None}
            ),
        )
        return result

    backend._poll_chunk_children = _sweep_then_foreign_write  # type: ignore[method-assign]

    await backend.check_status(parent_id)

    row = await foundation.persistence.fetch_workflow(parent_id)
    assert row["status"] == "failed", (
        "the read-repair must settle the parent from the freshest plan, "
        f"got {row['status']!r}"
    )


@pytest.mark.asyncio
async def test_forced_interleave_cannot_double_submit(
    tmp_path, monkeypatch
) -> None:
    """Deterministic version of the run-31 window: process A has WON a
    reservation and is stalled mid-remote-call; process B polls the same
    parent to completion of its refill. B must skip A's reserved slot — the
    old code's read-then-write gap would have ordered it twice."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    found_a, persistence_a = _make_foundation(tmp_path)
    await persistence_a.initialise()
    found_b, persistence_b = _make_foundation(tmp_path)
    await persistence_b.initialise()
    try:
        status: dict[str, str] = {}
        _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
        _, submitted_years, _ = _patch_cdsapi_children(monkeypatch, status)
        a = CdsBackend(
            foundation=_with_max_inflight(found_a, 2), credentials=_fake_creds()
        )
        b = CdsBackend(
            foundation=_with_max_inflight(found_b, 3), credentials=_fake_creds()
        )

        seed = CdsBackend(
            foundation=_with_max_inflight(found_b, 1), credentials=_fake_creds()
        )
        out = await seed.submit(_splittable_params())
        parent_id = out["request_id"]
        for c in await persistence_a.list_child_workflows(parent_id):
            status[c["request_id"]] = "successful"

        gate = asyncio.Event()
        started = asyncio.Event()
        original = a._submit_chunk_child

        async def _stalled(**kwargs):
            started.set()
            await gate.wait()
            return await original(**kwargs)

        a._submit_chunk_child = _stalled  # type: ignore[method-assign]

        a_poll = asyncio.ensure_future(a.check_status(parent_id))
        await asyncio.wait_for(started.wait(), timeout=10)
        # A holds a persisted reservation and is stalled before the remote
        # call. B's full poll must not touch that slot.
        await b.check_status(parent_id)
        assert "2021" not in submitted_years, (
            "B ordered the slot A had reserved"
        )

        gate.set()
        await a_poll
        counts = {y: submitted_years.count(y) for y in set(submitted_years)}
        assert all(n == 1 for n in counts.values()), (
            f"duplicate remote submissions: {counts}"
        )
        assert submitted_years.count("2021") == 1  # A completed its slot
    finally:
        await persistence_a.close()
        await persistence_b.close()


@pytest.mark.asyncio
async def test_two_instances_one_db_never_duplicate_a_chunk(
    tmp_path, monkeypatch
) -> None:
    """Two backend instances over two connections to ONE database file (the
    two-process shape) poll the same parent concurrently, repeatedly. Every
    chunk is submitted to the remote at most once."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    found_a, persistence_a = _make_foundation(tmp_path)
    await persistence_a.initialise()
    found_b, persistence_b = _make_foundation(tmp_path)
    await persistence_b.initialise()
    try:
        status: dict[str, str] = {}
        _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
        _, submitted_years, _ = _patch_cdsapi_children(monkeypatch, status)
        a = CdsBackend(
            foundation=_with_max_inflight(found_a, 2), credentials=_fake_creds()
        )
        b = CdsBackend(
            foundation=_with_max_inflight(found_b, 2), credentials=_fake_creds()
        )

        out = await a.submit(_splittable_params())
        parent_id = out["request_id"]

        for _ in range(8):
            for c in await persistence_a.list_child_workflows(parent_id):
                status[c["request_id"]] = "successful"
            results = await asyncio.gather(
                a.check_status(parent_id),
                b.check_status(parent_id),
                return_exceptions=True,
            )
            for r in results:
                assert not isinstance(r, BaseException), r
            if all(
                not isinstance(r, BaseException)
                and r.get("status") == "successful"
                for r in results
            ):
                break

        # Review LOW: guard against a vacuous pass — the property "no
        # duplicates" must be proven over actual progress, not an empty set.
        assert set(submitted_years) == {"2020", "2021", "2022", "2023", "2024"}
        counts = {y: submitted_years.count(y) for y in set(submitted_years)}
        assert all(n == 1 for n in counts.values()), (
            f"duplicate remote submissions: {counts}"
        )
        children = await persistence_a.list_child_workflows(parent_id)
        assert len(children) == len(set(submitted_years))
    finally:
        await persistence_a.close()
        await persistence_b.close()
