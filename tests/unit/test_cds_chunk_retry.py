"""Bounded per-chunk retry for capacity failures (T-CDS-RESIL-003).

Field run 31: one refused child killed twenty good ones — any terminally
failed child failed the parent instantly, which is right for a bad request
and wrong for a busy service. Principle: *a parent is not terminal while
any of its chunks is still retryable.* A chunk whose child failed with the
capacity signature (RESIL-001 classification, or late corroboration by a
sibling that succeeded after the failure was recorded) is re-submitted —
same overrides, new child id — at most ``cds_chunk_retry_limit`` times,
spaced by ``cds_chunk_retry_backoff_seconds``. Content failures still fail
the parent promptly; a cancelled parent cancels everything, including a
chunk mid-backoff.
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


def _patch_cdsapi_children(
    monkeypatch, status_by_request, remote_json_by_request=None, *, raise_on_call=None
):
    """Like the chunk-lifecycle fake, plus ``remote_json_by_request`` to inject
    full remote payloads (empty-log failures vs content failures) and
    ``raise_on_call`` to make the Nth ``retrieve`` blow up (submit-time error)."""
    import sys
    import types

    fake_module = types.ModuleType("cdsapi")
    instance = MagicMock()
    counter = {"n": 0}

    def _retrieve(name, request, target):
        counter["n"] += 1
        if raise_on_call is not None and counter["n"] == raise_on_call:
            raise RuntimeError("submit refused")
        return _fake_remote(f"child-{counter['n']}")

    instance.retrieve = MagicMock(side_effect=_retrieve)
    inner = MagicMock()

    def _get_remote(request_id):
        rem = MagicMock()
        if remote_json_by_request and request_id in remote_json_by_request:
            rem.json = dict(remote_json_by_request[request_id])
        else:
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


async def _plan_of(persistence, parent_id: str) -> dict[str, Any]:
    row = await persistence.fetch_workflow(parent_id)
    assert row is not None
    return json.loads(row["chunk_plan_json"])


async def _active_children(persistence, parent_id: str) -> int:
    children = await persistence.list_child_workflows(parent_id)
    return sum(1 for c in children if c["status"] in ("queued", "running"))


# ---------------------------------------------------------------------------
# knobs + pure aggregation
# ---------------------------------------------------------------------------


def test_retry_knob_defaults() -> None:
    from copernicus_mcp.config import ConfigLoader

    config = ConfigLoader().load()
    assert config.budget.cds_chunk_retry_limit == 3
    assert config.budget.cds_chunk_retry_backoff_seconds == 120.0


def test_parent_status_masks_failed_chunk_awaiting_retry() -> None:
    from copernicus_mcp.backends.cds.chunking import compute_parent_status

    plan = {
        "stopped": False,
        "chunks": [
            {"index": 0, "child_request_id": "a", "retry_pending": True},
            {"index": 1, "child_request_id": "b"},
        ],
    }
    status = compute_parent_status(plan, {"a": "failed", "b": "running"})
    assert status == "running"


def test_parent_status_fails_on_non_retryable_failed_chunk() -> None:
    """A legacy plan entry (no retry fields) keeps today's fail-fast."""
    from copernicus_mcp.backends.cds.chunking import compute_parent_status

    plan = {
        "stopped": False,
        "chunks": [
            {"index": 0, "child_request_id": "a"},
            {"index": 1, "child_request_id": "b"},
        ],
    }
    assert compute_parent_status(plan, {"a": "failed", "b": "running"}) == "failed"


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capacity_failed_chunk_is_resubmitted_and_parent_completes(
    foundation, monkeypatch
) -> None:
    """Field acceptance check #1+#2: an empty-log child failure with a
    successful sibling is re-submitted (same overrides, new id) and the
    parent completes over waves — it is never terminal while a chunk is
    retryable."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(
        foundation=_with_budget(
            foundation,
            cds_chunk_max_inflight=2,
            cds_chunk_retry_backoff_seconds=0.0,
        ),
        credentials=_fake_creds(),
    )
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]
    assert sdk.retrieve.call_count == 2

    status["child-1"] = "failed"  # empty log: no error message in the payload
    status["child-2"] = "successful"  # corroborating sibling
    st = await backend.check_status(parent_id)

    assert st["status"] == "running"  # NOT failed
    plan = await _plan_of(foundation.persistence, parent_id)
    entry = plan["chunks"][0]
    assert entry["attempt"] == 1
    assert entry["superseded_child_ids"] == ["child-1"]
    assert entry["child_request_id"] not in (None, "child-1")

    # Every subsequently assigned id succeeds; drive to completion.
    for i in range(3, 9):
        status[f"child-{i}"] = "successful"
    final = None
    for _ in range(8):
        final = await backend.check_status(parent_id)
        assert await _active_children(foundation.persistence, parent_id) <= 2
        if final["status"] == "successful":
            break
    assert final is not None and final["status"] == "successful"
    assert sdk.retrieve.call_count == 6  # 5 chunks + 1 retry


@pytest.mark.asyncio
async def test_content_failure_still_fails_parent_promptly(
    foundation, monkeypatch
) -> None:
    """Field acceptance check #3 and §5: a failure WITH a server-side log
    is about the request — no retry, parent fails on the same poll."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    remote_json = {
        "child-1": {
            "status": "failed",
            "error": {"message": "invalid field 'frequency'"},
            "jobID": "child-1",
        }
    }
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status, remote_json)
    backend = CdsBackend(
        foundation=_with_budget(
            foundation,
            cds_chunk_max_inflight=2,
            cds_chunk_retry_backoff_seconds=0.0,
        ),
        credentials=_fake_creds(),
    )
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]

    status["child-2"] = "successful"
    st = await backend.check_status(parent_id)

    assert st["status"] == "failed"
    assert sdk.retrieve.call_count == 2  # no resubmission
    plan = await _plan_of(foundation.persistence, parent_id)
    assert plan["chunks"][0].get("attempt", 0) == 0


@pytest.mark.asyncio
async def test_retry_exhaustion_fails_parent(foundation, monkeypatch) -> None:
    """Attempts are bounded: when the limit is spent the parent fails, with
    every superseded child id preserved in the plan."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(
        foundation=_with_budget(
            foundation,
            cds_chunk_max_inflight=2,
            cds_chunk_retry_limit=1,
            cds_chunk_retry_backoff_seconds=0.0,
        ),
        credentials=_fake_creds(),
    )
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]

    status["child-1"] = "failed"
    status["child-2"] = "successful"
    st = await backend.check_status(parent_id)
    assert st["status"] == "running"  # retry 1 in flight
    plan = await _plan_of(foundation.persistence, parent_id)
    retried_id = plan["chunks"][0]["child_request_id"]

    status[retried_id] = "failed"  # the retry fails the same way
    final = await backend.check_status(parent_id)

    assert final["status"] == "failed"
    plan = await _plan_of(foundation.persistence, parent_id)
    assert plan["chunks"][0]["attempt"] == 1
    assert plan["chunks"][0]["superseded_child_ids"] == ["child-1"]


@pytest.mark.asyncio
async def test_backoff_defers_resubmission(foundation, monkeypatch) -> None:
    """Within the backoff window the chunk waits (parent stays running, no
    submit); once the clock passes the window the retry fires."""
    from copernicus_mcp.backends.cds import backend as backend_mod
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(
        foundation=_with_budget(
            foundation,
            cds_chunk_max_inflight=2,
            cds_chunk_retry_backoff_seconds=3600.0,
        ),
        credentials=_fake_creds(),
    )
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]

    status["child-1"] = "failed"
    status["child-2"] = "successful"
    st = await backend.check_status(parent_id)
    assert st["status"] == "running"
    plan = await _plan_of(foundation.persistence, parent_id)
    assert plan["chunks"][0]["retry_pending"] is True
    assert plan["chunks"][0]["child_request_id"] == "child-1"  # unchanged
    assert plan["chunks"][0].get("attempt", 0) == 0

    # Freed slots go to NEW chunks while the window runs (no idle slots);
    # the retry itself stays deferred across further in-window polls.
    for i in range(3, 6):
        status[f"child-{i}"] = "successful"
    await backend.check_status(parent_id)
    st = await backend.check_status(parent_id)
    assert st["status"] == "running"
    count_in_window = sdk.retrieve.call_count
    assert count_in_window == 5  # chunks 2-5 ordered once each; no retry
    plan = await _plan_of(foundation.persistence, parent_id)
    assert plan["chunks"][0]["retry_pending"] is True
    assert plan["chunks"][0]["child_request_id"] == "child-1"

    monkeypatch.setattr(
        backend_mod, "_iso_now", lambda: "2099-01-01T00:00:00Z"
    )
    st = await backend.check_status(parent_id)
    assert st["status"] == "running"
    assert sdk.retrieve.call_count == count_in_window + 1  # the retry fired
    plan = await _plan_of(foundation.persistence, parent_id)
    assert plan["chunks"][0]["attempt"] == 1
    assert plan["chunks"][0]["retry_pending"] is False
    retried_id = plan["chunks"][0]["child_request_id"]
    assert retried_id != "child-1"

    status[retried_id] = "successful"
    final = await backend.check_status(parent_id)
    assert final["status"] == "successful"


@pytest.mark.asyncio
async def test_cancel_during_backoff_cancels_everything(
    foundation, monkeypatch
) -> None:
    """Field acceptance check #4: cancel wins over a pending retry — no
    resubmission after the parent is cancelled."""
    from copernicus_mcp.backends.cds import backend as backend_mod
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(
        foundation=_with_budget(
            foundation,
            cds_chunk_max_inflight=2,
            cds_chunk_retry_backoff_seconds=3600.0,
        ),
        credentials=_fake_creds(),
    )
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]

    status["child-1"] = "failed"
    status["child-2"] = "successful"
    await backend.check_status(parent_id)
    count_before = sdk.retrieve.call_count

    cancelled = await backend.cancel(parent_id)
    assert cancelled["status"] == "cancelled"

    monkeypatch.setattr(
        backend_mod, "_iso_now", lambda: "2099-01-01T00:00:00Z"
    )
    st = await backend.check_status(parent_id)
    assert st["status"] == "cancelled"
    assert sdk.retrieve.call_count == count_before  # no retry after cancel


# ---------------------------------------------------------------------------
# a declined retry must never leave the parent immortal
#
# ``retry_pending`` masks a failed chunk as active in ``compute_parent_status``.
# Every path that DECLINES to retry such a chunk must therefore clear the flag,
# or the parent stays non-terminal forever — it can never succeed (the chunk has
# no file) and can never fail (the mask hides it). Found by self-review; the
# trigger is any config change between polls, which an operator restart makes
# ordinary.
# ---------------------------------------------------------------------------


async def _park_chunk_in_retry_pending(backend, status, sdk):
    """Drive chunk 0 into ``retry_pending`` with the backoff window still open."""
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]
    status["child-1"] = "failed"
    status["child-2"] = "successful"
    st = await backend.check_status(parent_id)
    assert st["status"] == "running"
    return parent_id


@pytest.mark.asyncio
async def test_retry_disabled_midflight_settles_parent_not_hangs(
    foundation, monkeypatch
) -> None:
    """A chunk parked in ``retry_pending``, then retry turned OFF by config:
    the parent must settle as ``failed``, not run forever."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)
    budget = dict(cds_chunk_max_inflight=2, cds_chunk_retry_backoff_seconds=3600.0)
    backend = CdsBackend(
        foundation=_with_budget(foundation, **budget), credentials=_fake_creds()
    )
    parent_id = await _park_chunk_in_retry_pending(backend, status, sdk)

    # Operator disables retry and the server restarts: same state DB, new config.
    restarted = CdsBackend(
        foundation=_with_budget(foundation, **budget, cds_chunk_retry_limit=0),
        credentials=_fake_creds(),
    )
    st = await restarted.check_status(parent_id)

    assert st["status"] == "failed"


@pytest.mark.asyncio
async def test_retry_limit_lowered_below_spent_attempts_settles_parent(
    foundation, monkeypatch
) -> None:
    """Same hazard via a lowered (not zeroed) limit: a chunk parked in
    ``retry_pending`` having already spent more attempts than the new ceiling
    must fail the parent, not mask forever."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(monkeypatch, status)
    inflight = dict(cds_chunk_max_inflight=2)

    # Attempt 1 fires immediately (zero backoff).
    eager = CdsBackend(
        foundation=_with_budget(
            foundation,
            cds_chunk_retry_limit=3,
            cds_chunk_retry_backoff_seconds=0.0,
            **inflight,
        ),
        credentials=_fake_creds(),
    )
    out = await eager.submit(_splittable_params())
    parent_id = out["request_id"]
    status["child-1"] = "failed"
    status["child-2"] = "successful"
    assert (await eager.check_status(parent_id))["status"] == "running"
    plan = await _plan_of(foundation.persistence, parent_id)
    assert plan["chunks"][0]["attempt"] == 1
    status[plan["chunks"][0]["child_request_id"]] = "failed"

    # A long backoff parks the chunk with retry_pending set and attempt == 1.
    patient = CdsBackend(
        foundation=_with_budget(
            foundation,
            cds_chunk_retry_limit=3,
            cds_chunk_retry_backoff_seconds=3600.0,
            **inflight,
        ),
        credentials=_fake_creds(),
    )
    assert (await patient.check_status(parent_id))["status"] == "running"
    plan = await _plan_of(foundation.persistence, parent_id)
    assert plan["chunks"][0]["retry_pending"] is True
    assert plan["chunks"][0]["attempt"] == 1

    # Operator lowers the ceiling to 1 — the parked chunk has already spent it.
    restarted = CdsBackend(
        foundation=_with_budget(
            foundation,
            cds_chunk_retry_limit=1,
            cds_chunk_retry_backoff_seconds=3600.0,
            **inflight,
        ),
        credentials=_fake_creds(),
    )
    st = await restarted.check_status(parent_id)

    assert st["status"] == "failed"


@pytest.mark.asyncio
async def test_failed_retry_submit_stops_the_advance_not_just_itself(
    foundation, monkeypatch
) -> None:
    """A retry that errors at submit time settles the parent ``failed``. The
    SAME advance must then stop — the refill runs next and only inspects the
    ``stopped`` flag and child statuses, never the parent's own status, so
    without a settled-signal it goes on ordering fresh REMOTE jobs under a dead
    parent, burning the concurrency slots this feature exists to conserve."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    # calls 1-2 = first wave; call 3 = the retry, which is refused.
    _, sdk = _patch_cdsapi_children(monkeypatch, status, raise_on_call=3)
    backend = CdsBackend(
        foundation=_with_budget(
            foundation,
            cds_chunk_max_inflight=2,
            cds_chunk_retry_backoff_seconds=0.0,
        ),
        credentials=_fake_creds(),
    )
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]
    assert sdk.retrieve.call_count == 2

    status["child-1"] = "failed"
    status["child-2"] = "successful"
    st = await backend.check_status(parent_id)

    assert st["status"] == "failed"
    assert sdk.retrieve.call_count == 3, (
        "the refill ordered fresh CDS jobs after the parent had already failed"
    )


@pytest.mark.asyncio
async def test_a_doomed_parent_stops_the_retry_loop_too(
    foundation, monkeypatch
) -> None:
    """A content failure dooms the parent. The retry loop must stop there — it
    was continuing to the next chunk and re-submitting a capacity-classified
    one, so a real CDS job got ordered under a parent that was already dead and
    was reclaimed only by a best-effort remote delete.

    Companion to the refill fix: same defect, other loop."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    remote_json = {
        "child-1": {
            "status": "failed",
            "error": {"message": "invalid field 'frequency'"},
            "jobID": "child-1",
        }
    }
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status, remote_json)
    backend = CdsBackend(
        foundation=_with_budget(
            foundation,
            cds_chunk_max_inflight=3,
            cds_chunk_retry_backoff_seconds=0.0,
        ),
        credentials=_fake_creds(),
    )
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]
    assert sdk.retrieve.call_count == 3

    status["child-2"] = "failed"  # empty log -> capacity, corroborated by...
    status["child-3"] = "successful"  # ...this sibling
    st = await backend.check_status(parent_id)

    assert st["status"] == "failed"  # child-1 is a content failure
    assert sdk.retrieve.call_count == 3, (
        "the retry loop ordered a CDS job under an already-doomed parent"
    )


@pytest.mark.asyncio
async def test_rejected_chunk_retries_without_needing_a_successful_sibling(
    foundation, monkeypatch
) -> None:
    """The recorded CORDEX shape, inside a chunked parent: admission control
    refuses a child (``status: "rejected"``, empty log) while nothing has
    succeeded yet.

    This is the case sibling-corroboration cannot reach — capacity refusals
    come back in seconds while successes take minutes, so the first wave is
    exactly when no sibling has succeeded, and a parent that fails there is
    terminal forever. The remote status carries the signal on its own."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {"child-1": "rejected"}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _, sdk = _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(
        foundation=_with_budget(
            foundation,
            cds_chunk_max_inflight=2,
            cds_chunk_retry_backoff_seconds=0.0,
        ),
        credentials=_fake_creds(),
    )
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]
    assert sdk.retrieve.call_count == 2

    st = await backend.check_status(parent_id)

    assert st["status"] == "running", "a refused-at-admission chunk killed the parent"
    plan = await _plan_of(foundation.persistence, parent_id)
    assert plan["chunks"][0]["attempt"] == 1
    assert plan["chunks"][0]["superseded_child_ids"] == ["child-1"]
