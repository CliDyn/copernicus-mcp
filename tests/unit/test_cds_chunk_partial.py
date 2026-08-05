"""Failed chunked parent: partial-result adoption (T-CDS-RESIL-004).

Field run 31: a parent finished ``failed`` with three successful children —
three years of real, delivered data — and they were unreachable through the
parent. A terminally failed parent's ``check_status`` envelope must expose
the successful children's descriptors as an explicitly labelled
``partial_result`` block (never inside ``result`` — a failed parent must not
look successful), and each child must remain individually resolvable via
``fetch_result(child_request_id)``.
"""

from __future__ import annotations

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


def _patch_cdsapi_children(monkeypatch, status_by_request, remote_json_by_request=None):
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


_CONTENT_FAILURE = {
    "status": "failed",
    "error": {"message": "invalid field 'frequency'"},
    "jobID": "child-3",
}


async def _fail_parent_with_two_successes(backend, status):
    """Submit the 5-chunk plan, land chunks 0+1, content-fail chunk 2."""
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]
    status["child-1"] = "successful"
    status["child-2"] = "successful"
    st = await backend.check_status(parent_id)
    assert st["status"] == "failed"
    return parent_id, st


@pytest.mark.asyncio
async def test_failed_parent_exposes_partial_result(foundation, monkeypatch) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(
        monkeypatch, status, {"child-3": dict(_CONTENT_FAILURE)}
    )
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    _, st = await _fail_parent_with_two_successes(backend, status)

    partial = st["partial_result"]
    assert partial["chunk_indices"] == [0, 1]
    assert [f["chunk_index"] for f in partial["files"]] == [0, 1]
    assert all(f["filepath"] for f in partial["files"])
    assert partial["missing_chunk_indices"] == [2, 3, 4]
    # A failed parent must never look successful: result stays the stub.
    assert "files" not in st["result"]


@pytest.mark.asyncio
async def test_partial_result_stable_on_repolls_of_terminal_parent(
    foundation, monkeypatch
) -> None:
    """A later poll of the already-terminal parent still carries the block."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(
        monkeypatch, status, {"child-3": dict(_CONTENT_FAILURE)}
    )
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    parent_id, _ = await _fail_parent_with_two_successes(backend, status)

    again = await backend.check_status(parent_id)
    assert again["status"] == "failed"
    assert [f["chunk_index"] for f in again["partial_result"]["files"]] == [0, 1]


@pytest.mark.asyncio
async def test_successful_parent_has_no_partial_result(foundation, monkeypatch) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]
    for i in range(1, 6):
        status[f"child-{i}"] = "successful"
    final = None
    for _ in range(7):
        final = await backend.check_status(parent_id)
        if final["status"] == "successful":
            break
    assert final is not None and final["status"] == "successful"
    assert "partial_result" not in final


@pytest.mark.asyncio
async def test_no_partial_result_when_nothing_succeeded(foundation, monkeypatch) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(
        monkeypatch, status, {"child-1": dict(_CONTENT_FAILURE)}
    )
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.submit(_splittable_params())

    st = await backend.check_status(out["request_id"])

    assert st["status"] == "failed"
    assert "partial_result" not in st


@pytest.mark.asyncio
async def test_evicted_successful_child_reported_missing_not_listed(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend, _cache_storage_key

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(
        monkeypatch, status, {"child-3": dict(_CONTENT_FAILURE)}
    )
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    parent_id, _ = await _fail_parent_with_two_successes(backend, status)

    children = await foundation.persistence.list_child_workflows(parent_id)
    chunk0 = next(c for c in children if c["request_id"] == "child-1")
    evicted_key = _cache_storage_key(chunk0["cache_key"])
    real_lookup = foundation.cache.lookup_file

    async def _evicting_lookup(key):
        return None if key == evicted_key else await real_lookup(key)

    monkeypatch.setattr(foundation.cache, "lookup_file", _evicting_lookup)
    st = await backend.check_status(parent_id)

    partial = st["partial_result"]
    assert [f["chunk_index"] for f in partial["files"]] == [1]
    assert 0 in partial["missing_chunk_indices"]


@pytest.mark.asyncio
async def test_successful_child_of_failed_parent_fetches_by_id(
    foundation, monkeypatch, tmp_path: Path
) -> None:
    """Adoption path: children are first-class request ids — fetch_result on a
    successful child of a FAILED parent resolves its file."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(
        monkeypatch, status, {"child-3": dict(_CONTENT_FAILURE)}
    )
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await _fail_parent_with_two_successes(backend, status)

    res = await backend.fetch_result("child-1", target=tmp_path / "ignored")

    assert res["status"] == "successful"
    assert res["result"]["filepath"]


# ---------------------------------------------------------------------------
# a successful parent must not advertise a short file set as the whole thing
# ---------------------------------------------------------------------------


async def _drive_all_chunks_successful(backend, status, parent_id):
    for i in range(1, 6):
        status[f"child-{i}"] = "successful"
    for _ in range(8):
        st = await backend.check_status(parent_id)
        if st["status"] == "successful":
            return st
    raise AssertionError("parent never reached successful")


@pytest.mark.asyncio
async def test_successful_parent_marks_its_file_set_complete(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.submit(_splittable_params())

    st = await _drive_all_chunks_successful(backend, status, out["request_id"])

    assert st["result"]["complete"] is True
    assert len(st["result"]["files"]) == 5


@pytest.mark.asyncio
async def test_successful_parent_with_evicted_chunk_is_flagged_incomplete(
    foundation, monkeypatch
) -> None:
    """All jobs succeeded, so the workflow status stays ``successful`` — but one
    chunk file has since been evicted from the cache, so the delivered set is
    short. ``fetch_result`` already raises CacheError on this state; the status
    envelope must not meanwhile present the remaining files as the whole
    retrieval (Phase 1 constraint: never report success on a partial set)."""
    from copernicus_mcp.backends.cds.backend import CdsBackend, _cache_storage_key

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]
    await _drive_all_chunks_successful(backend, status, parent_id)

    children = await foundation.persistence.list_child_workflows(parent_id)
    evicted_key = _cache_storage_key(
        next(c for c in children if c["request_id"] == "child-3")["cache_key"]
    )
    real_lookup = foundation.cache.lookup_file

    async def _evicting_lookup(key):
        return None if key == evicted_key else await real_lookup(key)

    monkeypatch.setattr(foundation.cache, "lookup_file", _evicting_lookup)
    st = await backend.check_status(parent_id)

    assert st["status"] == "successful"  # the jobs really did succeed
    assert st["result"]["complete"] is False
    assert st["result"]["evicted_chunk_indices"] == [2]
    assert len(st["result"]["files"]) == 4
    assert st["result"]["recovery_hint"]


@pytest.mark.asyncio
async def test_fetch_result_success_payload_also_carries_complete(
    foundation, monkeypatch, tmp_path: Path
) -> None:
    """``complete`` must mean the same thing on both success surfaces. This
    path only returns when every chunk file is present (it raises CacheError
    otherwise), so it is always true here — but the key has to BE there, or a
    caller that checks it on check_status gets nothing from fetch_result."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    status: dict[str, str] = {}
    _patch_costing_by_shape(monkeypatch, per_year=365.4, limit=400.0)
    _patch_cdsapi_children(monkeypatch, status)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.submit(_splittable_params())
    parent_id = out["request_id"]
    await _drive_all_chunks_successful(backend, status, parent_id)

    res = await backend.fetch_result(parent_id, target=tmp_path / "ignored")

    assert res["status"] == "successful"
    assert res["result"]["complete"] is True


def test_cancelled_parent_also_exposes_what_landed() -> None:
    """Cancelling does not un-pay for the chunks that already completed, so a
    cancelled parent surfaces them the same way a failed one does."""
    from copernicus_mcp.backends.cds.backend import _chunk_parent_response

    plan = {"granularity": "year", "stopped": True, "chunks": [
        {"index": 0, "child_request_id": "a"},
        {"index": 1, "child_request_id": "b"},
    ]}
    out = _chunk_parent_response(
        parent_id="p",
        cache_key="k",
        plan=plan,
        child_status={"a": "successful", "b": "cancelled"},
        status="cancelled",
        partial_files=[{"chunk_index": 0, "filepath": "/tmp/a.grib"}],
        missing_chunk_indices=[1],
    )
    assert out["partial_result"]["chunk_indices"] == [0]
    assert out["partial_result"]["missing_chunk_indices"] == [1]
    assert "files" not in out["result"]
