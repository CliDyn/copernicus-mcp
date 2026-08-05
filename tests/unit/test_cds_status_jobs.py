"""Per-account active-jobs count in ``copernicus_mcp_status``
(T-CDS-OPS-001).

Downstream consumers pace their own submissions against CDS's undocumented per-user
concurrency ceiling and previously had to GUESS how many jobs the account
had in flight — its own counter misses everything submitted by anything
else. The status tool now reports the number. Deliberately far from the
retry classifier: the Phase-1 lesson (no job census for CLASSIFICATION)
stands — this is operator reporting, exactly what the remark asked for.
Failure-tolerant: the status tool must answer even when CDS is down.
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

    config = ConfigLoader().load(
        cli_overrides={
            "storage": {
                "state_database": str(tmp_path / "state.db"),
                "cache_directory": str(tmp_path / "cache"),
            }
        }
    )
    persistence = SqliteBackend(config.storage.state_database)
    cache = CacheManager(
        cache_directory=config.storage.cache_directory,
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
async def setup(tmp_path):
    from copernicus_mcp.backends.registry import BackendRegistry
    from copernicus_mcp.workflow.orchestrator import WorkflowOrchestrator

    foundation, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    registry = BackendRegistry()
    try:
        yield foundation, registry, WorkflowOrchestrator(
            registry=registry, foundation=foundation
        )
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


class _FakeJobs:
    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self.json = {"jobs": entries}


def _patch_cdsapi_jobs(monkeypatch, entries_or_exc):
    import sys
    import types

    fake_module = types.ModuleType("cdsapi")
    instance = MagicMock()
    inner = MagicMock()
    if isinstance(entries_or_exc, Exception):
        inner.get_jobs = MagicMock(side_effect=entries_or_exc)
    else:
        inner.get_jobs = MagicMock(return_value=_FakeJobs(entries_or_exc))
    instance.client = inner
    fake_module.Client = MagicMock(return_value=instance)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cdsapi", fake_module)
    return inner


@pytest.mark.asyncio
async def test_status_reports_active_remote_jobs(setup, monkeypatch) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend

    foundation, registry, orchestrator = setup
    _patch_cdsapi_jobs(
        monkeypatch,
        [
            {"jobID": "a", "status": "accepted"},
            {"jobID": "b", "status": "accepted"},
            {"jobID": "c", "status": "running"},
        ],
    )
    registry.register(
        CdsBackend(foundation=foundation, credentials=_fake_creds())
    )

    out = await orchestrator.status()

    block = out["backends"]["cds"]["active_remote_jobs"]
    assert block["count"] == 3
    assert block["by_status"] == {"accepted": 2, "running": 1}
    assert block["fetched_at"]
    assert "truncated" not in block  # partial page == the full truth


@pytest.mark.asyncio
async def test_status_flags_a_full_page_as_truncated(setup, monkeypatch) -> None:
    """Review round 1 (codex MEDIUM): ``get_jobs`` is paginated. A page filled
    to the limit means "at least this many" — reporting it as an exact count
    would make pacing callers underestimate the account's remote load."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    foundation, registry, orchestrator = setup
    _patch_cdsapi_jobs(
        monkeypatch,
        [{"jobID": str(i), "status": "accepted"} for i in range(100)],
    )
    registry.register(
        CdsBackend(foundation=foundation, credentials=_fake_creds())
    )

    out = await orchestrator.status()

    block = out["backends"]["cds"]["active_remote_jobs"]
    assert block["count"] == 100
    assert block["truncated"] is True


@pytest.mark.asyncio
async def test_status_survives_a_cds_outage(setup, monkeypatch) -> None:
    """The status tool must never break because CDS is down."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    foundation, registry, orchestrator = setup
    _patch_cdsapi_jobs(monkeypatch, RuntimeError("api down"))
    registry.register(
        CdsBackend(foundation=foundation, credentials=_fake_creds())
    )

    out = await orchestrator.status()

    assert out["backends"]["cds"]["active_remote_jobs"] == "unavailable"
    assert out["backends"]["cds"]["registered"] is True


@pytest.mark.asyncio
async def test_status_without_credentials_omits_the_block(
    setup, monkeypatch
) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend

    foundation, registry, orchestrator = setup
    _patch_cdsapi_jobs(monkeypatch, [])
    registry.register(CdsBackend(foundation=foundation, credentials=None))

    out = await orchestrator.status()

    assert "active_remote_jobs" not in out["backends"]["cds"]


@pytest.mark.asyncio
async def test_status_details_errors_never_break_status(setup) -> None:
    """A backend whose status_details RAISES must not take the whole
    diagnostic down."""
    from copernicus_mcp.backends.abstract import AbstractBackend

    foundation, registry, orchestrator = setup

    class _Explosive(AbstractBackend):
        backend_id = "cmems"

        async def search(self, params): return {}
        async def describe(self, identifier): return {}
        async def validate(self, params): return {"valid": True}
        async def estimate(self, params): return {}
        async def submit(self, params): return {}
        async def check_status(self, request_id): return {}
        async def fetch_result(self, request_id, target): return {}
        async def cancel(self, request_id): return {"cancelled": True}
        @property
        def supports_async(self): return False
        @property
        def supports_dry_run(self): return True
        @property
        def requires_terms_acceptance(self): return False

        async def status_details(self):
            raise RuntimeError("kaboom")

    registry.register(_Explosive(foundation=foundation))

    out = await orchestrator.status()

    assert "backends" in out  # the tool still answered
    assert out["backends"]["cmems"]["registered"] is True


# ---------------------------------------------------------------------------
# T-CDS-OPS-002: batch check-status
# ---------------------------------------------------------------------------


def _batch_input(**kwargs):
    from copernicus_mcp.backends.cds.tools import CdsCheckRequestStatusInput

    return CdsCheckRequestStatusInput(**kwargs)


def test_check_status_input_takes_one_id_or_a_list_never_both() -> None:
    import pydantic

    assert _batch_input(request_id="a").request_id == "a"
    assert _batch_input(request_ids=["a", "b"]).request_ids == ["a", "b"]
    with pytest.raises(pydantic.ValidationError):
        _batch_input(request_id="a", request_ids=["b"])
    with pytest.raises(pydantic.ValidationError):
        _batch_input()
    with pytest.raises(pydantic.ValidationError):
        _batch_input(request_ids=[])
    # Local review round (MEDIUM): an empty ELEMENT must fail exactly like
    # request_id="" does in single mode.
    with pytest.raises(pydantic.ValidationError):
        _batch_input(request_ids=["", "abc"])


@pytest.mark.asyncio
async def test_batch_check_status_preserves_order_and_inlines_errors() -> None:
    """A 21-part window polled every 30 s over 10 h was ~18k CLI spawns
    (field report). One call takes several ids; a bad id yields an inline error entry
    and must not fail the batch."""
    from unittest.mock import AsyncMock

    from copernicus_mcp.backends.cds.tools import cds_check_request_status

    async def _run(*, backend, operation, params):
        rid = params["request_id"]
        if rid == "missing":
            return {"error": {"error_class": "NotFoundError", "message": "no"}}
        return {"result": {"status": "successful", "request_id": rid}}

    orch = AsyncMock()
    orch.run.side_effect = _run

    out = await cds_check_request_status(
        _batch_input(request_ids=["a", "missing", "b"]), orchestrator=orch
    )

    results = out["results"]
    # Local review round (MEDIUM): a failed entry carries ITS id — callers
    # must not have to zip against their input to correlate failures.
    assert [r.get("request_id") for r in results] == ["a", "missing", "b"]
    assert results[0]["status"] == "successful"
    assert results[1]["error"]["error_class"] == "NotFoundError"
    assert results[2]["status"] == "successful"
    assert out["count"] == 3


@pytest.mark.asyncio
async def test_single_id_behaviour_is_byte_identical() -> None:
    from unittest.mock import AsyncMock

    from copernicus_mcp.backends.cds.tools import cds_check_request_status

    orch = AsyncMock()
    orch.run.return_value = {"result": {"status": "running", "request_id": "a"}}

    out = await cds_check_request_status(
        _batch_input(request_id="a"), orchestrator=orch
    )

    assert out == {"status": "running", "request_id": "a"}


def test_cli_check_status_accepts_several_ids(monkeypatch) -> None:
    import json as _json
    from unittest.mock import AsyncMock

    from typer.testing import CliRunner

    from copernicus_mcp import cli

    fake = AsyncMock()

    async def _run(*, backend, operation, params):
        return {"result": {"status": "running", "request_id": params["request_id"]}}

    fake.run.side_effect = _run

    class _Builder:
        def __call__(self):
            return self

        async def __aenter__(self):
            return fake

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _Builder())
    runner = CliRunner()

    res = runner.invoke(cli.app, ["cds", "check-status", "id1", "id2", "--json"])
    assert res.exit_code == 0, res.stdout + str(res.exception)
    parsed = _json.loads(res.stdout)
    assert [r["request_id"] for r in parsed["results"]] == ["id1", "id2"]

    # single id keeps the old single-envelope output
    res = runner.invoke(cli.app, ["cds", "check-status", "id1", "--json"])
    assert res.exit_code == 0
    assert _json.loads(res.stdout)["request_id"] == "id1"
