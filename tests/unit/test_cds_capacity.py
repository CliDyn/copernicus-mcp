"""Capacity-vs-content classification of empty-log remote failures
(T-CDS-RESIL-001).

A failure with no server-side log covers both "the request is wrong" and
"the service refused it under load" — and the two need opposite responses
(change the request vs resubmit unchanged after a pause). The classifier
corroborates before calling it capacity: the job's own remote status is
``rejected`` (turned away by admission control, never ran — the recorded
CORDEX incident, where a per-dataset queue cap refused part of our own
burst and the identical requests succeeded once it drained), or a sibling
chunk of the same parent already succeeded.

A job that reached ``failed`` on its own with no log and no successful
sibling stays ``unknown`` and is never retried: that is the shape of a
request that RAN and found nothing, like the E-OBS phantom-version case,
and retrying it is a slower way to fail (field report §5).
"""

from __future__ import annotations

import json
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


def _fake_creds():
    from copernicus_mcp.auth.resolver import ResolvedCredentials

    return ResolvedCredentials(
        backend="cds",
        source="explicit",
        source_detail="test",
        fields={"key": "abcdef01-2345-6789-abcd-ef0123456789"},
    )


def _backend(foundation):
    from copernicus_mcp.backends.cds.backend import CdsBackend

    return CdsBackend(foundation=foundation, credentials=_fake_creds())


def _wf(
    request_id: str,
    *,
    status: str = "running",
    parent_request_id: str | None = None,
    chunk_plan_json: str | None = None,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "backend_id": "cds",
        "operation": "submit",
        "status": status,
        "cache_key": f"cds:test:{request_id}",
        "request_json": json.dumps(
            {"dataset_id": "insitu-observations-woudc-ozone", "inputs": {}}
        ),
        "response_json": None,
        "error_record_json": None,
        "created_at": "2026-08-03T00:00:00+00:00",
        "updated_at": "2026-08-03T00:00:00+00:00",
        "parent_request_id": parent_request_id,
        "chunk_plan_json": chunk_plan_json,
    }


async def _error_record(persistence, request_id: str) -> dict[str, Any]:
    row = await persistence.fetch_workflow(request_id)
    assert row is not None and row["status"] == "failed"
    assert row["error_record_json"]
    return json.loads(row["error_record_json"])


# ---------------------------------------------------------------------------
# pure classifier truth table
# ---------------------------------------------------------------------------


def test_classifier_content_when_server_log_present() -> None:
    from copernicus_mcp.backends.cds.capacity import classify_remote_failure

    assert (
        classify_remote_failure(
            empty_log=False,
            sibling_succeeded=True,
            remote_status_rejected=True,
        )
        == "content"
    )


def test_classifier_capacity_on_successful_sibling() -> None:
    from copernicus_mcp.backends.cds.capacity import classify_remote_failure

    assert (
        classify_remote_failure(
            empty_log=True,
            sibling_succeeded=True,
            remote_status_rejected=False,
        )
        == "capacity_suspected"
    )


def test_classifier_capacity_on_saturated_account() -> None:
    from copernicus_mcp.backends.cds.capacity import classify_remote_failure

    assert (
        classify_remote_failure(
            empty_log=True,
            sibling_succeeded=False,
            remote_status_rejected=True,
        )
        == "capacity_suspected"
    )


def test_classifier_unknown_on_a_plain_failed_job() -> None:
    """Ran and failed, no log, no sibling — the E-OBS phantom-version shape.
    Must never be auto-retried."""
    from copernicus_mcp.backends.cds.capacity import classify_remote_failure

    assert (
        classify_remote_failure(
            empty_log=True,
            sibling_succeeded=False,
            remote_status_rejected=False,
        )
        == "unknown"
    )


def test_classifier_takes_no_job_census() -> None:
    """The signature must not grow an account/threshold parameter back. Three
    review rounds showed a job census cannot discriminate: our own load scales
    with how many retrievals we run, and the one refusal on record was our own
    burst against a per-dataset cap with nothing else on the account."""
    import inspect

    from copernicus_mcp.backends.cds.capacity import classify_remote_failure

    params = set(inspect.signature(classify_remote_failure).parameters)
    assert params == {"empty_log", "sibling_succeeded", "remote_status_rejected"}


# ---------------------------------------------------------------------------
# _record_terminal wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejected_status_is_capacity_even_when_we_caused_it(
    foundation,
) -> None:
    """The recorded CORDEX incident: a 9-request burst tripped a PER-DATASET
    queued-request cap, two were refused with ``status: "rejected"`` and an
    empty log, nothing else was on the account, and the identical requests
    succeeded once the burst drained. The load was entirely our own, so no
    count of foreign jobs can identify this — the remote status can."""
    backend = _backend(foundation)
    await foundation.persistence.record_workflow(_wf("cordex-3"))

    await backend._record_terminal(
        "cordex-3",
        "failed",
        remote_json={"status": "rejected", "log": []},
        client=None,
    )

    record = await _error_record(foundation.persistence, "cordex-3")
    fc = record["context"]["failure_classification"]
    assert record["retryable"] is True
    assert record["automatic_retry_recommended"] is True
    assert fc["class"] == "capacity_suspected"
    assert fc["remote_status"] == "rejected"


@pytest.mark.asyncio
async def test_empty_log_child_with_successful_sibling_is_retryable(
    foundation,
) -> None:
    """The run-31 shape: a chunk child fails with the empty-log signature
    while a sibling of the same parent already succeeded."""
    backend = _backend(foundation)
    await foundation.persistence.record_workflow(
        _wf("parent-1", chunk_plan_json=json.dumps({"chunks": []}))
    )
    await foundation.persistence.record_workflow(
        _wf("child-ok", status="successful", parent_request_id="parent-1")
    )
    await foundation.persistence.record_workflow(
        _wf("child-bad", status="running", parent_request_id="parent-1")
    )

    await backend._record_terminal(
        "child-bad", "failed", remote_json={"status": "failed"}, client=None
    )

    record = await _error_record(foundation.persistence, "child-bad")
    assert record["retryable"] is True
    fc = record["context"]["failure_classification"]
    assert fc["class"] == "capacity_suspected"
    assert fc["sibling_succeeded"] is True


@pytest.mark.asyncio
async def test_plain_failed_job_stays_unretryable(foundation) -> None:
    """Ran, produced nothing, no sibling to vouch for the shape → never retry.
    This is the leg that protects against resubmitting a bad request."""
    backend = _backend(foundation)
    await foundation.persistence.record_workflow(_wf("solo-1"))

    await backend._record_terminal(
        "solo-1", "failed", remote_json={"status": "failed"}, client=None
    )

    record = await _error_record(foundation.persistence, "solo-1")
    assert record["retryable"] is False
    assert record["automatic_retry_recommended"] is False
    assert record["context"]["failure_classification"]["class"] == "unknown"


@pytest.mark.asyncio
async def test_paced_siblings_do_not_make_a_content_failure_retryable(
    foundation,
) -> None:
    """A content-invalid chunk fails while its paced siblings are legitimately
    running. Sibling activity is not sibling SUCCESS, so it must not
    corroborate — otherwise every content failure gets resubmitted."""
    backend = _backend(foundation)
    await foundation.persistence.record_workflow(
        _wf("parent-p", chunk_plan_json=json.dumps({"chunks": []}))
    )
    for i in range(4):
        await foundation.persistence.record_workflow(
            _wf(f"sib-{i}", status="running", parent_request_id="parent-p")
        )
    await foundation.persistence.record_workflow(
        _wf("bad-chunk", status="running", parent_request_id="parent-p")
    )

    await backend._record_terminal(
        "bad-chunk", "failed", remote_json={"status": "failed"}, client=None
    )

    record = await _error_record(foundation.persistence, "bad-chunk")
    assert record["retryable"] is False
    assert record["context"]["failure_classification"]["class"] == "unknown"


@pytest.mark.asyncio
async def test_server_log_failure_is_content_and_unclassified(foundation) -> None:
    """A real server-side message is a content failure: no retry flags, no
    classification block, message preserved. Even a `rejected` status does not
    override an explicit reason."""
    backend = _backend(foundation)
    await foundation.persistence.record_workflow(_wf("solo-3"))

    await backend._record_terminal(
        "solo-3",
        "failed",
        remote_json={
            "status": "rejected",
            "error": {"message": "invalid field 'x'"},
        },
        client=None,
    )

    record = await _error_record(foundation.persistence, "solo-3")
    assert record["retryable"] is False
    assert "failure_classification" not in record["context"]
    assert "invalid field 'x'" in record["message"]
