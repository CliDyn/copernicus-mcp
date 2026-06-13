from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal, NotRequired, Protocol, TypedDict

WorkflowStatus = Literal["queued", "running", "successful", "failed", "cancelled"]
"""Five canonical workflow statuses (the project conventions invariant #5).

``existing_success`` is **not** a status — it is a response flag emitted by
the orchestrator (``{"status": "successful", "cache_hit": true}``). It must
never reach the database.
"""


class WorkflowRecord(TypedDict):
    request_id: str
    backend_id: str
    operation: str
    status: WorkflowStatus
    cache_key: str | None
    request_json: str
    response_json: str | None
    error_record_json: str | None
    created_at: str
    updated_at: str
    # T-CDS-CHUNK-001: parent/child linkage for auto-chunked workflows.
    # NotRequired so existing single-request constructors that omit them stay
    # valid; ``_row_to_workflow`` always populates them from the DB.
    parent_request_id: NotRequired[str | None]
    chunk_plan_json: NotRequired[str | None]


class ProvenanceRecord(TypedDict):
    record_id: str
    workflow_request_id: str | None
    provenance_json: str
    created_at: str


class AcceptanceEvent(TypedDict):
    event_id: str
    workflow_request_id: str | None
    accepted_at: str
    accepted_by: str
    request_hash: str


class CacheEntry(TypedDict):
    namespace: str
    key: str
    value_json: str
    file_path: str | None
    size_bytes: int
    content_type: str | None
    created_at: str
    last_accessed_at: str


class PersistenceBackend(Protocol):
    """Async durable store for workflows, provenance, acceptance, and cache."""

    async def initialise(self) -> None: ...
    async def close(self) -> None: ...

    async def record_workflow(self, record: WorkflowRecord) -> None: ...
    async def update_workflow_status(
        self, request_id: str, status: WorkflowStatus
    ) -> None: ...
    async def update_workflow_error(
        self,
        request_id: str,
        status: WorkflowStatus,
        error_record_json: str,
    ) -> None: ...
    async def update_workflow_status_if_pending(
        self, request_id: str, status: WorkflowStatus
    ) -> bool: ...
    async def update_workflow_error_if_pending(
        self,
        request_id: str,
        status: WorkflowStatus,
        error_record_json: str,
    ) -> bool: ...
    async def lookup_workflow_by_cache_key(
        self, cache_key: str
    ) -> WorkflowRecord | None: ...
    async def fetch_workflow(self, request_id: str) -> WorkflowRecord | None: ...

    # T-CDS-CHUNK-001: parent/child auto-chunk linkage.
    async def list_child_workflows(
        self, parent_request_id: str
    ) -> list[WorkflowRecord]: ...

    # T-JOBS-RECOVERY: cross-session discovery — recent workflows, newest-first.
    async def list_workflows(
        self,
        *,
        status: Sequence[str] | None = None,
        created_after: str | None = None,
        limit: int = 50,
    ) -> list[WorkflowRecord]: ...
    async def update_chunk_plan(
        self, request_id: str, chunk_plan_json: str
    ) -> None: ...

    # T-CDS-EST2-003: size calibration observations.
    async def record_size_observation(
        self, observation: dict[str, Any]
    ) -> None: ...
    async def list_size_observations(
        self, backend_id: str, dataset_id: str, signature: str | None
    ) -> list[dict[str, Any]]: ...

    async def record_provenance(self, record: ProvenanceRecord) -> None: ...
    async def fetch_provenance(self, record_id: str) -> ProvenanceRecord | None: ...

    async def record_acceptance(self, event: AcceptanceEvent) -> None: ...
    async def lookup_acceptance(
        self, workflow_request_id: str
    ) -> AcceptanceEvent | None: ...

    async def record_cache_entry(self, entry: CacheEntry) -> None: ...
    async def lookup_cache_entry(
        self, namespace: str, key: str
    ) -> CacheEntry | None: ...
    async def delete_cache_entry(self, namespace: str, key: str) -> bool: ...
    def iter_cache_entries_by_namespace(
        self, namespace: str
    ) -> AsyncIterator[CacheEntry]: ...
