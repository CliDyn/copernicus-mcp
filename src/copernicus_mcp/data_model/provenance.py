"""Provenance recorder: sidecar JSON + SQLite row for every successful retrieve.

Research §12.4. Iter 1 omits ``STAC Item`` integration (research §12.4.4 — deferred).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from copernicus_mcp.errors.sanitiser import Sanitiser
from copernicus_mcp.persistence import PersistenceBackend
from copernicus_mcp.persistence import ProvenanceRecord as _PersistenceRow

_SANITISER = Sanitiser()

_FORBID = ConfigDict(extra="forbid", frozen=True)
_SCHEMA_VERSION = "1.0"
_SIDECAR_SUFFIX = ".provenance.json"


def _iso_now() -> str:
    """Return ``YYYY-MM-DDTHH:MM:SS.fffZ`` (millisecond precision)."""
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _new_record_id() -> str:
    """``prv-YYYY-MM-DD-HHMM-{12 hex}`` (48 bits — collision-safe under load)."""
    now = datetime.now(UTC)
    return f"prv-{now:%Y-%m-%d-%H%M}-{secrets.token_hex(6)}"


class CreatedBy(BaseModel):
    model_config = _FORBID

    software: str = "copernicus-mcp"
    software_version: str
    mcp_server_id: str | None = None


class RecordHeader(BaseModel):
    model_config = _FORBID

    record_id: str
    created_at: str
    created_by: CreatedBy


class BackendBlock(BaseModel):
    model_config = _FORBID

    id: str
    provider: str
    endpoint_url: str
    api_version: str


class DatasetBlock(BaseModel):
    model_config = _FORBID

    dataset_id: str
    dataset_version: str | None = None
    dataset_part: str | None = None
    product_id: str | None = None
    service_name: str | None = None
    doi: str | None = None
    license: str | None = None
    citation_formal: str | None = None


class RequestBlock(BaseModel):
    model_config = _FORBID

    operation: str
    request_id_backend: str | None = None
    submitted_at: str
    started_at: str | None = None
    finished_at: str
    user_request: dict[str, Any]
    normalized_request: dict[str, Any]
    options_applied: dict[str, Any] = Field(default_factory=dict)


class SpatialBlock(BaseModel):
    model_config = _FORBID

    native_crs: str
    output_crs: str | None = None
    bbox_native_crs: list[float] | None = None
    bbox_epsg_4326: list[float] | None = None
    resolution_native_units: float | None = None
    resolution_unit: str | None = None


class TemporalBlock(BaseModel):
    model_config = _FORBID

    start_datetime: str
    end_datetime: str
    temporal_resolution: str | None = None


class Variable(BaseModel):
    model_config = _FORBID

    name: str
    long_name: str | None = None
    units: str | None = None


class FileEntry(BaseModel):
    model_config = _FORBID

    path: str
    size_bytes: int
    format: str | None = None
    md5: str
    sha256: str


class CostConsumed(BaseModel):
    model_config = _FORBID

    type: str
    advisory_message: str | None = None


class CacheRef(BaseModel):
    model_config = _FORBID

    cache_key: str
    cache_hit: bool


class AgentContext(BaseModel):
    model_config = _FORBID

    session_id: str | None = None
    tool_name: str | None = None
    trace_id: str | None = None


class ProvenanceRecord(BaseModel):
    """Top-level provenance record matching research §12.4.2."""

    model_config = _FORBID

    schema_version: str
    record: RecordHeader
    backend: BackendBlock
    dataset: DatasetBlock
    request: RequestBlock
    spatial: SpatialBlock | None = None
    temporal: TemporalBlock | None = None
    variables: list[Variable] = Field(default_factory=list)
    files: list[FileEntry]
    cost_consumed: CostConsumed
    source_urls: list[str] = Field(default_factory=list)
    software_versions: dict[str, str]
    cache: CacheRef
    agent_context: AgentContext | None = None


def _hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        while chunk := fh.read(8192):
            h.update(chunk)
    return h.hexdigest()


def _format_for(path: Path) -> str | None:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "nc":
        return "netcdf4"
    return suffix or None


class ProvenanceRecorder:
    def __init__(
        self,
        persistence: PersistenceBackend,
        software_versions: dict[str, str],
    ) -> None:
        self._persistence = persistence
        self._software_versions = dict(software_versions)

    async def record_successful_retrieve(
        self,
        *,
        backend: BackendBlock,
        dataset: DatasetBlock,
        request: RequestBlock,
        spatial: SpatialBlock | None,
        temporal: TemporalBlock | None,
        variables: list[Variable],
        files: list[Path],
        cost_consumed: CostConsumed,
        source_urls: list[str],
        cache: CacheRef,
        agent_context: AgentContext | None = None,
        workflow_request_id: str | None = None,
    ) -> str:
        record_id = _new_record_id()
        created_at = _iso_now()

        file_entries = await asyncio.gather(
            *(self._build_file_entry(p) for p in files)
        )

        # Defensive sanitation: even though callers shouldn't include
        # credentials in user_request/normalized_request/options_applied,
        # the project conventions makes the recorder the last line of defence.
        scrubbed_request = request.model_copy(
            update={
                "user_request": _SANITISER.sanitise(request.user_request),
                "normalized_request": _SANITISER.sanitise(
                    request.normalized_request
                ),
                "options_applied": _SANITISER.sanitise(request.options_applied),
            }
        )

        record = ProvenanceRecord(
            schema_version=_SCHEMA_VERSION,
            record=RecordHeader(
                record_id=record_id,
                created_at=created_at,
                created_by=CreatedBy(
                    software_version=self._software_versions.get(
                        "copernicus-mcp", "unknown"
                    ),
                ),
            ),
            backend=backend,
            dataset=dataset,
            request=scrubbed_request,
            spatial=spatial,
            temporal=temporal,
            variables=list(variables),
            files=list(file_entries),
            cost_consumed=cost_consumed,
            source_urls=list(source_urls),
            software_versions=dict(self._software_versions),
            cache=cache,
            agent_context=agent_context,
        )

        # codex T-013 MEDIUM: scrub the entire record at the serialisation
        # boundary, not just the request blocks. ``source_urls`` may contain
        # signed download URLs with token-shaped query params; backend /
        # dataset blocks are also free-form. Sanitise the dumped dict once.
        record_dict = _SANITISER.sanitise(record.model_dump(mode="json"))
        record_json = json.dumps(record_dict, sort_keys=False)
        pretty = json.dumps(record_dict, indent=2, sort_keys=False)

        # SQLite first: the DB row is the system-of-record. If the process
        # is cancelled or crashes during sidecar writes, an orphan row is
        # easier to reconcile than orphan sidecars without a row. Cancellation
        # mid-sidecar-gather may leave a subset of files with sidecars — that
        # is a documented best-effort property, not a correctness guarantee.
        await self._persistence.record_provenance(
            _PersistenceRow(
                record_id=record_id,
                workflow_request_id=workflow_request_id,
                provenance_json=record_json,
                created_at=created_at,
            )
        )
        await asyncio.gather(
            *(asyncio.to_thread(self._write_sidecar, p, pretty) for p in files)
        )
        return record_id

    async def _build_file_entry(self, path: Path) -> FileEntry:
        size = await asyncio.to_thread(lambda: path.stat().st_size)
        md5 = await asyncio.to_thread(_hash_file, path, "md5")
        sha256 = await asyncio.to_thread(_hash_file, path, "sha256")
        return FileEntry(
            path=str(path),
            size_bytes=size,
            format=_format_for(path),
            md5=md5,
            sha256=sha256,
        )

    @staticmethod
    def _write_sidecar(path: Path, pretty_json: str) -> None:
        sidecar = path.with_suffix(path.suffix + _SIDECAR_SUFFIX)
        # Atomic write: unique tmp + rename. Using a per-write random suffix
        # avoids the race where two concurrent writers for the same output
        # path clobber each other's ``<sidecar>.tmp``.
        tmp = sidecar.with_name(f".{sidecar.name}.{secrets.token_hex(4)}.tmp")
        tmp.write_text(pretty_json)
        tmp.replace(sidecar)
