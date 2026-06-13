from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import aiosqlite

from copernicus_mcp.errors import BackendError, ValidationError, build_error_record
from copernicus_mcp.persistence.protocol import (
    AcceptanceEvent,
    CacheEntry,
    ProvenanceRecord,
    WorkflowRecord,
    WorkflowStatus,
)
from copernicus_mcp.persistence.schema import (
    ADDITIVE_COLUMN_MIGRATIONS,
    ALL_DDL,
    POST_MIGRATION_INDICES,
)


def _iso_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _wrap_validation(message: str, exc: BaseException) -> ValidationError:
    return ValidationError(
        message,
        record=build_error_record(
            "ValidationError",
            message=message,
            recovery_action="modify_request_parameters",
            backend_diagnostics={"db_error_class": type(exc).__name__},
        ),
    )


class SqliteBackend:
    """Async SQLite-backed persistence layer.

    A single ``aiosqlite.Connection`` is held for the lifetime of the
    backend (preserving WAL benefits). All DB operations serialise through
    one ``asyncio.Lock`` — codex T-009 review: WAL does not enable
    concurrent reads on the same connection, so locking only writes would
    risk reads observing uncommitted state.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def _apply_schema(self, conn: aiosqlite.Connection) -> None:
        """Create base tables/indices, then run additive column migrations and
        the columns-dependent indices (T-CDS-CHUNK-001). Idempotent."""
        for stmt in ALL_DDL:
            await conn.execute(stmt)
        # Additive column migration: on a fresh DB the columns already exist
        # (the CREATE includes them) so this is a no-op; on an existing DB the
        # CREATE was a no-op and we ALTER in the missing columns.
        for table, column, coltype in ADDITIVE_COLUMN_MIGRATIONS:
            cur = await conn.execute(f"PRAGMA table_info({table})")
            existing = {row[1] for row in await cur.fetchall()}
            if column not in existing:
                await conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {coltype}"
                )
        for stmt in POST_MIGRATION_INDICES:
            await conn.execute(stmt)

    async def initialise(self) -> None:
        if self._conn is not None:
            # Re-running on an open backend is fine (idempotent CREATEs).
            async with self._lock:
                await self._apply_schema(self._conn)
                await self._conn.commit()
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(self._db_path))
        # codex T-015 MEDIUM: assign ``_conn`` immediately so ``close()``
        # can clean up if any subsequent PRAGMA/DDL await fails or is
        # cancelled. try/finally + success flag avoids ``except BaseException``
        # (forbidden by the AST test in test_errors.py).
        self._conn = conn
        success = False
        try:
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA journal_mode=WAL;")
            await conn.execute("PRAGMA foreign_keys=ON;")
            await conn.execute("PRAGMA busy_timeout=5000;")
            await self._apply_schema(conn)
            await conn.commit()
            success = True
        finally:
            if not success:
                with contextlib.suppress(Exception):
                    await conn.close()
                self._conn = None

    async def close(self) -> None:
        # Hold the lock so close() cannot race an in-flight operation
        # (codex T-009 diff review).
        async with self._lock:
            if self._conn is None:
                return
            await self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # workflows
    # ------------------------------------------------------------------

    async def record_workflow(self, record: WorkflowRecord) -> None:
        async with self._lock:
            try:
                await self._conn_required().execute(
                    "INSERT INTO workflows (request_id, backend_id, operation, "
                    "status, cache_key, request_json, response_json, "
                    "error_record_json, created_at, updated_at, "
                    "parent_request_id, chunk_plan_json) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        record["request_id"],
                        record["backend_id"],
                        record["operation"],
                        record["status"],
                        record.get("cache_key"),
                        record["request_json"],
                        record.get("response_json"),
                        record.get("error_record_json"),
                        record["created_at"],
                        record["updated_at"],
                        record.get("parent_request_id"),
                        record.get("chunk_plan_json"),
                    ),
                )
                await self._conn_required().commit()
            except aiosqlite.IntegrityError as exc:
                await self._safe_rollback()
                raise _wrap_validation(
                    f"workflow CHECK/FK constraint failed: {exc}", exc
                ) from exc

    async def update_workflow_status(
        self, request_id: str, status: WorkflowStatus
    ) -> None:
        async with self._lock:
            try:
                await self._conn_required().execute(
                    "UPDATE workflows SET status = ?, updated_at = ? "
                    "WHERE request_id = ?",
                    (status, _iso_now(), request_id),
                )
                await self._conn_required().commit()
            except aiosqlite.IntegrityError as exc:
                await self._safe_rollback()
                raise _wrap_validation(
                    f"workflow status CHECK failed: {exc}", exc
                ) from exc

    async def update_workflow_status_if_pending(
        self, request_id: str, status: WorkflowStatus
    ) -> bool:
        """Conditional UPDATE: set ``status`` only if current row is in
        a non-terminal state.

        Codex T-039 round 3 MEDIUM: ``cancel()`` previously did a fetch
        + non-atomic update, so a runner committing ``successful`` between
        the two steps could be overwritten. The atomic SQL guard closes
        the TOCTOU window. Returns ``True`` if a row was updated.
        """
        async with self._lock:
            try:
                cur = await self._conn_required().execute(
                    "UPDATE workflows SET status = ?, updated_at = ? "
                    "WHERE request_id = ? AND status IN ('queued','running')",
                    (status, _iso_now(), request_id),
                )
                await self._conn_required().commit()
                return bool(cur.rowcount)
            except aiosqlite.IntegrityError as exc:
                await self._safe_rollback()
                raise _wrap_validation(
                    f"workflow status CHECK failed: {exc}", exc
                ) from exc

    async def update_workflow_error(
        self,
        request_id: str,
        status: WorkflowStatus,
        error_record_json: str,
    ) -> None:
        """Set ``status`` and ``error_record_json`` atomically.

        Async submit (T-039) needs the polling caller to see *why* a row went
        ``failed`` via ``check_status``; status alone is not enough.
        """
        async with self._lock:
            try:
                await self._conn_required().execute(
                    "UPDATE workflows SET status = ?, error_record_json = ?, "
                    "updated_at = ? WHERE request_id = ?",
                    (status, error_record_json, _iso_now(), request_id),
                )
                await self._conn_required().commit()
            except aiosqlite.IntegrityError as exc:
                await self._safe_rollback()
                raise _wrap_validation(
                    f"workflow status CHECK failed: {exc}", exc
                ) from exc

    async def update_workflow_error_if_pending(
        self,
        request_id: str,
        status: WorkflowStatus,
        error_record_json: str,
    ) -> bool:
        """Conditional ``update_workflow_error``: only writes if the
        current row is non-terminal. T-CDS-005 round-1: prevents the
        finaliser's ``failed`` commit from clobbering a row that just
        transitioned to ``cancelled`` via a concurrent ``cancel()``.
        Returns ``True`` if a row was updated.
        """
        async with self._lock:
            try:
                cur = await self._conn_required().execute(
                    "UPDATE workflows SET status = ?, error_record_json = ?, "
                    "updated_at = ? WHERE request_id = ? "
                    "AND status IN ('queued','running')",
                    (status, error_record_json, _iso_now(), request_id),
                )
                await self._conn_required().commit()
                return bool(cur.rowcount)
            except aiosqlite.IntegrityError as exc:
                await self._safe_rollback()
                raise _wrap_validation(
                    f"workflow status CHECK failed: {exc}", exc
                ) from exc

    async def lookup_workflow_by_cache_key(
        self, cache_key: str
    ) -> WorkflowRecord | None:
        async with self._lock:
            cur = await self._conn_required().execute(
                # rowid tiebreaker: created_at is second precision and two
                # force-refresh rows in the same second would otherwise be
                # ordered nondeterministically.
                "SELECT * FROM workflows WHERE cache_key = ? "
                "ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (cache_key,),
            )
            row = await cur.fetchone()
            return _row_to_workflow(row) if row else None

    async def fetch_workflow(self, request_id: str) -> WorkflowRecord | None:
        async with self._lock:
            cur = await self._conn_required().execute(
                "SELECT * FROM workflows WHERE request_id = ?", (request_id,)
            )
            row = await cur.fetchone()
            return _row_to_workflow(row) if row else None

    async def list_child_workflows(
        self, parent_request_id: str
    ) -> list[WorkflowRecord]:
        """T-CDS-CHUNK-001: child rows of a chunked parent, oldest-first."""
        async with self._lock:
            cur = await self._conn_required().execute(
                "SELECT * FROM workflows WHERE parent_request_id = ? "
                "ORDER BY created_at ASC, rowid ASC",
                (parent_request_id,),
            )
            rows = await cur.fetchall()
            return [_row_to_workflow(row) for row in rows]

    async def list_workflows(
        self,
        *,
        status: Sequence[str] | None = None,
        created_after: str | None = None,
        limit: int = 50,
    ) -> list[WorkflowRecord]:
        """T-JOBS-RECOVERY: recent workflows, newest-first, for cross-session
        discovery (a fresh agent enumerating jobs without a ``request_id``).

        ``status`` filters to the given values; ``created_after`` is a strict
        ``created_at > ?`` lower bound. ``limit`` is clamped to ``1..500`` —
        SQLite reads ``LIMIT <= 0`` as *unbounded*, so a stray non-positive
        limit must not be able to dump the whole table.
        """
        capped = max(1, min(int(limit), 500))
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            placeholders = ",".join("?" for _ in status)
            clauses.append(f"status IN ({placeholders})")
            params.extend(status)
        if created_after:
            clauses.append("created_at > ?")
            params.append(created_after)
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        params.append(capped)
        async with self._lock:
            cur = await self._conn_required().execute(
                f"SELECT * FROM workflows {where}"
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                params,
            )
            rows = await cur.fetchall()
            return [_row_to_workflow(row) for row in rows]

    async def update_chunk_plan(
        self, request_id: str, chunk_plan_json: str
    ) -> None:
        """T-CDS-CHUNK-001: persist a parent's chunk plan."""
        async with self._lock:
            await self._conn_required().execute(
                "UPDATE workflows SET chunk_plan_json = ?, updated_at = ? "
                "WHERE request_id = ?",
                (chunk_plan_json, _iso_now(), request_id),
            )
            await self._conn_required().commit()

    # ------------------------------------------------------------------
    # provenance
    # ------------------------------------------------------------------

    async def record_provenance(self, record: ProvenanceRecord) -> None:
        async with self._lock:
            try:
                await self._conn_required().execute(
                    "INSERT INTO provenance_records (record_id, "
                    "workflow_request_id, provenance_json, created_at) "
                    "VALUES (?,?,?,?)",
                    (
                        record["record_id"],
                        record.get("workflow_request_id"),
                        record["provenance_json"],
                        record["created_at"],
                    ),
                )
                await self._conn_required().commit()
            except aiosqlite.IntegrityError as exc:
                await self._safe_rollback()
                raise _wrap_validation(
                    f"provenance FK violation: {exc}", exc
                ) from exc

    async def fetch_provenance(self, record_id: str) -> ProvenanceRecord | None:
        async with self._lock:
            cur = await self._conn_required().execute(
                "SELECT * FROM provenance_records WHERE record_id = ?",
                (record_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            return cast(
                "ProvenanceRecord",
                {
                    "record_id": row["record_id"],
                    "workflow_request_id": row["workflow_request_id"],
                    "provenance_json": row["provenance_json"],
                    "created_at": row["created_at"],
                },
            )

    # ------------------------------------------------------------------
    # acceptance
    # ------------------------------------------------------------------

    async def record_acceptance(self, event: AcceptanceEvent) -> None:
        async with self._lock:
            try:
                await self._conn_required().execute(
                    "INSERT INTO acceptance_events (event_id, "
                    "workflow_request_id, accepted_at, accepted_by, "
                    "request_hash) VALUES (?,?,?,?,?)",
                    (
                        event["event_id"],
                        event.get("workflow_request_id"),
                        event["accepted_at"],
                        event["accepted_by"],
                        event["request_hash"],
                    ),
                )
                await self._conn_required().commit()
            except aiosqlite.IntegrityError as exc:
                await self._safe_rollback()
                raise _wrap_validation(
                    f"acceptance FK violation: {exc}", exc
                ) from exc

    async def lookup_acceptance(
        self, workflow_request_id: str
    ) -> AcceptanceEvent | None:
        async with self._lock:
            cur = await self._conn_required().execute(
                "SELECT * FROM acceptance_events "
                "WHERE workflow_request_id = ? "
                "ORDER BY accepted_at DESC LIMIT 1",
                (workflow_request_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            return cast(
                "AcceptanceEvent",
                {
                    "event_id": row["event_id"],
                    "workflow_request_id": row["workflow_request_id"],
                    "accepted_at": row["accepted_at"],
                    "accepted_by": row["accepted_by"],
                    "request_hash": row["request_hash"],
                },
            )

    # ------------------------------------------------------------------
    # cache_entries
    # ------------------------------------------------------------------

    async def record_cache_entry(self, entry: CacheEntry) -> None:
        async with self._lock:
            await self._conn_required().execute(
                "INSERT INTO cache_entries (namespace, key, value_json, "
                "file_path, size_bytes, content_type, created_at, "
                "last_accessed_at) VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(namespace, key) DO UPDATE SET "
                "value_json = excluded.value_json, "
                "file_path = excluded.file_path, "
                "size_bytes = excluded.size_bytes, "
                "content_type = excluded.content_type, "
                "last_accessed_at = excluded.last_accessed_at",
                (
                    entry["namespace"],
                    entry["key"],
                    entry["value_json"],
                    entry.get("file_path"),
                    entry["size_bytes"],
                    entry.get("content_type"),
                    entry["created_at"],
                    entry["last_accessed_at"],
                ),
            )
            await self._conn_required().commit()

    async def lookup_cache_entry(
        self, namespace: str, key: str
    ) -> CacheEntry | None:
        async with self._lock:
            cur = await self._conn_required().execute(
                "SELECT * FROM cache_entries WHERE namespace = ? AND key = ?",
                (namespace, key),
            )
            row = await cur.fetchone()
            return _row_to_cache_entry(row) if row else None

    async def delete_cache_entry(self, namespace: str, key: str) -> bool:
        async with self._lock:
            cur = await self._conn_required().execute(
                "DELETE FROM cache_entries WHERE namespace = ? AND key = ?",
                (namespace, key),
            )
            await self._conn_required().commit()
            return cur.rowcount > 0

    async def iter_cache_entries_by_namespace(
        self, namespace: str
    ) -> AsyncIterator[CacheEntry]:
        """Yield cache entries in ``namespace`` ordered by ``created_at``.

        Materialises the result set under the DB lock and yields outside it
        (snapshot semantics, not streaming) — keeps the lock window short
        and avoids holding the connection across caller-controlled awaits.
        Iter 1: cache size is bounded so memory pressure is acceptable.
        """
        async with self._lock:
            cur = await self._conn_required().execute(
                "SELECT * FROM cache_entries WHERE namespace = ? "
                "ORDER BY created_at",
                (namespace,),
            )
            rows = await cur.fetchall()
        for row in rows:
            yield _row_to_cache_entry(row)

    # ------------------------------------------------------------------
    # introspection helpers (for tests + diagnostics)
    # ------------------------------------------------------------------

    async def list_tables(self) -> list[str]:
        async with self._lock:
            cur = await self._conn_required().execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            rows = await cur.fetchall()
            return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _conn_required(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise BackendError(
                "SqliteBackend used before initialise() or after close().",
                record=build_error_record(
                    "BackendError",
                    message="SqliteBackend connection is not open.",
                    recovery_action="report_to_administrator",
                ),
            )
        return self._conn

    async def _safe_rollback(self) -> None:
        """Rollback that never masks the caller's intended exception.

        Codex T-009 diff review: ``rollback()`` itself can raise; if it does
        inside an ``except IntegrityError`` block, the rollback exception
        replaces our ``ValidationError``.

        T-CDS-EST2-003 bugfix: this previously called ``self._safe_rollback()``
        (itself), so the rollback never ran and the resulting ``RecursionError``
        was swallowed. Roll back the connection.
        """
        try:
            conn = self._conn
            if conn is not None:
                await conn.rollback()
        except Exception:  # noqa: BLE001 — last-ditch swallow on cleanup
            pass

    async def record_size_observation(self, observation: dict[str, Any]) -> None:
        """Insert one ``size_observations`` row (T-CDS-EST2-003).

        ``cost_units`` may be ``None`` (restart / FIFO-evicted completion).
        ``area_fraction`` defaults to 1.0 if absent.
        """
        async with self._lock:
            await self._conn_required().execute(
                "INSERT INTO size_observations (observation_id, backend_id, "
                "dataset_id, signature, cost_units, size_bytes, area_fraction, "
                "request_id, observed_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    observation["observation_id"],
                    observation["backend_id"],
                    observation["dataset_id"],
                    observation["signature"],
                    observation.get("cost_units"),
                    observation["size_bytes"],
                    observation.get("area_fraction", 1.0),
                    observation.get("request_id"),
                    observation["observed_at"],
                ),
            )
            await self._conn_required().commit()

    async def list_size_observations(
        self, backend_id: str, dataset_id: str, signature: str | None
    ) -> list[dict[str, Any]]:
        """Return observations for ``(backend_id, dataset_id)``, optionally
        narrowed to one ``signature``. Ordered oldest-first (EWMA-friendly)."""
        async with self._lock:
            if signature is None:
                cur = await self._conn_required().execute(
                    "SELECT * FROM size_observations WHERE backend_id = ? "
                    "AND dataset_id = ? ORDER BY observed_at ASC, rowid ASC",
                    (backend_id, dataset_id),
                )
            else:
                cur = await self._conn_required().execute(
                    "SELECT * FROM size_observations WHERE backend_id = ? "
                    "AND dataset_id = ? AND signature = ? "
                    "ORDER BY observed_at ASC, rowid ASC",
                    (backend_id, dataset_id, signature),
                )
            rows = await cur.fetchall()
            return [dict(row) for row in rows]


def _row_to_workflow(row: Any) -> WorkflowRecord:
    return cast(
        "WorkflowRecord",
        {
            "request_id": row["request_id"],
            "backend_id": row["backend_id"],
            "operation": row["operation"],
            "status": row["status"],
            "cache_key": row["cache_key"],
            "request_json": row["request_json"],
            "response_json": row["response_json"],
            "error_record_json": row["error_record_json"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "parent_request_id": row["parent_request_id"],
            "chunk_plan_json": row["chunk_plan_json"],
        },
    )


def _row_to_cache_entry(row: Any) -> CacheEntry:
    return cast(
        "CacheEntry",
        {
            "namespace": row["namespace"],
            "key": row["key"],
            "value_json": row["value_json"],
            "file_path": row["file_path"],
            "size_bytes": row["size_bytes"],
            "content_type": row["content_type"],
            "created_at": row["created_at"],
            "last_accessed_at": row["last_accessed_at"],
        },
    )
