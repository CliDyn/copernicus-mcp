"""SQL DDL for the persistence layer.

Iter 1 freezes this schema. Additive changes (new columns / tables / indices)
are allowed in later iterations; breaking changes require a migration story
and an entry in ``the project decision log``.

Timestamp deviation from research §11.5.6:
    The research models timestamps as INTEGER (epoch). We store them as ISO
    8601 UTC TEXT (``YYYY-MM-DDTHH:MM:SSZ``) for human-readable ``.dump``
    output and direct comparability with ``ErrorRecord.timestamp_utc``.
"""

from __future__ import annotations

WORKFLOWS_DDL = """
CREATE TABLE IF NOT EXISTS workflows (
    request_id TEXT PRIMARY KEY,
    backend_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('queued','running','successful','failed','cancelled')
    ),
    cache_key TEXT,
    request_json TEXT NOT NULL,
    response_json TEXT,
    error_record_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    parent_request_id TEXT,
    chunk_plan_json TEXT,
    chunk_plan_version INTEGER NOT NULL DEFAULT 0
);
"""

# T-CDS-CHUNK-001: parent/child columns for auto-chunked workflows. Additive,
# nullable, NO foreign key (kept plain TEXT so a fresh CREATE and an ALTER-ed
# upgrade produce byte-identical schemas; the link is logical, not FK-enforced).
# On an existing DB the CREATE above is a no-op, so the columns are added by
# ``ALTER TABLE`` in ``SqliteBackend.initialise`` — see ADDITIVE_COLUMN_MIGRATIONS.
# The 5-status CHECK is deliberately untouched (invariant 5): an unsubmitted chunk
# lives in ``chunk_plan_json``, never as a sixth status value.
ADDITIVE_COLUMN_MIGRATIONS: list[tuple[str, str, str]] = [
    ("workflows", "parent_request_id", "TEXT"),
    ("workflows", "chunk_plan_json", "TEXT"),
    # T-CDS-RESIL-006: monotonic version for compare-and-swap plan writes.
    # The per-parent asyncio lock is process-local; two processes polling one
    # parent serialise their plan mutations through this counter instead of
    # clobbering each other's whole-JSON overwrites.
    ("workflows", "chunk_plan_version", "INTEGER NOT NULL DEFAULT 0"),
]

# Index created AFTER the column migration (it references a possibly-just-added
# column), so it lives here rather than in INDICES_DDL/ALL_DDL.
POST_MIGRATION_INDICES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_workflows_parent "
    "ON workflows(parent_request_id);",
]

PROVENANCE_DDL = """
CREATE TABLE IF NOT EXISTS provenance_records (
    record_id TEXT PRIMARY KEY,
    workflow_request_id TEXT REFERENCES workflows(request_id),
    provenance_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

ACCEPTANCE_DDL = """
CREATE TABLE IF NOT EXISTS acceptance_events (
    event_id TEXT PRIMARY KEY,
    workflow_request_id TEXT REFERENCES workflows(request_id),
    accepted_at TEXT NOT NULL,
    accepted_by TEXT NOT NULL,
    request_hash TEXT NOT NULL
);
"""

CACHE_DDL = """
CREATE TABLE IF NOT EXISTS cache_entries (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    file_path TEXT,
    size_bytes INTEGER NOT NULL,
    content_type TEXT,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    PRIMARY KEY (namespace, key)
);
"""

# T-CDS-EST2-003: per-retrieval actual-size observations feeding size
# calibration. ``cost_units`` is nullable (a restart / FIFO-evicted completion
# loses the in-memory costing → row written with NULL cost, used only by
# ``--from-history`` replay). ``area_fraction`` normalises bytes (cost is
# area-independent; bytes scale with area). New table, so a plain
# ``CREATE IF NOT EXISTS`` migrates old databases on next ``initialise``.
SIZE_OBSERVATIONS_DDL = """
CREATE TABLE IF NOT EXISTS size_observations (
    observation_id TEXT PRIMARY KEY,
    backend_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    signature TEXT NOT NULL,
    cost_units REAL,
    size_bytes INTEGER NOT NULL,
    area_fraction REAL NOT NULL DEFAULT 1.0,
    request_id TEXT REFERENCES workflows(request_id),
    observed_at TEXT NOT NULL
);
"""

INDICES_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_workflows_cache_key "
    "ON workflows(cache_key);",
    # T-JOBS-RECOVERY: newest-first cross-session listing orders by created_at.
    "CREATE INDEX IF NOT EXISTS idx_workflows_created "
    "ON workflows(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_provenance_created_at "
    "ON provenance_records(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_cache_entries_last_accessed "
    "ON cache_entries(last_accessed_at);",
    "CREATE INDEX IF NOT EXISTS idx_size_obs_lookup "
    "ON size_observations(backend_id, dataset_id, signature);",
]


ALL_DDL: list[str] = [
    WORKFLOWS_DDL,
    PROVENANCE_DDL,
    ACCEPTANCE_DDL,
    CACHE_DDL,
    SIZE_OBSERVATIONS_DDL,
    *INDICES_DDL,
]
