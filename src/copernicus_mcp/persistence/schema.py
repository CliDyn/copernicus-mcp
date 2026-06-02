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
    updated_at TEXT NOT NULL
);
"""

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

INDICES_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_workflows_cache_key "
    "ON workflows(cache_key);",
    "CREATE INDEX IF NOT EXISTS idx_provenance_created_at "
    "ON provenance_records(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_cache_entries_last_accessed "
    "ON cache_entries(last_accessed_at);",
]


ALL_DDL: list[str] = [
    WORKFLOWS_DDL,
    PROVENANCE_DDL,
    ACCEPTANCE_DDL,
    CACHE_DDL,
    *INDICES_DDL,
]
