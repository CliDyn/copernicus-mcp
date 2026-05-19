"""Multi-file ``get`` manifest helpers (T-CMEMS-GET-002).

The cache contract for ``copernicusmarine.get`` is:

  cache_key  →  subdirectory containing N data files + 1 ``manifest.json``

The manifest IS the cache entry — the cache table stores its path and
content_type. These helpers handle building, reading, and shaping it
into the canonical ``{files: [LargeDataResult, ...]}`` envelope that
``backend.get_files`` returns to the orchestrator in T-CMEMS-GET-003.

Why this shape (vs. one cache entry per file):
- Preserves the single-cache-entry-per-cache-key invariant.
- Eviction tears down the whole subdirectory in lockstep, so we never
  leak data files for a manifest that has been removed.
- A single ``copernicus://files/{cache_key}`` URI resolves to the
  manifest; per-file URIs append ``?file=<rel>`` (see
  ``to_files_envelope``).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

MANIFEST_FILENAME = "manifest.json"
MANIFEST_CONTENT_TYPE = "application/x.cmems-get-manifest+json"
MANIFEST_SCHEMA_VERSION = "v1.0"

_MD5_CHUNK = 1024 * 1024


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while chunk := fh.read(_MD5_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def iter_data_files(directory: Path) -> list[Path]:
    """Walk ``directory`` recursively, returning data files only.

    Skips:
    - ``manifest.json`` (metadata, not data).
    - Hidden files and directories (defence in depth — toolbox
      shouldn't produce them).
    - Symlinks: ``Path.is_file()`` follows symlinks, so a symlinked
      entry inside the bundle would otherwise pull in size + md5 of a
      file outside the bundle. cr+codex round-1 MEDIUM on T-CMEMS-GET-002.

    Public because ``backend.get_files`` reuses the same skip rules to
    decide whether the SDK produced any *real* data — without this,
    an SDK that drops a session-state dotfile but no data files would
    bypass the empty-match guard (cr round-1 MEDIUM on
    T-CMEMS-GET-003).
    """
    out: list[Path] = []
    for child in sorted(directory.rglob("*")):
        if child.is_symlink():
            continue
        if not child.is_file():
            continue
        rel = child.relative_to(directory)
        parts = rel.parts
        if any(p == MANIFEST_FILENAME for p in parts):
            continue
        if any(p.startswith(".") for p in parts):
            continue
        out.append(child)
    return out


def build_manifest(
    *,
    directory: Path,
    cache_key: str,
    dataset_id: str,
    dataset_version: str | None,
) -> Path:
    """Walk ``directory`` and write ``manifest.json`` at its root.

    Raises ``ValueError`` if the directory contains no data files — a
    zero-file ``get`` is a defect, not a cacheable result.
    """
    data_files = iter_data_files(directory)
    if not data_files:
        raise ValueError(
            f"build_manifest: directory {directory!s} has no data files; "
            "refusing to register an empty cache entry"
        )

    entries: list[dict[str, Any]] = []
    for f in data_files:
        rel = f.relative_to(directory).as_posix()  # POSIX form for portability
        entries.append(
            {
                "relative_path": rel,
                "size_bytes": f.stat().st_size,
                "md5": _md5_file(f),
            }
        )

    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "cache_key": cache_key,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "files": entries,
    }
    manifest_path = directory / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return manifest_path


def read_manifest(directory: Path) -> dict[str, Any] | None:
    """Read ``manifest.json`` from ``directory``; ``None`` if absent.

    A corrupted manifest (invalid JSON) propagates the ``JSONDecodeError``
    — the cache layer treats that as a defect and lets the caller invalidate.
    """
    path = directory / MANIFEST_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def to_files_envelope(
    *,
    manifest: dict[str, Any],
    data_dir: Path,
) -> dict[str, Any]:
    """Shape a manifest into ``{"files": [LargeDataResult, ...]}``.

    Each descriptor follows the project's large-data envelope:
    ``{filepath, uri, metadata, provenance}``. ``provenance`` is left
    empty here; the backend (T-CMEMS-GET-003) splices in the per-bundle
    provenance reference.
    """
    cache_key = manifest["cache_key"]
    # codex round-1 LOW: percent-encode ``cache_key`` too. Schema only
    # rejects blank ``dataset_id`` (which ``cache_key`` embeds), so a
    # weird ID containing ``?``/``#``/``/`` would otherwise be embedded
    # raw in the URI path segment and break the resource parser.
    encoded_key = quote(cache_key, safe=":")
    files: list[dict[str, Any]] = []
    for entry in manifest["files"]:
        rel = entry["relative_path"]
        filepath = data_dir / rel
        # ``safe=""`` so ``/`` in nested paths is also percent-encoded —
        # the URI carries one logical token, not a route.
        encoded_rel = quote(rel, safe="")
        files.append(
            {
                "filepath": str(filepath),
                "uri": f"copernicus://files/{encoded_key}?file={encoded_rel}",
                "metadata": {
                    "size_bytes": entry["size_bytes"],
                    "md5": entry["md5"],
                },
                "provenance": {},
            }
        )
    return {"files": files}
