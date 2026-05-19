"""Atomic file-write helpers shared across the package.

Used by the CMEMS catalogue builder (``backends.cmems._catalogue_build``)
and the Layer 2 index store (``backends.cmems._index_store``, T-CMEMS-GET-INDEX-003).
Both call sites need the same tempfile + ``os.replace`` semantics so a
mid-write failure cannot leave a half-written cache file in place.

The underscore prefix keeps these package-private — they are NOT part
of the public API surface that end users (or tool callers) consume.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Atomically write ``payload`` to ``path``.

    Tempfile in the target directory + ``fsync`` + ``os.replace`` so a
    concurrent reader either sees the previous file's contents or the
    new file's contents, never a partial write. Best-effort cleanup of
    the tempfile on any failure between ``mkstemp`` and ``replace``.
    """
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Same atomic-write pattern as :func:`_atomic_write_bytes` but
    JSON-serialises ``payload`` first. ``sort_keys=True`` and
    ``indent=2`` produce stable, diff-friendly output for snapshots
    committed to git. ``ensure_ascii=False`` preserves UTF-8 strings.
    """
    serialised = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialised)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
