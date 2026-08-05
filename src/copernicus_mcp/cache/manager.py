"""Filesystem cache + SQLite index for downloaded files.

Cache key namespacing: ``CacheManager`` itself is opaque. Producers (the
data model in T-012) prefix keys with ``file:``, ``metadata:``, ``search:``;
``CacheManager`` operates only in the ``"file"`` namespace of
``PersistenceBackend``.

T-CMEMS-GET-002: ``store_manifest`` registers a multi-file ``get``
bundle as ONE cache entry whose ``file_path`` points at the bundle's
``manifest.json``. The cache entry's ``content_type`` is
``MANIFEST_CONTENT_TYPE``; LRU eviction and ``invalidate`` dispatch on
this marker and ``rmtree`` the whole subdirectory in lockstep (instead
of unlinking a single file). Defence in depth: paths are resolved and
verified to live under ``cache_directory`` at register time, so a
tampered SQLite row cannot trick eviction into ``rmtree`` of an
arbitrary tree.
"""

from __future__ import annotations

import asyncio
import errno
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from copernicus_mcp.observability.logger import get_logger
from copernicus_mcp.persistence import PersistenceBackend
from copernicus_mcp.persistence.protocol import CacheEntry

logger = get_logger(__name__)

_FILE_NS = "file"

MANIFEST_CONTENT_TYPE = "application/x.cmems-get-manifest+json"
"""content_type marker for multi-file ``get`` bundles (T-CMEMS-GET-002)."""


def _iso_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _dir_size(directory: Path) -> int:
    """Recursive byte total of regular files in ``directory``.

    Skips symlinks (uses ``is_symlink()`` before ``is_file()``) so a
    symlinked entry doesn't double-count its target's size — cr+codex
    round-1 LOW/MEDIUM.
    """
    total = 0
    for child in directory.rglob("*"):
        try:
            if child.is_symlink():
                continue
            if child.is_file():
                total += child.stat().st_size
        except OSError:
            # Race with concurrent eviction or partial download — count
            # what we can; the next pass will reconcile.
            continue
    return total


def _path_chain_has_symlink(path: Path, root: Path) -> bool:
    """``True`` if any component of ``path`` from ``path`` up to but not
    including ``root`` is itself a symlink. Lets us reject paths whose
    *unresolved* form goes through a symlink even though
    ``resolve()`` of the full path would collapse cleanly inside
    ``root``.

    Returns ``False`` (defer to other checks) if the path is not
    lexically inside ``root`` — the ``is_relative_to`` check rejects
    that case with a clearer error.

    cr round-1 MEDIUM: a ``data_dir`` symlink that points at a sibling
    bundle would pass ``is_relative_to(cache_directory)`` after
    ``resolve()`` and silently make eviction ``rmtree`` the sibling.
    """
    current = path
    while True:
        if current == root:
            return False
        try:
            if current.is_symlink():
                return True
        except OSError:
            return True
        parent = current.parent
        if parent == current:
            # Walked past root without hitting it — the caller's path
            # isn't under root. Defer to the ``is_relative_to`` check.
            return False
        current = parent


class CacheManager:
    """Thin wrapper combining a filesystem zone with the persistence index."""

    def __init__(
        self,
        cache_directory: Path,
        persistence: PersistenceBackend,
        size_limit_bytes: int,
    ) -> None:
        self._cache_directory = cache_directory
        self._persistence = persistence
        self._size_limit_bytes = size_limit_bytes
        # cr+codex round-1 MEDIUM: serialise the record-and-enforce
        # critical section. Without this lock two concurrent
        # ``store_*`` calls each snapshot the cache, each thinks its
        # snapshot is the source of truth, and the post-enforcement
        # total can exceed ``size_limit_bytes``. With multi-GB
        # ``get`` bundles the overshoot is material.
        self._cache_lock = asyncio.Lock()

    def cache_zone_for(self, backend_id: str) -> Path:
        now = datetime.now(UTC)
        zone = (
            self._cache_directory
            / "downloads"
            / backend_id
            / f"{now.year:04d}"
            / f"{now.month:02d}"
            / f"{now.day:02d}"
        )
        zone.mkdir(parents=True, exist_ok=True)
        return zone

    def staging_root_for(self, backend_id: str) -> Path:
        """Day-INDEPENDENT staging area for resumable partial transfers.

        ``cache_zone_for`` partitions by calendar day — right for published
        files, wrong for state that must survive a UTC-midnight boundary: a
        transfer interrupted at 23:58 must be findable (and its leftovers
        sweepable) at 00:05, and forever after. Pure path derivation — the
        caller creates it when it first writes."""
        return self._cache_directory / "downloads" / backend_id / ".staging"

    async def store_file(
        self,
        cache_key: str,
        source_path: Path,
        backend_id: str,
        content_type: str,
    ) -> Path:
        async with self._cache_lock:
            # Drop any existing entry/file for this key first
            # (idempotent overwrite).
            existing = await self._persistence.lookup_cache_entry(
                _FILE_NS, cache_key
            )
            if existing is not None:
                self._remove_entry_files(existing)

            target = self.cache_zone_for(backend_id) / source_path.name
            try:
                os.replace(source_path, target)
            except OSError as exc:
                if exc.errno == errno.EXDEV:
                    shutil.copy2(source_path, target)
                    source_path.unlink()
                else:
                    raise

            size_bytes = target.stat().st_size
            now = _iso_now()
            await self._persistence.record_cache_entry(
                {
                    "namespace": _FILE_NS,
                    "key": cache_key,
                    "value_json": "{}",
                    "file_path": str(target),
                    "size_bytes": size_bytes,
                    "content_type": content_type,
                    "created_at": now,
                    "last_accessed_at": now,
                }
            )
            await self._enforce_size_limit()
            return target

    async def store_manifest(
        self,
        cache_key: str,
        manifest_path: Path,
        data_dir: Path,
        backend_id: str,
    ) -> Path:
        """Register a multi-file ``get`` bundle as ONE cache entry.

        Caller's responsibility:
        - ``data_dir`` exists and is fully populated (data files +
          ``manifest.json``).
        - ``manifest_path`` is ``data_dir / "manifest.json"`` (the
          name is enforced).
        - Both paths are real (no symlinked components) and live under
          ``cache_directory``.

        We verify each of those at register time and refuse otherwise —
        defence in depth so a buggy or hostile caller cannot poison the
        SQLite row with a path that would, on eviction, ``rmtree``
        outside the cache OR onto another live bundle reached through
        a symlink.
        """
        if manifest_path.name != "manifest.json":
            raise ValueError(
                f"manifest_path must be named 'manifest.json'; got "
                f"{manifest_path.name!r}"
            )
        # codex round-2 MEDIUM: a non-existent manifest or data_dir
        # would let the row be recorded, then ``lookup_file`` would
        # delete it on first access — leaving the data files (if any)
        # orphaned. Fail fast at register time.
        if not data_dir.is_dir():
            raise ValueError(f"data_dir {data_dir} must exist and be a directory")
        if not manifest_path.is_file():
            raise ValueError(
                f"manifest_path {manifest_path} must exist and be a regular file"
            )

        # cr round-1 MEDIUM: reject symlinks BEFORE ``resolve()`` so we
        # see the original path shape. A symlinked ``data_dir`` would
        # otherwise ``resolve()`` to a live sibling bundle, pass the
        # ``is_relative_to`` and ``parent ==`` checks, and let
        # eviction tear down the sibling.
        for name, p in (("manifest_path", manifest_path), ("data_dir", data_dir)):
            if _path_chain_has_symlink(p, self._cache_directory):
                raise ValueError(
                    f"{name} {p} contains a symlinked component; refusing "
                    "to register"
                )

        resolved_root = self._cache_directory.resolve(strict=False)
        resolved_data_dir = data_dir.resolve(strict=False)
        resolved_manifest = manifest_path.resolve(strict=False)

        if not resolved_data_dir.is_relative_to(resolved_root):
            raise ValueError(
                f"data_dir {data_dir} escapes cache_directory "
                f"{self._cache_directory}"
            )
        if not resolved_manifest.is_relative_to(resolved_root):
            raise ValueError(
                f"manifest_path {manifest_path} escapes cache_directory "
                f"{self._cache_directory}"
            )
        if resolved_data_dir == resolved_root:
            raise ValueError(
                "data_dir must be a subdirectory of cache_directory, "
                "not the cache root itself"
            )
        if resolved_manifest.parent != resolved_data_dir:
            raise ValueError(
                f"manifest_path {manifest_path} must reside directly in "
                f"data_dir {data_dir}"
            )

        async with self._cache_lock:
            # Idempotent: drop any prior entry for this key (tearing
            # down its subdirectory if it was a manifest).
            existing = await self._persistence.lookup_cache_entry(
                _FILE_NS, cache_key
            )
            if existing is not None:
                self._remove_entry_files(existing)

            size_bytes = _dir_size(resolved_data_dir)
            now = _iso_now()
            await self._persistence.record_cache_entry(
                {
                    "namespace": _FILE_NS,
                    "key": cache_key,
                    "value_json": "{}",
                    "file_path": str(resolved_manifest),
                    "size_bytes": size_bytes,
                    "content_type": MANIFEST_CONTENT_TYPE,
                    "created_at": now,
                    "last_accessed_at": now,
                }
            )
            await self._enforce_size_limit()
            return resolved_manifest

    async def lookup_file(self, cache_key: str) -> Path | None:
        entry = await self.lookup_entry(cache_key)
        if entry is None:
            return None
        return Path(entry["file_path"]) if entry.get("file_path") else None

    async def lookup_entry(self, cache_key: str) -> CacheEntry | None:
        """Lookup the full cache entry (path + content_type) under
        the same lock + semantics as ``lookup_file``.

        T-CMEMS-GET-006 cr round-1 MEDIUM: the file resource handler
        needs ``content_type`` to dispatch between single-file and
        manifest envelopes — without this method it would have to
        either skip the LRU bump (regression) or do two lookups
        (race). Returning the full entry keeps the LRU bump and
        dead-row cleanup centralised here.
        """
        # codex round-2 MEDIUM: serialise against ``store_*`` /
        # ``invalidate`` / ``_enforce_size_limit``. Without the lock,
        # a concurrent eviction between our row read and our
        # ``last_accessed_at`` UPSERT would resurrect the row pointing
        # at an already-deleted file.
        async with self._cache_lock:
            entry = await self._persistence.lookup_cache_entry(
                _FILE_NS, cache_key
            )
            if entry is None:
                return None
            path_str = entry.get("file_path")
            if path_str is None:
                await self._persistence.delete_cache_entry(_FILE_NS, cache_key)
                return None
            path = Path(path_str)
            if not path.exists():
                await self._persistence.delete_cache_entry(_FILE_NS, cache_key)
                return None
            # Bump last_accessed_at for LRU eviction.
            refreshed = dict(entry)
            refreshed["last_accessed_at"] = _iso_now()
            await self._persistence.record_cache_entry(refreshed)  # type: ignore[arg-type]
            return refreshed  # type: ignore[return-value]

    async def invalidate(self, cache_key: str) -> bool:
        async with self._cache_lock:
            entry = await self._persistence.lookup_cache_entry(
                _FILE_NS, cache_key
            )
            if entry is None:
                return False
            removed_file = self._remove_entry_files(entry)
            removed_entry = await self._persistence.delete_cache_entry(
                _FILE_NS, cache_key
            )
            return removed_file or removed_entry

    def _safe_manifest_data_dir(self, manifest_path_str: str) -> Path | None:
        """Return the data_dir for a manifest entry, or ``None`` if the
        stored ``file_path`` would resolve outside ``cache_directory``.

        Even though ``store_manifest`` validates at register time, we
        re-check here so a tampered SQLite row cannot redirect
        ``rmtree`` to an arbitrary tree.
        """
        try:
            manifest = Path(manifest_path_str).resolve(strict=False)
            root = self._cache_directory.resolve(strict=False)
        except OSError:
            return None
        if not manifest.is_relative_to(root):
            return None
        data_dir = manifest.parent
        if data_dir == root or root.is_relative_to(data_dir):
            return None
        return data_dir

    def _remove_entry_files(self, entry: CacheEntry) -> bool:
        """Remove the on-disk artefacts for ``entry``. Dispatches on
        ``content_type``:

        - Manifest entries → ``rmtree`` the bundle subdirectory.
        - Single-file entries → ``unlink`` the file + its provenance
          sidecar (NFA-6).

        Returns ``True`` if anything was removed, ``False`` otherwise.
        Never raises; on transient ``OSError`` we log at debug and
        move on (eviction is best-effort).
        """
        path_str = entry.get("file_path")
        if not path_str:
            return False

        if entry.get("content_type") == MANIFEST_CONTENT_TYPE:
            data_dir = self._safe_manifest_data_dir(path_str)
            if data_dir is None:
                logger.warning(
                    "refused to rmtree manifest bundle outside cache_directory",
                    extra={"path": path_str},
                )
                return False
            try:
                shutil.rmtree(data_dir)
                return True
            except FileNotFoundError:
                return False
            except OSError as exc:
                logger.debug(
                    "failed to rmtree manifest bundle",
                    extra={
                        "path": str(data_dir),
                        "error_class": type(exc).__name__,
                    },
                )
                return False

        # Single-file path.
        file_path = Path(path_str)
        removed = False
        try:
            file_path.unlink()
            removed = True
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug(
                "failed to remove cache file",
                extra={"path": path_str, "error_class": type(exc).__name__},
            )
        # NFA-6: provenance sidecar in lockstep.
        sidecar = file_path.with_suffix(file_path.suffix + ".provenance.json")
        try:
            sidecar.unlink()
            removed = True
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug(
                "failed to remove provenance sidecar",
                extra={"path": str(sidecar), "error_class": type(exc).__name__},
            )
        return removed

    async def _enforce_size_limit(self) -> None:
        entries = [
            e
            async for e in self._persistence.iter_cache_entries_by_namespace(_FILE_NS)
        ]
        total = sum(e.get("size_bytes") or 0 for e in entries)
        # Oldest first (iter is ordered by created_at).
        for entry in entries:
            if total <= self._size_limit_bytes:
                break
            self._remove_entry_files(entry)
            await self._persistence.delete_cache_entry(_FILE_NS, entry["key"])
            total -= entry.get("size_bytes") or 0
            logger.info(
                "evicted cache entry",
                extra={
                    "key": entry["key"],
                    "size_bytes": entry.get("size_bytes"),
                    "content_type": entry.get("content_type"),
                },
            )
