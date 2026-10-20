"""T-CDS-FINALISE-ATOMIC Task 002a: eviction tombstones + conditional cache
maintenance.

Two ``CacheManager`` instances on two connections to one state DB stand in for
two processes: each has its own ``_cache_lock``, so only the SQL conditions
protect one process's registration from the other's stale snapshot.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from copernicus_mcp.cache import CacheManager
from copernicus_mcp.persistence import SqliteBackend

_NS = "file"


@pytest_asyncio.fixture
async def two_managers(tmp_path: Path) -> AsyncIterator[tuple[Any, ...]]:
    db = tmp_path / "state.db"
    cache_dir = tmp_path / "cache"
    a_store = SqliteBackend(db)
    await a_store.initialise()
    b_store = SqliteBackend(db)
    await b_store.initialise()
    limit = 25 * 1024 * 1024
    a = CacheManager(cache_directory=cache_dir, persistence=a_store, size_limit_bytes=limit)
    b = CacheManager(cache_directory=cache_dir, persistence=b_store, size_limit_bytes=limit)
    try:
        yield a, b, a_store, b_store, tmp_path
    finally:
        await a_store.close()
        await b_store.close()


def _src(tmp_path: Path, name: str, size: int = 64) -> Path:
    p = tmp_path / "src" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    return p


async def _store(mgr: CacheManager, tmp_path: Path, key: str, name: str, size: int = 64) -> Path:
    return await mgr.store_file(key, _src(tmp_path, name, size), backend_id="cds", content_type="application/x-netcdf")


@pytest.mark.asyncio
async def test_schema_has_cache_evictions_table(two_managers) -> None:
    _, _, a_store, _, _ = two_managers
    assert "cache_evictions" in await a_store.list_tables()


@pytest.mark.asyncio
async def test_lru_eviction_writes_lru_tombstone(two_managers) -> None:
    a, _, a_store, _, tmp_path = two_managers
    a._size_limit_bytes = 100
    first = await _store(a, tmp_path, "k1", "one.nc", size=80)
    await _store(a, tmp_path, "k2", "two.nc", size=80)  # pushes k1 out
    assert await a_store.lookup_cache_entry(_NS, "k1") is None
    assert not first.exists()
    tomb = await a_store.latest_cache_eviction(_NS, "k1")
    assert tomb is not None
    assert tomb["reason"] == "lru"
    assert tomb["file_path"] == str(first)


@pytest.mark.asyncio
async def test_lookup_of_vanished_file_writes_file_missing_tombstone(two_managers) -> None:
    a, _, a_store, _, tmp_path = two_managers
    path = await _store(a, tmp_path, "k1", "one.nc")
    path.unlink()
    assert await a.lookup_file("k1") is None
    tomb = await a_store.latest_cache_eviction(_NS, "k1")
    assert tomb is not None
    assert tomb["reason"] == "file_missing"


@pytest.mark.asyncio
async def test_registration_clears_tombstones_for_the_key(two_managers) -> None:
    a, _, a_store, _, tmp_path = two_managers
    path = await _store(a, tmp_path, "k1", "one.nc")
    path.unlink()
    assert await a.lookup_file("k1") is None
    assert await a_store.latest_cache_eviction(_NS, "k1") is not None
    await _store(a, tmp_path, "k1", "again.nc")
    assert await a_store.latest_cache_eviction(_NS, "k1") is None


@pytest.mark.asyncio
async def test_conditional_evict_refuses_a_replaced_entry(two_managers) -> None:
    a, _, a_store, _, tmp_path = two_managers
    path = await _store(a, tmp_path, "k1", "one.nc")
    entry = await a_store.lookup_cache_entry(_NS, "k1")
    assert entry is not None
    removed = await a_store.evict_cache_entry_if_matches(
        _NS, "k1", str(path) + ".other", "lru", generation=entry.get("generation")
    )
    assert removed is False
    assert await a_store.lookup_cache_entry(_NS, "k1") is not None
    assert await a_store.latest_cache_eviction(_NS, "k1") is None


@pytest.mark.asyncio
async def test_conditional_touch_refuses_a_replaced_entry(two_managers) -> None:
    a, _, a_store, _, tmp_path = two_managers
    path = await _store(a, tmp_path, "k1", "one.nc")
    before = await a_store.lookup_cache_entry(_NS, "k1")
    assert before is not None
    touched = await a_store.touch_cache_entry_if_matches(
        _NS, "k1", str(path) + ".other", "2999-01-01T00:00:00Z", generation=before.get("generation")
    )
    assert touched is False
    after = await a_store.lookup_cache_entry(_NS, "k1")
    assert after is not None
    assert after["last_accessed_at"] == before["last_accessed_at"]


@pytest.mark.asyncio
async def test_stale_lru_touch_does_not_resurrect_a_superseded_entry(two_managers, monkeypatch) -> None:
    """Process A reads K→F1, process B re-registers K→F2 (removing F1), then A
    finishes its LRU bump. 0.7.0 UPSERTed A's stale snapshot, restoring K→F1."""
    a, b, a_store, b_store, tmp_path = two_managers
    await _store(a, tmp_path, "k1", "one.nc")

    snapshot_taken = asyncio.Event()
    release = asyncio.Event()
    real_lookup = a_store.lookup_cache_entry

    async def paused_lookup(ns: str, key: str):  # type: ignore[no-untyped-def]
        entry = await real_lookup(ns, key)
        snapshot_taken.set()
        await release.wait()
        return entry

    monkeypatch.setattr(a_store, "lookup_cache_entry", paused_lookup)
    reader = asyncio.create_task(a.lookup_file("k1"))
    await snapshot_taken.wait()
    new_path = await _store(b, tmp_path, "k1", "two.nc")
    release.set()
    await reader

    entry = await b_store.lookup_cache_entry(_NS, "k1")
    assert entry is not None
    assert entry["file_path"] == str(new_path)
    assert new_path.exists()


@pytest.mark.asyncio
async def test_stale_eviction_does_not_delete_a_superseded_entry(two_managers, monkeypatch) -> None:
    """A snapshots entries for size enforcement, B re-registers the key, A's
    eviction by key must not delete B's entry or B's file."""
    a, b, a_store, b_store, tmp_path = two_managers
    await _store(a, tmp_path, "k1", "one.nc", size=80)

    snapshot_taken = asyncio.Event()
    release = asyncio.Event()
    real_iter = a_store.iter_cache_entries_by_namespace

    async def paused_iter(ns: str):  # type: ignore[no-untyped-def]
        rows = [e async for e in real_iter(ns)]
        snapshot_taken.set()
        await release.wait()
        for row in rows:
            yield row

    monkeypatch.setattr(a_store, "iter_cache_entries_by_namespace", paused_iter)
    a._size_limit_bytes = 10  # A will try to evict k1 from its snapshot
    enforcer = asyncio.create_task(a._enforce_size_limit())
    await snapshot_taken.wait()
    new_path = await _store(b, tmp_path, "k1", "two.nc", size=80)
    release.set()
    await enforcer

    entry = await b_store.lookup_cache_entry(_NS, "k1")
    assert entry is not None
    assert entry["file_path"] == str(new_path)
    assert new_path.exists()


@pytest.mark.asyncio
async def test_stale_eviction_spares_a_same_path_reregistration(two_managers, monkeypatch) -> None:
    """Codex review MEDIUM-4: CMEMS republishes a deterministic file name, so
    a replacement can reuse the SAME path. The condition must track the
    registration generation, not just the path."""
    a, b, a_store, b_store, tmp_path = two_managers
    await _store(a, tmp_path, "k1", "same.nc", size=80)

    snapshot_taken = asyncio.Event()
    release = asyncio.Event()
    real_iter = a_store.iter_cache_entries_by_namespace

    async def paused_iter(ns: str):  # type: ignore[no-untyped-def]
        rows = [e async for e in real_iter(ns)]
        snapshot_taken.set()
        await release.wait()
        for row in rows:
            yield row

    monkeypatch.setattr(a_store, "iter_cache_entries_by_namespace", paused_iter)
    a._size_limit_bytes = 10
    enforcer = asyncio.create_task(a._enforce_size_limit())
    await snapshot_taken.wait()
    new_path = await _store(b, tmp_path, "k1", "same.nc", size=80)
    release.set()
    await enforcer

    entry = await b_store.lookup_cache_entry(_NS, "k1")
    assert entry is not None and entry["file_path"] == str(new_path)
    assert new_path.exists()
