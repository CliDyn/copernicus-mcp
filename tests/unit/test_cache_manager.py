from __future__ import annotations

import asyncio
import errno
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def cache_manager(tmp_path: Path):
    from copernicus_mcp.cache import CacheManager
    from copernicus_mcp.persistence import SqliteBackend

    backend = SqliteBackend(tmp_path / "state.db")
    await backend.initialise()
    cache_dir = tmp_path / "cache"
    mgr = CacheManager(
        cache_directory=cache_dir,
        persistence=backend,
        size_limit_bytes=25 * 1024 * 1024,
    )
    try:
        yield mgr, cache_dir, backend
    finally:
        await backend.close()


def _make_source(tmp_path: Path, name: str, size: int = 32) -> Path:
    src = tmp_path / "src" / name
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"x" * size)
    return src


@pytest.mark.asyncio
async def test_cache_zone_layout(cache_manager) -> None:
    mgr, cache_dir, _ = cache_manager
    zone = mgr.cache_zone_for("cmems")
    today = datetime.now(UTC)
    expected = (
        cache_dir
        / "downloads"
        / "cmems"
        / f"{today.year:04d}"
        / f"{today.month:02d}"
        / f"{today.day:02d}"
    )
    assert zone == expected
    assert zone.is_dir()


@pytest.mark.asyncio
async def test_store_and_lookup_round_trip(cache_manager, tmp_path: Path) -> None:
    mgr, _, _ = cache_manager
    src = _make_source(tmp_path, "data.nc", size=128)
    target = await mgr.store_file(
        cache_key="ck-1",
        source_path=src,
        backend_id="cmems",
        content_type="application/x-netcdf",
    )
    assert target.exists()
    assert not src.exists(), "source should be moved/copied away"
    looked = await mgr.lookup_file("ck-1")
    assert looked == target
    assert looked.read_bytes() == b"x" * 128


@pytest.mark.asyncio
async def test_lookup_missing_on_disk_cleans_stale_entry(
    cache_manager, tmp_path: Path
) -> None:
    mgr, _, _ = cache_manager
    src = _make_source(tmp_path, "ghost.nc")
    target = await mgr.store_file("ck-ghost", src, "cmems", "application/x-netcdf")
    target.unlink()  # simulate filesystem rot
    assert await mgr.lookup_file("ck-ghost") is None
    # second lookup confirms entry was deleted (no auto-recreate)
    assert await mgr.lookup_file("ck-ghost") is None


@pytest.mark.asyncio
async def test_invalidate(cache_manager, tmp_path: Path) -> None:
    mgr, _, _ = cache_manager
    src = _make_source(tmp_path, "x.nc")
    await mgr.store_file("ck-inv", src, "cmems", "application/x-netcdf")
    assert await mgr.invalidate("ck-inv") is True
    assert await mgr.lookup_file("ck-inv") is None
    assert await mgr.invalidate("ck-inv") is False


@pytest.mark.asyncio
async def test_eviction_drops_oldest(tmp_path: Path) -> None:
    """Limit 25 MB, three 10 MB files → oldest evicted on 3rd store."""
    from copernicus_mcp.cache import CacheManager
    from copernicus_mcp.persistence import SqliteBackend

    backend = SqliteBackend(tmp_path / "state.db")
    await backend.initialise()
    try:
        mgr = CacheManager(
            cache_directory=tmp_path / "cache",
            persistence=backend,
            size_limit_bytes=25 * 1024 * 1024,
        )
        sizes = [10 * 1024 * 1024] * 3
        targets: list[Path] = []
        for idx, sz in enumerate(sizes):
            src = _make_source(tmp_path, f"f{idx}.nc", size=sz)
            t = await mgr.store_file(f"ck-{idx}", src, "cmems", "application/x-netcdf")
            targets.append(t)
            await asyncio.sleep(0.01)  # ensure distinct created_at
        # Oldest should be evicted.
        assert await mgr.lookup_file("ck-0") is None
        assert not targets[0].exists()
        assert await mgr.lookup_file("ck-1") is not None
        assert await mgr.lookup_file("ck-2") is not None
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_idempotent_replace(cache_manager, tmp_path: Path) -> None:
    """Storing the same cache_key twice replaces the previous file."""
    mgr, _, _ = cache_manager
    src1 = _make_source(tmp_path, "v1.nc")
    src1.write_bytes(b"first")
    t1 = await mgr.store_file("ck-rep", src1, "cmems", "application/x-netcdf")
    assert t1.read_bytes() == b"first"

    src2 = _make_source(tmp_path, "v2.nc")
    src2.write_bytes(b"second-version")
    t2 = await mgr.store_file("ck-rep", src2, "cmems", "application/x-netcdf")
    assert t2.read_bytes() == b"second-version"
    # Old file (if path differs) should not linger.
    if t1 != t2:
        assert not t1.exists()


# ---------------------------------------------------------------------------
# T-CMEMS-GET-002: store_manifest + subdirectory LRU eviction
# ---------------------------------------------------------------------------

MANIFEST_CONTENT_TYPE = "application/x.cmems-get-manifest+json"


def _populate_subdir(subdir: Path, files: dict[str, bytes]) -> Path:
    """Create ``subdir`` and write the given files. Returns the
    ``manifest.json`` path (which must be one of the entries)."""
    subdir.mkdir(parents=True, exist_ok=True)
    for relpath, content in files.items():
        f = subdir / relpath
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(content)
    return subdir / "manifest.json"


@pytest.mark.asyncio
async def test_store_manifest_registers_cache_entry(cache_manager) -> None:
    """``store_manifest`` records ONE cache entry whose ``file_path`` is
    the manifest and whose ``content_type`` flags it as a multi-file
    bundle. ``size_bytes`` reflects the *whole subdirectory* so LRU
    eviction accounts for the real disk footprint."""
    mgr, cache_dir, backend = cache_manager
    subdir = cache_dir / "bundle-a"
    manifest = _populate_subdir(
        subdir,
        {"a.nc": b"a" * 100, "b.nc": b"b" * 200, "manifest.json": b"{}"},
    )

    result = await mgr.store_manifest(
        cache_key="ck-manifest",
        manifest_path=manifest,
        data_dir=subdir,
        backend_id="cmems",
    )
    assert result == manifest.resolve()

    entry = await backend.lookup_cache_entry("file", "ck-manifest")
    assert entry is not None
    assert entry["file_path"] == str(manifest.resolve())
    assert entry["content_type"] == MANIFEST_CONTENT_TYPE
    # 100 + 200 + 2 ("{}") = 302; subdir size includes manifest itself.
    assert entry["size_bytes"] >= 302


@pytest.mark.asyncio
async def test_store_manifest_lookup_returns_manifest_path(cache_manager) -> None:
    """``lookup_file`` works uniformly for manifest entries — it returns
    the manifest path, which the caller can hand to ``read_manifest``."""
    mgr, cache_dir, _ = cache_manager
    subdir = cache_dir / "bundle-l"
    manifest = _populate_subdir(subdir, {"a.nc": b"x", "manifest.json": b"{}"})

    result = await mgr.store_manifest(
        cache_key="ck-look",
        manifest_path=manifest,
        data_dir=subdir,
        backend_id="cmems",
    )
    looked = await mgr.lookup_file("ck-look")
    assert looked == result


@pytest.mark.asyncio
async def test_store_manifest_rejects_path_outside_cache_dir(
    cache_manager, tmp_path: Path
) -> None:
    """Defence in depth: a manifest_path / data_dir outside
    ``cache_directory`` would let a buggy caller cause ``rmtree`` of
    arbitrary trees at eviction. Refuse at register time."""
    mgr, _, _ = cache_manager
    outside = tmp_path / "outside"
    manifest = _populate_subdir(
        outside, {"a.nc": b"x", "manifest.json": b"{}"}
    )
    with pytest.raises(ValueError, match="cache_directory"):
        await mgr.store_manifest(
            cache_key="ck-evil",
            manifest_path=manifest,
            data_dir=outside,
            backend_id="cmems",
        )


@pytest.mark.asyncio
async def test_store_manifest_rejects_data_dir_equal_to_root(
    cache_manager,
) -> None:
    """``data_dir == cache_directory`` would mean rmtree-ing the whole
    cache on eviction. Refuse."""
    mgr, cache_dir, _ = cache_manager
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = cache_dir / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError):
        await mgr.store_manifest(
            cache_key="ck-root",
            manifest_path=manifest,
            data_dir=cache_dir,
            backend_id="cmems",
        )


@pytest.mark.asyncio
async def test_store_manifest_rejects_missing_manifest_file(cache_manager) -> None:
    """codex round-2 MEDIUM: ``manifest_path`` not existing means a
    later ``lookup_file`` will see ``path.exists() == False`` and
    delete the DB row — leaving the data files orphaned. Fail fast
    at register time."""
    mgr, cache_dir, _ = cache_manager
    subdir = cache_dir / "no-manifest"
    subdir.mkdir(parents=True)
    (subdir / "a.nc").write_bytes(b"data")
    missing = subdir / "manifest.json"  # not written
    with pytest.raises(ValueError, match="must exist"):
        await mgr.store_manifest(
            cache_key="ck-missing-m",
            manifest_path=missing,
            data_dir=subdir,
            backend_id="cmems",
        )


@pytest.mark.asyncio
async def test_store_manifest_rejects_missing_data_dir(cache_manager) -> None:
    """codex round-2 MEDIUM: same reasoning as missing manifest —
    fail fast if the data_dir isn't actually present."""
    mgr, cache_dir, _ = cache_manager
    nonexistent = cache_dir / "ghost-dir"
    bogus_manifest = nonexistent / "manifest.json"
    with pytest.raises(ValueError, match="must exist"):
        await mgr.store_manifest(
            cache_key="ck-ghost",
            manifest_path=bogus_manifest,
            data_dir=nonexistent,
            backend_id="cmems",
        )


@pytest.mark.asyncio
async def test_store_manifest_rejects_manifest_named_other(cache_manager) -> None:
    """cr+codex round-1 MEDIUM: a caller who passes
    ``manifest_path = data_dir / "manifest.json"`` always works, but a
    bug that passes ``data_dir / "info.json"`` would still satisfy
    ``parent == data_dir`` and let lookup_file return a non-manifest
    file. Fail-fast on the well-known name."""
    mgr, cache_dir, _ = cache_manager
    subdir = cache_dir / "bundle-bad-name"
    subdir.mkdir(parents=True)
    bogus = subdir / "info.json"
    bogus.write_text("{}")
    with pytest.raises(ValueError, match="manifest.json"):
        await mgr.store_manifest(
            cache_key="ck-badname",
            manifest_path=bogus,
            data_dir=subdir,
            backend_id="cmems",
        )


@pytest.mark.asyncio
async def test_store_manifest_rejects_symlinked_data_dir(
    cache_manager,
) -> None:
    """cr round-1 MEDIUM: a ``data_dir`` symlink that points at a
    sibling bundle would, after ``resolve()``, collapse to that bundle's
    path. ``is_relative_to(cache_directory)`` would pass — and eviction
    would ``rmtree`` the sibling. Reject any symlink in the chain."""
    mgr, cache_dir, _ = cache_manager
    real_bundle = cache_dir / "real-bundle"
    real_manifest = _populate_subdir(
        real_bundle, {"a.nc": b"precious", "manifest.json": b"{}"}
    )
    await mgr.store_manifest("ck-real", real_manifest, real_bundle, "cmems")

    link_bundle = cache_dir / "link-bundle"
    link_bundle.symlink_to(real_bundle, target_is_directory=True)
    bogus_manifest = link_bundle / "manifest.json"
    with pytest.raises(ValueError, match="symlink"):
        await mgr.store_manifest(
            cache_key="ck-evil",
            manifest_path=bogus_manifest,
            data_dir=link_bundle,
            backend_id="cmems",
        )


@pytest.mark.asyncio
async def test_store_manifest_rejects_symlinked_manifest(cache_manager) -> None:
    """A symlinked ``manifest.json`` (pointing inside a real bundle)
    would, after resolution, satisfy every other check but let an
    attacker register an arbitrary file in the cache as a manifest
    entry."""
    mgr, cache_dir, _ = cache_manager
    real_bundle = cache_dir / "real-2"
    real_manifest = _populate_subdir(
        real_bundle, {"a.nc": b"x", "manifest.json": b"{}"}
    )
    await mgr.store_manifest("ck-real-2", real_manifest, real_bundle, "cmems")

    sneaky_dir = cache_dir / "sneaky"
    sneaky_dir.mkdir()
    sneaky_manifest = sneaky_dir / "manifest.json"
    sneaky_manifest.symlink_to(real_manifest)
    with pytest.raises(ValueError, match="symlink"):
        await mgr.store_manifest(
            cache_key="ck-sneaky",
            manifest_path=sneaky_manifest,
            data_dir=sneaky_dir,
            backend_id="cmems",
        )


@pytest.mark.asyncio
async def test_concurrent_stores_serialised_under_size_limit(
    tmp_path: Path,
) -> None:
    """cr+codex round-1 MEDIUM: two concurrent ``store_manifest`` calls
    each snapshot the cache state; without serialisation the post-
    enforcement total can exceed ``size_limit_bytes`` because each
    caller thinks its own snapshot is the truth. A per-manager
    ``asyncio.Lock`` around the record-and-enforce critical section
    keeps the final total at or below the limit."""
    import asyncio as _asyncio

    from copernicus_mcp.cache import CacheManager
    from copernicus_mcp.persistence import SqliteBackend

    backend = SqliteBackend(tmp_path / "state.db")
    await backend.initialise()
    try:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mgr = CacheManager(
            cache_directory=cache_dir,
            persistence=backend,
            size_limit_bytes=15 * 1024 * 1024,
        )

        async def store_bundle(name: str) -> None:
            sub = cache_dir / name
            manifest = _populate_subdir(
                sub,
                {
                    "a.nc": b"a" * (10 * 1024 * 1024),
                    "manifest.json": b"{}",
                },
            )
            await mgr.store_manifest(f"ck-{name}", manifest, sub, "cmems")

        await _asyncio.gather(
            store_bundle("c-1"),
            store_bundle("c-2"),
        )

        # Whatever survives, the post-enforcement total must respect
        # the limit. The lock guarantees serialisation, so the older
        # bundle is evicted before the newer one's enforcement runs.
        entries = [
            e
            async for e in backend.iter_cache_entries_by_namespace("file")
        ]
        total = sum(e.get("size_bytes") or 0 for e in entries)
        assert total <= 15 * 1024 * 1024
        # Exactly one bundle should remain (the one whose store wins
        # the lock last); the other is evicted in lockstep.
        assert len(entries) == 1
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_store_manifest_rejects_manifest_not_in_data_dir(
    cache_manager,
) -> None:
    """``manifest_path.parent`` must equal ``data_dir`` — otherwise
    ``lookup_file`` would resolve to a file outside the bundle and
    eviction-by-parent would tear down the wrong tree."""
    mgr, cache_dir, _ = cache_manager
    bundle = cache_dir / "bundle-m"
    bundle.mkdir(parents=True)
    sibling = cache_dir / "elsewhere"
    sibling.mkdir(parents=True)
    stray_manifest = sibling / "manifest.json"
    stray_manifest.write_text("{}")
    with pytest.raises(ValueError):
        await mgr.store_manifest(
            cache_key="ck-stray",
            manifest_path=stray_manifest,
            data_dir=bundle,
            backend_id="cmems",
        )


@pytest.mark.asyncio
async def test_store_manifest_eviction_removes_whole_subdir(
    tmp_path: Path,
) -> None:
    """LRU eviction must tear down the WHOLE subdirectory (manifest +
    data files + any sidecars) — leaving orphans would skew the size
    accounting on the next pass and waste disk."""
    from copernicus_mcp.cache import CacheManager
    from copernicus_mcp.persistence import SqliteBackend

    backend = SqliteBackend(tmp_path / "state.db")
    await backend.initialise()
    try:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mgr = CacheManager(
            cache_directory=cache_dir,
            persistence=backend,
            size_limit_bytes=25 * 1024 * 1024,
        )

        sub1 = cache_dir / "bundle-1"
        manifest1 = _populate_subdir(
            sub1,
            {"a.nc": b"a" * (10 * 1024 * 1024), "manifest.json": b"{}"},
        )
        await mgr.store_manifest("ck-1", manifest1, sub1, "cmems")
        await asyncio.sleep(0.01)

        sub2 = cache_dir / "bundle-2"
        manifest2 = _populate_subdir(
            sub2,
            {"a.nc": b"a" * (10 * 1024 * 1024), "manifest.json": b"{}"},
        )
        await mgr.store_manifest("ck-2", manifest2, sub2, "cmems")
        await asyncio.sleep(0.01)

        sub3 = cache_dir / "bundle-3"
        manifest3 = _populate_subdir(
            sub3,
            {"a.nc": b"a" * (10 * 1024 * 1024), "manifest.json": b"{}"},
        )
        await mgr.store_manifest("ck-3", manifest3, sub3, "cmems")

        # ck-1's subdir is gone (manifest + data files):
        assert await mgr.lookup_file("ck-1") is None
        assert not manifest1.exists()
        assert not sub1.exists(), "the whole subdirectory must be removed"
        # ck-2 / ck-3 survived:
        assert await mgr.lookup_file("ck-2") is not None
        assert await mgr.lookup_file("ck-3") is not None
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_store_manifest_idempotent_replace_clears_prior_subdir(
    cache_manager,
) -> None:
    """Re-storing the same cache_key replaces the manifest AND tears
    down the prior subdirectory (otherwise old data files would leak)."""
    mgr, cache_dir, _ = cache_manager
    sub1 = cache_dir / "rep-1"
    manifest1 = _populate_subdir(
        sub1, {"a.nc": b"alpha", "manifest.json": b"{}"}
    )
    await mgr.store_manifest("ck-rep", manifest1, sub1, "cmems")

    sub2 = cache_dir / "rep-2"
    manifest2 = _populate_subdir(
        sub2, {"a.nc": b"alpha2", "manifest.json": b"{}"}
    )
    await mgr.store_manifest("ck-rep", manifest2, sub2, "cmems")

    assert await mgr.lookup_file("ck-rep") == manifest2.resolve()
    assert not sub1.exists()


@pytest.mark.asyncio
async def test_invalidate_manifest_removes_subdir(cache_manager) -> None:
    """``invalidate`` works for manifest entries too — same lockstep
    semantics as eviction."""
    mgr, cache_dir, _ = cache_manager
    subdir = cache_dir / "inv-m"
    manifest = _populate_subdir(
        subdir, {"a.nc": b"x", "manifest.json": b"{}"}
    )
    await mgr.store_manifest("ck-inv-m", manifest, subdir, "cmems")

    assert await mgr.invalidate("ck-inv-m") is True
    assert not subdir.exists()
    assert await mgr.lookup_file("ck-inv-m") is None


@pytest.mark.asyncio
async def test_eviction_mixed_file_and_manifest_entries(
    tmp_path: Path,
) -> None:
    """Mix single-file and manifest entries; LRU eviction must dispatch
    correctly per entry — ``unlink`` for files, ``rmtree`` for manifests
    — without crashing the loop."""
    from copernicus_mcp.cache import CacheManager
    from copernicus_mcp.persistence import SqliteBackend

    backend = SqliteBackend(tmp_path / "state.db")
    await backend.initialise()
    try:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mgr = CacheManager(
            cache_directory=cache_dir,
            persistence=backend,
            size_limit_bytes=15 * 1024 * 1024,
        )

        # 1. single file, 10 MB (oldest).
        src = _make_source(tmp_path, "single.nc", size=10 * 1024 * 1024)
        single_target = await mgr.store_file(
            "ck-single", src, "cmems", "application/x-netcdf"
        )
        await asyncio.sleep(0.01)

        # 2. manifest bundle, 10 MB → triggers eviction of ck-single.
        sub = cache_dir / "bundle"
        manifest = _populate_subdir(
            sub,
            {"a.nc": b"a" * (10 * 1024 * 1024), "manifest.json": b"{}"},
        )
        await mgr.store_manifest("ck-bundle", manifest, sub, "cmems")

        assert not single_target.exists()
        assert await mgr.lookup_file("ck-single") is None
        assert await mgr.lookup_file("ck-bundle") == manifest.resolve()
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_cross_filesystem_fallback(
    cache_manager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If os.replace raises EXDEV, fall back to copy+delete."""
    mgr, _, _ = cache_manager
    src = _make_source(tmp_path, "xfs.nc")
    src.write_bytes(b"cross-fs content")

    real_replace = os.replace
    state = {"raised": False}

    def fake_replace(a, b):
        if not state["raised"]:
            state["raised"] = True
            raise OSError(errno.EXDEV, "fake cross-device")
        return real_replace(a, b)

    monkeypatch.setattr("copernicus_mcp.cache.manager.os.replace", fake_replace)
    target = await mgr.store_file("ck-xfs", src, "cmems", "application/x-netcdf")
    assert target.exists()
    assert target.read_bytes() == b"cross-fs content"
    assert not src.exists()
    assert state["raised"] is True
