"""State DB on a network filesystem: fail loud, fall back, self-heal
(T-STATEDB-001).

Six recorded incidents (2026-06-20 … 07-27): the state DB defaults to the
user's home, on HPC clusters that home is NFS, and after any unclean exit the
stale ``-wal``/``-shm`` side files make the next ``PRAGMA journal_mode=WAL``
hang or error — the MCP client's connect times out and the agent proceeds
tool-less, which is the expensive failure mode. Asks, in order: (1) never
hang — fail loud with the cause and the ``COPERNICUS_MCP_STATE_DB`` remedy;
(2) don't WAL on a network filesystem (DELETE journal is fine for a
single-writer stdio server); (3) self-heal stale side files by moving them
aside (never deleting); (4) document the env var. The default path must stay
shared-FS-compatible: a consumer polls from a different node than the submitter, so
relocating the default is NOT an option.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from pathlib import Path

import pytest

_MOUNTS_LINUX_NFS_HOME = """\
/dev/mapper/vg0-root / ext4 rw,relatime 0 0
proc /proc proc rw 0 0
fileserver:/export/lab /cluster/lab nfs4 rw,relatime,vers=4.2 0 0
/dev/sda1 /work ext4 rw 0 0
"""

_MOUNTS_LUSTRE_WORK = """\
/dev/mapper/vg0-root / ext4 rw 0 0
10.0.0.1@tcp:/lustre /work lustre rw,flock 0 0
"""


# ---------------------------------------------------------------------------
# filesystem detection (pure)
# ---------------------------------------------------------------------------


def test_nfs_home_is_detected_as_network() -> None:
    from copernicus_mcp.persistence.fs_detect import is_network_filesystem

    assert (
        is_network_filesystem(
            Path("/cluster/lab/alice/.local/state/copernicus-mcp"),
            mounts_text=_MOUNTS_LINUX_NFS_HOME,
        )
        is True
    )


def test_local_path_is_not_network_and_longest_prefix_wins() -> None:
    from copernicus_mcp.persistence.fs_detect import is_network_filesystem

    # /work is ext4 even though / is also a mount point.
    assert (
        is_network_filesystem(
            Path("/work/run42/state.db"), mounts_text=_MOUNTS_LINUX_NFS_HOME
        )
        is False
    )
    # Lustre on /work IS network-class (POSIX locking unreliable).
    assert (
        is_network_filesystem(
            Path("/work/run42/state.db"), mounts_text=_MOUNTS_LUSTRE_WORK
        )
        is True
    )


def test_unreadable_mounts_means_unknown() -> None:
    """No /proc (macOS, containers): unknown — the caller attempts WAL with
    the loud-fallback path, never a silent guess."""
    from copernicus_mcp.persistence.fs_detect import is_network_filesystem

    assert (
        is_network_filesystem(Path("/anywhere"), mounts_text=None, proc_path="/nonexistent")
        is None
    )


# ---------------------------------------------------------------------------
# journal-mode selection + fallback
# ---------------------------------------------------------------------------


async def _journal_mode(backend) -> str:
    rows = await backend._conn_required().execute("PRAGMA journal_mode;")
    row = await rows.fetchone()
    return str(row[0]).lower()


@pytest.mark.asyncio
async def test_network_detection_skips_wal_entirely(tmp_path: Path, monkeypatch) -> None:
    from copernicus_mcp.persistence import SqliteBackend
    from copernicus_mcp.persistence import sqlite_backend as mod

    monkeypatch.setattr(mod, "is_network_filesystem", lambda _p: True)
    backend = SqliteBackend(tmp_path / "state.db")
    await backend.initialise()
    try:
        assert await _journal_mode(backend) == "delete"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_local_default_stays_wal(tmp_path: Path, monkeypatch) -> None:
    from copernicus_mcp.persistence import SqliteBackend
    from copernicus_mcp.persistence import sqlite_backend as mod

    monkeypatch.setattr(mod, "is_network_filesystem", lambda _p: False)
    backend = SqliteBackend(tmp_path / "state.db")
    await backend.initialise()
    try:
        assert await _journal_mode(backend) == "wal"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_wal_failure_falls_back_to_delete_and_serves(
    tmp_path: Path, monkeypatch
) -> None:
    """Ask 1+2: a locked WAL open does not hang and does not kill the server —
    it comes up in DELETE mode, fully functional (a DELETE-journal DB is fine
    for a single-writer stdio server and keeps cross-node polling working)."""
    from copernicus_mcp.persistence import SqliteBackend
    from copernicus_mcp.persistence import sqlite_backend as mod

    monkeypatch.setattr(mod, "is_network_filesystem", lambda _p: None)
    monkeypatch.setattr(mod, "_probe_journal_mode", lambda *a: "ok")
    real_open = SqliteBackend._open_configured

    async def _wal_refuses(self, journal_mode: str):
        if journal_mode == "WAL":
            raise sqlite3.OperationalError("locking protocol")
        return await real_open(self, journal_mode)

    monkeypatch.setattr(SqliteBackend, "_open_configured", _wal_refuses)
    backend = SqliteBackend(tmp_path / "state.db")
    await backend.initialise()
    try:
        assert await _journal_mode(backend) == "delete"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_stale_sidefiles_are_moved_aside_and_wal_retried(
    tmp_path: Path, monkeypatch
) -> None:
    """Ask 3: side files older than the stale threshold are RENAMED (never
    deleted) after a failed WAL open, and the retry succeeds in WAL mode."""
    from copernicus_mcp.persistence import SqliteBackend
    from copernicus_mcp.persistence import sqlite_backend as mod

    db = tmp_path / "state.db"
    for suffix in ("-wal", "-shm"):
        side = Path(str(db) + suffix)
        side.write_bytes(b"orphaned")
        old = time.time() - 3600
        os.utime(side, (old, old))

    monkeypatch.setattr(mod, "is_network_filesystem", lambda _p: None)
    monkeypatch.setattr(mod, "_probe_journal_mode", lambda *a: "ok")
    real_open = SqliteBackend._open_configured
    calls = {"wal": 0}

    async def _wal_fails_while_sidefiles_present(self, journal_mode: str):
        if journal_mode == "WAL":
            calls["wal"] += 1
            if calls["wal"] == 1:
                raise sqlite3.OperationalError("locking protocol")
        return await real_open(self, journal_mode)

    monkeypatch.setattr(
        SqliteBackend, "_open_configured", _wal_fails_while_sidefiles_present
    )
    backend = SqliteBackend(db)
    await backend.initialise()
    try:
        assert await _journal_mode(backend) == "wal"
    finally:
        await backend.close()

    # OUR code never deletes: whatever the lock probe's sqlite recovery did
    # not consume is renamed aside; no live side file remains either way.
    assert len(list(tmp_path.glob("state.db-*.stale-*"))) >= 1
    assert not Path(str(db) + "-wal").exists()
    assert not Path(str(db) + "-shm").exists()


@pytest.mark.asyncio
async def test_fresh_sidefiles_are_left_alone(tmp_path: Path, monkeypatch) -> None:
    """A recently-touched -wal may belong to a LIVE writer — moving it aside
    would discard committed transactions. Fall back to DELETE… no: DELETE
    open would also fight the live writer; the point is only that the files
    must not be renamed. The open falls through the normal fallback chain."""
    from copernicus_mcp.persistence import SqliteBackend
    from copernicus_mcp.persistence import sqlite_backend as mod

    db = tmp_path / "state.db"
    wal = Path(str(db) + "-wal")
    wal.write_bytes(b"live")

    monkeypatch.setattr(mod, "is_network_filesystem", lambda _p: None)
    real_open = SqliteBackend._open_configured

    async def _wal_refuses(self, journal_mode: str):
        if journal_mode == "WAL":
            raise sqlite3.OperationalError("locking protocol")
        return await real_open(self, journal_mode)

    monkeypatch.setattr(SqliteBackend, "_open_configured", _wal_refuses)
    backend = SqliteBackend(db)
    try:
        await backend.initialise()
        await backend.close()
    except Exception:  # noqa: BLE001 — a live-writer conflict may refuse; fine
        pass

    # OUR code must not have moved it aside; sqlite itself may legitimately
    # recover/discard an invalid hot WAL during the DELETE open, so only the
    # absence of rename artefacts is asserted.
    assert not list(tmp_path.glob("*.stale-*"))
    del wal


@pytest.mark.asyncio
async def test_every_mode_failing_raises_loud_canonical_error(
    tmp_path: Path, monkeypatch
) -> None:
    """Ask 1: the terminal failure names the path and the env-var remedy in a
    canonical error — never a hang, never a bare traceback."""
    from copernicus_mcp.errors import BackendError
    from copernicus_mcp.persistence import SqliteBackend
    from copernicus_mcp.persistence import sqlite_backend as mod

    monkeypatch.setattr(mod, "is_network_filesystem", lambda _p: None)

    async def _always_refuses(self, journal_mode: str):
        raise sqlite3.OperationalError("locking protocol")

    monkeypatch.setattr(SqliteBackend, "_open_configured", _always_refuses)
    backend = SqliteBackend(tmp_path / "state.db")

    with pytest.raises(BackendError) as exc:
        await backend.initialise()

    record = exc.value.error_record
    assert record.error_subclass == "state_db_unavailable"
    assert str(tmp_path / "state.db") in record.message
    assert "COPERNICUS_MCP_STATE_DB" in record.message


@pytest.mark.asyncio
async def test_hung_wal_pragma_times_out_instead_of_hanging_forever(
    tmp_path: Path, monkeypatch
) -> None:
    """The recorded failure mode is a HANG on the WAL pragma, not an error.
    A bounded timeout converts it into the fallback chain."""
    from copernicus_mcp.persistence import SqliteBackend
    from copernicus_mcp.persistence import sqlite_backend as mod

    monkeypatch.setattr(mod, "is_network_filesystem", lambda _p: None)
    real_open = SqliteBackend._open_configured

    async def _wal_hangs(self, journal_mode: str):
        if journal_mode == "WAL":
            await asyncio.Event().wait()  # never returns
        return await real_open(self, journal_mode)

    monkeypatch.setattr(SqliteBackend, "_open_configured", _wal_hangs)
    backend = SqliteBackend(tmp_path / "state.db", pragma_timeout_seconds=0.2)

    await asyncio.wait_for(backend.initialise(), timeout=5.0)
    try:
        assert await _journal_mode(backend) == "delete"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_wedged_wal_probe_never_touches_aiosqlite(
    tmp_path: Path, monkeypatch
) -> None:
    """codex M4: aiosqlite's worker thread is NOT a daemon — a WAL pragma
    wedged inside it outlives close() and blocks interpreter exit (a one-shot
    CLI would hang at shutdown). The journal mode is therefore probed first in
    a plain-sqlite DAEMON thread we own: if the probe wedges, only that
    disposable thread is abandoned and aiosqlite is never touched in WAL mode."""
    import threading

    from copernicus_mcp.persistence import SqliteBackend
    from copernicus_mcp.persistence import sqlite_backend as mod

    monkeypatch.setattr(mod, "is_network_filesystem", lambda _p: None)
    hang = threading.Event()
    real_connect = mod._probe_connect

    def _wedging_connect(path, *a, **kw):
        conn = real_connect(path, *a, **kw)

        class _Wedge:
            def execute(self, sql, *args):
                if "journal_mode=WAL" in sql:
                    hang.wait(timeout=30)  # simulate the NFS lock wedge
                return conn.execute(sql, *args)

            def close(self) -> None:
                conn.close()

        return _Wedge()

    monkeypatch.setattr(mod, "_probe_connect", _wedging_connect)
    backend = SqliteBackend(tmp_path / "state.db", pragma_timeout_seconds=0.2)
    try:
        await asyncio.wait_for(backend.initialise(), timeout=5.0)
        assert await _journal_mode(backend) == "delete"
        # The wedged probe thread is a DAEMON, so it cannot block process exit.
        probe_threads = [
            t for t in threading.enumerate() if t.name == "state-db-probe"
        ]
        assert probe_threads, "the wedged probe thread should still be alive"
        assert all(t.daemon for t in probe_threads)
    finally:
        hang.set()
        await backend.close()


@pytest.mark.asyncio
async def test_network_path_also_self_heals_stale_sidefiles(
    tmp_path: Path, monkeypatch
) -> None:
    """local M3: the six-incident scenario IS the network path — orphaned WAL
    sidefiles on NFS. Converting the db to DELETE journal needs the stale WAL
    recovered, so the self-heal must run there too, not only on the WAL
    branch."""
    from copernicus_mcp.persistence import SqliteBackend
    from copernicus_mcp.persistence import sqlite_backend as mod

    db = tmp_path / "state.db"
    for suffix in ("-wal", "-shm"):
        side = Path(str(db) + suffix)
        side.write_bytes(b"orphaned")
        old = time.time() - 3600
        os.utime(side, (old, old))

    monkeypatch.setattr(mod, "is_network_filesystem", lambda _p: True)
    monkeypatch.setattr(mod, "_probe_journal_mode", lambda *a: "ok")
    real_open = SqliteBackend._open_configured
    calls = {"delete": 0}

    async def _delete_fails_once(self, journal_mode: str):
        if journal_mode == "DELETE":
            calls["delete"] += 1
            if calls["delete"] == 1:
                raise sqlite3.OperationalError("locking protocol")
        return await real_open(self, journal_mode)

    monkeypatch.setattr(SqliteBackend, "_open_configured", _delete_fails_once)
    backend = SqliteBackend(db)
    await backend.initialise()
    try:
        assert await _journal_mode(backend) == "delete"
    finally:
        await backend.close()
    assert len(list(tmp_path.glob("state.db-*.stale-*"))) >= 1
    assert not Path(str(db) + "-wal").exists()
    assert not Path(str(db) + "-shm").exists()


def test_mount_points_with_octal_escapes_and_symlinks() -> None:
    """local LOW: /proc mounts escape spaces as \\040 (octal, latin-1 safe);
    unicode_escape mojibakes UTF-8. And a symlinked home must resolve to its
    real NFS mount to be detected."""
    from copernicus_mcp.persistence.fs_detect import parse_mounts

    entries = dict(parse_mounts(
        "srv:/exp /mnt/nfs\\040share nfs4 rw 0 0\n"
        "/dev/sda1 /données ext4 rw 0 0\n"
    ))
    assert entries["/mnt/nfs share"] == "nfs4"
    assert entries["/données"] == "ext4"


def test_symlinked_path_resolves_to_its_real_mount(tmp_path: Path) -> None:
    from copernicus_mcp.persistence.fs_detect import is_network_filesystem

    real = tmp_path / "real-nfs-home"
    real.mkdir()
    link = tmp_path / "link-home"
    link.symlink_to(real)
    mounts = f"srv:/exp {real} nfs4 rw 0 0\n/dev/sda1 / ext4 rw 0 0\n"

    assert is_network_filesystem(link / "state-dir", mounts_text=mounts) is True


@pytest.mark.asyncio
async def test_heal_declines_while_a_live_holder_keeps_the_write_lock(
    tmp_path: Path, monkeypatch
) -> None:
    """local M4 (medium): a live-but-IDLE holder does not touch its -wal, so
    mtime alone conflates idle with dead — and healing under a live writer
    discards its committed-but-uncheckpointed transactions. Before renaming,
    probe the write lock; while someone holds it, decline."""
    from copernicus_mcp.persistence import SqliteBackend
    from copernicus_mcp.persistence import sqlite_backend as mod

    db = tmp_path / "state.db"
    holder = sqlite3.connect(db)
    holder.execute("PRAGMA journal_mode=WAL;")
    holder.execute("CREATE TABLE t (x)")
    holder.commit()
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("INSERT INTO t VALUES (1)")
    # -wal exists now; make it LOOK stale while the holder is alive.
    old = time.time() - 3600
    os.utime(str(db) + "-wal", (old, old))

    monkeypatch.setattr(mod, "is_network_filesystem", lambda _p: None)
    backend = SqliteBackend(db, pragma_timeout_seconds=2.0)
    try:
        healed = await backend._heal_stale_sidefiles()
        assert healed is False
        assert Path(str(db) + "-wal").exists()
        assert not list(tmp_path.glob("*.stale-*"))
    finally:
        holder.rollback()
        holder.close()


def test_symlink_into_shorter_network_prefix_is_still_network(tmp_path: Path) -> None:
    """codex/local round-2 MEDIUM: prefix lengths must not be compared across
    the absolute and resolved namespaces. A symlink under a LONG local mount
    pointing into a SHORT network mount is network — IO follows the symlink."""
    from copernicus_mcp.persistence.fs_detect import is_network_filesystem

    local_mount = tmp_path / "very" / "long" / "local-home-mount"
    nfs_root = tmp_path / "n"
    (nfs_root / "u").mkdir(parents=True)
    local_mount.mkdir(parents=True)
    link = local_mount / "state"
    link.symlink_to(nfs_root / "u")
    mounts = (
        f"/dev/sda1 {local_mount} ext4 rw 0 0\n"
        f"srv:/exp {nfs_root} nfs4 rw 0 0\n"
        "/dev/sda2 / ext4 rw 0 0\n"
    )

    assert is_network_filesystem(link / "db-dir", mounts_text=mounts) is True


def test_resolve_failure_falls_back_to_the_literal_path(monkeypatch) -> None:
    """closing pass MEDIUM: resolve() can raise (permission-protected parent,
    ELOOP symlink cycles) — detection must fall back to the literal path, not
    escape with an exception out of the persistence layer."""
    from copernicus_mcp.persistence.fs_detect import is_network_filesystem

    monkeypatch.setattr(
        Path, "resolve", lambda self, strict=False: (_ for _ in ()).throw(PermissionError("nope"))
    )
    mounts = "srv:/exp /cluster/lab nfs4 rw 0 0\n/dev/sda1 / ext4 rw 0 0\n"
    assert (
        is_network_filesystem(Path("/cluster/lab/alice"), mounts_text=mounts)
        is True
    )


def test_root_only_resolution_defers_to_a_specific_literal_match(tmp_path: Path, monkeypatch) -> None:
    """closing pass MEDIUM: when host automounts rewrite a foreign path so its
    resolved form matches ONLY the catch-all root mount, the literal path's
    specific match must answer — otherwise a foreign NFS path reads local."""
    from copernicus_mcp.persistence.fs_detect import is_network_filesystem

    monkeypatch.setattr(
        Path, "resolve", lambda self, strict=False: Path("/local/rewritten/elsewhere")
    )
    mounts = (
        "/dev/sda1 / ext4 rw 0 0\n"
        "srv:/exp /foreign/nfs nfs4 rw 0 0\n"
    )
    assert (
        is_network_filesystem(Path("/foreign/nfs/user/state"), mounts_text=mounts)
        is True
    )
