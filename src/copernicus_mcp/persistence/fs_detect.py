"""Network-filesystem detection for the state database (T-STATEDB-001).

SQLite's WAL mode needs ``-shm``/``-wal`` side files plus POSIX locking that
NFS and Lustre do not provide reliably: after any unclean exit the next
process to open the database hangs or errors on ``PRAGMA journal_mode=WAL``
(six recorded incidents on an HPC home directory). WAL buys nothing for a
single-writer stdio server, so on a network filesystem the DELETE journal is
simply the right mode — and it keeps the cross-node submit-on-compute /
poll-on-login pattern working, which relocating the database would break.

Detection reads ``/proc/self/mounts`` (Linux — the environment where every
incident occurred). Platforms without it (macOS, some containers) answer
``None`` = unknown; the caller then attempts WAL with a loud, bounded
fallback rather than guessing.
"""

from __future__ import annotations

import re
from pathlib import Path

_PROC_MOUNTS = "/proc/self/mounts"

# Filesystem types whose locking semantics are known-unreliable for WAL.
_NETWORK_FSTYPES = frozenset(
    {
        "nfs",
        "nfs3",
        "nfs4",
        "lustre",
        "cifs",
        "smbfs",
        "smb3",
        "fuse.sshfs",
        "glusterfs",
        "beegfs",
        "ceph",
        "fuse.ceph",
        "gpfs",
    }
)


_OCTAL_ESCAPE = re.compile(r"\\([0-7]{3})")


def parse_mounts(text: str) -> list[tuple[str, str]]:
    """``(mount_point, fstype)`` pairs from ``/proc/self/mounts`` content.
    The kernel escapes space/tab/newline/backslash in mount points as
    3-digit OCTAL (``\\040`` = space); decoding via ``unicode_escape``
    would mojibake UTF-8 mount points, so only those octal escapes are
    decoded. Rows that don't parse are skipped rather than failing
    detection."""
    entries: list[tuple[str, str]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        mount_point = _OCTAL_ESCAPE.sub(
            lambda m: chr(int(m.group(1), 8)), parts[1]
        )
        entries.append((mount_point, parts[2]))
    return entries


def is_network_filesystem(
    path: Path,
    *,
    mounts_text: str | None = None,
    proc_path: str = _PROC_MOUNTS,
) -> bool | None:
    """Is ``path`` on a network filesystem? ``True``/``False`` when the mounts
    table answers it, ``None`` when there is no table to consult (non-Linux).
    Longest-prefix match, so ``/work`` on ext4 wins over ``/`` even when both
    are mounted."""
    if mounts_text is None:
        try:
            mounts_text = Path(proc_path).read_text()
        except OSError:
            return None
    # Prefix lengths are only comparable WITHIN one namespace (round-2 review,
    # both reviewers): a symlink under a long local mount pointing into a
    # short network mount lives on the network — IO follows the symlink. So:
    # find the longest-prefix mount for each candidate INDEPENDENTLY, and the
    # RESOLVED path's verdict is decisive (that is where IO actually goes);
    # the literal absolute path only answers when the resolved form matches
    # nothing in the table (e.g. host automounts rewriting a foreign path).
    entries = parse_mounts(mounts_text)

    def _best_match(candidate: str) -> tuple[str, str] | None:
        """``(mount_prefix, fstype)`` of the longest matching mount."""
        best: tuple[str, str] | None = None
        for mount_point, fstype in entries:
            normalized = mount_point.rstrip("/") or "/"
            if candidate == normalized or candidate.startswith(
                normalized if normalized == "/" else normalized + "/"
            ):
                if best is None or len(normalized) > len(best[0]) or (
                    len(normalized) == len(best[0])
                    and fstype in _NETWORK_FSTYPES
                ):
                    best = (normalized, fstype)
        return best

    expanded = path.expanduser()
    literal = str(expanded.absolute())
    try:
        resolved: str | None = str(expanded.resolve())
    except OSError:
        # Permission-protected parents / ELOOP symlink cycles: detection must
        # degrade to the literal path, never escape (closing-pass review).
        resolved = None

    resolved_match = _best_match(resolved) if resolved is not None else None
    literal_match = _best_match(literal)
    # The resolved path is where IO actually goes, so its verdict is decisive
    # — EXCEPT when the host's own automounts rewrote a foreign path so that
    # it matches nothing more specific than the catch-all root mount, while
    # the literal path has a real entry in the table being consulted
    # (closing-pass review): a zero-information root match must not shadow a
    # specific one.
    chosen = resolved_match
    if chosen is None or (
        chosen[0] == "/"
        and literal_match is not None
        and literal_match[0] != "/"
    ):
        chosen = literal_match
    if chosen is None:
        return None
    return chosen[1] in _NETWORK_FSTYPES
