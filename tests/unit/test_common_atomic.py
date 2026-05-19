"""Focused unit tests for ``copernicus_mcp.common.atomic``.

The helpers ``_atomic_write_bytes`` / ``_atomic_write_json`` were
extracted from ``backends.cmems._catalogue_build`` in
T-CMEMS-GET-INDEX-003a so that T-CMEMS-GET-INDEX-003's ``IndexStore``
can share the same atomic-write semantics for the Parquet cache.

The catalogue's broader test suite continues to monkey-patch the
helpers via ``cb._atomic_write_json``; these tests pin the
fundamental atomic-rename guarantee in isolation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from copernicus_mcp.common.atomic import _atomic_write_bytes, _atomic_write_json


class TestAtomicWriteBytes:
    def test_writes_payload_to_target(self, tmp_path: Path) -> None:
        target = tmp_path / "out.bin"
        _atomic_write_bytes(target, b"hello\x00world")
        assert target.read_bytes() == b"hello\x00world"

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "out.bin"
        target.write_bytes(b"original")
        _atomic_write_bytes(target, b"replacement")
        assert target.read_bytes() == b"replacement"

    def test_mid_write_failure_leaves_previous_file_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Previous file MUST survive a failed atomic write — that's the
        # whole point of the tempfile + os.replace pattern.
        target = tmp_path / "out.bin"
        target.write_bytes(b"survives")

        def flaky_fdopen(fd: int, *args: object, **kwargs: object) -> object:
            os.close(fd)
            raise OSError("simulated mid-write failure")

        monkeypatch.setattr(os, "fdopen", flaky_fdopen)

        with pytest.raises(OSError, match="simulated mid-write failure"):
            _atomic_write_bytes(target, b"never lands")

        assert target.read_bytes() == b"survives"

    def test_mid_write_failure_cleans_up_tempfile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "out.bin"

        def flaky_fdopen(fd: int, *args: object, **kwargs: object) -> object:
            os.close(fd)
            raise OSError("simulated mid-write failure")

        monkeypatch.setattr(os, "fdopen", flaky_fdopen)

        with pytest.raises(OSError):
            _atomic_write_bytes(target, b"payload")

        orphans = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
        assert orphans == [], f"tempfile orphans: {orphans!r}"


class TestAtomicWriteJson:
    def test_writes_json_payload(self, tmp_path: Path) -> None:
        target = tmp_path / "out.json"
        _atomic_write_json(target, {"k": "v", "n": 1})
        loaded = json.loads(target.read_text(encoding="utf-8"))
        assert loaded == {"k": "v", "n": 1}

    def test_serialises_with_sorted_keys_and_indent(self, tmp_path: Path) -> None:
        # The existing catalogue snapshots depend on stable serialisation
        # (key order, indent) for diff-able commits. Pin that contract.
        target = tmp_path / "out.json"
        _atomic_write_json(target, {"b": 1, "a": 2})
        text = target.read_text(encoding="utf-8")
        assert text == '{\n  "a": 2,\n  "b": 1\n}'

    def test_unicode_round_trips_without_escapes(self, tmp_path: Path) -> None:
        # ``ensure_ascii=False`` keeps UTF-8 strings readable in the
        # bundled snapshots.
        target = tmp_path / "out.json"
        _atomic_write_json(target, {"region": "mediterrané"})
        assert "mediterrané" in target.read_text(encoding="utf-8")
