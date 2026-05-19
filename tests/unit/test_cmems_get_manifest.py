"""T-CMEMS-GET-002: manifest helpers for multi-file ``get`` results.

The cache contract for ``copernicusmarine.get`` is:

  cache_key  →  subdirectory containing N data files + 1 manifest.json

The manifest IS the cache entry (the cache table stores its path). These
helpers handle building, reading, and shaping it into the canonical
``{files: [LargeDataResult, ...]}`` envelope that ``backend.get_files``
will return in T-CMEMS-GET-003.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------


def _write_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_build_manifest_creates_manifest_json(tmp_path: Path) -> None:
    """``build_manifest`` writes ``manifest.json`` at the root of the data
    directory. Returns the manifest path."""
    from copernicus_mcp.backends.cmems._get_manifest import build_manifest

    _write_file(tmp_path / "a.nc", b"alpha")
    _write_file(tmp_path / "b.nc", b"bravo")

    manifest_path = build_manifest(
        directory=tmp_path,
        cache_key="cmems:get:ds:abc123",
        dataset_id="ds",
        dataset_version=None,
    )
    assert manifest_path == tmp_path / "manifest.json"
    assert manifest_path.exists()


def test_build_manifest_records_each_file_with_size_and_md5(tmp_path: Path) -> None:
    """Each file gets ``relative_path``, ``size_bytes``, and ``md5`` so
    the cache layer can later verify integrity and resolve per-file URIs."""
    from copernicus_mcp.backends.cmems._get_manifest import build_manifest

    _write_file(tmp_path / "a.nc", b"alpha")
    _write_file(tmp_path / "b.nc", b"bravo")

    manifest_path = build_manifest(
        directory=tmp_path,
        cache_key="cmems:get:ds:abc",
        dataset_id="ds",
        dataset_version=None,
    )
    data = json.loads(manifest_path.read_text())
    files = {entry["relative_path"]: entry for entry in data["files"]}
    assert files["a.nc"]["size_bytes"] == len(b"alpha")
    assert files["b.nc"]["size_bytes"] == len(b"bravo")
    assert files["a.nc"]["md5"] == hashlib.md5(b"alpha").hexdigest()
    assert files["b.nc"]["md5"] == hashlib.md5(b"bravo").hexdigest()


def test_build_manifest_recurses_into_subdirectories(tmp_path: Path) -> None:
    """The toolbox preserves CMEMS native directory structure
    (``YYYY/MM/file.nc`` etc). Manifest must walk recursively and store
    POSIX-style relative paths so the manifest is portable across OSes."""
    from copernicus_mcp.backends.cmems._get_manifest import build_manifest

    _write_file(tmp_path / "1990" / "01" / "obs.nc", b"deep")
    _write_file(tmp_path / "top.nc", b"shallow")

    manifest_path = build_manifest(
        directory=tmp_path,
        cache_key="ck",
        dataset_id="ds",
        dataset_version=None,
    )
    rels = {f["relative_path"] for f in json.loads(manifest_path.read_text())["files"]}
    assert "1990/01/obs.nc" in rels
    assert "top.nc" in rels


def test_build_manifest_does_not_list_itself(tmp_path: Path) -> None:
    """``manifest.json`` is metadata, not a data file — it must not appear
    in its own ``files`` list, or the round trip would diverge."""
    from copernicus_mcp.backends.cmems._get_manifest import build_manifest

    _write_file(tmp_path / "a.nc", b"x")

    manifest_path = build_manifest(
        directory=tmp_path,
        cache_key="ck",
        dataset_id="ds",
        dataset_version=None,
    )
    rels = [f["relative_path"] for f in json.loads(manifest_path.read_text())["files"]]
    assert "manifest.json" not in rels


def test_build_manifest_persists_logical_fields(tmp_path: Path) -> None:
    """``cache_key``, ``dataset_id``, ``dataset_version``, and
    ``schema_version`` are written so a manifest read from disk is
    self-describing (cache eviction or external inspection doesn't need
    the database)."""
    from copernicus_mcp.backends.cmems._get_manifest import build_manifest

    _write_file(tmp_path / "a.nc", b"x")
    manifest_path = build_manifest(
        directory=tmp_path,
        cache_key="cmems:get:ds:abc",
        dataset_id="ds",
        dataset_version="202411",
    )
    data = json.loads(manifest_path.read_text())
    assert data["cache_key"] == "cmems:get:ds:abc"
    assert data["dataset_id"] == "ds"
    assert data["dataset_version"] == "202411"
    assert data["schema_version"]  # non-empty


def test_build_manifest_rejects_empty_directory(tmp_path: Path) -> None:
    """A ``get`` that produced zero files is a defect (the SDK should
    have raised). Refuse to register a cache entry for nothing."""
    from copernicus_mcp.backends.cmems._get_manifest import build_manifest

    with pytest.raises(ValueError):
        build_manifest(
            directory=tmp_path,
            cache_key="ck",
            dataset_id="ds",
            dataset_version=None,
        )


# ---------------------------------------------------------------------------
# read_manifest
# ---------------------------------------------------------------------------


def test_read_manifest_round_trip(tmp_path: Path) -> None:
    from copernicus_mcp.backends.cmems._get_manifest import (
        build_manifest,
        read_manifest,
    )

    _write_file(tmp_path / "a.nc", b"alpha")
    build_manifest(
        directory=tmp_path,
        cache_key="ck",
        dataset_id="ds",
        dataset_version=None,
    )
    parsed = read_manifest(tmp_path)
    assert parsed["cache_key"] == "ck"
    assert len(parsed["files"]) == 1
    assert parsed["files"][0]["relative_path"] == "a.nc"


def test_read_manifest_missing_returns_none(tmp_path: Path) -> None:
    from copernicus_mcp.backends.cmems._get_manifest import read_manifest

    assert read_manifest(tmp_path) is None


# ---------------------------------------------------------------------------
# to_files_envelope
# ---------------------------------------------------------------------------


def test_to_files_envelope_yields_per_file_descriptor(tmp_path: Path) -> None:
    """The envelope returned to the orchestrator is
    ``{"files": [{filepath, uri, metadata, provenance}, ...]}`` — one
    descriptor per data file. the large-data invariant forbids inlining the
    file bytes; each entry is a path + metadata."""
    from copernicus_mcp.backends.cmems._get_manifest import (
        build_manifest,
        read_manifest,
        to_files_envelope,
    )

    _write_file(tmp_path / "a.nc", b"alpha")
    _write_file(tmp_path / "sub" / "b.nc", b"bravo")
    build_manifest(
        directory=tmp_path,
        cache_key="cmems:get:ds:abc",
        dataset_id="ds",
        dataset_version=None,
    )
    manifest = read_manifest(tmp_path)
    assert manifest is not None

    envelope = to_files_envelope(manifest=manifest, data_dir=tmp_path)
    assert set(envelope) == {"files"}
    descriptors = {d["filepath"]: d for d in envelope["files"]}
    assert str(tmp_path / "a.nc") in descriptors
    assert str(tmp_path / "sub" / "b.nc") in descriptors

    a_desc = descriptors[str(tmp_path / "a.nc")]
    assert a_desc["uri"].startswith("copernicus://files/cmems:get:ds:abc")
    assert "file=a.nc" in a_desc["uri"]
    assert a_desc["metadata"]["size_bytes"] == len(b"alpha")
    assert a_desc["metadata"]["md5"]
    # Provenance is filled in by the caller (T-CMEMS-GET-003); the helper
    # returns an empty dict so the orchestrator can splice the reference.
    assert a_desc["provenance"] == {}


def test_build_manifest_skips_symlinked_files(tmp_path: Path) -> None:
    """cr+codex round-1 MEDIUM: ``Path.is_file()`` follows symlinks. A
    symlinked entry inside the bundle would silently pull in size + md5
    of a file outside the bundle, breaking the cache invariant that
    the manifest describes only what lives in ``data_dir``."""
    from copernicus_mcp.backends.cmems._get_manifest import build_manifest

    target = tmp_path.parent / "outside.nc"
    target.write_bytes(b"outside-bytes")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "real.nc").write_bytes(b"real")
    (bundle / "link.nc").symlink_to(target)

    manifest_path = build_manifest(
        directory=bundle,
        cache_key="ck",
        dataset_id="ds",
        dataset_version=None,
    )
    rels = {f["relative_path"] for f in json.loads(manifest_path.read_text())["files"]}
    assert rels == {"real.nc"}


def test_to_files_envelope_uri_quotes_cache_key(tmp_path: Path) -> None:
    """codex round-1 LOW: weird dataset IDs containing ``?``/``#``/
    ``/`` would otherwise be embedded raw in the URI and break the
    resource parser. Percent-encode the whole cache_key path segment."""
    from copernicus_mcp.backends.cmems._get_manifest import (
        build_manifest,
        read_manifest,
        to_files_envelope,
    )

    _write_file(tmp_path / "a.nc", b"x")
    build_manifest(
        directory=tmp_path,
        cache_key="cmems:get:weird?id#bad:abc",
        dataset_id="weird?id#bad",
        dataset_version=None,
    )
    manifest = read_manifest(tmp_path)
    assert manifest is not None
    uri = to_files_envelope(manifest=manifest, data_dir=tmp_path)["files"][0]["uri"]
    assert "?id" not in uri.split("?file=")[0]  # ``?`` only as query separator
    assert "#bad" not in uri  # ``#`` would be a URI fragment otherwise
    assert "weird%3Fid%23bad" in uri


def test_to_files_envelope_uri_quotes_unsafe_chars(tmp_path: Path) -> None:
    """The per-file URI encodes the relative path safely so a filename
    with spaces or ``?`` cannot break the query-string parser."""
    from copernicus_mcp.backends.cmems._get_manifest import (
        build_manifest,
        read_manifest,
        to_files_envelope,
    )

    _write_file(tmp_path / "weird name?.nc", b"x")
    build_manifest(
        directory=tmp_path,
        cache_key="ck",
        dataset_id="ds",
        dataset_version=None,
    )
    manifest = read_manifest(tmp_path)
    assert manifest is not None
    envelope = to_files_envelope(manifest=manifest, data_dir=tmp_path)
    assert "weird%20name%3F.nc" in envelope["files"][0]["uri"]
