"""Slim-record builder for the CMEMS catalogue snapshot.

Imported by the dev/ops refresh script
(``scripts/refresh_marine_catalogue.py``) and by unit tests. NOT
imported by ``CmemsBackend`` at runtime — runtime reads the bundled
``_data/marine.json`` that this builder produces.

Schema lock and rationale: see
``spikes/T-CMEMS-CAT-000-describe-shape/FINDINGS.md`` Stage 3.

The leading underscore in the module name signals "private to the
catalogue build pipeline" — agents and users should not import this
directly. The module ships in the wheel anyway (Hatch packs the whole
``src/copernicus_mcp`` tree); the footprint is a couple of KB.
"""

from __future__ import annotations

import json
import ntpath
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from copernicus_mcp.common.atomic import _atomic_write_bytes, _atomic_write_json

_DESCRIPTION_CAP: int = 500
_ELLIPSIS: str = "..."
# Variable bboxes set to all-zeros are sentinels — common for OMI-style
# datasets whose variables don't have a meaningful spatial extent. Skip
# them when aggregating so the dataset-level extent is None rather than
# a bogus 0-area box. Deliberate divergence from describe-path (see
# FINDINGS §Future for the harmonisation follow-up).
_ZERO_SENTINEL_BBOX: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)


def slim_marine_record(product: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    """Project a CMEMS describe() product+dataset pair into the slim
    catalogue row.

    The slim row preserves every field the live ``_map_dataset`` envelope
    emits (so search consumers see a byte-compatible shape) plus a few
    cheap discovery additions (``product_title``, ``description``,
    ``doi``, ``versions``). Variable bboxes are aggregated into a
    dataset-level ``spatial_extent`` via the same union strategy the
    describe-path uses, skipping ``[0,0,0,0]`` sentinels.
    """
    (
        service_types,
        variables,
        var_bboxes,
        var_time_ranges,
    ) = _walk_latest_version(dataset)
    versions = [
        str(v.get("label"))
        for v in (dataset.get("versions") or [])
        if isinstance(v, dict) and v.get("label") is not None
    ]

    return {
        "dataset_id": dataset.get("dataset_id"),
        "dataset_name": dataset.get("dataset_name"),
        # FINDINGS Stage 3: ``title`` mirrors ``dataset_name`` so consumers
        # don't have to know the equivalence the live mapper relied on.
        "title": dataset.get("dataset_name"),
        "product_id": product.get("product_id"),
        "product_title": product.get("title"),
        "description": _truncate_description(product.get("description") or ""),
        "doi": dataset.get("digital_object_identifier") or product.get("digital_object_identifier"),
        "service_types": service_types,
        "variables": variables,
        "versions": versions,
        "spatial_extent": _union_bboxes(var_bboxes),
        # T-CMEMS-HIER-001: previously always ``dataset.get("temporal_extent")``
        # which is ``None`` in real SDK output. Now aggregated from
        # variable.coordinates[time] same way bboxes are aggregated —
        # this puts real time coverage on slim records.
        "temporal_extent": _aggregate_temporal_extent(var_time_ranges),
    }


def _walk_latest_version(
    dataset: dict[str, Any],
) -> tuple[list[str], list[str], list[list[float]], list[tuple[int, int]]]:
    """Walk the SDK shape ``versions[].parts[].services[]`` and return
    ``(service_types, variables, var_bboxes, var_time_ranges)`` for the
    latest version only (lex-sorted by label, matching ``_latest_version``
    in ``backends/cmems/backend.py``).

    Returns empty lists when the dataset has no versions or no
    variables / services — safe for slim records of OMI-style datasets
    whose top-level fields exist but versions are empty.

    ``var_time_ranges`` is a list of ``(start_ms, end_ms)`` tuples — one
    per variable that surfaces a ``coordinate_id == "time"`` coordinate
    with a populated range. T-CMEMS-HIER-001: lets ``slim_marine_record``
    aggregate dataset-level ``temporal_extent`` without a second walk.
    """
    versions = [v for v in (dataset.get("versions") or []) if isinstance(v, dict)]
    if not versions:
        return ([], [], [], [])

    latest = max(
        versions,
        key=lambda v: str(v.get("label") or ""),
    )

    service_types: list[str] = []
    variables: list[str] = []
    var_bboxes: list[list[float]] = []
    var_time_ranges: list[tuple[int, int]] = []

    for part in latest.get("parts") or []:
        if not isinstance(part, dict):
            continue
        for svc in part.get("services") or []:
            if not isinstance(svc, dict):
                continue
            st = svc.get("service_name") or svc.get("name")
            if isinstance(st, str) and st not in service_types:
                service_types.append(st)
            for var in svc.get("variables") or []:
                if not isinstance(var, dict):
                    continue
                name = var.get("short_name") or var.get("name")
                if isinstance(name, str) and name not in variables:
                    variables.append(name)
                bbox = var.get("bbox")
                if (
                    isinstance(bbox, (list, tuple))
                    and len(bbox) == 4
                    and tuple(bbox) != _ZERO_SENTINEL_BBOX
                ):
                    try:
                        var_bboxes.append([float(x) for x in bbox])
                    except (TypeError, ValueError):
                        # Skip a variable with a non-numeric bbox slot
                        # rather than crash the whole refresh.
                        continue
                time_range = _extract_time_range(var.get("coordinates"))
                if time_range is not None:
                    var_time_ranges.append(time_range)
    return (service_types, variables, var_bboxes, var_time_ranges)


def _is_numeric(v: Any) -> bool:
    """Numeric check that excludes ``bool`` (Python ``bool`` is a
    subclass of ``int``; ``isinstance(True, int)`` is ``True``).
    cr round-1 MEDIUM: prevents a stray ``True``/``False`` in
    coordinate fields from producing a bogus ``(0,1)`` extent.
    """
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _extract_time_range(
    coordinates: Any,
) -> tuple[int, int] | None:
    """Find the ``time`` coordinate inside a variable's
    ``coordinates`` list and return ``(start_ms, end_ms)`` or
    ``None``. T-CMEMS-HIER-001.

    Prefers ``minimum_value`` / ``maximum_value`` on the time
    coordinate. Falls back to ``min(values)`` / ``max(values)`` when
    those convenience fields are null but the ``values`` list is
    non-empty (the empirical OMI Sea Ice Extent shape from the
    T-CMEMS-CAT-000 smoke).

    Defensive against unexpected SDK shapes (cr round-1 MEDIUM):
    - rejects ``bool`` values (subclass of int);
    - rejects all-zero sentinels (mirrors the ``[0,0,0,0]`` bbox
      sentinel filter — a ``(0,0)`` time range maps to
      ``1970-01-01`` which is never real data);
    - rejects ``values`` lists that mix numeric and non-numeric
      entries (returns None so the dataset gets the
      ``no_temporal_extent`` quality flag, rather than silently
      slicing to the numeric subset).
    """
    if not isinstance(coordinates, list):
        return None
    for coord in coordinates:
        if not isinstance(coord, dict):
            continue
        if coord.get("coordinate_id") != "time":
            continue
        lo = coord.get("minimum_value")
        hi = coord.get("maximum_value")
        if _is_numeric(lo) and _is_numeric(hi):
            if lo == 0 and hi == 0:
                # Zero-sentinel — mirrors the bbox [0,0,0,0] filter.
                return None
            return (int(lo), int(hi))
        values = coord.get("values")
        if isinstance(values, list) and values:
            # All-or-nothing: if any non-numeric (e.g. ISO string)
            # appears, treat the whole list as suspect and skip.
            if not all(_is_numeric(v) for v in values):
                return None
            if all(v == 0 for v in values):
                return None
            return (int(min(values)), int(max(values)))
        return None
    return None


def _aggregate_temporal_extent(
    var_time_ranges: list[tuple[int, int]],
) -> dict[str, str] | None:
    """Union of per-variable ``(start_ms, end_ms)`` tuples into a
    dataset-level ``{"start": ISO, "end": ISO}`` dict. T-CMEMS-HIER-001.

    Returns ``None`` when no variable surfaced a time coordinate
    (mirrors the defensive shape of ``_union_bboxes``).
    """
    if not var_time_ranges:
        return None
    start_ms = min(r[0] for r in var_time_ranges)
    end_ms = max(r[1] for r in var_time_ranges)
    return {
        "start": _ms_to_iso(start_ms),
        "end": _ms_to_iso(end_ms),
    }


def _ms_to_iso(ms: int) -> str:
    """Format milliseconds-since-epoch (UTC) as ``YYYY-MM-DDTHH:MM:SSZ``.

    CMEMS coordinates are documented as ``milliseconds since
    1970-01-01 00:00:00Z (no leap seconds)`` — direct division by
    1000 is sound.
    """
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _union_bboxes(bboxes: list[list[float]]) -> dict[str, float] | None:
    """Union variable bboxes into a dataset-level ``spatial_extent`` dict.

    Same shape the describe-path ``_union_bboxes`` returns
    (``backends/cmems/backend.py:1526-1541``). Slim path filters
    ``[0,0,0,0]`` sentinels before calling this; describe-path
    currently does not (filed in FINDINGS §Future for harmonisation).
    """
    if not bboxes:
        return None
    return {
        "min_lon": min(b[0] for b in bboxes),
        "min_lat": min(b[1] for b in bboxes),
        "max_lon": max(b[2] for b in bboxes),
        "max_lat": max(b[3] for b in bboxes),
    }


def _truncate_description(text: str) -> str:
    """Cap a product description at ``_DESCRIPTION_CAP`` chars, cut at a
    word boundary, append ``"..."`` only when truncation actually
    happens.

    FINDINGS Stage 3 + LOW-2: pass-through verbatim when ``len(text)
    <= 500``. When longer, find the last word boundary at or before
    char index ``500 - 3 = 497`` and append ``"..."`` — total length
    is then at most 500 chars.
    """
    if len(text) <= _DESCRIPTION_CAP:
        return text
    budget = _DESCRIPTION_CAP - len(_ELLIPSIS)
    # Find the last whitespace at-or-before the budget. Falls back to a
    # hard cut at ``budget`` if there is no whitespace in the prefix
    # (degenerate one-word descriptions).
    head = text[: budget + 1]
    last_space = head.rfind(" ")
    if last_space <= 0:
        cut = text[:budget].rstrip()
    else:
        cut = text[:last_space].rstrip()
    return cut + _ELLIPSIS


# ---------------------------------------------------------------------------
# Catalogue-level helpers (walk full SDK response + write atomic snapshot)
# ---------------------------------------------------------------------------


def build_slim_catalogue(sdk_response: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten an SDK ``describe()`` response to a list of slim records,
    one per dataset, in upstream traversal order.

    Empty ``products`` list returns ``[]`` (a real edge case for a
    future SDK call against a user with zero entitlements). Products
    without a ``datasets`` list are skipped silently — a single
    malformed upstream entry should not abort the refresh.

    Raises ``ValueError`` when ``products`` is missing or not a list
    (codex round-1 M1): a malformed-but-keyed response like
    ``{"products": "error"}`` or ``{"products": null}`` previously
    produced an empty slim list, which then overwrote the bundled
    snapshot. Surface this as an explicit failure instead so the
    refresh script aborts before touching disk.
    """
    products = sdk_response.get("products")
    if not isinstance(products, list):
        raise ValueError(
            f"SDK response 'products' must be a list, got "
            f"{type(products).__name__}; refusing to build slim catalogue"
        )
    records: list[dict[str, Any]] = []
    for product in products:
        if not isinstance(product, dict):
            continue
        datasets = product.get("datasets")
        if not isinstance(datasets, list):
            continue
        for dataset in datasets:
            if not isinstance(dataset, dict):
                continue
            records.append(slim_marine_record(product, dataset))
    return records


def assert_no_credential_leak(records: list[dict[str, Any]]) -> None:
    """Run the project ``Sanitiser`` over the slim records and assert
    the post-sanitise JSON is byte-identical to the pre-sanitise JSON.

    Belt-and-braces against the very-unlikely case that
    ``copernicusmarine.describe()`` echoes a credential into a record
    field. If the sanitiser would have stripped anything, raise
    ``RuntimeError`` so the refresh script exits non-zero and the
    existing snapshot stays untouched.

    Implementation note: import the sanitiser lazily so this module
    stays cheap to import from a context (e.g. a test fixture) that
    doesn't need the full ``errors`` subpackage on the import path.
    """
    from copernicus_mcp.errors.sanitiser import Sanitiser

    sanitiser = Sanitiser()
    pre = json.dumps(records, sort_keys=True, ensure_ascii=False)
    cleaned = sanitiser.sanitise(records)
    post = json.dumps(cleaned, sort_keys=True, ensure_ascii=False)
    if pre != post:
        raise RuntimeError(
            "credential-shaped pattern detected in slim records; "
            "refresh aborted before writing snapshot. Inspect "
            "the offending product/dataset and re-run."
        )


def _validate_snapshot_key(name: str) -> None:
    """Reject mapping keys that aren't plain filenames.

    codex round-1 PR #88 MEDIUM: ``data_dir / name`` follows POSIX
    rules — an absolute path replaces ``data_dir``, ``..`` escapes
    upward, and any embedded separator lets the rollback's
    ``os.replace``/``unlink`` touch unrelated files. Reject early
    with a ``ValueError`` carrying the offending key.

    Round-2 widened to defense-in-depth on the same class:
    - reject Windows drive-relative shapes like ``C:foo.json`` (codex
      round-2 MEDIUM) via ``ntpath.splitdrive``;
    - reject embedded NUL/control characters and backslash, both of
      which the POSIX separator check missed (cr round-2 MEDIUM);
    - reject leading/trailing whitespace, which would also produce a
      surprising on-disk filename.

    None of these are exploitable through the current call sites
    (the refresh script uses hard-coded keys), but the function
    exists precisely to give a clean ValueError up front before any
    I/O — so it should not delegate to a downstream ``open()`` or
    ``Path`` for half of the rejection logic.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"write_snapshot keys must be non-empty strings, got {name!r}")
    # Embedded NUL / control chars never appear in a legitimate
    # filename and would either crash or surprise the FS layer.
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in name):
        raise ValueError(f"write_snapshot keys must not contain control characters; got {name!r}")
    # Whitespace fence-posts produce filenames that look identical to
    # their stripped form in most UIs.
    if name != name.strip():
        raise ValueError(
            f"write_snapshot keys must not have leading/trailing whitespace; got {name!r}"
        )
    # Catch Windows drive-relative names (``C:foo.json`` → drive='C:'
    # in ntpath, which would resolve relative to the drive's CWD on
    # Windows rather than under ``data_dir``). ``ntpath.splitdrive``
    # works on POSIX too and returns ``('', name)`` for plain names.
    if ntpath.splitdrive(name)[0]:
        raise ValueError(f"write_snapshot keys must not include a drive letter; got {name!r}")
    if (
        os.path.isabs(name)
        or "/" in name
        or "\\" in name  # Windows separator; also rejects literal backslash on POSIX.
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
        or os.path.normpath(name) != name
        or name in (".", "..")
    ):
        raise ValueError(
            f"write_snapshot keys must be plain filenames "
            f"(no separators, no '..', no absolute paths); got {name!r}"
        )


def write_snapshot(
    files: Mapping[str, Any],
    *,
    data_dir: Path,
) -> None:
    """Atomically write every ``{filename: payload}`` entry in
    ``files`` into ``data_dir`` as JSON.

    Each file uses POSIX ``os.replace`` for atomicity, but a series
    of N writes is not jointly atomic on POSIX. The invariant this
    helper provides is *post-failure consistency*, not *read-time
    consistency*:

    1. Take an in-memory backup of every target file that already
       exists in ``data_dir``.
    2. Replace each target via tempfile + ``os.replace``, in mapping
       insertion order, recording which paths were successfully
       written.
    3. If any write raises, walk the recorded set of completed writes
       and restore each one: backups go back via the same atomic
       tempfile+``os.replace`` pattern as the forward write (codex
       round-2 MEDIUM — an in-place rewrite can truncate on a second
       failure), and paths that had no previous version are deleted.
       Re-raise the original error so the caller exits non-zero.

    Outcome: on any failure, the on-disk state for every key in
    ``files`` is byte-identical to its pre-call content. The
    "previous snapshot left untouched" guarantee scales to any number
    of bundled manifests, and the restore itself is atomic.

    What this does NOT provide: a concurrent reader between forward
    writes 1 and 2 can observe a new file N and an old file M
    together (codex round-1 PR #88 MEDIUM). The bundled snapshot is
    refreshed by a dev/ops script that runs ahead of release, so a
    single-writer assumption is fine here; do not reuse this helper
    in a server context without serialising callers.

    Keys must be plain filenames — no ``/``, no ``..``, no absolute
    paths (codex round-1 PR #88 MEDIUM). Anything else escapes
    ``data_dir`` and would let the rollback overwrite or delete
    unrelated files.
    """
    if not files:
        return
    for name in files:
        _validate_snapshot_key(name)
    data_dir.mkdir(parents=True, exist_ok=True)

    backups: dict[Path, bytes | None] = {}
    for name in files:
        path = data_dir / name
        backups[path] = path.read_bytes() if path.exists() else None

    written: list[Path] = []
    try:
        for name, payload in files.items():
            path = data_dir / name
            _atomic_write_json(path, payload)
            written.append(path)
    except Exception:
        # Best-effort rollback. Restore each successfully-written
        # path to its pre-call content; failures inside the rollback
        # are swallowed so the caller sees the original write
        # exception, which is the actionable one.
        for path in written:
            backup = backups.get(path)
            try:
                if backup is not None:
                    _atomic_write_bytes(path, backup)
                else:
                    path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


