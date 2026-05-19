"""Bundled CMEMS / Marine catalogue snapshot — runtime read module.

T-CMEMS-CAT-002. Counterpart to ``_catalogue_build.py``: the build-side
module slims an SDK response and writes the snapshot; this module
loads the bundled snapshot once at first call and serves ``search()``
queries offline — no credentials, no network.

The snapshot lives at ``_data/marine.json`` (slim records,
schema-locked in ``spikes/T-CMEMS-CAT-000-describe-shape/FINDINGS.md``
Stage 3) plus ``_data/fetched_at.json`` (``{"marine": "<ISO>"}``).

Mirrors ``backends/cds/catalogue.py`` (T-CDS-003) where the patterns
transfer; differences are CMEMS-specific (single-store, ``product_id``
filter, schema is already slim on disk so there's no ``_slim_record``
projection layer here).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Module-private cache. Loaded lazily on first call; never mutated after.
_DATA_DIR: Path = Path(__file__).resolve().parent / "_data"
_FETCHED_AT_KEY: str = "marine"

_catalogue_cache: list[dict[str, Any]] | None = None
_fetched_at_cache: str | None = None


def load_catalogue() -> list[dict[str, Any]]:
    """Return the bundled catalogue as a list of slim records.

    Loaded once per process and cached. Callers must NOT mutate the
    returned list or its records — it is the module cache. The
    ``search`` helper deep-copies before returning to protect against
    accidental mutation.

    Fails fast (``ValueError``) on malformed snapshot shape (codex
    round-1 MEDIUM on T-CMEMS-CAT-002): non-list top level, or any
    non-dict record. Silently dropping bad records would let CAT-003
    serve a truncated catalogue + wrong ``total_count`` with no
    signal.
    """
    global _catalogue_cache  # noqa: PLW0603 — module-level cache is intentional
    if _catalogue_cache is None:
        path = _DATA_DIR / "marine.json"
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            # cr round-1 LOW-1: a broken install (wheel built without
            # ``_data/marine.json``) used to surface as a generic
            # ``BackendError`` from the orchestrator's "unexpected"
            # net. Raise a specific message naming the snapshot path
            # so the diagnostic points at the packaging issue.
            raise FileNotFoundError(
                f"CMEMS catalogue snapshot missing at {path}: this "
                "indicates a broken install — the bundled snapshot is "
                "shipped with the wheel. Try `pip install --force-"
                "reinstall copernicus-mcp` or refresh the snapshot via "
                "`python scripts/refresh_marine_catalogue.py`."
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"could not parse catalogue snapshot at {path}: {exc}"
            ) from exc
        if not isinstance(loaded, list):
            raise ValueError(
                f"unexpected catalogue shape in {path}: top level is "
                f"not a list (got {type(loaded).__name__})"
            )
        for i, r in enumerate(loaded):
            if not isinstance(r, dict):
                raise ValueError(
                    f"corrupt catalogue snapshot at {path}: record {i} "
                    f"is {type(r).__name__}, expected dict"
                )
        _catalogue_cache = loaded
    return _catalogue_cache


def fetched_at() -> str:
    """Return the ISO-8601 UTC timestamp the snapshot was refreshed.

    Useful for surfacing snapshot age in search responses and status
    diagnostics. The string is formatted ``YYYY-MM-DDTHH:MM:SSZ`` to
    match the refresh script's output.

    Fails fast (``ValueError``) on missing ``marine`` key, non-string
    value, or empty string (codex round-1 LOW on T-CMEMS-CAT-002):
    a ``None`` or empty value would have leaked into the search
    envelope's ``catalogue_fetched_at`` as ``"None"`` / ``""``.
    """
    global _fetched_at_cache  # noqa: PLW0603
    if _fetched_at_cache is None:
        path = _DATA_DIR / "fetched_at.json"
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or _FETCHED_AT_KEY not in loaded:
            raise ValueError(
                f"fetched_at.json missing 'marine' key: {path}"
            )
        value = loaded[_FETCHED_AT_KEY]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"fetched_at.json 'marine' value must be a non-empty "
                f"string, got {value!r} at {path}"
            )
        _fetched_at_cache = value
    return _fetched_at_cache


_SEARCHABLE_TEXT_FIELDS: tuple[str, ...] = (
    "dataset_id",
    "dataset_name",
    "title",
    "product_id",
    "product_title",
    "description",
)


def _record_matches(record: dict[str, Any], needle: str) -> bool:
    """Case-insensitive substring match against the searchable fields
    of a slim record.

    Searchable fields per FINDINGS Stage 3: ``dataset_id``,
    ``dataset_name``, ``title``, ``product_id``, ``product_title``,
    ``description``, plus the ``variables`` list.

    Per-field independence (cr round-1 LOW-2): each field is searched
    independently — the substring must be fully contained in ONE
    field to match. An earlier version joined fields with ``"\\n"``
    which let a needle bleed across two adjacent fields. Pathological
    in practice but trivial to close.
    """
    needle_lower = needle.lower()
    for field in _SEARCHABLE_TEXT_FIELDS:
        value = record.get(field)
        if value and needle_lower in str(value).lower():
            return True
    variables = record.get("variables") or []
    if isinstance(variables, list):
        for v in variables:
            if needle_lower in str(v).lower():
                return True
    return False


def _iter_matches(
    *,
    keyword: str | None,
    product_id: str | None,
) -> Iterator[dict[str, Any]]:
    """Yield records (from the module cache, not deep-copies) matching
    the given filters. Internal helper shared by ``search`` and
    ``count_matches`` so the filter logic has one source of truth.

    Filtering order: ``product_id`` exact (case-sensitive — see
    ``search`` docstring), then case-insensitive substring match
    across the searchable fields.
    """
    needle = keyword.strip() if keyword else ""
    for record in load_catalogue():
        if product_id is not None and record.get("product_id") != product_id:
            continue
        if needle and not _record_matches(record, needle):
            continue
        yield record


def search(
    *,
    keyword: str | None = None,
    product_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return slim records matching the filters.

    Filtering order:

    1. ``product_id`` — **exact case-sensitive match** against the
       slim record's ``product_id``. CMEMS product ids are
       identifiers (e.g. ``GLOBAL_ANALYSISFORECAST_PHY_001_024``)
       and ~10% of the catalogue uses mixed-case
       (``ANTARCTIC_OMI_SI_extent``). Case-folding would alter the
       inherited live-SDK semantics and risks silently widening
       results; pass the exact id from a prior ``search()`` call.
       ``product_id=None`` matches everything.
    2. ``keyword`` — case-insensitive substring across the searchable
       fields (see ``_record_matches``). ``keyword=None``, empty
       string, or whitespace-only are all treated as no-filter.
    3. ``limit`` — slice the result. ``limit=None`` or ``limit<=0``
       returns the unbounded match list. Defence-in-depth — the MCP
       schema rejects ``limit<1`` but direct callers bypass it.

    Returns independent dicts (deep-copied from the module cache) so
    callers can freely mutate without corrupting the in-process
    catalogue.

    For the unfiltered match count (e.g. to populate ``total_count``
    in a search envelope without paying the deep-copy of the full
    match list), use ``count_matches`` instead — see cr round-1 M2 /
    codex round-1 LOW-1.
    """
    effective_limit = limit if limit is not None and limit > 0 else None
    out: list[dict[str, Any]] = []
    for record in _iter_matches(keyword=keyword, product_id=product_id):
        out.append(copy.deepcopy(record))
        if effective_limit is not None and len(out) >= effective_limit:
            break
    return out


def count_matches(
    *,
    keyword: str | None = None,
    product_id: str | None = None,
) -> int:
    """Return the number of records matching the filters, WITHOUT
    slicing or deep-copying.

    CAT-003 uses this to compute ``total_count`` for the search
    envelope cheaply: a full scan of the 1251-record cache is
    millisecond-fast, while deep-copying the full match list just to
    take ``len()`` allocates ~3 MB per call.
    """
    return sum(
        1
        for _ in _iter_matches(keyword=keyword, product_id=product_id)
    )
