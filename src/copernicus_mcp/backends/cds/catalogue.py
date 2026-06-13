"""Bundled CDS / ADS / EWDS catalogue snapshot (T-CDS-003).

Reads STAC collection lists from
``src/copernicus_mcp/backends/cds/_data/{cds,ads,ewds}.json``. The
snapshots are refreshed manually via ``scripts/refresh_cds_catalogue.py``
and committed to git — runtime never hits the network for catalogue
discovery.

Why a bundled snapshot:
- ``cdsapi`` 0.7.7 has zero programmatic catalogue access (research §6.7.1).
- ``ecmwf-datastores-client`` 0.4.2 (incubating successor) does, but
  exposes a raw ``get_collections()`` that we'd cache locally anyway.
- Total slim payload across all 3 stores is ~135 KB / ~34k tokens —
  fits in any modern LLM context. RAG / vector store overkill.
- Manual refresh is auditable: ops runs the script, commits a diff,
  no surprise content drift in production.

Two-stage select (intended LLM flow):
1. ``search()`` returns slim catalogue (id, title, first-paragraph
   description, keywords). LLM picks N candidates.
2. ``describe(dataset_id)`` returns the full STAC record for one
   dataset (extent, summaries, providers, etc.) for final selection.

Cross-store lookup: ``describe`` searches all three stores so the
agent does not need to remember which store a dataset_id came from.
``search`` results are tagged with ``store`` so the agent can route
subsequent operations correctly.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from copernicus_mcp.errors import NotFoundError, ValidationError
from copernicus_mcp.errors.records import build_error_record

# Module-private cache. Loaded lazily on first call; never mutated after.
_DATA_DIR: Path = Path(__file__).resolve().parent / "_data"
_STORES: tuple[str, ...] = ("cds", "ads", "ewds")
_DESCRIPTION_CHAR_CAP: int = 500


_catalogue_cache: dict[str, list[dict[str, Any]]] | None = None
_fetched_at_cache: dict[str, str] | None = None
# T-CDS-015 (Layer A): bundled constraints (empty-inputs) per dataset.
# Merged across stores at load time; describe() reads from the merged
# dict so cross-store lookup stays O(1). Cache invalidates only on
# explicit monkeypatch (tests) or fresh process.
_constraints_cache: dict[str, dict[str, Any]] | None = None


def _load_store_records(store: str) -> list[dict[str, Any]]:
    path = _DATA_DIR / f"{store}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    collections = payload.get("collections", [])
    if not isinstance(collections, list):
        raise ValueError(
            f"unexpected catalogue shape in {path}: 'collections' is not a list"
        )
    return [c for c in collections if isinstance(c, dict)]


def load_catalogue() -> dict[str, list[dict[str, Any]]]:
    """Return the in-memory catalogue, loading from disk on first call.

    The returned dict maps ``store_id -> [stac_record, ...]``. Callers
    must NOT mutate the returned structure — it is the module cache.
    """
    global _catalogue_cache  # noqa: PLW0603 — module-level cache is intentional
    if _catalogue_cache is None:
        _catalogue_cache = {store: _load_store_records(store) for store in _STORES}
    return _catalogue_cache


def load_constraints() -> dict[str, dict[str, Any]]:
    """Return ``{dataset_id: {field: [valid_values, ...]}}`` from the
    bundled constraints snapshot, merged across all three stores.

    Missing bundle files are tolerated — describes() simply omits
    ``available_inputs`` when a dataset has no entry. T-CDS-015.
    """
    global _constraints_cache  # noqa: PLW0603
    if _constraints_cache is None:
        merged: dict[str, dict[str, Any]] = {}
        for store in _STORES:
            path = _DATA_DIR / f"{store}_constraints.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            for dataset_id, constraints in payload.items():
                # cr round-1 IMPORTANT-2: use setdefault so first-write-
                # wins. describe() iterates _STORES in the same order
                # and returns on first match, so a collision (no real
                # case today, but future-proof) keeps the SAME store's
                # STAC record + constraints paired.
                if isinstance(constraints, dict):
                    merged.setdefault(dataset_id, constraints)
        _constraints_cache = merged
    return _constraints_cache


def fetched_at() -> dict[str, str]:
    """Return ``{store: ISO8601-UTC-timestamp}`` when the snapshot was last
    refreshed. Useful for surfacing staleness in status / docs."""
    global _fetched_at_cache  # noqa: PLW0603
    if _fetched_at_cache is None:
        path = _DATA_DIR / "fetched_at.json"
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"fetched_at.json is not a dict: {path}")
        _fetched_at_cache = {str(k): str(v) for k, v in loaded.items()}
    return dict(_fetched_at_cache)


def _slim_record(record: dict[str, Any], store: str) -> dict[str, Any]:
    """Project a STAC record down to the fields the LLM needs to PICK."""
    raw_desc = record.get("description") or ""
    first_para = (
        str(raw_desc).strip().split("\n\n", 1)[0][:_DESCRIPTION_CHAR_CAP]
    )
    return {
        "id": record.get("id"),
        "title": record.get("title"),
        "description": first_para,
        "keywords": list(record.get("keywords") or []),
        "store": store,
    }


# T-CDS-020: bbox / time / variable filter primitives ----------------------


def _normalise_lon(lon: float) -> float:
    """Wrap a longitude into [-180, 180]. CDS catalogue records use a
    mix of 0..360 (most ERA5 entries) and ±180 (CERRA, satellite, etc.);
    normalising before comparison removes the foot-gun for callers."""
    # (((lon + 180) % 360) - 180) maps any real lon into (-180, 180].
    wrapped = ((lon + 180.0) % 360.0) - 180.0
    return -180.0 if wrapped == -180.0 else wrapped


def _bbox_lon_intervals(west: float, east: float) -> list[tuple[float, float]]:
    """Express a bbox's longitude span as one or two ±180-normalised
    intervals. A bbox that crosses the antimeridian becomes two pieces.

    Examples:
      (170, 190) → [(170, 180), (-180, -170)]   (crosses dateline once wrapped)
      (-10, 30)  → [(-10, 30)]
      (0, 360)   → [(-180, 180)]                (global)
    """
    if east - west >= 360.0:
        return [(-180.0, 180.0)]
    nw = _normalise_lon(west)
    ne = _normalise_lon(east)
    if nw <= ne:
        return [(nw, ne)]
    # Crosses the antimeridian after normalisation.
    return [(nw, 180.0), (-180.0, ne)]


def _intervals_overlap(
    a: tuple[float, float], b: tuple[float, float]
) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _bbox_4d_from_stac(raw_bbox: Any) -> tuple[float, float, float, float] | None:
    """Project a STAC ``extent.spatial.bbox[i]`` to a 4-element WSEN
    tuple. STAC spec allows two shapes per bbox entry:
      - 4D: ``[west, south, east, north]``
      - 6D: ``[west, south, min_elev, east, north, max_elev]``
    Returns ``None`` if neither shape matches (caller treats as
    unknown coverage). Codex retro PR #124 MEDIUM."""
    if not isinstance(raw_bbox, list):
        return None
    if len(raw_bbox) == 4:
        try:
            return (
                float(raw_bbox[0]),
                float(raw_bbox[1]),
                float(raw_bbox[2]),
                float(raw_bbox[3]),
            )
        except (TypeError, ValueError):
            return None
    if len(raw_bbox) == 6:
        try:
            return (
                float(raw_bbox[0]),  # west
                float(raw_bbox[1]),  # south
                float(raw_bbox[3]),  # east
                float(raw_bbox[4]),  # north
            )
        except (TypeError, ValueError):
            return None
    return None


def _record_intersects_bbox(
    record: dict[str, Any], query: tuple[float, float, float, float]
) -> bool:
    """STAC-formatted record bbox(es) ∩ query bbox (w, s, e, n).

    Per STAC spec, ``extent.spatial.bbox`` is a LIST of bbox entries
    (multi-region datasets can have multiple). Each entry is either
    4D ``[w,s,e,n]`` or 6D ``[w,s,min_elev,e,n,max_elev]``. The record
    matches if ANY of its bboxes intersects the query (codex retro
    PR #124 MEDIUM — prior implementation only read ``raw[0]`` and
    rejected 6D shapes).

    Datasets without a usable ``extent.spatial.bbox`` are treated as
    "unknown coverage" and INCLUDED — better to surface a candidate
    the LLM can verify than silently hide it."""
    extent = record.get("extent") or {}
    spatial = extent.get("spatial") or {}
    raw = spatial.get("bbox")
    if not (isinstance(raw, list) and raw):
        return True
    qw, qs, qe, qn = query
    qry_lons = _bbox_lon_intervals(qw, qe)
    any_usable = False
    for entry in raw:
        coords = _bbox_4d_from_stac(entry)
        if coords is None:
            continue
        any_usable = True
        rw, rs, re_, rn = coords
        if not _intervals_overlap((rs, rn), (qs, qn)):
            continue
        # Longitude: handle 0..360 records and ±180 queries (or vice
        # versa) by normalising both sides to ±180 and checking
        # interval overlap.
        rec_lons = _bbox_lon_intervals(rw, re_)
        if any(_intervals_overlap(r, q) for r in rec_lons for q in qry_lons):
            return True
    # Any bbox entries were structurally valid but none intersected →
    # genuine miss. No usable entries → unknown coverage → INCLUDE.
    return not any_usable


def _parse_iso(value: str) -> datetime:
    """Parse an ISO 8601 string to a tz-aware UTC ``datetime``.

    Round-1 cr H1: catalogue records carry tz-aware timestamps
    (``"...+00:00"``); naive query inputs would crash the
    cross-comparison in ``_record_overlaps_time`` with raw ``TypeError``.
    Normalise here: strip whitespace, accept trailing ``Z``, and treat
    any naive result as UTC. Bare dates (``"2010-01-01"``) are also
    accepted (parsed as midnight UTC)."""
    cleaned = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _record_overlaps_time(
    record: dict[str, Any], query: tuple[datetime, datetime]
) -> bool:
    """STAC-formatted record temporal extent ∩ query window.

    Records may have ``None`` for start or end (open-ended datasets that
    are still being updated). ``None`` is treated as ±∞ accordingly.
    Records without a usable extent are INCLUDED (same policy as bbox)."""
    extent = record.get("extent") or {}
    temporal = extent.get("temporal") or {}
    raw = temporal.get("interval")
    if not (isinstance(raw, list) and raw and isinstance(raw[0], list)):
        return True
    interval = raw[0]
    if not (isinstance(interval, list) and len(interval) == 2):
        return True
    rec_start_raw, rec_end_raw = interval
    qs, qe = query
    rec_start = _parse_iso(rec_start_raw) if rec_start_raw else None
    rec_end = _parse_iso(rec_end_raw) if rec_end_raw else None
    if rec_end is not None and rec_end < qs:
        return False
    if rec_start is not None and rec_start > qe:
        return False
    return True


_VARIABLE_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def _tokenise_variable_name(name: str) -> set[str]:
    """Split a variable name into lowercase alphanumeric tokens.
    ``10m_u_component_of_wind`` → ``{"10m", "u", "component", "of",
    "wind"}``. Empty strings dropped so the empty-token from a leading
    separator doesn't pollute the set."""
    return {tok for tok in _VARIABLE_TOKEN_SPLIT.split(name.lower()) if tok}


def _record_matches_variable(record: dict[str, Any], needle: str) -> bool:
    """Two-tier variable filter (codex retro PR #124 MEDIUM-2 rewrite).

    Precision-first contract:
      1. **Tier 1 — bundled-constraints variable enum.** Tokenise each
         variable name in ``constraints.get("variable")`` on ``_`` (and
         other non-alphanumeric) and check whether ``needle`` (also
         tokenised) is a subset of any single variable's token set.
         This is the strongest signal — it means the user could actually
         request that variable in the API.
      2. **Tier 2 — word-boundary text match.** Regex ``\\bneedle\\b``
         against keywords + summaries + id + title + description.
         Catches "wind speed" in keywords but NOT "window" in prose.

    The previous substring-everywhere logic over-matched (``"wind"``
    found ``satellite-albedo`` via "moving temporal window") and
    crowded out real matches under ``limit``."""
    needle_clean = needle.strip().lower()
    if not needle_clean:
        return False
    needle_tokens = _tokenise_variable_name(needle_clean)
    # Round-1 cr HIGH: ``set().issubset(...)`` is always True. A needle
    # like ``","`` or ``"!!!"`` strips non-empty but tokenises to the
    # empty set; without this guard the Tier-1 subset check would match
    # every record that has a constraints variable enum. Skip both
    # tiers when the needle has no signal-bearing tokens.
    if not needle_tokens:
        return False

    # Tier 1 — constraints' variable enum (primary signal).
    dataset_id = record.get("id")
    if isinstance(dataset_id, str):
        constraints = load_constraints().get(dataset_id)
        if isinstance(constraints, dict):
            variables = constraints.get("variable")
            if isinstance(variables, list):
                for var_name in variables:
                    if not isinstance(var_name, str):
                        continue
                    var_tokens = _tokenise_variable_name(var_name)
                    if needle_tokens.issubset(var_tokens):
                        return True

    # Tier 2 — word-boundary text fallback over the record's text
    # surfaces. Build a regex with \b anchors so substrings of unrelated
    # words ("window" containing "wind") don't match.
    boundary_re = re.compile(
        r"\b" + re.escape(needle_clean) + r"\b", re.IGNORECASE
    )
    text_parts: list[str] = []
    for field in ("id", "title", "description"):
        value = record.get(field)
        if value:
            text_parts.append(str(value))
    kws = record.get("keywords") or []
    if isinstance(kws, list):
        text_parts.extend(str(k) for k in kws)
    summaries = record.get("summaries") or {}
    if isinstance(summaries, dict):
        for v in summaries.values():
            if isinstance(v, list):
                text_parts.extend(str(item) for item in v)
            elif isinstance(v, dict):
                text_parts.append(json.dumps(v))
            else:
                text_parts.append(str(v))
    return bool(boundary_re.search("\n".join(text_parts)))


def _validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    w, s, e, n = bbox
    if not (-180.0 <= w <= 180.0 and -180.0 <= e <= 180.0):
        raise ValidationError(
            f"bbox longitudes must be in [-180, 180], got west={w} east={e}",
            record=build_error_record(
                "ValidationError",
                message=(
                    f"bbox longitudes out of range; got west={w} east={e}"
                ),
                recovery_action="modify_request_parameters",
            ),
        )
    if not (-90.0 <= s <= 90.0 and -90.0 <= n <= 90.0):
        raise ValidationError(
            f"bbox latitudes must be in [-90, 90], got south={s} north={n}",
            record=build_error_record(
                "ValidationError",
                message=(
                    f"bbox latitudes out of range; got south={s} north={n}"
                ),
                recovery_action="modify_request_parameters",
            ),
        )
    if s > n:
        raise ValidationError(
            f"bbox south ({s}) must be <= north ({n})",
            record=build_error_record(
                "ValidationError",
                message=f"bbox south ({s}) must be <= north ({n})",
                recovery_action="modify_request_parameters",
            ),
        )
    if w > e:
        # CMEMS rejects antimeridian-crossing bboxes (the project conventions inv 7).
        # CDS catalogue search has the same policy — the agent should
        # split the query rather than have us guess intent.
        raise ValidationError(
            f"bbox west ({w}) must be <= east ({e}); antimeridian-crossing "
            "bboxes are rejected — split into two requests",
            record=build_error_record(
                "ValidationError",
                message=(
                    f"bbox west ({w}) must be <= east ({e}); "
                    "antimeridian-crossing bboxes are not supported"
                ),
                recovery_action="modify_request_parameters",
            ),
        )


def _validate_time_range(
    time_range: tuple[str, str],
) -> tuple[datetime, datetime]:
    try:
        start = _parse_iso(time_range[0])
        end = _parse_iso(time_range[1])
    except ValueError as exc:
        raise ValidationError(
            f"time_range values must be ISO 8601 timestamps: {exc}",
            record=build_error_record(
                "ValidationError",
                message=f"time_range values must be ISO 8601 timestamps: {exc}",
                recovery_action="modify_request_parameters",
            ),
        ) from None
    if start > end:
        raise ValidationError(
            f"time_range start ({start}) must be <= end ({end})",
            record=build_error_record(
                "ValidationError",
                message=f"time_range start ({start}) must be <= end ({end})",
                recovery_action="modify_request_parameters",
            ),
        )
    return start, end


# T-CDS-021 PR-2: hierarchical grouping helpers ---------------------------


_VARIABLE_DOMAIN_PREFIX = "Variable domain:"
_UNSPECIFIED_DOMAIN = "unspecified"


def _record_domains(record: dict[str, Any]) -> list[str]:
    """Extract all ``Variable domain:`` keywords from a record. CDS
    datasets are commonly tagged with multiple domains
    (e.g. ERA5 carries both ``"Atmosphere (surface)"`` and ``"Atmosphere
    (upper air)"``). Returns ``["unspecified"]`` if no domain keyword
    exists so the record is still discoverable via grouping."""
    keywords = record.get("keywords") or []
    out: list[str] = []
    if isinstance(keywords, list):
        for kw in keywords:
            if isinstance(kw, str) and kw.startswith(_VARIABLE_DOMAIN_PREFIX):
                out.append(kw[len(_VARIABLE_DOMAIN_PREFIX):].strip())
    return out or [_UNSPECIFIED_DOMAIN]


def _record_category(record: dict[str, Any]) -> str:
    """Derive the dataset category from the ID's first token. CDS dataset
    IDs follow a family naming convention: ``reanalysis-era5-*``,
    ``satellite-*``, ``cams-*``, ``efas-*``, ``sis-*``, etc. The first
    token is the natural grouping axis."""
    dataset_id = record.get("id") or ""
    if not isinstance(dataset_id, str):
        return ""
    first, _, _ = dataset_id.partition("-")
    return first


_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    """Reduce ``value`` to lowercase ``[a-z0-9-]+``. Used for group ids
    so future taxonomy values (commas, slashes, the ``|`` separator
    itself) can't collide with the ``"<domain-slug>|<category-slug>"``
    chaining key (cr round-1 MEDIUM)."""
    return _SLUG_NON_ALNUM.sub("-", value.lower()).strip("-")


def _group_id(domain: str, category: str) -> str:
    """Stable chaining key for ``cds_search_datasets(domain=..., category=...)``.
    Both halves are slugified independently and joined with ``|`` so the
    full id is URL-safe and round-trippable in scripts even if a future
    catalogue introduces punctuation in either axis."""
    return f"{_slugify(domain)}|{_slugify(category)}"


def search_groups(
    *,
    query: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Return ranked (domain, category) groups of CDS / ADS / EWDS datasets.

    Mirrors the CMEMS hierarchical-search UX (``marine_search_groups``)
    for free-text discovery: the LLM passes a natural-language query
    and gets a short list of groups + sample dataset titles, then
    narrows via ``cds_search_datasets(domain=..., category=...)``.

    Without ``query``: every populated group, sorted by
    ``dataset_count`` descending.

    With ``query``: scored against the query — substring matches on
    domain name (+2), category name (+2), and member-dataset
    keywords/titles/descriptions (+1 per match). Ties broken by
    ``dataset_count``.

    ``top_k`` caps the returned list (after ranking).
    """
    if query is not None and not query.strip():
        query = None

    catalogue = load_catalogue()
    # (domain, category) → list of (store, record). Multi-domain
    # datasets land in every matching group (ERA5 surface AND upper-air).
    # Round-1 cr LOW: skip records whose category resolves to empty
    # (would yield unchainable groups — no advertised cds_search_datasets
    # filter could re-narrow them).
    grouped: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
    for store, records in catalogue.items():
        for record in records:
            cat = _record_category(record)
            if not cat:
                continue
            for dom in _record_domains(record):
                if not dom:
                    continue
                grouped.setdefault((dom, cat), []).append((store, record))

    needle = query.lower() if query else None
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for (domain, category), members in grouped.items():
        if needle is None:
            score = 0.0
        else:
            score = 0.0
            dom_l = domain.lower()
            cat_l = category.lower()
            # Full-query substring on either axis: strongest signal
            # (e.g. ``"atmosphere (surface)"`` query → exact axis hit).
            if needle in dom_l:
                score += 2.0
            if needle in cat_l:
                score += 2.0
            # Round-1 codex M1: per-token hits on the axes also get
            # the full +2 boost (was +0.5 — too weak for multi-word
            # queries; "atmosphere reanalysis" should surface both
            # domain=atmosphere AND category=reanalysis groups first).
            tokens = needle.split()
            for token in tokens:
                if token in dom_l:
                    score += 2.0
                if token in cat_l:
                    score += 2.0
                # Membership match (capped) — secondary signal so
                # huge groups don't drown axis relevance. Deliberately
                # still the simple substring ``_record_matches`` (not the
                # token-AND ``_record_matches_keyword`` used by search()):
                # here ``token`` is already a single split word and this is
                # only a tie-break weight on group ranking, not a
                # dataset-level filter. Routing it through the richer
                # keyword matcher would change reviewed group scores for no
                # discovery benefit (T-CDS-KWFIX divergence; see
                # the project decision log).
                hits = sum(
                    1
                    for _, rec in members
                    if _record_matches(rec, token)
                )
                score += min(hits, 5) * 0.2
            if score == 0.0:
                continue
        sample = [
            rec.get("title") or rec.get("id") or ""
            for _, rec in members[:3]
        ]
        scored.append(
            (
                score,
                len(members),
                {
                    "id": _group_id(domain, category),
                    "domain": domain,
                    "category": category,
                    "dataset_count": len(members),
                    "sample_titles": sample,
                    "score": round(score, 3),
                },
            )
        )

    scored.sort(key=lambda t: (-t[0], -t[1]))
    # Round-1 local cr M2: ``top_k=0`` means "zero results" (matches
    # the schema's ``ge=1`` semantics for the tool path — direct
    # callers passing 0 should also see 0, not the full list).
    if top_k is not None and top_k >= 0:
        scored = scored[:top_k]
    out_groups = [g for _, _, g in scored]
    return {"groups": out_groups, "total_count": len(out_groups)}


def _record_matches(record: dict[str, Any], needle: str) -> bool:
    """Case-insensitive substring match against id, title, description,
    keywords. Designed for human-typed queries — exact equality on
    keyword/id matches always succeeds."""
    haystack_parts: list[str] = []
    for field in ("id", "title", "description"):
        value = record.get(field)
        if value:
            haystack_parts.append(str(value))
    keywords = record.get("keywords") or []
    if isinstance(keywords, list):
        haystack_parts.extend(str(k) for k in keywords)
    haystack = "\n".join(haystack_parts).lower()
    return needle.lower() in haystack


def _record_matches_keyword(record: dict[str, Any], needle: str) -> bool:
    """Token-AND keyword match (T-CDS-KWFIX).

    Splits ``needle`` on whitespace and keeps the record when EVERY token
    matches via :func:`_record_matches_variable` — i.e. the token is a
    subset of some bundled-constraints ``variable`` name (Tier 1) OR a
    word-boundary hit in the record's text surface (Tier 2: id + title +
    description + keywords + summaries).

    This replaces the previous single literal-substring test, which
    returned 0 for any natural multi-word query whose words were not
    contiguous in the prose — e.g. a smaller model typing
    ``"2m air temperature"`` got 0 results and wrongly concluded the
    keyword filter was broken, even though every word individually maps to
    real temperature datasets. Word order no longer matters, and variable
    names (which live in constraints, not the STAC prose) become
    searchable.

    Reusing :func:`_record_matches_variable` per token is deliberate: it
    already does precision-first word-boundary matching, so ``"sea ice"``
    does NOT match the ubiquitous "Copernicus Climate Change **Service**"
    boilerplate (``ice`` ⊄ ``service`` at a word boundary) the way a naive
    per-token substring would. An empty token set (whitespace/punctuation-
    only needle, already coerced to ``None`` by the schema validator)
    matches nothing."""
    tokens = needle.split()
    if not tokens:
        return False
    return all(_record_matches_variable(record, token) for token in tokens)


def _keyword_relevance(record: dict[str, Any], needle: str) -> int:
    """Rank a keyword match so the most on-topic datasets survive ``limit``.

    A record only reaches here once it has already passed
    ``_record_matches_keyword`` (every token matched). Token-AND matching
    is deliberately broad — generic ERA5 prose satisfies ``air`` and
    ``quality`` as *separate* tokens — so without ranking a query like
    ``"air quality"`` would let dozens of reanalysis records (scanned
    first, in snapshot order) fill ``limit`` and bury the real CAMS
    air-quality datasets in the later-scanned ADS store (codex
    T-CDS-KWFIX review). Tiers, highest first:

      3 — the full query phrase is a contiguous substring of a
          human-meaningful field (id / title / a keyword)
      2 — the full phrase is a contiguous substring of the description
      1 — matched only token-by-token (no contiguous phrase anywhere)

    Within a tier the caller keeps snapshot order (stable sort), so this
    only *promotes* exact-phrase hits; it never reshuffles equally-ranked
    records."""
    # Collapse internal whitespace runs the SAME way ``_record_matches_keyword``
    # does (``str.split``), so ``"air  quality"`` / ``"air\tquality"`` score
    # the exact-phrase tier against single-spaced titles instead of silently
    # collapsing to tier 1 and re-burying the match (codex r2).
    phrase = " ".join(needle.split()).lower()
    if not phrase:
        return 1
    strong: list[str] = []
    for field in ("id", "title"):
        value = record.get(field)
        if value:
            strong.append(str(value))
    keywords = record.get("keywords") or []
    if isinstance(keywords, list):
        strong.extend(str(k) for k in keywords)
    if any(phrase in part.lower() for part in strong):
        return 3
    description = record.get("description")
    if description and phrase in str(description).lower():
        return 2
    return 1


def search(
    *,
    keyword: str | None = None,
    store: str | None = None,
    limit: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    time_range: tuple[str, str] | None = None,
    variable: str | None = None,
    domain: str | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Return slim catalogue records for the LLM to pick from.

    ``store=None`` combines all three stores. ``keyword=None`` returns
    every record (subject to ``limit``). ``keyword`` is a case-insensitive
    token-AND match: the query is split on whitespace and a record is kept
    when EVERY token matches its searchable surface — either as a
    word-boundary hit in id + title + description + keywords + summaries,
    or as a token-subset of the bundled constraints' ``variable`` enum. So
    word order is irrelevant and ``keyword="2m temperature"`` finds ERA5
    even though that phrase is not contiguous in the prose and the
    variable name lives only in constraints (T-CDS-KWFIX).

    T-CDS-020 filters (all AND-combined with each other and with
    ``keyword``):
      - ``bbox=(w, s, e, n)`` in WGS84 degrees. Antimeridian-crossing
        queries are rejected (mirrors the CMEMS subset invariant 7).
        Records with no usable ``extent.spatial.bbox`` are included.
      - ``time_range=(start_iso, end_iso)``. Records whose temporal
        extent overlaps the window are kept; open-ended dataset ends
        (``None``) are treated as +∞. Records with no usable
        ``extent.temporal.interval`` are included.
      - ``variable=substring`` — case-insensitive over keywords +
        summaries + id/title/description.

    Returns the slim shape ``{id, title, description, keywords, store}``.
    Without ``keyword`` the order is the bundled snapshot order. With
    ``keyword`` the results are relevance-ranked (exact-phrase hits in
    id/title/keywords first, then description, then token-only matches),
    snapshot order breaking ties — so a small ``limit`` keeps the most
    on-topic datasets rather than whichever store is scanned first.
    """
    if store is not None and store not in _STORES:
        raise ValidationError(
            f"unknown CDS store {store!r}; must be one of {list(_STORES)}",
            record=build_error_record(
                "ValidationError",
                message=(
                    f"unknown CDS store {store!r}; "
                    f"must be one of {list(_STORES)}"
                ),
                recovery_action="modify_request_parameters",
            ),
        )

    if bbox is not None:
        _validate_bbox(bbox)
    parsed_time: tuple[datetime, datetime] | None = None
    if time_range is not None:
        parsed_time = _validate_time_range(time_range)

    catalogue = load_catalogue()
    target_stores = (store,) if store else _STORES

    # Defence-in-depth: schema rejects ``limit < 1`` but harden the
    # catalogue helper independently — direct callers (tests, future
    # internal use) bypass the schema.
    effective_limit = limit if limit is not None and limit > 0 else None

    out: list[dict[str, Any]] = []
    # Keyword searches are relevance-ranked (see ``_keyword_relevance``)
    # because token-AND matching is broad and ``limit`` truncation would
    # otherwise bury exact-phrase hits behind generic per-token matches in
    # whichever store is scanned first. We therefore collect ALL matches
    # for the keyword path and sort before truncating, instead of the
    # snapshot-order early-return used when no keyword is given.
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    order = 0
    for s in target_stores:
        for record in catalogue[s]:
            if keyword is not None and not _record_matches_keyword(
                record, keyword
            ):
                continue
            if bbox is not None and not _record_intersects_bbox(record, bbox):
                continue
            if parsed_time is not None and not _record_overlaps_time(
                record, parsed_time
            ):
                continue
            if variable is not None and not _record_matches_variable(
                record, variable
            ):
                continue
            if domain is not None and domain not in _record_domains(record):
                continue
            if category is not None and _record_category(record) != category:
                continue
            slim = _slim_record(record, s)
            if keyword is not None:
                # Negate the tier so a plain ascending sort keeps the
                # highest tier first while ``order`` preserves snapshot
                # order within a tier (stable tie-break).
                ranked.append((-_keyword_relevance(record, keyword), order, slim))
                order += 1
                continue
            out.append(slim)
            if effective_limit is not None and len(out) >= effective_limit:
                return out
    if keyword is not None:
        ranked.sort(key=lambda t: (t[0], t[1]))
        out = [slim for _, _, slim in ranked]
        if effective_limit is not None:
            out = out[:effective_limit]
    return out


def store_for(dataset_id: str) -> str | None:
    """Light lookup: return the store name (``cds`` / ``ads`` / ``ewds``)
    for a dataset id, or ``None`` if unknown.

    T-CDS-011: used by the backend's per-store URL routing on the
    submit/poll/cancel/fetch path. ``describe()`` also returns ``store``
    but deep-copies the full STAC record, which is wasteful when the
    caller only needs the one-word routing key.
    """
    catalogue = load_catalogue()
    for store in _STORES:
        for record in catalogue[store]:
            if record.get("id") == dataset_id:
                return store
    return None


def _is_timeseries_product(record: dict[str, Any]) -> bool:
    """T-TS-007: the ARCO ``*-timeseries`` products take a nested ``location``
    point that the upstream machine-readable form omits. Detect them by id
    suffix or the ``Data type: Time-series`` keyword."""
    if str(record.get("id", "")).endswith("-timeseries"):
        return True
    keywords = record.get("keywords") or []
    return isinstance(keywords, list) and "Data type: Time-series" in keywords


def describe(dataset_id: str) -> dict[str, Any]:
    """Return the full STAC record for ``dataset_id``.

    Cross-store lookup: searches all three stores and returns the first
    match. The returned record includes a ``store`` field so the agent
    knows where the dataset lives without a separate call.

    Raises ``NotFoundError`` if no record exists in any store.
    """
    catalogue = load_catalogue()
    for store in _STORES:
        for record in catalogue[store]:
            if record.get("id") == dataset_id:
                # Codex/code-reviewer T-CDS-003 LOW: shallow ``dict(record)``
                # protected only top-level keys; nested ``links`` / ``extent``
                # / ``summaries`` remained shared with the module cache, so a
                # caller mutating those would poison the cache for the
                # process lifetime. Deep copy is cheap (~10 KB per record)
                # and removes the foot-gun for direct catalogue users.
                augmented = copy.deepcopy(record)
                augmented["store"] = store
                # T-CDS-015 (Layer A): attach bundled constraints if
                # available. This gives the LLM the dataset's input
                # contract (`data_format`, `variable` values, etc.) so
                # it doesn't guess `format: ...` vs `data_format: ...`.
                constraints = load_constraints().get(dataset_id)
                if constraints is not None:
                    augmented["available_inputs"] = copy.deepcopy(constraints)
                if _is_timeseries_product(augmented):
                    # T-TS-007: upstream omits the required ``location`` point
                    # from the machine-readable form (STAC summaries empty;
                    # constraints enumerate only discrete fields). Inject the
                    # shape so an agent can compose a valid request from
                    # ``available_inputs`` alone. ``setdefault`` so a future
                    # upstream addition of ``location`` is never clobbered.
                    ai = augmented.setdefault("available_inputs", {})
                    if isinstance(ai, dict):
                        ai.setdefault(
                            "location",
                            {
                                "latitude": "<float, WGS84 degrees, -90..90>",
                                "longitude": "<float, WGS84 degrees, -180..180>",
                            },
                        )
                return augmented
    raise NotFoundError(
        f"CDS dataset {dataset_id!r} not found in bundled catalogue",
        record=build_error_record(
            "NotFoundError",
            message=(
                f"CDS dataset {dataset_id!r} not found in bundled catalogue; "
                "run scripts/refresh_cds_catalogue.py if the dataset was "
                "added upstream after the snapshot"
            ),
            recovery_action="modify_request_parameters",
        ),
    )
