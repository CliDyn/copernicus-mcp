"""Runtime hierarchical routing over the bundled CMEMS manifests.

T-CMEMS-HIER-005. The agent pipeline is:

    query → search_groups(query)
          → search_products(group_ids, query)
          → search_datasets_in_products(product_ids, query, bbox?, time_range?)

Each level returns the same envelope so the orchestrator can pass
the result through without re-shaping::

    {
        "selected": [...],         # ranked matches
        "rejected": [...],         # close-but-not-quite, for debug
        "reason": "<str>",         # one-line human explanation
        "confidence": "high"|"medium"|"low",
        "fallback_available": <bool>,  # True iff caller should try a
                                       # different strategy (flat search)
    }

All three loader functions are lazy + module-cached, matching
``catalogue.load_catalogue``. No network, no credentials.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Word-boundary phrase match. Avoids the "arctic" ⊂ "antarctic" false
# positive: a phrase matches only when surrounded by start/end-of-string
# or non-alphanumeric characters. Case-insensitive.
_NON_WORD = re.compile(r"[^a-z0-9]")


def _phrase_in(phrase: str, haystack: str) -> bool:
    """Return True iff ``phrase`` appears in ``haystack`` as a
    whole-word run (not as a substring of a longer word).

    Normalises hyphens to spaces in BOTH inputs (round-1 codex
    MEDIUM): manifest phrases like ``sea-surface height`` should
    match user queries spelled ``sea surface height`` and vice
    versa. Hyphens are visual / typographic, not semantic."""
    if not phrase or not haystack:
        return False
    norm_phrase = phrase.lower().replace("-", " ")
    norm_haystack = haystack.lower().replace("-", " ")
    # ``\b`` doesn't work cleanly for multi-word phrases with spaces
    # in them; explicit boundary check on the two ends.
    pattern = r"(?:^|[^a-z0-9])" + re.escape(norm_phrase) + r"(?:$|[^a-z0-9])"
    return bool(re.search(pattern, norm_haystack))


_DATA_DIR: Path = Path(__file__).resolve().parent / "_data"

_groups_cache: list[dict[str, Any]] | None = None
_products_cache: list[dict[str, Any]] | None = None
_cards_cache: list[dict[str, Any]] | None = None

# Scoring knobs. Tuned conservatively — easy to bump in v2.
_INCLUDE_HIT_WEIGHT: float = 2.0
_EXCLUDE_HIT_WEIGHT: float = 3.0  # penalise more than include rewards
_TITLE_HIT_WEIGHT: float = 1.0
_SUMMARY_HIT_WEIGHT: float = 0.5

# Confidence thresholds applied to the top-1 score.
_HIGH_CONFIDENCE_THRESHOLD: float = 4.0
_MEDIUM_CONFIDENCE_THRESHOLD: float = 1.5

# Default response caps.
_DEFAULT_GROUPS_TOP_K: int = 5
_DEFAULT_PRODUCTS_TOP_K: int = 20
_DEFAULT_DATASETS_LIMIT: int = 10
# Upper bound on `limit` so a caller can't request all 1251 cards
# inline (round-1 LOW). 50 is generous enough for any agent UI.
_MAX_DATASETS_LIMIT: int = 50


# ---------------------------------------------------------------------------
# Lazy loaders
# ---------------------------------------------------------------------------


def load_groups() -> list[dict[str, Any]]:
    """Return the bundled ``groups.json`` content (47 entries)."""
    global _groups_cache  # noqa: PLW0603 — module-level cache by design
    if _groups_cache is None:
        _groups_cache = _load_json("groups.json")
    return _groups_cache


def load_products() -> list[dict[str, Any]]:
    """Return the bundled ``products.json`` content (306 entries)."""
    global _products_cache  # noqa: PLW0603
    if _products_cache is None:
        _products_cache = _load_json("products.json")
    return _products_cache


def load_cards() -> list[dict[str, Any]]:
    """Return the bundled ``dataset_cards.json`` content (1251 entries)."""
    global _cards_cache  # noqa: PLW0603
    if _cards_cache is None:
        _cards_cache = _load_json("dataset_cards.json")
    return _cards_cache


def _load_json(filename: str) -> list[dict[str, Any]]:
    """Read a bundled manifest. Fails fast on missing or
    structurally-wrong shape (mirrors ``catalogue.load_catalogue``
    diagnostics)."""
    path = _DATA_DIR / filename
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"CMEMS manifest missing at {path}: broken install or "
            f"missing refresh. Run `python scripts/refresh_marine_catalogue.py` "
            f"to rebuild."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse {path}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"{path}: top level is not a list (got {type(data).__name__})")
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entry #{i} is {type(entry).__name__}, expected dict")
    return data


# ---------------------------------------------------------------------------
# search_groups
# ---------------------------------------------------------------------------


def search_groups(
    query: str,
    *,
    top_k: int = _DEFAULT_GROUPS_TOP_K,
) -> dict[str, Any]:
    """Rank groups by phrase-overlap against the query.

    Scoring per group:
      +2 for every include_when_query_mentions phrase substring-found
      in the query.
      -3 for every exclude_when_query_mentions phrase substring-found
      in the query.
      +1 for any token of group_title substring-found in the query.
      +0.5 for any token of summary substring-found in the query.

    Negative scores get dropped from ``selected`` and surfaced in
    ``rejected`` (with their score) so a developer debugging a bad
    match can see what nearly won.
    """
    needle = query.lower().strip()
    groups = load_groups()
    scored: list[tuple[float, dict[str, Any]]] = []
    for g in groups:
        score = _score_group(g, needle)
        scored.append((score, g))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    positive = [(s, g) for s, g in scored if s > 0]
    rejected_with_penalty = [(s, g) for s, g in scored if s < 0]

    selected_payload = [
        {
            "group_id": g["group_id"],
            "group_title": g.get("group_title", ""),
            "summary": g.get("summary", ""),
            "product_ids": list(g.get("product_ids") or []),
            "score": round(s, 3),
        }
        for s, g in positive[:top_k]
    ]
    rejected_payload = [
        {
            "group_id": g["group_id"],
            "score": round(s, 3),
            "reason": "exclude_when_query_mentions matched",
        }
        for s, g in rejected_with_penalty[:5]
    ]

    confidence = _confidence_from_top_score(
        selected_payload[0]["score"] if selected_payload else 0.0
    )
    return {
        "selected": selected_payload,
        "rejected": rejected_payload,
        "reason": _reason_for_groups(query, selected_payload, confidence),
        "confidence": confidence,
        "fallback_available": confidence == "low",
        "catalogue_fetched_at": _fetched_at_iso(),
    }


def _score_group(group: dict[str, Any], needle: str) -> float:
    """Compute the phrase-overlap score for one group + query.
    Word-boundary matching prevents ``arctic`` ⊂ ``antarctic``
    false positives."""
    score = 0.0
    for phrase in group.get("include_when_query_mentions") or []:
        if isinstance(phrase, str) and _phrase_in(phrase, needle):
            score += _INCLUDE_HIT_WEIGHT
    for phrase in group.get("exclude_when_query_mentions") or []:
        if isinstance(phrase, str) and _phrase_in(phrase, needle):
            score -= _EXCLUDE_HIT_WEIGHT
    # Title and summary tokens contribute small additional signal.
    title = (group.get("group_title") or "").lower()
    for token in _meaningful_tokens(title):
        if _phrase_in(token, needle):
            score += _TITLE_HIT_WEIGHT
            break  # one hit per axis is enough; avoid double-counting
    summary = (group.get("summary") or "").lower()
    for token in _meaningful_tokens(summary):
        if _phrase_in(token, needle):
            score += _SUMMARY_HIT_WEIGHT
            break
    return score


def _meaningful_tokens(text: str) -> list[str]:
    """Tokenise on whitespace + common punctuation, drop stopwords
    and tokens shorter than 4 characters. Used for title/summary
    matching where we don't want ``the``, ``and``, ``of`` to count."""
    stopwords = {
        "the",
        "and",
        "for",
        "from",
        "with",
        "into",
        "over",
        "this",
        "that",
        "are",
        "all",
        "any",
    }
    raw = (
        text.replace(",", " ")
        .replace(".", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("/", " ")
        .replace("—", " ")
        .replace("-", " ")
        .split()
    )
    return [t for t in raw if len(t) >= 4 and t not in stopwords]


def _confidence_from_top_score(top_score: float) -> str:
    if top_score >= _HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if top_score >= _MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def _reason_for_groups(query: str, selected: list[dict[str, Any]], confidence: str) -> str:
    if not selected:
        return (
            f"no group matched query {query!r} — try the flat search "
            "(marine_search_datasets) or refine the query."
        )
    top = selected[0]
    return (
        f"top group {top['group_id']!r} matched with score "
        f"{top['score']} ({confidence} confidence)."
    )


# ---------------------------------------------------------------------------
# search_products
# ---------------------------------------------------------------------------


def search_products(
    group_ids: list[str],
    *,
    query: str | None = None,
    top_k: int = _DEFAULT_PRODUCTS_TOP_K,
) -> dict[str, Any]:
    """Filter products by membership in any of ``group_ids``.

    If ``query`` is provided, products are re-ranked by phrase-overlap
    against their ``product_title`` + ``summary`` + ``variables_normalized``.
    Otherwise the original sort (by ``product_id``) is preserved.
    """
    groups_by_id = {g["group_id"]: g for g in load_groups()}
    products_by_id = {p["product_id"]: p for p in load_products()}

    # Union of product_ids + per-product group-rank score. ``group_ids``
    # is the agent's ranked list (most-relevant first); a product cited
    # by the top group should sort above one cited only by the third.
    # Multi-membership across groups adds to the rank score.
    rank_score: dict[str, float] = {}
    unknown_group_ids: list[str] = []
    n = len(group_ids)
    for i, gid in enumerate(group_ids):
        group = groups_by_id.get(gid)
        if group is None:
            unknown_group_ids.append(gid)
            continue
        # Position-decayed weight: top group contributes ``n``, next
        # ``n-1``, etc. Multi-membership bubbles a product above
        # single-membership without overwhelming keyword score.
        weight = float(n - i)
        for pid in group.get("product_ids") or []:
            rank_score[pid] = rank_score.get(pid, 0.0) + weight

    # Score each candidate product.
    needle = (query or "").lower().strip()
    scored: list[tuple[float, float, dict[str, Any]]] = []
    for pid, group_rank in rank_score.items():
        product = products_by_id.get(pid)
        if product is None:
            continue  # group references a product not in the manifest — skip
        keyword_score = _score_product(product, needle) if needle else 0.0
        scored.append((keyword_score, group_rank, product))

    # Sort by (keyword_score desc, group_rank desc, product_id asc).
    # Keyword signal dominates when provided; group rank breaks ties
    # (and is the only signal when no query).
    scored.sort(key=lambda triple: (-triple[0], -triple[1], triple[2]["product_id"]))

    selected_payload = [
        {
            "product_id": p["product_id"],
            "product_title": p.get("product_title", ""),
            "summary": p.get("summary", ""),
            "domains": list(p.get("domains") or []),
            "regions": list(p.get("regions") or []),
            "data_types": list(p.get("data_types") or []),
            "dataset_count": p.get("dataset_count", 0),
            "score": round(kw, 3) if needle else round(gr, 3),
        }
        for kw, gr, p in scored[:top_k]
    ]
    rejected_payload = [
        {"group_id": gid, "reason": "unknown group_id"} for gid in unknown_group_ids
    ]

    if not selected_payload:
        confidence = "low"
        reason = (
            f"no products matched group_ids {group_ids!r} (unknown: {unknown_group_ids or 'none'})"
        )
    elif needle and scored[0][0] >= _HIGH_CONFIDENCE_THRESHOLD:
        confidence = "high"
        reason = f"{len(selected_payload)} products in groups {group_ids!r}"
    else:
        # Membership match without keyword refinement, or modest keyword
        # signal — call it medium so the caller knows to keep the agent
        # in the loop.
        confidence = "medium"
        reason = f"{len(selected_payload)} products in groups {group_ids!r}"
    return {
        "selected": selected_payload,
        "rejected": rejected_payload,
        "reason": reason,
        "confidence": confidence,
        "fallback_available": confidence == "low",
        "catalogue_fetched_at": _fetched_at_iso(),
    }


def _score_product(product: dict[str, Any], needle: str) -> float:
    score = 0.0
    title = (product.get("product_title") or "").lower()
    summary = (product.get("summary") or "").lower()
    for token in _meaningful_tokens(title):
        if _phrase_in(token, needle):
            score += _TITLE_HIT_WEIGHT
    for token in _meaningful_tokens(summary):
        if _phrase_in(token, needle):
            score += _SUMMARY_HIT_WEIGHT
    for var in product.get("variables_normalized") or []:
        if isinstance(var, str) and _phrase_in(var, needle):
            score += _INCLUDE_HIT_WEIGHT
    return score


# ---------------------------------------------------------------------------
# search_datasets_in_products
# ---------------------------------------------------------------------------


def search_datasets_in_products(
    product_ids: list[str],
    *,
    query: str | None = None,
    limit: int = _DEFAULT_DATASETS_LIMIT,
    bbox: tuple[float, float, float, float] | None = None,
    time_range: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Filter cards by membership in ``product_ids``, optionally
    refining by query / bbox / time_range.

    Bbox semantics (``[min_lon, min_lat, max_lon, max_lat]``): a card
    is kept when its ``spatial_extent`` overlaps the bbox. Cards with
    ``spatial_extent is None`` are EXCLUDED when bbox is set — better
    under-select than risk a mismatch (codex spec-review).

    time_range semantics (ISO-8601 ``(start, end)``): a card is kept
    when its ``temporal_extent`` overlaps the range. Cards with
    ``temporal_extent is None`` are EXCLUDED when time_range is set.

    Query (optional): plain substring rank against ``title`` /
    ``description`` / ``variables_normalized``.
    """
    # Preserve the caller's product_ids order: a product earlier in
    # the list is more relevant (the agent's search_products step
    # already ranked them), so its cards should sort first when no
    # keyword query is provided to re-rank.
    pid_rank = {pid: i for i, pid in enumerate(product_ids)}
    eligible_pids = set(product_ids)
    cards = load_cards()
    pool = [c for c in cards if c.get("product_id") in eligible_pids]

    rejected: list[dict[str, Any]] = []

    if bbox is not None:
        before = len(pool)
        pool = [c for c in pool if _spatial_overlaps(c.get("spatial_extent"), bbox)]
        if len(pool) < before:
            rejected.append(
                {
                    "filter": "bbox",
                    "removed": before - len(pool),
                    "reason": "no spatial overlap or null spatial_extent",
                }
            )

    if time_range is not None:
        before = len(pool)
        pool = [c for c in pool if _temporal_overlaps(c.get("temporal_extent"), time_range)]
        if len(pool) < before:
            rejected.append(
                {
                    "filter": "time_range",
                    "removed": before - len(pool),
                    "reason": "no temporal overlap or null temporal_extent",
                }
            )

    needle = (query or "").lower().strip()
    top_card_score: float = 0.0
    if needle:
        # Sort by (keyword_score desc, product_id rank asc, dataset_id asc).
        scored = [
            (
                _score_card(c, needle),
                pid_rank.get(c.get("product_id"), len(pid_rank)),
                c.get("dataset_id", ""),
                c,
            )
            for c in pool
        ]
        scored.sort(key=lambda t: (-t[0], t[1], t[2]))
        if scored:
            top_card_score = scored[0][0]
        pool = [c for _, _, _, c in scored]
    else:
        # No query: rank by (product_id position, then dataset_id).
        pool.sort(
            key=lambda c: (
                pid_rank.get(c.get("product_id"), len(pid_rank)),
                c.get("dataset_id", ""),
            )
        )

    # Round-1 LOW: cap `limit` so a caller can't pull all 1251 cards.
    effective_limit = max(1, min(limit, _MAX_DATASETS_LIMIT))
    pool_size = len(pool)
    selected_payload = pool[:effective_limit]

    # Round-1 HIGH: confidence dead branch.
    #   * empty pool → low + fallback (no cards to surface).
    #   * non-empty pool, but top keyword-score is 0 → low + fallback
    #     (the query didn't match anything; results are arbitrary).
    #   * pool > limit → medium ("paginate or refine").
    #   * else → high.
    if not selected_payload:
        confidence = "low"
        reason = f"no cards left after filtering {len(eligible_pids)} product(s)"
    elif needle and top_card_score == 0.0:
        confidence = "low"
        reason = (
            f"query {query!r} matched no card text — results are ordered "
            f"by product priority only; try widening or rephrasing."
        )
    elif pool_size > effective_limit:
        confidence = "medium"
        reason = (
            f"{len(selected_payload)} of {pool_size} cards from "
            f"{len(eligible_pids)} product(s); paginate or refine for more."
        )
    else:
        confidence = "high"
        reason = f"{len(selected_payload)} cards from {len(eligible_pids)} product(s)"

    return {
        "selected": selected_payload,
        "rejected": rejected,
        "reason": reason,
        "confidence": confidence,
        "fallback_available": confidence == "low",
        "catalogue_fetched_at": _fetched_at_iso(),
    }


def _score_card(card: dict[str, Any], needle: str) -> float:
    score = 0.0
    for field in ("title", "description"):
        value = (card.get(field) or "").lower()
        for token in _meaningful_tokens(value):
            if _phrase_in(token, needle):
                score += _TITLE_HIT_WEIGHT
                break
    for var in card.get("variables_normalized") or []:
        if isinstance(var, str) and _phrase_in(var, needle):
            score += _INCLUDE_HIT_WEIGHT
    return score


def _spatial_overlaps(
    extent: dict[str, Any] | None,
    bbox: tuple[float, float, float, float],
) -> bool:
    """Standard axis-aligned-rectangle overlap. Returns False when
    extent is None — cards without spatial info are excluded when
    a bbox filter is applied (under-selection by design)."""
    if not isinstance(extent, dict):
        return False
    try:
        min_lon = float(extent["min_lon"])
        max_lon = float(extent["max_lon"])
        min_lat = float(extent["min_lat"])
        max_lat = float(extent["max_lat"])
    except (KeyError, TypeError, ValueError):
        return False
    q_min_lon, q_min_lat, q_max_lon, q_max_lat = bbox
    return (
        min_lon <= q_max_lon
        and max_lon >= q_min_lon
        and min_lat <= q_max_lat
        and max_lat >= q_min_lat
    )


def _temporal_overlaps(
    extent: dict[str, Any] | None,
    time_range: tuple[str, str],
) -> bool:
    """Inclusive overlap on ISO-8601 time ranges. Returns False
    when extent is None — under-selection by design.

    cr round-1 MEDIUM: lex comparison broke on mixed precision
    (``2020-06-01`` vs ``2020-06-01T00:00:00Z``) and on tz-offset
    strings. Parse both ends with ``datetime.fromisoformat`` and
    compare as datetime objects."""
    if not isinstance(extent, dict):
        return False
    start = extent.get("start_datetime") or extent.get("start")
    end = extent.get("end_datetime") or extent.get("end")
    if not isinstance(start, str) or not isinstance(end, str):
        return False
    try:
        extent_start = _parse_iso(start)
        extent_end = _parse_iso(end)
        query_start = _parse_iso(time_range[0])
        query_end = _parse_iso(time_range[1])
    except ValueError:
        return False
    # codex round-2 MEDIUM: a corrupt card with ``start > end``
    # could otherwise spuriously "overlap" with broad queries that
    # straddle both endpoints. Treat as no extent (under-select).
    if extent_start > extent_end:
        return False
    return extent_start <= query_end and extent_end >= query_start


def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp into a tz-aware ``datetime``.

    Accepts ``YYYY-MM-DD``, ``YYYY-MM-DDTHH:MM:SS``, and the ``Z``
    suffix. Naive timestamps are anchored to UTC so all comparisons
    happen in the same frame."""
    text = value.strip()
    # ``fromisoformat`` (3.11+) accepts trailing ``Z`` and pure dates.
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


# ---------------------------------------------------------------------------
# Snapshot freshness marker (shared with catalogue.fetched_at when present)
# ---------------------------------------------------------------------------


def _fetched_at_iso() -> str:
    """Return the bundled snapshot timestamp, or a placeholder if
    the file is missing. Mirrors ``catalogue.fetched_at`` but does
    not raise — routing is read-only and the timestamp is metadata,
    not load-bearing."""
    path = _DATA_DIR / "fetched_at.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    value = data.get("marine") if isinstance(data, dict) else None
    if isinstance(value, str) and value.strip():
        return value
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
