"""Aggregate dataset cards into product manifest entries.

T-CMEMS-HIER-003. Runtime routing (T-CMEMS-HIER-005) consults
``products.json`` first to shortlist a small set of CMEMS products
matching a query, then drills into ``dataset_cards.json`` for the
selected products. The product layer carries unioned categorical
axes (domain / region / data_type / variables) plus a rule-based
``summary`` string that gives the router enough text to confirm
relevance without re-reading every member card.

Imported by the dev/ops refresh script
(``scripts/refresh_marine_catalogue.py``) and by unit tests.
``CmemsBackend`` will import this only via the bundled
``_data/products.json`` artefact — no runtime aggregation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

_MAX_SUMMARY_VARIABLES: int = 6

# Variable shortnames that carry no signal for a router skimming the
# summary: pure grid metrics, mask layers, depth bookkeeping. Filter
# them out of the summary preview so the slots get filled by science
# variables instead. cr round-1 PR #89 MEDIUM.
_BOOKKEEPING_VARS: frozenset[str] = frozenset(
    {
        "e1t",
        "e2t",
        "e3t",
        "e1u",
        "e2u",
        "e3u",
        "e1v",
        "e2v",
        "e3v",
        "deptho",
        "deptho_lev",
        # cr round-2 PR #89 MEDIUM: NWSHELF reanalysis exposes an
        # interpolated grid-depth variant.
        "deptho_lev_interp",
        "mask",
        # Some SDK records use the abbreviated "_bnds" form alongside
        # the canonical CF "_bounds" suffix; both are time-axis
        # bookkeeping.
        "climatology_bounds",
        "climatology_bnds",
    }
)

# Suffixes that indicate a derived variant of an underlying variable
# (mean, standard deviation, percentile, error bound). Drop them from
# the summary preview when an unsuffixed name is also present — the
# router cares about the underlying variable, not its statistics.
_DERIVED_SUFFIXES: tuple[str, ...] = (
    "_mean",
    "_std",
    "_lower",
    "_upper",
    "_min",
    "_max",
    # cr round-2 PR #89 MEDIUM: MEDSEA reanalysis uses "_avg" instead
    # of "_mean" for the same statistical aggregate.
    "_avg",
)


def build_products(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group ``cards`` by ``product_id``, union categorical axes, and
    emit one entry per product.

    Output is sorted by ``product_id``; each entry's ``dataset_ids``
    and union axes are sorted as well, so the bundled
    ``products.json`` has minimal diffs across refreshes.
    """
    by_product: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        raw_pid = card.get("product_id")
        if not isinstance(raw_pid, str):
            # Cards without a product_id can't be grouped. Silently
            # skip — the refresh sanitiser / shape guards upstream
            # should catch this before it reaches here.
            continue
        # cr round-1 PR #89 LOW: normalise whitespace so ``" FOO"`` and
        # ``"FOO"`` collide into one bucket rather than producing
        # duplicate product entries on SDK shape drift.
        pid = raw_pid.strip()
        if not pid:
            continue
        by_product.setdefault(pid, []).append(card)

    products: list[dict[str, Any]] = []
    for pid in sorted(by_product):
        members = by_product[pid]

        dataset_ids = sorted(
            {str(m.get("dataset_id")) for m in members if m.get("dataset_id") is not None}
        )
        domains = _union_strings(m.get("domain") for m in members)
        regions = _union_string_lists(m.get("region") for m in members)
        data_types = _union_string_lists(m.get("data_type") for m in members)
        variables = _union_string_lists(m.get("variables") for m in members)
        variables_normalized = _union_string_lists(m.get("variables_normalized") for m in members)

        product = {
            "product_id": pid,
            "product_title": _first_non_empty(m.get("product_title") for m in members) or pid,
            "description": _first_non_empty(m.get("description") for m in members) or "",
            "doi": _first_non_empty(m.get("doi") for m in members) or "",
            "dataset_ids": dataset_ids,
            "dataset_count": len(dataset_ids),
            "domains": domains,
            "regions": regions,
            "data_types": data_types,
            "variables": variables,
            "variables_normalized": variables_normalized,
            "summary": "",  # placeholder, filled in below.
        }
        product["summary"] = _render_summary(product)
        products.append(product)

    return products


def _union_strings(values: Iterable[Any]) -> list[str]:
    """Collect non-empty string scalars into a sorted unique list.

    cr round-1 PR #89 MEDIUM: split on commas so a card carrying
    ``domain="physics, biogeochemistry"`` (latent SDK shape drift)
    contributes two tokens, not one composite. Real-world cards
    today emit one bare token, but the split is cheap and prevents
    the summary line from reading "covering global physics,
    biogeochemistry (analysis)" with an unintended embedded comma.
    """
    out: set[str] = set()
    for v in values:
        if not isinstance(v, str):
            continue
        for piece in v.split(","):
            piece = piece.strip()
            if piece:
                out.add(piece)
    return sorted(out)


def _union_string_lists(values: Iterable[Any]) -> list[str]:
    """Collect non-empty string entries from a sequence of lists into
    a sorted unique list. Non-list / non-string entries are skipped."""
    out: set[str] = set()
    for v in values:
        if not isinstance(v, list):
            continue
        for item in v:
            if isinstance(item, str) and item:
                out.add(item)
    return sorted(out)


def _first_non_empty(values: Iterable[Any]) -> str | None:
    """Return the first truthy string in ``values`` (or ``None``).
    Used for fields like ``product_title`` / ``description`` / ``doi``
    where all member cards carry the same value at the SDK level."""
    for v in values:
        if isinstance(v, str) and v.strip():
            return v
    return None


def _render_summary(product: dict[str, Any]) -> str:
    """One-line rule-based summary surfacing the categorical axes a
    router needs to confirm relevance.

    The template keeps every union axis in line so the router does
    not need to consult the entry's structured fields just to see
    "what is this product about". A few sample variable shortnames
    follow so the router can match on common CF / SDK names — picked
    with ``_pick_summary_variables`` to skip grid bookkeeping and
    derived statistics in favour of the underlying science variables.
    """
    title = product["product_title"]
    n = product["dataset_count"]
    dataset_word = "dataset" if n == 1 else "datasets"
    domains = ", ".join(product["domains"]) or "unknown"
    regions = ", ".join(product["regions"]) or "unknown"
    # cr round-1 PR #89 LOW: consistent comma-space separator across
    # axes; the previous "+" was inconsistent with the other lists.
    data_types = ", ".join(product["data_types"]) or "unknown"

    pieces = [
        f"{title} — {n} {dataset_word} covering {regions} {domains} ({data_types}).",
    ]
    sample = _pick_summary_variables(
        product["variables_normalized"],
        product["variables"],
    )
    if sample:
        pieces.append("Variables include " + ", ".join(sample) + ".")

    return " ".join(pieces)


def _pick_summary_variables(normalized: list[str], raw: list[str]) -> list[str]:
    """Select up to ``_MAX_SUMMARY_VARIABLES`` informative variable
    names for the product summary.

    cr round-1 PR #89 MEDIUM: alphabetical-first on the raw list
    biased large physics products toward grid bookkeeping
    (``bottomT, deptho, e1t, e2t, e3t, mask``) and obscured the
    actually-important science variables (``thetao``, ``so``,
    ``uo``, ``vo``, ``zos``).

    Strategy:
    1. Drop names that match a known bookkeeping shortname
       (``e1t``/``deptho``/``mask`` and friends).
    2. If an entry has a derived-statistic suffix
       (``_mean``/``_std``/...) AND the unsuffixed base exists in
       the same list, drop the derived variant.
    3. Prefer CF-canonical names from ``normalized`` (heuristic:
       ``len > 6 chars and contains '_'``) over short shortnames.
       Falls back to short shortnames if no canonical names exist
       (true for many indicator / observation products).
    4. Within each tier, sort alphabetically so consecutive
       refreshes produce byte-identical output.
    """
    universe = list(dict.fromkeys(list(normalized) + list(raw)))
    universe_set = set(universe)

    def _has_unsuffixed_sibling(name: str) -> bool:
        """True iff ``name`` ends in a derived suffix AND the
        unsuffixed base is itself present in ``universe``. Keeps
        e.g. ``mlotst_mean`` when ``mlotst`` is the only form, but
        drops ``bottomT_mean`` when ``bottomT`` is also present."""
        for s in _DERIVED_SUFFIXES:
            if name.endswith(s) and name[: -len(s)] in universe_set:
                return True
        return False

    filtered = [
        n for n in universe if n and n not in _BOOKKEEPING_VARS and not _has_unsuffixed_sibling(n)
    ]

    canonical = sorted({n for n in filtered if len(n) > 6 and "_" in n})
    short = sorted({n for n in filtered if not (len(n) > 6 and "_" in n)})
    return (canonical + short)[:_MAX_SUMMARY_VARIABLES]
