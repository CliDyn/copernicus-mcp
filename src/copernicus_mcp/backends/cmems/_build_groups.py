"""Lint helper for the v1 groups manifest.

T-CMEMS-HIER-004. The groups manifest layers human-curated routing
intent on top of the rule-derived product manifest: each group has
include/exclude phrase hints, a one-line summary, and a list of
``product_ids`` the runtime router should consider when the query
matches the include phrases.

``validate_groups`` returns a list of human-readable problems —
empty when the manifest is consistent. The refresh pipeline (and
this PR's bundled-manifest test) call it before shipping a new
``groups.json`` so a broken cross-reference, missing field, or
duplicate id surfaces at bundle-time, not at runtime.

Lint, not schema: we pin the cross-reference invariants the
router cares about, not every nice-to-have JSON shape.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

_REQUIRED_STRING_FIELDS: tuple[str, ...] = ("group_id", "group_title", "summary")


def validate_groups(
    groups: list[dict[str, Any]],
    products: list[dict[str, Any]],
) -> list[str]:
    """Lint ``groups`` against ``products``. Returns a list of
    problem strings. Empty list = consistent manifest."""
    problems: list[str] = []
    known_product_ids = {
        p.get("product_id") for p in products if isinstance(p.get("product_id"), str)
    }
    referenced_product_ids: set[str] = set()
    seen_group_ids: set[str] = set()

    for idx, group in enumerate(groups):
        gid = group.get("group_id") if isinstance(group, dict) else None
        label = gid if isinstance(gid, str) and gid else f"<group #{idx}>"

        # Invariant 5: required string fields present + non-empty.
        for field in _REQUIRED_STRING_FIELDS:
            if field not in group:
                problems.append(f"{label}: missing required field '{field}'")
                continue
            value = group[field]
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{label}: '{field}' must be a non-empty string")

        # Invariant 6: group_id uniqueness.
        if isinstance(gid, str) and gid:
            if gid in seen_group_ids:
                problems.append(f"{gid}: duplicate group_id")
            seen_group_ids.add(gid)

        # Invariant 3: include_when_query_mentions non-empty.
        include = group.get("include_when_query_mentions")
        if not isinstance(include, list) or not include:
            problems.append(f"{label}: 'include_when_query_mentions' must be a non-empty list")

        # Invariant 4: exclude_when_query_mentions field must be
        # present (may be empty list).
        if "exclude_when_query_mentions" not in group:
            problems.append(f"{label}: missing field 'exclude_when_query_mentions'")
        elif not isinstance(group["exclude_when_query_mentions"], list):
            problems.append(f"{label}: 'exclude_when_query_mentions' must be a list")

        # Invariant 2: non-empty product_ids.
        product_ids = group.get("product_ids")
        if not isinstance(product_ids, list) or not product_ids:
            problems.append(f"{label}: empty product_ids list")
        else:
            # Invariant 1: every cited product_id exists.
            for pid in product_ids:
                if not isinstance(pid, str):
                    problems.append(f"{label}: non-string product_id {pid!r}")
                    continue
                if pid not in known_product_ids:
                    problems.append(f"{label}: unknown product_id '{pid}'")
                referenced_product_ids.add(pid)

    # Invariant 7: orphan products (informational; surface so the
    # curator notices, but not a hard fail).
    for pid in sorted(known_product_ids - referenced_product_ids):
        problems.append(f"orphan product not cited by any group: {pid}")

    return problems


def collect_referenced_products(
    groups: Iterable[dict[str, Any]],
) -> set[str]:
    """Helper for callers that want the set of all product_ids any
    group references (e.g. to compute coverage stats)."""
    out: set[str] = set()
    for g in groups:
        for pid in g.get("product_ids") or []:
            if isinstance(pid, str) and pid:
                out.add(pid)
    return out
