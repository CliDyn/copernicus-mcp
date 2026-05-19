"""Canonicalisation primitives for deterministic cache-key construction.

Research §12.6.3. Floats are normalised to 6 decimals (sufficient for ~10cm
ground precision); lists are sorted; dicts are key-sorted recursively.
"""

from __future__ import annotations

from typing import Any

from copernicus_mcp.common.time import iso8601_utc


def _escape_string(s: str) -> str:
    """Self-delimiting quoted form so ``a,b=c`` cannot collide with ``a``+``b=c``."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def canonicalise_value(v: Any) -> str:
    """Render any JSON-ish value as a deterministic, self-delimiting string."""
    if isinstance(v, bool):
        # bool must be checked before int (bool is a subclass of int).
        return "true" if v else "false"
    if isinstance(v, float):
        # Collapse +0.0 / -0.0 to a single canonical form.
        if v == 0.0:
            return "0"
        formatted = f"{v:.6f}".rstrip("0").rstrip(".")
        return formatted if formatted else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, list):
        return "[" + ",".join(sorted(canonicalise_value(item) for item in v)) + "]"
    if isinstance(v, tuple):
        # Tuples preserve order (positional semantics, e.g. bbox).
        return "(" + ",".join(canonicalise_value(item) for item in v) + ")"
    if isinstance(v, dict):
        items = sorted(v.items())
        return (
            "{"
            + ",".join(
                f"{_escape_string(str(k))}={canonicalise_value(val)}"
                for k, val in items
            )
            + "}"
        )
    if isinstance(v, str):
        return _escape_string(v)
    if v is None:
        return "null"
    return _escape_string(str(v))


def canonicalise_bbox(bbox: tuple[float, ...]) -> str:
    """Canonicalise a bbox tuple. Order is preserved (positional semantics).

    For CMEMS we always pass ``[min_lon, min_lat, max_lon, max_lat]``.
    Antimeridian-crossing tuples are preserved as-is — splitting (or rejecting)
    is the schema layer's concern, not this one's.
    """
    return ",".join(canonicalise_value(float(c)) for c in bbox)


def canonicalise_datetime_range(start: str, end: str) -> str:
    """Normalise both endpoints via ``iso8601_utc`` and join with ``..``."""
    return f"{iso8601_utc(start)}..{iso8601_utc(end)}"
