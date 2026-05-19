"""Datetime helpers shared across the package."""

from __future__ import annotations

from datetime import UTC, datetime

from copernicus_mcp.errors import ValidationError


def iso8601_utc(value: str) -> str:
    """Normalise a tz-aware ISO 8601 string to ``YYYY-MM-DDTHH:MM:SSZ`` (UTC).

    Drops sub-second precision. Naive (tz-less) inputs are rejected so that
    cache keys derived from datetimes never depend on the caller's local clock.
    """
    if not isinstance(value, str):
        raise ValidationError(
            f"expected ISO 8601 string, got {type(value).__name__}"
        )
    raw = value.strip()
    candidate = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        # codex T-026 HIGH: do not echo the raw user input in the error message
        # (an attacker / careless caller could put ``"password=hunter2"`` in a
        # datetime field and have it surface in tool output via validate()).
        raise ValidationError(
            "invalid ISO 8601 datetime; expected e.g. 2024-01-01T00:00:00Z"
        ) from exc
    if parsed.tzinfo is None:
        raise ValidationError(
            "datetime is missing a timezone offset; "
            "supply 'Z' or an explicit ±HH:MM offset"
        )
    utc = parsed.astimezone(UTC).replace(microsecond=0)
    return utc.strftime("%Y-%m-%dT%H:%M:%SZ")
