from __future__ import annotations

import pytest

from copernicus_mcp.common.time import iso8601_utc
from copernicus_mcp.errors import ValidationError


def test_z_suffix_passthrough() -> None:
    assert iso8601_utc("2024-01-02T03:04:05Z") == "2024-01-02T03:04:05Z"


def test_offset_normalised_to_utc() -> None:
    # +02:00 means local clock is 2h ahead of UTC; 03:04:05+02:00 == 01:04:05Z
    assert iso8601_utc("2024-01-02T03:04:05+02:00") == "2024-01-02T01:04:05Z"


def test_negative_offset() -> None:
    assert iso8601_utc("2024-01-02T03:04:05-05:00") == "2024-01-02T08:04:05Z"


def test_fractional_seconds_dropped() -> None:
    assert iso8601_utc("2024-01-02T03:04:05.123456Z") == "2024-01-02T03:04:05Z"


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        iso8601_utc("2024-01-02T03:04:05")
    assert "timezone" in str(exc_info.value).lower() or "tz" in str(exc_info.value).lower()


def test_garbage_rejected() -> None:
    with pytest.raises(ValidationError):
        iso8601_utc("not-a-date")
