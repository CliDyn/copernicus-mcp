from __future__ import annotations

import pytest

from copernicus_mcp.data_model.canonicalisation import (
    canonicalise_bbox,
    canonicalise_datetime_range,
    canonicalise_value,
)


def test_float_trailing_zeros_stripped() -> None:
    assert canonicalise_value(10.0) == "10"
    assert canonicalise_value(10.500000) == "10.5"


def test_float_six_decimal_precision() -> None:
    assert canonicalise_value(-10.000001) == "-10.000001"
    # Past 6 decimals → rounded
    assert canonicalise_value(-10.0000001) == "-10"


def test_float_zero() -> None:
    assert canonicalise_value(0.0) == "0"


def test_list_sorted() -> None:
    assert canonicalise_value(["b", "a", "c"]) == '["a","b","c"]'


def test_dict_sorted_recursively() -> None:
    out = canonicalise_value({"b": 1, "a": [2.0, 1.0]})
    assert out == '{"a"=[1,2],"b"=1}'


def test_none_str_int() -> None:
    assert canonicalise_value(None) == "null"
    assert canonicalise_value("hello") == '"hello"'
    assert canonicalise_value(7) == "7"


def test_string_self_delimiting() -> None:
    """A string containing canonical delimiters must not be confused with structure."""
    a = canonicalise_value({"keyword": "temperature,limit=5"})
    b = canonicalise_value({"keyword": "temperature", "limit": 5})
    assert a != b


def test_negative_zero_canonical() -> None:
    assert canonicalise_value(-0.0) == "0"
    assert canonicalise_value(0.0) == "0"


def test_canonicalise_bbox_simple() -> None:
    assert canonicalise_bbox((-10.0, 35.0, 5.0, 45.0)) == "-10,35,5,45"


def test_canonicalise_bbox_precision() -> None:
    a = canonicalise_bbox((-10.000001, 35.0, 5.0, 45.0))
    b = canonicalise_bbox((-10.0, 35.0, 5.0, 45.0))
    assert a != b


def test_canonicalise_datetime_range_normalises_offsets() -> None:
    out = canonicalise_datetime_range(
        "2024-01-01T02:00:00+02:00", "2024-01-02T00:00:00Z"
    )
    assert out == "2024-01-01T00:00:00Z..2024-01-02T00:00:00Z"


def test_canonicalise_datetime_range_rejects_naive() -> None:
    from copernicus_mcp.errors import ValidationError

    with pytest.raises(ValidationError):
        canonicalise_datetime_range("2024-01-01T00:00:00", "2024-01-02T00:00:00Z")
