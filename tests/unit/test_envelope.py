from __future__ import annotations

import pytest
from pydantic import ValidationError as PydValidationError

from copernicus_mcp.data_model.envelope import RequestEnvelope, RequestOptions


def test_envelope_minimal_construction() -> None:
    env = RequestEnvelope(backend="cmems", operation="search", request={"keyword": "sst"})
    assert env.backend == "cmems"
    assert env.operation == "search"
    assert env.request == {"keyword": "sst"}
    assert isinstance(env.options, RequestOptions)


def test_envelope_unknown_backend_rejected() -> None:
    with pytest.raises(PydValidationError):
        RequestEnvelope(backend="cdse", operation="search", request={})  # type: ignore[arg-type]


def test_envelope_unknown_operation_rejected() -> None:
    with pytest.raises(PydValidationError):
        RequestEnvelope(backend="cmems", operation="frobnicate", request={})  # type: ignore[arg-type]


def test_envelope_extra_field_forbidden() -> None:
    with pytest.raises(PydValidationError):
        RequestEnvelope(  # type: ignore[call-arg]
            backend="cmems", operation="search", request={}, surprise=1
        )


def test_request_options_all_optional() -> None:
    opts = RequestOptions()
    assert opts.dry_run is None
    assert opts.confirmed is None
    assert opts.cache_policy is None


def test_request_options_extra_forbidden() -> None:
    with pytest.raises(PydValidationError):
        RequestOptions(unknown_option=True)  # type: ignore[call-arg]


def test_request_options_cache_policy_literal() -> None:
    RequestOptions(cache_policy="use")
    RequestOptions(cache_policy="skip")
    RequestOptions(cache_policy="refresh")
    with pytest.raises(PydValidationError):
        RequestOptions(cache_policy="bogus")  # type: ignore[arg-type]


def test_envelope_frozen() -> None:
    env = RequestEnvelope(backend="cmems", operation="search", request={})
    with pytest.raises((PydValidationError, TypeError, AttributeError)):
        env.backend = "cmems"  # type: ignore[misc]
