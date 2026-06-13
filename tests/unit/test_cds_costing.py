"""T-CDS-EST2-001: CDS ``/costing`` client + ``CostingResult``.

No network: a fake ``HttpClientFactory`` hands ``fetch_costing`` a real
``httpx.AsyncClient`` backed by ``httpx.MockTransport`` so real status-code,
JSON and transport-error handling is exercised, not a hand-rolled mock.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from copernicus_mcp.backends.cds.costing import CostingResult, fetch_costing


class _FakeCatalogue:
    def __init__(self, store: str | None) -> None:
        self._store = store

    def store_for(self, dataset_id: str) -> str | None:  # noqa: ARG002
        return self._store


class _FakeFactory:
    """Returns an AsyncClient whose transport is the supplied handler.

    Records the last requested URL so routing can be asserted.
    """

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self._handler = handler
        self.last_url: str | None = None

    def create(self, backend_id: str) -> httpx.AsyncClient:  # noqa: ARG002
        def _record(request: httpx.Request) -> httpx.Response:
            self.last_url = str(request.url)
            return self._handler(request)

        return httpx.AsyncClient(transport=httpx.MockTransport(_record))


def _json_handler(payload: Any, status: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(status, json=payload)

    return handler


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# --- CostingResult ------------------------------------------------------


def test_costing_result_exceeds_limit_true_when_over() -> None:
    assert CostingResult(units=1827.0, limit=400.0).exceeds_limit is True


def test_costing_result_not_exceeds_when_equal() -> None:
    assert CostingResult(units=400.0, limit=400.0).exceeds_limit is False


def test_costing_result_not_exceeds_when_under() -> None:
    assert CostingResult(units=24.0, limit=121000.0).exceeds_limit is False


# --- fetch_costing happy path -------------------------------------------


def test_fetch_costing_parses_units_and_limit() -> None:
    factory = _FakeFactory(_json_handler({"id": "size", "cost": 1827.0, "limit": 400.0}))
    result = _run(
        fetch_costing(
            "derived-era5-single-levels-daily-statistics",
            {"variable": ["2m_temperature"]},
            http_client_factory=factory,
            catalogue=_FakeCatalogue("cds"),
        )
    )
    assert result is not None
    assert result.units == 1827.0
    assert result.limit == 400.0
    assert result.exceeds_limit is True


def test_fetch_costing_routes_to_store_endpoint() -> None:
    factory = _FakeFactory(_json_handler({"id": "size", "cost": 2.0, "limit": 10.0}))
    _run(
        fetch_costing(
            "cams-global-reanalysis-eac4",
            {},
            http_client_factory=factory,
            catalogue=_FakeCatalogue("ads"),
        )
    )
    assert factory.last_url is not None
    assert factory.last_url.startswith("https://ads.atmosphere.copernicus.eu/api")
    assert factory.last_url.endswith(
        "/retrieve/v1/processes/cams-global-reanalysis-eac4/costing"
    )


# --- fetch_costing failure modes all return None ------------------------


@pytest.mark.parametrize("status", [400, 404, 422, 500, 503])
def test_fetch_costing_non_2xx_returns_none(status: int) -> None:
    factory = _FakeFactory(_json_handler({"detail": "nope"}, status=status))
    result = _run(
        fetch_costing("x", {}, http_client_factory=factory, catalogue=_FakeCatalogue("cds"))
    )
    assert result is None


def test_fetch_costing_non_json_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(200, text="not json")

    result = _run(
        fetch_costing(
            "x", {}, http_client_factory=_FakeFactory(handler), catalogue=_FakeCatalogue("cds")
        )
    )
    assert result is None


@pytest.mark.parametrize("payload", [{"id": "size"}, {"cost": 1.0}, {"limit": 5.0}, {}])
def test_fetch_costing_missing_keys_returns_none(payload: dict[str, Any]) -> None:
    result = _run(
        fetch_costing(
            "x",
            {},
            http_client_factory=_FakeFactory(_json_handler(payload)),
            catalogue=_FakeCatalogue("cds"),
        )
    )
    assert result is None


def test_fetch_costing_transport_error_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        raise httpx.ConnectError("boom")

    result = _run(
        fetch_costing(
            "x", {}, http_client_factory=_FakeFactory(handler), catalogue=_FakeCatalogue("cds")
        )
    )
    assert result is None


def test_fetch_costing_propagates_cancelled_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        _run(
            fetch_costing(
                "x", {}, http_client_factory=_FakeFactory(handler), catalogue=_FakeCatalogue("cds")
            )
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"cost": float("nan"), "limit": 400.0},
        {"cost": float("inf"), "limit": 400.0},
        {"cost": 24.0, "limit": float("nan")},
        {"cost": 24.0, "limit": 0.0},
        {"cost": 24.0, "limit": -5.0},
    ],
)
def test_fetch_costing_non_finite_or_nonpositive_limit_returns_none(payload: dict) -> None:
    """A malformed/advisory-hostile costing payload (NaN, Inf, non-positive
    limit) must degrade to None, not crash a later int(units)."""
    result = _run(
        fetch_costing(
            "x",
            {},
            http_client_factory=_FakeFactory(_json_handler(payload)),
            catalogue=_FakeCatalogue("cds"),
        )
    )
    assert result is None


def test_fetch_costing_unknown_store_falls_back_to_cds() -> None:
    factory = _FakeFactory(_json_handler({"id": "size", "cost": 1.0, "limit": 9.0}))
    _run(
        fetch_costing(
            "mystery", {}, http_client_factory=factory, catalogue=_FakeCatalogue(None)
        )
    )
    assert factory.last_url is not None
    assert factory.last_url.startswith("https://cds.climate.copernicus.eu/api")
