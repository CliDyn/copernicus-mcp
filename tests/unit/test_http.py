from __future__ import annotations

import asyncio

import httpx
import pytest


class _Retryable(Exception):
    pass


class _Fatal(Exception):
    pass


async def _fail_then_succeed(call_log: list[int], *, fail_first: int) -> str:
    call_log.append(1)
    if len(call_log) <= fail_first:
        raise _Retryable("nope")
    return "ok"


@pytest.mark.asyncio
async def test_with_retry_returns_value_when_succeeds_first_try() -> None:
    from copernicus_mcp.http import with_retry

    calls = []

    async def factory():
        calls.append(1)
        return 42

    result = await with_retry(
        factory,
        max_attempts=3,
        base_delay=0.0,
        max_delay=0.0,
        retryable=(_Retryable,),
    )
    assert result == 42
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_with_retry_succeeds_on_second_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from copernicus_mcp.http import with_retry

    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr("copernicus_mcp.http.retry.asyncio.sleep", fake_sleep)

    calls: list[int] = []

    async def factory():
        return await _fail_then_succeed(calls, fail_first=1)

    result = await with_retry(
        factory,
        max_attempts=3,
        base_delay=0.01,
        max_delay=1.0,
        retryable=(_Retryable,),
    )
    assert result == "ok"
    assert len(calls) == 2
    assert len(sleeps) == 1


@pytest.mark.asyncio
async def test_with_retry_exhausts_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from copernicus_mcp.http import with_retry

    async def _no_sleep(_d: float) -> None:
        return None

    monkeypatch.setattr("copernicus_mcp.http.retry.asyncio.sleep", _no_sleep)

    calls: list[int] = []

    async def factory():
        calls.append(1)
        raise _Retryable("always fails")

    with pytest.raises(_Retryable):
        await with_retry(
            factory,
            max_attempts=3,
            base_delay=0.0,
            max_delay=0.0,
            retryable=(_Retryable,),
        )
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_with_retry_does_not_retry_non_retryable() -> None:
    from copernicus_mcp.http import with_retry

    calls: list[int] = []

    async def factory():
        calls.append(1)
        raise _Fatal("boom")

    with pytest.raises(_Fatal):
        await with_retry(
            factory,
            max_attempts=5,
            base_delay=0.0,
            max_delay=0.0,
            retryable=(_Retryable,),
        )
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_with_retry_propagates_cancelled_immediately() -> None:
    from copernicus_mcp.http import with_retry

    calls: list[int] = []

    async def factory():
        calls.append(1)
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await with_retry(
            factory,
            max_attempts=5,
            base_delay=0.0,
            max_delay=0.0,
            retryable=(_Retryable, asyncio.CancelledError),
        )
    # Even if CancelledError is in retryable, it must NOT retry — invariant #3.
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_with_retry_delay_capped_at_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from copernicus_mcp.http import with_retry

    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr("copernicus_mcp.http.retry.asyncio.sleep", fake_sleep)
    # Force jitter to its upper bound (uniform returns hi).
    monkeypatch.setattr(
        "copernicus_mcp.http.retry.random.uniform",
        lambda lo, hi: hi,
    )

    calls: list[int] = []

    async def factory():
        calls.append(1)
        raise _Retryable

    with pytest.raises(_Retryable):
        await with_retry(
            factory,
            max_attempts=4,
            base_delay=1.0,
            max_delay=2.5,
            retryable=(_Retryable,),
        )
    assert all(s <= 2.5 + 1e-9 for s in sleeps)
    assert any(s == 2.5 for s in sleeps)


def test_http_client_factory_returns_async_client_with_timeout() -> None:
    from copernicus_mcp.config import HttpConfig
    from copernicus_mcp.http import HttpClientFactory

    cfg = HttpConfig(default_timeout_seconds=42)
    factory = HttpClientFactory(cfg)
    client = factory.create("cmems")
    try:
        assert isinstance(client, httpx.AsyncClient)
        assert client.timeout.read == 42
    finally:
        # Sync close — AsyncClient supports it for unconnected clients.
        asyncio.run(client.aclose())


def test_http_client_factory_unknown_backend_warns(caplog) -> None:
    import logging

    from copernicus_mcp.config import HttpConfig
    from copernicus_mcp.http import HttpClientFactory

    factory = HttpClientFactory(HttpConfig())
    with caplog.at_level(logging.WARNING, logger="copernicus_mcp.http.client_factory"):
        client = factory.create("unknown-backend")
    assert any("unknown-backend" in rec.message for rec in caplog.records)
    asyncio.run(client.aclose())
