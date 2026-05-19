from __future__ import annotations

import time

import httpx

from copernicus_mcp.config.schema import HttpConfig
from copernicus_mcp.observability.logger import get_logger

logger = get_logger(__name__)

_KNOWN_BACKENDS: frozenset[str] = frozenset({"cmems"})

_DEFAULT_LIMITS = httpx.Limits(max_connections=10, max_keepalive_connections=5)

_REQUEST_START_ATTR = "_copernicus_mcp_started_at"


async def _log_request(request: httpx.Request) -> None:
    setattr(request, _REQUEST_START_ATTR, time.monotonic())
    # URL without query string — query may carry credentials in some backends.
    safe_url = f"{request.url.scheme}://{request.url.host}{request.url.path}"
    logger.debug(
        "http request",
        extra={"method": request.method, "url": safe_url},
    )


async def _log_response(response: httpx.Response) -> None:
    started = getattr(response.request, _REQUEST_START_ATTR, None)
    elapsed_ms = round((time.monotonic() - started) * 1000, 2) if started else None
    safe_url = (
        f"{response.request.url.scheme}://{response.request.url.host}"
        f"{response.request.url.path}"
    )
    logger.info(
        "http response",
        extra={
            "method": response.request.method,
            "url": safe_url,
            "status_code": response.status_code,
            "elapsed_ms": elapsed_ms,
        },
    )


class HttpClientFactory:
    """Builds configured ``httpx.AsyncClient`` instances per backend."""

    def __init__(self, http_config: HttpConfig) -> None:
        self._cfg = http_config

    def create(self, backend_id: str) -> httpx.AsyncClient:
        if backend_id not in _KNOWN_BACKENDS:
            logger.warning(
                "creating http client for unknown backend %s; using defaults",
                backend_id,
                extra={"backend_id": backend_id},
            )
        return httpx.AsyncClient(
            timeout=self._cfg.default_timeout_seconds,
            limits=_DEFAULT_LIMITS,
            event_hooks={"request": [_log_request], "response": [_log_response]},
        )
