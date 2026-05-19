from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from copernicus_mcp.observability.logger import get_logger

T = TypeVar("T")

logger = get_logger(__name__)


async def with_retry(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
    base_delay: float,
    max_delay: float,
    retryable: tuple[type[BaseException], ...],
) -> T:
    """Run ``coro_factory()`` with decorrelated-jitter retries.

    - ``asyncio.CancelledError`` is **never** retried, even if listed in
      ``retryable``. It's a control-flow signal owned by the orchestrator
      (the project conventions invariant #3).
    - Non-retryable exceptions propagate immediately.
    - After ``max_attempts``, the last exception is re-raised.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if base_delay < 0 or max_delay < 0:
        raise ValueError("base_delay and max_delay must be >= 0")
    if max_delay < base_delay:
        raise ValueError("max_delay must be >= base_delay")

    prev_delay = base_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_factory()
        except asyncio.CancelledError:
            raise
        except retryable as exc:
            if attempt >= max_attempts:
                raise
            hi = max(prev_delay * 3, base_delay)
            delay = min(max_delay, random.uniform(base_delay, hi))
            prev_delay = delay if delay > 0 else base_delay
            logger.warning(
                "retrying after %s",
                type(exc).__name__,
                extra={
                    "attempt": attempt,
                    "delay_seconds": delay,
                    "exception_type": type(exc).__name__,
                },
            )
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover
