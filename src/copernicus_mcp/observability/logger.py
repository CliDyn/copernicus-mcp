from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import IO, Any

from copernicus_mcp.config.schema import ObservabilityConfig

trace_id_context: ContextVar[str | None] = ContextVar("trace_id", default=None)

# stdlib LogRecord attributes we never want to serialise — they're noise or
# already represented by the canonical fields below.
_STDLIB_NOISE = frozenset(
    {
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


@contextmanager
def bind_trace_id(trace_id: str) -> Iterator[None]:
    """Bind ``trace_id`` for the duration of the with-block.

    Nested calls form a stack via ContextVar tokens. ``ContextVar.reset`` is
    cancellation-safe: an awaited cancellation inside the block still restores
    the previous value on exit.
    """
    token = trace_id_context.set(trace_id)
    try:
        yield
    finally:
        trace_id_context.reset(token)


class JsonFormatter(logging.Formatter):
    """Emit each record as a single-line JSON object on stderr-friendly form."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": trace_id_context.get(),
        }
        for key, value in record.__dict__.items():
            if key in _STDLIB_NOISE or key in payload:
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(
    config: ObservabilityConfig, stream: IO[str] | None = None
) -> None:
    """Configure the root logger with a single stderr handler.

    ``stream`` is exposed only for tests — production code passes
    ``ObservabilityConfig`` and gets stderr. stdout is reserved for the MCP
    transport (the project conventions invariant #4).
    """
    target = stream if stream is not None else sys.stderr
    handler = logging.StreamHandler(target)
    if config.log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )

    root = logging.getLogger()
    for old in list(root.handlers):
        root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
