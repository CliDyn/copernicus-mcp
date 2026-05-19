from __future__ import annotations

import json
import logging
import sys
from io import StringIO

import pytest


def _make_record(
    msg: str = "hi", level: int = logging.INFO, extra: dict | None = None
) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger",
        level=level,
        pathname=__file__,
        lineno=10,
        msg=msg,
        args=(),
        exc_info=None,
    )
    if extra:
        for k, v in extra.items():
            setattr(record, k, v)
    return record


def test_json_formatter_emits_parseable_json() -> None:
    from copernicus_mcp.observability.logger import JsonFormatter

    out = JsonFormatter().format(_make_record("hello"))
    parsed = json.loads(out)
    assert parsed["message"] == "hello"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test.logger"
    assert "timestamp" in parsed
    assert parsed["trace_id"] is None


def test_trace_id_binds_and_resets() -> None:
    from copernicus_mcp.observability.logger import JsonFormatter, bind_trace_id

    fmt = JsonFormatter()
    assert json.loads(fmt.format(_make_record()))["trace_id"] is None
    with bind_trace_id("trace-123"):
        assert json.loads(fmt.format(_make_record()))["trace_id"] == "trace-123"
    assert json.loads(fmt.format(_make_record()))["trace_id"] is None


def test_extra_fields_merged() -> None:
    from copernicus_mcp.observability.logger import JsonFormatter

    record = _make_record(extra={"foo": "bar", "count": 7})
    parsed = json.loads(JsonFormatter().format(record))
    assert parsed["foo"] == "bar"
    assert parsed["count"] == 7


def test_console_format_is_not_json() -> None:
    from copernicus_mcp.config import ObservabilityConfig
    from copernicus_mcp.observability.logger import get_logger, setup_logging

    cfg = ObservabilityConfig(structured_logging=False, log_format="console")
    buf = StringIO()
    setup_logging(cfg, stream=buf)
    get_logger("t.console").warning("plain message")
    output = buf.getvalue()
    assert output, "expected non-empty output"
    assert not output.lstrip().startswith("{")
    assert "plain message" in output


def test_setup_logging_writes_to_stderr_not_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from copernicus_mcp.config import ObservabilityConfig
    from copernicus_mcp.observability.logger import get_logger, setup_logging

    setup_logging(ObservabilityConfig())
    get_logger("t.stderr").error("boom")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "boom" in captured.err


def test_setup_logging_default_stream_is_sys_stderr() -> None:
    from copernicus_mcp.config import ObservabilityConfig
    from copernicus_mcp.observability.logger import setup_logging

    setup_logging(ObservabilityConfig())
    handlers = logging.getLogger().handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)
    assert handlers[0].stream is sys.stderr


def test_get_logger_returns_named_logger() -> None:
    from copernicus_mcp.observability.logger import get_logger

    assert get_logger("foo.bar").name == "foo.bar"
