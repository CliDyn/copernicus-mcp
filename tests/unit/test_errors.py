from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

# ---------- canonical ErrorClass strings (from research §13.2) -----------

CANONICAL_CLASS_NAMES: dict[str, str] = {
    "AuthError": "AuthError",
    "QuotaError": "QuotaError",
    "TermsNotAcceptedError": "TermsNotAcceptedError",
    "ValidationError": "ValidationError",
    "NotFoundError": "NotFoundError",
    "CoverageUnavailableError": "CoverageUnavailableError",
    "NetworkError": "NetworkError",
    "TimeoutError": "TimeoutError",
    "BackendError": "BackendError",
    "OperationCancelledError": "CancelledError",  # wire name preserved
    "CacheError": "CacheError",
}


def _all_classes():
    from copernicus_mcp import errors

    return [getattr(errors, name) for name in CANONICAL_CLASS_NAMES.keys()]


# ---------- record / build_error_record --------------------------------------


def test_new_error_id_unique() -> None:
    from copernicus_mcp.errors import new_error_id

    ids = {new_error_id() for _ in range(20)}
    assert len(ids) == 20


def test_build_error_record_defaults_and_overrides() -> None:
    from copernicus_mcp.errors import ErrorRecord, build_error_record

    rec = build_error_record("AuthError", message="bad creds")
    assert isinstance(rec, ErrorRecord)
    assert rec.error_class == "AuthError"
    assert rec.message == "bad creds"
    assert rec.severity == "error"
    assert rec.error_id.startswith("err-")
    # Override.
    rec2 = build_error_record("AuthError", message="x", severity="critical")
    assert rec2.severity == "critical"


def test_error_record_is_frozen_top_level() -> None:
    from pydantic import ValidationError as PydValidationError

    from copernicus_mcp.errors import build_error_record

    rec = build_error_record("AuthError", message="x")
    with pytest.raises((PydValidationError, AttributeError, TypeError)):
        rec.message = "mutated"  # type: ignore[misc]


def test_error_record_context_is_deeply_immutable() -> None:
    """A frozen Pydantic model is shallow — codex T-007 review.

    The implementation must defensively deep-copy and freeze nested
    structures so mutating the original input dict does not show through.
    """
    from copernicus_mcp.errors import build_error_record

    src_ctx = {"key": "value", "nested": {"a": 1}}
    rec = build_error_record("AuthError", message="x", context=src_ctx)
    src_ctx["key"] = "MUTATED"
    src_ctx["nested"]["a"] = 999
    assert rec.context["key"] == "value"
    assert rec.context["nested"]["a"] == 1


def test_error_record_extra_forbid() -> None:
    from datetime import UTC, datetime

    from pydantic import ValidationError as PydValidationError

    from copernicus_mcp.errors import ErrorRecord

    with pytest.raises(PydValidationError):
        ErrorRecord.model_validate(
            {
                "error_class": "AuthError",
                "message": "x",
                "timestamp_utc": datetime.now(UTC),
                "error_id": "err-test",
                "surprise_field": 1,
            }
        )


def test_error_record_timestamp_is_iso_utc() -> None:
    from copernicus_mcp.errors import build_error_record

    rec = build_error_record("AuthError", message="x")
    iso = rec.timestamp_utc.isoformat()
    # UTC offset must be present (no naive datetimes).
    assert iso.endswith("+00:00") or iso.endswith("Z")


# ---------- exception classes ------------------------------------------------


@pytest.mark.parametrize(
    "py_class_name,wire_name", list(CANONICAL_CLASS_NAMES.items())
)
def test_each_class_canonical_wire_name(
    py_class_name: str, wire_name: str
) -> None:
    from copernicus_mcp import errors

    cls = getattr(errors, py_class_name)
    exc = cls("test message")
    assert exc.error_record.error_class == wire_name


def test_eleven_classes_are_distinct() -> None:
    classes = _all_classes()
    assert len(classes) == 11
    assert len(set(classes)) == 11
    from copernicus_mcp.errors import CopernicusMcpError

    for c in classes:
        assert issubclass(c, CopernicusMcpError)
        assert c is not CopernicusMcpError
    # Pairwise siblings (no class is subclass of another among the 11).
    for a in classes:
        for b in classes:
            if a is b:
                continue
            assert not issubclass(a, b)


def test_message_is_preserved() -> None:
    from copernicus_mcp.errors import AuthError

    exc = AuthError("hello world")
    assert exc.error_record.message == "hello world"
    # __str__ returns only message — never structured context.
    assert str(exc) == "hello world"


def test_trace_id_from_context_is_captured() -> None:
    from copernicus_mcp.errors import AuthError
    from copernicus_mcp.observability.logger import bind_trace_id

    with bind_trace_id("trc-1"):
        exc = AuthError("oops")
    assert exc.error_record.trace_id == "trc-1"
    # outside: None
    exc2 = AuthError("no trace")
    assert exc2.error_record.trace_id is None


def test_explicit_record_overrides_auto() -> None:
    from copernicus_mcp.errors import AuthError, build_error_record

    rec = build_error_record(
        "AuthError",
        message="explicit",
        recovery_action="rotate_credentials",
    )
    exc = AuthError("ignored", record=rec)
    assert exc.error_record is rec
    assert exc.error_record.recovery_action == "rotate_credentials"


def test_error_record_is_readonly_property() -> None:
    """Handlers must not be able to swap the record on a raised exception."""
    from copernicus_mcp.errors import AuthError, build_error_record

    exc = AuthError("x")
    other = build_error_record("AuthError", message="other")
    with pytest.raises(AttributeError):
        exc.error_record = other  # type: ignore[misc]


def test_str_does_not_dump_context() -> None:
    """Defense against accidental leaks in raw __str__ output."""
    from copernicus_mcp.errors import AuthError, build_error_record

    rec = build_error_record(
        "AuthError",
        message="auth failed",
        context={"sensitive_key": "TOPSECRET-VALUE"},
    )
    exc = AuthError("auth failed", record=rec)
    assert "TOPSECRET-VALUE" not in str(exc)
    assert "TOPSECRET-VALUE" not in repr(exc)
    assert str(exc) == "auth failed"


# ---------- per-class default recovery_action --------------------------------


@pytest.mark.parametrize(
    "py_class,expected_action",
    [
        ("AuthError", "configure_credentials"),
        ("TermsNotAcceptedError", "accept_terms"),
        ("ValidationError", "modify_request_parameters"),
        ("NotFoundError", "modify_request_parameters"),
        ("CoverageUnavailableError", "modify_request_parameters"),
        ("NetworkError", "retry_automatic"),
        ("TimeoutError", "retry_automatic"),
        ("BackendError", "report_to_administrator"),
        ("OperationCancelledError", "no_action"),
        ("QuotaError", "wait_for_quota_reset"),
        ("CacheError", "report_to_administrator"),
    ],
)
def test_per_class_default_recovery_action(
    py_class: str, expected_action: str
) -> None:
    from copernicus_mcp import errors

    cls = getattr(errors, py_class)
    exc = cls("msg")
    assert exc.error_record.recovery_action == expected_action


# ---------- cancellation isolation --------------------------------------------


def test_cancellation_classes_unrelated() -> None:
    from copernicus_mcp.errors import OperationCancelledError

    assert not issubclass(asyncio.CancelledError, OperationCancelledError)
    assert not issubclass(OperationCancelledError, asyncio.CancelledError)


def test_no_conflated_except_clause_in_src() -> None:
    """AST walk: no ``except`` clause may catch both OperationCancelledError
    and ``asyncio.CancelledError`` together (the project conventions invariant #3).

    Covers tuple form ``except (A, B)``, union form ``except A | B`` (3.10+),
    ``except*`` (PEP 654), and aliasing variants. Bare ``except:`` and
    ``except BaseException`` are also flagged because they implicitly trap
    ``asyncio.CancelledError``.
    """
    import ast

    src = Path(__file__).resolve().parents[2] / "src"

    def _names(node: ast.expr) -> list[str]:
        """Flatten a handler's type expression to dotted-name strings."""
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, ast.Attribute):
            parts = []
            cur: ast.expr = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            return [".".join(reversed(parts))]
        if isinstance(node, ast.Tuple):
            out: list[str] = []
            for elt in node.elts:
                out.extend(_names(elt))
            return out
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return _names(node.left) + _names(node.right)
        return []

    bad: list[str] = []
    for py in src.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            continue
        for h in ast.walk(tree):
            if not isinstance(h, ast.ExceptHandler):
                continue
            if h.type is None:
                bad.append(f"{py}:{h.lineno} bare except")
                continue
            names = _names(h.type)
            if "BaseException" in names:
                bad.append(f"{py}:{h.lineno} except BaseException")
                continue
            has_async_cancel = any(
                n.endswith("CancelledError") and n != "OperationCancelledError"
                for n in names
            )
            has_app_cancel = "OperationCancelledError" in names
            if has_async_cancel and has_app_cancel:
                bad.append(f"{py}:{h.lineno} conflated cancellation")
    assert not bad, f"forbidden except clauses: {bad}"


def test_error_record_round_trips_through_json() -> None:
    """ErrorRecord must be JSON-serializable for the wire contract."""
    import json

    from copernicus_mcp.errors import ErrorRecord, build_error_record

    rec = build_error_record(
        "AuthError",
        message="hi",
        context={"k": "v", "nested": {"x": 1}},
        backend_diagnostics={"http_status": 401},
    )
    text = rec.model_dump_json()
    payload = json.loads(text)
    assert payload["error_class"] == "AuthError"
    assert payload["message"] == "hi"
    assert payload["context"] == {"k": "v", "nested": {"x": 1}}
    assert payload["backend_diagnostics"] == {"http_status": 401}
    # And model_validate accepts the round-trip.
    rec2 = ErrorRecord.model_validate(payload)
    assert rec2.error_class == rec.error_class
    assert rec2.context == rec.context


def test_backend_diagnostics_is_isolated() -> None:
    from copernicus_mcp.errors import build_error_record

    src = {"http_status": 500, "headers": {"x-ratelimit": "0"}}
    rec = build_error_record(
        "BackendError",
        message="x",
        backend_diagnostics=src,
    )
    src["http_status"] = 999
    src["headers"]["x-ratelimit"] = "MUTATED"
    assert rec.backend_diagnostics["http_status"] == 500
    assert rec.backend_diagnostics["headers"]["x-ratelimit"] == "0"


def test_direct_subclass_without_canonical_name_raises() -> None:
    """Subclassing CopernicusMcpError directly without setting both ClassVars
    must fail loudly when constructed without an explicit record (codex)."""
    from copernicus_mcp.errors import CopernicusMcpError

    class BadCustom(CopernicusMcpError):
        pass

    with pytest.raises(NotImplementedError):
        BadCustom("oops")
