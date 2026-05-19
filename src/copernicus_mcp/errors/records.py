from __future__ import annotations

import copy
import secrets
import time
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from copernicus_mcp.observability.logger import trace_id_context

# Canonical wire names — must match research §13.2 exactly.
ErrorClass = Literal[
    "AuthError",
    "QuotaError",
    "TermsNotAcceptedError",
    "ValidationError",
    "NotFoundError",
    "CoverageUnavailableError",
    "NetworkError",
    "TimeoutError",
    "BackendError",
    "CancelledError",
    "CacheError",
]

RecoveryAction = Literal[
    "retry_automatic",
    "retry_with_modification",
    "configure_credentials",
    "rotate_credentials",
    "accept_terms",
    "order_dad_product",
    "wait_for_quota_reset",
    "use_alternative_tool",
    "modify_request_parameters",
    "report_to_administrator",
    "no_action",
]

Severity = Literal["info", "warning", "error", "critical"]


def new_error_id() -> str:
    """Stable, sortable, collision-resistant error identifier."""
    return f"err-{int(time.time())}-{secrets.token_hex(4)}"


def _isolate(value: Any) -> Any:
    """Deep-copy ``value`` so caller mutations of the source cannot leak in.

    Codex T-007 review: Pydantic ``frozen=True`` is shallow — without this
    helper a caller could keep a reference to the dict they handed us and
    mutate it post-construction. We deliberately do NOT wrap in
    ``MappingProxyType`` — that breaks ``model_dump_json()`` (the wire
    contract). Defense-in-depth here is "no shared reference"; tamper
    resistance against handlers mutating ``record.context`` directly is
    out of scope for Iter 1 (frozen=True still blocks attribute reassign).
    """
    return copy.deepcopy(value)


class ErrorRecord(BaseModel):
    """Structured payload that travels alongside every CopernicusMcpError.

    Wire contract — any change here is a breaking change to MCP responses.
    See the upstream documentation §13.2.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    error_class: ErrorClass
    error_subclass: str | None = None
    retryable: bool = False
    automatic_retry_recommended: bool = False
    recovery_action: RecoveryAction = "report_to_administrator"
    recovery_url: str | None = None
    next_action_hint: str | None = None
    message: str
    context: Any = Field(default_factory=dict)
    timestamp_utc: datetime
    error_id: str
    trace_id: str | None = None
    backend_diagnostics: Any = None
    severity: Severity = "error"

    @field_validator("context", mode="after")
    @classmethod
    def _isolate_context(cls, v: Any) -> Any:
        return _isolate(v) if v is not None else {}

    @field_validator("backend_diagnostics", mode="after")
    @classmethod
    def _isolate_diag(cls, v: Any) -> Any:
        return _isolate(v) if v is not None else None


def build_error_record(
    error_class: ErrorClass,
    message: str,
    *,
    error_subclass: str | None = None,
    retryable: bool = False,
    automatic_retry_recommended: bool = False,
    recovery_action: RecoveryAction = "report_to_administrator",
    recovery_url: str | None = None,
    next_action_hint: str | None = None,
    context: Any = None,
    backend_diagnostics: Any = None,
    severity: Severity = "error",
    trace_id: str | None = None,
    error_id: str | None = None,
    timestamp_utc: datetime | None = None,
) -> ErrorRecord:
    """Construct an ``ErrorRecord`` with sensible defaults.

    ``trace_id`` defaults to the current ``trace_id_context`` value;
    ``error_id`` defaults to a fresh ``new_error_id()``;
    ``timestamp_utc`` defaults to ``datetime.now(UTC)``.

    Caller-supplied kwargs override the defaults.
    """
    return ErrorRecord(
        error_class=error_class,
        error_subclass=error_subclass,
        retryable=retryable,
        automatic_retry_recommended=automatic_retry_recommended,
        recovery_action=recovery_action,
        recovery_url=recovery_url,
        next_action_hint=next_action_hint,
        message=message,
        context=context if context is not None else {},
        timestamp_utc=timestamp_utc or datetime.now(UTC),
        error_id=error_id or new_error_id(),
        trace_id=trace_id if trace_id is not None else trace_id_context.get(),
        backend_diagnostics=backend_diagnostics,
        severity=severity,
    )
