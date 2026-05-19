"""Eleven canonical error classes + ``CopernicusMcpError`` base.

Cancellation distinction (the project conventions invariant #3):
- ``OperationCancelledError`` is the **app-level** "operation was cancelled"
  exception. Its ``error_record.error_class`` is the wire-name string
  ``"CancelledError"``.
- ``asyncio.CancelledError`` is a ``BaseException`` and is a control-flow
  signal owned by the orchestrator. **Never** catch both in the same
  ``except`` clause; never wrap ``asyncio.CancelledError`` into
  ``OperationCancelledError`` — let it propagate.
"""

from __future__ import annotations

from typing import ClassVar

from copernicus_mcp.errors.records import (
    ErrorClass,
    ErrorRecord,
    RecoveryAction,
    build_error_record,
)


class CopernicusMcpError(Exception):
    """Base class for the eleven canonical error types."""

    # Subclasses MUST override both. Leaving them None on the base prevents
    # the foot-gun where someone subclasses CopernicusMcpError directly and
    # silently inherits BackendError's wire name (codex T-007 diff review).
    _canonical_class_name: ClassVar[ErrorClass | None] = None
    _default_recovery_action: ClassVar[RecoveryAction | None] = None

    def __init__(
        self,
        message: str,
        *,
        record: ErrorRecord | None = None,
    ) -> None:
        # Initialise Exception with the message string only — never the
        # structured record — so default str/repr cannot dump context.
        super().__init__(message)
        if record is None:
            if self._canonical_class_name is None:
                raise NotImplementedError(
                    f"{type(self).__name__} must set _canonical_class_name "
                    "(or pass an explicit ``record=``)."
                )
            if self._default_recovery_action is None:
                raise NotImplementedError(
                    f"{type(self).__name__} must set _default_recovery_action "
                    "(or pass an explicit ``record=``)."
                )
            record = build_error_record(
                self._canonical_class_name,
                message=message,
                recovery_action=self._default_recovery_action,
            )
        # Use object.__setattr__ to populate the underlying slot so the
        # public ``error_record`` property remains read-only.
        object.__setattr__(self, "_error_record", record)

    @property
    def error_record(self) -> ErrorRecord:
        return self._error_record  # type: ignore[attr-defined,no-any-return]

    def __str__(self) -> str:
        # Surface only the human-readable message — never the context dict.
        return self.error_record.message


class AuthError(CopernicusMcpError):
    _canonical_class_name: ClassVar[ErrorClass] = "AuthError"
    _default_recovery_action: ClassVar[RecoveryAction] = "configure_credentials"


class QuotaError(CopernicusMcpError):
    _canonical_class_name: ClassVar[ErrorClass] = "QuotaError"
    _default_recovery_action: ClassVar[RecoveryAction] = "wait_for_quota_reset"


class TermsNotAcceptedError(CopernicusMcpError):
    _canonical_class_name: ClassVar[ErrorClass] = "TermsNotAcceptedError"
    _default_recovery_action: ClassVar[RecoveryAction] = "accept_terms"


class ValidationError(CopernicusMcpError):
    _canonical_class_name: ClassVar[ErrorClass] = "ValidationError"
    _default_recovery_action: ClassVar[RecoveryAction] = "modify_request_parameters"


class NotFoundError(CopernicusMcpError):
    _canonical_class_name: ClassVar[ErrorClass] = "NotFoundError"
    _default_recovery_action: ClassVar[RecoveryAction] = "modify_request_parameters"


class CoverageUnavailableError(CopernicusMcpError):
    _canonical_class_name: ClassVar[ErrorClass] = "CoverageUnavailableError"
    _default_recovery_action: ClassVar[RecoveryAction] = "modify_request_parameters"


class NetworkError(CopernicusMcpError):
    _canonical_class_name: ClassVar[ErrorClass] = "NetworkError"
    _default_recovery_action: ClassVar[RecoveryAction] = "retry_automatic"


class TimeoutError(CopernicusMcpError):  # noqa: A001 — deliberate shadow of builtin
    _canonical_class_name: ClassVar[ErrorClass] = "TimeoutError"
    _default_recovery_action: ClassVar[RecoveryAction] = "retry_automatic"


class BackendError(CopernicusMcpError):
    _canonical_class_name: ClassVar[ErrorClass] = "BackendError"
    _default_recovery_action: ClassVar[RecoveryAction] = "report_to_administrator"


class OperationCancelledError(CopernicusMcpError):
    """App-level "operation was cancelled".

    ``error_record.error_class == "CancelledError"`` (wire name preserved).
    Distinct from ``asyncio.CancelledError`` (BaseException, control-flow).
    """

    _canonical_class_name: ClassVar[ErrorClass] = "CancelledError"
    _default_recovery_action: ClassVar[RecoveryAction] = "no_action"


class CacheError(CopernicusMcpError):
    _canonical_class_name: ClassVar[ErrorClass] = "CacheError"
    _default_recovery_action: ClassVar[RecoveryAction] = "report_to_administrator"
