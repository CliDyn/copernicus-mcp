"""Workflow orchestrator (T-028).

A thin layer between the MCP server and the backends. Responsibilities:

1. Bind a per-call ``trace_id`` so structured logs from the dispatched
   backend method correlate.
2. Resolve the backend by id via the registry.
3. Dispatch the named operation to the matching backend method.
4. Catch ``ConfirmationRequired`` and return its payload directly (no
   ``error`` key).
5. Catch any ``CopernicusMcpError`` and return ``{"error": <record>}``.
6. Catch unexpected exceptions, sanitise the message, wrap in
   ``BackendError``, log, and return as above.
7. Pass any result through ``Sanitiser`` once more (defence-in-depth).

``asyncio.CancelledError`` is NEVER caught by the orchestrator — it
propagates to the asyncio runtime so the higher layer can decide policy
(the project conventions invariant 3).
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from copernicus_mcp.backends.abstract import FoundationServices
from copernicus_mcp.backends.registry import BackendRegistry
from copernicus_mcp.errors import BackendError, CopernicusMcpError, ValidationError
from copernicus_mcp.errors.records import build_error_record
from copernicus_mcp.observability.logger import bind_trace_id, get_logger
from copernicus_mcp.version import __version__
from copernicus_mcp.workflow.confirmation import ConfirmationRequired

logger = get_logger(__name__)


# Map ``OperationType`` from the envelope to the backend method name.
# ``poll`` is named ``check_status`` on the backend; ``fetch`` is
# ``fetch_result``. The other six operations match 1-to-1.
_OPERATION_METHOD: dict[str, str] = {
    "search": "search",
    "search_groups": "search_groups",
    "search_products": "search_products",
    "describe": "describe",
    "validate": "validate",
    "estimate": "estimate",
    "submit": "submit",
    "get": "get_files",  # T-CMEMS-GET-006: native-file retrieval.
    "list_files": "list_files",  # T-CMEMS-GET-INDEX-004: Layer 2 index listing.
    "get_coordinates": "get_coordinates",  # T-022 second half: coordinate axes.
    "apply_constraints": "apply_constraints",  # T-CDS-016: CDS narrowing.
    "poll": "check_status",
    "fetch": "fetch_result",
    "cancel": "cancel",
}


class WorkflowOrchestrator:
    def __init__(
        self,
        registry: BackendRegistry,
        foundation: FoundationServices,
    ) -> None:
        self._registry = registry
        self._foundation = foundation

    async def run(
        self,
        *,
        backend: str,
        operation: str,
        params: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        trace_id = uuid.uuid4().hex
        # Merge options under the ``__options`` magic key so CmemsBackend.submit
        # can consume them (force_refresh, confirmed, etc.). T-031 will plumb
        # this through a properly typed RequestOptions envelope.
        merged_params = dict(params)
        if options:
            merged_params["__options"] = dict(options)
        with bind_trace_id(trace_id):
            return await self._run_inner(backend, operation, merged_params, options)

    async def _run_inner(
        self,
        backend_id: str,
        operation: str,
        params: dict[str, Any],
        options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        # 1. Resolve backend.
        try:
            backend = self._registry.get(backend_id)
        except CopernicusMcpError as exc:
            return self._error_envelope(exc)

        # 2. Resolve method name.
        method_name = _OPERATION_METHOD.get(operation)
        if method_name is None:
            return self._error_envelope(
                BackendError(
                    f"unsupported operation {operation!r}",
                    record=build_error_record(
                        "BackendError",
                        message=f"unsupported operation {operation!r}",
                        error_subclass="unsupported_operation",
                        recovery_action="modify_request_parameters",
                    ),
                )
            )

        method = getattr(backend, method_name, None)
        if method is None:
            return self._error_envelope(
                BackendError(
                    f"backend {backend_id!r} does not implement {method_name!r}",
                    record=build_error_record(
                        "BackendError",
                        message=(f"backend {backend_id!r} does not implement {method_name!r}"),
                        error_subclass="not_implemented",
                        recovery_action="report_to_administrator",
                    ),
                )
            )

        # 3. Dispatch.
        try:
            result = await self._dispatch(method, operation, params)
        except ConfirmationRequired as exc:
            # Confirmation flow: return payload directly, no ``error`` key.
            return self._foundation.sanitiser.sanitise(  # type: ignore[no-any-return]
                exc.payload
            )
        except CopernicusMcpError as exc:
            return self._error_envelope(exc)
        except ExceptionGroup as eg:
            # codex-batch-3 HIGH: ExceptionGroup containing exactly one
            # canonical leaf is unwrapped so the class+recovery_action
            # survives. Nested groups recurse (asyncio.TaskGroup/anyio).
            # Mixed groups (canonical + non-canonical) still unwrap the
            # single canonical leaf — user-actionable info wins over the
            # wrapper. The non-canonical extras are logged for diagnostics
            # but not folded into the wire response.
            matched, rest = eg.split(CopernicusMcpError)
            leaves = _flatten_canonical_leaves(matched)
            if len(leaves) == 1:
                if rest is not None:
                    logger.warning(
                        "ExceptionGroup contained additional non-canonical "
                        "exceptions; only the canonical leaf is reported",
                        extra={"backend_id": backend_id, "operation": operation},
                    )
                return self._error_envelope(leaves[0])
            # Otherwise: treat as unexpected.
            safe = self._foundation.sanitiser.sanitise(str(eg))
            return self._error_envelope(
                BackendError(
                    f"unexpected exception group during {operation}: {safe}",
                    record=build_error_record(
                        "BackendError",
                        message=(f"unexpected exception group during {operation}: {safe}"),
                        error_subclass="exception_group",
                        recovery_action="report_to_administrator",
                        context={"backend_id": backend_id, "operation": operation},
                    ),
                )
            )
        except Exception as exc:
            # Unexpected — sanitise the message, wrap into BackendError, log.
            safe = self._foundation.sanitiser.sanitise(str(exc))
            wrapped = BackendError(
                f"unexpected backend error during {operation}: {safe}",
                record=build_error_record(
                    "BackendError",
                    message=(f"unexpected backend error during {operation}: {safe}"),
                    error_subclass="unexpected",
                    recovery_action="report_to_administrator",
                    context={"backend_id": backend_id, "operation": operation},
                ),
            )
            logger.error(
                "unexpected backend error",
                extra={
                    "backend_id": backend_id,
                    "operation": operation,
                    "error_class": type(exc).__name__,
                },
                exc_info=True,
            )
            return self._error_envelope(wrapped)

        # 4. Defence-in-depth: sanitise the result before returning.
        return {"result": self._foundation.sanitiser.sanitise(result)}

    async def _dispatch(
        self,
        method: Any,
        operation: str,
        params: dict[str, Any],
    ) -> Any:
        """Dispatch with the per-operation argument shape.

        Validates the required keys are present and non-empty before
        dispatching — silently coercing to ``""``/``Path('.')`` would have
        the backend raise an opaque error with the wrong recovery hint.
        """
        # Warn loudly if the caller passed options for an op that takes
        # positional args — those would otherwise silently disappear.
        if operation in ("describe", "poll", "cancel", "fetch"):
            if params.get("__options"):
                logger.warning(
                    "options ignored for positional-arg operation",
                    extra={"operation": operation},
                )
        if operation == "describe":
            identifier = params.get("identifier")
            if identifier is None or identifier == "":
                raise _missing_field("identifier", operation)
            return await method(identifier)
        if operation in ("poll", "cancel"):
            request_id = params.get("request_id")
            if request_id is None or request_id == "":
                raise _missing_field("request_id", operation)
            return await method(request_id)
        if operation == "fetch":
            request_id = params.get("request_id")
            target = params.get("target")
            if request_id is None or request_id == "":
                raise _missing_field("request_id", operation)
            if target is None or target == "":
                raise _missing_field("target", operation)
            return await method(request_id, Path(target))
        return await method(params)

    def _error_envelope(self, exc: CopernicusMcpError) -> dict[str, Any]:
        """Render a canonical error as ``{"error": <record dict>}``."""
        record = exc.error_record.model_dump(mode="json")
        return {"error": self._foundation.sanitiser.sanitise(record)}

    async def status(self) -> dict[str, Any]:
        """Cross-cutting "is everything plugged in" probe (T-029).

        Wrapped in the same trace_id + error-envelope discipline as ``run()``
        per the project conventions §3 — diagnostic tools must not crash mid-call.
        """
        trace_id = uuid.uuid4().hex
        with bind_trace_id(trace_id):
            try:
                return await self._status_inner()
            except CopernicusMcpError as exc:
                return self._error_envelope(exc)
            except Exception as exc:
                safe = self._foundation.sanitiser.sanitise(str(exc))
                return self._error_envelope(
                    BackendError(
                        f"unexpected error in status(): {safe}",
                        record=build_error_record(
                            "BackendError",
                            message=f"unexpected error in status(): {safe}",
                            error_subclass="status_failure",
                            recovery_action="report_to_administrator",
                        ),
                    )
                )

    async def _status_inner(self) -> dict[str, Any]:
        cfg = self._foundation.config
        backends_block: dict[str, dict[str, Any]] = {}
        # List every enabled backend whether or not it's registered, so the
        # caller sees "cmems is enabled in config but the factory failed
        # to construct it" as a clear signal.
        seen_ids: set[str] = set(cfg.enabled_backends)
        for backend in self._registry.iter_backends():
            seen_ids.add(backend.backend_id)
        for bid in sorted(seen_ids):
            registered = self._registry.is_configured(bid)
            enabled = bid in cfg.enabled_backends
            # Resolver may hit disk / secret-manager — push to threadpool.
            creds = await asyncio.to_thread(self._foundation.credential_resolver.resolve, bid)
            backends_block[bid] = {
                "registered": registered,
                "enabled_in_config": enabled,
                "configured": creds is not None,
                "credential_source": creds.source if creds is not None else "missing",
            }

        cache_dir = cfg.storage.cache_directory
        # Recursive filesystem walk → threadpool so a large cache doesn't
        # block the event loop.
        cache_size, cache_count = await asyncio.to_thread(_cache_metrics, cache_dir)

        # Sanitise the structured backend block AND the storage paths.
        # the credential-isolation invariant wins over diagnostic value: a user who
        # configures cache_directory=/tmp/password=hunter2/ must not see
        # "hunter2" leak through status output. The diagnostic cost
        # (a path containing literal ``token=...`` is rewritten) is accepted.
        sanitised_backends = self._foundation.sanitiser.sanitise(backends_block)
        sanitised_cache_dir = self._foundation.sanitiser.sanitise(str(cache_dir))
        sanitised_db_path = self._foundation.sanitiser.sanitise(str(cfg.storage.state_database))
        sanitised_config = self._foundation.sanitiser.sanitise(
            {
                "log_level": cfg.server.log_level,
                "transport": cfg.server.transport,
                "cache": cfg.cache.model_dump(),
                "budget": cfg.budget.model_dump(),
                "enabled_backends": list(cfg.enabled_backends),
            }
        )
        return {
            "version": __version__,
            "backends": sanitised_backends,
            "cache": {
                "directory": sanitised_cache_dir,
                "size_bytes": cache_size,
                "entry_count": cache_count,
                # Welcome-message companion: agents that didn't read
                # the once-per-session ``instructions`` block still see
                # the override mechanisms here on every status call.
                "override_hint": (
                    "Override this path via COPERNICUS_MCP_CACHE_DIR env "
                    "var, --cache-dir CLI flag, or storage.cache_directory "
                    "in config.yaml. Default is the per-OS user cache "
                    "(platformdirs)."
                ),
            },
            "persistence": {
                "database_path": sanitised_db_path,
            },
            "config": sanitised_config,
        }


def _cache_metrics(cache_dir: Path) -> tuple[int, int]:
    """Best-effort recursive size + entry count under ``cache_dir``.

    All ``OSError`` paths are swallowed: a broken symlink, restricted
    permission, or symlink loop produces (or continues with) a partial
    metric rather than crashing the diagnostic tool. ``status()`` calls
    this via ``asyncio.to_thread`` so a large cache does not block the
    event loop.
    """
    try:
        if not cache_dir.exists():
            return (0, 0)
    except OSError:
        return (0, 0)
    total = 0
    count = 0
    try:
        iterator = cache_dir.rglob("*")
    except OSError:
        return (0, 0)
    for entry in iterator:
        try:
            if entry.is_file():
                count += 1
                total += entry.stat().st_size
        except OSError:
            continue
    return (total, count)


def _flatten_canonical_leaves(
    eg: ExceptionGroup[CopernicusMcpError] | None,
) -> list[CopernicusMcpError]:
    """Recursively collect ``CopernicusMcpError`` leaves from a nested group."""
    if eg is None:
        return []
    out: list[CopernicusMcpError] = []
    for exc in eg.exceptions:
        if isinstance(exc, ExceptionGroup):
            out.extend(_flatten_canonical_leaves(exc))  # type: ignore[arg-type]
        elif isinstance(exc, CopernicusMcpError):
            out.append(exc)
    return out


def _missing_field(field: str, operation: str) -> ValidationError:
    msg = f"missing required field {field!r} for operation {operation!r}"
    return ValidationError(
        msg,
        record=build_error_record(
            "ValidationError",
            message=msg,
            recovery_action="modify_request_parameters",
            context={"missing": [field], "operation": operation},
        ),
    )
