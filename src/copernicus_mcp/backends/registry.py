"""Process-wide backend registry.

The registry is a thin in-memory map ``backend_id -> BackendProtocol``.
Construction happens once at startup in ``bootstrap.build_backend_registry``;
the orchestrator and CLI consume registered instances by id.

``get`` raises a canonical ``BackendError`` (``error_subclass="backend_not_configured"``)
rather than ``KeyError`` so the orchestrator can surface a structured response
without translating exception types at every call site.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from copernicus_mcp.backends.protocol import BackendProtocol
from copernicus_mcp.errors import BackendError
from copernicus_mcp.errors.records import build_error_record

# Backend ids surface as dict keys in ``status()`` output and as cache-key
# fragments. Rejecting credential-shaped ids at registration time prevents
# the dict-key-leak class entirely (a key cannot be safely rewritten without
# collision risk — see Sanitiser._walk for the rationale).
_BACKEND_ID_RE: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")


class BackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, BackendProtocol] = {}

    def register(self, backend: BackendProtocol) -> None:
        """Register a backend by its ``backend_id``. Re-registration replaces.

        Validates the id against ``_BACKEND_ID_RE`` so credential-shaped or
        otherwise unsafe ids cannot leak through downstream string outputs.
        """
        bid = backend.backend_id
        if not isinstance(bid, str) or not _BACKEND_ID_RE.match(bid):
            raise BackendError(
                f"invalid backend_id {bid!r}",
                record=build_error_record(
                    "BackendError",
                    message=(
                        f"invalid backend_id {bid!r} — must match "
                        f"{_BACKEND_ID_RE.pattern}"
                    ),
                    error_subclass="invalid_backend_id",
                    recovery_action="report_to_administrator",
                ),
            )
        self._backends[bid] = backend

    def get(self, backend_id: str) -> BackendProtocol:
        try:
            return self._backends[backend_id]
        except KeyError:
            raise BackendError(
                f"backend {backend_id!r} is not configured",
                record=build_error_record(
                    "BackendError",
                    message=f"backend {backend_id!r} is not configured",
                    error_subclass="backend_not_configured",
                    recovery_action="configure_credentials",
                ),
            ) from None

    def is_configured(self, backend_id: str) -> bool:
        return backend_id in self._backends

    def iter_backends(self) -> Iterator[BackendProtocol]:
        return iter(self._backends.values())
