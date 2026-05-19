"""CMEMS basic-auth adapter.

This module is the **only** in-process exit point for raw CMEMS credential
values, via ``CmemsBasicAuthAdapter.get_username_password()``. The CMEMS
backend layer calls that method during initialization to hand the values to
the ``copernicusmarine`` toolbox; nothing else may.

the project conventions invariant #2: credentials live in ``CredentialResolver`` and
``AuthAdapter`` instances only — never tool input/output, logs, cache keys,
provenance records, or error records.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import httpx

from copernicus_mcp.auth.resolver import ResolvedCredentials

_REDACTED_SUMMARY: Mapping[str, str] = MappingProxyType(
    {"username": "<set>", "password": "<set>"}
)


class CmemsBasicAuthAdapter:
    """Static-credential adapter for CMEMS.

    The ``copernicusmarine`` toolbox performs HTTP-level auth itself, so
    ``apply_credentials`` is a no-op and our ``httpx`` client never sees
    these credentials. The backend layer pulls username/password via
    ``get_username_password`` during construction and passes them to the
    toolbox.
    """

    backend_id: str = "cmems"
    supports_refresh: bool = False

    def __init__(self, credentials: ResolvedCredentials) -> None:
        if credentials.backend != "cmems":
            # Do not echo the rejected backend value — defense in depth in
            # case it is ever something secret-shaped.
            raise ValueError(
                "CmemsBasicAuthAdapter requires credentials for backend 'cmems'."
            )
        if credentials.source == "missing":
            raise ValueError("CmemsBasicAuthAdapter requires resolved credentials.")
        username = credentials.fields.get("username", "")
        password = credentials.fields.get("password", "")
        if not username or not password:
            raise ValueError(
                "CmemsBasicAuthAdapter requires non-empty 'username' and "
                "'password' credential fields."
            )
        self._username = username
        self._password = password
        self._closed = False

    async def apply_credentials(self, request: httpx.Request) -> httpx.Request:
        # CMEMS toolbox handles HTTP-level auth itself; nothing to apply here.
        return request

    async def handle_unauthorized(self, response: httpx.Response) -> bool:
        # Static basic auth has no refreshable state.
        return False

    async def close(self) -> None:
        # Idempotent — nothing to release.
        self._closed = True

    def credentials_summary(self) -> Mapping[str, str]:
        return _REDACTED_SUMMARY

    def get_username_password(self) -> tuple[str, str]:
        """Return raw ``(username, password)`` for the in-process backend layer.

        SENSITIVE: this is the **only** authorised exit point for raw CMEMS
        credentials. Must never be exposed through MCP tool results, logs,
        cache keys, provenance, or error messages.
        """
        return self._username, self._password

    def __repr__(self) -> str:
        return (
            f"CmemsBasicAuthAdapter(backend_id={self.backend_id!r}, "
            f"supports_refresh={self.supports_refresh!r})"
        )
