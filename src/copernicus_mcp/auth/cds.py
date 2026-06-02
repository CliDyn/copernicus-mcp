"""CDS PAT auth adapter (T-CDS-001 completion).

The exit point for the Common Data Store Personal Access Token. Per
``the project research notes`` §6.8.1 the CDS API uses a single
UUID PAT injected as ``PRIVATE-TOKEN: <UUID>`` (NOT ``Authorization:
Bearer``). One PAT covers CDS + ADS + EWDS — the store is selected by
``endpoint_url``, not by per-store credentials (research §6.8.2).

the project conventions invariant #2 applies in full: the PAT lives only inside this
adapter and the ``cdsapi.Client`` constructor that consumes it. It must
never appear in tool I/O, logs, cache keys, provenance records, or error
messages.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import httpx

from copernicus_mcp.auth.resolver import ResolvedCredentials

_REDACTED_SUMMARY: Mapping[str, str] = MappingProxyType({"key": "<set>"})


class CdsApiKeyAdapter:
    """Static-PAT adapter for the Common Data Store family.

    Constructor accepts a ``ResolvedCredentials`` shaped ``{"key": "<UUID>",
    "url": "<endpoint>"}``. The url is optional — falls back to the public
    CDS endpoint when absent.

    The ``cdsapi`` toolbox handles its own HTTP layer so production code
    routes credentials via ``get_pat()`` + the ``cdsapi.Client(url=, key=)``
    constructor. ``apply_credentials`` exists for any direct httpx-driven
    paths (e.g. T-CDS-005's ``ecmwf-datastores-client`` migration) and
    sets the canonical ``PRIVATE-TOKEN`` header.
    """

    backend_id: str = "cds"
    supports_refresh: bool = False

    def __init__(self, credentials: ResolvedCredentials) -> None:
        if credentials.backend != "cds":
            raise ValueError(
                "CdsApiKeyAdapter requires credentials for backend 'cds'."
            )
        if credentials.source == "missing":
            raise ValueError("CdsApiKeyAdapter requires resolved credentials.")
        key = credentials.fields.get("key", "")
        if not key:
            raise ValueError(
                "CdsApiKeyAdapter requires a non-empty 'key' credential field."
            )
        self._key = key
        # ``url`` is optional; cdsapi defaults to the CDS endpoint when None.
        self._url = credentials.fields.get("url") or None
        self._closed = False

    async def apply_credentials(self, request: httpx.Request) -> httpx.Request:
        """Inject the PAT as ``PRIVATE-TOKEN: <UUID>`` (research §6.8.1).

        Mutates the request headers in place and returns the same object.
        Existing headers (User-Agent, Accept, etc.) are preserved.
        """
        request.headers["PRIVATE-TOKEN"] = self._key
        return request

    async def handle_unauthorized(self, response: httpx.Response) -> bool:
        # PATs are static; nothing to refresh. Always returns False.
        return False

    async def close(self) -> None:
        self._closed = True

    def credentials_summary(self) -> Mapping[str, str]:
        return _REDACTED_SUMMARY

    def get_pat(self) -> tuple[str, str | None]:
        """Return ``(key, url_or_None)`` for the in-process backend layer.

        SENSITIVE: the **only** authorised exit point for the raw PAT.
        Mirrors ``CmemsBasicAuthAdapter.get_username_password``.
        """
        return self._key, self._url

    def __repr__(self) -> str:
        return (
            f"CdsApiKeyAdapter(backend_id={self.backend_id!r}, "
            f"supports_refresh={self.supports_refresh!r})"
        )
