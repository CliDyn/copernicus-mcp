from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import httpx


class AuthAdapter(Protocol):
    """Backend-specific authorization handler.

    Concrete adapters live next to their backend (e.g. ``CmemsBasicAuthAdapter``).
    Implementations must not log credential values; ``credentials_summary``
    is the only public introspection surface and returns ``<set>`` / ``<unset>``
    placeholders only.

    See ``the project research notes`` §9.6 for the canonical protocol shape.
    """

    backend_id: str
    supports_refresh: bool

    async def apply_credentials(
        self, request: httpx.Request
    ) -> httpx.Request: ...

    async def handle_unauthorized(self, response: httpx.Response) -> bool:
        """Return ``True`` if a retry should be attempted after re-auth."""
        ...

    async def close(self) -> None: ...

    def credentials_summary(self) -> Mapping[str, str]: ...
