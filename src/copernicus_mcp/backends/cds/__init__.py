"""CDS backend package — registers its factory at import time."""

from copernicus_mcp.auth.resolver import ResolvedCredentials
from copernicus_mcp.backends.abstract import FoundationServices
from copernicus_mcp.backends.cds.backend import CdsBackend
from copernicus_mcp.backends.protocol import BackendProtocol
from copernicus_mcp.bootstrap import register_backend_factory


def _factory(
    foundation: FoundationServices, creds: ResolvedCredentials | None
) -> BackendProtocol:
    # T-CDS-001 completion: with ``CdsApiKeyAdapter.apply_credentials`` and
    # the ``CredentialResolver`` extension reviewed, the factory now plumbs
    # resolver-supplied creds through to the backend.
    return CdsBackend(foundation=foundation, credentials=creds)


register_backend_factory("cds", _factory)

__all__ = ["CdsBackend"]
