"""CMEMS backend package — registers its factory at import time.

T-015 left ``bootstrap._BACKEND_FACTORIES`` empty; importing this package
plugs the CMEMS factory in so ``build_backend_registry`` can construct a
``CmemsBackend`` instance.
"""

from copernicus_mcp.auth.resolver import ResolvedCredentials
from copernicus_mcp.backends.abstract import FoundationServices
from copernicus_mcp.backends.cmems.backend import CmemsBackend
from copernicus_mcp.backends.protocol import BackendProtocol
from copernicus_mcp.bootstrap import register_backend_factory


def _factory(
    foundation: FoundationServices, creds: ResolvedCredentials | None
) -> BackendProtocol:
    return CmemsBackend(foundation=foundation, credentials=creds)


register_backend_factory("cmems", _factory)

__all__ = ["CmemsBackend"]
