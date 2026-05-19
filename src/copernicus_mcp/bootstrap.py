"""Single-place wiring of foundation services and backend registry.

T-015. Iter 1 only knows the ``cmems`` backend; T-024 plugs ``CmemsBackend``
into ``_BACKEND_FACTORIES`` so this file does not need to change again.

Graceful-degradation rule: even when credentials
for a configured backend are missing, the backend is still **registered** —
operations that need credentials raise a clean ``AuthError`` with
``recovery_action="configure_credentials"``. This lets ``copernicus_mcp_status``
(T-029) report the backend as "not configured" without hiding it from the user.
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import Awaitable, Callable

from copernicus_mcp.auth import CredentialResolver
from copernicus_mcp.auth.resolver import ResolvedCredentials
from copernicus_mcp.backends.abstract import FoundationServices
from copernicus_mcp.backends.protocol import BackendProtocol
from copernicus_mcp.backends.registry import BackendRegistry
from copernicus_mcp.cache import CacheManager
from copernicus_mcp.config import CopernicusMcpConfig
from copernicus_mcp.data_model.coordinator import DataModelCoordinator
from copernicus_mcp.data_model.provenance import ProvenanceRecorder
from copernicus_mcp.errors.sanitiser import Sanitiser
from copernicus_mcp.http import HttpClientFactory
from copernicus_mcp.observability.logger import get_logger
from copernicus_mcp.persistence import SqliteBackend
from copernicus_mcp.version import __version__

logger = get_logger(__name__)

# Each entry takes ``(foundation, credentials_or_None)`` and returns a backend.
# T-024 adds the CMEMS factory; subsequent iterations append more entries.
BackendFactory = Callable[
    [FoundationServices, ResolvedCredentials | None],
    Awaitable[BackendProtocol] | BackendProtocol,
]
_BACKEND_FACTORIES: dict[str, BackendFactory] = {}


async def build_foundation(config: CopernicusMcpConfig) -> FoundationServices:
    """Construct and initialise the singleton foundation services."""
    persistence = SqliteBackend(config.storage.state_database)
    await persistence.initialise()
    # try/finally + success flag avoids ``except BaseException`` (forbidden by
    # the AST test in test_errors.py) while still closing the SQLite handle on
    # both ordinary exceptions and ``CancelledError``.
    success = False
    try:
        cache_size_bytes = int(config.storage.cache_size_limit_gb * 1024**3)
        cache = CacheManager(
            cache_directory=config.storage.cache_directory,
            persistence=persistence,
            size_limit_bytes=cache_size_bytes,
        )
        foundation = FoundationServices(
            config=config,
            credential_resolver=CredentialResolver(),
            http_client_factory=HttpClientFactory(http_config=config.http),
            persistence=persistence,
            cache=cache,
            sanitiser=Sanitiser(),
            data_model=DataModelCoordinator(persistence=persistence),
            provenance=ProvenanceRecorder(
                persistence=persistence,
                software_versions={"copernicus-mcp": __version__},
            ),
        )
        success = True
        return foundation
    finally:
        if not success:
            await persistence.close()


_BACKEND_PACKAGES: dict[str, str] = {
    "cmems": "copernicus_mcp.backends.cmems",
    "cds": "copernicus_mcp.backends.cds",
}


async def build_backend_registry(
    foundation: FoundationServices,
) -> BackendRegistry:
    """Register every enabled backend, even if its credentials are missing."""
    registry = BackendRegistry()
    for backend_id in foundation.config.enabled_backends:
        # codex T-015 MEDIUM: ensure the backend package's side-effect
        # registration has run before we look up its factory. Importing
        # ``copernicus_mcp.bootstrap`` alone is not enough — the CMEMS
        # factory only plugs itself in when its package is imported.
        package = _BACKEND_PACKAGES.get(backend_id)
        if package is not None:
            import importlib

            with contextlib.suppress(ImportError):
                importlib.import_module(package)
        factory = _BACKEND_FACTORIES.get(backend_id)
        if factory is None:
            # Transitional: between T-015 (this file) and T-024 (CMEMS factory
            # plug-in) no factories exist. Logged at ERROR — once T-024 lands
            # this branch indicates a real config bug..
            logger.error(
                "no factory registered for enabled backend",
                extra={"backend_id": backend_id},
            )
            continue

        creds = foundation.credential_resolver.resolve(backend_id)
        if creds is None:
            logger.warning(
                "backend credentials missing; backend will be exposed but "
                "operations will return AuthError",
                extra={"backend_id": backend_id},
            )

        result = factory(foundation, creds)
        backend = await result if inspect.isawaitable(result) else result
        # codex T-015 LOW: a miswired factory returning a backend with the
        # wrong id would silently leave the requested backend unconfigured.
        if backend.backend_id != backend_id:
            logger.error(
                "factory returned mismatched backend_id; skipping registration",
                extra={
                    "expected": backend_id,
                    "got": backend.backend_id,
                },
            )
            continue
        registry.register(backend)

    return registry


def register_backend_factory(backend_id: str, factory: BackendFactory) -> None:
    """T-024 and later iterations call this at import time to plug factories in."""
    _BACKEND_FACTORIES[backend_id] = factory
