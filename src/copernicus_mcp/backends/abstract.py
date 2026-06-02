"""Abstract backend base class wired to the foundation services.

To add a new backend:

1. Subclass ``AbstractBackend`` in ``backends/<backend>/backend.py``.
2. Implement every abstract method (``search``, ``describe``, ``validate``,
   ``estimate``, ``submit``, ``check_status``, ``fetch_result``, ``cancel``)
   plus the three capability properties.
3. Register an instance in ``backends/registry.py`` via the bootstrapper.
4. Expose tools in ``backends/<backend>/tools.py`` following
   ``marine_<verb>_<noun>`` (or analogous) naming.

Backend-specific specs live in:

- ``the project research notes`` — CMEMS (Iter 1).
- ``the project research notes`` — CDS / ADS / EWDS family (deferred).
- ``the project research notes`` — CDSE OData/STAC + Sentinel Hub (deferred).
- ``the project research notes`` — WEkEO Harmonised Data Access (deferred).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from copernicus_mcp.auth import CredentialResolver
from copernicus_mcp.cache import CacheManager
from copernicus_mcp.config import CopernicusMcpConfig
from copernicus_mcp.data_model.coordinator import DataModelCoordinator
from copernicus_mcp.data_model.provenance import ProvenanceRecorder
from copernicus_mcp.errors.sanitiser import Sanitiser
from copernicus_mcp.http import HttpClientFactory
from copernicus_mcp.persistence import PersistenceBackend


@dataclass(frozen=True, slots=True)
class FoundationServices:
    """Singleton bundle of process-wide services injected into every backend.

    ``frozen=True`` prevents rebinding fields; the referenced services manage
    their own internal mutable state (DB connections, cache index, etc.).
    """

    config: CopernicusMcpConfig
    credential_resolver: CredentialResolver
    http_client_factory: HttpClientFactory
    persistence: PersistenceBackend
    cache: CacheManager
    sanitiser: Sanitiser
    data_model: DataModelCoordinator
    provenance: ProvenanceRecorder


class AbstractBackend(ABC):
    """Base class concrete backends extend.

    The base ``__init__`` only stores ``foundation``; subclasses can override
    to construct their auth adapter / HTTP client. ``close()`` is a no-op
    here — **subclasses owning HTTP clients or auth adapters MUST override**.
    """

    backend_id: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Concrete subclasses must declare a non-empty ``backend_id``.
        # Abstract subclasses (still carrying unimplemented methods) are
        # permitted to defer the assignment.
        if not getattr(cls, "__abstractmethods__", None) and not cls.backend_id:
            raise TypeError(
                f"{cls.__name__} must define a non-empty class attribute "
                "`backend_id` (e.g. `backend_id = \"cmems\"`)."
            )

    def __init__(self, foundation: FoundationServices) -> None:
        self.foundation = foundation

    # --- discovery -------------------------------------------------------

    @abstractmethod
    async def search(self, params: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    async def describe(self, identifier: str) -> dict[str, Any]: ...

    # --- pre-flight ------------------------------------------------------

    @abstractmethod
    async def validate(self, params: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    async def estimate(self, params: dict[str, Any]) -> dict[str, Any]: ...

    # --- lifecycle -------------------------------------------------------

    @abstractmethod
    async def submit(self, params: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    async def check_status(self, request_id: str) -> dict[str, Any]: ...

    @abstractmethod
    async def fetch_result(
        self, request_id: str, target: Path
    ) -> dict[str, Any]: ...

    @abstractmethod
    async def cancel(self, request_id: str) -> dict[str, Any]: ...

    # --- capabilities ----------------------------------------------------

    @property
    @abstractmethod
    def supports_async(self) -> bool: ...

    @property
    @abstractmethod
    def supports_dry_run(self) -> bool: ...

    @property
    @abstractmethod
    def requires_terms_acceptance(self) -> bool: ...

    # --- shutdown --------------------------------------------------------

    async def close(self) -> None:  # noqa: B027 — deliberate no-op default
        """No-op default. Subclasses override to release HTTP clients, etc."""
