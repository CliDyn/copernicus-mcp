"""Structural contract every backend implements.

A backend is anything that satisfies ``BackendProtocol``: discovery
(``search``/``describe``), pre-flight (``validate``/``estimate``), submission
and lifecycle (``submit``/``check_status``/``fetch_result``/``cancel``), and a
clean ``close``. Capability flags expose async semantics, dry-run support,
and terms-acceptance requirements.

Iter 1 has a single backend (CMEMS); subsequent iterations subclass
``AbstractBackend`` and register through ``backends/registry.py`` (T-015).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Protocol


class BackendProtocol(Protocol):
    backend_id: ClassVar[str]

    async def search(self, params: dict[str, Any]) -> dict[str, Any]: ...

    async def describe(self, identifier: str) -> dict[str, Any]: ...

    async def validate(self, params: dict[str, Any]) -> dict[str, Any]: ...

    async def estimate(self, params: dict[str, Any]) -> dict[str, Any]: ...

    async def submit(self, params: dict[str, Any]) -> dict[str, Any]: ...

    async def check_status(self, request_id: str) -> dict[str, Any]: ...

    async def fetch_result(
        self, request_id: str, target: Path
    ) -> dict[str, Any]: ...

    async def cancel(self, request_id: str) -> dict[str, Any]: ...

    @property
    def supports_async(self) -> bool: ...

    @property
    def supports_dry_run(self) -> bool: ...

    @property
    def requires_terms_acceptance(self) -> bool: ...

    async def close(self) -> None: ...
