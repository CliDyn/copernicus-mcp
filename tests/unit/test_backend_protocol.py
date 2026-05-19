from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio


def _build_foundation(tmp_path: Path):
    from copernicus_mcp.auth import CredentialResolver
    from copernicus_mcp.backends.abstract import FoundationServices
    from copernicus_mcp.cache import CacheManager
    from copernicus_mcp.config import ConfigLoader
    from copernicus_mcp.data_model.coordinator import DataModelCoordinator
    from copernicus_mcp.data_model.provenance import ProvenanceRecorder
    from copernicus_mcp.errors.sanitiser import Sanitiser
    from copernicus_mcp.http import HttpClientFactory
    from copernicus_mcp.persistence import SqliteBackend

    config = ConfigLoader().load()
    persistence = SqliteBackend(tmp_path / "state.db")
    http_factory = HttpClientFactory(http_config=config.http)
    cache = CacheManager(
        cache_directory=tmp_path / "cache",
        persistence=persistence,
        size_limit_bytes=10 * 1024 * 1024,
    )
    return (
        FoundationServices(
            config=config,
            credential_resolver=CredentialResolver(),
            http_client_factory=http_factory,
            persistence=persistence,
            cache=cache,
            sanitiser=Sanitiser(),
            data_model=DataModelCoordinator(persistence=persistence),
            provenance=ProvenanceRecorder(
                persistence=persistence,
                software_versions={"copernicus-mcp": "0.0.1"},
            ),
        ),
        persistence,
    )


@pytest_asyncio.fixture
async def foundation(tmp_path: Path):
    found, persistence = _build_foundation(tmp_path)
    await persistence.initialise()
    try:
        yield found
    finally:
        await persistence.close()


def _make_concrete():
    from copernicus_mcp.backends.abstract import AbstractBackend

    class _NoopBackend(AbstractBackend):
        backend_id = "noop"

        async def search(self, params: dict) -> dict:
            return {}

        async def describe(self, identifier: str) -> dict:
            return {"id": identifier}

        async def validate(self, params: dict) -> dict:
            return {}

        async def estimate(self, params: dict) -> dict:
            return {}

        async def submit(self, params: dict) -> dict:
            return {}

        async def check_status(self, request_id: str) -> dict:
            return {}

        async def fetch_result(self, request_id: str, target):
            return {}

        async def cancel(self, request_id: str) -> dict:
            return {}

        @property
        def supports_async(self) -> bool:
            return False

        @property
        def supports_dry_run(self) -> bool:
            return True

        @property
        def requires_terms_acceptance(self) -> bool:
            return False

    return _NoopBackend


def test_abstract_subclass_missing_methods_cannot_instantiate(foundation) -> None:
    from copernicus_mcp.backends.abstract import AbstractBackend

    class _Partial(AbstractBackend):
        backend_id = "partial"

        async def search(self, params: dict) -> dict:
            return {}

    with pytest.raises(TypeError):
        _Partial(foundation=foundation)  # type: ignore[abstract]


@pytest.mark.asyncio
async def test_concrete_subclass_instantiates_and_closes(foundation) -> None:
    Backend = _make_concrete()
    inst = Backend(foundation=foundation)
    assert inst.backend_id == "noop"
    assert inst.foundation is foundation
    assert (await inst.search({})) == {}
    await inst.close()  # base no-op


def test_concrete_subclass_without_backend_id_rejected() -> None:
    from copernicus_mcp.backends.abstract import AbstractBackend

    with pytest.raises(TypeError, match="backend_id"):

        class _NoId(AbstractBackend):  # missing backend_id assignment
            async def search(self, params: dict) -> dict:
                return {}

            async def describe(self, identifier: str) -> dict:
                return {}

            async def validate(self, params: dict) -> dict:
                return {}

            async def estimate(self, params: dict) -> dict:
                return {}

            async def submit(self, params: dict) -> dict:
                return {}

            async def check_status(self, request_id: str) -> dict:
                return {}

            async def fetch_result(self, request_id: str, target):
                return {}

            async def cancel(self, request_id: str) -> dict:
                return {}

            @property
            def supports_async(self) -> bool:
                return False

            @property
            def supports_dry_run(self) -> bool:
                return False

            @property
            def requires_terms_acceptance(self) -> bool:
                return False


def test_protocol_runtime_check_optional() -> None:
    """``BackendProtocol`` is a structural type — no runtime registration required."""
    from copernicus_mcp.backends.protocol import BackendProtocol

    Backend = _make_concrete()
    # Just ensure the symbol is importable and not a runtime_checkable abuse.
    assert BackendProtocol is not None
    # Concrete class has the right attributes:
    expected = {
        "search",
        "describe",
        "validate",
        "estimate",
        "submit",
        "check_status",
        "fetch_result",
        "cancel",
        "close",
        "supports_async",
        "supports_dry_run",
        "requires_terms_acceptance",
        "backend_id",
    }
    assert expected.issubset(set(dir(Backend)))
