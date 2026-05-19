from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def _make_stub_backend(bid: str = "stub"):
    from copernicus_mcp.backends.abstract import AbstractBackend, FoundationServices

    class _Stub(AbstractBackend):
        backend_id = bid

        def __init__(self, foundation: FoundationServices) -> None:
            super().__init__(foundation=foundation)

        async def search(self, params: dict[str, Any]) -> dict[str, Any]:
            return {}

        async def describe(self, identifier: str) -> dict[str, Any]:
            return {}

        async def validate(self, params: dict[str, Any]) -> dict[str, Any]:
            return {}

        async def estimate(self, params: dict[str, Any]) -> dict[str, Any]:
            return {}

        async def submit(self, params: dict[str, Any]) -> dict[str, Any]:
            return {}

        async def check_status(self, request_id: str) -> dict[str, Any]:
            return {}

        async def fetch_result(self, request_id: str, target: Path) -> dict[str, Any]:
            return {}

        async def cancel(self, request_id: str) -> dict[str, Any]:
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

    return _Stub


def test_register_and_get(tmp_path: Path) -> None:
    from copernicus_mcp.backends.registry import BackendRegistry

    reg = BackendRegistry()
    Stub = _make_stub_backend("stub")
    foundation = _make_foundation_for_test(tmp_path)
    inst = Stub(foundation=foundation)
    reg.register(inst)

    assert reg.get("stub") is inst
    assert reg.is_configured("stub")
    assert list(reg.iter_backends()) == [inst]


def test_get_unknown_raises_backend_error() -> None:
    from copernicus_mcp.backends.registry import BackendRegistry
    from copernicus_mcp.errors import BackendError

    reg = BackendRegistry()
    with pytest.raises(BackendError) as exc_info:
        reg.get("does-not-exist")
    assert exc_info.value.error_record.error_subclass == "backend_not_configured"


def test_register_duplicate_replaces(tmp_path: Path) -> None:
    from copernicus_mcp.backends.registry import BackendRegistry

    reg = BackendRegistry()
    Stub = _make_stub_backend("stub")
    foundation = _make_foundation_for_test(tmp_path)
    a = Stub(foundation=foundation)
    b = Stub(foundation=foundation)
    reg.register(a)
    reg.register(b)
    # Second registration replaces the first.
    assert reg.get("stub") is b


def _make_foundation_for_test(tmp_path: Path):
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
    cache = CacheManager(
        cache_directory=tmp_path / "cache",
        persistence=persistence,
        size_limit_bytes=10 * 1024 * 1024,
    )
    return FoundationServices(
        config=config,
        credential_resolver=CredentialResolver(),
        http_client_factory=HttpClientFactory(http_config=config.http),
        persistence=persistence,
        cache=cache,
        sanitiser=Sanitiser(),
        data_model=DataModelCoordinator(persistence=persistence),
        provenance=ProvenanceRecorder(
            persistence=persistence,
            software_versions={"copernicus-mcp": "0.0.1"},
        ),
    )


@pytest.mark.asyncio
async def test_build_foundation_initialises_persistence(tmp_path: Path) -> None:
    from copernicus_mcp.bootstrap import build_foundation
    from copernicus_mcp.config import ConfigLoader

    config = ConfigLoader().load(
        cli_overrides={
            "storage": {
                "state_database": str(tmp_path / "state.db"),
                "cache_directory": str(tmp_path / "cache"),
            }
        }
    )
    foundation = await build_foundation(config)
    try:
        # Persistence must be ready for round-trips.
        assert foundation.persistence is not None
        assert (tmp_path / "state.db").exists()
    finally:
        await foundation.persistence.close()


@pytest.fixture
def isolated_factories(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Snapshot/restore the module-level _BACKEND_FACTORIES so tests don't leak."""
    from copernicus_mcp import bootstrap

    fresh: dict = {}
    monkeypatch.setattr(bootstrap, "_BACKEND_FACTORIES", fresh)
    return fresh


def _stub_factory(bid: str = "stub"):
    Stub = _make_stub_backend(bid)

    def factory(foundation, creds):
        return Stub(foundation=foundation)

    return factory


async def _build(tmp_path: Path):
    from copernicus_mcp.bootstrap import build_foundation
    from copernicus_mcp.config import ConfigLoader

    config = ConfigLoader().load(
        cli_overrides={
            "storage": {
                "state_database": str(tmp_path / "state.db"),
                "cache_directory": str(tmp_path / "cache"),
            }
        }
    )
    return await build_foundation(config)


@pytest.mark.asyncio
async def test_build_backend_registry_registers_with_factory(
    tmp_path: Path, isolated_factories: dict
) -> None:
    from copernicus_mcp.backends.registry import BackendRegistry
    from copernicus_mcp.bootstrap import build_backend_registry

    isolated_factories["cmems"] = _stub_factory("cmems")
    foundation = await _build(tmp_path)
    try:
        registry = await build_backend_registry(foundation)
        assert isinstance(registry, BackendRegistry)
        assert registry.is_configured("cmems")
    finally:
        await foundation.persistence.close()


@pytest.mark.asyncio
async def test_build_backend_registry_registers_without_credentials(
    tmp_path: Path,
    isolated_factories: dict,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even with no creds resolvable, the backend is still registered."""
    from copernicus_mcp.bootstrap import build_backend_registry

    isolated_factories["cmems"] = _stub_factory("cmems")

    # Force resolver to return None for any backend.
    from copernicus_mcp.auth import CredentialResolver

    monkeypatch.setattr(CredentialResolver, "resolve", lambda self, b, override=None: None)

    foundation = await _build(tmp_path)
    try:
        with caplog.at_level("WARNING"):
            registry = await build_backend_registry(foundation)
        assert registry.is_configured("cmems")
        assert any(
            "credentials missing" in r.getMessage() for r in caplog.records
        )
    finally:
        await foundation.persistence.close()


@pytest.mark.asyncio
async def test_build_backend_registry_logs_error_when_factory_missing(
    tmp_path: Path,
    isolated_factories: dict,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Factory absence for an enabled backend is a configuration error."""
    from copernicus_mcp.bootstrap import build_backend_registry

    foundation = await _build(tmp_path)
    try:
        with caplog.at_level("ERROR"):
            registry = await build_backend_registry(foundation)
        assert not registry.is_configured("cmems")
        assert any(
            "no factory registered" in r.getMessage() for r in caplog.records
        )
    finally:
        await foundation.persistence.close()
