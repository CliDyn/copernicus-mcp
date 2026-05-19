from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio


def _make_foundation(tmp_path: Path):
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
    return (
        FoundationServices(
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
        ),
        persistence,
    )


@pytest_asyncio.fixture
async def foundation(tmp_path: Path):
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        yield found
    finally:
        await persistence.close()


def _creds():
    from copernicus_mcp.auth.resolver import ResolvedCredentials

    return ResolvedCredentials(
        backend="cmems",
        source="explicit",
        source_detail="test",
        fields={"username": "u", "password": "p"},
    )


def test_imports_without_copernicusmarine(monkeypatch) -> None:
    """Importing CmemsBackend must succeed even when copernicusmarine is absent."""
    import sys

    # Pretend copernicusmarine is not installed.
    monkeypatch.setitem(sys.modules, "copernicusmarine", None)
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    assert CmemsBackend.backend_id == "cmems"


def test_constructs_with_credentials(foundation) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    assert backend.backend_id == "cmems"
    # T-039: CMEMS gained opt-in async submit mode.
    assert backend.supports_async is True
    assert backend.supports_dry_run is True
    assert backend.requires_terms_acceptance is False


def test_constructs_without_credentials(foundation) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=None)
    assert backend.backend_id == "cmems"


def test_check_credentials_raises_authentication_error_when_missing(
    foundation,
) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import AuthError

    backend = CmemsBackend(foundation=foundation, credentials=None)
    with pytest.raises(AuthError) as exc_info:
        backend._check_credentials_or_raise()
    record = exc_info.value.error_record
    assert record.recovery_action == "configure_credentials"
    assert record.recovery_url == "https://data.marine.copernicus.eu/register"


def test_check_credentials_returns_when_present(foundation) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    creds = _creds()
    backend = CmemsBackend(foundation=foundation, credentials=creds)
    assert backend._check_credentials_or_raise() is creds


# All eight protocol methods are now implemented (search T-021, describe T-022,
# estimate T-023, submit T-024, validate/check_status/fetch_result/cancel T-026).
# Coverage in dedicated test modules.


def test_factory_registered_for_cmems() -> None:
    """Importing the cmems package wires the factory into the bootstrap.

    Re-imports the package via importlib.reload so this test is independent
    of any earlier test that swapped ``bootstrap._BACKEND_FACTORIES`` via
    monkeypatch.
    """
    import importlib

    import copernicus_mcp.backends.cmems
    from copernicus_mcp import bootstrap

    importlib.reload(copernicus_mcp.backends.cmems)
    assert "cmems" in bootstrap._BACKEND_FACTORIES
