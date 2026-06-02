"""CDS backend skeleton tests (T-CDS-001 partial).

Pins the scaffold contract:
- CdsBackend imports without ``cdsapi`` installed (mirrors CMEMS).
- Constructs with and without credentials.
- Capabilities advertise async-only, no dry-run, T&C-required.
- All eight protocol methods raise BackendError(error_subclass="not_implemented").
- Bootstrap registers the backend when ``cds`` is in ``enabled_backends``.
- ``copernicus_mcp_status`` reports cds with ``configured=False`` (no creds
  in CredentialResolver yet).
"""

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


def test_imports_without_cdsapi(monkeypatch) -> None:
    """``CdsBackend`` must import in environments without the ``cds`` extra."""
    import sys

    monkeypatch.setitem(sys.modules, "cdsapi", None)
    from copernicus_mcp.backends.cds.backend import CdsBackend

    assert CdsBackend.backend_id == "cds"


def test_constructs_without_credentials(foundation) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend

    backend = CdsBackend(foundation=foundation, credentials=None)
    assert backend.backend_id == "cds"


def test_capabilities_match_research(foundation) -> None:
    """Per ``the project research notes`` §6.5 + §6.6:
    CDS is async-only, has no SDK-level dry-run, requires per-dataset T&C."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    backend = CdsBackend(foundation=foundation, credentials=None)
    assert backend.supports_async is True
    assert backend.supports_dry_run is False
    assert backend.requires_terms_acceptance is True


# Scaffold ``not_implemented`` parametrize removed — every protocol
# method now has a real implementation:
# - ``search`` / ``describe`` — T-CDS-003 (test_cds_search_describe.py,
#   test_cds_catalogue.py)
# - ``validate`` — T-CDS-002 (test_cds_validate.py)
# - ``estimate`` — T-CDS-004 (test_cds_estimate.py)
# - ``submit`` / ``check_status`` / ``fetch_result`` / ``cancel`` —
#   T-CDS-005 (test_cds_submit_check_cancel.py)


@pytest.mark.asyncio
async def test_bootstrap_registers_cds_backend(tmp_path: Path) -> None:
    """When ``cds`` is in ``enabled_backends``, the bootstrap factory
    plugs CdsBackend into the registry."""
    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.config import ConfigLoader

    config = ConfigLoader().load(
        cli_overrides={
            "enabled_backends": ["cmems", "cds"],
            "storage": {
                "state_database": str(tmp_path / "state.db"),
                "cache_directory": str(tmp_path / "cache"),
            },
        }
    )
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        assert registry.is_configured("cds")
        backend = registry.get("cds")
        assert backend.backend_id == "cds"
    finally:
        await foundation.persistence.close()


@pytest.mark.asyncio
async def test_factory_passes_credentials_through(
    tmp_path: Path, monkeypatch
) -> None:
    """T-CDS-001 completion: the scaffold factory previously ``del``'d
    resolver-supplied creds defensively. Now that ``CdsApiKeyAdapter``
    is implemented and Tier-A reviewed, the factory passes
    ``ResolvedCredentials`` through and the backend constructs an
    auth adapter."""
    from copernicus_mcp.auth import CredentialResolver
    from copernicus_mcp.auth.resolver import ResolvedCredentials
    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.config import ConfigLoader

    fake_creds = ResolvedCredentials(
        backend="cds",
        source="explicit",
        source_detail="test",
        fields={"key": "abcdef01-2345-6789-abcd-ef0123456789"},
    )

    def fake_resolve(self, backend: str, override=None):
        return fake_creds if backend == "cds" else None

    monkeypatch.setattr(CredentialResolver, "resolve", fake_resolve)

    config = ConfigLoader().load(
        cli_overrides={
            "enabled_backends": ["cds"],
            "storage": {
                "state_database": str(tmp_path / "state.db"),
                "cache_directory": str(tmp_path / "cache"),
            },
        }
    )
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        backend = registry.get("cds")
        # Factory now plumbs the resolver creds through.
        assert backend._credentials is fake_creds  # type: ignore[attr-defined]
        # Auth adapter constructed.
        assert backend._auth_adapter is not None  # type: ignore[attr-defined]
    finally:
        await foundation.persistence.close()


@pytest.mark.asyncio
async def test_status_reports_cds_unconfigured_without_creds(
    tmp_path: Path, monkeypatch
) -> None:
    """``copernicus_mcp_status`` reports CDS as registered + enabled +
    configured=False when no creds are present."""
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    monkeypatch.delenv("CDSAPI_RC", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "no-rc"))

    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.config import ConfigLoader
    from copernicus_mcp.workflow.orchestrator import WorkflowOrchestrator

    config = ConfigLoader().load(
        cli_overrides={
            "enabled_backends": ["cmems", "cds"],
            "storage": {
                "state_database": str(tmp_path / "state.db"),
                "cache_directory": str(tmp_path / "cache"),
            },
        }
    )
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        orch = WorkflowOrchestrator(registry=registry, foundation=foundation)
        out = await orch.status()
        cds_block = out["backends"]["cds"]
        assert cds_block["registered"] is True
        assert cds_block["enabled_in_config"] is True
        assert cds_block["configured"] is False
        assert cds_block["credential_source"] == "missing"
    finally:
        await foundation.persistence.close()


@pytest.mark.asyncio
async def test_status_reports_cds_configured_with_env_creds(
    tmp_path: Path, monkeypatch
) -> None:
    """T-CDS-001 completion: with ``CDSAPI_KEY`` set, status must report
    ``configured=True, credential_source=env``."""
    monkeypatch.setenv("CDSAPI_KEY", "abcdef01-2345-6789-abcd-ef0123456789")
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    monkeypatch.delenv("CDSAPI_RC", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "no-rc"))

    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.config import ConfigLoader
    from copernicus_mcp.workflow.orchestrator import WorkflowOrchestrator

    config = ConfigLoader().load(
        cli_overrides={
            "enabled_backends": ["cds"],
            "storage": {
                "state_database": str(tmp_path / "state.db"),
                "cache_directory": str(tmp_path / "cache"),
            },
        }
    )
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        orch = WorkflowOrchestrator(registry=registry, foundation=foundation)
        out = await orch.status()
        cds_block = out["backends"]["cds"]
        assert cds_block["configured"] is True
        assert cds_block["credential_source"] == "env"
    finally:
        await foundation.persistence.close()
