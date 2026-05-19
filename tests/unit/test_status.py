from __future__ import annotations

import json
import re
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

    config = ConfigLoader().load(
        cli_overrides={
            "storage": {
                "state_database": str(tmp_path / "state.db"),
                "cache_directory": str(tmp_path / "cache"),
            }
        }
    )
    persistence = SqliteBackend(config.storage.state_database)
    cache = CacheManager(
        cache_directory=config.storage.cache_directory,
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
async def setup(tmp_path):
    from copernicus_mcp.backends.registry import BackendRegistry
    from copernicus_mcp.workflow.orchestrator import WorkflowOrchestrator

    foundation, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    registry = BackendRegistry()
    try:
        yield foundation, registry, WorkflowOrchestrator(
            registry=registry, foundation=foundation
        )
    finally:
        await persistence.close()


def _stub_backend(bid: str = "cmems"):
    from copernicus_mcp.backends.abstract import AbstractBackend

    class _Stub(AbstractBackend):
        backend_id = bid

        async def search(self, params): return {}
        async def describe(self, identifier): return {}
        async def validate(self, params): return {"valid": True}
        async def estimate(self, params): return {}
        async def submit(self, params): return {}
        async def check_status(self, request_id): return {}
        async def fetch_result(self, request_id, target): return {}
        async def cancel(self, request_id): return {"cancelled": True}
        @property
        def supports_async(self): return False
        @property
        def supports_dry_run(self): return True
        @property
        def requires_terms_acceptance(self): return False

    return _Stub


@pytest.mark.asyncio
async def test_status_returns_expected_keys(setup) -> None:
    foundation, registry, orch = setup
    out = await orch.status()
    assert set(out.keys()) >= {"version", "backends", "cache", "persistence", "config"}
    assert isinstance(out["backends"], dict)
    assert "directory" in out["cache"]
    assert "database_path" in out["persistence"]
    assert out["version"]


@pytest.mark.asyncio
async def test_status_lists_registered_backend(setup) -> None:
    foundation, registry, orch = setup
    Stub = _stub_backend("cmems")
    registry.register(Stub(foundation=foundation))
    out = await orch.status()
    assert "cmems" in out["backends"]
    assert out["backends"]["cmems"]["registered"] is True


@pytest.mark.asyncio
async def test_status_credential_source_missing_when_none(
    setup, monkeypatch
) -> None:
    # Isolate HOME so a developer with real CMEMS credentials at
    # ``~/.copernicusmarine/.copernicusmarine-credentials`` does not see
    # this test flip to "configured: true".
    foundation, registry, orch = setup
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_USERNAME", raising=False)
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_PASSWORD", raising=False)
    monkeypatch.setenv("HOME", "/nonexistent-test-home")
    Stub = _stub_backend("cmems")
    registry.register(Stub(foundation=foundation))
    out = await orch.status()
    assert out["backends"]["cmems"]["configured"] is False
    assert out["backends"]["cmems"]["credential_source"] == "missing"


@pytest.mark.asyncio
async def test_status_credential_source_when_resolved(setup, monkeypatch) -> None:
    from copernicus_mcp.auth.resolver import ResolvedCredentials

    foundation, registry, orch = setup
    Stub = _stub_backend("cmems")
    registry.register(Stub(foundation=foundation))

    monkeypatch.setattr(
        foundation.credential_resolver.__class__,
        "resolve",
        lambda self, b, override=None: ResolvedCredentials(
            backend=b,
            source="env",
            source_detail="env_vars",
            fields={"username": "u", "password": "p"},
        ),
    )
    out = await orch.status()
    assert out["backends"]["cmems"]["configured"] is True
    assert out["backends"]["cmems"]["credential_source"] == "env"


@pytest.mark.asyncio
async def test_status_unregistered_backend_listed_when_enabled(setup) -> None:
    """An enabled-but-not-registered backend is still surfaced as configured=False."""
    foundation, registry, orch = setup
    # default config has enabled_backends=["cmems"]; we register nothing.
    out = await orch.status()
    assert "cmems" in out["backends"]
    assert out["backends"]["cmems"]["registered"] is False


@pytest.mark.asyncio
async def test_status_no_credential_value_in_serialised_output(setup, monkeypatch) -> None:
    """the project conventions §2: no credential value may appear in status output."""
    from copernicus_mcp.auth.resolver import ResolvedCredentials

    foundation, registry, orch = setup
    Stub = _stub_backend("cmems")
    registry.register(Stub(foundation=foundation))

    secret = "super-secret-please-redact-1234567890"
    monkeypatch.setattr(
        foundation.credential_resolver.__class__,
        "resolve",
        lambda self, b, override=None: ResolvedCredentials(
            backend=b,
            source="explicit",
            source_detail="cli",
            fields={"username": "u", "password": secret},
        ),
    )
    out = await orch.status()
    blob = json.dumps(out, default=str)
    assert secret not in blob
    assert not re.search(r"password\s*[:=]\s*\w{4,}", blob)
    # Stronger: the credential dict's structural keys must never appear
    # anywhere in the output (catches accidental future leaks via repr/dump).
    assert '"fields"' not in blob
    assert '"username"' not in blob


@pytest.mark.asyncio
async def test_status_cache_metrics_populated(setup) -> None:
    foundation, registry, orch = setup
    out = await orch.status()
    assert isinstance(out["cache"]["size_bytes"], int)
    assert isinstance(out["cache"]["entry_count"], int)
    assert out["cache"]["size_bytes"] >= 0


@pytest.mark.asyncio
async def test_status_config_block_no_credentials(setup) -> None:
    foundation, registry, orch = setup
    out = await orch.status()
    cfg_blob = json.dumps(out["config"], default=str)
    # No credential-shaped tokens should leak through the config dump.
    assert "password" not in cfg_blob.lower()


@pytest.mark.asyncio
async def test_status_path_with_credential_shaped_segment_redacted(
    setup, tmp_path: Path
) -> None:
    """codex-batch-T028-T029-followup HIGH: invariant #2 wins over diagnostics.

    A user who sets cache_directory at a path containing credential-shaped
    segments (``/tmp/password=hunter2/``) must not see the secret leak through
    status() output. Diagnostic value is sacrificed to prevent any chance of
    upstream credential exposure via mis-configured paths.
    """
    foundation, registry, orch = setup
    weird_dir = tmp_path / "password=hunter2"
    weird_dir.mkdir(parents=True, exist_ok=True)
    object.__setattr__(foundation.config.storage, "cache_directory", weird_dir)
    out = await orch.status()
    assert "hunter2" not in out["cache"]["directory"]
    assert "[REDACTED]" in out["cache"]["directory"]


@pytest.mark.asyncio
async def test_status_database_path_credential_shaped_redacted(
    setup, tmp_path: Path
) -> None:
    """database_path must also be sanitised, not just cache directory."""
    foundation, registry, orch = setup
    weird_db = tmp_path / "access_token=abc123" / "state.db"
    weird_db.parent.mkdir(parents=True, exist_ok=True)
    object.__setattr__(foundation.config.storage, "state_database", weird_db)
    out = await orch.status()
    assert "abc123" not in out["persistence"]["database_path"]


@pytest.mark.asyncio
async def test_status_rejects_credential_shaped_backend_id(setup) -> None:
    """codex-batch-T028-T029-followup MEDIUM: dict-key leak prevented at source.

    Backend ids surface as dict keys in status() output. Sanitising the keys
    in-place is collision-prone (two distinct credential-shaped keys would
    collapse to the same redacted string and silently drop one value). Fix
    upstream: BackendRegistry.register validates the id, so the leak class
    is impossible by construction.
    """
    from copernicus_mcp.errors import BackendError

    foundation, registry, orch = setup
    Stub = _stub_backend("evil-password=hunter2")
    with pytest.raises(BackendError):
        registry.register(Stub(foundation=foundation))


@pytest.mark.asyncio
async def test_status_no_unhandled_exception_on_resolver_failure(
    setup, monkeypatch
) -> None:
    """codex-batch-T-029-followup MEDIUM 4: status() must wrap errors."""
    foundation, registry, orch = setup
    Stub = _stub_backend("cmems")
    registry.register(Stub(foundation=foundation))

    def boom(self, b, override=None):
        raise RuntimeError("resolver kaboom: password=hunter2")

    monkeypatch.setattr(
        foundation.credential_resolver.__class__, "resolve", boom
    )
    out = await orch.status()
    assert "error" in out
    assert out["error"]["error_class"] == "BackendError"
    # Sanitiser must scrub the credential-shaped substring in the error.
    assert "hunter2" not in json.dumps(out, default=str)


@pytest.mark.asyncio
async def test_status_enabled_in_config_flag(setup) -> None:
    """codex-batch-T-029-followup MEDIUM 2: enabled_in_config disambiguates registry-only backends."""
    foundation, registry, orch = setup
    Stub = _stub_backend("cmems")
    registry.register(Stub(foundation=foundation))
    out = await orch.status()
    assert out["backends"]["cmems"]["enabled_in_config"] is True


@pytest.mark.asyncio
async def test_status_config_round_trips_through_sanitiser_unchanged(setup) -> None:
    """Pass-2 L4: lock the invariant that no current config field name
    collides with Sanitiser keywords (search/budget/cache/etc.).
    A future field named ``token``/``password``/``secret`` etc. would
    break this test loudly — that's the point.
    """
    foundation, registry, orch = setup
    out = await orch.status()
    assert out["config"]["cache"] == foundation.config.cache.model_dump()
    assert out["config"]["budget"] == foundation.config.budget.model_dump()
