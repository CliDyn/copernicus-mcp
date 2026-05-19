"""``CdsBackend.validate`` tests (T-CDS-002).

Schema-only validation per research §6.9.1 — server-side
``apply_constraints`` is unavailable. ``validate`` returns
``{valid: bool, errors: [...]}`` without any network call.

Errors are passed through Sanitiser before returning — Pydantic
``msg``/``loc`` fields can echo raw user input, which would otherwise
surface a credential-shaped value in the tool output (mirrors CMEMS
``validate`` discipline).
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


def _good_params() -> dict[str, object]:
    return {
        "dataset_id": "reanalysis-era5-single-levels",
        "inputs": {
            "variable": ["2m_temperature"],
            "year": ["2024"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00"],
            "data_format": "grib",
        },
    }


@pytest.mark.asyncio
async def test_validate_returns_valid_for_good_params(foundation) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend

    backend = CdsBackend(foundation=foundation, credentials=None)
    out = await backend.validate(_good_params())
    assert out == {"valid": True}


@pytest.mark.asyncio
async def test_validate_returns_errors_for_blank_dataset_id(
    foundation,
) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend

    backend = CdsBackend(foundation=foundation, credentials=None)
    params = _good_params()
    params["dataset_id"] = ""
    out = await backend.validate(params)
    assert out["valid"] is False
    assert len(out["errors"]) >= 1


@pytest.mark.asyncio
async def test_validate_returns_errors_for_empty_inputs(foundation) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend

    backend = CdsBackend(foundation=foundation, credentials=None)
    params = _good_params()
    params["inputs"] = {}
    out = await backend.validate(params)
    assert out["valid"] is False


@pytest.mark.asyncio
async def test_validate_returns_errors_for_nested_dict(foundation) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend

    backend = CdsBackend(foundation=foundation, credentials=None)
    params = _good_params()
    params["inputs"] = {"variable": ["t"], "options": {"nested": "bad"}}
    out = await backend.validate(params)
    assert out["valid"] is False


@pytest.mark.asyncio
async def test_validate_strips_options_magic_key(foundation) -> None:
    """Orchestrator injects ``__options`` with confirmed/async_mode/etc.
    ``validate`` must strip it before Pydantic ``extra=forbid`` rejects."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    backend = CdsBackend(foundation=foundation, credentials=None)
    params = _good_params()
    params["__options"] = {"confirmed": True}
    out = await backend.validate(params)
    assert out["valid"] is True


@pytest.mark.asyncio
async def test_validate_sanitises_error_output(foundation) -> None:
    """Round 1 review: an earlier version of this test planted the value
    in a nested dict, which the schema rejects with a generic
    "nested dicts not allowed" message that NEVER echoes the planted
    value — passing whether or not the sanitiser ran. Use a top-level
    extra-forbidden key instead: Pydantic emits the offending key in
    the ``loc`` of the error, so the sanitiser's structural walk is
    actually exercised. The redaction proves the value is filtered."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    backend = CdsBackend(foundation=foundation, credentials=None)
    # Top-level extra forbidden key carries the credential-shaped value
    # in its OWN name — Pydantic surfaces it in ``loc``.
    params: dict[str, object] = {
        "dataset_id": "ds-1",
        "inputs": {"variable": ["t"]},
        "password=hunter2-very-secret": "x",  # extra-forbidden + sensitive
    }
    out = await backend.validate(params)
    assert out["valid"] is False
    raw = repr(out)
    assert "hunter2-very-secret" not in raw, (
        "credential-shaped value survived sanitiser walk through error loc"
    )
    # And confirm the redaction marker DID land in the loc.
    assert "[REDACTED]" in raw
