"""``CdsBackend.search`` and ``CdsBackend.describe`` integration tests
(T-CDS-003).

The backend reads from the bundled catalogue snapshot — no network. These
tests validate the orchestrator-shaped contract:
- ``search(params)`` accepts ``{keyword?, store?, limit?}`` and returns
  ``{datasets, total_count}``.
- ``describe(identifier)`` returns the full STAC record by id, doing a
  cross-store lookup so the agent doesn't have to remember which store
  it came from.
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


@pytest.mark.asyncio
async def test_search_returns_datasets_envelope(foundation) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend

    backend = CdsBackend(foundation=foundation, credentials=None)
    out = await backend.search({"keyword": "ERA5", "store": "cds"})
    assert "datasets" in out
    assert "total_count" in out
    assert out["total_count"] == len(out["datasets"])
    assert out["total_count"] > 0


@pytest.mark.asyncio
async def test_search_no_filters_returns_all_stores(foundation) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend

    backend = CdsBackend(foundation=foundation, credentials=None)
    out = await backend.search({})
    assert out["total_count"] >= 100  # CDS alone has >100 datasets


@pytest.mark.asyncio
async def test_search_store_filter(foundation) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend

    backend = CdsBackend(foundation=foundation, credentials=None)
    out = await backend.search({"store": "ads"})
    stores = {d["store"] for d in out["datasets"]}
    assert stores == {"ads"}


@pytest.mark.asyncio
async def test_search_invalid_store_returns_validation_error(
    foundation,
) -> None:
    """Schema rejects unknown stores at the backend boundary."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import ValidationError

    backend = CdsBackend(foundation=foundation, credentials=None)
    with pytest.raises(ValidationError):
        await backend.search({"store": "not-a-store"})


@pytest.mark.asyncio
async def test_search_strips_options_magic_key(foundation) -> None:
    """Orchestrator ``__options`` must be stripped before Pydantic
    ``extra=forbid`` rejects."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    backend = CdsBackend(foundation=foundation, credentials=None)
    out = await backend.search({"keyword": "ERA5", "__options": {"force_refresh": True}})
    assert out["total_count"] > 0


@pytest.mark.asyncio
async def test_search_result_is_sanitised(foundation) -> None:
    """Even though the bundled catalogue is scrubbed, the boundary
    runs the response through Sanitiser. Smoke-test that no
    credential-shaped string survives in titles or descriptions."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    backend = CdsBackend(foundation=foundation, credentials=None)
    out = await backend.search({"keyword": "ERA5"})
    flat = repr(out)
    assert "password=" not in flat
    assert "Bearer " not in flat


@pytest.mark.asyncio
async def test_describe_returns_full_stac_record(foundation) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend

    backend = CdsBackend(foundation=foundation, credentials=None)
    record = await backend.describe("reanalysis-era5-single-levels")
    assert record["id"] == "reanalysis-era5-single-levels"
    assert record["store"] == "cds"
    # Full STAC fields are present (not just slim).
    assert "description" in record


@pytest.mark.asyncio
async def test_describe_carries_available_inputs_from_bundle(foundation) -> None:
    """T-CDS-015 (Layer A): the bundled constraints snapshot ships
    empty-input valid values for each dataset; describe() merges them
    in. This is what tells the LLM 'use data_format / download_format,
    not legacy format'."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    backend = CdsBackend(foundation=foundation, credentials=None)
    record = await backend.describe("reanalysis-era5-single-levels")
    avail = record.get("available_inputs")
    assert avail is not None, "available_inputs must be present for ERA5"
    # Modern key names from the new CDS engine — agents that copy the
    # legacy ``format`` would otherwise get a silent remote_job_failed.
    assert "data_format" in avail
    assert "download_format" in avail


@pytest.mark.asyncio
async def test_describe_unknown_id_raises_not_found(foundation) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import NotFoundError

    backend = CdsBackend(foundation=foundation, credentials=None)
    with pytest.raises(NotFoundError):
        await backend.describe("nonexistent-dataset-id-zzz9999")


@pytest.mark.asyncio
async def test_describe_blank_id_raises_validation_error(foundation) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import ValidationError

    backend = CdsBackend(foundation=foundation, credentials=None)
    with pytest.raises(ValidationError):
        await backend.describe("   ")
