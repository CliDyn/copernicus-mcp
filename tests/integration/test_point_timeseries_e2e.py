"""End-to-end integration tests for point time-series (T-TS-009).

Gated by ``RUN_INTEGRATION_TESTS=1``; individual tests additionally skip
when the relevant backend's credentials are not resolvable. Covers the
T-TIMESERIES acceptance paths against the real services:

  - CDS ``reanalysis-era5-land-timeseries`` with a nested ``location`` point
    submits past validation (T-TS-001) — the headline unblock.
  - CMEMS point estimate is KB-scale, not the ~1.2 GB transfer figure
    (T-TS-002), so the confirmation gate does not false-fire.
  - CMEMS point subset with ``file_format="csv"`` returns a ``.csv``
    descriptor (T-TS-005); the default stays netcdf.

    export RUN_INTEGRATION_TESTS=1
    export CDSAPI_KEY=<uuid-pat>          # for the CDS test
    # CMEMS creds via ~/.copernicusmarine or config for the marine tests
    pytest tests/integration/test_point_timeseries_e2e.py -q
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="set RUN_INTEGRATION_TESTS=1 to run integration tests",
)


@pytest.fixture
def _require_cds_creds() -> None:
    from copernicus_mcp.auth import CredentialResolver

    if CredentialResolver().resolve("cds") is None:
        pytest.skip("CDS credentials not resolvable (set CDSAPI_KEY or ~/.cdsapirc)")


@pytest.fixture
def _require_cmems_creds() -> None:
    from copernicus_mcp.auth import CredentialResolver

    if CredentialResolver().resolve("cmems") is None:
        pytest.skip("CMEMS credentials not resolvable (set ~/.copernicusmarine or config)")


@pytest_asyncio.fixture
async def orchestrator(tmp_path: Path, monkeypatch):
    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.config import ConfigLoader
    from copernicus_mcp.config import loader as _loader_module
    from copernicus_mcp.workflow.orchestrator import WorkflowOrchestrator

    monkeypatch.setattr(_loader_module, "_USER_CONFIG_PATHS", ())
    config = ConfigLoader().load(
        cli_overrides={
            "storage": {
                "state_database": str(tmp_path / "state.db"),
                "cache_directory": str(tmp_path / "cache"),
            },
            "enabled_backends": ["cmems", "cds"],
        },
    )
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        yield WorkflowOrchestrator(registry=registry, foundation=foundation)
    finally:
        await foundation.persistence.close()


# A long single-point series in the open Baltic (thetao = potential temperature).
_CMEMS_POINT: dict[str, Any] = {
    "dataset_id": "cmems_mod_bal_phy_my_P1D-m",
    "variables": ["thetao"],
    "minimum_longitude": 20.0,
    "maximum_longitude": 20.0,
    "minimum_latitude": 58.0,
    "maximum_latitude": 58.0,
    "minimum_depth": 0.0,
    "maximum_depth": 1.0,
    "start_datetime": "2014-01-01T00:00:00Z",
    "end_datetime": "2015-12-31T00:00:00Z",
    "coordinates_selection_method": "nearest",
}

_ERA5_LAND_TS: dict[str, Any] = {
    "dataset_id": "reanalysis-era5-land-timeseries",
    "inputs": {
        "variable": "2m_temperature",
        "location": {"latitude": 47.4979, "longitude": 19.0402},
        "date": "2023-06-01/2023-06-07",
        "data_format": "csv",
    },
}


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_cds_era5_land_timeseries_passes_validation(
    _require_cds_creds, orchestrator
) -> None:
    """T-TS-001 headline: the nested ``location`` request reaches the server
    instead of failing client-side. We assert the absence of the old
    'nested dicts are not allowed' ValidationError — a server-side T&C or
    queue error is acceptable (still proves validation passed)."""
    out = await orchestrator.run(
        backend="cds", operation="estimate", params=_ERA5_LAND_TS
    )
    blob = str(out).lower()
    # The point: the request passed client-side validation rather than failing
    # on the nested ``location``. A server-side T&C/queue error is acceptable.
    assert "nested dicts are not allowed" not in blob


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_cmems_point_estimate_is_kb_not_gb(
    _require_cmems_creds, orchestrator
) -> None:
    """T-TS-002: a point's gated size reflects the ~KB output file, not the
    ~1.2 GB chunk-read transfer that used to trip the 1 GB gate."""
    out = await orchestrator.run(
        backend="cmems", operation="estimate", params=_CMEMS_POINT
    )
    result = out.get("result", out)
    assert "error" not in out, f"estimate errored: {out}"
    size = result["estimated_size_bytes"]
    # Generous ceiling: the real output is ~60-160 KB; anything under 50 MB
    # proves we are no longer reporting the GB-scale transfer figure.
    assert 0 < size < 50 * 1024 * 1024, f"point estimate not KB-scale: {size}"


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_cmems_point_csv_download(
    _require_cmems_creds, orchestrator
) -> None:
    """T-TS-005: ``file_format='csv'`` yields a ``.csv`` descriptor."""
    params = {**_CMEMS_POINT, "file_format": "csv"}
    out = await orchestrator.run(
        backend="cmems",
        operation="submit",
        params=params,
        options={"confirmed": True},
    )
    result = out.get("result", out)
    assert "error" not in out, f"submit errored: {out}"
    filepath = result["filepath"]
    assert filepath.endswith(".csv")
    assert Path(filepath).exists()
