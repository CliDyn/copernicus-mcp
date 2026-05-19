from __future__ import annotations

import json
from pathlib import Path

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def recorder(tmp_path: Path):
    from copernicus_mcp.data_model.provenance import ProvenanceRecorder
    from copernicus_mcp.persistence import SqliteBackend

    backend = SqliteBackend(tmp_path / "state.db")
    await backend.initialise()
    try:
        yield (
            ProvenanceRecorder(
                persistence=backend,
                software_versions={
                    "python": "3.14.4",
                    "copernicus-mcp": "0.0.1",
                    "copernicusmarine": "2.4.0",
                },
            ),
            backend,
        )
    finally:
        await backend.close()


def _make_file(path: Path, content: bytes = b"hello world") -> Path:
    path.write_bytes(content)
    return path


def _record_kwargs(file_path: Path) -> dict:
    from copernicus_mcp.data_model.provenance import (
        AgentContext,
        BackendBlock,
        CacheRef,
        CostConsumed,
        DatasetBlock,
        RequestBlock,
        SpatialBlock,
        TemporalBlock,
        Variable,
    )

    return dict(
        backend=BackendBlock(
            id="cmems",
            provider="Mercator Ocean International",
            endpoint_url="https://my.cmems-du.eu",
            api_version="marine-data-store-v1",
        ),
        dataset=DatasetBlock(
            dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            dataset_version="202411",
            dataset_part="default",
            product_id="GLOBAL_ANALYSISFORECAST_PHY_001_024",
            service_name="arco-geo-series",
        ),
        request=RequestBlock(
            operation="subset",
            submitted_at="2026-04-25T17:40:11Z",
            started_at="2026-04-25T17:40:11Z",
            finished_at="2026-04-25T17:42:31Z",
            user_request={"variables": ["thetao"]},
            normalized_request={"variables": ["thetao"]},
            options_applied={"dry_run": False},
        ),
        spatial=SpatialBlock(
            native_crs="EPSG:4326",
            output_crs="EPSG:4326",
            bbox_epsg_4326=[-10.0, 35.0, 5.0, 45.0],
        ),
        temporal=TemporalBlock(
            start_datetime="2024-01-01T00:00:00Z",
            end_datetime="2024-01-02T00:00:00Z",
        ),
        variables=[Variable(name="thetao", units="degree_Celsius")],
        files=[file_path],
        cost_consumed=CostConsumed(type="free"),
        source_urls=["https://my.cmems-du.eu/foo"],
        cache=CacheRef(
            cache_key="cmems:submit:cmems_mod_glo_phy_anfc_0.083deg_P1D-m:abc123",
            cache_hit=False,
        ),
        agent_context=AgentContext(tool_name="marine_subset_dataset"),
    )


@pytest.mark.asyncio
async def test_sidecar_written_and_persisted(recorder, tmp_path: Path) -> None:
    rec, backend = recorder
    f = _make_file(tmp_path / "data.nc", b"x" * 4096)
    record_id = await rec.record_successful_retrieve(**_record_kwargs(f))

    sidecar = f.with_suffix(f.suffix + ".provenance.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["schema_version"] == "1.0"
    assert payload["record"]["record_id"] == record_id
    assert payload["files"][0]["md5"]
    assert payload["files"][0]["sha256"]
    assert payload["files"][0]["size_bytes"] == 4096

    persisted = await backend.fetch_provenance(record_id)
    assert persisted is not None
    assert persisted["record_id"] == record_id


@pytest.mark.asyncio
async def test_sidecar_round_trips(recorder, tmp_path: Path) -> None:
    from copernicus_mcp.data_model.provenance import ProvenanceRecord

    rec, _ = recorder
    f = _make_file(tmp_path / "x.nc")
    await rec.record_successful_retrieve(**_record_kwargs(f))
    sidecar = f.with_suffix(f.suffix + ".provenance.json")
    parsed = ProvenanceRecord.model_validate_json(sidecar.read_text())
    assert parsed.schema_version == "1.0"
    assert parsed.dataset.dataset_id.startswith("cmems_")


@pytest.mark.asyncio
async def test_distinct_record_ids(recorder, tmp_path: Path) -> None:
    rec, _ = recorder
    f1 = _make_file(tmp_path / "a.nc", b"a")
    f2 = _make_file(tmp_path / "b.nc", b"b")
    id1 = await rec.record_successful_retrieve(**_record_kwargs(f1))
    id2 = await rec.record_successful_retrieve(**_record_kwargs(f2))
    assert id1 != id2


@pytest.mark.asyncio
async def test_extra_field_forbidden() -> None:
    from pydantic import ValidationError as PydValidationError

    from copernicus_mcp.data_model.provenance import BackendBlock

    with pytest.raises(PydValidationError):
        BackendBlock(  # type: ignore[call-arg]
            id="cmems",
            provider="x",
            endpoint_url="https://x",
            api_version="v1",
            surprise="boom",
        )


@pytest.mark.asyncio
async def test_md5_sha256_match_known_content(recorder, tmp_path: Path) -> None:
    import hashlib

    rec, _ = recorder
    content = b"hello world"
    f = _make_file(tmp_path / "h.nc", content)
    rid = await rec.record_successful_retrieve(**_record_kwargs(f))
    sidecar = f.with_suffix(f.suffix + ".provenance.json")
    payload = json.loads(sidecar.read_text())
    assert payload["files"][0]["md5"] == hashlib.md5(content).hexdigest()
    assert payload["files"][0]["sha256"] == hashlib.sha256(content).hexdigest()
    assert rid.startswith("prv-")
