"""Unit tests for ``CmemsBackend.list_files`` (T-CMEMS-GET-INDEX-004).

Wires the IndexStore + filter primitives into a backend method and
verifies the canonical envelope shape pinned in the sub-plan
"Acceptance for the sub-plan" section.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pandas as pd
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


def _sample_df() -> pd.DataFrame:
    """A tiny synthetic IndexRow DataFrame for filter-path tests."""
    return pd.DataFrame(
        {
            "file_path": ["med_a.nc", "med_b.nc", "atl.nc"],
            "lon_min": [10.0, 15.0, -30.0],
            "lon_max": [12.0, 17.0, -20.0],
            "lat_min": [35.0, 40.0, 5.0],
            "lat_max": [38.0, 42.0, 10.0],
            "time_start": pd.to_datetime(
                ["2010-06-01T00:00:00Z", "2011-06-01T00:00:00Z", "2010-06-01T00:00:00Z"],
                utc=True,
            ),
            "time_end": pd.to_datetime(
                ["2010-06-30T00:00:00Z", "2011-06-30T00:00:00Z", "2010-06-30T00:00:00Z"],
                utc=True,
            ),
            "platform_type": ["PF", "CT", "PF"],
            "variables": [("TEMP", "PSAL"), ("TEMP",), ("PSAL",)],
            "size_bytes": [1000, 2000, None],
        }
    )


def _backend_with_mock_store(foundation: Any, df: pd.DataFrame, mode: str = "offline"):
    """Build CmemsBackend with its IndexStore.load mocked to return ``(df, mode)``."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    backend._index_store.load = AsyncMock(return_value=(df, mode))  # type: ignore[method-assign]
    return backend


class TestListFilesEnvelope:
    @pytest.mark.asyncio
    async def test_returns_canonical_envelope_shape(self, foundation: Any) -> None:
        backend = _backend_with_mock_store(foundation, _sample_df(), mode="offline")
        envelope = await backend.list_files({"dataset_id": "x"})
        # Pinned acceptance lines 265-271.
        assert set(envelope.keys()) >= {
            "files",
            "matched_count",
            "matched_count_uncapped",
            "truncated",
            "total_count_in_index",
            "total_size_bytes_known",
            "rows_with_unknown_size",
            "filters_applied",
            "index_fetched_at",
            "mode",
        }

    @pytest.mark.asyncio
    async def test_all_filters_none_returns_full_index(self, foundation: Any) -> None:
        backend = _backend_with_mock_store(foundation, _sample_df())
        envelope = await backend.list_files({"dataset_id": "x"})
        assert envelope["matched_count"] == 3
        assert envelope["total_count_in_index"] == 3
        assert envelope["truncated"] is False

    @pytest.mark.asyncio
    async def test_bbox_filter_narrows_files(self, foundation: Any) -> None:
        backend = _backend_with_mock_store(foundation, _sample_df())
        envelope = await backend.list_files(
            {"dataset_id": "x", "bbox": (5.0, 30.0, 20.0, 46.0)}  # Mediterranean
        )
        assert envelope["matched_count"] == 2
        assert {f["file_path"] for f in envelope["files"]} == {"med_a.nc", "med_b.nc"}

    @pytest.mark.asyncio
    async def test_time_filter_narrows_files(self, foundation: Any) -> None:
        backend = _backend_with_mock_store(foundation, _sample_df())
        envelope = await backend.list_files(
            {
                "dataset_id": "x",
                "time_range": ("2010-01-01T00:00:00Z", "2010-12-31T23:59:59Z"),
            }
        )
        assert envelope["matched_count"] == 2
        assert {f["file_path"] for f in envelope["files"]} == {"med_a.nc", "atl.nc"}

    @pytest.mark.asyncio
    async def test_size_aggregation_distinguishes_known_and_unknown(
        self, foundation: Any
    ) -> None:
        backend = _backend_with_mock_store(foundation, _sample_df())
        envelope = await backend.list_files({"dataset_id": "x"})
        # Two rows have size_bytes (1000 + 2000 = 3000); one is None.
        assert envelope["total_size_bytes_known"] == 3000
        assert envelope["rows_with_unknown_size"] == 1

    @pytest.mark.asyncio
    async def test_limit_truncation_is_deterministic_by_file_path(
        self, foundation: Any
    ) -> None:
        backend = _backend_with_mock_store(foundation, _sample_df())
        envelope = await backend.list_files({"dataset_id": "x", "limit": 2})
        assert envelope["matched_count_uncapped"] == 3
        assert envelope["matched_count"] == 2
        assert envelope["truncated"] is True
        # Deterministic order: file_path ASC.
        names = [f["file_path"] for f in envelope["files"]]
        assert names == sorted(names)

    @pytest.mark.asyncio
    async def test_limit_higher_than_matched_does_not_truncate(
        self, foundation: Any
    ) -> None:
        backend = _backend_with_mock_store(foundation, _sample_df())
        envelope = await backend.list_files({"dataset_id": "x", "limit": 100})
        assert envelope["truncated"] is False
        assert envelope["matched_count"] == envelope["matched_count_uncapped"] == 3

    @pytest.mark.asyncio
    async def test_mode_passes_through_from_store(self, foundation: Any) -> None:
        backend = _backend_with_mock_store(foundation, _sample_df(), mode="fresh")
        envelope = await backend.list_files({"dataset_id": "x"})
        assert envelope["mode"] == "fresh"

    @pytest.mark.asyncio
    async def test_filters_applied_reflects_input(self, foundation: Any) -> None:
        backend = _backend_with_mock_store(foundation, _sample_df())
        envelope = await backend.list_files(
            {
                "dataset_id": "x",
                "bbox": (5.0, 30.0, 20.0, 46.0),
                "variables": ["TEMP"],
            }
        )
        applied = envelope["filters_applied"]
        assert "bbox" in applied
        assert "variables" in applied
        assert "time_range" not in applied  # not provided
        assert "platform_types" not in applied

    @pytest.mark.asyncio
    async def test_combined_filters_and_their_intersection(self, foundation: Any) -> None:
        backend = _backend_with_mock_store(foundation, _sample_df())
        envelope = await backend.list_files(
            {
                "dataset_id": "x",
                "bbox": (5.0, 30.0, 20.0, 46.0),
                "platform_types": ["PF"],
            }
        )
        # Mediterranean + PF = med_a.nc only.
        assert {f["file_path"] for f in envelope["files"]} == {"med_a.nc"}

    @pytest.mark.asyncio
    async def test_credentials_required(self, foundation: Any) -> None:
        from copernicus_mcp.backends.cmems.backend import CmemsBackend
        from copernicus_mcp.errors import AuthError

        backend = CmemsBackend(foundation=foundation, credentials=None)
        with pytest.raises(AuthError):
            await backend.list_files({"dataset_id": "x"})

    @pytest.mark.asyncio
    async def test_files_carry_canonical_row_fields(self, foundation: Any) -> None:
        backend = _backend_with_mock_store(foundation, _sample_df())
        envelope = await backend.list_files({"dataset_id": "x"})
        for row in envelope["files"]:
            assert set(row.keys()) >= {
                "file_path",
                "lon_min",
                "lon_max",
                "lat_min",
                "lat_max",
                "time_start",
                "time_end",
                "platform_type",
                "variables",
                "size_bytes",
            }


class TestRoundOneRegressions:
    """cr round-1 HIGH+MEDIUM regressions."""

    @pytest.mark.asyncio
    async def test_H1_unknown_field_does_not_leak_value_into_error_message(
        self, foundation: Any
    ) -> None:
        # the credential-isolation invariant: pydantic stringifies "input_value='X'" into
        # the error message; that path can leak credential-shaped extras.
        # _validate_list_files must catch pydantic and re-raise with
        # structured field_errors (loc/msg/type only, no raw value).
        from copernicus_mcp.errors import ValidationError

        backend = _backend_with_mock_store(foundation, _sample_df())
        with pytest.raises(ValidationError) as exc_info:
            await backend.list_files(
                {"dataset_id": "x", "password": "hunter2"}
            )
        msg = str(exc_info.value)
        assert "hunter2" not in msg
        # Field-error metadata is in the structured record context,
        # not the message.
        record = exc_info.value.error_record
        ctx = record.context or {}
        assert "field_errors" in ctx
        # The structured field-error entries themselves should not
        # echo raw values either.
        for fe in ctx["field_errors"]:
            assert "hunter2" not in fe.get("msg", "")

    @pytest.mark.asyncio
    async def test_H2_grid_dataset_wrong_format_suggests_subset_not_get(
        self, foundation: Any
    ) -> None:
        # When marine_list_files is invoked on a grid dataset, the SDK
        # raises WrongFormatRequested. The recovery hint must point at
        # marine_subset_dataset (not marine_get_files, which would route
        # back through the same SDK call).
        from copernicus_mcp.backends.cmems.backend import (
            _wrap_subset_exception,
        )

        class _FakeWrongFormat(Exception):
            pass

        marine = types.SimpleNamespace(WrongFormatRequested=_FakeWrongFormat)
        wrapped = _wrap_subset_exception(
            marine, _FakeWrongFormat("grid dataset"), "list_files"
        )
        msg = str(wrapped)
        # Must point at marine_subset_dataset, not marine_get_files.
        assert "marine_subset_dataset" in msg or "subset" in msg.lower()
        assert "marine_get_files" not in msg


class TestFetchedAtContract:
    """cr round-1 M3: lock the (offline → fetched_at not None) contract."""

    @pytest.mark.asyncio
    async def test_offline_mode_surfaces_fetched_at(self, foundation: Any) -> None:
        # Construct a backend whose store has a real Parquet cache so
        # fetched_at returns a real timestamp.
        from datetime import UTC, datetime, timedelta

        from copernicus_mcp.backends.cmems.backend import CmemsBackend

        backend = CmemsBackend(foundation=foundation, credentials=_creds())
        # Seed a fake parquet for one dataset_id.
        dataset_id = "x"
        cache_dir = (
            foundation.config.storage.cache_directory / "marine_indices"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        df = _sample_df()
        df.to_parquet(cache_dir / f"{dataset_id}.parquet")
        # Override only load() — we want to keep the real fetched_at.
        backend._index_store.load = AsyncMock(return_value=(df, "offline"))  # type: ignore[method-assign]
        envelope = await backend.list_files({"dataset_id": dataset_id})
        assert envelope["index_fetched_at"] is not None
        now = datetime.now(UTC)
        # Parse and verify it's recent (within last minute).
        parsed = datetime.strptime(
            envelope["index_fetched_at"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=UTC)
        assert now - parsed < timedelta(seconds=60)


class TestListFilesViaOrchestrator:
    @pytest.mark.asyncio
    async def test_orchestrator_routes_list_files_to_backend(
        self, foundation: Any
    ) -> None:
        # Verify the new "list_files" OperationType reaches the
        # backend.list_files method via _OPERATION_METHOD.
        from copernicus_mcp.backends.registry import BackendRegistry
        from copernicus_mcp.workflow.orchestrator import WorkflowOrchestrator

        backend = _backend_with_mock_store(foundation, _sample_df())
        registry = BackendRegistry()
        registry.register(backend)
        orch = WorkflowOrchestrator(registry=registry, foundation=foundation)
        envelope = await orch.run(
            backend="cmems",
            operation="list_files",
            params={"dataset_id": "x"},
        )
        # cr round-1 M2: assert the backend output actually reached us.
        # WorkflowOrchestrator wraps the backend response, so the
        # backend envelope sits under the "result" key.
        result = envelope.get("result", envelope)
        assert result["matched_count"] == 3
        assert result["mode"] == "offline"
