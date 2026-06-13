"""T-CDS-EST2-003: ``size_observations`` persistence + the ``_safe_rollback``
recursion bugfix."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _obs(
    observation_id: str = "obs-1",
    *,
    dataset_id: str = "cds-x",
    signature: str = "sig-a",
    cost_units: float | None = 24.0,
    size_bytes: int = 1000,
    area_fraction: float = 1.0,
    request_id: str | None = None,
) -> dict:
    return {
        "observation_id": observation_id,
        "backend_id": "cds",
        "dataset_id": dataset_id,
        "signature": signature,
        "cost_units": cost_units,
        "size_bytes": size_bytes,
        "area_fraction": area_fraction,
        "request_id": request_id,
        "observed_at": _ts(),
    }


@pytest.mark.asyncio
async def test_record_and_list_size_observation(sqlite_backend) -> None:
    await sqlite_backend.record_size_observation(_obs())
    rows = await sqlite_backend.list_size_observations("cds", "cds-x", "sig-a")
    assert len(rows) == 1
    assert rows[0]["size_bytes"] == 1000
    assert rows[0]["cost_units"] == 24.0
    assert rows[0]["area_fraction"] == 1.0


@pytest.mark.asyncio
async def test_list_filters_by_signature(sqlite_backend) -> None:
    await sqlite_backend.record_size_observation(_obs("o1", signature="sig-a"))
    await sqlite_backend.record_size_observation(_obs("o2", signature="sig-b"))
    only_a = await sqlite_backend.list_size_observations("cds", "cds-x", "sig-a")
    both = await sqlite_backend.list_size_observations("cds", "cds-x", None)
    assert [r["observation_id"] for r in only_a] == ["o1"]
    assert len(both) == 2


@pytest.mark.asyncio
async def test_cost_units_is_nullable(sqlite_backend) -> None:
    await sqlite_backend.record_size_observation(_obs(cost_units=None))
    rows = await sqlite_backend.list_size_observations("cds", "cds-x", "sig-a")
    assert rows[0]["cost_units"] is None


@pytest.mark.asyncio
async def test_list_other_dataset_returns_empty(sqlite_backend) -> None:
    await sqlite_backend.record_size_observation(_obs())
    assert await sqlite_backend.list_size_observations("cds", "other", None) == []


@pytest.mark.asyncio
async def test_safe_rollback_actually_calls_conn_rollback(sqlite_backend) -> None:
    """Regression: ``_safe_rollback`` previously called *itself* recursively
    (sqlite_backend.py) so the rollback never happened and the
    RecursionError was swallowed. It must call the connection's rollback."""
    sqlite_backend._conn.rollback = AsyncMock()
    await sqlite_backend._safe_rollback()
    sqlite_backend._conn.rollback.assert_awaited_once()
