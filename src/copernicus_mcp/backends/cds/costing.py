"""CDS ``/costing`` pre-flight client (T-CDS-EST2-001).

The CDS engine exposes ``POST {store}/retrieve/v1/processes/{id}/costing``
(same anonymous family as ``/constraints``). It returns an abstract *cost*
unit and the dataset's per-request *limit* — never a byte size — e.g.
``{"id": "size", "cost": 1827.0, "limit": 400.0}`` (see
``spikes/T-CDS-EST2-000-costing/FINDINGS.md``).

``fetch_costing`` is advisory: any failure (timeout, network, non-2xx,
non-JSON, missing keys) yields ``None`` and a ``warning`` log so the estimate
path can fall back to the local heuristic without raising.
``asyncio.CancelledError`` is the one exception that is re-raised, never
swallowed (project invariant: cancellation is a control-flow signal).
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, computed_field

from copernicus_mcp.observability.logger import get_logger

logger = get_logger(__name__)


class _Catalogue(Protocol):
    def store_for(self, dataset_id: str) -> str | None: ...


class _HttpClientFactory(Protocol):
    def create(self, backend_id: str) -> httpx.AsyncClient: ...


class CostingResult(BaseModel):
    """Parsed ``/costing`` response. ``units``/``limit`` are abstract cost
    units (field count for gridded adaptors, file count for whole-file
    products), not bytes."""

    model_config = {"frozen": True}

    units: float
    limit: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def exceeds_limit(self) -> bool:
        return self.units > self.limit


async def fetch_costing(
    dataset_id: str,
    inputs: dict[str, Any],
    *,
    http_client_factory: _HttpClientFactory,
    catalogue: _Catalogue,
) -> CostingResult | None:
    """Return the dataset's costing for ``inputs``, or ``None`` on any failure.

    Routes to the dataset's store endpoint (cds/ads/ewds, falling back to
    cds). Anonymous, like ``apply_constraints``.
    """
    # Lazy import to avoid a circular import at module load (backend imports
    # this module from its estimate path); mirrors estimator.py's lazy
    # ``runtime_supports`` import.
    from copernicus_mcp.backends.cds.backend import _STORE_ENDPOINT_URLS

    store = catalogue.store_for(dataset_id) or "cds"
    base_url = _STORE_ENDPOINT_URLS.get(store, _STORE_ENDPOINT_URLS["cds"])
    url = f"{base_url}/retrieve/v1/processes/{dataset_id}/costing"

    try:
        async with http_client_factory.create("cds") as client:
            response = await client.post(url, json={"inputs": inputs})
            response.raise_for_status()
            payload = response.json()
        units = float(payload["cost"])
        limit = float(payload["limit"])
        # A NaN/Inf/non-positive-limit payload is unusable — downstream
        # ``int(units)`` would crash. Treat it as "costing unavailable" so the
        # heuristic path runs, honouring the advisory contract.
        if not (math.isfinite(units) and math.isfinite(limit) and limit > 0 and units >= 0):
            raise ValueError(f"unusable costing payload: units={units} limit={limit}")
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — costing is advisory; never raise
        logger.warning(
            "costing unavailable for %s; falling back to heuristic: %s",
            dataset_id,
            type(exc).__name__,
            extra={"dataset_id": dataset_id, "error_type": type(exc).__name__},
        )
        return None

    return CostingResult(units=units, limit=limit)
