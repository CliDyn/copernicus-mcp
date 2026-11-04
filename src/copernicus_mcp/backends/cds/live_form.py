"""Live CDS form probe for the submit key check (T-CDS-KEYCHECK-002).

The submit path refuses input keys the bundled constraints snapshot does not
list (T-CDS-KEYCHECK-001). A snapshot goes stale when CDS adds a key — E-OBS
v33 gained a required ``licence_type`` that the 2026-05-16 snapshot lacked, so
every valid E-OBS v33 request was refused. Before refusing, the key check asks
the live form: ``POST {store}/retrieve/v1/processes/{id}/constraints`` with
empty inputs (anonymous, the same call ``apply_constraints`` makes) lists every
selectable field of the dataset.

``fetch_live_form_keys`` is advisory, like ``fetch_costing``: any failure
(timeout, network, non-2xx, non-JSON, a body that is not a ``{field: [values]}``
mapping) yields ``None`` and a ``warning`` log, and the caller keeps its
refusal. ``asyncio.CancelledError`` is re-raised, never swallowed.
"""

from __future__ import annotations

import asyncio
from typing import Any

from copernicus_mcp.backends.cds.costing import _Catalogue, _HttpClientFactory
from copernicus_mcp.observability.logger import get_logger

logger = get_logger(__name__)

# The probe runs on the submit path only when the snapshot flags a key; a slow
# endpoint must not hold the submit for the 60 s HTTP default.
LIVE_FORM_TIMEOUT_SECONDS = 15.0


def _form_keys(payload: Any) -> set[str] | None:
    """The field names of a ``{field: [values, ...]}`` constraints payload, or
    ``None`` for any other shape. Empty value lists are allowed (continuous
    fields such as ``area`` come back that way); an empty mapping is not a form.
    """
    if not isinstance(payload, dict) or not payload:
        return None
    for key, values in payload.items():
        if not isinstance(key, str) or not key or not isinstance(values, list):
            return None
    return set(payload)


async def fetch_live_form_keys(
    dataset_id: str,
    *,
    http_client_factory: _HttpClientFactory,
    catalogue: _Catalogue,
) -> set[str] | None:
    """Return the input keys the dataset's live CDS form lists, or ``None``
    when the form cannot be read or has an unexpected shape."""
    # Lazy import: backend imports this module (same reason as costing.py).
    from copernicus_mcp.backends.cds.backend import _STORE_ENDPOINT_URLS

    store = catalogue.store_for(dataset_id) or "cds"
    base_url = _STORE_ENDPOINT_URLS.get(store, _STORE_ENDPOINT_URLS["cds"])
    url = f"{base_url}/retrieve/v1/processes/{dataset_id}/constraints"

    try:
        async with http_client_factory.create("cds") as client:
            response = await client.post(
                url, json={"inputs": {}}, timeout=LIVE_FORM_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            payload = response.json()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 — advisory; the caller keeps its refusal
        logger.warning(
            "live CDS form unavailable for %s: %s",
            dataset_id,
            type(exc).__name__,
            extra={"dataset_id": dataset_id, "error_type": type(exc).__name__},
        )
        return None

    keys = _form_keys(payload)
    if keys is None:
        logger.warning(
            "live CDS form for %s has an unexpected shape",
            dataset_id,
            extra={"dataset_id": dataset_id},
        )
    return keys
