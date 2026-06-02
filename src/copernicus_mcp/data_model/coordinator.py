"""``DataModelCoordinator`` — pure cache-key derivation for backend requests.

The coordinator holds a ``PersistenceBackend`` reference for future read/write
of cache metadata, but the cache-key methods themselves are pure (no I/O).
"""

from __future__ import annotations

from typing import Any

from copernicus_mcp.common.time import iso8601_utc
from copernicus_mcp.data_model.cache_key import construct_cache_key
from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest
from copernicus_mcp.data_model.schemas_cmems import (
    CmemsGetRequest,
    CmemsSearchRequest,
    CmemsSubsetRequest,
)
from copernicus_mcp.persistence import PersistenceBackend

_SCHEMA_VERSION = "v1.0"
_CMEMS_API_VERSION = "marine-data-store-v1"
_CMEMS_BACKEND = "cmems"
_CDS_API_VERSION = "cds-engine-v1"
_CDS_BACKEND = "cds"
# Search has no dataset_id; use a stable sentinel so the cache-key prefix is
# still well-formed.
_SEARCH_SENTINEL = "_search"


def _lists_to_tuples(value: Any) -> Any:
    """Recursively convert lists → tuples to preserve positional order in
    ``canonicalise_value``. See ``cache_key_for_cds_retrieve`` rationale."""
    if isinstance(value, dict):
        return {k: _lists_to_tuples(v) for k, v in value.items()}
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    return value


class DataModelCoordinator:
    def __init__(self, persistence: PersistenceBackend) -> None:
        self._persistence = persistence

    def cache_key_for_subset(self, req: CmemsSubsetRequest) -> str:
        spatial = {
            "min_lon": req.minimum_longitude,
            "max_lon": req.maximum_longitude,
            "min_lat": req.minimum_latitude,
            "max_lat": req.maximum_latitude,
            "min_depth": req.minimum_depth,
            "max_depth": req.maximum_depth,
        }
        temporal = {
            "start": req.start_datetime,
            "end": req.end_datetime,
        }
        format_options = {
            "file_format": req.file_format,
            # csv ignores compression (the csv download path strips it before
            # read_dataframe) — keep the key independent of it so two equal
            # csv requests dedupe (the project conventions inv #6).
            "netcdf_compression_level": (
                None if req.file_format == "csv" else req.netcdf_compression_level
            ),
            "coordinates_selection_method": req.coordinates_selection_method,
            "service": req.service,
            "dataset_part": req.dataset_part,
        }
        return construct_cache_key(
            backend=_CMEMS_BACKEND,
            operation="submit",
            dataset_id=req.dataset_id,
            dataset_version=req.dataset_version,
            api_version=_CMEMS_API_VERSION,
            schema_version=_SCHEMA_VERSION,
            spatial=spatial,
            temporal=temporal,
            variables=list(req.variables),
            format_options=format_options,
        )

    def cache_key_for_cds_retrieve(self, req: CdsRetrieveRequest) -> str:
        """Build a deterministic cache key for a CDS retrieve request.

        Per ``the project research notes`` §6.9.1 the per-dataset
        ``inputs`` dict is opaque — keys differ across datasets and cannot
        be statically enumerated. We pack the dict into the
        ``format_options`` bucket of ``construct_cache_key`` after
        converting lists to tuples recursively.

        Why list→tuple: ``canonicalise_value`` sorts lists (set-like
        semantics; correct for CMEMS ``variables``) but preserves tuple
        order (positional; correct for ``area: [N, W, S, E]``). Since
        CDS inputs mix positional fields (``area``, ``grid``) and
        set-like fields (``variable``, ``year``, ``month``) without
        per-key signal, we treat ALL lists as positional. This means
        equivalent variable lists in different order generate distinct
        keys (cache miss on the second submit) — accepted trade-off vs
        risking ``area`` collisions for genuinely different bboxes.

        ``construct_cache_key`` recursively strips credential-shaped keys
        (``api_key``, ``password`` …) so a buggy caller that smuggles a
        secret into ``inputs`` cannot taint the digest.

        Limitation (codex T-CDS-002 LOW): the sanitiser's
        ``SENSITIVE_REQUEST_KEYS`` set matches keys EXACTLY (case-folded)
        — variants like ``backend-password``, ``clientSecret``, or
        ``password=<value>`` (where the literal value is part of the
        key) do NOT collapse and therefore can change the digest. They
        do not leak the literal value into the returned cache_key
        (which is a sha-256 prefix). Documented residual risk per
        ``docs/pending_codex_review.md`` accepted-residuals.
        """
        return construct_cache_key(
            backend=_CDS_BACKEND,
            operation="submit",
            dataset_id=req.dataset_id,
            dataset_version=None,
            api_version=_CDS_API_VERSION,
            schema_version=_SCHEMA_VERSION,
            spatial=None,
            temporal=None,
            variables=None,
            format_options={"inputs": _lists_to_tuples(req.inputs)},
        )

    def cache_key_for_get(self, req: CmemsGetRequest) -> str:
        """Cache key for a ``copernicusmarine.get`` bundle (T-CMEMS-GET-002).

        Selection state (``filter`` / ``regex`` / ``file_list``) and
        ``dataset_part`` are packed into ``format_options``. SDK-forwarded
        operational flags (``sync`` / ``skip_existing`` / ``overwrite``)
        do NOT contribute — they don't change the data delivered, and
        including them would cache-miss on otherwise-identical requests.

        ``file_list`` is canonicalised set-wise: ``["a","b"]`` and
        ``["b","a"]`` request the same files and must collide.
        ``canonicalise_value`` sorts lists, which gives that property
        for free.
        """
        format_options: dict[str, object] = {}
        if req.filter is not None:
            format_options["filter"] = req.filter
        if req.regex is not None:
            format_options["regex"] = req.regex
        if req.file_list is not None:
            format_options["file_list"] = list(req.file_list)
        if req.dataset_part is not None:
            format_options["dataset_part"] = req.dataset_part
        return construct_cache_key(
            backend=_CMEMS_BACKEND,
            operation="get",
            dataset_id=req.dataset_id,
            dataset_version=req.dataset_version,
            api_version=_CMEMS_API_VERSION,
            schema_version=_SCHEMA_VERSION,
            spatial=None,
            temporal=None,
            variables=None,
            format_options=format_options or None,
        )

    def cache_key_for_search(self, req: CmemsSearchRequest) -> str:
        spatial: dict[str, object] | None = None
        if req.bbox is not None:
            spatial = {"bbox": list(req.bbox)}
        temporal: dict[str, object] | None = None
        if req.time_range is not None:
            # Normalise tz-aware offsets so equivalent times collide on the key.
            temporal = {
                "start": iso8601_utc(req.time_range[0]),
                "end": iso8601_utc(req.time_range[1]),
            }
        format_options: dict[str, object] = {}
        if req.product_id is not None:
            format_options["product_id"] = req.product_id
        if req.service_types is not None:
            format_options["service_types"] = list(req.service_types)
        if req.limit is not None:
            format_options["limit"] = req.limit
        if req.keyword is not None:
            format_options["keyword"] = req.keyword
        return construct_cache_key(
            backend=_CMEMS_BACKEND,
            operation="search",
            dataset_id=_SEARCH_SENTINEL,
            dataset_version=None,
            api_version=_CMEMS_API_VERSION,
            schema_version=_SCHEMA_VERSION,
            spatial=spatial,
            temporal=temporal,
            variables=None,
            format_options=format_options or None,
        )
