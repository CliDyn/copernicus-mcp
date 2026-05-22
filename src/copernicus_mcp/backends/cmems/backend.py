"""CMEMS backend.

Iter 1 fills in the eight protocol methods one tool at a time (T-021 .. T-026).
Methods not yet implemented raise ``BackendError`` with
``error_subclass="not_implemented"``.

The ``copernicusmarine`` SDK is imported **inside** the methods that need it,
not at module top — the ``cmems`` extra is opt-in and ``import CmemsBackend``
must succeed in environments without the SDK installed.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import shutil
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from copernicus_mcp.auth.cmems import CmemsBasicAuthAdapter
from copernicus_mcp.auth.resolver import ResolvedCredentials
from copernicus_mcp.backends.abstract import AbstractBackend, FoundationServices
from copernicus_mcp.backends.cmems import _index_filter
from copernicus_mcp.backends.cmems import catalogue as _catalogue
from copernicus_mcp.backends.cmems import routing as _routing
from copernicus_mcp.backends.cmems._get_manifest import (
    build_manifest,
    iter_data_files,
    read_manifest,
    to_files_envelope,
)
from copernicus_mcp.backends.cmems._index_registry import IndexRegistry
from copernicus_mcp.backends.cmems._index_store import IndexStore
from copernicus_mcp.data_model.provenance import (
    BackendBlock,
    CacheRef,
    CostConsumed,
    DatasetBlock,
    RequestBlock,
    SpatialBlock,
    TemporalBlock,
    Variable,
)
from copernicus_mcp.data_model.schemas_cmems import (
    CmemsGetCoordinatesRequest,
    CmemsGetRequest,
    CmemsListFilesRequest,
    CmemsSearchRequest,
    CmemsSubsetRequest,
)
from copernicus_mcp.errors import (
    AuthError,
    BackendError,
    CacheError,
    CopernicusMcpError,
    CoverageUnavailableError,
    NetworkError,
    NotFoundError,
)
from copernicus_mcp.errors import TimeoutError as CmcpTimeoutError
from copernicus_mcp.errors import ValidationError as CmcpValidationError
from copernicus_mcp.errors.records import build_error_record
from copernicus_mcp.observability.logger import get_logger
from copernicus_mcp.workflow.confirmation import build_size_confirmation

logger = get_logger(__name__)

# T-CMEMS-CAT-003: ``_SEARCH_NAMESPACE`` removed — search now reads
# the bundled catalogue snapshot offline and no longer persists a
# search-results cache (disk reads are 1 ms; the cache was redundant).
_METADATA_NAMESPACE = "metadata"

_REGISTRATION_URL = "https://data.marine.copernicus.eu/register"

_AXIS_FULL_LIMIT_TIME = 5000
_AXIS_FULL_LIMIT_SPATIAL = 10000

# T-CMEMS-GET-001 (cr round-2 MEDIUM): subset-shape discriminator
# for `validate`. If any of these fields is present in the params
# dict, the request is treated as subset and validated against
# `CmemsSubsetRequest`. Otherwise it falls through to
# `CmemsGetRequest`.
#
# Derived dynamically from the schemas so a future subset-only
# field can't drift out of sync with the discriminator (round-1's
# hand-curated list missed 5 optional fields — `file_format`,
# `service`, etc. — reproducing the exact UX trap the
# discriminator was meant to close). A test in
# `tests/unit/test_cmems_routing.py` pins set parity.
_SUBSET_DISCRIMINATOR_FIELDS: frozenset[str] = frozenset(
    set(CmemsSubsetRequest.model_fields) - set(CmemsGetRequest.model_fields)
)


def _not_implemented(method: str) -> BackendError:
    return BackendError(
        f"CmemsBackend.{method} not implemented yet",
        record=build_error_record(
            "BackendError",
            message=f"CmemsBackend.{method} not implemented yet",
            error_subclass="not_implemented",
            recovery_action="report_to_administrator",
        ),
    )


class CmemsBackend(AbstractBackend):
    backend_id = "cmems"

    def __init__(
        self,
        foundation: FoundationServices,
        credentials: ResolvedCredentials | None,
    ) -> None:
        super().__init__(foundation=foundation)
        self._credentials = credentials
        self._auth_adapter: CmemsBasicAuthAdapter | None = (
            CmemsBasicAuthAdapter(credentials) if credentials is not None else None
        )
        # T-039: async submit registry. Instance-scoped (CmemsBackend is a
        # process-wide singleton via BackendRegistry). Persistence is the
        # source of truth for status; this dict only holds live tasks so
        # cancel() can interrupt them. After a server restart the dict is
        # empty and orphaned ``running`` rows reconcile via the existing
        # 60-second staleness guard in ``check_status``.
        self._background_tasks: dict[str, asyncio.Task[Any]] = {}
        # codex/code-reviewer round 1 HIGH (H4): the dedupe-by-cache_key
        # check has a TOCTOU window between lookup and ``record_workflow``
        # insert. A single lock around the async-mode preflight section
        # serialises submits within one process — sufficient because the
        # registry is also process-scoped.
        self._async_submit_lock: asyncio.Lock = asyncio.Lock()
        # T-CMEMS-GET-INDEX-003/004: Layer 2 index store. Reads the
        # bundled index_registry.json on first construction. The
        # Parquet cache lives under foundation.config.storage.cache_directory.
        self._index_registry: IndexRegistry = IndexRegistry()
        self._index_store: IndexStore = IndexStore(
            registry=self._index_registry,
            cache_directory=foundation.config.storage.cache_directory,
            marine_loader=_load_copernicusmarine,
            auth_adapter=self._auth_adapter,
            wrap_exception=_wrap_subset_exception,
            sanitiser=foundation.sanitiser,
        )

    # --- helpers ---------------------------------------------------------

    def _check_credentials_or_raise(self) -> ResolvedCredentials:
        if self._credentials is None:
            raise AuthError(
                "CMEMS credentials are not configured",
                record=build_error_record(
                    "AuthError",
                    message="CMEMS credentials are not configured",
                    recovery_action="configure_credentials",
                    recovery_url=_REGISTRATION_URL,
                ),
            )
        return self._credentials

    # --- protocol --------------------------------------------------------

    async def search(self, params: dict[str, Any]) -> dict[str, Any]:
        """Catalogue search — hybrid offline / live mode.

        ``live=False`` (default): pure-Python over the bundled
        snapshot (``backends.cmems.catalogue``). Fast, credless,
        may be days/weeks stale.

        ``live=True``: call ``copernicusmarine.describe(contains=
        [keyword], product_id=..., disable_progress_bar=True)``
        against the live Marine Service. Fresh, requires
        credentials, ~10 s network round-trip. Useful for
        post-snapshot product updates or to verify freshness for a
        known id.

        Envelope (both modes):
          - ``datasets``: slim records up to ``limit``.
          - ``total_count``: PRE-slice match count.
          - ``mode``: ``"offline"`` or ``"live"`` — explicit
            provenance so consumers can distinguish results.
          - ``catalogue_fetched_at``: snapshot ISO timestamp in
            offline mode; ``None`` in live mode.
        """
        # T-CMEMS-HIER-005: when params include ``product_ids``,
        # ``bbox``, or ``time_range``, route through the
        # hierarchical cards path (offline) which actually filters
        # on those axes. Previously bbox/time_range were rejected
        # with ValidationError per T-CMEMS-CATALOGUE punt.
        if params.get("product_ids") or params.get("bbox") or params.get("time_range"):
            return self._search_in_products(params)
        req = _validate_search(params)
        if req.live:
            return await self._search_live(req)
        return self._search_offline(req)

    async def search_groups(self, params: dict[str, Any]) -> dict[str, Any]:
        """T-CMEMS-HIER-005: shortlist groups for a query.

        Offline-only — reads the bundled ``groups.json``. Returns the
        standard routing envelope (``selected``, ``rejected``,
        ``reason``, ``confidence``, ``fallback_available``)."""
        query = params.get("query")
        if not isinstance(query, str) or not query.strip():
            raise CmcpValidationError(
                "search_groups: 'query' must be a non-empty string",
                record=build_error_record(
                    "ValidationError",
                    message="search_groups: 'query' must be a non-empty string",
                    recovery_action="modify_request_parameters",
                ),
            )
        top_k = params.get("top_k")
        kwargs: dict[str, Any] = {}
        if isinstance(top_k, int) and top_k > 0:
            kwargs["top_k"] = top_k
        envelope = _routing.search_groups(query, **kwargs)
        return self.foundation.sanitiser.sanitise(envelope)  # type: ignore[no-any-return]

    async def search_products(self, params: dict[str, Any]) -> dict[str, Any]:
        """T-CMEMS-HIER-005: filter products by group membership,
        optionally refining by keyword. Offline-only."""
        group_ids = params.get("group_ids")
        if not isinstance(group_ids, list) or not group_ids:
            raise CmcpValidationError(
                "search_products: 'group_ids' must be a non-empty list of strings",
                record=build_error_record(
                    "ValidationError",
                    message=("search_products: 'group_ids' must be a non-empty list of strings"),
                    recovery_action="modify_request_parameters",
                ),
            )
        kwargs: dict[str, Any] = {}
        query = params.get("query")
        if isinstance(query, str) and query.strip():
            kwargs["query"] = query
        top_k = params.get("top_k")
        if isinstance(top_k, int) and top_k > 0:
            kwargs["top_k"] = top_k
        envelope = _routing.search_products(group_ids, **kwargs)
        return self.foundation.sanitiser.sanitise(envelope)  # type: ignore[no-any-return]

    def _search_in_products(self, params: dict[str, Any]) -> dict[str, Any]:
        """Cards-level search routed through ``routing.search_datasets_in_products``.

        Accepts ``product_ids`` (explicit shortlist) and/or ``bbox`` /
        ``time_range`` filters. When ``product_ids`` is absent, the
        full bundled product set is the candidate pool — useful for a
        spatial-only query like "anything intersecting the Black Sea
        bbox".

        Validates input shapes loudly (round-1 review):
          * service_types is still unimplemented and rejected.
          * bbox must be a 4-tuple of floats with min_lon < max_lon
            (antimeridian-crossing rejected per the antimeridian rejection rule).
          * time_range must be a 2-tuple of non-empty strings.
        """
        # codex round-1 HIGH: the cards path bypassed _validate_search,
        # silently dropping unsupported filters. Reject service_types
        # here too with the same recovery message.
        if params.get("service_types"):
            raise CmcpValidationError(
                "CMEMS search filters not yet implemented: ['service_types']",
                record=build_error_record(
                    "ValidationError",
                    message="CMEMS search filters not yet implemented: ['service_types']",
                    recovery_action="modify_request_parameters",
                    next_action_hint=(
                        "Use ``keyword``, ``product_id``, ``product_ids``, "
                        "``bbox``, or ``time_range``; omit service_types."
                    ),
                ),
            )

        product_ids = params.get("product_ids")
        singular_product_id = params.get("product_id")
        # codex round-2 HIGH: ``product_id=A, product_ids=[A,B]`` used
        # to pass the old "in list" check, then the cards path included
        # B too — the agent's intent was ambiguous and the implicit
        # union was a silent footgun. Always reject when both are set.
        if singular_product_id and product_ids:
            raise CmcpValidationError(
                "search: pass either 'product_id' or 'product_ids', not both",
                record=build_error_record(
                    "ValidationError",
                    message="search: pass either 'product_id' or 'product_ids', not both",
                    recovery_action="modify_request_parameters",
                ),
            )
        if product_ids is not None:
            if not isinstance(product_ids, list) or not product_ids:
                raise CmcpValidationError(
                    "search: 'product_ids' must be a non-empty list of strings",
                    record=build_error_record(
                        "ValidationError",
                        message=("search: 'product_ids' must be a non-empty list of strings"),
                        recovery_action="modify_request_parameters",
                    ),
                )
        elif singular_product_id:
            # Promote singular ``product_id`` to a single-element list
            # so it composes with bbox/time_range on the cards path.
            product_ids = [singular_product_id]
        else:
            # All bundled product_ids — bbox/time_range narrow the set.
            product_ids = [p["product_id"] for p in _routing.load_products()]

        kwargs: dict[str, Any] = {}
        query = params.get("query") or params.get("keyword")
        if isinstance(query, str) and query.strip():
            kwargs["query"] = query
        limit = params.get("limit")
        if isinstance(limit, int) and limit > 0:
            kwargs["limit"] = limit
        bbox = params.get("bbox")
        if bbox is not None:
            kwargs["bbox"] = _validate_bbox(bbox)
        time_range = params.get("time_range")
        if time_range is not None:
            kwargs["time_range"] = _validate_time_range(time_range)
        envelope = _routing.search_datasets_in_products(product_ids, **kwargs)
        return self.foundation.sanitiser.sanitise(envelope)  # type: ignore[no-any-return]

    def _search_offline(self, req: CmemsSearchRequest) -> dict[str, Any]:
        """Bundled-snapshot search path. No credentials, no network."""
        total = _catalogue.count_matches(keyword=req.keyword, product_id=req.product_id)
        rows = _catalogue.search(
            keyword=req.keyword,
            product_id=req.product_id,
            limit=req.limit,
        )
        result = {
            "datasets": rows,
            "total_count": total,
            "mode": "offline",
            "catalogue_fetched_at": _catalogue.fetched_at(),
        }
        # Sanitiser pass on the offline envelope (cr round-1 LOW-5):
        # the slim records were already vetted by the refresh
        # script's ``assert_no_credential_leak`` self-check at build
        # time, so this is defence-in-depth for the invariant 2
        # "everything leaving a backend goes through Sanitiser"
        # boundary discipline.
        return self.foundation.sanitiser.sanitise(result)  # type: ignore[no-any-return]

    async def _search_live(self, req: CmemsSearchRequest) -> dict[str, Any]:
        """Live-SDK search path — opt-in via ``live=True``.

        Calls ``copernicusmarine.describe(...)`` and projects the
        raw SDK response into the same slim-record envelope offline
        mode produces (via ``_catalogue_build.slim_marine_record``,
        the same helper the refresh script uses).
        """
        # Re-use the build-side slim helper so live + offline
        # records have byte-identical shape.
        from copernicus_mcp.backends.cmems._catalogue_build import (
            slim_marine_record,
        )

        self._check_credentials_or_raise()
        kwargs: dict[str, Any] = {"disable_progress_bar": True}
        # codex round-1 MEDIUM: offline strips whitespace and treats
        # ``"   "`` as no filter (mirrors ``catalogue._iter_matches``).
        # Live must match — forwarding ``contains=["   "]`` to the SDK
        # behaves as OR / no-op per the empirical SDK note historically
        # carried in ``_describe_kwargs``. Strip first; only forward
        # truthy needles.
        keyword = req.keyword.strip() if req.keyword else ""
        if keyword:
            kwargs["contains"] = [keyword]
        if req.product_id:
            kwargs["product_id"] = req.product_id

        marine = _load_copernicusmarine()
        try:
            raw = await asyncio.to_thread(marine.describe, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise _wrap_marine_exception(marine, exc, "search") from exc

        sdk_response = raw.model_dump(exclude_none=False) if hasattr(raw, "model_dump") else raw
        rows: list[dict[str, Any]] = []
        if isinstance(sdk_response, dict):
            for product in sdk_response.get("products") or []:
                if not isinstance(product, dict):
                    continue
                for dataset in product.get("datasets") or []:
                    if not isinstance(dataset, dict):
                        continue
                    rows.append(slim_marine_record(product, dataset))
        total = len(rows)
        if req.limit is not None and req.limit > 0 and req.limit < total:
            rows = rows[: req.limit]

        result = {
            "datasets": rows,
            "total_count": total,
            "mode": "live",
            # Live results are not snapshot-derived; explicit None so
            # consumers can tell the modes apart without inferring.
            "catalogue_fetched_at": None,
        }
        return self.foundation.sanitiser.sanitise(result)  # type: ignore[no-any-return]

    async def describe(self, identifier: str) -> dict[str, Any]:
        if not isinstance(identifier, str) or not identifier.strip():
            raise CmcpValidationError(
                "dataset identifier must be a non-empty string",
                record=build_error_record(
                    "ValidationError",
                    message="dataset identifier must be a non-empty string",
                    recovery_action="modify_request_parameters",
                ),
            )
        self._check_credentials_or_raise()

        cached = await self._lookup_metadata_cache(identifier)
        if cached is not None:
            return self.foundation.sanitiser.sanitise(cached)  # type: ignore[no-any-return]

        raw = await self._call_describe(dataset_id=identifier)
        product, dataset = _find_dataset(raw, identifier)
        if dataset is None:
            raise NotFoundError(
                f"CMEMS dataset {identifier!r} not found",
                record=build_error_record(
                    "NotFoundError",
                    message=f"CMEMS dataset {identifier!r} not found",
                    recovery_action="modify_request_parameters",
                ),
            )

        result = _map_full_metadata(product, dataset)
        sanitised = self.foundation.sanitiser.sanitise(result)
        await self._persist_metadata_cache(identifier, sanitised)
        return sanitised  # type: ignore[no-any-return]

    async def get_coordinates(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return the dataset's coordinate axes (T-022 second half).

        Validates the request via ``CmemsGetCoordinatesRequest`` and
        delegates to the existing private ``_get_coordinates`` helper.
        See its docstring for axis summarisation thresholds.
        """
        req = _validate_get_coordinates(params)
        return await self._get_coordinates(
            dataset_id=req.dataset_id,
            dataset_version=req.dataset_version,
            service=req.service,
        )

    async def _get_coordinates(
        self,
        dataset_id: str,
        dataset_version: str | None = None,
        service: str | None = None,
    ) -> dict[str, Any]:
        """Return coordinate axes; summarise large axes per plan T-022 step 3."""
        self._check_credentials_or_raise()
        raw = await self._call_describe(dataset_id=dataset_id)
        _, dataset = _find_dataset(raw, dataset_id)
        if dataset is None:
            raise NotFoundError(f"CMEMS dataset {dataset_id!r} not found")
        coords = _extract_coordinates(
            dataset, requested_version=dataset_version, requested_service=service
        )
        # Sanitise uniformly even though axis values are floats/dates — keeps
        # the "everything leaving a backend goes through Sanitiser" invariant.
        return self.foundation.sanitiser.sanitise(coords)  # type: ignore[no-any-return]

    async def _call_describe(self, **kwargs: Any) -> Any:
        marine = _load_copernicusmarine()
        try:
            return await asyncio.to_thread(marine.describe, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise _wrap_marine_exception(marine, exc, "describe") from exc

    async def _lookup_metadata_cache(self, identifier: str) -> dict[str, Any] | None:
        return await _lookup_json_cache(
            self.foundation.persistence,
            namespace=_METADATA_NAMESPACE,
            key=identifier,
            ttl_seconds=self.foundation.config.cache.metadata_ttl_seconds,
        )

    async def _persist_metadata_cache(self, identifier: str, payload: dict[str, Any]) -> None:
        await _persist_json_cache(
            self.foundation.persistence,
            namespace=_METADATA_NAMESPACE,
            key=identifier,
            payload=payload,
        )

    async def validate(self, params: dict[str, Any]) -> dict[str, Any]:
        """Schema-only validation (no toolbox call) per plan §T-026.1.

        T-CMEMS-GET-001: dispatches by request shape. A request
        carrying subset-mandatory fields (``variables`` is the
        cheapest discriminator — required by ``CmemsSubsetRequest``,
        never present on ``CmemsGetRequest``) is validated as a
        subset request; otherwise it is validated as a get
        request. This lets a future ``marine_get_files`` caller
        reach ``validate`` from the orchestrator without a new
        operation name.

        Errors are passed through Sanitiser before returning —
        Pydantic ``msg``/``loc`` fields can echo raw user input,
        which would otherwise surface a credential-shaped value
        in the tool output (codex HIGH on T-026).
        """
        from pydantic import ValidationError as PydValidationError

        clean = {k: v for k, v in params.items() if k != "__options"}
        schema: type[CmemsSubsetRequest] | type[CmemsGetRequest]
        # cr+codex round-1 MEDIUM on PR #96: the original
        # discriminator was "variables" alone, which routed
        # `{"dataset_id": "x"}` (and any subset-shape payload
        # missing only `variables`) to the get path — producing
        # confusing `extra_forbidden` errors instead of a clean
        # "variables required". Widen to any subset-mandatory
        # field so a typo on `variables` still lands on the
        # subset schema.
        if any(field in clean for field in _SUBSET_DISCRIMINATOR_FIELDS):
            schema = CmemsSubsetRequest
        else:
            schema = CmemsGetRequest
        try:
            schema.model_validate(clean)
        except PydValidationError as exc:
            errors = [
                {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
            ]
            return self.foundation.sanitiser.sanitise(  # type: ignore[no-any-return]
                {"valid": False, "errors": errors}
            )
        except CmcpValidationError as exc:
            return self.foundation.sanitiser.sanitise(  # type: ignore[no-any-return]
                {
                    "valid": False,
                    "errors": [{"msg": exc.error_record.message}],
                }
            )
        return {"valid": True}

    async def estimate(self, params: dict[str, Any]) -> dict[str, Any]:
        """Dispatch on params shape (matches ``validate`` discriminator).

        T-CMEMS-GET-004: a get-shape request runs ``marine.get(dry_run=True)``;
        a subset-shape request keeps the existing ``marine.subset(dry_run=True)``
        path.
        """
        clean = (
            {k: v for k, v in params.items() if k != "__options"}
            if isinstance(params, dict)
            else params
        )
        if any(field in clean for field in _SUBSET_DISCRIMINATOR_FIELDS):
            return await self._estimate_subset(params)
        return await self._estimate_get(params)

    async def _estimate_subset(self, params: dict[str, Any]) -> dict[str, Any]:
        req = _validate_subset(params)
        self._check_credentials_or_raise()

        marine = _load_copernicusmarine()
        kwargs = _subset_kwargs(req, dry_run=True)
        if self._auth_adapter is not None:
            user, password = self._auth_adapter.get_username_password()
            kwargs["username"] = user
            kwargs["password"] = password
        timeout = self.foundation.config.budget.cmems_estimate_timeout_seconds

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(marine.subset, **kwargs),
                timeout=timeout,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:  # asyncio.TimeoutError is the builtin alias.
            # Per the project conventions gotcha #8: the underlying toolbox thread cannot be
            # cancelled and may run to completion in the background. Accepted.
            raise CmcpTimeoutError(f"CMEMS estimate timed out after {timeout}s") from exc
        except Exception as exc:
            raise _wrap_subset_exception(marine, exc, "estimate") from exc

        result = _map_estimate_response(response)
        advisory = await self._build_coverage_advisory(req)
        if advisory is not None:
            result["coverage_advisory"] = advisory
        return self.foundation.sanitiser.sanitise(result)  # type: ignore[no-any-return]

    async def _estimate_get(self, params: dict[str, Any]) -> dict[str, Any]:
        """``marine.get(dry_run=True)`` size estimate.

        The SDK's dry-run shape for ``get`` is not stable across
        versions: some return a precise ``total_size`` / ``file_size``,
        others return only a status. ``_map_get_estimate_response``
        is field-tolerant and falls back to
        ``epistemic_status="approximate"``, which the gate in
        ``get_files`` treats as always-gate per sub-plan D3.
        """
        req = _validate_get(params)
        self._check_credentials_or_raise()

        marine = _load_copernicusmarine()
        kwargs = _get_kwargs(req)
        kwargs["dry_run"] = True
        if self._auth_adapter is not None:
            user, password = self._auth_adapter.get_username_password()
            kwargs["username"] = user
            kwargs["password"] = password
        timeout = self.foundation.config.budget.cmems_estimate_timeout_seconds

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(marine.get, **kwargs),
                timeout=timeout,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            raise CmcpTimeoutError(
                f"CMEMS get estimate timed out after {timeout}s"
            ) from exc
        except Exception as exc:
            # cr round-1 MEDIUM: pass ``op="get"`` so the
            # WrongFormatRequested hint suggests
            # ``marine_subset_dataset`` (not ``marine_get_files``,
            # which is the caller). ``_estimate_get`` is the get-
            # shape estimate path; the get-aware branch is correct
            # here.
            raise _wrap_subset_exception(marine, exc, "get") from exc

        result = _map_get_estimate_response(response)
        return self.foundation.sanitiser.sanitise(result)  # type: ignore[no-any-return]

    async def _build_coverage_advisory(self, req: Any) -> dict[str, Any] | None:
        """Compare user bbox to dataset's spatial_extent.

        Returns an advisory only when bbox does NOT effectively match the
        extent. Emitted on every ``estimate`` so the LLM agent sees coverage
        mismatch before downloading. Real session that motivated this: agent
        asked for "all of Mediterranean" with lon[-6, 36.5]; actual dataset
        extent is lon[-17.29, 36.29] — the Atlantic buffer west of Gibraltar
        was silently dropped.

        Per-axis comparison: each axis is classified as exact / inside (user
        narrower) / exceeds (user wider) / disjoint (no overlap). The
        aggregate verdict is mapped to one of four statuses:
          - all-axes exact (within tolerance)  → no advisory (omit field)
          - any-axis disjoint                  → ``bbox_disjoint_from_dataset_extent``
          - all axes inside (some exact)       → ``bbox_inside_dataset_extent``
          - all axes exceed (some exact)       → ``bbox_exceeds_dataset_extent``
          - mix of inside + exceeds (overlap)  → ``bbox_mismatch_dataset_extent``

        Failures / missing metadata return ``None`` so estimate stays
        usable for datasets whose describe we can't probe (review L3 note:
        when called from submit, this holds the async-submit lock during
        the describe — cache hits dominate via the persisted describe
        cache, so in practice the network call is rare).

        Review L5: ``force_refresh`` on the parent submit is intentionally
        not propagated here — stale dataset extents change on the order of
        years, not minutes, so the cost of a fresh describe per submit
        outweighs the rare staleness risk.
        """
        try:
            meta = await self.describe(req.dataset_id)
        except Exception:
            return None
        extent = meta.get("spatial_extent") if isinstance(meta, dict) else None
        if not isinstance(extent, dict):
            return None
        try:
            ds_min_lon = float(extent["min_lon"])
            ds_max_lon = float(extent["max_lon"])
            ds_min_lat = float(extent["min_lat"])
            ds_max_lat = float(extent["max_lat"])
        except (KeyError, TypeError, ValueError):
            return None
        u_min_lon = float(req.minimum_longitude)
        u_max_lon = float(req.maximum_longitude)
        u_min_lat = float(req.minimum_latitude)
        u_max_lat = float(req.maximum_latitude)

        lon_state = _axis_state(u_min_lon, u_max_lon, ds_min_lon, ds_max_lon)
        lat_state = _axis_state(u_min_lat, u_max_lat, ds_min_lat, ds_max_lat)
        states = (lon_state, lat_state)
        ext_dict = {
            "min_lon": ds_min_lon,
            "max_lon": ds_max_lon,
            "min_lat": ds_min_lat,
            "max_lat": ds_max_lat,
        }

        if all(s == "exact" for s in states):
            return None

        if any(s == "disjoint" for s in states):
            return {
                "status": "bbox_disjoint_from_dataset_extent",
                "dataset_extent": ext_dict,
                "advisory_message": (
                    "Your bbox does not overlap the dataset's spatial "
                    "extent on at least one axis. The download will "
                    "produce an empty or clamped-to-empty result. "
                    "Dataset extent: "
                    f"lon[{ds_min_lon}, {ds_max_lon}] x "
                    f"lat[{ds_min_lat}, {ds_max_lat}]."
                ),
            }

        if all(s in ("exact", "inside") for s in states):
            return {
                "status": "bbox_inside_dataset_extent",
                "dataset_extent": ext_dict,
                "advisory_message": (
                    "Your bbox is inside the dataset's spatial extent but "
                    "does not cover it fully. Dataset extent: "
                    f"lon[{ds_min_lon}, {ds_max_lon}] x "
                    f"lat[{ds_min_lat}, {ds_max_lat}]. Widen the bbox "
                    "only if you intended to retrieve the full extent."
                ),
            }

        if all(s in ("exact", "exceeds") for s in states):
            return {
                "status": "bbox_exceeds_dataset_extent",
                "dataset_extent": ext_dict,
                "advisory_message": (
                    "Your bbox extends beyond the dataset's spatial extent. "
                    "The toolbox will clamp the subset to the dataset's "
                    "coordinates. Dataset extent: "
                    f"lon[{ds_min_lon}, {ds_max_lon}] x "
                    f"lat[{ds_min_lat}, {ds_max_lat}]."
                ),
            }

        # Mixed: at least one axis inside, another exceeds — review M2.
        # Build a per-axis breakdown so both behaviors are visible.
        return {
            "status": "bbox_mismatch_dataset_extent",
            "dataset_extent": ext_dict,
            "advisory_message": (
                "Your bbox does not align with the dataset's spatial "
                f"extent: longitude is {lon_state} (dataset "
                f"[{ds_min_lon}, {ds_max_lon}], request "
                f"[{u_min_lon}, {u_max_lon}]); latitude is {lat_state} "
                f"(dataset [{ds_min_lat}, {ds_max_lat}], request "
                f"[{u_min_lat}, {u_max_lat}]). The toolbox will clamp "
                "exceed axes and you will not retrieve data outside "
                "narrowed axes."
            ),
        }

    async def submit(self, params: dict[str, Any]) -> dict[str, Any]:
        # TODO(T-031): orchestrator plumbs options under ``__options``; T-031
        # will give us a properly typed RequestOptions envelope.
        # Non-mutating read — ``_validate_subset`` strips the key before
        # Pydantic; mutating the caller's dict was a footgun.
        options = dict(params.get("__options") or {}) if isinstance(params, dict) else {}
        req = _validate_subset(params)
        self._check_credentials_or_raise()
        # H1: sanitise params at the boundary so any credential-shaped key the
        # caller forgot to strip cannot reach SQLite or the provenance sidecar.
        safe_params = self.foundation.sanitiser.sanitise(params)
        cache_key = self.foundation.data_model.cache_key_for_subset(req)
        async_mode = bool(options.get("async_mode"))

        # --- Idempotency: file-cache hit -----------------------------------
        if not options.get("force_refresh"):
            # codex round 2 HIGH (originally fixed by dropping ``file:``
            # prefix; codex round 3 HIGH reverted): keep the ``file:``
            # prefix on the storage key for backwards compat with v0.1.2
            # caches. The URL ``copernicus://files/{cache_key}`` resolves
            # via ``_file_resource`` which prepends the prefix internally
            # — see ``server.py``. ``_cache_storage_key()`` is the single
            # source of truth.
            existing_path = await self.foundation.cache.lookup_file(_cache_storage_key(cache_key))
            if existing_path is not None:
                # Reuse the original workflow row if we can find one.
                existing_wf = await self.foundation.persistence.lookup_workflow_by_cache_key(
                    cache_key
                )
                req_id = existing_wf["request_id"] if existing_wf is not None else _new_request_id()
                return self.foundation.sanitiser.sanitise(  # type: ignore[no-any-return]
                    _success_response(
                        request_id=req_id,
                        cache_key=cache_key,
                        filepath=existing_path,
                        cache_hit=True,
                        is_existing=True,
                        provenance_ref=None,
                    )
                )

        # codex/code-reviewer round 1 H4: serialise the async-mode preflight
        # (dedupe lookup → record_workflow INSERT → task spawn) under a
        # per-backend lock so two concurrent ``gather``'d submits with the
        # same cache_key cannot both pass the lookup. Sync submits skip the
        # lock entirely. ``contextlib.nullcontext`` avoids branching in the
        # caller.
        critical_section: contextlib.AbstractAsyncContextManager[Any] = (
            self._async_submit_lock if async_mode else contextlib.nullcontext()
        )

        async with critical_section:
            # --- Async mode: dedupe against an in-flight submit ---------------
            # Two callers with identical params racing should observe a single
            # background task. ``lookup_workflow_by_cache_key`` returns the most
            # recent row; if that row is queued/running we reuse its request_id
            # rather than spawning a duplicate.
            if async_mode and not options.get("force_refresh"):
                inflight = await self.foundation.persistence.lookup_workflow_by_cache_key(cache_key)
                if inflight is not None and inflight["status"] in (
                    "queued",
                    "running",
                ):
                    return self.foundation.sanitiser.sanitise(  # type: ignore[no-any-return]
                        _running_response(
                            request_id=inflight["request_id"],
                            cache_key=cache_key,
                        )
                    )

            return await self._submit_after_dedupe(
                params=params,
                req=req,
                safe_params=safe_params,
                cache_key=cache_key,
                options=options,
                async_mode=async_mode,
            )

    async def _submit_after_dedupe(
        self,
        *,
        params: dict[str, Any],
        req: CmemsSubsetRequest,
        safe_params: dict[str, Any],
        cache_key: str,
        options: dict[str, Any],
        async_mode: bool,
    ) -> dict[str, Any]:
        """Estimate + confirmation gate + workflow row + dispatch.

        Extracted so the async-mode lock in ``submit`` can wrap the entire
        dedupe-to-spawn critical section without duplicating logic.
        """
        # --- Pre-flight estimate + confirmation gate -----------------------
        estimate = await self.estimate(params)
        threshold_bytes = int(
            self.foundation.config.budget.cmems_per_request_size_warning_gb * 1_000_000_000
        )
        # codex T-023 MEDIUM: an ``approximate`` estimate (toolbox didn't
        # report sizes) must NOT silently bypass confirmation just because
        # the bytes-count defaults to 0 — fall back to "ask the user".
        size_unknown = estimate.get("epistemic_status") == "approximate"
        size_over_threshold = estimate["estimated_size_bytes"] > threshold_bytes
        if not options.get("confirmed") and (size_over_threshold or size_unknown):
            raise build_size_confirmation(
                tool_name="marine_subset_dataset",
                backend="cmems",
                estimated_size_bytes=estimate["estimated_size_bytes"],
                threshold_bytes=threshold_bytes,
                source="config.budget.cmems_per_request_size_warning_gb",
            )

        # --- Workflow row + staging dir ------------------------------------
        request_id = _new_request_id()
        now = _iso_now_seconds()
        request_json = json.dumps(safe_params, sort_keys=True, default=str)

        await self.foundation.persistence.record_workflow(
            {
                "request_id": request_id,
                "backend_id": "cmems",
                "operation": "submit",
                "status": "queued",
                "cache_key": cache_key,
                "request_json": request_json,
                "response_json": None,
                "error_record_json": None,
                "created_at": now,
                "updated_at": now,
            }
        )
        await self.foundation.persistence.update_workflow_status(request_id, "running")

        # codex round 2 MEDIUM (M1): the row is already ``running`` — any
        # raise from staging setup or kwargs prep would leak that state.
        # Wrap the prep block so a setup failure ends the row in ``failed``
        # before the exception propagates to the caller.
        try:
            # Toolbox writes to a temp staging dir; cache.store_file then
            # moves the file into the canonical cache zone. Writing directly
            # into the cache zone would conflict with store_file's
            # "remove old entry first" idempotency on force_refresh.
            # codex HIGH H1: dataset_id is user input — never let it
            # influence the filesystem path. Slug to a safe basename and rely
            # on the cache_key hash for uniqueness.
            zone = self.foundation.cache.cache_zone_for("cmems")
            staging = zone / ".staging" / uuid.uuid4().hex
            staging.mkdir(parents=True, exist_ok=False)
            short_hash = cache_key.rsplit(":", 1)[-1]
            safe_id = _slug_dataset_id(req.dataset_id)
            filename = f"{safe_id}_{short_hash}.nc"
            target_path = staging / filename
            # Defence-in-depth: ensure target stays inside staging.
            # Both sides must resolve so a symlinked cache_directory
            # (e.g. macOS ``/tmp`` → ``/private/tmp``) doesn't trigger
            # a false positive — ``target_path.resolve()`` would follow
            # the symlink while an un-resolved ``staging`` would not.
            if not target_path.resolve().is_relative_to(staging.resolve()):
                raise BackendError(
                    "internal: target path escaped staging directory",
                    record=build_error_record(
                        "BackendError",
                        message=("internal: target path escaped staging directory"),
                        error_subclass="path_escape",
                        recovery_action="report_to_administrator",
                    ),
                )

            marine = _load_copernicusmarine()
            kwargs = _subset_kwargs(req, dry_run=False)
            kwargs["output_directory"] = str(staging)
            kwargs["output_filename"] = filename
            # codex HIGH H2: pass resolved credentials explicitly. The toolbox
            # otherwise relies on ambient state and silently ignores creds we
            # accepted from secret_manager / explicit override.
            if self._auth_adapter is not None:
                user, password = self._auth_adapter.get_username_password()
                kwargs["username"] = user
                kwargs["password"] = password
        except CopernicusMcpError as exc:
            error_record_json = self._serialise_error_record(exc)
            with contextlib.suppress(Exception):
                await self.foundation.persistence.update_workflow_error(
                    request_id, "failed", error_record_json
                )
            raise
        except Exception as exc:
            wrapped = BackendError(
                f"staging-dir setup failed: {exc!s}",
                record=build_error_record(
                    "BackendError",
                    message="staging-dir setup failed",
                    error_subclass="staging_setup_failure",
                    recovery_action="report_to_administrator",
                ),
            )
            error_record_json = self._serialise_error_record(wrapped)
            with contextlib.suppress(Exception):
                await self.foundation.persistence.update_workflow_error(
                    request_id, "failed", error_record_json
                )
            raise wrapped from exc

        # --- Dispatch: sync vs async --------------------------------------
        if async_mode:
            task = asyncio.create_task(
                self._async_subset_runner(
                    request_id=request_id,
                    cache_key=cache_key,
                    marine=marine,
                    kwargs=kwargs,
                    target_path=target_path,
                    staging=staging,
                    req=req,
                    safe_params=safe_params,
                    options=options,
                    estimate=estimate,
                    started_at=now,
                )
            )
            self._background_tasks[request_id] = task

            def _drop(t: asyncio.Task[Any], rid: str = request_id) -> None:
                self._background_tasks.pop(rid, None)

            task.add_done_callback(_drop)
            return self.foundation.sanitiser.sanitise(  # type: ignore[no-any-return]
                _running_response(request_id=request_id, cache_key=cache_key)
            )

        return await self._execute_subset_download(
            request_id=request_id,
            cache_key=cache_key,
            marine=marine,
            kwargs=kwargs,
            target_path=target_path,
            staging=staging,
            req=req,
            safe_params=safe_params,
            options=options,
            estimate=estimate,
            started_at=now,
        )

    async def _async_subset_runner(self, **kwargs: Any) -> None:
        """Background-task wrapper around ``_execute_subset_download``.

        Re-raises ``CancelledError`` so the asyncio Task settles in cancelled
        state. Swallows other exceptions (already persisted on the workflow
        row by ``_execute_subset_download``'s error handler) so the asyncio
        runtime does not log "Task exception was never retrieved" — the
        polling caller observes the failure via ``check_status``.
        """
        request_id: str = kwargs["request_id"]
        try:
            await self._execute_subset_download(**kwargs)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "async submit failed; error_record_json persisted on workflow row",
                extra={"request_id": request_id},
                exc_info=True,
            )

    async def _execute_subset_download(
        self,
        *,
        request_id: str,
        cache_key: str,
        marine: Any,
        kwargs: dict[str, Any],
        target_path: Path,
        staging: Path,
        req: CmemsSubsetRequest,
        safe_params: dict[str, Any],
        options: dict[str, Any],
        estimate: dict[str, Any],
        started_at: str,
    ) -> dict[str, Any]:
        """Run the toolbox call, then store + record provenance + finalize.

        Shared by the sync ``submit`` path and the async background task.
        Error handlers update the workflow row before re-raising so both
        callers leave a consistent persistence state behind.

        codex/code-reviewer round 1 HIGH: a CancelledError or generic
        Exception arriving after ``marine.subset`` succeeds (e.g. during
        ``cache.store_file`` or ``provenance``) used to leave the row
        stuck ``running``. The full body now lives inside one try/except
        envelope so every termination path settles the row, and status
        updates inside the handlers are ``asyncio.shield``'d so a second
        cancellation cannot interrupt them.
        """
        try:
            await asyncio.to_thread(marine.subset, **kwargs)

            if not target_path.exists():
                raise BackendError(
                    f"CMEMS toolbox reported success but file is missing at {target_path}",
                    record=build_error_record(
                        "BackendError",
                        message=("CMEMS toolbox reported success but file is missing"),
                        error_subclass="missing_output_file",
                        recovery_action="report_to_administrator",
                    ),
                )

            # Register file in cache index.
            cache_path = await self.foundation.cache.store_file(
                cache_key=_cache_storage_key(cache_key),
                source_path=target_path,
                backend_id="cmems",
                content_type="application/x-netcdf",
            )
            # M5: clean up the empty staging dir so it doesn't accumulate.
            with contextlib.suppress(OSError):
                shutil.rmtree(staging, ignore_errors=True)

            # Provenance — a failure here is non-fatal (logged warning).
            provenance_ref: str | None = None
            try:
                record_id = await self.foundation.provenance.record_successful_retrieve(
                    backend=BackendBlock(
                        id="cmems",
                        provider="Mercator Ocean International",
                        endpoint_url="https://my.cmems-du.eu",
                        api_version="marine-data-store-v1",
                    ),
                    dataset=DatasetBlock(
                        dataset_id=req.dataset_id,
                        dataset_version=req.dataset_version,
                        dataset_part=req.dataset_part,
                        service_name=estimate.get("service_used"),
                    ),
                    request=RequestBlock(
                        operation="submit",
                        submitted_at=started_at,
                        started_at=started_at,
                        finished_at=_iso_now_seconds(),
                        user_request=safe_params,
                        normalized_request=req.model_dump(),
                        options_applied=options,
                    ),
                    spatial=SpatialBlock(
                        native_crs="EPSG:4326",
                        output_crs="EPSG:4326",
                        bbox_epsg_4326=[
                            req.minimum_longitude,
                            req.minimum_latitude,
                            req.maximum_longitude,
                            req.maximum_latitude,
                        ],
                    ),
                    temporal=TemporalBlock(
                        start_datetime=req.start_datetime,
                        end_datetime=req.end_datetime,
                    ),
                    variables=[Variable(name=v) for v in req.variables],
                    files=[cache_path],
                    cost_consumed=CostConsumed(
                        type=estimate.get("type", "free"),
                        advisory_message=estimate.get("advisory_message"),
                    ),
                    source_urls=[],
                    cache=CacheRef(cache_key=cache_key, cache_hit=False),
                    workflow_request_id=request_id,
                )
                provenance_ref = f"copernicus://provenance/{record_id}"
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("provenance recording failed", exc_info=True)

            # Final status flip is shielded — once the file is in the cache
            # and provenance is logged, a late cancel must not leave the
            # row ``running``. The cache file/provenance commitment is the
            # logical commit point; the SQL update is just metadata.
            await asyncio.shield(
                self.foundation.persistence.update_workflow_status(request_id, "successful")
            )

            return self.foundation.sanitiser.sanitise(  # type: ignore[no-any-return]
                _success_response(
                    request_id=request_id,
                    cache_key=cache_key,
                    filepath=cache_path,
                    cache_hit=False,
                    is_existing=False,
                    provenance_ref=provenance_ref,
                )
            )

        except asyncio.CancelledError:
            # Shielded so a second cancellation cannot interrupt the row
            # update — a CancelledError raised by the shielded coroutine
            # itself is suppressed via the BaseException catch.
            with contextlib.suppress(BaseException):
                await asyncio.shield(
                    self.foundation.persistence.update_workflow_status(request_id, "cancelled")
                )
            _safe_unlink(target_path)
            raise
        except CopernicusMcpError as exc:
            # Already canonical (e.g. ``BackendError`` from
            # missing_output_file above). Persist the record before
            # re-raising so async callers can read it via ``check_status``.
            error_record_json = self._serialise_error_record(exc)
            with contextlib.suppress(Exception):
                await asyncio.shield(
                    self.foundation.persistence.update_workflow_error(
                        request_id, "failed", error_record_json
                    )
                )
            _safe_unlink(target_path)
            raise
        except Exception as exc:
            # Toolbox or post-toolbox raw exception → wrap into our
            # taxonomy then persist + re-raise. ``_wrap_subset_exception``
            # already sanitises the message; ``_serialise_error_record``
            # walks the full record for defence in depth.
            wrapped = _wrap_subset_exception(marine, exc, "submit")
            error_record_json = self._serialise_error_record(wrapped)
            with contextlib.suppress(Exception):
                await asyncio.shield(
                    self.foundation.persistence.update_workflow_error(
                        request_id, "failed", error_record_json
                    )
                )
            _safe_unlink(target_path)
            raise wrapped from exc

    # ------------------------------------------------------------------
    # T-CMEMS-GET-003: ``get_files`` happy path
    # ------------------------------------------------------------------

    async def list_files(self, params: dict[str, Any]) -> dict[str, Any]:
        """Layer 2 index-driven file listing (T-CMEMS-GET-INDEX-004).

        Validates the request, loads the parsed-index DataFrame via
        ``IndexStore.load`` (offline cache hit or fresh SDK round-trip),
        applies the four AND-combined filter primitives, computes the
        size aggregation, applies caller-imposed ``limit`` truncation
        with deterministic ordering, and emits the canonical envelope.
        """
        req = _validate_list_files(params)
        self._check_credentials_or_raise()

        df, mode = await self._index_store.load(req.dataset_id)
        total_in_index = int(len(df))

        filtered = _index_filter.apply_filters(
            df,
            bbox=req.bbox,
            time_range=req.time_range,
            variables=req.variables,
            platform_types=req.platform_types,
        )
        matched_uncapped = int(len(filtered))

        # Deterministic ordering for limit truncation.
        if matched_uncapped:
            filtered = filtered.sort_values("file_path").reset_index(drop=True)

        truncated = False
        if req.limit is not None and matched_uncapped > req.limit:
            filtered = filtered.head(req.limit)
            truncated = True

        size_series = filtered["size_bytes"].dropna()
        total_size_bytes_known = int(size_series.sum()) if not size_series.empty else 0
        rows_with_unknown_size = int(filtered["size_bytes"].isna().sum())

        files = [
            self._row_to_dict(row) for _, row in filtered.iterrows()
        ]

        filters_applied: dict[str, Any] = {}
        if req.bbox is not None:
            filters_applied["bbox"] = list(req.bbox)
        if req.time_range is not None:
            filters_applied["time_range"] = list(req.time_range)
        if req.variables is not None:
            filters_applied["variables"] = list(req.variables)
        if req.platform_types is not None:
            filters_applied["platform_types"] = list(req.platform_types)

        index_fetched_at = await self._index_store.fetched_at(req.dataset_id)

        envelope: dict[str, Any] = {
            "files": files,
            "matched_count": len(files),
            "matched_count_uncapped": matched_uncapped,
            "truncated": truncated,
            "total_count_in_index": total_in_index,
            "total_size_bytes_known": total_size_bytes_known,
            "rows_with_unknown_size": rows_with_unknown_size,
            "filters_applied": filters_applied,
            "index_fetched_at": (
                index_fetched_at.strftime("%Y-%m-%dT%H:%M:%SZ")
                if index_fetched_at is not None
                else None
            ),
            "mode": mode,
        }
        return self.foundation.sanitiser.sanitise(envelope)  # type: ignore[no-any-return]

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        """Convert an IndexRow DataFrame row to a JSON-serialisable dict."""
        def _iso(ts: Any) -> str | None:
            if ts is None or (hasattr(ts, "to_pydatetime") and pd.isna(ts)):
                return None
            return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

        size = row["size_bytes"]
        if pd.isna(size):
            size_out: int | None = None
        else:
            size_out = int(size)
        variables = row["variables"]
        if variables is None or (
            isinstance(variables, float) and pd.isna(variables)
        ):
            vars_out: list[str] | None = None
        else:
            vars_out = list(variables)
        return {
            "file_path": str(row["file_path"]),
            "lon_min": float(row["lon_min"]),
            "lon_max": float(row["lon_max"]),
            "lat_min": float(row["lat_min"]),
            "lat_max": float(row["lat_max"]),
            "time_start": _iso(row["time_start"]),
            "time_end": _iso(row["time_end"]),
            "platform_type": (
                None if row["platform_type"] is None else str(row["platform_type"])
            ),
            "variables": vars_out,
            "size_bytes": size_out,
        }

    async def get_files(self, params: dict[str, Any]) -> dict[str, Any]:
        """Native-file retrieval via ``copernicusmarine.get``.

        Happy-path-only for T-CMEMS-GET-003:
        - validate + check creds + cache-key derivation;
        - cache-hit short-circuit;
        - serialise concurrent submits via the shared
          ``_async_submit_lock``;
        - one ``marine.get`` call (sync, wrapped in ``asyncio.to_thread``);
        - register the produced bundle as a manifest cache entry;
        - record one provenance row;
        - workflow row flips to ``successful``.

        Out of scope here (later sub-plan tasks):
        - confirmation gate (T-CMEMS-GET-004),
        - cancellation parity + commit-point shielding (T-CMEMS-GET-005).
        """
        options = (
            dict(params.get("__options") or {}) if isinstance(params, dict) else {}
        )
        req = _validate_get(params)
        self._check_credentials_or_raise()
        safe_params = self.foundation.sanitiser.sanitise(params)
        cache_key = self.foundation.data_model.cache_key_for_get(req)

        # --- Idempotency: manifest-cache hit (fast path, no lock) -----
        if not options.get("force_refresh"):
            hit = await self._build_get_cache_hit_response(cache_key)
            if hit is not None:
                return hit

        # Serialise concurrent identical submits behind the per-backend
        # lock; the lock-holder records the workflow row, the second
        # caller observes the freshly-populated cache on its post-lock
        # re-check.
        async with self._async_submit_lock:
            if not options.get("force_refresh"):
                hit = await self._build_get_cache_hit_response(cache_key)
                if hit is not None:
                    return hit

            # T-CMEMS-GET-004: confirmation gate. Mirrors
            # ``_submit_after_dedupe`` for subset. ``approximate``
            # epistemic status always gates per sub-plan D3 even at
            # 0 bytes (SDK didn't surface a precise size → we can't
            # reason about cost).
            estimate = await self.estimate(params)
            threshold_bytes = int(
                self.foundation.config.budget.cmems_per_request_size_warning_gb
                * 1_000_000_000
            )
            size_unknown = estimate.get("epistemic_status") == "approximate"
            size_over_threshold = (
                estimate["estimated_size_bytes"] > threshold_bytes
            )
            if not options.get("confirmed") and (
                size_over_threshold or size_unknown
            ):
                raise build_size_confirmation(
                    tool_name="marine_get_files",
                    backend="cmems",
                    estimated_size_bytes=estimate["estimated_size_bytes"],
                    threshold_bytes=threshold_bytes,
                    source="config.budget.cmems_per_request_size_warning_gb",
                )

            return await self._execute_get_download(
                req=req,
                safe_params=safe_params,
                cache_key=cache_key,
                options=options,
            )

    async def _build_get_cache_hit_response(
        self, cache_key: str
    ) -> dict[str, Any] | None:
        existing_manifest = await self.foundation.cache.lookup_file(
            _cache_storage_key(cache_key)
        )
        if existing_manifest is None:
            return None
        # cr+codex round-1 MEDIUM: ``read_manifest`` propagates
        # ``json.JSONDecodeError`` for a corrupt manifest. Treat that
        # the same as a missing file — invalidate the bundle so the
        # download path repopulates, instead of wedging the cache_key.
        try:
            manifest = read_manifest(existing_manifest.parent)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "corrupt or unreadable manifest at cache-hit; invalidating",
                extra={
                    "path": str(existing_manifest),
                    "error_class": type(exc).__name__,
                },
            )
            await self.foundation.cache.invalidate(_cache_storage_key(cache_key))
            return None
        if manifest is None:
            # Cache entry without a readable manifest — treat as
            # missing and let the download path repopulate.
            await self.foundation.cache.invalidate(_cache_storage_key(cache_key))
            return None
        envelope = to_files_envelope(
            manifest=manifest, data_dir=existing_manifest.parent
        )
        envelope["provenance"] = {}  # parity with subset cache-hit shape
        existing_wf = await self.foundation.persistence.lookup_workflow_by_cache_key(
            cache_key
        )
        req_id = (
            existing_wf["request_id"] if existing_wf is not None else _new_request_id()
        )
        return self.foundation.sanitiser.sanitise(  # type: ignore[no-any-return]
            _success_response_get_files(
                request_id=req_id,
                cache_key=cache_key,
                files_envelope=envelope,
                cache_hit=True,
                is_existing=True,
            )
        )

    async def _execute_get_download(
        self,
        *,
        req: CmemsGetRequest,
        safe_params: dict[str, Any],
        cache_key: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        # cr round-1 LOW + round-2 MEDIUM: load the SDK BEFORE the
        # workflow row is recorded. If the SDK is missing (``cmems``
        # extra not installed) the loader raises here and no
        # ``running`` row is left behind. After the row is recorded,
        # the captured ``marine`` is reused by the except handler so
        # a second lookup can't lose the original exception.
        marine = _load_copernicusmarine()

        request_id = _new_request_id()
        submitted_at = _iso_now_seconds()
        request_json = json.dumps(safe_params, sort_keys=True, default=str)
        await self.foundation.persistence.record_workflow(
            {
                "request_id": request_id,
                "backend_id": "cmems",
                "operation": "get",
                "status": "queued",
                "cache_key": cache_key,
                "request_json": request_json,
                "response_json": None,
                "error_record_json": None,
                "created_at": submitted_at,
                "updated_at": submitted_at,
            }
        )
        await self.foundation.persistence.update_workflow_status(request_id, "running")
        started_at = _iso_now_seconds()

        zone = self.foundation.cache.cache_zone_for("cmems")
        # Bundle directory name = SHA-256 hex digest of the cache_key
        # (the cache_key itself contains the raw dataset_id and is
        # therefore unsafe as a filesystem basename). The cache
        # layer's path-safety check (T-CMEMS-GET-002) verifies the
        # final location is inside ``cache_directory``.
        bundle_basename = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:32]
        bundle_dir = zone / bundle_basename
        staging = zone / ".staging" / uuid.uuid4().hex

        # T-CMEMS-GET-005 round-2 LOW: hoist ``commit_state`` so the
        # except branches read it unconditionally (no
        # ``locals().get(...)`` idiom needed). Flips to ``True``
        # inside ``_commit_get_bundle`` the instant ``store_manifest``
        # returns.
        commit_state: dict[str, bool] = {"committed": False}

        try:
            staging.mkdir(parents=True, exist_ok=False)
            kwargs = _get_kwargs(req)
            kwargs["output_directory"] = str(staging)
            if self._auth_adapter is not None:
                user, password = self._auth_adapter.get_username_password()
                kwargs["username"] = user
                kwargs["password"] = password

            await asyncio.to_thread(marine.get, **kwargs)

            # cr round-1 MEDIUM: use the same skip rules as
            # ``build_manifest`` (dotfiles + manifest.json + symlinks)
            # so an SDK that drops a session-state dotfile doesn't
            # bypass the empty-match guard.
            data_files = iter_data_files(staging)
            if not data_files:
                msg = (
                    "CMEMS get produced no files; the selection "
                    "(filter/regex/file_list) matched nothing"
                )
                raise NotFoundError(
                    msg,
                    record=build_error_record(
                        "NotFoundError",
                        message=msg,
                        recovery_action="modify_request_parameters",
                        next_action_hint=(
                            "Loosen filter/regex, verify the dataset has "
                            "files for the requested period, or call "
                            "marine_describe_dataset to inspect the file "
                            "tree."
                        ),
                    ),
                )

            # cr round-1 HIGH: when ``force_refresh`` re-runs over a
            # pre-existing bundle, our deterministic ``bundle_dir``
            # already holds the previous bundle. A naive
            # ``rmtree(bundle_dir)`` here would race with
            # ``store_manifest``'s own idempotent-overwrite
            # ``rmtree`` (which fires from inside the cache lock on
            # ``record_cache_entry``), so the new bundle would be
            # registered against a directory the cache layer is about
            # to destroy. We invalidate the prior cache entry first
            # (the cache layer's own ``rmtree`` of the old data_dir
            # runs under its lock), then rmtree any leftover orphan
            # (e.g. from a crashed prior run that placed the dir but
            # never registered the entry), then move the new bundle
            # in. ``store_manifest`` now sees no existing entry and
            # skips the inner rmtree.
            await self.foundation.cache.invalidate(_cache_storage_key(cache_key))
            if bundle_dir.exists():
                shutil.rmtree(bundle_dir)
            bundle_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging), str(bundle_dir))
            staging = bundle_dir  # for cleanup-on-error symmetry

            manifest_path = build_manifest(
                directory=bundle_dir,
                cache_key=cache_key,
                dataset_id=req.dataset_id,
                dataset_version=req.dataset_version,
            )

            # T-CMEMS-GET-005: commit point. The ``store_manifest``
            # call is the moment the bundle becomes externally
            # visible (cache row published). Cancellation BEFORE
            # this point rolls back; cancellation / failure AFTER
            # must NOT delete the committed bundle.
            #
            # Sub-plan codex round-3 LOW explicitly warned that
            # shielding only the row update would still let a
            # cancel interleave between the cache entry and the
            # provenance record. We shield the whole block
            # (store_manifest + provenance + workflow finalize)
            # via an explicit Task so the cancel handler can drain
            # it before deciding rollback.
            #
            # ``commit_state["committed"]`` flips to True INSIDE
            # the shielded coro the moment ``store_manifest``
            # returns. The drain handler reads it to distinguish
            # pre-commit failures (roll back: bundle dir + row →
            # cancelled) from post-commit failures (preserve the
            # bundle; the data is already published).
            commit_task = asyncio.ensure_future(
                self._commit_get_bundle(
                    request_id=request_id,
                    cache_key=cache_key,
                    manifest_path=manifest_path,
                    bundle_dir=bundle_dir,
                    req=req,
                    safe_params=safe_params,
                    options=options,
                    submitted_at=submitted_at,
                    started_at=started_at,
                    commit_state=commit_state,
                )
            )
            return await asyncio.shield(commit_task)

        except asyncio.CancelledError:
            # T-CMEMS-GET-005: pre- vs post-commit cancellation.
            # If we never reached the shield call, ``commit_task``
            # isn't bound — treat as pre-commit. Otherwise drain
            # the shielded task before deciding rollback.
            commit_task = locals().get("commit_task")
            if commit_task is not None:
                # Drain the shielded task, absorbing additional
                # cancels delivered to the outer task. ``shield``
                # re-raises the inner exception when the task is
                # done, so we also absorb that to keep the drain
                # loop converging on a stable state.
                while not commit_task.done():
                    try:
                        await asyncio.shield(commit_task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                # Surface a post-commit inner failure: the bundle is
                # already published (commit_state["committed"]==True)
                # so we keep it, but the row may still be stuck
                # ``running`` and we want operators to see the
                # exception in logs.
                inner_exc = (
                    commit_task.exception()
                    if commit_task.done() and not commit_task.cancelled()
                    else None
                )
                if inner_exc is not None and commit_state["committed"]:
                    logger.warning(
                        "get_files commit-task post-commit step failed",
                        extra={"request_id": request_id},
                        exc_info=(type(inner_exc), inner_exc, inner_exc.__traceback__),
                    )
                    # cr+codex round-2 MEDIUM: workflow row would
                    # otherwise stay ``running`` because NFA-3
                    # reconcile only fires when the cache file is
                    # missing. The bundle IS committed, so the
                    # honest terminal state is ``successful``.
                    with contextlib.suppress(Exception):
                        await asyncio.shield(
                            self.foundation.persistence.update_workflow_status_if_pending(
                                request_id, "successful"
                            )
                        )
            if not commit_state["committed"]:
                # Pre-commit rollback. Note: the underlying
                # ``marine.get`` thread cannot be cancelled
                # (the project conventions §10.9 gotcha #8) and may continue
                # writing to ``staging`` after the rmtree below.
                # That residual race is accepted — the thread will
                # eventually finish into a now-deleted dir, and any
                # files it manages to recreate are leaked under
                # ``.staging`` to be reaped by ops cleanup. Same
                # contract as subset's ``_safe_unlink(target_path)``.
                with contextlib.suppress(BaseException):
                    await asyncio.shield(
                        self.foundation.persistence.update_workflow_status_if_pending(
                            request_id, "cancelled"
                        )
                    )
                _safe_rmtree(staging)
            raise
        except CopernicusMcpError as exc:
            if commit_state["committed"]:
                # cr+codex round-1 MEDIUM: post-commit failure must
                # NOT delete the published bundle. Surface the
                # exception unchanged so operators see the real
                # cause; the cache row stays live.
                logger.warning(
                    "get_files commit-task post-commit failure (canonical)",
                    extra={"request_id": request_id},
                    exc_info=True,
                )
                # cr+codex round-2 MEDIUM: flip row to ``successful``
                # (best-effort) so the workflow log matches the
                # cache state.
                with contextlib.suppress(Exception):
                    await asyncio.shield(
                        self.foundation.persistence.update_workflow_status_if_pending(
                            request_id, "successful"
                        )
                    )
                raise
            error_record_json = self._serialise_error_record(exc)
            with contextlib.suppress(Exception):
                await asyncio.shield(
                    self.foundation.persistence.update_workflow_error(
                        request_id, "failed", error_record_json
                    )
                )
            _safe_rmtree(staging)
            raise
        except Exception as exc:
            if commit_state["committed"]:
                # cr+codex round-1 MEDIUM: same as the canonical
                # branch — preserve the bundle. cr round-2 MEDIUM:
                # the project conventions §6 forbids raw ``RuntimeError`` /
                # ``Exception`` from leaving the backend; wrap into
                # ``BackendError`` so the orchestrator sees a
                # canonical class. Also flip the workflow row to
                # ``successful`` so it matches the cache state.
                logger.warning(
                    "get_files commit-task post-commit failure (uncategorised)",
                    extra={"request_id": request_id},
                    exc_info=True,
                )
                with contextlib.suppress(Exception):
                    await asyncio.shield(
                        self.foundation.persistence.update_workflow_status_if_pending(
                            request_id, "successful"
                        )
                    )
                wrapped = BackendError(
                    f"get_files post-commit finalize failed: {exc!s}",
                    record=build_error_record(
                        "BackendError",
                        message="get_files post-commit finalize failed",
                        error_subclass="post_commit_finalize_failure",
                        recovery_action="report_to_administrator",
                    ),
                )
                raise wrapped from exc
            # T-CMEMS-GET-007: ``_wrap_subset_exception`` now branches
            # on ``op``; the ``get`` path produces a non-circular
            # hint pointing at ``marine_subset_dataset``.
            wrapped = _wrap_subset_exception(marine, exc, "get")
            error_record_json = self._serialise_error_record(wrapped)
            with contextlib.suppress(Exception):
                await asyncio.shield(
                    self.foundation.persistence.update_workflow_error(
                        request_id, "failed", error_record_json
                    )
                )
            _safe_rmtree(staging)
            raise wrapped from exc

    async def _commit_get_bundle(
        self,
        *,
        request_id: str,
        cache_key: str,
        manifest_path: Path,
        bundle_dir: Path,
        req: CmemsGetRequest,
        safe_params: dict[str, Any],
        options: dict[str, Any],
        submitted_at: str,
        started_at: str,
        commit_state: dict[str, bool],
    ) -> dict[str, Any]:
        """T-CMEMS-GET-005 shielded commit block.

        Runs as one atomic unit inside ``asyncio.shield``:
        - register the manifest in the cache,
        - record provenance (best-effort),
        - flip the workflow row to ``successful``.

        A cancellation delivered during this block is held at the
        ``shield`` boundary in the caller — the block finishes; the
        awaiter then sees ``CancelledError`` *after* the commit is
        durable.
        """
        await self.foundation.cache.store_manifest(
            cache_key=_cache_storage_key(cache_key),
            manifest_path=manifest_path,
            data_dir=bundle_dir,
            backend_id="cmems",
        )
        # store_manifest just returned → cache row published →
        # bundle is externally visible. Mark this turn the commit
        # point. The outer ``except asyncio.CancelledError``
        # handler reads ``commit_state["committed"]`` to know
        # that ANY failure or cancel from here on must NOT delete
        # the bundle (cr+codex round-1 MEDIUM: prior code keyed
        # the decision off ``commit_task.exception() is None``,
        # which mistakenly classified a post-commit ``read_manifest``
        # or finalize failure as pre-commit and tore down the
        # committed bundle).
        commit_state["committed"] = True

        manifest = read_manifest(bundle_dir)
        if manifest is None:
            raise BackendError(
                "get_files: manifest disappeared after build",
                record=build_error_record(
                    "BackendError",
                    message="get_files: manifest disappeared after build",
                    error_subclass="missing_manifest_after_build",
                    recovery_action="report_to_administrator",
                ),
            )
        envelope = to_files_envelope(manifest=manifest, data_dir=bundle_dir)

        # Provenance: best-effort. The cache entry is already
        # committed at this point — a provenance failure must not
        # leave the row stuck ``running``, so we log-and-continue.
        provenance_ref: str | None = None
        try:
            bundle_files = [Path(e["filepath"]) for e in envelope["files"]]
            record_id = await self.foundation.provenance.record_successful_retrieve(
                backend=BackendBlock(
                    id="cmems",
                    provider="Mercator Ocean International",
                    endpoint_url="https://my.cmems-du.eu",
                    api_version="marine-data-store-v1",
                ),
                dataset=DatasetBlock(
                    dataset_id=req.dataset_id,
                    dataset_version=req.dataset_version,
                    dataset_part=req.dataset_part,
                    service_name=None,
                ),
                request=RequestBlock(
                    operation="get",
                    submitted_at=submitted_at,
                    started_at=started_at,
                    finished_at=_iso_now_seconds(),
                    user_request=safe_params,
                    normalized_request=req.model_dump(),
                    options_applied=options,
                ),
                spatial=None,
                temporal=None,
                variables=[],
                files=bundle_files,
                cost_consumed=CostConsumed(
                    type="free",
                    advisory_message=(
                        "CMEMS data is free under the Mercator Ocean licence."
                    ),
                ),
                source_urls=[],
                cache=CacheRef(cache_key=cache_key, cache_hit=False),
                workflow_request_id=request_id,
            )
            provenance_ref = f"copernicus://provenance/{record_id}"
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "provenance recording failed for get_files", exc_info=True
            )

        envelope["provenance"] = (
            {"reference": provenance_ref} if provenance_ref else {}
        )

        await self.foundation.persistence.update_workflow_status(
            request_id, "successful"
        )
        return self.foundation.sanitiser.sanitise(  # type: ignore[no-any-return]
            _success_response_get_files(
                request_id=request_id,
                cache_key=cache_key,
                files_envelope=envelope,
                cache_hit=False,
                is_existing=False,
            )
        )

    def _serialise_error_record(self, exc: Any) -> str:
        """Serialise a canonical error to a sanitised JSON string for
        ``update_workflow_error``. Defence-in-depth: the wrapped exception's
        ``error_record.message`` is already sanitised at construction time;
        we sanitise the full record again before writing to SQLite."""
        record_dict = exc.error_record.model_dump(mode="json")
        sanitised = self.foundation.sanitiser.sanitise(record_dict)
        return json.dumps(sanitised, sort_keys=True, default=str)

    async def check_status(self, request_id: str) -> dict[str, Any]:
        row = await self.foundation.persistence.fetch_workflow(request_id)
        if row is None:
            raise NotFoundError(
                f"workflow {request_id!r} not found",
                record=build_error_record(
                    "NotFoundError",
                    message=f"workflow {request_id!r} not found",
                    recovery_action="modify_request_parameters",
                ),
            )
        # NFA-3: reconcile a stale ``running`` row left behind by a crashed
        # server. If the cache file for its cache_key is gone, the workflow
        # cannot have completed cleanly — flip to ``failed`` on observation.
        # Successful runs publish a file before updating to ``successful``,
        # so an absent file with a ``running`` row is unambiguous.
        # Staleness guard: only reconcile rows last touched > 60 s ago, so
        # an in-flight submit between ``marine.subset`` returning and
        # ``store_file`` completing cannot be mistaken for a crash.
        status = row["status"]
        if status == "running" and _row_is_stale(row, threshold_seconds=60):
            cache_key = row.get("cache_key")
            if cache_key:
                existing = await self.foundation.cache.lookup_file(_cache_storage_key(cache_key))
                if existing is None:
                    await self.foundation.persistence.update_workflow_status(request_id, "failed")
                    status = "failed"

        # codex round 2 HIGH: a successful async submit must surface the
        # large-data descriptor through ``check_status`` — otherwise the
        # MCP agent that polled and got ``status=successful`` cannot reach
        # the file. The sync path returns this inline; the async path
        # exposes it here.
        # codex/code-reviewer round 3 MEDIUM: if the row claims successful
        # but the file was LRU-evicted, fall through with a synthetic
        # ``cache_eviction`` error_details so the agent doesn't see
        # ``status=successful`` with an unusable empty result. Mirrors
        # ``fetch_result``'s ``CacheError(error_subclass="cache_eviction")``.
        result_block: dict[str, Any] = {}
        cache_evicted = False
        if status == "successful":
            cache_key_value = row.get("cache_key")
            if cache_key_value:
                cache_path = await self.foundation.cache.lookup_file(
                    _cache_storage_key(cache_key_value)
                )
                if cache_path is not None:
                    metadata: dict[str, Any] = {}
                    try:
                        metadata["size_bytes"] = cache_path.stat().st_size
                    except OSError:
                        pass
                    result_block = {
                        "filepath": str(cache_path),
                        "uri": f"copernicus://files/{cache_key_value}",
                        "metadata": metadata,
                        "provenance": {},
                    }
                else:
                    cache_evicted = True

        # codex/code-reviewer round 2 MEDIUM: surface ``error_details`` as a
        # structured dict so the agent doesn't have to ``json.loads`` a
        # JSON-string-inside-JSON. Falls back to the raw string if the row
        # holds a malformed payload.
        error_details: Any = row.get("error_record_json")
        if isinstance(error_details, str):
            try:
                error_details = json.loads(error_details)
            except json.JSONDecodeError:
                pass

        if cache_evicted:
            # Surface as a synthetic error so the agent doesn't observe
            # ``status=successful`` with an empty result block. Don't
            # mutate the persisted row — eviction is a present-tense
            # observation, not a permanent state.
            status = "failed"
            error_details = build_error_record(
                "CacheError",
                message=(
                    f"file for {request_id!r} was evicted from the cache; "
                    "re-run the subset to repopulate"
                ),
                error_subclass="cache_eviction",
                recovery_action="retry_with_modification",
            ).model_dump(mode="json")

        return self.foundation.sanitiser.sanitise(  # type: ignore[no-any-return]
            {
                "status": status,
                "request_id": row["request_id"],
                "submitted_at": row["created_at"],
                "updated_at": row["updated_at"],
                "cache_key": row["cache_key"],
                "error_details": error_details,
                "result": result_block,
            }
        )

    async def fetch_result(self, request_id: str, target: Path) -> dict[str, Any]:
        row = await self.foundation.persistence.fetch_workflow(request_id)
        if row is None:
            raise NotFoundError(
                f"workflow {request_id!r} not found",
                record=build_error_record(
                    "NotFoundError",
                    message=f"workflow {request_id!r} not found",
                    recovery_action="modify_request_parameters",
                ),
            )
        if row["status"] != "successful":
            raise BackendError(
                f"workflow {request_id!r} status={row['status']!r}, no result available",
                record=build_error_record(
                    "BackendError",
                    message=(
                        f"workflow {request_id!r} status={row['status']!r}, no result available"
                    ),
                    error_subclass="result_not_ready",
                    recovery_action="report_to_administrator",
                ),
            )
        # ``target`` is ignored for CMEMS — file is already in the cache zone.
        logger.debug(
            "fetch_result ignoring caller target; file already in cache zone",
            extra={"request_id": request_id, "target": str(target)},
        )
        cache_path = await self.foundation.cache.lookup_file(
            _cache_storage_key(row["cache_key"] or "")
        )
        if cache_path is None:
            raise CacheError(
                f"workflow {request_id!r} successful but file missing from cache",
                record=build_error_record(
                    "CacheError",
                    message=(f"workflow {request_id!r} successful but file missing"),
                    error_subclass="cache_eviction",
                    recovery_action="retry_with_modification",
                ),
            )
        return self.foundation.sanitiser.sanitise(  # type: ignore[no-any-return]
            _success_response(
                request_id=request_id,
                cache_key=row["cache_key"] or "",
                filepath=cache_path,
                cache_hit=True,
                is_existing=True,
                provenance_ref=None,
            )
        )

    async def cancel(self, request_id: str) -> dict[str, Any]:
        row = await self.foundation.persistence.fetch_workflow(request_id)
        if row is None:
            raise NotFoundError(
                f"workflow {request_id!r} not found",
                record=build_error_record(
                    "NotFoundError",
                    message=f"workflow {request_id!r} not found",
                    recovery_action="modify_request_parameters",
                ),
            )
        terminal = {"successful", "failed", "cancelled"}
        if row["status"] in terminal:
            return {
                "cancelled": False,
                "reason": f"already terminal (status={row['status']})",
                "request_id": request_id,
            }

        # T-039: if the request has a live background task, cancel it and
        # let its CancelledError handler mark the row + unlink the partial.
        # Awaiting with a 5-s timeout bounds the wait — the toolbox worker
        # thread may keep running (the project conventions gotcha #8).
        task = self._background_tasks.get(request_id)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait({task}, timeout=5)

        # codex round 2 HIGH (force-settle) + round 3 MEDIUM (TOCTOU):
        # use a conditional UPDATE so we never overwrite a row that
        # raced to ``successful`` / ``failed`` between fetch and write.
        # Idempotent with the task's own shielded update.
        await self.foundation.persistence.update_workflow_status_if_pending(request_id, "cancelled")
        final = await self.foundation.persistence.fetch_workflow(request_id)
        final_status = final["status"] if final is not None else "unknown"
        return {
            "cancelled": final_status == "cancelled",
            "request_id": request_id,
            "status": final_status,
        }

    # --- capabilities ----------------------------------------------------

    @property
    def supports_async(self) -> bool:
        return True

    @property
    def supports_dry_run(self) -> bool:
        return True

    @property
    def requires_terms_acceptance(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Module-level helpers — keep CmemsBackend itself thin.
# ---------------------------------------------------------------------------


def _load_copernicusmarine() -> Any:
    try:
        import copernicusmarine as marine
    except ImportError as exc:
        raise BackendError(
            "copernicusmarine SDK is not installed",
            record=build_error_record(
                "BackendError",
                message="copernicusmarine SDK is not installed",
                error_subclass="dependency_missing",
                recovery_action="report_to_administrator",
            ),
        ) from exc
    return marine


def _validate_bbox(
    raw: Any,
) -> tuple[float, float, float, float]:
    """Coerce a bbox input to ``(min_lon, min_lat, max_lon, max_lat)``.

    Round-1 review HIGHs:
      * malformed shapes (3-element list, non-numeric entries, etc.)
        raise ValidationError instead of being silently dropped.
      * antimeridian-crossing (``min_lon > max_lon``) is rejected per
        the antimeridian rejection rule — caller must split into two non-crossing bboxes.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise CmcpValidationError(
            f"search: 'bbox' must be a 4-element list "
            f"(min_lon, min_lat, max_lon, max_lat); got {raw!r}",
            record=build_error_record(
                "ValidationError",
                message=(
                    "search: 'bbox' must be a 4-element list (min_lon, min_lat, max_lon, max_lat)"
                ),
                recovery_action="modify_request_parameters",
            ),
        )
    try:
        min_lon, min_lat, max_lon, max_lat = (float(x) for x in raw)
    except (TypeError, ValueError) as exc:
        raise CmcpValidationError(
            f"search: 'bbox' entries must be numeric; got {raw!r}",
            record=build_error_record(
                "ValidationError",
                message=f"search: 'bbox' entries must be numeric; got {raw!r}",
                recovery_action="modify_request_parameters",
            ),
        ) from exc
    # codex round-2 MEDIUM: ``>=`` rejected zero-width (point/meridian)
    # bboxes as if they were antimeridian-crossing, inconsistent with
    # the subset invariant which allows ``min <= max`` on both axes
    # (schemas_cmems.py:126). Allow zero-width; only reject true
    # ``min_lon > max_lon`` (antimeridian) here.
    if min_lon > max_lon:
        raise CmcpValidationError(
            "antimeridian-crossing bboxes are not supported in Iter 1; "
            f"got min_lon={min_lon} > max_lon={max_lon}. Split into "
            "two non-crossing bboxes (e.g. (min_lon, min_lat, 180, "
            "max_lat) and (-180, min_lat, max_lon, max_lat)).",
            record=build_error_record(
                "ValidationError",
                message=("antimeridian-crossing bboxes are not supported in Iter 1"),
                recovery_action="modify_request_parameters",
                next_action_hint=(
                    "Split into two non-crossing bboxes (one ending at 180, one starting at -180)."
                ),
            ),
        )
    if min_lat > max_lat:
        raise CmcpValidationError(
            f"search: 'bbox' min_lat ({min_lat}) > max_lat ({max_lat})",
            record=build_error_record(
                "ValidationError",
                message=f"search: 'bbox' min_lat ({min_lat}) > max_lat ({max_lat})",
                recovery_action="modify_request_parameters",
            ),
        )
    return (min_lon, min_lat, max_lon, max_lat)


def _validate_time_range(raw: Any) -> tuple[str, str]:
    """Coerce a time_range input to ``(start_iso, end_iso)``.

    Round-1 HIGH: malformed shape silently dropped — now raises.
    Round-2 codex MEDIUM: also parse the ISO timestamps and require
    ``start < end`` so a reversed or non-ISO range fails loudly here
    rather than degrading to a silent no-match downstream.
    """
    from copernicus_mcp.backends.cmems.routing import _parse_iso

    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise CmcpValidationError(
            f"search: 'time_range' must be a 2-element list "
            f"(start, end) of ISO-8601 strings; got {raw!r}",
            record=build_error_record(
                "ValidationError",
                message=(
                    "search: 'time_range' must be a 2-element list (start, end) of ISO-8601 strings"
                ),
                recovery_action="modify_request_parameters",
            ),
        )
    start, end = raw
    if not (isinstance(start, str) and isinstance(end, str) and start.strip() and end.strip()):
        raise CmcpValidationError(
            f"search: 'time_range' entries must be non-empty ISO-8601 strings; got {raw!r}",
            record=build_error_record(
                "ValidationError",
                message=("search: 'time_range' entries must be non-empty ISO-8601 strings"),
                recovery_action="modify_request_parameters",
            ),
        )
    try:
        start_dt = _parse_iso(start)
        end_dt = _parse_iso(end)
    except ValueError as exc:
        raise CmcpValidationError(
            f"search: 'time_range' entries must be ISO-8601 timestamps; "
            f"got start={start!r}, end={end!r}",
            record=build_error_record(
                "ValidationError",
                message=("search: 'time_range' entries must be ISO-8601 timestamps"),
                recovery_action="modify_request_parameters",
            ),
        ) from exc
    if start_dt >= end_dt:
        raise CmcpValidationError(
            f"search: 'time_range' start ({start}) must be strictly earlier than end ({end})",
            record=build_error_record(
                "ValidationError",
                message=("search: 'time_range' start must be strictly earlier than end"),
                recovery_action="modify_request_parameters",
            ),
        )
    return (start, end)


def _validate_search(params: dict[str, Any]) -> CmemsSearchRequest:
    from pydantic import ValidationError as PydValidationError

    # Strip orchestrator-injected ``__options`` before Pydantic extra="forbid".
    params = {k: v for k, v in params.items() if k != "__options"}
    # T-CMEMS-HIER-005: bbox and time_range are now honest filters
    # routed through the cards path (``_search_in_products``).
    # ``service_types`` filtering is still unimplemented.
    unsupported = [f for f in ("service_types",) if params.get(f)]
    if unsupported:
        raise CmcpValidationError(
            f"CMEMS search filters not yet implemented: {unsupported}",
            record=build_error_record(
                "ValidationError",
                message=f"CMEMS search filters not yet implemented: {unsupported}",
                recovery_action="modify_request_parameters",
                next_action_hint=(
                    "Use ``keyword``, ``product_id``, ``product_ids``, "
                    f"``bbox``, or ``time_range``; omit {unsupported}."
                ),
            ),
        )
    try:
        return CmemsSearchRequest.model_validate(params)
    except PydValidationError as exc:
        field_errors = [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
        ]
        raise CmcpValidationError(
            "invalid CMEMS search params",
            record=build_error_record(
                "ValidationError",
                message="invalid CMEMS search params",
                recovery_action="modify_request_parameters",
                context={"field_errors": field_errors},
            ),
        ) from exc


# T-CMEMS-CAT-003: ``_describe_kwargs`` / ``_map_dataset`` /
# ``_map_describe_response`` removed — search is now offline against
# the bundled catalogue snapshot. The schema/envelope contract those
# helpers maintained is preserved by ``backends/cmems/catalogue.py``
# +  ``backends/cmems/_catalogue_build.py`` (the build side that
# produces the slim records). See
# ``spikes/T-CMEMS-CAT-000-describe-shape/FINDINGS.md`` for the
# schema lock.


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read an attribute or dict key uniformly (toolbox returns objects or dicts).

    Still used by the describe / estimate / subset paths which call
    the live SDK and need to handle both dict- and Pydantic-shaped
    responses.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _row_is_stale(row: Mapping[str, Any], *, threshold_seconds: int) -> bool:
    """Return True if ``row['updated_at']`` is older than ``threshold_seconds``.

    Falls open (returns True) if the field is missing or unparseable so a
    malformed row still gets reconciled rather than silently lingering.

    Accepts any read-only string-keyed mapping — ``TypedDict``
    (``WorkflowRecord`` from the persistence layer) is not assignable to
    ``dict[str, Any]`` in mypy's type system, so use ``Mapping`` to cover
    both the plain-dict and TypedDict call sites without a runtime cast.
    """
    raw = row.get("updated_at") or row.get("created_at")
    if not raw:
        return True
    try:
        # Persistence stores ISO-8601 UTC strings ending in ``Z``.
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return True
    age = (datetime.now(UTC) - ts).total_seconds()
    return age >= threshold_seconds


def _wrap_marine_exception(marine: Any, exc: BaseException, op: str) -> Exception:
    """Map a ``copernicusmarine`` exception to our canonical taxonomy.

    Sanitises the toolbox message before embedding it in our ErrorRecord —
    the SDK has been observed echoing query-string-shaped URLs that may carry
    signed-token parameters.
    """
    from copernicus_mcp.errors.sanitiser import Sanitiser

    raw_msg = str(exc)
    safe_msg = Sanitiser().sanitise(raw_msg)

    login_error = getattr(marine, "LoginError", None)
    not_found = getattr(marine, "DatasetNotFound", None)

    if login_error is not None and isinstance(exc, login_error):
        return AuthError(
            f"CMEMS authentication failed during {op}",
            record=build_error_record(
                "AuthError",
                message=f"CMEMS authentication failed during {op}",
                recovery_action="configure_credentials",
                recovery_url=_REGISTRATION_URL,
            ),
        )
    if not_found is not None and isinstance(exc, not_found):
        return NotFoundError(f"CMEMS dataset not found during {op}: {safe_msg}")

    lc = raw_msg.lower()
    # FA-3: detect coverage-shaped errors (request bbox / depth / time
    # outside the dataset's actual extent). CMEMS toolbox phrasings include
    # "exceed the dataset coordinates" and "outside ... extent".
    if ("exceed" in lc and "coordinates" in lc) or ("outside" in lc and "extent" in lc):
        return CoverageUnavailableError(
            f"CMEMS dataset coverage error during {op}: {safe_msg}",
            record=build_error_record(
                "CoverageUnavailableError",
                message=f"CMEMS dataset coverage error during {op}: {safe_msg}",
                recovery_action="modify_request_parameters",
                next_action_hint=(
                    "Call marine_describe_dataset to inspect the dataset's "
                    "spatial / temporal / depth extent and narrow the request."
                ),
            ),
        )
    if any(s in lc for s in ("timeout", "connection", "network", "dns")):
        return NetworkError(f"CMEMS network error during {op}: {safe_msg}")

    return BackendError(
        f"CMEMS toolbox error during {op}: {safe_msg}",
        record=build_error_record(
            "BackendError",
            message=f"CMEMS toolbox error during {op}: {safe_msg}",
            error_subclass="toolbox_error",
            recovery_action="report_to_administrator",
        ),
    )


# ---------------------------------------------------------------------------
# Cache helpers (shared by search + metadata).
# ---------------------------------------------------------------------------


async def _lookup_json_cache(
    persistence: Any, *, namespace: str, key: str, ttl_seconds: int
) -> dict[str, Any] | None:
    entry = await persistence.lookup_cache_entry(namespace, key)
    if entry is None:
        return None
    try:
        iso = entry["created_at"].replace("Z", "+00:00")
        created = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if datetime.now(UTC) - created > timedelta(seconds=ttl_seconds):
        await persistence.delete_cache_entry(namespace, key)
        return None
    try:
        return json.loads(entry["value_json"])  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        return None


async def _persist_json_cache(
    persistence: Any, *, namespace: str, key: str, payload: dict[str, Any]
) -> None:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    value_json = json.dumps(payload, sort_keys=True)
    await persistence.record_cache_entry(
        {
            "namespace": namespace,
            "key": key,
            "value_json": value_json,
            "file_path": None,
            "size_bytes": len(value_json.encode("utf-8")),
            "content_type": "application/json",
            "created_at": now,
            "last_accessed_at": now,
        }
    )


# ---------------------------------------------------------------------------
# Metadata mapping helpers.
# ---------------------------------------------------------------------------


def _find_dataset(response: Any, dataset_id: str) -> tuple[Any, Any]:
    """Return (product, dataset) tuple for ``dataset_id``; ``(None, None)`` if missing."""
    for product in _get(response, "products") or []:
        for dataset in _get(product, "datasets") or []:
            if _get(dataset, "dataset_id") == dataset_id or _get(dataset, "id") == dataset_id:
                return product, dataset
    return None, None


def _latest_version(dataset: Any) -> Any:
    """Return the version with the highest ``label``.

    CMEMS labels are ``YYYYMM`` strings (e.g. ``"202411"``) which sort
    lexicographically into chronological order, so we don't depend on the
    toolbox preserving any particular list ordering.
    """
    versions = list(_get(dataset, "versions") or [])
    if not versions:
        return None
    labelled = [(_get(v, "label") or "", v) for v in versions]
    labelled.sort(key=lambda t: t[0])
    return labelled[-1][1]


def _map_full_metadata(product: Any, dataset: Any) -> dict[str, Any]:
    """Map a dataset to the describe-result shape.

    Walks ``versions → parts → services → variables`` per real SDK shape
    (codex T-022 HIGH). Falls back to fixture-shaped ``parts[].variables``
    for older tests.
    """
    version = _latest_version(dataset)
    parts = _get(version, "parts") or [] if version is not None else []

    services_seen: list[dict[str, Any]] = []
    variables_seen: dict[str, dict[str, Any]] = {}
    part_names: list[str] = []
    # Aggregate variable bboxes — real SDK puts extent on variables, not
    # dataset. Empirically: MEDSEA_ANALYSISFORECAST_PHY_006_013 leaves
    # ``dataset.spatial_extent = None`` while each ``variable.bbox`` carries
    # ``[min_lon, min_lat, max_lon, max_lat]``. Agents need the dataset-level
    # extent to form correct subset bboxes without clamping surprises.
    var_bboxes: list[list[float]] = []

    for part in parts:
        name = _get(part, "name")
        if name is not None:
            part_names.append(str(name))
        # First collect from services[].variables (real SDK shape).
        for svc in _get(part, "services") or []:
            services_seen.append(
                {
                    "service_type": _get(svc, "service_type"),
                    "service_name": _get(svc, "service_name") or _get(svc, "name"),
                    "base_url": _get(svc, "uri") or _get(svc, "base_url"),
                }
            )
            for var in _get(svc, "variables") or []:
                _accumulate_variable(variables_seen, var)
                bbox = _get(var, "bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    # Review L4: if the SDK ever emits a non-numeric value
                    # in a bbox slot (e.g. ``"30.5N"``), skip that variable
                    # rather than raising — describe stays usable for
                    # malformed datasets, advisory just won't fire for them.
                    try:
                        var_bboxes.append([float(x) for x in bbox])
                    except (TypeError, ValueError):
                        continue
        # Fallback for fixture-shaped tests with parts[].variables directly.
        for var in _get(part, "variables") or []:
            _accumulate_variable(variables_seen, var)

    return {
        "dataset_id": (
            _get(dataset, "dataset_id")
            or _get(dataset, "id")
            or _get(dataset, "dataset_name")  # SDK
        ),
        "dataset_version": _get(version, "label") if version is not None else None,
        "dataset_parts": part_names,
        "product_id": _get(product, "product_id"),
        "title": (
            _get(dataset, "title")
            or _get(dataset, "dataset_name")  # SDK
            or _get(product, "title")
        ),
        "services": services_seen,
        "variables": list(variables_seen.values()),
        "spatial_extent": (_get(dataset, "spatial_extent") or _union_bboxes(var_bboxes)),
        "temporal_extent": _get(dataset, "temporal_extent"),
        "doi": (
            _get(product, "doi")
            or _get(dataset, "doi")
            or _get(product, "digital_object_identifier")  # SDK
            or _get(dataset, "digital_object_identifier")
        ),
        "license": _get(product, "license") or _get(dataset, "license"),
        "citation": _get(product, "citation") or _get(dataset, "citation"),
    }


def _union_bboxes(bboxes: list[list[float]]) -> dict[str, float] | None:
    """Union variable bboxes into a dataset-level spatial_extent dict.

    Real CMEMS SDK stores extent on each ``variable.bbox`` as
    ``[min_lon, min_lat, max_lon, max_lat]``. The dataset extent is the
    enclosing box across all variables — what an agent needs to form a
    subset request that covers everything available.
    """
    if not bboxes:
        return None
    return {
        "min_lon": min(b[0] for b in bboxes),
        "min_lat": min(b[1] for b in bboxes),
        "max_lon": max(b[2] for b in bboxes),
        "max_lat": max(b[3] for b in bboxes),
    }


# Tolerance for advisory exact-match comparison. Float-roundtrip via JSON
# can introduce sub-ulp drift; a user typing 36.290000001 vs the SDK's
# 36.29 shouldn't trigger a spurious advisory (review L2).
_AXIS_EXACT_TOLERANCE = 1e-9


def _axis_state(u_min: float, u_max: float, ds_min: float, ds_max: float) -> str:
    """Classify one axis of a user bbox against the dataset extent.

    Returns one of:
      - ``"exact"``    — user range matches dataset within tolerance.
      - ``"inside"``   — user range is strictly inside the dataset's.
      - ``"exceeds"``  — user range fully contains or extends past on at
                         least one side, while overlapping the dataset.
      - ``"disjoint"`` — user range does not overlap the dataset at all.
      - ``"partial"``  — one side inside, other outside (rare; classified
                         conservatively as "exceeds" by callers).
    """
    if (
        abs(u_min - ds_min) <= _AXIS_EXACT_TOLERANCE
        and abs(u_max - ds_max) <= _AXIS_EXACT_TOLERANCE
    ):
        return "exact"
    if u_max < ds_min or u_min > ds_max:
        return "disjoint"
    if u_min >= ds_min and u_max <= ds_max:
        return "inside"
    if u_min <= ds_min and u_max >= ds_max:
        return "exceeds"
    # Mixed: e.g. u_min < ds_min but u_max < ds_max — one side wider, the
    # other narrower. Treat as exceeds; the advisory message in the
    # mixed-axis branch will show the per-axis numbers anyway.
    return "exceeds"


def _accumulate_variable(seen: dict[str, dict[str, Any]], var: Any) -> None:
    """Add a variable entry to ``seen`` (first wins on short_name conflict)."""
    short = _get(var, "short_name") or _get(var, "name")
    if short is None or str(short) in seen:
        return
    seen[str(short)] = {
        "name": str(short),
        "long_name": _get(var, "long_name"),
        "units": _get(var, "units"),
        "valid_min": _get(var, "valid_min"),
        "valid_max": _get(var, "valid_max"),
    }


# ---------------------------------------------------------------------------
# Coordinate axis extraction.
# ---------------------------------------------------------------------------


def _select_part(dataset: Any, version_label: str | None, service: str | None) -> Any:
    versions = _get(dataset, "versions") or []
    chosen_version: Any = None
    if version_label is not None:
        for v in versions:
            if _get(v, "label") == version_label:
                chosen_version = v
                break
        if chosen_version is None:
            raise NotFoundError(
                f"dataset version {version_label!r} not found",
                record=build_error_record(
                    "NotFoundError",
                    message=f"dataset version {version_label!r} not found",
                    recovery_action="modify_request_parameters",
                ),
            )
    elif versions:
        chosen_version = _latest_version(dataset)
    if chosen_version is None:
        return None
    parts = _get(chosen_version, "parts") or []
    if service is not None:
        for part in parts:
            for svc in _get(part, "services") or []:
                if _get(svc, "service_name") == service or _get(svc, "service_type") == service:
                    return part
        raise NotFoundError(
            f"service {service!r} not found on dataset",
            record=build_error_record(
                "NotFoundError",
                message=f"service {service!r} not found on dataset",
                recovery_action="modify_request_parameters",
            ),
        )
    return parts[0] if parts else None


def _summarise_axis(coord: Any, values: list[Any], axis: str) -> dict[str, Any]:
    """Summarise an axis. Field name reflects unit: ``stride_seconds`` for
    time, plain ``stride`` for spatial/depth axes (degrees, metres)."""
    out: dict[str, Any] = {
        "start": _get(coord, "minimum_value")
        if _get(coord, "minimum_value") is not None
        else (values[0] if values else None),
        "end": _get(coord, "maximum_value")
        if _get(coord, "maximum_value") is not None
        else (values[-1] if values else None),
        "count": len(values),
    }
    out["stride_seconds" if axis == "time" else "stride"] = _get(coord, "step")
    return out


def _extract_coordinates(
    dataset: Any,
    *,
    requested_version: str | None,
    requested_service: str | None,
) -> dict[str, Any]:
    part = _select_part(dataset, requested_version, requested_service)
    if part is None:
        return {"longitude": [], "latitude": [], "depth": [], "time": []}

    # Accumulate across variables (CMEMS axes are shared per part in practice).
    # Prefer the SDK ``part.get_coordinates()`` API when available.
    by_axis: dict[str, Any] = {}
    by_axis_full: dict[str, list[Any]] = {}
    get_coords = getattr(part, "get_coordinates", None)
    if callable(get_coords):
        try:
            sdk_coords = get_coords()
        except Exception:
            sdk_coords = None
        if isinstance(sdk_coords, dict):
            for cid, entry in sdk_coords.items():
                # SDK shape: {coord_id: (coord, variable_ids, service_names)}
                coord_obj = entry[0] if isinstance(entry, tuple) else entry
                values = list(_get(coord_obj, "values") or [])
                by_axis[cid] = coord_obj
                by_axis_full[cid] = values
    if not by_axis:
        # Fallback: walk services[].variables[].coordinates (real SDK), then
        # parts[].variables[].coordinates (fixture).
        sources: list[Any] = []
        for svc in _get(part, "services") or []:
            sources.extend(_get(svc, "variables") or [])
        sources.extend(_get(part, "variables") or [])
        for var in sources:
            for coord in _get(var, "coordinates") or []:
                cid = _get(coord, "coordinate_id")
                if cid is None or cid in by_axis:
                    continue
                values = list(_get(coord, "values") or [])
                by_axis[cid] = coord
                by_axis_full[cid] = values

    out: dict[str, Any] = {}
    for axis in ("longitude", "latitude", "depth", "time"):
        if axis not in by_axis:
            out[axis] = []
            continue
        values = by_axis_full[axis]
        limit = _AXIS_FULL_LIMIT_TIME if axis == "time" else _AXIS_FULL_LIMIT_SPATIAL
        if len(values) > limit:
            out[axis] = _summarise_axis(by_axis[axis], values, axis)
        else:
            out[axis] = values
    return out


# ---------------------------------------------------------------------------
# Subset / estimate helpers.
# ---------------------------------------------------------------------------


def _validate_subset(params: dict[str, Any]) -> CmemsSubsetRequest:
    from pydantic import ValidationError as PydValidationError

    # Strip orchestrator-injected ``__options`` before Pydantic extra="forbid".
    params = {k: v for k, v in params.items() if k != "__options"}
    try:
        return CmemsSubsetRequest.model_validate(params)
    except PydValidationError as exc:
        field_errors = [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
        ]
        raise CmcpValidationError(
            "invalid CMEMS subset params",
            record=build_error_record(
                "ValidationError",
                message="invalid CMEMS subset params",
                recovery_action="modify_request_parameters",
                context={"field_errors": field_errors},
            ),
        ) from exc


def _validate_get_coordinates(
    params: dict[str, Any],
) -> CmemsGetCoordinatesRequest:
    """Validate ``marine_get_coordinates`` params with credential-safe error mapping.

    Same pattern as ``_validate_list_files``: catch pydantic, project to
    structured ``field_errors`` so no raw input values leak into the wire
    message.
    """
    from pydantic import ValidationError as PydValidationError

    params = {k: v for k, v in params.items() if k != "__options"}
    try:
        return CmemsGetCoordinatesRequest.model_validate(params)
    except PydValidationError as exc:
        field_errors = [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
            for e in exc.errors()
        ]
        raise CmcpValidationError(
            "invalid CMEMS get_coordinates params",
            record=build_error_record(
                "ValidationError",
                message="invalid CMEMS get_coordinates params",
                recovery_action="modify_request_parameters",
                context={"field_errors": field_errors},
            ),
        ) from exc


def _validate_list_files(params: dict[str, Any]) -> CmemsListFilesRequest:
    """Validate ``marine_list_files`` params with credential-safe error mapping.

    cr round-1 H1: raw ``CmemsListFilesRequest(**params)`` would surface
    pydantic's ``input_value=...`` strings (which echo raw inputs) through
    the wire. We catch the pydantic error, project it to structured
    ``field_errors`` (loc/msg/type — no raw values), and re-raise the
    project's canonical ``ValidationError``.

    **Contributor note (cr round-2 M1):** the loc/msg/type projection
    relies on every ``CmemsListFilesRequest`` field-validator emitting
    fixed-string ``ValueError`` messages WITHOUT interpolating raw input
    values. A validator like ``raise ValueError(f"invalid {v}")`` would
    push the value into ``msg`` and defeat the credential-safety here.
    """
    from pydantic import ValidationError as PydValidationError

    params = {k: v for k, v in params.items() if k != "__options"}
    try:
        return CmemsListFilesRequest.model_validate(params)
    except PydValidationError as exc:
        field_errors = [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
            for e in exc.errors()
        ]
        raise CmcpValidationError(
            "invalid CMEMS list_files params",
            record=build_error_record(
                "ValidationError",
                message="invalid CMEMS list_files params",
                recovery_action="modify_request_parameters",
                context={"field_errors": field_errors},
            ),
        ) from exc


def _subset_kwargs(req: CmemsSubsetRequest, *, dry_run: bool) -> dict[str, Any]:
    """Translate our schema to ``copernicusmarine.subset`` kwargs."""
    kwargs: dict[str, Any] = {
        "dataset_id": req.dataset_id,
        "variables": list(req.variables),
        "minimum_longitude": req.minimum_longitude,
        "maximum_longitude": req.maximum_longitude,
        "minimum_latitude": req.minimum_latitude,
        "maximum_latitude": req.maximum_latitude,
        "minimum_depth": req.minimum_depth,
        "maximum_depth": req.maximum_depth,
        "start_datetime": req.start_datetime,
        "end_datetime": req.end_datetime,
        "coordinates_selection_method": req.coordinates_selection_method,
        "file_format": req.file_format,
        "netcdf_compression_level": req.netcdf_compression_level,
        "dry_run": dry_run,
    }
    if req.dataset_version is not None:
        kwargs["dataset_version"] = req.dataset_version
    if req.dataset_part is not None:
        kwargs["dataset_part"] = req.dataset_part
    if req.service is not None:
        kwargs["service"] = req.service
    return kwargs


def _human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    size_f = float(num_bytes) / 1024
    for unit in ("KB", "MB", "GB", "TB"):
        if size_f < 1024:
            return f"{size_f:.1f} {unit}"
        size_f /= 1024
    return f"{size_f:.1f} PB"


def _map_estimate_response(response: Any) -> dict[str, Any]:
    # Plan §1 also lists ``n_chunks``, ``temporal_chunks``, ``spatial_chunks``;
    # the toolbox ``ResponseSubset`` (see spikes/T-000-cmems-smoke/FINDINGS.md)
    # does not expose them. Iter 1 punts on those fields.
    transfer_mb = _get(response, "data_transfer_size")
    file_mb = _get(response, "file_size")
    raw_service = _get(response, "service")
    service: str = str(raw_service) if raw_service is not None else "unknown"

    epistemic = "precise"
    if transfer_mb is None and file_mb is None:
        bytes_estimate = 0
        epistemic = "approximate"
    else:
        # Prefer data_transfer_size (network bytes) per T-000 spike findings —
        # confirmation gates compare against IO/network budgets, not on-disk size.
        chosen_mb = transfer_mb if transfer_mb is not None else file_mb
        bytes_estimate = int(float(chosen_mb) * 1024 * 1024)

    msg = _get(response, "message")
    advisory = (
        str(msg) if msg is not None else "CMEMS data is free under the Mercator Ocean licence."
    )
    return {
        "type": "free",
        "estimated_size_bytes": bytes_estimate,
        "estimated_size_human": _human_size(bytes_estimate),
        "service_used": service,
        "advisory_message": advisory,
        "epistemic_status": epistemic,
    }


def _map_get_estimate_response(response: Any) -> dict[str, Any]:
    """T-CMEMS-GET-004: shape a ``marine.get(dry_run=True)`` return.

    Field-tolerant — the SDK's dry-run output for ``get`` varies
    across versions. We probe ``total_size`` / ``data_transfer_size``
    / ``file_size`` (all in MB by the toolbox's convention) and fall
    back to ``epistemic_status="approximate"`` if none is present. The
    gate in ``get_files`` treats ``approximate`` as always-gate per
    sub-plan D3 (v2 dropped v1's "small-get-skips-gate" rule).
    """
    total_mb = _get(response, "total_size")
    if total_mb is None:
        total_mb = _get(response, "data_transfer_size")
    file_mb = _get(response, "file_size")

    epistemic = "precise"
    if total_mb is None and file_mb is None:
        bytes_estimate = 0
        epistemic = "approximate"
    else:
        chosen_mb = total_mb if total_mb is not None else file_mb
        bytes_estimate = int(float(chosen_mb) * 1024 * 1024)

    msg = _get(response, "message")
    advisory = (
        str(msg)
        if msg is not None
        else "CMEMS data is free under the Mercator Ocean licence."
    )
    return {
        "type": "free",
        "estimated_size_bytes": bytes_estimate,
        "estimated_size_human": _human_size(bytes_estimate),
        # ``service_used`` is reported by ``marine.subset`` (Zarr
        # routing decision); ``marine.get`` doesn't pick a service —
        # native files are served directly. Surface "n/a" so a
        # caller that inspects the field doesn't crash.
        "service_used": "n/a",
        "advisory_message": advisory,
        "epistemic_status": epistemic,
    }


def _validate_get(params: dict[str, Any]) -> CmemsGetRequest:
    from pydantic import ValidationError as PydValidationError

    params = {k: v for k, v in params.items() if k != "__options"}
    try:
        return CmemsGetRequest.model_validate(params)
    except PydValidationError as exc:
        field_errors = [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
            for e in exc.errors()
        ]
        raise CmcpValidationError(
            "invalid CMEMS get params",
            record=build_error_record(
                "ValidationError",
                message="invalid CMEMS get params",
                recovery_action="modify_request_parameters",
                context={"field_errors": field_errors},
            ),
        ) from exc


def _get_kwargs(req: CmemsGetRequest) -> dict[str, Any]:
    """Translate our schema to ``copernicusmarine.get`` kwargs.

    Only forwards fields the user set explicitly; the SDK's own
    defaults fill in the rest. Omits ``output_directory`` — the caller
    sets that to its staging path.
    """
    kwargs: dict[str, Any] = {"dataset_id": req.dataset_id}
    if req.dataset_version is not None:
        kwargs["dataset_version"] = req.dataset_version
    if req.dataset_part is not None:
        kwargs["dataset_part"] = req.dataset_part
    if req.filter is not None:
        kwargs["filter"] = req.filter
    if req.regex is not None:
        kwargs["regex"] = req.regex
    if req.file_list is not None:
        kwargs["file_list"] = list(req.file_list)
    if req.sync is not None:
        kwargs["sync"] = req.sync
    if req.skip_existing is not None:
        kwargs["skip_existing"] = req.skip_existing
    if req.overwrite is not None:
        kwargs["overwrite"] = req.overwrite
    return kwargs


def _wrap_subset_exception(marine: Any, exc: BaseException, op: str) -> Exception:
    """Marine submit/get exception mapping (extends ``_wrap_marine_exception``).

    T-CMEMS-GET-007: ``WrongFormatRequested`` now appears in BOTH the
    subset and the ``get`` paths. The hint is op-aware:
    - subset / estimate context → "use marine_get_files" (sparse
      datasets only support native files).
    - get / list_files context → "use marine_subset_dataset" (grid
      datasets only support Zarr subset; ``marine.get`` raised because
      the dataset isn't shipped as native files).
    Before this fix the hint was circular when raised from inside
    ``get_files`` / ``list_files`` (T-CMEMS-GET-INDEX-004 cr round-1 H2).
    """
    wrong_format = getattr(marine, "WrongFormatRequested", None)
    if wrong_format is not None and isinstance(exc, wrong_format):
        if op in ("get", "list_files"):
            msg = (
                "this dataset's service (likely arco-geo-series or "
                "arco-time-series) does not support native-file get; "
                "use marine_subset_dataset instead"
            )
            hint = (
                "Call marine_subset_dataset for grid datasets that only "
                "expose a Zarr subset path."
            )
        else:
            msg = (
                "this dataset's service (likely arco-platform-series) "
                "does not support subset; use marine_get_files instead"
            )
            hint = (
                "Call marine_get_files for sparse / platform-series datasets."
            )
        return CmcpValidationError(
            msg,
            record=build_error_record(
                "ValidationError",
                message=msg,
                recovery_action="modify_request_parameters",
                next_action_hint=hint,
            ),
        )
    return _wrap_marine_exception(marine, exc, op)


# ---------------------------------------------------------------------------
# Submit helpers.
# ---------------------------------------------------------------------------


def _cache_storage_key(cache_key: str) -> str:
    """Map a logical CMEMS cache_key to the cache-manager storage key.

    History: v0.1.2 stored under ``f"file:{cache_key}"``. Round 2 of the
    T-039 review naively dropped the prefix to make the URI
    ``copernicus://files/{cache_key}`` resolve through ``_file_resource``,
    which broke v0.1.2 → v0.1.3 cache migration (codex round 3 HIGH:
    double-count + shared file_path → eviction unlinked the live file).
    Round 3 reverts: storage stays prefixed, the URI stays bare, and
    ``_file_resource`` calls this helper to bridge.
    """
    return f"file:{cache_key}"


def _new_request_id() -> str:
    return f"req-{uuid.uuid4().hex[:16]}"


def _iso_now_seconds() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _running_response(*, request_id: str, cache_key: str) -> dict[str, Any]:
    """Envelope for an async submit that has been accepted but not yet
    completed. The ``result.uri`` points at the jobs resource so the LLM
    client can render a stable handle even before any file exists."""
    return {
        "status": "running",
        "cache_hit": False,
        "is_existing": False,
        "request_id": request_id,
        "cache_key": cache_key,
        "result": {
            "uri": f"copernicus://jobs/{request_id}",
            "metadata": {},
            "provenance": {},
        },
    }


def _safe_unlink(path: Path) -> None:
    """Best-effort unlink; never raises. Used by the submit error/cancel
    handlers so the staging dir doesn't accumulate partials when the
    toolbox aborts mid-write."""
    with contextlib.suppress(FileNotFoundError, OSError):
        if path.exists():
            path.unlink()


def _safe_rmtree(path: Path) -> None:
    """Best-effort recursive removal; never raises. Used by
    ``get_files`` error handlers to clean up staging / partial bundle
    directories without masking the original exception."""
    with contextlib.suppress(FileNotFoundError, OSError):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def _success_response(
    *,
    request_id: str,
    cache_key: str,
    filepath: Path,
    cache_hit: bool,
    is_existing: bool,
    provenance_ref: str | None,
) -> dict[str, Any]:
    """Canonical large-data success envelope (the project conventions invariant 1)."""
    metadata: dict[str, Any] = {}
    try:
        metadata["size_bytes"] = filepath.stat().st_size
    except OSError:
        pass
    return {
        "status": "successful",
        "cache_hit": cache_hit,
        "is_existing": is_existing,
        "request_id": request_id,
        "cache_key": cache_key,
        "result": {
            "filepath": str(filepath),
            "uri": f"copernicus://files/{cache_key}",
            "metadata": metadata,
            "provenance": ({"reference": provenance_ref} if provenance_ref else {}),
        },
    }


def _success_response_get_files(
    *,
    request_id: str,
    cache_key: str,
    files_envelope: dict[str, Any],
    cache_hit: bool,
    is_existing: bool,
) -> dict[str, Any]:
    """Multi-file ``get`` success envelope (T-CMEMS-GET-003).

    Mirrors ``_success_response`` but the ``result`` block carries a
    ``files`` list (one descriptor per data file) plus an
    envelope-level ``provenance`` reference. Per-file descriptors keep
    an empty ``provenance: {}`` for shape parity with ``_success_response``
    — there is one provenance record per bundle, attached at the
    envelope level rather than duplicated per file.
    """
    return {
        "status": "successful",
        "cache_hit": cache_hit,
        "is_existing": is_existing,
        "request_id": request_id,
        "cache_key": cache_key,
        "mode": "offline",
        "result": files_envelope,
    }


_SAFE_DATASET_ID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def _slug_dataset_id(dataset_id: str) -> str:
    r"""Slug a dataset_id for safe use as a filename basename.

    Defends against path-traversal attacks (``/``, ``..``, ``\``, NUL).
    Replaces unsafe chars with ``_``.
    """
    cleaned = "".join(c if c in _SAFE_DATASET_ID_CHARS else "_" for c in dataset_id)
    # Strip leading dots so a malicious id can't become a hidden file.
    cleaned = cleaned.lstrip(".")
    return cleaned or "_unnamed"
