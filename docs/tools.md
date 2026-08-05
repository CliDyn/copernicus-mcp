# MCP Tool Reference

`copernicus-mcp` registers a diagnostic plus per-backend tool surfaces:

- **Diagnostic** (always registered): `copernicus_mcp_status`, `copernicus_mcp_list_jobs`.
- **CMEMS** (eleven tools, registered when the `cmems` backend is enabled): `marine_search_groups`, `marine_search_products`, `marine_search_datasets`, `marine_describe_dataset`, `marine_get_coordinates`, `marine_estimate_subset`, `marine_subset_dataset`, `marine_list_files`, `marine_get_files`, `marine_check_status`, `marine_cancel_subset`. The first three implement the three-step hierarchical pipeline (T-CMEMS-HIER-005): start with `marine_search_groups` for free-text routing, drill into `marine_search_products` with the chosen `group_ids`, then resolve datasets via `marine_search_datasets` with the chosen `product_ids` (plus optional `bbox` / `time_range`). The pipeline is the default path for any agentic query; the bare `marine_search_datasets` (`keyword=` only) flat path stays available for known dataset ids.
- **CDS / ADS / EWDS** (nine tools, registered when the `cds` backend is enabled AND credentials resolve): `cds_search_datasets`, `cds_describe_dataset`, `cds_apply_constraints`, `cds_estimate_request`, `cds_submit_request`, `cds_check_request_status`, `cds_download_request_result`, `cds_cancel_request`, `cds_list_licences` — plus `cds_accept_licence` as a tenth when the operator opts in via `budget.cds_licence_accept_enabled`.

All tools share these conventions:

- **Large-data invariant.** Tools that produce scientific data return a descriptor (`filepath` + metadata + `provenance`), never inline bytes. The MCP client opens the file from the returned path.
- **Errors.** A canonical error returns `{"error": <ErrorRecord>}` over the wire (FastMCP wraps this as `isError=true`; clients can recover the structured record from the message body — see [`docs/architecture.md`](architecture.md)). Each section below lists the relevant error classes.
- **Sanitisation.** Every outbound payload passes through `Sanitiser`; credentials never appear in tool output regardless of upstream library behaviour.

---

## `copernicus_mcp_status`

Server diagnostics: registered backends, credential sources (without values), cache metrics, persistence path, configuration snapshot.

### Inputs

None.

### Output

```jsonc
{
  "version": "0.0.1",
  "backends": {
    "cmems": {
      "registered": true,
      "enabled_in_config": true,
      "configured": true,
      "credential_source": "env"   // env | config_file | secret_manager | explicit | missing
    },
    "cds": {
      "registered": true, "enabled_in_config": true, "configured": true,
      "credential_source": "config_file",
      // Live count of the ACCOUNT's in-flight remote jobs (everything the
      // account has queued/running, not just this server's). "truncated":
      // true means the page filled — read the count as "at least". On any
      // probe failure the field is the string "unavailable" (with an
      // "active_remote_jobs_reason") — the status tool answers regardless.
      // Omitted entirely when no CDS credentials resolve.
      "active_remote_jobs": {
        "count": 3,
        "by_status": {"accepted": 2, "running": 1},
        "fetched_at": "2026-08-05T12:00:00Z"
      }
    }
  },
  "cache": {
    "directory": "~/.cache/copernicus-mcp",
    "size_bytes": 0,
    "entry_count": 0
  },
  "persistence": {
    "database_path": "~/.local/state/copernicus-mcp/state.db"
  },
  "config": { /* sanitised non-secret subset */ }
}
```

### Errors

- `BackendError` (`error_subclass="status_failure"`) — internal failure during diagnostics. Wraps the underlying cause; the original message is sanitised before reaching the client.

### Examples

```jsonc
// request
{"name": "copernicus_mcp_status", "arguments": {}}

// response (structured)
{ "version": "0.0.1", "backends": { "cmems": { "registered": true, ... } }, ... }
```

---

## `copernicus_mcp_list_jobs`

Enumerate recent submitted jobs (downloads) from the local state store so a fresh session can recover work after a restart — no `request_id` required. Feed a returned `request_id` to the per-backend status / fetch / cancel tools.

### Inputs

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `status` | `string[]` \| null | `null` | Keep only jobs in these statuses. Allowed: `queued`, `running`, `successful`, `failed`, `cancelled`. |
| `limit` | integer | `50` | Maximum jobs returned, newest first. Clamped to `1..500`. |
| `created_after` | string \| null | `null` | ISO-8601 UTC lower bound (strict `>`), e.g. `2026-06-01T00:00:00Z`. |

### Output

```jsonc
{
  "results": [
    {
      "request_id": "…",
      "backend": "cds",             // cmems | cds
      "operation": "submit",
      "status": "successful",        // queued | running | successful | failed | cancelled
      "dataset": "reanalysis-era5-single-levels",
      "created_at": "2026-06-01T12:00:00Z",
      "updated_at": "2026-06-01T12:03:00Z",
      "error_class": "BackendError"  // present only when status == "failed"
    }
  ],
  "count": 1
}
```

The listing does not carry the result file path — the workflow row stores no result descriptor. Retrieve a completed job's file with the per-backend download/fetch tool (or the `copernicus://jobs/{request_id}` resource) using its `request_id`.

### Errors

- `ValidationError` (`recovery_action="modify_request_parameters"`) — `status` contains a value outside the five canonical statuses, or `created_after` is not a valid ISO-8601 timestamp.
- `BackendError` (`error_subclass="list_jobs_failure"`) — internal failure while listing; the underlying message is sanitised before it reaches the client.

### Examples

```jsonc
// request — all recent jobs
{"name": "copernicus_mcp_list_jobs", "arguments": {}}

// request — only in-flight, newest 10
{"name": "copernicus_mcp_list_jobs", "arguments": {"status": ["queued", "running"], "limit": 10}}

// response (structured)
{ "results": [ { "request_id": "…", "backend": "cds", "status": "running", "dataset": "…" } ], "count": 1 }
```

---

## `marine_search_groups`

First step of the hierarchical search pipeline (T-CMEMS-HIER-005). Shortlist CMEMS routing groups for a free-text query. Each group bundles related products by region, domain, and intent (e.g. `physics-mediterranean-state`, `ocean-acidification-monitoring`, `climate-reanalysis`, `arctic-comprehensive`).

Offline-only — reads the bundled `groups.json` (47 groups). No credentials, no network.

### Inputs

| Field   | Type                         | Required | Default | Description                                                                |
| ------- | ---------------------------- | -------- | ------- | -------------------------------------------------------------------------- |
| `query` | `string` (non-blank)         | yes      | —       | Free-text — what the user is looking for.                                  |
| `top_k` | `integer (1..20) \| null`    | no       | `5`     | Maximum groups to surface in `selected`.                                   |

### Output

The standard hierarchical-search envelope:

```jsonc
{
  "selected": [
    {
      "group_id": "physics-arctic-state",
      "group_title": "Arctic Ocean Physics — analysis, forecast, and reanalysis",
      "summary": "Arctic gridded ocean state ...",
      "product_ids": ["ARCTIC_ANALYSISFORECAST_PHY_002_001", ...],
      "score": 6.0
    }
  ],
  "rejected": [
    {"group_id": "...", "score": -1.0, "reason": "exclude_when_query_mentions matched"}
  ],
  "reason": "top group 'physics-arctic-state' matched with score 6.0 (high confidence).",
  "confidence": "high",          // high | medium | low
  "fallback_available": false,   // true when confidence == "low"
  "catalogue_fetched_at": "2026-05-14T09:01:22Z"
}
```

- Scoring: +2 per `include_when_query_mentions` phrase substring-found in the query (word-boundary match), −3 per `exclude_when_query_mentions` phrase, +1 for any meaningful title token, +0.5 for any summary token.
- `confidence = "high"` when the top score is ≥ 4.0, `"medium"` when ≥ 1.5, otherwise `"low"`.
- `fallback_available = true` signals the caller should fall back to the flat `marine_search_datasets` path (or rephrase).

### Errors

- `ValidationError` — `query` blank or whitespace-only.

### Examples

```jsonc
// request
{"name": "marine_search_groups",
 "arguments": {"input": {"query": "arctic sea ice extent"}}}

// response
{"selected": [
   {"group_id": "sea-ice-arctic", "score": 6.0, ...},
   {"group_id": "arctic-comprehensive", "score": 3.0, ...}],
 "confidence": "high", "fallback_available": false, ...}
```

---

## `marine_search_products`

Second step of the hierarchical pipeline. Take the `group_ids` from `marine_search_groups` and surface the candidate CMEMS products with their summaries — feed the resulting `product_ids` to `marine_search_datasets` for the final dataset shortlist.

Offline-only.

### Inputs

| Field       | Type                         | Required | Default | Description                                                            |
| ----------- | ---------------------------- | -------- | ------- | ---------------------------------------------------------------------- |
| `group_ids` | `array<string>` (non-empty)  | yes      | —       | Group ids from `marine_search_groups.selected[*].group_id`.            |
| `query`     | `string \| null`             | no       | `null`  | Optional keyword to re-rank within the union of the named groups.       |
| `top_k`     | `integer (1..50) \| null`    | no       | `20`    | Maximum products to surface.                                            |

### Output

Standard envelope. Each `selected` entry carries `product_id`, `product_title`, `summary`, `domains`, `regions`, `data_types`, `dataset_count`, and `score`. Products are ranked by (keyword score desc, then group-rank position from the input `group_ids` desc, then alphabetical product id) so the top-ranked group's products land first.

### Errors

- `ValidationError` — `group_ids` empty.

### Examples

```jsonc
// request
{"name": "marine_search_products",
 "arguments": {"input": {"group_ids": ["sea-ice-arctic"], "query": "drift"}}}

// response
{"selected": [{"product_id": "SEAICE_ARC_PHY_L4_NRT_011_006", "score": 2.0, ...}],
 "confidence": "medium", ...}
```

---

## `marine_search_datasets`

Third step of the hierarchical pipeline, or a standalone flat search by keyword/product id. Two paths plus two modes:

- **Hierarchical path** — set `product_ids` (and optionally `bbox` / `time_range`). Reads the enriched dataset cards (`dataset_cards.json`), filters by product membership + spatial / temporal overlap, returns cards with full enriched fields (`domain`, `region`, `data_type`, `variables_normalized`, `best_for`, `not_good_for`, `spatial_label`, `temporal_label`, `quality_flags`). Cards with `spatial_extent` / `temporal_extent` `None` are EXCLUDED when the corresponding filter is set — better under-select than risk a mismatch.
- **Flat path** — set `keyword` and/or `product_id` (singular), no `bbox`/`time_range`/`product_ids`. Reads the slim catalogue (`marine.json`) and returns the smaller per-dataset envelope.

Each path supports two modes:

- **Offline (default).** Reads bundled snapshots. Fast, no credentials, no network. Age exposed as `catalogue_fetched_at`.
- **Live (`live=true`).** Calls `copernicusmarine.describe()` against the live service. Requires CMEMS credentials and ~10 s. Use for products published after the last bundled refresh. Note: `live=true` is only honoured on the flat path; the hierarchical path is always offline (the cards manifest is the source of truth).

### Inputs

| Field           | Type                            | Required | Default | Description                                                                |
| --------------- | ------------------------------- | -------- | ------- | -------------------------------------------------------------------------- |
| `keyword`       | `string \| null`                | no       | `null`  | Flat-path: case-insensitive substring match against dataset ids/titles/etc. Hierarchical path: re-ranks cards by phrase overlap. |
| `product_ids`   | `array<string> \| null`         | no       | `null`  | Hierarchical-path shortlist (usually from `marine_search_products`). Routes through the enriched cards.                          |
| `bbox`          | `[min_lon, min_lat, max_lon, max_lat]` | no | `null` | Spatial filter. Hierarchical path keeps only cards whose `spatial_extent` overlaps the bbox; null-extent cards are excluded. Antimeridian-crossing bboxes (min_lon > max_lon) are rejected with `ValidationError` per the project conventions inv-7 — split into two non-crossing bboxes. |
| `time_range`    | `[start_iso, end_iso]`          | no       | `null`  | Temporal filter. ISO-8601 strings; `start < end` required. Hierarchical path keeps only cards whose `temporal_extent` overlaps; null-extent cards are excluded. |
| `service_types` | `array<enum>`                   | no       | `null`  | Filter by service kind — `timeseries`, `geoseries`, `omi-arco`, `static-arco`, `platformseries` (short names, mapped to the catalogue's `arco-*` names). Returns only datasets exposing that service; e.g. `["timeseries"]` finds the datasets that support fast point time-series. Not yet combinable with `bbox` / `time_range` / `product_ids`. |
| `limit`         | `integer (>=1) \| null`         | no       | `null`  | Maximum dataset records returned. Capped at 50 on the hierarchical path. Flat path's `total_count` reflects the full match count before slicing. |
| `live`          | `boolean`                       | no       | `false` | Flat path only. When `true`, call the live SDK. Requires CMEMS credentials. |

### Output

**Hierarchical path** (when `product_ids` / `bbox` / `time_range` is set) returns the standard search envelope (`selected`, `rejected`, `reason`, `confidence`, `fallback_available`, `catalogue_fetched_at`). Each `selected` entry is a full enriched card.

**Flat path** returns the slim-catalogue envelope:

```jsonc
{
  "datasets": [
    {
      "dataset_id": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
      "dataset_name": "Daily mean physics",
      "title": "Daily mean physics",
      "product_id": "GLOBAL_ANALYSISFORECAST_PHY_001_024",
      "product_title": "Global Ocean Physics Analysis and Forecast",
      "description": "Daily mean fields...",
      "doi": "10.48670/moi-00016",
      "service_types": ["original-files"],
      "variables": ["thetao", "so"],
      "versions": ["202406"],
      "spatial_extent": {"min_lon": -180.0, "min_lat": -90.0, "max_lon": 180.0, "max_lat": 90.0},
      "temporal_extent": null
    }
  ],
  "total_count": 1,
  "mode": "offline",
  "catalogue_fetched_at": "2026-05-14T09:01:22Z"
}
```

- `mode` is `"offline"` for the bundled snapshot, `"live"` for SDK results.
- `catalogue_fetched_at` is `null` when `mode == "live"`.
- `spatial_extent` is `null` for datasets whose variable bboxes are all sentinel `[0, 0, 0, 0]`. `temporal_extent` is `null` in slim records (the upstream SDK does not surface a dataset-level time range on the flat path; the hierarchical cards aggregate it from variable coordinates).

### Errors

- `ValidationError` — `limit < 1`; `service_types` combined with `bbox` / `time_range` / `product_ids` (use it on its own for now); `bbox` with wrong shape or antimeridian-crossing; `time_range` with non-ISO entries or `start >= end`; both `product_id` and `product_ids` set.
- `AuthError` — `live=true` and CMEMS credentials missing or invalid. Offline default never raises this.
- `BackendError` — bundled snapshot missing (broken install) or live SDK call failed after retries.

### Examples

Hierarchical (from `marine_search_products` output):

```jsonc
// request
{"name": "marine_search_datasets",
 "arguments": {"input": {
    "product_ids": ["GLOBAL_ANALYSISFORECAST_PHY_001_024"],
    "bbox": [20.0, 41.0, 30.0, 47.0],
    "limit": 5}}}

// response
{"selected": [{"dataset_id": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
                "domain": "physics", "region": ["global"],
                "data_type": ["analysis", "forecast"],
                "best_for": [...], "not_good_for": [...], ...}],
 "confidence": "high", "fallback_available": false, ...}
```

Flat offline:

```jsonc
// request
{"name": "marine_search_datasets",
 "arguments": {"input": {"keyword": "temperature", "limit": 3}}}

// response
{"datasets": [...], "total_count": 87, "mode": "offline",
 "catalogue_fetched_at": "2026-05-14T09:01:22Z"}
```

Live (flat, requires credentials):

```jsonc
// request
{"name": "marine_search_datasets",
 "arguments": {"input": {"keyword": "fresh-product", "live": true}}}

// response
{"datasets": [...], "total_count": 1, "mode": "live",
 "catalogue_fetched_at": null}
```

---

## `marine_describe_dataset`

Return full metadata for a single CMEMS dataset: variables, axes, services, terms.

### Inputs

| Field        | Type     | Required | Default | Description                                            |
| ------------ | -------- | -------- | ------- | ------------------------------------------------------ |
| `dataset_id` | `string` | yes      | —       | Dataset id from `marine_search_datasets`. Non-blank.   |

### Output

```jsonc
{
  "dataset_id": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
  "title": "...",
  "abstract": "...",
  "variables": [{"name": "thetao", "units": "degrees_C", "standard_name": "..."}],
  "axes": {"longitude": {"min": -180.0, "max": 180.0}, "latitude": {...}, ...},
  "services": [{"service_type": "geoseries", "service_format": "zarr", ...}],
  "doi": "https://doi.org/...",
  "terms_url": "https://marine.copernicus.eu/..."
}
```

### Errors

- `ValidationError` — empty or whitespace `dataset_id`.
- `NotFoundError` — dataset id does not exist in the catalogue (`recovery_action="retry_with_modification"`).
- `AuthError` — credentials missing or invalid.
- `BackendError` — toolbox-level failure.

### Examples

```jsonc
// request
{"name": "marine_describe_dataset",
 "arguments": {"input": {"dataset_id": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"}}}

// response
{"dataset_id": "...", "variables": [{"name": "thetao", ...}], ...}
```

---

## `marine_get_coordinates`

Return the dataset's coordinate axes (lat / lon / depth / time, plus any non-spatio-temporal axes). Use this BEFORE `marine_estimate_subset` or `marine_subset_dataset` when you need the real extent of the dataset — its actual depth levels, time stride, or to verify your bbox lies inside coverage. Cheaper than `marine_describe_dataset` for the same purpose because the response is just axes, not full metadata.

### Inputs

| Field             | Type             | Required | Default | Description                                                                                                       |
| ----------------- | ---------------- | -------- | ------- | ----------------------------------------------------------------------------------------------------------------- |
| `dataset_id`      | `string`         | yes      | —       | Dataset id from `marine_search_datasets`. Non-blank.                                                              |
| `dataset_version` | `string \| null` | no       | `null`  | Version label (e.g. `"202411"`); defaults to the latest.                                                          |
| `service`         | `string \| null` | no       | `null`  | Service id (`geoseries`, `timeseries`, `omi-arco`, …) to disambiguate when a version exposes multiple services.   |

### Output

A dict keyed by axis name. Short axes (≤ 5000 entries) return as full lists; long axes return as a summary dict.

```jsonc
{
  "longitude": [-180.0, -179.9, ...],         // short axis → full list
  "latitude":  [-90.0, -89.9, ...],           // short axis → full list
  "depth":     [0.49, 1.54, 2.65, ...],
  "time": {                                    // long axis → summary
    "start": "1993-01-01T00:00:00Z",
    "end":   "2024-12-31T23:59:59Z",
    "count": 11688,
    "stride_seconds": 86400
  }
}
```

### Errors

- `ValidationError` — empty / whitespace `dataset_id`, unknown extra fields.
- `NotFoundError` — dataset id (or version) not in the catalogue.
- `AuthError` — credentials missing or invalid.
- `BackendError` — toolbox-level failure.

### Examples

```jsonc
// request
{"name": "marine_get_coordinates",
 "arguments": {"input": {"dataset_id": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"}}}

// response — most axes elided
{"longitude": [...], "latitude": [...], "depth": [...], "time": {"start": "...", "end": "...", "count": 11688, "stride_seconds": 86400}}
```

---

## `marine_estimate_subset`

Preview byte-size, variable list and confirmation status for a subset request without downloading. Always inexpensive; safe to call before `marine_subset_dataset`.

### Inputs

Same shape as `marine_subset_dataset`. See the table below.

### Output

```jsonc
{
  "estimated_size_bytes": 4123456,
  "estimated_size_mb": 3.93,
  "epistemic_status": "exact",        // "exact" or "approximate"
  "variables": ["thetao"],
  "service_used": "geoseries",
  "advisory_message": null            // populated for approximate estimates
}
```

When `epistemic_status="approximate"`, the output also contains `confirmation_required: true` so a downstream subset call can gate on user approval (see "Confirmation flow" in [`docs/architecture.md`](architecture.md)).

### Errors

- `ValidationError` — same set as `marine_subset_dataset`.
- `AuthError`, `BackendError`.

### Examples

```jsonc
{"name": "marine_estimate_subset",
 "arguments": {"input": {
    "dataset_id": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
    "variables": ["thetao"],
    "minimum_longitude": -1.0, "maximum_longitude": 0.0,
    "minimum_latitude": 45.0, "maximum_latitude": 46.0,
    "minimum_depth": 0.0, "maximum_depth": 5.0,
    "start_datetime": "2024-06-01T00:00:00Z",
    "end_datetime":   "2024-06-01T23:59:59Z"
 }}}
```

---

## `marine_subset_dataset`

Download a spatio-temporal subset of a CMEMS dataset. Returns a descriptor; the file lives on disk in the cache directory until eviction.

**This tool returns a descriptor (filepath + metadata + provenance), not the data itself.** Open the file via the returned `filepath` using xarray, netCDF4, or your preferred I/O library.

### Inputs

| Field                           | Type                  | Required | Default     | Description                                                                  |
| ------------------------------- | --------------------- | -------- | ----------- | ---------------------------------------------------------------------------- |
| `dataset_id`                    | `string` (non-blank)  | yes      | —           | Dataset id from `marine_search_datasets`.                                    |
| `dataset_version`               | `string \| null`      | no       | `null`      | Pin a specific dataset version.                                              |
| `dataset_part`                  | `string \| null`      | no       | `null`      | Pin a specific part within a multi-part dataset.                             |
| `variables`                     | `array<string>` (≥1)  | yes      | —           | Names from the dataset metadata, e.g. `["thetao"]`.                          |
| `minimum_longitude`             | `float [-180, 180]`   | yes      | —           | Western edge.                                                                |
| `maximum_longitude`             | `float [-180, 180]`   | yes      | —           | Eastern edge. Antimeridian-crossing bboxes are rejected with a recovery hint. |
| `minimum_latitude`              | `float [-90, 90]`     | yes      | —           | Southern edge.                                                               |
| `maximum_latitude`              | `float [-90, 90]`     | yes      | —           | Northern edge.                                                               |
| `minimum_depth`                 | `float (>=0) \| null` | no       | `null`      | Shallowest depth in metres. Optional — omit for a surface / 2-D dataset that has no depth axis. |
| `maximum_depth`                 | `float (>=0) \| null` | no       | `null`      | Deepest depth in metres. Optional; omit both bounds to retrieve the dataset's full depth range. |
| `start_datetime`                | `string` (ISO 8601)   | yes      | —           | Inclusive start. Date-only (`2023-06-01`) and naive datetimes are accepted and assumed UTC; tz-aware inputs are converted to UTC. |
| `end_datetime`                  | `string` (ISO 8601)   | yes      | —           | Strictly after `start_datetime`. Same date-only / naive / UTC handling.      |
| `coordinates_selection_method`  | enum                  | no       | `inside`    | `inside` \| `strict-inside` \| `nearest` \| `outside`.                       |
| `service`                       | `string \| null`      | no       | `null`      | Force a specific CMEMS service (`geoseries`, `timeseries`, …).               |
| `file_format`                   | enum                  | no       | `netcdf`    | `netcdf` \| `zarr` \| `csv`. `csv` is opt-in and **restricted to a single point** (`min==max` on both longitude and latitude) — it is retrieved via the toolbox `read_dataframe`, which loads the subset into memory, so an area request with `csv` is rejected with a hint to use `netcdf`. The default stays `netcdf`. |
| `netcdf_compression_level`      | `integer [0, 9]`      | no       | `1`         | NetCDF deflate level.                                                        |

### Output

```jsonc
{
  "filepath": "~/.cache/copernicus-mcp/cmems/<cache_key>/data.nc",
  "uri": "copernicus://files/<cache_key>",
  "cache_key": "<sha256-derived>",
  "cache_hit": false,
  "metadata": {"file_format": "netcdf", "file_size_bytes": 4123456, ...},
  "provenance": {
    "record_id": "<uuid>",
    "workflow_request_id": "<uuid>",
    "uri": "copernicus://provenance/<record_id>"
  }
}
```

### Confirmation flow

If the estimate exceeds `budget.cmems_per_request_size_warning_gb` (default 1 GB) or returns `epistemic_status="approximate"`, the tool returns:

```jsonc
{
  "confirmation_required": true,
  "advisory_message": "estimated 2.5 GB — confirm to proceed",
  "estimated_size_bytes": 2500000000,
  "tool_name": "marine_subset_dataset",
  "backend": "cmems",
  "source": "config.budget.cmems_per_request_size_warning_gb"
}
```

The agent must then call the tool again with `options.confirmed=true` (CLI: `--yes`) to actually download. There is no global "skip-all" by design.

### Errors

- `ValidationError` — malformed inputs, antimeridian bbox, sparse-dataset request to `subset` (use `marine_get_files` instead).
- `AuthError` — credentials missing or invalid.
- `NotFoundError` — dataset id not in catalogue.
- `CoverageUnavailableError` — bbox or time range outside the dataset's actual coverage.
- `QuotaError` — backend rate-limited or daily quota exceeded.
- `OperationCancelledError` — caller cancelled (Ctrl-C, MCP `notifications/cancelled`).
- `BackendError` — toolbox-level failure (network, server-side error).
- `TermsNotAcceptedError` — dataset requires terms acceptance not yet recorded for this user.

### Examples

```jsonc
// confirmation gate
{"name": "marine_subset_dataset", "arguments": {"input": { ... }}}
// → {"confirmation_required": true, "advisory_message": "...", "estimated_size_bytes": ...}

// confirmed retrieval
{"name": "marine_subset_dataset",
 "arguments": {"input": { /* same input */ }, "options": {"confirmed": true}}}
// → {"filepath": "...", "metadata": {...}, "provenance": {...}}
```

The `provenance` block points at a sidecar `.provenance.json` next to the file and at a SQLite row keyed by `record_id`. Either source is sufficient to reproduce the request.

---

## `marine_list_files`

**Use this BEFORE `marine_get_files` when you want a precise subset of a sparse dataset.** Datasets like CORA, EasyCORA, and INSITU-BGC contain millions of native files; downloading the entire bundle for one region/time-window is wasteful. `marine_list_files` consults a local index (Parquet-cached after the first call) and returns a precise `file_list` filtered by bbox / time / variables / platform.

Workflow: `marine_search_datasets` → `marine_list_files` → `marine_get_files(file_list=[...])`.

### Inputs

| Field            | Type                                       | Required | Default | Description                                                                                                                                                                                                                                          |
| ---------------- | ------------------------------------------ | -------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset_id`     | `string` (non-blank)                       | yes      | —       | Dataset id from `marine_search_datasets`.                                                                                                                                                                                                          |
| `bbox`           | `[min_lon, min_lat, max_lon, max_lat]`     | no       | `null`  | **Antimeridian-crossing bboxes (`min_lon > max_lon`) ARE accepted here** — unlike `marine_subset_dataset`. The wrap is treated as `[min_lon, 180] ∪ [-180, max_lon]`. The resulting `file_list` is usable with `marine_get_files` but **NOT** with `marine_subset_dataset`. |
| `time_range`     | `[start, end]` ISO 8601                    | no       | `null`  | UTC-normalised; inverted ranges rejected.                                                                                                                                                                                                          |
| `variables`      | `array<string> \| null`                    | no       | `null`  | Whitelist of variable names. Rows whose `variables` field is `None` (no info) are kept; rows with explicit `()` are dropped.                                                                                                                       |
| `platform_types` | `array<string> \| null`                    | no       | `null`  | Whitelist of platform-type codes (PF, CT, MO, DB, …). Rows with unknown platform are dropped.                                                                                                                                                      |
| `limit`          | `int (≥1) \| null`                         | no       | `null`  | Cap on returned rows. Truncation is sorted by `file_path` ASC for determinism; the envelope surfaces `matched_count_uncapped` and `truncated=true` so callers see the true match count.                                                            |

### Output

```jsonc
{
  "status": "successful",
  "result": {
    "files": [
      {
        "file_path": "INSITU_.../GL_TS_CO_2EIF7-SOCATv2025.nc",
        "lon_min": -97.8386, "lon_max": 3.5688,
        "lat_min": 10.1022, "lat_max": 51.942,
        "time_start": "2009-09-16T17:11:48Z",
        "time_end": "2012-08-07T06:45:30Z",
        "platform_type": null,
        "variables": ["FCO2", "PSAL", "TEMP"],
        "size_bytes": null
      },
      // ...
    ],
    "matched_count": 12,
    "matched_count_uncapped": 12,
    "truncated": false,
    "total_count_in_index": 305,
    "total_size_bytes_known": 0,
    "rows_with_unknown_size": 12,
    "filters_applied": {"bbox": [-5, 30, 40, 46], "time_range": ["2010-01-01T00:00:00Z", "2015-12-31T23:59:59Z"]},
    "index_fetched_at": "2026-05-16T17:00:00Z",
    "mode": "offline"
  }
}
```

`mode` is `"offline"` when the Parquet cache hit and `"fresh"` when the SDK was hit. First call per dataset is "fresh"; subsequent calls are "offline" (milliseconds).

### First-call latency

- **INSITU-BGC** (GLODAP, SOCAT, multi-obs): one-time ~1–5 s SDK fetch of the `index_file.txt`.
- **CORA / EasyCORA**: one-time **~210 s** SDK dry-run listing of ~1M file paths. This is sync — the MCP call blocks. Subsequent calls read the Parquet cache and return in milliseconds.
- **Other sparse datasets**: live discovery via SDK probe (~1–5 s). Datasets without a parseable index raise `NotFoundError`.

### Errors

- `ValidationError` — malformed inputs, inverted `time_range`, naive datetimes, lat/lon out of canonical ranges, grid-only dataset (hint: use `marine_subset_dataset`).
- `AuthError` — credentials missing or invalid.
- `NotFoundError` — dataset has no parseable index (hint: use `marine_get_files(filter=...)` instead). Repeat probes are short-circuited via a 5-min negative cache to prevent auth-storm.

### Cache location

Indices are NOT bundled in the wheel and NOT committed to git. Per-dataset Parquet caches live under `<cache_directory>/marine_indices/<dataset_id>.parquet` (see [`setup.md`](./setup.md#cache-directory)).

---

## `marine_get_files`

Download native CMEMS files (no Zarr slicing). Use this for datasets whose service is `original-files` or `arco-platform-series` (sparse / in-situ observations) — these don't support `marine_subset_dataset`. Returns one descriptor per file the toolbox produced.

**Tip: for sparse datasets, call `marine_list_files` first** to get a precise filtered `file_list` instead of downloading the full multi-GB bundle.

**Like `marine_subset_dataset`, this tool returns descriptors, not bytes.** Open each file via the returned `filepath`.

### Inputs

| Field             | Type                                  | Required | Default | Description                                                                       |
| ----------------- | ------------------------------------- | -------- | ------- | --------------------------------------------------------------------------------- |
| `dataset_id`      | `string` (non-blank)                  | yes      | —       | Dataset id from `marine_search_datasets`.                                         |
| `dataset_version` | `string \| null`                      | no       | `null`  | Pin a specific dataset version.                                                   |
| `dataset_part`    | `string \| null`                      | no       | `null`  | Pin a specific part within a multi-part dataset.                                  |
| `filter`          | `string \| null` (non-blank)          | no       | `null`  | Glob pattern, e.g. `"*1990*"`. Mutually exclusive with `regex` / `file_list`.     |
| `regex`           | `string \| null` (non-blank)          | no       | `null`  | Python regex matching file paths. Mutually exclusive with the other selectors.    |
| `file_list`       | `array<string>` (≥1, each non-blank)  | no       | `null`  | Explicit list of file paths to fetch. Set-semantic (order doesn't affect cache).  |
| `sync`            | `bool \| null`                        | no       | `null`  | Forwarded to the SDK unchanged.                                                   |
| `skip_existing`   | `bool \| null`                        | no       | `null`  | Forwarded to the SDK unchanged.                                                   |
| `overwrite`       | `bool \| null`                        | no       | `null`  | Forwarded to the SDK unchanged.                                                   |
| `confirmed`       | `bool`                                | no       | `false` | Set to `true` to bypass the size confirmation gate (see below).                   |

Omit `filter` / `regex` / `file_list` to download whatever the toolbox defaults to for the dataset. At most one of them may be set.

### Output

```jsonc
{
  "status": "successful",
  "cache_hit": false,
  "is_existing": false,
  "request_id": "<uuid>",
  "cache_key": "cmems:get:<dataset_id>:<hash>",
  "mode": "offline",
  "result": {
    "files": [
      {
        "filepath": "/.../<bundle>/a_1990.nc",
        "uri": "copernicus://files/<cache_key>?file=a_1990.nc",
        "metadata": {"size_bytes": 12345, "md5": "..."},
        "provenance": {}
      },
      // ... one descriptor per data file in the bundle
    ],
    "provenance": {"reference": "copernicus://provenance/<record_id>"}
  }
}
```

The whole bundle lives in a single subdirectory under the cache zone; eviction tears it down in lockstep with the manifest cache entry.

### Confirmation flow

`copernicusmarine.get` does NOT always surface a precise `dry_run` size. The gate uses:
- precise size > `budget.cmems_per_request_size_warning_gb` → gate fires.
- `epistemic_status="approximate"` (SDK didn't report any size — common for sparse formats) → gate ALWAYS fires, regardless of bytes. Pass `confirmed=true` (CLI `--yes`) to bypass.

### Errors

- `ValidationError` — malformed inputs, grid-only dataset that doesn't support `get` (hint: use `marine_subset_dataset`), or mutually-exclusive selectors set together.
- `AuthError` — credentials missing or invalid.
- `NotFoundError` — empty match (`filter` / `regex` matched zero files) or dataset id not in catalogue.
- `OperationCancelledError` — caller cancelled before the commit point (`store_manifest`).
- `BackendError` — toolbox-level failure or post-commit finalize failure (the bundle is still committed when this fires).

### Examples

```jsonc
// First call — gate fires for sparse formats with no precise estimate
{"name": "marine_get_files",
 "arguments": {"input": {"dataset_id": "cmems_obs-ins_glo_phy-temp-sal_my_easycora_irr", "filter": "*1990*"}}}
// → {"confirmation_required": true, ...}

// Confirmed retrieval
{"name": "marine_get_files",
 "arguments": {"input": {"dataset_id": "cmems_obs-ins_glo_phy-temp-sal_my_easycora_irr", "filter": "*1990*", "confirmed": true}}}
// → {"status": "successful", "result": {"files": [...]}}
```

The bundle's URI (`copernicus://files/<cache_key>`) resolves through the MCP file resource as a JSON envelope of descriptors (not a single path string), letting the agent enumerate per-file paths without re-fetching.

---

## CDS / ADS / EWDS tools

The `cds` backend is queue-backed: most operations are async by design. A single PAT works across all three stores; the backend dispatches by dataset id (research §6.8.2). Discovery uses a bundled catalogue snapshot rather than a live API — there is no programmatic search endpoint upstream.

Enable the backend by setting `enabled_backends: [cmems, cds]` in your config or `COPERNICUS_MCP_ENABLED_BACKENDS=cmems,cds` in env, then ensure `CDSAPI_KEY` (UUID) is set. See [`setup.md`](./setup.md#credentials-for-cds--ads--ewds).

## `cds_search_datasets`

Discover dataset ids by keyword across the bundled CDS / ADS / EWDS catalogue snapshot.

### Inputs

- `keyword` (string, optional) — free-text match against dataset id, title, description.
- `store` (literal, optional) — restrict to one of `cds` / `ads` / `ewds`. Omit to search all three.
- `limit` (int, optional, ≥1) — cap the result count.

### Output

`{datasets: [{id, title, description, store, variables, …}], total_count: int}`. The slim per-record shape is ~hundreds of bytes; the full catalogue across stores fits in ~30k tokens.

### Errors

- `ValidationError` — limit ≤ 0.

## `cds_describe_dataset`

Return the full STAC item for one dataset.

### Inputs

- `dataset_id` (string, required) — id from search results (e.g. `reanalysis-era5-single-levels`).

### Output

Slim STAC item: `{id, description, extent, keywords, license, links, providers, sci:doi, store, available_inputs, …}`. Note `title` lives on search summaries, not full items.

`available_inputs` lists every parameter the dataset accepts and the valid values for each (`data_format: [netcdf, grib]`, `download_format: [zip, unarchived]`, the canonical `variable` enum, …). For the ARCO `*-timeseries` products (e.g. `reanalysis-era5-land-timeseries`) it also carries a synthetic `location: {latitude, longitude}` entry — those products take a single point, not an `area`/grid, and can emit `data_format: csv`. Compose your `cds_submit_request` using only these field names and values; the legacy `format: …` key was deprecated by the new CDS processes engine and is silently rejected by the server.

**Caveat — `available_inputs` is a snapshot.** It ships in the bundled catalogue refreshed manually (every few weeks). CDS occasionally rotates field names / adds new constraints. If a submit composed from this snapshot fails with `remote_job_failed`, call `cds_apply_constraints(dataset_id, inputs={})` for the LIVE server-side valid values verbatim — they are the canonical source of truth. (Mind that tool's own caveats: for some datasets the constraints response is a flat, non-narrowing union, so the real submit remains the decisive validator.)

### Errors

- `ValidationError` — blank id.
- `NotFoundError` — id not in the bundled snapshot.

## `cds_apply_constraints`

Server-side narrowing: given a PARTIAL CDS / ADS / EWDS request, return the remaining valid values for unfilled fields. Hits the live constraints endpoint of the dataset's home store.

Use this to compose a submit request step-by-step instead of guessing field names / enum values. The bundled `cds_describe_dataset → available_inputs` shows the static top-level choices; this tool returns the LIVE narrowing as you fill the partial request. Particularly useful when:

- You don't know whether a field accepts the legacy `format` or modern `data_format` / `download_format` keys (this endpoint uses the modern keys exclusively).
- You picked a `variable` and want to know whether it accepts a time range (auxiliary time-invariant variables like EFAS `elevation` will NOT return `hyear/hmonth/hday/time` in the response — that's the signal to drop those fields).
- You want to confirm a dataset's required additional fields (e.g. EFAS v5.0 needs `hydrological_model: [lisflood]`).

**Caveats (observed on live datasets):**

- Some datasets return a **non-narrowing flat union**: `valid_remaining` echoes the full vocabulary no matter what you pin, and values coupled to your selection may be silently omitted. Per-field membership does **not** guarantee the combination is retrievable.
- Cross-field couplings (e.g. a spatial grid that only exists for certain temporal aggregations) surface only when **all related axes are pinned together**. When probing whether a combination exists, pin variable + product type + spatial + temporal + version in the same call — narrowing one field at a time can pass on every step and still fail at submit.
- This live endpoint and the bundled `cds_describe_dataset` snapshot can be different vintages; for vocabulary this endpoint wins. Either way the **real submit is the decisive validator** — treat a passing constraints probe as advisory, not as proof.

### Inputs

- `dataset_id` (string, required) — CDS / ADS / EWDS dataset id.
- `inputs` (dict, optional) — partial request dict. Empty `{}` returns top-level valid values for every field. Each subsequent narrowing call appends to `inputs`.

### Output

`{dataset_id, store, inputs_provided, valid_remaining}`. `inputs_provided` is the partial echo (for traceability); `valid_remaining` maps each remaining field to its valid values given the partial selection.

### Errors

- `ValidationError` — blank id, malformed `inputs` shape.
- `NotFoundError` — `dataset_id` not exposed by the constraints endpoint (HTTP 404; usually means the dataset moved between stores even if the bundled snapshot still lists it).
- `NetworkError` / `TimeoutError` — store endpoint unreachable.
- `BackendError` — non-404 4xx/5xx from the server, or a non-JSON response body.

Note: `cds_apply_constraints` is read-only and does **not** require credentials. It will work without a CDS PAT.

## `cds_estimate_request`

Size + queue estimator. Combines the CDS `/costing` pre-flight (the server's own
cost units and the dataset's per-request cost limit) with a calibrated or curated
bytes-per-unit factor, and learns from every completed download.

### Inputs

- `dataset_id` (string, required).
- `inputs` (dict, required) — cdsapi-shaped retrieve dict (`variable`, `year`, `month`, `day`, `time`, `area`, `pressure_level`, …). **`area` ordering**: CDS uses `[north, west, south, east]` (NWSE), opposite of common GIS `[w, s, e, n]`. Sending the wrong order does not error — it silently retrieves the wrong region.

### Output

`{estimated_size_bytes, estimated_size_human, epistemic_status, cost, fields_count, queue_latency_tier, runtime_compatible, advisory_message}` (plus `calibration_observations` when calibrated).

- **`estimated_size_bytes` / `estimated_size_human` may be `null`** — for whole-file products whose size cannot be derived from request shape, the estimate is honestly reported as unknown rather than an invented number. The first successful retrieval calibrates it.
- `epistemic_status`: `calibrated` (learned from prior downloads of this request shape), `curated_approximate` (±50%), `default_heuristic` (unknown dataset, ±10×), or `unknown` (whole-file, size unknowable until first retrieval).
- `cost`: `{units, limit, exceeds_limit, source}` from the costing endpoint, or `null` if it was unreachable. **`exceeds_limit: true` means the server will reject the request** — narrow it (split along year, then month).
- `queue_latency_tier` (`light` / `medium` / `heavy`) is cost-unit / field-count driven per research §6.5.4. `runtime_compatible` is `true` iff the dataset is in the bundled catalogue snapshot — treat `false` as a strong hint submit will 404.

### Errors

- `ValidationError` — missing required key in `inputs`, malformed structure.

## `cds_submit_request`

Queue a retrieve. Returns immediately after the server acknowledges; downloads happen via `cds_download_request_result` once status reaches `successful`.

**Unknown input keys are rejected up front.** The CDS server accepts keys a dataset does not use and silently ignores them — the request then delivers the wrong selection, or fails minutes later with an empty log. Keys are validated against the dataset's known input set (the same source as `cds_describe_dataset`'s `available_inputs`, which includes `area`/`data_format`/… where the dataset genuinely accepts them). A dataset missing from the bundled snapshot is not checked; pass `__options.skip_input_validation=true` if a key is newly added upstream and the snapshot is stale.

**Multi-model requests fan out on `projections-cmip6` and `projections-cordex-domains-single-levels`.** Those datasets execute ONE model per request — a list is accepted by the API but only the first model is delivered, silently. A request naming several models (for CORDEX: several `gcm_model`/`rcm_model` values) therefore always becomes a chunked parent with one part per model combination, and every downloaded part is verified to actually contain its requested model before it is cached (`delivered_content_mismatch` otherwise). Other datasets are untouched.

### Inputs

- `dataset_id`, `inputs` — same as `cds_estimate_request`.
- `confirmed` (bool, default false) — bypass the size + queue-tier confirmation gate; also accepts an auto-chunk split and clears the first fan-out tier.
- `confirm_large_fanout` (bool, default false) — second, deliberate ack for a very large auto-chunk split (more parts than `cds_auto_chunk_reconfirm_above`); required *in addition to* `confirmed`.

### Output (queued)

`{status: "queued", request_id, cache_key, result: {uri: "copernicus://jobs/<request_id>"}}`. `request_id` is a canonical UUID assigned by the CDS server.

### Output (cache hit — idempotent re-submit)

`{status: "successful", cache_hit: true, request_id, cache_key, result: {filepath, uri, metadata, provenance}}`.

### Confirmation flow

The gate fires when **either** the estimated bytes exceed `cds_per_request_size_warning_gb` (default 1 GB) **or** the queue tier is `medium` / `heavy` (cardinality-driven). The first call returns:

```jsonc
{"confirmation_required": true,
 "reason": "estimated_size_threshold_exceeded",
 "estimated_size_gb": 3.2,
 "next_action": "call cds_submit_request with confirmed=true to proceed",
 "context": {"queue_latency_tier": "medium", "fields_count": 372}}
```

Resubmit with `confirmed: true` to proceed.

### Auto-chunking (requests over the dataset cost limit)

CDS enforces a per-request cost limit (a field-count budget returned by the server's pre-flight). A request above that limit is rejected with HTTP 403 ("cost limits exceeded"). Rather than failing, `cds_submit_request` proposes a split along the calendar axis and runs the parts as one logical workflow.

When an over-limit request can be split — it has a list-valued `year`, and optionally `month` / `day` — the first call returns a proposal instead of queueing:

```jsonc
{"confirmation_required": true,
 "reason": "cost_limit_requires_chunking",
 "chunked": true,
 "estimated_cost": {"type": "cds_cost_units", "cost_units": 1827, "cost_limit": 400},
 "chunking": {"suggested_granularity": "year",
              "min_chunks": {"year": 5, "month": 5, "day": 60}},
 "next_action": "re-submit ... with chunk_by set to year/month/day (or confirmed=true for year)"}
```

Choose a granularity and re-submit with `chunk_by` set to `year`, `month`, or `day` (or `confirmed: true` to accept the suggested `year`). The server validates each chunk's cost, creates a parent workflow, and submits **all** the child jobs at once — CDS queues any excess, so there is no inflight throttle. The response is the parent:

```jsonc
{"status": "queued", "request_id": "<parent-id>", "cache_key": "...",
 "chunked": true, "chunk_count": 5,
 "result": {"uri": "copernicus://jobs/<parent-id>"}}
```

The parent `request_id` is a single handle: poll it with `cds_check_request_status`, download all parts with `cds_download_request_result`, cancel the whole set with `cds_cancel_request`. The parts advance automatically on each poll until every chunk completes.

**Large fan-outs require confirmation.** Because all parts submit at once, a big split launches many CDS jobs together. A validated plan with more than `cds_auto_chunk_confirm_above` chunks (default 30) and no `confirmed` returns `{"confirmation_required": true, "reason": "auto_chunk_job_count", "chunk_count": N, …}`; re-submit with `confirmed: true` to proceed (a human should approve a large batch). A plan over `cds_auto_chunk_reconfirm_above` (default 100) demands a **second, deliberate** ack (`reason: "auto_chunk_job_count_large"`): `confirmed: true` alone is not enough — also pass `confirm_large_fanout: true`, so a glitched agent blanket-setting `confirmed` cannot launch a runaway batch. A plan over `cds_auto_chunk_max_chunks` (default 366) is rejected outright (`too_many_chunks`) — no confirm bypasses it. All three thresholds are configurable under `budget.*`.

A request that cannot be split (no list-valued calendar axis, or a single calendar cell already over the limit) is rejected with a `ValidationError` suggesting a manual narrowing. Auto-chunking is on by default; disable it per request with `auto_chunk: false`, or globally with `budget.cds_auto_chunk_enabled: false`. The split axes must hold plain calendar tokens (a non-numeric value on `year` / `month` / `day` disables chunking for that request).

### Errors

- `ValidationError`, `AuthError` — standard.
- `BackendError` with `error_subclass="remote_job_failed"` — the CDS server marked the job failed. `context.backend_diagnostics` carries the structured server-side job state (status, error.code/message, attempt, queue metadata) for debugging. `next_action_hint` mentions the empirical CDS concurrent-quota pattern (~5-6 active jobs/user; excess reaped after ~5 min) so callers know to serialise submits rather than retry-storm.
- `TermsNotAcceptedError` — surfaced when the CDS / ADS / EWDS server returns HTTP 403 + "user didn't accept all required site policies". `recovery_url` points at the first missing licence page on the correct store host (T-CDS-011.1 — routing picks the right endpoint per dataset's `store`). `context.missing_policies` enumerates all of them. Open each URL, accept, and re-submit the same request.

### Cross-store routing

Submit / poll / cancel **and the download triggered by a successful poll** are routed automatically to the right Data Store endpoint based on each dataset's `store` field in the bundled catalogue snapshot. (`cds_download_request_result` itself is cache-only — by the time you call it the file is already on disk; it does not hit the network.)

- CDS datasets → `https://cds.climate.copernicus.eu/api`
- ADS datasets → `https://ads.atmosphere.copernicus.eu/api`
- EWDS datasets → `https://ewds.climate.copernicus.eu/api`

A single PAT works across all three stores (research §6.8.2). The catalogue tags every record with `store`, so callers do not need to remember which Data Store a dataset id lives in.

## `cds_check_request_status`

Look up the workflow row for a request_id — or several at once.

### Inputs

Exactly one of:

- `request_id` (string) — single mode; the envelope below.
- `request_ids` (list of strings, each non-empty) — batch mode: poll several requests in one call with bounded concurrency (at most 4 in flight). Returns `{"results": [...], "count": N}` preserving input order. A bad id yields an inline `{"request_id": "<that id>", "error": {...}}` entry and does not fail the batch. One call for a 21-part window instead of 21 separate invocations.

### Output

`{status, request_id, submitted_at, updated_at, cache_key, result, error_details}`. Status is one of `queued` / `running` / `successful` / `failed` / `cancelled`. On `successful` the file lives in the canonical cache and `result` carries `{filepath, uri, metadata, provenance}` — the same shape as `cds_download_request_result`. On `failed` `error_details` carries the canonical error record.

`result.metadata.content_type` is `application/x-netcdf` / `application/x-grib` / `application/zip` / `text/csv` / `application/octet-stream`, derived from the actual bytes on disk (the cached filename's extension reflects the real format, not whatever the inputs requested — see `cds_download_request_result`).

### Output (downloading)

When the CDS job has finished server-side and the result file is being fetched, `check_status` returns `{status: "running", phase: "downloading"}` **immediately** rather than blocking on the transfer — the download runs in the background, so the agent stays free. Keep polling; a later poll returns `successful` with the `filepath`. (A small file may finish within a brief inline grace and return `successful` in one poll, so `phase` is absent there.)

> **Transfers are resumable across polls and processes.** The staged download survives the process that started it: if a poll (or the whole server) exits mid-transfer, the partial bytes are kept, and the next `check_status` for the same request appends from where the previous one stopped instead of restarting from zero. Even short-lived one-shot pollers — a fresh CLI invocation per poll — therefore make forward progress on a large file; each poll accumulates more bytes. Still, when you see `phase: "downloading"` the slow part (the queue wait) is over, and one long-running `copernicus-mcp cds wait <request_id>` finishes the transfer fastest — in a single uninterrupted pass rather than poll-sized instalments. Abandoned partial transfers are cleaned up automatically after a week; set `budget.cds_resume_downloads: false` to restore per-attempt throwaway staging.

### Output (chunked parent)

A request that was auto-split returns the aggregate instead:

```jsonc
{"status": "running", "request_id": "<parent-id>", "chunked": true, "chunk_count": 5,
 "progress": {"completed": 2, "total": 5},
 "chunks": {"total": 5, "successful": 2, "running": 1, "downloading": 1,
            "retrying": 0, "queued": 1, "failed": 0, "cancelled": 0},
 "per_chunk": [{"index": 0, "request_id": "<child-id>", "status": "successful"},
               {"index": 1, "request_id": "<child-id>", "status": "downloading",
                "phase": "downloading"}, "..."]}
```

`status` is the aggregate: `successful` once every chunk completes, `failed` if any chunk fails for a reason that will not resolve on its own (the remaining in-flight chunks are then cancelled), `cancelled` if you cancel the parent. Each poll advances the workflow — it finalises completed children and submits the next ones, so poll the parent until it reaches a terminal state. The individual `per_chunk[].request_id` values are also pollable on their own.

The `chunks` counts **partition** the parts: every part is in exactly one bucket, so they sum to `total`.

**Parts are submitted a few at a time,** not all at once — the Climate Data Store throttles an account to a small number of concurrent jobs and refuses the excess rather than queueing it. Each poll tops the level back up. A large split therefore completes over several waves; this is expected, not a stall. See `docs/setup.md` for the pacing knobs.

**A part refused for capacity is re-submitted automatically** (a bounded number of times, spaced out) and shows as `retrying`; one refused part no longer destroys the whole retrieval. A part that failed because the *request* is wrong is never retried — it fails the parent promptly, because retrying a malformed request is only a slower way to fail.

**`phase: "downloading"` on the parent** means every part has finished server-side and only local file transfers remain. Transfers resume across polls (see above), so continued polling does finish them part by part — but this is the natural moment to hand the job to one long-running `copernicus-mcp cds wait`, which completes the remaining transfers in a single uninterrupted pass.

**A failed parent still gives you what landed.** It carries `partial_result: {files, chunk_indices, missing_chunk_indices}` with descriptors for the parts that did complete — deliberately outside `result`, so a failed parent never looks like a complete delivery. Every part is a first-class request id, so `cds_download_request_result(<per_chunk request_id>)` also resolves a completed part's file directly.

**On a successful parent, check `result.complete`.** It is `false` when a part's file has since been evicted from the cache: the `files` list is then short, `evicted_chunk_indices` names the gaps, and `recovery_hint` explains how to refill them (re-run with `force_refresh`). `cds_download_request_result` refuses outright on that state rather than handing back a partial set.

When `status` is `successful`, the response already carries the full multi-file `result` (the same `files` / `merge_hint` set that `cds_download_request_result` returns) — the chunk files were downloaded during polling, so there is no need to call download separately. If a chunk file has since been evicted, its index is listed in `result.evicted_chunk_indices`.

### Errors

- `NotFoundError` — request_id not known locally.

## `cds_download_request_result`

Fetch a completed result from the canonical cache. Returns a file descriptor — never inline bytes.

### Inputs

- `request_id` (string, required).
- `target` (string, optional) — reserved for future use. The CDS backend stores files under the canonical cache key regardless of this value.

### Output

`{request_id, cache_key, status: "successful", cache_hit: true, result: {filepath, uri, metadata, provenance}}`.

`metadata` carries `{size_bytes, content_type}`. The cached filename ends in `.nc` / `.grib` / `.zip` / `.bin` reflecting the real bytes on disk:

- The backend derives an initial extension from the submit `inputs` (`download_format: zip` → `.zip`; `data_format: netcdf | netcdf3 | netcdf4 | netcdf_legacy` → `.nc`; `data_format: grib | grib1 | grib2` → `.grib`; `data_format: csv` → `.csv`).
- After download a magic-byte sniff overrides the extension if the actual content disagrees (e.g. ECMWF wraps multi-variable NetCDF requests in a ZIP — the cached file lands as `.zip` regardless of what the agent asked for, and `content_type` reports `application/zip`). Trust the on-disk extension and the reported `content_type`, not the original request shape.

### Output (chunked parent)

A successful chunked parent returns a descriptor **set** — one file per chunk, ordered by chunk index — not a single recombined file:

```jsonc
{"status": "successful", "request_id": "<parent-id>", "chunked": true, "chunk_count": 5,
 "result": {"files": [{"chunk_index": 0, "filepath": "...", "size_bytes": 12345,
                       "content_type": "application/x-grib", "span": {"year": ["2020"]}}, "..."],
            "formats": ["application/x-grib"],
            "heterogeneous_formats": false,
            "merge_hint": "..."}}
```

The server does not stitch or re-encode the parts — merging is the consumer's job (for example `xarray.open_mfdataset(sorted_filepaths, combine="nested", concat_dim="time")`). The files are non-overlapping and ordered by `chunk_index`. If `heterogeneous_formats` is `true`, CDS returned different formats for different chunks (it occasionally zips one); convert them to a single format before merging. If a chunk's cached file was evicted, download raises `CacheError` (`cache_eviction`) naming the missing chunk indices — re-submit with `force_refresh: true` to repopulate the whole set. A parent that is not yet `successful` raises a `BackendError` (`result_not_ready`) whose `context.partial_files` lists the chunks already done.

### Errors

- `NotFoundError` — request_id not known or workflow row missing.
- `BackendError` — workflow not yet `successful`, or file was evicted from the cache (rare; re-submit to repopulate).

## `cds_cancel_request`

Cancel a queued or running request. Idempotent on already-terminal rows.

### Inputs

- `request_id` (string, required).

### Output

`{cancelled: bool, request_id, status}`. `status` is the final workflow status; `cancelled: false` on already-terminal rows.

### Errors

- `NotFoundError` — request_id not known locally.

### Notes

Cancellation is best-effort: the server may have processed the request between submit and cancel. The cancel call returns successfully in that race window with `status: "successful"`.

Cancelling a chunked parent stops the plan and cascades to its children: in-flight chunks are cancelled (best-effort), already-completed chunks keep their files. The parent ends `cancelled`.

> Two deliberate asymmetries of the key check: `cds_estimate_request` is NOT key-checked (an estimate is free and harmless; the submit is where a silently-ignored key costs you a wrong delivery), and the check runs BEFORE the cache lookup — a previously cached request whose keys a snapshot refresh now flags returns the validation error rather than the stale cache hit, because the cached file was produced by a request the server had silently mis-interpreted.

## `cds_list_licences`

The store's licence catalogue plus the account's already-accepted set. Use it when a submit fails with `TermsNotAcceptedError`: find the missing licence's `id` and `revision` here, then accept it (in-band via `cds_accept_licence` if the operator enabled it, otherwise via the web URL in the error's `recovery_url` or the CLI `copernicus-mcp cds accept-licence`).

### Inputs

- `dataset_id` (string, optional) — routes the call to the right store (CDS / ADS / EWDS); licence ids are per-store.

### Output

`{store, available: [{id, revision, label, scope, contents_url}], accepted: [...]}`. Only these known fields are passed through; both lists are sanitised.

### Errors

- `AuthError` — no CDS credentials resolve (listing reads the account's accepted set).
- `NetworkError` / `BackendError` — store profile API unreachable / errored.

## `cds_accept_licence`

Accept a dataset licence **on behalf of the account owner**. Registered only when the operator opts in with `budget.cds_licence_accept_enabled: true` — acceptance is a legally binding act, so the agent-visible surface is off by default (the CLI `copernicus-mcp cds accept-licence` works regardless: the CLI is the operator's own hands).

### Inputs

- `licence_id` (string, required) — from `cds_list_licences`.
- `revision` (integer ≥ 0, strict, required) — the revision `cds_list_licences` reports.
- `dataset_id` (string, optional) — store routing.

### Output

`{accepted: true, licence_id, revision, store}`.

### Errors

- `ValidationError` — missing/blank `licence_id`, non-integer or negative `revision`.
- `AuthError`, `NetworkError`, `BackendError` — credential / transport / server failures.
