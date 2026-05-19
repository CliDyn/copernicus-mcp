# MCP Tool Reference

`copernicus-mcp` registers a diagnostic plus per-backend tool surfaces:

- **Diagnostic** (always registered): `copernicus_mcp_status`.
- **CMEMS** (eleven tools, registered when the `cmems` backend is enabled): `marine_search_groups`, `marine_search_products`, `marine_search_datasets`, `marine_describe_dataset`, `marine_get_coordinates`, `marine_estimate_subset`, `marine_subset_dataset`, `marine_list_files`, `marine_get_files`, `marine_check_status`, `marine_cancel_subset`. The first three implement the three-step hierarchical pipeline: start with `marine_search_groups` for free-text routing, drill into `marine_search_products` with the chosen `group_ids`, then resolve datasets via `marine_search_datasets` with the chosen `product_ids` (plus optional `bbox` / `time_range`). The pipeline is the default path for any agentic query; the bare `marine_search_datasets` (`keyword=` only) flat path stays available for known dataset ids.
- **CDS / ADS / EWDS** (eight tools, registered when the `cds` backend is enabled AND credentials resolve): `cds_search_datasets`, `cds_describe_dataset`, `cds_apply_constraints`, `cds_estimate_request`, `cds_submit_request`, `cds_check_request_status`, `cds_download_request_result`, `cds_cancel_request`.

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
    }
  },
  "cache": {
    "directory": "/Users/you/.cache/copernicus-mcp",
    "size_bytes": 0,
    "entry_count": 0
  },
  "persistence": {
    "database_path": "/Users/you/.local/state/copernicus-mcp/state.db"
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

## `marine_search_groups`

First step of the hierarchical search pipeline. Shortlist CMEMS routing groups for a free-text query. Each group bundles related products by region, domain, and intent (e.g. `physics-mediterranean-state`, `ocean-acidification-monitoring`, `climate-reanalysis`, `arctic-comprehensive`).

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
| `bbox`          | `[min_lon, min_lat, max_lon, max_lat]` | no | `null` | Spatial filter. Hierarchical path keeps only cards whose `spatial_extent` overlaps the bbox; null-extent cards are excluded. Antimeridian-crossing bboxes (min_lon > max_lon) are rejected with `ValidationError` per the antimeridian rejection rule — split into two non-crossing bboxes. |
| `time_range`    | `[start_iso, end_iso]`          | no       | `null`  | Temporal filter. ISO-8601 strings; `start < end` required. Hierarchical path keeps only cards whose `temporal_extent` overlaps; null-extent cards are excluded. |
| `service_types` | `array<enum>`                   | no       | `null`  | Still rejected with `ValidationError` (filter not yet implemented).        |
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

- `ValidationError` — `limit < 1`; `service_types` set (unimplemented); `bbox` with wrong shape or antimeridian-crossing; `time_range` with non-ISO entries or `start >= end`; both `product_id` and `product_ids` set.
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
| `minimum_depth`                 | `float (>=0)`         | yes      | —           | Shallowest depth in metres.                                                  |
| `maximum_depth`                 | `float (>=0)`         | yes      | —           | Deepest depth in metres.                                                     |
| `start_datetime`                | `string` (ISO 8601 UTC) | yes    | —           | Inclusive start. Naive datetimes rejected.                                   |
| `end_datetime`                  | `string` (ISO 8601 UTC) | yes    | —           | Strictly after `start_datetime`.                                             |
| `coordinates_selection_method`  | enum                  | no       | `inside`    | `inside` \| `strict-inside` \| `nearest` \| `outside`.                       |
| `service`                       | `string \| null`      | no       | `null`      | Force a specific CMEMS service (`geoseries`, `timeseries`, …).               |
| `file_format`                   | enum                  | no       | `netcdf`    | `netcdf` \| `zarr`.                                                          |
| `netcdf_compression_level`      | `integer [0, 9]`      | no       | `1`         | NetCDF deflate level.                                                        |

### Output

```jsonc
{
  "filepath": "/Users/you/.cache/copernicus-mcp/cmems/<cache_key>/data.nc",
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

`available_inputs` lists every parameter the dataset accepts and the valid values for each (`data_format: [netcdf, grib]`, `download_format: [zip, unarchived]`, the canonical `variable` enum, …). Compose your `cds_submit_request` using only these field names and values; the legacy `format: …` key was deprecated by the new CDS processes engine and is silently rejected by the server.

**Caveat — `available_inputs` is a snapshot.** It ships in the bundled catalogue refreshed manually (every few weeks). CDS occasionally rotates field names / adds new constraints. If a submit composed from this snapshot fails with `remote_job_failed`, call `cds_apply_constraints(dataset_id, inputs={})` for the LIVE server-side valid values verbatim — they are the canonical source of truth.

### Errors

- `ValidationError` — blank id.
- `NotFoundError` — id not in the bundled snapshot.

## `cds_apply_constraints`

Server-side narrowing: given a PARTIAL CDS / ADS / EWDS request, return the remaining valid values for unfilled fields. Hits the live constraints endpoint of the dataset's home store.

Use this to compose a submit request step-by-step instead of guessing field names / enum values. The bundled `cds_describe_dataset → available_inputs` shows the static top-level choices; this tool returns the LIVE narrowing as you fill the partial request. Particularly useful when:

- You don't know whether a field accepts the legacy `format` or modern `data_format` / `download_format` keys (this endpoint uses the modern keys exclusively).
- You picked a `variable` and want to know whether it accepts a time range (auxiliary time-invariant variables like EFAS `elevation` will NOT return `hyear/hmonth/hday/time` in the response — that's the signal to drop those fields).
- You want to confirm a dataset's required additional fields (e.g. EFAS v5.0 needs `hydrological_model: [lisflood]`).

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

Heuristic byte-size estimator + queue-tier classification.

### Inputs

- `dataset_id` (string, required).
- `inputs` (dict, required) — cdsapi-shaped retrieve dict (`variable`, `year`, `month`, `day`, `time`, `area`, `pressure_level`, …). **`area` ordering**: CDS uses `[north, west, south, east]` (NWSE), opposite of common GIS `[w, s, e, n]`. Sending the wrong order does not error — it silently retrieves the wrong region.

### Output

`{estimated_size_bytes, estimated_size_human, fields_count, queue_latency_tier, epistemic_status, runtime_compatible, advisory_message}`. `queue_latency_tier` is one of `light` / `medium` / `heavy` — driven by field count per research §6.5.4 (server pulls fields independently from tape). `epistemic_status` is `curated_approximate` (dataset in the curated bytes-per-field map, ±50%) or `default_heuristic` (unknown dataset, fallback to 2 MB/field, ±10×). `runtime_compatible` is `true` iff the dataset id is in the bundled CDS / ADS / EWDS catalogue snapshot — agents should treat `false` as a strong hint that submit will likely 404.

### Errors

- `ValidationError` — missing required key in `inputs`, malformed structure.

## `cds_submit_request`

Queue a retrieve. Returns immediately after the server acknowledges; downloads happen via `cds_download_request_result` once status reaches `successful`.

### Inputs

- `dataset_id`, `inputs` — same as `cds_estimate_request`.
- `confirmed` (bool, default false) — bypass the size + queue-tier confirmation gate.

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

### Errors

- `ValidationError`, `AuthError` — standard.
- `BackendError` with `error_subclass="remote_job_failed"` — the CDS server marked the job failed. `context.backend_diagnostics` carries the structured server-side job state (status, error.code/message, attempt, queue metadata) for debugging. `next_action_hint` mentions the empirical CDS concurrent-quota pattern (~5-6 active jobs/user; excess reaped after ~5 min) so callers know to serialise submits rather than retry-storm.
- `TermsNotAcceptedError` — surfaced when the CDS / ADS / EWDS server returns HTTP 403 + "user didn't accept all required site policies". `recovery_url` points at the first missing licence page on the correct store host (routing picks the right endpoint per dataset's `store`). `context.missing_policies` enumerates all of them. Open each URL, accept, and re-submit the same request.

### Cross-store routing

Submit / poll / cancel **and the download triggered by a successful poll** are routed automatically to the right Data Store endpoint based on each dataset's `store` field in the bundled catalogue snapshot. (`cds_download_request_result` itself is cache-only — by the time you call it the file is already on disk; it does not hit the network.)

- CDS datasets → `https://cds.climate.copernicus.eu/api`
- ADS datasets → `https://ads.atmosphere.copernicus.eu/api`
- EWDS datasets → `https://ewds.climate.copernicus.eu/api`

A single PAT works across all three stores (research §6.8.2). The catalogue tags every record with `store`, so callers do not need to remember which Data Store a dataset id lives in.

## `cds_check_request_status`

Look up the workflow row for a request_id.

### Inputs

- `request_id` (string, required).

### Output

`{status, request_id, submitted_at, updated_at, cache_key, result, error_details}`. Status is one of `queued` / `running` / `successful` / `failed` / `cancelled`. On `successful` the file lives in the canonical cache and `result` carries `{filepath, uri, metadata, provenance}` — the same shape as `cds_download_request_result`. On `failed` `error_details` carries the canonical error record.

`result.metadata.content_type` is `application/x-netcdf` / `application/x-grib` / `application/zip` / `application/octet-stream`, derived from the actual bytes on disk (the cached filename's extension reflects the real format, not whatever the inputs requested — see `cds_download_request_result`).

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

- The backend derives an initial extension from the submit `inputs` (`download_format: zip` → `.zip`; `data_format: netcdf | netcdf3 | netcdf4 | netcdf_legacy` → `.nc`; `data_format: grib | grib1 | grib2` → `.grib`).
- After download a magic-byte sniff overrides the extension if the actual content disagrees (e.g. ECMWF wraps multi-variable NetCDF requests in a ZIP — the cached file lands as `.zip` regardless of what the agent asked for, and `content_type` reports `application/zip`). Trust the on-disk extension and the reported `content_type`, not the original request shape.

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
