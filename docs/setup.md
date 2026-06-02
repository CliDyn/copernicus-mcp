# Setup

This document covers everything you need to run `copernicus-mcp` against the live Copernicus services: credentials, configuration files, where state and cache live, how to enable verbose logging, and the most common things that go wrong.

Two backends are shipped: `cmems` (Copernicus Marine Service) and `cds` (Climate Data Store family — CDS / ADS / EWDS share one PAT). **Both are enabled by default.** Trim the tool surface (and the LLM context cost) by setting `COPERNICUS_MCP_ENABLED_BACKENDS=cmems` (or `=cds`), or override `enabled_backends` in `config.yaml`. CDS tools auto-skip registration when no PAT is configured, so leaving the default on is safe even if you only have CMEMS credentials.

### CMEMS hierarchical search (T-CMEMS-HIER-005)

The CMEMS backend ships a three-level routing hierarchy on top of the bundled catalogue snapshot:

1. `groups.json` (~47 entries) — human-curated routing groups (region × domain × intent).
2. `products.json` (~306 entries) — one entry per CMEMS product with union axes.
3. `dataset_cards.json` (~1251 entries) — full enriched dataset cards (domain, region, data_type, variables_normalized, best_for, not_good_for, spatial_label, temporal_label, quality_flags).

Use the hierarchical pipeline when the query is free-text or open-ended ("Arctic sea ice extent", "Mediterranean salinity", "ocean acidification"): call `marine_search_groups(query)` → `marine_search_products(group_ids)` → `marine_search_datasets(product_ids, [bbox], [time_range])`. Each step returns the standard `{selected, rejected, reason, confidence, fallback_available}` envelope. When `confidence == "low"`, the response sets `fallback_available: true`; drop back to the flat `marine_search_datasets(keyword=...)` path.

Use the flat path when the dataset id pattern is already known: `marine_search_datasets(keyword="...", product_id="...")` returns the smaller slim-catalogue envelope. The flat path also supports `live=true` to hit the live SDK for products newer than the bundled snapshot; the hierarchical path is always offline (the cards manifest is the source of truth).

## Credentials for CMEMS

The `cmems` backend authenticates against the Copernicus Marine Service. Create a free account at <https://data.marine.copernicus.eu/register> if you do not have one. The same username and password work for the toolbox, the web portal, and `copernicus-mcp`.

**Which operations need credentials?**

- `marine_search_datasets` — **no credentials needed** by default. Search reads a bundled catalogue snapshot shipped inside the wheel. Credentials are only required when you opt in to the live SDK call with `live=true`.
- `marine_check_status`, `marine_cancel_subset` — **no credentials needed**. Both operate on the local workflow database; they look up a request id you obtained from a prior `marine_subset_dataset` submission and either return its current state or mark it cancelled. The Copernicus Marine Service is never contacted.
- `marine_describe_dataset`, `marine_estimate_subset`, `marine_subset_dataset` — all require credentials. These call the live Copernicus Marine Service.

If you only need dataset discovery, you can run `copernicus-mcp` without configuring credentials at all. They become mandatory the moment you describe, estimate, or download.

Three resolution sources are supported, in this precedence order:

1. **Explicit override** passed in code (used by tests and by future programmatic embedders).
2. **Secret manager** (a placeholder hook in Iteration 1; we do not ship a provider, but the resolver will call one if you wire it in).
3. **Environment variables** — the recommended option for everyday use.
4. **Credentials file** — `~/.copernicusmarine/.copernicusmarine-credentials`, the same file the official `copernicusmarine` CLI writes when you run `copernicusmarine login`.

The resolver returns the first source that produces a complete `(username, password)` pair. If none does, the backend is loaded but every operation that needs CMEMS credentials returns a structured `AuthError` with `recovery_action="configure_credentials"`.

### Option A — environment variables (recommended)

```bash
export COPERNICUSMARINE_SERVICE_USERNAME=your_user
export COPERNICUSMARINE_SERVICE_PASSWORD=your_pass
copernicus-mcp status            # verifies the backend is "configured"
```

The `copernicus-mcp status` command prints `credential_source: env` when this option is in effect. It never prints the password.

### Option B — toolbox credentials file

Run the official toolbox login once and `copernicus-mcp` will pick it up:

```bash
pip install copernicusmarine
copernicusmarine login           # writes ~/.copernicusmarine/.copernicusmarine-credentials
copernicus-mcp status            # credential_source: config_file
```

### Option C — secret manager

The resolver accepts a `SecretManagerProvider` that returns `{"username": ..., "password": ...}` for a given backend. Wire one up in your own embedder if your environment uses Vault, AWS Secrets Manager, GCP Secret Manager, or similar. We do not ship a provider in Iteration 1.

### What the resolver never does

- It never reads or writes credentials to the project state database, cache, logs, provenance records, or tool output. The `Sanitiser` runs as defence-in-depth on every outbound payload.
- It never falls back to anonymous access. Missing credentials produce a structured `AuthError`, not a half-broken request.

## Credentials for CDS / ADS / EWDS

The `cds` backend authenticates against the Climate Data Store family. A single Personal Access Token (PAT) — a canonical UUID — works across all three stores per ECMWF policy. Create a free account at <https://cds.climate.copernicus.eu/> if you do not have one, log in, open the user profile, and copy the "Personal Access Token".

Each new dataset requires accepting its licence once via the web UI. If you submit a request for a dataset whose licence you have not accepted, the backend surfaces a structured `TermsNotAcceptedError` with `recovery_url` pointing at the licence page — open the URL, accept, and re-submit.

Resolution sources, in precedence order:

1. **Environment variable** `CDSAPI_KEY=<your-uuid-pat>` in your shell profile (recommended).
2. **`~/.cdsapirc`** — the same file the official `cdsapi` CLI reads. Format:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <your-uuid-pat>
   ```
   Override the location with `CDSAPI_RC=/path/to/custom.cdsapirc` (mirrors the upstream cdsapi convention).

Advanced: `CDSAPI_URL` overrides the endpoint base URL. The default (`https://cds.climate.copernicus.eu/api`) routes CDS requests directly; the backend selects ADS / EWDS endpoints automatically based on dataset id, so most users never need to set this.

The resolver returns the first source that produces a complete PAT. Missing credentials yield a structured `AuthError` on every CDS operation.

### Enabling the backend

By default only `cmems` is enabled. Opt in to CDS either in your config file:

```yaml
enabled_backends: [cmems, cds]
```

…or via env var (overrides the config value):

```bash
export COPERNICUS_MCP_ENABLED_BACKENDS=cmems,cds
```

Verify with `copernicus-mcp status` — the `backends.cds` entry should report `configured: true` and `credential_source: env` or `config_file`.

## Configuration file

Every Pydantic field has a default; you only need a config file to override what does not fit your machine. The defaults are in [`src/copernicus_mcp/config/defaults.yaml`](../src/copernicus_mcp/config/defaults.yaml) and reproduced here for reference:

```yaml
server:
  name: copernicus-mcp
  log_level: INFO            # DEBUG | INFO | WARNING | ERROR
  transport: stdio

storage:
  # cache_directory and state_database are resolved per-OS via
  # platformdirs (Linux: ~/.cache/... + ~/.local/state/...; macOS:
  # ~/Library/Caches/... + ~/Library/Application Support/...; Windows:
  # %LOCALAPPDATA%/copernicus-mcp/Cache + %LOCALAPPDATA%/copernicus-mcp).
  # Uncomment to pin a custom path; otherwise leave to the OS default.
  # cache_directory: /custom/cache/path
  # state_database: /custom/state.db
  cache_size_limit_gb: 50.0
  cache_eviction_policy: lru

http:
  default_timeout_seconds: 60
  default_retry_max_attempts: 5
  default_retry_base_delay_seconds: 1.0
  default_retry_max_delay_seconds: 60.0

cache:
  search_results_ttl_seconds: 3600        # 1 hour
  metadata_ttl_seconds: 86400             # 24 hours

budget:
  cmems_max_concurrent_subset_operations: 2
  cmems_per_request_size_warning_gb: 1.0
  cmems_estimate_timeout_seconds: 30.0

observability:
  structured_logging: true
  log_format: json

enabled_backends:
  - cmems
```

### Where the config file lives

The loader reads the first existing file in this list, then layers env vars and explicit overrides on top:

1. `~/.config/copernicus-mcp/config.yaml`
2. `~/.copernicus-mcp.yaml`

Both paths are optional. If neither exists, the package defaults are used as-is.

### Selected env-var overrides

These are convenient one-line overrides without a config file:

| Environment variable                 | Maps to                       |
| ------------------------------------ | ----------------------------- |
| `COPERNICUS_MCP_LOG_LEVEL`           | `server.log_level`            |
| `COPERNICUS_MCP_CACHE_DIR`           | `storage.cache_directory`     |
| `COPERNICUS_MCP_STATE_DB`            | `storage.state_database`      |

Nested fields can also be set via `COPERNICUS_MCP_<SECTION>__<FIELD>` (double underscore between section and field), e.g. `COPERNICUS_MCP_STORAGE__CACHE_DIRECTORY=/tmp/copernicus-mcp-cache`.

## Cache and state directories

Two directories hold all persistent state:

- **`storage.cache_directory`** — downloaded NetCDF/Zarr files, plus their `.provenance.json` sidecars. One subdirectory per backend (`cache/cmems/`). Also holds the Layer 2 index cache (see below).
- **`storage.state_database`** — SQLite database with one row per workflow request, cache index entries, persisted provenance records, and acceptance events.

### `marine_indices/` — Layer 2 index cache

`marine_list_files` writes one Parquet file per dataset under `<cache_directory>/marine_indices/<dataset_id>.parquet`. First call per dataset pays an SDK round-trip; subsequent calls read the local Parquet and return in milliseconds.

**First-call latency** (one-time per dataset):

- **INSITU-BGC** (GLODAP, SOCAT): ~1–5 s.
- **CORA / EasyCORA**: ~210 s — the SDK enumerates ~1M file paths on a dry-run listing. The MCP call blocks for this duration; plan accordingly (run once at deploy time if you want zero perceived latency for users).

Both are operator-pre-warmable: run `marine_list_files(dataset_id, ...)` from a Python REPL once at deploy time and the Parquet will be in place when real users arrive.

Both are recreated on demand. To clear everything, target the per-OS paths:

```bash
# Linux (XDG)
rm -rf ~/.cache/copernicus-mcp ~/.local/state/copernicus-mcp

# macOS
rm -rf "~/Library/Caches/copernicus-mcp" \
       "~/Library/Application Support/copernicus-mcp"

# Windows (PowerShell)
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\copernicus-mcp"
```

To move them off the default location, set `COPERNICUS_MCP_CACHE_DIR` / `COPERNICUS_MCP_STATE_DB`, pass `--cache-dir` to the CLI, or pin `storage.cache_directory` / `storage.state_database` in `config.yaml`.

The cache enforces `cache_size_limit_gb` after every successful download via the `cache_eviction_policy` (LRU in Iteration 1). Layer 2 Parquet indices are small (sub-100 KB for INSITU; ~20–50 MB for CORA/EasyCORA) and not subject to LRU eviction in Iter 1 — they live under `marine_indices/` separately from the bundle cache.

## Logging

All logs go to **stderr**, always. Stdout is reserved for the MCP wire protocol when `copernicus-mcp serve` is running, so any log line on stdout would corrupt the connection.

To enable DEBUG output (useful when troubleshooting):

```bash
COPERNICUS_MCP_LOG_LEVEL=DEBUG copernicus-mcp marine search-datasets --keyword temp
```

By default logs are JSON-formatted, one record per line, with a `trace_id` field that ties every log entry from a single tool invocation together. Set `observability.log_format: console` in `config.yaml` to switch to a human-readable formatter for local development. Credentials never appear in logs (the value is replaced with `<set>` or `<unset>`).

## Claude Desktop integration

In your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "copernicus": {
      "command": "copernicus-mcp",
      "args": ["serve"],
      "env": {
        "COPERNICUSMARINE_SERVICE_USERNAME": "your_user",
        "COPERNICUSMARINE_SERVICE_PASSWORD": "your_pass"
      }
    }
  }
}
```

Restart Claude Desktop. The four `marine_*` tools and the `copernicus_mcp_status` diagnostic become available to the assistant. Tool results that wrap large data return a `filepath` plus metadata and provenance — never inline bytes.

## Troubleshooting

### Tool returns `AuthError`

Check that credentials are visible:

```bash
copernicus-mcp status            # backends.cmems.configured should be true
```

If `configured` is `false`, your env vars are not exported into the process running the server (a common Claude Desktop pitfall — restart the desktop client after editing the config), or the credentials file is missing or unreadable.

### Subset hangs or times out

The toolbox-level download is governed by `http.default_timeout_seconds` and the toolbox's own settings. CMEMS can be slow under load. If a request hangs:

1. Run the same call with `COPERNICUS_MCP_LOG_LEVEL=DEBUG` and watch for retry messages.
2. Reduce the bbox or time range — most hangs are very large requests close to the dataset extent.
3. Check the Copernicus Marine status page: <https://marine.copernicus.eu/news>.

The structured logger emits a `trace_id` per request so you can correlate the slow path across log lines.

### `CoverageUnavailableError`

The bbox or time range falls outside the dataset's actual coverage. Use `marine_describe_dataset` (or `copernicus-mcp marine describe DATASET_ID`) to inspect the spatial and temporal extent, then narrow the request.

### `ValidationError` with `recovery_action="modify_request_parameters"`

The request was structurally invalid (e.g. `minimum_latitude > maximum_latitude`, antimeridian-crossing bbox, or `start_datetime >= end_datetime`). Iteration 1 explicitly rejects antimeridian-crossing bboxes and asks you to split the request into two non-crossing halves. The `next_action_hint` field on the error record tells you exactly how.

### Confirmation prompt on every subset

Subsets above `budget.cmems_per_request_size_warning_gb` (default 1 GB) trigger a confirmation gate. To skip the prompt on a single call, pass `--yes` to the CLI. To raise the global threshold, set `budget.cmems_per_request_size_warning_gb` higher in `config.yaml`. There is no global "skip-all" flag — by design.
