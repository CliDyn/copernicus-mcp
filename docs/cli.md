# CLI Reference

The `copernicus-mcp` command-line tool exposes the same workflows as the MCP server, but for terminal users and shell scripts. The package installs a `copernicus-mcp` console script (defined in `pyproject.toml`); `python -m copernicus_mcp` works equivalently.

All commands accept a global `--json` flag that emits the raw orchestrator response on **stdout** (pipe-safe for `jq`). Without `--json` the output is a Rich table or panel. Diagnostics, error panels, confirmation prompts and the non-TTY abort message always go to **stderr**, so a `--json` consumer never sees mixed output.

## Global options

| Option              | Description                                                                                                                                                          |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--cache-dir PATH`  | Override the cache directory for this invocation. Wins over `COPERNICUS_MCP_CACHE_DIR` env var and the yaml/defaults layers (priority: CLI > env > yaml > defaults). |

Without `--cache-dir`, the cache lives in the OS user-cache location resolved by `platformdirs`: `~/.cache/copernicus-mcp` on Linux, `~/Library/Caches/copernicus-mcp` on macOS, `%LOCALAPPDATA%\copernicus-mcp\Cache` on Windows.

## Synopsis

```
copernicus-mcp [--help] [--cache-dir PATH] SUBCOMMAND ...
copernicus-mcp version
copernicus-mcp serve
copernicus-mcp status [--json]
copernicus-mcp marine search-groups   --query ...      [OPTIONS]
copernicus-mcp marine search-products --groups ...     [OPTIONS]
copernicus-mcp marine search-datasets                  [OPTIONS]
copernicus-mcp marine describe DATASET_ID              [--json]
copernicus-mcp marine estimate    --dataset ...        [OPTIONS]
copernicus-mcp marine subset      --dataset ...        [OPTIONS]
copernicus-mcp marine get-files   --dataset ...        [OPTIONS]
copernicus-mcp marine check-status REQUEST_ID          [--json]
copernicus-mcp jobs list                               [OPTIONS]
```

---

## `version`

Prints the installed `copernicus-mcp` version.

```bash
copernicus-mcp version
# 0.0.1
```

---

## `serve`

Starts the MCP server over stdio. The process reads JSON-RPC frames from stdin and writes them to stdout; the structured logger writes to stderr. Used as the `command` of an MCP client (Claude Desktop, IDEs, etc.).

```bash
copernicus-mcp serve
```

The server runs until SIGINT, SIGTERM, or stdin EOF. A clean shutdown closes the SQLite persistence handle in a `finally` block.

---

## `status`

Prints the server diagnostics block: enabled backends, credential sources (without values), cache size and entry count, persistence path, and a non-secret subset of the configuration. Identical content to the `copernicus_mcp_status` MCP tool.

| Option   | Description                                |
| -------- | ------------------------------------------ |
| `--json` | Emit the raw status object on stdout.      |

Example:

```bash
copernicus-mcp status --json | jq '.backends.cmems.configured'
# true
```

---

## Hierarchical search pipeline

For agentic / free-text discovery the recommended path is the three-step pipeline introduced in T-CMEMS-HIER-005:

```bash
# 1. Free-text → 1-3 groups
copernicus-mcp marine search-groups --query "arctic sea ice extent" --json

# 2. Group ids → ~10-20 candidate products
copernicus-mcp marine search-products --groups sea-ice-arctic --json

# 3. Product ids (+ optional bbox / time) → ≤50 enriched dataset cards
copernicus-mcp marine search-datasets \
    --product-ids SEAICE_ARC_PHY_L4_NRT_011_006 \
    --time 2024-01-01T00:00:00Z,2024-12-31T00:00:00Z --json
```

Each step returns the standard `{selected, rejected, reason, confidence, fallback_available}` envelope. When `confidence == "low"`, the response sets `fallback_available: true` — drop back to the flat `marine search-datasets --keyword ...` path.

The flat `marine search-datasets --keyword ...` path is still available and is what to use when you already know the dataset id pattern; the hierarchical path returns enriched cards (domain / region / data_type / variables_normalized / best_for / not_good_for / spatial_label / temporal_label) which the flat path does not.

## `marine search-groups`

Shortlist routing groups for a free-text query. Offline-only — reads the bundled `groups.json`. No credentials, no network.

| Option     | Type             | Description                                                          |
| ---------- | ---------------- | -------------------------------------------------------------------- |
| `--query`  | `string`         | Required. Free-text — what the user is looking for.                  |
| `--top-k`  | `integer >= 1`   | Max groups to return. Default `5`. The MCP-tool schema caps at 20; the CLI relies on the backend's `> 0` check, no upper bound. |
| `--json`   |                  | Emit raw orchestrator response on stdout.                            |

Example:

```bash
copernicus-mcp marine search-groups --query "AMOC strength trend" --json
# selected[0].group_id == "ocean-circulation-indices", confidence "high"
```

## `marine search-products`

Filter products by group membership. Offline-only.

| Option      | Type                              | Description                                                                  |
| ----------- | --------------------------------- | ---------------------------------------------------------------------------- |
| `--groups`  | comma-separated `group_id` list   | Required. From `marine search-groups`.                                       |
| `--query`   | `string`                          | Optional. Re-rank products within the union of the named groups by keyword.  |
| `--top-k`   | `integer >= 1`                    | Max products to return. Default `20`. MCP-tool schema caps at 50; CLI relies on the backend's `> 0` check. |
| `--json`    |                                   | Emit raw orchestrator response on stdout.                                    |

Example:

```bash
copernicus-mcp marine search-products \
    --groups physics-arctic-state,arctic-comprehensive \
    --query "sea ice drift" --json
```

## `marine search-datasets`

Search the CMEMS catalogue. Two paths plus two modes:

- **Hierarchical path** — pass `--product-ids` (and optionally `--bbox` / `--time`). Returns enriched dataset cards filtered by product membership and spatial / temporal overlap. Cards with null spatial/temporal extent on the filtered axis are excluded — better under-select than risk a mismatch.
- **Flat path** — pass `--keyword` (no `--product-ids` / `--bbox` / `--time`). Reads the slim catalogue and returns the smaller per-dataset envelope.

| Option           | Type                              | Description                                                              |
| ---------------- | --------------------------------- | ------------------------------------------------------------------------ |
| `--keyword`      | `string`                          | Flat path: case-insensitive substring match against dataset ids, names, titles, product ids, descriptions, and variables. Hierarchical path (when `--product-ids` is set): forwarded as a re-rank phrase against the cards. |
| `--product-ids`  | comma-separated `product_id` list | Routes through the hierarchical cards path. Usually from `marine search-products`. |
| `--bbox`         | `min_lon,min_lat,max_lon,max_lat` | Spatial filter; antimeridian-crossing rejected (split into two non-crossing bboxes). |
| `--time`         | `start,end` ISO 8601 UTC          | Temporal filter; `start < end` required.                                  |
| `--limit`        | `integer >= 1`                    | Max dataset records returned. Capped at 50 on the hierarchical path.     |
| `--live`         |                                   | Flat path only. Call the live SDK instead of the snapshot. Needs credentials. |
| `--json`         |                                   | Emit raw orchestrator response on stdout.                                |

### Offline vs live

- **Offline (default)** is fast and works without credentials. The snapshot lives inside the wheel and is refreshed manually; the response includes `catalogue_fetched_at` so callers can see snapshot age.
- **Live (`--live`)** calls `copernicusmarine.describe()` and takes roughly ten seconds. Use it when you need a product or version that was published after the bundled snapshot was last refreshed. Credentials are required (see `docs/setup.md`). `--live` is only honoured on the flat path.

### Keyword tips

The flat-path keyword filter is a case-insensitive substring match. Empty and whitespace-only values are treated as no filter. Single-token keywords work best.

Examples:

```bash
# Hierarchical path (with the product ids from search-products)
copernicus-mcp marine search-datasets \
    --product-ids GLOBAL_ANALYSISFORECAST_PHY_001_024 \
    --bbox 20,41,30,47 --json

# Flat path
copernicus-mcp marine search-datasets --keyword temperature --limit 5 --json
copernicus-mcp marine search-datasets --keyword fresh-product --live --json
```

---

## `marine describe`

Show metadata for one dataset.

```
copernicus-mcp marine describe DATASET_ID [--json]
```

Example:

```bash
copernicus-mcp marine describe cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m --json \
  | jq '.variables[].name'
```

---

## `marine estimate`

Estimate the size and confirmation status of a subset request without downloading.

| Option         | Required | Description                                                  |
| -------------- | -------- | ------------------------------------------------------------ |
| `--dataset`    | yes      | Dataset id from `search-datasets`.                           |
| `--bbox`       | yes      | `min_lon,min_lat,max_lon,max_lat`.                           |
| `--time`       | yes      | `start,end` ISO 8601 UTC.                                    |
| `--variables`  | yes      | Comma-separated, e.g. `thetao,so`.                           |
| `--depth`      | no (default `0,5000`) | `min_depth,max_depth` in metres.                  |
| `--json`       |          | Emit raw orchestrator response on stdout.                    |

Example:

```bash
copernicus-mcp marine estimate \
  --dataset cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m \
  --variables thetao \
  --bbox -1,45,0,46 \
  --time 2024-06-01T00:00:00Z,2024-06-01T23:59:59Z \
  --depth 0,5 \
  --json
```

---

## `marine subset`

Download a spatio-temporal subset. **Returns a descriptor**: a path to a NetCDF/Zarr file in the cache directory plus metadata and a provenance record. Open the file with xarray, `netCDF4`, or your tool of choice.

Same options as `estimate`, plus:

| Option   | Description                                                                    |
| -------- | ------------------------------------------------------------------------------ |
| `--yes`  | Skip the confirmation prompt and proceed (assumes `confirmed=True`).           |

### Confirmation behaviour

If the request exceeds `budget.cmems_per_request_size_warning_gb` (default 1 GB) or returns an approximate estimate, the CLI shows a yellow Rich panel on stderr describing the size and either:

- **Prompts** with `Proceed? [y/N]` if stdin is a TTY.
- **Aborts** with exit code 3 if stdin is not a TTY and `--yes` was not given. This is the safe default for CI and unattended scripts.

Pass `--yes` to skip the prompt entirely. To raise the threshold, set `budget.cmems_per_request_size_warning_gb` higher in `~/.config/copernicus-mcp/config.yaml`.

Example:

```bash
copernicus-mcp marine subset \
  --dataset cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m \
  --variables thetao \
  --bbox -1,45,0,46 \
  --time 2024-06-01T00:00:00Z,2024-06-01T23:59:59Z \
  --depth 0,5 \
  --yes --json \
  | jq -r .filepath
# ~/.cache/copernicus-mcp/cmems/<cache_key>/data.nc
```

---

## `marine get-files`

Download native CMEMS files (no Zarr slicing). Use for sparse / in-situ datasets and `original-files` services that `marine subset` doesn't handle. Like `marine subset`, returns a descriptor — but with **multiple files per bundle**.

| Option            | Description                                                                |
| ----------------- | -------------------------------------------------------------------------- |
| `--dataset`       | Dataset id (required).                                                     |
| `--filter`        | Glob pattern (e.g. `*1990*`). Mutually exclusive with `--regex` / `--file-list`. |
| `--regex`         | Python regex matching file paths.                                          |
| `--file-list`     | Comma-separated list of explicit file paths.                               |
| `--version`       | Pin a specific dataset version.                                            |
| `--part`          | Pin a specific dataset part.                                               |
| `--yes`           | Skip the confirmation prompt (assumes `confirmed=True`).                   |
| `--json`          | Emit JSON to stdout.                                                       |

The gate behaves the same as `marine subset` — it almost always fires for sparse formats because `copernicusmarine.get` doesn't surface a precise dry-run size; pass `--yes` to bypass after reading the panel.

Example:

```bash
copernicus-mcp marine get-files \
  --dataset cmems_obs-ins_glo_phy-temp-sal_my_easycora_irr \
  --filter '*1990*' \
  --yes --json \
  | jq '.files[].filepath'
# "~/.cache/copernicus-mcp/cmems/<bundle>/a_1990.nc"
# "~/.cache/copernicus-mcp/cmems/<bundle>/b_1990.nc"
```

The bundle URI (`copernicus://files/<cache_key>`) resolves to a JSON envelope of per-file descriptors via the MCP file resource — useful if an agent wants to enumerate the bundle later without re-running `get-files`.

---

## `marine check-status`

Look up the status of an in-flight or completed workflow by `request_id` (returned in the provenance block of `subset`).

```
copernicus-mcp marine check-status REQUEST_ID [--json]
```

Example:

```bash
copernicus-mcp marine check-status 7b1ef... --json | jq '.status'
# "successful"
```

---

## `jobs list`

List recent jobs (downloads) recorded in the local state store, newest first — across sessions. Every per-job command needs a `request_id`, so this is how a fresh session rediscovers work submitted before a restart: list, then drive a specific job by its `request_id`. Identical content to the `copernicus_mcp_list_jobs` MCP tool.

```
copernicus-mcp jobs list [--status S] [--limit N] [--created-after TS] [--json]
```

| Option | Description |
|--------|-------------|
| `--status` | Comma-separated statuses to keep: `queued,running,successful,failed,cancelled`. |
| `--limit` | Maximum jobs returned (newest first; clamped to `1..500`; default 50). |
| `--created-after` | ISO-8601 UTC lower bound (strict), e.g. `2026-06-01T00:00:00Z`. |
| `--json` | Emit the raw listing object on stdout. |

Example:

```bash
# everything recent
copernicus-mcp jobs list

# only in-flight, as JSON, pull the request ids
copernicus-mcp jobs list --status queued,running --json | jq -r '.results[].request_id'
```

---

## Exit codes

| Code | Meaning                                                                            |
| ---- | ---------------------------------------------------------------------------------- |
| `0`  | Success.                                                                           |
| `1`  | Generic error — orchestrator returned `{"error": ...}` and it is not one of the more specific cases below. The error record is printed (Rich panel on stderr; JSON on stdout when `--json`). |
| `2`  | User input error — malformed `--bbox`, `--time`, `--depth`, missing required option, or empty `--variables`. Caught by the CLI before any backend call. |
| `3`  | Confirmation aborted — non-interactive stdin and `--yes` not given for a request that requires confirmation. |
| `4`  | Backend not configured — `error_subclass="backend_not_configured"`, typically because the requested backend is not enabled or its dependencies are missing. |

Scripts can rely on these codes; tests in `tests/unit/test_cli.py` assert them.

## Tips for scripting

- Prefer `--json` for any pipeline. The contract is "valid JSON on stdout for both success and error", so `jq -e` works without special-casing failures.
- Use `--yes` in cron/CI to keep subset jobs non-interactive — but estimate first to avoid surprise downloads.
- The cache directory is shared with the MCP server, so a CLI-driven retrieval shows up in the next `copernicus_mcp_status` call as well.

---

## CDS subcommands

The `cds` Typer group mirrors `marine` for the Climate Data Store family (CDS / ADS / EWDS). Enable the backend first — see [`setup.md`](./setup.md#enabling-the-backend).

CDS is queue-backed and async by design. The typical flow:

```bash
# 1. Discover.
copernicus-mcp cds search --keyword reanalysis --limit 5

# 2. Inspect one dataset.
copernicus-mcp cds describe reanalysis-era5-single-levels

# 3. (optional) Estimate before submit — heuristic only.
echo '{"product_type":["reanalysis"],"variable":["2m_temperature"],
       "year":["2024"],"month":["01"],"day":["01"],"time":["00:00"],
       "area":[50.0,0.0,49.0,1.0],"data_format":"grib"}' \
  | copernicus-mcp cds estimate \
      --dataset-id reanalysis-era5-single-levels --inputs-file - --json

# 4. Submit, poll, download.
copernicus-mcp cds submit \
    --dataset-id reanalysis-era5-single-levels --inputs-file request.json --yes
copernicus-mcp cds wait <request_id>           # blocks until terminal
copernicus-mcp cds download <request_id> --json
```

## `cds search`

```text
Usage: copernicus-mcp cds search [OPTIONS]

  --keyword TEXT  Free-text match against dataset id / title / description.
  --store TEXT    cds | ads | ewds — restrict to a single store. Omit
                  to search across all three.
  --limit INTEGER Cap result count.
  --json          Emit raw JSON instead of a Rich table.
```

Returns `{datasets, total_count}` against the bundled catalogue snapshot — no network call.

## `cds describe`

```text
Usage: copernicus-mcp cds describe [OPTIONS] DATASET_ID

  DATASET_ID  Dataset id from cds search. [required, positional]
  --json
```

Returns the full STAC item. Note that the dataset id is a **positional**
argument here (matches the `marine describe` shape) — `cds estimate`
and `cds submit` instead take `--dataset-id` because the file-input flow
makes a flag less ambiguous.

## `cds estimate`

```text
Usage: copernicus-mcp cds estimate [OPTIONS]

  --dataset-id TEXT       Required.
  --inputs-file PATH      Path to a JSON file with the cdsapi inputs dict, or
                          ``-`` to read from stdin.
  --json
```

The `inputs` dict mirrors the cdsapi retrieve shape (`variable`, `year`, `month`, …, `area`). **`area` ordering: CDS uses `[north, west, south, east]` (NWSE)**, opposite of GIS WSEN. The estimator returns `estimated_size_bytes` (which **may be `null`** for whole-file products — honest "size unknown"), `epistemic_status` (`calibrated` / `curated_approximate` / `default_heuristic` / `unknown`), a `cost` block (`{units, limit, exceeds_limit}` — `exceeds_limit: true` means the server will reject the request), and `queue_latency_tier`. Estimates self-calibrate from completed downloads.

## `cds apply-constraints`

```text
Usage: copernicus-mcp cds apply-constraints [OPTIONS]

  --dataset-id TEXT     Required.
  --inputs-file PATH    JSON file with a PARTIAL inputs dict, or ``-`` for
                        stdin. Omit for empty inputs → full per-field vocabulary.
  --json
```

Returns the still-valid values per field (`valid_remaining`). Empty inputs give every field's full value set; a partial selection narrows the rest. Read-only and anonymous (no PAT needed) — useful for discovering valid field values before composing a submit.

## `cds submit`

```text
Usage: copernicus-mcp cds submit [OPTIONS]

  --dataset-id TEXT     Required.
  --inputs-file PATH    JSON file with the inputs dict; ``-`` for stdin.
  --yes                 Bypass the size + queue-tier confirmation gate.
  --chunk-by TEXT       Split an over-limit request along the calendar axis
                        (year|month|day); run the parts as one workflow.
  --force-refresh       Re-run from scratch, bypassing the cache.
  --json
```

Returns `{status: "queued", request_id, cache_key}` immediately. Without `--yes`, large or heavy requests print a confirmation prompt and exit with code 3 on non-interactive stdin.

### Auto-chunking

A request over the dataset's server-side cost limit is split automatically. With `--chunk-by year|month|day` the split happens on the first call; with `--yes` (and no `--chunk-by`) an over-limit request is split by year. The command then returns a parent `request_id` that drives the whole multi-file set:

```text
copernicus-mcp cds submit --dataset-id <id> --inputs-file req.json --chunk-by year
copernicus-mcp cds wait <parent_id>            # advances every chunk to completion
copernicus-mcp cds download <parent_id> --json # lists one file descriptor per chunk
```

`cds wait` / `cds download` / `cds cancel` operate on the parent `request_id` transparently. `download` on a chunked parent prints the descriptor set (one file per chunk, ordered by chunk index) with a `merge_hint` — the CLI never recombines the parts.

While it waits, `cds wait` prints a live progress line whose states partition the parts:

```text
parts: 3/21 done (running 2, downloading 1, retrying 1, queued 14)
```

`downloading` is a part whose server-side job is finished and whose file is transferring; `retrying` is a part the Climate Data Store refused because it was busy, which is re-submitted automatically. Parts are submitted a few at a time rather than all at once, so a large split completes over several waves — a slowly-advancing count is expected, not a stall. See the pacing knobs in `docs/setup.md`.

A large split asks for confirmation: more than `budget.cds_auto_chunk_confirm_above` parts (default 30) prints a confirmation prompt; more than `cds_auto_chunk_reconfirm_above` (default 100) is a heavier batch that prompts a second time. At the CLI a single `--yes` (or one interactive confirm) covers both tiers — the repeat gate exists for the agent path, where the model must escalate to a human. Over `cds_auto_chunk_max_chunks` (default 366) the request is rejected; raise that config to allow a bigger fan-out.

### T&C errors

If the user has not accepted the dataset's licence(s), the server returns HTTP 403 with `"user didn't accept all required site policies"`. The CLI exits with `error_class: TermsNotAcceptedError`, `recovery_action: accept_terms`, and `recovery_url` pointing at the licence page. Open the URL, accept the licence, and re-submit the same request.

## `cds check-status`

```text
Usage: copernicus-mcp cds check-status REQUEST_ID [OPTIONS]

  --json
```

One-shot poll. Prints the workflow row. Status is one of `queued` / `running` / `successful` / `failed` / `cancelled`.

## `cds wait`

```text
Usage: copernicus-mcp cds wait REQUEST_ID [OPTIONS]

  --timeout FLOAT       Hard timeout in seconds (default 7200, i.e. 2h —
                        higher than `marine wait` because CDS queues can
                        run hours on heavy datasets per research §6.5.4).
  --interval FLOAT      Seconds between polls (default 10.0).
  --json
```

Blocks until status is terminal (`successful` / `failed` / `cancelled`), or exits 1 on `--timeout` expiry.

## `cds download`

```text
Usage: copernicus-mcp cds download REQUEST_ID [OPTIONS]

  --target PATH         Reserved. CDS backend stores files in the canonical
                        cache regardless; this option is accepted but ignored
                        in the current release.
  --json
```

Returns the file descriptor `{filepath, uri, metadata, provenance}`. Fails with `error_subclass="result_not_ready"` if the workflow has not reached `successful`.

## `cds cancel`

```text
Usage: copernicus-mcp cds cancel REQUEST_ID [OPTIONS]

  --json
```

Idempotent on already-terminal rows. Returns `{cancelled, request_id, status}`.
