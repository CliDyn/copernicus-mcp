# copernicus-mcp

**A self-hosted [Model Context Protocol](https://modelcontextprotocol.io) server that gives MCP-compatible LLM agents local, reproducible access to Copernicus environmental data — observations, reanalysis, forecasts, and climate indicators.**

Two backends are currently supported, both using Copernicus services that are free to register for: [Copernicus Marine](https://data.marine.copernicus.eu/register), [CDS](https://cds.climate.copernicus.eu/), [ADS](https://ads.atmosphere.copernicus.eu/), and [EWDS](https://ewds.climate.copernicus.eu/).

- **[Copernicus Marine](https://marine.copernicus.eu/) (CMEMS)** — 1,251 datasets across 306 products in the bundled catalogue snapshot: physics, biogeochemistry, sea ice, ocean colour, SST, sea level, waves, wind, and in-situ observations. Supports discovery, subsetting, native-file retrieval, and sync or async downloads.
- **[Climate Data Store family](https://cds.climate.copernicus.eu/) (CDS / ADS / EWDS)** — 164 datasets in the bundled snapshot across reanalysis, satellite, in-situ, atmospheric composition (CAMS), and emergency-management (EFAS / GloFAS / CEMS) data. Uses queue-based asynchronous retrieval with offline discovery from a bundled catalogue snapshot.

Ask in plain English. The server finds, filters, estimates, and downloads. Large downloads are size-estimated, gated for explicit confirmation, cached, and returned as a `filepath + metadata + provenance` descriptor rather than inline bytes. Every retrieval lands with an MD5-sealed sidecar JSON so the exact request is reproducible later.

> *"Get me Mediterranean Sea salinity forecasts for next week, then fetch the AMOC strength time series for the last 5 years."*

---

## Install

No hosted endpoint. No vendor account. No data upload. Python 3.11+ required.

### With `venv` (stdlib, no extra tools)

```bash
python -m venv .venv
source .venv/bin/activate           # macOS / Linux
# .venv\Scripts\activate             # Windows

pip install "copernicus-mcp[cmems,cds]"     # both backends (recommended)
# pip install "copernicus-mcp[cmems]"       # CMEMS only
# pip install "copernicus-mcp[cds]"         # CDS / ADS / EWDS only
```

### With `conda` / `mamba` / `micromamba`

The package is currently published on PyPI only (no conda-forge feedstock yet) — so `pip install` inside a fresh conda environment is the path:

```bash
mamba create -n copernicus python=3.11 pip       # or `conda create ...`
mamba activate copernicus
pip install "copernicus-mcp[cmems,cds]"
```

The MCP client config (next section) then points at `/path/to/your/conda/envs/copernicus/bin/copernicus-mcp` instead of the venv path. Run `which copernicus-mcp` inside the activated environment to find the exact location.

### Credentials

The Copernicus services own the credentials — we never see them.

**CMEMS** (free account at https://data.marine.copernicus.eu/register):

```bash
copernicusmarine login            # writes ~/.copernicusmarine/.copernicusmarine-credentials
```

**CDS / ADS / EWDS** — a single Personal Access Token works across all three stores under ECMWF's unified-token policy. Get it at https://cds.climate.copernicus.eu/ → user profile:

```bash
export CDSAPI_KEY=<your-uuid-pat>
# or write to ~/.cdsapirc (same format the cdsapi CLI uses)
```

Some CDS-family datasets require accepting their licence once; when that happens the server returns the acceptance URL in a structured `TermsNotAcceptedError`.

---

## Configure your MCP client

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or the equivalent on your platform:

```json
{
  "mcpServers": {
    "copernicus": {
      "command": "/absolute/path/to/.venv/bin/copernicus-mcp",
      "args": ["serve"]
    }
  }
}
```

Restart Claude Desktop. Tools become available the next time you open a chat.

### Claude Code

```bash
claude mcp add copernicus -- /absolute/path/to/.venv/bin/copernicus-mcp serve
```

### Other MCP clients

Any client that speaks MCP over stdio works. Point it at the `copernicus-mcp serve` command in your virtualenv.

### Smoke test

After install, confirm the server starts and your credentials resolve:

```bash
copernicus-mcp status
```

The output lists configured backends and where credentials were resolved from — without ever printing the credential values themselves.

---

## Try these prompts

Drop these into a chat with the server connected. Each one walks through the discovery → estimate → download flow and lands a real file on your disk.

| # | Prompt | What the agent does |
|---|---|---|
| 1 | *"Mediterranean salinity forecast for the next 7 days."* | Routes to the `physics-mediterranean-state` group → picks `MEDSEA_ANALYSISFORECAST_PHY_006_013` → downloads a daily-mean NetCDF subset. |
| 2 | *"How has Arctic sea-ice extent changed over the last 5 years?"* | Routes to the `sea-ice-arctic` group → finds `ARCTIC_OMI_SI_extent` indicator timeseries + `SEAICE_ARC_PHY_AUTO_L3_MYNRT_011_023` satellite L3. |
| 3 | *"Get me the AMOC strength timeseries at 26°N."* | Intent-heavy query — routes to `ocean-monitoring-indicators` → returns `GLOBAL_OMI_NATLANTIC_amoc_26N_profile` and `_amoc_max26N_timeseries`. |
| 4 | *"ERA5 hourly 2m temperature over Europe for January 2024."* | CDS path → describes `reanalysis-era5-single-levels` → submits a request, polls until the queue settles, lands a GRIB / NetCDF file. |
| 5 | *"Global CO₂ atmospheric forecasts for next week."* | ADS path → finds `cams-global-greenhouse-gas-forecasts` → submits + waits + downloads. |

The discovery routing is covered offline by `bench/marine_routing_bench.py`; the full submit-download flows are covered by `tests/integration/` and require live CMEMS / CDS credentials with `RUN_INTEGRATION_TESTS=1`.

---

## Tools

| Backend | Tool | Purpose |
|---|---|---|
| **diagnostic** | `copernicus_mcp_status` | Configured backends, cache size, override hints. No credentials in output. |
| **CMEMS** | `marine_search_groups` → `marine_search_products` → `marine_search_datasets` | Hierarchical discovery — narrows ~1251 datasets in two hops via 47 hand-curated routing groups. Offline, no embeddings, no LLM at query time. |
| | `marine_describe_dataset` | Full metadata: variables, axes, spatial / temporal extent, services, DOI. |
| | `marine_get_coordinates` | The dataset's actual lon/lat/depth/time axes — summarised for long axes. |
| | `marine_estimate_subset` | Preview download size before running it. |
| | `marine_subset_dataset` | Download a spatio-temporal subset. Large requests require explicit confirmation. `async_mode=true` returns immediately. |
| | `marine_list_files` → `marine_get_files` | For sparse / in-situ datasets (CORA, EasyCORA, INSITU-BGC, MULTIOBS): filter by bbox / time / variables, then download the precise file list. |
| | `marine_check_status`, `marine_cancel_subset` | Async lifecycle. |
| **CDS / ADS / EWDS** | `cds_search_groups` → `cds_search_datasets` | Hierarchical group discovery + filters (bbox / time_range / variable / domain / category). |
| | `cds_describe_dataset`, `cds_apply_constraints` | Bundled snapshot + live narrowing against the store's constraints endpoint. |
| | `cds_estimate_request` | Heuristic byte-size + queue-tier (light / medium / heavy). |
| | `cds_submit_request`, `cds_check_request_status`, `cds_download_request_result`, `cds_cancel_request` | Async queue lifecycle. T&C-not-accepted surfaces as a structured error with the accept-URL. |

Tools that return large data return `{filepath, uri, metadata, provenance}` — never inline bytes. The `copernicus://files/{cache_key}`, `copernicus://jobs/{request_id}`, and `copernicus://provenance/{record_id}` resources surface the cached artifacts to MCP clients that prefer the resource API.

For complete schemas read the inline tool descriptions your MCP client surfaces, or the detailed reference in [`docs/tools.md`](docs/tools.md).

---

## How it works

```
   ┌──────────────────────────┐       ┌─────────────────────────────┐
   │  Claude / Claude Code /  │       │   copernicus-mcp serve      │
   │  any MCP client          │ stdio │   (local Python process)    │
   └──────────────────────────┘ ────▶ │                             │
                                      │  hierarchical discovery     │
                                      │  size estimate + gate       │
                                      │  retrieval + provenance     │
                                      │  cache + idempotent re-use  │
                                      └──────────────┬──────────────┘
                                                     │
                          ┌──────────────────────────┼────────────────────────┐
                          ▼                          ▼                        ▼
                  Copernicus Marine          Climate Data Store          local cache
                  (Mercator Ocean)           CDS / ADS / EWDS            ~/Library/Caches/...
                  copernicusmarine SDK       cdsapi SDK                  (per-OS paths)
```

Hierarchical discovery uses bundled JSON manifests (slim records → enriched cards → product summaries → routing groups). Scoring is deterministic phrase matching. The catalogue + groups are committed JSON, so "why did this query return that group" is auditable.

---

## Limitations

- The server does not bypass Copernicus licences or access controls — credentials and licence acceptance remain the user's responsibility.
- CDS-family downloads depend on upstream queue availability; large requests may take minutes to hours.
- Catalogue snapshots are bundled at release time and may lag behind the live Copernicus catalogues. Re-publish a new version when the snapshots are refreshed.
- Hierarchical routing is pattern-based; it does not use embeddings or an LLM at query time. Misroutes on highly ambiguous queries are possible.

---

## Configuration

The system runs out of the box. Override via env vars (`COPERNICUS_MCP_CACHE_DIR`, `COPERNICUS_MCP_LOG_LEVEL`, `COPERNICUS_MCP_ENABLED_BACKENDS=cmems,cds`), a YAML file at `~/.config/copernicus-mcp/config.yaml`, or `--cache-dir PATH` on the entry-point binary.

Cache directories are per-OS via [`platformdirs`](https://platformdirs.readthedocs.io/): Linux `~/.cache/copernicus-mcp/`, macOS `~/Library/Caches/copernicus-mcp/`, Windows `%LOCALAPPDATA%\copernicus-mcp\Cache\`.

Full reference: [`docs/setup.md`](docs/setup.md).

---

## Status

Latest release: see [releases page](https://github.com/CliDyn/copernicus-mcp/releases) (current: `v0.4.2`). Two backends in production: CMEMS + Climate Data Store family. CDSE, Sentinel Hub, WEkEO planned for subsequent iterations.

---

## License

BSD 3-Clause. See [`LICENSE`](LICENSE). Dependencies are EUPL-1.2 ([`copernicusmarine`](https://github.com/mercator-ocean/copernicus-marine-toolbox)), Apache-2.0 ([`cdsapi`](https://github.com/ecmwf/cdsapi) and most others), MIT or BSD.

---

## Acknowledgements

- [Mercator Ocean International](https://www.mercator-ocean.eu/) for [`copernicusmarine`](https://github.com/mercator-ocean/copernicus-marine-toolbox).
- [ECMWF](https://www.ecmwf.int/) for [`cdsapi`](https://github.com/ecmwf/cdsapi) and for operating the Climate / Atmosphere / Early Warning data stores.
- The [Copernicus programme](https://www.copernicus.eu/) for the underlying data.
- The Anthropic team for the [Model Context Protocol](https://modelcontextprotocol.io/) specification and Python SDK.

---

## Related projects

- **[AQUAVIEW MCP](https://github.com/AQUAVIEW-DAH/mcp)** — hosted MCP server unifying ~700K datasets across 68 NOAA / IOOS / EMODnet / Argo / GOES-R / Sentinel collections. If your question reaches beyond Copernicus into US / global multi-agency oceanographic and atmospheric data, that's the natural complement to this server.
