"""CLI companion for copernicus-mcp (T-032).

Top-level ``copernicus-mcp`` Typer app with must-have subcommands:
``serve``, ``version``, ``status``, and the ``marine`` group
(``search-datasets``, ``describe``, ``estimate``, ``subset``,
``check-status``).

Output: human-readable Rich tables/panels by default; ``--json`` emits
the raw orchestrator response for scripting.

Confirmation flow (subset): orchestrator returns
``{"confirmation_required": True, ...}``. CLI prompts via
``rich.prompt.Confirm`` if stdin is a TTY, or aborts with exit code 3
if non-interactive and ``--yes`` was not given.

Exit codes (per plan T-037):
  0 success, 1 generic error, 2 user input error, 3 confirmation aborted,
  4 backend not configured.

This module is excluded from ``mypy --strict`` (per plan T-031 step 6 —
MCP/Pydantic interop with CLI helpers is broad in surface).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.json import JSON as RichJSON
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
from copernicus_mcp.config import ConfigLoader
from copernicus_mcp.version import __version__
from copernicus_mcp.workflow.orchestrator import WorkflowOrchestrator

app = typer.Typer(name="copernicus-mcp", help="Copernicus MCP server CLI.")
marine_app = typer.Typer(name="marine", help="CMEMS Marine subcommands.")
app.add_typer(marine_app, name="marine")
cds_app = typer.Typer(
    name="cds",
    help="Copernicus Climate / Atmosphere / Emergency Data Store subcommands (T-CDS-007).",
)
app.add_typer(cds_app, name="cds")

# Two consoles: ``console`` for primary payload output (non-JSON tables /
# panels go to stdout). ``err_console`` for diagnostics — error panels,
# confirmation prompts, non-TTY messages — always to stderr so they never
# corrupt JSON on stdout in ``--json`` mode (the stdio-cleanliness invariant
# spirit applied to CLI output).
console = Console(stderr=False)
err_console = Console(stderr=True)

# Exit codes — keep in sync with docs/cli.md (T-037).
EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_USER_INPUT = 2
EXIT_CONFIRMATION_ABORTED = 3
EXIT_BACKEND_NOT_CONFIGURED = 4


# T-CDS-019: ``--cache-dir`` is a global CLI flag that takes precedence
# over ``COPERNICUS_MCP_CACHE_DIR`` and the yaml/defaults layers
# (cli > env > yaml > defaults). Stored at module scope so subcommands
# can see it without each declaring the option.
_cli_cache_dir: Path | None = None


def _cli_config_overrides() -> dict[str, Any]:
    """CLI-layer overrides to merge into ``ConfigLoader().load(...)``.

    Currently only holds the optional ``--cache-dir``. Future flags
    follow the same pattern."""
    overrides: dict[str, Any] = {}
    if _cli_cache_dir is not None:
        overrides["storage"] = {"cache_directory": str(_cli_cache_dir)}
    return overrides


@app.callback()
def _root(
    cache_dir: Path | None = typer.Option(
        None,
        "--cache-dir",
        help=(
            "Override cache directory. Wins over COPERNICUS_MCP_CACHE_DIR. "
            "Default is the OS user cache (e.g. ~/Library/Caches/copernicus-mcp "
            "on macOS, %LOCALAPPDATA%\\copernicus-mcp\\Cache on Windows)."
        ),
    ),
) -> None:
    """Top-level group; subcommands attach here."""
    global _cli_cache_dir
    _cli_cache_dir = cache_dir


@contextlib.asynccontextmanager
async def _build_orchestrator_for_cli() -> AsyncIterator[WorkflowOrchestrator]:
    """Build foundation + registry + orchestrator for a one-shot CLI command.

    Tests monkeypatch this symbol to inject a mocked orchestrator.
    """
    config = ConfigLoader().load(cli_overrides=_cli_config_overrides() or None)
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        yield WorkflowOrchestrator(registry=registry, foundation=foundation)
    finally:
        await foundation.persistence.close()


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _emit(payload: dict[str, Any], *, json_out: bool, title: str | None = None) -> None:
    """Render either as raw JSON (pipe-safe) or as a Rich panel/table."""
    if json_out:
        typer.echo(json.dumps(payload, default=str))
        return
    if "results" in payload and isinstance(payload["results"], list):
        table = Table(title=title or "Results")
        rows = payload["results"]
        if rows:
            for col in rows[0].keys():
                table.add_column(col)
            for row in rows:
                table.add_row(*[str(row.get(c, "")) for c in rows[0].keys()])
        console.print(table)
        return
    console.print(Panel(RichJSON.from_data(payload), title=title or "Response"))


def _unwrap_result(envelope: dict[str, Any]) -> dict[str, Any]:
    if "result" in envelope and isinstance(envelope["result"], dict):
        return envelope["result"]
    return envelope


def _handle_error(envelope: dict[str, Any], *, json_out: bool = False) -> None:
    """Print a structured error and exit with the right code.

    In ``--json`` mode the canonical record is emitted as JSON to stdout
    so scripts piping into ``jq`` always receive valid JSON regardless
    of success/failure. Otherwise a red Rich panel goes to stderr.
    """
    record = envelope.get("error") or {}
    if json_out:
        typer.echo(json.dumps({"error": record}, default=str))
    else:
        err_console.print(
            Panel(
                RichJSON.from_data(record),
                title=f"Error: {record.get('error_class', 'Error')}",
                style="red",
            )
        )
    code = (
        EXIT_BACKEND_NOT_CONFIGURED
        if record.get("error_subclass") == "backend_not_configured"
        else EXIT_GENERIC
    )
    raise typer.Exit(code=code)


@app.command()
def version() -> None:
    """Print the installed copernicus-mcp version."""
    typer.echo(__version__)


@app.command()
def serve() -> None:
    """Start the MCP server over stdio (Claude Desktop, MCP clients)."""
    from copernicus_mcp.server import main as server_main

    # T-CDS-019: forward the CLI --cache-dir (and future flags) so
    # ``copernicus-mcp --cache-dir X serve`` honours the override
    # rather than silently re-loading the bare default.
    server_main(cli_overrides=_cli_config_overrides() or None)


@app.command()
def status(
    json_out: bool = typer.Option(False, "--json", help="Emit raw JSON."),
) -> None:
    """Print server diagnostics (backends, cache, persistence, config)."""

    async def _go() -> dict[str, Any]:
        async with _build_orchestrator_for_cli() as orch:
            return await orch.status()

    payload = _run(_go())
    _emit(payload, json_out=json_out, title="Status")


def _parse_csv_floats(spec: str, *, count: int, name: str) -> tuple[float, ...]:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != count:
        typer.echo(
            f"Error: --{name} expects {count} comma-separated numbers, got {len(parts)}",
            err=True,
        )
        raise typer.Exit(code=EXIT_USER_INPUT)
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        typer.echo(f"Error: --{name} values must be numeric", err=True)
        raise typer.Exit(code=EXIT_USER_INPUT) from None


def _parse_csv_strs(spec: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in spec.split(",") if p.strip())


@marine_app.command("search-groups")
def marine_search_groups(
    query: str = typer.Option(..., "--query", help="Free-text query — what you are looking for."),
    top_k: int | None = typer.Option(None, "--top-k", help="Max groups to return."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """First step of the hierarchical search: shortlist routing groups
    for a free-text query (T-CMEMS-HIER-005)."""
    params: dict[str, Any] = {"query": query}
    if top_k is not None:
        params["top_k"] = top_k

    async def _go() -> dict[str, Any]:
        async with _build_orchestrator_for_cli() as orch:
            return await orch.run(backend="cmems", operation="search_groups", params=params)

    envelope = _run(_go())
    if "error" in envelope:
        _handle_error(envelope, json_out=json_out)
    _emit(_unwrap_result(envelope), json_out=json_out, title="Groups")


@marine_app.command("search-products")
def marine_search_products(
    groups: str = typer.Option(
        ...,
        "--groups",
        help="Comma-separated group ids from `marine search-groups`.",
    ),
    query: str | None = typer.Option(
        None,
        "--query",
        help="Optional keyword to re-rank products within the chosen groups.",
    ),
    top_k: int | None = typer.Option(None, "--top-k"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Second step of the hierarchical search: shortlist products in
    the chosen routing groups."""
    group_ids = [g for g in _parse_csv_strs(groups) if g]
    if not group_ids:
        typer.echo("Error: --groups expects a non-empty list", err=True)
        raise typer.Exit(code=EXIT_USER_INPUT)
    params: dict[str, Any] = {"group_ids": group_ids}
    if query:
        params["query"] = query
    if top_k is not None:
        params["top_k"] = top_k

    async def _go() -> dict[str, Any]:
        async with _build_orchestrator_for_cli() as orch:
            return await orch.run(backend="cmems", operation="search_products", params=params)

    envelope = _run(_go())
    if "error" in envelope:
        _handle_error(envelope, json_out=json_out)
    _emit(_unwrap_result(envelope), json_out=json_out, title="Products")


@marine_app.command("search-datasets")
def marine_search_datasets(
    keyword: str | None = typer.Option(None, "--keyword"),
    bbox: str | None = typer.Option(None, "--bbox", help="min_lon,min_lat,max_lon,max_lat"),
    time: str | None = typer.Option(None, "--time", help="start,end ISO 8601 UTC"),
    product_ids: str | None = typer.Option(
        None,
        "--product-ids",
        help=(
            "Comma-separated product ids from `marine search-products`. Routes "
            "through the hierarchical cards path and returns enriched cards."
        ),
    ),
    limit: int | None = typer.Option(None, "--limit"),
    live: bool = typer.Option(
        False,
        "--live",
        help=(
            "Hit the live Copernicus Marine Service instead of the bundled "
            "catalogue snapshot. Requires CMEMS credentials and a network "
            "round-trip; use when you need products newer than the snapshot."
        ),
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Search the CMEMS catalogue. Hierarchical path (with --product-ids,
    --bbox, or --time) returns enriched dataset cards; flat path
    (--keyword only) returns the slim catalogue envelope."""
    params: dict[str, Any] = {}
    if keyword:
        params["keyword"] = keyword
    if bbox:
        bb = _parse_csv_floats(bbox, count=4, name="bbox")
        params["bbox"] = list(bb)
    if time:
        parts = _parse_csv_strs(time)
        if len(parts) != 2:
            typer.echo("Error: --time expects start,end", err=True)
            raise typer.Exit(code=EXIT_USER_INPUT)
        params["time_range"] = list(parts)
    if product_ids:
        pids = [p for p in _parse_csv_strs(product_ids) if p]
        if not pids:
            typer.echo("Error: --product-ids expects a non-empty list", err=True)
            raise typer.Exit(code=EXIT_USER_INPUT)
        params["product_ids"] = pids
    if limit is not None:
        params["limit"] = limit
    if live:
        params["live"] = True

    async def _go() -> dict[str, Any]:
        async with _build_orchestrator_for_cli() as orch:
            return await orch.run(backend="cmems", operation="search", params=params)

    envelope = _run(_go())
    if "error" in envelope:
        _handle_error(envelope, json_out=json_out)
    _emit(_unwrap_result(envelope), json_out=json_out, title="Datasets")


@marine_app.command("describe")
def marine_describe(
    dataset_id: str = typer.Argument(..., help="Dataset id from search-datasets."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Show full metadata for a single CMEMS dataset."""

    async def _go() -> dict[str, Any]:
        async with _build_orchestrator_for_cli() as orch:
            return await orch.run(
                backend="cmems",
                operation="describe",
                params={"identifier": dataset_id},
            )

    envelope = _run(_go())
    if "error" in envelope:
        _handle_error(envelope, json_out=json_out)
    _emit(_unwrap_result(envelope), json_out=json_out, title=dataset_id)


def _build_subset_params(
    *,
    dataset_id: str,
    bbox: str,
    time: str,
    variables: str,
    depth: str,
) -> dict[str, Any]:
    bb = _parse_csv_floats(bbox, count=4, name="bbox")
    time_parts = _parse_csv_strs(time)
    if len(time_parts) != 2:
        typer.echo("Error: --time expects start,end", err=True)
        raise typer.Exit(code=EXIT_USER_INPUT)
    depths = _parse_csv_floats(depth, count=2, name="depth")
    var_list = list(_parse_csv_strs(variables))
    if not var_list:
        typer.echo("Error: --variables must list at least one variable", err=True)
        raise typer.Exit(code=EXIT_USER_INPUT)
    return {
        "dataset_id": dataset_id,
        "variables": var_list,
        "minimum_longitude": bb[0],
        "minimum_latitude": bb[1],
        "maximum_longitude": bb[2],
        "maximum_latitude": bb[3],
        "minimum_depth": depths[0],
        "maximum_depth": depths[1],
        "start_datetime": time_parts[0],
        "end_datetime": time_parts[1],
    }


@marine_app.command("estimate")
def marine_estimate(
    dataset: str = typer.Option(..., "--dataset"),
    bbox: str = typer.Option(..., "--bbox"),
    time: str = typer.Option(..., "--time"),
    variables: str = typer.Option(..., "--variables"),
    depth: str = typer.Option("0,5000", "--depth", help="min_depth,max_depth"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Preview byte-size and confirmation requirements without downloading."""
    params = _build_subset_params(
        dataset_id=dataset, bbox=bbox, time=time, variables=variables, depth=depth
    )

    async def _go() -> dict[str, Any]:
        async with _build_orchestrator_for_cli() as orch:
            return await orch.run(backend="cmems", operation="estimate", params=params)

    envelope = _run(_go())
    if "error" in envelope:
        _handle_error(envelope, json_out=json_out)
    _emit(_unwrap_result(envelope), json_out=json_out, title="Estimate")


@marine_app.command("subset")
def marine_subset(
    dataset: str = typer.Option(..., "--dataset"),
    bbox: str = typer.Option(..., "--bbox"),
    time: str = typer.Option(..., "--time"),
    variables: str = typer.Option(..., "--variables"),
    depth: str = typer.Option("0,5000", "--depth"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt (assume yes)."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Subset a CMEMS dataset; prompts for confirmation on large requests.

    The CLI is one-shot — fire-and-forget async submit is intentionally
    not exposed here (it would race the process shutdown that closes
    persistence and cancels in-flight tasks). Use the MCP server
    (``copernicus-mcp serve``) and ``marine_subset_dataset`` with
    ``async_mode=True`` for the agent flow; ``marine wait REQUEST_ID``
    polls a workflow row written by either path.
    """
    params = _build_subset_params(
        dataset_id=dataset, bbox=bbox, time=time, variables=variables, depth=depth
    )

    async def _go() -> dict[str, Any]:
        async with _build_orchestrator_for_cli() as orch:
            envelope = await orch.run(backend="cmems", operation="submit", params=params)
            if envelope.get("confirmation_required"):
                _show_confirmation(envelope)
                if not yes and not _interactive_confirm():
                    raise typer.Exit(code=EXIT_CONFIRMATION_ABORTED)
                envelope = await orch.run(
                    backend="cmems",
                    operation="submit",
                    params=params,
                    options={"confirmed": True},
                )
            return envelope

    envelope = _run(_go())
    if "error" in envelope:
        _handle_error(envelope, json_out=json_out)
    _emit(_unwrap_result(envelope), json_out=json_out, title="Subset")


def _show_confirmation(envelope: dict[str, Any]) -> None:
    """Print confirmation prompt to stderr (so ``--json`` stdout stays clean)."""
    msg = envelope.get("advisory_message", "Large request")
    size = envelope.get("estimated_size_bytes")
    title = "Confirmation required"
    if size:
        title = f"{title} (~{size / 1_000_000_000:.2f} GB)"
    err_console.print(Panel(str(msg), title=title, style="yellow"))


def _interactive_confirm() -> bool:
    """Return True if the user confirms; False if non-interactive or 'no'.

    All prompt output goes to stderr.
    """
    if not sys.stdin.isatty():
        err_console.print(
            "[yellow]Non-interactive stdin and --yes not set: aborting confirmation.[/yellow]"
        )
        return False
    return Confirm.ask("Proceed?", default=False, console=err_console)


@marine_app.command("get-files")
def marine_get_files_cmd(
    dataset: str = typer.Option(..., "--dataset"),
    filter_: str | None = typer.Option(
        None, "--filter", help="Glob pattern, e.g. ``*1990*``."
    ),
    regex: str | None = typer.Option(
        None, "--regex", help="Python regex matching file paths."
    ),
    file_list: str | None = typer.Option(
        None,
        "--file-list",
        help="Comma-separated list of explicit file paths to fetch.",
    ),
    dataset_version: str | None = typer.Option(None, "--version"),
    dataset_part: str | None = typer.Option(None, "--part"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Download native CMEMS files (T-CMEMS-GET-006).

    Use this for sparse / in-situ datasets and ``original-files``
    services that ``marine subset`` doesn't handle. Prompts for
    confirmation on large or approximate-sized downloads — the gate
    almost always fires for sparse formats because the SDK doesn't
    surface a precise dry-run size.

    At most one of ``--filter`` / ``--regex`` / ``--file-list`` may
    be set; omit them to download whatever the toolbox defaults to.
    """
    params: dict[str, Any] = {"dataset_id": dataset}
    if filter_ is not None:
        params["filter"] = filter_
    if regex is not None:
        params["regex"] = regex
    if file_list is not None:
        params["file_list"] = [p.strip() for p in file_list.split(",") if p.strip()]
    if dataset_version is not None:
        params["dataset_version"] = dataset_version
    if dataset_part is not None:
        params["dataset_part"] = dataset_part

    async def _go() -> dict[str, Any]:
        async with _build_orchestrator_for_cli() as orch:
            envelope = await orch.run(
                backend="cmems", operation="get", params=params
            )
            if envelope.get("confirmation_required"):
                _show_confirmation(envelope)
                if not yes and not _interactive_confirm():
                    raise typer.Exit(code=EXIT_CONFIRMATION_ABORTED)
                envelope = await orch.run(
                    backend="cmems",
                    operation="get",
                    params=params,
                    options={"confirmed": True},
                )
            return envelope

    envelope = _run(_go())
    if "error" in envelope:
        _handle_error(envelope, json_out=json_out)
    _emit(_unwrap_result(envelope), json_out=json_out, title="Get files")


@marine_app.command("check-status")
def marine_check_status(
    request_id: str = typer.Argument(...),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Look up the status of an in-flight or completed workflow."""

    async def _go() -> dict[str, Any]:
        async with _build_orchestrator_for_cli() as orch:
            return await orch.run(
                backend="cmems",
                operation="poll",
                params={"request_id": request_id},
            )

    envelope = _run(_go())
    if "error" in envelope:
        _handle_error(envelope, json_out=json_out)
    _emit(_unwrap_result(envelope), json_out=json_out, title=request_id)


_TERMINAL_STATUSES = {"successful", "failed", "cancelled"}


@marine_app.command("wait")
def marine_wait(
    request_id: str = typer.Argument(...),
    interval: float = typer.Option(10.0, "--interval", help="Seconds between polls."),
    timeout: float = typer.Option(
        3600.0, "--timeout", help="Hard timeout in seconds. Exit 1 on expiry."
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Block until an async submit reaches a terminal status.

    Polls ``check_status`` every ``--interval`` seconds. Exits 0 with the
    final payload on stdout when status is ``successful`` / ``failed`` /
    ``cancelled``; exits 1 if ``--timeout`` elapses first.
    """
    import time as _time

    async def _go() -> dict[str, Any]:
        async with _build_orchestrator_for_cli() as orch:
            deadline = _time.monotonic() + timeout
            while True:
                envelope = await orch.run(
                    backend="cmems",
                    operation="poll",
                    params={"request_id": request_id},
                )
                if "error" in envelope:
                    return envelope
                payload = _unwrap_result(envelope)
                status = payload.get("status")
                if status in _TERMINAL_STATUSES:
                    return envelope
                if _time.monotonic() >= deadline:
                    # Round 1 M2: route through the canonical record so
                    # downstream JSON consumers see the same shape as
                    # any other error envelope.
                    from copernicus_mcp.errors.records import (
                        build_error_record,
                    )

                    record = build_error_record(
                        "TimeoutError",
                        message=(
                            f"marine wait timed out after {timeout}s; last "
                            f"observed status={status!r}"
                        ),
                        recovery_action="retry_with_modification",
                    )
                    return {"error": record.model_dump(mode="json")}
                # Yield to the loop for ``interval`` seconds.
                await asyncio.sleep(max(0.0, interval))

    envelope = _run(_go())
    if "error" in envelope:
        _handle_error(envelope, json_out=json_out)
    _emit(_unwrap_result(envelope), json_out=json_out, title=request_id)


# ---------------------------------------------------------------------------
# CDS subcommands (T-CDS-007)
# ---------------------------------------------------------------------------


def _read_inputs_json(spec: str | None) -> dict[str, Any]:
    """Read the cdsapi-shaped ``inputs`` dict from a JSON file path or
    ``-`` for stdin. Used by ``cds estimate`` / ``cds submit``."""
    if not spec:
        typer.echo(
            "Error: --inputs-file is required (path to JSON or '-' for stdin)",
            err=True,
        )
        raise typer.Exit(code=EXIT_USER_INPUT)
    if spec == "-":
        text = sys.stdin.read()
    else:
        try:
            text = Path(spec).read_text()
        except OSError as exc:
            typer.echo(f"Error: cannot read {spec!r}: {exc}", err=True)
            raise typer.Exit(code=EXIT_USER_INPUT) from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        typer.echo(f"Error: --inputs-file is not valid JSON: {exc}", err=True)
        raise typer.Exit(code=EXIT_USER_INPUT) from exc
    if not isinstance(parsed, dict):
        typer.echo("Error: --inputs-file must contain a JSON object", err=True)
        raise typer.Exit(code=EXIT_USER_INPUT)
    return parsed


@cds_app.command("search")
def cds_search(
    keyword: str | None = typer.Option(None, "--keyword"),
    store: str | None = typer.Option(
        None, "--store", help="cds | ads | ewds — restrict to a single store."
    ),
    limit: int | None = typer.Option(None, "--limit"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Search the bundled CDS / ADS / EWDS catalogue snapshot."""
    params: dict[str, Any] = {}
    if keyword:
        params["keyword"] = keyword
    if store:
        params["store"] = store
    if limit is not None:
        params["limit"] = limit

    async def _go() -> dict[str, Any]:
        async with _build_orchestrator_for_cli() as orch:
            return await orch.run(backend="cds", operation="search", params=params)

    envelope = _run(_go())
    if "error" in envelope:
        _handle_error(envelope, json_out=json_out)
    _emit(_unwrap_result(envelope), json_out=json_out, title="CDS datasets")


@cds_app.command("describe")
def cds_describe(
    dataset_id: str = typer.Argument(..., help="Dataset id from cds search."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Show the full STAC record for a CDS / ADS / EWDS dataset."""

    async def _go() -> dict[str, Any]:
        async with _build_orchestrator_for_cli() as orch:
            return await orch.run(
                backend="cds",
                operation="describe",
                params={"identifier": dataset_id},
            )

    envelope = _run(_go())
    if "error" in envelope:
        _handle_error(envelope, json_out=json_out)
    _emit(_unwrap_result(envelope), json_out=json_out, title=dataset_id)


@cds_app.command("estimate")
def cds_estimate(
    dataset_id: str = typer.Option(..., "--dataset-id"),
    inputs_file: str | None = typer.Option(
        None,
        "--inputs-file",
        help="Path to a JSON file with the cdsapi-shaped inputs dict, or '-' for stdin.",
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Heuristic byte-size estimate for a CDS retrieve request."""
    inputs = _read_inputs_json(inputs_file)
    params = {"dataset_id": dataset_id, "inputs": inputs}

    async def _go() -> dict[str, Any]:
        async with _build_orchestrator_for_cli() as orch:
            return await orch.run(backend="cds", operation="estimate", params=params)

    envelope = _run(_go())
    if "error" in envelope:
        _handle_error(envelope, json_out=json_out)
    _emit(_unwrap_result(envelope), json_out=json_out, title="Estimate")


@cds_app.command("submit")
def cds_submit(
    dataset_id: str = typer.Option(..., "--dataset-id"),
    inputs_file: str | None = typer.Option(
        None,
        "--inputs-file",
        help="Path to a JSON file with the cdsapi-shaped inputs dict, or '-' for stdin.",
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Skip the size + queue-tier confirmation prompt."
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Submit a CDS / ADS / EWDS retrieve request.

    The CDS server queues every request; this command returns immediately
    after the server acknowledges. Track progress via
    ``cds check-status <request_id>`` or block via
    ``cds wait <request_id>``; download via ``cds download <request_id>
    --target <path>`` once the row reaches ``successful``.
    """
    inputs = _read_inputs_json(inputs_file)
    params = {"dataset_id": dataset_id, "inputs": inputs}

    async def _go() -> dict[str, Any]:
        async with _build_orchestrator_for_cli() as orch:
            envelope = await orch.run(backend="cds", operation="submit", params=params)
            if envelope.get("confirmation_required"):
                _show_confirmation(envelope)
                if not yes and not _interactive_confirm():
                    raise typer.Exit(code=EXIT_CONFIRMATION_ABORTED)
                envelope = await orch.run(
                    backend="cds",
                    operation="submit",
                    params=params,
                    options={"confirmed": True},
                )
            return envelope

    envelope = _run(_go())
    if "error" in envelope:
        _handle_error(envelope, json_out=json_out)
    _emit(_unwrap_result(envelope), json_out=json_out, title="Submit")


@cds_app.command("check-status")
def cds_check_status(
    request_id: str = typer.Argument(...),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Look up the status of an in-flight or completed CDS request."""

    async def _go() -> dict[str, Any]:
        async with _build_orchestrator_for_cli() as orch:
            return await orch.run(
                backend="cds",
                operation="poll",
                params={"request_id": request_id},
            )

    envelope = _run(_go())
    if "error" in envelope:
        _handle_error(envelope, json_out=json_out)
    _emit(_unwrap_result(envelope), json_out=json_out, title=request_id)


@cds_app.command("wait")
def cds_wait(
    request_id: str = typer.Argument(...),
    interval: float = typer.Option(10.0, "--interval", help="Seconds between polls."),
    timeout: float = typer.Option(
        7200.0,
        "--timeout",
        help=(
            "Hard timeout in seconds. CDS queue latency can run into "
            "hours; default is 2 hours so a long ERA5 request resolves "
            "in one wait. Exit 1 on expiry."
        ),
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Block until a CDS request reaches a terminal status.

    Polls ``check_status`` every ``--interval`` seconds. Exits 0 with
    the final payload on stdout when status is ``successful`` /
    ``failed`` / ``cancelled``; exits 1 if ``--timeout`` elapses.
    """
    import time as _time

    async def _go() -> dict[str, Any]:
        async with _build_orchestrator_for_cli() as orch:
            deadline = _time.monotonic() + timeout
            while True:
                envelope = await orch.run(
                    backend="cds",
                    operation="poll",
                    params={"request_id": request_id},
                )
                if "error" in envelope:
                    return envelope
                payload = _unwrap_result(envelope)
                status = payload.get("status")
                if status in _TERMINAL_STATUSES:
                    return envelope
                if _time.monotonic() >= deadline:
                    from copernicus_mcp.errors.records import build_error_record

                    record = build_error_record(
                        "TimeoutError",
                        message=(
                            f"cds wait timed out after {timeout}s; last observed status={status!r}"
                        ),
                        recovery_action="retry_with_modification",
                    )
                    return {"error": record.model_dump(mode="json")}
                await asyncio.sleep(max(0.0, interval))

    envelope = _run(_go())
    if "error" in envelope:
        _handle_error(envelope, json_out=json_out)
    _emit(_unwrap_result(envelope), json_out=json_out, title=request_id)


@cds_app.command("download")
def cds_download(
    request_id: str = typer.Argument(...),
    target: str | None = typer.Option(
        None,
        "--target",
        help=(
            "Optional target path (currently ignored — the backend "
            "always returns a descriptor pointing at the canonical "
            "cache location). Reserved for future use."
        ),
    ),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Resolve the cached file descriptor for a successful CDS request.

    No bytes are returned (the large-data invariant). The descriptor's
    ``filepath`` points at the canonical cache location; ``--target``
    is informational and need not be supplied.
    """

    from copernicus_mcp.backends.cds.tools import _TARGET_UNUSED_SENTINEL

    async def _go() -> dict[str, Any]:
        async with _build_orchestrator_for_cli() as orch:
            return await orch.run(
                backend="cds",
                operation="fetch",
                params={
                    "request_id": request_id,
                    "target": target or _TARGET_UNUSED_SENTINEL,
                },
            )

    envelope = _run(_go())
    if "error" in envelope:
        _handle_error(envelope, json_out=json_out)
    _emit(_unwrap_result(envelope), json_out=json_out, title=request_id)


@cds_app.command("cancel")
def cds_cancel(
    request_id: str = typer.Argument(...),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Cancel an in-flight CDS request (best-effort)."""

    async def _go() -> dict[str, Any]:
        async with _build_orchestrator_for_cli() as orch:
            return await orch.run(
                backend="cds",
                operation="cancel",
                params={"request_id": request_id},
            )

    envelope = _run(_go())
    if "error" in envelope:
        _handle_error(envelope, json_out=json_out)
    _emit(_unwrap_result(envelope), json_out=json_out, title=request_id)
