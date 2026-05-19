"""MCP server entrypoint (T-031).

Wires the orchestrator + CMEMS tools to a FastMCP stdio server. Three
resource templates are registered: ``copernicus://datasets/cmems/{id}``,
``copernicus://files/{cache_key}``, and ``copernicus://provenance/{record_id}``.
``copernicus://jobs/{request_id}`` is deferred (logged in
).

project invariants this layer enforces:
- #1 large-data: tools return descriptors only (enforced upstream in
  ``backends/cmems``); resource ``files/{key}`` returns a path string,
  never bytes.
- #2 credential isolation: orchestrator results pass through Sanitiser
  before reaching FastMCP's structuredContent.
- #3 cancellation: no ``except`` for CancelledError.
- #4 stdout discipline: stdio is the MCP transport; logger writes to
  stderr per ``observability.setup_logging`` config.

Type-strictness: this module is deliberately excluded from
``mypy --strict`` per plan T-031 step 6 (MCP SDK stubs evolving).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from copernicus_mcp.backends.abstract import FoundationServices
from copernicus_mcp.backends.cds.tools import register_cds_tools
from copernicus_mcp.backends.cmems.tools import register_marine_tools
from copernicus_mcp.backends.registry import BackendRegistry
from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
from copernicus_mcp.config import ConfigLoader, CopernicusMcpConfig
from copernicus_mcp.observability.logger import get_logger, setup_logging
from copernicus_mcp.version import __version__
from copernicus_mcp.workflow.orchestrator import WorkflowOrchestrator

logger = get_logger(__name__)


def build_server(
    *,
    config: CopernicusMcpConfig,
    foundation: FoundationServices,
    registry: BackendRegistry,
) -> FastMCP:
    """Construct a FastMCP server with all tools and resources registered.

    Split out from ``serve()`` so tests can drive the server in-process via
    ``call_tool`` / ``read_resource`` without spawning the stdio loop.
    """
    # mcp 1.27 FastMCP does NOT expose ``serverInfo.version`` via the
    # constructor; the wire value defaults to the SDK package version.
    # Surfacing our version here would require subclassing the low-level
    # Server. Deferred — clients can still get our version via the
    # ``copernicus_mcp_status`` tool..
    # The cache path embedded below is either the platformdirs-derived
    # OS default (our code, no user input) OR a user-configured path
    # via env/CLI/yaml. We pass it through as-is: by design we don't
    # sanitise user-controlled config — if a user names their directory
    # ``/tmp/key=hunter2/...`` that's their decision. project invariants
    # invariant 2 protects against US injecting credentials; it does
    # not require us to scrub user-owned paths.
    server = FastMCP(
        "copernicus-mcp",
        instructions=(
            f"Copernicus MCP server v{__version__}. Provides discovery, "
            "estimation, and async/sync subset retrieval tools for "
            "Copernicus Marine (CMEMS) datasets and Climate Data Store "
            "family datasets (CDS / ADS / EWDS), plus a status "
            "diagnostic. CDS tools are registered only when the cds "
            "backend is in enabled_backends AND credentials resolve. "
            "Tip: both backends are on by default; if you only need "
            "one, trim the tool surface (and the LLM context cost) by "
            "setting COPERNICUS_MCP_ENABLED_BACKENDS=cmems (or =cds), "
            "or override enabled_backends in config.yaml. "
            f"Downloaded files land under {config.storage.cache_directory} "
            "(per-OS default via platformdirs). Override with the "
            "COPERNICUS_MCP_CACHE_DIR env var, the --cache-dir CLI flag, "
            "or storage.cache_directory in config.yaml — when a user asks "
            "where their data is, surface this path."
        ),
    )
    orchestrator = WorkflowOrchestrator(registry=registry, foundation=foundation)

    register_marine_tools(server, orchestrator=orchestrator)

    # T-CDS-007: register CDS tools only when CDS is both opted-in
    # (``cds`` in ``enabled_backends`` ⇒ registry has the backend
    # registered) AND credentials actually resolve.
    #
    # The two signals are NOT redundant — earlier rounds showed each
    # alone misroutes a real user case:
    #   - Round-2 (``is_configured`` only): registers when user listed
    #     ``cds`` in config but has no PAT — bootstrap registers the
    #     backend even when credentials are missing (with a warning),
    #     so ``is_configured`` is True. Phantom tools advertised.
    #   - Round-3 round-2-fix (``resolve != None`` only): registers
    #     when user has ``CDSAPI_KEY`` in env (common — set for the
    #     standalone ``cdsapi`` CLI) but kept the default
    #     ``enabled_backends=["cmems"]``. Registry has no cds entry,
    #     every tool call returns ``backend_not_configured`` with a
    #     misleading ``configure_credentials`` recovery action.
    # Combining catches both. Regression tests pin each direction
    # individually.
    #
    # Round-1 cr M2: ``register_marine_tools`` runs unconditionally
    # above (legacy behaviour). The CDS gating is intentional; aligning
    # CMEMS to the same discipline is deferred to T-CDS-009.
    if (
        registry.is_configured("cds")
        and foundation.credential_resolver.resolve("cds") is not None
    ):
        register_cds_tools(server, orchestrator=orchestrator)

    @server.tool(
        name="copernicus_mcp_status",
        description=(
            "Return server diagnostics: registered backends, credential "
            "sources, cache metrics, persistence path, and config snapshot. "
            "No credential values appear in the output."
        ),
    )
    async def _status() -> dict[str, Any]:
        return await orchestrator.status()

    @server.resource("copernicus://datasets/cmems/{dataset_id}")
    async def _dataset_resource(dataset_id: str) -> str:
        envelope = await orchestrator.run(
            backend="cmems",
            operation="describe",
            params={"identifier": dataset_id},
        )
        return json.dumps(envelope, default=str)

    @server.resource("copernicus://files/{cache_key}")
    async def _file_resource(cache_key: str) -> str:
        # Returns the filesystem path as a plain string. The MCP client
        # opens it locally — large-data invariant #1: never serve bytes.
        # Cache miss → raise ResourceError so the MCP client gets a proper
        # protocol-level not-found instead of an empty success body
        # (codex T-031 round 2 MEDIUM).
        # Path sanitisation: same discipline as ``status()`` — a user who
        # configures cache_directory at e.g. ``/tmp/password=hunter2/...``
        # must not see the secret leak through the resource boundary
        # (codex final-batch HIGH; the credential-isolation invariant).
        # T-039 round 3: the URL carries a bare cache_key for clean URIs,
        # while the cache stores under ``f"file:{cache_key}"``. Try both
        # so URIs from any historical caller resolve.
        #
        # T-CMEMS-GET-006: a manifest entry (content_type
        # ``application/x.cmems-get-manifest+json``) resolves to a
        # JSON envelope listing all bundle files — single-path return
        # would only point at ``manifest.json`` and force the client
        # to parse it. Dispatch on content_type so manifest entries
        # expand and single-file entries keep their existing string
        # shape.
        from mcp.server.fastmcp.exceptions import ResourceError

        from copernicus_mcp.backends.cmems._get_manifest import (
            read_manifest,
            to_files_envelope,
        )
        from copernicus_mcp.cache import MANIFEST_CONTENT_TYPE

        # ``lookup_entry`` preserves the LRU bump and dead-row
        # cleanup that ``lookup_file`` did (cr round-1 MEDIUMs:
        # the previous T-CMEMS-GET-006 dispatch bypassed both).
        entry = await foundation.cache.lookup_entry(
            f"file:{cache_key}"
        ) or await foundation.cache.lookup_entry(cache_key)
        if entry is None:
            raise ResourceError(f"cache_key {cache_key!r} not found")
        path_str = entry.get("file_path")
        if path_str is None:  # already covered by lookup_entry; defence in depth
            raise ResourceError(f"cache_key {cache_key!r} not found")
        from pathlib import Path as _Path

        path = _Path(path_str)

        if entry.get("content_type") == MANIFEST_CONTENT_TYPE:
            # cr round-1 HIGH: a corrupt ``manifest.json`` raised
            # ``json.JSONDecodeError`` straight out of MCP. Wrap it
            # as a ``ResourceError`` so the client sees a clean
            # protocol-level miss (matches the ``_jobs_resource``
            # pattern at the corrupt error_record_json branch).
            try:
                manifest = read_manifest(path.parent)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "files resource: manifest unreadable; treating as miss",
                    extra={
                        "cache_key": cache_key,
                        "error_class": type(exc).__name__,
                    },
                )
                raise ResourceError(
                    f"cache_key {cache_key!r} manifest unreadable"
                ) from None
            if manifest is None:
                raise ResourceError(f"cache_key {cache_key!r} manifest unreadable")
            envelope = to_files_envelope(manifest=manifest, data_dir=path.parent)
            sanitised_envelope = foundation.sanitiser.sanitise(envelope)
            return json.dumps(sanitised_envelope, default=str)

        sanitised: str = foundation.sanitiser.sanitise(str(path))
        return sanitised

    @server.resource("copernicus://jobs/{request_id}")
    async def _jobs_resource(request_id: str) -> str:
        # T-039: surface the workflow row for an in-flight or completed
        # submit. Same not-found discipline as files/provenance — raise
        # ResourceError so the MCP client sees a protocol-level miss.
        # codex round 2 MEDIUM: ``error_record_json`` is a JSON STRING; the
        # sanitiser regex-walks string values but can't see structural
        # context like sensitive dict-key names. Parse it so the structural
        # walk applies, then re-serialise.
        from mcp.server.fastmcp.exceptions import ResourceError

        row = await foundation.persistence.fetch_workflow(request_id)
        if row is None:
            raise ResourceError(f"workflow {request_id!r} not found")
        payload: dict[str, Any] = dict(row)
        raw_err = payload.get("error_record_json")
        if isinstance(raw_err, str):
            try:
                payload["error_record_json"] = json.loads(raw_err)
            except json.JSONDecodeError:
                # Leave as string; the regex layer still applies.
                pass
        sanitised = foundation.sanitiser.sanitise(payload)
        return json.dumps(sanitised, default=str)

    @server.resource("copernicus://provenance/{record_id}")
    async def _provenance_resource(record_id: str) -> str:
        # Persistence row is a TypedDict with ``provenance_json`` holding the
        # canonicalised record. Parse and re-sanitise — defence-in-depth: the
        # row was already sanitised at write time, but a future schema change
        # could regress and the resource boundary is the right place to guard.
        from mcp.server.fastmcp.exceptions import ResourceError

        row = await foundation.persistence.fetch_provenance(record_id)
        if row is None:
            # Same not-found contract as the files resource — protocol
            # error rather than a JSON-shaped success body.
            raise ResourceError(f"provenance record {record_id!r} not found")
        record_payload = json.loads(row["provenance_json"])
        sanitised = foundation.sanitiser.sanitise(record_payload)
        return json.dumps(sanitised, default=str)

    return server


async def serve(config: CopernicusMcpConfig) -> None:
    """Start the MCP server over stdio.

    Builds foundation + registry + orchestrator, registers tools and
    resources, then runs the FastMCP stdio loop until cancelled.
    """
    setup_logging(config.observability)
    logger.info("starting copernicus-mcp server")

    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        server = build_server(
            config=config, foundation=foundation, registry=registry
        )
        # FastMCP's stdio runner reads stdin / writes stdout. The structured
        # logger continues writing to stderr (invariant #4).
        # TODO(T-031 cancellation): MCP ``notifications/cancelled`` is handled
        # by FastMCP's stdio loop internally — verified end-to-end in T-038
        # release-readiness checks. If a future SDK release exposes a
        # cancellation hook, wire it here so OperationCancelledError
        # propagates with our canonical error_class.
        await server.run_stdio_async()
    finally:
        await foundation.persistence.close()


def main(cli_overrides: dict[str, Any] | None = None) -> None:
    """Synchronous entrypoint used by the ``copernicus-mcp serve`` script.

    ``cli_overrides`` lets the CLI forward top-level flags like
    ``--cache-dir`` (T-CDS-019) into the config-load merge so a one-shot
    ``copernicus-mcp --cache-dir X serve`` honours the override.
    """
    config = ConfigLoader().load(cli_overrides=cli_overrides)
    asyncio.run(serve(config))
