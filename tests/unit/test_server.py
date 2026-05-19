from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _config(tmp_path: Path) -> Any:
    from copernicus_mcp.config import ConfigLoader

    return ConfigLoader().load(
        cli_overrides={
            "storage": {
                "state_database": str(tmp_path / "state.db"),
                "cache_directory": str(tmp_path / "cache"),
            }
        }
    )


@pytest.mark.asyncio
async def test_instructions_mention_token_savings_via_disabling_backends(
    tmp_path: Path,
) -> None:
    """Server `instructions` (shown to clients on initialize handshake)
    must include a one-line nudge so users with only-CMEMS or only-CDS
    needs know they can trim the tool surface."""
    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.server import build_server

    config = _config(tmp_path)
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        server = build_server(config=config, foundation=foundation, registry=registry)
        instructions = server.instructions or ""
        # Welcome-style hint visible on the once-per-session initialize.
        assert "enabled_backends" in instructions
        assert "COPERNICUS_MCP_ENABLED_BACKENDS" in instructions
    finally:
        await foundation.persistence.close()


@pytest.mark.asyncio
async def test_instructions_surface_cache_location_and_override(
    tmp_path: Path,
) -> None:
    """`instructions` is the once-per-session handshake message the agent
    sees first. Without it knowing where downloaded files land (and how
    to change it), the agent can't tell the user "your NetCDF is at
    <path>". Must mention:
      - the actual resolved cache_directory path for this server, AND
      - the override mechanisms (COPERNICUS_MCP_CACHE_DIR or --cache-dir),
        so the agent can guide a user who wants a different location."""
    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.server import build_server

    config = _config(tmp_path)
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        server = build_server(config=config, foundation=foundation, registry=registry)
        instructions = server.instructions or ""
        # The actual resolved path, so the agent always knows.
        assert str(config.storage.cache_directory) in instructions, instructions
        # Override mechanism — at least one must appear, both is fine.
        assert "COPERNICUS_MCP_CACHE_DIR" in instructions or "--cache-dir" in (
            instructions
        ), instructions
    finally:
        await foundation.persistence.close()


@pytest.mark.asyncio
async def test_status_carries_override_hint(tmp_path: Path) -> None:
    """`copernicus_mcp_status` already returns ``cache.directory``; add
    a sibling ``cache.override_hint`` so the agent has a copy-paste-ready
    string for the user instead of synthesising one. Same shape as the
    instructions but per-call, so an agent that ignored the handshake
    can still find it."""
    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.server import build_server

    config = _config(tmp_path)
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        server = build_server(config=config, foundation=foundation, registry=registry)
        result = await server.call_tool("copernicus_mcp_status", {})
        structured = result[1] if isinstance(result, tuple) else result
        cache_block = structured.get("cache") or {}
        assert "directory" in cache_block, cache_block
        assert "override_hint" in cache_block, cache_block
        hint = cache_block["override_hint"]
        assert isinstance(hint, str) and hint, cache_block
        # Hint should reference at least one override mechanism by name.
        assert (
            "COPERNICUS_MCP_CACHE_DIR" in hint
            or "--cache-dir" in hint
            or "config.yaml" in hint
        ), hint
    finally:
        await foundation.persistence.close()


@pytest.mark.asyncio
async def test_build_server_registers_must_have_tools(tmp_path: Path) -> None:
    """Server must register all 6 CMEMS tools + copernicus_mcp_status.

    T-039 round 1 H1 added ``marine_check_status`` and
    ``marine_cancel_subset`` so MCP agents can drive the async submit
    lifecycle. Without them, ``async_mode=True`` returns a request_id
    that the agent cannot poll or cancel from MCP.
    """
    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.server import build_server

    config = _config(tmp_path)
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        server = build_server(config=config, foundation=foundation, registry=registry)
        tools = await server.list_tools()
        names = {t.name for t in tools}
        assert {
            "marine_search_datasets",
            "marine_describe_dataset",
            "marine_estimate_subset",
            "marine_subset_dataset",
            "marine_check_status",
            "marine_cancel_subset",
            "copernicus_mcp_status",
        } <= names
    finally:
        await foundation.persistence.close()


@pytest.mark.asyncio
async def test_status_tool_returns_expected_shape(tmp_path: Path) -> None:
    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.server import build_server

    config = _config(tmp_path)
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        server = build_server(config=config, foundation=foundation, registry=registry)
        result = await server.call_tool("copernicus_mcp_status", {})
        structured = result[1] if isinstance(result, tuple) else result
        assert "version" in structured
        assert "backends" in structured
    finally:
        await foundation.persistence.close()


@pytest.mark.asyncio
async def test_resources_registered(tmp_path: Path) -> None:
    """Iter 1.5 resources: datasets, files, provenance, jobs. ``jobs`` was
    deferred at T-031; T-039 wires it in alongside the async submit path."""
    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.server import build_server

    config = _config(tmp_path)
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        server = build_server(config=config, foundation=foundation, registry=registry)
        templates = await server.list_resource_templates()
        uri_patterns = {str(t.uriTemplate) for t in templates}
        assert any("datasets/cmems" in u for u in uri_patterns), uri_patterns
        assert any("files/" in u for u in uri_patterns), uri_patterns
        assert any("provenance/" in u for u in uri_patterns), uri_patterns
        assert any("jobs/" in u for u in uri_patterns), uri_patterns
    finally:
        await foundation.persistence.close()


@pytest.mark.asyncio
async def test_jobs_resource_returns_workflow_row(tmp_path: Path) -> None:
    """T-039: ``copernicus://jobs/{request_id}`` returns the workflow row
    serialised to JSON. Sanitised on the way out — defence in depth even
    though the row was sanitised at write time."""
    import json as _json

    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.server import build_server

    config = _config(tmp_path)
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        server = build_server(config=config, foundation=foundation, registry=registry)
        # Plant a row.
        await foundation.persistence.record_workflow(
            {
                "request_id": "req-test-jobs-1",
                "backend_id": "cmems",
                "operation": "submit",
                "status": "running",
                "cache_key": "cmems:submit:abc",
                "request_json": "{}",
                "response_json": None,
                "error_record_json": None,
                "created_at": "2026-05-05T00:00:00Z",
                "updated_at": "2026-05-05T00:00:00Z",
            }
        )

        body = await server.read_resource("copernicus://jobs/req-test-jobs-1")
        contents = body[0] if isinstance(body, (list, tuple)) else body
        text = getattr(contents, "content", contents)
        payload = _json.loads(str(text))
        assert payload["request_id"] == "req-test-jobs-1"
        assert payload["status"] == "running"
    finally:
        await foundation.persistence.close()


@pytest.mark.asyncio
async def test_jobs_resource_raises_on_missing_request_id(tmp_path: Path) -> None:
    """Missing request_id → protocol-level not-found, same contract as
    files / provenance resources."""
    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.server import build_server

    config = _config(tmp_path)
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        server = build_server(config=config, foundation=foundation, registry=registry)
        with pytest.raises((ValueError, Exception)) as exc_info:
            await server.read_resource("copernicus://jobs/no-such-request")
        assert "no-such-request" in str(exc_info.value)
    finally:
        await foundation.persistence.close()


@pytest.mark.asyncio
async def test_invalid_input_produces_structured_error(tmp_path: Path) -> None:
    """Validation failure surfaces through MCP without crashing the server."""
    from mcp.server.fastmcp.exceptions import ToolError

    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.server import build_server

    config = _config(tmp_path)
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        server = build_server(config=config, foundation=foundation, registry=registry)
        # describe with a blank dataset_id triggers Pydantic ValidationError;
        # FastMCP surfaces this as a ToolError. The server must NOT crash.
        with pytest.raises(ToolError):
            await server.call_tool(
                "marine_describe_dataset", {"input": {"dataset_id": "   "}}
            )
    finally:
        await foundation.persistence.close()


@pytest.mark.asyncio
async def test_files_resource_raises_on_cache_miss(tmp_path: Path) -> None:
    """codex T-031 round 2 MEDIUM: empty-string body ambiguates a miss
    from a successful empty read. Miss raises so the MCP client gets a
    protocol-level not-found. FastMCP wraps the inner ResourceError as
    a ValueError at the template-resolution layer; the wire effect is
    that ``read_resource`` does not return a success-with-empty-body."""
    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.server import build_server

    config = _config(tmp_path)
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        server = build_server(config=config, foundation=foundation, registry=registry)
        with pytest.raises((ValueError, Exception)) as exc_info:
            await server.read_resource("copernicus://files/nonexistent-key")
        assert "nonexistent-key" in str(exc_info.value)
    finally:
        await foundation.persistence.close()


@pytest.mark.asyncio
async def test_files_resource_path_sanitised_against_credential_shaped_dir(
    tmp_path: Path,
) -> None:
    """codex final-batch HIGH: the file resource returned the raw cache
    path without sanitisation. A user who misconfigured cache_directory
    at a credential-shaped path (or whose path was rewritten by an
    upstream tool) would leak the secret via ``read_resource``.

    Same discipline as ``status()`` — sanitise at the MCP output boundary.
    """
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.backends.registry import BackendRegistry
    from copernicus_mcp.bootstrap import build_foundation
    from copernicus_mcp.server import build_server

    weird_dir = tmp_path / "password=hunter2"
    weird_dir.mkdir()
    config = _config(tmp_path)
    object.__setattr__(config.storage, "cache_directory", weird_dir)
    foundation = await build_foundation(config)
    try:
        registry = BackendRegistry()
        server = build_server(config=config, foundation=foundation, registry=registry)

        # Plant a real cache entry pointing under the weird directory.
        target_dir = weird_dir / "cmems"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "data.nc"
        target.write_bytes(b"x")
        # Use the cache manager API to record the entry so lookup_file finds it.
        from datetime import UTC, datetime

        # T-039 round 3: cache stores under ``file:{cache_key}`` and the
        # resource handler bridges. URL path is the bare key.
        await foundation.persistence.record_cache_entry(
            {
                "namespace": "file",
                "key": "file:ck-leak",
                "value_json": "{}",
                "file_path": str(target),
                "size_bytes": 1,
                "content_type": "application/octet-stream",
                "created_at": datetime.now(UTC).isoformat(),
                "last_accessed_at": datetime.now(UTC).isoformat(),
            }
        )

        # Suppress the unused-import warning for CmemsBackend (kept as
        # documentation for the test's intent — file resource is owned by
        # the server layer, not a specific backend).
        _ = CmemsBackend

        body = await server.read_resource("copernicus://files/ck-leak")
        contents = body[0] if isinstance(body, (list, tuple)) else body
        text = getattr(contents, "content", contents)
        text_str = str(text)
        assert "hunter2" not in text_str, text_str
        assert "[REDACTED]" in text_str
    finally:
        await foundation.persistence.close()


@pytest.mark.asyncio
async def test_files_resource_expands_manifest_entry(tmp_path: Path) -> None:
    """T-CMEMS-GET-006: a cache entry with ``content_type ==
    application/x.cmems-get-manifest+json`` resolves through the
    file resource as a JSON envelope of per-file descriptors,
    rather than the manifest path string. Non-manifest entries keep
    the existing string shape (covered by the existing tests above).
    """
    from datetime import UTC, datetime

    from copernicus_mcp.backends.cmems._get_manifest import build_manifest
    from copernicus_mcp.backends.registry import BackendRegistry
    from copernicus_mcp.bootstrap import build_foundation
    from copernicus_mcp.cache import MANIFEST_CONTENT_TYPE
    from copernicus_mcp.server import build_server

    config = _config(tmp_path)
    foundation = await build_foundation(config)
    try:
        registry = BackendRegistry()
        server = build_server(config=config, foundation=foundation, registry=registry)

        # Build a real bundle directory inside the cache zone.
        zone = foundation.cache.cache_zone_for("cmems")
        bundle = zone / "bundle-abc"
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "a.nc").write_bytes(b"alpha")
        (bundle / "b.nc").write_bytes(b"bravo")
        cache_key = "cmems:get:ds:abc"
        manifest_path = build_manifest(
            directory=bundle,
            cache_key=cache_key,
            dataset_id="ds",
            dataset_version=None,
        )
        # Register via the cache manager so the path-safety + size
        # accounting matches production.
        await foundation.cache.store_manifest(
            cache_key=f"file:{cache_key}",
            manifest_path=manifest_path,
            data_dir=bundle,
            backend_id="cmems",
        )

        body = await server.read_resource(f"copernicus://files/{cache_key}")
        contents = body[0] if isinstance(body, (list, tuple)) else body
        text = getattr(contents, "content", contents)
        text_str = str(text)
        # Returned JSON envelope, not a single path string.
        payload = json.loads(text_str)
        assert "files" in payload
        rels = {Path(f["filepath"]).name for f in payload["files"]}
        assert rels == {"a.nc", "b.nc"}
        # The dispatch trigger is the manifest content-type marker
        # — verify the persistence row indeed carries it.
        registered = await foundation.persistence.lookup_cache_entry(
            "file", f"file:{cache_key}"
        )
        assert registered is not None
        assert registered["content_type"] == MANIFEST_CONTENT_TYPE
        # Sanity: ``last_accessed_at`` is set (the LRU bump path
        # is exercised, cr round-1 MEDIUM regression guard).
        assert registered.get("last_accessed_at")
        # Datetime parsing happens elsewhere; just confirm the bump
        # is recent (within the test run).
        bumped = datetime.fromisoformat(
            str(registered["last_accessed_at"]).replace("Z", "+00:00")
        )
        assert (datetime.now(UTC) - bumped).total_seconds() < 10
    finally:
        await foundation.persistence.close()


@pytest.mark.asyncio
async def test_files_resource_corrupt_manifest_raises_resource_error(
    tmp_path: Path,
) -> None:
    """cr round-1 HIGH: a corrupt ``manifest.json`` used to propagate
    ``json.JSONDecodeError`` out of MCP. The handler now treats it
    as a protocol-level miss via ``ResourceError``."""
    from copernicus_mcp.backends.cmems._get_manifest import build_manifest
    from copernicus_mcp.backends.registry import BackendRegistry
    from copernicus_mcp.bootstrap import build_foundation
    from copernicus_mcp.server import build_server

    config = _config(tmp_path)
    foundation = await build_foundation(config)
    try:
        registry = BackendRegistry()
        server = build_server(config=config, foundation=foundation, registry=registry)

        zone = foundation.cache.cache_zone_for("cmems")
        bundle = zone / "bundle-corrupt"
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "a.nc").write_bytes(b"x")
        cache_key = "cmems:get:ds:corrupt"
        manifest_path = build_manifest(
            directory=bundle,
            cache_key=cache_key,
            dataset_id="ds",
            dataset_version=None,
        )
        await foundation.cache.store_manifest(
            cache_key=f"file:{cache_key}",
            manifest_path=manifest_path,
            data_dir=bundle,
            backend_id="cmems",
        )
        # Tamper with the manifest after registration.
        manifest_path.write_text("{ not valid json")

        with pytest.raises((ValueError, Exception)) as exc_info:
            await server.read_resource(f"copernicus://files/{cache_key}")
        assert "manifest unreadable" in str(exc_info.value)
    finally:
        await foundation.persistence.close()


def test_main_loads_config_without_crashing(monkeypatch: Any, tmp_path: Path) -> None:
    """``main()`` must not import-error or crash on stub invocation."""
    import copernicus_mcp.server as server_mod

    # Patch ``serve`` so we don't actually start the stdio loop.
    called: dict[str, bool] = {}

    async def _fake_serve(config: Any) -> None:
        called["yes"] = True

    monkeypatch.setattr(server_mod, "serve", _fake_serve)
    monkeypatch.setenv(
        "COPERNICUS_MCP_STORAGE__STATE_DATABASE", str(tmp_path / "state.db")
    )
    monkeypatch.setenv(
        "COPERNICUS_MCP_STORAGE__CACHE_DIRECTORY", str(tmp_path / "cache")
    )
    server_mod.main()
    assert called.get("yes") is True
