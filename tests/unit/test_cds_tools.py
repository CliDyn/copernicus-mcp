"""Tests for the CDS MCP tool surface (T-CDS-007).

Mirror of ``test_cmems_tools.py`` for the seven CDS tools. Each tool
maps a Pydantic-validated input to an orchestrator call and unwraps
the envelope. Errors raised by the orchestrator are surfaced as
``ToolReturnedError`` so FastMCP renders them as ``isError=true``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest


def _good_submit_payload() -> dict[str, Any]:
    return {
        "dataset_id": "reanalysis-era5-single-levels",
        "inputs": {
            "variable": ["2m_temperature"],
            "year": ["2024"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00"],
        },
    }


# ---------------------------------------------------------------------------
# Per-tool orchestrator-call shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cds_search_datasets_dispatches_search() -> None:
    from copernicus_mcp.backends.cds.tools import (
        CdsSearchDatasetsInput,
        cds_search_datasets,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"datasets": [], "total_count": 0}}
    await cds_search_datasets(
        CdsSearchDatasetsInput(keyword="temperature", limit=5),
        orchestrator=orch,
    )
    assert orch.run.call_args.kwargs["backend"] == "cds"
    assert orch.run.call_args.kwargs["operation"] == "search"
    assert orch.run.call_args.kwargs["params"]["keyword"] == "temperature"


@pytest.mark.asyncio
async def test_cds_search_datasets_forwards_bbox_time_variable() -> None:
    """T-CDS-020 PR-1: the new filter fields reach the orchestrator
    params dict so the catalogue layer can apply them."""
    from copernicus_mcp.backends.cds.tools import (
        CdsSearchDatasetsInput,
        cds_search_datasets,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"datasets": [], "total_count": 0}}
    await cds_search_datasets(
        CdsSearchDatasetsInput(
            bbox=(20.0, 30.0, 30.0, 60.0),
            time_range=("2010-01-01T00:00:00Z", "2010-12-31T23:59:59Z"),
            variable="temperature",
            limit=5,
        ),
        orchestrator=orch,
    )
    params = orch.run.call_args.kwargs["params"]
    assert params["bbox"] == (20.0, 30.0, 30.0, 60.0)
    assert params["time_range"] == (
        "2010-01-01T00:00:00Z",
        "2010-12-31T23:59:59Z",
    )
    assert params["variable"] == "temperature"


@pytest.mark.asyncio
async def test_cds_describe_dataset_maps_dataset_id_to_identifier() -> None:
    """The orchestrator's ``describe`` dispatch reads ``params['identifier']``;
    the tool input field is ``dataset_id`` for user clarity."""
    from copernicus_mcp.backends.cds.tools import (
        CdsDescribeDatasetInput,
        cds_describe_dataset,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"id": "reanalysis-era5-single-levels"}}
    await cds_describe_dataset(
        CdsDescribeDatasetInput(dataset_id="reanalysis-era5-single-levels"),
        orchestrator=orch,
    )
    params = orch.run.call_args.kwargs["params"]
    assert params == {"identifier": "reanalysis-era5-single-levels"}


@pytest.mark.asyncio
async def test_cds_estimate_request_dispatches_estimate() -> None:
    from copernicus_mcp.backends.cds.tools import (
        CdsEstimateRequestInput,
        cds_estimate_request,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"estimated_size_bytes": 1234}}
    await cds_estimate_request(
        CdsEstimateRequestInput(**_good_submit_payload()),
        orchestrator=orch,
    )
    assert orch.run.call_args.kwargs["operation"] == "estimate"


@pytest.mark.asyncio
async def test_cds_submit_request_forwards_confirmed_flag() -> None:
    """The ``confirmed`` field is plumbed through ``__options`` (orchestrator
    kwarg ``options``) — not as a Pydantic-validated request field."""
    from copernicus_mcp.backends.cds.tools import (
        CdsSubmitRequestInput,
        cds_submit_request,
    )

    orch = AsyncMock()
    orch.run.return_value = {
        "result": {"status": "queued", "request_id": "abc"}
    }
    payload = {**_good_submit_payload(), "confirmed": True}
    await cds_submit_request(
        CdsSubmitRequestInput(**payload), orchestrator=orch
    )
    kwargs = orch.run.call_args.kwargs
    assert kwargs["operation"] == "submit"
    # confirmed lives in options, not in params
    assert "confirmed" not in kwargs["params"]
    assert kwargs["options"] == {"confirmed": True}


@pytest.mark.asyncio
async def test_cds_submit_request_forwards_chunk_options() -> None:
    """T-CDS-CHUNK: chunk_by / auto_chunk / force_refresh are plumbed through
    ``__options`` and never reach the validated request params."""
    from copernicus_mcp.backends.cds.tools import (
        CdsSubmitRequestInput,
        cds_submit_request,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"status": "queued", "request_id": "p"}}
    payload = {
        **_good_submit_payload(),
        "chunk_by": "month",
        "auto_chunk": False,
        "force_refresh": True,
    }
    await cds_submit_request(CdsSubmitRequestInput(**payload), orchestrator=orch)
    kwargs = orch.run.call_args.kwargs
    for key in ("chunk_by", "auto_chunk", "force_refresh"):
        assert key not in kwargs["params"]
    assert kwargs["options"] == {
        "chunk_by": "month",
        "auto_chunk": False,
        "force_refresh": True,
    }


@pytest.mark.asyncio
async def test_cds_submit_request_forwards_confirm_large_fanout() -> None:
    """T-CDS-CHUNK fan-out: confirm_large_fanout is the second, deliberate ack for
    a > reconfirm_above split; it rides __options, never the request params."""
    from copernicus_mcp.backends.cds.tools import (
        CdsSubmitRequestInput,
        cds_submit_request,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"status": "queued", "request_id": "p"}}
    payload = {
        **_good_submit_payload(),
        "confirmed": True,
        "confirm_large_fanout": True,
    }
    await cds_submit_request(CdsSubmitRequestInput(**payload), orchestrator=orch)
    kwargs = orch.run.call_args.kwargs
    assert "confirm_large_fanout" not in kwargs["params"]
    assert kwargs["options"] == {"confirmed": True, "confirm_large_fanout": True}


@pytest.mark.asyncio
async def test_cds_submit_request_omits_default_chunk_options() -> None:
    """Defaults (chunk_by=None, auto_chunk=True, force_refresh=False) add nothing
    to options — a plain submit stays option-free."""
    from copernicus_mcp.backends.cds.tools import (
        CdsSubmitRequestInput,
        cds_submit_request,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"status": "queued", "request_id": "p"}}
    await cds_submit_request(
        CdsSubmitRequestInput(**_good_submit_payload()), orchestrator=orch
    )
    assert orch.run.call_args.kwargs["options"] is None


@pytest.mark.asyncio
async def test_cds_submit_request_default_no_options() -> None:
    """Without ``confirmed=True``, options must be ``None`` so the
    orchestrator does not inject ``__options`` at all."""
    from copernicus_mcp.backends.cds.tools import (
        CdsSubmitRequestInput,
        cds_submit_request,
    )

    orch = AsyncMock()
    orch.run.return_value = {
        "result": {"status": "queued", "request_id": "abc"}
    }
    await cds_submit_request(
        CdsSubmitRequestInput(**_good_submit_payload()),
        orchestrator=orch,
    )
    assert orch.run.call_args.kwargs["options"] is None


@pytest.mark.asyncio
async def test_cds_check_request_status_passes_request_id() -> None:
    from copernicus_mcp.backends.cds.tools import (
        CdsCheckRequestStatusInput,
        cds_check_request_status,
    )

    orch = AsyncMock()
    orch.run.return_value = {
        "result": {"status": "running", "request_id": "rid"}
    }
    await cds_check_request_status(
        CdsCheckRequestStatusInput(request_id="rid"),
        orchestrator=orch,
    )
    assert orch.run.call_args.kwargs["operation"] == "poll"
    assert orch.run.call_args.kwargs["params"] == {"request_id": "rid"}


@pytest.mark.asyncio
async def test_cds_download_request_result_dispatches_fetch() -> None:
    """``download_request_result`` maps to the orchestrator's ``fetch``
    operation (which calls ``CdsBackend.fetch_result``)."""
    from copernicus_mcp.backends.cds.tools import (
        CdsDownloadRequestResultInput,
        cds_download_request_result,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"filepath": "/tmp/out.bin"}}
    await cds_download_request_result(
        CdsDownloadRequestResultInput(request_id="rid", target="/tmp/out.bin"),
        orchestrator=orch,
    )
    kwargs = orch.run.call_args.kwargs
    assert kwargs["operation"] == "fetch"
    assert kwargs["params"]["request_id"] == "rid"
    assert kwargs["params"]["target"] == "/tmp/out.bin"


@pytest.mark.asyncio
async def test_cds_cancel_request_dispatches_cancel() -> None:
    from copernicus_mcp.backends.cds.tools import (
        CdsCancelRequestInput,
        cds_cancel_request,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"cancelled": True, "request_id": "rid"}}
    await cds_cancel_request(
        CdsCancelRequestInput(request_id="rid"), orchestrator=orch
    )
    assert orch.run.call_args.kwargs["operation"] == "cancel"


# ---------------------------------------------------------------------------
# Envelope unwrap behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_returned_error_raised_on_orchestrator_error() -> None:
    from copernicus_mcp.backends.cds.tools import (
        CdsCheckRequestStatusInput,
        ToolReturnedError,
        cds_check_request_status,
    )

    orch = AsyncMock()
    orch.run.return_value = {
        "error": {
            "error_class": "TermsNotAcceptedError",
            "recovery_url": "https://x/licence",
            "context": {},
        }
    }
    with pytest.raises(ToolReturnedError) as exc_info:
        await cds_check_request_status(
            CdsCheckRequestStatusInput(request_id="rid"),
            orchestrator=orch,
        )
    assert exc_info.value.record["error_class"] == "TermsNotAcceptedError"


@pytest.mark.asyncio
async def test_confirmation_required_envelope_passed_through() -> None:
    """``confirmation_required`` envelopes are not errors — they need to
    reach the LLM agent / CLI so the user can approve / decline.
    Mirror of the CMEMS tool behaviour."""
    from copernicus_mcp.backends.cds.tools import (
        CdsSubmitRequestInput,
        cds_submit_request,
    )

    confirm_payload = {
        "confirmation_required": True,
        "reason": "estimated_size_threshold_exceeded",
        "estimated_size_bytes": 5_000_000_000,
    }
    orch = AsyncMock()
    orch.run.return_value = confirm_payload
    out = await cds_submit_request(
        CdsSubmitRequestInput(**_good_submit_payload()),
        orchestrator=orch,
    )
    assert out == confirm_payload


# ---------------------------------------------------------------------------
# register_cds_tools wiring
# ---------------------------------------------------------------------------


def test_check_request_status_docstring_discourages_sleep_loops() -> None:
    """Poll-pacing guidance for the agent. The Claude Code harness blocks
    standalone `sleep N` commands precisely to discourage naive
    sleep-then-retry patterns; if our docstring doesn't tell the agent
    what to do instead, the agent invents brittle workarounds (e.g.
    `until [ -f /tmp/...]; do sleep 5; ...`). Surface the correct
    pattern: don't sleep on the client side; let the user/MCP host
    pace the next call, or use the CLI ``cds wait`` for offline polling."""
    from copernicus_mcp.backends.cds.tools import cds_check_request_status

    doc = cds_check_request_status.__doc__ or ""
    lo = doc.lower()
    assert "sleep" in lo, doc
    assert "cds wait" in lo, doc


def test_describe_docstring_marks_available_inputs_as_snapshot() -> None:
    """T-CDS-017: agents must know ``available_inputs`` on
    cds_describe_dataset is a SNAPSHOT — for live server-side truth they
    should call ``cds_apply_constraints``. Docstring is registered as the
    MCP tool description, so it's the agent's primary signal."""
    from copernicus_mcp.backends.cds.tools import cds_describe_dataset

    doc = cds_describe_dataset.__doc__ or ""
    lo = doc.lower()
    # Marks the snapshot nature explicitly.
    assert "snapshot" in lo or "may be stale" in lo, doc
    # Points at the live tool by name.
    assert "cds_apply_constraints" in doc, doc


def test_register_cds_tools_registers_all_nine() -> None:
    from copernicus_mcp.backends.cds.tools import register_cds_tools

    registered: dict[str, Any] = {}

    class _FakeServer:
        def tool(self, *, name: str, description: str | None = None) -> Any:
            def _decorator(fn: Any) -> Any:
                registered[name] = (fn, description)
                return fn

            return _decorator

    register_cds_tools(_FakeServer(), orchestrator=AsyncMock())
    expected = {
        "cds_search_groups",  # T-CDS-021 PR-2: hierarchical discovery
        "cds_search_datasets",
        "cds_describe_dataset",
        "cds_apply_constraints",  # T-CDS-016 Layer B
        "cds_estimate_request",
        "cds_submit_request",
        "cds_check_request_status",
        "cds_download_request_result",
        "cds_cancel_request",
    }
    assert expected <= set(registered.keys())
    for name in expected:
        _, description = registered[name]
        assert description, f"{name} has empty description"


@pytest.mark.asyncio
async def test_cds_search_groups_dispatches_search_groups() -> None:
    """T-CDS-021 PR-2: the tool calls orchestrator with
    ``operation="search_groups"`` and forwards query/top_k."""
    from copernicus_mcp.backends.cds.tools import (
        CdsSearchGroupsInput,
        cds_search_groups,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"groups": [], "total_count": 0}}
    await cds_search_groups(
        CdsSearchGroupsInput(query="atmosphere temperature", top_k=3),
        orchestrator=orch,
    )
    assert orch.run.call_args.kwargs["backend"] == "cds"
    assert orch.run.call_args.kwargs["operation"] == "search_groups"
    params = orch.run.call_args.kwargs["params"]
    assert params["query"] == "atmosphere temperature"
    assert params["top_k"] == 3


@pytest.mark.asyncio
async def test_cds_search_datasets_forwards_domain_and_category() -> None:
    """T-CDS-021 PR-2: chaining from groups → datasets passes the
    ``domain``/``category`` filter pair through to the orchestrator."""
    from copernicus_mcp.backends.cds.tools import (
        CdsSearchDatasetsInput,
        cds_search_datasets,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"datasets": [], "total_count": 0}}
    await cds_search_datasets(
        CdsSearchDatasetsInput(
            domain="Atmosphere (surface)",
            category="reanalysis",
        ),
        orchestrator=orch,
    )
    params = orch.run.call_args.kwargs["params"]
    assert params["domain"] == "Atmosphere (surface)"
    assert params["category"] == "reanalysis"


def test_register_cds_tools_registered_callable_has_typed_input() -> None:
    """FastMCP derives JSON schema from the registered callable's
    annotations; ``input: Any`` would yield an empty schema. The
    registered wrapper must declare a concrete Pydantic model
    annotation. Same property as the CMEMS-side regression test."""
    import inspect

    from copernicus_mcp.backends.cds.tools import (
        CdsSearchDatasetsInput,
        register_cds_tools,
    )

    captured: dict[str, Any] = {}

    class _FakeServer:
        def tool(self, *, name: str, description: str | None = None) -> Any:
            def _decorator(fn: Any) -> Any:
                captured[name] = fn
                return fn

            return _decorator

    register_cds_tools(_FakeServer(), orchestrator=AsyncMock())
    import typing

    fn = captured["cds_search_datasets"]
    sig = inspect.signature(fn)
    # ``from __future__ import annotations`` stringifies the annotation;
    # get_type_hints resolves it to the actual class FastMCP needs.
    hints = typing.get_type_hints(fn)
    assert hints["input"] is CdsSearchDatasetsInput
    assert "input" in sig.parameters


@pytest.mark.asyncio
async def test_cds_download_request_result_target_is_optional() -> None:
    """Round-1 cr M1: ``target`` is informational only — the file lives
    at the canonical cache path. The previous Pydantic + Typer
    declaration required a non-empty value, baiting the LLM to invent
    a path it would then ignore. Make it optional; the wrapper fills
    in a sentinel."""
    from copernicus_mcp.backends.cds.tools import (
        CdsDownloadRequestResultInput,
        cds_download_request_result,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"filepath": "/tmp/x.bin"}}
    # No target — should still dispatch.
    await cds_download_request_result(
        CdsDownloadRequestResultInput(request_id="rid"), orchestrator=orch
    )
    kwargs = orch.run.call_args.kwargs
    assert kwargs["operation"] == "fetch"
    assert kwargs["params"]["request_id"] == "rid"
    # Sentinel string, not None — the orchestrator's fetch dispatch
    # rejects empty/None target.
    assert kwargs["params"]["target"]


@pytest.mark.asyncio
async def test_register_cds_tools_skipped_when_no_credentials(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """Round-2 cr H1: ``registry.is_configured("cds")`` returns True
    even when credentials are missing (bootstrap registers the
    backend with a warning rather than refusing). The previous gate
    in ``server.py`` therefore did NOT deliver phantom-tool
    suppression. Fix gates on credential resolution directly. With
    no CDS creds in env, ``register_cds_tools`` must NOT run.
    """

    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.config import ConfigLoader
    from copernicus_mcp.server import build_server

    # Wipe any ambient CDS credentials so resolution returns None.
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    monkeypatch.delenv("CDSAPI_RC", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "no-rc"))

    # Stub mcp.server.fastmcp.FastMCP so build_server doesn't pull a
    # real SDK instance.
    registered: list[str] = []

    class _StubFastMCP:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def tool(self, *, name: str, description: str | None = None) -> Any:
            def _decorator(fn: Any) -> Any:
                registered.append(name)
                return fn

            return _decorator

        def resource(self, _uri: str) -> Any:
            def _decorator(fn: Any) -> Any:
                return fn

            return _decorator

    # ``server.py`` does ``from mcp.server.fastmcp import FastMCP`` at
    # module load — patching ``sys.modules`` after the fact is too late.
    # Patch the bound name on the already-imported module instead.
    import copernicus_mcp.server as _server_mod

    monkeypatch.setattr(_server_mod, "FastMCP", _StubFastMCP)

    config = ConfigLoader().load(
        cli_overrides={
            "enabled_backends": ["cmems", "cds"],
            "storage": {
                "state_database": str(tmp_path / "state.db"),
                "cache_directory": str(tmp_path / "cache"),
            },
        }
    )
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        build_server(config=config, foundation=foundation, registry=registry)
    finally:
        await foundation.persistence.close()

    # Sanity probe: marine_* must register so we know build_server ran.
    assert any(n.startswith("marine_") for n in registered), (
        f"build_server skipped registering everything? registered={registered!r}"
    )
    cds_registered = [n for n in registered if n.startswith("cds_")]
    assert cds_registered == [], (
        f"phantom CDS tools registered without credentials: {cds_registered}"
    )


@pytest.mark.asyncio
async def test_register_cds_tools_skipped_when_cds_not_in_enabled_backends(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """Round-3 cr H1: round-2's gate ``resolve("cds") is not None``
    over-registers when the user has ``CDSAPI_KEY`` in env (common
    for users who also run the standalone ``cdsapi`` CLI) but did
    NOT add ``cds`` to ``enabled_backends``. Registry has no cds
    entry, so any tool call would hit ``BackendError("backend 'cds'
    is not configured")`` — but the LLM client sees 7 phantom tools
    advertised with a recovery_action ("configure_credentials") that
    is wrong for this case. Real fix: combine both signals.
    """
    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.config import ConfigLoader
    from copernicus_mcp.server import build_server

    # Real CDSAPI_KEY-shaped env value but ``cds`` NOT in enabled_backends.
    monkeypatch.setenv("CDSAPI_KEY", "abcdef01-2345-6789-abcd-ef0123456789")
    monkeypatch.setenv(
        "CDSAPI_URL", "https://cds.climate.copernicus.eu/api"
    )

    registered: list[str] = []

    class _StubFastMCP:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def tool(self, *, name: str, description: str | None = None) -> Any:
            def _decorator(fn: Any) -> Any:
                registered.append(name)
                return fn

            return _decorator

        def resource(self, _uri: str) -> Any:
            def _decorator(fn: Any) -> Any:
                return fn

            return _decorator

    import copernicus_mcp.server as _server_mod

    monkeypatch.setattr(_server_mod, "FastMCP", _StubFastMCP)

    config = ConfigLoader().load(
        cli_overrides={
            "enabled_backends": ["cmems"],  # NB: no cds
            "storage": {
                "state_database": str(tmp_path / "state.db"),
                "cache_directory": str(tmp_path / "cache"),
            },
        }
    )
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        build_server(config=config, foundation=foundation, registry=registry)
    finally:
        await foundation.persistence.close()

    assert any(n.startswith("marine_") for n in registered)
    cds_registered = [n for n in registered if n.startswith("cds_")]
    assert cds_registered == [], (
        f"phantom CDS tools registered when cds not in enabled_backends: "
        f"{cds_registered}"
    )
