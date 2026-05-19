from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest


def test_search_input_accepts_minimal_payload() -> None:
    from copernicus_mcp.backends.cmems.tools import MarineSearchDatasetsInput

    obj = MarineSearchDatasetsInput()
    assert obj.keyword is None
    assert obj.bbox is None


def test_search_input_accepts_full_payload() -> None:
    from copernicus_mcp.backends.cmems.tools import MarineSearchDatasetsInput

    obj = MarineSearchDatasetsInput(
        keyword="temperature",
        bbox=(-10.0, 30.0, 10.0, 50.0),
        time_range=("2024-01-01T00:00:00Z", "2024-12-31T00:00:00Z"),
        limit=5,
    )
    assert obj.limit == 5


def test_search_input_rejects_negative_limit() -> None:
    from pydantic import ValidationError

    from copernicus_mcp.backends.cmems.tools import MarineSearchDatasetsInput

    with pytest.raises(ValidationError):
        MarineSearchDatasetsInput(limit=0)


def test_describe_input_requires_dataset_id() -> None:
    from pydantic import ValidationError

    from copernicus_mcp.backends.cmems.tools import MarineDescribeDatasetInput

    with pytest.raises(ValidationError):
        MarineDescribeDatasetInput()  # type: ignore[call-arg]
    obj = MarineDescribeDatasetInput(dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m")
    assert obj.dataset_id


@pytest.mark.parametrize("blank", ["", "   ", "\t", "\n"])
def test_describe_input_rejects_blank_or_whitespace(blank: str) -> None:
    """codex round 2 MEDIUM: orchestrator only checks ``== ''``; Pydantic
    ``min_length=1`` accepts whitespace. Tool input must reject both."""
    from pydantic import ValidationError

    from copernicus_mcp.backends.cmems.tools import MarineDescribeDatasetInput

    with pytest.raises(ValidationError):
        MarineDescribeDatasetInput(dataset_id=blank)


@pytest.mark.parametrize("blank", ["   ", "\t", "\n"])
def test_subset_estimate_inputs_reject_blank_dataset_id(blank: str) -> None:
    """codex round 3 MEDIUM: same blank-rejection must apply to subset/estimate
    inputs (inherit from CmemsSubsetRequest)."""
    from pydantic import ValidationError

    from copernicus_mcp.backends.cmems.tools import (
        MarineEstimateSubsetInput,
        MarineSubsetDatasetInput,
    )

    payload = _valid_subset_payload()
    payload["dataset_id"] = blank
    with pytest.raises(ValidationError):
        MarineEstimateSubsetInput(**payload)
    with pytest.raises(ValidationError):
        MarineSubsetDatasetInput(**payload)


def _valid_subset_payload() -> dict[str, Any]:
    return {
        "dataset_id": "cmems_mod_glo_phy_my",
        "variables": ["thetao"],
        "minimum_longitude": -10.0,
        "maximum_longitude": 10.0,
        "minimum_latitude": 30.0,
        "maximum_latitude": 50.0,
        "minimum_depth": 0.0,
        "maximum_depth": 100.0,
        "start_datetime": "2024-01-01T00:00:00Z",
        "end_datetime": "2024-12-31T00:00:00Z",
    }


def test_estimate_input_validates_payload() -> None:
    from copernicus_mcp.backends.cmems.tools import MarineEstimateSubsetInput

    obj = MarineEstimateSubsetInput(**_valid_subset_payload())
    assert obj.variables == ["thetao"]


def test_subset_input_rejects_inverted_latitude() -> None:
    from copernicus_mcp.backends.cmems.tools import MarineSubsetDatasetInput
    from copernicus_mcp.errors import ValidationError

    payload = _valid_subset_payload()
    payload["minimum_latitude"] = 60.0
    payload["maximum_latitude"] = 30.0
    with pytest.raises(ValidationError):
        MarineSubsetDatasetInput(**payload)


@pytest.mark.asyncio
async def test_search_tool_calls_orchestrator_with_search_op() -> None:
    from copernicus_mcp.backends.cmems.tools import (
        MarineSearchDatasetsInput,
        marine_search_datasets,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"results": []}}
    out = await marine_search_datasets(
        MarineSearchDatasetsInput(keyword="temp"),
        orchestrator=orch,
    )
    # codex round 2 MEDIUM: orchestrator wraps success as {"result": ...};
    # tool unwraps so the user sees the flat payload its docstring promises.
    assert out == {"results": []}
    orch.run.assert_awaited_once()
    kwargs = orch.run.call_args.kwargs
    assert kwargs["backend"] == "cmems"
    assert kwargs["operation"] == "search"
    assert kwargs["params"]["keyword"] == "temp"


@pytest.mark.asyncio
async def test_describe_tool_maps_dataset_id_to_identifier() -> None:
    """codex round 1 HIGH: orchestrator.describe expects ``identifier``,
    NOT ``dataset_id``. The tool input keeps the user-friendly name and
    maps to ``identifier`` at the boundary."""
    from copernicus_mcp.backends.cmems.tools import (
        MarineDescribeDatasetInput,
        marine_describe_dataset,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"dataset_id": "x"}}
    await marine_describe_dataset(
        MarineDescribeDatasetInput(dataset_id="x"),
        orchestrator=orch,
    )
    kwargs = orch.run.call_args.kwargs
    assert kwargs["operation"] == "describe"
    assert kwargs["params"] == {"identifier": "x"}


@pytest.mark.asyncio
async def test_estimate_tool_routes_to_estimate_op() -> None:
    from copernicus_mcp.backends.cmems.tools import (
        MarineEstimateSubsetInput,
        marine_estimate_subset,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"size_bytes": 100}}
    await marine_estimate_subset(
        MarineEstimateSubsetInput(**_valid_subset_payload()),
        orchestrator=orch,
    )
    assert orch.run.call_args.kwargs["operation"] == "estimate"


@pytest.mark.asyncio
async def test_subset_tool_routes_to_submit_op() -> None:
    from copernicus_mcp.backends.cmems.tools import (
        MarineSubsetDatasetInput,
        marine_subset_dataset,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"filepath": "/tmp/x.nc"}}
    await marine_subset_dataset(
        MarineSubsetDatasetInput(**_valid_subset_payload()),
        orchestrator=orch,
    )
    assert orch.run.call_args.kwargs["operation"] == "submit"


@pytest.mark.asyncio
async def test_marine_subset_dataset_forwards_confirmed_flag() -> None:
    """Bug fix: gate emits ``next_action: "call marine_subset_dataset with
    options.confirmed=true"`` but the MCP-tool schema had no way to receive
    ``confirmed``. Mirrors the CDS submit pattern — field on input model,
    plumbed via orchestrator ``options`` not request ``params``."""
    from copernicus_mcp.backends.cmems.tools import (
        MarineSubsetDatasetInput,
        marine_subset_dataset,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"filepath": "/tmp/x.nc"}}
    payload = {**_valid_subset_payload(), "confirmed": True}
    await marine_subset_dataset(
        MarineSubsetDatasetInput(**payload), orchestrator=orch
    )
    kwargs = orch.run.call_args.kwargs
    assert kwargs["operation"] == "submit"
    assert "confirmed" not in kwargs["params"]
    assert kwargs["options"] == {"confirmed": True}


@pytest.mark.asyncio
async def test_marine_subset_dataset_default_no_options() -> None:
    """Without ``confirmed=True`` or ``async_mode=True``, options must stay
    ``None`` — regression guard so the gate keeps firing for unconfirmed
    large requests."""
    from copernicus_mcp.backends.cmems.tools import (
        MarineSubsetDatasetInput,
        marine_subset_dataset,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"filepath": "/tmp/x.nc"}}
    await marine_subset_dataset(
        MarineSubsetDatasetInput(**_valid_subset_payload()),
        orchestrator=orch,
    )
    assert orch.run.call_args.kwargs["options"] is None


@pytest.mark.asyncio
async def test_get_files_tool_routes_to_get_op() -> None:
    """T-CMEMS-GET-006: ``marine_get_files`` dispatches with
    ``operation="get"`` so the orchestrator routes to
    ``backend.get_files``."""
    from copernicus_mcp.backends.cmems.tools import (
        MarineGetFilesInput,
        marine_get_files,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"files": []}}
    await marine_get_files(
        MarineGetFilesInput(dataset_id="ds"),
        orchestrator=orch,
    )
    assert orch.run.call_args.kwargs["operation"] == "get"


@pytest.mark.asyncio
async def test_get_files_forwards_confirmed_flag() -> None:
    """``confirmed`` is pulled out of the params dict and plumbed via
    orchestrator ``options`` so the request envelope stays clean (the
    backend's ``CmemsGetRequest`` has ``extra='forbid'`` and would
    reject a stray ``confirmed`` field)."""
    from copernicus_mcp.backends.cmems.tools import (
        MarineGetFilesInput,
        marine_get_files,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"files": []}}
    await marine_get_files(
        MarineGetFilesInput(dataset_id="ds", confirmed=True),
        orchestrator=orch,
    )
    kwargs = orch.run.call_args.kwargs
    assert kwargs["operation"] == "get"
    assert "confirmed" not in kwargs["params"]
    assert kwargs["options"] == {"confirmed": True}


@pytest.mark.asyncio
async def test_get_files_default_no_options() -> None:
    """Without ``confirmed=True``, options stays ``None`` — gate must
    keep firing for unconfirmed requests."""
    from copernicus_mcp.backends.cmems.tools import (
        MarineGetFilesInput,
        marine_get_files,
    )

    orch = AsyncMock()
    orch.run.return_value = {"result": {"files": []}}
    await marine_get_files(
        MarineGetFilesInput(dataset_id="ds"),
        orchestrator=orch,
    )
    assert orch.run.call_args.kwargs["options"] is None


def test_marine_check_status_docstring_discourages_sleep_loops() -> None:
    """Same polling-pacing guidance as cds_check_request_status. The
    Claude Code harness blocks standalone sleep — we need to nudge the
    agent toward the correct pattern (user-paced re-call, or CLI
    ``marine wait``) instead of letting it invent until-loop
    workarounds."""
    from copernicus_mcp.backends.cmems.tools import marine_check_status

    doc = marine_check_status.__doc__ or ""
    lo = doc.lower()
    assert "sleep" in lo, doc
    assert "marine wait" in lo, doc


def test_register_marine_tools_registers_must_have_tools() -> None:
    from copernicus_mcp.backends.cmems.tools import register_marine_tools

    registered: dict[str, Any] = {}

    class _FakeServer:
        def tool(self, *, name: str, description: str | None = None) -> Any:
            def _decorator(fn: Any) -> Any:
                registered[name] = (fn, description)
                return fn

            return _decorator

    register_marine_tools(_FakeServer(), orchestrator=AsyncMock())
    must_have = {
        "marine_search_datasets",
        # T-CMEMS-HIER-005: two new hierarchical-search tools.
        "marine_search_groups",
        "marine_search_products",
        "marine_describe_dataset",
        "marine_estimate_subset",
        "marine_subset_dataset",
        # T-CMEMS-GET-006: native-file retrieval.
        "marine_get_files",
        # T-CMEMS-GET-INDEX-004: Layer 2 index-driven listing.
        "marine_list_files",
        # T-022 second half: coordinate axes.
        "marine_get_coordinates",
    }
    assert must_have <= set(registered.keys())
    # Each registered tool's description is the underlying tool's docstring
    # (not None, not the wrapper's empty doc) — FastMCP uses this.
    for name in must_have:
        _, description = registered[name]
        assert description, f"{name} has empty description"


def test_register_marine_tools_registered_callable_has_typed_input() -> None:
    """codex round 1 HIGH: FastMCP derives JSON schema from the registered
    callable's annotations. ``input: Any`` produces an empty schema, breaking
    LLM clients. The registered wrapper must declare a concrete Pydantic
    model annotation."""
    from copernicus_mcp.backends.cmems.tools import (
        MarineSearchDatasetsInput,
        register_marine_tools,
    )

    captured: dict[str, Any] = {}

    class _FakeServer:
        def tool(self, *, name: str, description: str | None = None) -> Any:
            def _decorator(fn: Any) -> Any:
                captured[name] = fn
                return fn

            return _decorator

    register_marine_tools(_FakeServer(), orchestrator=AsyncMock())
    search_fn = captured["marine_search_datasets"]
    # ``from __future__ import annotations`` stores annotations as strings;
    # get_type_hints resolves to the actual class.
    from typing import get_type_hints

    hints = get_type_hints(search_fn)
    assert hints["input"] is MarineSearchDatasetsInput


@pytest.mark.asyncio
async def test_register_against_real_fastmcp_derives_schema_and_dispatches() -> None:
    """codex round 1 HIGH + round 2 LOW: end-to-end smoke for all 4 tools.

    Verifies for every must-have tool:
      (a) FastMCP derives a non-trivial input schema (not the empty ``{}``
          that ``input: Any`` would produce).
      (b) ``call_tool`` round-trips through the orchestrator without
          crashing on ``model_dump`` of a raw dict (would catch the
          original HIGH-1 regression).
      (c) The wrapper unwraps the orchestrator ``{"result": ...}`` envelope
          to a flat payload (codex round-2 MEDIUM).
    """
    from mcp.server.fastmcp import FastMCP

    from copernicus_mcp.backends.cmems.tools import register_marine_tools

    orch = AsyncMock()
    orch.run.return_value = {"result": {"echoed": "ok"}}
    server = FastMCP("t-030-smoke")
    register_marine_tools(server, orchestrator=orch)

    tools = await server.list_tools()
    by_name = {t.name: t for t in tools}

    must_have_calls = {
        "marine_search_datasets": {"keyword": "temp"},
        "marine_describe_dataset": {"dataset_id": "ds"},
        "marine_estimate_subset": _valid_subset_payload(),
        "marine_subset_dataset": _valid_subset_payload(),
    }

    for tool_name, tool_input in must_have_calls.items():
        schema = by_name[tool_name].inputSchema
        # FastMCP wraps single Pydantic-model param as {"input": <ref>}.
        assert "input" in schema.get("properties", {}), (tool_name, schema)
        input_schema = schema["properties"]["input"]
        assert input_schema.get("properties") or input_schema.get(
            "$ref"
        ), (tool_name, input_schema)

        result = await server.call_tool(tool_name, {"input": tool_input})
        structured = result[1] if isinstance(result, tuple) else result
        # Unwrap: should be the inner payload, NOT {"result": {...}}.
        assert structured == {"echoed": "ok"}, (tool_name, structured)


def test_unwrap_raises_on_error_envelope_preserves_record_and_json() -> None:
    """codex T-031 round 1 HIGH + round 2: error envelopes surface as MCP
    protocol-level errors. ``_unwrap`` raises ``ToolReturnedError``
    carrying:
      - ``.record`` attribute = the canonical ErrorRecord dict (for
        in-process callers like tests / CLI).
      - ``str(exc)`` = JSON-encoded record so clients that parse the
        FastMCP-prefixed message can recover error_class / recovery_action.
    """
    import json as _json

    from copernicus_mcp.backends.cmems.tools import ToolReturnedError, _unwrap

    err = {
        "error": {
            "error_class": "ValidationError",
            "message": "x",
            "recovery_action": "modify_request_parameters",
        }
    }
    with pytest.raises(ToolReturnedError) as exc_info:
        _unwrap(err)
    assert exc_info.value.record == err["error"]
    # str(exc) must round-trip through json.loads — clients depend on this.
    parsed = _json.loads(str(exc_info.value))
    assert parsed == err["error"]


def test_unwrap_preserves_confirmation_envelope() -> None:
    """ConfirmationRequired payloads pass through unchanged — the LLM
    client must see the structured prompt, not an MCP error."""
    from copernicus_mcp.backends.cmems.tools import _unwrap

    confirm = {"confirmation_required": True, "advisory_message": "y"}
    assert _unwrap(confirm) == confirm


def test_subset_output_advertises_large_data_descriptor() -> None:
    """Large-data invariant #1: subset tool docstring must promise descriptor only."""
    from copernicus_mcp.backends.cmems.tools import marine_subset_dataset

    doc = marine_subset_dataset.__doc__ or ""
    assert "filepath" in doc.lower()
    assert "metadata" in doc.lower()
    assert "provenance" in doc.lower()
