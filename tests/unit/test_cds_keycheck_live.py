"""A stale constraints snapshot never blocks a valid CDS request (T-CDS-KEYCHECK-002).

Field defect 2026-11-04: E-OBS v33 needs ``licence_type``; the bundled
snapshot (2026-05-16) did not list it, so the submit key check refused every
request CDS would accept, and the escape hatch the error named
(``__options.skip_input_validation``) was reachable only from Python.

Covered here:
- ``fetch_live_form_keys``: the anonymous ``/constraints`` probe the key check
  consults before refusing; advisory, ``None`` on any failure, strict shape.
- the submit key check: a key the live form lists passes; a live-form failure
  keeps the refusal; the hint names only routes a caller can take.
- the MCP tool field and the CLI flag that expose ``skip_input_validation``.

No network: the live probe gets a ``httpx.MockTransport``; the submit tests
patch the backend's ``fetch_live_form_keys`` symbol (the autouse fixture in
``tests/conftest.py`` defaults it to "unavailable").
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from typer.testing import CliRunner

from tests.unit.test_cds_keycheck import (
    _backend,
    _eobs_with_area,
    _era5_with_area,
    _make_foundation,
    _patch_cdsapi,
    _patch_costing_flat,
    _with_budget,
)

_SECRET = "deadbeef-cafe-1234-5678-aabbccddeeff"


@pytest_asyncio.fixture
async def foundation(tmp_path: Path):
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        yield found
    finally:
        await persistence.close()


# ---------------------------------------------------------------------------
# fetch_live_form_keys
# ---------------------------------------------------------------------------


class _FakeCatalogue:
    def __init__(self, store: str | None) -> None:
        self._store = store

    def store_for(self, dataset_id: str) -> str | None:  # noqa: ARG002
        return self._store


class _FakeFactory:
    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self._handler = handler
        self.requests: list[httpx.Request] = []

    def create(self, backend_id: str) -> httpx.AsyncClient:  # noqa: ARG002
        def _record(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return self._handler(request)

        return httpx.AsyncClient(transport=httpx.MockTransport(_record))


def _json(payload: Any, status: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(status, json=payload)

    return handler


_EOBS_LIVE = {
    "licence_type": ["open_data_licence", "restricted_research_licence"],
    "product_type": ["elevation", "ensemble_mean", "ensemble_spread"],
    "variable": ["mean_temperature"],
    "grid_resolution": ["0_1deg", "0_25deg"],
    "version": ["32_0e", "33_0e"],
}


def _fetch(
    handler: Callable[[httpx.Request], httpx.Response],
    store: str | None = "cds",
    dataset_id: str = "insitu-gridded-observations-europe",
) -> tuple[Any, _FakeFactory]:
    from copernicus_mcp.backends.cds.live_form import fetch_live_form_keys

    factory = _FakeFactory(handler)
    result = asyncio.run(
        fetch_live_form_keys(
            dataset_id,
            http_client_factory=factory,
            catalogue=_FakeCatalogue(store),
        )
    )
    return result, factory


def test_live_form_keys_parsed_from_constraints_payload() -> None:
    keys, factory = _fetch(_json(_EOBS_LIVE))
    assert keys == set(_EOBS_LIVE)
    (request,) = factory.requests
    assert request.method == "POST"
    assert str(request.url) == (
        "https://cds.climate.copernicus.eu/api/retrieve/v1/processes/"
        "insitu-gridded-observations-europe/constraints"
    )
    assert json.loads(request.content) == {"inputs": {}}


def test_live_form_routes_to_the_dataset_store() -> None:
    _, factory = _fetch(
        _json({"variable": ["x"]}), store="ads", dataset_id="cams-global-reanalysis-eac4"
    )
    assert str(factory.requests[0].url).startswith("https://ads.atmosphere.copernicus.eu/api/")


def test_live_form_allows_empty_value_lists() -> None:
    """Continuous fields (``area``) come back with an empty value list."""
    keys, _ = _fetch(_json({"area": [], "variable": ["x"]}))
    assert keys == {"area", "variable"}


def test_live_form_uses_a_short_timeout() -> None:
    """The probe sets its own timeout; the fake client's default (5 s) differs
    from it, so a dropped ``timeout=`` would fail this equality."""
    from copernicus_mcp.backends.cds.live_form import LIVE_FORM_TIMEOUT_SECONDS

    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.extensions.get("timeout") or {})
        return httpx.Response(200, json={"variable": ["x"]})

    _fetch(handler)
    assert LIVE_FORM_TIMEOUT_SECONDS == 15.0
    assert seen["read"] == LIVE_FORM_TIMEOUT_SECONDS
    assert seen["connect"] == LIVE_FORM_TIMEOUT_SECONDS


@pytest.mark.parametrize(
    "handler",
    [
        _json({"message": "temporarily unavailable"}),  # dict, wrong shape
        _json({"variable": "x"}),  # value not a list
        _json({}),  # empty form
        _json(["variable", "year"]),  # not a dict
        _json({"variable": ["x"]}, status=500),
        _json({"detail": "not found"}, status=404),
        lambda request: httpx.Response(200, content=b"<html>oops</html>"),
    ],
    ids=[
        "malformed-dict",
        "non-list-value",
        "empty",
        "list-body",
        "http-500",
        "http-404",
        "non-json",
    ],
)
def test_live_form_failure_or_bad_shape_returns_none(handler: Any) -> None:
    keys, _ = _fetch(handler)
    assert keys is None


@pytest.mark.parametrize("exc", [httpx.ConnectError("boom"), httpx.ReadTimeout("slow")])
def test_live_form_transport_errors_return_none(exc: Exception) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        raise exc

    keys, _ = _fetch(handler)
    assert keys is None


def test_live_form_cancellation_propagates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        _fetch(handler)


# ---------------------------------------------------------------------------
# submit key check with the live-form fallback
# ---------------------------------------------------------------------------


def _patch_live(monkeypatch: Any, keys: set[str] | None) -> list[str]:
    calls: list[str] = []

    async def _fake(dataset_id: str, **_kwargs: Any) -> set[str] | None:
        calls.append(dataset_id)
        return None if keys is None else set(keys)

    monkeypatch.setattr("copernicus_mcp.backends.cds.backend.fetch_live_form_keys", _fake)
    return calls


def _era5_live_keys(*extra: str) -> set[str]:
    from copernicus_mcp.backends.cds import catalogue

    return set(catalogue.load_constraints()["reanalysis-era5-single-levels"]) | set(extra)


@pytest.mark.asyncio
async def test_key_listed_by_live_form_passes_and_logs_stale_snapshot(
    foundation, monkeypatch, caplog
) -> None:
    _patch_costing_flat(monkeypatch)
    sdk = _patch_cdsapi(monkeypatch)
    calls = _patch_live(monkeypatch, _era5_live_keys("brand_new_key"))

    with caplog.at_level(logging.WARNING):
        out = await _backend(foundation).submit(_era5_with_area(brand_new_key="x"))

    assert out["status"] == "queued"
    sdk.retrieve.assert_called_once()
    assert calls == ["reanalysis-era5-single-levels"]
    assert any(
        "brand_new_key" in r.getMessage() and "stale" in r.getMessage() for r in caplog.records
    )


@pytest.mark.asyncio
async def test_key_missing_from_live_form_is_refused_with_live_keys(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.errors import ValidationError

    _patch_costing_flat(monkeypatch)
    sdk = _patch_cdsapi(monkeypatch)
    _patch_live(monkeypatch, _era5_live_keys("only_live_key"))

    with pytest.raises(ValidationError) as exc:
        await _backend(foundation).submit(_era5_with_area(brand_new_key="x"))

    record = exc.value.error_record
    hint = record.next_action_hint or ""
    assert "'brand_new_key'" in record.message
    assert "only_live_key" in hint  # the LIVE key set is listed
    assert "live" in hint.lower()
    assert record.context.get("live_form_checked") is True
    sdk.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_mixed_keys_only_the_live_unknown_one_is_named(foundation, monkeypatch) -> None:
    from copernicus_mcp.errors import ValidationError

    _patch_costing_flat(monkeypatch)
    _patch_cdsapi(monkeypatch)
    _patch_live(monkeypatch, _era5_live_keys("new_key_a"))

    with pytest.raises(ValidationError) as exc:
        await _backend(foundation).submit(_era5_with_area(new_key_a="x", new_key_b="y"))

    message = exc.value.error_record.message
    assert "'new_key_b'" in message
    assert "new_key_a" not in message


@pytest.mark.asyncio
async def test_live_form_unavailable_keeps_the_refusal(foundation, monkeypatch) -> None:
    from copernicus_mcp.errors import ValidationError

    _patch_costing_flat(monkeypatch)
    sdk = _patch_cdsapi(monkeypatch)
    _patch_live(monkeypatch, None)

    with pytest.raises(ValidationError) as exc:
        await _backend(foundation).submit(_eobs_with_area())

    record = exc.value.error_record
    hint = record.next_action_hint or ""
    assert "'area'" in record.message
    assert "could not be checked" in hint
    assert "grid_resolution" in hint  # snapshot keys still listed
    assert record.context.get("live_form_checked") is False
    sdk.retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_refusal_hint_names_only_reachable_routes(foundation, monkeypatch) -> None:
    from copernicus_mcp.errors import ValidationError

    _patch_costing_flat(monkeypatch)
    _patch_cdsapi(monkeypatch)
    for live in (None, _era5_live_keys()):
        _patch_live(monkeypatch, live)
        with pytest.raises(ValidationError) as exc:
            await _backend(foundation).submit(_era5_with_area(brand_new_key="x"))
        hint = exc.value.error_record.next_action_hint or ""
        assert "__options" not in hint
        assert "skip_input_validation=true" in hint
        assert "--skip-input-validation" in hint
        assert "cds_apply_constraints" in hint


@pytest.mark.asyncio
async def test_known_keys_make_no_live_call(foundation, monkeypatch) -> None:
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi(monkeypatch)
    calls = _patch_live(monkeypatch, set())

    out = await _backend(foundation).submit(_era5_with_area())

    assert out["status"] == "queued"
    assert calls == []


@pytest.mark.asyncio
async def test_skip_option_and_kill_switch_make_no_live_call(foundation, monkeypatch) -> None:
    _patch_costing_flat(monkeypatch)
    _patch_cdsapi(monkeypatch)
    calls = _patch_live(monkeypatch, None)

    out = await _backend(foundation).submit(_eobs_with_area(skip_input_validation=True))
    assert out["status"] == "queued"
    out = await _backend(_with_budget(foundation, cds_input_key_validation=False)).submit(
        _eobs_with_area(force_refresh=True)
    )
    assert out["status"] == "queued"
    assert calls == []


@pytest.mark.asyncio
async def test_live_keys_are_sanitised_in_hint_and_log(foundation, monkeypatch, caplog) -> None:
    """Live key names are remote input: a credential-shaped one must not echo
    into the error record or the stderr log verbatim."""
    from copernicus_mcp.errors import ValidationError

    _patch_costing_flat(monkeypatch)
    _patch_cdsapi(monkeypatch)

    # Refusal: the secret is in the LIVE key list shown in the hint.
    _patch_live(monkeypatch, _era5_live_keys(_SECRET))
    with pytest.raises(ValidationError) as exc:
        await _backend(foundation).submit(_era5_with_area(brand_new_key="x"))
    assert _SECRET not in (exc.value.error_record.next_action_hint or "")

    # Acceptance: the secret is a request key the live form lists → logged.
    # (Our own loggers only: aiosqlite's DEBUG echo of the persisted request
    # is the request itself, unchanged by this feature.)
    with caplog.at_level(logging.DEBUG, logger="copernicus_mcp"):
        params = _era5_with_area()
        params["inputs"][_SECRET] = "x"
        await _backend(foundation).submit(params)
    ours = [r for r in caplog.records if r.name.startswith("copernicus_mcp")]
    assert any("stale" in r.getMessage() for r in ours)
    assert all(_SECRET not in r.getMessage() for r in ours)
    assert all(_SECRET not in json.dumps(r.__dict__, default=str) for r in ours)


@pytest.mark.asyncio
async def test_eobs_v33_with_licence_type_passes_on_the_bundled_snapshot(
    foundation, monkeypatch
) -> None:
    """The field request itself, against the refreshed snapshot: no live call."""
    _patch_costing_flat(monkeypatch)
    sdk = _patch_cdsapi(monkeypatch)
    calls = _patch_live(monkeypatch, None)

    params = _eobs_with_area()
    del params["inputs"]["area"]
    out = await _backend(foundation).submit(params)

    assert out["status"] == "queued"
    sdk.retrieve.assert_called_once()
    assert calls == []


def test_eobs_period_is_unknown_on_the_refreshed_snapshot() -> None:
    """The 2026-11-04 E-OBS form dropped ``period``: CDS would ignore it."""
    from copernicus_mcp.backends.cds.backend import _unknown_input_keys

    inputs = {**_eobs_with_area()["inputs"], "period": "full_period"}
    del inputs["area"]
    assert _unknown_input_keys("insitu-gridded-observations-europe", inputs) == ["period"]


@pytest.mark.asyncio
async def test_live_key_list_keeps_the_synthetic_timeseries_location(
    foundation, monkeypatch
) -> None:
    """The live form of the ``*-timeseries`` products omits ``location`` like
    the snapshot does; the accepted keys shown in the hint must still name it."""
    from copernicus_mcp.errors import ValidationError

    _patch_live(monkeypatch, {"variable", "date", "data_format"})

    with pytest.raises(ValidationError) as exc:
        await _backend(foundation)._check_input_keys(
            "reanalysis-era5-land-timeseries",
            {"variable": ["2m_temperature"], "brand_new_key": "x"},
        )

    assert "'location'" in (exc.value.error_record.next_action_hint or "")


@pytest.mark.asyncio
async def test_backend_wires_the_real_probe(foundation, monkeypatch) -> None:
    """Contract: the backend calls the REAL probe with the arguments it takes.
    The other submit tests replace the probe with a ``**kwargs`` fake, which
    would hide a renamed argument; that ``TypeError`` would be swallowed as
    "live form unavailable" and bring the 0.7.1 refusal back silently."""
    import dataclasses

    from copernicus_mcp.backends.cds import live_form

    _patch_costing_flat(monkeypatch)
    sdk = _patch_cdsapi(monkeypatch)
    monkeypatch.setattr(
        "copernicus_mcp.backends.cds.backend.fetch_live_form_keys",
        live_form.fetch_live_form_keys,
    )
    live = {k: [] for k in _era5_live_keys("brand_new_key")}
    factory = _FakeFactory(_json(live))
    wired = dataclasses.replace(foundation, http_client_factory=factory)

    out = await _backend(wired).submit(_era5_with_area(brand_new_key="x"))

    assert out["status"] == "queued"
    sdk.retrieve.assert_called_once()
    (request,) = factory.requests
    assert str(request.url).endswith("/processes/reanalysis-era5-single-levels/constraints")


@pytest.mark.asyncio
async def test_skip_via_tool_reaches_every_chunk_through_the_real_orchestrator(
    tmp_path: Path, monkeypatch
) -> None:
    """End to end (codex R1 LOW): tool field → real orchestrator → backend →
    auto-chunked parent. The snapshot-unknown key must reach the SDK in every
    submitted part, and the same request without the option is refused."""
    from copernicus_mcp.backends.cds.costing import CostingResult
    from copernicus_mcp.backends.cds.tools import CdsSubmitRequestInput, cds_submit_request
    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.config import ConfigLoader
    from copernicus_mcp.workflow.orchestrator import WorkflowOrchestrator

    monkeypatch.setenv("CDSAPI_KEY", "abcdef01-2345-6789-abcd-ef0123456789")
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    monkeypatch.delenv("CDSAPI_RC", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "no-rc"))

    async def _costing(dataset_id, inputs, **_kwargs):  # noqa: ARG001
        return CostingResult(units=300.0 * len(inputs.get("month", [])), limit=400.0)

    monkeypatch.setattr("copernicus_mcp.backends.cds.backend.fetch_costing", _costing)
    sdk = _patch_cdsapi(monkeypatch)

    config = ConfigLoader().load(
        cli_overrides={
            "enabled_backends": ["cds"],
            "storage": {
                "state_database": str(tmp_path / "state.db"),
                "cache_directory": str(tmp_path / "cache"),
            },
        }
    )
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        orch = WorkflowOrchestrator(registry=registry, foundation=foundation)
        request = _era5_with_area(brand_new_key="x")
        request["inputs"]["month"] = ["01", "02"]
        del request["__options"]

        with pytest.raises(Exception, match="brand_new_key"):
            await cds_submit_request(
                CdsSubmitRequestInput(**request, chunk_by="month"), orchestrator=orch
            )
        sdk.retrieve.assert_not_called()

        out = await cds_submit_request(
            CdsSubmitRequestInput(**request, chunk_by="month", skip_input_validation=True),
            orchestrator=orch,
        )
    finally:
        await foundation.persistence.close()

    assert out["status"] == "queued"
    assert out.get("chunked") is True
    assert sdk.retrieve.call_count >= 1
    for call in sdk.retrieve.call_args_list:
        sent = call.args[1] if len(call.args) > 1 else call.kwargs["request"]
        assert sent.get("brand_new_key") == "x"
        assert len(sent["month"]) == 1


# ---------------------------------------------------------------------------
# Other submit hints name reachable routes too (same defect class)
# ---------------------------------------------------------------------------


def test_chunk_proposal_names_the_tool_field_and_cli_flag() -> None:
    from copernicus_mcp.backends.cds.backend import _build_chunk_proposal

    confirmation = _build_chunk_proposal(
        estimate={"queue_latency_tier": "medium", "estimated_size_bytes": 1},
        inputs={"year": ["2020", "2021"], "month": ["01", "02"], "day": ["01"]},
        cost_units=800.0,
        cost_limit=400.0,
    )
    next_action = confirmation.payload["next_action"]
    assert "__options" not in next_action
    assert "chunk_by=" in next_action
    assert "--chunk-by" in next_action


def test_large_fanout_reconfirm_names_the_tool_field() -> None:
    from copernicus_mcp.backends.cds.backend import _build_chunk_count_confirmation

    confirmation = _build_chunk_count_confirmation(
        chunk_count=150,
        threshold=100,
        reason="auto_chunk_job_count_large",
        require_large_ack=True,
    )
    next_action = confirmation.payload["next_action"]
    assert "__options" not in next_action
    assert "confirm_large_fanout=true" in next_action


@pytest.mark.asyncio
async def test_invalid_chunk_by_message_has_no_dunder_options(foundation, monkeypatch) -> None:
    """A direct caller can pass any chunk_by; the refusal must not name a
    route the MCP tool refuses."""
    from copernicus_mcp.errors import ValidationError

    _patch_costing_flat(monkeypatch, units=800.0, limit=400.0)
    _patch_cdsapi(monkeypatch)
    params = _era5_with_area()
    params["inputs"]["month"] = ["01", "02"]
    params["__options"] = {"confirmed": True, "chunk_by": "week"}

    with pytest.raises(ValidationError) as exc:
        await _backend(foundation).submit(params)

    message = exc.value.error_record.message
    assert "__options" not in message
    assert "chunk_by" in message


# ---------------------------------------------------------------------------
# MCP tool field
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_forwards_skip_input_validation() -> None:
    from copernicus_mcp.backends.cds.tools import CdsSubmitRequestInput, cds_submit_request

    orch = AsyncMock()
    orch.run.return_value = {"result": {"status": "queued", "request_id": "r"}}
    payload = {**_era5_with_area(), "skip_input_validation": True}
    del payload["__options"]

    await cds_submit_request(CdsSubmitRequestInput(**payload), orchestrator=orch)

    kwargs = orch.run.call_args.kwargs
    assert "skip_input_validation" not in kwargs["params"]
    assert kwargs["options"] == {"skip_input_validation": True}


@pytest.mark.asyncio
async def test_tool_default_does_not_skip() -> None:
    from copernicus_mcp.backends.cds.tools import CdsSubmitRequestInput, cds_submit_request

    orch = AsyncMock()
    orch.run.return_value = {"result": {"status": "queued", "request_id": "r"}}
    payload = _era5_with_area()
    del payload["__options"]

    await cds_submit_request(CdsSubmitRequestInput(**payload), orchestrator=orch)

    assert orch.run.call_args.kwargs["options"] is None


def test_tool_docstring_names_the_field_not_dunder_options() -> None:
    from copernicus_mcp.backends.cds.tools import cds_submit_request

    doc = cds_submit_request.__doc__ or ""
    assert "skip_input_validation" in doc
    assert "__options" not in doc


# ---------------------------------------------------------------------------
# CLI flag
# ---------------------------------------------------------------------------


def _fake_builder(orch: AsyncMock) -> Any:
    @contextlib.asynccontextmanager
    async def _builder() -> Any:
        yield orch

    return _builder


@pytest.fixture
def cli_env(monkeypatch: Any, tmp_path: Path) -> Path:
    monkeypatch.setenv("COPERNICUS_MCP_STATE_DB", str(tmp_path / "state.db"))
    monkeypatch.setenv("COPERNICUS_MCP_CACHE_DIR", str(tmp_path / "cache"))
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(_eobs_with_area()["inputs"]))
    return inputs


def test_cli_skip_flag_reaches_options_also_on_reconfirm(cli_env: Path, monkeypatch: Any) -> None:
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.side_effect = [
        {"confirmation_required": True, "reason": "size", "message": "big"},
        {"result": {"status": "queued", "request_id": "r"}},
    ]
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = CliRunner().invoke(
        cli.app,
        [
            "cds",
            "submit",
            "--dataset-id",
            "insitu-gridded-observations-europe",
            "--inputs-file",
            str(cli_env),
            "--skip-input-validation",
            "--yes",
            "--json",
        ],
    )

    assert res.exit_code == 0, res.stdout + res.stderr
    first, second = (c.kwargs for c in fake.run.call_args_list)
    assert first["options"] == {"skip_input_validation": True}
    assert second["options"]["skip_input_validation"] is True
    assert second["options"]["confirmed"] is True
    assert "skip_input_validation" not in first["params"]["inputs"]


def test_cli_default_does_not_skip(cli_env: Path, monkeypatch: Any) -> None:
    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.return_value = {"result": {"status": "queued", "request_id": "r"}}
    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _fake_builder(fake))

    res = CliRunner().invoke(
        cli.app,
        [
            "cds",
            "submit",
            "--dataset-id",
            "insitu-gridded-observations-europe",
            "--inputs-file",
            str(cli_env),
            "--json",
        ],
    )

    assert res.exit_code == 0, res.stdout + res.stderr
    assert fake.run.call_args.kwargs["options"] is None


def test_cli_help_documents_the_flag() -> None:
    from copernicus_mcp import cli

    res = CliRunner().invoke(cli.app, ["cds", "submit", "--help"])
    assert "--skip-input-validation" in res.stdout
