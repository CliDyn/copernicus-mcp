"""In-band licence tools (T-CDS-LICENCE-001).

The standing-authorisation decision (2026-06-19): every new dataset licence used to
require a human web gate — the data agent surfaced the ``recovery_url``, a
human accepted in the browser, the agent re-submitted. The operator wants the
agent to accept under the operator's standing authority instead. Acceptance
legally binds the ACCOUNT owner, so the MCP tool surface is opt-in
(``budget.cds_licence_accept_enabled``, default false); the CLI commands work
regardless — the CLI *is* the operator. The datastores client ships the API:
``get_licences(scope)`` / ``get_accepted_licences(scope)`` / list[dict], and
``accept_licence(licence_id, revision)`` (PUT, pinned against 0.5.1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio


def _make_foundation(tmp_path: Path):
    from copernicus_mcp.auth import CredentialResolver
    from copernicus_mcp.backends.abstract import FoundationServices
    from copernicus_mcp.cache import CacheManager
    from copernicus_mcp.config import ConfigLoader
    from copernicus_mcp.data_model.coordinator import DataModelCoordinator
    from copernicus_mcp.data_model.provenance import ProvenanceRecorder
    from copernicus_mcp.errors.sanitiser import Sanitiser
    from copernicus_mcp.http import HttpClientFactory
    from copernicus_mcp.persistence import SqliteBackend

    config = ConfigLoader().load()
    persistence = SqliteBackend(tmp_path / "state.db")
    cache = CacheManager(
        cache_directory=tmp_path / "cache",
        persistence=persistence,
        size_limit_bytes=10 * 1024 * 1024,
    )
    return (
        FoundationServices(
            config=config,
            credential_resolver=CredentialResolver(),
            http_client_factory=HttpClientFactory(http_config=config.http),
            persistence=persistence,
            cache=cache,
            sanitiser=Sanitiser(),
            data_model=DataModelCoordinator(persistence=persistence),
            provenance=ProvenanceRecorder(
                persistence=persistence,
                software_versions={"copernicus-mcp": "0.0.1"},
            ),
        ),
        persistence,
    )


@pytest_asyncio.fixture
async def foundation(tmp_path: Path):
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        yield found
    finally:
        await persistence.close()


def _fake_creds():
    from copernicus_mcp.auth.resolver import ResolvedCredentials

    return ResolvedCredentials(
        backend="cds",
        source="explicit",
        source_detail="test",
        fields={"key": "abcdef01-2345-6789-abcd-ef0123456789"},
    )


def _backend(foundation):
    from copernicus_mcp.backends.cds.backend import CdsBackend

    return CdsBackend(foundation=foundation, credentials=_fake_creds())


_LICENCES = [
    {"id": "licence-to-use-copernicus-products", "revision": 12,
     "label": "Licence to use Copernicus Products",
     "contents_url": "https://cds.climate.copernicus.eu/api/.../licence.md"},
    {"id": "licence-to-use-E-OBS-products", "revision": 1,
     "label": "E-OBS product licence",
     "contents_url": "https://cds.climate.copernicus.eu/api/.../eobs.md"},
]


def _patch_cdsapi_licences(monkeypatch):
    import sys
    import types

    fake_module = types.ModuleType("cdsapi")
    instance = MagicMock()
    inner = MagicMock()
    inner.get_licences = MagicMock(return_value=list(_LICENCES))
    inner.get_accepted_licences = MagicMock(return_value=[_LICENCES[0]])
    inner.accept_licence = MagicMock(
        return_value={"id": "licence-to-use-E-OBS-products", "revision": 1}
    )
    instance.client = inner
    urls: list[Any] = []

    def _cls(**kwargs):
        urls.append(kwargs.get("url"))
        return instance

    fake_module.Client = _cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cdsapi", fake_module)
    return inner, urls


# ---------------------------------------------------------------------------
# backend methods
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_licences_returns_available_and_accepted(
    foundation, monkeypatch
) -> None:
    inner, _ = _patch_cdsapi_licences(monkeypatch)
    backend = _backend(foundation)

    out = await backend.list_licences({})

    assert [entry["id"] for entry in out["available"]] == [
        "licence-to-use-copernicus-products",
        "licence-to-use-E-OBS-products",
    ]
    assert [entry["id"] for entry in out["accepted"]] == [
        "licence-to-use-copernicus-products"
    ]
    inner.get_licences.assert_called_once()
    inner.get_accepted_licences.assert_called_once()


@pytest.mark.asyncio
async def test_licence_calls_route_to_the_dataset_store(
    foundation, monkeypatch
) -> None:
    """Licence ids are per-store: an ADS dataset must talk to the ADS
    endpoint, not the CDS default."""
    _, urls = _patch_cdsapi_licences(monkeypatch)
    backend = _backend(foundation)

    await backend.list_licences({"dataset_id": "cams-global-reanalysis-eac4"})

    assert urls and "ads" in (urls[0] or "")


@pytest.mark.asyncio
async def test_accept_licence_puts_the_revision_and_reports(
    foundation, monkeypatch
) -> None:
    inner, _ = _patch_cdsapi_licences(monkeypatch)
    backend = _backend(foundation)

    out = await backend.accept_licence(
        {"licence_id": "licence-to-use-E-OBS-products", "revision": 1}
    )

    assert out["accepted"] is True
    assert out["licence_id"] == "licence-to-use-E-OBS-products"
    assert out["revision"] == 1
    inner.accept_licence.assert_called_once_with(
        "licence-to-use-E-OBS-products", 1
    )


@pytest.mark.asyncio
async def test_licence_methods_require_credentials(foundation, monkeypatch) -> None:
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import AuthError

    _patch_cdsapi_licences(monkeypatch)
    backend = CdsBackend(foundation=foundation, credentials=None)

    with pytest.raises(AuthError):
        await backend.list_licences({})
    with pytest.raises(AuthError):
        await backend.accept_licence({"licence_id": "x", "revision": 1})


@pytest.mark.asyncio
async def test_licence_sdk_errors_are_wrapped_canonically(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.errors import CopernicusMcpError

    inner, _ = _patch_cdsapi_licences(monkeypatch)
    inner.get_licences.side_effect = RuntimeError("boom")
    backend = _backend(foundation)

    with pytest.raises(CopernicusMcpError):
        await backend.list_licences({})


# ---------------------------------------------------------------------------
# registration gating + T&C hint
# ---------------------------------------------------------------------------


def _registered_names(**register_kwargs) -> set[str]:
    from copernicus_mcp.backends.cds.tools import register_cds_tools

    registered: dict[str, Any] = {}

    class _FakeServer:
        def tool(self, *, name: str, description: str | None = None) -> Any:
            def _decorator(fn: Any) -> Any:
                registered[name] = (fn, description)
                return fn

            return _decorator

    register_cds_tools(_FakeServer(), orchestrator=AsyncMock(), **register_kwargs)
    return set(registered)


def test_list_tool_always_registered_accept_tool_gated() -> None:
    """Acceptance legally binds the account owner: the MCP-facing accept tool
    exists only when the operator opted in. Listing is harmless and always
    available."""
    default = _registered_names()
    assert "cds_list_licences" in default
    assert "cds_accept_licence" not in default

    enabled = _registered_names(licence_accept_enabled=True)
    assert "cds_accept_licence" in enabled


def test_licence_accept_knob_defaults_off() -> None:
    from copernicus_mcp.config import ConfigLoader

    assert ConfigLoader().load().budget.cds_licence_accept_enabled is False


@pytest.mark.asyncio
async def test_terms_error_hint_mentions_the_in_band_path(
    foundation, monkeypatch
) -> None:
    """The licence error already carries recovery_url; it now also points at
    the in-band path so an authorised agent knows it exists."""
    import sys
    import types

    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import TermsNotAcceptedError

    message = (
        "required licences not accepted; please visit "
        "https://cds.climate.copernicus.eu/datasets/insitu-gridded-observations-europe"
        "?tab=download#manage-licences to accept the required licence(s) "
        "[('licence-to-use-E-OBS-products', 1)]"
    )
    fake_module = types.ModuleType("cdsapi")
    instance = MagicMock()
    instance.retrieve = MagicMock(side_effect=RuntimeError(message))
    instance.client = MagicMock()
    fake_module.Client = MagicMock(return_value=instance)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cdsapi", fake_module)

    from copernicus_mcp.backends.cds.costing import CostingResult

    async def _fake(dataset_id, inputs, **_kwargs):
        return CostingResult(units=1.0, limit=400.0)

    monkeypatch.setattr("copernicus_mcp.backends.cds.backend.fetch_costing", _fake)
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())

    with pytest.raises(TermsNotAcceptedError) as exc:
        await backend.submit(
            {
                "dataset_id": "insitu-gridded-observations-europe",
                "inputs": {
                    "product_type": "ensemble_mean",
                    "variable": "mean_temperature",
                    "grid_resolution": "0_25deg",
                    "period": "full_period",
                    "version": "30_0e",
                },
                "__options": {"confirmed": True},
            }
        )

    hint = exc.value.error_record.next_action_hint or ""
    assert "cds_accept_licence" in hint or "accept-licence" in hint


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_licences_and_accept(monkeypatch) -> None:
    """The CLI is the operator: both commands work without the MCP opt-in."""
    import json as _json

    from typer.testing import CliRunner

    from copernicus_mcp import cli

    fake = AsyncMock()
    fake.run.side_effect = [
        {"result": {"store": "cds", "available": [], "accepted": []}},
        {"result": {"accepted": True, "licence_id": "lic-x", "revision": 2}},
    ]

    class _Builder:
        def __call__(self):
            return self

        async def __aenter__(self):
            return fake

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(cli, "_build_orchestrator_for_cli", _Builder())
    runner = CliRunner()

    res = runner.invoke(cli.app, ["cds", "licences", "--json"])
    assert res.exit_code == 0, res.stdout + str(res.exception)
    assert _json.loads(res.stdout)["store"] == "cds"
    # Local review round (MEDIUM): pin EACH command to its operation — a
    # side_effect list answers in order regardless of what was asked.
    assert fake.run.call_args.kwargs["operation"] == "list_licences"

    res = runner.invoke(cli.app, ["cds", "accept-licence", "lic-x", "2", "--json"])
    assert res.exit_code == 0, res.stdout + str(res.exception)
    assert _json.loads(res.stdout)["accepted"] is True
    kwargs = fake.run.call_args.kwargs
    assert kwargs["operation"] == "accept_licence"
    assert kwargs["params"]["licence_id"] == "lic-x"
    assert kwargs["params"]["revision"] == 2


def test_accept_licence_input_is_strict_about_revision() -> None:
    """Local review round (MEDIUM): the pydantic layer is the real MCP
    boundary — plain ``int`` would coerce ``revision=True`` to 1 before the
    backend's bool guard could see it, and a negative revision should fail
    fast locally, not on the live API."""
    import pydantic

    from copernicus_mcp.backends.cds.tools import CdsAcceptLicenceInput

    ok = CdsAcceptLicenceInput(licence_id="x", revision=2)
    assert ok.revision == 2
    with pytest.raises(pydantic.ValidationError):
        CdsAcceptLicenceInput(licence_id="x", revision=True)
    with pytest.raises(pydantic.ValidationError):
        CdsAcceptLicenceInput(licence_id="x", revision=-5)
