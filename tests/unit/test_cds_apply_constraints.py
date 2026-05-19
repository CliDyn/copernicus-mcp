"""``CdsBackend.apply_constraints`` — T-CDS-016 Layer B.

Live POST to ``<store>/api/retrieve/v1/processes/<id>/constraints``
returning the remaining valid values for unfilled fields. We mock the
HTTP transport via ``httpx.MockTransport`` so no real CDS call fires.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio


def _make_foundation(tmp_path: Path, mock_transport: httpx.MockTransport):
    """Foundation whose ``http_client_factory.create`` returns an
    AsyncClient backed by the provided ``MockTransport``."""
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

    factory = HttpClientFactory(http_config=config.http)

    def create_mock(backend_id: str) -> httpx.AsyncClient:
        # cr round-1 M6: previous version created a real client just to
        # snapshot its timeout, then leaked it. Use config.http directly
        # to read the timeout — no real client constructed.
        return httpx.AsyncClient(
            timeout=config.http.default_timeout_seconds,
            transport=mock_transport,
        )

    factory.create = create_mock  # type: ignore[method-assign]

    return (
        FoundationServices(
            config=config,
            credential_resolver=CredentialResolver(),
            http_client_factory=factory,
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


def _creds():
    from copernicus_mcp.auth.resolver import ResolvedCredentials

    return ResolvedCredentials(
        backend="cds",
        source="explicit",
        source_detail="test",
        fields={"key": "abcdef01-2345-6789-abcd-ef0123456789"},
    )


@pytest_asyncio.fixture
async def foundation_with_handler(tmp_path: Path):
    """Test-friendly: ``foundation_with_handler(handler) -> foundation``."""
    factories = []

    async def build(handler):
        transport = httpx.MockTransport(handler)
        found, persistence = _make_foundation(tmp_path, transport)
        await persistence.initialise()
        factories.append((found, persistence))
        return found

    yield build

    for _, persistence in factories:
        await persistence.close()


class TestApplyConstraints:
    @pytest.mark.asyncio
    async def test_empty_inputs_returns_top_level_valid_values(
        self, foundation_with_handler
    ) -> None:
        from copernicus_mcp.backends.cds.backend import CdsBackend

        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["method"] = request.method
            return httpx.Response(
                200,
                json={
                    "data_format": ["netcdf", "grib"],
                    "download_format": ["zip", "unarchived"],
                    "variable": ["elevation", "river_discharge_in_the_last_24_hours"],
                },
            )

        foundation = await foundation_with_handler(handler)
        backend = CdsBackend(foundation=foundation, credentials=_creds())
        out = await backend.apply_constraints(
            {"dataset_id": "efas-historical", "inputs": {}}
        )
        # URL routes to the EWDS store via the catalogue lookup.
        assert "ewds.climate.copernicus.eu" in captured["url"]
        assert "/processes/efas-historical/constraints" in captured["url"]
        assert captured["method"] == "POST"
        assert out["dataset_id"] == "efas-historical"
        assert out["store"] == "ewds"
        assert out["valid_remaining"]["data_format"] == ["netcdf", "grib"]
        assert out["valid_remaining"]["download_format"] == ["zip", "unarchived"]

    @pytest.mark.asyncio
    async def test_partial_inputs_forwarded_in_post_body(
        self, foundation_with_handler
    ) -> None:
        """The 'inputs' field in the POST body must match the request."""
        import json

        from copernicus_mcp.backends.cds.backend import CdsBackend

        body_seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            body_seen.update(json.loads(request.content))
            return httpx.Response(200, json={"data_format": ["netcdf"]})

        foundation = await foundation_with_handler(handler)
        backend = CdsBackend(foundation=foundation, credentials=_creds())
        await backend.apply_constraints(
            {
                "dataset_id": "efas-historical",
                "inputs": {"variable": ["elevation"], "system_version": ["version_5_0"]},
            }
        )
        assert body_seen == {
            "inputs": {
                "variable": ["elevation"],
                "system_version": ["version_5_0"],
            }
        }

    @pytest.mark.asyncio
    async def test_404_raises_notfound(self, foundation_with_handler) -> None:
        from copernicus_mcp.backends.cds.backend import CdsBackend
        from copernicus_mcp.errors import NotFoundError

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"message": "not found"})

        foundation = await foundation_with_handler(handler)
        backend = CdsBackend(foundation=foundation, credentials=_creds())
        with pytest.raises(NotFoundError):
            await backend.apply_constraints(
                {"dataset_id": "efas-historical", "inputs": {}}
            )

    @pytest.mark.asyncio
    async def test_5xx_raises_backend_error(self, foundation_with_handler) -> None:
        from copernicus_mcp.backends.cds.backend import CdsBackend
        from copernicus_mcp.errors import BackendError

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="upstream down")

        foundation = await foundation_with_handler(handler)
        backend = CdsBackend(foundation=foundation, credentials=_creds())
        with pytest.raises(BackendError):
            await backend.apply_constraints(
                {"dataset_id": "efas-historical", "inputs": {}}
            )

    @pytest.mark.asyncio
    async def test_works_without_credentials(self, foundation_with_handler) -> None:
        """cr round-1 M3: constraints endpoint is anonymous-friendly
        (empirically verified). apply_constraints is a read-only op
        like search/describe/validate/estimate — must NOT gate on
        credentials so credential-less agents can still compose
        requests interactively."""
        from copernicus_mcp.backends.cds.backend import CdsBackend

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data_format": ["netcdf"]})

        foundation = await foundation_with_handler(handler)
        backend = CdsBackend(foundation=foundation, credentials=None)
        out = await backend.apply_constraints(
            {"dataset_id": "efas-historical", "inputs": {}}
        )
        assert out["valid_remaining"]["data_format"] == ["netcdf"]

    @pytest.mark.asyncio
    async def test_inputs_provided_is_sanitised_in_response(
        self, foundation_with_handler
    ) -> None:
        """cr round-1 M4: inputs_provided echoes user-controlled keys
        into the response. Sanitiser must scrub credential-shaped
        values from the echo so a careless caller passing
        ``password=hunter2`` in inputs doesn't leak via response /
        logs."""
        from copernicus_mcp.backends.cds.backend import CdsBackend

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        foundation = await foundation_with_handler(handler)
        backend = CdsBackend(foundation=foundation, credentials=_creds())
        out = await backend.apply_constraints(
            {
                "dataset_id": "efas-historical",
                "inputs": {"variable": "password=hunter2"},
            }
        )
        # Either the password= pattern is scrubbed, or the whole
        # value gets REDACTED. Either way, the literal secret cannot
        # appear verbatim.
        assert "hunter2" not in str(out)

    @pytest.mark.asyncio
    async def test_timeout_maps_to_timeout_error(
        self, foundation_with_handler
    ) -> None:
        """cr round-1 M5: httpx.TimeoutException → canonical TimeoutError,
        not BackendError. Distinct error_class on the wire matters for
        retry semantics."""
        from copernicus_mcp.backends.cds.backend import CdsBackend
        from copernicus_mcp.errors import TimeoutError as CmcpTimeoutError

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("simulated timeout")

        foundation = await foundation_with_handler(handler)
        backend = CdsBackend(foundation=foundation, credentials=_creds())
        with pytest.raises(CmcpTimeoutError):
            await backend.apply_constraints(
                {"dataset_id": "efas-historical", "inputs": {}}
            )

    @pytest.mark.asyncio
    async def test_connect_error_maps_to_network_error(
        self, foundation_with_handler
    ) -> None:
        """cr round-1 M5: transport-layer connection failure → canonical
        NetworkError, not BackendError."""
        from copernicus_mcp.backends.cds.backend import CdsBackend
        from copernicus_mcp.errors import NetworkError

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("simulated connect failure")

        foundation = await foundation_with_handler(handler)
        backend = CdsBackend(foundation=foundation, credentials=_creds())
        with pytest.raises(NetworkError):
            await backend.apply_constraints(
                {"dataset_id": "efas-historical", "inputs": {}}
            )

    @pytest.mark.asyncio
    async def test_unknown_field_does_not_leak_value(
        self, foundation_with_handler
    ) -> None:
        from copernicus_mcp.backends.cds.backend import CdsBackend
        from copernicus_mcp.errors import ValidationError

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        foundation = await foundation_with_handler(handler)
        backend = CdsBackend(foundation=foundation, credentials=_creds())
        with pytest.raises(ValidationError) as exc_info:
            await backend.apply_constraints(
                {"dataset_id": "efas-historical", "password": "hunter2"}
            )
        assert "hunter2" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_routes_to_cds_store_for_era5(
        self, foundation_with_handler
    ) -> None:
        from copernicus_mcp.backends.cds.backend import CdsBackend

        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            return httpx.Response(200, json={})

        foundation = await foundation_with_handler(handler)
        backend = CdsBackend(foundation=foundation, credentials=_creds())
        await backend.apply_constraints(
            {"dataset_id": "reanalysis-era5-single-levels", "inputs": {}}
        )
        assert "cds.climate.copernicus.eu" in captured["url"]

    @pytest.mark.asyncio
    async def test_routes_to_ads_store_for_cams(
        self, foundation_with_handler
    ) -> None:
        from copernicus_mcp.backends.cds.backend import CdsBackend

        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            return httpx.Response(200, json={})

        foundation = await foundation_with_handler(handler)
        backend = CdsBackend(foundation=foundation, credentials=_creds())
        await backend.apply_constraints(
            {"dataset_id": "cams-global-reanalysis-eac4", "inputs": {}}
        )
        assert "ads.atmosphere.copernicus.eu" in captured["url"]
