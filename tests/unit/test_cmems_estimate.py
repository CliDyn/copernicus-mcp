from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

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


def _creds():
    from copernicus_mcp.auth.resolver import ResolvedCredentials

    return ResolvedCredentials(
        backend="cmems",
        source="explicit",
        source_detail="test",
        fields={"username": "u", "password": "p"},
    )


def _subset_params() -> dict[str, Any]:
    return dict(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        variables=["thetao"],
        minimum_longitude=-1.0,
        maximum_longitude=1.0,
        minimum_latitude=0.0,
        maximum_latitude=1.0,
        minimum_depth=0.0,
        maximum_depth=100.0,
        start_datetime="2024-01-01T00:00:00Z",
        end_datetime="2024-01-02T00:00:00Z",
    )


def _make_response(file_size_mb: float, transfer_mb: float, service: str = "arco-geo-series"):
    """Mimic the toolbox's ResponseSubset shape."""
    return types.SimpleNamespace(
        file_size=file_size_mb,
        data_transfer_size=transfer_mb,
        status="DRY_RUN",
        message="dry-run only",
        file_path="/tmp/x.nc",
        file_status="DOWNLOADED",
        variables=["thetao"],
        service=service,
    )


def _install_fake_module(monkeypatch, subset_fn) -> types.ModuleType:
    mod = types.ModuleType("copernicusmarine")
    mod.subset = subset_fn  # type: ignore[attr-defined]
    # Review L7: ``backend.estimate`` calls ``self.describe`` to build the
    # coverage_advisory. Every estimate test therefore implicitly exercises
    # the describe path. The default ``{"products": []}`` stub makes
    # ``_find_dataset`` raise NotFoundError, which ``_build_coverage_advisory``
    # swallows (returns None) — so estimate tests pass without advisory.
    # Tests that want a specific extent override ``mod.describe`` directly
    # via ``_describe_with_extent(...)`` after this fixture runs.
    mod.describe = lambda **kw: {"products": []}  # type: ignore[attr-defined]

    class LoginError(Exception):
        pass

    class DatasetNotFound(Exception):
        pass

    class WrongFormatRequested(Exception):
        pass

    mod.LoginError = LoginError  # type: ignore[attr-defined]
    mod.DatasetNotFound = DatasetNotFound  # type: ignore[attr-defined]
    mod.WrongFormatRequested = WrongFormatRequested  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "copernicusmarine", mod)
    return mod


@pytest.mark.asyncio
async def test_estimate_precise_when_toolbox_returns_size(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    def fake_subset(**kwargs):
        assert kwargs.get("dry_run") is True
        return _make_response(file_size_mb=2.0, transfer_mb=2.5)

    _install_fake_module(monkeypatch, fake_subset)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    out = await backend.estimate(_subset_params())

    assert out["type"] == "free"
    # data_transfer_size (2.5 MB) is the threshold-relevant number per T-000 findings.
    assert out["estimated_size_bytes"] == int(2.5 * 1024 * 1024)
    assert out["estimated_size_human"]
    assert out["service_used"] == "arco-geo-series"
    assert out["epistemic_status"] == "precise"


@pytest.mark.asyncio
async def test_estimate_approximate_when_size_missing(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    def fake_subset(**kwargs):
        # Toolbox returns a response without data_transfer_size populated.
        return types.SimpleNamespace(
            file_size=None,
            data_transfer_size=None,
            status="DRY_RUN",
            variables=["thetao"],
            service="arco-geo-series",
        )

    _install_fake_module(monkeypatch, fake_subset)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    out = await backend.estimate(_subset_params())
    assert out["epistemic_status"] == "approximate"
    assert out["estimated_size_bytes"] >= 0


@pytest.mark.asyncio
async def test_estimate_sparse_dataset_raises_validation(
    foundation, monkeypatch
) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import ValidationError

    def fake_subset(**kwargs):
        from copernicusmarine import WrongFormatRequested  # type: ignore[attr-defined]

        raise WrongFormatRequested("arco-platform-series unsupported")

    _install_fake_module(monkeypatch, fake_subset)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(ValidationError) as exc_info:
        await backend.estimate(_subset_params())
    record = exc_info.value.error_record
    assert record.recovery_action == "modify_request_parameters"
    assert "marine_get_files" in (record.next_action_hint or "").lower() or \
           "marine_get_files" in record.message.lower()


def _describe_with_extent(min_lon, min_lat, max_lon, max_lat):
    """Build a fake describe response with the given dataset extent."""

    def fn(**kwargs):
        return {
            "products": [
                {
                    "product_id": "TEST_PRODUCT",
                    "datasets": [
                        {
                            "dataset_id": kwargs.get("dataset_id"),
                            "versions": [
                                {
                                    "label": "1.0",
                                    "parts": [
                                        {
                                            "name": "default",
                                            "services": [
                                                {
                                                    "service_name": "arco-geo-series",
                                                    "variables": [
                                                        {
                                                            "short_name": "thetao",
                                                            "bbox": [
                                                                min_lon,
                                                                min_lat,
                                                                max_lon,
                                                                max_lat,
                                                            ],
                                                        }
                                                    ],
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        }

    return fn


@pytest.mark.asyncio
async def test_estimate_emits_coverage_advisory_when_bbox_inside_extent(
    foundation, monkeypatch
) -> None:
    """When user bbox is strictly inside the dataset's spatial extent, the
    response carries a ``coverage_advisory`` so the LLM agent sees the
    mismatch instead of guessing from geography.

    Why: in a real session an agent asked for "all of Mediterranean" salinity
    with bbox lon[-6, 36.5]; the dataset's actual extent is lon[-17.29, 36.29]
    (an Atlantic buffer west of Gibraltar). Without an advisory, the agent
    silently dropped 11° of longitude and reported success.
    """
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    def fake_subset(**kwargs):
        return _make_response(file_size_mb=2.0, transfer_mb=2.5)

    fake_describe = _describe_with_extent(-17.29, 30.19, 36.29, 45.98)
    mod = _install_fake_module(monkeypatch, fake_subset)
    mod.describe = fake_describe  # type: ignore[attr-defined]

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    # User bbox strictly inside dataset extent on all axes
    params = {
        **_subset_params(),
        "minimum_longitude": -6.0,
        "maximum_longitude": 36.0,
        "minimum_latitude": 31.0,
        "maximum_latitude": 45.5,
    }
    out = await backend.estimate(params)

    advisory = out.get("coverage_advisory")
    assert advisory is not None, "expected coverage_advisory when bbox shrinks"
    assert advisory["status"] == "bbox_inside_dataset_extent"
    ext = advisory["dataset_extent"]
    assert ext["min_lon"] == -17.29
    assert ext["max_lon"] == 36.29
    assert "does not cover" in advisory["advisory_message"].lower()


@pytest.mark.asyncio
async def test_estimate_no_advisory_when_bbox_matches_extent(
    foundation, monkeypatch
) -> None:
    """Idempotent case: bbox covers the full extent → advisory is omitted
    so successful flows don't pay extra tokens."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    def fake_subset(**kwargs):
        return _make_response(file_size_mb=2.0, transfer_mb=2.5)

    fake_describe = _describe_with_extent(-17.29, 30.19, 36.29, 45.98)
    mod = _install_fake_module(monkeypatch, fake_subset)
    mod.describe = fake_describe  # type: ignore[attr-defined]

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    params = {
        **_subset_params(),
        "minimum_longitude": -17.29,
        "maximum_longitude": 36.29,
        "minimum_latitude": 30.19,
        "maximum_latitude": 45.98,
    }
    out = await backend.estimate(params)

    assert "coverage_advisory" not in out


@pytest.mark.asyncio
async def test_estimate_advisory_when_bbox_disjoint_from_extent(
    foundation, monkeypatch
) -> None:
    """Review M1: when the user bbox does NOT overlap the dataset extent
    at all, the download produces an empty / clamped-to-empty file with no
    warning at the estimate or submit step. Advisory must catch this — it
    is the most user-facing failure the feature was built to prevent."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    def fake_subset(**kwargs):
        return _make_response(file_size_mb=2.0, transfer_mb=2.5)

    fake_describe = _describe_with_extent(-17.29, 30.19, 36.29, 45.98)
    mod = _install_fake_module(monkeypatch, fake_subset)
    mod.describe = fake_describe  # type: ignore[attr-defined]

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    # User bbox entirely outside dataset extent (lon east of dataset's east edge)
    params = {
        **_subset_params(),
        "minimum_longitude": 100.0,
        "maximum_longitude": 110.0,
        "minimum_latitude": 10.0,
        "maximum_latitude": 20.0,
    }
    out = await backend.estimate(params)

    advisory = out.get("coverage_advisory")
    assert advisory is not None
    assert advisory["status"] == "bbox_disjoint_from_dataset_extent"
    assert "overlap" in advisory["advisory_message"].lower() or \
           "disjoint" in advisory["advisory_message"].lower() or \
           "does not overlap" in advisory["advisory_message"].lower()


@pytest.mark.asyncio
async def test_estimate_advisory_when_bbox_mismatch(
    foundation, monkeypatch
) -> None:
    """Review M2: when the user bbox shrinks on one axis AND exceeds on
    another (overlap exists), the advisory must mention BOTH behaviors —
    not just clamping. Previously the single ``bbox_exceeds`` message
    silently dropped the lon-shrink info."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    def fake_subset(**kwargs):
        return _make_response(file_size_mb=2.0, transfer_mb=2.5)

    fake_describe = _describe_with_extent(-17.29, 30.19, 36.29, 45.98)
    mod = _install_fake_module(monkeypatch, fake_subset)
    mod.describe = fake_describe  # type: ignore[attr-defined]

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    # lon shrinks INSIDE extent, lat EXCEEDS extent — mixed case
    params = {
        **_subset_params(),
        "minimum_longitude": -5.0,
        "maximum_longitude": 30.0,
        "minimum_latitude": 25.0,
        "maximum_latitude": 50.0,
    }
    out = await backend.estimate(params)

    advisory = out.get("coverage_advisory")
    assert advisory is not None
    assert advisory["status"] == "bbox_mismatch_dataset_extent"
    msg = advisory["advisory_message"].lower()
    # Must mention both behaviors
    assert "longitude" in msg or "lon" in msg
    assert "latitude" in msg or "lat" in msg


@pytest.mark.asyncio
async def test_estimate_advisory_when_bbox_exceeds_extent(
    foundation, monkeypatch
) -> None:
    """When user bbox is wider than the dataset's extent, the toolbox will
    silently clamp on download. Advisory must tell the agent up-front."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend

    def fake_subset(**kwargs):
        return _make_response(file_size_mb=2.0, transfer_mb=2.5)

    fake_describe = _describe_with_extent(-17.29, 30.19, 36.29, 45.98)
    mod = _install_fake_module(monkeypatch, fake_subset)
    mod.describe = fake_describe  # type: ignore[attr-defined]

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    params = {
        **_subset_params(),
        "minimum_longitude": -20.0,
        "maximum_longitude": 40.0,
        "minimum_latitude": 25.0,
        "maximum_latitude": 50.0,
    }
    out = await backend.estimate(params)

    advisory = out.get("coverage_advisory")
    assert advisory is not None
    assert advisory["status"] == "bbox_exceeds_dataset_extent"
    assert "clamp" in advisory["advisory_message"].lower()


@pytest.mark.asyncio
async def test_estimate_timeout_raises_timeout_error(
    foundation, monkeypatch
) -> None:

    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import TimeoutError as CmcpTimeoutError

    def fake_subset(**kwargs):
        import time

        time.sleep(2.0)  # simulate slow toolbox
        return _make_response(1.0, 1.0)

    _install_fake_module(monkeypatch, fake_subset)
    # Force tiny timeout to trigger wait_for cancellation.
    object.__setattr__(
        foundation.config.budget, "cmems_estimate_timeout_seconds", 0.001
    )

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(CmcpTimeoutError):
        await backend.estimate(_subset_params())


@pytest.mark.asyncio
async def test_estimate_invalid_request_raises_validation(foundation) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import ValidationError

    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    # T-CMEMS-GET-004: ``estimate`` now dispatches on params shape.
    # ``{}`` has no subset-discriminator fields and routes to
    # ``_estimate_get``, where ``CmemsGetRequest`` rejects the missing
    # ``dataset_id``. Either dispatch branch surfaces a
    # canonical ``ValidationError``.
    with pytest.raises(ValidationError):
        await backend.estimate({})
    # And the subset-shape branch is still wired up — an incomplete
    # subset request (variables present but no spatial/temporal
    # fields) routes to ``_estimate_subset`` and gets rejected.
    with pytest.raises(ValidationError):
        await backend.estimate({"dataset_id": "x", "variables": ["thetao"]})


@pytest.mark.asyncio
async def test_estimate_approximate_forces_confirmation_in_submit(
    foundation, monkeypatch
) -> None:
    """codex T-023 MEDIUM: approximate (zero) estimate must NOT bypass confirmation."""
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.workflow.confirmation import ConfirmationRequired

    def fake(**kwargs):
        return types.SimpleNamespace(
            file_size=None, data_transfer_size=None, status="DRY_RUN",
        )

    _install_fake_module(monkeypatch, fake)
    backend = CmemsBackend(foundation=foundation, credentials=_creds())
    with pytest.raises(ConfirmationRequired):
        await backend.submit(_subset_params())


def test_sanitiser_redacts_python_dict_repr_credentials() -> None:
    """codex T-023 HIGH: ``{'password': 'p'}`` Python repr must be redacted."""
    from copernicus_mcp.errors.sanitiser import Sanitiser

    out = Sanitiser().sanitise("kwargs={'password': 'hunter2'}")
    assert "hunter2" not in out


@pytest.mark.asyncio
async def test_estimate_no_credentials_raises_auth(foundation, monkeypatch) -> None:
    from copernicus_mcp.backends.cmems.backend import CmemsBackend
    from copernicus_mcp.errors import AuthError

    _install_fake_module(monkeypatch, lambda **k: _make_response(1, 1))
    backend = CmemsBackend(foundation=foundation, credentials=None)
    with pytest.raises(AuthError):
        await backend.estimate(_subset_params())
