from __future__ import annotations

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


def _stub_backend_class():
    from copernicus_mcp.backends.abstract import AbstractBackend

    class _Stub(AbstractBackend):
        backend_id = "stub"

        async def search(self, params):
            return {"datasets": [], "total_count": 0}

        async def describe(self, identifier):
            return {"dataset_id": identifier}

        async def validate(self, params):
            if not params:
                from copernicus_mcp.errors import ValidationError
                raise ValidationError("empty")
            return {"valid": True}

        async def estimate(self, params):
            return {"estimated_size_bytes": 100, "epistemic_status": "precise"}

        async def submit(self, params):
            return {"status": "successful", "request_id": "x"}

        async def get_files(self, params):
            # T-CMEMS-GET-006: orchestrator dispatch for
            # ``operation="get"`` resolves to this method via
            # ``_OPERATION_METHOD``.
            return {
                "status": "successful",
                "request_id": "x",
                "result": {"files": []},
            }

        async def check_status(self, request_id):
            return {"status": "running", "request_id": request_id}

        async def fetch_result(self, request_id, target):
            return {"status": "successful"}

        async def cancel(self, request_id):
            return {"cancelled": True}

        @property
        def supports_async(self):
            return False

        @property
        def supports_dry_run(self):
            return True

        @property
        def requires_terms_acceptance(self):
            return False

    return _Stub


def _confirmation_backend_class():
    from copernicus_mcp.backends.abstract import AbstractBackend

    class _ConfirmBackend(AbstractBackend):
        backend_id = "stub"

        async def search(self, params):
            return {}

        async def describe(self, identifier):
            return {}

        async def validate(self, params):
            return {"valid": True}

        async def estimate(self, params):
            return {"estimated_size_bytes": 0}

        async def submit(self, params):
            from copernicus_mcp.workflow.confirmation import (
                build_size_confirmation,
            )
            raise build_size_confirmation(
                tool_name="marine_subset_dataset",
                backend="stub",
                estimated_size_bytes=10**10,
                threshold_bytes=10**9,
                source="test",
            )

        async def check_status(self, request_id):
            return {}

        async def fetch_result(self, request_id, target):
            return {}

        async def cancel(self, request_id):
            return {"cancelled": False}

        @property
        def supports_async(self):
            return False

        @property
        def supports_dry_run(self):
            return True

        @property
        def requires_terms_acceptance(self):
            return False

    return _ConfirmBackend


def _crashing_backend_class():
    from copernicus_mcp.backends.abstract import AbstractBackend

    class _Crash(AbstractBackend):
        backend_id = "stub"

        async def search(self, params):
            raise RuntimeError("password=hunter2 leaked into ad-hoc message")

        async def describe(self, identifier): return {}
        async def validate(self, params): return {"valid": True}
        async def estimate(self, params): return {}
        async def submit(self, params): return {}
        async def check_status(self, request_id): return {}
        async def fetch_result(self, request_id, target): return {}
        async def cancel(self, request_id): return {"cancelled": False}
        @property
        def supports_async(self): return False
        @property
        def supports_dry_run(self): return True
        @property
        def requires_terms_acceptance(self): return False

    return _Crash


@pytest_asyncio.fixture
async def setup(tmp_path):
    from copernicus_mcp.backends.registry import BackendRegistry
    from copernicus_mcp.workflow.orchestrator import WorkflowOrchestrator

    foundation, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    registry = BackendRegistry()
    try:
        yield foundation, registry, lambda backend: (
            registry.register(backend),
            WorkflowOrchestrator(registry=registry, foundation=foundation),
        )[1]
    finally:
        await persistence.close()


@pytest.mark.asyncio
async def test_run_dispatches_to_backend_method(setup) -> None:
    foundation, registry, build = setup
    Stub = _stub_backend_class()
    orch = build(Stub(foundation=foundation))
    out = await orch.run(backend="stub", operation="search", params={})
    assert "result" in out
    assert out["result"] == {"datasets": [], "total_count": 0}


@pytest.mark.asyncio
async def test_run_unknown_backend_returns_error_record(setup) -> None:
    foundation, registry, build = setup
    Stub = _stub_backend_class()
    orch = build(Stub(foundation=foundation))
    out = await orch.run(backend="does-not-exist", operation="search", params={})
    assert "error" in out
    assert out["error"]["error_class"] == "BackendError"


@pytest.mark.asyncio
async def test_run_validation_error_returns_error_record(setup) -> None:
    foundation, registry, build = setup
    Stub = _stub_backend_class()
    orch = build(Stub(foundation=foundation))
    out = await orch.run(backend="stub", operation="validate", params={})
    assert "error" in out
    assert out["error"]["error_class"] == "ValidationError"


@pytest.mark.asyncio
async def test_run_confirmation_required_returns_payload(setup) -> None:
    foundation, registry, build = setup
    Backend = _confirmation_backend_class()
    orch = build(Backend(foundation=foundation))
    out = await orch.run(backend="stub", operation="submit", params={})
    assert "error" not in out
    assert out["confirmation_required"] is True
    assert "estimated_size_gb" in out


@pytest.mark.asyncio
async def test_run_unexpected_exception_wrapped_and_sanitised(setup) -> None:
    foundation, registry, build = setup
    Backend = _crashing_backend_class()
    orch = build(Backend(foundation=foundation))
    out = await orch.run(backend="stub", operation="search", params={})
    assert "error" in out
    assert out["error"]["error_class"] == "BackendError"
    # Sanitiser must scrub the credential-shaped substring.
    serialised = str(out)
    assert "hunter2" not in serialised


@pytest.mark.asyncio
async def test_run_unsupported_operation_returns_error(setup) -> None:
    foundation, registry, build = setup
    Stub = _stub_backend_class()
    orch = build(Stub(foundation=foundation))
    out = await orch.run(backend="stub", operation="frobnicate", params={})
    assert "error" in out
    assert out["error"]["error_class"] == "BackendError"
    assert out["error"]["error_subclass"] == "unsupported_operation"


@pytest.mark.asyncio
async def test_run_dispatches_get_to_get_files(setup) -> None:
    """T-CMEMS-GET-006: ``operation="get"`` resolves via
    ``_OPERATION_METHOD`` to ``backend.get_files``."""
    foundation, registry, build = setup
    Stub = _stub_backend_class()
    orch = build(Stub(foundation=foundation))
    out = await orch.run(backend="stub", operation="get", params={"dataset_id": "ds"})
    # The orchestrator wraps the backend's return under ``result``;
    # the stub's nested response.result then surfaces as
    # ``out["result"]["result"]["files"]``.
    assert "result" in out
    assert out["result"]["status"] == "successful"
    assert "files" in out["result"]["result"]


@pytest.mark.asyncio
async def test_run_dispatches_describe(setup) -> None:
    foundation, registry, build = setup
    Stub = _stub_backend_class()
    orch = build(Stub(foundation=foundation))
    out = await orch.run(
        backend="stub", operation="describe", params={"identifier": "ds-1"}
    )
    assert out["result"]["dataset_id"] == "ds-1"


@pytest.mark.asyncio
async def test_run_dispatches_check_status(setup) -> None:
    foundation, registry, build = setup
    Stub = _stub_backend_class()
    orch = build(Stub(foundation=foundation))
    out = await orch.run(
        backend="stub",
        operation="poll",
        params={"request_id": "rid-1"},
    )
    assert out["result"]["request_id"] == "rid-1"


@pytest.mark.asyncio
async def test_run_dispatches_cancel(setup) -> None:
    foundation, registry, build = setup
    Stub = _stub_backend_class()
    orch = build(Stub(foundation=foundation))
    out = await orch.run(
        backend="stub",
        operation="cancel",
        params={"request_id": "rid-2"},
    )
    assert out["result"]["cancelled"] is True


@pytest.mark.asyncio
async def test_dispatch_missing_required_field_raises_validation(setup) -> None:
    """codex M1: missing identifier/request_id/target → ValidationError, not opaque."""
    foundation, registry, build = setup
    Stub = _stub_backend_class()
    orch = build(Stub(foundation=foundation))

    out = await orch.run(backend="stub", operation="describe", params={})
    assert out["error"]["error_class"] == "ValidationError"
    assert "identifier" in out["error"]["message"]

    out = await orch.run(backend="stub", operation="poll", params={})
    assert out["error"]["error_class"] == "ValidationError"
    assert "request_id" in out["error"]["message"]

    out = await orch.run(
        backend="stub", operation="fetch", params={"request_id": "x"}
    )
    assert out["error"]["error_class"] == "ValidationError"
    assert "target" in out["error"]["message"]


@pytest.mark.asyncio
async def test_trace_id_appears_in_dispatched_logs(setup, caplog) -> None:
    """Plan acceptance bullet 5: trace_id in backend-method log records."""
    import logging

    from copernicus_mcp.backends.abstract import AbstractBackend
    from copernicus_mcp.observability.logger import (
        get_logger,
        trace_id_context,
    )

    captured: dict[str, str | None] = {}

    class _LogBackend(AbstractBackend):
        backend_id = "stub"

        async def search(self, params):
            captured["trace_id"] = trace_id_context.get()
            get_logger("test.backend").info("hello from backend")
            return {}

        async def describe(self, identifier): return {}
        async def validate(self, params): return {"valid": True}
        async def estimate(self, params): return {}
        async def submit(self, params): return {}
        async def check_status(self, request_id): return {}
        async def fetch_result(self, request_id, target): return {}
        async def cancel(self, request_id): return {"cancelled": False}
        @property
        def supports_async(self): return False
        @property
        def supports_dry_run(self): return True
        @property
        def requires_terms_acceptance(self): return False

    foundation, registry, build = setup
    orch = build(_LogBackend(foundation=foundation))

    with caplog.at_level(logging.INFO):
        await orch.run(backend="stub", operation="search", params={})

    assert captured["trace_id"] is not None
    assert len(captured["trace_id"]) == 32  # uuid hex


@pytest.mark.asyncio
async def test_exception_group_with_single_canonical_unwrapped(setup) -> None:
    """codex-batch-3 HIGH 2: ExceptionGroup[ValidationError] preserves class."""
    from copernicus_mcp.backends.abstract import AbstractBackend
    from copernicus_mcp.errors import ValidationError as CmcpValidationError

    class _Group(AbstractBackend):
        backend_id = "stub"

        async def search(self, params):
            raise ExceptionGroup(
                "wrap", [CmcpValidationError("bad input from inner task")]
            )

        async def describe(self, identifier): return {}
        async def validate(self, params): return {"valid": True}
        async def estimate(self, params): return {}
        async def submit(self, params): return {}
        async def check_status(self, request_id): return {}
        async def fetch_result(self, request_id, target): return {}
        async def cancel(self, request_id): return {"cancelled": False}
        @property
        def supports_async(self): return False
        @property
        def supports_dry_run(self): return True
        @property
        def requires_terms_acceptance(self): return False

    foundation, registry, build = setup
    orch = build(_Group(foundation=foundation))
    out = await orch.run(backend="stub", operation="search", params={})
    assert out["error"]["error_class"] == "ValidationError"
    assert out["error"]["recovery_action"] == "modify_request_parameters"


@pytest.mark.asyncio
async def test_nested_exception_group_unwrapped(setup) -> None:
    """codex-batch-3-followup HIGH 2: nested ExceptionGroup unwraps recursively."""
    from copernicus_mcp.backends.abstract import AbstractBackend
    from copernicus_mcp.errors import ValidationError as CmcpValidationError

    class _Nested(AbstractBackend):
        backend_id = "stub"

        async def search(self, params):
            inner = ExceptionGroup(
                "inner", [CmcpValidationError("nested validation")]
            )
            raise ExceptionGroup("outer", [inner])

        async def describe(self, identifier): return {}
        async def validate(self, params): return {"valid": True}
        async def estimate(self, params): return {}
        async def submit(self, params): return {}
        async def check_status(self, request_id): return {}
        async def fetch_result(self, request_id, target): return {}
        async def cancel(self, request_id): return {"cancelled": False}
        @property
        def supports_async(self): return False
        @property
        def supports_dry_run(self): return True
        @property
        def requires_terms_acceptance(self): return False

    foundation, registry, build = setup
    orch = build(_Nested(foundation=foundation))
    out = await orch.run(backend="stub", operation="search", params={})
    assert out["error"]["error_class"] == "ValidationError"
    # codex-batch-3-pass4 MEDIUM 1: recovery_action must survive unwrapping —
    # it's the user-actionable info that distinguishes "fix your request"
    # from "report a bug".
    assert out["error"]["recovery_action"] == "modify_request_parameters"


@pytest.mark.asyncio
async def test_mixed_exception_group_unwraps_canonical_leaf(setup) -> None:
    """codex-batch-3-pass4 MEDIUM 2: EG(canonical, non-canonical) still surfaces canonical."""
    from copernicus_mcp.backends.abstract import AbstractBackend
    from copernicus_mcp.errors import ValidationError as CmcpValidationError

    class _Mixed(AbstractBackend):
        backend_id = "stub"

        async def search(self, params):
            raise ExceptionGroup(
                "mixed",
                [
                    CmcpValidationError("user fixable"),
                    RuntimeError("cleanup also failed"),
                ],
            )

        async def describe(self, identifier): return {}
        async def validate(self, params): return {"valid": True}
        async def estimate(self, params): return {}
        async def submit(self, params): return {}
        async def check_status(self, request_id): return {}
        async def fetch_result(self, request_id, target): return {}
        async def cancel(self, request_id): return {"cancelled": False}
        @property
        def supports_async(self): return False
        @property
        def supports_dry_run(self): return True
        @property
        def requires_terms_acceptance(self): return False

    foundation, registry, build = setup
    orch = build(_Mixed(foundation=foundation))
    out = await orch.run(backend="stub", operation="search", params={})
    assert out["error"]["error_class"] == "ValidationError"
    assert out["error"]["recovery_action"] == "modify_request_parameters"


@pytest.mark.asyncio
async def test_options_warned_for_positional_op(setup, caplog) -> None:
    """codex-batch-3-followup MEDIUM 1: options for describe/poll/etc. logged."""
    import logging

    foundation, registry, build = setup
    Stub = _stub_backend_class()
    orch = build(Stub(foundation=foundation))
    with caplog.at_level(logging.WARNING):
        await orch.run(
            backend="stub",
            operation="describe",
            params={"identifier": "x"},
            options={"force_refresh": True},
        )
    assert any("options ignored" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_options_threaded_into_params(setup) -> None:
    """codex-batch-3 MEDIUM 1: options must reach the backend (e.g. confirmed=True)."""
    from copernicus_mcp.backends.abstract import AbstractBackend

    captured: dict[str, Any] = {}

    class _Capture(AbstractBackend):
        backend_id = "stub"

        async def submit(self, params):
            captured.update(params)
            return {"status": "successful"}

        async def search(self, params): return {}
        async def describe(self, identifier): return {}
        async def validate(self, params): return {"valid": True}
        async def estimate(self, params): return {}
        async def check_status(self, request_id): return {}
        async def fetch_result(self, request_id, target): return {}
        async def cancel(self, request_id): return {"cancelled": False}
        @property
        def supports_async(self): return False
        @property
        def supports_dry_run(self): return True
        @property
        def requires_terms_acceptance(self): return False

    foundation, registry, build = setup
    orch = build(_Capture(foundation=foundation))
    await orch.run(
        backend="stub",
        operation="submit",
        params={"x": 1},
        options={"confirmed": True},
    )
    assert captured.get("__options", {}).get("confirmed") is True


@pytest.mark.asyncio
async def test_cancellation_propagates_unwrapped(setup) -> None:
    """asyncio.CancelledError must NOT be wrapped into a BackendError."""
    import asyncio

    from copernicus_mcp.backends.abstract import AbstractBackend

    class _Hang(AbstractBackend):
        backend_id = "stub"

        async def search(self, params):
            await asyncio.sleep(10)
            return {}

        async def describe(self, identifier): return {}
        async def validate(self, params): return {"valid": True}
        async def estimate(self, params): return {}
        async def submit(self, params): return {}
        async def check_status(self, request_id): return {}
        async def fetch_result(self, request_id, target): return {}
        async def cancel(self, request_id): return {"cancelled": False}
        @property
        def supports_async(self): return False
        @property
        def supports_dry_run(self): return True
        @property
        def requires_terms_acceptance(self): return False

    foundation, registry, build = setup
    orch = build(_Hang(foundation=foundation))

    task = asyncio.create_task(
        orch.run(backend="stub", operation="search", params={})
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
