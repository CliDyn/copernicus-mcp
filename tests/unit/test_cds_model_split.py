"""Model as a leading fan-out axis (T-CDS-MODEL-001).

field run 19: one request for five CMIP6 models returned a zip containing ONLY
the first model — no error, no per-model status. A controlled cdsapi probe
confirmed the cause is service-side: the Rook backend executes ONE model per
request while the constraints endpoint happily accepts a list. The loss is
semantic, not size, so for the registered dataset families a multi-model
request ALWAYS fans out into one child per model (per combination of model
axes for CORDEX, which has two), unconditionally — composed with the existing
calendar escalation when a per-model child exceeds the cost limit.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

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


def _with_budget(foundation, **updates):
    budget = foundation.config.budget.model_copy(update=updates)
    config = foundation.config.model_copy(update={"budget": budget})
    return dataclasses.replace(foundation, config=config)


def _cmip6_params(models: list[str] | str) -> dict[str, Any]:
    return {
        "dataset_id": "projections-cmip6",
        "inputs": {
            "temporal_resolution": "monthly",
            "experiment": "historical",
            "variable": "near_surface_air_temperature",
            "model": models,
            "year": ["2000", "2001"],
            "month": [f"{m:02d}" for m in range(1, 13)],
        },
    }


def _cordex_params(rcms: list[str]) -> dict[str, Any]:
    return {
        "dataset_id": "projections-cordex-domains-single-levels",
        "inputs": {
            "domain": "europe",
            "horizontal_resolution": "0_11_degree_x_0_11_degree",
            "experiment": "historical",
            "temporal_resolution": "monthly_mean",
            "variable": ["2m_air_temperature"],
            "gcm_model": "mpi_m_mpi_esm_lr",
            "rcm_model": rcms,
            "ensemble_member": "r1i1p1",
            # The LIVE retrieve form (snapshot refreshed 2026-08-04): year +
            # month, not the silently-ignored start_year/end_year blocks.
            "year": ["1971"],
            "month": ["01"],
        },
    }


def _patch_costing_flat(monkeypatch, *, units: float, limit: float) -> None:
    """Every child costs the same — no calendar escalation triggered."""
    from copernicus_mcp.backends.cds.costing import CostingResult

    async def _fake(dataset_id, inputs, **_kwargs):
        return CostingResult(units=units, limit=limit)

    monkeypatch.setattr("copernicus_mcp.backends.cds.backend.fetch_costing", _fake)


def _patch_costing_per_model_year(
    monkeypatch, *, per_cell: float, limit: float
) -> None:
    """Cost scales with n_models × n_years, so a multi-year single-model child
    exceeds the limit while a one-year one fits — the composition path."""
    from copernicus_mcp.backends.cds.costing import CostingResult

    async def _fake(dataset_id, inputs, **_kwargs):
        models = inputs.get("model")
        years = inputs.get("year")
        n_models = len(models) if isinstance(models, list) else 1
        n_years = len(years) if isinstance(years, list) else 1
        return CostingResult(units=per_cell * n_models * n_years, limit=limit)

    monkeypatch.setattr("copernicus_mcp.backends.cds.backend.fetch_costing", _fake)


def _fake_remote(request_id: str) -> MagicMock:
    remote = MagicMock()
    remote.request_id = request_id
    return remote


def _patch_cdsapi_recording(monkeypatch, status_by_request):
    """The chunk-lifecycle fake, plus a log of every submitted request dict so
    tests can assert what actually went to CDS."""
    import sys
    import types

    fake_module = types.ModuleType("cdsapi")
    instance = MagicMock()
    counter = {"n": 0}
    submitted: list[dict[str, Any]] = []

    def _retrieve(name, request, target):
        counter["n"] += 1
        submitted.append(dict(request))
        return _fake_remote(f"child-{counter['n']}")

    instance.retrieve = MagicMock(side_effect=_retrieve)
    inner = MagicMock()

    def _get_remote(request_id):
        rem = MagicMock()
        rem.json = {
            "status": status_by_request.get(request_id, "running"),
            "jobID": request_id,
        }
        return rem

    inner.get_remote = MagicMock(side_effect=_get_remote)
    inner.delete = MagicMock(return_value={"deleted": True})

    def _download_results(request_id, target):
        Path(target).write_bytes(b"GRIB-chunk")
        return target

    inner.download_results = MagicMock(side_effect=_download_results)
    instance.client = inner
    fake_class = MagicMock(return_value=instance)
    fake_module.Client = fake_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cdsapi", fake_module)
    return instance, submitted


def _backend(foundation):
    from copernicus_mcp.backends.cds.backend import CdsBackend

    return CdsBackend(foundation=foundation, credentials=_fake_creds())


# ---------------------------------------------------------------------------
# pure proposal
# ---------------------------------------------------------------------------


def test_propose_model_chunks_one_per_model() -> None:
    from copernicus_mcp.backends.cds.chunking import propose_model_chunks

    overrides = propose_model_chunks(
        {"model": ["access_cm2", "miroc6"], "year": ["2000"]}, ("model",)
    )
    assert overrides == [{"model": ["access_cm2"]}, {"model": ["miroc6"]}]


def test_propose_model_chunks_none_without_a_multi_list() -> None:
    from copernicus_mcp.backends.cds.chunking import propose_model_chunks

    assert propose_model_chunks({"model": "access_cm2"}, ("model",)) is None
    assert propose_model_chunks({"model": ["access_cm2"]}, ("model",)) is None
    assert propose_model_chunks({"year": ["2000"]}, ("model",)) is None


def test_propose_model_chunks_cartesian_over_two_axes() -> None:
    """CORDEX: gcm_model × rcm_model. Only the multi-valued axes fan out."""
    from copernicus_mcp.backends.cds.chunking import propose_model_chunks

    overrides = propose_model_chunks(
        {
            "gcm_model": ["g1", "g2"],
            "rcm_model": ["r1", "r2"],
            "start_year": ["1971"],
        },
        ("gcm_model", "rcm_model"),
    )
    assert overrides == [
        {"gcm_model": ["g1"], "rcm_model": ["r1"]},
        {"gcm_model": ["g1"], "rcm_model": ["r2"]},
        {"gcm_model": ["g2"], "rcm_model": ["r1"]},
        {"gcm_model": ["g2"], "rcm_model": ["r2"]},
    ]

    single_gcm = propose_model_chunks(
        {"gcm_model": "g1", "rcm_model": ["r1", "r2"]},
        ("gcm_model", "rcm_model"),
    )
    assert single_gcm == [{"rcm_model": ["r1"]}, {"rcm_model": ["r2"]}]


def test_registry_names_the_two_rook_datasets() -> None:
    from copernicus_mcp.backends.cds.chunking import SINGLE_MODEL_EXECUTION_AXES

    assert SINGLE_MODEL_EXECUTION_AXES["projections-cmip6"] == ("model",)
    assert SINGLE_MODEL_EXECUTION_AXES[
        "projections-cordex-domains-single-levels"
    ] == ("gcm_model", "rcm_model")
    # Conservative: SIS derived products are NOT gated without evidence.
    assert "sis-extreme-indices-cmip6" not in SINGLE_MODEL_EXECUTION_AXES


# ---------------------------------------------------------------------------
# end-to-end submit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_model_cmip6_fans_out_one_child_per_model(
    foundation, monkeypatch
) -> None:
    """The run-19 shape (reduced to two models): each child request that
    reaches CDS carries EXACTLY one model; the caller gets a chunked parent."""
    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch, units=10.0, limit=400.0)
    _, submitted = _patch_cdsapi_recording(monkeypatch, status)
    backend = _backend(foundation)

    out = await backend.submit(_cmip6_params(["access_cm2", "miroc6"]))

    assert out["chunked"] is True
    assert out["chunk_count"] == 2
    assert [r["model"] for r in submitted] == [["access_cm2"], ["miroc6"]]
    # Everything except the model axis is inherited untouched.
    assert all(r["year"] == ["2000", "2001"] for r in submitted)

    parent = await foundation.persistence.fetch_workflow(out["request_id"])
    plan = json.loads(parent["chunk_plan_json"])
    assert plan["granularity"] == "model"
    assert len(plan["chunks"]) == 2


@pytest.mark.asyncio
async def test_over_limit_model_child_composes_with_calendar_split(
    foundation, monkeypatch
) -> None:
    """2 models × 2 years, each (model, year) cell = 300 units, limit 400: a
    per-model child costs 600 → escalate to year → 4 children carrying BOTH
    the model and the year override."""
    status: dict[str, str] = {}
    _patch_costing_per_model_year(monkeypatch, per_cell=300.0, limit=400.0)
    _, submitted = _patch_cdsapi_recording(monkeypatch, status)
    backend = _backend(foundation)

    params = _cmip6_params(["access_cm2", "miroc6"])
    params["__options"] = {"confirmed": True}  # the seed-backed size trips the gate
    out = await backend.submit(params)

    assert out["chunk_count"] == 4
    assert [(r["model"], r["year"]) for r in submitted] == [
        (["access_cm2"], ["2000"]),
        (["access_cm2"], ["2001"]),
        (["miroc6"], ["2000"]),
        (["miroc6"], ["2001"]),
    ]
    parent = await foundation.persistence.fetch_workflow(out["request_id"])
    plan = json.loads(parent["chunk_plan_json"])
    assert plan["granularity"] == "model+year"


@pytest.mark.asyncio
async def test_single_model_request_is_not_chunked(foundation, monkeypatch) -> None:
    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch, units=10.0, limit=400.0)
    _, submitted = _patch_cdsapi_recording(monkeypatch, status)
    backend = _backend(foundation)

    out = await backend.submit(_cmip6_params(["access_cm2"]))

    assert "chunked" not in out
    assert len(submitted) == 1
    assert submitted[0]["model"] == ["access_cm2"]


@pytest.mark.asyncio
async def test_cordex_rcm_list_fans_out_calendar_axes_untouched(
    foundation, monkeypatch
) -> None:
    """A CORDEX model fan-out under the cost limit leaves the calendar axes
    untouched in every child."""
    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch, units=10.0, limit=400.0)
    _, submitted = _patch_cdsapi_recording(monkeypatch, status)
    backend = _backend(foundation)

    out = await backend.submit(_cordex_params(["knmi_racmo22e", "smhi_rca4"]))

    assert out["chunk_count"] == 2
    assert [r["rcm_model"] for r in submitted] == [["knmi_racmo22e"], ["smhi_rca4"]]
    assert all(r["year"] == ["1971"] and r["month"] == ["01"] for r in submitted)


@pytest.mark.asyncio
async def test_unknown_model_token_is_rejected_before_any_sdk_call(
    foundation, monkeypatch
) -> None:
    """Invariant-2 parity with the calendar guard: split overrides are
    persisted verbatim in the plan, so a token outside the dataset's known
    vocabulary refuses the split loudly — naming the token — instead of
    persisting it."""
    from copernicus_mcp.errors import ValidationError

    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch, units=10.0, limit=400.0)
    instance, submitted = _patch_cdsapi_recording(monkeypatch, status)
    backend = _backend(foundation)

    with pytest.raises(ValidationError) as exc:
        await backend.submit(
            _cmip6_params(["access_cm2", "abcdef0123456789abcdef0123456789"])
        )

    assert "abcdef0123456789abcdef0123456789" in str(exc.value.error_record.message)
    assert not submitted


@pytest.mark.asyncio
async def test_auto_chunk_off_refuses_loudly_instead_of_losing_models(
    foundation, monkeypatch
) -> None:
    """Disabling auto-chunk must NOT restore the silent single-model
    execution — the request is refused with a per-model instruction."""
    from copernicus_mcp.errors import ValidationError

    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch, units=10.0, limit=400.0)
    _, submitted = _patch_cdsapi_recording(monkeypatch, status)
    backend = _backend(foundation)

    params = _cmip6_params(["access_cm2", "miroc6"])
    params["__options"] = {"auto_chunk": False}
    with pytest.raises(ValidationError) as exc:
        await backend.submit(params)

    assert "one request per model" in exc.value.error_record.message
    assert not submitted


@pytest.mark.asyncio
async def test_non_registry_dataset_with_model_list_is_untouched(
    foundation, monkeypatch
) -> None:
    """SIS derived products carry a model axis but have no evidence of
    single-model execution — they submit as-is (MODEL-000 decision)."""
    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch, units=10.0, limit=400.0)
    _, submitted = _patch_cdsapi_recording(monkeypatch, status)
    backend = _backend(foundation)

    out = await backend.submit(
        {
            "dataset_id": "sis-extreme-indices-cmip6",
            "inputs": {
                "version": "2_0",
                "product_type": "base_independent",
                "variable": "cold_days",
                "model": ["access_cm2", "miroc6"],
                "ensemble_member": "r1i1p1",
                "experiment": "historical",
                "temporal_aggregation": "yearly",
                "period": "195101_201412",
            },
        }
    )

    assert "chunked" not in out
    assert len(submitted) == 1
    assert submitted[0]["model"] == ["access_cm2", "miroc6"]


# ---------------------------------------------------------------------------
# review round 1 (codex) — gate parity, pre-flight refusal, sibling precision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_split_honours_the_size_queue_confirmation_gate(
    foundation, monkeypatch
) -> None:
    """codex M1: a single-model request whose size is unknown (with
    confirm-on-unknown enabled) raises ConfirmationRequired — the two-model
    version of the SAME request must not sail past that gate just because it
    fans out."""
    from copernicus_mcp.workflow.confirmation import ConfirmationRequired

    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch, units=10.0, limit=400.0)
    _, submitted = _patch_cdsapi_recording(monkeypatch, status)
    backend = _backend(
        _with_budget(foundation, cds_confirm_on_unknown_size=True)
    )

    with pytest.raises(ConfirmationRequired):
        await backend.submit(_cmip6_params(["access_cm2", "miroc6"]))
    assert not submitted

    params = _cmip6_params(["access_cm2", "miroc6"])
    params["__options"] = {"confirmed": True}
    out = await backend.submit(params)
    assert out["chunk_count"] == 2


@pytest.mark.asyncio
async def test_known_over_limit_unsplittable_model_child_is_refused_preflight(
    foundation, monkeypatch
) -> None:
    """codex M2: a per-model child KNOWN to exceed the cost limit whose
    single-cell calendar (one year, one month) cannot be sub-split any
    further must be refused up front — submitting it anyway would burn a
    job slot on a guaranteed 403."""
    from copernicus_mcp.errors import ValidationError

    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch, units=800.0, limit=400.0)
    _, submitted = _patch_cdsapi_recording(monkeypatch, status)
    backend = _backend(foundation)

    params = _cordex_params(["knmi_racmo22e", "smhi_rca4"])
    params["__options"] = {"confirmed": True}  # past the size gate, to the refusal
    with pytest.raises(ValidationError) as exc:
        await backend.submit(params)

    assert "one request per model" in exc.value.error_record.message
    assert not submitted


@pytest.mark.asyncio
async def test_cross_model_sibling_success_does_not_corroborate_a_retry(
    foundation, monkeypatch
) -> None:
    """codex M3: a successful model-A chunk proves nothing about model-B —
    the shapes differ. An empty-log model-B failure next to a successful
    model-A must NOT be retried; the parent fails promptly."""
    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch, units=10.0, limit=400.0)
    _, submitted = _patch_cdsapi_recording(monkeypatch, status)
    backend = _backend(
        _with_budget(
            foundation,
            cds_chunk_max_inflight=2,
            cds_chunk_retry_backoff_seconds=0.0,
        )
    )
    out = await backend.submit(_cmip6_params(["access_cm2", "miroc6"]))
    parent_id = out["request_id"]
    n_submitted = len(submitted)

    status["child-1"] = "successful"  # model A
    status["child-2"] = "failed"  # model B, empty log
    st = await backend.check_status(parent_id)

    assert st["status"] == "failed"
    assert len(submitted) == n_submitted  # no resubmission
    parent = await foundation.persistence.fetch_workflow(parent_id)
    plan = json.loads(parent["chunk_plan_json"])
    assert plan["chunks"][1].get("attempt", 0) == 0


@pytest.mark.asyncio
async def test_same_model_calendar_sibling_still_corroborates(
    foundation, monkeypatch
) -> None:
    """The precise rule keeps the Phase-1 behaviour for calendar chunks OF THE
    SAME MODEL: model-A/2000 succeeded, model-A/2001 refused empty-log →
    retried."""
    status: dict[str, str] = {}
    _patch_costing_per_model_year(monkeypatch, per_cell=300.0, limit=400.0)
    _, submitted = _patch_cdsapi_recording(monkeypatch, status)
    backend = _backend(
        _with_budget(
            foundation,
            cds_chunk_max_inflight=2,
            cds_chunk_retry_backoff_seconds=0.0,
        )
    )
    # 2 models x 2 years, per-model child 600 > 400 -> model+year, 4 chunks:
    # (A,2000) (A,2001) (B,2000) (B,2001); max_inflight=2 -> A-chunks first.
    params = _cmip6_params(["access_cm2", "miroc6"])
    params["__options"] = {"confirmed": True}  # the seed-backed size trips the gate
    out = await backend.submit(params)
    parent_id = out["request_id"]

    status["child-1"] = "successful"  # (A, 2000)
    status["child-2"] = "failed"  # (A, 2001): empty log, same model succeeded
    st = await backend.check_status(parent_id)

    assert st["status"] == "running"  # retried, not fatal
    parent = await foundation.persistence.fetch_workflow(parent_id)
    plan = json.loads(parent["chunk_plan_json"])
    assert plan["chunks"][1]["attempt"] == 1


# ---------------------------------------------------------------------------
# review round 1 (local reviewer) — hygiene gate, invalid pairs, small holes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unclean_calendar_axis_refuses_the_model_sub_split() -> None:
    """local M2: the model path must apply the same invariant-2 calendar
    hygiene the cost branch does — a credential-shaped year token must never
    be persisted into chunk_plan_json via a model+calendar sub-split."""
    from copernicus_mcp.backends.cds.chunking import (
        ChunkPlanError,
        build_model_chunk_plan,
    )

    async def _costing(child_inputs):
        years = child_inputs.get("year")
        if isinstance(years, list) and len(years) == 1:
            return 100.0  # a single-"year" chunk fits — the leak path
        return 800.0  # the whole model child is over -> sub-split by year

    with pytest.raises(ChunkPlanError):
        await build_model_chunk_plan(
            {
                "model": ["access_cm2", "miroc6"],
                "year": ["2000", "hunter2secret"],
                "month": ["01"],
            },
            [{"model": ["access_cm2"]}, {"model": ["miroc6"]}],
            cost_limit=400.0,
            chunk_by="year",
            costing_fn=_costing,
        )


@pytest.mark.asyncio
async def test_costing_failure_amid_successful_siblings_names_the_pair() -> None:
    """local M5: costing answering for some combos but not others is the
    signature of an invalid GCM×RCM pair — refuse pre-flight naming it,
    instead of submitting it whole and letting the sync 400 abort the wave
    and cancel every valid sibling."""
    from copernicus_mcp.backends.cds.chunking import (
        ChunkPlanError,
        build_model_chunk_plan,
    )

    async def _costing(child_inputs):
        if child_inputs.get("rcm_model") == ["bad_pair_rcm"]:
            return None
        return 10.0

    with pytest.raises(ChunkPlanError) as exc:
        await build_model_chunk_plan(
            {"gcm_model": ["g1"], "rcm_model": ["knmi_racmo22e", "bad_pair_rcm"]},
            [{"rcm_model": ["knmi_racmo22e"]}, {"rcm_model": ["bad_pair_rcm"]}],
            cost_limit=400.0,
            chunk_by="year",
            costing_fn=_costing,
        )
    assert exc.value.reason == "invalid_model_combination"
    assert "bad_pair_rcm" in (exc.value.detail or "")


@pytest.mark.asyncio
async def test_all_costing_unavailable_still_fans_out() -> None:
    """Spike §5: when costing is down for EVERYTHING the split must still
    happen (the loss is semantic) — units 0.0, 403 as the fallback."""
    from copernicus_mcp.backends.cds.chunking import build_model_chunk_plan

    async def _costing(child_inputs):
        return None

    plan = await build_model_chunk_plan(
        {"model": ["access_cm2", "miroc6"], "year": ["2000"]},
        [{"model": ["access_cm2"]}, {"model": ["miroc6"]}],
        cost_limit=400.0,
        chunk_by="year",
        costing_fn=_costing,
    )
    assert [c.units for c in plan.chunks] == [0.0, 0.0]


def test_duplicate_model_tokens_are_deduplicated() -> None:
    from copernicus_mcp.backends.cds.chunking import propose_model_chunks

    overrides = propose_model_chunks(
        {"model": ["access_cm2", "miroc6", "access_cm2"]}, ("model",)
    )
    assert overrides == [{"model": ["access_cm2"]}, {"model": ["miroc6"]}]


@pytest.mark.asyncio
async def test_invalid_chunk_by_is_rejected_loudly(foundation, monkeypatch) -> None:
    """local LOW: the cost branch validates __options.chunk_by; the model
    branch silently fell back to year."""
    from copernicus_mcp.errors import ValidationError

    status: dict[str, str] = {}
    _patch_costing_flat(monkeypatch, units=10.0, limit=400.0)
    _, submitted = _patch_cdsapi_recording(monkeypatch, status)
    backend = _backend(foundation)

    params = _cmip6_params(["access_cm2", "miroc6"])
    params["__options"] = {"confirmed": True, "chunk_by": "decade"}
    with pytest.raises(ValidationError) as exc:
        await backend.submit(params)

    assert "chunk_by" in exc.value.error_record.message
    assert not submitted


def test_model_member_match_is_segment_exact_not_substring(tmp_path: Path) -> None:
    """local LOW: requested ec_earth3 must NOT be satisfied by a delivered
    EC-Earth3-Veg file — CMOR names are _-segmented; match whole segments."""
    import io
    import zipfile as _zipfile

    from copernicus_mcp.backends.cds.backend import _missing_model_tokens

    buf = io.BytesIO()
    with _zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("tas_Amon_EC-Earth3-Veg_historical_r1i1p1f1_gn_2000.nc", b"x")
    path = tmp_path / "d.zip"
    path.write_bytes(buf.getvalue())

    assert _missing_model_tokens(path, {"model": ["ec_earth3"]}, ("model",)) == [
        "ec_earth3"
    ]
    assert (
        _missing_model_tokens(path, {"model": ["ec_earth3_veg"]}, ("model",)) == []
    )


@pytest.mark.asyncio
async def test_transient_costing_blip_is_retried_before_invalid_verdict() -> None:
    """local round-2 MEDIUM: fetch_costing returns None on ANY failure —
    timeout, 5xx, burst throttling — not only 'combination does not exist'.
    One blip among N sequential costings must not produce a refusal whose
    guidance makes the agent drop a valid model. Re-cost once; only a
    REPEATED failure amid successful siblings is treated as invalid."""
    from copernicus_mcp.backends.cds.chunking import build_model_chunk_plan

    calls = {"n": 0}

    async def _costing(child_inputs):
        if child_inputs.get("rcm_model") == ["knmi_racmo22e"]:
            return 10.0
        calls["n"] += 1
        return None if calls["n"] == 1 else 10.0  # blip once, then fine

    plan = await build_model_chunk_plan(
        {"gcm_model": ["g1"], "rcm_model": ["knmi_racmo22e", "smhi_rca4"]},
        [{"rcm_model": ["knmi_racmo22e"]}, {"rcm_model": ["smhi_rca4"]}],
        cost_limit=400.0,
        chunk_by="year",
        costing_fn=_costing,
    )
    assert len(plan.chunks) == 2


@pytest.mark.asyncio
async def test_invalid_combination_detail_never_says_remove() -> None:
    """The wording is part of the contract: an LLM agent follows recovery
    hints literally, and 'Remove it' launders silent model loss through a
    validation error. The detail must lead with the transient possibility."""
    from copernicus_mcp.backends.cds.chunking import (
        ChunkPlanError,
        build_model_chunk_plan,
    )

    async def _costing(child_inputs):
        if child_inputs.get("rcm_model") == ["bad_pair_rcm"]:
            return None
        return 10.0

    with pytest.raises(ChunkPlanError) as exc:
        await build_model_chunk_plan(
            {"gcm_model": ["g1"], "rcm_model": ["knmi_racmo22e", "bad_pair_rcm"]},
            [{"rcm_model": ["knmi_racmo22e"]}, {"rcm_model": ["bad_pair_rcm"]}],
            cost_limit=400.0,
            chunk_by="year",
            costing_fn=_costing,
        )
    detail = exc.value.detail or ""
    assert "Remove it" not in detail
    assert "transient" in detail


def test_entry_copy_with_same_child_id_does_not_self_corroborate() -> None:
    """local round-2 LOW: the same-parse identity contract is fragile — a
    future caller pairing an entry COPY with the chunks list must not let a
    chunk corroborate itself via its own id."""
    from copernicus_mcp.backends.cds.chunking import sibling_corroborates

    chunks = [
        {"index": 0, "child_request_id": "a", "overrides": {"model": ["m1"]}},
    ]
    entry_copy = dict(chunks[0])
    assert (
        sibling_corroborates(entry_copy, chunks, {"a": "successful"}) is False
    )
