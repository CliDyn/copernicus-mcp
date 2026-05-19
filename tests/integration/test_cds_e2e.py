"""End-to-end integration tests against the real CDS API (T-CDS-008).

**Gating.** The whole module is skipped unless ``RUN_INTEGRATION_TESTS=1``.
Individual tests additionally skip if CDS credentials are not resolvable
via ``CredentialResolver`` (env var ``CDSAPI_KEY`` or ``~/.cdsapirc``).
On a contributor machine without credentials the build stays green.

**Credentials needed.** A free CDS account
(https://cds.climate.copernicus.eu/) with a PAT. Set:

    export CDSAPI_KEY=<your-uuid-pat>
    export RUN_INTEGRATION_TESTS=1
    pytest tests/integration/test_cds_e2e.py -q

**Dataset.** ``reanalysis-era5-single-levels`` — T&C accepted by most
CDS users; the empirical smoke spike (``spikes/T-CDS-000-cdsapi-smoke``)
confirmed this is a reliable happy-path dataset. The request shape mirrors
the smoke spike's ``happy`` scenario: 2m_temperature, 1 hour, 1°x1° bbox.

**Expected runtime.** ~3-8 minutes when CDS is healthy. The submit
lifecycle test sleeps up to 10 minutes waiting for the queue.

**Cost.** The happy request produces a ~few-KB GRIB file (single hour
over 1 sq° at 0.25° resolution). Far under daily CDS quotas.

T-CDS-006 (T&C elicitation) is exercised by unit tests against fixtures
captured in the smoke spike; integration coverage would require a user
who has never accepted a particular licence — not portable, so omitted.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="set RUN_INTEGRATION_TESTS=1 to run integration tests",
)


# T&C-accepted-on-most-accounts dataset. Confirmed via T-CDS-000 smoke.
_KNOWN_DATASET_ID = "reanalysis-era5-single-levels"

# Happy-path request from spikes/T-CDS-000-cdsapi-smoke/smoke.py.
# Tiny: 1 hour x 1° x 1° single variable -> ~few KB GRIB.
# Note CDS ``area`` ordering: [north, west, south, east] (NWSE) — opposite
# of GIS WSEN; see CdsRetrieveRequest.inputs description.
_HAPPY_INPUTS: dict[str, Any] = {
    "product_type": ["reanalysis"],
    "variable": ["2m_temperature"],
    "year": ["2024"],
    "month": ["01"],
    "day": ["01"],
    "time": ["00:00"],
    "area": [50.0, 0.0, 49.0, 1.0],
    "data_format": "grib",
}


# UUID 8-4-4-4-12 — used to defensive-validate the resolved PAT before
# hitting CDS (review cr L4 / codex CX-L2): a typo like CDSAPI_KEY=foo
# resolves successfully but the online tests then hard-fail with a
# confusing CDS auth error instead of skipping clearly.
_PAT_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


@pytest.fixture
def _require_creds() -> None:
    """Skip if CDS credentials are not resolvable.

    Same fixture-time resolution pattern as test_cmems_e2e — pytest plugins
    that mutate env during collection still have their effect before this
    check. Additionally validates the resolved PAT is UUID-shape (CDS PAT
    contract — research §6.8.1) so a malformed key produces a clear skip,
    not a confusing CDS auth error mid-test.
    """
    from copernicus_mcp.auth import CredentialResolver

    resolved = CredentialResolver().resolve("cds")
    if resolved is None:
        pytest.skip(
            "CDS credentials not resolvable "
            "(set CDSAPI_KEY env var or populate ~/.cdsapirc)"
        )
    # Resolved CDS credentials surface the PAT under ``fields["key"]``.
    # Codex Round 2 CX-R2-L2: distinguish "no resolve" (skip — contributor
    # has no credentials) from "resolved but missing/malformed key" (fail
    # — real auth-shape regression we don't want to silently hide).
    pat = resolved.fields.get("key") if hasattr(resolved, "fields") else None
    if pat is None:
        pytest.fail(
            "CDS credentials resolved but ``fields['key']`` is missing — "
            "likely an auth-shape regression in CredentialResolver."
        )
    if not isinstance(pat, str) or not _PAT_UUID_RE.match(pat):
        pytest.skip(
            "CDSAPI_KEY is not UUID-shape — looks malformed (CDS PATs are "
            "canonical UUIDs). Set a valid PAT or unset to skip."
        )


@pytest_asyncio.fixture
async def orchestrator(tmp_path: Path, monkeypatch):
    """Build a fresh foundation + orchestrator scoped to ``tmp_path``.

    Enables both CMEMS and CDS — the orchestrator only dispatches to the
    backend named in each ``run()`` call, so other backends being
    registered has no effect on the CDS path.

    Review cr L5: skips reading the contributor's
    ``~/.config/copernicus-mcp/config.yaml`` so per-backend settings
    (observability, http timeouts, budget) don't leak in and skew the
    test. ``explicit_config_path`` alone is not enough — the loader
    merges user files BEFORE the explicit path, so we monkeypatch the
    user-paths tuple to empty.
    """
    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.config import ConfigLoader
    from copernicus_mcp.config import loader as _loader_module
    from copernicus_mcp.workflow.orchestrator import WorkflowOrchestrator

    monkeypatch.setattr(_loader_module, "_USER_CONFIG_PATHS", ())
    config = ConfigLoader().load(
        cli_overrides={
            "storage": {
                "state_database": str(tmp_path / "state.db"),
                "cache_directory": str(tmp_path / "cache"),
            },
            "enabled_backends": ["cmems", "cds"],
        },
    )
    foundation = await build_foundation(config)
    try:
        registry = await build_backend_registry(foundation)
        yield WorkflowOrchestrator(registry=registry, foundation=foundation)
    finally:
        await foundation.persistence.close()


def _ok(envelope: dict[str, Any]) -> dict[str, Any]:
    """Strip orchestrator success envelope or fail with the error record."""
    if "error" in envelope:
        pytest.fail(f"orchestrator returned error: {envelope['error']}")
    if "result" in envelope and isinstance(envelope["result"], dict):
        return envelope["result"]
    return envelope


def _happy_params() -> dict[str, Any]:
    return {"dataset_id": _KNOWN_DATASET_ID, "inputs": _HAPPY_INPUTS}


# ---------------------------------------------------------------------------
# Offline smoke tests — exercise the bundled catalogue snapshot + heuristic
# estimator. These pass without network if the snapshot is current; they
# verify the wiring of the orchestrator + backend, not the CDS server.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_search_returns_known_dataset(
    _require_creds, orchestrator
) -> None:
    """Bundled catalogue snapshot returns the well-known ERA5 dataset.

    Search results use the slim shape: ``{id, title, description, store}``
    (no ``dataset_id`` alias — research §6.3.1 catalogue snapshot).
    """
    payload = _ok(
        await orchestrator.run(
            backend="cds",
            operation="search",
            params={"keyword": "ERA5", "limit": 50},
        )
    )
    results = payload.get("datasets") or []
    assert results, f"search returned no datasets: {payload}"
    ids = {r.get("id") for r in results}
    assert _KNOWN_DATASET_ID in ids, (
        f"known dataset {_KNOWN_DATASET_ID!r} not found in {ids}"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_describe_known_dataset(_require_creds, orchestrator) -> None:
    """Describe returns the full STAC item for the known dataset.

    Full STAC shape: ``{id, description, extent, keywords, license,
    links, providers, ...}`` — note ``title`` is absent on full STAC
    items, only on search summaries.
    """
    payload = _ok(
        await orchestrator.run(
            backend="cds",
            operation="describe",
            params={"identifier": _KNOWN_DATASET_ID},
        )
    )
    assert payload.get("id") == _KNOWN_DATASET_ID, payload
    # STAC items always carry a description.
    assert payload.get("description"), payload


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_estimate_small_request(_require_creds, orchestrator) -> None:
    """Heuristic estimator returns a positive size well under the gate."""
    payload = _ok(
        await orchestrator.run(
            backend="cds",
            operation="estimate",
            params=_happy_params(),
        )
    )
    size = payload.get("estimated_size_bytes")
    assert isinstance(size, int) and size > 0, payload
    # Default size threshold is 1 GB; this single-field, single-hour
    # request must stay well under it.
    assert size < 1_000_000_000, f"estimate too large: {size} bytes"
    # Queue tier classification — research §6.5.4. Tiny request -> light.
    assert payload.get("queue_latency_tier") == "light", payload


# ---------------------------------------------------------------------------
# Online lifecycle tests — submit a real request and walk it to
# successful + download.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
# Codex Round 2 CX-R2-L4: 100 attempts × 5s = 500s poll budget, leaving
# 100s of pytest-timeout headroom for submit + fetch + finally cleanup.
@pytest.mark.timeout(600)
async def test_submit_check_download_lifecycle(
    _require_creds, orchestrator, tmp_path: Path
) -> None:
    """T-CDS-008 happy path: submit -> poll -> successful -> fetch.

    Verifies:
      - submit returns a non-empty request_id and queued/running status.
      - poll converges to ``successful`` within the timeout.
      - fetch returns a descriptor pointing at an on-disk file.
      - file is non-empty (sanity check; CDS sometimes returns 0-byte
        files for malformed requests we'd expect to fail at submit).
    """
    submit_envelope = await orchestrator.run(
        backend="cds",
        operation="submit",
        params=_happy_params(),
        options={"confirmed": True},
    )
    submit_payload = _ok(submit_envelope)
    request_id = submit_payload.get("request_id")
    assert request_id, submit_payload
    # First response shape: queued or already-cached.
    initial_status = submit_payload.get("status")
    assert initial_status in {"queued", "running", "successful"}, (
        submit_payload
    )

    # Review codex CX-M1 / cr L1: ensure the in-flight CDS request is
    # cancelled if pytest-timeout or any assertion bails out before
    # terminal status. Otherwise the queued job lingers on the server
    # burning CDS quota.
    status: str | None = initial_status
    status_payload: dict[str, Any] = submit_payload
    try:
        # Poll until terminal. 5s × 100 attempts = 500s — leaves 100s
        # of pytest-timeout headroom for submit + fetch + cleanup.
        for _ in range(100):
            if status in {"successful", "failed", "cancelled"}:
                break
            await asyncio.sleep(5)
            status_envelope = await orchestrator.run(
                backend="cds",
                operation="poll",
                params={"request_id": request_id},
            )
            status_payload = _ok(status_envelope)
            status = status_payload.get("status")

        assert status == "successful", (
            f"lifecycle did not converge to successful: {status_payload}"
        )

        # Fetch — CDS backend stores the file under the canonical cache
        # key regardless of ``target``; the wrapper uses a sentinel
        # string.
        fetch_envelope = await orchestrator.run(
            backend="cds",
            operation="fetch",
            params={
                "request_id": request_id,
                "target": str(tmp_path / "fetch_target_unused"),
            },
        )
        fetch_payload = _ok(fetch_envelope)
        inner = fetch_payload.get("result") or fetch_payload
        filepath = inner.get("filepath")
        assert filepath, fetch_payload
        file_path = Path(filepath)
        assert file_path.is_file(), f"fetched file missing: {file_path}"
        assert file_path.stat().st_size > 0, "fetched file is empty"
    finally:
        # Best-effort cleanup: only if the request never reached a
        # terminal state. ``contextlib.suppress`` so cleanup never
        # masks the real failure.
        if status not in {"successful", "failed", "cancelled"}:
            with contextlib.suppress(Exception):
                await orchestrator.run(
                    backend="cds",
                    operation="cancel",
                    params={"request_id": request_id},
                )


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_submit_then_cancel(_require_creds, orchestrator) -> None:
    """Submit and immediately cancel; status should be ``cancelled``
    or ``successful`` (race window where the server completed first)."""
    submit_payload = _ok(
        await orchestrator.run(
            backend="cds",
            operation="submit",
            params=_happy_params(),
            options={"confirmed": True},
        )
    )
    request_id = submit_payload.get("request_id")
    assert request_id, submit_payload

    cancel_payload = _ok(
        await orchestrator.run(
            backend="cds",
            operation="cancel",
            params={"request_id": request_id},
        )
    )
    assert cancel_payload.get("request_id") == request_id, cancel_payload
    # The cancel envelope's ``status`` is the workflow row's terminal
    # status. ``successful`` is acceptable if the toolbox finished
    # between submit and cancel; otherwise expect ``cancelled``.
    assert cancel_payload.get("status") in {"cancelled", "successful"}, (
        cancel_payload
    )


# ---------------------------------------------------------------------------
# CLI smoke — exercises Typer wiring end-to-end without going through
# pytest-asyncio. Subprocess invocation matches the marine_e2e pattern.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(90)
def test_cli_cds_search_json(_require_creds) -> None:
    """`copernicus-mcp cds search --json` returns at least one dataset.

    Subprocess invocation reads the loader's env override
    ``COPERNICUS_MCP_ENABLED_BACKENDS`` to opt cds in for this one call —
    avoids mutating the user's ``~/.config/copernicus-mcp/config.yaml``.

    Review cr L6: strip any pre-existing ``COPERNICUS_MCP_*`` overrides
    from the inherited environment so a contributor's unrelated dev
    overrides cannot skew the test.
    """
    inherited = {
        k: v for k, v in os.environ.items()
        if not k.startswith("COPERNICUS_MCP_")
    }
    env = {**inherited, "COPERNICUS_MCP_ENABLED_BACKENDS": "cmems,cds"}
    cmd = [
        sys.executable,
        "-m",
        "copernicus_mcp",
        "cds",
        "search",
        "--keyword",
        "ERA5",
        "--limit",
        "1",
        "--json",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=60, env=env
    )
    assert result.returncode == 0, (result.returncode, result.stderr)
    parsed = json.loads(result.stdout)
    datasets = parsed.get("datasets")
    assert datasets, parsed
    assert len(datasets) >= 1


# ---------------------------------------------------------------------------
# T-033 expansion: post-T-CDS-008 surface (apply_constraints, search_groups,
# bundled-snapshot filters, smart-extension downloads). Each test still
# requires resolved CDS credentials because the CDS backend's registration
# gates on them (double-gate per T-CDS-007) — the orchestrator fixture
# otherwise dispatches to an unregistered backend.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_apply_constraints_live_returns_remaining_fields(
    _require_creds, orchestrator
) -> None:
    """T-CDS-016: ``apply_constraints`` with an empty inputs dict must
    return the dataset's top-level valid values from the LIVE store
    endpoint (not the bundled snapshot). Asserts the canonical
    ``valid_remaining`` shape and that core ERA5 axes show up."""
    payload = _ok(
        await orchestrator.run(
            backend="cds",
            operation="apply_constraints",
            params={"dataset_id": _KNOWN_DATASET_ID, "inputs": {}},
        )
    )
    assert payload.get("dataset_id") == _KNOWN_DATASET_ID, payload
    # Round-1 cr LOW (codex): ERA5 currently lives in `cds` but the
    # snapshot's store mapping can change; assert membership of the
    # known store set rather than pinning the current placement (the
    # backend-behaviour regression we want to catch is shape, not the
    # specific store).
    assert payload.get("store") in {"cds", "ads", "ewds"}, payload
    valid = payload.get("valid_remaining") or {}
    assert isinstance(valid, dict) and valid, payload
    # Core ERA5 axes the server always exposes — fail loudly on schema drift.
    for key in ("variable", "year", "month", "day", "time"):
        assert key in valid, f"missing {key!r} in {list(valid.keys())}"


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_search_groups_returns_populated_groups(
    _require_creds, orchestrator
) -> None:
    """T-CDS-021 PR-2: bundled-snapshot exercise of the hierarchical
    discovery tool. Confirms the orchestrator wiring + the snapshot
    parse together produce a non-trivial group list with the documented
    record shape."""
    payload = _ok(
        await orchestrator.run(
            backend="cds",
            operation="search_groups",
            params={"query": "atmosphere reanalysis", "top_k": 5},
        )
    )
    groups = payload.get("groups") or []
    assert groups, payload
    assert payload.get("total_count") == len(groups), payload
    sample = groups[0]
    for key in ("id", "domain", "category", "dataset_count", "sample_titles"):
        assert key in sample, sample
    # Round-1 cr LOW (codex): integration test asserts the shape +
    # presence of a relevant group rather than the exact top-rank
    # ordering (which is ranking-weight-sensitive and already pinned
    # by unit tests). A group with atmosphere domain + reanalysis
    # category MUST appear somewhere in the returned shortlist.
    assert any(
        g["category"] == "reanalysis" and "atmosphere" in g["domain"].lower()
        for g in groups
    ), groups


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_search_datasets_bbox_filter_narrows_to_regional(
    _require_creds, orchestrator
) -> None:
    """T-CDS-020 PR-1: bbox filter excludes datasets whose spatial
    extent doesn't intersect. CERRA (regional Europe) survives a
    European query bbox but a South-Pacific bbox excludes it."""
    european = _ok(
        await orchestrator.run(
            backend="cds",
            operation="search",
            params={"bbox": (20.0, 30.0, 30.0, 60.0), "category": "reanalysis"},
        )
    )
    eu_ids = {r.get("id") for r in european.get("datasets") or []}
    assert "reanalysis-cerra-single-levels" in eu_ids, eu_ids

    pacific = _ok(
        await orchestrator.run(
            backend="cds",
            operation="search",
            params={
                "bbox": (-150.0, -50.0, -120.0, -30.0),
                "category": "reanalysis",
            },
        )
    )
    pac_ids = {r.get("id") for r in pacific.get("datasets") or []}
    assert "reanalysis-cerra-single-levels" not in pac_ids, pac_ids


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_search_datasets_domain_category_chaining(
    _require_creds, orchestrator
) -> None:
    """T-CDS-021 PR-2: picking a (domain, category) from
    ``cds_search_groups`` and chaining into ``cds_search_datasets``
    yields a narrow list of category-prefixed dataset ids."""
    payload = _ok(
        await orchestrator.run(
            backend="cds",
            operation="search",
            params={
                "domain": "Atmosphere (surface)",
                "category": "reanalysis",
            },
        )
    )
    datasets = payload.get("datasets") or []
    assert datasets, payload
    # Every returned id must be in the reanalysis family.
    assert all(
        (r.get("id") or "").startswith("reanalysis-") for r in datasets
    ), [r.get("id") for r in datasets]


@pytest.mark.asyncio
# CDS netcdf submit + poll + fetch can take ~5 min on a healthy queue;
# tighter timeout would mask infrastructure issues but be too generous
# under a stuck queue.
@pytest.mark.timeout(600)
async def test_submit_check_download_netcdf_yields_smart_extension(
    _require_creds, orchestrator, tmp_path: Path
) -> None:
    """T-CDS-018: a submit requesting ``data_format=netcdf`` lands a
    file with a ``.nc`` extension (or ``.zip`` if ECMWF wraps the
    response — the magic-byte sniff overrides the input-derived
    extension) and ``metadata.content_type`` matches the on-disk shape."""
    # Round-1 cr LOW (codex): deepcopy so the shared ``area`` list in
    # _HAPPY_INPUTS can't be mutated by anything downstream and bleed
    # into other tests via the module-level constant.
    netcdf_inputs = copy.deepcopy(_HAPPY_INPUTS)
    netcdf_inputs["data_format"] = "netcdf"
    netcdf_inputs["download_format"] = "unarchived"

    submit_payload = _ok(
        await orchestrator.run(
            backend="cds",
            operation="submit",
            params={"dataset_id": _KNOWN_DATASET_ID, "inputs": netcdf_inputs},
            options={"confirmed": True},
        )
    )
    request_id = submit_payload.get("request_id")
    assert request_id, submit_payload

    status: str | None = submit_payload.get("status")
    last_payload: dict[str, Any] = submit_payload
    try:
        for _ in range(100):
            if status in {"successful", "failed", "cancelled"}:
                break
            await asyncio.sleep(5)
            last_payload = _ok(
                await orchestrator.run(
                    backend="cds",
                    operation="poll",
                    params={"request_id": request_id},
                )
            )
            status = last_payload.get("status")
        assert status == "successful", last_payload

        result = last_payload.get("result") or {}
        filepath = result.get("filepath")
        assert filepath, last_payload
        # Smart extension: netcdf request -> .nc unless ECMWF wrapped
        # multi-variable into a zip (single-variable here, so .nc is
        # the expected path; .zip is the documented fallback we accept).
        assert filepath.endswith((".nc", ".zip")), filepath
        metadata = result.get("metadata") or {}
        ctype = metadata.get("content_type")
        expected_ctypes = {
            ".nc": "application/x-netcdf",
            ".zip": "application/zip",
        }
        suffix = Path(filepath).suffix
        assert ctype == expected_ctypes.get(suffix), (
            f"content_type {ctype!r} mismatches suffix {suffix!r}"
        )
    finally:
        if status not in {"successful", "failed", "cancelled"}:
            with contextlib.suppress(Exception):
                await orchestrator.run(
                    backend="cds",
                    operation="cancel",
                    params={"request_id": request_id},
                )
