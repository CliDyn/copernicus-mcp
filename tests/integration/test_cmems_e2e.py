"""End-to-end integration tests against the real CMEMS API.

**Gating.** The whole module is skipped unless the environment variable
``RUN_INTEGRATION_TESTS=1`` is set. Individual tests skip if CMEMS
credentials are not resolvable via ``CredentialResolver`` (env vars
``COPERNICUSMARINE_SERVICE_USERNAME`` / ``..._PASSWORD`` or the
configured config file). With ``RUN_INTEGRATION_TESTS`` unset, ``pytest``
on a contributor machine without credentials produces a green build —
the module never runs.

**Credentials needed.** A free CMEMS account
(https://data.marine.copernicus.eu/register). Set:

    export COPERNICUSMARINE_SERVICE_USERNAME=your_user
    export COPERNICUSMARINE_SERVICE_PASSWORD=your_pass
    export RUN_INTEGRATION_TESTS=1
    pytest tests/integration -q

**Expected total runtime.** ~3-5 minutes when CMEMS is healthy. Each
individual test has a 5-minute timeout so a stuck call fails loudly
rather than hanging CI.

**Cost.** All tests use 1° × 1° × 1-day single-variable subsets so the
download is well under 10 MB and well under daily CMEMS quotas.
"""

from __future__ import annotations

import json
import os
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


# A well-known CMEMS dataset present across all environments. PSY4QV3R1
# global ocean physics analysis-and-forecast — 1/12° resolution, daily means.
_KNOWN_DATASET_ID = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
_KNOWN_PRODUCT_KEYWORD = "GLOBAL_ANALYSISFORECAST_PHY"


@pytest.fixture
def _require_creds() -> None:
    """Skip if CMEMS credentials are not resolvable.

    Resolution happens at fixture-time, not import-time — so pytest plugins
    that mutate env during collection (pytest-env, dotenv loaders) still
    have their effect before the check.
    """
    from copernicus_mcp.auth import CredentialResolver

    if CredentialResolver().resolve("cmems") is None:
        pytest.skip(
            "CMEMS credentials not resolvable "
            "(set COPERNICUSMARINE_SERVICE_USERNAME/_PASSWORD)"
        )


@pytest_asyncio.fixture
async def orchestrator(tmp_path: Path):
    """Build a fresh foundation + orchestrator scoped to ``tmp_path``."""
    from copernicus_mcp.bootstrap import build_backend_registry, build_foundation
    from copernicus_mcp.config import ConfigLoader
    from copernicus_mcp.workflow.orchestrator import WorkflowOrchestrator

    config = ConfigLoader().load(
        cli_overrides={
            "storage": {
                "state_database": str(tmp_path / "state.db"),
                "cache_directory": str(tmp_path / "cache"),
            }
        }
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


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_search_returns_known_dataset(_require_creds, orchestrator) -> None:
    """T-033 Test 1: a well-known dataset appears in search results."""
    envelope = await orchestrator.run(
        backend="cmems",
        operation="search",
        params={"keyword": _KNOWN_PRODUCT_KEYWORD, "limit": 50},
    )
    payload = _ok(envelope)
    # Backend returns {"datasets": [...], "total_count": int} — preserved
    # verbatim by the orchestrator unwrap.
    results = payload.get("datasets", [])
    assert results, f"search returned no datasets: {payload}"
    ids = {r.get("dataset_id") for r in results}
    assert any(_KNOWN_DATASET_ID in (i or "") for i in ids), (
        f"known dataset {_KNOWN_DATASET_ID!r} not found in {ids}"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_describe_known_dataset(_require_creds, orchestrator) -> None:
    """T-033 Test 2: describe a known dataset; ``thetao`` is among variables."""
    envelope = await orchestrator.run(
        backend="cmems",
        operation="describe",
        params={"identifier": _KNOWN_DATASET_ID},
    )
    payload = _ok(envelope)
    variables = payload.get("variables") or []
    var_names = {v.get("name") if isinstance(v, dict) else v for v in variables}
    assert "thetao" in var_names, f"thetao not in {var_names}"


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_estimate_small_subset(_require_creds, orchestrator) -> None:
    """T-033 Test 3: estimate returns a positive byte count and stays well
    under the confirmation threshold (1 GB) for a 1°×1°×1-day request.

    Note: the SDK's ``data_transfer_size`` reports the *chunk* size pulled
    over the wire, not the final filtered output. For a 0.083° dataset
    with ~50 vertical levels, that is ~20-30 MB even when the actually
    filtered output is ~16 KB.
    """
    payload = _ok(
        await orchestrator.run(
            backend="cmems",
            operation="estimate",
            params=_tiny_subset_params(),
        )
    )
    size = payload.get("estimated_size_bytes")
    assert isinstance(size, int) and size > 0, payload
    # Confirmation threshold is 1 GB; tiny request stays well under it.
    assert size < 1_000_000_000, f"estimate too large: {size} bytes"


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_subset_small_region(_require_creds, orchestrator, tmp_path: Path) -> None:
    """T-033 Test 4: download a tiny subset; verify file + sidecar + workflow row."""
    import hashlib

    payload = _ok(
        await orchestrator.run(
            backend="cmems",
            operation="submit",
            params=_tiny_subset_params(),
            options={"confirmed": True},
        )
    )
    # Backend submit envelope: {status, cache_hit, is_existing, request_id,
    # cache_key, result: {filepath, uri, metadata, provenance}}.
    inner = payload.get("result") or {}
    fp = inner.get("filepath")
    assert fp, payload
    file_path = Path(fp)
    assert file_path.is_file(), f"file not present: {file_path}"

    sidecar = file_path.with_suffix(file_path.suffix + ".provenance.json")
    assert sidecar.is_file(), f"sidecar missing: {sidecar}"

    # Sidecar MD5 matches the file.
    record = json.loads(sidecar.read_text())
    file_block = record.get("files", [None])[0] or {}
    md5_expected = file_block.get("md5")
    assert md5_expected, record

    md5_actual = hashlib.md5(file_path.read_bytes()).hexdigest()
    assert md5_actual == md5_expected, (md5_actual, md5_expected)

    # Workflow row is ``successful``. Backend uses ``request_id`` at the
    # top level of the submit envelope.
    request_id = payload.get("request_id")
    assert request_id, payload
    status_envelope = await orchestrator.run(
        backend="cmems",
        operation="poll",
        params={"request_id": request_id},
    )
    status_payload = _ok(status_envelope)
    assert status_payload.get("status") == "successful", status_payload


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_invalid_dataset_returns_not_found(_require_creds, orchestrator) -> None:
    """T-033 Test 5: describe with bogus id returns NotFoundError envelope."""
    envelope = await orchestrator.run(
        backend="cmems",
        operation="describe",
        params={"identifier": "this-dataset-does-not-exist-xyzzy"},
    )
    assert "error" in envelope, envelope
    record = envelope["error"]
    assert record.get("error_class") in {"NotFoundError", "BackendError"}, record


# ---------------------------------------------------------------------------
# T-CMEMS-GET-008: native-file retrieval against the real SDK.
# ---------------------------------------------------------------------------


# A sparse (in-situ) dataset — EasyCORA is the trimmed CORA T/S
# product. The download test below targets the family-standard
# ``index_history.txt`` (small text file, present at every
# CORA-family product's root) via ``file_list`` so a layout shift
# fails loudly rather than silently skipping.
_SPARSE_DATASET_ID = "cmems_obs-ins_glo_phy-temp-sal_my_easycora_irr"


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_get_estimate_dispatches_to_get(
    _require_creds, orchestrator
) -> None:
    """Estimate for a get-shape request goes through ``_estimate_get``
    (``marine.get(dry_run=True)``) rather than ``_estimate_subset``.

    The dry-run path for ``marine.get`` does not always surface a
    precise size for sparse formats — that's the whole reason the
    confirmation gate's ``approximate`` branch always-fires. We
    accept both ``precise`` and ``approximate`` to avoid coupling
    the test to a specific SDK version; the unit suite already
    pins the gate behaviour for each branch (cr round-1 LOW
    flagged this; the SDK shape is what it is).
    """
    envelope = await orchestrator.run(
        backend="cmems",
        operation="estimate",
        params={"dataset_id": _SPARSE_DATASET_ID, "filter": "*1990*"},
    )
    payload = _ok(envelope)
    # Service field reports "n/a" for get (no service routing).
    assert payload.get("service_used") == "n/a", payload
    assert payload.get("epistemic_status") in {"precise", "approximate"}, payload
    assert "estimated_size_bytes" in payload, payload


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_get_files_index_file_download(
    _require_creds, orchestrator
) -> None:
    """Download a single small index file from EasyCORA via
    ``file_list`` — pinned by exact path so a layout shift fails
    loudly instead of silently skipping (cr round-1 HIGH).

    ``index_history.txt`` is the standard CORA-family index file
    (<1 MB) — every product in the family ships one at the root.
    Verifies the full happy path: ``marine.get`` invocation, bundle
    directory placement, manifest write, store_manifest cache
    registration, per-file descriptors with size + md5, workflow
    row ``successful``.

    If the index file path drifts in a future EasyCORA release
    this test will fail with a clear NotFoundError — the right
    signal for an integration test.
    """
    envelope = await orchestrator.run(
        backend="cmems",
        operation="get",
        params={
            "dataset_id": _SPARSE_DATASET_ID,
            "file_list": ["index_history.txt"],
        },
        options={"confirmed": True},
    )
    payload = _ok(envelope)
    assert payload.get("status") == "successful", payload
    assert payload.get("mode") == "offline", payload
    files = (payload.get("result") or {}).get("files") or []
    assert files, payload
    # Hard cap so a future product reshuffle that re-broadens the
    # filter doesn't silently start downloading hundreds of MB.
    total_bytes = sum(
        (d.get("metadata") or {}).get("size_bytes", 0) for d in files
    )
    assert total_bytes < 10 * 1024 * 1024, (
        f"bundle exceeded 10 MB cap: {total_bytes} bytes, {len(files)} files"
    )
    for desc in files:
        fp = desc.get("filepath")
        assert fp, desc
        assert Path(fp).is_file(), f"missing data file: {fp}"
        meta = desc.get("metadata") or {}
        assert meta.get("size_bytes", 0) > 0, desc
        assert meta.get("md5"), desc


@pytest.mark.timeout(300)
def test_cli_search_datasets_json(_require_creds) -> None:
    """T-033 Test 6: subprocess `copernicus-mcp marine search-datasets --json`."""
    cmd = [
        sys.executable,
        "-m",
        "copernicus_mcp",
        "marine",
        "search-datasets",
        "--keyword",
        "temperature",
        "--limit",
        "1",
        "--json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, (result.returncode, result.stderr)
    parsed = json.loads(result.stdout)
    # CLI unwraps the orchestrator envelope, so the top level is the
    # backend's search response shape: {datasets, total_count}.
    datasets = parsed.get("datasets")
    assert datasets is not None, parsed
    assert len(datasets) >= 1


def _tiny_subset_params() -> dict[str, Any]:
    """1° × 1° × 1-day single-variable subset (well under 10 MB)."""
    return {
        "dataset_id": _KNOWN_DATASET_ID,
        "variables": ["thetao"],
        "minimum_longitude": -1.0,
        "maximum_longitude": 0.0,
        "minimum_latitude": 45.0,
        "maximum_latitude": 46.0,
        "minimum_depth": 0.0,
        "maximum_depth": 5.0,
        "start_datetime": "2024-06-01T00:00:00Z",
        "end_datetime": "2024-06-01T23:59:59Z",
    }


# ---------------------------------------------------------------------------
# T-033 expansion: post-original-cut surface — marine_get_coordinates
# (T-022 second half), marine_list_files (T-CMEMS-GET-INDEX-004).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_get_coordinates_returns_axes(
    _require_creds, orchestrator
) -> None:
    """T-022 second half: ``marine_get_coordinates`` returns the
    dataset's axes. Long axes (time) come back as ``{start, end,
    count, stride_*}`` summaries; short axes (depth) come back as
    full lists. Asserts the canonical key set is present."""
    payload = _ok(
        await orchestrator.run(
            backend="cmems",
            operation="get_coordinates",
            params={"dataset_id": _KNOWN_DATASET_ID},
        )
    )
    # Spatio-temporal core axes are non-negotiable for any gridded
    # CMEMS product.
    for key in ("longitude", "latitude", "time"):
        assert key in payload, f"missing axis {key!r} in {list(payload.keys())}"
    time_axis = payload["time"]
    # Round-2 cr HIGH (both reviewers): the backend's
    # ``_AXIS_FULL_LIMIT_TIME = 5000`` only summarises when an axis
    # exceeds that count. The reference dataset has ~1.4k daily steps
    # (2022-06-01 → present), well under the threshold, so the backend
    # returns ``time`` as a list — NOT a summary dict.
    #
    # The right shape contract: ``list[str] | {start, end, count}``,
    # always non-empty. Branch the inner check rather than picking one
    # form (round-1's mistake).
    assert isinstance(time_axis, list | dict), (
        f"time axis expected as list or summary dict, got "
        f"{type(time_axis).__name__}"
    )
    if isinstance(time_axis, dict):
        for k in ("start", "end", "count"):
            assert k in time_axis, time_axis
        assert time_axis.get("count", 0) > 0, time_axis
    else:
        assert time_axis, "time axis came back empty"


@pytest.mark.asyncio
# Index fetch for EasyCORA can take ~210s on the SDK's first call
# (full file enumeration); the subsequent Parquet-cache read is sub-
# second. Generous timeout matches the SDK reality.
@pytest.mark.timeout(420)
async def test_list_files_fetches_index_for_sparse_dataset(
    _require_creds, orchestrator, tmp_path: Path
) -> None:
    """T-CDS-INDEX-004: ``marine_list_files`` on a sparse dataset
    triggers the index fetch path (first call), drops a Parquet file
    under ``marine_indices/`` in the cache, and returns a non-empty
    ``files`` list with per-file metadata.

    Tightly bounded: a narrow bbox + 30-day window keeps the matched
    file count small (<100) so the test runs quickly after the index
    is in place."""
    payload = _ok(
        await orchestrator.run(
            backend="cmems",
            operation="list_files",
            params={
                "dataset_id": _SPARSE_DATASET_ID,
                "bbox": (-10.0, 40.0, 10.0, 50.0),
                "time_range": ("1990-01-01T00:00:00Z", "1990-01-31T23:59:59Z"),
                "limit": 50,
            },
        )
    )
    files = payload.get("files") or []
    assert files, payload
    # File records carry path + per-axis bounds + time-range + variables —
    # verify shape rather than exact contents (snapshot drift is OK;
    # schema drift is the regression we'd want to catch). Round-1 cr
    # HIGH (codex): the row dict uses ``lon_min/lon_max/lat_min/lat_max``
    # and ``time_start/time_end`` (per ``_row_to_dict`` at
    # backends/cmems/backend.py:1291-1304), NOT the bbox/time_min/
    # time_max shape my initial draft assumed.
    sample = files[0]
    for key in (
        "file_path",
        "lon_min",
        "lon_max",
        "lat_min",
        "lat_max",
        "time_start",
        "time_end",
        # Round-2 cr LOW (codex): also pin the per-file metadata fields
        # the unit suite treats as canonical. Values can legitimately
        # be None (unknown size, unknown platform) — assert presence
        # of the key only.
        "platform_type",
        "variables",
        "size_bytes",
    ):
        assert key in sample, sample

    # Parquet cache landed under <cache>/marine_indices/<dataset>.parquet.
    indices_dir = tmp_path / "cache" / "marine_indices"
    assert indices_dir.is_dir(), f"marine_indices/ missing: {indices_dir}"
    parquets = list(indices_dir.glob("*.parquet"))
    assert parquets, f"no parquet index materialised: {indices_dir}"


