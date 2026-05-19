"""``CdsBackend.estimate`` tests (T-CDS-004).

 heuristic
estimate from ``N_fields × bytes_per_field``, where ``N_fields`` is the
product of cardinalities of cdsapi-canonical list fields (``year``,
``month``, ``day``, ``time``, ``variable``, ``pressure_level`` …).

The CDS engine has no ``apply_constraints`` API in the legacy ``cdsapi``
client (research §6.7.4 option 2), and using a real submit as a sizing
probe (option 3) abuses the queue. Heuristic stays empirical-only;
``epistemic_status`` is always ``"approximate"``.
"""

from __future__ import annotations

from pathlib import Path

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


def _backend(foundation):
    from copernicus_mcp.backends.cds.backend import CdsBackend

    return CdsBackend(foundation=foundation, credentials=None)


def _single_day_request() -> dict[str, object]:
    """1 variable × 1 year × 1 month × 1 day × 1 time = 1 field."""
    return {
        "dataset_id": "reanalysis-era5-single-levels",
        "inputs": {
            "variable": ["2m_temperature"],
            "year": ["2024"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00"],
            "data_format": "grib",
        },
    }


def _multi_day_request() -> dict[str, object]:
    """2 variables × 1 year × 1 month × 31 days × 24 hours = 1488 fields."""
    return {
        "dataset_id": "reanalysis-era5-single-levels",
        "inputs": {
            "variable": ["2m_temperature", "10m_u_component_of_wind"],
            "year": ["2024"],
            "month": ["01"],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "data_format": "grib",
        },
    }


# ---------------------------------------------------------------------------
# Canonical envelope shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_returns_canonical_envelope(foundation) -> None:
    """Mirrors CMEMS estimate envelope: type, estimated_size_bytes,
    estimated_size_human, epistemic_status, advisory_message."""
    backend = _backend(foundation)
    out = await backend.estimate(_single_day_request())
    assert out["type"] == "free"
    assert isinstance(out["estimated_size_bytes"], int)
    assert isinstance(out["estimated_size_human"], str)
    # T-CDS-011.6: epistemic_status is now per-dataset; ERA5 is curated.
    assert out["epistemic_status"] in {
        "curated_approximate",
        "default_heuristic",
    }
    assert isinstance(out["advisory_message"], str)


@pytest.mark.asyncio
async def test_estimate_includes_cds_specific_fields(foundation) -> None:
    """Beyond the CMEMS-shared shape, CDS estimate exposes the
    fields-count (debugging) and the queue-latency tier (research §6.7.4)."""
    backend = _backend(foundation)
    out = await backend.estimate(_single_day_request())
    assert isinstance(out["fields_count"], int)
    assert out["fields_count"] >= 1
    assert out["queue_latency_tier"] in ("light", "medium", "heavy")


# ---------------------------------------------------------------------------
# Heuristic correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_scales_with_cardinality(foundation) -> None:
    """N_fields = product of cardinalities of canonical list fields."""
    backend = _backend(foundation)
    small = await backend.estimate(_single_day_request())
    big = await backend.estimate(_multi_day_request())
    assert big["fields_count"] > small["fields_count"]
    assert big["estimated_size_bytes"] > small["estimated_size_bytes"]
    # 2 vars × 31 days × 24 hours = 1488; small was 1.
    assert big["fields_count"] == 1488


@pytest.mark.asyncio
async def test_estimate_handles_pressure_level_dimension(foundation) -> None:
    """``pressure_level`` is a multiplicative dimension on ERA5 PL data."""
    backend = _backend(foundation)
    base = {
        "dataset_id": "reanalysis-era5-pressure-levels",
        "inputs": {
            "variable": ["temperature"],
            "year": ["2024"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00"],
            "data_format": "grib",
        },
    }
    out_one_level = await backend.estimate(base)
    base["inputs"]["pressure_level"] = [  # type: ignore[index]
        "500", "700", "850", "1000",
    ]
    out_four_levels = await backend.estimate(base)
    assert out_four_levels["fields_count"] == 4 * out_one_level["fields_count"]


@pytest.mark.asyncio
async def test_estimate_area_subset_reduces_size(foundation) -> None:
    """``area: [N, W, S, E]`` (research §6.9.2) restricts spatial coverage,
    which reduces download size proportionally."""
    backend = _backend(foundation)
    global_req = _single_day_request()
    out_global = await backend.estimate(global_req)

    europe_req = _single_day_request()
    europe_req["inputs"]["area"] = [60.0, -10.0, 35.0, 40.0]  # type: ignore[index]
    out_europe = await backend.estimate(europe_req)
    # Europe area is much smaller than global — bytes proportionally smaller.
    assert out_europe["estimated_size_bytes"] < out_global["estimated_size_bytes"]
    # Field count is unchanged — area only scales bytes per field.
    assert out_europe["fields_count"] == out_global["fields_count"]


@pytest.mark.asyncio
async def test_estimate_unknown_dataset_uses_default_bytes_per_field(
    foundation,
) -> None:
    """Datasets not in the bytes-per-field map fall back to a conservative
    default (overestimate is safer for the confirmation gate)."""
    backend = _backend(foundation)
    out = await backend.estimate({
        "dataset_id": "made-up-dataset-not-in-the-map",
        "inputs": {
            "variable": ["x"],
            "year": ["2024"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00"],
        },
    })
    # Unknown dataset still produces a usable estimate.
    assert out["estimated_size_bytes"] > 0
    # Advisory mentions the fallback so the user knows the estimate is rougher.
    assert "default" in out["advisory_message"].lower() or \
           "unknown" in out["advisory_message"].lower()


# ---------------------------------------------------------------------------
# Queue-latency tier
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_tier_light_for_tiny_request(foundation) -> None:
    """1 field is well below the light threshold (100)."""
    backend = _backend(foundation)
    out = await backend.estimate(_single_day_request())
    assert out["queue_latency_tier"] == "light"
    assert out["fields_count"] < 100


@pytest.mark.asyncio
async def test_estimate_tier_heavy_for_huge_request(foundation) -> None:
    """A multi-year, all-vars, all-pressure-levels request is HEAVY."""
    backend = _backend(foundation)
    huge = {
        "dataset_id": "reanalysis-era5-pressure-levels",
        "inputs": {
            "variable": [
                "temperature", "u_component_of_wind", "v_component_of_wind",
                "specific_humidity", "geopotential",
            ],
            "year": [str(y) for y in range(2000, 2025)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "pressure_level": [str(p) for p in [
                10, 50, 100, 200, 300, 500, 700, 850, 925, 1000,
            ]],
            "data_format": "grib",
        },
    }
    out = await backend.estimate(huge)
    assert out["queue_latency_tier"] == "heavy"
    assert out["fields_count"] > 10_000


@pytest.mark.asyncio
async def test_estimate_tier_uses_field_count_not_area_scaled_bytes(
    foundation,
) -> None:
    """Codex T-CDS-004 MEDIUM: queue tier should reflect field count
    (server pulls fields independently from tape) not area-scaled
    download bytes. A tiny-area request with thousands of fields still
    queues long even though bytes are small."""
    backend = _backend(foundation)
    # Tiny area (0.1° × 0.1°) with many fields: 31 days × 24 hours × 5 vars
    # = 3720 fields. By cardinality semantics this is medium-tier;
    # by area-scaled bytes it would be ~tens of KB, formerly "light".
    out = await backend.estimate({
        "dataset_id": "reanalysis-era5-single-levels",
        "inputs": {
            "variable": ["t2m", "u10", "v10", "sp", "tp"],
            "year": ["2024"],
            "month": ["01"],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [0.1, 0.0, 0.0, 0.1],
        },
    })
    assert out["fields_count"] == 5 * 31 * 24
    assert out["queue_latency_tier"] in ("medium", "heavy"), out


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_strips_options_magic_key(foundation) -> None:
    """Orchestrator ``__options`` must be stripped before Pydantic
    ``extra=forbid`` rejects."""
    backend = _backend(foundation)
    params = _single_day_request()
    params["__options"] = {"confirmed": True}
    out = await backend.estimate(params)
    assert out["estimated_size_bytes"] > 0


@pytest.mark.asyncio
async def test_estimate_invalid_request_raises_validation_error(
    foundation,
) -> None:
    """Schema rejection surfaces as a structured ValidationError."""
    from copernicus_mcp.errors import ValidationError

    backend = _backend(foundation)
    with pytest.raises(ValidationError):
        await backend.estimate({"dataset_id": "", "inputs": {}})


@pytest.mark.asyncio
async def test_estimate_handles_scalar_string_fields(foundation) -> None:
    """MARS-style requests use slash-delimited STRINGS instead of lists
    (research §6.9.5: ``time: "00/06/12/18"``). Heuristic must not crash;
    a non-list field counts as cardinality 1 (we cannot expand MARS
    syntax without a full parser, accept the underestimate for these
    legacy datasets)."""
    backend = _backend(foundation)
    out = await backend.estimate({
        "dataset_id": "reanalysis-era5-complete",
        "inputs": {
            "date": "2013-01-01",
            "levelist": "1/10/100/137",
            "levtype": "ml",
            "param": "129/130/131",
            "stream": "oper",
            "time": "00/06/12/18",
            "type": "an",
            "grid": "1.0/1.0",
        },
    })
    assert out["estimated_size_bytes"] > 0
    assert out["fields_count"] >= 1


@pytest.mark.asyncio
async def test_estimate_result_is_sanitised(foundation) -> None:
    """Defence-in-depth: estimate output passes through Sanitiser."""
    backend = _backend(foundation)
    out = await backend.estimate(_single_day_request())
    flat = repr(out)
    assert "password=" not in flat
    assert "Bearer " not in flat


# ---------------------------------------------------------------------------
# Pure helper tests (heuristic logic in isolation)
# ---------------------------------------------------------------------------


def test_count_fields_product_of_list_cardinalities() -> None:
    from copernicus_mcp.backends.cds.estimator import count_fields

    # 2 × 1 × 1 × 31 × 24 = 1488
    n = count_fields({
        "variable": ["a", "b"],
        "year": ["2024"],
        "month": ["01"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
    })
    assert n == 1488


def test_count_fields_ignores_non_cardinality_fields() -> None:
    """``data_format``, ``download_format``, ``area``, ``grid`` are not
    multiplicative — they're scalars or transforms, not field counters."""
    from copernicus_mcp.backends.cds.estimator import count_fields

    n = count_fields({
        "variable": ["a"],
        "year": ["2024"],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [60, -10, 35, 40],
        "grid": [0.25, 0.25],
    })
    assert n == 1


def test_count_fields_handles_empty_inputs() -> None:
    from copernicus_mcp.backends.cds.estimator import count_fields

    # No cardinality fields → cardinality 1 (degenerate but consistent).
    n = count_fields({"data_format": "grib"})
    assert n == 1


def test_area_fraction_global_returns_one() -> None:
    from copernicus_mcp.backends.cds.estimator import area_fraction

    # No area key → global.
    assert area_fraction({}) == 1.0


def test_area_fraction_europe_is_small() -> None:
    """Europe roughly N=60, W=-10, S=35, E=40 → 25° lat × 50° lon = 1250 deg²
    out of 64800 deg² global. ~1.9% of global."""
    from copernicus_mcp.backends.cds.estimator import area_fraction

    frac = area_fraction({"area": [60, -10, 35, 40]})
    assert 0.01 < frac < 0.03


def test_area_fraction_handles_nan() -> None:
    """Codex/code-reviewer T-CDS-004 MEDIUM: NaN in any area coordinate
    used to crash with ``ValueError: cannot convert float NaN to integer``
    when the bytes estimate cast to int. Now treated as malformed →
    fall back to global (1.0). Caller never sees a Python traceback."""
    from copernicus_mcp.backends.cds.estimator import area_fraction

    assert area_fraction({"area": [float("nan"), 0.0, 0.0, 1.0]}) == 1.0
    assert area_fraction({"area": [60.0, -10.0, 35.0, float("nan")]}) == 1.0


def test_area_fraction_handles_infinity() -> None:
    from copernicus_mcp.backends.cds.estimator import area_fraction

    assert area_fraction({"area": [float("inf"), 0.0, 0.0, 0.0]}) == 1.0
    assert area_fraction({"area": [0.0, float("-inf"), 0.0, 0.0]}) == 1.0


def test_area_fraction_antimeridian_overestimates_safely() -> None:
    """Documented behaviour: antimeridian-crossing bbox computes
    ``|E-W|`` literally (340° instead of 20°) which OVERestimates
    coverage. This is safe for the confirmation gate (errs toward
    asking the user) — pin the direction in a test."""
    from copernicus_mcp.backends.cds.estimator import area_fraction

    # antimeridian: N=70, W=170, S=60, E=-170 — naive |E-W|=340°
    naive_frac = area_fraction({"area": [70.0, 170.0, 60.0, -170.0]})
    # True coverage would be 10° × 20° / (180×360) ≈ 0.31%; the naive
    # computation gives 10° × 340° / (180×360) ≈ 5.2%. Verify naive
    # output stays in the expected bracket — overestimate, not crash.
    assert 0.04 < naive_frac < 0.07


def test_area_fraction_inverted_returns_zero() -> None:
    from copernicus_mcp.backends.cds.estimator import area_fraction

    # N=S=0, W=E=0 → degenerate.
    assert area_fraction({"area": [0.0, 0.0, 0.0, 0.0]}) == 0.0


def test_count_fields_date_is_multiplicative() -> None:
    """Codex T-CDS-004 MEDIUM: ``date`` field used to be missing from
    ``_CARDINALITY_FIELDS`` so a 336-element date list silently
    bypassed the cardinality calculation, leading to gross
    underestimates. Pin the fix: list-valued ``date`` is a multiplier;
    scalar ``date`` (MARS string form) still counts as 1 because we
    don't ship a slash/range parser."""
    from copernicus_mcp.backends.cds.estimator import count_fields

    # List form: 14 days × 4 times = 56 fields.
    n = count_fields({
        "variable": ["t2m"],
        "date": [f"2024-01-{d:02d}" for d in range(1, 15)],
        "time": [f"{h:02d}:00" for h in (0, 6, 12, 18)],
    })
    assert n == 14 * 4

    # Scalar form (MARS-shape) — counts as 1 (parser-free underestimate).
    n_mars = count_fields({
        "variable": ["t"],
        "date": "2024-01-01/2024-01-14",
        "time": "00/06/12/18",
    })
    assert n_mars == 1


def test_count_fields_empty_list_value_counts_as_one() -> None:
    """``len(value) or 1`` collapses empty list → cardinality 1.
    Acceptable: an empty cardinality field would error server-side, but
    locally we don't want to silently produce zero-byte estimates that
    would skip the confirmation gate."""
    from copernicus_mcp.backends.cds.estimator import count_fields

    n = count_fields({"variable": [], "year": ["2024"], "month": ["01"]})
    assert n == 1


def test_estimate_returns_exact_bytes_for_known_dataset() -> None:
    """Code-reviewer T-CDS-004 LOW: pin the formula. ERA5-SL is
    2_100_000 bytes/field × 1 field × 1.0 area = 2_100_000."""
    from copernicus_mcp.backends.cds.estimator import estimate

    out = estimate(
        "reanalysis-era5-single-levels",
        {
            "variable": ["t2m"],
            "year": ["2024"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00"],
        },
    )
    assert out["estimated_size_bytes"] == 2_100_000
    assert out["fields_count"] == 1


def test_estimate_does_not_crash_on_nan_area() -> None:
    """End-to-end: NaN in area used to bubble up as ValueError from
    the int() cast. Now the area_fraction guard returns 1.0, the
    estimate stays usable, no Python traceback at the boundary."""
    from copernicus_mcp.backends.cds.estimator import estimate

    out = estimate(
        "reanalysis-era5-single-levels",
        {
            "variable": ["t2m"],
            "year": ["2024"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00"],
            "area": [float("nan"), 0.0, 0.0, 1.0],
        },
    )
    # Treated as global, so full bytes-per-field estimate.
    assert out["estimated_size_bytes"] == 2_100_000


# ---------------------------------------------------------------------------
# T-CDS-011.2 + .6 — epistemic_status split + runtime_compatible flag.
# Pre-11.x estimator emitted a uniform ``epistemic_status="approximate"``
# regardless of dataset and only advisory_message text disambiguated
# curated vs default-heuristic. Promote to a structured signal so LLM
# clients can branch on confidence (and on runtime support, post-11.1).
# ---------------------------------------------------------------------------


def test_estimate_curated_dataset_marks_curated_approximate() -> None:
    """ERA5 single-levels is in the curated bytes-per-field map → the
    estimate carries ``epistemic_status="curated_approximate"``."""
    from copernicus_mcp.backends.cds.estimator import estimate

    out = estimate(
        "reanalysis-era5-single-levels",
        {"variable": ["t2m"], "year": ["2024"], "month": ["01"],
         "day": ["01"], "time": ["00:00"]},
    )
    assert out["epistemic_status"] == "curated_approximate"


def test_estimate_unknown_dataset_marks_default_heuristic() -> None:
    """A dataset id not in the curated map falls back to default
    heuristic (±10×). Caller can branch on this signal rather than
    parsing the advisory_message string."""
    from copernicus_mcp.backends.cds.estimator import estimate

    out = estimate(
        "not-a-real-dataset-xyzzy",
        {"variable": ["x"], "year": ["2024"], "month": ["01"],
         "day": ["01"], "time": ["00:00"]},
    )
    assert out["epistemic_status"] == "default_heuristic"


def test_estimate_runtime_compatible_when_dataset_in_catalogue() -> None:
    """Datasets in the bundled CDS / ADS / EWDS catalogue snapshot are
    fully supported by the runtime after T-CDS-011.1. Estimate sets
    ``runtime_compatible=true`` so LLM clients can submit confidently
    instead of guessing from the dataset id prefix."""
    from copernicus_mcp.backends.cds.estimator import estimate

    out = estimate(
        "reanalysis-era5-single-levels",
        {"variable": ["t2m"], "year": ["2024"], "month": ["01"],
         "day": ["01"], "time": ["00:00"]},
    )
    assert out["runtime_compatible"] is True


def test_estimate_runtime_compatible_for_ads_dataset() -> None:
    """ADS datasets (CAMS) route through T-CDS-011.1 — fully supported."""
    from copernicus_mcp.backends.cds.estimator import estimate

    out = estimate(
        "cams-global-reanalysis-eac4",
        {"variable": ["total_column_ozone"],
         "date": ["2024-01-01/2024-01-01"], "time": ["00:00"]},
    )
    assert out["runtime_compatible"] is True


def test_estimate_runtime_incompatible_for_unknown_dataset() -> None:
    """Unknown dataset id (not in any of the three catalogue snapshots)
    is flagged ``runtime_compatible=false`` so an LLM agent can WARN
    the user before paying the submit token cost on a request that
    will likely 404."""
    from copernicus_mcp.backends.cds.estimator import estimate

    out = estimate(
        "not-a-real-dataset-xyzzy",
        {"variable": ["x"], "year": ["2024"], "month": ["01"],
         "day": ["01"], "time": ["00:00"]},
    )
    assert out["runtime_compatible"] is False
