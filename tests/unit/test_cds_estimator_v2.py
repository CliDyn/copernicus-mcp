"""T-CDS-EST2-001: pure estimator envelope v2 (costing-aware, nullable size).

Tests the pure ``estimate`` function directly with an injected ``CostingResult``
(or ``None``) — no network, no backend. ``costing=None`` must reproduce the v1
envelope (backward-compatible) plus new optional fields.
"""

from __future__ import annotations

from copernicus_mcp.backends.cds.costing import CostingResult
from copernicus_mcp.backends.cds.estimator import estimate

_ERA5_SL = "reanalysis-era5-single-levels"
_ERA5_SL_BYTES = 2_100_000  # curated bytes/field

# A whole-file-shaped request: no list-valued multiplicative axis -> count_fields == 1.
_WHOLE_FILE_INPUTS = {
    "processing_level": "level_3",
    "variable": "xco2",
    "sensor_and_algorithm": "merged_obs4mips",
    "version": "4_5",
}
# A gridded request with a 24-value time axis -> count_fields == 24.
_GRIDDED_INPUTS = {
    "product_type": ["reanalysis"],
    "variable": ["2m_temperature"],
    "year": ["2024"],
    "month": ["01"],
    "day": ["01"],
    "time": [f"{h:02d}:00" for h in range(24)],
}


# --- costing None: v1-compatible envelope + new optional fields ----------


def test_costing_none_uncalibrated_is_unknown() -> None:
    """v2 honesty: no costing + no local calibration → size unknown (no guess);
    fields_count still surfaced."""
    out = estimate(_ERA5_SL, _GRIDDED_INPUTS, costing=None, calibration=None)
    assert out["estimated_size_bytes"] is None
    assert out["estimated_size_human"] is None
    assert out["epistemic_status"] == "unknown"
    assert out["cost"] is None
    assert out["fields_count"] == 24


def test_size_estimate_caveat_always_present() -> None:
    """The agent must always be told, very clearly (not vaguely), that the byte
    size is approximate and must not be relied on — present whether a size is
    shown or unknown."""
    out = estimate(_ERA5_SL, _GRIDDED_INPUTS, costing=None, calibration=None)
    caveat = out["size_estimate_caveat"]
    assert "APPROXIMATE" in caveat
    assert "do not rely" in caveat.lower()


def test_costing_none_unknown_dataset_is_unknown() -> None:
    out = estimate("mystery-dataset", _GRIDDED_INPUTS, costing=None, calibration=None)
    assert out["epistemic_status"] == "unknown"
    assert out["estimated_size_bytes"] is None
    assert out["cost"] is None


# --- costing available: cost block present, size still unknown if uncalibrated


def test_costing_available_emits_cost_block_size_unknown() -> None:
    costing = CostingResult(units=24.0, limit=121000.0)
    out = estimate(_ERA5_SL, _GRIDDED_INPUTS, costing=costing, calibration=None)
    assert out["cost"] == {
        "units": 24.0,
        "limit": 121000.0,
        "exceeds_limit": False,
        "source": "costing_api",
    }
    # The cost units are exact; the byte size is unknown without local calibration.
    assert out["estimated_size_bytes"] is None
    assert out["epistemic_status"] == "unknown"


def test_costing_exceeds_limit_reflected_in_cost_block() -> None:
    costing = CostingResult(units=1827.0, limit=400.0)
    out = estimate(
        "derived-era5-single-levels-daily-statistics",
        {"variable": ["2m_temperature"], "year": ["2020"]},
        costing=costing,
        calibration=None,
    )
    assert out["cost"]["exceeds_limit"] is True


# --- demotion (decision 4): cost==1 AND count_fields==1 AND no calibration


def test_whole_file_demoted_to_unknown() -> None:
    costing = CostingResult(units=1.0, limit=10000.0)
    out = estimate("satellite-carbon-dioxide", _WHOLE_FILE_INPUTS, costing=costing, calibration=None)
    assert out["estimated_size_bytes"] is None
    assert out["estimated_size_human"] is None
    assert out["epistemic_status"] == "unknown"
    assert out["cost"]["units"] == 1.0


def test_curated_whole_file_also_demoted() -> None:
    # CORDEX is in the curated map, but cost==1 + count_fields==1 still demotes.
    costing = CostingResult(units=1.0, limit=1000000.0)
    out = estimate(
        "projections-cordex-domains-single-levels",
        {"domain": "europe", "experiment": "rcp_4_5", "variable": "2m_air_temperature"},
        costing=costing,
        calibration=None,
    )
    assert out["estimated_size_bytes"] is None
    assert out["epistemic_status"] == "unknown"


def test_cost_one_multiple_fields_uncalibrated_is_unknown() -> None:
    # v2: field count no longer rescues an uncalibrated estimate — unknown.
    costing = CostingResult(units=1.0, limit=10000.0)
    out = estimate(
        _ERA5_SL,
        {"variable": ["2m_temperature", "10m_wind_speed"], "year": ["2024"], "month": ["01"], "day": ["01"], "time": ["00:00"]},
        costing=costing,
        calibration=None,
    )
    assert out["estimated_size_bytes"] is None
    assert out["epistemic_status"] == "unknown"


def test_cost_gt_one_single_field_uncalibrated_is_unknown() -> None:
    # v2: a curated dataset with cost>1 is still unknown without local calibration.
    costing = CostingResult(units=5.0, limit=10000.0)
    out = estimate(_ERA5_SL, _WHOLE_FILE_INPUTS, costing=costing, calibration=None)
    assert out["estimated_size_bytes"] is None
    assert out["epistemic_status"] == "unknown"


# --- queue tier derives from cost.units when costing is available --------


def test_tier_from_cost_units_when_available() -> None:
    # 50_000 units -> heavy, regardless of the tiny local field count.
    costing = CostingResult(units=50_000.0, limit=1_000_000.0)
    out = estimate(_ERA5_SL, {"variable": ["x"]}, costing=costing, calibration=None)
    assert out["queue_latency_tier"] == "heavy"


def test_tier_from_fields_when_costing_none() -> None:
    out = estimate(_ERA5_SL, _GRIDDED_INPUTS, costing=None, calibration=None)
    assert out["queue_latency_tier"] == "light"  # 24 fields < 100


# --- calibration (T-CDS-EST2-004) ---------------------------------------

from copernicus_mcp.backends.cds.calibration import (  # noqa: E402
    CalibrationLookup,
    signature,
)


def _lookup(dataset_inputs: dict, size_bytes: int, cost_units: float, **kw):
    """A CalibrationLookup with one observation matching dataset_inputs' shape."""
    sig = signature(dataset_inputs)
    obs = {
        "signature": sig,
        "size_bytes": size_bytes,
        "cost_units": cost_units,
        "area_fraction": 1.0,
        "observed_at": "2026-01-01T00:00:00Z",
    }
    return CalibrationLookup(observations=[obs], seed=kw.get("seed", {}))


def test_calibrated_status_and_size() -> None:
    costing = CostingResult(units=24.0, limit=121000.0)
    cal = _lookup(_GRIDDED_INPUTS, size_bytes=24_000, cost_units=24.0)  # bpu=1000
    out = estimate(_ERA5_SL, _GRIDDED_INPUTS, costing=costing, calibration=cal)
    assert out["epistemic_status"] == "calibrated"
    assert out["estimated_size_bytes"] == 24_000  # 24 units × 1000 bpu × 1.0
    assert out["calibration_observations"] == 1


def test_calibration_miss_is_unknown() -> None:
    costing = CostingResult(units=24.0, limit=121000.0)
    # Lookup holds a DIFFERENT signature → resolve miss → no local data → unknown.
    cal = _lookup({"variable": ["unrelated"], "temporal_resolution": "x"}, 1, 1.0)
    out = estimate(_ERA5_SL, _GRIDDED_INPUTS, costing=costing, calibration=cal)
    assert out["epistemic_status"] == "unknown"
    assert out["estimated_size_bytes"] is None


def test_field_case_a_calibrated_to_actual_size() -> None:
    """XCO2 whole-file: one observation of 62.7 MB at cost 1 → estimate == 62.7
    MB, and the cost==1 demotion is overridden by the calibration entry."""
    costing = CostingResult(units=1.0, limit=10000.0)
    cal = _lookup(_WHOLE_FILE_INPUTS, size_bytes=62_751_933, cost_units=1.0)
    out = estimate("satellite-carbon-dioxide", _WHOLE_FILE_INPUTS, costing=costing, calibration=cal)
    assert out["epistemic_status"] == "calibrated"
    assert out["estimated_size_bytes"] == 62_751_933


def test_field_case_b_cordex_unknown_when_uncalibrated() -> None:
    """CORDEX whole-file (cost==1, single field): no confident 3.8 MB — honest
    'unknown' until calibrated (vs the old curated-4MB under-estimate)."""
    costing = CostingResult(units=1.0, limit=1_000_000.0)
    out = estimate(
        "projections-cordex-domains-single-levels",
        {"domain": "europe", "experiment": "rcp_4_5", "variable": "2m_air_temperature"},
        costing=costing,
        calibration=None,
    )
    assert out["estimated_size_bytes"] is None
    assert out["epistemic_status"] == "unknown"


def test_field_case_d_cost_exceeds_limit_flagged() -> None:
    """Daily-stats 5yr: costing exceeds the limit; the estimate surfaces it so
    submit can reject pre-flight (full submit-path test in test_cds_submit_*)."""
    costing = CostingResult(units=1827.0, limit=400.0)
    out = estimate(
        "derived-era5-single-levels-daily-statistics",
        {"variable": ["2m_temperature"], "year": ["2020"]},
        costing=costing,
        calibration=None,
    )
    assert out["cost"]["exceeds_limit"] is True


def test_field_case_c_cmip6_seed_only_is_approximate() -> None:
    """A seed-only estimate (no LOCAL observations) is now SHOWN but clearly marked
    ``approximate`` — the clean global-only seed gives a rough out-of-the-box number,
    and the loud ``size_estimate_caveat`` warns it can be off by a large factor. A
    later local download of this shape upgrades it to ``calibrated``."""
    cmip6_inputs = {
        "temporal_resolution": "monthly",
        "experiment": "historical",
        "variable": "precipitation",
        "model": "mpi_esm1_2_lr",
    }
    sig = signature(cmip6_inputs)
    cal = CalibrationLookup(
        observations=[], seed={("projections-cmip6", sig): {"bytes_per_unit": 80_000.0, "n_obs": 5}}
    )
    costing = CostingResult(units=180.0, limit=1_000_000.0)
    out = estimate("projections-cmip6", cmip6_inputs, costing=costing, calibration=cal)
    assert out["epistemic_status"] == "approximate"
    assert out["estimated_size_bytes"] == 14_400_000  # 180 units × 80_000 bpu × 1.0
    assert out["calibration_observations"] == 0
    assert "size_estimate_caveat" in out


def test_cross_signature_median_stays_unknown() -> None:
    """A cross-signature dataset-median (≥3 obs of OTHER shapes, target shape unseen,
    no seed) is too weak to show — it stays ``unknown`` (no estimate ⇒ say unknown),
    kept only as a weak blending prior, never displayed as a number."""
    other = [
        {
            "signature": signature({"variable": [f"v{i}"]}),
            "size_bytes": 1_000_000,
            "cost_units": 1.0,
            "area_fraction": 1.0,
            "observed_at": f"2026-01-0{i}T00:00:00Z",
        }
        for i in (1, 2, 3)
    ]
    cal = CalibrationLookup(observations=other, seed={})
    costing = CostingResult(units=10.0, limit=1_000_000.0)
    out = estimate(_ERA5_SL, _GRIDDED_INPUTS, costing=costing, calibration=cal)
    assert out["epistemic_status"] == "unknown"
    assert out["estimated_size_bytes"] is None
