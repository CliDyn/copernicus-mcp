"""T-CDS-EST2-003/004: calibration ``signature`` behaviour (decision 8)."""

from __future__ import annotations

from copernicus_mcp.backends.cds.calibration import signature


def test_signature_is_deterministic_key_order() -> None:
    a = signature({"experiment": "historical", "model": "x", "temporal_resolution": "monthly"})
    b = signature({"temporal_resolution": "monthly", "model": "x", "experiment": "historical"})
    assert a == b


def test_signature_scalar_and_single_list_match() -> None:
    assert signature({"variable": "xco2"}) == signature({"variable": ["xco2"]})


def test_signature_multi_variable_collapses_to_token() -> None:
    multi = signature({"variable": ["a", "b"]})
    single = signature({"variable": ["a"]})
    assert "__multi" in multi
    assert multi != single


def test_signature_level_count_distinguishes_depth() -> None:
    three = signature({"variable": ["t"], "level": ["1", "2", "3"]})
    seventeen = signature({"variable": ["t"], "level": [str(i) for i in range(17)]})
    assert three != seventeen


def test_signature_model_discriminates_cmip6() -> None:
    a = signature({"temporal_resolution": "monthly", "model": "mpi_esm1_2_lr"})
    b = signature({"temporal_resolution": "monthly", "model": "access_cm2"})
    assert a != b


def test_signature_system_version_discriminates_glofas() -> None:
    a = signature({"system_version": "version_4_0", "hydrological_model": "lisflood"})
    b = signature({"system_version": "version_3_0", "hydrological_model": "lisflood"})
    assert a != b


def test_signature_ignores_multiplicative_axes() -> None:
    # year/month/day/time/area are NOT in the signature — they live in cost.
    base = {"temporal_resolution": "daily", "variable": ["t"]}
    with_dates = {**base, "year": ["2020", "2021"], "month": ["01"], "area": [10, 0, 0, 10]}
    assert signature(base) == signature(with_dates)


# --- CalibrationLookup (decision 9) -------------------------------------

from copernicus_mcp.backends.cds.calibration import CalibrationLookup  # noqa: E402


def _obs(
    sig: str,
    size_bytes: int,
    cost_units: float | None,
    *,
    area_fraction: float = 1.0,
    observed_at: str = "2026-01-01T00:00:00Z",
) -> dict:
    return {
        "signature": sig,
        "size_bytes": size_bytes,
        "cost_units": cost_units,
        "area_fraction": area_fraction,
        "observed_at": observed_at,
    }


def test_lookup_single_observation_bytes_per_unit() -> None:
    lk = CalibrationLookup(observations=[_obs("s", 1000, 10)], seed={})
    r = lk.resolve("ds", "s")
    assert r is not None
    assert r.bytes_per_unit == 100.0  # 1000 / 10 / 1.0
    assert r.n_obs == 1


def test_lookup_area_normalised() -> None:
    lk = CalibrationLookup(observations=[_obs("s", 1000, 10, area_fraction=0.5)], seed={})
    r = lk.resolve("ds", "s")
    assert r is not None
    assert r.bytes_per_unit == 200.0  # 1000 / 10 / 0.5


def test_lookup_excludes_null_cost_rows() -> None:
    lk = CalibrationLookup(observations=[_obs("s", 1000, None)], seed={})
    assert lk.resolve("ds", "s") is None


def test_lookup_excludes_zero_cost_and_zero_area() -> None:
    lk = CalibrationLookup(
        observations=[_obs("s", 1000, 0.0), _obs("s", 1000, 10, area_fraction=0.0)],
        seed={},
    )
    assert lk.resolve("ds", "s") is None


def test_lookup_seed_only_when_no_local() -> None:
    lk = CalibrationLookup(
        observations=[], seed={("ds", "s"): {"bytes_per_unit": 500.0, "n_obs": 4}}
    )
    r = lk.resolve("ds", "s")
    assert r is not None
    assert r.bytes_per_unit == 500.0


def test_lookup_local_dominates_at_three_observations() -> None:
    obs = [
        _obs("s", 1000, 10, observed_at=f"2026-01-0{i}T00:00:00Z") for i in range(1, 4)
    ]
    lk = CalibrationLookup(
        observations=obs, seed={("ds", "s"): {"bytes_per_unit": 9999.0, "n_obs": 100}}
    )
    r = lk.resolve("ds", "s")
    assert r is not None
    assert r.bytes_per_unit == 100.0  # seed ignored once n_local >= 3


def test_lookup_blends_below_three_local() -> None:
    lk = CalibrationLookup(
        observations=[_obs("s", 1000, 10)],  # bpu 100, n_local 1
        seed={("ds", "s"): {"bytes_per_unit": 200.0, "n_obs": 9}},
    )
    r = lk.resolve("ds", "s")
    assert r is not None
    # (100*1 + 200*min(9,3)) / (1 + 3) = 700/4 = 175
    assert r.bytes_per_unit == 175.0


def test_lookup_dataset_median_fallback_for_unseen_signature() -> None:
    obs = [_obs("a", 1000, 10), _obs("b", 2000, 10), _obs("c", 3000, 10)]
    lk = CalibrationLookup(observations=obs, seed={})
    r = lk.resolve("ds", "unseen")
    assert r is not None
    assert r.bytes_per_unit == 200.0  # median of 100, 200, 300


def test_lookup_no_dataset_median_below_three() -> None:
    obs = [_obs("a", 1000, 10), _obs("b", 2000, 10)]  # only 2 eligible
    lk = CalibrationLookup(observations=obs, seed={})
    assert lk.resolve("ds", "unseen") is None


def test_lookup_returns_none_when_empty() -> None:
    assert CalibrationLookup(observations=[], seed={}).resolve("ds", "s") is None


# --- seed loader --------------------------------------------------------

from copernicus_mcp.backends.cds.calibration import load_seed  # noqa: E402


def test_load_seed_bundled_has_calibration_entries() -> None:
    """The shipped seed (T-CDS-EST2-005, from developer history) is populated;
    WP3 case A (XCO2 L3 OBS4MIPS) is calibrated out-of-the-box."""
    seed = load_seed()
    assert any(ds == "satellite-carbon-dioxide" for (ds, _sig) in seed)
    assert any(ds == "projections-cmip6" for (ds, _sig) in seed)


def test_load_seed_parses_entries(tmp_path) -> None:
    import json

    p = tmp_path / "seed.json"
    p.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-12T00:00:00Z",
                "entries": [
                    {
                        "dataset_id": "projections-cmip6",
                        "signature": "sig-x",
                        "bytes_per_unit": 80000.0,
                        "n_obs": 5,
                        "source": "history",
                    }
                ],
            }
        )
    )
    seed = load_seed(p)
    assert seed[("projections-cmip6", "sig-x")] == {"bytes_per_unit": 80000.0, "n_obs": 5}


def test_load_seed_missing_file_returns_empty(tmp_path) -> None:
    assert load_seed(tmp_path / "nope.json") == {}


def test_load_seed_corrupt_file_returns_empty(tmp_path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert load_seed(p) == {}
