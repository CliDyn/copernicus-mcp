"""T-CDS-EST2-005: refresh_cds_calibration.py core logic (no network)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent.parent / "scripts" / "refresh_cds_calibration.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("refresh_cds_calibration", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_M = _load_module()


def _prov(dataset_id: str, inputs: dict, size_bytes: int) -> str:
    return json.dumps(
        {
            "backend": {"id": "cds"},
            "request": {"user_request": {"dataset_id": dataset_id, "inputs": inputs}},
            "files": [{"size_bytes": size_bytes}],
        }
    )


def test_extract_observation_happy() -> None:
    out = _M._extract_observation(_prov("ds", {"variable": ["x"]}, 1000))
    assert out == ("ds", {"variable": ["x"]}, 1000)


def test_extract_observation_skips_non_cds() -> None:
    row = json.dumps({"backend": {"id": "cmems"}, "request": {"user_request": {}}})
    assert _M._extract_observation(row) is None


def test_extract_observation_skips_malformed() -> None:
    assert _M._extract_observation("{not json") is None
    assert _M._extract_observation(json.dumps({"backend": {"id": "cds"}})) is None


def test_build_seed_entries_aggregates_by_signature() -> None:
    rows = [
        _prov("ds", {"variable": ["x"], "temporal_resolution": "monthly"}, 2000),
        _prov("ds", {"variable": ["x"], "temporal_resolution": "monthly"}, 4000),
    ]
    # cost 10 units, area 1.0 → bytes_per_unit 200 and 400 → mean 300.
    entries = _M.build_seed_entries(rows, costing_fn=lambda d, i: 10.0)
    assert len(entries) == 1
    assert entries[0]["bytes_per_unit"] == 300.0
    assert entries[0]["n_obs"] == 2


def test_build_seed_entries_skips_when_costing_none() -> None:
    rows = [_prov("ds", {"variable": ["x"]}, 2000)]
    assert _M.build_seed_entries(rows, costing_fn=lambda d, i: None) == []


def test_build_seed_entries_area_normalises() -> None:
    rows = [_prov("ds", {"variable": ["x"], "area": [10.0, 0.0, 0.0, 10.0]}, 1000)]
    # area_fraction = (10*10)/(180*360) ≈ 0.001543; bpu = 1000/10/0.001543 ≈ 64800
    # (normalising a tiny-area observation up to the full-globe per-unit rate).
    entries = _M.build_seed_entries(rows, costing_fn=lambda d, i: 10.0)
    assert entries[0]["bytes_per_unit"] > 60_000


def test_write_seed_round_trip(tmp_path: Path) -> None:
    from copernicus_mcp.backends.cds.calibration import load_seed

    entries = [
        {"dataset_id": "ds", "signature": "sig", "bytes_per_unit": 300.0, "n_obs": 2, "source": "history"}
    ]
    seed_file = tmp_path / "seed.json"
    _M.write_seed(entries, path=seed_file, generated_at="2026-06-12T00:00:00Z")
    loaded = load_seed(seed_file)
    assert loaded[("ds", "sig")] == {"bytes_per_unit": 300.0, "n_obs": 2}


def test_signature_of_mode(tmp_path: Path, capsys) -> None:
    req = tmp_path / "req.json"
    req.write_text(json.dumps({"inputs": {"variable": ["x"], "model": "m"}}))
    rc = _M.main(["--signature-of", str(req), "--dataset-id", "ds"])
    assert rc == 0
    from copernicus_mcp.backends.cds.calibration import signature

    assert capsys.readouterr().out.strip() == signature({"variable": ["x"], "model": "m"})
