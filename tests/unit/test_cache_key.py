from __future__ import annotations

import subprocess
import sys
import textwrap

from copernicus_mcp.data_model.cache_key import construct_cache_key


def _base_kwargs() -> dict:
    return dict(
        backend="cmems",
        operation="submit",
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        dataset_version="202411",
        api_version="marine-data-store-v1",
        schema_version="v1.0",
        spatial={"min_lon": -10.0, "max_lon": 5.0, "min_lat": 35.0, "max_lat": 45.0},
        temporal={"start": "2024-01-01T00:00:00Z", "end": "2024-01-02T00:00:00Z"},
        variables=["thetao"],
        format_options={"file_format": "netcdf", "netcdf_compression_level": 1},
    )


def test_format_prefix() -> None:
    key = construct_cache_key(**_base_kwargs())
    assert key.startswith(
        "cmems:submit:cmems_mod_glo_phy_anfc_0.083deg_P1D-m:"
    )
    suffix = key.rsplit(":", 1)[-1]
    assert len(suffix) == 16
    assert all(c in "0123456789abcdef" for c in suffix)


def test_same_input_same_key() -> None:
    a = construct_cache_key(**_base_kwargs())
    b = construct_cache_key(**_base_kwargs())
    assert a == b


def test_variables_reorder_invariant() -> None:
    kw = _base_kwargs()
    kw["variables"] = ["thetao", "salinity"]
    a = construct_cache_key(**kw)
    kw["variables"] = ["salinity", "thetao"]
    b = construct_cache_key(**kw)
    assert a == b


def test_dataset_version_changes_key() -> None:
    kw = _base_kwargs()
    a = construct_cache_key(**kw)
    kw["dataset_version"] = "202412"
    b = construct_cache_key(**kw)
    assert a != b


def test_sensitive_keys_in_format_options_ignored() -> None:
    kw = _base_kwargs()
    a = construct_cache_key(**kw)
    kw["format_options"] = {
        "file_format": "netcdf",
        "netcdf_compression_level": 1,
        "password": "leak-me",
        "token": "abc",
        "username": "ivan",
        "client_secret": "x",
        "authorization": "Bearer xyz",
    }
    b = construct_cache_key(**kw)
    assert a == b


def test_sensitive_keys_filtered_recursively() -> None:
    kw = _base_kwargs()
    a = construct_cache_key(**kw)
    kw["format_options"] = {
        "file_format": "netcdf",
        "netcdf_compression_level": 1,
        "nested": {"password": "leak", "actual": "data"},
    }
    b = construct_cache_key(**kw)
    # `actual` IS semantic, so b != a; but if we drop `actual` too, same as base
    # only the nested password should be filtered
    assert a != b
    kw["format_options"] = {
        "file_format": "netcdf",
        "netcdf_compression_level": 1,
        "nested": {"actual": "data"},
    }
    c = construct_cache_key(**kw)
    assert b == c


def test_float_precision_distinguishes_microbox() -> None:
    """6-decimal precision: -10.000001 vs -10.0 must produce different keys."""
    kw = _base_kwargs()
    a = construct_cache_key(**kw)
    kw["spatial"] = {**kw["spatial"], "min_lon": -10.000001}
    b = construct_cache_key(**kw)
    assert a != b


def test_none_components_omitted() -> None:
    kw = _base_kwargs()
    kw["dataset_version"] = None
    kw["spatial"] = None
    kw["temporal"] = None
    kw["variables"] = None
    kw["format_options"] = None
    # Should not raise; should still produce a deterministic key.
    key = construct_cache_key(**kw)
    assert key.startswith("cmems:submit:")


def test_cross_process_determinism() -> None:
    """Spawn three separate Pythons; all must produce the same key."""
    script = textwrap.dedent(
        """
        from copernicus_mcp.data_model.cache_key import construct_cache_key
        key = construct_cache_key(
            backend="cmems",
            operation="submit",
            dataset_id="ds-1",
            dataset_version="v1",
            api_version="marine-data-store-v1",
            schema_version="v1.0",
            spatial={"min_lon": -10.0, "max_lon": 5.0},
            temporal={"start": "2024-01-01T00:00:00Z"},
            variables=["b", "a"],
            format_options={"format": "netcdf"},
        )
        print(key)
        """
    ).strip()
    keys = set()
    for _ in range(3):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )
        keys.add(result.stdout.strip())
    assert len(keys) == 1
