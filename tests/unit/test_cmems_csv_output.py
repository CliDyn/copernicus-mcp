"""T-TS-005: opt-in CSV output for CMEMS subsets.

``file_format="csv"`` retrieves via ``copernicusmarine.read_dataframe`` (subset
has no csv writer) and writes a ``.csv``; the default stays ``netcdf``. The
download branch lives in ``_download_to`` so it is unit-testable without the
full submit/staging/cache machinery; the large-data invariant holds (a file is
written and the tool returns a descriptor, never inline rows).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _valid(**over):
    base = dict(
        dataset_id="cmems_mod_bal_phy_my_P1D-m",
        variables=["thetao"],
        minimum_longitude=20.0,
        maximum_longitude=20.0,
        minimum_latitude=58.0,
        maximum_latitude=58.0,
        minimum_depth=0.0,
        maximum_depth=1.0,
        start_datetime="2014-01-01T00:00:00Z",
        end_datetime="2015-12-31T00:00:00Z",
    )
    base.update(over)
    return base


def test_file_format_csv_accepted() -> None:
    from copernicus_mcp.data_model.schemas_cmems import CmemsSubsetRequest

    req = CmemsSubsetRequest(**_valid(file_format="csv"))
    assert req.file_format == "csv"


def test_subset_kwargs_coerces_csv_to_netcdf_for_dryrun() -> None:
    """subset() has no csv writer, so the dry-run estimate must run as
    netcdf; the real csv download goes through read_dataframe instead."""
    from copernicus_mcp.backends.cmems.backend import _subset_kwargs
    from copernicus_mcp.data_model.schemas_cmems import CmemsSubsetRequest

    req = CmemsSubsetRequest(**_valid(file_format="csv"))
    assert _subset_kwargs(req, dry_run=True)["file_format"] == "netcdf"


class _FakeDF:
    def __init__(self) -> None:
        self.to_csv_path: str | None = None

    def to_csv(self, path, index: bool = True) -> None:  # noqa: FBT001, FBT002
        self.to_csv_path = str(path)
        Path(path).write_text("time,thetao\n2014-01-01,3.1\n", encoding="utf-8")


def test_download_to_csv_uses_read_dataframe(tmp_path) -> None:
    from copernicus_mcp.backends.cmems.backend import _download_to

    seen: dict[str, object] = {}
    df = _FakeDF()

    class FakeMarine:
        def subset(self, **k):
            seen["subset"] = True

        def read_dataframe(self, **k):
            seen["rdf_kwargs"] = k
            return df

    target = tmp_path / "x.csv"
    kwargs = {
        "dataset_id": "d",
        "variables": ["thetao"],
        "output_directory": "/x",
        "output_filename": "x.csv",
        "file_format": "netcdf",  # coerced upstream; must be stripped here
        "netcdf_compression_level": 1,
        "dry_run": False,
        "username": "u",
        "password": "p",
    }
    _download_to(FakeMarine(), kwargs, target, file_format="csv")

    assert target.exists()
    assert "subset" not in seen  # csv path must NOT call subset
    rdf = seen["rdf_kwargs"]
    assert rdf["variables"] == ["thetao"]
    # subset-only keys are stripped before read_dataframe
    for k in ("output_directory", "output_filename", "file_format", "dry_run"):
        assert k not in rdf


def test_download_to_netcdf_uses_subset(tmp_path) -> None:
    from copernicus_mcp.backends.cmems.backend import _download_to

    seen: dict[str, object] = {}

    class FakeMarine:
        def subset(self, **k):
            seen["subset_kwargs"] = k

        def read_dataframe(self, **k):
            seen["rdf"] = True

    _download_to(
        FakeMarine(),
        {"dataset_id": "d", "output_directory": str(tmp_path), "dry_run": False},
        tmp_path / "x.nc",
        file_format="netcdf",
    )
    assert "subset_kwargs" in seen
    assert "rdf" not in seen


def test_csv_requires_point_bbox() -> None:
    """Review M1: csv loads the whole subset into memory via read_dataframe,
    so restrict it to a single point — a large-area csv could OOM and bypass
    the netcdf-based size gate."""
    from copernicus_mcp.data_model.schemas_cmems import CmemsSubsetRequest
    from copernicus_mcp.errors import ValidationError

    with pytest.raises(ValidationError):
        CmemsSubsetRequest(
            **_valid(file_format="csv", minimum_longitude=10.0, maximum_longitude=20.0)
        )


def test_csv_point_accepted() -> None:
    from copernicus_mcp.data_model.schemas_cmems import CmemsSubsetRequest

    # _valid() is a degenerate point (lon 20==20, lat 58==58).
    assert CmemsSubsetRequest(**_valid(file_format="csv")).file_format == "csv"


def test_csv_cache_key_ignores_compression_level() -> None:
    """Review LOW: the csv path ignores netcdf_compression_level, so two csv
    point requests differing only by it must share a cache key (inv #6)."""
    from copernicus_mcp.data_model.coordinator import DataModelCoordinator
    from copernicus_mcp.data_model.schemas_cmems import CmemsSubsetRequest

    coord = DataModelCoordinator(persistence=None)  # cache_key is pure
    a = coord.cache_key_for_subset(
        CmemsSubsetRequest(**_valid(file_format="csv", netcdf_compression_level=1))
    )
    b = coord.cache_key_for_subset(
        CmemsSubsetRequest(**_valid(file_format="csv", netcdf_compression_level=9))
    )
    assert a == b
    # netcdf still distinguishes compression level.
    c = coord.cache_key_for_subset(
        CmemsSubsetRequest(**_valid(netcdf_compression_level=1))
    )
    d = coord.cache_key_for_subset(
        CmemsSubsetRequest(**_valid(netcdf_compression_level=9))
    )
    assert c != d
