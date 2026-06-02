"""T-TS-006: CMEMS subset input ergonomics — datetime normalisation and
optional depth.

Naive / date-only datetimes are accepted and normalised to UTC (a CMEMS-local
relaxation; the global ``common.time.iso8601_utc`` stays strict — codex LOW-9).
Depth bounds become optional so a surface request need not invent them.
"""

from __future__ import annotations

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


def test_date_only_datetime_normalised() -> None:
    from copernicus_mcp.data_model.schemas_cmems import CmemsSubsetRequest

    req = CmemsSubsetRequest(
        **_valid(start_datetime="2023-06-01", end_datetime="2023-06-08")
    )
    assert req.start_datetime == "2023-06-01T00:00:00Z"
    assert req.end_datetime == "2023-06-08T00:00:00Z"


def test_naive_datetime_assumed_utc() -> None:
    from copernicus_mcp.data_model.schemas_cmems import CmemsSubsetRequest

    req = CmemsSubsetRequest(
        **_valid(
            start_datetime="2023-06-01T06:00:00", end_datetime="2023-06-01T18:00:00"
        )
    )
    assert req.start_datetime == "2023-06-01T06:00:00Z"
    assert req.end_datetime == "2023-06-01T18:00:00Z"


def test_tz_aware_datetime_unchanged() -> None:
    from copernicus_mcp.data_model.schemas_cmems import CmemsSubsetRequest

    req = CmemsSubsetRequest(
        **_valid(
            start_datetime="2024-01-01T02:00:00+02:00",
            end_datetime="2024-01-02T00:00:00Z",
        )
    )
    assert req.start_datetime == "2024-01-01T00:00:00Z"


def test_depth_optional_when_omitted() -> None:
    from copernicus_mcp.data_model.schemas_cmems import CmemsSubsetRequest

    kw = _valid()
    kw.pop("minimum_depth")
    kw.pop("maximum_depth")
    req = CmemsSubsetRequest(**kw)
    assert req.minimum_depth is None
    assert req.maximum_depth is None


def test_subset_kwargs_omits_depth_when_none() -> None:
    from copernicus_mcp.backends.cmems.backend import _subset_kwargs
    from copernicus_mcp.data_model.schemas_cmems import CmemsSubsetRequest

    kw = _valid()
    kw.pop("minimum_depth")
    kw.pop("maximum_depth")
    req = CmemsSubsetRequest(**kw)
    sk = _subset_kwargs(req, dry_run=False)
    assert "minimum_depth" not in sk
    assert "maximum_depth" not in sk


def test_global_iso8601_utc_still_rejects_naive() -> None:
    """codex LOW-9: the CMEMS relaxation must NOT leak into the global helper."""
    from copernicus_mcp.common.time import iso8601_utc
    from copernicus_mcp.errors import ValidationError

    with pytest.raises(ValidationError):
        iso8601_utc("2023-06-01T00:00:00")  # naive — still rejected globally
