"""T-TS-007: CDS timeseries discoverability (`location` in available_inputs),
csv descriptors, and a non-steering `inputs` description."""

from __future__ import annotations


def test_is_timeseries_product_by_suffix() -> None:
    from copernicus_mcp.backends.cds.catalogue import _is_timeseries_product

    assert _is_timeseries_product({"id": "reanalysis-era5-land-timeseries"}) is True
    assert _is_timeseries_product({"id": "reanalysis-era5-single-levels"}) is False


def test_is_timeseries_product_by_keyword() -> None:
    from copernicus_mcp.backends.cds.catalogue import _is_timeseries_product

    rec = {"id": "x", "keywords": ["Product type: Reanalysis", "Data type: Time-series"]}
    assert _is_timeseries_product(rec) is True


def test_describe_injects_location_for_era5_land_timeseries() -> None:
    """The product needs a nested ``location`` point, but the upstream
    machine-readable form omits it — we inject the shape so an agent can
    compose a valid request from ``available_inputs`` alone."""
    from copernicus_mcp.backends.cds import catalogue

    ai = catalogue.describe("reanalysis-era5-land-timeseries").get("available_inputs") or {}
    assert set(ai.get("location") or {}) == {"latitude", "longitude"}


def test_csv_data_format_yields_csv_filename() -> None:
    from copernicus_mcp.backends.cds.backend import _cds_target_filename

    assert _cds_target_filename("abc123", {"data_format": "csv"}).endswith(".csv")


def test_csv_extension_content_type() -> None:
    from copernicus_mcp.backends.cds.backend import _cds_content_type_for_extension

    assert _cds_content_type_for_extension("cds_abc123.csv") == "text/csv"


def test_cds_inputs_description_mentions_location() -> None:
    """The submit/estimate ``inputs`` field must not steer only to the gridded
    (year/month/day/area) shape — it must also describe the timeseries point."""
    from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest

    desc = (CdsRetrieveRequest.model_fields["inputs"].description or "").lower()
    assert "location" in desc
