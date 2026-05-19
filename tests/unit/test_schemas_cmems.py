from __future__ import annotations

import pytest
from pydantic import ValidationError as PydValidationError

from copernicus_mcp.data_model.schemas_cmems import (
    CmemsDescribeRequest,
    CmemsGetCoordinatesRequest,
    CmemsGetRequest,
    CmemsListFilesRequest,
    CmemsSearchRequest,
    CmemsSubsetRequest,
)
from copernicus_mcp.errors import ValidationError


def _valid_subset_kwargs() -> dict:
    return dict(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        variables=["thetao"],
        minimum_longitude=-10.0,
        maximum_longitude=10.0,
        minimum_latitude=30.0,
        maximum_latitude=45.0,
        minimum_depth=0.0,
        maximum_depth=100.0,
        start_datetime="2024-01-01T00:00:00Z",
        end_datetime="2024-01-02T00:00:00Z",
    )


def test_subset_valid_construction() -> None:
    req = CmemsSubsetRequest(**_valid_subset_kwargs())
    assert req.dataset_id.startswith("cmems_")
    assert req.variables == ["thetao"]
    # datetimes normalised
    assert req.start_datetime.endswith("Z")


def test_subset_datetime_offset_normalised() -> None:
    kw = _valid_subset_kwargs()
    kw["start_datetime"] = "2024-01-01T02:00:00+02:00"
    req = CmemsSubsetRequest(**kw)
    assert req.start_datetime == "2024-01-01T00:00:00Z"


def test_subset_antimeridian_rejected_with_hint() -> None:
    kw = _valid_subset_kwargs()
    kw["minimum_longitude"] = 170.0
    kw["maximum_longitude"] = -170.0
    with pytest.raises(ValidationError) as exc_info:
        CmemsSubsetRequest(**kw)
    msg = str(exc_info.value).lower()
    assert "antimeridian" in msg
    record = exc_info.value.error_record
    assert record.recovery_action == "modify_request_parameters"
    assert record.next_action_hint is not None
    assert "split" in record.next_action_hint.lower()


def test_subset_min_lon_gt_max_lon_non_antimeridian() -> None:
    kw = _valid_subset_kwargs()
    kw["minimum_longitude"] = 20.0
    kw["maximum_longitude"] = 10.0
    with pytest.raises(ValidationError):
        CmemsSubsetRequest(**kw)


def test_subset_start_after_end_rejected() -> None:
    kw = _valid_subset_kwargs()
    kw["start_datetime"] = "2024-02-01T00:00:00Z"
    kw["end_datetime"] = "2024-01-01T00:00:00Z"
    with pytest.raises(ValidationError):
        CmemsSubsetRequest(**kw)


def test_subset_start_equals_end_rejected() -> None:
    kw = _valid_subset_kwargs()
    kw["start_datetime"] = "2024-01-01T00:00:00Z"
    kw["end_datetime"] = "2024-01-01T00:00:00Z"
    with pytest.raises(ValidationError):
        CmemsSubsetRequest(**kw)


def test_subset_min_depth_gt_max_depth_rejected() -> None:
    kw = _valid_subset_kwargs()
    kw["minimum_depth"] = 500.0
    kw["maximum_depth"] = 100.0
    with pytest.raises(ValidationError):
        CmemsSubsetRequest(**kw)


def test_subset_lat_out_of_range() -> None:
    kw = _valid_subset_kwargs()
    kw["minimum_latitude"] = -91.0
    with pytest.raises(PydValidationError):
        CmemsSubsetRequest(**kw)


def test_subset_lon_out_of_range() -> None:
    kw = _valid_subset_kwargs()
    kw["minimum_longitude"] = -181.0
    with pytest.raises(PydValidationError):
        CmemsSubsetRequest(**kw)


def test_subset_negative_depth_rejected() -> None:
    kw = _valid_subset_kwargs()
    kw["minimum_depth"] = -1.0
    with pytest.raises(PydValidationError):
        CmemsSubsetRequest(**kw)


def test_subset_naive_datetime_rejected() -> None:
    kw = _valid_subset_kwargs()
    kw["start_datetime"] = "2024-01-01T00:00:00"
    with pytest.raises(ValidationError):
        CmemsSubsetRequest(**kw)


def test_subset_extra_field_forbidden() -> None:
    kw = _valid_subset_kwargs()
    kw["surprise"] = "boom"
    with pytest.raises(PydValidationError):
        CmemsSubsetRequest(**kw)


def test_subset_default_file_format() -> None:
    req = CmemsSubsetRequest(**_valid_subset_kwargs())
    assert req.file_format == "netcdf"


def test_search_all_optional() -> None:
    CmemsSearchRequest()
    CmemsSearchRequest(keyword="temperature", limit=5)


def test_describe_dataset_id_required() -> None:
    with pytest.raises(PydValidationError):
        CmemsDescribeRequest()  # type: ignore[call-arg]
    CmemsDescribeRequest(dataset_id="x")


# ---------------------------------------------------------------------------
# T-CMEMS-GET-001: CmemsGetRequest schema reconciliation
# ---------------------------------------------------------------------------


def test_get_dataset_id_is_required() -> None:
    """The v1 schema accepted ``CmemsGetRequest()`` with no
    fields. The reconciled schema (T-CMEMS-GET-001) makes
    ``dataset_id`` required and non-empty so a malformed call
    fails fast instead of silently reaching the SDK."""
    with pytest.raises(PydValidationError):
        CmemsGetRequest()  # type: ignore[call-arg]
    CmemsGetRequest(dataset_id="cmems_obs-ins_glo_phy-temp-sal_my_easycora_irr")


@pytest.mark.parametrize("blank", ["", "   ", "\t", "\n"])
def test_get_rejects_blank_dataset_id(blank: str) -> None:
    """Whitespace-only ``dataset_id`` is just as bad as empty —
    reject both."""
    with pytest.raises(PydValidationError):
        CmemsGetRequest(dataset_id=blank)


def test_get_accepts_dataset_version() -> None:
    """``dataset_version`` is optional; mirrors ``CmemsSubsetRequest``."""
    req = CmemsGetRequest(dataset_id="x", dataset_version="202411")
    assert req.dataset_version == "202411"


def test_get_accepts_at_most_one_selection_filter() -> None:
    """``filter`` / ``regex`` / ``file_list`` are mutually
    exclusive — one of them, or none (download everything).
    Setting two is a validation error."""
    # Each alone is fine.
    CmemsGetRequest(dataset_id="x", filter="*1990*")
    CmemsGetRequest(dataset_id="x", regex=".*1990.*")
    CmemsGetRequest(dataset_id="x", file_list=["a.nc", "b.nc"])
    # None set is fine (downloads everything the toolbox allows).
    CmemsGetRequest(dataset_id="x")
    # Two set raises.
    with pytest.raises(ValidationError):
        CmemsGetRequest(dataset_id="x", filter="*1990*", regex=".*1991.*")
    with pytest.raises(ValidationError):
        CmemsGetRequest(dataset_id="x", filter="*1990*", file_list=["a.nc"])
    with pytest.raises(ValidationError):
        CmemsGetRequest(dataset_id="x", regex=".*1990.*", file_list=["a.nc"])


def test_get_rejects_empty_filter_string() -> None:
    """``filter=""`` is meaningless (the SDK would match nothing or
    everything depending on the day); reject up front."""
    with pytest.raises(PydValidationError):
        CmemsGetRequest(dataset_id="x", filter="")


def test_get_rejects_empty_regex_string() -> None:
    with pytest.raises(PydValidationError):
        CmemsGetRequest(dataset_id="x", regex="")


def test_get_rejects_empty_file_list() -> None:
    """``file_list=[]`` is "no files" — same shape error as
    ``filter=""``."""
    with pytest.raises(PydValidationError):
        CmemsGetRequest(dataset_id="x", file_list=[])


@pytest.mark.parametrize("blank", ["   ", "\t", "\n"])
def test_get_rejects_whitespace_only_filter(blank: str) -> None:
    """cr+codex round-1 MEDIUM: ``min_length=1`` accepts pure
    whitespace; mirrors the dataset_id rule and rejects."""
    with pytest.raises(PydValidationError):
        CmemsGetRequest(dataset_id="x", filter=blank)


@pytest.mark.parametrize("blank", ["   ", "\t", "\n"])
def test_get_rejects_whitespace_only_regex(blank: str) -> None:
    with pytest.raises(PydValidationError):
        CmemsGetRequest(dataset_id="x", regex=blank)


def test_get_rejects_blank_items_in_file_list() -> None:
    """cr+codex round-1 MEDIUM: ``file_list=[""]`` and
    ``file_list=["   "]`` pass the outer length check but are
    semantically empty. Reject each item."""
    with pytest.raises(PydValidationError):
        CmemsGetRequest(dataset_id="x", file_list=[""])
    with pytest.raises(PydValidationError):
        CmemsGetRequest(dataset_id="x", file_list=["   "])
    with pytest.raises(PydValidationError):
        CmemsGetRequest(dataset_id="x", file_list=["good.nc", ""])


def test_get_model_dump_round_trip() -> None:
    """cr round-1 LOW: a valid request → model_dump → re-validate
    survives. Locks the round-trip for T-CMEMS-GET-002's
    cache_key construction."""
    req = CmemsGetRequest(dataset_id="x", filter="*1990*")
    dumped = req.model_dump()
    rebuilt = CmemsGetRequest(**{k: v for k, v in dumped.items() if v is not None})
    assert rebuilt.dataset_id == req.dataset_id
    assert rebuilt.filter == req.filter


def test_get_keeps_sync_skip_existing_overwrite_flags() -> None:
    """Forwarded-to-SDK flags survive the reconciliation."""
    req = CmemsGetRequest(
        dataset_id="x",
        sync=True,
        skip_existing=False,
        overwrite=True,
    )
    assert req.sync is True
    assert req.skip_existing is False
    assert req.overwrite is True


# ---------------------------------------------------------------------------
# CmemsListFilesRequest (T-CMEMS-GET-INDEX-004)
# ---------------------------------------------------------------------------


def test_list_files_dataset_id_required() -> None:
    with pytest.raises(PydValidationError):
        CmemsListFilesRequest()  # type: ignore[call-arg]


def test_list_files_all_filters_optional() -> None:
    req = CmemsListFilesRequest(dataset_id="cmems_obs-ins_glo_bgc-car_my_socat-obs_irr")
    assert req.bbox is None
    assert req.time_range is None
    assert req.variables is None
    assert req.platform_types is None
    assert req.limit is None


def test_list_files_blank_dataset_id_rejected() -> None:
    with pytest.raises(PydValidationError):
        CmemsListFilesRequest(dataset_id="")
    with pytest.raises(PydValidationError):
        CmemsListFilesRequest(dataset_id="   ")


def test_list_files_accepts_antimeridian_bbox() -> None:
    # Decision #10/#18: antimeridian-crossing user bbox is accepted here
    # because offline index filtering supports it.
    req = CmemsListFilesRequest(
        dataset_id="x",
        bbox=(170.0, -10.0, -170.0, 10.0),
    )
    assert req.bbox == (170.0, -10.0, -170.0, 10.0)


def test_list_files_bbox_lat_out_of_range_rejected() -> None:
    with pytest.raises(PydValidationError):
        CmemsListFilesRequest(dataset_id="x", bbox=(-10.0, -91.0, 10.0, 10.0))
    with pytest.raises(PydValidationError):
        CmemsListFilesRequest(dataset_id="x", bbox=(-10.0, -10.0, 10.0, 91.0))


def test_list_files_bbox_lon_out_of_range_rejected() -> None:
    with pytest.raises(PydValidationError):
        CmemsListFilesRequest(dataset_id="x", bbox=(-181.0, -10.0, 10.0, 10.0))
    with pytest.raises(PydValidationError):
        CmemsListFilesRequest(dataset_id="x", bbox=(-10.0, -10.0, 181.0, 10.0))


def test_list_files_time_range_normalises_offset_to_utc() -> None:
    req = CmemsListFilesRequest(
        dataset_id="x",
        time_range=("2010-01-01T02:00:00+02:00", "2010-12-31T02:00:00+02:00"),
    )
    assert req.time_range == ("2010-01-01T00:00:00Z", "2010-12-31T00:00:00Z")


def test_list_files_time_range_inverted_rejected() -> None:
    with pytest.raises(ValidationError):
        CmemsListFilesRequest(
            dataset_id="x",
            time_range=("2011-01-01T00:00:00Z", "2010-01-01T00:00:00Z"),
        )


def test_list_files_time_range_naive_rejected() -> None:
    with pytest.raises(ValidationError):
        CmemsListFilesRequest(
            dataset_id="x",
            time_range=("2010-01-01T00:00:00", "2010-12-31T23:59:59"),
        )


def test_list_files_limit_must_be_positive() -> None:
    with pytest.raises(PydValidationError):
        CmemsListFilesRequest(dataset_id="x", limit=0)
    with pytest.raises(PydValidationError):
        CmemsListFilesRequest(dataset_id="x", limit=-1)


def test_list_files_extra_fields_rejected() -> None:
    with pytest.raises(PydValidationError):
        CmemsListFilesRequest(dataset_id="x", unknown_field="boom")


def test_list_files_is_frozen() -> None:
    req = CmemsListFilesRequest(dataset_id="x")
    with pytest.raises(PydValidationError):
        req.dataset_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CmemsGetCoordinatesRequest (T-022 second half)
# ---------------------------------------------------------------------------


def test_get_coordinates_dataset_id_required() -> None:
    with pytest.raises(PydValidationError):
        CmemsGetCoordinatesRequest()  # type: ignore[call-arg]


def test_get_coordinates_minimal() -> None:
    req = CmemsGetCoordinatesRequest(dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m")
    assert req.dataset_version is None
    assert req.service is None


def test_get_coordinates_blank_dataset_id_rejected() -> None:
    with pytest.raises(PydValidationError):
        CmemsGetCoordinatesRequest(dataset_id="")
    with pytest.raises(PydValidationError):
        CmemsGetCoordinatesRequest(dataset_id="   ")


def test_get_coordinates_extra_fields_rejected() -> None:
    with pytest.raises(PydValidationError):
        CmemsGetCoordinatesRequest(dataset_id="x", unknown_field="boom")


def test_get_coordinates_is_frozen() -> None:
    req = CmemsGetCoordinatesRequest(dataset_id="x")
    with pytest.raises(PydValidationError):
        req.dataset_id = "other"  # type: ignore[misc]


def test_get_coordinates_optional_version_and_service() -> None:
    req = CmemsGetCoordinatesRequest(
        dataset_id="x",
        dataset_version="202411",
        service="geoseries",
    )
    assert req.dataset_version == "202411"
    assert req.service == "geoseries"
