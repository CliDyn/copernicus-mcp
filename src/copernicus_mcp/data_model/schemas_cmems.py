"""Strongly-typed CMEMS request schemas (research §12.2.2).

Field-level numeric ranges use Pydantic ``Field`` constraints — these surface
as ``pydantic.ValidationError`` and are caught at the MCP boundary. Cross-field
invariants (``min <= max``, ``start < end``) raise our canonical
``copernicus_mcp.errors.ValidationError`` so the orchestrator can attach a
recovery hint.

# TODO(antimeridian): Iter 1 deliberately rejects bboxes whose intent is to
# wrap across the antimeridian (min_lon=170, max_lon=-170). The cross-field
# validator below produces a ValidationError with an explicit recovery hint
# pointing the user at "split into two non-crossing bboxes". Auto-splitting is
# out of scope (the project conventions invariant 7, plan T-011 step 3).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from copernicus_mcp.common.time import iso8601_utc
from copernicus_mcp.errors import ValidationError as CmcpValidationError
from copernicus_mcp.errors.records import build_error_record

# T-TS-006: accept date-only / naive datetimes on the CMEMS subset path and
# treat them as UTC. The global ``iso8601_utc`` stays strict (codex LOW-9) —
# this pre-normaliser only tags a tz onto naive input; ``iso8601_utc`` then does
# the canonical conversion and the strict error for genuinely bad strings.
_DATE_ONLY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NAIVE_DATETIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?")


def _ensure_utc_iso(value: str) -> str:
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if _DATE_ONLY_RE.fullmatch(raw):
        return f"{raw}T00:00:00Z"
    if _NAIVE_DATETIME_RE.fullmatch(raw):
        return raw.replace(" ", "T") + "Z"
    return raw

_CMEMS_FORBID_FROZEN = ConfigDict(extra="forbid", frozen=True)


class CmemsSearchRequest(BaseModel):
    """Discovery filter for ``marine_search_datasets`` (research §12.2.2)."""

    model_config = _CMEMS_FORBID_FROZEN

    keyword: str | None = None
    product_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    time_range: tuple[str, str] | None = None
    service_types: (
        list[
            Literal[
                "geoseries",
                "timeseries",
                "omi-arco",
                "static-arco",
                "platformseries",
            ]
        ]
        | None
    ) = None
    limit: int | None = Field(default=None, ge=1)
    # T-CMEMS-CAT-003a: opt-in hybrid search.
    # ``live=False`` (default) → offline read of the bundled catalogue
    # snapshot (fast, no credentials, may be days/weeks stale).
    # ``live=True``            → live ``copernicusmarine.describe()``
    # call against the Marine Service (fresh, requires credentials,
    # network round-trip ~10 s). Surfaced for callers who need
    # post-snapshot product updates or live-only metadata.
    live: bool = False


class CmemsDescribeRequest(BaseModel):
    """Metadata lookup for a single dataset."""

    model_config = _CMEMS_FORBID_FROZEN

    dataset_id: str
    dataset_version: str | None = None
    dataset_part: str | None = None


class CmemsSubsetRequest(BaseModel):
    """Spatio-temporal subset request for the ``subset`` toolbox call.

    Longitude is restricted to ``[-180, 180]`` for Iter 1: ``[0, 360]`` and
    antimeridian-crossing bboxes are out of scope (plan T-011 step 3).
    """

    model_config = _CMEMS_FORBID_FROZEN

    dataset_id: str = Field(min_length=1)
    dataset_version: str | None = None
    dataset_part: str | None = None

    variables: list[str]

    @field_validator("dataset_id")
    @classmethod
    def _reject_blank_dataset_id(cls, v: str) -> str:
        # codex T-030 round 3 MEDIUM: ``min_length=1`` accepts whitespace-only;
        # downstream validators don't strip. Reject pure whitespace at the
        # input boundary so Pydantic produces a clear field-path error.
        if not v.strip():
            raise ValueError("dataset_id must not be blank or whitespace-only")
        return v

    minimum_longitude: float = Field(ge=-180.0, le=180.0)
    maximum_longitude: float = Field(ge=-180.0, le=180.0)
    minimum_latitude: float = Field(ge=-90.0, le=90.0)
    maximum_latitude: float = Field(ge=-90.0, le=90.0)
    minimum_depth: float | None = Field(default=None, ge=0.0)
    maximum_depth: float | None = Field(default=None, ge=0.0)

    start_datetime: str
    end_datetime: str

    coordinates_selection_method: Literal["inside", "strict-inside", "nearest", "outside"] = (
        "inside"
    )
    service: str | None = None

    file_format: Literal["netcdf", "zarr", "csv"] = "netcdf"
    netcdf_compression_level: int = Field(default=1, ge=0, le=9)

    @model_validator(mode="after")
    def _normalise_and_check(self) -> CmemsSubsetRequest:
        # Datetime normalisation (also rejects naive inputs).
        start_norm = iso8601_utc(_ensure_utc_iso(self.start_datetime))
        end_norm = iso8601_utc(_ensure_utc_iso(self.end_datetime))

        if start_norm >= end_norm:
            raise CmcpValidationError(
                f"start_datetime ({start_norm}) must be strictly earlier than "
                f"end_datetime ({end_norm})"
            )

        # Latitude order.
        if self.minimum_latitude > self.maximum_latitude:
            raise CmcpValidationError(
                f"minimum_latitude ({self.minimum_latitude}) must be <= "
                f"maximum_latitude ({self.maximum_latitude})"
            )

        # Longitude order — distinguish antimeridian-shaped requests.
        if self.minimum_longitude > self.maximum_longitude:
            looks_antimeridian = self.minimum_longitude > 0.0 and self.maximum_longitude < 0.0
            if looks_antimeridian:
                msg = (
                    "antimeridian-crossing bboxes are not supported in "
                    "Iteration 1; split your request into two non-crossing "
                    "bboxes (one ending at 180, one starting at -180)"
                )
                raise CmcpValidationError(
                    msg,
                    record=build_error_record(
                        "ValidationError",
                        message=msg,
                        recovery_action="modify_request_parameters",
                        next_action_hint=(
                            "Split into two non-crossing bboxes: "
                            f"[{self.minimum_longitude}, 180] and "
                            f"[-180, {self.maximum_longitude}]."
                        ),
                    ),
                )
            raise CmcpValidationError(
                f"minimum_longitude ({self.minimum_longitude}) must be <= "
                f"maximum_longitude ({self.maximum_longitude})"
            )

        # Depth order — both bounds are optional (T-TS-006); only compare when
        # both are present.
        if (
            self.minimum_depth is not None
            and self.maximum_depth is not None
            and self.minimum_depth > self.maximum_depth
        ):
            raise CmcpValidationError(
                f"minimum_depth ({self.minimum_depth}) must be <= "
                f"maximum_depth ({self.maximum_depth})"
            )

        # Review M1: csv loads the whole subset into memory via
        # ``read_dataframe`` — restrict it to a single point so a large area
        # cannot OOM or slip under the netcdf-based size gate.
        if self.file_format == "csv" and not (
            self.minimum_longitude == self.maximum_longitude
            and self.minimum_latitude == self.maximum_latitude
        ):
            raise CmcpValidationError(
                "file_format='csv' is supported only for a single point",
                record=build_error_record(
                    "ValidationError",
                    message=(
                        "file_format='csv' is supported only for a single point "
                        "(set min==max longitude and latitude); use 'netcdf' for "
                        "an area"
                    ),
                    recovery_action="modify_request_parameters",
                    next_action_hint=(
                        "Set minimum_longitude==maximum_longitude and "
                        "minimum_latitude==maximum_latitude, or use "
                        "file_format='netcdf'."
                    ),
                ),
            )

        # Persist normalised datetimes so cache keys are deterministic.
        if start_norm != self.start_datetime or end_norm != self.end_datetime:
            object.__setattr__(self, "start_datetime", start_norm)
            object.__setattr__(self, "end_datetime", end_norm)
        return self


class CmemsGetRequest(BaseModel):
    """Native-files retrieval (toolbox ``get``).

    T-CMEMS-GET-001: reconciled with the v1 placeholder. Hard
    requirements (codex spec-review):

    - ``dataset_id`` is required, non-blank.
    - ``filter`` / ``regex`` / ``file_list`` are mutually
      exclusive; setting two raises the canonical
      ``ValidationError`` so a router can attach a recovery hint.
      All three unset is fine (downloads whatever the toolbox
      defaults to for the dataset).
    - Empty selection values (``filter=""``, ``regex=""``,
      ``file_list=[]``) are rejected at the input boundary.
    - ``dataset_version`` mirrors ``CmemsSubsetRequest`` so a
      caller can pin a non-default version.
    - ``sync`` / ``skip_existing`` / ``overwrite`` are forwarded
      to the SDK unchanged.

    Bbox / time_range live in the Layer 2 sub-plan
    (``T-CMEMS-GET-INDEX``) and are intentionally NOT fields
    here.
    """

    model_config = _CMEMS_FORBID_FROZEN

    dataset_id: str = Field(min_length=1)
    dataset_version: str | None = None
    # cr round-2 MEDIUM on PR #96: ``dataset_part`` is shared with
    # ``CmemsSubsetRequest`` — the upstream ``copernicusmarine.get``
    # signature accepts it with the same "pin a non-default product
    # part" semantics. Because it now lives on both schemas, it is
    # NOT a subset-discriminator (codex round-3 LOW).
    dataset_part: str | None = None
    filter: str | None = Field(default=None, min_length=1)
    regex: str | None = Field(default=None, min_length=1)
    file_list: list[str] | None = Field(default=None, min_length=1)
    sync: bool | None = None
    skip_existing: bool | None = None
    overwrite: bool | None = None

    @field_validator("dataset_id")
    @classmethod
    def _reject_blank_dataset_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("dataset_id must not be blank or whitespace-only")
        return v

    @field_validator("filter", "regex")
    @classmethod
    def _reject_blank_selector(cls, v: str | None) -> str | None:
        # cr+codex round-1 MEDIUM on PR #96: ``min_length=1``
        # checks raw length, not stripped length — ``filter="   "``
        # would otherwise reach the SDK as a malformed glob.
        if v is not None and not v.strip():
            raise ValueError("must not be blank or whitespace-only")
        return v

    @field_validator("file_list")
    @classmethod
    def _reject_blank_file_list_items(cls, v: list[str] | None) -> list[str] | None:
        # cr+codex round-1 MEDIUM on PR #96: the outer
        # ``min_length=1`` only catches ``file_list=[]``; an empty
        # or whitespace-only entry would otherwise reach the SDK.
        if v is None:
            return v
        for i, item in enumerate(v):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"file_list[{i}] must be a non-blank string; got {item!r}")
        return v

    @model_validator(mode="after")
    def _selection_filters_mutually_exclusive(self) -> CmemsGetRequest:
        set_filters = [
            name
            for name, value in (
                ("filter", self.filter),
                ("regex", self.regex),
                ("file_list", self.file_list),
            )
            if value is not None
        ]
        if len(set_filters) > 1:
            raise CmcpValidationError(
                f"CmemsGetRequest: filter / regex / file_list are mutually "
                f"exclusive — got {set_filters}. Pass at most one."
            )
        return self


class CmemsGetCoordinatesRequest(BaseModel):
    """Layer 1 ``marine_get_coordinates`` input (T-022 second half).

    Returns the dataset's four canonical coordinate axes (lon/lat/depth/
    time) without downloading any data. Each axis comes back as either a
    full list of values (short axes: ≤10000 for spatial/depth, ≤5000 for
    time) or a summary dict ``{start, end, count, stride[_seconds]}``
    (longer axes). ``depth`` is ``[]`` for surface-only datasets.
    Useful for an agent that wants to know the dataset's actual extent
    before composing a subset request.
    """

    model_config = _CMEMS_FORBID_FROZEN

    dataset_id: str = Field(min_length=1)
    dataset_version: str | None = None
    service: str | None = None

    @field_validator("dataset_id")
    @classmethod
    def _reject_blank_dataset_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("dataset_id must not be blank or whitespace-only")
        return v


class CmemsListFilesRequest(BaseModel):
    """Layer 2 ``marine_list_files`` input (T-CMEMS-GET-INDEX-004).

    All filters are optional and AND-combined. ``time_range`` endpoints
    are normalised to canonical UTC via ``iso8601_utc``; an inverted
    range (start > end) raises ``ValidationError`` at the input boundary.

    **Antimeridian carve-out (decisions #10 + #18):** unlike
    ``CmemsSubsetRequest`` (which rejects antimeridian-crossing bboxes
    because the SDK cannot split the request), this schema accepts
    them. The offline index filter operates on pre-parsed per-row
    bboxes in pure Python and the intersection is unambiguous. The
    resulting ``file_list`` is downloadable via ``marine_get_files``
    but is NOT a valid input for ``marine_subset_dataset``.
    """

    model_config = _CMEMS_FORBID_FROZEN

    dataset_id: str = Field(min_length=1)
    bbox: tuple[float, float, float, float] | None = None
    time_range: tuple[str, str] | None = None
    variables: list[str] | None = None
    platform_types: list[str] | None = None
    limit: int | None = Field(default=None, ge=1)

    @field_validator("dataset_id")
    @classmethod
    def _reject_blank_dataset_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("dataset_id must not be blank or whitespace-only")
        return v

    @field_validator("bbox")
    @classmethod
    def _check_bbox_ranges(
        cls, v: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if v is None:
            return v
        lon_min, lat_min, lon_max, lat_max = v
        if not -180.0 <= lon_min <= 180.0:
            raise ValueError("bbox lon_min must be in [-180, 180]")
        if not -180.0 <= lon_max <= 180.0:
            raise ValueError("bbox lon_max must be in [-180, 180]")
        if not -90.0 <= lat_min <= 90.0:
            raise ValueError("bbox lat_min must be in [-90, 90]")
        if not -90.0 <= lat_max <= 90.0:
            raise ValueError("bbox lat_max must be in [-90, 90]")
        if lat_min > lat_max:
            raise ValueError("bbox lat_min must be <= lat_max")
        return v

    @model_validator(mode="after")
    def _normalise_time_range(self) -> CmemsListFilesRequest:
        if self.time_range is None:
            return self
        start_norm = iso8601_utc(self.time_range[0])
        end_norm = iso8601_utc(self.time_range[1])
        if start_norm >= end_norm:
            raise CmcpValidationError(
                "time_range start must be strictly before end",
                record=build_error_record(
                    "ValidationError",
                    message="time_range start must be strictly before end",
                    recovery_action="modify_request_parameters",
                ),
            )
        object.__setattr__(self, "time_range", (start_norm, end_norm))
        return self
