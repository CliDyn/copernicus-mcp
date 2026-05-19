"""CDS request schemas (T-CDS-002).

 the per-dataset
constraint catalogue is server-side; static validation is limited to
**structural** checks. ``CdsRetrieveRequest`` accepts any cdsapi-shaped
``inputs`` dict (modern structured form OR legacy MARS-keywords form,
research §6.9.5) but enforces:

- non-empty ``dataset_id``
- non-empty ``inputs`` dict with non-empty string keys
- only primitive values (``str | int | float | bool``) or flat lists
  thereof — no nested dicts (cdsapi requires a flat request) and no
  ``None`` values (cdsapi rejects them server-side)

Semantic correctness (does ``year=2024+month=02+day=30`` exist? does
this dataset support ``pressure_level``?) is NOT checked at this
layer. T-CDS-005 surfaces server-side error envelopes through the
canonical error taxonomy.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_CDS_FORBID_FROZEN = ConfigDict(extra="forbid", frozen=True)

# Primitive value types allowed at leaves of the ``inputs`` dict.
# ``bool`` is a subclass of ``int`` in Python; we list it explicitly for
# isinstance clarity even though ``isinstance(True, int)`` is True.
_ALLOWED_LEAF_TYPES: tuple[type, ...] = (str, int, float, bool)


def _check_value(val: Any, path: str) -> None:
    """Reject nested dicts, ``None``, and non-primitive list items.

    Raises ``ValueError`` so Pydantic surfaces it through its standard
    ``ValidationError`` mechanism with a clear ``loc``.
    """
    if val is None:
        raise ValueError(f"input {path!r}: None values are not allowed")
    if isinstance(val, dict):
        raise ValueError(
            f"input {path!r}: nested dicts are not allowed; cdsapi "
            "requires a flat request dict"
        )
    if isinstance(val, list):
        for i, item in enumerate(val):
            # ``bool`` is a subclass of ``int`` and is included in the
            # leaf-types tuple — single isinstance check is sufficient.
            if isinstance(item, _ALLOWED_LEAF_TYPES):
                continue
            raise ValueError(
                f"input {path!r}[{i}]: list items must be str/int/float/bool, "
                f"got {type(item).__name__}"
            )
        return
    if isinstance(val, _ALLOWED_LEAF_TYPES):
        return
    raise ValueError(
        f"input {path!r}: must be primitive (str/int/float/bool) or list "
        f"thereof, got {type(val).__name__}"
    )


class CdsRetrieveRequest(BaseModel):
    """Structural envelope for a CDS / ADS / EWDS retrieve request.

    The ``inputs`` dict is passed as-is to ``cdsapi.Client.retrieve()``
    (research §6.3.2). Per-dataset semantic validation requires
    ``apply_constraints`` (research §6.4.2) which the legacy ``cdsapi``
    client does not expose; T-CDS-005 surfaces server-side errors
    through the canonical error taxonomy.

    The ``area`` field, when present, is in cdsapi-native order
    ``[north, west, south, east]`` (research §6.9.2 — non-standard).
    Tool wrappers (T-CDS-007) document this prominently.
    """

    model_config = _CDS_FORBID_FROZEN

    dataset_id: str = Field(min_length=1)
    inputs: dict[str, Any] = Field(
        description=(
            "cdsapi-shaped request dict (variable, year, month, day, "
            "time, area, pressure_level, ...). WARNING — ``area`` "
            "ordering: CDS uses ``[north, west, south, east]`` (NWSE), "
            "OPPOSITE of common GIS ``[west, south, east, north]``. "
            "Sending the wrong order does not error — it silently "
            "retrieves the wrong region. Mediterranean example "
            "(lon -6..36.5, lat 30..46): area=[46, -6, 30, 36.5]."
        ),
    )

    @field_validator("dataset_id")
    @classmethod
    def _reject_blank_dataset_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("dataset_id must not be blank or whitespace-only")
        # T-CDS-005 round-1 HIGH (codex, defense in depth for invariant 6):
        # dataset_id flows into cache keys and the file URI. Reject
        # whitespace and ``=`` so a credential-shaped string like
        # "X password=hunter2" cannot reach the cache index. Real CDS
        # collection ids match ``[a-z0-9][a-z0-9._-]*``; we accept a
        # superset that allows common alphanumeric forms but blocks
        # the obvious credential shapes.
        import re

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", v):
            raise ValueError(
                "dataset_id must contain only alphanumerics and "
                "``._-``; whitespace, '=' and other punctuation are "
                "rejected for cache-key safety"
            )
        return v

    @field_validator("inputs")
    @classmethod
    def _validate_inputs_structure(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("inputs dict must not be empty")
        for key, val in v.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    f"input keys must be non-empty strings; got {key!r}"
                )
            _check_value(val, key)
        return v


class CdsApplyConstraintsRequest(BaseModel):
    """Input for ``cds_apply_constraints`` (T-CDS-016, Layer B).

    Mirrors ``CdsRetrieveRequest`` shape but ``inputs`` MAY be empty —
    an empty dict POSTed to ``/processes/<id>/constraints`` returns
    every field's top-level valid values (same as the bundled snapshot
    from Layer A). Progressive narrowing: agent fills part of the
    request, asks for valid values of the remaining fields, repeats.
    """

    model_config = _CDS_FORBID_FROZEN

    dataset_id: str = Field(min_length=1)
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Partial cdsapi-shaped request. Empty dict → return top-level "
            "valid values for every field. Non-empty → CDS narrows the "
            "remaining valid values given the partial selection."
        ),
    )

    @field_validator("dataset_id")
    @classmethod
    def _reject_blank_dataset_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("dataset_id must not be blank or whitespace-only")
        import re

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", v):
            raise ValueError(
                "dataset_id must contain only alphanumerics and "
                "``._-``; whitespace, '=' and other punctuation are "
                "rejected for cache-key safety"
            )
        return v

    @field_validator("inputs")
    @classmethod
    def _validate_inputs_structure(cls, v: dict[str, Any]) -> dict[str, Any]:
        for key, val in v.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    f"input keys must be non-empty strings; got {key!r}"
                )
            _check_value(val, key)
        return v


CdsStoreId = Literal["cds", "ads", "ewds"]


class CdsSearchRequest(BaseModel):
    """Discovery filter for ``cds_search_datasets`` (T-CDS-003).

    Default is *all stores combined* — the bundled snapshot is small
    enough that returning everything (~150 datasets) costs only ~30k
    tokens, which fits in any modern LLM context. Two-stage select:
    LLM uses search → narrows to N candidates → ``describe`` for full
    record on each.
    """

    model_config = _CDS_FORBID_FROZEN

    keyword: str | None = None
    store: CdsStoreId | None = None
    limit: int | None = Field(default=None, ge=1)
    # T-CDS-020 PR-1: bbox/time/variable filters (AND-combined with
    # ``keyword`` and with each other).
    bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description=(
            "Spatial filter `(west, south, east, north)` in WGS84 degrees "
            "(longitudes -180..180, latitudes -90..90). Records whose "
            "spatial extent intersects this bbox are returned. "
            "Antimeridian-crossing queries (west > east) are rejected — "
            "split into two requests."
        ),
    )
    time_range: tuple[str, str] | None = Field(
        default=None,
        description=(
            "Temporal filter `(start_iso, end_iso)` (ISO 8601 with timezone). "
            "Records whose temporal extent overlaps the window are returned. "
            "Open-ended datasets (no end timestamp in catalogue) match any "
            "future end."
        ),
    )
    variable: str | None = Field(
        default=None,
        description=(
            "Variable-name substring (case-insensitive). Matches against "
            "keywords, title, description, summaries, and the bundled "
            "constraints' `variable` enum — so e.g. `variable=\"temperature\"` "
            "finds ERA5 datasets even though the STAC keywords don't "
            "mention specific variables."
        ),
    )
    # T-CDS-021 PR-2: hierarchical filters. Use ``cds_search_groups`` to
    # discover valid (domain, category) combos first; then pass them
    # here for narrowing.
    domain: str | None = Field(
        default=None,
        description=(
            "Exact match against the dataset's `Variable domain: X` keyword "
            "(e.g. `\"Atmosphere (surface)\"`, `\"Ocean (physics)\"`). Multi-"
            "domain datasets (ERA5 carries both surface and upper-air) "
            "match if any of their tags equals this value."
        ),
    )
    category: str | None = Field(
        default=None,
        description=(
            "Exact match against the dataset's ID family prefix "
            "(e.g. `\"reanalysis\"`, `\"satellite\"`, `\"cams\"`, `\"efas\"`, "
            "`\"sis\"`, `\"projections\"`, `\"derived\"`)."
        ),
    )

    @field_validator("keyword", "variable", "domain", "category")
    @classmethod
    def _normalise_optional_str(cls, v: str | None) -> str | None:
        """Codex/code-reviewer T-CDS-003 LOW: a whitespace-only filter
        is meaningless intent — treat it as no filter. Also catches a
        literal empty-string from a buggy caller. Applied to every
        optional-string filter field."""
        if v is None:
            return None
        stripped = v.strip()
        return stripped or None


class CdsSearchGroupsRequest(BaseModel):
    """Discovery filter for ``cds_search_groups`` (T-CDS-021 PR-2).

    Free-text discovery over the bundled snapshot's (domain, category)
    hierarchy. Mirrors CMEMS's ``marine_search_groups`` UX so the LLM
    can narrow ~164 CDS / ADS / EWDS datasets in two hops instead of
    one overloaded keyword query.
    """

    model_config = _CDS_FORBID_FROZEN

    query: str | None = Field(
        default=None,
        description=(
            "Free-text query — what you are looking for. Ranks groups by "
            "substring matches against domain name + category + member "
            "dataset keywords/titles. Omit to return every populated group "
            "sorted by dataset_count."
        ),
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Cap the returned group count after ranking.",
    )

    @field_validator("query")
    @classmethod
    def _normalise_query(cls, v: str | None) -> str | None:
        if v is None:
            return None
        stripped = v.strip()
        return stripped or None
