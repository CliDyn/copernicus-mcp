"""``CdsRetrieveRequest`` schema tests (T-CDS-002).

 the per-dataset
constraint catalogue is server-side; static validation is limited to
**structural** checks. The schema accepts any cdsapi-shaped ``inputs``
dict (structured CDS form OR MARS-keywords form) but rejects:
- empty / blank ``dataset_id``
- empty ``inputs`` dict
- nested dicts inside ``inputs`` (cdsapi requires a flat dict)
- ``None`` values (cdsapi rejects them server-side)
- non-string keys
- tuples / sets in values (JSON has only lists)
"""

from __future__ import annotations

import pytest


def _structured_inputs() -> dict[str, object]:
    """Modern CDS shape (research §6.9.1)."""
    return {
        "product_type": ["reanalysis"],
        "variable": ["temperature"],
        "year": ["2024"],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],
        "pressure_level": ["1000"],
        "data_format": "grib",
        "download_format": "unarchived",
    }


def _mars_inputs() -> dict[str, object]:
    """Legacy MARS-keywords shape (research §6.9.5)."""
    return {
        "date": "2013-01-01",
        "levelist": "1/10/100/137",
        "levtype": "ml",
        "param": "129/130/131",
        "stream": "oper",
        "time": "00/06/12/18",
        "type": "an",
        "grid": "1.0/1.0",
        "area": "90/-180/-90/180",
    }


def test_constructs_with_structured_inputs() -> None:
    from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest

    req = CdsRetrieveRequest(
        dataset_id="reanalysis-era5-pressure-levels",
        inputs=_structured_inputs(),
    )
    assert req.dataset_id == "reanalysis-era5-pressure-levels"
    assert req.inputs["variable"] == ["temperature"]


def test_constructs_with_mars_inputs() -> None:
    """Per research §6.9.5 archival datasets use MARS keywords."""
    from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest

    req = CdsRetrieveRequest(
        dataset_id="reanalysis-era5-complete",
        inputs=_mars_inputs(),
    )
    assert req.inputs["levtype"] == "ml"


def test_rejects_empty_dataset_id() -> None:
    from pydantic import ValidationError

    from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest

    with pytest.raises(ValidationError):
        CdsRetrieveRequest(dataset_id="", inputs=_structured_inputs())


def test_rejects_whitespace_dataset_id() -> None:
    from pydantic import ValidationError

    from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest

    with pytest.raises(ValidationError):
        CdsRetrieveRequest(dataset_id="   ", inputs=_structured_inputs())


def test_rejects_empty_inputs() -> None:
    from pydantic import ValidationError

    from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest

    with pytest.raises(ValidationError):
        CdsRetrieveRequest(dataset_id="ds-1", inputs={})


def test_rejects_nested_dict_in_inputs() -> None:
    """cdsapi requires a flat request dict (research §6.9.1). A
    nested dict signals a programmer error on the user side."""
    from pydantic import ValidationError

    from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest

    with pytest.raises(ValidationError):
        CdsRetrieveRequest(
            dataset_id="ds-1",
            inputs={"variable": ["temperature"], "options": {"nested": "bad"}},
        )


def test_rejects_none_value_in_inputs() -> None:
    """cdsapi rejects None server-side; catch it earlier."""
    from pydantic import ValidationError

    from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest

    with pytest.raises(ValidationError):
        CdsRetrieveRequest(
            dataset_id="ds-1",
            inputs={"variable": None},
        )


def test_rejects_blank_input_key() -> None:
    from pydantic import ValidationError

    from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest

    with pytest.raises(ValidationError):
        CdsRetrieveRequest(
            dataset_id="ds-1",
            inputs={"": ["temperature"]},
        )


def test_rejects_non_primitive_in_list() -> None:
    from pydantic import ValidationError

    from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest

    with pytest.raises(ValidationError):
        CdsRetrieveRequest(
            dataset_id="ds-1",
            inputs={"variable": [{"nested": "in-list"}]},
        )


def test_accepts_numeric_area() -> None:
    """Bbox with floats: ``area: [90.0, -180.0, -90.0, 180.0]`` per
    research §6.9.2 (north, west, south, east)."""
    from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest

    req = CdsRetrieveRequest(
        dataset_id="reanalysis-era5-single-levels",
        inputs={
            "variable": ["2m_temperature"],
            "year": ["2024"],
            "month": ["01"],
            "day": ["01"],
            "time": ["00:00"],
            "area": [60.0, -10.0, 50.0, 5.0],
        },
    )
    assert req.inputs["area"] == [60.0, -10.0, 50.0, 5.0]


def test_accepts_int_and_bool_values() -> None:
    from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest

    req = CdsRetrieveRequest(
        dataset_id="ds-1",
        inputs={"some_int": 42, "some_bool": True, "list_int": [1, 2, 3]},
    )
    assert req.inputs["some_int"] == 42


def test_extra_top_level_field_forbidden() -> None:
    """Top-level ``extra=forbid`` discipline."""
    from pydantic import ValidationError

    from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest

    with pytest.raises(ValidationError):
        CdsRetrieveRequest(
            dataset_id="ds-1",
            inputs=_structured_inputs(),
            stray_field="should-be-rejected",  # type: ignore[call-arg]
        )


def test_frozen_instance_immutable() -> None:
    """``frozen=True`` consistent with CMEMS schemas."""
    from pydantic import ValidationError

    from copernicus_mcp.data_model.schemas_cds import CdsRetrieveRequest

    req = CdsRetrieveRequest(dataset_id="ds-1", inputs=_structured_inputs())
    with pytest.raises(ValidationError):
        req.dataset_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CdsApplyConstraintsRequest (T-CDS-016 Layer B)
# ---------------------------------------------------------------------------


def test_apply_constraints_dataset_id_required() -> None:
    from pydantic import ValidationError

    from copernicus_mcp.data_model.schemas_cds import CdsApplyConstraintsRequest

    with pytest.raises(ValidationError):
        CdsApplyConstraintsRequest()  # type: ignore[call-arg]


def test_apply_constraints_inputs_default_empty() -> None:
    """Empty inputs is the canonical 'give me top-level valid values' case."""
    from copernicus_mcp.data_model.schemas_cds import CdsApplyConstraintsRequest

    req = CdsApplyConstraintsRequest(dataset_id="efas-historical")
    assert req.inputs == {}


def test_apply_constraints_inputs_with_partial_selection() -> None:
    from copernicus_mcp.data_model.schemas_cds import CdsApplyConstraintsRequest

    req = CdsApplyConstraintsRequest(
        dataset_id="efas-historical",
        inputs={"variable": ["elevation"]},
    )
    assert req.inputs == {"variable": ["elevation"]}


def test_apply_constraints_blank_dataset_id_rejected() -> None:
    from pydantic import ValidationError

    from copernicus_mcp.data_model.schemas_cds import CdsApplyConstraintsRequest

    with pytest.raises(ValidationError):
        CdsApplyConstraintsRequest(dataset_id="")
    with pytest.raises(ValidationError):
        CdsApplyConstraintsRequest(dataset_id="   ")


def test_apply_constraints_blocks_credential_shaped_dataset_id() -> None:
    """Cache-key safety mirror of CdsRetrieveRequest regression."""
    from pydantic import ValidationError

    from copernicus_mcp.data_model.schemas_cds import CdsApplyConstraintsRequest

    with pytest.raises(ValidationError):
        CdsApplyConstraintsRequest(dataset_id="ds password=hunter2")


def test_apply_constraints_extra_fields_rejected() -> None:
    from pydantic import ValidationError

    from copernicus_mcp.data_model.schemas_cds import CdsApplyConstraintsRequest

    with pytest.raises(ValidationError):
        CdsApplyConstraintsRequest(
            dataset_id="ds",
            inputs={},
            unknown="boom",  # type: ignore[call-arg]
        )


def test_apply_constraints_rejects_none_value_in_inputs() -> None:
    from pydantic import ValidationError

    from copernicus_mcp.data_model.schemas_cds import CdsApplyConstraintsRequest

    with pytest.raises(ValidationError):
        CdsApplyConstraintsRequest(
            dataset_id="ds",
            inputs={"variable": None},
        )
