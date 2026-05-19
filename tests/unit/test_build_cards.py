"""T-CMEMS-HIER-001: dataset card builder tests.

``_build_cards.py`` derives the new card-level fields (domain,
region, data_type, normalized variables, spatial/temporal labels,
quality flags) from a slim record produced by ``_catalogue_build.
slim_marine_record``. Rule-based; no LLM. Pure Python, no I/O.

Schema lock: hierarchical-search Stage 1 dataset
cards table.
"""

from __future__ import annotations

from typing import Any


def _slim_record(
    *,
    dataset_id: str = "antarctic_omi_si_extent",
    dataset_name: str = "Sea Ice Extent for Southern Hemisphere",
    product_id: str = "ANTARCTIC_OMI_SI_extent",
    product_title: str = "Antarctic Sea Ice Extent from Reanalysis",
    description: str = "...",
    doi: str = "10.48670/moi-00186",
    service_types: list[str] | None = None,
    variables: list[str] | None = None,
    versions: list[str] | None = None,
    spatial_extent: Any = None,
    temporal_extent: Any = None,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "title": dataset_name,
        "product_id": product_id,
        "product_title": product_title,
        "description": description,
        "doi": doi,
        "service_types": service_types or ["original-files"],
        "variables": variables or ["siextents_cglo"],
        "versions": versions or ["202411"],
        "spatial_extent": spatial_extent,
        "temporal_extent": temporal_extent,
    }


# ---------------------------------------------------------------------------
# infer_domain — one value per dataset
# ---------------------------------------------------------------------------


def test_infer_domain_physics_from_PHY_token() -> None:
    from copernicus_mcp.backends.cmems._build_cards import infer_domain

    assert infer_domain("GLOBAL_ANALYSISFORECAST_PHY_001_024", "...") == "physics"


def test_infer_domain_biogeochemistry_from_BGC_token() -> None:
    from copernicus_mcp.backends.cmems._build_cards import infer_domain

    assert infer_domain("GLOBAL_ANALYSISFORECAST_BGC_001_028", "...") == "biogeochemistry"


def test_infer_domain_waves_from_WAV_token() -> None:
    from copernicus_mcp.backends.cmems._build_cards import infer_domain

    assert infer_domain("GLOBAL_ANALYSISFORECAST_WAV_001_027", "...") == "waves"


def test_infer_domain_sea_ice_from_SI_or_SEAICE_token() -> None:
    from copernicus_mcp.backends.cmems._build_cards import infer_domain

    assert infer_domain("ANTARCTIC_OMI_SI_extent", "...") == "sea_ice"
    assert infer_domain("SEAICE_GLO_PHY_L4_NRT_011_001", "...") == "sea_ice"


def test_infer_domain_ocean_color_from_OCEANCOLOUR_token() -> None:
    """The thematic-token-first product naming pattern codex
    flagged in spec review MEDIUM-3 — prefix-only inference would
    misclassify this as 'arctic' instead of 'ocean_color'."""
    from copernicus_mcp.backends.cmems._build_cards import infer_domain

    assert infer_domain("OCEANCOLOUR_ARC_BGC_L4_MY_009_124", "...") == "ocean_color"


def test_infer_domain_sst_from_SST_token() -> None:
    from copernicus_mcp.backends.cmems._build_cards import infer_domain

    assert infer_domain("SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001", "...") == "sst"


def test_infer_domain_unknown_when_no_token_matches() -> None:
    from copernicus_mcp.backends.cmems._build_cards import infer_domain

    assert infer_domain("FOO_BAR_BAZ", "unrelated title") == "unknown"


# ---------------------------------------------------------------------------
# infer_region — multi-valued
# ---------------------------------------------------------------------------


def test_infer_region_arctic_from_ARCTIC_token() -> None:
    from copernicus_mcp.backends.cmems._build_cards import infer_region

    assert infer_region("ARCTIC_ANALYSISFORECAST_PHY_002_001", "...") == ["arctic"]


def test_infer_region_antarctic_carries_southern_hemisphere() -> None:
    from copernicus_mcp.backends.cmems._build_cards import infer_region

    assert sorted(infer_region("ANTARCTIC_OMI_SI_extent", "...")) == sorted(
        ["antarctic", "southern_hemisphere"]
    )


def test_infer_region_global_from_GLOBAL_token() -> None:
    from copernicus_mcp.backends.cmems._build_cards import infer_region

    assert infer_region("GLOBAL_ANALYSISFORECAST_PHY_001_024", "...") == ["global"]


def test_infer_region_mediterranean_from_MEDSEA_token() -> None:
    from copernicus_mcp.backends.cmems._build_cards import infer_region

    assert infer_region("MEDSEA_ANALYSISFORECAST_PHY_006_013", "...") == ["mediterranean"]


def test_infer_region_black_sea_from_BLKSEA_token() -> None:
    from copernicus_mcp.backends.cmems._build_cards import infer_region

    assert infer_region("BLKSEA_ANALYSISFORECAST_PHY_007_001", "...") == ["black_sea"]


def test_infer_region_thematic_first_OCEANCOLOUR_ARC() -> None:
    """codex spec-review MEDIUM-3: region token can appear after a
    thematic prefix. ``OCEANCOLOUR_ARC_*`` is arctic."""
    from copernicus_mcp.backends.cmems._build_cards import infer_region

    assert infer_region("OCEANCOLOUR_ARC_BGC_L4_MY_009_124", "...") == ["arctic"]


def test_infer_region_unknown_when_no_token_matches() -> None:
    from copernicus_mcp.backends.cmems._build_cards import infer_region

    assert infer_region("FOO_BAR_BAZ", "unrelated title") == ["unknown"]


# ---------------------------------------------------------------------------
# infer_data_type — multi-valued
# ---------------------------------------------------------------------------


def test_infer_data_type_reanalysis_from_my_substring() -> None:
    """``_my_`` (multi-year) marks reanalysis products."""
    from copernicus_mcp.backends.cmems._build_cards import infer_data_type

    assert infer_data_type(
        "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "GLOBAL_MULTIYEAR_PHY_001_030",
        "Global Ocean Physics Reanalysis",
    ) == ["reanalysis"]


def test_infer_data_type_forecast_from_anfc_substring() -> None:
    """``_anfc_`` marks analysis + forecast products."""
    from copernicus_mcp.backends.cmems._build_cards import infer_data_type

    assert sorted(
        infer_data_type(
            "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            "GLOBAL_ANALYSISFORECAST_PHY_001_024",
            "Global Ocean Physics Analysis and Forecast",
        )
    ) == sorted(["analysis", "forecast"])


def test_infer_data_type_observation_from_obs_substring() -> None:
    from copernicus_mcp.backends.cmems._build_cards import infer_data_type

    assert infer_data_type(
        "cmems_obs_glo_phy_my_l4_P1D",
        "INSITU_GLO_PHY_TS_OBSERVATIONS_013_002",
        "Global In-Situ Observations",
    ) == ["observation"]


def test_infer_data_type_indicator_from_OMI_token() -> None:
    from copernicus_mcp.backends.cmems._build_cards import infer_data_type

    assert infer_data_type(
        "antarctic_omi_si_extent",
        "ANTARCTIC_OMI_SI_extent",
        "...",
    ) == ["indicator"]


def test_infer_data_type_unknown_when_no_marker() -> None:
    from copernicus_mcp.backends.cmems._build_cards import infer_data_type

    assert infer_data_type("foo", "BAR", "no markers in title") == ["unknown"]


# ---------------------------------------------------------------------------
# normalize_variables — lookup with fallback
# ---------------------------------------------------------------------------


def test_normalize_variables_maps_known_short_names() -> None:
    """The curated lookup maps SDK short names (``thetao``, ``so``,
    ``siextents_cglo``) to canonical CF-style names. Dedupes the
    result when several short names normalise to the same canonical."""
    from copernicus_mcp.backends.cmems._build_cards import normalize_variables

    result = normalize_variables(["thetao", "so", "siextents_cglo"])
    assert "sea_water_potential_temperature" in result
    assert "sea_water_salinity" in result
    assert "sea_ice_extent" in result


def test_normalize_variables_dedupes_when_multiple_short_names_collapse() -> None:
    """Multiple SDK short names (e.g. ``siextents_cglo`` /
    ``siextents_glor`` / ``siextents_mean``) all collapse to
    ``sea_ice_extent`` — return one entry, not three."""
    from copernicus_mcp.backends.cmems._build_cards import normalize_variables

    result = normalize_variables(["siextents_cglo", "siextents_glor", "siextents_mean"])
    assert result == ["sea_ice_extent"]


def test_normalize_variables_falls_back_to_short_name_when_unknown() -> None:
    """An unmapped SDK short name appears verbatim in the output —
    never silently dropped (codex MEDIUM-3 fallback rule)."""
    from copernicus_mcp.backends.cmems._build_cards import normalize_variables

    result = normalize_variables(["xyzzy_no_such_variable"])
    assert result == ["xyzzy_no_such_variable"]


def test_normalize_variables_empty_input_returns_empty_list() -> None:
    from copernicus_mcp.backends.cmems._build_cards import normalize_variables

    assert normalize_variables([]) == []


# ---------------------------------------------------------------------------
# bbox_to_spatial_label — human-readable summary
# ---------------------------------------------------------------------------


def test_bbox_to_spatial_label_global_extent() -> None:
    from copernicus_mcp.backends.cmems._build_cards import bbox_to_spatial_label

    label = bbox_to_spatial_label(
        {"min_lon": -180.0, "min_lat": -90.0, "max_lon": 180.0, "max_lat": 90.0}
    )
    assert label is not None
    assert "global" in label.lower()


def test_bbox_to_spatial_label_antarctic() -> None:
    from copernicus_mcp.backends.cmems._build_cards import bbox_to_spatial_label

    label = bbox_to_spatial_label(
        {"min_lon": -180.0, "min_lat": -90.0, "max_lon": 180.0, "max_lat": -50.0}
    )
    assert label is not None
    assert "antarctic" in label.lower()


def test_bbox_to_spatial_label_none_when_extent_is_none() -> None:
    from copernicus_mcp.backends.cmems._build_cards import bbox_to_spatial_label

    assert bbox_to_spatial_label(None) is None


# ---------------------------------------------------------------------------
# temporal_extent_to_label — human-readable summary
# ---------------------------------------------------------------------------


def test_temporal_extent_to_label_yyyy_mm_range() -> None:
    from copernicus_mcp.backends.cmems._build_cards import temporal_extent_to_label

    label = temporal_extent_to_label(
        {"start": "1993-01-01T00:00:00Z", "end": "2020-06-15T00:00:00Z"}
    )
    assert label == "1993-01 to 2020-06"


def test_temporal_extent_to_label_present_when_end_recent() -> None:
    """When ``end`` is within the last 30 days of "now", render the
    label as ``"YYYY-MM to present"`` rather than a stale date."""
    import datetime as dt

    from copernicus_mcp.backends.cmems._build_cards import temporal_extent_to_label

    end_recent = dt.datetime.now(dt.UTC) - dt.timedelta(days=5)
    end_iso = end_recent.strftime("%Y-%m-%dT%H:%M:%SZ")
    label = temporal_extent_to_label({"start": "2020-01-01T00:00:00Z", "end": end_iso})
    assert label is not None
    assert label.endswith("to present")
    assert "2020-01" in label


def test_temporal_extent_to_label_none_when_extent_is_none() -> None:
    from copernicus_mcp.backends.cmems._build_cards import temporal_extent_to_label

    assert temporal_extent_to_label(None) is None


# ---------------------------------------------------------------------------
# quality_flags — derived from extent presence + low-confidence inference
# ---------------------------------------------------------------------------


def test_quality_flags_empty_when_card_is_clean() -> None:
    """A card with full spatial + temporal coverage + a confident
    domain inference has no quality flags."""
    from copernicus_mcp.backends.cmems._build_cards import quality_flags

    card = {
        "spatial_extent": {"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        "temporal_extent": {"start": "2020-01-01T00:00:00Z", "end": "2024-01-01T00:00:00Z"},
        "domain": "physics",
        "region": ["global"],
    }
    assert quality_flags(card) == []


def test_quality_flags_no_spatial_extent() -> None:
    from copernicus_mcp.backends.cmems._build_cards import quality_flags

    card = {
        "spatial_extent": None,
        "temporal_extent": {"start": "2020-01-01T00:00:00Z", "end": "2024-01-01T00:00:00Z"},
        "domain": "physics",
        "region": ["global"],
    }
    assert "no_spatial_extent" in quality_flags(card)


def test_quality_flags_no_temporal_extent() -> None:
    from copernicus_mcp.backends.cmems._build_cards import quality_flags

    card = {
        "spatial_extent": {"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        "temporal_extent": None,
        "domain": "physics",
        "region": ["global"],
    }
    assert "no_temporal_extent" in quality_flags(card)


def test_quality_flags_low_confidence_domain() -> None:
    from copernicus_mcp.backends.cmems._build_cards import quality_flags

    card = {
        "spatial_extent": {"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        "temporal_extent": {"start": "2020-01-01T00:00:00Z", "end": "2024-01-01T00:00:00Z"},
        "domain": "unknown",
        "region": ["global"],
    }
    assert "low_confidence_domain" in quality_flags(card)


def test_quality_flags_low_confidence_region() -> None:
    from copernicus_mcp.backends.cmems._build_cards import quality_flags

    card = {
        "spatial_extent": {"min_lon": 0.0, "min_lat": 0.0, "max_lon": 1.0, "max_lat": 1.0},
        "temporal_extent": {"start": "2020-01-01T00:00:00Z", "end": "2024-01-01T00:00:00Z"},
        "domain": "physics",
        "region": ["unknown"],
    }
    assert "low_confidence_region" in quality_flags(card)


# ---------------------------------------------------------------------------
# build_dataset_card — top-level builder ties everything together
# ---------------------------------------------------------------------------


def test_build_dataset_card_produces_full_schema() -> None:
    """``build_dataset_card`` returns a dict with every field the
    HIER plan locks for level-1 cards."""
    from copernicus_mcp.backends.cmems._build_cards import build_dataset_card

    slim = _slim_record(
        product_id="GLOBAL_ANALYSISFORECAST_PHY_001_024",
        product_title="Global Ocean Physics Analysis and Forecast",
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        dataset_name="Daily mean",
        variables=["thetao", "so"],
        spatial_extent={"min_lon": -180.0, "min_lat": -90.0, "max_lon": 180.0, "max_lat": 90.0},
        temporal_extent={"start": "2020-01-01T00:00:00Z", "end": "2024-01-01T00:00:00Z"},
    )
    card = build_dataset_card(slim)
    expected_keys = {
        "dataset_id",
        "dataset_name",
        "title",
        "product_id",
        "product_title",
        "description",
        "doi",
        "service_types",
        "variables",
        "variables_normalized",
        "versions",
        "spatial_extent",
        "spatial_label",
        "temporal_extent",
        "temporal_label",
        "domain",
        "region",
        "data_type",
        "quality_flags",
        "best_for",
        "not_good_for",
    }
    assert set(card.keys()) == expected_keys


def test_build_dataset_card_passthrough_fields_preserved() -> None:
    """Slim record fields appear verbatim on the card."""
    from copernicus_mcp.backends.cmems._build_cards import build_dataset_card

    slim = _slim_record(
        dataset_id="ds-x",
        product_id="GLOBAL_ANALYSISFORECAST_PHY_001_024",
        variables=["thetao"],
    )
    card = build_dataset_card(slim)
    assert card["dataset_id"] == "ds-x"
    assert card["variables"] == ["thetao"]


def test_build_dataset_card_derived_fields_populated() -> None:
    from copernicus_mcp.backends.cmems._build_cards import build_dataset_card

    slim = _slim_record(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        product_id="GLOBAL_ANALYSISFORECAST_PHY_001_024",
        product_title="Global Ocean Physics Analysis and Forecast",
        variables=["thetao"],
    )
    card = build_dataset_card(slim)
    assert card["domain"] == "physics"
    assert card["region"] == ["global"]
    assert sorted(card["data_type"]) == sorted(["analysis", "forecast"])
    assert card["variables_normalized"] == ["sea_water_potential_temperature"]


def test_build_dataset_card_best_for_populated_by_annotator() -> None:
    """T-CMEMS-HIER-004: ``build_dataset_card`` now calls
    ``annotate_card`` to fill ``best_for`` / ``not_good_for`` from
    rule-based templates. Both fields are non-empty lists of
    strings, populated based on the card's ``data_type`` and
    ``domain``."""
    from copernicus_mcp.backends.cmems._build_cards import build_dataset_card

    card = build_dataset_card(_slim_record())
    assert isinstance(card["best_for"], list) and card["best_for"]
    assert all(isinstance(x, str) for x in card["best_for"])
    assert isinstance(card["not_good_for"], list) and card["not_good_for"]
    assert all(isinstance(x, str) for x in card["not_good_for"])


# ---------------------------------------------------------------------------
# Round-1 cr+codex HIGH findings — coverage gaps in region/data_type/domain
# ---------------------------------------------------------------------------


def test_infer_region_recognises_GLO_token() -> None:
    """codex+cr round-1 HIGH: ``GLO`` appears in many CMEMS product_ids
    (e.g. ``INSITU_GLO_PHY_TS_OA_MY_013_052``, ``SEAICE_GLO_PHY_L4_MY_011_020``)
    and should map to global, mirroring the ``GLOBAL`` token."""
    from copernicus_mcp.backends.cmems._build_cards import infer_region

    assert infer_region("INSITU_GLO_PHY_TS_OA_MY_013_052", "...") == ["global"]
    assert infer_region("SEAICE_GLO_PHY_L4_MY_011_020", "...") == ["global"]


def test_infer_region_recognises_BALTICSEA_token() -> None:
    """codex+cr round-1 HIGH: real product
    ``BALTICSEA_ANALYSISFORECAST_BGC_003_007`` uses ``BALTICSEA`` not
    just ``BAL``."""
    from copernicus_mcp.backends.cmems._build_cards import infer_region

    assert infer_region("BALTICSEA_ANALYSISFORECAST_BGC_003_007", "...") == ["baltic"]


def test_infer_region_recognises_NWSHELF_token() -> None:
    """codex+cr round-1 HIGH: real product ``NWSHELF_REANALYSIS_WAV_*`` uses
    ``NWSHELF``."""
    from copernicus_mcp.backends.cmems._build_cards import infer_region

    assert infer_region("NWSHELF_REANALYSIS_WAV_004_015", "...") == ["north_west_shelf"]


def test_infer_region_recognises_BLK_and_BS_tokens() -> None:
    """codex+cr round-1 HIGH: Black Sea variants. ``BLK`` (alongside
    existing ``BLKSEA``) and ``BS`` (e.g. ``SST_BS_*``)."""
    from copernicus_mcp.backends.cmems._build_cards import infer_region

    assert infer_region("BLK_SOMETHING", "...") == ["black_sea"]
    assert infer_region("SST_BS_SST_L4_NRT_OBSERVATIONS_010_006", "...") == ["black_sea"]


def test_infer_region_recognises_ATL_token() -> None:
    """codex+cr round-1 HIGH: Atlantic — real products ``OCEANCOLOUR_ATL_*``,
    ``SST_ATL_*``."""
    from copernicus_mcp.backends.cmems._build_cards import infer_region

    assert infer_region("OCEANCOLOUR_ATL_BGC_L4_MY_009_098", "...") == ["atlantic"]


def test_infer_region_recognises_ANT_token() -> None:
    """codex+cr round-1 HIGH: Antarctic short form ``ANT`` used by
    ``SEAICE_ANT_PHY_*``."""
    from copernicus_mcp.backends.cmems._build_cards import infer_region

    assert sorted(infer_region("SEAICE_ANT_PHY_AUTO_L3_NRT_011_012", "...")) == sorted(
        ["antarctic", "southern_hemisphere"]
    )


def test_infer_data_type_case_insensitive_for_MY_in_product_id() -> None:
    """codex+cr round-1 HIGH: uppercase ``_MY_`` in product_id
    (``OCEANCOLOUR_ARC_BGC_L4_MY_009_124``) didn't match the
    lowercase ``_my_`` marker. 312/1251 datasets land on ``unknown``
    because of this.

    Round-1b refinement: ``obs-`` (with hyphen) is now the
    observation marker. ``cmems_obs-oc_*_my_*`` is multi-year
    **observation**, not reanalysis — the obs- tier wins. Same
    case-insensitivity property still holds: an uppercase product
    without obs- prefix correctly hits reanalysis."""
    from copernicus_mcp.backends.cmems._build_cards import infer_data_type

    # obs- prefix → observation (multi-year obs is still obs).
    assert infer_data_type(
        "cmems_obs-oc_arc_bgc_optics_my_l3-multi-1km_P1D",
        "OCEANCOLOUR_ARC_BGC_L4_MY_009_124",
        "...",
    ) == ["observation"]
    # No obs- prefix, uppercase _MY_ still in product_id → reanalysis.
    assert infer_data_type(
        "cmems_mod_glo_phy_my_grid",
        "GLOBAL_MULTIYEAR_PHY_001_030",
        "...",
    ) == ["reanalysis"]


def test_infer_data_type_recognises_REANALYSIS_title_token() -> None:
    """codex+cr round-1 HIGH: ``NWSHELF_REANALYSIS_WAV_004_015`` lacks
    ``_my_`` but the product_id literally contains ``REANALYSIS``.
    The marker set should include the long-form name too."""
    from copernicus_mcp.backends.cmems._build_cards import infer_data_type

    assert infer_data_type(
        "cmems_mod_nws_wav_my_1.5km",
        "NWSHELF_REANALYSIS_WAV_004_015",
        "...",
    ) == ["reanalysis"]


def test_infer_domain_recognises_WAVE_alias() -> None:
    """codex round-1 MEDIUM: ``WAVE`` (alongside existing ``WAV``)
    appears in real CMEMS product_ids."""
    from copernicus_mcp.backends.cmems._build_cards import infer_domain

    assert infer_domain("GLOBAL_REANALYSIS_WAVE_001_032", "...") == "waves"


def test_infer_domain_recognises_SEALEVEL() -> None:
    """codex round-1 MEDIUM: sea-level products use ``SEALEVEL``."""
    from copernicus_mcp.backends.cmems._build_cards import infer_domain

    assert infer_domain("SEALEVEL_GLO_PHY_L4_MY_008_047", "...") == "sea_level"


def test_infer_domain_recognises_WIND() -> None:
    """codex round-1 MEDIUM: wind products use ``WIND``."""
    from copernicus_mcp.backends.cmems._build_cards import infer_domain

    assert infer_domain("WIND_GLO_PHY_L4_NRT_OBSERVATIONS_012_004", "...") == "wind"


def test_infer_domain_recognises_BATHYMETRY() -> None:
    """codex round-1 MEDIUM: bathymetry products use the long form
    ``BATHYMETRY`` (the existing ``BATHY`` marker only catches short
    variants)."""
    from copernicus_mcp.backends.cmems._build_cards import infer_domain

    assert infer_domain("BATHYMETRY_GLO_PHY_L4_MY_002_001", "...") == "bathymetry"


def test_temporal_extent_to_label_forecast_horizon_explicit_label() -> None:
    """cr round-1 HIGH: when ``end`` is in the future (forecast), the
    label should NOT be ``"to present"`` for arbitrarily far horizons.
    Treat any end > now as a forecast horizon and label explicitly."""
    import datetime as dt

    from copernicus_mcp.backends.cmems._build_cards import temporal_extent_to_label

    # End 1 year in the future — definitely a forecast.
    end_future = dt.datetime.now(dt.UTC) + dt.timedelta(days=365)
    end_iso = end_future.strftime("%Y-%m-%dT%H:%M:%SZ")
    label = temporal_extent_to_label({"start": "2020-01-01T00:00:00Z", "end": end_iso})
    assert label is not None
    # Acceptance: any future end either ends with "(forecast)" or is
    # an explicit month label, but NEVER bare "to present" for far
    # futures.
    assert (
        "(forecast)" in label
        or "to present" not in label
        or end_future <= dt.datetime.now(dt.UTC) + dt.timedelta(days=30)
    )


def test_temporal_extent_to_label_short_forecast_horizon_still_present() -> None:
    """A forecast with end ~10 days in the future is "to present"
    (the threshold catches both recent past AND near-future)."""
    import datetime as dt

    from copernicus_mcp.backends.cmems._build_cards import temporal_extent_to_label

    end_near = dt.datetime.now(dt.UTC) + dt.timedelta(days=10)
    end_iso = end_near.strftime("%Y-%m-%dT%H:%M:%SZ")
    label = temporal_extent_to_label({"start": "2020-01-01T00:00:00Z", "end": end_iso})
    assert label is not None
    assert label.endswith("to present")


def test_extract_time_range_rejects_bool() -> None:
    """cr round-1 MEDIUM: ``bool`` is a subclass of ``int``. A stray
    ``True``/``False`` in coordinate fields shouldn't produce a
    bogus (0,1) extent."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    dataset = {
        "dataset_id": "x",
        "dataset_name": "x",
        "versions": [
            {
                "label": "1",
                "parts": [
                    {
                        "services": [
                            {
                                "service_name": "s",
                                "variables": [
                                    {
                                        "short_name": "v",
                                        "coordinates": [
                                            {
                                                "coordinate_id": "time",
                                                "minimum_value": True,
                                                "maximum_value": True,
                                            }
                                        ],
                                    }
                                ],
                            }
                        ]
                    }
                ],
            }
        ],
    }
    product = {"product_id": "p", "title": "t"}
    rec = slim_marine_record(product, dataset)
    assert rec["temporal_extent"] is None


def test_extract_time_range_rejects_zero_sentinel() -> None:
    """cr round-1 MEDIUM: ``min=0, max=0`` is a sentinel — same
    pattern as the ``[0,0,0,0]`` bbox sentinel that's already
    filtered. Return None rather than emit a bogus 1970-01-01
    extent."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    dataset = {
        "dataset_id": "x",
        "dataset_name": "x",
        "versions": [
            {
                "label": "1",
                "parts": [
                    {
                        "services": [
                            {
                                "service_name": "s",
                                "variables": [
                                    {
                                        "short_name": "v",
                                        "coordinates": [
                                            {
                                                "coordinate_id": "time",
                                                "minimum_value": 0,
                                                "maximum_value": 0,
                                            }
                                        ],
                                    }
                                ],
                            }
                        ]
                    }
                ],
            }
        ],
    }
    product = {"product_id": "p", "title": "t"}
    rec = slim_marine_record(product, dataset)
    assert rec["temporal_extent"] is None


def test_infer_data_type_omi_leading_token_classifies_as_indicator() -> None:
    """cr round-2 MEDIUM: ``OMI_CLIMATE_*`` products (OMI at the
    leading token, no preceding underscore) used to fall through
    the ``_omi_`` marker (which requires both delimiters) and land
    on a title-driven false-positive of ``reanalysis``. They are
    indicator products and must classify as such."""
    from copernicus_mcp.backends.cmems._build_cards import infer_data_type

    # Title intentionally contains "reanalysis" to exercise the
    # round-1 regression cr caught.
    assert infer_data_type(
        "omi_climate_ofc_baltic_area_averaged_anomalies",
        "OMI_CLIMATE_OFC_BALTIC_area_averaged_anomalies",
        "Baltic Sea Anomaly from Reanalysis",
    ) == ["indicator"]


def test_infer_data_type_bare_reanalysis_in_title_no_longer_triggers() -> None:
    """cr round-2 MEDIUM: the round-1 ``'reanalysis'`` literal
    marker caused 4 OMI products with that word in their title to
    be mis-classified. The fix uses the delimited ``_reanalysis_``
    marker instead. A product whose title contains the word but
    whose dataset_id / product_id don't have the delimited form
    should NOT classify as reanalysis."""
    from copernicus_mcp.backends.cmems._build_cards import infer_data_type

    # Hypothetical product with reanalysis only in its title prose.
    assert infer_data_type(
        "foo_anomaly",
        "FOO_ANOMALY",
        "The Anomaly Computed From Reanalysis Data",
    ) == ["unknown"]


def test_infer_data_type_delimited_reanalysis_still_works() -> None:
    """The original NWSHELF case (delimited ``_REANALYSIS_`` token
    in product_id) is still caught after the round-2 fix."""
    from copernicus_mcp.backends.cmems._build_cards import infer_data_type

    assert infer_data_type(
        "cmems_mod_nws_wav_my_1.5km",
        "NWSHELF_REANALYSIS_WAV_004_015",
        "...",
    ) == ["reanalysis"]


def test_infer_data_type_obs_hyphen_prefix_classifies_as_observation() -> None:
    """Round-1b coverage: CMEMS uses ``obs-`` (with hyphen) in
    dataset_ids — ``cmems_obs-wind_glo_phy_my_l4`` is a multi-year
    observation product, NOT a reanalysis. ``_obs_`` (with both
    underscores) misses it; need ``obs-`` as an observation marker."""
    from copernicus_mcp.backends.cmems._build_cards import infer_data_type

    assert infer_data_type(
        "cmems_obs-wind_glo_phy_my_l4_P1M",
        "WIND_GLO_PHY_CLIMATE_L4_MY_012_003",
        "...",
    ) == ["observation"]


def test_infer_data_type_obs_hyphen_at_start_of_dataset_id() -> None:
    """Some CMEMS dataset_ids start with ``cmems-obs-*`` (hyphen-only,
    no underscore separator). The marker must catch both
    ``cmems_obs-*`` and ``cmems-obs-*``."""
    from copernicus_mcp.backends.cmems._build_cards import infer_data_type

    assert infer_data_type(
        "cmems-obs-wave_glo_phy_spc-fwk_nrt_cfo_l3_PT15S-i",
        "WAVE_GLO_PHY_SPC-FWK_L3_NRT_014_002",
        "...",
    ) == ["observation"]


def test_infer_data_type_pure_nrt_falls_back_to_analysis() -> None:
    """When ``_nrt_`` appears WITHOUT an ``obs-`` prefix, it's a
    model near-real-time product (analysis). Cosmetic separation
    from pure reanalysis and observation."""
    from copernicus_mcp.backends.cmems._build_cards import infer_data_type

    # Hypothetical pure-NRT model product (no obs- prefix).
    assert infer_data_type(
        "cmems_mod_glo_phy_nrt_5km",
        "GLOBAL_MOD_PHY_NRT_001_999",
        "Global Ocean Physics Near-Real-Time Analysis",
    ) == ["analysis"]


def test_infer_region_recognises_NORTHWESTSHELF_ATLANTIC_PACIFIC_EUROPE() -> None:
    """Round-1b: 13 unknown-region cards all use NORTHWESTSHELF /
    ATLANTIC / PACIFIC / EUROPE long forms. Adding them drops to
    zero unknowns."""
    from copernicus_mcp.backends.cmems._build_cards import infer_region

    assert infer_region("NORTHWESTSHELF_OMI_TEMPSAL_extreme", "...") == ["north_west_shelf"]
    assert infer_region("OMI_CIRCULATION_BOUNDARY_ATLANTIC_gulf_stream", "...") == ["atlantic"]
    assert infer_region("OMI_CIRCULATION_BOUNDARY_PACIFIC_kuroshio", "...") == ["pacific"]
    assert infer_region("OMI_CLIMATE_SL_EUROPE_area_averaged_anomalies", "...") == ["europe"]


def test_infer_domain_recognises_omi_subtokens() -> None:
    """OMI products encode the subject domain in the token after
    OMI. ``OMI_TEMPSAL_*`` is physics, ``OMI_HEALTH_*`` is BGC,
    ``OMI_SEASTATE_*`` is waves, etc."""
    from copernicus_mcp.backends.cmems._build_cards import infer_domain

    assert infer_domain("BLKSEA_OMI_TEMPSAL_sst_trend", "...") == "physics"
    assert infer_domain("GLOBAL_OMI_HEALTH_carbon_ph_trend", "...") == ("biogeochemistry")
    assert infer_domain("BLKSEA_OMI_SEASTATE_extreme_var_swh", "...") == "waves"
    assert infer_domain("OMI_CIRCULATION_BOUNDARY_ATLANTIC_gulf_stream", "...") == "physics"


def test_infer_domain_recognises_SL_sea_level_token() -> None:
    """OMI_CLIMATE_SL_* products encode sea_level in the SL token."""
    from copernicus_mcp.backends.cmems._build_cards import infer_domain

    assert infer_domain("OMI_CLIMATE_SL_EUROPE_area_averaged_anomalies", "...") == "sea_level"


def test_infer_domain_recognises_PHYBGCWAV_composite() -> None:
    """INSITU_*_PHYBGCWAV_* products are multi-discipline observations
    spanning physics, biogeochemistry, and waves. Classify as
    'physics' (the dominant first component) rather than 'unknown'."""
    from copernicus_mcp.backends.cmems._build_cards import infer_domain

    assert infer_domain("INSITU_GLO_PHYBGCWAV_DISCRETE_MYNRT_013_030", "...") == ("physics")


def test_infer_domain_recognises_more_omi_subtokens() -> None:
    """Round-1c: a tail of OMI sub-tokens. Most map to physics
    (ocean heat content, currents, etc.); ``STATE`` maps to waves
    because the empirical case ``OMI_EXTREME_STATE_GLOBAL_trend``
    is a Significant Wave Height indicator (codex round-1
    MEDIUM)."""
    from copernicus_mcp.backends.cmems._build_cards import infer_domain

    cases = [
        ("IBI_OMI_CURRENTS_cui", "physics"),
        ("OMI_CLIMATE_OHC_GLOBAL_area_averaged_anomalies_0_2000", "physics"),
        ("OMI_CLIMATE_OFC_BALTIC_area_averaged_anomalies", "physics"),
        ("OMI_CLIMATE_OSC_MEDSEA_volume_mean", "physics"),
        ("OMI_CLIMATE_THSL_GLOBAL_trend", "physics"),
        ("OMI_EXTREME_MHW_ARCTIC_area_averaged_anomalies", "physics"),
        ("OMI_EXTREME_STATE_GLOBAL_trend", "waves"),
        ("OMI_EXTREME_WMF_MEDSEA_area_averaged_mean", "physics"),
    ]
    for product_id, expected in cases:
        assert infer_domain(product_id, "...") == expected, f"{product_id} → expected {expected}"


def test_infer_data_type_observations_in_title_does_not_match_omi() -> None:
    """cr round-1 HIGH on PR #87: the bare ``observations`` marker
    (not delimited) matched OMI product titles containing the word
    "observations" prose (e.g. "from Satellite Observations") and
    mis-classified 79 OMI products as ``observation`` instead of
    ``indicator``. Same shape as the PR #86 ``reanalysis`` bug.
    Fix: delimit to ``_observations_``."""
    from copernicus_mcp.backends.cmems._build_cards import infer_data_type

    # OMI product whose title contains prose "from Satellite
    # Observations Reprocessing".
    assert infer_data_type(
        "blksea_omi_seastate_extreme_var_swh_mean_and_anomaly",
        "BLKSEA_OMI_SEASTATE_extreme_var_swh_mean_and_anomaly",
        "Black Sea Significant Wave Height Mean From Satellite Observations Reprocessing",
    ) == ["indicator"]


def test_infer_data_type_OBSERVATIONS_token_in_product_id_still_works() -> None:
    """Delimited ``_observations_`` (e.g. ``INSITU_GLO_PHY_TS_OBSERVATIONS_*``)
    is still observation. The fix narrows from bare to delimited,
    preserving the legitimate cases."""
    from copernicus_mcp.backends.cmems._build_cards import infer_data_type

    assert infer_data_type(
        "cmems_obs_glo_phy_my_l4_P1D",
        "INSITU_GLO_PHY_TS_OBSERVATIONS_013_002",
        "Global Ocean Physics In-Situ Observations",
    ) == ["observation"]


def test_infer_data_type_analysisforecast_one_word() -> None:
    """``ARCTIC_ANALYSISFORECAST_PHY_TIDE_*`` carries the
    ``ANALYSISFORECAST`` token in one word — needs to map to
    analysis+forecast same as the ``_anfc_`` short form."""
    from copernicus_mcp.backends.cmems._build_cards import infer_data_type

    assert sorted(
        infer_data_type(
            "dataset-topaz6-arc-15min-3km-be",
            "ARCTIC_ANALYSISFORECAST_PHY_TIDE_002_015",
            "...",
        )
    ) == sorted(["analysis", "forecast"])


def test_infer_data_type_analysis_underscore_forecast() -> None:
    """``ARCTIC_ANALYSIS_FORECAST_WAV_*`` carries the two-word
    ``_ANALYSIS_FORECAST_`` form with an underscore between."""
    from copernicus_mcp.backends.cmems._build_cards import infer_data_type

    assert sorted(
        infer_data_type(
            "dataset-wam-arctic-1hr3km-be",
            "ARCTIC_ANALYSIS_FORECAST_WAV_002_014",
            "...",
        )
    ) == sorted(["analysis", "forecast"])


def test_extract_time_range_rejects_mixed_string_values() -> None:
    """cr round-1 MEDIUM: a ``values`` list with strings shouldn't
    silently slice to the numeric subset — that's "fail open with
    wrong data". Return None so the dataset gets the
    no_temporal_extent quality flag instead."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    dataset = {
        "dataset_id": "x",
        "dataset_name": "x",
        "versions": [
            {
                "label": "1",
                "parts": [
                    {
                        "services": [
                            {
                                "service_name": "s",
                                "variables": [
                                    {
                                        "short_name": "v",
                                        "coordinates": [
                                            {
                                                "coordinate_id": "time",
                                                "minimum_value": None,
                                                "maximum_value": None,
                                                "values": [
                                                    1577836800000,
                                                    "2020-12-31",
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ]
                    }
                ],
            }
        ],
    }
    product = {"product_id": "p", "title": "t"}
    rec = slim_marine_record(product, dataset)
    assert rec["temporal_extent"] is None
