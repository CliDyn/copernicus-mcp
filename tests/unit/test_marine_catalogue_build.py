"""T-CMEMS-CAT-001: slim-record builder tests.

The slim helpers live in ``backends/cmems/_catalogue_build.py`` and are
imported by both the dev/ops refresh script
(``scripts/refresh_marine_catalogue.py``) and these tests. Runtime
``CmemsBackend.search`` will NOT import them — search reads the bundled
``_data/marine.json``, which the script writes.

Schema reference: ``spikes/T-CMEMS-CAT-000-describe-shape/FINDINGS.md``
Stage 3.
"""

from __future__ import annotations

from typing import Any

import pytest


def _make_sdk_dataset(
    *,
    dataset_id: str = "antarctic_omi_si_extent",
    dataset_name: str = "Sea Ice Extent for Southern Hemisphere",
    versions: list[dict[str, Any]] | None = None,
    spatial_extent: Any = None,
    temporal_extent: Any = None,
) -> dict[str, Any]:
    """Build an SDK-shaped dataset dict that matches the empirical
    shape captured in the T-CMEMS-CAT-000 smoke."""
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "product_id": "ANTARCTIC_OMI_SI_extent",
        "digital_object_identifier": "10.48670/moi-00186",
        "versions": versions
        if versions is not None
        else [
            {
                "label": "202411",
                "parts": [
                    {
                        "name": "default",
                        "services": [
                            {
                                "service_name": "original-files",
                                "service_short_name": "files",
                                "variables": [
                                    {
                                        "short_name": "siextents_cglo",
                                        "standard_name": "sea_ice_extent",
                                        "units": "km2",
                                        "bbox": [-180.0, -90.0, 180.0, -50.0],
                                        "coordinates": [],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
        "spatial_extent": spatial_extent,
        "temporal_extent": temporal_extent,
    }


def _make_sdk_product(
    *,
    product_id: str = "ANTARCTIC_OMI_SI_extent",
    title: str = "Antarctic Sea Ice Extent from Reanalysis",
    description: str = "Short description.",
) -> dict[str, Any]:
    return {
        "product_id": product_id,
        "title": title,
        "description": description,
        "digital_object_identifier": "10.48670/moi-00186",
        "sources": ["Numerical models"],
        "processing_level": None,
        "production_center": "Mercator Ocean International",
        "keywords": ["target-application#seaiceinformation"],
        "thumbnail_url": "https://example.org/thumb.png",
    }


# ---------------------------------------------------------------------------
# slim_marine_record — happy-path schema lock
# ---------------------------------------------------------------------------


def test_slim_marine_record_has_all_locked_schema_fields() -> None:
    """FINDINGS Stage 3 lists 12 keys for the slim record. Pin them all
    in one go so an accidental removal breaks the test."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    rec = slim_marine_record(_make_sdk_product(), _make_sdk_dataset())
    assert sorted(rec.keys()) == sorted(
        [
            "dataset_id",
            "dataset_name",
            "title",
            "product_id",
            "product_title",
            "description",
            "doi",
            "service_types",
            "variables",
            "versions",
            "spatial_extent",
            "temporal_extent",
        ]
    )


def test_slim_marine_record_dataset_fields_passthrough() -> None:
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    rec = slim_marine_record(_make_sdk_product(), _make_sdk_dataset())
    assert rec["dataset_id"] == "antarctic_omi_si_extent"
    assert rec["dataset_name"] == "Sea Ice Extent for Southern Hemisphere"
    # ``title`` mirrors ``dataset_name`` per FINDINGS Stage 3.
    assert rec["title"] == "Sea Ice Extent for Southern Hemisphere"
    assert rec["product_id"] == "ANTARCTIC_OMI_SI_extent"
    assert rec["doi"] == "10.48670/moi-00186"


def test_slim_marine_record_product_context_attached() -> None:
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    rec = slim_marine_record(
        _make_sdk_product(
            product_id="GLOBAL_ANALYSISFORECAST_PHY_001_024",
            title="Global Ocean Physics Analysis and Forecast",
            description="Daily mean fields ...",
        ),
        _make_sdk_dataset(),
    )
    # product_id, product_title, description come from the product, not
    # the dataset. They give search a richer surface than the live
    # _map_dataset envelope did.
    assert rec["product_id"] == "GLOBAL_ANALYSISFORECAST_PHY_001_024"
    assert rec["product_title"] == "Global Ocean Physics Analysis and Forecast"
    assert rec["description"] == "Daily mean fields ..."


def test_slim_marine_record_walks_versions_for_variables_and_services() -> None:
    """variables / service_types are nested 4 levels deep under
    versions[].parts[].services[]. The slim path mirrors the live
    _map_dataset walk so search consumers see the same flat list."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    rec = slim_marine_record(_make_sdk_product(), _make_sdk_dataset())
    assert rec["service_types"] == ["original-files"]
    assert rec["variables"] == ["siextents_cglo"]


def test_slim_marine_record_collects_all_version_labels() -> None:
    """FINDINGS Stage 3: ``versions`` carries every label so consumers
    don't need to call describe just to discover a non-default
    version."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    dataset = _make_sdk_dataset(
        versions=[
            {"label": "202105", "parts": []},
            {"label": "202211", "parts": []},
        ]
    )
    rec = slim_marine_record(_make_sdk_product(), dataset)
    assert rec["versions"] == ["202105", "202211"]


# ---------------------------------------------------------------------------
# spatial_extent — _union_bboxes with sentinel skip (round-2 H1 decision)
# ---------------------------------------------------------------------------


def test_slim_marine_record_spatial_extent_unions_variable_bboxes() -> None:
    """Slim path aggregates variable.bbox via the SAME _union_bboxes
    helper the describe-path uses. Skips [0,0,0,0] sentinels per the
    deliberate divergence documented in FINDINGS §Future."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    dataset = _make_sdk_dataset(
        versions=[
            {
                "label": "202411",
                "parts": [
                    {
                        "name": "default",
                        "services": [
                            {
                                "service_name": "files",
                                "variables": [
                                    {"short_name": "v1", "bbox": [-10.0, -5.0, 10.0, 5.0]},
                                    {"short_name": "v2", "bbox": [0.0, -20.0, 30.0, 20.0]},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    )
    rec = slim_marine_record(_make_sdk_product(), dataset)
    assert rec["spatial_extent"] == {
        "min_lon": -10.0,
        "min_lat": -20.0,
        "max_lon": 30.0,
        "max_lat": 20.0,
    }


def test_slim_marine_record_spatial_extent_skips_zero_sentinel_bboxes() -> None:
    """A variable with bbox=[0,0,0,0] (the empirical OMI Sea Ice
    sentinel) is dropped from the union. If ALL variables have the
    sentinel, spatial_extent is None (not a bogus 0-area bbox)."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    dataset = _make_sdk_dataset(
        versions=[
            {
                "label": "202411",
                "parts": [
                    {
                        "name": "default",
                        "services": [
                            {
                                "service_name": "files",
                                "variables": [
                                    {"short_name": "real", "bbox": [-10.0, -5.0, 10.0, 5.0]},
                                    {"short_name": "sentinel", "bbox": [0.0, 0.0, 0.0, 0.0]},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    )
    rec = slim_marine_record(_make_sdk_product(), dataset)
    # Sentinel skipped; union is the one real bbox.
    assert rec["spatial_extent"] == {
        "min_lon": -10.0,
        "min_lat": -5.0,
        "max_lon": 10.0,
        "max_lat": 5.0,
    }


def test_slim_marine_record_spatial_extent_all_sentinels_returns_none() -> None:
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    dataset = _make_sdk_dataset(
        versions=[
            {
                "label": "202411",
                "parts": [
                    {
                        "name": "default",
                        "services": [
                            {
                                "service_name": "files",
                                "variables": [
                                    {"short_name": "s1", "bbox": [0.0, 0.0, 0.0, 0.0]},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    )
    rec = slim_marine_record(_make_sdk_product(), dataset)
    assert rec["spatial_extent"] is None


def test_slim_marine_record_temporal_extent_none_when_no_time_coordinate() -> None:
    """T-CMEMS-HIER-001: a dataset whose variables have no time
    coordinate (e.g. OMI-style index with only `coordinates: []`)
    still gets ``temporal_extent: None``. The aggregator is silent on
    datasets where the SDK doesn't surface a time series."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    rec = slim_marine_record(_make_sdk_product(), _make_sdk_dataset())
    assert rec["temporal_extent"] is None


def test_slim_marine_record_temporal_extent_from_min_max_values() -> None:
    """T-CMEMS-HIER-001: when ``variable.coordinates`` has a
    ``coordinate_id=="time"`` entry with non-null ``minimum_value``
    and ``maximum_value`` (ms since epoch UTC), aggregate them into
    a dataset-level ``{"start": ISO, "end": ISO}``."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    # 2010-01-01 00:00 UTC = 1262304000000 ms
    # 2024-12-31 00:00 UTC = 1735603200000 ms
    dataset = _make_sdk_dataset(
        versions=[
            {
                "label": "202411",
                "parts": [
                    {
                        "name": "default",
                        "services": [
                            {
                                "service_name": "files",
                                "variables": [
                                    {
                                        "short_name": "thetao",
                                        "bbox": [-180.0, -90.0, 180.0, 90.0],
                                        "coordinates": [
                                            {
                                                "coordinate_id": "time",
                                                "minimum_value": 1262304000000,
                                                "maximum_value": 1735603200000,
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    )
    rec = slim_marine_record(_make_sdk_product(), dataset)
    assert rec["temporal_extent"] == {
        "start": "2010-01-01T00:00:00Z",
        "end": "2024-12-31T00:00:00Z",
    }


def test_slim_marine_record_temporal_extent_unions_across_variables() -> None:
    """T-CMEMS-HIER-001: multiple variables with disjoint time ranges
    aggregate to min of mins, max of maxes — same pattern as
    ``_union_bboxes`` for spatial."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    dataset = _make_sdk_dataset(
        versions=[
            {
                "label": "202411",
                "parts": [
                    {
                        "name": "default",
                        "services": [
                            {
                                "service_name": "files",
                                "variables": [
                                    {
                                        "short_name": "v1",
                                        "coordinates": [
                                            {
                                                "coordinate_id": "time",
                                                "minimum_value": 1577836800000,  # 2020-01-01
                                                "maximum_value": 1640995200000,  # 2022-01-01
                                            }
                                        ],
                                    },
                                    {
                                        "short_name": "v2",
                                        "coordinates": [
                                            {
                                                "coordinate_id": "time",
                                                "minimum_value": 1609459200000,  # 2021-01-01
                                                "maximum_value": 1704067200000,  # 2024-01-01
                                            }
                                        ],
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    )
    rec = slim_marine_record(_make_sdk_product(), dataset)
    assert rec["temporal_extent"] == {
        "start": "2020-01-01T00:00:00Z",
        "end": "2024-01-01T00:00:00Z",
    }


def test_slim_marine_record_temporal_extent_falls_back_to_values_list() -> None:
    """T-CMEMS-HIER-001: when min/max are null but ``values`` is a
    non-empty list of timestamps, derive start/end from the list.
    Empirically the OMI Sea Ice Extent dataset behaves this way
    (smoke 2026-05-13)."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    dataset = _make_sdk_dataset(
        versions=[
            {
                "label": "202411",
                "parts": [
                    {
                        "name": "default",
                        "services": [
                            {
                                "service_name": "files",
                                "variables": [
                                    {
                                        "short_name": "siextents",
                                        "coordinates": [
                                            {
                                                "coordinate_id": "time",
                                                "minimum_value": None,
                                                "maximum_value": None,
                                                "values": [
                                                    725846400000,  # 1993-01-01
                                                    1735603200000,  # 2024-12-31
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    )
    rec = slim_marine_record(_make_sdk_product(), dataset)
    assert rec["temporal_extent"] is not None
    assert rec["temporal_extent"]["start"] == "1993-01-01T00:00:00Z"
    assert rec["temporal_extent"]["end"] == "2024-12-31T00:00:00Z"


def test_slim_marine_record_temporal_extent_ignores_non_time_coordinates() -> None:
    """T-CMEMS-HIER-001: ``coordinate_id`` other than ``"time"``
    (e.g. ``"depth"``, ``"latitude"``) is skipped, not aggregated."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    dataset = _make_sdk_dataset(
        versions=[
            {
                "label": "202411",
                "parts": [
                    {
                        "name": "default",
                        "services": [
                            {
                                "service_name": "files",
                                "variables": [
                                    {
                                        "short_name": "v",
                                        "coordinates": [
                                            {
                                                "coordinate_id": "depth",
                                                "minimum_value": 0.0,
                                                "maximum_value": 5500.0,
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    )
    rec = slim_marine_record(_make_sdk_product(), dataset)
    assert rec["temporal_extent"] is None


# ---------------------------------------------------------------------------
# Description truncation (FINDINGS LOW-2)
# ---------------------------------------------------------------------------


def test_slim_marine_record_description_passthrough_when_short() -> None:
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    short = "A short product description, well under the cap."
    rec = slim_marine_record(_make_sdk_product(description=short), _make_sdk_dataset())
    assert rec["description"] == short


def test_slim_marine_record_description_capped_with_word_boundary_ellipsis() -> None:
    """FINDINGS LOW-2: descriptions > 500 chars are cut at the last
    word boundary <= 497 chars and "..." is appended. Total <= 500."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    long_desc = ("word " * 200).strip()  # ~999 chars, all word boundaries
    rec = slim_marine_record(_make_sdk_product(description=long_desc), _make_sdk_dataset())
    assert len(rec["description"]) <= 500
    assert rec["description"].endswith("...")
    # Word-boundary: shouldn't end mid-word (no chopped "wor...")
    assert not rec["description"][:-3].rstrip().endswith("wor")


def test_slim_marine_record_description_exactly_500_chars_passthrough() -> None:
    """Boundary case from cr round-3 LOW: a description of exactly 500
    chars passes through verbatim, no ellipsis. (500 is the cap, not
    "<= 499 or ellipsis".)"""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    exact = "x" * 500
    rec = slim_marine_record(_make_sdk_product(description=exact), _make_sdk_dataset())
    assert rec["description"] == exact
    assert len(rec["description"]) == 500


# ---------------------------------------------------------------------------
# build_slim_catalogue — walks a full SDK response
# ---------------------------------------------------------------------------


def test_build_slim_catalogue_returns_one_record_per_dataset() -> None:
    """The SDK response is ``{"products": [{..., "datasets": [...]}]}``;
    the slim catalogue flattens to a single list of dataset rows
    (product context attached to each)."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        build_slim_catalogue,
    )

    sdk_response = {
        "products": [
            {
                **_make_sdk_product(),
                "datasets": [
                    _make_sdk_dataset(dataset_id="ds-a"),
                    _make_sdk_dataset(dataset_id="ds-b"),
                ],
            },
            {
                **_make_sdk_product(product_id="PROD_X", title="Other"),
                "datasets": [_make_sdk_dataset(dataset_id="ds-c")],
            },
        ]
    }

    catalogue = build_slim_catalogue(sdk_response)
    assert isinstance(catalogue, list)
    assert [r["dataset_id"] for r in catalogue] == ["ds-a", "ds-b", "ds-c"]


def test_build_slim_catalogue_handles_empty_products_list() -> None:
    from copernicus_mcp.backends.cmems._catalogue_build import (
        build_slim_catalogue,
    )

    assert build_slim_catalogue({"products": []}) == []


# ---------------------------------------------------------------------------
# Sanitiser self-check (sub-plan T-CMEMS-CAT-001 + codex spec-review LOW)
# ---------------------------------------------------------------------------


def test_assert_no_credential_leak_passes_on_clean_records() -> None:
    """The refresh script runs ``Sanitiser.sanitise`` over the slim
    list as a self-check. On a clean snapshot the post-sanitise JSON
    must be byte-identical to the pre-sanitise JSON, else the script
    fails rather than commit a leaky snapshot."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        assert_no_credential_leak,
    )

    records = [slim_record_for_test() for _ in range(3)]
    # Should NOT raise.
    assert_no_credential_leak(records)


def test_assert_no_credential_leak_raises_on_bearer_token() -> None:
    """A record carrying a Bearer-shaped token in its description must
    trip the sanitiser pass and abort the refresh."""
    import pytest

    from copernicus_mcp.backends.cmems._catalogue_build import (
        assert_no_credential_leak,
    )

    leaky = slim_record_for_test()
    leaky["description"] = "Bearer abc123def456ghi789jkl012mno345pqr678"
    with pytest.raises(RuntimeError, match="credential"):
        assert_no_credential_leak([leaky])


def slim_record_for_test() -> dict[str, Any]:
    """Helper — a minimal slim record useful for sanitiser tests."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        slim_marine_record,
    )

    return slim_marine_record(_make_sdk_product(), _make_sdk_dataset())


# ---------------------------------------------------------------------------
# Atomic snapshot write
# ---------------------------------------------------------------------------


def test_write_snapshot_writes_both_marine_and_fetched_at(tmp_path) -> None:
    """The refresh script writes ``marine.json`` (the slim list) and
    ``fetched_at.json`` (UTC ISO timestamp). Both files appear in the
    target directory."""
    import json

    from copernicus_mcp.backends.cmems._catalogue_build import (
        write_snapshot,
    )

    records = [slim_record_for_test()]
    write_snapshot(
        {
            "marine.json": records,
            "fetched_at.json": {"marine": "2026-05-13T18:00:00Z"},
        },
        data_dir=tmp_path,
    )

    marine_path = tmp_path / "marine.json"
    fetched_path = tmp_path / "fetched_at.json"
    assert marine_path.exists()
    assert fetched_path.exists()

    loaded = json.loads(marine_path.read_text())
    assert isinstance(loaded, list)
    assert loaded[0]["dataset_id"] == records[0]["dataset_id"]

    ts = json.loads(fetched_path.read_text())
    # Snapshot stores ``{"marine": "<iso>"}`` to mirror the CDS multi-store shape.
    assert ts == {"marine": "2026-05-13T18:00:00Z"}


def test_write_snapshot_does_not_leave_tempfiles(tmp_path) -> None:
    """Atomic write uses tempfile + os.replace. After a successful
    write only the two final files should exist — no ``.tmp`` orphans."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        write_snapshot,
    )

    write_snapshot(
        {
            "marine.json": [slim_record_for_test()],
            "fetched_at.json": {"marine": "2026-05-13T18:00:00Z"},
        },
        data_dir=tmp_path,
    )

    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert leftovers == [], f"tempfile orphans: {leftovers!r}"


# ---------------------------------------------------------------------------
# Round-1 codex MEDIUMs
# ---------------------------------------------------------------------------


def test_write_snapshot_restores_previous_marine_if_fetched_at_fails(tmp_path, monkeypatch) -> None:
    """codex round-1 M2: ``write_snapshot`` writes two files. If
    ``fetched_at.json`` fails after ``marine.json`` was already
    replaced, we end up with new marine + stale fetched_at — the
    "previous snapshot left untouched" claim is false in that path.

    The fix: keep a backup of the existing ``marine.json`` before
    the new write, and restore it if the second write fails."""
    import json

    import pytest

    from copernicus_mcp.backends.cmems import _catalogue_build as cb

    # Seed the data_dir with an existing snapshot we want preserved.
    previous = [{"dataset_id": "ds-prev", "title": "previous"}]
    (tmp_path / "marine.json").write_text(json.dumps(previous))
    (tmp_path / "fetched_at.json").write_text('{"marine": "old"}')

    # Patch the atomic-write helper so it succeeds for marine.json
    # and FAILS for fetched_at.json.
    real_atomic = cb._atomic_write_json
    call_count = {"n": 0}

    def flaky(path, payload):
        call_count["n"] += 1
        if call_count["n"] == 1:
            real_atomic(path, payload)
        else:
            raise OSError("disk full (simulated)")

    monkeypatch.setattr(cb, "_atomic_write_json", flaky)

    with pytest.raises(OSError):
        cb.write_snapshot(
            {
                "marine.json": [{"dataset_id": "ds-new", "title": "new"}],
                "fetched_at.json": {"marine": "2026-05-13T18:00:00Z"},
            },
            data_dir=tmp_path,
        )

    # The previous marine.json must be restored — the new write
    # is not visible.
    loaded = json.loads((tmp_path / "marine.json").read_text())
    assert loaded == previous, f"marine.json was clobbered by a failed two-file write: {loaded!r}"
    # fetched_at.json keeps its old value because we never replaced it.
    fetched = json.loads((tmp_path / "fetched_at.json").read_text())
    assert fetched == {"marine": "old"}


def test_write_snapshot_first_run_no_previous_no_marine_left_on_failure(
    tmp_path, monkeypatch
) -> None:
    """codex round-1 M2 edge case: first-time refresh (no previous
    snapshot). If fetched_at.json write fails, marine.json should be
    cleaned up rather than left as an orphan claiming to be a fresh
    snapshot."""
    import pytest

    from copernicus_mcp.backends.cmems import _catalogue_build as cb

    real_atomic = cb._atomic_write_json
    call_count = {"n": 0}

    def flaky(path, payload):
        call_count["n"] += 1
        if call_count["n"] == 1:
            real_atomic(path, payload)
        else:
            raise OSError("disk full (simulated)")

    monkeypatch.setattr(cb, "_atomic_write_json", flaky)

    with pytest.raises(OSError):
        cb.write_snapshot(
            {
                "marine.json": [{"dataset_id": "ds-new"}],
                "fetched_at.json": {"marine": "2026-05-13T18:00:00Z"},
            },
            data_dir=tmp_path,
        )

    # No previous file existed → orphan marine.json must be removed.
    assert not (tmp_path / "marine.json").exists()


# ---------------------------------------------------------------------------
# build_slim_catalogue input validation (codex round-1 M1)
# ---------------------------------------------------------------------------


def test_build_slim_catalogue_raises_on_non_list_products() -> None:
    """codex round-1 M1: the script's shape guard checks
    ``"products" in sdk_response`` only — a malformed
    ``{"products": "error"}`` or ``{"products": null}`` would
    pass the guard and produce an empty snapshot that overwrites the
    existing one. Make ``build_slim_catalogue`` itself enforce the
    list-shape so the refresh script can rely on that signal."""
    import pytest

    from copernicus_mcp.backends.cmems._catalogue_build import (
        build_slim_catalogue,
    )

    with pytest.raises(ValueError, match="products"):
        build_slim_catalogue({"products": "error"})
    with pytest.raises(ValueError, match="products"):
        build_slim_catalogue({"products": None})
    with pytest.raises(ValueError, match="products"):
        build_slim_catalogue({"products": {}})
    # Existing "empty list" semantics still work — that's a real case
    # for a future SDK call against a user with zero entitlements.
    assert build_slim_catalogue({"products": []}) == []
    # Missing key is also a real error.
    with pytest.raises(ValueError, match="products"):
        build_slim_catalogue({})


def test_write_snapshot_restore_is_atomic_on_failure(tmp_path, monkeypatch) -> None:
    """codex round-2 MEDIUM: the round-1 restore path used
    ``marine_path.write_bytes(marine_backup)`` — NOT atomic. A crash
    during the restore would leave ``marine.json`` truncated.

    We pin TWO invariants the new ``_atomic_write_bytes`` path
    provides:
      1. The on-disk ``marine.json`` matches the previous content
         byte-identical (rollback succeeded).
      2. No ``.tmp`` orphans remain in the data dir — the restore
         used tempfile + ``os.replace``, not an in-place rewrite.
    """
    import json

    import pytest

    from copernicus_mcp.backends.cmems import _catalogue_build as cb

    previous = [{"dataset_id": "ds-prev"}]
    marine_path = tmp_path / "marine.json"
    marine_path.write_text(json.dumps(previous))
    fetched_path = tmp_path / "fetched_at.json"
    fetched_path.write_text('{"marine": "old"}')

    real_atomic = cb._atomic_write_json
    call_count = {"n": 0}

    def flaky(path, payload):
        call_count["n"] += 1
        if call_count["n"] == 1:
            real_atomic(path, payload)  # forward marine.json write
        else:
            raise OSError("disk full (simulated, fetched_at.json)")

    monkeypatch.setattr(cb, "_atomic_write_json", flaky)

    with pytest.raises(OSError):
        cb.write_snapshot(
            {
                "marine.json": [{"dataset_id": "ds-new"}],
                "fetched_at.json": {"marine": "2026-05-13T18:00:00Z"},
            },
            data_dir=tmp_path,
        )

    # 1. Restore landed: on-disk content matches the previous backup.
    assert json.loads(marine_path.read_text()) == previous
    # 2. No tempfile orphans in the data dir — the restore was atomic.
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert leftovers == [], f"tempfile orphans after restore: {leftovers!r}"


def test_atomic_write_json_cleans_up_tempfile_when_write_fails(tmp_path, monkeypatch) -> None:
    """codex round-2 LOW-2: exercise the cleanup branch INSIDE
    ``_atomic_write_json`` — the one that fires after ``mkstemp``
    succeeded but the write/fsync/replace failed. Pre-mkstemp errors
    (e.g. JSON-serialisation) leave nothing to clean up; this test
    pins the post-mkstemp orphan-cleanup branch."""
    import os as _os

    import pytest

    from copernicus_mcp.backends.cmems._catalogue_build import (
        _atomic_write_json,
    )

    target = tmp_path / "out.json"

    def flaky_fdopen(fd, *args, **kwargs):
        # Close the real fd to avoid a leak, then raise so the
        # ``except Exception`` branch in _atomic_write_json runs.
        _os.close(fd)
        raise OSError("simulated write failure after mkstemp")

    monkeypatch.setattr(_os, "fdopen", flaky_fdopen)

    with pytest.raises(OSError, match="simulated write failure"):
        _atomic_write_json(target, {"k": "v"})

    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert leftovers == [], f"tempfile orphans: {leftovers!r}"


# ---------------------------------------------------------------------------
# T-CMEMS-HIER-002: generic N-file atomic writer
# ---------------------------------------------------------------------------


def test_write_snapshot_writes_all_files_in_mapping(tmp_path) -> None:
    """Generic N-file writer: every key in the mapping becomes a
    JSON file under ``data_dir`` with the value as payload."""
    import json

    from copernicus_mcp.backends.cmems._catalogue_build import (
        write_snapshot,
    )

    files = {
        "marine.json": [{"dataset_id": "ds-a"}],
        "dataset_cards.json": [{"dataset_id": "ds-a", "domain": "physics"}],
        "fetched_at.json": {"marine": "2026-05-13T18:00:00Z"},
    }
    write_snapshot(files, data_dir=tmp_path)

    for name, expected in files.items():
        path = tmp_path / name
        assert path.exists(), f"{name} missing"
        assert json.loads(path.read_text()) == expected


def test_write_snapshot_empty_mapping_is_noop(tmp_path) -> None:
    """An empty files mapping must not touch the directory or fail."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        write_snapshot,
    )

    write_snapshot({}, data_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "bad_key",
    [
        # codex round-1 PR #88 MEDIUM
        "../escape.json",
        "/etc/passwd",
        "a/b.json",
        "..",
        ".",
        "",
        "./marine.json",
        # codex round-2 PR #88 MEDIUM — Windows drive-relative.
        "C:foo.json",
        "C:\\tmp\\x.json",
        # cr round-2 PR #88 MEDIUM — control chars, whitespace, backslash.
        "foo\x00.json",
        "foo\nbar.json",
        "foo\\bar.json",
        " foo.json",
        "foo.json ",
    ],
)
def test_write_snapshot_rejects_non_filename_keys(tmp_path, bad_key) -> None:
    """codex round-1 PR #88 MEDIUM (and round-2 widening): keys must
    be plain filenames so rollback's ``os.replace`` / ``unlink``
    cannot touch files outside ``data_dir``. Anything else raises
    ``ValueError`` before any write attempt."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        write_snapshot,
    )

    with pytest.raises(ValueError):
        write_snapshot({bad_key: [1, 2, 3]}, data_dir=tmp_path)

    # And nothing was written under data_dir.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "good_key",
    [
        "marine.json",
        "dataset_cards.json",
        "fetched_at.json",
        "foo.bar.baz.json",
        "a-b_c.json",
        ".hidden.json",
        "UPPER.JSON",
    ],
)
def test_write_snapshot_accepts_legitimate_filenames(tmp_path, good_key) -> None:
    """Round-2: the widened validator must not have over-rejected. Pin
    that ordinary filenames (mixed case, hidden dot-prefix, multi-dot
    extensions, dashes, underscores) still go through."""
    from copernicus_mcp.backends.cmems._catalogue_build import (
        write_snapshot,
    )

    write_snapshot({good_key: [1, 2, 3]}, data_dir=tmp_path)
    assert (tmp_path / good_key).is_file()


@pytest.mark.parametrize("fail_index", [0, 1, 2])
def test_write_snapshot_rolls_back_all_files_when_any_write_fails(
    tmp_path, monkeypatch, fail_index
) -> None:
    """T-CMEMS-HIER-002 atomicity invariant: if the write at index
    ``fail_index`` raises, every previously-replaced file is restored
    to its pre-call content, and any new file (one with no previous
    version) is removed. The original error propagates."""
    import json

    from copernicus_mcp.backends.cmems import _catalogue_build as cb

    # Seed two of the three target files with previous content. The
    # third ("dataset_cards.json") has no previous version — rollback
    # must delete it if it was already written.
    (tmp_path / "marine.json").write_text(json.dumps([{"dataset_id": "old-marine"}]))
    (tmp_path / "fetched_at.json").write_text('{"marine": "old"}')

    files = {
        "marine.json": [{"dataset_id": "new-marine"}],
        "dataset_cards.json": [{"dataset_id": "new-card"}],
        "fetched_at.json": {"marine": "2026-05-13T18:00:00Z"},
    }

    real_atomic = cb._atomic_write_json
    call_count = {"n": 0}

    def flaky(path, payload):
        idx = call_count["n"]
        call_count["n"] += 1
        if idx == fail_index:
            raise OSError(f"simulated failure at write index {idx}")
        real_atomic(path, payload)

    monkeypatch.setattr(cb, "_atomic_write_json", flaky)

    with pytest.raises(OSError, match=f"index {fail_index}"):
        cb.write_snapshot(files, data_dir=tmp_path)

    # The two pre-existing files must be byte-identical to their
    # pre-call content (rollback succeeded).
    assert json.loads((tmp_path / "marine.json").read_text()) == [{"dataset_id": "old-marine"}]
    assert json.loads((tmp_path / "fetched_at.json").read_text()) == {"marine": "old"}
    # The new file (no previous version) must NOT exist after rollback.
    assert not (tmp_path / "dataset_cards.json").exists()
    # No tempfile orphans from the rollback either.
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert leftovers == [], f"tempfile orphans after rollback: {leftovers!r}"
