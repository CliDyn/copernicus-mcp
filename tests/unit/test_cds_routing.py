"""Per-store URL routing tests for the CDS backend (T-CDS-011).

The CDS backend uses a single ``cdsapi.Client``, but the actual HTTP
endpoint differs per Data Store: CDS, ADS, EWDS. Catalogue snapshot
(T-CDS-003) tags each dataset with ``store``; this module verifies that
the backend picks the correct URL when constructing the cdsapi client.

Closes a scope gap in T-CDS-005: that PR shipped submit/poll/cancel/
fetch against the CDS endpoint only. ADS / EWDS datasets surfaced in
search and describe but submit returned HTTP 404 (caught only by
post-v0.3.0 hands-on testing — see ``era5_session_report_2026-05-13.md``).
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# _endpoint_url_for: pure-function lookup
# ---------------------------------------------------------------------------


def test_endpoint_url_for_cds_dataset() -> None:
    from copernicus_mcp.backends.cds.backend import _endpoint_url_for

    url = _endpoint_url_for(
        "reanalysis-era5-single-levels",
        default="https://default.example/api",
    )
    assert url == "https://cds.climate.copernicus.eu/api"


def test_endpoint_url_for_ads_dataset() -> None:
    """CAMS lives on ADS — must NOT route to CDS endpoint.

    This is the bug user surfaced via real-PAT smoke against
    ``cams-global-reanalysis-eac4``: v0.3.0 sent submit to
    ``cds.climate.copernicus.eu`` and got 404 because that processes
    catalogue does not know the CAMS id.
    """
    from copernicus_mcp.backends.cds.backend import _endpoint_url_for

    url = _endpoint_url_for(
        "cams-global-reanalysis-eac4",
        default="https://default.example/api",
    )
    assert url == "https://ads.atmosphere.copernicus.eu/api"


def test_routing_table_matches_catalogue_stores() -> None:
    """T-CDS-011 cr round-2 M1: pin the drift CR round-1 M1 actually
    described — catalogue snapshot stores vs routing table.

    Earlier the pin asserted ``_RUNTIME_SUPPORTED_STORES ==
    frozenset(_STORE_ENDPOINT_URLS.keys())`` which is tautological by
    construction (the LHS is defined as exactly the RHS). The real
    risk is: catalogue grows to include CDSE/WEkEO collections before
    the runtime supports them, ``runtime_compatible`` over-reports
    ``true`` and submits 404. Assert against ``catalogue._STORES`` so
    a catalogue update without a matching routing-table update fails
    the test instead of silently regressing.
    """
    from copernicus_mcp.backends.cds.backend import _STORE_ENDPOINT_URLS
    from copernicus_mcp.backends.cds.catalogue import _STORES

    assert frozenset(_STORES) == frozenset(_STORE_ENDPOINT_URLS.keys()), (
        f"catalogue stores {_STORES} drifted from routing table "
        f"{tuple(_STORE_ENDPOINT_URLS.keys())}; add the new store to "
        "_STORE_ENDPOINT_URLS or revert the catalogue addition"
    )


def test_runtime_supports_helper_known_and_unknown() -> None:
    from copernicus_mcp.backends.cds.backend import runtime_supports

    # CDS dataset — supported.
    assert runtime_supports("reanalysis-era5-single-levels") is True
    # ADS dataset — supported.
    assert runtime_supports("cams-global-reanalysis-eac4") is True
    # Unknown — not supported.
    assert runtime_supports("not-a-real-dataset-xyzzy") is False


def test_endpoint_url_for_unknown_dataset_falls_back_to_default() -> None:
    """An id not in the bundled catalogue snapshot returns the default —
    chosen by the caller (typically the adapter's configured URL).
    Defensive: a new catalogue version that drops an old id should not
    break running clients; they retry against their default and surface
    whatever the server returns (most likely a 404 with the right error
    class)."""
    from copernicus_mcp.backends.cds.backend import _endpoint_url_for

    url = _endpoint_url_for(
        "not-a-real-dataset-xyzzy",
        default="https://fallback.example/api",
    )
    assert url == "https://fallback.example/api"


# ---------------------------------------------------------------------------
# CdsBackend.submit / check_status / cancel route through the right URL.
# Each test stubs cdsapi.Client and asserts the constructor's ``url=`` kwarg.
# Re-uses the helpers from test_cds_submit_check_cancel.py via direct
# import to avoid duplicating fixtures.
# ---------------------------------------------------------------------------


import sys  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from test_cds_submit_check_cancel import (  # noqa: E402
    _fake_creds,
    _fake_remote,
    _good_params,
    _patch_cdsapi,
    foundation,  # noqa: F401  (pytest fixture re-export)
)


@pytest.mark.asyncio
async def test_submit_for_cds_dataset_uses_cds_endpoint(
    foundation,  # noqa: F811  — re-exported pytest_asyncio fixture
    monkeypatch,
) -> None:
    """Default happy path — ``reanalysis-era5-single-levels`` is a CDS
    dataset, so the constructed cdsapi.Client must point at the CDS API."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    fake_class, _ = _patch_cdsapi(
        monkeypatch, retrieve_returns=_fake_remote("req-cds")
    )
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await backend.submit(_good_params())

    assert (
        fake_class.call_args.kwargs["url"]
        == "https://cds.climate.copernicus.eu/api"
    )


@pytest.mark.asyncio
async def test_submit_for_ads_dataset_uses_ads_endpoint(
    foundation,  # noqa: F811  — re-exported pytest_asyncio fixture
    monkeypatch,
) -> None:
    """The bug user surfaced: submit for CAMS (ADS) v0.3.0 hits the CDS
    API and returns 404. After T-CDS-011, it must route to ADS."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    fake_class, _ = _patch_cdsapi(
        monkeypatch, retrieve_returns=_fake_remote("req-ads")
    )
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    ads_params = {
        "dataset_id": "cams-global-reanalysis-eac4",
        "inputs": {
            "variable": ["total_column_ozone"],
            "date": ["2024-01-01/2024-01-01"],
            "time": ["00:00"],
            "data_format": "grib",
        },
    }
    await backend.submit(ads_params)

    assert (
        fake_class.call_args.kwargs["url"]
        == "https://ads.atmosphere.copernicus.eu/api"
    )


@pytest.mark.asyncio
async def test_poll_for_ads_workflow_row_uses_ads_endpoint(
    foundation,  # noqa: F811
    monkeypatch,
) -> None:
    """codex round-1 L2: poll path recovers dataset_id from
    workflow row's request_json. For an ADS-originated row it must
    construct the cdsapi client against the ADS endpoint, not CDS."""
    import json as _json
    from datetime import UTC, datetime

    from copernicus_mcp.backends.cds.backend import CdsBackend

    iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    await foundation.persistence.record_workflow(
        {
            "request_id": "rid-poll-ads",
            "backend_id": "cds",
            "operation": "submit",
            "status": "running",
            "cache_key": "ck-poll-ads",
            "request_json": _json.dumps(
                {
                    "dataset_id": "cams-global-reanalysis-eac4",
                    "inputs": {"variable": ["x"]},
                }
            ),
            "response_json": None,
            "error_record_json": None,
            "created_at": iso,
            "updated_at": iso,
        }
    )
    fake_class, _ = _patch_cdsapi(monkeypatch, get_remote_json={"status": "running"})

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await backend.check_status("rid-poll-ads")

    assert (
        fake_class.call_args.kwargs["url"]
        == "https://ads.atmosphere.copernicus.eu/api"
    )


@pytest.mark.asyncio
async def test_cancel_for_ads_workflow_row_uses_ads_endpoint(
    foundation,  # noqa: F811
    monkeypatch,
) -> None:
    """codex round-1 L2: cancel against an ADS-originated workflow
    row must construct the cdsapi client with the ADS endpoint URL
    so the SDK ``delete`` call hits the right server."""
    import json as _json
    from datetime import UTC, datetime

    from copernicus_mcp.backends.cds.backend import CdsBackend

    iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    await foundation.persistence.record_workflow(
        {
            "request_id": "rid-cancel-ads",
            "backend_id": "cds",
            "operation": "submit",
            "status": "queued",
            "cache_key": "ck-cancel-ads",
            "request_json": _json.dumps(
                {
                    "dataset_id": "cams-global-reanalysis-eac4",
                    "inputs": {"variable": ["x"]},
                }
            ),
            "response_json": None,
            "error_record_json": None,
            "created_at": iso,
            "updated_at": iso,
        }
    )
    fake_class, _ = _patch_cdsapi(monkeypatch)

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await backend.cancel("rid-cancel-ads")

    assert (
        fake_class.call_args.kwargs["url"]
        == "https://ads.atmosphere.copernicus.eu/api"
    )
