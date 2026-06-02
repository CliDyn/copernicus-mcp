"""``CdsBackend`` Terms-of-use elicitation tests (T-CDS-006).

The CDS / ADS / EWDS server returns a per-dataset T&C-not-accepted
error when the user has not yet accepted every required licence on
the dataset's web page. The empirical error shape (T-CDS-000 smoke
F-2 in 2026) differs from the form documented in
``the project research notes`` §6.6.2; we parse the actual
HTTP 403 body:

    user didn't accept all required site policies
    Missing policies are: Data protection and privacy statement
        (rev. 1) - https://ads.atmosphere.copernicus.eu/licences/...,
        Terms of use of the Copernicus Atmosphere Data Store (rev. 1)
        - https://ads.atmosphere.copernicus.eu/licences/...

Backend behaviour:

- ``submit``: detect T&C in the SDK exception, raise canonical
  ``TermsNotAcceptedError`` with ``recovery_url`` set to the first
  missing-policy URL and ``context.missing_policies`` listing all
  parsed entries.
- ``check_status``: when ``remote_json["status"] == "failed"`` and
  the error message matches the T&C marker, persist the row as
  ``failed`` with the canonical ``TermsNotAcceptedError`` record so
  the polling caller observes the structured envelope.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio


def _make_foundation(tmp_path: Path):
    from copernicus_mcp.auth import CredentialResolver
    from copernicus_mcp.backends.abstract import FoundationServices
    from copernicus_mcp.cache import CacheManager
    from copernicus_mcp.config import ConfigLoader
    from copernicus_mcp.data_model.coordinator import DataModelCoordinator
    from copernicus_mcp.data_model.provenance import ProvenanceRecorder
    from copernicus_mcp.errors.sanitiser import Sanitiser
    from copernicus_mcp.http import HttpClientFactory
    from copernicus_mcp.persistence import SqliteBackend

    config = ConfigLoader().load()
    persistence = SqliteBackend(tmp_path / "state.db")
    cache = CacheManager(
        cache_directory=tmp_path / "cache",
        persistence=persistence,
        size_limit_bytes=10 * 1024 * 1024,
    )
    return (
        FoundationServices(
            config=config,
            credential_resolver=CredentialResolver(),
            http_client_factory=HttpClientFactory(http_config=config.http),
            persistence=persistence,
            cache=cache,
            sanitiser=Sanitiser(),
            data_model=DataModelCoordinator(persistence=persistence),
            provenance=ProvenanceRecorder(
                persistence=persistence,
                software_versions={"copernicus-mcp": "0.0.1"},
            ),
        ),
        persistence,
    )


@pytest_asyncio.fixture
async def foundation(tmp_path: Path):
    found, persistence = _make_foundation(tmp_path)
    await persistence.initialise()
    try:
        yield found
    finally:
        await persistence.close()


def _fake_creds():
    from copernicus_mcp.auth.resolver import ResolvedCredentials

    return ResolvedCredentials(
        backend="cds",
        source="explicit",
        source_detail="test",
        fields={"key": "abcdef01-2345-6789-abcd-ef0123456789"},
    )


def _good_params() -> dict[str, Any]:
    return {
        "dataset_id": "cams-global-reanalysis-eac4",
        "inputs": {"variable": ["x"], "year": ["2024"]},
    }


def _patch_cdsapi(monkeypatch, retrieve_side_effect=None, get_remote_json=None):
    import sys
    import types

    fake_module = types.ModuleType("cdsapi")
    instance = MagicMock()
    instance.retrieve = MagicMock(side_effect=retrieve_side_effect)
    inner = MagicMock()
    poll_remote = MagicMock()
    poll_remote.json = get_remote_json or {"status": "running"}
    inner.get_remote = MagicMock(return_value=poll_remote)
    inner.delete = MagicMock(return_value={"deleted": True})
    inner.download_results = MagicMock()
    instance.client = inner
    fake_class = MagicMock(return_value=instance)
    fake_module.Client = fake_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cdsapi", fake_module)
    return fake_class, instance


_REAL_TC_MESSAGE = (
    "403 Client Error: Forbidden for url: "
    "https://ads.atmosphere.copernicus.eu/api/retrieve/v1/processes/"
    "cams-global-reanalysis-eac4/execution\n"
    "user didn't accept all required site policies\n"
    "Missing policies are: Data protection and privacy statement (rev. 1) - "
    "https://ads.atmosphere.copernicus.eu/licences/ads-data-protection-privacy-statement, "
    "Terms of use of the Copernicus Atmosphere Data Store (rev. 1) - "
    "https://ads.atmosphere.copernicus.eu/licences/terms-of-use-ads"
)


# ---------------------------------------------------------------------------
# Parser unit tests
# ---------------------------------------------------------------------------


def test_parse_terms_not_accepted_returns_none_on_unrelated_message() -> None:
    from copernicus_mcp.backends.cds.backend import _parse_terms_not_accepted

    assert _parse_terms_not_accepted("HTTP 500 internal server error") is None
    assert _parse_terms_not_accepted("connection refused") is None


def test_parse_terms_not_accepted_extracts_two_policies() -> None:
    from copernicus_mcp.backends.cds.backend import _parse_terms_not_accepted

    parsed = _parse_terms_not_accepted(_REAL_TC_MESSAGE)
    assert parsed is not None
    assert len(parsed) == 2
    assert parsed[0]["name"] == "Data protection and privacy statement"
    assert parsed[0]["rev"] == 1
    assert parsed[0]["url"].endswith("ads-data-protection-privacy-statement")
    assert parsed[1]["name"] == "Terms of use of the Copernicus Atmosphere Data Store"
    assert parsed[1]["url"].endswith("terms-of-use-ads")


# ---------------------------------------------------------------------------
# T-CDS-012 — EWDS uses a different T&C error wording than CDS/ADS.
# Captured empirically against the real EWDS server on 2026-05-13
# (efas-historical submit before licence acceptance):
#
#   403 Client Error: Forbidden for url:
#     https://ewds.climate.copernicus.eu/api/retrieve/v1/processes/efas-historical/execution
#   required licences not accepted
#   Not all the required licences have been accepted; please visit
#   https://ewds.climate.copernicus.eu/datasets/efas-historical?tab=download#manage-licences
#   to accept the required licence(s).
#
# Differences from the CDS/ADS shape (research §6.6.2 + T-CDS-000):
#   - marker is "required licences not accepted" (British spelling) instead
#     of "user didn't accept all required site policies"
#   - one recovery URL is embedded inline in the prose; no per-policy
#     (name, rev, URL) tuples to extract
# ---------------------------------------------------------------------------

_REAL_EWDS_TC_MESSAGE = (
    "403 Client Error: Forbidden for url: "
    "https://ewds.climate.copernicus.eu/api/retrieve/v1/processes/"
    "efas-historical/execution\n"
    "required licences not accepted\n"
    "Not all the required licences have been accepted; please visit "
    "https://ewds.climate.copernicus.eu/datasets/efas-historical?"
    "tab=download#manage-licences to accept the required licence(s)."
)


def test_parse_terms_not_accepted_ewds_extracts_inline_url() -> None:
    """T-CDS-012: EWDS T&C error must be recognised and the inline
    licence-management URL extracted as the single policy."""
    from copernicus_mcp.backends.cds.backend import _parse_terms_not_accepted

    parsed = _parse_terms_not_accepted(_REAL_EWDS_TC_MESSAGE)
    assert parsed is not None, "EWDS marker not recognised — falls through to generic BackendError"
    assert len(parsed) == 1
    assert (
        parsed[0]["url"]
        == "https://ewds.climate.copernicus.eu/datasets/efas-historical"
        "?tab=download#manage-licences"
    )


def test_parse_terms_not_accepted_ewds_marker_only_returns_empty_list() -> None:
    """T-CDS-012 cr LOW-2: marker present but URL not extractable —
    parser still recognises this as a T&C error (returns ``[]``,
    caller falls back to the EWDS landing page detected by the host
    in ``Forbidden for url:``)."""
    from copernicus_mcp.backends.cds.backend import _parse_terms_not_accepted

    msg = (
        "403 Client Error: Forbidden for url: "
        "https://ewds.climate.copernicus.eu/api/x\n"
        "required licences not accepted\n"
        "(no recoverable URL in this variant of the message)"
    )
    assert _parse_terms_not_accepted(msg) == []


def test_parse_terms_not_accepted_ewds_strips_trailing_punctuation() -> None:
    """T-CDS-012 cr LOW-2: empirical EWDS body sometimes drops a
    trailing ``.`` or ``,`` immediately after the URL. The
    ``rstrip(".,")`` step preserves the live URL while not
    breaking the simpler case."""
    from copernicus_mcp.backends.cds.backend import _parse_terms_not_accepted

    msg = (
        "required licences not accepted\n"
        "please visit https://ewds.climate.copernicus.eu/dataset/x. "
        "to accept the required licence(s)."
    )
    parsed = _parse_terms_not_accepted(msg)
    assert parsed is not None and len(parsed) == 1
    assert parsed[0]["url"] == "https://ewds.climate.copernicus.eu/dataset/x"


def test_parse_terms_not_accepted_marker_only_returns_empty_list() -> None:
    """The marker is present but the ``Missing policies are:`` block is
    absent (server returned only the headline). Treat as T&C error
    with no parsed policies — caller still surfaces the canonical
    class."""
    from copernicus_mcp.backends.cds.backend import _parse_terms_not_accepted

    msg = "403 Forbidden\nuser didn't accept all required site policies"
    parsed = _parse_terms_not_accepted(msg)
    assert parsed == []


# ---------------------------------------------------------------------------
# T-CDS-014 — codex retro LOWs on PR #69 (T-CDS-012). Both pin
# pathological-but-real edge cases the EWDS branch didn't handle.
# ---------------------------------------------------------------------------


def test_parse_terms_not_accepted_falls_through_to_cds_when_ewds_url_missing(
) -> None:
    """T-CDS-014 codex LOW-1: when a message contains BOTH the EWDS
    marker (without a parseable inline URL) AND the CDS canonical
    ``Missing policies are:`` block, the parser must NOT short-circuit
    on the EWDS branch and drop the CDS per-policy URLs. EWDS wins
    only when its URL is actually found; otherwise fall through to
    CDS/ADS parsing.

    Empirical likelihood is low (no real-world message captured with
    both shapes), but a defensive deployment that quotes the EWDS
    headline in CDS error prose would lose the actionable URLs."""
    from copernicus_mcp.backends.cds.backend import _parse_terms_not_accepted

    msg = (
        "required licences not accepted\n"
        "user didn't accept all required site policies\n"
        "Missing policies are: CDS terms of use (rev. 2) - "
        "https://cds.climate.copernicus.eu/licences/cds-terms-of-use"
    )
    parsed = _parse_terms_not_accepted(msg)
    assert parsed is not None
    assert len(parsed) == 1
    assert parsed[0]["name"] == "CDS terms of use"
    assert parsed[0]["rev"] == 2
    assert parsed[0]["url"].endswith("cds-terms-of-use")


def test_parse_terms_not_accepted_ewds_strips_semicolon_colon_paren_trailing(
) -> None:
    """T-CDS-014 codex LOW-2: the trailing-punctuation strip on the
    inline EWDS URL handled ``.`` and ``,`` but not ``;``, ``:``,
    ``)``. Empirically rare in EWDS messages today, but a server that
    parenthesises the URL (``please visit (https://x) to accept``)
    or semicolon-joins clauses would leave a junk character on the
    canonical recovery URL — the agent then opens a 404."""
    from copernicus_mcp.backends.cds.backend import _parse_terms_not_accepted

    for trailing in (";", ":", ")"):
        msg = (
            "required licences not accepted\n"
            f"please visit https://ewds.climate.copernicus.eu/dataset/x{trailing} "
            "to accept the required licence(s)."
        )
        parsed = _parse_terms_not_accepted(msg)
        assert parsed is not None and len(parsed) == 1
        assert (
            parsed[0]["url"] == "https://ewds.climate.copernicus.eu/dataset/x"
        ), f"trailing {trailing!r} not stripped"


def test_parse_terms_not_accepted_cds_strips_semicolon_colon_paren_trailing(
) -> None:
    """T-CDS-014 round-1 cr HIGH: LOW-2 broadened the trailing-
    punctuation strip on the EWDS inline URL but missed the same
    rstrip on the CDS canonical ``Missing policies are:`` per-policy
    URLs — the more empirically common code path. A CDS policy URL
    ending in ``;``, ``:``, or ``)`` left junk on the canonical
    recovery URL and the agent opened a 404 on click."""
    from copernicus_mcp.backends.cds.backend import _parse_terms_not_accepted

    for trailing in (";", ":", ")"):
        msg = (
            "user didn't accept all required site policies\n"
            f"Missing policies are: CDS terms of use (rev. 3) - "
            f"https://cds.climate.copernicus.eu/licences/cds-terms-of-use{trailing}"
        )
        parsed = _parse_terms_not_accepted(msg)
        assert parsed is not None and len(parsed) == 1, (
            f"unexpected parse result for trailing {trailing!r}: {parsed!r}"
        )
        assert (
            parsed[0]["url"]
            == "https://cds.climate.copernicus.eu/licences/cds-terms-of-use"
        ), f"trailing {trailing!r} not stripped from CDS canonical URL"


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_raises_canonical_terms_not_accepted_on_403(
    foundation, monkeypatch
) -> None:
    """Real-world T&C-rejected ``client.retrieve`` raises a runtime
    error with the documented body. Backend must surface a canonical
    ``TermsNotAcceptedError`` with structured ``recovery_url`` and
    ``missing_policies`` context — not a generic ``BackendError``."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import TermsNotAcceptedError

    _patch_cdsapi(monkeypatch, retrieve_side_effect=RuntimeError(_REAL_TC_MESSAGE))

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    with pytest.raises(TermsNotAcceptedError) as exc_info:
        await backend.submit(_good_params())

    record = exc_info.value.error_record
    assert record.error_class == "TermsNotAcceptedError"
    assert record.recovery_action == "accept_terms"
    # First missing-policy URL surfaces as the canonical recovery URL.
    assert record.recovery_url is not None
    assert record.recovery_url.startswith("https://")
    # Both policies enumerated for the agent / CLI.
    policies = record.context.get("missing_policies")
    assert isinstance(policies, list)
    assert len(policies) == 2
    assert all("url" in p for p in policies)


@pytest.mark.asyncio
async def test_submit_non_tc_sdk_error_still_wraps_as_backend_error(
    foundation, monkeypatch
) -> None:
    """Don't accidentally classify every SDK exception as T&C —
    non-matching message must still raise the generic
    ``BackendError(sdk_submit_failure)``."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import BackendError

    _patch_cdsapi(
        monkeypatch,
        retrieve_side_effect=RuntimeError("503 Service Unavailable: queue saturated"),
    )

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    with pytest.raises(BackendError) as exc_info:
        await backend.submit(_good_params())
    assert exc_info.value.error_record.error_subclass == "sdk_submit_failure"


# ---------------------------------------------------------------------------
# check_status (failed remote with T&C in message)
# ---------------------------------------------------------------------------


async def _seed_running_row(foundation, request_id: str) -> None:
    now = "2026-05-09T12:00:00Z"
    await foundation.persistence.record_workflow(
        {
            "request_id": request_id,
            "backend_id": "cds",
            "operation": "submit",
            "status": "running",
            "cache_key": "ck-tc",
            "request_json": "{}",
            "response_json": None,
            "error_record_json": None,
            "created_at": now,
            "updated_at": now,
        }
    )


@pytest.mark.asyncio
async def test_check_status_failed_remote_with_tc_persists_canonical_record(
    foundation, monkeypatch
) -> None:
    """If CDS server returns ``status=failed`` with the T&C marker in
    the error message, ``check_status`` must persist the row as
    failed but with a ``TermsNotAcceptedError``-shaped error_record
    (not a generic ``remote_job_failed``). The polling caller
    observes the canonical envelope and can route to the elicitation
    UI."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_running_row(foundation, "rid-tc-poll")
    _patch_cdsapi(
        monkeypatch,
        get_remote_json={
            "status": "failed",
            "error": {"message": _REAL_TC_MESSAGE},
        },
    )

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    out = await backend.check_status("rid-tc-poll")
    assert out["status"] == "failed"

    row = await foundation.persistence.fetch_workflow("rid-tc-poll")
    assert row is not None
    err = json.loads(row["error_record_json"] or "{}")
    assert err.get("error_class") == "TermsNotAcceptedError"
    assert err.get("recovery_action") == "accept_terms"
    assert err.get("recovery_url", "").startswith("https://")
    assert isinstance(err.get("context", {}).get("missing_policies"), list)


# ---------------------------------------------------------------------------
# Round-1 review fixes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_status_failed_remote_with_tc_in_top_level_message(
    foundation, monkeypatch
) -> None:
    """Round-1 HIGH (cr H-1 / codex M-1): _record_terminal must find
    the T&C marker even when CDS surfaces it at top-level
    ``remote_json["message"]`` rather than ``error.message``."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_running_row(foundation, "rid-top-msg")
    _patch_cdsapi(
        monkeypatch,
        get_remote_json={
            "status": "failed",
            "message": _REAL_TC_MESSAGE,  # no ``error`` key at all
        },
    )

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await backend.check_status("rid-top-msg")
    row = await foundation.persistence.fetch_workflow("rid-top-msg")
    assert row is not None
    err = json.loads(row["error_record_json"] or "{}")
    assert err.get("error_class") == "TermsNotAcceptedError"


@pytest.mark.asyncio
async def test_check_status_failed_remote_with_tc_in_error_string(
    foundation, monkeypatch
) -> None:
    """Round-1 HIGH (cr H-1): legacy ecmwf-datastores shape returns
    ``error`` as a bare string (not a dict). Same canonical routing
    must apply."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_running_row(foundation, "rid-err-str")
    _patch_cdsapi(
        monkeypatch,
        get_remote_json={"status": "failed", "error": _REAL_TC_MESSAGE},
    )

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await backend.check_status("rid-err-str")
    row = await foundation.persistence.fetch_workflow("rid-err-str")
    assert row is not None
    err = json.loads(row["error_record_json"] or "{}")
    assert err.get("error_class") == "TermsNotAcceptedError"


def test_parse_terms_not_accepted_handles_extra_prose_between_policies() -> None:
    """Round-1 MEDIUM (cr M-2 / codex M-2): the previous greedy +
    trailing-lookahead regex collapsed two policies when prose sat
    between a URL and the next comma boundary. Anchored URL pattern
    must parse both policies cleanly."""
    from copernicus_mcp.backends.cds.backend import _parse_terms_not_accepted

    msg = (
        "user didn't accept all required site policies\n"
        "Missing policies are: A (rev. 1) - https://x/a, "
        "B (rev. 2) - https://x/b"
    )
    parsed = _parse_terms_not_accepted(msg)
    assert parsed is not None
    assert len(parsed) == 2
    assert parsed[0]["url"] == "https://x/a"
    assert parsed[1]["url"] == "https://x/b"


@pytest.mark.asyncio
async def test_terms_marker_only_falls_back_to_landing_page(
    foundation, monkeypatch
) -> None:
    """Round-1 MEDIUM (cr M-3): when the marker is present but no
    policy block is parseable, ``recovery_url`` falls back to the
    per-store landing page so the agent always has an actionable URL.
    """
    from copernicus_mcp.backends.cds.backend import (
        _TC_FALLBACK_LANDING_DEFAULT,
        CdsBackend,
    )
    from copernicus_mcp.errors import TermsNotAcceptedError

    # No host in the message → falls back to the default (CDS) landing.
    marker_only = "403 Forbidden\nuser didn't accept all required site policies"
    _patch_cdsapi(monkeypatch, retrieve_side_effect=RuntimeError(marker_only))

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    with pytest.raises(TermsNotAcceptedError) as exc_info:
        await backend.submit(_good_params())
    record = exc_info.value.error_record
    assert record.recovery_url == _TC_FALLBACK_LANDING_DEFAULT
    assert record.context["missing_policies"] == []


def test_terms_recovery_url_survives_persistence_for_today_urls(foundation) -> None:
    """Round-1 HIGH (cr H-2): pin the contract that the empirical CDS
    licence URL slugs (e.g. ``ads-data-protection-privacy-statement``)
    survive ``_serialise_error_record``'s sanitiser pass intact. If a
    future CDS deployment switches to UUID-shaped paths, this test
    will fail and force a deliberate decision (allow-list vs. accept
    redaction)."""
    import json

    from copernicus_mcp.backends.cds.backend import CdsBackend

    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    real_url = (
        "https://ads.atmosphere.copernicus.eu/licences/"
        "terms-of-use-of-the-copernicus-atmosphere-data-store"
    )
    policies = [{"name": "T", "rev": 1, "url": real_url}]
    err = backend._build_terms_not_accepted_error(  # type: ignore[attr-defined]
        "user didn't accept all required site policies\n"
        f"Missing policies are: T (rev. 1) - {real_url}",
        policies,
    )
    serialised = backend._serialise_error_record(err)  # type: ignore[attr-defined]
    record = json.loads(serialised)
    assert record["recovery_url"] == real_url
    assert record["context"]["missing_policies"][0]["url"] == real_url


# ---------------------------------------------------------------------------
# Round-2 review fixes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_status_scans_all_message_slots_for_tc_marker(
    foundation, monkeypatch
) -> None:
    """Round-2 HIGH (codex / cr M-2): when the failure message lives at
    ``remote_json["message"]`` but ``error.message`` is a non-empty
    generic string (e.g. ``"403 Forbidden"``), the round-1 fix used
    ``next(iter(candidates), ...)`` which took the first non-empty
    slot — and missed the T&C marker further down. Now we prefer the
    T&C-bearing candidate when one is present, falling back to the
    first slot for generic errors."""
    from copernicus_mcp.backends.cds.backend import CdsBackend

    await _seed_running_row(foundation, "rid-mixed")
    _patch_cdsapi(
        monkeypatch,
        get_remote_json={
            "status": "failed",
            "error": {"message": "403 Forbidden"},  # generic
            "message": _REAL_TC_MESSAGE,  # T&C in top-level
        },
    )
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    await backend.check_status("rid-mixed")
    row = await foundation.persistence.fetch_workflow("rid-mixed")
    assert row is not None
    err = json.loads(row["error_record_json"] or "{}")
    assert err.get("error_class") == "TermsNotAcceptedError"


@pytest.mark.asyncio
async def test_terms_marker_only_uses_per_store_landing_url(
    foundation, monkeypatch
) -> None:
    """Round-2 MEDIUM (cr M-1 / codex LOW-2): the marker-only fallback
    URL was hardcoded to the CDS landing even when the failing
    request was on ADS or EWDS. Now we derive the store from the
    hostname in the raw message and pick the matching landing page."""
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import TermsNotAcceptedError

    ads_marker_only = (
        "403 Client Error: Forbidden for url: "
        "https://ads.atmosphere.copernicus.eu/api/retrieve/v1/processes/"
        "cams-global-reanalysis-eac4/execution\n"
        "user didn't accept all required site policies"
    )
    _patch_cdsapi(monkeypatch, retrieve_side_effect=RuntimeError(ads_marker_only))
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    with pytest.raises(TermsNotAcceptedError) as exc_info:
        await backend.submit(_good_params())
    assert (
        exc_info.value.error_record.recovery_url
        == "https://ads.atmosphere.copernicus.eu/datasets"
    )


@pytest.mark.asyncio
async def test_detect_store_landing_anchors_to_forbidden_url_host(
    foundation, monkeypatch
) -> None:
    """Round-3 MEDIUM (codex): when the marker-only fallback path
    fires, ``_detect_store_landing`` previously did a free-form
    ``host in message`` substring match. If a future deployment
    cross-links — e.g. an ADS rejection whose unparseable policy
    block contains a CDS hostname in prose — the substring match
    could route to the wrong store. Anchor detection to the
    canonical ``Forbidden for url: <URL>`` prefix.
    """
    from copernicus_mcp.backends.cds.backend import CdsBackend
    from copernicus_mcp.errors import TermsNotAcceptedError

    # Pathological message: request URL is on CDS, but ADS is mentioned
    # in prose first. The free-form ``host in message`` substring scan
    # iterates ADS→EWDS→CDS and picks ADS by order of the tuple, even
    # though the actual request was a CDS rejection. The fix anchors
    # detection to ``Forbidden for url: <URL>`` so the request host
    # wins regardless of mentions elsewhere.
    cross_store = (
        "403 Client Error: Forbidden for url: "
        "https://cds.climate.copernicus.eu/api/retrieve/v1/processes/"
        "reanalysis-era5-single-levels/execution\n"
        "user didn't accept all required site policies\n"
        "See ads.atmosphere.copernicus.eu/how-to-api for help."
    )
    _patch_cdsapi(monkeypatch, retrieve_side_effect=RuntimeError(cross_store))
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    with pytest.raises(TermsNotAcceptedError) as exc_info:
        await backend.submit(_good_params())
    # Must use CDS landing (the actual request host), not ADS (which
    # is only mentioned in prose).
    assert (
        exc_info.value.error_record.recovery_url
        == "https://cds.climate.copernicus.eu/datasets"
    )


@pytest.mark.asyncio
async def test_detect_store_landing_unknown_anchored_host_does_not_fall_through(
    foundation, monkeypatch
) -> None:
    """Round-4 MEDIUM (codex): when ``Forbidden for url:`` matches but
    the extracted host is not a known store hostname (e.g. has a
    ``:443`` port suffix, or a future/proxy host), the previous code
    fell through to the substring scan — re-opening the same
    prose-mention misrouting that round-3 closed for the canonical
    shape. The fix returns the default landing immediately when the
    anchored host is unknown, never re-entering the substring scan.
    """
    from copernicus_mcp.backends.cds.backend import (
        _TC_FALLBACK_LANDING_DEFAULT,
        CdsBackend,
    )
    from copernicus_mcp.errors import TermsNotAcceptedError

    # CDS request with explicit port; ADS in prose.
    msg_with_port = (
        "403 Client Error: Forbidden for url: "
        "https://cds.climate.copernicus.eu:443/api/retrieve/v1/processes/"
        "reanalysis-era5-single-levels/execution\n"
        "user didn't accept all required site policies\n"
        "See ads.atmosphere.copernicus.eu/how-to-api for help."
    )
    _patch_cdsapi(monkeypatch, retrieve_side_effect=RuntimeError(msg_with_port))
    backend = CdsBackend(foundation=foundation, credentials=_fake_creds())
    with pytest.raises(TermsNotAcceptedError) as exc_info:
        await backend.submit(_good_params())
    # Anchored host was matched but unknown — must NOT fall through to
    # substring scan and pick ADS from prose. Default to CDS landing.
    assert (
        exc_info.value.error_record.recovery_url == _TC_FALLBACK_LANDING_DEFAULT
    )
