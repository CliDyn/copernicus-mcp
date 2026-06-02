"""CDS backend skeleton (Iteration 2 — T-CDS-001 partial).

All eight protocol methods raise ``BackendError(error_subclass="not_implemented")``.
The full implementation is broken into T-CDS-002 .. T-CDS-007 per
``the project conventions``. Each follow-up task is its own Tier-A PR with
codex review.

This file intentionally has no business logic — it exists so:
- the package layout mirrors ``backends/cmems``;
- ``bootstrap`` can register the factory;
- ``copernicus_mcp_status`` reports the backend (with ``configured=False``);
- a test in ``tests/unit/test_cds_backend_skeleton.py`` pins the contract.

The ``cdsapi`` SDK is NOT imported at module top — the ``cds`` extra is
opt-in and ``import CdsBackend`` must succeed in environments without the
SDK installed (matches CMEMS pattern).
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import re
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import httpx

from copernicus_mcp.auth.cds import CdsApiKeyAdapter
from copernicus_mcp.auth.resolver import ResolvedCredentials
from copernicus_mcp.backends.abstract import AbstractBackend, FoundationServices
from copernicus_mcp.backends.cds import catalogue as _catalogue
from copernicus_mcp.backends.cds import estimator as _estimator
from copernicus_mcp.data_model.provenance import (
    BackendBlock,
    CacheRef,
    CostConsumed,
    DatasetBlock,
    RequestBlock,
)
from copernicus_mcp.data_model.schemas_cds import (
    CdsApplyConstraintsRequest,
    CdsRetrieveRequest,
    CdsSearchGroupsRequest,
    CdsSearchRequest,
)
from copernicus_mcp.errors import (
    AuthError,
    BackendError,
    CacheError,
    NetworkError,
    NotFoundError,
    TermsNotAcceptedError,
)
from copernicus_mcp.errors import TimeoutError as CmcpTimeoutError
from copernicus_mcp.errors import ValidationError as CmcpValidationError
from copernicus_mcp.errors.classes import CopernicusMcpError
from copernicus_mcp.errors.records import build_error_record
from copernicus_mcp.observability.logger import get_logger

logger = get_logger(__name__)

_REGISTRATION_URL = "https://cds.climate.copernicus.eu/how-to-api"

# T&C elicitation (T-CDS-006). The empirical error shape captured by
# the T-CDS-000 smoke (F-2) differs from research §6.6.2's documented
# string. We detect on a stable substring marker and parse the
# ``Missing policies are: <name> (rev. N) - <URL>, ...`` tail. The
# server emits a single line with comma-joined policies.
#
# Round-1 LOW (cr + codex M): the marker is case-sensitive by
# deliberate choice. CDS error strings are emitted by a fixed
# `raise_for_status` chain in the upstream `ecmwf-datastores-client`
# with literal lowercase headlines; a server-side casing change would
# be a breaking deployment, and case-insensitive matching risks false
# positives on quoted prose like ``User didn't accept the cookie
# banner``. Revisit if the wire format changes.
_TC_MARKER = "user didn't accept all required site policies"
_TC_BLOCK_HEADER = "Missing policies are:"
# T-CDS-012: EWDS uses a different T&C error wording entirely (caught by
# empirical EWDS smoke 2026-05-13 against ``efas-historical``). British
# spelling "licences", inline single recovery URL embedded in prose.
_TC_EWDS_MARKER = "required licences not accepted"
_TC_EWDS_URL_RE = re.compile(
    r"please visit\s+(https?://[^\s]+?)\s+to accept",
    re.IGNORECASE,
)
# Round-1 MEDIUM (cr M-2 + codex M-2): the previous greedy-name +
# trailing-lookahead regex collapsed two policies when extra prose sat
# between a policy URL and the next comma boundary. Anchor the URL
# with ``[^\s,]+`` instead — a URL terminates at the first whitespace
# or comma, which is the empirical CDS shape.
_TC_POLICY_RE = re.compile(
    r"(?P<name>[^,]+?)\s*\(rev\.\s*(?P<rev>\d+)\)\s*-\s*"
    r"(?P<url>https?://[^\s,]+)",
)
# Per-store fallback landing page (round-1 MEDIUM cr M-3 + round-2
# MEDIUM cr M-1 / codex LOW-2). The empirical 403 message includes
# the API hostname in ``Forbidden for url: <host>/api/...``; when no
# per-policy URL is parseable we map the host to the matching web
# UI dataset index so the agent / CLI is directed to the correct
# store. CDS is the default if no host match.
_TC_STORE_LANDINGS: tuple[tuple[str, str], ...] = (
    ("ads.atmosphere.copernicus.eu", "https://ads.atmosphere.copernicus.eu/datasets"),
    ("ewds.climate.copernicus.eu", "https://ewds.climate.copernicus.eu/datasets"),
    ("cds.climate.copernicus.eu", "https://cds.climate.copernicus.eu/datasets"),
)
_TC_FALLBACK_LANDING_DEFAULT = "https://cds.climate.copernicus.eu/datasets"


_TC_FORBIDDEN_HOST_RE = re.compile(
    r"Forbidden for url:\s+https?://(?P<host>[^/\s]+)/", re.IGNORECASE
)


def _detect_store_landing(message: str) -> str:
    """Pick the per-store landing URL by matching the API hostname
    embedded in the SDK error body.

    Round-3 MEDIUM (codex): prefer the host extracted from the
    canonical ``Forbidden for url: <URL>`` prefix so detection is
    anchored to the actual request URL. Without anchoring, prose
    mentioning a hostname (or a policy URL on a different store)
    could win and route to the wrong landing.

    Round-4 MEDIUM (codex): when the ``Forbidden`` regex matches but
    the extracted host is not in the known set (e.g. ``host:443``,
    a proxy host, a future deployment), DO NOT fall back to the
    substring scan — that would re-open the prose-mention
    misrouting. Normalise the host (lowercase, strip a trailing dot,
    strip an explicit port) and compare against the known set; if
    still unknown, return the default landing immediately.

    The free-form substring fallback only fires when no
    ``Forbidden for url:`` prefix is present at all (e.g. a
    custom-shaped server error body without that header).
    """
    forbidden = _TC_FORBIDDEN_HOST_RE.search(message)
    if forbidden:
        host = forbidden.group("host").lower().rstrip(".")
        # Strip an explicit port (``host:443``) — DNS is port-agnostic
        # and the canonical CDS / ADS / EWDS hostnames have no port.
        host = host.split(":", 1)[0]
        for known_host, landing in _TC_STORE_LANDINGS:
            if host == known_host:
                return landing
        # Anchored match failed — return default WITHOUT the substring
        # scan, otherwise prose mentions could win.
        return _TC_FALLBACK_LANDING_DEFAULT
    for known_host, landing in _TC_STORE_LANDINGS:
        if known_host in message:
            return landing
    return _TC_FALLBACK_LANDING_DEFAULT


def _parse_terms_not_accepted(message: str) -> list[dict[str, Any]] | None:
    """Detect the CDS T&C-not-accepted error and parse the policy list.

    Returns:
    - ``None`` if the marker is absent (caller falls through to the
      generic SDK-error path).
    - ``[]`` if the marker is present but the ``Missing policies are:``
      block is missing or unparseable (still a T&C error; caller still
      surfaces the canonical class but with no per-policy URLs).
    - List of ``{"name": str, "rev": int, "url": str}`` dicts otherwise.
    """
    # T-CDS-012: EWDS uses a different wording. Detect it first so a
    # message that happens to contain both markers (unlikely but
    # defensively bounded) still surfaces something parseable.
    #
    # T-CDS-014 codex retro LOW-1: when the EWDS marker is present
    # but its inline-URL regex misses (defensive prose, both-marker
    # collision), fall through to the CDS/ADS branch if its canonical
    # marker is present too — otherwise we'd drop parseable CDS
    # ``Missing policies are:`` URLs. EWDS branch wins only when its
    # URL is actually found.
    #
    # T-CDS-014 codex retro LOW-2: extend the URL trailing-punctuation
    # strip beyond ``.,`` to cover ``;``, ``:``, ``)`` — a message
    # that parenthesises or semicolon-joins the URL leaves junk on
    # the canonical recovery URL and the agent then opens a 404.
    if _TC_EWDS_MARKER in message:
        url_match = _TC_EWDS_URL_RE.search(message)
        if url_match:
            return [
                {
                    "name": "linked licence-management page",
                    "rev": 0,
                    "url": url_match.group(1).rstrip(".,;:)"),
                }
            ]
        # EWDS marker without a parseable URL: prefer CDS/ADS parsing
        # if its canonical marker is also present; otherwise preserve
        # the legacy "T&C error with no per-policy detail" signal so
        # the caller still raises ``TermsNotAcceptedError`` and uses
        # the per-store landing fallback.
        if _TC_MARKER not in message:
            return []
    elif _TC_MARKER not in message:
        return None
    if _TC_BLOCK_HEADER not in message:
        return []
    tail = message.split(_TC_BLOCK_HEADER, 1)[1].strip()
    parsed: list[dict[str, Any]] = []
    for match in _TC_POLICY_RE.finditer(tail):
        # T-CDS-014 round-1 cr HIGH: the same trailing-punctuation
        # widening applied to the EWDS branch (``.,;:)``) — a CDS
        # policy URL ending with ``;``, ``:``, or ``)`` left junk
        # on the canonical recovery URL and the agent opened a 404.
        url = match.group("url").rstrip(".,;:)")
        parsed.append(
            {
                "name": match.group("name").strip().lstrip(",").strip(),
                "rev": int(match.group("rev")),
                "url": url,
            }
        )
    return parsed


# Mapping cdsapi remote-state values (research §6.5.2) to our canonical
# workflow status enum (the project conventions invariant 5). Anything not in this
# table surfaces as ``BackendError(unknown_remote_status)`` — never a
# DB CHECK violation.
_REMOTE_STATUS_MAP: dict[str, str] = {
    "accepted": "queued",
    "queued": "queued",
    "running": "running",
    "successful": "successful",
    "failed": "failed",
    "rejected": "failed",
    "dismissed": "cancelled",
    "deleted": "cancelled",
}


def _cache_storage_key(cache_key: str) -> str:
    """Single source of truth for the storage prefix; keeps file-cache
    keys distinct from search-cache and metadata-cache rows that share
    the same SQLite blob table. Mirrors the CMEMS convention."""
    return f"file:{cache_key}"


def _iso_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_request_id() -> str:
    return uuid.uuid4().hex


def _decode_error_record(raw: Any) -> Any:
    """Surface stored ``error_record_json`` as a structured dict to the
    caller — saves the agent ``json.loads(json_string)`` round-trips.
    Falls back to the raw string if malformed."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _load_cdsapi() -> Any:
    """Late import — the ``cds`` extra is opt-in (see the project conventions gotcha
    #1 / module docstring)."""
    try:
        import cdsapi
    except ImportError as exc:
        raise BackendError(
            "cdsapi extra not installed",
            record=build_error_record(
                "BackendError",
                message=(
                    "the ``cds`` extra is required for live CDS operations; "
                    "install with ``pip install copernicus-mcp[cds]``"
                ),
                error_subclass="missing_dependency",
                recovery_action="report_to_administrator",
            ),
        ) from exc
    return cdsapi


# T-CDS-011: per-store endpoint URLs. Single PAT works across all three
# stores (research §6.8.2 — common ECMWF identity layer) but each store
# has its own HTTP endpoint. Submit/poll/cancel/fetch route to the right
# one based on the dataset's ``store`` field in the bundled catalogue
# snapshot (T-CDS-003). Hard-coded here because they are public ECMWF
# constants — a future endpoint shift would land as a code change, not
# a config knob.
_STORE_ENDPOINT_URLS: Final[dict[str, str]] = {
    "cds": "https://cds.climate.copernicus.eu/api",
    "ads": "https://ads.atmosphere.copernicus.eu/api",
    "ewds": "https://ewds.climate.copernicus.eu/api",
}

# T-CDS-011 cr round-1 M1: single source of truth for "runtime knows
# how to route this store". The catalogue snapshot (T-CDS-003) may
# grow to include CDSE collections before the runtime supports them;
# ``runtime_compatible`` in ``estimate`` keys off this set, not
# catalogue presence, to avoid silent false positives in that
# transitional window.
_RUNTIME_SUPPORTED_STORES: Final[frozenset[str]] = frozenset(
    _STORE_ENDPOINT_URLS.keys()
)


# Whole-string UUID pattern reused for the surgical jobID-preservation
# branch in ``_record_terminal`` (T-CDS-011 round-2 codex HIGH). Matches
# the sanitiser's internal regex but lives here to avoid importing a
# private symbol.
#
# Round-2 cr L4: this regex is character-identical to
# ``copernicus_mcp.errors.sanitiser._UUID_FULL_RE``. If you tweak one
# (e.g. tighten the hex boundary or allow uppercase variants), update
# the other in lockstep — the sanitiser is the source of truth.
_UUID_FULL_RE: Final[re.Pattern[str]] = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def runtime_supports(dataset_id: str) -> bool:
    """True iff this backend currently knows how to route runtime ops
    (submit / poll / cancel / fetch-download) for ``dataset_id``.

    Defined here, not in the estimator, so the catalogue lookup +
    routing table membership are co-located and can't drift.
    """
    store = _catalogue.store_for(dataset_id)
    return store in _RUNTIME_SUPPORTED_STORES


def _coerce_scalar_str(value: Any) -> str | None:
    """Return ``value`` as a string when it's a scalar string or a
    single-element list of strings; otherwise ``None``.

    T-CDS-018 round-2 (local-MED2): the CDS request schema accepts
    list-valued inputs (``data_format: ["netcdf"]``); cdsapi accepts
    them too. A list-of-one is unambiguous — match the scalar. A
    multi-element list is an agent error (these slots are scalars in
    the upstream API); fall through to ``None`` and let the magic-byte
    sniff settle it after download.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    return None


# T-CDS-018 round-2 (local-MED1 + codex-LOW): the bundled catalogue and
# CDS UI both expose ``netcdf4`` and ``netcdf_legacy`` as legitimate
# data_format values; an agent copying from ``cds_apply_constraints``
# would pass them verbatim. Treat the whole family as ``.nc``.
_NETCDF_ALIASES = frozenset({"netcdf", "netcdf3", "netcdf4", "netcdf_legacy"})
_GRIB_ALIASES = frozenset({"grib", "grib1", "grib2"})


def _cds_target_filename(short_hash: str, inputs: Mapping[str, Any]) -> str:
    """Derive the cache filename for a CDS download from request inputs.

    T-CDS-018 (breaking change for v0.4): the cached file is named after
    the actual content shape, not the opaque ``cds_<hex>.bin`` we used
    before. A user opening the file in Finder / Explorer can immediately
    tell what format it is, and OS file associations work.

    Resolution order:
      1. ``download_format: zip`` → ``.zip`` (outer wrapper wins).
      2. ``data_format`` in the netcdf family → ``.nc``.
      3. ``data_format`` in the grib family → ``.grib``.
      4. Legacy ``format: ...`` key consulted only when ``data_format``
         is absent or falsy (same family rules).
      5. ``.bin`` when nothing parseable — the magic-byte sniff after
         download gets the final word.
    """
    df = _coerce_scalar_str(inputs.get("download_format"))
    if df is not None and df.lower() == "zip":
        return f"cds_{short_hash}.zip"
    fmt = _coerce_scalar_str(inputs.get("data_format")) or _coerce_scalar_str(
        inputs.get("format")
    )
    if fmt is not None:
        lo = fmt.lower()
        if lo in _NETCDF_ALIASES:
            return f"cds_{short_hash}.nc"
        if lo in _GRIB_ALIASES:
            return f"cds_{short_hash}.grib"
        if lo == "csv":  # T-TS-007: ARCO *-timeseries products emit native CSV.
            return f"cds_{short_hash}.csv"
    return f"cds_{short_hash}.bin"


_CONTENT_TYPE_BY_EXTENSION: dict[str, str] = {
    ".nc": "application/x-netcdf",
    ".grib": "application/x-grib",
    ".zip": "application/zip",
    ".csv": "text/csv",
    ".bin": "application/octet-stream",
}


def _cds_content_type_for_extension(filename: str) -> str:
    """Map a derived filename's extension to a stable MIME-ish string
    we expose in the response envelope. Defaults to
    ``application/octet-stream`` for unknown extensions.

    Round-2 (local-MED4): single ``Path.suffix`` lookup avoids the
    earlier ``endswith`` ordering trap (``.bin.gz`` etc.)."""
    return _CONTENT_TYPE_BY_EXTENSION.get(
        Path(filename).suffix, "application/octet-stream"
    )


# Round-2 (codex-MED1): ECMWF documents that NetCDF conversion may
# wrap multiple files in a ZIP even when the user asked for
# ``download_format: unarchived`` (ERA5 pressure-levels, multi-variable
# requests). Trust the bytes on disk, not the request: sniff the magic
# header after download and override the input-derived extension when
# they disagree. Mapping covers the formats we actually receive.
_MAGIC_TO_EXT: tuple[tuple[bytes, str], ...] = (
    # ZIP: normal, empty-archive, and spanned variants — all three are
    # valid ZIPs an OS-level zip reader handles. Round-2 codex LOW.
    (b"PK\x03\x04", ".zip"),
    (b"PK\x05\x06", ".zip"),
    (b"PK\x07\x08", ".zip"),
    (b"GRIB", ".grib"),
    # NetCDF3 magic is ``CDF`` followed by a version byte (1/2/5).
    # Require the full 4-byte signature so a truncated ``CDF`` blob
    # isn't misclassified as NetCDF (codex round-2 LOW).
    (b"CDF\x01", ".nc"),
    (b"CDF\x02", ".nc"),
    (b"CDF\x05", ".nc"),
    # HDF5 (== NetCDF4) requires the full 8-byte signature.
    (b"\x89HDF\r\n\x1a\n", ".nc"),
)


def _cds_sniff_extension(path: Path) -> str | None:
    """Return the canonical extension (``.zip`` / ``.nc`` / ``.grib``)
    based on the first bytes of ``path``, or ``None`` if no signature
    matches (in which case the caller keeps the input-derived extension
    or falls through to ``.bin``).
    """
    try:
        with path.open("rb") as fh:
            header = fh.read(8)
    except OSError:
        return None
    for magic, ext in _MAGIC_TO_EXT:
        if header.startswith(magic):
            return ext
    return None


def _cds_result_metadata(cache_path: Path) -> dict[str, Any]:
    """Build the canonical ``metadata`` block for a successful CDS
    response. Single source of truth — used by submit-cache-hit,
    check_status, and fetch_result so the three surfaces agree
    (round-2 codex-MED2)."""
    metadata: dict[str, Any] = {}
    try:
        metadata["size_bytes"] = cache_path.stat().st_size
    except OSError:
        pass
    metadata["content_type"] = _cds_content_type_for_extension(cache_path.name)
    return metadata


def _endpoint_url_for(dataset_id: str, *, default: str) -> str:
    """Return the per-store HTTP endpoint URL for ``dataset_id``.

    Looks up ``store`` via ``catalogue.store_for(dataset_id)``. Unknown
    dataset ids fall back to ``default`` — the caller's configured URL
    from the adapter. This keeps the old behaviour for any dataset not
    yet (or no longer) in the bundled catalogue: the request still goes
    somewhere sane, and the server's response (most likely 404) flows
    through the normal error path.
    """
    store = _catalogue.store_for(dataset_id)
    if store is None:
        return default
    return _STORE_ENDPOINT_URLS.get(store, default)


def _inputs_from_workflow_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the persisted ``inputs`` dict from a workflow row.

    T-CDS-018: used by the finaliser to pick a sensible filename
    extension from ``data_format`` / ``download_format``. Returns an
    empty dict on any parse failure — callers fall back to ``.bin``.
    """
    raw = row.get("request_json")
    if not isinstance(raw, str):
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    inputs = payload.get("inputs")
    return inputs if isinstance(inputs, dict) else {}


def _dataset_id_from_workflow_row(row: Mapping[str, Any]) -> str | None:
    """Extract the original ``dataset_id`` from a persisted workflow row.

    T-CDS-011: poll / cancel / fetch operate on ``request_id`` only, so
    they need to recover the dataset id from the row's ``request_json``
    blob to route to the right per-store endpoint URL.

    Returns ``None`` on any parse failure or missing field — caller
    falls back to the adapter's default URL.
    """
    raw = row.get("request_json")
    if not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("dataset_id")
    return value if isinstance(value, str) and value else None


# T-CDS-013: cap cdsapi retry policy. Upstream defaults are
# ``retry_max=500`` and ``sleep_max=120`` — a single hard-failing HTTP
# request can busy-wait for ~17 hours before giving up. The 2026-05-13
# real-PAT EWDS smoke (``efas-historical``) hit an HTTP 500 server-side
# fault and the SDK started looping "Retrying in 120 seconds, attempt N
# of 500". MCP must fail fast so the orchestrator / user decides when
# to retry; treat the SDK as a per-request HTTP client, not a long-lived
# waiter.
#
# Naming caveat (round-1 codex LOW, verified empirically against the
# installed ``cdsapi==0.7.7``): cdsapi treats ``retry_max`` as the
# **total number of attempts**, not the number of retries on top of
# one initial call. Internally cdsapi 0.7.x routes through its own
# ``cdsapi.Client.robust`` wrapper (and via ``multiurl`` for the
# datastores download path). So ``retry_max=3`` => 3 attempts (= 2
# retries) with up to 2 inter-attempt sleeps capped at
# ``sleep_max=10`` => ~20s of back-off + per-attempt request latency
# (typically a few seconds) => roughly 20-30s worst case per HTTP
# call. Do not set ``retry_max=0``: legacy download paths interpret
# it as "no attempts", not "no retries".
_CDSAPI_RETRY_MAX: Final[int] = 3
_CDSAPI_SLEEP_MAX: Final[int] = 10


def _make_cdsapi_client(
    adapter: CdsApiKeyAdapter, *, dataset_id: str | None = None
) -> Any:
    """Construct a cdsapi.Client with the kwargs we have audited.

    Codex spec review: ``LegacyClient.__init__`` debug-logs the raw
    ``key`` (legacy_client.py:122) when ``debug=True``, and the tqdm
    progress bar misbehaves under MCP stdio. ``wait_until_complete=True``
    would block the event loop. Lock the four kwargs that matter.

    T-CDS-011: when ``dataset_id`` is provided, route to the
    corresponding store's endpoint (CDS / ADS / EWDS). Unknown dataset
    ids fall back to the adapter's configured URL — typically the CDS
    default — so a stale catalogue snapshot does not regress callers.
    When ``dataset_id`` is None the adapter URL is used verbatim
    (legacy behaviour, mostly tests and the auth-summary path).

    T-CDS-013: cap ``retry_max`` / ``sleep_max`` so a hard-failing
    upstream cannot keep us looping for hours (see module-level
    constants above for the rationale).
    """
    cdsapi = _load_cdsapi()
    key, adapter_url = adapter.get_pat()
    fallback_url = adapter_url or _STORE_ENDPOINT_URLS["cds"]
    url: str | None
    if dataset_id is not None:
        url = _endpoint_url_for(dataset_id, default=fallback_url)
    else:
        url = adapter_url
    return cdsapi.Client(
        url=url,
        key=key,
        wait_until_complete=False,
        quiet=True,
        progress=False,
        debug=False,
        retry_max=_CDSAPI_RETRY_MAX,
        sleep_max=_CDSAPI_SLEEP_MAX,
    )


def _pending_response(*, request_id: str, cache_key: str, status: str) -> dict[str, Any]:
    """Envelope for an async submit that is in-flight on the CDS queue.

    Distinct from CMEMS's ``_running_response`` which hard-codes
    ``status: "running"`` because CMEMS has no ``queued`` state. CDS
    canonical statuses include ``queued`` (research §6.5.2) — surface
    it accurately so callers can distinguish queue-wait from active
    processing.
    """
    return {
        "status": status,
        "cache_hit": False,
        "is_existing": False,
        "request_id": request_id,
        "cache_key": cache_key,
        "result": {
            "uri": f"copernicus://jobs/{request_id}",
            "metadata": {},
            "provenance": {},
        },
    }


def _success_response_from_cache(
    *,
    request_id: str,
    cache_key: str,
    filepath: Path,
) -> dict[str, Any]:
    """Canonical large-data success envelope for an idempotent cache
    hit. Mirrors CMEMS shape (the project conventions invariant 1)."""
    return {
        "status": "successful",
        "cache_hit": True,
        "is_existing": True,
        "request_id": request_id,
        "cache_key": cache_key,
        "result": {
            "filepath": str(filepath),
            "uri": f"copernicus://files/{cache_key}",
            "metadata": _cds_result_metadata(filepath),
            "provenance": {},
        },
    }


def _build_cds_confirmation(
    *,
    estimate: dict[str, Any],
    threshold_bytes: int,
    reason: str,
) -> Any:
    """Wrap ``build_size_confirmation`` adding CDS-specific tier context
    (codex spec review): the agent needs the queue-tier when the gate
    triggers on field count rather than bytes, otherwise a 50 MB-but-
    heavy-queue prompt looks like noise."""
    from copernicus_mcp.workflow.confirmation import build_size_confirmation

    bytes_estimate = int(estimate.get("estimated_size_bytes", 0))
    confirmation = build_size_confirmation(
        tool_name="cds_submit_request",
        backend="cds",
        estimated_size_bytes=bytes_estimate,
        threshold_bytes=threshold_bytes,
        source="config.budget.cds_per_request_size_warning_gb",
    )
    confirmation.payload["reason"] = reason
    confirmation.payload["context"]["queue_latency_tier"] = estimate.get("queue_latency_tier")
    confirmation.payload["context"]["fields_count"] = estimate.get("fields_count")
    confirmation.payload["context"]["epistemic_status"] = "approximate"
    return confirmation


def _not_implemented(method: str) -> BackendError:
    return BackendError(
        f"CdsBackend.{method} not implemented yet (T-CDS scaffold)",
        record=build_error_record(
            "BackendError",
            message=(f"CdsBackend.{method} not implemented yet — see the project conventions"),
            error_subclass="not_implemented",
            recovery_action="report_to_administrator",
        ),
    )


class CdsBackend(AbstractBackend):
    backend_id = "cds"

    def __init__(
        self,
        foundation: FoundationServices,
        credentials: ResolvedCredentials | None,
    ) -> None:
        super().__init__(foundation=foundation)
        self._credentials = credentials
        self._auth_adapter: CdsApiKeyAdapter | None = (
            CdsApiKeyAdapter(credentials) if credentials is not None else None
        )
        # Per-cache-key submit locks with ref-counted lifecycle (round-3
        # HIGH, codex): the round-2 ``setdefault → pop`` pattern split
        # the lock when an early caller failed before recording a row.
        # A waiter held lock_v1; a later arrival did ``setdefault`` after
        # the pop and got lock_v2; both raced. Now we ref-count the
        # entry under a small dict mutex so the lock survives until
        # every waiter is gone. Memory bounded: O(active submits).
        self._submit_locks: dict[str, asyncio.Lock] = {}
        self._submit_lock_refs: dict[str, int] = {}
        self._submit_locks_dict_mutex = asyncio.Lock()
        # Per-request finalisation lock (codex spec review HIGH-2):
        # ``check_status`` does the download-and-cache step, so two
        # concurrent pollers observing remote=successful must not both
        # download. Locks are popped from the dict after the row reaches
        # a terminal state to bound memory in long-running servers
        # (round-1 HIGH-5, code-reviewer).
        self._finalise_locks: dict[str, asyncio.Lock] = {}

    def _check_credentials_or_raise(self) -> ResolvedCredentials:
        if self._credentials is None:
            raise AuthError(
                "CDS credentials are not configured",
                record=build_error_record(
                    "AuthError",
                    message="CDS credentials are not configured",
                    recovery_action="configure_credentials",
                    recovery_url=_REGISTRATION_URL,
                ),
            )
        return self._credentials

    # --- protocol --------------------------------------------------------

    async def search(self, params: dict[str, Any]) -> dict[str, Any]:
        """Catalogue discovery (T-CDS-003).

        Returns the slim catalogue (``{id, title, description, keywords,
        store}`` per dataset) so the LLM can pick candidates from a
        single tool-call payload (~30k tokens for the full catalogue).
        Two-stage flow: ``search`` narrows, then ``describe`` returns the
        full STAC record for each candidate.

        Reads from the bundled snapshot at
        ``backends/cds/_data/{cds,ads,ewds}.json``. NEVER hits the network.
        """
        from pydantic import ValidationError as PydValidationError

        clean = {k: v for k, v in params.items() if k != "__options"}
        try:
            req = CdsSearchRequest.model_validate(clean)
        except PydValidationError as exc:
            field_errors = [
                {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
            ]
            raise CmcpValidationError(
                "invalid CDS search params",
                record=build_error_record(
                    "ValidationError",
                    message="invalid CDS search params",
                    recovery_action="modify_request_parameters",
                    context={"field_errors": field_errors},
                ),
            ) from exc

        datasets = _catalogue.search(
            keyword=req.keyword,
            store=req.store,
            limit=req.limit,
            bbox=req.bbox,
            time_range=req.time_range,
            variable=req.variable,
            domain=req.domain,
            category=req.category,
        )
        return self.foundation.sanitiser.sanitise(  # type: ignore[no-any-return]
            {"datasets": datasets, "total_count": len(datasets)}
        )

    async def search_groups(self, params: dict[str, Any]) -> dict[str, Any]:
        """T-CDS-021 PR-2: hierarchical group discovery.

        Returns (domain, category) groups ranked against an optional
        free-text query. Two-hop discovery flow: agent calls this with a
        natural-language intent, picks a group, then calls
        ``cds_search_datasets(domain=..., category=...)`` for the
        narrowed candidate list.

        Reads from the bundled snapshot — NEVER hits the network.
        """
        from pydantic import ValidationError as PydValidationError

        clean = {k: v for k, v in params.items() if k != "__options"}
        try:
            req = CdsSearchGroupsRequest.model_validate(clean)
        except PydValidationError as exc:
            field_errors = [
                {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
                for e in exc.errors()
            ]
            raise CmcpValidationError(
                "invalid CDS search_groups params",
                record=build_error_record(
                    "ValidationError",
                    message="invalid CDS search_groups params",
                    recovery_action="modify_request_parameters",
                    context={"field_errors": field_errors},
                ),
            ) from exc

        groups = _catalogue.search_groups(query=req.query, top_k=req.top_k)
        return self.foundation.sanitiser.sanitise(  # type: ignore[no-any-return]
            groups
        )

    async def describe(self, identifier: str) -> dict[str, Any]:
        """Full STAC record for a single dataset (T-CDS-003).

        Cross-store lookup — caller does not need to know which of CDS /
        ADS / EWDS the id belongs to. Result is augmented with a
        ``store`` field. Raises ``NotFoundError`` if the id is not in
        the bundled snapshot.
        """
        if not isinstance(identifier, str) or not identifier.strip():
            raise CmcpValidationError(
                "dataset identifier must be a non-empty string",
                record=build_error_record(
                    "ValidationError",
                    message="dataset identifier must be a non-empty string",
                    recovery_action="modify_request_parameters",
                ),
            )
        record = _catalogue.describe(identifier)
        return self.foundation.sanitiser.sanitise(record)  # type: ignore[no-any-return]

    async def validate(self, params: dict[str, Any]) -> dict[str, Any]:
        """Schema-only validation (T-CDS-002).

        Per ``the project research notes`` §6.9.1 the per-dataset
        constraint catalogue is server-side; we surface only structural
        problems here. Mirrors the CMEMS ``validate`` discipline:
        sanitise the error output before returning so a Pydantic ``msg``
        echoing raw input cannot leak credential-shaped values.
        """
        from pydantic import ValidationError as PydValidationError

        clean = {k: v for k, v in params.items() if k != "__options"}
        try:
            CdsRetrieveRequest.model_validate(clean)
        except PydValidationError as exc:
            errors = [
                {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
            ]
            return self.foundation.sanitiser.sanitise(  # type: ignore[no-any-return]
                {"valid": False, "errors": errors}
            )
        return {"valid": True}

    async def apply_constraints(self, params: dict[str, Any]) -> dict[str, Any]:
        """Progressive constraints narrowing for CDS / ADS / EWDS (T-CDS-016).

        POSTs ``{"inputs": <partial_request>}`` to
        ``<store>/api/retrieve/v1/processes/<dataset_id>/constraints``
        and returns the parsed JSON of remaining valid values. Used to
        compose a submit request step-by-step instead of guessing field
        names / values.

        Credential-required: same as other CDS write paths.
        """
        from pydantic import ValidationError as PydValidationError

        clean = {k: v for k, v in params.items() if k != "__options"}
        try:
            req = CdsApplyConstraintsRequest.model_validate(clean)
        except PydValidationError as exc:
            field_errors = [
                {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
                for e in exc.errors()
            ]
            raise CmcpValidationError(
                "invalid cds_apply_constraints params",
                record=build_error_record(
                    "ValidationError",
                    message="invalid cds_apply_constraints params",
                    recovery_action="modify_request_parameters",
                    context={"field_errors": field_errors},
                ),
            ) from exc

        # cr round-1 M3: constraints endpoint is anonymous-friendly
        # (empirically verified 2026-05-16: a credential-less POST
        # against /processes/efas-historical/constraints returns the
        # full valid-values payload). This is a READ-ONLY op like
        # search/describe/validate/estimate, so we do NOT gate on
        # credentials — consistent with the rest of the read-only
        # surface.
        store = _catalogue.store_for(req.dataset_id) or "cds"
        base_url = _STORE_ENDPOINT_URLS[store]
        url = f"{base_url}/retrieve/v1/processes/{req.dataset_id}/constraints"

        async with self.foundation.http_client_factory.create("cds") as client:
            try:
                response = await client.post(url, json={"inputs": req.inputs})
            except asyncio.CancelledError:
                raise
            except httpx.TimeoutException as exc:
                # cr round-1 M5: timeout maps to canonical TimeoutError,
                # not BackendError — same discipline as CMEMS subset/get.
                raise CmcpTimeoutError(
                    f"apply_constraints: request to CDS timed out: {exc}"
                ) from exc
            except (httpx.ConnectError, httpx.ReadError, httpx.WriteError) as exc:
                # cr round-1 M5: transport-level connect/read/write
                # failures are NetworkError (retry-friendly), not the
                # generic BackendError "SDK broken" signal.
                raise NetworkError(
                    f"apply_constraints: network error contacting CDS: {exc}"
                ) from exc
            except httpx.HTTPError as exc:
                raise BackendError(
                    "apply_constraints: HTTP error against CDS",
                    record=build_error_record(
                        "BackendError",
                        message=self.foundation.sanitiser.sanitise(
                            f"apply_constraints: HTTP error: {exc}"
                        ),
                        error_subclass="apply_constraints_http",
                        recovery_action="retry_automatic",
                    ),
                ) from exc
            if response.status_code == 404:
                raise NotFoundError(
                    f"apply_constraints: dataset_id={req.dataset_id!r} "
                    "not exposed by the constraints endpoint. The "
                    "bundled snapshot ('available_inputs' on "
                    "cds_describe_dataset) may still describe valid "
                    "fields for this dataset."
                )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise BackendError(
                    "apply_constraints: server rejected the request",
                    record=build_error_record(
                        "BackendError",
                        message=self.foundation.sanitiser.sanitise(
                            f"apply_constraints: HTTP {response.status_code} "
                            f"from server"
                        ),
                        error_subclass="apply_constraints_http",
                        recovery_action="modify_request_parameters",
                    ),
                ) from exc
            try:
                payload = response.json()
            except ValueError as exc:
                raise BackendError(
                    "apply_constraints: server returned non-JSON",
                    record=build_error_record(
                        "BackendError",
                        message="apply_constraints: server returned non-JSON",
                        error_subclass="apply_constraints_decode",
                        recovery_action="report_to_administrator",
                    ),
                ) from exc

        envelope = {
            "dataset_id": req.dataset_id,
            "store": store,
            "inputs_provided": dict(req.inputs),
            "valid_remaining": payload if isinstance(payload, dict) else {},
        }
        return self.foundation.sanitiser.sanitise(envelope)  # type: ignore[no-any-return]

    async def estimate(self, params: dict[str, Any]) -> dict[str, Any]:
        """Heuristic byte-size estimate for a CDS retrieve request (T-CDS-004).

        Per ``the project research notes`` §6.7.4 option 1:
        the legacy ``cdsapi`` 0.7.7 client has no estimation API; we
        derive the estimate from request shape alone (product of
        list-cardinalities × dataset-specific bytes-per-field × area
        fraction). Always ``epistemic_status="approximate"``.

        Used by ``submit`` (T-CDS-005) to drive the confirmation gate
        before queueing a real download.
        """
        from pydantic import ValidationError as PydValidationError

        clean = {k: v for k, v in params.items() if k != "__options"}
        try:
            req = CdsRetrieveRequest.model_validate(clean)
        except PydValidationError as exc:
            field_errors = [
                {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
            ]
            raise CmcpValidationError(
                "invalid CDS retrieve params",
                record=build_error_record(
                    "ValidationError",
                    message="invalid CDS retrieve params",
                    recovery_action="modify_request_parameters",
                    context={"field_errors": field_errors},
                ),
            ) from exc

        result = _estimator.estimate(req.dataset_id, dict(req.inputs))
        return self.foundation.sanitiser.sanitise(result)  # type: ignore[no-any-return]

    async def submit(self, params: dict[str, Any]) -> dict[str, Any]:
        from pydantic import ValidationError as PydValidationError

        clean = {k: v for k, v in params.items() if k != "__options"}
        try:
            req = CdsRetrieveRequest.model_validate(clean)
        except PydValidationError as exc:
            field_errors = [
                {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
            ]
            raise CmcpValidationError(
                "invalid CDS retrieve params",
                record=build_error_record(
                    "ValidationError",
                    message="invalid CDS retrieve params",
                    recovery_action="modify_request_parameters",
                    context={"field_errors": field_errors},
                ),
            ) from exc

        self._check_credentials_or_raise()
        assert self._auth_adapter is not None

        options = dict(params.get("__options") or {}) if isinstance(params, dict) else {}
        safe_params = self.foundation.sanitiser.sanitise(params)
        cache_key = self.foundation.data_model.cache_key_for_cds_retrieve(req)

        # Round-3 HIGH (codex): ref-counted per-key lock. While ANY
        # task is using the entry for ``cache_key``, the dict keeps the
        # same Lock instance — closing the split-lock race that round-2
        # introduced.
        async with self._submit_locks_dict_mutex:
            if cache_key not in self._submit_locks:
                self._submit_locks[cache_key] = asyncio.Lock()
                self._submit_lock_refs[cache_key] = 0
            self._submit_lock_refs[cache_key] += 1
            lock = self._submit_locks[cache_key]
        try:
            async with lock:
                # Idempotency: cache hit short-circuits without any SDK call.
                if not options.get("force_refresh"):
                    existing_path = await self.foundation.cache.lookup_file(
                        _cache_storage_key(cache_key)
                    )
                    if existing_path is not None:
                        existing_wf = (
                            await self.foundation.persistence.lookup_workflow_by_cache_key(
                                cache_key
                            )
                        )
                        req_id = (
                            existing_wf["request_id"]
                            if existing_wf is not None
                            else _new_request_id()
                        )
                        # T-CDS-000 smoke regression: do NOT pass the
                        # envelope through the sanitiser — its UUID
                        # pattern would redact the server-generated
                        # ``request_id`` (UUID-shape) and the agent
                        # would lose the only handle for poll/download.
                        # The envelope is safe by construction; user
                        # input only enters the persistence layer
                        # via ``safe_params`` above.
                        return _success_response_from_cache(
                            request_id=req_id,
                            cache_key=cache_key,
                            filepath=existing_path,
                        )

                    # In-flight dedupe.
                    inflight = await self.foundation.persistence.lookup_workflow_by_cache_key(
                        cache_key
                    )
                    if inflight is not None and inflight["status"] in (
                        "queued",
                        "running",
                    ):
                        return _pending_response(
                            request_id=inflight["request_id"],
                            cache_key=cache_key,
                            status=inflight["status"],
                        )

                # Confirmation gate (codex spec review HIGH-3 hybrid):
                #   bytes > threshold OR queue tier in {medium, heavy}
                # Round-1 MEDIUM (code-reviewer): use ``self.estimate`` so
                # any future side-effect (telemetry, provenance) added to
                # the public estimate path also fires for submit.
                estimate = await self.estimate(params)
                budget = self.foundation.config.budget
                threshold_bytes = int(
                    budget.cds_per_request_size_warning_gb * 1_000_000_000
                )
                bytes_estimate = int(estimate["estimated_size_bytes"])
                tier = estimate.get("queue_latency_tier")
                size_over = bytes_estimate > threshold_bytes
                tier_over = tier in budget.cds_confirm_on_queue_tier
                if not options.get("confirmed") and (size_over or tier_over):
                    reason = (
                        "estimated_size_threshold_exceeded"
                        if size_over
                        else "queue_latency_tier_exceeded"
                    )
                    raise _build_cds_confirmation(
                        estimate=estimate,
                        threshold_bytes=threshold_bytes,
                        reason=reason,
                    )

                # Round-1 MEDIUM (codex): wrap SDK exceptions so any PAT
                # embedded in an exception string is rewritten through
                # the sanitiser before reaching the caller.
                try:
                    client = _make_cdsapi_client(
                        self._auth_adapter, dataset_id=req.dataset_id
                    )
                    remote = await asyncio.to_thread(
                        client.retrieve, req.dataset_id, dict(req.inputs), None
                    )
                except CopernicusMcpError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    # T-CDS-006: detect the per-dataset T&C-not-accepted
                    # error before the generic SDK wrap, so the agent /
                    # CLI gets the canonical ``TermsNotAcceptedError``
                    # envelope with the licence ``recovery_url``.
                    raw = str(exc)
                    policies = _parse_terms_not_accepted(raw)
                    if policies is not None:
                        raise self._build_terms_not_accepted_error(
                            raw, policies
                        ) from exc
                    raise self._wrap_sdk_error(exc, op="submit") from exc

                request_id = str(remote.request_id)
                now = _iso_now()
                # Round-1 MEDIUM (codex): record_workflow failure between
                # remote submit and local row commit ⇒ orphan queue slot.
                # Round-4 MEDIUM (codex async re-review): use ``try/finally``
                # so the orphan cleanup also runs on ``CancelledError``;
                # ``except Exception`` would let cancellation skip the
                # delete. ``asyncio.shield`` keeps the in-flight delete
                # alive even if the outer task is being torn down.
                # the project conventions invariant 3 preserved: we don't catch
                # ``CancelledError`` — it propagates after the finally.
                recorded = False
                try:
                    await self.foundation.persistence.record_workflow(
                        {
                            "request_id": request_id,
                            "backend_id": "cds",
                            "operation": "submit",
                            "status": "queued",
                            "cache_key": cache_key,
                            "request_json": json.dumps(
                                safe_params, sort_keys=True, default=str
                            ),
                            "response_json": None,
                            "error_record_json": None,
                            "created_at": now,
                            "updated_at": now,
                        }
                    )
                    recorded = True
                finally:
                    if not recorded:
                        with contextlib.suppress(Exception):
                            await asyncio.shield(
                                asyncio.to_thread(
                                    client.client.delete, request_id
                                )
                            )
        finally:
            # Round-3 HIGH (codex): drop the entry only when refcount
            # reaches zero. Otherwise a waiter exists and we'd split
            # the lock by leaving them with a stale reference while a
            # new arrival creates a fresh Lock.
            async with self._submit_locks_dict_mutex:
                self._submit_lock_refs[cache_key] -= 1
                if self._submit_lock_refs[cache_key] == 0:
                    self._submit_locks.pop(cache_key, None)
                    self._submit_lock_refs.pop(cache_key, None)

        return _pending_response(
            request_id=request_id, cache_key=cache_key, status="queued"
        )

    async def check_status(self, request_id: str) -> dict[str, Any]:
        row = await self.foundation.persistence.fetch_workflow(request_id)
        if row is None:
            raise NotFoundError(
                f"workflow {request_id!r} not found",
                record=build_error_record(
                    "NotFoundError",
                    message=f"workflow {request_id!r} not found",
                    recovery_action="modify_request_parameters",
                ),
            )

        status = row["status"]
        # Already-terminal rows return their cached state without an SDK
        # call. The cache-eviction synthetic error (CMEMS parity) is
        # applied on the way out. Codex spec review HIGH-1: do NOT mark
        # stale running rows failed — CDS jobs can validly stay
        # queued/running for hours; the source of truth is the remote
        # poll below, never local age.
        if status in ("successful", "failed", "cancelled"):
            return await self._build_status_envelope(row, status)

        return await self._poll_and_maybe_finalise(request_id, row)

    async def _poll_and_maybe_finalise(
        self, request_id: str, row: Mapping[str, Any]
    ) -> dict[str, Any]:
        self._check_credentials_or_raise()
        assert self._auth_adapter is not None

        client = _make_cdsapi_client(
            self._auth_adapter,
            dataset_id=_dataset_id_from_workflow_row(row),
        )
        try:
            remote = await asyncio.to_thread(client.client.get_remote, request_id)
            remote_json = await asyncio.to_thread(lambda: remote.json)
        except Exception as exc:  # noqa: BLE001 — wrap SDK errors uniformly
            raise self._wrap_sdk_error(exc, op="poll") from exc

        remote_status = str(remote_json.get("status", "")).strip()
        canonical = _REMOTE_STATUS_MAP.get(remote_status)
        if canonical is None:
            raise BackendError(
                f"unknown remote status {remote_status!r}",
                record=build_error_record(
                    "BackendError",
                    message=(
                        f"CDS returned unrecognised status "
                        f"{remote_status!r} for request {request_id!r}"
                    ),
                    error_subclass="unknown_remote_status",
                    recovery_action="report_to_administrator",
                ),
            )

        if canonical == "successful":
            return await self._finalise_successful(
                request_id=request_id,
                cache_key=row["cache_key"] or "",
                client=client,
            )

        if canonical in ("failed", "cancelled"):
            await self._record_terminal(request_id, canonical, remote_json=remote_json)
            # Round-2 MEDIUM-E (codex): build the envelope from the
            # *actual* row state, not from the intended canonical
            # status. If a concurrent cancel won the race, the row is
            # already ``cancelled`` and the response must say so —
            # otherwise the envelope contradicts persistence.
            fresh = await self.foundation.persistence.fetch_workflow(request_id)
            actual = (fresh or row)["status"]
            return await self._build_status_envelope(fresh or row, actual)

        # Non-terminal: persist the transition (queued ↔ running) and
        # return the canonical envelope.
        # Round-2 HIGH-A (codex): conditional UPDATE so a concurrent
        # cancel that committed between our row fetch and this write
        # is not silently re-promoted to ``running``.
        if canonical != row["status"]:
            await self.foundation.persistence.update_workflow_status_if_pending(
                request_id,
                canonical,  # type: ignore[arg-type]
            )
        fresh = await self.foundation.persistence.fetch_workflow(request_id)
        actual = (fresh or row)["status"]
        return await self._build_status_envelope(fresh or row, actual)

    async def _finalise_successful(
        self,
        *,
        request_id: str,
        cache_key: str,
        client: Any,
    ) -> dict[str, Any]:
        """Locked, idempotent download + cache.store_file commit.

        Round-1 HIGH-2 (codex + code-reviewer): all terminal commits
        use ``*_if_pending`` so a row that flipped to ``cancelled``
        between the lock re-check and the commit is not overwritten.

        Round-1 HIGH-5 (code-reviewer): the per-request lock is popped
        from ``self._finalise_locks`` after the row reaches a terminal
        state to bound memory in long-running servers.
        """
        lock = self._finalise_locks.setdefault(request_id, asyncio.Lock())
        try:
            async with lock:
                # Re-check under the lock — another caller may have
                # already finalised. In that case skip download entirely.
                row = await self.foundation.persistence.fetch_workflow(
                    request_id
                )
                if row is not None and row["status"] in (
                    "successful",
                    "failed",
                    "cancelled",
                ):
                    # Already terminal; nothing to do, return the cached
                    # state.
                    return await self._build_status_envelope(row, row["status"])

                zone = self.foundation.cache.cache_zone_for("cds")
                staging = zone / ".staging" / uuid.uuid4().hex
                staging.mkdir(parents=True, exist_ok=False)
                short_hash = (
                    cache_key.rsplit(":", 1)[-1] if cache_key else "unknown"
                )
                # T-CDS-018: derive extension from the persisted submit
                # inputs so the cached file is human-recognisable.
                row_inputs = _inputs_from_workflow_row(row) if row else {}
                target_path = staging / _cds_target_filename(
                    short_hash, row_inputs
                )

                stored: Path | None = None
                try:
                    # Download to staging only — do NOT publish to the
                    # canonical cache yet. Round-3 MEDIUM (codex):
                    # publishing before the conditional UPDATE would
                    # overwrite an unrelated workflow's cached file
                    # under the same cache_key (force_refresh case),
                    # then "invalidate" it on race-loss, destroying
                    # the older successful workflow's result.
                    await asyncio.to_thread(
                        client.client.download_results,
                        request_id,
                        str(target_path),
                    )
                except CopernicusMcpError as exc:
                    error_record_json = self._serialise_error_record(exc)
                    await self.foundation.persistence.update_workflow_error_if_pending(
                        request_id, "failed", error_record_json
                    )
                    with contextlib.suppress(OSError):
                        if target_path.exists():
                            target_path.unlink()
                        staging.rmdir()
                    raise
                except Exception as exc:  # noqa: BLE001
                    wrapped = self._wrap_sdk_error(exc, op="download")
                    error_record_json = self._serialise_error_record(wrapped)
                    await self.foundation.persistence.update_workflow_error_if_pending(
                        request_id, "failed", error_record_json
                    )
                    with contextlib.suppress(OSError):
                        if target_path.exists():
                            target_path.unlink()
                        staging.rmdir()
                    raise wrapped from exc

                # T-CDS-018 round-2 (codex-MED1): ECMWF may return a
                # ZIP even when the user asked for ``data_format=netcdf``
                # + ``download_format=unarchived`` (multi-variable ERA5).
                # Trust the bytes on disk: sniff the magic header and
                # rename the staged file if the input-derived extension
                # disagrees, so the on-disk name reflects content and
                # downstream openers don't choke on a "netcdf" that's
                # really a zip.
                #
                # Round-2 HIGH (local cr): the rename is a real syscall
                # (``ENOSPC``, ``EACCES``, file vanished, etc.); without
                # error handling the workflow row would stay ``running``
                # forever and leak the staged file. Wrap with the same
                # pattern as download/store_file: flip the row to
                # ``failed`` with a canonical error record, clean
                # staging, re-raise wrapped.
                sniffed = _cds_sniff_extension(target_path)
                if sniffed is not None and Path(target_path).suffix != sniffed:
                    renamed = target_path.with_name(f"cds_{short_hash}{sniffed}")
                    try:
                        target_path.replace(renamed)
                    except OSError as exc:
                        wrapped = self._wrap_sdk_error(exc, op="sniff_rename")
                        error_record_json = self._serialise_error_record(wrapped)
                        await self.foundation.persistence.update_workflow_error_if_pending(
                            request_id, "failed", error_record_json
                        )
                        with contextlib.suppress(OSError):
                            if target_path.exists():
                                target_path.unlink()
                            staging.rmdir()
                        raise wrapped from exc
                    target_path = renamed

                # Round-1 HIGH-2 + Round-3 MEDIUM (codex): commit
                # conditionally; only publish to the canonical cache
                # AFTER we've won the race. If the row already
                # transitioned to ``cancelled`` (or any other terminal
                # state), drop the staged file and leave the cache
                # entry untouched.
                committed = await self.foundation.persistence.update_workflow_status_if_pending(
                    request_id, "successful"
                )

                if not committed:
                    # Race-loser: another caller settled the row. Drop
                    # staging without touching the cache. Round-4 MEDIUM
                    # (codex async): log cleanup failures explicitly so
                    # leaked staging files (potentially hundreds of MB
                    # for ERA5 pressure-levels) are visible in
                    # observability rather than silently piling up.
                    try:
                        if target_path.exists():
                            target_path.unlink()
                        staging.rmdir()
                    except OSError as cleanup_exc:
                        logger.warning(
                            "race_loser_staging_cleanup_failed",
                            extra={
                                "request_id": request_id,
                                "staging_path": str(staging),
                                "error_class": type(cleanup_exc).__name__,
                                "error_message": str(cleanup_exc),
                            },
                        )
                    fresh = await self.foundation.persistence.fetch_workflow(
                        request_id
                    )
                    return await self._build_status_envelope(
                        fresh or {"request_id": request_id, "cache_key": cache_key},
                        (fresh or {"status": "failed"})["status"],
                    )

                # Won the race — now publish to the canonical cache.
                # ``store_file`` moves the staged file out of the
                # staging dir and atomically replaces the previous
                # entry for this key (if any). Round-3 MEDIUM: this
                # only happens after the workflow row commits to
                # ``successful``, so a force_refresh that loses the
                # race no longer overwrites another workflow's content.
                #
                # Round-4 HIGH (codex async re-review): if ``store_file``
                # raises after the row has already been committed
                # ``successful``, the row would be stuck terminal-success
                # with no cache file, and ``check_status`` would
                # synthesise ``cache_eviction`` forever. Revert the row
                # to ``failed`` (unconditionally — the row is at
                # ``successful`` which we just set ourselves under the
                # finalise lock; cancel can't have raced past us).
                try:
                    # Round-2 (local-MED3): persist the same content_type
                    # we report in the envelope, so the cache_entries row
                    # and the agent-visible metadata agree.
                    stored = await self.foundation.cache.store_file(
                        _cache_storage_key(cache_key),
                        target_path,
                        backend_id="cds",
                        content_type=_cds_content_type_for_extension(
                            target_path.name
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    wrapped = self._wrap_sdk_error(exc, op="cache_store")
                    error_record_json = self._serialise_error_record(wrapped)
                    await self.foundation.persistence.update_workflow_error(
                        request_id, "failed", error_record_json
                    )
                    with contextlib.suppress(OSError):
                        if target_path.exists():
                            target_path.unlink()
                        staging.rmdir()
                    raise wrapped from exc
                with contextlib.suppress(OSError):
                    if target_path.exists():
                        target_path.unlink()
                    staging.rmdir()

                # T-CDS-011.5: write provenance sidecar + persistence
                # row, mirroring CMEMS behaviour (T-024). Failure here
                # is non-fatal — the row is already ``successful`` and
                # the file is in the cache; sidecar is recoverable from
                # the SQLite provenance table even if the JSON write
                # raced. CancelledError must NOT be swallowed
                # (the project conventions invariant 3).
                try:
                    await self._record_cds_provenance(
                        request_id=request_id,
                        cache_key=cache_key,
                        stored_path=stored,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning("provenance recording failed", exc_info=True)

                fresh = await self.foundation.persistence.fetch_workflow(
                    request_id
                )
                return await self._build_status_envelope(
                    fresh or {"request_id": request_id, "cache_key": cache_key},
                    "successful",
                    _resolved_filepath=stored,
                )
        finally:
            # Round-1 HIGH-5: drop the per-request lock once we exit so
            # ``self._finalise_locks`` cannot grow without bound. Safe
            # because terminal status is permanent — any later
            # ``check_status`` takes the early-return terminal branch
            # and never reaches ``_finalise_successful``.
            self._finalise_locks.pop(request_id, None)

    async def _record_terminal(
        self,
        request_id: str,
        status: str,
        *,
        remote_json: dict[str, Any],
    ) -> None:
        """Persist a row as failed / cancelled with sanitised error
        details when the SDK reports a terminal non-success state.

        Round-1 HIGH-3 (codex): the server-supplied error message is
        passed through ``Sanitiser`` before persistence — CDS may echo
        request inputs or, in pathological cases, credential-shaped
        identifiers it received in headers.

        Round-1 HIGH-2 (codex + code-reviewer): the failed-write uses
        ``update_workflow_error_if_pending`` so a row that already
        flipped to ``cancelled`` is not overwritten.
        """
        if status == "failed":
            # Round-1 HIGH (cr H-1 / codex M-1): the previous extraction
            # only looked at ``error.message``. CDS / ADS / EWDS may
            # surface the failure string in any of three slots —
            # ``error.message`` (dict-shaped), ``error`` (string-shaped,
            # legacy ecmwf-datastores), or top-level ``message``. Try
            # each in turn so a T&C-not-accepted body that happens to
            # land at the top level still flows through the canonical
            # TermsNotAcceptedError path instead of the generic
            # ``remote_job_failed`` fallback.
            err = remote_json.get("error")
            candidates: list[str] = []
            if isinstance(err, dict):
                msg = err.get("message")
                if isinstance(msg, str) and msg:
                    candidates.append(msg)
            elif isinstance(err, str) and err:
                candidates.append(err)
            top_msg = remote_json.get("message")
            if isinstance(top_msg, str) and top_msg:
                candidates.append(top_msg)
            # Round-2 HIGH (codex / cr M-2): when one slot carries a
            # generic message (e.g. ``"403 Forbidden"``) and another
            # carries the T&C marker, the round-1 ``next(iter(...))``
            # took the generic message and lost the T&C signal. Scan
            # all candidates for the T&C marker first; only fall back
            # to the first non-empty slot if no T&C marker is present.
            tc_candidate = next(
                (c for c in candidates if _parse_terms_not_accepted(c) is not None),
                None,
            )
            raw_str = (
                tc_candidate
                if tc_candidate is not None
                else next(iter(candidates), "remote job failed")
            )
            # T-CDS-006: if the failed-job message indicates a
            # per-dataset T&C-not-accepted error, persist the canonical
            # ``TermsNotAcceptedError`` record so the polling caller
            # gets the licence ``recovery_url`` instead of a generic
            # ``remote_job_failed``.
            policies = _parse_terms_not_accepted(raw_str)
            if policies is not None:
                tc_err = self._build_terms_not_accepted_error(raw_str, policies)
                error_record_json = self._serialise_error_record(tc_err)
                await self.foundation.persistence.update_workflow_error_if_pending(
                    request_id, "failed", error_record_json
                )
                return

            sanitised_message = self.foundation.sanitiser.sanitise(raw_str)
            # T-CDS-011.3: surface the structured CDS job-state JSON in
            # ``context.backend_diagnostics`` so callers can distinguish
            # "too many concurrent jobs" from "request too large" from
            # an internal server error. The full payload goes through
            # the sanitiser (defence-in-depth — PATs should never reach
            # ``remote_json`` but the sanitiser pass costs near-nothing).
            diagnostics: Any = self.foundation.sanitiser.sanitise(
                copy.deepcopy(remote_json)
            )
            # T-CDS-011 round-2 codex HIGH: server-side ``jobID`` /
            # ``job_id`` UUIDs are needed to file an ECMWF support
            # ticket, but the global ``_SAFE_UUID_KEYS`` allowlist
            # cannot include those keys (CDS ``inputs`` accepts
            # ``inputs.jobID`` from the user and would exfil a
            # UUID-shape secret into provenance). Preserve server
            # job ids LOCALLY here, after we've sanitised — values
            # came from ``client.get_remote(...).json``, never from
            # the caller's payload.
            if isinstance(remote_json, dict):
                # Round-2 cr L3: empirically CDS emits the lowercase-d
                # camelCase ``jobID``; ``job_id`` is forward-compat for
                # the ecmwf-datastores migration. We deliberately do
                # NOT match ``JobID`` / ``JOB_ID`` etc. — a casing
                # change would be a server-side breaking deployment
                # and we want the regression visible in tests rather
                # than silently survived.
                for key in ("jobID", "job_id"):
                    original = remote_json.get(key)
                    if (
                        isinstance(original, str)
                        and _UUID_FULL_RE.fullmatch(
                            original.strip().strip("\"'")
                        )
                    ):
                        diagnostics[key] = original
            err_record = build_error_record(
                "BackendError",
                message=sanitised_message,
                error_subclass="remote_job_failed",
                recovery_action="retry_with_modification",
                # T-CDS-011.4 + T-CDS-014: CDS / EWDS return
                # ``remote_job_failed`` with no server-side log for
                # several common request-shape mistakes. Static hint
                # so the agent / human reader can self-correct on
                # the first failure rather than retry-storming.
                next_action_hint=(
                    "remote_job_failed with no server-side log is "
                    "usually one of: "
                    "(a) legacy `format: ...` — modern CDS expects "
                    "`data_format: netcdf|grib` plus `download_format: "
                    "zip|unarchived`; "
                    "(b) time-invariant auxiliary variables (EFAS "
                    "elevation / soil_depth / upstream_area / etc.) "
                    "do NOT accept `hyear`/`hmonth`/`hday`/`time` — "
                    "drop those fields; "
                    "(c) concurrent-quota saturation — CDS empirically "
                    "throttles to a small handful of active jobs per "
                    "user; serialise submits one-at-a-time or wait "
                    "several minutes between batches; "
                    "(d) missing dataset-specific required field "
                    "(e.g. EFAS v5.0 needs `hydrological_model: "
                    "[lisflood]`). Use cds_describe_dataset to inspect "
                    "the dataset's form before composing the request. "
                    "If you previously relied on cds_describe_dataset's "
                    "`available_inputs` field, that snapshot may be "
                    "days/weeks stale — call cds_apply_constraints("
                    "dataset_id, inputs={}) for the LIVE server-side "
                    "valid values verbatim from the CDS engine before "
                    "blindly retrying."
                ),
                context={"backend_diagnostics": diagnostics},
            )
            error_record_json = err_record.model_dump_json()
            await self.foundation.persistence.update_workflow_error_if_pending(
                request_id, "failed", error_record_json
            )
        else:
            await self.foundation.persistence.update_workflow_status_if_pending(
                request_id,
                status,  # type: ignore[arg-type]
            )

    async def _record_cds_provenance(
        self,
        *,
        request_id: str,
        cache_key: str,
        stored_path: Path,
    ) -> None:
        """Write a provenance sidecar + persistence row for a CDS download.

        T-CDS-011.5: CDS files were previously stored without a
        ``.provenance.json`` sidecar (CMEMS has had this since T-024).
        A user holding a bare ``cds_<hex>.bin`` had no way to
        reconstruct what request produced it without re-querying
        SQLite. The minimal CDS shape:
          - BackendBlock: id="cds", per-store endpoint URL.
          - DatasetBlock: dataset_id from the persisted submit payload.
          - RequestBlock: user_request from the persisted JSON.
          - SpatialBlock / TemporalBlock / variables: omitted (CDS
            ``inputs`` is an opaque dict; extracting them requires
            per-dataset semantic knowledge we deliberately do not
            embed).
          - CacheRef + CostConsumed: minimal placeholders.
        """
        row = await self.foundation.persistence.fetch_workflow(request_id)
        if row is None:
            # cr round-1 M2: the previous comment claimed "logger upstream
            # will warn" — incorrect, the upstream catch only fires on
            # exceptions, not silent returns. Operators inspecting a
            # cached .bin without a sidecar otherwise have no signal
            # about why it's missing.
            logger.warning(
                "cds_provenance_skipped",
                extra={"request_id": request_id, "reason": "row_missing"},
            )
            return
        try:
            payload = json.loads(row["request_json"])
        except (json.JSONDecodeError, TypeError, KeyError):
            payload = {}
        dataset_id = (
            payload.get("dataset_id") if isinstance(payload, dict) else None
        )
        if not isinstance(dataset_id, str) or not dataset_id:
            logger.warning(
                "cds_provenance_skipped",
                extra={
                    "request_id": request_id,
                    "reason": "dataset_id_unparseable",
                },
            )
            return
        endpoint_url = _endpoint_url_for(
            dataset_id, default=_STORE_ENDPOINT_URLS["cds"]
        )
        sanitised_payload = self.foundation.sanitiser.sanitise(payload)
        await self.foundation.provenance.record_successful_retrieve(
            backend=BackendBlock(
                id="cds",
                provider="ECMWF Copernicus",
                endpoint_url=endpoint_url,
                api_version="cdsapi",
            ),
            dataset=DatasetBlock(dataset_id=dataset_id),
            request=RequestBlock(
                operation="submit",
                submitted_at=row.get("created_at") or _iso_now(),
                started_at=row.get("created_at") or _iso_now(),
                finished_at=_iso_now(),
                user_request=sanitised_payload,
                normalized_request=sanitised_payload,
                options_applied={},
            ),
            spatial=None,
            temporal=None,
            variables=[],
            files=[stored_path],
            cost_consumed=CostConsumed(type="free", advisory_message=None),
            source_urls=[],
            cache=CacheRef(cache_key=cache_key, cache_hit=False),
            workflow_request_id=request_id,
        )

    async def _build_status_envelope(
        self,
        row: Mapping[str, Any],
        status: str,
        *,
        _resolved_filepath: Path | None = None,
    ) -> dict[str, Any]:
        """Canonical ``check_status`` envelope. Resolves the cache file
        for successful rows; if the file was LRU-evicted, surface a
        synthetic ``CacheError(cache_eviction)`` rather than returning
        ``status=successful`` with an empty result (mirrors CMEMS at
        backends/cmems/backend.py:849)."""
        result_block: dict[str, Any] = {}
        error_details = _decode_error_record(row.get("error_record_json"))
        emitted_status = status

        if status == "successful":
            cache_key_value = row.get("cache_key") or ""
            cache_path = _resolved_filepath
            if cache_path is None and cache_key_value:
                cache_path = await self.foundation.cache.lookup_file(
                    _cache_storage_key(cache_key_value)
                )

            if cache_path is not None:
                # T-CDS-018: ``_cds_result_metadata`` is the shared
                # builder so submit-cache-hit / check_status /
                # fetch_result emit the same shape (round-2 codex-MED2).
                result_block = {
                    "filepath": str(cache_path),
                    "uri": f"copernicus://files/{cache_key_value}",
                    "metadata": _cds_result_metadata(cache_path),
                    "provenance": {},
                }
            else:
                # Synthetic eviction: don't mutate the persisted row,
                # but downgrade the response so the caller doesn't
                # observe ``successful`` with an unusable result.
                emitted_status = "failed"
                error_details = build_error_record(
                    "CacheError",
                    message=(
                        f"file for {row.get('request_id')!r} was evicted "
                        "from the cache; re-run the request to repopulate"
                    ),
                    error_subclass="cache_eviction",
                    recovery_action="retry_with_modification",
                ).model_dump(mode="json")

        # T-CDS-000 smoke regression: do NOT sanitise the whole
        # envelope — the sanitiser's UUID pattern would redact the
        # server-generated ``request_id``. ``error_details`` is the
        # only user-influenced field; sanitise it specifically. Other
        # fields are safe by construction (row column values, the
        # constructed ``filepath``/``uri``, and ``size_bytes``).
        return {
            "status": emitted_status,
            "request_id": row.get("request_id"),
            "submitted_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
            "cache_key": row.get("cache_key"),
            "error_details": (
                self.foundation.sanitiser.sanitise(error_details)
                if error_details is not None
                else None
            ),
            "result": result_block,
        }

    def _wrap_sdk_error(self, exc: Exception, *, op: str) -> BackendError:
        """Coerce any SDK exception to ``BackendError`` with the message
        passed through the sanitiser to redact a leaked PAT."""
        msg = self.foundation.sanitiser.sanitise(str(exc))
        return BackendError(
            f"CDS {op} failed: {msg}",
            record=build_error_record(
                "BackendError",
                message=f"CDS {op} failed: {msg}",
                error_subclass=f"sdk_{op}_failure",
                recovery_action="retry_with_modification",
            ),
        )

    def _build_terms_not_accepted_error(
        self, raw_message: str, policies: list[dict[str, Any]]
    ) -> TermsNotAcceptedError:
        """Build the canonical ``TermsNotAcceptedError`` from a parsed
        T&C-not-accepted message (T-CDS-006).

        ``recovery_url`` is the FIRST missing-policy URL when the
        server enumerated specific policies; round-1 MEDIUM (cr M-3):
        when the marker is present but no per-policy URL was parseable
        (``policies == []``), fall back to the generic CDS dataset
        landing page so the agent / CLI always has an actionable URL.

        Note on URL safety vs. Sanitiser (round-1 cr H-2): empirical
        CDS / ADS / EWDS licence URLs use slug paths (e.g.
        ``/licences/terms-of-use-ads``) and do not match the Sanitiser
        UUID rule, so the URL survives the persistence round-trip
        through ``_serialise_error_record``. If a future deployment
        emits a UUID-shaped licence URL, the recovery_url WOULD be
        redacted in persisted ``error_record_json``; revisit then by
        adding a known-safe-keys allow-list to the Sanitiser. Test
        ``test_terms_recovery_url_survives_persistence_for_today_urls``
        pins the current contract.
        """
        sanitised = self.foundation.sanitiser.sanitise(raw_message)
        first_url = (
            policies[0]["url"]
            if policies
            else _detect_store_landing(raw_message)
        )
        if policies:
            names = ", ".join(p["name"] for p in policies)
            hint = (
                "Visit the recovery_url (and any additional URLs in "
                "context.missing_policies), accept each licence, and "
                "then re-submit the same request."
            )
        else:
            names = "(server did not enumerate policies)"
            hint = (
                "The server flagged a T&C-acceptance issue but did not "
                "enumerate the missing policies in a parseable form. "
                "Open the dataset's page on the CDS / ADS / EWDS web UI, "
                "accept all required licences in the licence tab, and "
                "re-submit."
            )
        record = build_error_record(
            "TermsNotAcceptedError",
            message=(
                "CDS server rejected the request: the user has not yet "
                "accepted all required site policies for this dataset. "
                f"Missing: {names}. Open the licence page(s) and accept, "
                "then re-submit."
            ),
            error_subclass="terms_not_accepted",
            recovery_action="accept_terms",
            recovery_url=first_url,
            next_action_hint=hint,
            context={
                "missing_policies": policies,
                "raw_sdk_message": sanitised,
            },
        )
        return TermsNotAcceptedError(
            "CDS terms-of-use not accepted",
            record=record,
        )

    def _serialise_error_record(self, exc: Any) -> str:
        """Serialise the canonical ``error_record`` for persistence.

        Round-1 MEDIUM (code-reviewer): a missing ``error_record`` used
        to silently produce ``"{}"`` (an invalid ErrorRecord). Synthesise
        a structured record from ``str(exc)`` instead so the row never
        carries opaque corruption.

        Round-2 HIGH (codex + code-reviewer): sanitise the **dict**
        first, then ``json.dumps``. The previous order — dump JSON,
        then sanitise the string — corrupted JSON validity when the
        sanitiser substituted a value inside a quoted string context
        (``"password":"x"`` → ``"password":[REDACTED]``). CMEMS uses
        the dict-then-dumps pattern.
        """
        record = getattr(exc, "error_record", None)
        if record is None:
            record = build_error_record(
                "BackendError",
                message=self.foundation.sanitiser.sanitise(str(exc)),
                error_subclass="unknown_sdk_failure",
                recovery_action="retry_with_modification",
            )
        sanitised = self.foundation.sanitiser.sanitise(record.model_dump(mode="json"))
        return json.dumps(sanitised, sort_keys=True)

    async def fetch_result(self, request_id: str, target: Path) -> dict[str, Any]:
        row = await self.foundation.persistence.fetch_workflow(request_id)
        if row is None:
            raise NotFoundError(
                f"workflow {request_id!r} not found",
                record=build_error_record(
                    "NotFoundError",
                    message=f"workflow {request_id!r} not found",
                    recovery_action="modify_request_parameters",
                ),
            )
        if row["status"] != "successful":
            raise BackendError(
                f"workflow {request_id!r} status={row['status']!r}, no result",
                record=build_error_record(
                    "BackendError",
                    message=(
                        f"workflow {request_id!r} status={row['status']!r}, "
                        "no result available — call check_status first"
                    ),
                    error_subclass="result_not_ready",
                    recovery_action="report_to_administrator",
                ),
            )

        cache_key = row["cache_key"] or ""
        cache_path = await self.foundation.cache.lookup_file(_cache_storage_key(cache_key))
        if cache_path is None:
            raise CacheError(
                f"workflow {request_id!r} successful but file missing from cache",
                record=build_error_record(
                    "CacheError",
                    message=(f"workflow {request_id!r} successful but file evicted"),
                    error_subclass="cache_eviction",
                    recovery_action="retry_with_modification",
                ),
            )

        # T-CDS-000 smoke regression: envelope is safe by construction;
        # don't run the UUID-redacting sanitiser over the whole result.
        # Round-2 (codex-MED2): metadata via the shared helper so the
        # three success surfaces agree.
        return (
            {
                "status": "successful",
                "cache_hit": True,
                "is_existing": True,
                "request_id": request_id,
                "cache_key": cache_key,
                "result": {
                    "filepath": str(cache_path),
                    "uri": f"copernicus://files/{cache_key}",
                    "metadata": _cds_result_metadata(cache_path),
                    "provenance": {},
                },
            }
        )

    async def cancel(self, request_id: str) -> dict[str, Any]:
        row = await self.foundation.persistence.fetch_workflow(request_id)
        if row is None:
            raise NotFoundError(
                f"workflow {request_id!r} not found",
                record=build_error_record(
                    "NotFoundError",
                    message=f"workflow {request_id!r} not found",
                    recovery_action="modify_request_parameters",
                ),
            )

        terminal = {"successful", "failed", "cancelled"}
        if row["status"] in terminal:
            return {
                "cancelled": False,
                "reason": f"already terminal (status={row['status']})",
                "request_id": request_id,
                "status": row["status"],
            }

        # Best-effort SDK delete — failures are swallowed because the
        # remote may have already finalised. the project conventions gotcha #8 carries
        # over: cancellation is best-effort.
        if self._auth_adapter is not None:
            import contextlib as _ctx

            client = _make_cdsapi_client(
                self._auth_adapter,
                dataset_id=_dataset_id_from_workflow_row(row),
            )
            with _ctx.suppress(Exception):
                await asyncio.to_thread(client.client.delete, request_id)

        # Atomic conditional UPDATE: never overwrite a row that raced to
        # ``successful`` / ``failed`` between our fetch and write.
        await self.foundation.persistence.update_workflow_status_if_pending(request_id, "cancelled")
        final = await self.foundation.persistence.fetch_workflow(request_id)
        final_status = final["status"] if final is not None else "unknown"
        return {
            "cancelled": final_status == "cancelled",
            "request_id": request_id,
            "status": final_status,
        }

    # --- capabilities ----------------------------------------------------

    @property
    def supports_async(self) -> bool:
        # CDS is queue-backed; async is the only supported mode (research
        # §6.5.1). Reflected here even though no operations work yet.
        return True

    @property
    def supports_dry_run(self) -> bool:
        # No CDS analogue to ``copernicusmarine.subset(dry_run=True)``.
        # Estimation will be heuristic per research §6.7.4.
        return False

    @property
    def requires_terms_acceptance(self) -> bool:
        # Per research §6.6: per-dataset T&C acceptance is mandatory and
        # cannot be automated. T-CDS-006 wires up the elicitation flow.
        return True
