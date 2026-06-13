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
import math
import re
import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

import httpx

from copernicus_mcp.auth.cds import CdsApiKeyAdapter
from copernicus_mcp.auth.resolver import ResolvedCredentials
from copernicus_mcp.backends.abstract import AbstractBackend, FoundationServices
from copernicus_mcp.backends.cds import catalogue as _catalogue
from copernicus_mcp.backends.cds import estimator as _estimator
from copernicus_mcp.backends.cds.chunking import (
    ChunkPlan,
    ChunkPlanError,
    apply_chunk,
    build_chunk_plan,
    compute_parent_status,
    granularity_estimates,
    is_splittable,
)
from copernicus_mcp.backends.cds.costing import fetch_costing
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
from copernicus_mcp.persistence.protocol import WorkflowStatus
from copernicus_mcp.workflow.confirmation import ConfirmationRequired

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


_CALIBRATION_SEED: dict[tuple[str, str], dict[str, Any]] | None = None


def _calibration_seed() -> dict[tuple[str, str], dict[str, Any]]:
    """Bundled calibration seed, loaded once (T-CDS-EST2-004)."""
    global _CALIBRATION_SEED  # noqa: PLW0603
    if _CALIBRATION_SEED is None:
        from copernicus_mcp.backends.cds.calibration import load_seed

        _CALIBRATION_SEED = load_seed()
    return _CALIBRATION_SEED


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


def _load_chunk_plan(row: Mapping[str, Any]) -> dict[str, Any]:
    """Parse a workflow row's ``chunk_plan_json`` into a dict ({} if absent/bad)."""
    raw = row.get("chunk_plan_json")
    if isinstance(raw, str):
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                return loaded
    return {}


def _parent_multifile_result(descriptors: list[dict[str, Any]]) -> dict[str, Any]:
    """The multi-file ``result`` block for a successful chunked parent — one file
    per chunk plus the format flag + merge hint. Shared by ``check_status`` (so a
    successful poll returns the files directly, no second download call) and
    ``fetch_result`` / download."""
    formats = sorted({d["content_type"] for d in descriptors if d["content_type"]})
    return {
        "files": descriptors,
        "formats": formats,
        "heterogeneous_formats": len(formats) > 1,
        "merge_hint": (
            "files are a non-overlapping set ordered by chunk_index, split along "
            "the calendar/time axis — merge them yourself, e.g. "
            "xarray.open_mfdataset(sorted_filepaths, combine='nested', "
            "concat_dim='time'). The MCP does NOT stitch or re-encode; if "
            "heterogeneous_formats is true, convert to one format first."
        ),
        "metadata": {"chunk_count": len(descriptors)},
        "provenance": {},
    }


def _chunk_parent_response(
    *,
    parent_id: str,
    cache_key: str | None,
    plan: dict[str, Any],
    child_status: dict[str, str],
    status: str,
    files: list[dict[str, Any]] | None = None,
    evicted: list[int] | None = None,
) -> dict[str, Any]:
    """Aggregate ``check_status`` envelope for a chunked parent (T-CDS-CHUNK-003):
    the parent status + per-chunk breakdown + state counts. An unsubmitted chunk
    counts as ``queued``; a submitted child with no known status as ``running``.
    When the parent is successful, ``files`` carries the resolved per-chunk
    descriptors so the response includes the multi-file result directly (the
    agent does not need a second download call)."""
    chunks = plan.get("chunks", []) if isinstance(plan.get("chunks"), list) else []
    counts = {
        "total": len(chunks),
        "successful": 0,
        "running": 0,
        "queued": 0,
        "failed": 0,
        "cancelled": 0,
    }
    per_chunk: list[dict[str, Any]] = []
    for chunk in chunks:
        cid = chunk.get("child_request_id")
        state = (child_status.get(cid) or "running") if cid else "queued"
        if state in counts:
            counts[state] += 1
        per_chunk.append(
            {"index": chunk.get("index"), "request_id": cid, "status": state}
        )
    if status == "successful" and files is not None:
        result = _parent_multifile_result(files)
        if evicted:
            result["evicted_chunk_indices"] = evicted
    else:
        result = {
            "uri": f"copernicus://jobs/{parent_id}",
            "metadata": {"chunk_count": len(chunks)},
            "provenance": {},
        }
    return {
        "status": status,
        "cache_hit": False,
        "is_existing": True,
        "request_id": parent_id,
        "cache_key": cache_key,
        "chunked": True,
        "chunk_count": len(chunks),
        # T-DOWNLOAD-PROGRESS: explicit download position (completed parts / total)
        # over the per-state ``chunks`` breakdown, for an at-a-glance progress read.
        "progress": {"completed": counts["successful"], "total": counts["total"]},
        "chunks": counts,
        "per_chunk": per_chunk,
        "result": result,
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

    # T-CDS-EST2-002: ``estimated_size_bytes`` may be ``None`` (whole-file
    # product, epistemic_status="unknown") — pass it through; the builder
    # omits the GB field rather than crashing on ``int(None)``.
    bytes_estimate = estimate.get("estimated_size_bytes")
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
    # Pass the real epistemic_status (was hard-coded "approximate", a value the
    # v2 estimator never emits) plus the cost block so the agent sees why.
    confirmation.payload["context"]["epistemic_status"] = estimate.get("epistemic_status")
    confirmation.payload["context"]["cost"] = estimate.get("cost")
    return confirmation


def _missing_product_type(
    dataset_id: str, inputs: dict[str, Any]
) -> list[str] | None:
    """Return the dataset's valid ``product_type`` values if it REQUIRES
    product_type but ``inputs`` omits it (absent or empty); else ``None``.

    Scoped to the ECMWF **reanalysis** families (``reanalysis-*``: ERA5,
    ERA5-Land, CERRA, CARRA, ...), where ``product_type`` (reanalysis vs
    ensemble_*) is a required selection. Omitting it makes CDS split a
    multi-month request server-side into per-month MARS sub-requests, which
    dies with a cryptic ``UserError: Duplicate value for month`` after
    ~40 min in the queue (a small single request slips through, so the
    failure looks intermittent). We catch it here, pre-network, with the
    valid values (T-CDS-PT, WP3 field report 2026-06-12).

    The scope is deliberately narrow. The bundled constraints come from
    POSTing EMPTY inputs to CDS, which lists every *selectable* field — not
    a required flag (``data_format`` is listed too yet defaults). So merely
    appearing in the snapshot does NOT prove required; many non-reanalysis
    families (satellite/insitu/sis/derived/...) list a ``product_type`` that
    may be optional. Widen this from a field report, not from the snapshot
    alone (v2 review MEDIUM)."""
    if not dataset_id.startswith("reanalysis-"):
        return None
    constraints = _catalogue.load_constraints().get(dataset_id)
    if not isinstance(constraints, dict):
        return None
    values = constraints.get("product_type")
    if not isinstance(values, list) or not values:
        return None
    provided = inputs.get("product_type")
    if provided is None or (
        isinstance(provided, (list, tuple, str)) and len(provided) == 0
    ):
        return [str(v) for v in values]
    return None


def _build_chunk_proposal(
    *,
    estimate: dict[str, Any],
    inputs: dict[str, Any],
    cost_units: float,
    cost_limit: float,
) -> ConfirmationRequired:
    """T-CDS-CHUNK-002 (model B): the agent-facing split proposal.

    Raised when a request exceeds the dataset's server-side cost limit and is
    splittable, but the agent has not yet chosen a granularity. The MCP does the
    arithmetic (minimum chunk count per calendar axis, from the exact ``/costing``
    numbers); the agent owns the strategy and re-submits with
    ``__options.chunk_by ∈ {year, month, day}`` (or ``confirmed=true`` to accept
    the default ``year``). ``chunked: true`` tells the agent the result will be a
    multi-file set, one file per chunk."""
    # Per-granularity: clean chunk count, estimated cost per whole unit, and
    # whether that estimate fits the limit — so the agent sees e.g. "a single
    # year is over the limit, use months" instead of guessing.
    estimates = granularity_estimates(inputs, cost_units, cost_limit)
    # Suggest the COARSEST granularity that is estimated to fit (year < month <
    # day); fall back to the finest available if none clearly fits.
    suggested = next(
        (g for g in ("year", "month", "day") if estimates.get(g, {}).get("fits")),
        None,
    )
    if suggested is None:
        suggested = next(reversed(estimates), "month") if estimates else "month"
    payload: dict[str, Any] = {
        "confirmation_required": True,
        "reason": "cost_limit_requires_chunking",
        "chunked": True,
        "estimated_cost": {
            "type": "cds_cost_units",
            "cost_units": cost_units,
            "cost_limit": cost_limit,
        },
        "chunking": {
            "suggested_granularity": suggested,
            "granularities": estimates,
        },
        "next_action": (
            f"re-submit cds_submit_request with __options.chunk_by (suggested: "
            f"{suggested!r}; pick a granularity whose est cost per chunk is under "
            "the limit), or confirmed=true to accept the suggestion; the result is "
            "a multi-file set, one file per chunk"
        ),
        "context": {
            "tool_name": "cds_submit_request",
            "backend": "cds",
            "cost_units": cost_units,
            "cost_limit": cost_limit,
            "queue_latency_tier": estimate.get("queue_latency_tier"),
            "estimated_size_bytes": estimate.get("estimated_size_bytes"),
        },
    }
    return ConfirmationRequired(payload)


def _build_chunk_count_confirmation(
    *,
    chunk_count: int,
    threshold: int,
    reason: str,
    require_large_ack: bool,
) -> ConfirmationRequired:
    """Fan-out confirmation: the validated split would launch ``chunk_count`` CDS
    jobs at once (> ``threshold``). Tier 1 (``require_large_ack=False``) is cleared
    by ``confirmed=true`` — a human in the loop for a sizeable batch. Tier 2 needs
    a SECOND, deliberate ack (``__options.confirm_large_fanout``) so a glitched
    agent that blanket-sets ``confirmed`` cannot launch a runaway fan-out."""
    if require_large_ack:
        next_action = (
            f"this split launches {chunk_count} CDS jobs at once (> {threshold}) — a "
            "large fan-out. Re-submit cds_submit_request with confirmed=true AND "
            "__options.confirm_large_fanout=true to proceed, or narrow the request / "
            "use a coarser granularity for fewer parts."
        )
    else:
        next_action = (
            f"this split launches {chunk_count} CDS jobs at once (> {threshold}). "
            "Re-submit cds_submit_request with confirmed=true to proceed (a human "
            "should approve a large batch), or narrow the request."
        )
    payload: dict[str, Any] = {
        "confirmation_required": True,
        "reason": reason,
        "chunked": True,
        "chunk_count": chunk_count,
        "next_action": next_action,
        "context": {
            "tool_name": "cds_submit_request",
            "backend": "cds",
            "chunk_count": chunk_count,
            "confirm_threshold": threshold,
        },
    }
    return ConfirmationRequired(payload)


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
        # T-CDS-ASYNC-DOWNLOAD: in-flight background result-file downloads keyed by
        # request_id. The fetch of a successful job's file runs as a background task
        # (after a short inline grace) so check_status returns without blocking the
        # agent on the transfer. In-memory only — a restart loses the task and the
        # next poll re-spawns it (the row is still "running" + the CDS job is done).
        self._downloads: dict[str, asyncio.Task[Any]] = {}
        # T-CDS-ASYNC-DOWNLOAD review HIGH: serialise the spawn DECISION (is there a
        # live download? else create one) so two concurrent first-polls cannot
        # double-spawn and cancel / the done-callback always target the one live
        # task. Held only across one DB read — never the download itself.
        self._downloads_mutex = asyncio.Lock()
        # T-CDS-EST2-003: pre-flight costing keyed by request_id, carried from
        # submit to the (later, possibly different-process) finalise where the
        # size observation is written. Bounded FIFO so a submit never polled to
        # terminal cannot leak; a missing entry ⇒ observation gets NULL cost.
        self._inflight_costing: dict[str, dict[str, float]] = {}
        self._inflight_costing_cap = 256
        # T-CDS-CHUNK-003: per-parent ref-counted lock (decisions 5/8). Both the
        # poll-driven advancement (`check_status`) and `cancel` serialise on it so
        # two concurrent polls can never submit the same next wave, and a cancel
        # never races a poll into orphaning a fresh child under a stopped parent.
        # Same ref-count discipline as `_submit_locks`.
        self._parent_locks: dict[str, asyncio.Lock] = {}
        self._parent_lock_refs: dict[str, int] = {}
        self._parent_locks_dict_mutex = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def _parent_lock(self, parent_id: str) -> AsyncIterator[None]:
        """Acquire the ref-counted per-parent lock (T-CDS-CHUNK-003)."""
        async with self._parent_locks_dict_mutex:
            if parent_id not in self._parent_locks:
                self._parent_locks[parent_id] = asyncio.Lock()
                self._parent_lock_refs[parent_id] = 0
            self._parent_lock_refs[parent_id] += 1
            lock = self._parent_locks[parent_id]
        try:
            async with lock:
                yield
        finally:
            async with self._parent_locks_dict_mutex:
                self._parent_lock_refs[parent_id] -= 1
                if self._parent_lock_refs[parent_id] == 0:
                    self._parent_locks.pop(parent_id, None)
                    self._parent_lock_refs.pop(parent_id, None)

    def _remember_costing(self, request_id: str, cost: dict[str, Any]) -> None:
        """Store the pre-flight cost for ``request_id`` (FIFO-bounded)."""
        units = cost.get("units")
        limit = cost.get("limit")
        if units is None or limit is None:
            return
        if request_id not in self._inflight_costing:
            while len(self._inflight_costing) >= self._inflight_costing_cap:
                # Drop the oldest entry (insertion-ordered dict).
                oldest = next(iter(self._inflight_costing))
                self._inflight_costing.pop(oldest, None)
        self._inflight_costing[request_id] = {
            "units": float(units),
            "limit": float(limit),
        }

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

        costing = await fetch_costing(
            req.dataset_id,
            dict(req.inputs),
            http_client_factory=self.foundation.http_client_factory,
            catalogue=_catalogue,
        )
        # T-CDS-EST2-004: build a calibration lookup from this dataset's
        # observations (all signatures, for the dataset-median fallback) blended
        # with the bundled seed.
        from copernicus_mcp.backends.cds.calibration import CalibrationLookup

        observations = await self.foundation.persistence.list_size_observations(
            "cds", req.dataset_id, None
        )
        calibration = CalibrationLookup(
            observations=observations, seed=_calibration_seed()
        )
        result = _estimator.estimate(
            req.dataset_id,
            dict(req.inputs),
            costing=costing,
            calibration=calibration,
        )
        return self.foundation.sanitiser.sanitise(result)  # type: ignore[no-any-return]

    async def _submit_one(
        self,
        *,
        req: CdsRetrieveRequest,
        safe_params: dict[str, Any],
        cache_key: str,
        cost: Mapping[str, Any] | None,
        parent_request_id: str | None = None,
    ) -> str:
        """Submit ONE retrieve to cdsapi, record its workflow row, and remember
        the pre-flight cost for later calibration capture. Returns the CDS
        ``request_id``.

        Shared by the normal single-request path and the auto-chunk children
        (T-CDS-CHUNK-002). It does NOT run the idempotency lookup, confirmation
        gate, cost-limit, or chunk branch — the caller owns those and already
        holds the per-cache-key submit lock. ``parent_request_id`` links a chunk
        child to its logical parent (``None`` for a normal top-level submit).
        """
        assert self._auth_adapter is not None
        # Round-1 MEDIUM (codex): wrap SDK exceptions so any PAT embedded in an
        # exception string is rewritten through the sanitiser before reaching
        # the caller. T-CDS-006: detect the per-dataset T&C-not-accepted error
        # before the generic SDK wrap, so the agent/CLI gets the canonical
        # ``TermsNotAcceptedError`` envelope with the licence ``recovery_url``.
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
            raw = str(exc)
            policies = _parse_terms_not_accepted(raw)
            if policies is not None:
                raise self._build_terms_not_accepted_error(raw, policies) from exc
            raise self._wrap_sdk_error(exc, op="submit") from exc

        request_id = str(remote.request_id)
        now = _iso_now()
        # Round-1/4 MEDIUM (codex): a ``record_workflow`` failure between the
        # remote submit and the local row commit orphans a queue slot. The
        # ``try/finally`` + ``asyncio.shield`` cleanup runs even on
        # ``CancelledError`` (invariant 3: we never catch it — it propagates
        # after the finally).
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
                    "parent_request_id": parent_request_id,
                }
            )
            recorded = True
            # T-CDS-EST2-003: remember the pre-flight cost so the later finalise
            # can write a calibration observation.
            if isinstance(cost, Mapping):
                self._remember_costing(request_id, dict(cost))
        finally:
            if not recorded:
                with contextlib.suppress(Exception):
                    await asyncio.shield(
                        asyncio.to_thread(client.client.delete, request_id)
                    )
        return request_id

    def _build_cost_limit_error(
        self, units: float, limit: float
    ) -> CmcpValidationError:
        """The EST2 manual-split rejection — raised when a request exceeds the
        dataset cost limit but cannot (or must not) be auto-chunked. Context
        shape is stable (callers and tests depend on it)."""
        chunks = math.ceil(units / limit) if limit > 0 else 0
        return CmcpValidationError(
            "CDS request exceeds the dataset cost limit",
            record=build_error_record(
                "ValidationError",
                message=(
                    f"request cost {units:g} exceeds the dataset limit "
                    f"{limit:g}; split it along year (then month) so each part "
                    "stays under the limit"
                ),
                recovery_action="modify_request_parameters",
                context={
                    "cost_units": units,
                    "cost_limit": limit,
                    "suggested_split": {"dimension": "year", "chunks": chunks},
                },
            ),
        )

    def _build_chunk_plan_error(
        self, exc: ChunkPlanError, units: float, limit: float
    ) -> CmcpValidationError:
        """Map a ``ChunkPlanError`` (auto-chunk could not produce a usable plan)
        to a ValidationError the agent can act on."""
        messages = {
            "not_splittable": (
                "request exceeds the cost limit and has no list-shaped year "
                "axis to split along; narrow it manually"
            ),
            "too_many_chunks": (
                "splitting this request under the cost limit would need more "
                "than the allowed number of chunks; narrow the request "
                "(fewer years/variables/levels) and retry"
            ),
            "costing_unavailable": (
                "the cost endpoint became unavailable while validating the "
                "split; retry shortly"
            ),
            "exceeds_at_finest": (
                "the request cannot be split under the cost limit even at the "
                "finest calendar granularity; narrow it (fewer variables, "
                "levels, or a smaller area) and retry"
            ),
        }
        message = messages.get(exc.reason, "could not build a chunk plan")
        return CmcpValidationError(
            "CDS request could not be auto-chunked",
            record=build_error_record(
                "ValidationError",
                message=message,
                recovery_action="modify_request_parameters",
                context={
                    "cost_units": units,
                    "cost_limit": limit,
                    "chunk_plan_reason": exc.reason,
                },
            ),
        )

    async def _submit_chunk_child(
        self,
        *,
        dataset_id: str,
        parent_inputs: dict[str, Any],
        overrides: dict[str, Any],
        units: float,
        cost_limit: float,
        parent_id: str,
    ) -> str:
        """Submit ONE chunk child: build its narrowed request + cache key,
        sanitise, and submit via ``_submit_one`` with the parent link. Returns the
        child request_id. Shared by the first wave (``_submit_chunk_parent``) and
        the poll-driven refills (``check_status`` advancement); the caller writes
        the returned id back into the plan and persists it."""
        child_inputs = apply_chunk(parent_inputs, overrides)
        child_req = CdsRetrieveRequest.model_validate(
            {"dataset_id": dataset_id, "inputs": child_inputs}
        )
        child_cache_key = self.foundation.data_model.cache_key_for_cds_retrieve(
            child_req
        )
        # codex CHUNK-003 HIGH (r4/r5): a child can be durably submitted (its row
        # committed with this ``parent_request_id``) but lost from the persisted
        # plan if an interrupted ``update_chunk_plan`` never committed. ADOPT that
        # orphan by its cache_key instead of submitting a duplicate CDS job. Scope
        # the search to THIS parent's own children — the global newest-by-cache_key
        # lookup would be shadowed by a same-chunk child of a *different*
        # (legitimately separate) parent and miss the orphan.
        for child in await self.foundation.persistence.list_child_workflows(parent_id):
            if child.get("cache_key") == child_cache_key:
                return child["request_id"]
        child_safe = self.foundation.sanitiser.sanitise(
            {"dataset_id": dataset_id, "inputs": child_inputs}
        )
        return await self._submit_one(
            req=child_req,
            safe_params=child_safe,
            cache_key=child_cache_key,
            cost={"units": units, "limit": cost_limit},
            parent_request_id=parent_id,
        )

    async def _submit_chunk_parent(
        self,
        *,
        req: CdsRetrieveRequest,
        plan: ChunkPlan,
        cache_key: str,
        cost_limit: float,
    ) -> dict[str, Any]:
        """Create the logical parent workflow row for a chunked request and
        submit ALL of its children at once (T-CDS-CHUNK v2 — no inflight throttle;
        CDS queues any excess rather than rejecting it). Returns the parent
        (multi-file) envelope — same family as a single pending submit, with
        additive ``chunked``/``chunk_count`` fields.

        Called inside the per-cache-key submit lock, so the parent row and its
        children are created atomically with respect to a duplicate submit. If a
        child submit is interrupted, a later ``check_status`` poll completes the
        remaining children (orphan recovery)."""
        inputs = dict(req.inputs)
        parent_id = _new_request_id()
        now = _iso_now()
        chunk_entries: list[dict[str, Any]] = [
            {
                "index": index,
                "overrides": spec.overrides,
                "child_request_id": None,
                "units": spec.units,
            }
            for index, spec in enumerate(plan.chunks)
        ]
        plan_doc: dict[str, Any] = {
            "granularity": plan.granularity,
            "stopped": False,
            "cost_limit": cost_limit,
            "chunks": chunk_entries,
        }
        # The parent carries the canonical full request + the unchunked
        # cache_key, so a duplicate submit dedupes against it (decision 3). It is
        # a logical container only — no cdsapi job, no file under the parent key.
        parent_request_json = json.dumps(
            self.foundation.sanitiser.sanitise(
                {"dataset_id": req.dataset_id, "inputs": inputs}
            ),
            sort_keys=True,
            default=str,
        )
        await self.foundation.persistence.record_workflow(
            {
                "request_id": parent_id,
                "backend_id": "cds",
                "operation": "submit",
                "status": "queued",
                "cache_key": cache_key,
                "request_json": parent_request_json,
                "response_json": None,
                "error_record_json": None,
                "created_at": now,
                "updated_at": now,
                "chunk_plan_json": json.dumps(plan_doc, sort_keys=True, default=str),
            }
        )
        # Submit ALL children at once. The plan is persisted incrementally after
        # EACH child (codex/local Tier-A): a failure or cancellation mid-wave must
        # leave the plan a truthful resume point, not a stale "no children
        # submitted" snapshot. ``_submit_chunk_child`` adopts an already-submitted
        # child by cache_key, so a retry after an interrupted submit never
        # duplicates.
        wave_ok = False
        try:
            for entry in chunk_entries:
                child_id = await self._submit_chunk_child(
                    dataset_id=req.dataset_id,
                    parent_inputs=inputs,
                    overrides=entry["overrides"],
                    units=entry["units"],
                    cost_limit=cost_limit,
                    parent_id=parent_id,
                )
                entry["child_request_id"] = child_id
                await self.foundation.persistence.update_chunk_plan(
                    parent_id, json.dumps(plan_doc, sort_keys=True, default=str)
                )
            wave_ok = True
        finally:
            if not wave_ok:
                # First-wave failure (child T&C/SDK error) or cancellation. The
                # parent must NOT be left ``queued`` — otherwise the cache-key
                # dedupe would return this dead workflow forever, poisoning every
                # retry (codex Tier-A HIGH). Shielded so the cleanup completes
                # even under cancellation; CancelledError is NOT caught here — it
                # propagates after the finally (invariant 3).
                with contextlib.suppress(Exception):
                    await asyncio.shield(
                        self._abort_chunk_parent(
                            parent_id=parent_id,
                            dataset_id=req.dataset_id,
                            plan_doc=plan_doc,
                        )
                    )
        return {
            "status": "queued",
            "cache_hit": False,
            "is_existing": False,
            "request_id": parent_id,
            "cache_key": cache_key,
            "chunked": True,
            "chunk_count": len(chunk_entries),
            "result": {
                "uri": f"copernicus://jobs/{parent_id}",
                "metadata": {"chunk_count": len(chunk_entries)},
                "provenance": {},
            },
        }

    async def _abort_chunk_parent(
        self,
        *,
        parent_id: str,
        dataset_id: str,
        plan_doc: dict[str, Any],
    ) -> None:
        """Clean up after a failed/cancelled first wave (Tier-A HIGH, both
        reviewers).

        Order matters: the parent-status transition is the dedupe-poison CURE, so
        it runs FIRST and LOCAL-ONLY — it must never be gated behind the
        best-effort remote child cleanup (round-2 HIGH: a failing
        ``_make_cdsapi_client`` or a hanging remote delete would otherwise leave
        the parent ``queued`` and the poison would resurface). Each step is
        independently suppressed so a bookkeeping failure cannot mask the original
        error that triggered the abort."""
        # 1. CURE FIRST (local, unconditional): move the parent out of queued so
        #    the cache-key dedupe can never return this dead workflow.
        with contextlib.suppress(Exception):
            await self.foundation.persistence.update_workflow_status(
                parent_id, "failed"
            )
        # 2. Persist the plan as a truthful resume point.
        plan_doc["stopped"] = True
        with contextlib.suppress(Exception):
            await self.foundation.persistence.update_chunk_plan(
                parent_id, json.dumps(plan_doc, sort_keys=True, default=str)
            )
        # 3. Best-effort remote cleanup of any children already submitted (no
        #    leaked CDS jobs under a dead parent). Shared with cancel; entirely
        #    suppressed and decoupled from the cure above.
        await self._cancel_chunk_children(dataset_id, plan_doc)

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

        # T-CDS-PT: product_type is a required selection field for the
        # datasets whose constraints define it. Omitting it lets CDS split a
        # multi-month request server-side and fail ~40 min later with a
        # cryptic "Duplicate value for month". Reject it now (pre-network,
        # like the dataset_id shape check) with the valid values.
        missing_pt = _missing_product_type(req.dataset_id, req.inputs)
        if missing_pt is not None:
            raise CmcpValidationError(
                f"CDS dataset {req.dataset_id!r} requires a 'product_type' "
                f"selection; add one of {missing_pt}",
                record=build_error_record(
                    "ValidationError",
                    message=(
                        f"CDS dataset {req.dataset_id!r} requires a "
                        f"'product_type' selection (valid values: "
                        f"{missing_pt}). Omitting it makes CDS split a "
                        "multi-month request server-side and fail with "
                        "'Duplicate value for month'."
                    ),
                    recovery_action="modify_request_parameters",
                    context={"field": "product_type", "valid_values": missing_pt},
                ),
            )

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

                    # In-flight dedupe. A chunked parent (``chunk_plan_json``
                    # set) dedupes here too: an identical re-submit while the
                    # parent is queued/running must NOT fan out a second set of
                    # children — it returns the SAME parent id.
                    inflight = await self.foundation.persistence.lookup_workflow_by_cache_key(
                        cache_key
                    )
                    if inflight is not None and inflight["status"] in (
                        "queued",
                        "running",
                    ):
                        pending = _pending_response(
                            request_id=inflight["request_id"],
                            cache_key=cache_key,
                            status=inflight["status"],
                        )
                        plan_json = inflight.get("chunk_plan_json")
                        if plan_json:
                            with contextlib.suppress(
                                json.JSONDecodeError, TypeError, AttributeError
                            ):
                                chunks = json.loads(plan_json).get("chunks", [])
                                pending["chunked"] = True
                                pending["chunk_count"] = len(chunks)
                        return pending
                    # Terminal SUCCESSFUL chunked parent (decision 11a/b): rebuild
                    # the multi-file result from the child cache entries — zero CDS
                    # traffic, no duplicate fan-out. Raises CacheError if any chunk
                    # file was evicted (re-run via force_refresh, which skips this
                    # whole block). A failed/cancelled parent falls through to a
                    # fresh re-submit (retry).
                    if (
                        inflight is not None
                        and inflight.get("chunk_plan_json")
                        and inflight["status"] == "successful"
                    ):
                        return await self._fetch_chunk_parent_result(inflight)

                # Round-1 MEDIUM (code-reviewer): use ``self.estimate`` so
                # any future side-effect (telemetry, provenance) added to
                # the public estimate path also fires for submit. The v2
                # estimate also carries the ``/costing`` pre-flight.
                estimate = await self.estimate(params)
                budget = self.foundation.config.budget

                # T-CDS-CHUNK-002 (model B): pre-flight cost-limit handling,
                # BEFORE the confirmation gate. ``cost``/``limit`` are the CDS
                # server's own authoritative numbers (the logic behind the 403),
                # not our byte estimate — so the trigger is exact. Three exits:
                #   - not auto-chunkable (disabled, opted out, or a single
                #     calendar cell that can't be split) → the EST2 manual-split
                #     ValidationError (unchanged);
                #   - splittable but no granularity chosen → the chunk PROPOSAL
                #     (a ConfirmationRequired); the agent re-submits with
                #     ``__options.chunk_by`` (or ``confirmed=true`` ⇒ default year);
                #   - granularity chosen → build the validated plan, create the
                #     parent row, submit the first throttled wave, return the
                #     parent envelope.
                # (Costing unavailable ⇒ ``cost`` is None ⇒ no split, as before.)
                cost = estimate.get("cost")
                if cost is not None and cost.get("exceeds_limit"):
                    units = float(cost["units"])
                    limit = float(cost["limit"])
                    inputs = dict(req.inputs)
                    auto_chunk = (
                        budget.cds_auto_chunk_enabled
                        and options.get("auto_chunk", True) is not False
                        and is_splittable(inputs)
                    )
                    if not auto_chunk:
                        raise self._build_cost_limit_error(units, limit)

                    chunk_by = options.get("chunk_by")
                    if chunk_by is None:
                        if not options.get("confirmed"):
                            raise _build_chunk_proposal(
                                estimate=estimate,
                                inputs=inputs,
                                cost_units=units,
                                cost_limit=limit,
                            )
                        chunk_by = "year"  # confirmed ⇒ accept the default axis
                    if chunk_by not in ("year", "month", "day"):
                        raise CmcpValidationError(
                            f"invalid chunk_by {chunk_by!r}",
                            record=build_error_record(
                                "ValidationError",
                                message=(
                                    "__options.chunk_by must be one of "
                                    f"year/month/day, got {chunk_by!r}"
                                ),
                                recovery_action="modify_request_parameters",
                            ),
                        )

                    async def _costing_fn(
                        child_inputs: dict[str, Any],
                    ) -> float | None:
                        result = await fetch_costing(
                            req.dataset_id,
                            child_inputs,
                            http_client_factory=self.foundation.http_client_factory,
                            catalogue=_catalogue,
                        )
                        return result.units if result is not None else None

                    try:
                        plan = await build_chunk_plan(
                            inputs,
                            limit,
                            chunk_by,
                            costing_fn=_costing_fn,
                            max_chunks=budget.cds_auto_chunk_max_chunks,
                        )
                    except ChunkPlanError as exc:
                        raise self._build_chunk_plan_error(exc, units, limit) from exc

                    # Fan-out confirmation tiers (a large split = many CDS jobs
                    # launched at once). ``max_chunks`` (hard reject) already ran
                    # inside build_chunk_plan; these two gates are softer:
                    #   tier 1 (> confirm_above) ⇒ one confirm (confirmed=true);
                    #   tier 2 (> reconfirm_above) ⇒ a SECOND, deliberate ack
                    #   (confirm_large_fanout) that confirmed alone does not grant,
                    #   so a glitched agent can't blanket-confirm a runaway batch.
                    n_chunks = len(plan.chunks)
                    if (
                        n_chunks > budget.cds_auto_chunk_confirm_above
                        and not options.get("confirmed")
                    ):
                        raise _build_chunk_count_confirmation(
                            chunk_count=n_chunks,
                            threshold=budget.cds_auto_chunk_confirm_above,
                            reason="auto_chunk_job_count",
                            require_large_ack=False,
                        )
                    if (
                        n_chunks > budget.cds_auto_chunk_reconfirm_above
                        and not options.get("confirm_large_fanout")
                    ):
                        raise _build_chunk_count_confirmation(
                            chunk_count=n_chunks,
                            threshold=budget.cds_auto_chunk_reconfirm_above,
                            reason="auto_chunk_job_count_large",
                            require_large_ack=True,
                        )

                    return await self._submit_chunk_parent(
                        req=req,
                        plan=plan,
                        cache_key=cache_key,
                        cost_limit=limit,
                    )

                # Confirmation gate (codex spec review HIGH-3 hybrid):
                #   bytes > threshold OR queue tier in {medium, heavy} OR
                #   size unknown (whole-file product) when configured.
                threshold_bytes = int(
                    budget.cds_per_request_size_warning_gb * 1_000_000_000
                )
                bytes_estimate = estimate.get("estimated_size_bytes")
                tier = estimate.get("queue_latency_tier")
                size_over = bytes_estimate is not None and bytes_estimate > threshold_bytes
                unknown_over = (
                    bytes_estimate is None and budget.cds_confirm_on_unknown_size
                )
                tier_over = tier in budget.cds_confirm_on_queue_tier
                if not options.get("confirmed") and (size_over or unknown_over or tier_over):
                    if size_over:
                        reason = "estimated_size_threshold_exceeded"
                    elif unknown_over:
                        reason = "estimated_size_unknown"
                    else:
                        reason = "queue_latency_tier_exceeded"
                    raise _build_cds_confirmation(
                        estimate=estimate,
                        threshold_bytes=threshold_bytes,
                        reason=reason,
                    )

                # T-CDS-CHUNK-002: the SDK retrieve + workflow-row + costing
                # capture is shared with the auto-chunk children, so it lives in
                # ``_submit_one`` (which holds none of the gate/cost-limit logic
                # above — those have already run for this top-level request).
                request_id = await self._submit_one(
                    req=req,
                    safe_params=safe_params,
                    cache_key=cache_key,
                    cost=estimate.get("cost"),
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

    def _parent_request_inputs(
        self, row: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """``(dataset_id, inputs)`` from the parent's stored request_json, for
        refills. The request_json is sanitised, but the split (calendar) axes are
        gated calendar-clean and other CDS input fields are never
        credential-shaped, so the sanitised inputs faithfully reproduce the child
        requests."""
        dataset_id = _dataset_id_from_workflow_row(row) or ""
        inputs: dict[str, Any] = {}
        raw = row.get("request_json")
        if isinstance(raw, str):
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                payload = json.loads(raw)
                if isinstance(payload, dict) and isinstance(payload.get("inputs"), dict):
                    inputs = payload["inputs"]
        return dataset_id, inputs

    async def _child_status_map(self, plan: dict[str, Any]) -> dict[str, str]:
        """Map each submitted child's request_id → current status (a missing row
        is treated as ``running`` — never promotes the parent to successful)."""
        out: dict[str, str] = {}
        for chunk in plan.get("chunks", []):
            cid = chunk.get("child_request_id")
            if cid:
                r = await self.foundation.persistence.fetch_workflow(cid)
                out[cid] = r["status"] if r is not None else "running"
        return out

    async def _poll_chunk_children(self, plan: dict[str, Any]) -> None:
        """Poll each submitted, non-terminal child via the single-request path
        (which finalises + downloads on success). Best-effort per child — a poll
        error on one must not abort the whole parent advance."""
        for chunk in plan.get("chunks", []):
            cid = chunk.get("child_request_id")
            if not cid:
                continue
            row = await self.foundation.persistence.fetch_workflow(cid)
            if row is None or row["status"] in ("successful", "failed", "cancelled"):
                continue
            with contextlib.suppress(Exception):
                await self.check_status(cid)

    async def _refill_chunk_children(
        self, row: Mapping[str, Any], plan: dict[str, Any]
    ) -> None:
        """Submit any not-yet-submitted chunks (T-CDS-CHUNK v2 — no throttle, so
        normally a no-op because ``submit`` fans out everything; this completes a
        first wave that was interrupted mid-submit). Re-reads the persisted
        ``stopped`` flag inside the lock before each submit (decision 8); a failed
        child or a child-submit error stops further submissions and fails the
        parent (decisions 4/10)."""
        parent_id = row["request_id"]
        dataset_id, inputs = self._parent_request_inputs(row)
        cost_limit = float(plan.get("cost_limit", 0.0))
        while True:
            statuses = await self._child_status_map(plan)
            if any(s in ("failed", "cancelled") for s in statuses.values()):
                # decision 4 + codex HIGH: a failed OR (independently) cancelled
                # child means the aggregate can't complete — stop submitting.
                return
            fresh = await self.foundation.persistence.fetch_workflow(parent_id)
            if fresh is not None and _load_chunk_plan(fresh).get("stopped"):
                plan["stopped"] = True  # decision 8: a concurrent cancel won
                return
            nxt = next(
                (c for c in plan["chunks"] if not c.get("child_request_id")), None
            )
            if nxt is None:
                return  # all chunks submitted
            try:
                child_id = await self._submit_chunk_child(
                    dataset_id=dataset_id,
                    parent_inputs=inputs,
                    overrides=nxt["overrides"],
                    units=nxt["units"],
                    cost_limit=cost_limit,
                    parent_id=parent_id,
                )
            except CopernicusMcpError as exc:
                # decision 10: a child submit failure (T&C / SDK) fails the parent
                # WITH the child's canonical error record preserved (local Tier-A
                # MEDIUM — keeps the licence recovery_url on the refill path, not
                # just the first wave); then stop submitting the rest. No
                # ``stopped`` flag here: that is reserved for user cancel — the
                # explicit ``failed`` status + terminal short-circuit halt the loop.
                record = getattr(exc, "error_record", None)
                if record is not None:
                    err_json = json.dumps(
                        self.foundation.sanitiser.sanitise(record.model_dump(mode="json")),
                        sort_keys=True,
                        default=str,
                    )
                    await self.foundation.persistence.update_workflow_error_if_pending(
                        parent_id, "failed", err_json
                    )
                else:
                    await self.foundation.persistence.update_workflow_status_if_pending(
                        parent_id, "failed"
                    )
                return
            nxt["child_request_id"] = child_id
            await self.foundation.persistence.update_chunk_plan(
                parent_id, json.dumps(plan, sort_keys=True, default=str)
            )

    async def _advance_chunk_parent(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Advance + aggregate a chunked parent under the per-parent lock
        (decisions 4/5): poll children, refill freed waves, recompute the parent
        status, read-repair the row. A terminal parent is returned as-is (no
        re-poll)."""
        parent_id = row["request_id"]
        terminal = ("successful", "failed", "cancelled")
        async with self._parent_lock(parent_id):
            fresh = await self.foundation.persistence.fetch_workflow(parent_id)
            if fresh is None:
                raise NotFoundError(
                    f"workflow {parent_id!r} not found",
                    record=build_error_record(
                        "NotFoundError",
                        message=f"workflow {parent_id!r} not found",
                        recovery_action="modify_request_parameters",
                    ),
                )
            row = fresh
            if row["status"] not in terminal:
                plan = _load_chunk_plan(row)
                await self._poll_chunk_children(plan)
                await self._refill_chunk_children(row, plan)
                refetched = await self.foundation.persistence.fetch_workflow(parent_id)
                if refetched is not None:
                    row = refetched
                if row["status"] not in terminal:
                    statuses = await self._child_status_map(plan)
                    computed = compute_parent_status(plan, statuses)
                    if computed in terminal:
                        await self.foundation.persistence.update_workflow_status_if_pending(
                            parent_id, cast(WorkflowStatus, computed)
                        )
                        refetched = await self.foundation.persistence.fetch_workflow(
                            parent_id
                        )
                        if refetched is not None:
                            row = refetched
            plan = _load_chunk_plan(row)
            statuses = await self._child_status_map(plan)
            status = (
                row["status"]
                if row["status"] in terminal
                else compute_parent_status(plan, statuses)
            )
            # decision 4/8 + codex CHUNK-003 r1/r3: whenever the parent is
            # terminally FAILED or CANCELLED but stray non-terminal children
            # remain, best-effort cancel them. Placed here (after the status is
            # settled, on every poll) so it is RECOVERABLE: if a previous advance
            # or cancel was interrupted mid-cleanup, the next poll of the terminal
            # parent re-cleans the orphans — the poll-driven model (decision 5)
            # instead of a one-shot pre-terminal cleanup. Idempotent: once every
            # sibling is terminal there is nothing left to cancel.
            if status in ("failed", "cancelled") and any(
                s in ("queued", "running") for s in statuses.values()
            ):
                dataset_id = _dataset_id_from_workflow_row(row) or ""
                await self._cancel_chunk_children(dataset_id, plan)
                statuses = await self._child_status_map(plan)
            # When the parent is successful the child files are already downloaded
            # and cached (the poll above finalised them), so return the multi-file
            # descriptor set directly — the agent never needs a second download
            # call (WP3 part-2 feedback). Tolerant: an evicted chunk is flagged in
            # ``evicted_chunk_indices`` rather than raising (download is strict).
            files: list[dict[str, Any]] | None = None
            evicted: list[int] | None = None
            if status == "successful":
                files, evicted = await self._parent_chunk_descriptors(
                    plan, require_all=False
                )
            return _chunk_parent_response(
                parent_id=parent_id,
                cache_key=row.get("cache_key"),
                plan=plan,
                child_status=statuses,
                status=status,
                files=files,
                evicted=evicted,
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

        # T-CDS-CHUNK-003: a chunked PARENT is a logical container whose
        # request_id is synthetic — never poll CDS for it. Instead advance it
        # (poll its children, refill freed waves, recompute the aggregate status)
        # under the per-parent lock.
        if row.get("chunk_plan_json"):
            return await self._advance_chunk_parent(row)

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
            return await self._begin_or_report_download(
                request_id=request_id, row=row, client=client
            )

        if canonical in ("failed", "cancelled"):
            # T-CDS-EST2-003: terminal — drop the pre-flight cost (no download,
            # so no observation is written for this request).
            self._inflight_costing.pop(request_id, None)
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

    async def _begin_or_report_download(
        self, *, request_id: str, row: Mapping[str, Any], client: Any
    ) -> dict[str, Any]:
        """The CDS job is successful server-side; fetch the result file. A small
        file finishes inside the inline grace and this returns ``successful`` in one
        poll; a large one exceeds the grace and finishes in the BACKGROUND, so the
        poll returns ``running`` / ``phase="downloading"`` and the agent is not
        blocked on the transfer (T-CDS-ASYNC-DOWNLOAD). The row stays ``running``
        until the file lands (invariant 5: no new status). A lost background task
        (restart) is re-spawned by the next poll — the row is still ``running`` with
        a successful CDS job and no cached file."""
        # The registry decision (live task? else spawn) must be atomic across the
        # terminal-row recheck — there IS an await (fetch_workflow) between the slot
        # test and the store, so without the mutex two concurrent first-polls
        # double-spawn and cancel / the done-callback can target the wrong task
        # (review HIGH).
        async with self._downloads_mutex:
            task = self._downloads.get(request_id)
            if task is not None and not task.done():
                return await self._downloading_envelope(row)
            if task is not None:
                self._downloads.pop(request_id, None)
            # A finished task (this or a prior poll's) already wrote the terminal row.
            fresh = await self.foundation.persistence.fetch_workflow(request_id)
            if fresh is not None and fresh["status"] in (
                "successful",
                "failed",
                "cancelled",
            ):
                return await self._build_status_envelope(fresh, fresh["status"])
            new_task: asyncio.Task[Any] = asyncio.create_task(
                self._finalise_successful(
                    request_id=request_id,
                    cache_key=row["cache_key"] or "",
                    client=client,
                )
            )
            self._downloads[request_id] = new_task

            def _drop_download(t: asyncio.Task[Any]) -> None:
                self._on_download_done(request_id, t)

            new_task.add_done_callback(_drop_download)
        # Outside the mutex: give the freshly-spawned download a brief inline grace
        # (a small file finishes here and returns successful in one poll).
        grace = self.foundation.config.budget.cds_download_inline_grace_seconds
        done, _pending = await asyncio.wait({new_task}, timeout=max(0.0, grace))
        if new_task in done:
            refreshed = await self.foundation.persistence.fetch_workflow(request_id)
            if refreshed is not None and refreshed["status"] in (
                "successful",
                "failed",
                "cancelled",
            ):
                return await self._build_status_envelope(
                    refreshed, refreshed["status"]
                )
        return await self._downloading_envelope(row)

    async def _downloading_envelope(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """A non-terminal ``running`` envelope with a ``phase="downloading"`` hint —
        the CDS job is done but the result file is still being fetched in the
        background. The persisted row stays ``running`` (invariant 5 untouched)."""
        env = await self._build_status_envelope(row, "running")
        env["phase"] = "downloading"
        return env

    def _on_download_done(self, request_id: str, task: asyncio.Task[Any]) -> None:
        """Drop a finished background download from the in-flight map and log a
        failure (``_finalise_successful`` already marked the row ``failed``). A
        cancellation (user ``cancel`` mid-download) is expected, not an error."""
        # Identity-pop: a newer task for the same id must not be evicted by an older
        # one's callback.
        if self._downloads.get(request_id) is task:
            self._downloads.pop(request_id, None)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning(
                "background CDS download failed for %s: %s",
                request_id,
                self.foundation.sanitiser.sanitise(str(exc)),
            )

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
        # T-CDS-ASYNC-DOWNLOAD review MEDIUM: track staging so the ``finally`` can
        # clean it on the CANCELLED path — CancelledError is a BaseException, so it
        # bypasses every ``except`` cleanup below and would otherwise leak the staged
        # file (a detached download thread can finish writing after the task cancel).
        staging: Path | None = None
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
            # T-CDS-EST2-003: terminal transition — drop the pre-flight cost
            # (covers the success path after the observation write AND the
            # download/sniff/cache_store failure paths that raise through here).
            self._inflight_costing.pop(request_id, None)
            # T-CDS-ASYNC-DOWNLOAD review MEDIUM: best-effort remove a leftover staging
            # dir. Success + every except path already cleaned it; this catches the
            # CANCELLED path, which otherwise leaks a full result file under .staging.
            if staging is not None:
                with contextlib.suppress(OSError):
                    for leftover in staging.iterdir():
                        leftover.unlink()
                    staging.rmdir()

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

        # T-CDS-EST2-003: capture a calibration observation. ``cost`` is peeked
        # (not popped — ``_finalise_successful``'s finally owns the pop, so a
        # download-failure path still cleans up). A missing entry ⇒ NULL cost.
        raw_inputs = payload.get("inputs") if isinstance(payload, dict) else None
        obs_inputs = raw_inputs if isinstance(raw_inputs, dict) else {}
        cost = self._inflight_costing.get(request_id)
        await self._record_size_observation(
            request_id=request_id,
            dataset_id=dataset_id,
            inputs=obs_inputs,
            stored_path=stored_path,
            cost=cost,
        )

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
            cost_consumed=CostConsumed(
                type="free",
                advisory_message=None,
                cost_units=cost["units"] if cost else None,
                cost_limit=cost["limit"] if cost else None,
            ),
            source_urls=[],
            cache=CacheRef(cache_key=cache_key, cache_hit=False),
            workflow_request_id=request_id,
        )

    async def _record_size_observation(
        self,
        *,
        request_id: str,
        dataset_id: str,
        inputs: dict[str, Any],
        stored_path: Path,
        cost: dict[str, float] | None,
    ) -> None:
        """Insert one ``size_observations`` row. Best-effort: a bookkeeping
        failure must never fail an already-successful download. Skips degenerate
        rows where the normalisation would divide by zero (decision 9)."""
        from copernicus_mcp.backends.cds.calibration import signature

        try:
            fraction = _estimator.area_fraction(inputs)
            if fraction <= 0.0:
                return
            cost_units = cost["units"] if cost else None
            if cost_units is not None and cost_units <= 0.0:
                return
            size_bytes = stored_path.stat().st_size
            await self.foundation.persistence.record_size_observation(
                {
                    "observation_id": f"obs-{uuid.uuid4().hex}",
                    "backend_id": "cds",
                    "dataset_id": dataset_id,
                    "signature": signature(inputs),
                    "cost_units": cost_units,
                    "size_bytes": size_bytes,
                    "area_fraction": fraction,
                    "request_id": request_id,
                    "observed_at": _iso_now(),
                }
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("size_observation recording failed", exc_info=True)

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

    async def _parent_chunk_descriptors(
        self, plan: dict[str, Any], *, require_all: bool
    ) -> tuple[list[dict[str, Any]], list[int]]:
        """Resolve each chunk's downloaded file into an ordered descriptor.

        Returns ``(descriptors, evicted_indices)``. ``require_all=True`` means a
        missing/evicted child file is an error the caller raises; ``False`` (the
        partial-files path) just skips it. The MCP returns a descriptor SET — one
        file per chunk, ordered by chunk index — and NEVER stitches or re-encodes
        (decision: merging is the consumer's job)."""
        descriptors: list[dict[str, Any]] = []
        evicted: list[int] = []
        chunks = plan.get("chunks", []) if isinstance(plan.get("chunks"), list) else []
        for chunk in sorted(chunks, key=lambda c: c.get("index", 0)):
            index = chunk.get("index")
            cid = chunk.get("child_request_id")
            child = (
                await self.foundation.persistence.fetch_workflow(cid) if cid else None
            )
            child_key = child.get("cache_key") if child is not None else None
            path = None
            if child_key:
                path = await self.foundation.cache.lookup_file(
                    _cache_storage_key(child_key)
                )
            if path is None:
                evicted.append(index)
                continue
            meta = _cds_result_metadata(path)
            descriptors.append(
                {
                    "chunk_index": index,
                    "request_id": cid,
                    "filepath": str(path),
                    "uri": f"copernicus://files/{child_key}",
                    "size_bytes": path.stat().st_size,
                    "content_type": meta.get("content_type"),
                    "metadata": meta,
                    "span": chunk.get("overrides"),
                }
            )
        return descriptors, evicted

    async def _fetch_chunk_parent_result(
        self, row: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Multi-file result for a chunked parent (decision 9 + format flag)."""
        parent_id = row["request_id"]
        cache_key = row.get("cache_key")
        plan = _load_chunk_plan(row)
        if row["status"] != "successful":
            # Not ready / failed / cancelled → the single-request "no result"
            # error, plus partial_files so completed chunks are never thrown away.
            partial, _ = await self._parent_chunk_descriptors(plan, require_all=False)
            context: dict[str, Any] = {
                "parent_status": row["status"],
                "chunked": True,
                "partial_files": partial,
            }
            # decision 10 (local Tier-A MEDIUM): surface the child's canonical
            # error (e.g. TermsNotAcceptedError with the licence recovery_url) that
            # the refill/abort path preserved on the parent row.
            err_raw = row.get("error_record_json")
            if isinstance(err_raw, str):
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    context["cause"] = json.loads(err_raw)
            raise BackendError(
                f"chunked workflow {parent_id!r} status={row['status']!r}, no result",
                record=build_error_record(
                    "BackendError",
                    message=(
                        f"chunked workflow {parent_id!r} status={row['status']!r}, "
                        "no aggregate result yet — call check_status first"
                    ),
                    error_subclass="result_not_ready",
                    recovery_action="report_to_administrator",
                    context=context,
                ),
            )
        descriptors, evicted = await self._parent_chunk_descriptors(
            plan, require_all=True
        )
        if evicted:
            raise CacheError(
                f"chunked workflow {parent_id!r}: {len(evicted)} chunk file(s) evicted",
                record=build_error_record(
                    "CacheError",
                    message=(
                        f"chunked workflow {parent_id!r} successful but chunk "
                        f"file(s) at index {evicted} were evicted from cache; "
                        "re-submit with force_refresh=true to repopulate"
                    ),
                    error_subclass="cache_eviction",
                    recovery_action="retry_with_modification",
                    context={"evicted_chunk_indices": evicted, "chunked": True},
                ),
            )
        return {
            "status": "successful",
            "cache_hit": True,
            "is_existing": True,
            "request_id": parent_id,
            "cache_key": cache_key,
            "chunked": True,
            "chunk_count": len(descriptors),
            "result": _parent_multifile_result(descriptors),
        }

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
        # T-CDS-CHUNK-003: a chunked parent returns the multi-file descriptor set.
        if row.get("chunk_plan_json"):
            return await self._fetch_chunk_parent_result(row)
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

    async def _cancel_chunk_children(
        self, dataset_id: str, plan: dict[str, Any]
    ) -> None:
        """Best-effort: mark every submitted NON-terminal child cancelled and
        remote-delete its CDS job. Successful/failed children keep their state
        (their files + audit survive).

        Order (codex/local Tier-A): the LOCAL row cancel runs FIRST and
        unconditionally — it is the correctness-relevant part (keeps the parent
        aggregate consistent) and must not depend on building a cdsapi client or
        on a possibly-hanging remote delete completing. The remote delete is then
        pure best-effort (gotcha #8 — the underlying job may run to completion
        regardless). ``update_workflow_status_if_pending`` never overwrites a
        child that just finalised; everything is suppressed."""
        statuses = await self._child_status_map(plan)
        submitted = [
            c["child_request_id"]
            for c in plan.get("chunks", [])
            if c.get("child_request_id")
            and statuses.get(c["child_request_id"])
            not in ("successful", "failed", "cancelled")
        ]
        if not submitted:
            return
        # 1. Local cancel FIRST — fast, no remote dependency.
        for cid in submitted:
            with contextlib.suppress(Exception):
                await self.foundation.persistence.update_workflow_status_if_pending(
                    cid, "cancelled"
                )
            # T-CDS-ASYNC-DOWNLOAD: also stop any in-flight background download.
            child_dl = self._downloads.pop(cid, None)
            if child_dl is not None and not child_dl.done():
                child_dl.cancel()
        # 2. Best-effort remote delete (may run to completion regardless).
        if self._auth_adapter is None:
            return
        with contextlib.suppress(Exception):
            client = _make_cdsapi_client(self._auth_adapter, dataset_id=dataset_id)
            for cid in submitted:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(client.client.delete, cid)

    async def _cancel_chunk_parent(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Cancel a chunked parent under the per-parent lock (decision 8): mark
        the plan ``stopped`` (so a concurrent poll aborts before submitting),
        best-effort cancel in-flight children, then mark the parent cancelled."""
        parent_id = row["request_id"]
        async with self._parent_lock(parent_id):
            fresh = await self.foundation.persistence.fetch_workflow(parent_id)
            if fresh is None:
                raise NotFoundError(
                    f"workflow {parent_id!r} not found",
                    record=build_error_record(
                        "NotFoundError",
                        message=f"workflow {parent_id!r} not found",
                        recovery_action="modify_request_parameters",
                    ),
                )
            row = fresh
            if row["status"] in ("successful", "failed", "cancelled"):
                return {
                    "cancelled": False,
                    "reason": f"already terminal (status={row['status']})",
                    "request_id": parent_id,
                    "status": row["status"],
                    "chunked": True,
                }
            plan = _load_chunk_plan(row)
            plan["stopped"] = True
            with contextlib.suppress(Exception):
                await self.foundation.persistence.update_chunk_plan(
                    parent_id, json.dumps(plan, sort_keys=True, default=str)
                )
            dataset_id = _dataset_id_from_workflow_row(row) or ""
            await self._cancel_chunk_children(dataset_id, plan)
            await self.foundation.persistence.update_workflow_status_if_pending(
                parent_id, "cancelled"
            )
            final = await self.foundation.persistence.fetch_workflow(parent_id)
            final_status = final["status"] if final is not None else "unknown"
            return {
                "cancelled": final_status == "cancelled",
                "request_id": parent_id,
                "status": final_status,
                "chunked": True,
            }

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

        # T-CDS-CHUNK-003 (decision 8): a chunked parent cancels under the SAME
        # per-parent lock the advancement uses, cascading to its children.
        if row.get("chunk_plan_json"):
            return await self._cancel_chunk_parent(row)

        # T-CDS-ASYNC-DOWNLOAD review HIGH: commit ``cancelled`` FIRST, before any
        # further await, so a concurrent poll sees the terminal row and short-circuits
        # (check_status's terminal early-return) instead of re-spawning the download.
        committed = await self.foundation.persistence.update_workflow_status_if_pending(
            request_id, "cancelled"
        )
        # Stop an in-flight background download (best-effort, gotcha #8 — the
        # threadpool transfer may run to completion; ``_finalise_successful`` cannot
        # then resurrect the row, it only commits ``successful`` *_if_pending).
        dl = self._downloads.pop(request_id, None)
        if dl is not None and not dl.done():
            dl.cancel()
        # T-CDS-EST2-003 + codex Tier-A MEDIUM: drop the pre-flight cost ONLY if this
        # cancel won the terminal-state race. If a finalizer beat us to ``successful``,
        # leave the entry — its ``_finalise_successful`` ``finally`` pops it AFTER
        # recording the size observation; popping here would null that cost.
        if committed:
            self._inflight_costing.pop(request_id, None)

        # Best-effort SDK delete — failures are swallowed because the remote may have
        # already finalised. the project conventions gotcha #8: cancellation is best-effort.
        if self._auth_adapter is not None:
            import contextlib as _ctx

            client = _make_cdsapi_client(
                self._auth_adapter,
                dataset_id=_dataset_id_from_workflow_row(row),
            )
            with _ctx.suppress(Exception):
                await asyncio.to_thread(client.client.delete, request_id)

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
