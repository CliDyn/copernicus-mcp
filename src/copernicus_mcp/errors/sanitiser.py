"""Defensive sanitisation pass for outbound payloads.

This is the "last line" before an ``ErrorRecord`` (or any tool result) crosses
back into MCP / CLI output. Primary credential isolation is upstream
(invariant #2 — ``CredentialResolver`` and ``AuthAdapter`` keep raw values
out of payloads). Sanitiser catches accidental leaks via unforeseen paths
(exception traces, third-party library messages, backend diagnostics).

See the upstream documentation §13.5.3 and §9.5.3.

Iter 1 callers feed JSON-shaped Python primitives (output of
``BaseModel.model_dump(mode="json")``) — Pydantic models pass through
unchanged. The sanitiser is intentionally conservative: false positives
are accepted (e.g. UUID-shape strings get redacted) because under-redaction
is dangerous and over-redaction is just less convenient.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any, Final

REDACTED: Final[str] = "[REDACTED]"

_MAX_DEPTH: Final[int] = 32

# Dict keys (case-insensitive) whose values are unconditionally replaced
# with [REDACTED] regardless of value type. Codex T-008 diff review:
# string-pattern matching alone misses real Python dicts like
# {"password": "hunter2"} because the value 'hunter2' has no surrounding
# 'password=' literal.
_SENSITIVE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "password",
        "client_secret",
        "access_token",
        "refresh_token",
        "id_token",
        "api_key",
        "apikey",
        "x-api-key",
        "token",
        "private_token",
        "private-token",
        "personal_access_token",
        "secret",
        "secret_key",
        "ocp-apim-subscription-key",
        "cookie",
        "set-cookie",
        "authorization",
        "copernicusmarine_service_password",
    }
)

# Replacement that preserves scheme/host/path while wiping userinfo:
#   https://user:pw@host/x → https://[REDACTED]@host/x
_URL_USERINFO_RE = re.compile(
    r"(?P<scheme>https?://)(?P<userinfo>[^/\s:@]+:[^/\s:@]+)@"
)


def _url_userinfo_replace(match: re.Match[str]) -> str:
    return f"{match.group('scheme')}{REDACTED}@"


_Replacement = str | Callable[[re.Match[str]], str]

# Each entry is (compiled_pattern, replacement) where replacement is a string
# (with optional ``\1`` backrefs) or a callable that takes a Match.
_PATTERNS: list[tuple[re.Pattern[str], _Replacement]] = [
    # 1. URL userinfo — runs first so the rest of the URL stays diagnostic.
    (_URL_USERINFO_RE, _url_userinfo_replace),
    # 2. HTTP Basic auth header.
    (
        re.compile(r"(?i)Authorization:\s*Basic\s+[A-Za-z0-9+/=]+"),
        f"Authorization: Basic {REDACTED}",
    ),
    # 3. Bare `Basic <base64>` (catches stray header dumps).
    (re.compile(r"\bBasic\s+[A-Za-z0-9+/=]{8,}"), f"Basic {REDACTED}"),
    # 4. Bearer token. Disambiguate from English-word case ("Use Bearer
    #    authentication") by requiring EITHER (a) at least one
    #    non-alphabetic char in the token (digit/dot/dash/tilde/etc.), OR
    #    (b) the token is 20+ chars long (longer than any plausible
    #    English word that follows "Bearer" in tool messages).
    #    Charset includes ``~`` per RFC 6750 b64token. Trailing boundary is
    #    a NEGATIVE lookahead for any token-charset character — so the
    #    pattern terminates correctly regardless of what punctuation
    #    follows (``: ) ? # > | ! .`` etc.). This avoids the trap of
    #    enumerating an allow-list of terminators and missing one.
    (
        re.compile(
            r"\bBearer\s+"
            r"(?=[A-Za-z0-9._\-+/=~]{8,}(?![A-Za-z0-9._\-+/=~]))"
            r"(?:"
            r"[A-Za-z0-9._\-+/=~]*[\d._\-+/=~][A-Za-z0-9._\-+/=~]*"
            r"|[A-Za-z]{20,}"
            r")"
        ),
        f"Bearer {REDACTED}",
    ),
    # 4b. Authorization header — redact entire value regardless of shape.
    #     Lookahead requires the value to start with a non-bracket,
    #     non-whitespace char so the rule is idempotent (won't re-match
    #     ``authorization: [REDACTED]``) even after greedy ``\s*`` backtracks.
    #     Excludes ``"``/``'`` so JSON-shaped error messages stay parseable.
    (
        re.compile(
            r"(?i)\bauthorization\s*[:=]\s*"
            r"(?=[^\s\[])"
            r"[^\n,;}\[\]\"']+"
        ),
        f"authorization: {REDACTED}",
    ),
    # 5. PRIVATE-TOKEN header (CDS family).
    (
        re.compile(r"(?i)PRIVATE-TOKEN:\s*[A-Za-z0-9._\-+/=]{8,}"),
        f"PRIVATE-TOKEN: {REDACTED}",
    ),
    # 6. AWS access key.
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), REDACTED),
    # 7. UUID 8-4-4-4-12 (any version) — research §13.5.3 conservative pattern;
    #    CDS PATs are UUID-shaped so the FP cost is accepted.
    (
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{12}\b"
        ),
        REDACTED,
    ),
    # 8. AWS signature.
    (re.compile(r"(?i)X-Amz-Signature=[^&\s\"',}]+"), f"X-Amz-Signature={REDACTED}"),
    # NOTE: pattern 4 above already redacts ``Bearer <token>`` for tokens
    # that LOOK like tokens (8+ chars from the token char-class). Resist the
    # temptation to add a broader ``Bearer .*`` rule — it will eat the word
    # "authentication" out of legit prose like "Use Bearer authentication".
    # 9. Sensitive-key = value (env-var / query-string shape). Bounded so we
    #    don't eat neighboring keys. Longer alternatives first so ``token``
    #    doesn't shadow ``access_token``/``refresh_token``/``id_token`` —
    #    Python ``re`` is greedy-leftmost.
    (
        re.compile(
            r"(?i)\b("
            r"COPERNICUSMARINE_SERVICE_PASSWORD|"
            r"personal_access_token|"
            r"access_token|refresh_token|id_token|private_token|"
            r"client_secret|"
            r"secret_key|"
            r"api_key|apikey|"
            r"password|"
            r"set_cookie|cookie|"
            r"authorization|"
            r"secret|"
            r"token"
            r")=[^&\s\"',}\]\[]+"
        ),
        rf"\1={REDACTED}",
    ),
    # 10. Sensitive-key : "value" (JSON / YAML / Python-dict-repr / HTTP-header
    #     shape). Accepts unquoted, double-quoted, or single-quoted keys
    #     (codex T-023 finding); same expanded vocabulary as pattern 9.
    (
        re.compile(
            r"(?i)(['\"]?\b("
            r"personal_access_token|"
            r"access_token|refresh_token|id_token|private_token|"
            r"client_secret|"
            r"secret_key|"
            r"api_key|apikey|"
            r"password|"
            r"set_cookie|cookie|"
            r"authorization|"
            r"secret|"
            r"token"
            r")['\"]?\s*[:=]\s*)"
            r"(?:\"[^\"\\]*(?:\\.[^\"\\]*)*\"|'[^'\\]*(?:\\.[^'\\]*)*'|[^\s,}\]\[]+)"
        ),
        rf"\1{REDACTED}",
    ),
]


# Segment splitter for whole-token key matching. Splits ``backend-password``
# into ``["backend", "password"]`` so a sensitive keyword must appear as a
# whole segment, not as a substring of a longer benign word.
_KEY_SEGMENT_RE: Final[re.Pattern[str]] = re.compile(r"[-_.=/:\s]+")

# Whole-segment sensitive words. Deliberately narrow: ``token`` / ``cookie``
# are NOT here because ``next_token``, ``page_token``, ``token_count``,
# ``cookie_jar``, ``cookie_consent`` are routinely emitted as non-secret
# payload by upstream libraries. Compound names like ``access_token`` are
# matched via the whole-key check in ``_SENSITIVE_KEYS``.
_SENSITIVE_KEY_SEGMENTS: Final[frozenset[str]] = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "apikey",
        "authorization",
    }
)


def _key_is_sensitive(key: str) -> bool:
    """A dict key is sensitive if it matches whole-key OR any segment matches.

    Whole-key match against ``_SENSITIVE_KEYS`` covers compound names
    (``access_token``, ``client_secret``, ``set-cookie``).
    Segment-match against ``_SENSITIVE_KEY_SEGMENTS`` covers upstream-spliced
    keys (``backend-password``, ``X-API-Key-v2`` → segments include
    ``api`` + ``key`` separately, so ``apikey`` won't match — but the whole
    key ``x-api-key`` IS in ``_SENSITIVE_KEYS`` so the case is covered).
    Comparison is case-insensitive.
    """
    lower = key.lower()
    if lower in _SENSITIVE_KEYS:
        return True
    segments = set(_KEY_SEGMENT_RE.split(lower))
    return bool(segments & _SENSITIVE_KEY_SEGMENTS)


def _sanitise_str(value: str) -> str:
    out = value
    for pattern, replacement in _PATTERNS:
        out = pattern.sub(replacement, out)
    return out


# T-CDS-008: server-generated identifier keys whose values are canonical
# UUIDs. PATs are also UUID-shape, but they never reach response payloads
# (invariant #2 — they live in ``AuthAdapter``). Pattern 7's blanket UUID
# redaction would otherwise turn every CDS submit response into
# ``{"request_id": "[REDACTED]"}`` and break poll/cancel/download.
#
# Review round 2 (cr M1, codex L1): allowlist trimmed to only the keys
# that are actually UUID-shape in this codebase. ``trace_id`` is
# ``uuid4().hex`` (no dashes — pattern 7 already ignores it),
# ``error_id`` is ``err-…``, ``record_id`` is ``prv-…``,
# ``cache_key`` is ``backend:op:dataset:hash16``. None match pattern 7
# even before the safe-list. Keeping them was misleading dead weight.
#
# Other patterns (Bearer / Authorization / password=value) still apply
# under ``request_id`` — see ``_sanitise_str_safe_uuid`` below.
# ``request_id`` — primary CDS server-generated id (T-CDS-008). The
# allowlist is deliberately narrow: a key is universally safe only if
# user input can NEVER land under it at any structural depth. T-CDS-011
# Round-2 codex HIGH: an earlier addition of ``jobID`` / ``job_id``
# turned out to be an exfil hole because CDS ``inputs`` accepts
# arbitrary flat string keys, including ``jobID``. Server-generated
# jobIDs are preserved locally in ``CdsBackend._record_terminal``'s
# diagnostics path instead (the per-key restore loop right after the
# ``diagnostics = ... sanitise(deepcopy(remote_json))`` call).
_SAFE_UUID_KEYS: Final[frozenset[str]] = frozenset({"request_id"})

# UUID 8-4-4-4-12 (any version). Same shape as pattern 7 but as a full-string
# anchor so we can detect "the value is purely a UUID" rather than "contains
# a UUID inside a longer credential-bearing string". Use ``fullmatch`` at
# the call site (review codex CX-M2): ``re.match`` with ``$`` matches before
# a final ``\n`` in default mode, which would let ``"<uuid>\n"`` bypass
# pattern 7 — caught by codex on PR #63 review.
_UUID_FULL_RE: Final[re.Pattern[str]] = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def _sanitise_str_safe_uuid(value: str) -> str:
    """Like ``_sanitise_str`` but preserves bare UUIDs.

    A value qualifies for the bypass iff stripping ASCII whitespace and
    surrounding ASCII quotes leaves only the canonical UUID hyphen
    pattern (case-insensitive). The strip-then-match step makes the
    bypass robust against upstream deserialisation quirks (trailing
    newlines from line-buffered logs, surrounding quotes from naive
    JSON splicing, etc.).

    Codex Round 2 CX-R2-M1: the bypass returns the ORIGINAL ``value``,
    not the stripped form, so that the sanitised response stays
    byte-identical to what the backend persisted at ``record_workflow``
    time. Returning a stripped copy would de-sync the sanitised
    response from the persisted workflow row (the backend stores
    ``str(remote.request_id)`` raw at ``cds/backend.py:648``) and
    break ``poll``/``fetch``/``cancel`` lookups.

    Otherwise apply the full pattern set so ``"Bearer <uuid>"`` and
    ``"password=<uuid>"`` under a safe key still redact through the
    other patterns.
    """
    stripped = value.strip().strip("\"'")
    if _UUID_FULL_RE.fullmatch(stripped):
        return value
    return _sanitise_str(value)


class Sanitiser:
    """Recursive, depth-bounded, cycle-safe payload redactor."""

    SENSITIVE_PATTERNS: list[re.Pattern[str]] = [p for p, _ in _PATTERNS]

    def sanitise(self, payload: Any) -> Any:
        return self._walk(payload, depth=0, seen=set())

    def _walk_dict_value(
        self, key: Any, value: Any, *, depth: int, seen: set[int]
    ) -> Any:
        """Walk one dict entry, honouring the sensitive-key and safe-UUID-key
        rules. Sensitive key -> ``REDACTED`` regardless of value. Safe id key
        with a string value -> UUID-preserving sanitisation. Otherwise the
        usual recursive walk."""
        if isinstance(key, str) and _key_is_sensitive(key):
            return REDACTED
        if (
            isinstance(key, str)
            and key in _SAFE_UUID_KEYS
            and isinstance(value, str)
        ):
            return _sanitise_str_safe_uuid(value)
        return self._walk(value, depth=depth, seen=seen)

    def _walk(self, payload: Any, *, depth: int, seen: set[int]) -> Any:
        if depth > _MAX_DEPTH:
            return REDACTED
        if isinstance(payload, str):
            return _sanitise_str(payload)
        # Cycle / shared-container guard for mutable containers.
        if isinstance(payload, (dict, list, set, frozenset)):
            ident = id(payload)
            if ident in seen:
                return REDACTED
            seen = seen | {ident}
        if isinstance(payload, dict):
            # Keys are NOT rewritten in place: doing so would collapse
            # distinct keys (``password=foo`` and ``password=bar``) onto the
            # same redacted string and silently drop one value. Instead,
            # apply value-redaction whenever the key string CONTAINS any
            # credential-shaped segment (sensitive substring or ``key=value``
            # shape). The key text itself is left intact — callers that
            # JSON-encode the dict get a deterministic re-sanitisation pass
            # over the encoded string anyway.
            #
            # T-CDS-008: known server-generated id keys (``request_id``,
            # ``trace_id``, ...) preserve UUID-shape values; other patterns
            # still fire (defense-in-depth).
            return {
                k: self._walk_dict_value(k, v, depth=depth + 1, seen=seen)
                for k, v in payload.items()
            }
        if isinstance(payload, list):
            return [self._walk(v, depth=depth + 1, seen=seen) for v in payload]
        if isinstance(payload, tuple):
            return tuple(self._walk(v, depth=depth + 1, seen=seen) for v in payload)
        if isinstance(payload, (set, frozenset)):
            sanitised: Iterable[Any] = (
                self._walk(v, depth=depth + 1, seen=seen) for v in payload
            )
            return type(payload)(sanitised)
        # Non-container scalars (int, float, bool, None, Path, datetime,
        # bytes, Decimal, BaseModel) pass through unchanged. T-008 operates
        # on JSON-shaped primitives; callers dump Pydantic models first.
        return payload
