from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

REDACTED = "[REDACTED]"

# Each entry is (label, payload, must_not_contain) — must_not_contain is the
# secret that the sanitiser must scrub.
_PATTERN_CASES: list[tuple[str, str, str]] = [
    ("basic_auth_header", "Authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
    ("bare_basic_token", "Basic dXNlcjpwYXNzMTIz", "dXNlcjpwYXNzMTIz"),
    ("bearer_token", "Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig", "eyJhbGciOiJIUzI1NiJ9.payload.sig"),
    ("aws_access_key", "key=AKIAIOSFODNN7EXAMPLE rest", "AKIAIOSFODNN7EXAMPLE"),
    ("uuid_shape", "Token: 550e8400-e29b-41d4-a716-446655440000", "550e8400-e29b-41d4-a716-446655440000"),
    ("password_query", "https://example.com/api?password=hunter2&user=alice", "hunter2"),
    ("password_json", '{"password":"hunter2","user":"alice"}', "hunter2"),
    ("password_yaml", "password: secret-value", "secret-value"),
    ("private_token", "PRIVATE-TOKEN: abcdef0123456789", "abcdef0123456789"),
    ("client_secret_kv", 'client_secret="abcd1234efgh5678"', "abcd1234efgh5678"),
    ("client_secret_yaml", "client_secret: 'abcd1234efgh5678'", "abcd1234efgh5678"),
    ("aws_signature", "X-Amz-Signature=d41d8cd98f00b204e9800998ecf8427e", "d41d8cd98f00b204e9800998ecf8427e"),
    ("access_token", '{"access_token":"AAAA-BBBB-CCCC-DDDD"}', "AAAA-BBBB-CCCC-DDDD"),
    ("refresh_token", '{"refresh_token":"ref-1234567890"}', "ref-1234567890"),
    ("id_token", '{"id_token":"ID-9876543210"}', "ID-9876543210"),
    ("api_key", '{"api_key":"sk-live-abcdef0123456789"}', "sk-live-abcdef0123456789"),
    ("apikey_alias", '{"apikey":"alt-key-abcdef0123456789"}', "alt-key-abcdef0123456789"),
    ("env_var_password", "COPERNICUSMARINE_SERVICE_PASSWORD=hunter2", "hunter2"),
]


@pytest.mark.parametrize("label,payload,secret", _PATTERN_CASES)
def test_each_pattern_redacts(label: str, payload: str, secret: str) -> None:
    from copernicus_mcp.errors import Sanitiser

    out = Sanitiser().sanitise(payload)
    assert isinstance(out, str)
    assert secret not in out, f"{label}: secret leaked in {out!r}"
    assert REDACTED in out


def test_url_userinfo_preserves_host_and_path() -> None:
    from copernicus_mcp.errors import Sanitiser

    src = "https://alice:hunter2@cmems-du.eu/motu-web/Motu/path?x=1"
    out = Sanitiser().sanitise(src)
    assert "alice" not in out
    assert "hunter2" not in out
    assert "cmems-du.eu" in out
    assert "/motu-web/Motu/path" in out


def test_no_false_positive_on_username_only_dict() -> None:
    from copernicus_mcp.errors import Sanitiser

    payload = {
        "username": "alice@example.org",
        "dataset": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        "operation": "subset",
    }
    assert Sanitiser().sanitise(payload) == payload


def test_clean_fixture_unchanged() -> None:
    from copernicus_mcp.errors import Sanitiser

    fixture_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "clean_payload.json"
    )
    payload = json.loads(fixture_path.read_text())
    sanitised = Sanitiser().sanitise(payload)
    assert sanitised == payload, (
        "clean_payload fixture changed under sanitisation — false positive: "
        f"{json.dumps(sanitised, indent=2)}"
    )


def test_deeply_nested_redaction() -> None:
    from copernicus_mcp.errors import Sanitiser

    payload = {"a": {"b": {"c": {"d": "Bearer eyJsecretpayload12345"}}}}
    out = Sanitiser().sanitise(payload)
    assert "eyJsecretpayload12345" not in json.dumps(out)


def test_idempotence_on_complex_payload() -> None:
    from copernicus_mcp.errors import Sanitiser

    san = Sanitiser()
    payload = {
        "auth": "Basic dXNlcjpwYXNz",
        "list": ["Bearer abcd1234efgh", {"password": "x"}],
        "nested": {"again": "Bearer abcd1234efgh"},
    }
    once = san.sanitise(payload)
    twice = san.sanitise(once)
    assert once == twice


def test_non_string_scalars_pass_through() -> None:
    from copernicus_mcp.errors import Sanitiser

    san = Sanitiser()
    now = datetime.now(UTC)
    p = Path("/tmp/foo")
    payload = {
        "i": 42,
        "f": 3.14,
        "b": True,
        "n": None,
        "p": p,
        "t": now,
        "by": b"\x00\x01\x02",
    }
    out = san.sanitise(payload)
    assert out["i"] == 42
    assert out["f"] == 3.14
    assert out["b"] is True
    assert out["n"] is None
    assert out["p"] == p
    assert out["t"] == now
    assert out["by"] == b"\x00\x01\x02"


def test_tuple_and_set_shape_preserved() -> None:
    from copernicus_mcp.errors import Sanitiser

    out_tuple = Sanitiser().sanitise(("safe", "Bearer abcd1234efgh"))
    assert isinstance(out_tuple, tuple)
    assert "abcd1234efgh" not in str(out_tuple)

    out_set = Sanitiser().sanitise({"safe", "Bearer abcd1234efgh"})
    assert isinstance(out_set, set)
    joined = " ".join(out_set)
    assert "abcd1234efgh" not in joined


def test_redacted_token_does_not_get_redacted_again() -> None:
    """Sanitising a string already containing [REDACTED] is a no-op match-wise."""
    from copernicus_mcp.errors import Sanitiser

    src = f"prefix {REDACTED} suffix"
    assert Sanitiser().sanitise(src) == src


def test_input_is_not_mutated() -> None:
    from copernicus_mcp.errors import Sanitiser

    src: dict = {
        "k": "Bearer eyJsecretpayload12345",
        "nested": {"x": "AKIAIOSFODNN7EXAMPLE"},
        "list": ["Bearer xyz12345abcd"],
    }
    snapshot = json.loads(json.dumps(src))
    Sanitiser().sanitise(src)
    assert src == snapshot


def test_pydantic_model_passes_through() -> None:
    """Iter 1: Pydantic models pass through unchanged. Caller dumps then sanitises."""
    from copernicus_mcp.errors import Sanitiser, build_error_record

    rec = build_error_record("AuthError", message="x")
    out = Sanitiser().sanitise(rec)
    assert out is rec or out == rec


def test_cycle_detection_replaces_with_redacted() -> None:
    """Cyclic references must not crash; the sanitiser replaces the cycle
    branch with [REDACTED] rather than silently leaving it (codex T-008)."""
    from copernicus_mcp.errors import Sanitiser

    a: dict = {"name": "alice"}
    a["self"] = a  # cycle
    out = Sanitiser().sanitise(a)
    # No infinite recursion / RecursionError. Cycle node redacted.
    serialised = repr(out)
    assert "alice" in serialised  # leaf preserved
    assert REDACTED in serialised  # cycle replaced


def test_max_depth_caps_at_32() -> None:
    """Pathological deep nesting is capped — last branch becomes [REDACTED]."""
    from copernicus_mcp.errors import Sanitiser

    # Build a 50-level deep dict.
    payload: dict = {"v": "leaf"}
    for _ in range(50):
        payload = {"x": payload}
    out = Sanitiser().sanitise(payload)
    flat = json.dumps(out)
    # Sanitiser must not raise, and the deeply buried leaf is replaced by [REDACTED].
    assert REDACTED in flat


@pytest.mark.parametrize(
    "key,value",
    [
        ("password", "hunter2"),
        ("Password", "hunter2"),  # case-insensitive
        ("PASSWORD", "hunter2"),
        ("client_secret", "abcd1234"),
        ("access_token", "AAAA-BBBB"),
        ("refresh_token", "ref-12345"),
        ("id_token", "ID-789"),
        ("api_key", "sk-live-abc"),
        ("apikey", "alt-key-abc"),
        ("token", "raw-token-xyz"),
        ("secret", "shh"),
        ("secret_key", "shhh"),
        ("Authorization", "Basic xyz"),
        ("Cookie", "session=abc"),
        ("Set-Cookie", "session=abc; Path=/"),
        ("X-API-Key", "header-key-xyz"),
        ("Ocp-Apim-Subscription-Key", "azure-sub-key"),
        ("private_token", "cds-pat-uuid"),
        ("private-token", "cds-pat-uuid"),
        ("personal_access_token", "ghp_abc"),
        ("COPERNICUSMARINE_SERVICE_PASSWORD", "hunter2"),
    ],
)
def test_key_aware_dict_redaction(key: str, value: str) -> None:
    """Codex T-008 diff review: real dicts like {'password': 'hunter2'}
    must redact the VALUE based on the KEY, regardless of value content."""
    from copernicus_mcp.errors import Sanitiser

    out = Sanitiser().sanitise({key: value, "username": "alice"})
    assert out[key] == REDACTED
    assert out["username"] == "alice"  # neighboring key unchanged
    assert value not in json.dumps(out)


def test_key_aware_redaction_redacts_non_string_values() -> None:
    """Sensitive keys redact their value regardless of type."""
    from copernicus_mcp.errors import Sanitiser

    out = Sanitiser().sanitise(
        {
            "password": ["nested", "list"],
            "api_key": {"nested": "dict"},
            "client_secret": 12345,
            "username": "alice",
        }
    )
    assert out["password"] == REDACTED
    assert out["api_key"] == REDACTED
    assert out["client_secret"] == REDACTED
    assert out["username"] == "alice"


def test_idempotence_no_bracket_growth() -> None:
    """Codex T-008 diff review: pattern 9/10 used to grow `]` on each pass —
    `password=hunter2` → `password=[REDACTED]` → `password=[REDACTED]]`.
    The fix excludes `[` from the value char class so a sanitised value
    cannot re-trigger the pattern."""
    from copernicus_mcp.errors import Sanitiser

    san = Sanitiser()
    samples = [
        "password=hunter2",
        '{"password":"hunter2"}',
        "client_secret=abcdef1234",
        "access_token=AAAA-BBBB-CCCC",
        # codex-batch-3-pass4: authorization-header rule must be idempotent
        "Authorization: Bearer abc123def456",
        "authorization: Bearer xyz",
        "authorization=token123",
        "cookie=session=abc",
        '{"authorization": "Bearer xyz123abc", "user": "alice"}',
    ]
    for src in samples:
        once = san.sanitise(src)
        twice = san.sanitise(once)
        thrice = san.sanitise(twice)
        assert once == twice == thrice, f"non-idempotent on {src!r}: {once!r}"
        assert "]]" not in once, f"bracket grew on {src!r}"


def test_password_pattern_does_not_eat_neighboring_keys() -> None:
    """The password regex must be quote/boundary-bounded."""
    from copernicus_mcp.errors import Sanitiser

    src = '{"password":"hunter2","username":"alice","dataset":"GLOBAL_X"}'
    out = Sanitiser().sanitise(src)
    assert "hunter2" not in out
    assert "alice" in out
    assert "GLOBAL_X" in out


def test_extended_credential_keywords_redacted() -> None:
    """codex-batch-3 HIGH 1: token/secret/cookie/authorization were not redacted."""
    from copernicus_mcp.errors import Sanitiser

    s = Sanitiser()
    cases = [
        ("https://api?token=abc&foo=1", "abc"),
        ("secret_key=abcd1234", "abcd1234"),
        ("authorization: Bearer xyz", "xyz"),
        ("cookie=session=zzz", "zzz"),
        ("private_token=p123", "p123"),
        ("personal_access_token=pat_7", "pat_7"),
        ("set_cookie: id=abc", "abc"),
        ("secret=topsecret", "topsecret"),
    ]
    for src, leak in cases:
        out = s.sanitise(src)
        assert leak not in out, f"leak {leak!r} survived in {out!r} (from {src!r})"


def test_bearer_pattern_does_not_eat_english_words() -> None:
    """codex-batch-3-followup HIGH 1: pattern 4 must not redact 'authentication'."""
    from copernicus_mcp.errors import Sanitiser

    s = Sanitiser()
    assert s.sanitise("Use Bearer authentication") == "Use Bearer authentication"
    assert (
        s.sanitise("Bearer authentication is required")
        == "Bearer authentication is required"
    )
    # Real-shape token (8+ chars from token char-class) IS redacted.
    out = s.sanitise("Authorization: Bearer abc123def456")
    assert "abc123def456" not in out


def test_bearer_pattern_redacts_codex_high_examples() -> None:
    """codex-batch-T028-T029-followup HIGH: previously-leaking Bearer shapes."""
    from copernicus_mcp.errors import Sanitiser

    s = Sanitiser()
    # ``=``-terminated padded base64-ish — \\b after ``=`` previously failed.
    assert "AAAAAAA=" not in s.sanitise("Bearer AAAAAAA=")
    # ~-containing RFC 6750 b64token — ``~`` previously not in charset.
    assert "abc~def12" not in s.sanitise("Bearer abc~def12")
    # Long enough alpha-only (20+) catches real opaque tokens.
    assert "abcdefghijklmnopqrstuv" not in s.sanitise(
        "Bearer abcdefghijklmnopqrstuv"
    )
    # Pure-alpha < 20 chars deliberately NOT redacted: such tokens do not
    # exist in real auth systems and the false-positive cost on prose
    # ("Bearer characterization" — 16 chars) is higher.
    assert s.sanitise("Bearer abcdefgh") == "Bearer abcdefgh"
    assert (
        s.sanitise("Bearer characterization is")
        == "Bearer characterization is"
    )


@pytest.mark.parametrize(
    "terminator",
    [":", ")", "?", "#", ">", "|", "!", ",", ";", " ", "."],
)
def test_bearer_pattern_terminator_invariance(terminator: str) -> None:
    """codex-round2 HIGH: Bearer redaction must terminate on ANY non-token char."""
    from copernicus_mcp.errors import Sanitiser

    s = Sanitiser()
    out = s.sanitise(f"Bearer abc12345-xyz{terminator} retry")
    assert "abc12345-xyz" not in out, f"leak with terminator {terminator!r}"


def test_password_with_slash_or_question_mark_fully_redacted() -> None:
    """codex-round2 HIGH: passwords legitimately contain ``/`` and ``?``.

    Pattern 9 must NOT terminate at ``/`` or ``?`` for ``KEY=VALUE`` shapes —
    real passwords contain those. Diagnostic value on path-shaped strings
    (``/data/token=abc/file.nc``) is sacrificed: the ``/file.nc`` tail is
    folded into the redacted value. Acceptable trade-off.
    """
    from copernicus_mcp.errors import Sanitiser

    s = Sanitiser()
    assert "abc/def" not in s.sanitise("password=abc/def ghi")
    assert "abc?def" not in s.sanitise("access_token=abc?def&user=alice")
    assert "abc+ghi" not in s.sanitise(
        "https://api.example/data?access_token=abc/def+ghi&user=alice"
    )


def test_segment_sensitive_key_triggers_value_redaction() -> None:
    """codex-batch-T028-T029-followup MEDIUM + round-3 HIGH: whole-segment match.

    A key like ``"backend-password"`` (not literally in _SENSITIVE_KEYS) must
    still trigger value redaction because ``password`` is a whole segment.
    Keys themselves are NOT rewritten — rewriting would collapse distinct
    keys onto a single redacted string. Backend ids are validated upstream
    by BackendRegistry.register.
    """
    from copernicus_mcp.errors import Sanitiser

    s = Sanitiser()
    out = s.sanitise({"backend-password": "hunter2", "x-api-key": "abc"})
    assert out == {
        "backend-password": "[REDACTED]",
        "x-api-key": "[REDACTED]",
    }


@pytest.mark.parametrize(
    "key",
    [
        "next_token",
        "page_token",
        "token_count",
        "tokenize",
        "tokenizer",
        "tokens_used",
        "cookie_jar",
        "cookie_consent",
        "secretary",
        "secrets_manager_arn",
        "authentication_method",
    ],
)
def test_benign_keys_with_secret_substring_preserved(key: str) -> None:
    """codex-round3 HIGH: substring-match was over-redacting benign keys.

    Whole-segment match preserves these because ``token``/``cookie`` are not
    in the strict segment set, and ``secretary``/``secrets_manager_arn``
    don't have ``secret`` as a whole segment (it's a substring of ``secrets``
    which is itself the segment).
    """
    from copernicus_mcp.errors import Sanitiser

    s = Sanitiser()
    out = s.sanitise({key: "non-secret-value"})
    assert out == {key: "non-secret-value"}, f"benign key {key!r} over-redacted"


def test_dict_key_collision_does_not_drop_values() -> None:
    """codex-round2 HIGH: keys are NOT rewritten, so distinct keys survive."""
    from copernicus_mcp.errors import Sanitiser

    s = Sanitiser()
    out = s.sanitise({"foo": "Bearer abc12345-xyz", "bar": "Bearer def67890-uvw"})
    # Both keys preserved.
    assert set(out.keys()) == {"foo", "bar"}
    # Both values redacted.
    assert all("[REDACTED]" in v for v in out.values())


# ---------------------------------------------------------------------------
# T-CDS-008 / F-1 regression: safe UUID-shape values in known server-generated
# dict keys (request_id, trace_id, ...) must survive sanitisation. Without
# this, every CDS submit response routed through the orchestrator gets its
# request_id redacted to "[REDACTED]", breaking poll/cancel/download. Caught
# by integration smoke against the real CDS API; unit-test mocks used short
# non-UUID ids and missed it.
# ---------------------------------------------------------------------------


def test_safe_request_id_uuid_preserved() -> None:
    """CDS returns ``request_id`` as a canonical UUID; pattern 7's
    blanket UUID redaction would break poll/cancel/download. PATs are
    also UUID-shape but never reach response payloads (invariant #2 —
    they live in ``AuthAdapter``).

    Review round 2 (cr M1, codex L1): allowlist trimmed to only the
    keys that genuinely emit UUIDs. Other id keys (``trace_id`` is
    ``uuid4().hex`` without dashes, ``error_id``/``record_id`` use
    prefixed ``err-``/``prv-`` shapes, ``cache_key`` is
    ``backend:op:dataset:hash16``) never matched pattern 7 in the
    first place — keeping them in the allowlist was misleading.
    """
    from copernicus_mcp.errors import Sanitiser

    uuid_value = "cb290d05-40ea-42ad-90da-69ea09f2e1a7"
    s = Sanitiser()
    out = s.sanitise({"request_id": uuid_value})
    assert out == {"request_id": uuid_value}, out


@pytest.mark.parametrize(
    "key",
    ["trace_id", "error_id", "record_id", "cache_key"],
)
def test_non_uuid_id_keys_not_in_safe_list(key: str) -> None:
    """Keys whose values are NOT UUID-shape don't need the safe-list
    bypass. This test pins that contract — if any of these keys ever
    starts emitting UUIDs, this test fails and forces a deliberate
    decision (add to allowlist with regression coverage).
    """
    from copernicus_mcp.errors import Sanitiser

    uuid_value = "cb290d05-40ea-42ad-90da-69ea09f2e1a7"
    s = Sanitiser()
    out = s.sanitise({key: uuid_value})
    # Pattern 7 redacts UUIDs in *any* string context, so an unsafe
    # key wraps to ``[REDACTED]``.
    assert out[key] == "[REDACTED]", out


def test_safe_key_does_not_disable_other_redaction() -> None:
    """A safe UUID key bypasses only UUID redaction — other patterns
    (password / authorization / bearer / etc.) still apply if the value
    matches them. Defense-in-depth: misuse of the safe-list shouldn't
    leak credentials a careless caller stuck under a safe key."""
    from copernicus_mcp.errors import Sanitiser

    s = Sanitiser()
    out = s.sanitise({"request_id": "Bearer abc12345xyz67890"})
    assert "[REDACTED]" in out["request_id"], out


def test_safe_key_nested_in_envelope() -> None:
    """The CDS-orchestrator path passes
    ``{"result": {"request_id": "<uuid>", ...}}`` through the sanitiser.
    Walking into nested dicts must preserve the safe key at any depth."""
    from copernicus_mcp.errors import Sanitiser

    uuid_value = "cb290d05-40ea-42ad-90da-69ea09f2e1a7"
    s = Sanitiser()
    out = s.sanitise(
        {"result": {"request_id": uuid_value, "status": "queued"}}
    )
    assert out["result"]["request_id"] == uuid_value, out


# T-CDS-008 review round 2 — codex CX-M2 / cr L2:
# ``_UUID_FULL_RE.match`` with ``^...$`` is not a strict whole-string check;
# ``$`` matches before a final ``\n``. UUID + trailing newline would slip
# through. Use ``fullmatch`` and strip the value before the safe-list check
# so trailing whitespace, JSON-quote wrapping, and uppercase variants all
# survive the bypass.


@pytest.mark.parametrize(
    "value",
    [
        "cb290d05-40ea-42ad-90da-69ea09f2e1a7\n",  # trailing newline
        "cb290d05-40ea-42ad-90da-69ea09f2e1a7 ",  # trailing space
        " cb290d05-40ea-42ad-90da-69ea09f2e1a7",  # leading space
        "CB290D05-40EA-42AD-90DA-69EA09F2E1A7",  # uppercase
    ],
)
def test_safe_key_uuid_preserved_verbatim_under_wrapping(value: str) -> None:
    """Whitespace-wrapped or uppercase UUIDs under safe keys must
    survive sanitisation VERBATIM — same bytes in, same bytes out.

    Review codex Round 2 CX-R2-M1: returning a *stripped* value would
    de-sync the sanitised response from the persisted workflow row
    (which stores the raw ``str(remote.request_id)`` at
    ``cds/backend.py:648``). A subsequent ``poll``/``fetch``/``cancel``
    lookup with the stripped value would 404. Bypass must canonicalise
    only inside the match check, not in the return path.
    """
    from copernicus_mcp.errors import Sanitiser

    s = Sanitiser()
    out = s.sanitise({"request_id": value})
    assert out["request_id"] == value, out


def test_sensitive_key_with_uuid_value_still_redacted() -> None:
    """Review CX-L3: precedence pin — sensitive key check fires BEFORE
    safe-uuid check. ``password=<uuid>`` must redact even though the
    value happens to be UUID-shape. A future reordering of
    ``_walk_dict_value`` would silently leak PATs without this test."""
    from copernicus_mcp.errors import Sanitiser

    s = Sanitiser()
    uuid_value = "cb290d05-40ea-42ad-90da-69ea09f2e1a7"
    out = s.sanitise({"password": uuid_value})
    assert out["password"] == "[REDACTED]", out
    # Same for nested.
    out = s.sanitise({"context": {"access_token": uuid_value}})
    assert out["context"]["access_token"] == "[REDACTED]", out


@pytest.mark.parametrize(
    "secret,expected_marker",
    [
        # All 9 non-UUID patterns. The 10th (sensitive JSON-KV) is
        # already covered by sensitive-key redaction in test above.
        ("Authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
        ("Basic dXNlcjpwYXNzMTIz", "dXNlcjpwYXNzMTIz"),
        ("Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig", "eyJhbGciOiJIUzI1NiJ9"),
        ("AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
        ("https://user:pw@host/x", "user:pw"),
        ("authorization: foo123bar456", "foo123bar456"),
        ("PRIVATE-TOKEN: glpat-abcdefghij1234", "glpat-abcdefghij1234"),
        (
            "https://x/?X-Amz-Signature=abc123def456ghi789jklmnopqrst",
            "abc123def456ghi789jklmnopqrst",
        ),
        ("?password=hunter2&user=alice", "hunter2"),
    ],
)
def test_safe_key_does_not_disable_pattern(
    secret: str, expected_marker: str
) -> None:
    """Review CX-L3 / cr L3: parametrised version of the
    defense-in-depth claim. Each of the nine non-UUID patterns must
    still fire when the value sits under a safe-uuid key."""
    from copernicus_mcp.errors import Sanitiser

    s = Sanitiser()
    out = s.sanitise({"request_id": secret})
    assert expected_marker not in out["request_id"], (
        f"pattern leak under safe key: {secret!r} -> {out['request_id']!r}"
    )
