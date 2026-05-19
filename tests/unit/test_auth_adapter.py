from __future__ import annotations

import logging
from types import MappingProxyType

import httpx
import pytest


def _make_creds(username: str = "u", password: str = "p", source: str = "explicit"):
    from copernicus_mcp.auth import ResolvedCredentials

    return ResolvedCredentials(
        backend="cmems",
        fields={"username": username, "password": password},
        source=source,  # type: ignore[arg-type]
        source_detail=None,
    )


def test_class_attributes() -> None:
    from copernicus_mcp.auth import CmemsBasicAuthAdapter

    adapter = CmemsBasicAuthAdapter(_make_creds())
    assert adapter.backend_id == "cmems"
    assert adapter.supports_refresh is False


@pytest.mark.asyncio
async def test_apply_credentials_returns_request_unchanged() -> None:
    from copernicus_mcp.auth import CmemsBasicAuthAdapter

    adapter = CmemsBasicAuthAdapter(_make_creds())
    req = httpx.Request("GET", "https://example.com/path")
    out = await adapter.apply_credentials(req)
    assert out is req
    assert "Authorization" not in out.headers


@pytest.mark.asyncio
async def test_handle_unauthorized_returns_false() -> None:
    from copernicus_mcp.auth import CmemsBasicAuthAdapter

    adapter = CmemsBasicAuthAdapter(_make_creds())
    resp = httpx.Response(401, request=httpx.Request("GET", "https://example.com"))
    assert await adapter.handle_unauthorized(resp) is False


def test_credentials_summary_is_redacted_and_immutable() -> None:
    from copernicus_mcp.auth import CmemsBasicAuthAdapter

    adapter = CmemsBasicAuthAdapter(_make_creds("U", "P"))
    summary = adapter.credentials_summary()
    assert dict(summary) == {"username": "<set>", "password": "<set>"}
    assert isinstance(summary, MappingProxyType)
    with pytest.raises(TypeError):
        summary["username"] = "evil"  # type: ignore[index]
    assert "U" not in repr(summary)
    assert "P" not in repr(summary)


def test_get_username_password_returns_raw_tuple() -> None:
    from copernicus_mcp.auth import CmemsBasicAuthAdapter

    adapter = CmemsBasicAuthAdapter(_make_creds("user-1", "pw-1"))
    assert adapter.get_username_password() == ("user-1", "pw-1")


def test_repr_does_not_leak_credentials() -> None:
    from copernicus_mcp.auth import CmemsBasicAuthAdapter

    adapter = CmemsBasicAuthAdapter(_make_creds("TOPSECRET-USER", "TOPSECRET-PASS"))
    assert "TOPSECRET-USER" not in repr(adapter)
    assert "TOPSECRET-PASS" not in repr(adapter)
    assert "TOPSECRET-USER" not in str(adapter)
    assert "TOPSECRET-PASS" not in str(adapter)


def test_vars_do_not_leak_credentials_via_attribute_dump() -> None:
    """If anyone vars()s the adapter, raw values should not be obvious in attr names."""
    from copernicus_mcp.auth import CmemsBasicAuthAdapter

    adapter = CmemsBasicAuthAdapter(_make_creds("TOPSECRET-USER", "TOPSECRET-PASS"))
    # Values will be there if you reach for them — invariant is about defaults,
    # logs, and repr, not about Python introspection. This test asserts the
    # attribute names themselves are not credential values, which would be a
    # very dumb mistake.
    for k in vars(adapter).keys():
        assert "TOPSECRET" not in k


def test_constructing_with_missing_credentials_raises() -> None:
    from copernicus_mcp.auth import CmemsBasicAuthAdapter, ResolvedCredentials

    missing = ResolvedCredentials(
        backend="cmems",
        fields={},
        source="missing",
        source_detail=None,
    )
    with pytest.raises(ValueError) as exc:
        CmemsBasicAuthAdapter(missing)
    # Error message must not regurgitate any credential placeholder values either
    # (defense-in-depth — the error message is allowed to mention 'username',
    # 'password', 'cmems' but never raw values).
    assert "credential" in str(exc.value).lower()


def test_constructing_with_wrong_backend_raises() -> None:
    from copernicus_mcp.auth import CmemsBasicAuthAdapter, ResolvedCredentials

    cdse = ResolvedCredentials(
        backend="cdse",
        fields={"username": "u", "password": "p"},
        source="explicit",
        source_detail=None,
    )
    with pytest.raises(ValueError):
        CmemsBasicAuthAdapter(cdse)


def test_wrong_backend_error_does_not_echo_backend_or_fields() -> None:
    from copernicus_mcp.auth import CmemsBasicAuthAdapter, ResolvedCredentials

    suspicious = ResolvedCredentials(
        backend="TOPSECRET-BACKEND",
        fields={"username": "TOPSECRET-USER", "password": "TOPSECRET-PASS"},
        source="explicit",
        source_detail=None,
    )
    try:
        CmemsBasicAuthAdapter(suspicious)
    except ValueError as exc:
        full = repr(exc) + " " + str(exc)
        assert "TOPSECRET-BACKEND" not in full
        assert "TOPSECRET-USER" not in full
        assert "TOPSECRET-PASS" not in full


def test_constructing_with_empty_username_raises() -> None:
    from copernicus_mcp.auth import CmemsBasicAuthAdapter

    with pytest.raises(ValueError):
        CmemsBasicAuthAdapter(_make_creds(username=""))


def test_constructing_with_empty_password_raises() -> None:
    from copernicus_mcp.auth import CmemsBasicAuthAdapter

    with pytest.raises(ValueError):
        CmemsBasicAuthAdapter(_make_creds(password=""))


def test_value_error_does_not_contain_credential_values() -> None:
    from copernicus_mcp.auth import CmemsBasicAuthAdapter

    try:
        CmemsBasicAuthAdapter(_make_creds(username="", password="TOPSECRET-PW"))
    except ValueError as exc:
        assert "TOPSECRET-PW" not in repr(exc)
        assert "TOPSECRET-PW" not in str(exc)
        assert "TOPSECRET-PW" not in "".join(map(str, exc.args))


def test_no_log_record_contains_credential_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from copernicus_mcp.auth import CmemsBasicAuthAdapter

    with caplog.at_level(logging.DEBUG, logger="copernicus_mcp.auth"):
        adapter = CmemsBasicAuthAdapter(
            _make_creds("TOPSECRET-USER", "TOPSECRET-PASS")
        )
        adapter.credentials_summary()
        adapter.get_username_password()
    for rec in caplog.records:
        full = rec.getMessage() + " " + " ".join(
            f"{k}={v}" for k, v in rec.__dict__.items()
        )
        assert "TOPSECRET-USER" not in full
        assert "TOPSECRET-PASS" not in full


@pytest.mark.asyncio
async def test_close_is_idempotent() -> None:
    from copernicus_mcp.auth import CmemsBasicAuthAdapter

    adapter = CmemsBasicAuthAdapter(_make_creds())
    await adapter.close()
    await adapter.close()  # second call must not raise


def test_protocol_runtime_check_optional() -> None:
    """The AuthAdapter protocol is the static-typing contract.

    This test only exists to keep the module importable: classes implementing
    the protocol structurally do not need to inherit from it.
    """
    from copernicus_mcp.auth import AuthAdapter, CmemsBasicAuthAdapter

    adapter = CmemsBasicAuthAdapter(_make_creds())
    # Sanity: required protocol attributes exist.
    for attr in ("backend_id", "supports_refresh", "apply_credentials",
                 "handle_unauthorized", "close", "credentials_summary"):
        assert hasattr(adapter, attr)
    # No isinstance check — Protocol w/o @runtime_checkable.
    assert AuthAdapter is not None
