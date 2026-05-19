"""``CdsApiKeyAdapter`` tests (T-CDS-001 completion).

the CDS API expects the
PAT in a ``PRIVATE-TOKEN: <UUID>`` header — NOT the standard
``Authorization: Bearer`` form.

Defence-in-depth: the PAT must never appear in ``__repr__``,
``credentials_summary()``, or any logging surface.
"""

from __future__ import annotations

import httpx
import pytest

_CDS_TOKEN = "abcdef01-2345-6789-abcd-ef0123456789"


def _resolved_creds(
    *, key: str = _CDS_TOKEN, url: str | None = None
):
    from copernicus_mcp.auth.resolver import ResolvedCredentials

    fields: dict[str, str] = {"key": key}
    if url is not None:
        fields["url"] = url
    return ResolvedCredentials(
        backend="cds",
        source="explicit",
        source_detail="test",
        fields=fields,
    )


def test_constructs_with_key_only() -> None:
    from copernicus_mcp.auth.cds import CdsApiKeyAdapter

    adapter = CdsApiKeyAdapter(_resolved_creds())
    key, url = adapter.get_pat()
    assert key == _CDS_TOKEN
    assert url is None  # falls back to cdsapi default


def test_constructs_with_key_and_url() -> None:
    from copernicus_mcp.auth.cds import CdsApiKeyAdapter

    adapter = CdsApiKeyAdapter(
        _resolved_creds(url="https://ads.atmosphere.copernicus.eu/api")
    )
    key, url = adapter.get_pat()
    assert key == _CDS_TOKEN
    assert url == "https://ads.atmosphere.copernicus.eu/api"


def test_rejects_wrong_backend() -> None:
    from copernicus_mcp.auth.cds import CdsApiKeyAdapter
    from copernicus_mcp.auth.resolver import ResolvedCredentials

    bad = ResolvedCredentials(
        backend="cmems",
        source="explicit",
        source_detail="test",
        fields={"key": _CDS_TOKEN},
    )
    with pytest.raises(ValueError):
        CdsApiKeyAdapter(bad)


def test_rejects_missing_source() -> None:
    from copernicus_mcp.auth.cds import CdsApiKeyAdapter
    from copernicus_mcp.auth.resolver import ResolvedCredentials

    missing = ResolvedCredentials(
        backend="cds",
        source="missing",
        source_detail=None,
        fields={"key": _CDS_TOKEN},
    )
    with pytest.raises(ValueError):
        CdsApiKeyAdapter(missing)


def test_rejects_empty_key() -> None:
    from copernicus_mcp.auth.cds import CdsApiKeyAdapter

    with pytest.raises(ValueError):
        CdsApiKeyAdapter(_resolved_creds(key=""))


@pytest.mark.asyncio
async def test_apply_credentials_sets_private_token_header() -> None:
    """Per research §6.8.1 CDS expects ``PRIVATE-TOKEN: <UUID>`` — distinct
    from the standard Bearer form. Verify the adapter sets exactly that
    header and nothing else credential-shaped."""
    from copernicus_mcp.auth.cds import CdsApiKeyAdapter

    adapter = CdsApiKeyAdapter(_resolved_creds())
    request = httpx.Request(
        "GET", "https://cds.climate.copernicus.eu/api/some/path"
    )
    out = await adapter.apply_credentials(request)
    assert out.headers["PRIVATE-TOKEN"] == _CDS_TOKEN
    # No Bearer / Authorization header set.
    assert "Authorization" not in out.headers


@pytest.mark.asyncio
async def test_apply_credentials_returns_same_request_object() -> None:
    """Mutating in place rather than reconstructing keeps middleware that
    has cached references to the request working."""
    from copernicus_mcp.auth.cds import CdsApiKeyAdapter

    adapter = CdsApiKeyAdapter(_resolved_creds())
    request = httpx.Request("GET", "https://cds.climate.copernicus.eu/api")
    out = await adapter.apply_credentials(request)
    assert out is request


@pytest.mark.asyncio
async def test_apply_credentials_does_not_clobber_other_headers() -> None:
    from copernicus_mcp.auth.cds import CdsApiKeyAdapter

    adapter = CdsApiKeyAdapter(_resolved_creds())
    request = httpx.Request(
        "GET",
        "https://cds.climate.copernicus.eu/api",
        headers={"User-Agent": "copernicus-mcp/test", "Accept": "application/json"},
    )
    out = await adapter.apply_credentials(request)
    assert out.headers["User-Agent"] == "copernicus-mcp/test"
    assert out.headers["Accept"] == "application/json"
    assert out.headers["PRIVATE-TOKEN"] == _CDS_TOKEN


@pytest.mark.asyncio
async def test_apply_credentials_overwrites_stale_private_token() -> None:
    """Code-reviewer LOW: the adapter is the authority on the credential
    header. If a request already carries a stale ``PRIVATE-TOKEN`` (from
    a caller-set value or a retried request), ``apply_credentials`` must
    overwrite it. ``request.headers[k] = v`` does this; a regression to
    ``setdefault`` would silently keep stale credentials."""
    from copernicus_mcp.auth.cds import CdsApiKeyAdapter

    adapter = CdsApiKeyAdapter(_resolved_creds())
    request = httpx.Request(
        "GET",
        "https://cds.climate.copernicus.eu/api",
        headers={"PRIVATE-TOKEN": "stale-cached-pat-from-elsewhere"},
    )
    out = await adapter.apply_credentials(request)
    assert out.headers["PRIVATE-TOKEN"] == _CDS_TOKEN
    assert "stale-cached-pat-from-elsewhere" not in out.headers["PRIVATE-TOKEN"]


@pytest.mark.asyncio
async def test_handle_unauthorized_returns_false() -> None:
    """PATs are static — no refresh path."""
    from copernicus_mcp.auth.cds import CdsApiKeyAdapter

    adapter = CdsApiKeyAdapter(_resolved_creds())
    response = httpx.Response(status_code=401)
    assert await adapter.handle_unauthorized(response) is False


def test_repr_does_not_leak_pat() -> None:
    from copernicus_mcp.auth.cds import CdsApiKeyAdapter

    adapter = CdsApiKeyAdapter(_resolved_creds(key="TOPSECRET-CDS-PAT"))
    text = repr(adapter)
    assert "TOPSECRET-CDS-PAT" not in text
    assert "supports_refresh" in text


def test_credentials_summary_does_not_leak_pat() -> None:
    from copernicus_mcp.auth.cds import CdsApiKeyAdapter

    adapter = CdsApiKeyAdapter(_resolved_creds(key="TOPSECRET-CDS-PAT"))
    summary = adapter.credentials_summary()
    assert summary["key"] == "<set>"
    for v in summary.values():
        assert "TOPSECRET-CDS-PAT" not in v


@pytest.mark.asyncio
async def test_close_is_idempotent() -> None:
    from copernicus_mcp.auth.cds import CdsApiKeyAdapter

    adapter = CdsApiKeyAdapter(_resolved_creds())
    await adapter.close()
    await adapter.close()  # no error on second close


def test_supports_refresh_is_false() -> None:
    from copernicus_mcp.auth.cds import CdsApiKeyAdapter

    assert CdsApiKeyAdapter.supports_refresh is False
