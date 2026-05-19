from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import pytest


def _clear_cmems_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_USERNAME", raising=False)
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_PASSWORD", raising=False)


def test_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cmems_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))  # no creds file

    resolved = CredentialResolver().resolve(
        "cmems", override={"username": "u1", "password": "p1"}
    )
    assert resolved is not None
    assert resolved.source == "explicit"
    assert resolved.fields["username"] == "u1"
    assert resolved.fields["password"] == "p1"


def test_secret_manager_short_circuits_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "from-env")
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", "env-pw")
    monkeypatch.setenv("HOME", str(tmp_path))

    class FakeSM:
        def fetch(self, backend: str) -> Mapping[str, str] | None:
            assert backend == "cmems"
            return {"username": "sm-user", "password": "sm-pw"}

    resolved = CredentialResolver(secret_manager_provider=FakeSM()).resolve("cmems")
    assert resolved is not None
    assert resolved.source == "secret_manager"
    assert resolved.fields["username"] == "sm-user"


def test_secret_manager_partial_result_falls_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "from-env")
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", "env-pw")
    monkeypatch.setenv("HOME", str(tmp_path))

    class PartialSM:
        def fetch(self, backend: str) -> Mapping[str, str] | None:
            return {"username": "only-user"}  # missing password

    resolved = CredentialResolver(secret_manager_provider=PartialSM()).resolve("cmems")
    assert resolved is not None
    assert resolved.source == "env"


def test_env_resolves(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from copernicus_mcp.auth import CredentialResolver

    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "env-user")
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", "env-pw")
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = CredentialResolver().resolve("cmems")
    assert resolved is not None
    assert resolved.source == "env"
    assert resolved.fields["username"] == "env-user"


def test_partial_env_falls_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "only-user")
    monkeypatch.delenv("COPERNICUSMARINE_SERVICE_PASSWORD", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no creds file

    assert CredentialResolver().resolve("cmems") is None


def test_empty_env_string_falls_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "")
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", "")
    monkeypatch.setenv("HOME", str(tmp_path))  # no creds file

    assert CredentialResolver().resolve("cmems") is None


def test_config_file_resolves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cmems_env(monkeypatch)
    creds_dir = tmp_path / ".copernicusmarine"
    creds_dir.mkdir()
    (creds_dir / ".copernicusmarine-credentials").write_text(
        "# CMEMS credentials\n"
        "\n"
        "username=file-user\n"
        "password=file-pw\n"
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = CredentialResolver().resolve("cmems")
    assert resolved is not None
    assert resolved.source == "config_file"
    assert resolved.fields == {"username": "file-user", "password": "file-pw"}
    assert resolved.source_detail == "~/.copernicusmarine/.copernicusmarine-credentials"


def test_config_file_with_utf8_bom(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cmems_env(monkeypatch)
    creds_dir = tmp_path / ".copernicusmarine"
    creds_dir.mkdir()
    (creds_dir / ".copernicusmarine-credentials").write_bytes(
        "﻿username=bom-user\npassword=bom-pw\n".encode()
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = CredentialResolver().resolve("cmems")
    assert resolved is not None
    assert resolved.source == "config_file"
    assert resolved.fields["username"] == "bom-user"


def test_config_file_malformed_lines_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cmems_env(monkeypatch)
    creds_dir = tmp_path / ".copernicusmarine"
    creds_dir.mkdir()
    (creds_dir / ".copernicusmarine-credentials").write_text(
        "no_equals_here\n"
        "username=u\n"
        "  \n"
        "password=p\n"
        "= empty key\n"
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = CredentialResolver().resolve("cmems")
    assert resolved is not None
    assert resolved.fields == {"username": "u", "password": "p"}


def test_config_file_base64_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """copernicusmarine >= 2.0 writes the credentials file as a single
    base64-encoded line wrapping ``[credentials]\\nusername=...\\npassword=...``.
    Resolver must decode and parse it."""
    import base64

    from copernicus_mcp.auth import CredentialResolver

    _clear_cmems_env(monkeypatch)
    inner = "[credentials]\nusername=b64-user\npassword=b64-pw\n"
    encoded = base64.b64encode(inner.encode()).decode()
    creds_dir = tmp_path / ".copernicusmarine"
    creds_dir.mkdir()
    (creds_dir / ".copernicusmarine-credentials").write_text(encoded)
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = CredentialResolver().resolve("cmems")
    assert resolved is not None
    assert resolved.source == "config_file"
    assert resolved.fields == {"username": "b64-user", "password": "b64-pw"}


def test_config_file_legacy_plain_format_still_works(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The base64 detection must not break the legacy plain-INI format."""
    from copernicus_mcp.auth import CredentialResolver

    _clear_cmems_env(monkeypatch)
    creds_dir = tmp_path / ".copernicusmarine"
    creds_dir.mkdir()
    (creds_dir / ".copernicusmarine-credentials").write_text(
        "[credentials]\nusername=plain-user\npassword=plain-pw\n"
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = CredentialResolver().resolve("cmems")
    assert resolved is not None
    assert resolved.fields == {"username": "plain-user", "password": "plain-pw"}


def test_missing_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cmems_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert CredentialResolver().resolve("cmems") is None


def test_unknown_backend_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from copernicus_mcp.auth import CredentialResolver

    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "x")
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", "y")
    assert CredentialResolver().resolve("cdse") is None


def test_fields_mapping_is_readonly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cmems_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = CredentialResolver().resolve(
        "cmems", override={"username": "u", "password": "p"}
    )
    assert resolved is not None
    assert isinstance(resolved.fields, MappingProxyType)
    with pytest.raises(TypeError):
        resolved.fields["username"] = "evil"  # type: ignore[index]


def test_override_is_defensively_copied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cmems_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    src = {"username": "u", "password": "p"}
    resolved = CredentialResolver().resolve("cmems", override=src)
    assert resolved is not None
    src["username"] = "mutated"
    assert resolved.fields["username"] == "u"


def test_repr_does_not_leak_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cmems_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = CredentialResolver().resolve(
        "cmems",
        override={"username": "TOPSECRET-USER", "password": "TOPSECRET-PASS"},
    )
    assert resolved is not None
    text = repr(resolved)
    assert "TOPSECRET-USER" not in text
    assert "TOPSECRET-PASS" not in text


def test_no_credential_value_in_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "TOPSECRET-USER")
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", "TOPSECRET-PASS")
    monkeypatch.setenv("HOME", str(tmp_path))

    with caplog.at_level(logging.DEBUG, logger="copernicus_mcp.auth.resolver"):
        resolved = CredentialResolver().resolve("cmems")
    assert resolved is not None
    for rec in caplog.records:
        full = rec.getMessage() + " " + " ".join(
            f"{k}={v}" for k, v in rec.__dict__.items()
        )
        assert "TOPSECRET-USER" not in full
        assert "TOPSECRET-PASS" not in full


def test_config_file_oserror_treated_as_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from copernicus_mcp.auth import CredentialResolver
    from copernicus_mcp.auth import resolver as resolver_mod

    _clear_cmems_env(monkeypatch)
    creds_dir = tmp_path / ".copernicusmarine"
    creds_dir.mkdir()
    (creds_dir / ".copernicusmarine-credentials").write_text("username=u\npassword=p\n")
    monkeypatch.setenv("HOME", str(tmp_path))

    def _explode(self: Path, *args, **kwargs) -> str:
        raise PermissionError("denied")

    monkeypatch.setattr(resolver_mod.Path, "read_text", _explode)
    with caplog.at_level(logging.DEBUG, logger="copernicus_mcp.auth.resolver"):
        result = CredentialResolver().resolve("cmems")
    assert result is None


def test_extra_keys_are_stripped_from_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cmems_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = CredentialResolver().resolve(
        "cmems",
        override={
            "username": "u",
            "password": "p",
            "token": "SHOULD-NOT-APPEAR",
            "extra": "x",
        },
    )
    assert resolved is not None
    assert dict(resolved.fields) == {"username": "u", "password": "p"}


def test_extra_keys_are_stripped_from_secret_manager(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cmems_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    class ExtraSM:
        def fetch(self, backend: str) -> Mapping[str, str] | None:
            return {"username": "u", "password": "p", "api_key": "SHOULD-NOT-APPEAR"}

    resolved = CredentialResolver(secret_manager_provider=ExtraSM()).resolve("cmems")
    assert resolved is not None
    assert dict(resolved.fields) == {"username": "u", "password": "p"}


def test_secret_manager_dict_is_defensively_copied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cmems_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    src: dict[str, str] = {"username": "u", "password": "p"}

    class MutableSM:
        def fetch(self, backend: str) -> Mapping[str, str] | None:
            return src

    resolved = CredentialResolver(secret_manager_provider=MutableSM()).resolve("cmems")
    assert resolved is not None
    src["password"] = "mutated"
    assert resolved.fields["password"] == "p"


def test_secret_manager_exception_falls_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "env-user")
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", "env-pw")
    monkeypatch.setenv("HOME", str(tmp_path))

    class BoomSM:
        def fetch(self, backend: str) -> Mapping[str, str] | None:
            raise RuntimeError("provider exploded with TOPSECRET-VALUE inside")

    resolved = CredentialResolver(secret_manager_provider=BoomSM()).resolve("cmems")
    assert resolved is not None
    assert resolved.source == "env"


# ---------------------------------------------------------------------------
# T-CDS-001 completion: CDS credential resolution
# ---------------------------------------------------------------------------


def _clear_cds_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    monkeypatch.delenv("CDSAPI_URL", raising=False)
    monkeypatch.delenv("CDSAPI_RC", raising=False)


_CDS_TOKEN = "abcdef01-2345-6789-abcd-ef0123456789"


def test_cds_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cds_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = CredentialResolver().resolve("cds", override={"key": _CDS_TOKEN})
    assert resolved is not None
    assert resolved.source == "explicit"
    assert resolved.backend == "cds"
    assert resolved.fields["key"] == _CDS_TOKEN


def test_cds_override_with_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cds_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = CredentialResolver().resolve(
        "cds",
        override={
            "key": _CDS_TOKEN,
            "url": "https://ads.atmosphere.copernicus.eu/api",
        },
    )
    assert resolved is not None
    assert resolved.fields["key"] == _CDS_TOKEN
    assert resolved.fields["url"] == "https://ads.atmosphere.copernicus.eu/api"


def test_cds_env_resolves_key_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cds_env(monkeypatch)
    monkeypatch.setenv("CDSAPI_KEY", _CDS_TOKEN)
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = CredentialResolver().resolve("cds")
    assert resolved is not None
    assert resolved.source == "env"
    assert resolved.fields["key"] == _CDS_TOKEN
    # No url in env -> field absent (adapter falls back to cdsapi default).
    assert "url" not in resolved.fields


def test_cds_env_resolves_key_and_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cds_env(monkeypatch)
    monkeypatch.setenv("CDSAPI_KEY", _CDS_TOKEN)
    monkeypatch.setenv(
        "CDSAPI_URL", "https://ads.atmosphere.copernicus.eu/api"
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = CredentialResolver().resolve("cds")
    assert resolved is not None
    assert resolved.source == "env"
    assert resolved.fields["url"] == "https://ads.atmosphere.copernicus.eu/api"


def test_cds_empty_env_falls_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cds_env(monkeypatch)
    monkeypatch.setenv("CDSAPI_KEY", "")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert CredentialResolver().resolve("cds") is None


def test_cds_config_file_resolves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cds_env(monkeypatch)
    (tmp_path / ".cdsapirc").write_text(
        f"url: https://cds.climate.copernicus.eu/api\nkey: {_CDS_TOKEN}\n"
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = CredentialResolver().resolve("cds")
    assert resolved is not None
    assert resolved.source == "config_file"
    assert resolved.fields["key"] == _CDS_TOKEN
    assert resolved.fields["url"] == "https://cds.climate.copernicus.eu/api"
    assert resolved.source_detail == "~/.cdsapirc"


def test_cds_config_file_key_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A `.cdsapirc` containing only `key:` (no url) is valid — the
    adapter falls back to cdsapi's built-in default URL."""
    from copernicus_mcp.auth import CredentialResolver

    _clear_cds_env(monkeypatch)
    (tmp_path / ".cdsapirc").write_text(f"key: {_CDS_TOKEN}\n")
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = CredentialResolver().resolve("cds")
    assert resolved is not None
    assert resolved.fields["key"] == _CDS_TOKEN
    assert "url" not in resolved.fields


def test_cds_config_file_malformed_yaml_falls_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Malformed YAML must not crash the resolver — log debug, return None."""
    from copernicus_mcp.auth import CredentialResolver

    _clear_cds_env(monkeypatch)
    (tmp_path / ".cdsapirc").write_text("not: valid: yaml: at: all: [\n")
    monkeypatch.setenv("HOME", str(tmp_path))

    assert CredentialResolver().resolve("cds") is None


def test_cds_malformed_yaml_does_not_log_file_contents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Code-reviewer LOW: a malformed ``.cdsapirc`` could still contain
    a real PAT mid-edit. The resolver logs ``error_class`` only;
    ``yaml.YAMLError.__str__`` echoes a slice of the input which would
    leak if the log line ever included ``str(exc)`` or ``repr(exc)``."""
    from copernicus_mcp.auth import CredentialResolver

    _clear_cds_env(monkeypatch)
    leaked = "TOPSECRET-MID-EDIT-PAT"
    (tmp_path / ".cdsapirc").write_text(f"key: {leaked}: malformed-trailer\n")
    monkeypatch.setenv("HOME", str(tmp_path))

    with caplog.at_level(logging.DEBUG, logger="copernicus_mcp.auth.resolver"):
        result = CredentialResolver().resolve("cds")
    assert result is None
    for rec in caplog.records:
        full = rec.getMessage() + " " + " ".join(
            f"{k}={v}" for k, v in rec.__dict__.items()
        )
        assert leaked not in full


def test_cdsapi_rc_env_overrides_default_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``CDSAPI_RC`` env var points at an alternative rc-file location."""
    from copernicus_mcp.auth import CredentialResolver

    _clear_cds_env(monkeypatch)
    alt_path = tmp_path / "elsewhere" / "custom.rc"
    alt_path.parent.mkdir(parents=True)
    alt_path.write_text(f"key: {_CDS_TOKEN}\n")
    monkeypatch.setenv("CDSAPI_RC", str(alt_path))
    monkeypatch.setenv("HOME", str(tmp_path / "no-default-rc-here"))

    resolved = CredentialResolver().resolve("cds")
    assert resolved is not None
    assert resolved.source == "config_file"
    assert resolved.fields["key"] == _CDS_TOKEN
    # Codex LOW: source_detail must NOT echo the user-supplied path —
    # the env var name is enough to indicate provenance and avoids
    # leaking attacker-controlled path strings via repr.
    assert resolved.source_detail == "$CDSAPI_RC"
    assert str(alt_path) not in repr(resolved)


def test_cds_no_value_in_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    monkeypatch.setenv("CDSAPI_KEY", "TOPSECRET-CDS-PAT")
    monkeypatch.setenv("HOME", str(tmp_path))

    with caplog.at_level(logging.DEBUG, logger="copernicus_mcp.auth.resolver"):
        resolved = CredentialResolver().resolve("cds")
    assert resolved is not None
    for rec in caplog.records:
        full = rec.getMessage() + " " + " ".join(
            f"{k}={v}" for k, v in rec.__dict__.items()
        )
        assert "TOPSECRET-CDS-PAT" not in full


def test_cds_repr_no_value_leak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cds_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = CredentialResolver().resolve(
        "cds", override={"key": "TOPSECRET-CDS-VALUE"}
    )
    assert resolved is not None
    assert "TOPSECRET-CDS-VALUE" not in repr(resolved)
    assert "TOPSECRET-CDS-VALUE" not in str(resolved)


def test_cds_extra_keys_stripped_from_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cds_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = CredentialResolver().resolve(
        "cds",
        override={
            "key": _CDS_TOKEN,
            "username": "SHOULD-NOT-APPEAR",
            "password": "SHOULD-NOT-APPEAR",
        },
    )
    assert resolved is not None
    assert dict(resolved.fields) == {"key": _CDS_TOKEN}


def test_cds_listed_in_configured_backends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cmems_env(monkeypatch)
    _clear_cds_env(monkeypatch)
    monkeypatch.setenv("CDSAPI_KEY", _CDS_TOKEN)
    monkeypatch.setenv("HOME", str(tmp_path))

    configured = CredentialResolver().list_configured_backends()
    assert "cds" in configured
    assert "cmems" not in configured  # only cds env set


def test_str_does_not_leak_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    _clear_cmems_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = CredentialResolver().resolve(
        "cmems",
        override={"username": "TOPSECRET-USER", "password": "TOPSECRET-PASS"},
    )
    assert resolved is not None
    assert "TOPSECRET-USER" not in str(resolved)
    assert "TOPSECRET-PASS" not in str(resolved)


def test_list_configured_backends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from copernicus_mcp.auth import CredentialResolver

    monkeypatch.setenv("HOME", str(tmp_path))

    _clear_cmems_env(monkeypatch)
    # Codex LOW: this test enumerates configured backends. With CDS in
    # _KNOWN_BACKENDS, a host with ``CDSAPI_KEY`` set would otherwise
    # see ``["cds"]`` instead of ``[]``. Clear both env scopes.
    _clear_cds_env(monkeypatch)
    assert CredentialResolver().list_configured_backends() == []

    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_USERNAME", "u")
    monkeypatch.setenv("COPERNICUSMARINE_SERVICE_PASSWORD", "p")
    assert CredentialResolver().list_configured_backends() == ["cmems"]
