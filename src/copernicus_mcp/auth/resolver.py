from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol

from copernicus_mcp.observability.logger import get_logger

logger = get_logger(__name__)

CredentialSource = Literal[
    "explicit", "secret_manager", "env", "config_file", "missing"
]

_KNOWN_BACKENDS: tuple[str, ...] = ("cmems", "cds")
_CMEMS_USER_ENV = "COPERNICUSMARINE_SERVICE_USERNAME"
_CMEMS_PW_ENV = "COPERNICUSMARINE_SERVICE_PASSWORD"
_CMEMS_FILE_LITERAL = "~/.copernicusmarine/.copernicusmarine-credentials"

# T-CDS-001: the project research notes §6.8.3.
_CDS_KEY_ENV = "CDSAPI_KEY"
_CDS_URL_ENV = "CDSAPI_URL"
_CDS_RC_PATH_ENV = "CDSAPI_RC"
_CDS_FILE_LITERAL = "~/.cdsapirc"

# Per-backend allowed credential keys. Anything else a caller (override or
# secret manager) supplies is dropped at the boundary so it cannot reach
# logs, repr, downstream consumers, or callers that introspect ``fields``.
_BACKEND_ALLOWED_KEYS: dict[str, frozenset[str]] = {
    "cmems": frozenset({"username", "password"}),
    # ``url`` is optional — when absent the adapter falls back to cdsapi's
    # default. ``key`` is the PAT (UUID).
    "cds": frozenset({"key", "url"}),
}


class SecretManagerProvider(Protocol):
    def fetch(self, backend: str) -> Mapping[str, str] | None: ...


@dataclass(frozen=True, repr=False)
class ResolvedCredentials:
    """Frozen, redacted view of credentials for one backend."""

    backend: str
    fields: Mapping[str, str] = field(default_factory=dict)
    source: CredentialSource = "missing"
    source_detail: str | None = None

    def __post_init__(self) -> None:
        # Defensive copy + immutable view; bypass frozen to assign.
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))

    def __repr__(self) -> str:
        return (
            f"ResolvedCredentials(backend={self.backend!r}, "
            f"source={self.source!r}, "
            f"source_detail={self.source_detail!r}, "
            f"field_count={len(self.fields)})"
        )


def _is_complete_cmems(fields: Mapping[str, str] | None) -> bool:
    if not fields:
        return False
    user = fields.get("username")
    pw = fields.get("password")
    return bool(user) and bool(pw)


def _is_complete_cds(fields: Mapping[str, str] | None) -> bool:
    """CDS requires a non-empty ``key`` (PAT). ``url`` is optional —
    when absent the adapter falls back to cdsapi's built-in default."""
    if not fields:
        return False
    return bool(fields.get("key"))


def _parse_cdsapi_rc(path: Path) -> dict[str, str]:
    """Parse a ``~/.cdsapirc`` YAML file into ``{key, url}`` if present.

    Per ``the project research notes`` §6.8.3 the file is YAML
    with two recognised keys: ``url`` and ``key``. Malformed YAML returns
    an empty dict — caller falls through to ``None``.
    """
    import yaml  # type: ignore[import-untyped]

    out: dict[str, str] = {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (yaml.YAMLError, OSError, ValueError) as exc:
        logger.debug(
            "cdsapirc file unreadable or malformed",
            extra={"error_class": type(exc).__name__},
        )
        return out
    if not isinstance(loaded, dict):
        return out
    key = loaded.get("key")
    url = loaded.get("url")
    if isinstance(key, str) and key:
        out["key"] = key
    if isinstance(url, str) and url:
        out["url"] = url
    return out


def _maybe_b64_decode(text: str) -> str:
    """Decode a base64-wrapped credentials blob, or return ``text`` unchanged.

    ``copernicusmarine >= 2.0`` writes the credentials file as a single
    base64-encoded line wrapping the legacy INI body
    (``[credentials]\\nusername=...\\npassword=...``). Older versions wrote
    the INI directly.

    Strategy: try base64-decoding the *entire* trimmed file. If the decoded
    bytes are valid UTF-8 AND contain both ``username=`` and ``password=``
    markers, the wrapper was applied — return the decoded body. Otherwise
    return ``text`` unchanged. This is robust against the legacy plain-INI
    case (which won't decode cleanly as base64 because of newlines /
    non-base64 chars) and against the ambiguity of base64 padding ``=``
    looking like an INI key/value separator.
    """
    import base64
    import binascii

    stripped = text.strip()
    if not stripped:
        return text
    try:
        decoded_bytes = base64.b64decode(stripped, validate=True)
    except (binascii.Error, ValueError):
        return text
    try:
        decoded = decoded_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return text
    if "username=" in decoded and "password=" in decoded:
        return decoded
    return text


def _parse_credentials_file(path: Path) -> dict[str, str]:
    """Parse a key=value credentials file; ignore comments, blanks, malformed lines.

    Supports two formats: legacy plain INI / ``key=value`` lines, and the
    base64-wrapped INI used by ``copernicusmarine >= 2.0``. The
    ``[credentials]`` section header (and any other ``[...]`` lines) are
    ignored.
    """
    out: dict[str, str] = {}
    text = _maybe_b64_decode(path.read_text(encoding="utf-8-sig"))
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            logger.debug(
                "skipping malformed credentials line",
                extra={"reason": "no_equals"},
            )
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            logger.debug(
                "skipping malformed credentials line",
                extra={"reason": "empty_key"},
            )
            continue
        out[key] = value
    return out


class CredentialResolver:
    """Resolve credentials through a fixed precedence hierarchy.

    Iteration 1: only ``cmems``. Hierarchy:
        explicit override → secret_manager → env vars → ``~/.copernicusmarine/.copernicusmarine-credentials`` → None.
    """

    def __init__(
        self,
        secret_manager_provider: SecretManagerProvider | None = None,
    ) -> None:
        self._sm = secret_manager_provider

    def resolve(
        self,
        backend: str,
        override: Mapping[str, str] | None = None,
    ) -> ResolvedCredentials | None:
        if backend not in _KNOWN_BACKENDS:
            return None
        if backend == "cmems":
            return self._resolve_cmems(override)
        if backend == "cds":
            return self._resolve_cds(override)
        return None

    def _resolve_cmems(
        self, override: Mapping[str, str] | None
    ) -> ResolvedCredentials | None:
        backend = "cmems"

        # 1. Explicit override.
        if override is not None and _is_complete_cmems(override):
            return self._build(backend, override, "explicit", None)

        # 2. Secret manager. A faulty provider must not leak raw exception
        # messages through to callers — swallow as "missing" with DEBUG.
        if self._sm is not None:
            try:
                sm_fields = self._sm.fetch(backend)
            except Exception as exc:
                logger.debug(
                    "secret manager provider raised",
                    extra={
                        "backend": backend,
                        "source": "secret_manager",
                        "error_class": type(exc).__name__,
                    },
                )
                sm_fields = None
            if sm_fields is not None and _is_complete_cmems(sm_fields):
                return self._build(backend, sm_fields, "secret_manager", None)

        # 3. Env vars.
        env_user = os.environ.get(_CMEMS_USER_ENV) or ""
        env_pw = os.environ.get(_CMEMS_PW_ENV) or ""
        if env_user and env_pw:
            return self._build(
                backend,
                {"username": env_user, "password": env_pw},
                "env",
                f"{_CMEMS_USER_ENV},{_CMEMS_PW_ENV}",
            )

        # 4. Credentials file.
        file_path = Path(
            "~/.copernicusmarine/.copernicusmarine-credentials"
        ).expanduser()
        if file_path.exists():
            try:
                parsed = _parse_credentials_file(file_path)
            except OSError as exc:
                logger.debug(
                    "credentials file read failed",
                    extra={
                        "backend": backend,
                        "source": "config_file",
                        "error_class": type(exc).__name__,
                    },
                )
                return None
            if _is_complete_cmems(parsed):
                return self._build(
                    backend,
                    {
                        "username": parsed["username"],
                        "password": parsed["password"],
                    },
                    "config_file",
                    _CMEMS_FILE_LITERAL,
                )

        return None

    def _resolve_cds(
        self, override: Mapping[str, str] | None
    ) -> ResolvedCredentials | None:
        """Resolve CDS PAT through the same precedence as cmems but with
        the CDS-specific env vars and ``~/.cdsapirc`` YAML file.

        Per ``the project research notes`` §6.8.3.
        """
        backend = "cds"

        # 1. Explicit override.
        if override is not None and _is_complete_cds(override):
            return self._build(backend, override, "explicit", None)

        # 2. Secret manager.
        if self._sm is not None:
            try:
                sm_fields = self._sm.fetch(backend)
            except Exception as exc:
                logger.debug(
                    "secret manager provider raised",
                    extra={
                        "backend": backend,
                        "source": "secret_manager",
                        "error_class": type(exc).__name__,
                    },
                )
                sm_fields = None
            if sm_fields is not None and _is_complete_cds(sm_fields):
                return self._build(backend, sm_fields, "secret_manager", None)

        # 3. Env vars.
        env_key = os.environ.get(_CDS_KEY_ENV) or ""
        if env_key:
            fields: dict[str, str] = {"key": env_key}
            env_url = os.environ.get(_CDS_URL_ENV) or ""
            if env_url:
                fields["url"] = env_url
            detail = (
                f"{_CDS_KEY_ENV},{_CDS_URL_ENV}" if env_url else _CDS_KEY_ENV
            )
            return self._build(backend, fields, "env", detail)

        # 4. Config file. ``CDSAPI_RC`` overrides the default location
        # (mirrors the cdsapi 0.7.7 lookup hierarchy).
        # codex round-1 LOW: store the env-var name in ``source_detail``,
        # NOT the user-supplied path. ``ResolvedCredentials.__repr__``
        # echoes ``source_detail``, and an attacker-controlled path like
        # ``/tmp/TOPSECRET-PAT.rc`` would otherwise leak through repr.
        rc_override = os.environ.get(_CDS_RC_PATH_ENV)
        file_path: Path
        source_detail: str
        if rc_override:
            file_path = Path(rc_override).expanduser()
            source_detail = f"${_CDS_RC_PATH_ENV}"
        else:
            file_path = Path(_CDS_FILE_LITERAL).expanduser()
            source_detail = _CDS_FILE_LITERAL
        if file_path.exists():
            parsed = _parse_cdsapi_rc(file_path)
            if _is_complete_cds(parsed):
                return self._build(
                    backend, parsed, "config_file", source_detail
                )

        return None

    def list_configured_backends(self) -> list[str]:
        return [b for b in _KNOWN_BACKENDS if self.resolve(b) is not None]

    def _build(
        self,
        backend: str,
        fields: Mapping[str, str],
        source: CredentialSource,
        source_detail: str | None,
    ) -> ResolvedCredentials:
        # Whitelist keys at the boundary so callers cannot smuggle anything
        # beyond the expected credential shape into ResolvedCredentials.
        allowed = _BACKEND_ALLOWED_KEYS.get(backend, frozenset())
        filtered = {k: v for k, v in fields.items() if k in allowed}
        resolved = ResolvedCredentials(
            backend=backend,
            fields=filtered,
            source=source,
            source_detail=source_detail,
        )
        logger.debug(
            "resolved credentials",
            extra={
                "backend": backend,
                "source": source,
                "field_count": len(resolved.fields),
            },
        )
        return resolved
