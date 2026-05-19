"""Deterministic, credential-independent cache-key construction.

Research §12.6.3. Plan T-012 step 2 widens the sensitive-key set to include
``authorization`` and mandates **recursive** filtering through nested dicts so
a leaked credential inside ``format_options.nested.password`` cannot taint the
key.
"""

from __future__ import annotations

import hashlib
from typing import Any

from copernicus_mcp.data_model.canonicalisation import canonicalise_value

SENSITIVE_REQUEST_KEYS: frozenset[str] = frozenset(
    {
        "username",
        "password",
        "client_secret",
        "key",
        "token",
        "authorization",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "id_token",
        "secret",
        "secret_key",
        "private_token",
        "personal_access_token",
        "cookie",
        "set-cookie",
        "x-api-key",
        "ocp-apim-subscription-key",
        "copernicusmarine_service_password",
    }
)


def _strip_sensitive(value: Any, _seen: set[int] | None = None) -> Any:
    """Return a copy of ``value`` with any sensitive keys removed recursively.

    Cycle-safe: a value already on the descent stack is returned unchanged so a
    pathological caller cannot trigger ``RecursionError``.
    """
    seen = _seen if _seen is not None else set()
    if id(value) in seen:
        return value
    if isinstance(value, dict | list | tuple):
        seen = seen | {id(value)}
    if isinstance(value, dict):
        return {
            k: _strip_sensitive(v, seen)
            for k, v in value.items()
            if k.lower() not in SENSITIVE_REQUEST_KEYS
        }
    if isinstance(value, list):
        return [_strip_sensitive(item, seen) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_sensitive(item, seen) for item in value)
    return value


def construct_cache_key(
    *,
    backend: str,
    operation: str,
    dataset_id: str,
    dataset_version: str | None,
    api_version: str,
    schema_version: str,
    spatial: dict[str, Any] | None,
    temporal: dict[str, Any] | None,
    variables: list[str] | None,
    format_options: dict[str, Any] | None,
) -> str:
    """Build ``{backend}:{operation}:{dataset_id}:{hash16}`` cache key.

    Sensitive keys (see ``SENSITIVE_REQUEST_KEYS``) are stripped recursively
    from each component dict before canonicalisation, so credentials never
    influence cache-key derivation.
    """
    components: list[str] = [
        f"backend={backend}",
        f"op={operation}",
        f"api={api_version}",
        f"schema={schema_version}",
        f"id={dataset_id}",
        f"ver={dataset_version if dataset_version is not None else 'null'}",
    ]
    pairs: list[tuple[str, Any]] = [
        ("spatial", spatial),
        ("temporal", temporal),
        ("variables", variables),
        ("format", format_options),
    ]
    for prefix, params in pairs:
        if params is None:
            continue
        cleaned = _strip_sensitive(params)
        components.append(f"{prefix}={canonicalise_value(cleaned)}")

    canonical = "|".join(components)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{backend}:{operation}:{dataset_id}:{digest}"
