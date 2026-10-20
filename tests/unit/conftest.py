from __future__ import annotations

import os

import pytest

# Host variables that would let a unit test reach the developer's real
# credentials or redirect platformdirs away from the sandboxed HOME.
_HOST_ENV_VARS = (
    "CDSAPI_KEY",
    "CDSAPI_URL",
    "CDSAPI_RC",
    "COPERNICUSMARINE_SERVICE_USERNAME",
    "COPERNICUSMARINE_SERVICE_PASSWORD",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
)


@pytest.fixture(autouse=True)
def _isolate_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Keep the developer's own setup out of unit tests.

    Without this, a real ``~/.config/copernicus-mcp/config.yaml`` (e.g. one
    that enables ``cds_licence_accept_enabled``), an exported
    ``COPERNICUS_MCP_CACHE_DIR``, the real ``~/.local/state`` state DB or the
    real ``~/.cdsapirc`` leak into tests that build the default foundation.
    HOME points at a fresh per-test directory (outside ``tmp_path``, which
    some tests expect to start empty), so every ``~``/platformdirs path
    (state DB, cache, credential files) resolves inside the sandbox. Tests
    that exercise these layers set them explicitly via monkeypatch.
    Integration tests are not affected (they live outside this directory).
    """
    from copernicus_mcp.config import loader as loader_module

    # ``_USER_CONFIG_PATHS`` is expanded at import time, before HOME moves.
    monkeypatch.setattr(loader_module, "_USER_CONFIG_PATHS", ())
    for key in [k for k in os.environ if k.startswith("COPERNICUS_MCP_")]:
        monkeypatch.delenv(key)
    for key in _HOST_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
