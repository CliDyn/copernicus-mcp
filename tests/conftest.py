from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio


@pytest.fixture(autouse=True)
def _reset_cli_globals() -> AsyncIterator[None]:
    """T-CDS-019 round-2 (cr local + codex LOW): the ``--cache-dir`` CLI
    flag is stored on a module-level global ``cli._cli_cache_dir``.
    A test that leaves it set (via direct assignment or a Typer invoke
    that didn't go through the root callback to clear it) could leak
    into the next test's ``_build_orchestrator_for_cli``. Autouse-reset
    before AND after each test to make leakage impossible.
    """
    import copernicus_mcp.cli as cli_mod

    cli_mod._cli_cache_dir = None
    try:
        yield
    finally:
        cli_mod._cli_cache_dir = None


@pytest.fixture(autouse=True)
def _no_network_costing(monkeypatch: pytest.MonkeyPatch) -> None:
    """T-CDS-EST2-001: ``CdsBackend.estimate``/``submit`` now call the live
    ``/costing`` endpoint via ``fetch_costing``. Unit tests must not touch the
    network — default every test to "costing unavailable" (returns ``None`` →
    the legacy heuristic), exactly as a real offline/endpoint-down run behaves.
    Tests that exercise the costing-available path override this with their own
    monkeypatch. The costing-client unit tests call ``costing.fetch_costing``
    directly with a fake transport, so this patch of the backend symbol does
    not affect them.
    """

    async def _none(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "copernicus_mcp.backends.cds.backend.fetch_costing", _none, raising=False
    )


@pytest.fixture
def tmp_state_dir(tmp_path: Path) -> Path:
    """Temporary directory mimicking the runtime layout under XDG dirs.

    Creates ``cache/`` (≈ ``~/.cache/copernicus-mcp``) and ``state/``
    (≈ ``~/.local/state/copernicus-mcp``) inside ``tmp_path`` and returns
    ``tmp_path``. Tests can read the two subdirs as needed.
    """
    (tmp_path / "cache").mkdir()
    (tmp_path / "state").mkdir()
    return tmp_path


@pytest_asyncio.fixture
async def sqlite_backend(tmp_path: Path) -> AsyncIterator:
    """Initialised, temporary ``SqliteBackend`` for persistence tests."""
    from copernicus_mcp.persistence import SqliteBackend

    backend = SqliteBackend(tmp_path / "state.db")
    await backend.initialise()
    try:
        yield backend
    finally:
        await backend.close()
