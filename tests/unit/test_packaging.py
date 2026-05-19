from __future__ import annotations

from typer.testing import CliRunner


def test_version_importable() -> None:
    from copernicus_mcp import __version__

    assert isinstance(__version__, str)
    assert __version__


def test_cli_entrypoint_runs() -> None:
    from copernicus_mcp import __version__
    from copernicus_mcp.cli import app

    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output
