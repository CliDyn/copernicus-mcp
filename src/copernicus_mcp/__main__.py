"""Allow ``python -m copernicus_mcp ...`` to invoke the Typer CLI.

The installed console script ``copernicus-mcp`` (defined in
``pyproject.toml``) is the primary entrypoint; this module exists so
tests and tooling that use ``-m`` (without relying on PATH) reach the
same app.
"""

from copernicus_mcp.cli import app

if __name__ == "__main__":
    app()
