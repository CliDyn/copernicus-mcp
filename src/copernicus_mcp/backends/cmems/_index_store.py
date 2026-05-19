"""On-demand fetch + per-user Parquet cache for CMEMS Layer 2 indices.

``IndexStore.load(dataset_id)`` is the runtime entry point:

- Registered dataset (overlay hit): check the Parquet cache; on hit
  return ``(df, "offline")``; on miss fetch via the SDK, parse, write
  the cache atomically, return ``(df, "fresh")``.
- Unregistered dataset: delegate to ``IndexRegistry.discover`` (file-based
  live probe). If discover succeeds, write the Parquet cache and return
  ``(df, "fresh")``. If it fails, propagate canonical ``NotFoundError``.

Architectural choices (sub-plan decisions §10–§19):

- Per-dataset ``asyncio.Lock`` via ``dict.setdefault`` (atomic under CPython
  GIL) so concurrent first-touches share one lock.
- ``tempfile.TemporaryDirectory`` is exited **before** parsing so a
  cancelled fetch never leaves SDK partials behind.
- Parquet writes use ``_atomic_write_bytes`` so a crashed write cannot
  leave a half-written cache file in place.
"""

from __future__ import annotations

import asyncio
import io
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

from copernicus_mcp.backends.cmems._index_parser import (
    parse_cora_paths,
    parse_easycora_paths,
    parse_index,
)
from copernicus_mcp.backends.cmems._index_registry import IndexRegistry, RegistryEntry
from copernicus_mcp.common.atomic import _atomic_write_bytes
from copernicus_mcp.errors import BackendError, NotFoundError
from copernicus_mcp.errors.sanitiser import Sanitiser

if TYPE_CHECKING:
    from copernicus_mcp.auth import AuthAdapter


WrapException = Callable[[object, BaseException, str], Exception]
MarineLoader = Callable[[], object]


class IndexStore:
    """Runtime cache + fetch orchestrator for CMEMS index DataFrames."""

    def __init__(
        self,
        *,
        registry: IndexRegistry,
        cache_directory: Path,
        marine_loader: MarineLoader,
        auth_adapter: AuthAdapter | None = None,
        wrap_exception: WrapException | None = None,
        sanitiser: Sanitiser | None = None,
    ) -> None:
        self._registry = registry
        self._cache_directory = Path(cache_directory)
        self._marine_loader = marine_loader
        self._auth_adapter = auth_adapter
        self._wrap_exception = wrap_exception
        # cr round-1 H2: default Sanitiser scrubs credentials from SDK
        # exception messages so the wrap=None default never leaks creds.
        self._sanitiser = sanitiser if sanitiser is not None else Sanitiser()
        self._dataset_locks: dict[str, asyncio.Lock] = {}

    @property
    def sanitiser(self) -> Sanitiser:
        return self._sanitiser

    @property
    def wrap_exception(self) -> WrapException | None:
        return self._wrap_exception

    # ----- Public API -----

    async def load(
        self, dataset_id: str
    ) -> tuple[pd.DataFrame, Literal["offline", "fresh"]]:
        """Return the parsed index DataFrame and freshness mode for ``dataset_id``.

        Concurrency contract (cr round-1 H1): all SDK-bound work happens
        under the per-dataset lock. The cache-hit fast path stays
        outside the lock — atomic-rename semantics make
        ``pd.read_parquet`` consistent with concurrent writes.

        Negative-cache short-circuit (cr round-1 H3, decision #14): a
        previously-failed discovery for the same ``dataset_id`` raises
        ``NotFoundError`` immediately, without re-probing the SDK.
        """
        cache_path = self._cache_path(dataset_id)
        if cache_path.exists():
            return pd.read_parquet(cache_path), "offline"

        if self._registry.is_negatively_cached(dataset_id):
            raise NotFoundError(
                f"dataset_id={dataset_id!r} is in the negative-cache window — "
                "a recent index probe found nothing. Try again later, or "
                "use marine_get_files(filter=...) for unindexed datasets."
            )

        lock = self._dataset_locks.setdefault(dataset_id, asyncio.Lock())
        async with lock:
            # Re-check under the lock — another coroutine may have just
            # populated the cache between our outer check and the lock.
            if cache_path.exists():
                return pd.read_parquet(cache_path), "offline"

            entry = self._registry.get(dataset_id)
            if entry is None:
                new_entry, df = await self._registry.discover(
                    dataset_id, marine=self._marine_loader(), lock=lock, store=self
                )
                # discover may return df=None when the overlay already had
                # the entry (race resolved during lock acquire). In that
                # case the cache may already be fresh; otherwise fetch.
                if df is None:
                    if cache_path.exists():
                        return pd.read_parquet(cache_path), "offline"
                    df = await self._fetch_and_parse(new_entry)
            else:
                df = await self._fetch_and_parse(entry)

            self._write_cache(cache_path, df)
            return df, "fresh"

    async def refresh(self, dataset_id: str) -> None:
        """Force-fetch an already-registered dataset, overwriting the cache.

        Refresh operates on **registered datasets only** (sub-plan
        decision #13): unregistered dataset_ids raise ``NotFoundError``
        immediately without calling the SDK. To populate an unregistered
        dataset, call ``load(...)`` first — it goes through ``discover``.
        """
        entry = self._registry.get(dataset_id)
        if entry is None:
            raise NotFoundError(
                f"refresh: dataset_id={dataset_id!r} is not in the index registry. "
                "Call marine_list_files first to trigger live discovery."
            )
        lock = self._dataset_locks.setdefault(dataset_id, asyncio.Lock())
        cache_path = self._cache_path(dataset_id)
        async with lock:
            df = await self._fetch_and_parse(entry)
            self._write_cache(cache_path, df)

    async def fetched_at(self, dataset_id: str) -> datetime | None:
        """Return the Parquet cache file's mtime, or ``None`` if missing."""
        cache_path = self._cache_path(dataset_id)
        if not cache_path.exists():
            return None
        return datetime.fromtimestamp(cache_path.stat().st_mtime, tz=UTC)

    def list_cached_datasets(self) -> list[str]:
        """Enumerate dataset_ids with a Parquet cache file."""
        cache_dir = self._cache_directory / "marine_indices"
        if not cache_dir.exists():
            return []
        return sorted(p.stem for p in cache_dir.iterdir() if p.suffix == ".parquet")

    # ----- Internal helpers -----

    async def _fetch_index_bytes(
        self, dataset_id: str, index_filename: str
    ) -> bytes:
        """SDK fetch helper — downloads ``index_filename`` and returns the bytes.

        Used by both ``IndexRegistry.discover`` (when the filename is
        known but the format isn't yet) and ``_fetch_and_parse`` (when
        the registry entry pins the format already).
        """
        marine = self._marine_loader()
        kwargs: dict[str, object] = {
            "dataset_id": dataset_id,
            "file_list": [index_filename],
        }
        if self._auth_adapter is not None:
            user, password = self._auth_adapter.get_username_password()
            kwargs["username"] = user
            kwargs["password"] = password

        with tempfile.TemporaryDirectory(prefix="cmems_idx_") as tmp_str:
            tmp = Path(tmp_str)
            kwargs["output_directory"] = str(tmp)
            try:
                await asyncio.to_thread(marine.get, **kwargs)  # type: ignore[attr-defined]
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise self._wrap_sdk_error(marine, exc, "list_files") from exc

            matches = list(tmp.rglob(index_filename))
            if not matches:
                # cr round-1 M1: SDK accepted the call but wrote nothing
                # → the requested file does not exist for this dataset_id.
                # Route the user to "check dataset_id" via NotFoundError,
                # not "report SDK bug" via BackendError.
                raise NotFoundError(
                    f"no {index_filename!r} found in the SDK response for "
                    f"this dataset; verify the dataset_id is correct or use "
                    "marine_get_files(filter=...) for unindexed datasets."
                )
            return matches[0].read_bytes()

    async def _fetch_and_parse(self, entry: RegistryEntry) -> pd.DataFrame:
        """Dispatch on strategy: file_based downloads CSV, path_based runs dry_run listing."""
        if entry.strategy == "file_based":
            assert entry.index_filename is not None
            raw = await self._fetch_index_bytes(
                entry.dataset_id, entry.index_filename
            )
            return parse_index(raw, expected_format=entry.format_id)

        # path_based: dry_run=True returns a full file listing; we parse the paths.
        marine = self._marine_loader()
        kwargs: dict[str, object] = {
            "dataset_id": entry.dataset_id,
            "dry_run": True,
        }
        if self._auth_adapter is not None:
            user, password = self._auth_adapter.get_username_password()
            kwargs["username"] = user
            kwargs["password"] = password
        try:
            response = await asyncio.to_thread(marine.get, **kwargs)  # type: ignore[attr-defined]
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise self._wrap_sdk_error(marine, exc, "list_files") from exc

        paths = [
            getattr(f, "file_path", "")
            for f in (getattr(response, "files", []) or [])
            if getattr(f, "file_path", None)
        ]
        if entry.format_id == "cora_path_v1":
            return parse_cora_paths(paths)
        if entry.format_id == "easycora_path_v1":
            return parse_easycora_paths(paths)
        raise BackendError(
            f"_fetch_and_parse: unsupported path_based format_id={entry.format_id!r}"
        )

    def _wrap_sdk_error(
        self, marine: object, exc: BaseException, op: str
    ) -> Exception:
        """Wrap an SDK exception into a canonical class.

        If a ``wrap_exception`` callable was injected at construction,
        delegate to it (production path uses ``_wrap_subset_exception``
        from ``backends.cmems.backend``). Otherwise fall back to a
        sanitiser-based ``BackendError`` so credentials embedded in raw
        SDK messages never leak through the default path
        (cr round-1 H2).
        """
        if self._wrap_exception is not None:
            return self._wrap_exception(marine, exc, op)
        message = self._sanitiser.sanitise(str(exc))
        return BackendError(f"SDK {op} failed: {message}")

    def _cache_path(self, dataset_id: str) -> Path:
        return self._cache_directory / "marine_indices" / f"{dataset_id}.parquet"

    def _write_cache(self, cache_path: Path, df: pd.DataFrame) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        df.to_parquet(buf)
        _atomic_write_bytes(cache_path, buf.getvalue())
