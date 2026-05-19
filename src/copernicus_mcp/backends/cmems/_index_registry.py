"""Index strategy registry for the Layer 2 CMEMS index pipeline.

Two-tier source of truth:

1. **Bundled snapshot** — ``_data/index_registry.json`` ships in the wheel
   with the spike-known datasets (4 INSITU-BGC file_based + 2 CORA-family
   path_based). Read at startup, lives in the in-memory overlay.
2. **Live discovery** — ``IndexRegistry.discover`` populates the overlay
   on demand for datasets that ship an index file we haven't catalogued
   yet (file_based only; path_based datasets are bundled-registry-only
   per sub-plan decision #19).

The negative cache (decision #14) prevents an auth-storm of repeat probes
on bogus dataset_ids: each ``record_negative`` write is TTL'd (5 min) and
bounded (1024 LRU).
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from copernicus_mcp.backends.cmems._index_parser import (
    FormatId,
    detect_format,
    parse_index,
)
from copernicus_mcp.errors import NotFoundError, ValidationError

if TYPE_CHECKING:
    import pandas as pd

    from copernicus_mcp.backends.cmems._index_store import IndexStore

_DEFAULT_BUNDLED_PATH = Path(__file__).parent / "_data" / "index_registry.json"
_NEGATIVE_CACHE_TTL_SECONDS = 300  # 5 min
_NEGATIVE_CACHE_CAP = 1024
_INDEX_REGEX = r".*index.*\.(txt|csv|tsv)$"
_URI_PRECEDENCE = ("history", "latest", "monthly")


def _utcnow() -> datetime:
    """Indirected for clock-injection in tests."""
    return datetime.now(UTC)


class RegistryEntry(BaseModel):
    """A single dataset's index-strategy descriptor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    strategy: Literal["file_based", "path_based"]
    index_filename: str | None
    format_id: FormatId
    discovered_via: Literal["bundled", "live_probe"]


class IndexRegistry:
    """In-memory overlay over the bundled ``index_registry.json`` snapshot.

    The overlay is the single source of truth at runtime; bundled entries
    pre-populate it at construction time. ``discover`` populates the
    overlay on demand for unregistered datasets that ship an index file.
    """

    def __init__(self, *, bundled_path: Path | None = None) -> None:
        path = bundled_path if bundled_path is not None else _DEFAULT_BUNDLED_PATH
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self._overlay: dict[str, RegistryEntry] = {
            dataset_id: RegistryEntry(**payload) for dataset_id, payload in raw.items()
        }
        self._negative_cache: OrderedDict[str, datetime] = OrderedDict()

    def get(self, dataset_id: str) -> RegistryEntry | None:
        return self._overlay.get(dataset_id)

    def list_dataset_ids(self) -> list[str]:
        return list(self._overlay.keys())

    def is_negatively_cached(self, dataset_id: str) -> bool:
        """True if ``dataset_id`` failed a probe within the TTL.

        Opportunistic TTL eviction on read: stale entries are removed
        when callers query them, keeping the cache from growing without
        bound between writes.
        """
        recorded_at = self._negative_cache.get(dataset_id)
        if recorded_at is None:
            return False
        if _utcnow() - recorded_at > timedelta(seconds=_NEGATIVE_CACHE_TTL_SECONDS):
            del self._negative_cache[dataset_id]
            return False
        return True

    def record_negative(self, dataset_id: str) -> None:
        """Mark ``dataset_id`` as a known-bad probe target.

        LRU-capped at ``_NEGATIVE_CACHE_CAP``; oldest insertion evicted
        when the cap would otherwise be exceeded. Re-inserting an
        existing key refreshes its position.
        """
        if dataset_id in self._negative_cache:
            self._negative_cache.move_to_end(dataset_id)
        self._negative_cache[dataset_id] = _utcnow()
        while len(self._negative_cache) > _NEGATIVE_CACHE_CAP:
            self._negative_cache.popitem(last=False)

    async def discover(
        self,
        dataset_id: str,
        *,
        marine: object,
        lock: asyncio.Lock,
        store: IndexStore,
    ) -> tuple[RegistryEntry, pd.DataFrame | None]:
        """Probe the SDK for an index file and populate the overlay on hit.

        Only ``file_based`` discovery is supported (sub-plan decision #19):
        the regex matches ``*index*.{txt,csv,tsv}`` files; path-based
        datasets ship no such file and are reachable only via the bundled
        registry.

        The ``lock`` parameter is reserved for the spec'd contract; the
        actual serialisation against concurrent first-touches is now
        enforced by ``IndexStore.load`` (cr round-1 H1 fix).

        Returns ``(entry, df)``:
        - If ``dataset_id`` is already in the overlay (race resolved by
          another coroutine), returns ``(existing_entry, None)`` — the
          caller already has a cache to load from.
        - On a live-probe hit, returns the new entry + the parsed
          DataFrame so the caller can write Parquet without re-fetching.
        - On miss: records the failure in the negative cache and raises
          canonical ``NotFoundError`` with a recovery hint.
        """
        del lock  # spec-reserved parameter — serialisation is in IndexStore.load.
        existing = self._overlay.get(dataset_id)
        if existing is not None:
            return existing, None

        # Wrap the SDK probe through the store's wrap_sdk_error helper so
        # credentials embedded in raw SDK exception messages cannot leak
        # (the credential-isolation invariant; cr round-1 H2).
        try:
            response = await asyncio.to_thread(
                marine.get,  # type: ignore[attr-defined]
                dataset_id=dataset_id,
                regex=_INDEX_REGEX,
                dry_run=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise store._wrap_sdk_error(marine, exc, "discover") from exc
        uri = self._pick_preferred_uri(getattr(response, "files", []) or [])
        if uri is None:
            self.record_negative(dataset_id)
            raise NotFoundError(
                f"no index file found for dataset_id={dataset_id!r}; "
                "marine_get_files(filter=...) works for unindexed datasets, "
                "or a refreshed registry shipped in the next release "
                "would add this dataset."
            )

        index_filename = os.path.basename(uri)
        try:
            raw = await store._fetch_index_bytes(dataset_id, index_filename)
        except NotFoundError:
            # cr round-2 H-R2-1: probe matched a URI but the download
            # wrote nothing. Treat as a "no index" outcome and populate
            # the negative cache so the next probe is short-circuited
            # (decision #14 auth-storm prevention).
            self.record_negative(dataset_id)
            raise
        try:
            format_id = detect_format(raw)
        except ValidationError:
            # T-CMEMS-GET-INDEX-006 cr round-1 I1: the SDK shipped an
            # index file we can't parse (e.g. the canyon multiobs format).
            # Record negative so adversarial-iteration retries can't
            # auth-storm us re-probing the same unsupported dataset.
            self.record_negative(dataset_id)
            raise
        df = parse_index(raw, expected_format=format_id)
        entry = RegistryEntry(
            dataset_id=dataset_id,
            strategy="file_based",
            index_filename=index_filename,
            format_id=format_id,
            discovered_via="live_probe",
        )
        self._overlay[dataset_id] = entry
        return entry, df

    @staticmethod
    def _pick_preferred_uri(files: list[object]) -> str | None:
        """Pick the best-matched URI per the precedence ``history > latest > monthly > any``."""
        uris: list[str] = [
            getattr(f, "file_path", "")
            for f in files
            if getattr(f, "file_path", None)
        ]
        if not uris:
            return None
        for marker in _URI_PRECEDENCE:
            for uri in uris:
                if marker in os.path.basename(uri).lower():
                    return uri
        return uris[0]
