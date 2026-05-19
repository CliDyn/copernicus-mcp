"""Unit tests for ``copernicus_mcp.backends.cmems._index_registry``.

The registry owns the dataset_id → strategy mapping. T-CMEMS-GET-INDEX-003
ships a bundled JSON of six entries (spike-discovered) that the runtime
preloads at startup; ``discover()`` populates the in-memory overlay on
demand for any other dataset_id.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from copernicus_mcp.backends.cmems._index_registry import IndexRegistry, RegistryEntry

_BUNDLED_JSON = (
    Path(__file__).parent.parent.parent
    / "src"
    / "copernicus_mcp"
    / "backends"
    / "cmems"
    / "_data"
    / "index_registry.json"
)


class TestRegistryEntry:
    def _file_based_kwargs(self) -> dict:
        return dict(
            dataset_id="cmems_obs-ins_glo_bgc-car_my_socat-obs_irr",
            strategy="file_based",
            index_filename="index_file.txt",
            format_id="insitu_index_file_v3",
            discovered_via="bundled",
        )

    def _path_based_kwargs(self) -> dict:
        return dict(
            dataset_id="cmems_obs-ins_glo_phy-temp-sal_my_cora_irr",
            strategy="path_based",
            index_filename=None,
            format_id="cora_path_v1",
            discovered_via="bundled",
        )

    def test_file_based_entry_constructs(self) -> None:
        entry = RegistryEntry(**self._file_based_kwargs())
        assert entry.strategy == "file_based"
        assert entry.index_filename == "index_file.txt"

    def test_path_based_entry_constructs(self) -> None:
        entry = RegistryEntry(**self._path_based_kwargs())
        assert entry.strategy == "path_based"
        assert entry.index_filename is None

    def test_is_frozen(self) -> None:
        entry = RegistryEntry(**self._file_based_kwargs())
        with pytest.raises(PydanticValidationError):
            entry.dataset_id = "other"  # type: ignore[misc]

    def test_rejects_extra_fields(self) -> None:
        kwargs = self._file_based_kwargs()
        kwargs["unknown"] = "boom"
        with pytest.raises(PydanticValidationError):
            RegistryEntry(**kwargs)

    def test_rejects_unknown_strategy(self) -> None:
        kwargs = self._file_based_kwargs()
        kwargs["strategy"] = "magic_mode"
        with pytest.raises(PydanticValidationError):
            RegistryEntry(**kwargs)

    def test_rejects_unknown_format_id(self) -> None:
        kwargs = self._file_based_kwargs()
        kwargs["format_id"] = "not_a_format"
        with pytest.raises(PydanticValidationError):
            RegistryEntry(**kwargs)

    def test_rejects_unknown_discovered_via(self) -> None:
        kwargs = self._file_based_kwargs()
        kwargs["discovered_via"] = "guessed"
        with pytest.raises(PydanticValidationError):
            RegistryEntry(**kwargs)


class TestIndexRegistryBundled:
    """Bundled-JSON behaviour (no live discover)."""

    def test_loads_bundled_json_from_default_path(self) -> None:
        registry = IndexRegistry()
        # Six spike-known datasets must be present.
        assert registry.get("cmems_obs-ins_glo_bgc-car_my_socat-obs_irr") is not None
        assert registry.get("cmems_obs-ins_glo_phy-temp-sal_my_cora_irr") is not None

    def test_get_returns_none_for_unknown(self) -> None:
        registry = IndexRegistry()
        assert registry.get("not_a_real_dataset_id") is None

    def test_get_returns_typed_entry(self) -> None:
        registry = IndexRegistry()
        entry = registry.get("cmems_obs-ins_glo_phy-temp-sal_my_cora_irr")
        assert entry is not None
        assert isinstance(entry, RegistryEntry)
        assert entry.strategy == "path_based"
        assert entry.format_id == "cora_path_v1"
        assert entry.index_filename is None

    def test_bundled_entries_all_marked_discovered_via_bundled(self) -> None:
        registry = IndexRegistry()
        for dataset_id in registry.list_dataset_ids():
            entry = registry.get(dataset_id)
            assert entry is not None
            assert entry.discovered_via == "bundled"

    def test_list_dataset_ids_returns_all_bundled(self) -> None:
        registry = IndexRegistry()
        ids = set(registry.list_dataset_ids())
        # T-CMEMS-GET-INDEX-008: MULTIOBS canyon back in the bundled
        # registry — its `multiobs_canyon_v1` parser ships in this PR.
        expected = {
            "cmems_obs-ins_glo_bgc-car_my_glodap-gridded_irr",
            "cmems_obs-ins_glo_bgc-car_my_glodap-obs_irr",
            "cmems_obs-ins_glo_bgc-car_my_socat-obs_irr",
            "cmems_obs-mob_glo_bgc-nut-car_mynrt_irr_i",
            "cmems_obs-ins_glo_phy-temp-sal_my_cora_irr",
            "cmems_obs-ins_glo_phy-temp-sal_my_easycora_irr",
        }
        assert expected <= ids

    def test_accepts_custom_path(self, tmp_path: Path) -> None:
        # Empty registry — useful in tests where we don't want the bundled defaults.
        custom = tmp_path / "empty.json"
        custom.write_text("{}", encoding="utf-8")
        registry = IndexRegistry(bundled_path=custom)
        assert registry.list_dataset_ids() == []

    def test_loads_only_file_path_so_cli_args_dont_leak(self, tmp_path: Path) -> None:
        # The constructor must not silently swallow a malformed JSON
        # path; an unreadable file should surface as a real exception.
        with pytest.raises((FileNotFoundError, OSError)):
            IndexRegistry(bundled_path=tmp_path / "does_not_exist.json")


class TestNegativeCache:
    """TTL'd negative cache for failed-probe dataset ids (decision #14)."""

    def test_freshly_constructed_registry_has_empty_negative_cache(
        self, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty.json"
        empty.write_text("{}", encoding="utf-8")
        reg = IndexRegistry(bundled_path=empty)
        assert reg.is_negatively_cached("anything") is False

    def test_record_negative_persists_for_ttl(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.json"
        empty.write_text("{}", encoding="utf-8")
        reg = IndexRegistry(bundled_path=empty)
        reg.record_negative("bogus.dataset")
        assert reg.is_negatively_cached("bogus.dataset") is True

    def test_record_negative_evicts_after_ttl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Use a clock injection so the test is deterministic.
        from copernicus_mcp.backends.cmems import _index_registry as mod

        empty = tmp_path / "empty.json"
        empty.write_text("{}", encoding="utf-8")
        reg = IndexRegistry(bundled_path=empty)

        from datetime import UTC, datetime, timedelta

        now = datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)

        def fake_now() -> datetime:
            return now

        monkeypatch.setattr(mod, "_utcnow", fake_now)
        reg.record_negative("bogus.dataset")
        assert reg.is_negatively_cached("bogus.dataset") is True

        # Advance clock past the 5-minute TTL.
        now = now + timedelta(seconds=mod._NEGATIVE_CACHE_TTL_SECONDS + 1)
        assert reg.is_negatively_cached("bogus.dataset") is False

    def test_negative_cache_lru_cap(self, tmp_path: Path) -> None:
        from copernicus_mcp.backends.cmems import _index_registry as mod

        empty = tmp_path / "empty.json"
        empty.write_text("{}", encoding="utf-8")
        reg = IndexRegistry(bundled_path=empty)

        # Fill past the cap.
        cap = mod._NEGATIVE_CACHE_CAP
        for i in range(cap + 5):
            reg.record_negative(f"bogus_{i}")

        # The oldest 5 entries should have been evicted.
        for i in range(5):
            assert reg.is_negatively_cached(f"bogus_{i}") is False
        # The newest entry is still cached.
        assert reg.is_negatively_cached(f"bogus_{cap + 4}") is True


class TestDiscover:
    """``IndexRegistry.discover`` populates the overlay from a live SDK probe."""

    @pytest.mark.asyncio
    async def test_returns_existing_entry_if_already_in_overlay(
        self, tmp_path: Path
    ) -> None:
        # If the dataset_id is already known, discover returns the entry
        # without calling the SDK.
        import asyncio
        from unittest.mock import MagicMock

        reg = IndexRegistry()
        existing_id = "cmems_obs-ins_glo_bgc-car_my_socat-obs_irr"
        marine = MagicMock()
        store = MagicMock()
        store._fetch_index_bytes = MagicMock()
        lock = asyncio.Lock()
        entry, df = await reg.discover(
            existing_id, marine=marine, lock=lock, store=store
        )
        assert entry.dataset_id == existing_id
        # No SDK probe happened.
        marine.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_live_probe_no_match_raises_notfound_and_caches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio
        from unittest.mock import MagicMock

        from copernicus_mcp.errors import NotFoundError

        empty = tmp_path / "empty.json"
        empty.write_text("{}", encoding="utf-8")
        reg = IndexRegistry(bundled_path=empty)

        # marine.get(dry_run=True, regex=...) returns an object with .files = []
        marine = MagicMock()
        response = MagicMock()
        response.files = []
        marine.get.return_value = response
        store = MagicMock()
        store._fetch_index_bytes = MagicMock()
        lock = asyncio.Lock()

        with pytest.raises(NotFoundError):
            await reg.discover("unknown_id", marine=marine, lock=lock, store=store)

        # Negative cache populated.
        assert reg.is_negatively_cached("unknown_id") is True

    @pytest.mark.asyncio
    async def test_live_probe_hit_inserts_entry_and_returns_dataframe(
        self, tmp_path: Path
    ) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        empty = tmp_path / "empty.json"
        empty.write_text("{}", encoding="utf-8")
        reg = IndexRegistry(bundled_path=empty)

        # Synthetic INSITU index file payload.
        fixture = (
            Path(__file__).parent.parent
            / "fixtures"
            / "cmems_indices"
            / "insitu_index_file_v3.txt"
        )
        raw_bytes = fixture.read_bytes()

        marine = MagicMock()
        match = MagicMock()
        match.file_path = "/tmp/path/index_file.txt"
        marine.get.return_value = MagicMock(files=[match])
        store = MagicMock()
        store._fetch_index_bytes = AsyncMock(return_value=raw_bytes)
        lock = asyncio.Lock()

        entry, df = await reg.discover(
            "new_dataset_id", marine=marine, lock=lock, store=store
        )
        assert entry.dataset_id == "new_dataset_id"
        assert entry.strategy == "file_based"
        assert entry.format_id == "insitu_index_file_v3"
        assert entry.index_filename == "index_file.txt"
        assert entry.discovered_via == "live_probe"
        # DataFrame has canonical columns.
        assert "file_path" in df.columns
        assert len(df) > 0
        # Overlay now contains the new entry.
        assert reg.get("new_dataset_id") is entry

    @pytest.mark.asyncio
    async def test_uri_precedence_latest_wins_when_history_absent(
        self, tmp_path: Path
    ) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        empty = tmp_path / "empty.json"
        empty.write_text("{}", encoding="utf-8")
        reg = IndexRegistry(bundled_path=empty)

        fixture = (
            Path(__file__).parent.parent
            / "fixtures"
            / "cmems_indices"
            / "insitu_index_file_v3.txt"
        )
        raw_bytes = fixture.read_bytes()

        marine = MagicMock()
        marine.get.return_value = MagicMock(
            files=[
                MagicMock(file_path="/tmp/path/index_monthly.txt"),
                MagicMock(file_path="/tmp/path/index_other.txt"),
                MagicMock(file_path="/tmp/path/index_latest.txt"),
            ]
        )
        store = MagicMock()
        store._fetch_index_bytes = AsyncMock(return_value=raw_bytes)
        entry, _ = await reg.discover(
            "ds", marine=marine, lock=asyncio.Lock(), store=store
        )
        assert entry.index_filename == "index_latest.txt"

    @pytest.mark.asyncio
    async def test_uri_precedence_monthly_wins_when_history_and_latest_absent(
        self, tmp_path: Path
    ) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        empty = tmp_path / "empty.json"
        empty.write_text("{}", encoding="utf-8")
        reg = IndexRegistry(bundled_path=empty)

        fixture = (
            Path(__file__).parent.parent
            / "fixtures"
            / "cmems_indices"
            / "insitu_index_file_v3.txt"
        )
        raw_bytes = fixture.read_bytes()

        marine = MagicMock()
        marine.get.return_value = MagicMock(
            files=[
                MagicMock(file_path="/tmp/path/index_other.txt"),
                MagicMock(file_path="/tmp/path/index_monthly.txt"),
            ]
        )
        store = MagicMock()
        store._fetch_index_bytes = AsyncMock(return_value=raw_bytes)
        entry, _ = await reg.discover(
            "ds", marine=marine, lock=asyncio.Lock(), store=store
        )
        assert entry.index_filename == "index_monthly.txt"

    @pytest.mark.asyncio
    async def test_uri_precedence_falls_through_to_first_when_no_marker_matches(
        self, tmp_path: Path
    ) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        empty = tmp_path / "empty.json"
        empty.write_text("{}", encoding="utf-8")
        reg = IndexRegistry(bundled_path=empty)

        fixture = (
            Path(__file__).parent.parent
            / "fixtures"
            / "cmems_indices"
            / "insitu_index_file_v3.txt"
        )
        raw_bytes = fixture.read_bytes()

        marine = MagicMock()
        marine.get.return_value = MagicMock(
            files=[
                MagicMock(file_path="/tmp/path/index_first.txt"),
                MagicMock(file_path="/tmp/path/index_second.txt"),
            ]
        )
        store = MagicMock()
        store._fetch_index_bytes = AsyncMock(return_value=raw_bytes)
        entry, _ = await reg.discover(
            "ds", marine=marine, lock=asyncio.Lock(), store=store
        )
        assert entry.index_filename == "index_first.txt"

    @pytest.mark.asyncio
    async def test_uri_precedence_prefers_history_then_latest_then_monthly(
        self, tmp_path: Path
    ) -> None:
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        empty = tmp_path / "empty.json"
        empty.write_text("{}", encoding="utf-8")
        reg = IndexRegistry(bundled_path=empty)

        fixture = (
            Path(__file__).parent.parent
            / "fixtures"
            / "cmems_indices"
            / "insitu_index_file_v3.txt"
        )
        raw_bytes = fixture.read_bytes()

        marine = MagicMock()
        files = [
            MagicMock(file_path="/tmp/path/index_monthly.txt"),
            MagicMock(file_path="/tmp/path/index_latest.txt"),
            MagicMock(file_path="/tmp/path/index_history.txt"),
            MagicMock(file_path="/tmp/path/index_other.txt"),
        ]
        marine.get.return_value = MagicMock(files=files)
        store = MagicMock()
        store._fetch_index_bytes = AsyncMock(return_value=raw_bytes)
        lock = asyncio.Lock()

        entry, _ = await reg.discover(
            "new_dataset_id", marine=marine, lock=lock, store=store
        )
        assert entry.index_filename == "index_history.txt"
