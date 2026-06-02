"""Unit tests for ``copernicus_mcp.backends.cmems._index_store``.

The store is the runtime-facing surface: caller asks "give me the parsed
index for X", store fetches via the SDK + writes a Parquet cache + reads
the cache on subsequent calls.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from copernicus_mcp.backends.cmems._index_registry import IndexRegistry
from copernicus_mcp.backends.cmems._index_store import IndexStore
from copernicus_mcp.errors import NotFoundError

_INSITU_FIXTURE = (
    Path(__file__).parent.parent
    / "fixtures"
    / "cmems_indices"
    / "insitu_index_file_v3.txt"
)


def _empty_registry(tmp_path: Path) -> IndexRegistry:
    p = tmp_path / "empty_reg.json"
    p.write_text("{}", encoding="utf-8")
    return IndexRegistry(bundled_path=p)


def _mock_marine_with_index_payload(
    payload: bytes, *, expected_filename: str = "index_file.txt"
) -> MagicMock:
    """Build a marine mock whose ``get`` writes ``payload`` to a synthetic
    INSITU subtree under ``output_directory`` (matches real SDK shape)."""
    marine = MagicMock()

    def fake_get(**kwargs: object) -> object:
        out = Path(str(kwargs["output_directory"]))
        # SDK writes files under a product/dataset subtree; the store
        # uses rglob to find them.
        target_dir = out / "PRODUCT" / "DATASET"
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / expected_filename).write_bytes(payload)
        return MagicMock(files=[MagicMock(file_path=str(target_dir / expected_filename))])

    marine.get.side_effect = fake_get
    return marine


class TestFetchIndexBytes:
    @pytest.mark.asyncio
    async def test_returns_payload_from_sdk_download(self, tmp_path: Path) -> None:
        payload = _INSITU_FIXTURE.read_bytes()
        marine = _mock_marine_with_index_payload(payload)
        store = IndexStore(
            registry=_empty_registry(tmp_path),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
        )
        raw = await store._fetch_index_bytes("any_dataset", "index_file.txt")
        assert raw == payload

    @pytest.mark.asyncio
    async def test_temporary_directory_is_cleaned_up(self, tmp_path: Path) -> None:
        # The TemporaryDirectory must not leave content behind.
        payload = b"data"
        marine = _mock_marine_with_index_payload(payload, expected_filename="canyon.txt")
        store = IndexStore(
            registry=_empty_registry(tmp_path),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
        )
        before = set(Path(tmp_path).iterdir()) if tmp_path.exists() else set()
        await store._fetch_index_bytes("d", "canyon.txt")
        after = set(Path(tmp_path).iterdir()) if tmp_path.exists() else set()
        # Only the cache directory (if created) and our test fixtures are
        # left; no scratch tmpdir remains.
        new_paths = after - before
        for p in new_paths:
            assert "tmp" not in p.name.lower(), f"orphaned scratch dir: {p}"

    @pytest.mark.asyncio
    async def test_raises_notfound_if_sdk_returns_no_file(self, tmp_path: Path) -> None:
        # cr round-1 M1: "SDK wrote nothing" means the dataset/file does
        # not exist; the canonical class is NotFoundError.
        marine = MagicMock()

        def fake_get(**kwargs: object) -> object:
            return MagicMock(files=[])

        marine.get.side_effect = fake_get
        store = IndexStore(
            registry=_empty_registry(tmp_path),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
        )
        with pytest.raises(NotFoundError):
            await store._fetch_index_bytes("d", "index_file.txt")

    @pytest.mark.asyncio
    async def test_sdk_exception_is_wrapped_when_wrap_provided(
        self, tmp_path: Path
    ) -> None:
        from copernicus_mcp.errors import ValidationError

        marine = MagicMock()
        marine.get.side_effect = RuntimeError("simulated SDK failure")

        def wrap(marine_mod: object, exc: BaseException, op: str) -> Exception:
            return ValidationError(f"wrapped {op}: {exc}")

        store = IndexStore(
            registry=_empty_registry(tmp_path),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
            wrap_exception=wrap,
        )
        with pytest.raises(ValidationError, match="wrapped list_files"):
            await store._fetch_index_bytes("d", "index_file.txt")

    @pytest.mark.asyncio
    async def test_cancellation_propagates_unwrapped(self, tmp_path: Path) -> None:
        # the project conventions invariant 3: never catch / wrap CancelledError.
        marine = MagicMock()
        marine.get.side_effect = asyncio.CancelledError()

        def wrap(marine_mod: object, exc: BaseException, op: str) -> Exception:
            # If this gets called, we've broken invariant 3.
            raise AssertionError("CancelledError was wrapped")

        store = IndexStore(
            registry=_empty_registry(tmp_path),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
            wrap_exception=wrap,
        )
        with pytest.raises(asyncio.CancelledError):
            await store._fetch_index_bytes("d", "index_file.txt")

    @pytest.mark.asyncio
    async def test_uses_auth_adapter_when_present(self, tmp_path: Path) -> None:
        payload = b"x"
        marine = _mock_marine_with_index_payload(payload, expected_filename="i.txt")
        adapter = MagicMock()
        adapter.get_username_password.return_value = ("user@example.com", "secret")
        store = IndexStore(
            registry=_empty_registry(tmp_path),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
            auth_adapter=adapter,
        )
        await store._fetch_index_bytes("d", "i.txt")
        # The SDK was called with credentials.
        _, kwargs = marine.get.call_args
        assert kwargs["username"] == "user@example.com"
        assert kwargs["password"] == "secret"

    @pytest.mark.asyncio
    async def test_skips_auth_when_adapter_is_none(self, tmp_path: Path) -> None:
        payload = b"x"
        marine = _mock_marine_with_index_payload(payload, expected_filename="i.txt")
        store = IndexStore(
            registry=_empty_registry(tmp_path),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
            auth_adapter=None,
        )
        await store._fetch_index_bytes("d", "i.txt")
        _, kwargs = marine.get.call_args
        assert "username" not in kwargs
        assert "password" not in kwargs


class TestLoad:
    """``IndexStore.load`` — the runtime entry point."""

    @pytest.mark.asyncio
    async def test_registered_file_based_cache_miss_fetches_parses_writes(
        self, tmp_path: Path
    ) -> None:
        # Bundled registry has cmems_obs-ins_glo_bgc-car_my_socat-obs_irr
        # (file_based, index_file.txt, insitu_index_file_v3). Cache miss
        # triggers SDK fetch + Parquet write + "fresh" mode.
        dataset_id = "cmems_obs-ins_glo_bgc-car_my_socat-obs_irr"
        payload = _INSITU_FIXTURE.read_bytes()
        marine = _mock_marine_with_index_payload(payload)
        store = IndexStore(
            registry=IndexRegistry(),  # bundled
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
        )
        df, mode = await store.load(dataset_id)
        assert mode == "fresh"
        assert len(df) > 0
        assert list(df.columns)[0] == "file_path"
        # Parquet was written.
        cache_file = tmp_path / "cache" / "marine_indices" / f"{dataset_id}.parquet"
        assert cache_file.exists()

    @pytest.mark.asyncio
    async def test_registered_file_based_cache_hit_reads_parquet_no_sdk(
        self, tmp_path: Path
    ) -> None:
        dataset_id = "cmems_obs-ins_glo_bgc-car_my_socat-obs_irr"
        payload = _INSITU_FIXTURE.read_bytes()
        marine = _mock_marine_with_index_payload(payload)
        store = IndexStore(
            registry=IndexRegistry(),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
        )
        # First call: fresh.
        await store.load(dataset_id)
        marine.get.reset_mock()
        # Second call: cache hit, no SDK call.
        df, mode = await store.load(dataset_id)
        assert mode == "offline"
        assert len(df) > 0
        marine.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_registered_path_based_dispatches_to_dry_run_listing(
        self, tmp_path: Path
    ) -> None:
        # CORA — path_based dataset bundled in the registry.
        dataset_id = "cmems_obs-ins_glo_phy-temp-sal_my_cora_irr"

        marine = MagicMock()

        def fake_get(**kwargs: object) -> object:
            # dry_run=True listing — returns paths only, no download.
            assert kwargs.get("dry_run") is True
            return MagicMock(
                files=[
                    MagicMock(file_path="mediterrane/2010/CO_DMQCGL01_20100101_PR_CT.nc"),
                    MagicMock(file_path="mediterrane/2011/CO_DMQCGL01_20110315_PR_PF.nc"),
                ]
            )

        marine.get.side_effect = fake_get
        store = IndexStore(
            registry=IndexRegistry(),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
        )
        df, mode = await store.load(dataset_id)
        assert mode == "fresh"
        assert len(df) == 2
        # No file_list arg for path_based — dry_run only.
        _, kwargs = marine.get.call_args
        assert "file_list" not in kwargs

    @pytest.mark.asyncio
    async def test_unregistered_dataset_calls_discover(self, tmp_path: Path) -> None:
        # An unknown dataset_id goes through registry.discover (which
        # probes the SDK).
        payload = _INSITU_FIXTURE.read_bytes()
        marine = MagicMock()

        def fake_get(**kwargs: object) -> object:
            if kwargs.get("dry_run"):
                # Discovery probe: returns matched URIs.
                return MagicMock(
                    files=[MagicMock(file_path="/tmp/PRODUCT/DATASET/index_file.txt")]
                )
            # Real fetch.
            out = Path(str(kwargs["output_directory"]))
            sub = out / "PRODUCT" / "DATASET"
            sub.mkdir(parents=True, exist_ok=True)
            (sub / "index_file.txt").write_bytes(payload)
            return MagicMock(files=[])

        marine.get.side_effect = fake_get
        store = IndexStore(
            registry=_empty_registry(tmp_path),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
        )
        df, mode = await store.load("brand_new_dataset")
        assert mode == "fresh"
        assert len(df) > 0
        # Discover populated the overlay.
        assert store._registry.get("brand_new_dataset") is not None

    @pytest.mark.asyncio
    async def test_unregistered_dataset_with_no_index_raises_notfound(
        self, tmp_path: Path
    ) -> None:
        marine = MagicMock()
        marine.get.return_value = MagicMock(files=[])
        store = IndexStore(
            registry=_empty_registry(tmp_path),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
        )
        with pytest.raises(NotFoundError):
            await store.load("ghost_dataset")


class TestRefresh:
    @pytest.mark.asyncio
    async def test_refresh_unregistered_dataset_raises_without_sdk_call(
        self, tmp_path: Path
    ) -> None:
        marine = MagicMock()
        store = IndexStore(
            registry=_empty_registry(tmp_path),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
        )
        with pytest.raises(NotFoundError):
            await store.refresh("not_in_registry")
        marine.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_refresh_overwrites_existing_cache(self, tmp_path: Path) -> None:
        dataset_id = "cmems_obs-ins_glo_bgc-car_my_socat-obs_irr"
        payload = _INSITU_FIXTURE.read_bytes()
        marine = _mock_marine_with_index_payload(payload)
        store = IndexStore(
            registry=IndexRegistry(),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
        )
        await store.load(dataset_id)
        cache_file = tmp_path / "cache" / "marine_indices" / f"{dataset_id}.parquet"
        mtime_before = cache_file.stat().st_mtime_ns

        # Force-fetch — should overwrite.
        import time
        time.sleep(0.01)
        await store.refresh(dataset_id)
        mtime_after = cache_file.stat().st_mtime_ns
        assert mtime_after > mtime_before


class TestFetchedAtAndList:
    @pytest.mark.asyncio
    async def test_fetched_at_returns_none_before_any_load(self, tmp_path: Path) -> None:
        store = IndexStore(
            registry=IndexRegistry(),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: MagicMock(),
        )
        assert await store.fetched_at("anything") is None

    @pytest.mark.asyncio
    async def test_fetched_at_returns_timestamp_after_load(self, tmp_path: Path) -> None:
        dataset_id = "cmems_obs-ins_glo_bgc-car_my_socat-obs_irr"
        payload = _INSITU_FIXTURE.read_bytes()
        marine = _mock_marine_with_index_payload(payload)
        store = IndexStore(
            registry=IndexRegistry(),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
        )
        await store.load(dataset_id)
        ts = await store.fetched_at(dataset_id)
        assert ts is not None
        from datetime import UTC, datetime, timedelta
        now = datetime.now(UTC)
        assert now - ts < timedelta(seconds=10)

    def test_list_cached_datasets_returns_empty_before_load(
        self, tmp_path: Path
    ) -> None:
        store = IndexStore(
            registry=IndexRegistry(),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: MagicMock(),
        )
        assert store.list_cached_datasets() == []

    @pytest.mark.asyncio
    async def test_list_cached_datasets_enumerates_after_load(
        self, tmp_path: Path
    ) -> None:
        dataset_id = "cmems_obs-ins_glo_bgc-car_my_socat-obs_irr"
        payload = _INSITU_FIXTURE.read_bytes()
        marine = _mock_marine_with_index_payload(payload)
        store = IndexStore(
            registry=IndexRegistry(),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
        )
        await store.load(dataset_id)
        assert dataset_id in store.list_cached_datasets()


class TestRoundOneRegressions:
    """cr round-1 HIGH regressions."""

    @pytest.mark.asyncio
    async def test_H1_concurrent_unregistered_loads_share_single_fetch(
        self, tmp_path: Path
    ) -> None:
        # 5 concurrent load() calls on an unregistered dataset must share
        # ONE SDK round-trip, not five. Decision #1 + acceptance line 735.
        payload = _INSITU_FIXTURE.read_bytes()
        sdk_call_count = 0

        marine = MagicMock()

        def fake_get(**kwargs: object) -> object:
            nonlocal sdk_call_count
            sdk_call_count += 1
            if kwargs.get("dry_run"):
                return MagicMock(
                    files=[MagicMock(file_path="/tmp/PRODUCT/DATASET/index_file.txt")]
                )
            out = Path(str(kwargs["output_directory"]))
            sub = out / "PRODUCT" / "DATASET"
            sub.mkdir(parents=True, exist_ok=True)
            (sub / "index_file.txt").write_bytes(payload)
            return MagicMock(files=[])

        marine.get.side_effect = fake_get
        store = IndexStore(
            registry=_empty_registry(tmp_path),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
        )
        # 5 concurrent first-touches on the same unregistered dataset.
        results = await asyncio.gather(
            *(store.load("concurrent_new_dataset") for _ in range(5))
        )
        # Each returns a non-empty DataFrame; the SDK was probed at most
        # twice (one discover probe + one fetch). Without serialisation,
        # we'd see ~10 calls (5 probes + 5 fetches).
        assert all(len(df) > 0 for df, _ in results)
        assert sdk_call_count <= 2, f"expected ≤2 SDK calls, got {sdk_call_count}"

    @pytest.mark.asyncio
    async def test_H3_negative_cache_short_circuits_second_load(
        self, tmp_path: Path
    ) -> None:
        # First load on a ghost id raises NotFoundError and populates
        # the negative cache. Second load on the same id must NOT call
        # the SDK at all (decision #14 — auth-storm prevention).
        marine = MagicMock()
        marine.get.return_value = MagicMock(files=[])
        store = IndexStore(
            registry=_empty_registry(tmp_path),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
        )
        with pytest.raises(NotFoundError):
            await store.load("ghost_$1")
        marine.get.reset_mock()
        with pytest.raises(NotFoundError):
            await store.load("ghost_$1")
        marine.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_HR2_1_probe_hits_but_download_empty_populates_negative_cache(
        self, tmp_path: Path
    ) -> None:
        # cr round-2 H-R2-1: round-1's M1 introduced a new "no index"
        # outcome (probe matched a URI, but the actual download wrote
        # nothing → NotFoundError). The H3 negative-cache invariant
        # requires every "no index" outcome to populate the cache so
        # repeat-probes are short-circuited.
        marine = MagicMock()

        def fake_get(**kwargs: object) -> object:
            if kwargs.get("dry_run"):
                # Probe hits — returns a candidate URI.
                return MagicMock(
                    files=[MagicMock(file_path="/tmp/PRODUCT/DATASET/index_file.txt")]
                )
            # But the actual download writes nothing.
            return MagicMock(files=[])

        marine.get.side_effect = fake_get
        store = IndexStore(
            registry=_empty_registry(tmp_path),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
        )
        with pytest.raises(NotFoundError):
            await store.load("probe_hits_download_empty")
        marine.get.reset_mock()
        # Second call must short-circuit via the negative cache.
        with pytest.raises(NotFoundError):
            await store.load("probe_hits_download_empty")
        marine.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_MR2_1_default_path_sanitises_credentials(
        self, tmp_path: Path
    ) -> None:
        # cr round-2 M-R2-1: when wrap_exception is None (production
        # wiring before Task 004), the built-in Sanitiser-based fallback
        # must scrub credentials from raw SDK exception messages.
        from copernicus_mcp.errors import BackendError

        marine = MagicMock()
        marine.get.side_effect = RuntimeError(
            "login failed: user=alice password=hunter2 (host=example.com)"
        )
        store = IndexStore(
            registry=_empty_registry(tmp_path),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
            # wrap_exception intentionally not provided.
        )
        with pytest.raises(BackendError) as exc_info:
            await store._fetch_index_bytes("d", "index_file.txt")
        msg = str(exc_info.value)
        assert "hunter2" not in msg
        assert "REDACTED" in msg

    @pytest.mark.asyncio
    async def test_M1_no_file_after_sdk_call_raises_notfound_not_backend(
        self, tmp_path: Path
    ) -> None:
        # NotFoundError steers the user to "check dataset_id";
        # BackendError says "SDK is broken". The right canonical class
        # for "SDK accepted call but wrote nothing" is NotFoundError.
        marine = MagicMock()
        marine.get.return_value = MagicMock(files=[])
        store = IndexStore(
            registry=_empty_registry(tmp_path),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
        )
        with pytest.raises(NotFoundError):
            await store._fetch_index_bytes("d", "index_file.txt")


class TestAtomicWrite:
    @pytest.mark.asyncio
    async def test_mid_write_failure_leaves_previous_cache_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate a crash in the atomic write step — the previous
        # parquet file should remain unchanged.
        dataset_id = "cmems_obs-ins_glo_bgc-car_my_socat-obs_irr"
        payload = _INSITU_FIXTURE.read_bytes()
        marine = _mock_marine_with_index_payload(payload)
        store = IndexStore(
            registry=IndexRegistry(),
            cache_directory=tmp_path / "cache",
            marine_loader=lambda: marine,
        )
        await store.load(dataset_id)
        cache_file = tmp_path / "cache" / "marine_indices" / f"{dataset_id}.parquet"
        bytes_before = cache_file.read_bytes()

        from copernicus_mcp.backends.cmems import _index_store as mod

        def boom(*a: object, **kw: object) -> None:
            raise OSError("simulated write failure")

        monkeypatch.setattr(mod, "_atomic_write_bytes", boom)
        with pytest.raises(OSError, match="simulated write failure"):
            await store.refresh(dataset_id)

        # Previous cache survives.
        assert cache_file.read_bytes() == bytes_before
