"""CDS size calibration — request-shape signature (T-CDS-EST2-003/004).

A *signature* is a stable string identifying the **shape** of a request — the
non-multiplicative fields that materially determine bytes-per-cost-unit while
the cost itself does not (decision 8). Two requests with the same signature but
different calendar ranges should calibrate the same per-unit byte rate.

The multiplicative axes (``year/month/day/time/date/area``) are deliberately
excluded — they live in the cost. ``CalibrationLookup`` (T-CDS-EST2-004) keys
local observations and the bundled seed off this signature.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any

_EWMA_ALPHA = 0.3

# Non-multiplicative discriminators whose value changes bytes-per-unit. Kept in
# a frozenset; the signature sorts keys so order here is irrelevant.
_SIGNATURE_FIELDS: frozenset[str] = frozenset(
    {
        "temporal_resolution",
        "frequency",
        "daily_statistic",
        "product_type",
        "processing_level",
        "processinglevel",  # SST snapshot's actual key
        "sensor_and_algorithm",
        "sensor_on_satellite",
        "horizontal_resolution",
        "experiment",
        "model",  # CMIP6 models differ in native grid/resolution
        "system_version",  # GloFAS/EFAS product generation
        "hydrological_model",
        "time_aggregation",
        "type_of_sensor",
        "type_of_record",
        "version",
        "data_format",
        "download_format",
        "format",
    }
)

_LEVEL_FIELDS: tuple[str, ...] = ("pressure_level", "model_level", "level")


def _level_count(inputs: dict[str, Any]) -> int:
    """Integer count of requested levels (0 if none). Level *depth* scales
    bytes but not cost, so it belongs in the signature, not the cost."""
    for field in _LEVEL_FIELDS:
        value = inputs.get(field)
        if isinstance(value, list):
            return len(value)
        if value is not None:
            return 1
    return 0


def _variable_token(inputs: dict[str, Any]) -> str:
    """The single variable value, or ``"__multi"`` when more than one is
    requested (multi-variable requests price differently per unit)."""
    value = inputs.get("variable")
    if isinstance(value, list):
        if len(value) == 1:
            return str(value[0])
        return "__multi"
    if value is not None:
        return str(value)
    return "__none"


def signature(inputs: dict[str, Any]) -> str:
    """Return the canonical signature string for a request's ``inputs`` dict.

    Deterministic: sorted keys, compact separators. Scalar and single-element-
    list values normalise identically (``"x"`` and ``["x"]`` → same token) so a
    scalar MARS-style request and its list form share a signature.
    """
    shape: dict[str, Any] = {}
    for field in _SIGNATURE_FIELDS:
        value = inputs.get(field)
        if value is None:
            continue
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        shape[field] = value
    shape["__level_count"] = _level_count(inputs)
    shape["__variable"] = _variable_token(inputs)
    return json.dumps(shape, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class CalibrationResult:
    """Resolved bytes-per-cost-unit for a request shape, the number of *local*
    observations behind it (0 ⇒ seed-only / dataset-median), and the ``source`` of
    the figure: ``"local"`` (your own downloads), ``"seed"`` (bundled seed,
    signature-matched), or ``"median"`` (weak cross-signature fallback)."""

    bytes_per_unit: float
    n_obs: int
    source: str = "local"


def _eligible(obs: Mapping[str, Any]) -> bool:
    """An observation is usable only with a positive cost and area (decision 9
    excludes NULL-cost restart rows and degenerate-area rows)."""
    cost = obs.get("cost_units")
    area = obs.get("area_fraction", 1.0)
    return cost is not None and cost > 0 and area is not None and area > 0


def _bytes_per_unit(obs: Mapping[str, Any]) -> float:
    return float(obs["size_bytes"]) / float(obs["cost_units"]) / float(obs["area_fraction"])


def _ewma(values: Sequence[float]) -> float:
    """EWMA seeded with the first value (oldest), α=0.3."""
    acc = values[0]
    for v in values[1:]:
        acc = _EWMA_ALPHA * v + (1 - _EWMA_ALPHA) * acc
    return acc


class CalibrationLookup:
    """Resolves bytes-per-cost-unit for a ``(dataset_id, signature)`` from local
    observations blended with the bundled seed (decision 9). Pure: built once
    in the backend with the dataset's observations + seed, then handed to the
    estimator.
    """

    def __init__(
        self,
        *,
        observations: Sequence[Mapping[str, Any]],
        seed: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> None:
        # Keep only eligible rows; pre-sort oldest-first for the EWMA.
        self._obs = sorted(
            (o for o in observations if _eligible(o)),
            key=lambda o: o.get("observed_at", ""),
        )
        self._seed = seed

    def resolve(self, dataset_id: str, signature: str) -> CalibrationResult | None:
        sig_obs = [o for o in self._obs if o.get("signature") == signature]
        n_local = len(sig_obs)
        local_bpu = _ewma([_bytes_per_unit(o) for o in sig_obs]) if n_local else None

        seed_entry = self._seed.get((dataset_id, signature))
        seed_bpu = float(seed_entry["bytes_per_unit"]) if seed_entry else None
        n_seed = int(seed_entry["n_obs"]) if seed_entry else 0

        if n_local >= 3:
            # Local truth dominates once it exists (stale seed can't drown it).
            return CalibrationResult(bytes_per_unit=local_bpu, n_obs=n_local)  # type: ignore[arg-type]
        if n_local == 0:
            if seed_bpu is not None:
                return CalibrationResult(bytes_per_unit=seed_bpu, n_obs=0, source="seed")
            return self._dataset_median()
        # 1 or 2 local observations: blend with the seed if present.
        if seed_bpu is not None:
            weight = min(n_seed, 3)
            blended = (local_bpu * n_local + seed_bpu * weight) / (n_local + weight)  # type: ignore[operator]
            return CalibrationResult(bytes_per_unit=blended, n_obs=n_local)
        return CalibrationResult(bytes_per_unit=local_bpu, n_obs=n_local)  # type: ignore[arg-type]

    def _dataset_median(self) -> CalibrationResult | None:
        """Median bytes-per-unit across all signatures, only with ≥3 eligible
        observations (decision 9 fallback for an unseen signature). ``n_obs=0``:
        this is NOT signature-matched local data, so the estimator treats it as
        "size unknown" (v2 honesty) — the median is kept only as a weak prior for
        future blending, never shown as a confident number."""
        if len(self._obs) < 3:
            return None
        bpu = median(_bytes_per_unit(o) for o in self._obs)
        return CalibrationResult(bytes_per_unit=bpu, n_obs=0, source="median")


_SEED_PATH = __import__("pathlib").Path(__file__).parent / "_data" / "cds_calibration_seed.json"


def load_seed(
    path: Any = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load the bundled calibration seed into a ``(dataset_id, signature) ->
    {bytes_per_unit, n_obs}`` map. Missing or corrupt seed ⇒ ``{}`` + a warning
    (run seedless). ``path`` overrides the bundled file (tests / dev script)."""
    from pathlib import Path

    seed_path = Path(path) if path is not None else _SEED_PATH
    try:
        raw = json.loads(seed_path.read_text())
        entries = raw.get("entries", [])
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        try:
            key = (str(entry["dataset_id"]), str(entry["signature"]))
            out[key] = {
                "bytes_per_unit": float(entry["bytes_per_unit"]),
                "n_obs": int(entry["n_obs"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return out
