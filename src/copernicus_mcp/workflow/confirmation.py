"""Confirmation flow primitive (research §11.6.2).

The backend raises ``ConfirmationRequired`` rather than returning a flag dict
so the success path stays clean. The CLI catches and prompts; the MCP server
catches and surfaces the payload as the tool result.
"""

from __future__ import annotations

import copy
from typing import Any


class ConfirmationRequired(Exception):
    """Backend signal that a destructive/expensive operation needs user OK."""

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(payload.get("reason", "confirmation_required"))
        # Defensive copy so callers can't mutate stored payload.
        self.payload: dict[str, Any] = copy.deepcopy(payload)


def build_size_confirmation(
    *,
    tool_name: str,
    backend: str,
    estimated_size_bytes: int | None,
    threshold_bytes: int,
    source: str,
) -> ConfirmationRequired:
    """Build a size-threshold confirmation per research §11.6.2.

    ``estimated_size_bytes`` may be ``None`` (T-CDS-EST2: a whole-file product
    whose size is unknowable from request shape) — the ``estimated_size_gb``
    field is then omitted rather than computed from a missing number. CMEMS
    callers always pass an int, so their payload is unchanged.
    """
    payload: dict[str, Any] = {
        "confirmation_required": True,
        "reason": "estimated_size_threshold_exceeded",
        "estimated_cost": {
            "type": "size",
            "estimated_size_bytes": estimated_size_bytes,
        },
        "threshold": {
            "bytes": threshold_bytes,
            "source": source,
        },
        # codex CX-M1: MCP tools expose ``confirmed`` as a top-level
        # Pydantic field (``extra="forbid"`` rejects an ``options`` sub-
        # dict). The CLI-orchestrator-direct path builds
        # ``options={"confirmed": True}`` itself, so the wording stays
        # unprefixed and works for both callers.
        "next_action": (
            f"call {tool_name} with confirmed=true to proceed"
        ),
        "context": {
            "tool_name": tool_name,
            "backend": backend,
        },
    }
    if estimated_size_bytes is not None:
        payload["estimated_size_gb"] = round(estimated_size_bytes / 1_000_000_000, 3)
    return ConfirmationRequired(payload)
