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
    estimated_size_bytes: int,
    threshold_bytes: int,
    source: str,
) -> ConfirmationRequired:
    """Build a size-threshold confirmation per research §11.6.2."""
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
        "estimated_size_gb": round(estimated_size_bytes / 1_000_000_000, 3),
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
    return ConfirmationRequired(payload)
