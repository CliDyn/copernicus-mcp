from __future__ import annotations

import pytest


def test_build_size_confirmation_shape() -> None:
    from copernicus_mcp.workflow.confirmation import (
        ConfirmationRequired,
        build_size_confirmation,
    )

    exc = build_size_confirmation(
        tool_name="marine_subset_dataset",
        backend="cmems",
        estimated_size_bytes=2_400_000_000,
        threshold_bytes=1_000_000_000,
        source="config.budget.cmems_per_request_size_warning_gb",
    )
    assert isinstance(exc, ConfirmationRequired)
    p = exc.payload
    assert p["confirmation_required"] is True
    assert p["reason"] == "estimated_size_threshold_exceeded"
    assert p["estimated_size_gb"] == pytest.approx(2.4, rel=0.01)
    assert p["estimated_cost"]["estimated_size_bytes"] == 2_400_000_000
    assert p["threshold"]["bytes"] == 1_000_000_000
    assert p["threshold"]["source"] == "config.budget.cmems_per_request_size_warning_gb"
    # codex round 1 CX-M1: hint must be MCP-schema friendly — the tools
    # expose ``confirmed`` as a top-level Pydantic field (extra="forbid"),
    # so an LLM passing ``{options: {confirmed: true}}`` literally would
    # be rejected. Hint says ``confirmed=true`` without the ``options.``
    # prefix; the orchestrator-direct CLI path still works because it
    # builds ``options={"confirmed": True}`` itself, not from this hint.
    assert "confirmed=true" in p["next_action"]
    assert "options.confirmed" not in p["next_action"]
    assert p["context"]["tool_name"] == "marine_subset_dataset"
    assert p["context"]["backend"] == "cmems"


def test_raising_preserves_payload() -> None:
    from copernicus_mcp.workflow.confirmation import (
        ConfirmationRequired,
        build_size_confirmation,
    )

    with pytest.raises(ConfirmationRequired) as exc_info:
        raise build_size_confirmation(
            tool_name="marine_subset_dataset",
            backend="cmems",
            estimated_size_bytes=10,
            threshold_bytes=5,
            source="test",
        )
    assert exc_info.value.payload["estimated_size_gb"] >= 0


def test_payload_is_independent_copy() -> None:
    from copernicus_mcp.workflow.confirmation import build_size_confirmation

    a = build_size_confirmation(
        tool_name="x", backend="cmems",
        estimated_size_bytes=1, threshold_bytes=1, source="t",
    )
    a.payload["mutated"] = True
    b = build_size_confirmation(
        tool_name="x", backend="cmems",
        estimated_size_bytes=1, threshold_bytes=1, source="t",
    )
    assert "mutated" not in b.payload
