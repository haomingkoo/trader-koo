"""Fail-closed strategy evidence state from the latest audited baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SNAPSHOT_PATH = Path(__file__).with_name("strategy_evidence_20260822.json")
_ELIGIBLE_STATUS = "eligible_for_human_promotion_review"


def _unavailable_state(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "snapshot_asof": None,
        "lifecycle_stage": "descriptive",
        "readiness_status": "evidence_unavailable",
        "readiness_reasons": [reason],
        "observation_count": 0,
        "traded_signal_date_count": 0,
        "effective_non_overlapping_block_count": 0.0,
        "closed_trade_count": 0,
        "consumed_window": {
            "consumed": True,
            "reusable_for_policy_selection": False,
            "status": "unknown_fail_closed",
        },
        "causal_validity": {"valid": False, "reasons": [reason]},
        "return_basis": "unknown",
        "decision_eligible": False,
        "provenance": {
            "artifact_name": None,
            "artifact_sha256": None,
            "input_hash_sha256": None,
            "artifact_spec_hash_sha256": None,
            "href": None,
        },
    }


def strategy_evidence_state() -> dict[str, Any]:
    """Load the checked-in audit snapshot, returning an ineligible state on error."""
    try:
        payload = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _unavailable_state(f"strategy_evidence_snapshot_unavailable:{type(exc).__name__}")

    required = {
        "readiness_status",
        "readiness_reasons",
        "observation_count",
        "traded_signal_date_count",
        "effective_non_overlapping_block_count",
        "consumed_window",
        "causal_validity",
        "return_basis",
        "provenance",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        return _unavailable_state("strategy_evidence_snapshot_invalid")

    state = payload
    provenance = state.get("provenance")
    if not isinstance(provenance, dict):
        return _unavailable_state("strategy_evidence_provenance_invalid")
    artifact_hash = provenance.get("artifact_sha256")
    input_hash = provenance.get("input_hash_sha256")
    if not _is_sha256(artifact_hash) or not _is_sha256(input_hash):
        return _unavailable_state("strategy_evidence_hash_invalid")
    provenance["href"] = (
        f"/api/research/strategy-evidence/{artifact_hash}/inputs/{input_hash}"
    )

    causal = state.get("causal_validity")
    is_causal = isinstance(causal, dict) and causal.get("valid") is True
    state["decision_eligible"] = bool(
        state.get("readiness_status") == _ELIGIBLE_STATUS
        and state.get("lifecycle_stage") == "promotion_review"
        and is_causal
        and state.get("decision_eligible") is True
    )
    return state


def evidence_allows_action(state: dict[str, Any] | None) -> bool:
    """Return true only for an explicitly eligible, causally valid snapshot."""
    if not isinstance(state, dict):
        return False
    causal = state.get("causal_validity")
    return bool(
        state.get("readiness_status") == _ELIGIBLE_STATUS
        and state.get("lifecycle_stage") == "promotion_review"
        and state.get("decision_eligible") is True
        and isinstance(causal, dict)
        and causal.get("valid") is True
    )


def evidence_snapshot_by_hash(artifact_hash: str, input_hash: str) -> dict[str, Any] | None:
    """Resolve only the exact immutable snapshot named by both hashes."""
    state = strategy_evidence_state()
    provenance = state.get("provenance") or {}
    if (
        artifact_hash == provenance.get("artifact_sha256")
        and input_hash == provenance.get("input_hash_sha256")
    ):
        return state
    return None


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value)
