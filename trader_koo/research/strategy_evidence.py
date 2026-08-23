"""Fail-closed strategy evidence state from the latest audited baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_SNAPSHOT_PATH = Path(__file__).with_name("strategy_evidence_20260822.json")
_ELIGIBLE_STATUS = "eligible_for_human_promotion_review"
_MIN_OBSERVATIONS = 120
_MIN_TRADED_SIGNAL_DATES = 20
_MIN_EFFECTIVE_BLOCKS = 12.0
_PROMOTION_RETURN_BASIS = "split_adjusted_total_return_net_of_costs"


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
            "verified": False,
            "href": None,
        },
    }


def strategy_evidence_state() -> dict[str, Any]:
    """Load the checked-in audit snapshot, returning an ineligible state on error."""
    try:
        manifest = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _unavailable_state(f"strategy_evidence_snapshot_unavailable:{type(exc).__name__}")

    if not isinstance(manifest, dict):
        return _unavailable_state("strategy_evidence_manifest_invalid")
    artifact_path = _sibling_json_path(manifest.get("artifact_file"))
    input_path = _sibling_json_path(manifest.get("input_manifest_file"))
    if artifact_path is None or input_path is None:
        return _unavailable_state("strategy_evidence_manifest_invalid")

    try:
        artifact_bytes = artifact_path.read_bytes()
        input_bytes = input_path.read_bytes()
        payload = json.loads(artifact_bytes)
        input_manifest = json.loads(input_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        return _unavailable_state(f"strategy_evidence_provenance_unavailable:{type(exc).__name__}")
    artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
    input_hash = hashlib.sha256(input_bytes).hexdigest()
    if (
        artifact_hash != manifest.get("artifact_sha256")
        or input_hash != manifest.get("input_manifest_sha256")
    ):
        return _unavailable_state("strategy_evidence_provenance_hash_mismatch")
    if not isinstance(input_manifest, dict):
        return _unavailable_state("strategy_evidence_input_manifest_invalid")

    required = {
        "readiness_status",
        "readiness_reasons",
        "observation_count",
        "traded_signal_date_count",
        "effective_non_overlapping_block_count",
        "consumed_window",
        "causal_validity",
        "return_basis",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        return _unavailable_state("strategy_evidence_snapshot_invalid")

    state = payload
    upstream = manifest.get("upstream_audit")
    upstream = upstream if isinstance(upstream, dict) else {}
    state["provenance"] = {
        "artifact_name": artifact_path.name,
        "artifact_sha256": artifact_hash,
        "input_hash_sha256": input_hash,
        "artifact_spec_hash_sha256": upstream.get("artifact_spec_hash_sha256"),
        "upstream_artifact_name": upstream.get("artifact_name"),
        "upstream_artifact_sha256": upstream.get("artifact_sha256"),
        "upstream_input_hash_sha256": upstream.get("input_hash_sha256"),
        "verified": True,
        "href": f"/api/research/strategy-evidence/{artifact_hash}/inputs/{input_hash}",
    }

    state["decision_eligible"] = evidence_allows_action(state)
    return state


def evidence_allows_action(state: dict[str, Any] | None) -> bool:
    """Apply the complete promotion gate; assertions alone never authorize action."""
    if not isinstance(state, dict):
        return False
    causal = state.get("causal_validity")
    consumed = state.get("consumed_window")
    provenance = state.get("provenance")
    readiness_reasons = state.get("readiness_reasons")
    return bool(
        state.get("schema_version") == "1.0"
        and state.get("readiness_status") == _ELIGIBLE_STATUS
        and state.get("lifecycle_stage") == "promotion_review"
        and state.get("decision_eligible") is True
        and _at_least(state.get("observation_count"), _MIN_OBSERVATIONS)
        and _at_least(state.get("traded_signal_date_count"), _MIN_TRADED_SIGNAL_DATES)
        and _at_least(
            state.get("effective_non_overlapping_block_count"), _MIN_EFFECTIVE_BLOCKS
        )
        and isinstance(readiness_reasons, list)
        and not readiness_reasons
        and isinstance(causal, dict)
        and causal.get("valid") is True
        and isinstance(causal.get("reasons"), list)
        and not causal["reasons"]
        and isinstance(consumed, dict)
        and consumed.get("consumed") is True
        and consumed.get("reusable_for_policy_selection") is True
        and state.get("return_basis") == _PROMOTION_RETURN_BASIS
        and isinstance(provenance, dict)
        and provenance.get("verified") is True
        and _is_sha256(provenance.get("artifact_sha256"))
        and _is_sha256(provenance.get("input_hash_sha256"))
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


def _at_least(value: Any, minimum: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and value >= minimum
    )


def _sibling_json_path(value: Any) -> Path | None:
    if (
        not isinstance(value, str)
        or not value.endswith(".json")
        or Path(value).name != value
    ):
        return None
    return _SNAPSHOT_PATH.with_name(value)
