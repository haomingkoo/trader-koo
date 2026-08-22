"""Contract tests for the shared fail-closed strategy evidence state."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

import trader_koo.research.strategy_evidence as strategy_evidence_module
from trader_koo.research.strategy_evidence import (
    evidence_allows_action,
    evidence_snapshot_by_hash,
    strategy_evidence_state,
)
from trader_koo.paper_trade.family_edge import generate_edge_feedback


def test_current_audit_snapshot_is_descriptive_and_ineligible():
    state = strategy_evidence_state()

    assert state["readiness_status"] == "insufficient_history"
    assert state["lifecycle_stage"] == "descriptive"
    assert state["observation_count"] == 20
    assert state["traded_signal_date_count"] == 4
    assert state["effective_non_overlapping_block_count"] == 2.0
    assert state["consumed_window"]["consumed"] is True
    assert state["consumed_window"]["reusable_for_policy_selection"] is False
    assert state["causal_validity"]["valid"] is False
    assert state["return_basis"] == "split_adjusted_price_return_only_dividends_omitted"
    assert state["decision_eligible"] is False
    assert evidence_allows_action(state) is False


def test_snapshot_resolves_only_when_both_exact_hashes_match():
    state = strategy_evidence_state()
    provenance = state["provenance"]

    assert evidence_snapshot_by_hash(
        provenance["artifact_sha256"], provenance["input_hash_sha256"]
    ) == state
    assert evidence_snapshot_by_hash("0" * 64, provenance["input_hash_sha256"]) is None
    assert evidence_snapshot_by_hash(provenance["artifact_sha256"], "0" * 64) is None

    research_dir = Path(__file__).parents[1] / "trader_koo" / "research"
    artifact = research_dir / provenance["artifact_name"]
    inputs = research_dir / "strategy_evidence_inputs_20260822.json"
    assert artifact.is_file()
    assert inputs.is_file()
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == provenance["artifact_sha256"]
    assert hashlib.sha256(inputs.read_bytes()).hexdigest() == provenance["input_hash_sha256"]
    assert provenance["verified"] is True


def test_unknown_or_partial_state_fails_closed():
    assert evidence_allows_action(None) is False
    assert evidence_allows_action({}) is False
    assert evidence_allows_action(
        {
            "readiness_status": "eligible_for_human_promotion_review",
            "decision_eligible": True,
            "causal_validity": {"valid": False},
        }
    ) is False


def _eligible_state() -> dict:
    return {
        "schema_version": "1.0",
        "lifecycle_stage": "promotion_review",
        "readiness_status": "eligible_for_human_promotion_review",
        "readiness_reasons": [],
        "observation_count": 120,
        "traded_signal_date_count": 20,
        "effective_non_overlapping_block_count": 12.0,
        "consumed_window": {
            "consumed": True,
            "reusable_for_policy_selection": True,
            "status": "fresh_prospective_window_available_for_promotion_review",
        },
        "causal_validity": {"valid": True, "reasons": []},
        "return_basis": "split_adjusted_total_return_net_of_costs",
        "decision_eligible": True,
        "provenance": {
            "verified": True,
            "artifact_sha256": "a" * 64,
            "input_hash_sha256": "b" * 64,
        },
    }


def test_complete_promotion_schema_can_reach_human_review():
    assert evidence_allows_action(_eligible_state()) is True


def test_missing_packaged_artifact_fails_closed(monkeypatch, tmp_path):
    manifest = tmp_path / "strategy_evidence_20260822.json"
    manifest.write_text(
        '{"artifact_file":"missing.json","input_manifest_file":"inputs.json"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(strategy_evidence_module, "_SNAPSHOT_PATH", manifest)

    state = strategy_evidence_state()

    assert state["readiness_status"] == "evidence_unavailable"
    assert state["decision_eligible"] is False
    assert state["provenance"]["verified"] is False
    assert state["provenance"]["href"] is None


def test_tampered_packaged_artifact_fails_closed(monkeypatch, tmp_path):
    (tmp_path / "artifact.json").write_text('{"readiness_status":"eligible"}')
    (tmp_path / "inputs.json").write_text("{}")
    manifest = tmp_path / "strategy_evidence_20260822.json"
    manifest.write_text(
        """{
          "artifact_file":"artifact.json",
          "artifact_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
          "input_manifest_file":"inputs.json",
          "input_manifest_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        }""",
        encoding="utf-8",
    )
    monkeypatch.setattr(strategy_evidence_module, "_SNAPSHOT_PATH", manifest)

    state = strategy_evidence_state()

    assert state["readiness_status"] == "evidence_unavailable"
    assert state["readiness_reasons"] == [
        "strategy_evidence_provenance_hash_mismatch"
    ]
    assert state["decision_eligible"] is False


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    [
        (("observation_count",), 0),
        (("traded_signal_date_count",), 0),
        (("effective_non_overlapping_block_count",), 0),
        (("readiness_reasons",), ["unresolved"]),
        (("consumed_window", "reusable_for_policy_selection"), False),
        (("causal_validity", "reasons"), ["survivorship_bias"]),
        (("return_basis",), "unknown"),
        (("provenance", "verified"), False),
        (("provenance", "artifact_sha256"), "a" * 63),
    ],
)
def test_each_missing_promotion_gate_fails_closed(path, invalid_value):
    state = copy.deepcopy(_eligible_state())
    target = state
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid_value

    assert evidence_allows_action(state) is False


def test_five_trade_family_stays_descriptive_when_overall_state_is_eligible():
    feedback = generate_edge_feedback(
        [
            {
                "setup_family": "bullish_reversal",
                "direction": "long",
                "edge_label": "positive",
                "win_rate_pct": 60.0,
                "avg_pnl_pct": 1.2,
                "avg_r_multiple": 0.3,
                "trade_count": 5,
                "window_days": 180,
            }
        ],
        [
            {
                "regime": "bull_normal",
                "avg_pnl_pct": 1.2,
                "trade_count": 5,
            }
        ],
        actionable=True,
    )
    rendered = " ".join(
        f"{item['title']} {item['detail']} {item['action']}" for item in feedback
    ).lower()

    assert "research only" in rendered
    assert "size up" not in rendered
    assert "priority allocation" not in rendered
    assert "shows edge" not in rendered
