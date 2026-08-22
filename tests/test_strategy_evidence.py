"""Contract tests for the shared fail-closed strategy evidence state."""

from __future__ import annotations

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
