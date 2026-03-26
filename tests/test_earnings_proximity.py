"""Tests for earnings proximity flagging on setup rows."""
from __future__ import annotations

from typing import Any

import pytest

from trader_koo.report.setup_scoring import annotate_earnings_proximity


def _make_setup(ticker: str, risk_note: str = "none") -> dict[str, Any]:
    return {
        "ticker": ticker,
        "score": 75.0,
        "setup_tier": "B",
        "signal_bias": "bullish",
        "close": 150.0,
        "risk_note": risk_note,
    }


def _make_earnings_catalysts(
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {"ok": True, "rows": rows or []}


def _make_earnings_row(
    ticker: str,
    days_until: int,
    earnings_date: str = "2026-04-01",
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "earnings_date": earnings_date,
        "days_until": days_until,
        "earnings_session": "BMO",
    }


class TestAnnotateEarningsProximity:
    def test_flags_ticker_with_earnings_within_5d(self):
        setups = [_make_setup("AAPL")]
        catalysts = _make_earnings_catalysts(
            [_make_earnings_row("AAPL", 3, "2026-04-01")]
        )

        flagged = annotate_earnings_proximity(setups, catalysts)

        assert flagged == 1
        assert setups[0]["earnings_within_5d"] is True
        assert setups[0]["earnings_date"] == "2026-04-01"
        assert setups[0]["days_to_earnings"] == 3

    def test_does_not_flag_ticker_beyond_5d(self):
        setups = [_make_setup("MSFT")]
        catalysts = _make_earnings_catalysts(
            [_make_earnings_row("MSFT", 8, "2026-04-10")]
        )

        flagged = annotate_earnings_proximity(setups, catalysts)

        assert flagged == 0
        assert setups[0]["earnings_within_5d"] is False
        assert setups[0]["days_to_earnings"] == 8

    def test_does_not_flag_ticker_with_no_earnings(self):
        setups = [_make_setup("GOOG")]
        catalysts = _make_earnings_catalysts(
            [_make_earnings_row("AAPL", 3)]
        )

        flagged = annotate_earnings_proximity(setups, catalysts)

        assert flagged == 0
        assert setups[0]["earnings_within_5d"] is False
        assert setups[0]["earnings_date"] is None
        assert setups[0]["days_to_earnings"] is None

    def test_appends_earnings_to_risk_note(self):
        setups = [_make_setup("AAPL", risk_note="high volatility")]
        catalysts = _make_earnings_catalysts(
            [_make_earnings_row("AAPL", 2)]
        )

        annotate_earnings_proximity(setups, catalysts)

        assert "earnings within 2d" in setups[0]["risk_note"]
        assert "high volatility" in setups[0]["risk_note"]

    def test_replaces_none_risk_note(self):
        setups = [_make_setup("AAPL", risk_note="none")]
        catalysts = _make_earnings_catalysts(
            [_make_earnings_row("AAPL", 1)]
        )

        annotate_earnings_proximity(setups, catalysts)

        assert setups[0]["risk_note"] == "earnings within 1d"

    def test_does_not_duplicate_earnings_in_risk_note(self):
        setups = [_make_setup("AAPL", risk_note="earnings event pending")]
        catalysts = _make_earnings_catalysts(
            [_make_earnings_row("AAPL", 2)]
        )

        annotate_earnings_proximity(setups, catalysts)

        assert setups[0]["risk_note"] == "earnings event pending"
        assert setups[0]["earnings_within_5d"] is True

    def test_picks_nearest_earnings_date(self):
        setups = [_make_setup("AAPL")]
        catalysts = _make_earnings_catalysts([
            _make_earnings_row("AAPL", 7, "2026-04-10"),
            _make_earnings_row("AAPL", 3, "2026-04-01"),
        ])

        annotate_earnings_proximity(setups, catalysts)

        assert setups[0]["earnings_within_5d"] is True
        assert setups[0]["days_to_earnings"] == 3
        assert setups[0]["earnings_date"] == "2026-04-01"

    def test_multiple_tickers_mixed(self):
        setups = [
            _make_setup("AAPL"),
            _make_setup("MSFT"),
            _make_setup("GOOG"),
        ]
        catalysts = _make_earnings_catalysts([
            _make_earnings_row("AAPL", 2),
            _make_earnings_row("GOOG", 4),
        ])

        flagged = annotate_earnings_proximity(setups, catalysts)

        assert flagged == 2
        assert setups[0]["earnings_within_5d"] is True
        assert setups[1]["earnings_within_5d"] is False
        assert setups[2]["earnings_within_5d"] is True

    def test_empty_setups_returns_zero(self):
        flagged = annotate_earnings_proximity([], _make_earnings_catalysts())

        assert flagged == 0

    def test_empty_catalysts_sets_defaults(self):
        setups = [_make_setup("AAPL")]

        flagged = annotate_earnings_proximity(setups, {})

        assert flagged == 0
        assert setups[0]["earnings_within_5d"] is False
        assert setups[0]["earnings_date"] is None
        assert setups[0]["days_to_earnings"] is None

    def test_earnings_on_day_zero_flags(self):
        setups = [_make_setup("AAPL")]
        catalysts = _make_earnings_catalysts(
            [_make_earnings_row("AAPL", 0, "2026-03-26")]
        )

        flagged = annotate_earnings_proximity(setups, catalysts)

        assert flagged == 1
        assert setups[0]["earnings_within_5d"] is True
        assert setups[0]["days_to_earnings"] == 0

    def test_earnings_exactly_5d_flags(self):
        setups = [_make_setup("AAPL")]
        catalysts = _make_earnings_catalysts(
            [_make_earnings_row("AAPL", 5)]
        )

        flagged = annotate_earnings_proximity(setups, catalysts)

        assert flagged == 1
        assert setups[0]["earnings_within_5d"] is True

    def test_earnings_at_6d_does_not_flag(self):
        setups = [_make_setup("AAPL")]
        catalysts = _make_earnings_catalysts(
            [_make_earnings_row("AAPL", 6)]
        )

        flagged = annotate_earnings_proximity(setups, catalysts)

        assert flagged == 0
        assert setups[0]["earnings_within_5d"] is False

    def test_case_insensitive_ticker_matching(self):
        setups = [{"ticker": "aapl", "risk_note": "none"}]
        catalysts = _make_earnings_catalysts(
            [_make_earnings_row("AAPL", 3)]
        )

        flagged = annotate_earnings_proximity(setups, catalysts)

        assert flagged == 1
        assert setups[0]["earnings_within_5d"] is True


class TestEarningsHaircutIntegration:
    """Verify that earnings risk_note triggers the position haircut in decision.py."""

    def test_earnings_risk_note_triggers_haircut(self):
        from trader_koo.paper_trade.config import PaperTradeConfig
        from trader_koo.paper_trade.decision import compute_position_plan, evaluate_setup_for_paper_trade

        config = PaperTradeConfig(
            bot_version="v1.0.0",
            min_tier="B",
            min_score=55.0,
            max_open=10,
            expiry_days=10,
            stop_atr_mult=1.5,
            default_stop_pct=3.0,
            qualifying_tiers=frozenset({"A", "B"}),
            qualifying_actionability=frozenset({"higher-probability", "conditional"}),
            qualifying_directions=frozenset({"long", "short"}),
            tier_rank={"A": 0, "B": 1, "C": 2},
            decision_version="test-v1",
            debate_caution_agreement=40.0,
            high_vol_atr_pct=6.0,
            min_reward_r_multiple=1.5,
            min_position_pct=2.0,
            max_position_pct=12.0,
            tier_a_position_pct=8.0,
            tier_b_position_pct=6.0,
            tier_c_position_pct=4.0,
            caution_position_scale=0.75,
            high_vol_position_scale=0.70,
            earnings_position_scale=0.50,
        )

        row_no_earnings = {
            "ticker": "AAPL",
            "score": 80.0,
            "setup_tier": "A",
            "actionability": "higher-probability",
            "signal_bias": "bullish",
            "setup_family": "Bullish Breakout",
            "close": 150.0,
            "atr_pct_14": 2.5,
            "support_level": 140.0,
            "resistance_level": 165.0,
            "risk_note": "none",
            "debate_agreement_score": 80.0,
        }

        row_with_earnings = dict(row_no_earnings)
        row_with_earnings["risk_note"] = "earnings within 3d"

        eval_no = evaluate_setup_for_paper_trade(row_no_earnings, config=config)
        eval_with = evaluate_setup_for_paper_trade(row_with_earnings, config=config)

        from trader_koo.paper_trade.decision import compute_stop_and_target

        levels_no = compute_stop_and_target(row_no_earnings, "long", config=config)
        levels_with = compute_stop_and_target(row_with_earnings, "long", config=config)

        plan_no = compute_position_plan(
            row_no_earnings, eval_no, levels_no, config=config
        )
        plan_with = compute_position_plan(
            row_with_earnings, eval_with, levels_with, config=config
        )

        assert plan_with["position_size_pct"] < plan_no["position_size_pct"]
        assert "event-risk haircut" in str(plan_with["sizing_summary"])
        assert "event-risk haircut" not in str(plan_no["sizing_summary"])

    def test_earnings_flag_adds_risk_note_for_haircut(self):
        """End-to-end: annotate_earnings_proximity adds risk_note that
        causes the position haircut in compute_position_plan."""
        setups = [_make_setup("AAPL", risk_note="none")]
        catalysts = _make_earnings_catalysts(
            [_make_earnings_row("AAPL", 2)]
        )

        annotate_earnings_proximity(setups, catalysts)

        assert "earnings" in setups[0]["risk_note"].lower()
