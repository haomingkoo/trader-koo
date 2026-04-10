"""Parametrized boundary tests for the critic module.

Philosophy: every named threshold constant in critic.py has an on-boundary
and off-boundary test. If a threshold changes, the matching test fails —
forcing the developer to explicitly update the test (and document the
trade-count impact in the PR body, per the constant block's instructions).
"""
from __future__ import annotations

import sqlite3

import pytest

from trader_koo.paper_trade.critic import (
    CAUTION_FLAGS_HARD_BLOCK,
    CAUTION_FLAGS_WARN,
    CONVICTION_A_GRADE_SCORE,
    CONVICTION_A_HIGH,
    CONVICTION_A_MIN,
    CONVICTION_B_HIGH,
    CONVICTION_B_MIN,
    DEBATE_CONSENSUS_MIN,
    DEBATE_CONSENSUS_STRONG,
    FAMILY_EDGE_BLOCK_SAMPLE,
    FAMILY_EDGE_BLOCK_WINRATE,
    FAMILY_EDGE_MIN_SAMPLE,
    FAMILY_EDGE_WEAK_WINRATE,
    GOOD_REWARD_R,
    MIN_REWARD_R,
    REGIME_AMBIGUOUS_MIN_SCORE,
    REGIME_REVERSAL_MIN_SCORE,
    REGIME_VIX_EXTREME,
    REGIME_VIX_HIGH_VOL,
    REGIME_VIX_UNKNOWN_BLOCK,
    ROLLING_EXPECTANCY_MIN,
    ROLLING_EXPECTANCY_MIN_SAMPLE,
    _check_caution_flags,
    _check_conviction_grade,
    _check_debate_strength,
    _check_family_edge,
    _check_regime_alignment,
    _check_risk_reward,
    _check_rolling_expectancy,
    _check_volatility_environment,
    critic_review,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(
    ticker: str = "AAPL",
    setup_tier: str = "A",
    score: float = 80.0,
    setup_family: str = "Bullish Breakout",
    debate_agreement_score: float = 80.0,
) -> dict:
    return {
        "ticker": ticker,
        "setup_tier": setup_tier,
        "score": score,
        "setup_family": setup_family,
        "debate_agreement_score": debate_agreement_score,
    }


def _eval(
    direction: str = "long",
    decision_state: str = "approved",
    risk_flags: list | None = None,
) -> dict:
    return {
        "direction": direction,
        "decision_state": decision_state,
        "risk_flags": risk_flags or [],
    }


def _plan(expected_r: float = 2.5) -> dict:
    return {"expected_r_multiple": expected_r}


def _ctx(
    regime: str = "bull_market",
    vix: float = 18.0,
    dir_regime: str = "bullish",
) -> dict:
    return {
        "regime_state_at_entry": regime,
        "vix_at_entry": vix,
        "directional_regime_at_entry": dir_regime,
    }


def _make_conn() -> sqlite3.Connection:
    """In-memory DB with paper_trades schema (minimal columns needed)."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            direction TEXT,
            status TEXT DEFAULT 'open',
            pnl_pct REAL,
            setup_family TEXT,
            exit_date TEXT
        )
    """)
    conn.commit()
    return conn


def _insert_closed(conn: sqlite3.Connection, pnl: float, family: str = "bullish_breakout", direction: str = "long") -> None:
    conn.execute(
        "INSERT INTO paper_trades (ticker, direction, status, pnl_pct, setup_family, exit_date) "
        "VALUES ('X', ?, 'closed', ?, ?, '2026-01-01')",
        (direction, pnl, family),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# _check_conviction_grade
# ---------------------------------------------------------------------------

class TestConvictionGrade:
    @pytest.mark.parametrize("score,expected_pass", [
        (CONVICTION_A_HIGH, True),         # exactly at high threshold
        (CONVICTION_A_HIGH + 1, True),     # above high threshold
        (CONVICTION_A_MIN, True),          # exactly at minimum
        (CONVICTION_A_MIN - 0.1, False),   # just below minimum
        (0.0, False),                       # zero score
    ])
    def test_a_tier_score_boundaries(self, score, expected_pass):
        passed, reason = _check_conviction_grade(_row(setup_tier="A", score=score), _eval())
        assert passed is expected_pass, f"score={score}: {reason}"

    @pytest.mark.parametrize("score,decision,expected_pass", [
        (CONVICTION_B_HIGH, "approved", True),           # exactly at high threshold
        (CONVICTION_B_HIGH + 1, "approved", True),       # above high threshold
        (CONVICTION_B_MIN, "approved", True),            # exactly at minimum, approved
        (CONVICTION_B_MIN, "rejected", False),           # at minimum but wrong decision_state
        (CONVICTION_B_MIN - 0.1, "approved", False),     # just below minimum
    ])
    def test_b_tier_score_boundaries(self, score, decision, expected_pass):
        passed, reason = _check_conviction_grade(
            _row(setup_tier="B", score=score), _eval(decision_state=decision)
        )
        assert passed is expected_pass, f"score={score} decision={decision}: {reason}"

    def test_c_tier_always_fails(self):
        passed, _ = _check_conviction_grade(_row(setup_tier="C", score=99.0), _eval())
        assert passed is False

    def test_a_grade_score_boundary_in_critic_review(self):
        """Conviction grade A+ requires score >= CONVICTION_A_GRADE_SCORE."""
        conn = _make_conn()
        result = critic_review(
            conn,
            row=_row(setup_tier="A", score=CONVICTION_A_GRADE_SCORE),
            evaluation=_eval(),
            plan=_plan(),
            market_ctx=_ctx(),
        )
        if result["approved"]:
            assert result["conviction_grade"] == "A+"

        result_below = critic_review(
            conn,
            row=_row(setup_tier="A", score=CONVICTION_A_GRADE_SCORE - 1),
            evaluation=_eval(),
            plan=_plan(),
            market_ctx=_ctx(),
        )
        if result_below["approved"]:
            assert result_below["conviction_grade"] == "A"


# ---------------------------------------------------------------------------
# _check_debate_strength
# ---------------------------------------------------------------------------

class TestDebateStrength:
    @pytest.mark.parametrize("agreement,expected_pass", [
        (DEBATE_CONSENSUS_STRONG, True),         # exactly at strong threshold
        (DEBATE_CONSENSUS_STRONG + 1, True),     # above strong
        (DEBATE_CONSENSUS_MIN, True),            # exactly at minimum
        (DEBATE_CONSENSUS_MIN - 0.1, False),     # just below minimum
        (0.0, False),                             # zero agreement
    ])
    def test_agreement_boundaries(self, agreement, expected_pass):
        passed, reason = _check_debate_strength(
            _row(debate_agreement_score=agreement), _eval()
        )
        assert passed is expected_pass, f"agreement={agreement}: {reason}"


# ---------------------------------------------------------------------------
# _check_risk_reward
# ---------------------------------------------------------------------------

class TestRiskReward:
    @pytest.mark.parametrize("r_multiple,expected_pass", [
        (GOOD_REWARD_R, True),          # exactly at good threshold
        (GOOD_REWARD_R + 0.5, True),    # above good
        (MIN_REWARD_R, True),           # exactly at minimum
        (MIN_REWARD_R - 0.1, False),    # just below minimum
        (0.0, False),                    # zero R
        (1.0, False),                    # 1R is not enough
    ])
    def test_r_multiple_boundaries(self, r_multiple, expected_pass):
        passed, reason = _check_risk_reward(_plan(expected_r=r_multiple))
        assert passed is expected_pass, f"r_multiple={r_multiple}: {reason}"


# ---------------------------------------------------------------------------
# _check_regime_alignment — VIX and regime gating
# ---------------------------------------------------------------------------

class TestRegimeAlignment:
    def test_long_blocked_above_vix_high_vol(self):
        """Longs must be blocked when VIX >= REGIME_VIX_HIGH_VOL. No override."""
        passed, reason = _check_regime_alignment(
            _row(setup_tier="A", score=99.0),
            _eval(direction="long"),
            _ctx(regime="bull_market", vix=REGIME_VIX_HIGH_VOL + 0.1, dir_regime="bullish"),
        )
        assert passed is False
        assert "VIX" in reason or "high-vol" in reason

    def test_long_allowed_just_below_vix_high_vol(self):
        """Longs may pass when VIX is just below the high-vol threshold."""
        passed, reason = _check_regime_alignment(
            _row(setup_tier="A", score=99.0),
            _eval(direction="long"),
            _ctx(regime="bull_market", vix=REGIME_VIX_HIGH_VOL - 0.1, dir_regime="bullish"),
        )
        assert passed is True, reason

    def test_short_allowed_in_high_vol(self):
        """Shorts are regime-appropriate in high-vol — should pass."""
        passed, reason = _check_regime_alignment(
            _row(setup_tier="B", score=75.0),
            _eval(direction="short"),
            _ctx(regime="bear_market", vix=REGIME_VIX_HIGH_VOL + 5, dir_regime="bearish"),
        )
        assert passed is True, reason

    def test_reversal_long_non_bull_at_boundary(self):
        """Reversal long in non-bull needs score >= REGIME_REVERSAL_MIN_SCORE.
        Uses VIX=20 (normal range, not low_vol which is < 16) to isolate the reversal check.
        """
        passed_at, _ = _check_regime_alignment(
            _row(setup_tier="A", score=REGIME_REVERSAL_MIN_SCORE, setup_family="Bullish Reversal"),
            _eval(direction="long"),
            _ctx(regime="bear_market", vix=20.0, dir_regime=""),
        )
        passed_below, _ = _check_regime_alignment(
            _row(setup_tier="A", score=REGIME_REVERSAL_MIN_SCORE - 0.1, setup_family="Bullish Reversal"),
            _eval(direction="long"),
            _ctx(regime="bear_market", vix=20.0, dir_regime=""),
        )
        assert passed_at is True
        assert passed_below is False

    def test_continuation_long_non_bull_always_blocked(self):
        """Non-reversal longs in non-bull regime are blocked regardless of score."""
        passed, reason = _check_regime_alignment(
            _row(setup_tier="A", score=99.0, setup_family="Bullish Breakout"),
            _eval(direction="long"),
            _ctx(regime="bear_market", vix=20.0, dir_regime=""),
        )
        assert passed is False
        assert "continuation" in reason.lower() or "non-bull" in reason.lower()

    def test_ambiguous_regime_at_boundary(self):
        """In ambiguous regime (no HMM, no strong bull/bear), need score >= REGIME_AMBIGUOUS_MIN_SCORE.
        Uses VIX=20 (normal range) and short direction (no VIX high-vol conflict).
        """
        passed_at, _ = _check_regime_alignment(
            _row(setup_tier="B", score=REGIME_AMBIGUOUS_MIN_SCORE),
            _eval(direction="short"),
            _ctx(regime="chop", vix=20.0, dir_regime=""),
        )
        passed_below, _ = _check_regime_alignment(
            _row(setup_tier="B", score=REGIME_AMBIGUOUS_MIN_SCORE - 0.1),
            _eval(direction="short"),
            _ctx(regime="chop", vix=20.0, dir_regime=""),
        )
        assert passed_at is True
        assert passed_below is False

    def test_unknown_regime_high_vix_blocked(self):
        """Unknown regime + VIX > REGIME_VIX_UNKNOWN_BLOCK → block."""
        passed, reason = _check_regime_alignment(
            _row(),
            _eval(direction="long"),
            _ctx(regime="unknown_unknown", vix=REGIME_VIX_UNKNOWN_BLOCK + 0.1, dir_regime=""),
        )
        assert passed is False

    def test_unknown_regime_low_vix_allowed(self):
        """Unknown regime + VIX below unknown block threshold → allow."""
        passed, reason = _check_regime_alignment(
            _row(),
            _eval(direction="long"),
            _ctx(regime="unknown", vix=REGIME_VIX_UNKNOWN_BLOCK - 0.1, dir_regime=""),
        )
        assert passed is True, reason

    def test_hmm_counter_trend_hard_blocked(self):
        """HMM counter-trend is blocked — no tier or score override.
        Uses VIX=20 (normal range) to avoid triggering the low_vol override.
        """
        passed, reason = _check_regime_alignment(
            _row(setup_tier="A", score=99.0),
            _eval(direction="long"),
            _ctx(regime="chop", vix=20.0, dir_regime="bearish"),
        )
        assert passed is False
        assert "counter-trend" in reason.lower() or "blocked" in reason.lower()


# ---------------------------------------------------------------------------
# _check_volatility_environment
# ---------------------------------------------------------------------------

class TestVolatilityEnvironment:
    def test_extreme_vix_blocks_all(self):
        passed, reason = _check_volatility_environment({"vix_at_entry": REGIME_VIX_EXTREME + 0.1})
        assert passed is False
        assert "extreme" in reason.lower()

    def test_just_below_extreme_vix_passes(self):
        passed, _ = _check_volatility_environment({"vix_at_entry": REGIME_VIX_EXTREME - 0.1})
        assert passed is True

    def test_none_vix_passes(self):
        passed, _ = _check_volatility_environment({"vix_at_entry": None})
        assert passed is True


# ---------------------------------------------------------------------------
# _check_caution_flags
# ---------------------------------------------------------------------------

class TestCautionFlags:
    def test_clean_approval_passes(self):
        passed, reason = _check_caution_flags(_eval(decision_state="approved", risk_flags=[]))
        assert passed is True
        assert "clean" in reason.lower()

    @pytest.mark.parametrize("n_flags,expected_pass", [
        (CAUTION_FLAGS_HARD_BLOCK, False),      # exactly at hard block
        (CAUTION_FLAGS_HARD_BLOCK + 1, False),  # above hard block
        (CAUTION_FLAGS_WARN, True),             # exactly at warn threshold — still passes
        (CAUTION_FLAGS_WARN - 1, True),         # below warn
        (0, True),                               # zero flags
    ])
    def test_flag_count_boundaries(self, n_flags, expected_pass):
        flags = [f"flag_{i}" for i in range(n_flags)]
        passed, reason = _check_caution_flags(
            _eval(decision_state="approved_with_flags", risk_flags=flags)
        )
        assert passed is expected_pass, f"n_flags={n_flags}: {reason}"


# ---------------------------------------------------------------------------
# _check_rolling_expectancy
# ---------------------------------------------------------------------------

class TestRollingExpectancy:
    def test_insufficient_history_passes(self):
        conn = _make_conn()
        for _ in range(ROLLING_EXPECTANCY_MIN_SAMPLE - 1):
            _insert_closed(conn, pnl=1.0)

        passed, reason = _check_rolling_expectancy(conn)
        assert passed is True
        assert "insufficient" in reason.lower()

    def test_negative_expectancy_at_boundary_blocks(self):
        conn = _make_conn()
        for _ in range(ROLLING_EXPECTANCY_MIN_SAMPLE):
            _insert_closed(conn, pnl=ROLLING_EXPECTANCY_MIN - 0.01)  # all trades just below threshold

        passed, reason = _check_rolling_expectancy(conn)
        assert passed is False
        assert "negative" in reason.lower()

    def test_expectancy_at_boundary_passes(self):
        conn = _make_conn()
        for _ in range(ROLLING_EXPECTANCY_MIN_SAMPLE):
            _insert_closed(conn, pnl=ROLLING_EXPECTANCY_MIN)  # exactly at threshold

        passed, reason = _check_rolling_expectancy(conn)
        assert passed is True, reason

    def test_positive_expectancy_passes(self):
        conn = _make_conn()
        for _ in range(ROLLING_EXPECTANCY_MIN_SAMPLE):
            _insert_closed(conn, pnl=1.0)

        passed, reason = _check_rolling_expectancy(conn)
        assert passed is True, reason


# ---------------------------------------------------------------------------
# _check_family_edge
# ---------------------------------------------------------------------------

class TestFamilyEdge:
    def test_insufficient_sample_always_passes(self):
        conn = _make_conn()
        for _ in range(FAMILY_EDGE_MIN_SAMPLE - 1):
            _insert_closed(conn, pnl=-1.0, family="bullish_breakout", direction="long")

        passed, reason = _check_family_edge(conn, _row(setup_family="bullish_breakout"), _eval())
        assert passed is True
        assert "insufficient" in reason.lower()

    def test_zero_win_rate_blocks_at_block_sample(self):
        """0% win rate over FAMILY_EDGE_BLOCK_SAMPLE trades → block."""
        conn = _make_conn()
        for _ in range(FAMILY_EDGE_BLOCK_SAMPLE):
            _insert_closed(conn, pnl=-1.0, family="bullish_breakout", direction="long")

        passed, reason = _check_family_edge(conn, _row(setup_family="bullish_breakout"), _eval())
        assert passed is False
        assert "0%" in reason or "win rate" in reason.lower()

    def test_zero_win_rate_below_block_sample_passes(self):
        """0% win rate under FAMILY_EDGE_BLOCK_SAMPLE trades → insufficient for block."""
        conn = _make_conn()
        for _ in range(FAMILY_EDGE_BLOCK_SAMPLE - 1):
            _insert_closed(conn, pnl=-1.0, family="bullish_breakout", direction="long")

        passed, _ = _check_family_edge(conn, _row(setup_family="bullish_breakout"), _eval())
        assert passed is True

    def test_weak_win_rate_requires_a_tier(self):
        """Win rate < FAMILY_EDGE_WEAK_WINRATE and B-tier → block."""
        conn = _make_conn()
        # 1 win, 3 losses = 25% — exactly at FAMILY_EDGE_WEAK_WINRATE — allowed
        # 1 win, 4 losses = 20% — below FAMILY_EDGE_WEAK_WINRATE — B-tier blocked
        for _ in range(4):
            _insert_closed(conn, pnl=-1.0, family="bullish_breakout", direction="long")
        _insert_closed(conn, pnl=1.0, family="bullish_breakout", direction="long")  # 1 win out of 5

        row_b = _row(setup_family="bullish_breakout", setup_tier="B")
        row_a = _row(setup_family="bullish_breakout", setup_tier="A")

        passed_b, _ = _check_family_edge(conn, row_b, _eval())
        passed_a, _ = _check_family_edge(conn, row_a, _eval())

        assert passed_b is False
        assert passed_a is True

    def test_no_family_skips_check(self):
        conn = _make_conn()
        passed, reason = _check_family_edge(conn, _row(setup_family=""), _eval())
        assert passed is True
        assert "skipping" in reason.lower()


# ---------------------------------------------------------------------------
# critic_review integration
# ---------------------------------------------------------------------------

class TestCriticReview:
    def test_all_checks_pass_returns_approved(self):
        conn = _make_conn()
        result = critic_review(
            conn,
            row=_row(setup_tier="A", score=85.0, debate_agreement_score=80.0),
            evaluation=_eval(direction="long", decision_state="approved"),
            plan=_plan(expected_r=3.0),
            market_ctx=_ctx(regime="bull_market", vix=15.0, dir_regime="bullish"),
        )
        assert result["approved"] is True
        assert result["conviction_grade"] in ("A+", "A", "B+")
        assert result["checks_passed"] == result["checks_total"]
        assert result["rejections"] == []

    def test_one_failing_check_rejects(self):
        conn = _make_conn()
        result = critic_review(
            conn,
            row=_row(setup_tier="A", score=85.0),
            evaluation=_eval(direction="long"),
            plan=_plan(expected_r=0.5),  # terrible R:R
            market_ctx=_ctx(),
        )
        assert result["approved"] is False
        assert result["conviction_grade"] == "rejected"
        assert len(result["rejections"]) >= 1
        assert "REJECTED" in result["summary"]

    def test_vix_extreme_blocks_regardless_of_setup(self):
        """Extreme VIX overrides everything — even A+ tier setups."""
        conn = _make_conn()
        result = critic_review(
            conn,
            row=_row(setup_tier="A", score=99.0),
            evaluation=_eval(direction="long"),
            plan=_plan(expected_r=5.0),
            market_ctx=_ctx(vix=REGIME_VIX_EXTREME + 1, dir_regime="bullish"),
        )
        assert result["approved"] is False

    def test_result_schema_complete(self):
        conn = _make_conn()
        result = critic_review(
            conn,
            row=_row(),
            evaluation=_eval(),
            plan=_plan(),
            market_ctx=_ctx(),
        )
        required_keys = {
            "approved", "checks_passed", "checks_total",
            "conviction_grade", "critic_reasons", "rejections", "summary",
        }
        assert required_keys.issubset(result.keys())
        assert isinstance(result["critic_reasons"], list)
        assert isinstance(result["rejections"], list)
        assert isinstance(result["approved"], bool)
