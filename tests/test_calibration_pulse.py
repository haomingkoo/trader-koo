"""Tests for the calibration pulse module."""
from __future__ import annotations

import sqlite3

import pytest

from trader_koo.report.calibration_pulse import (
    BLOCK_EXPECTANCY_THRESHOLD,
    BLOCK_HIT_RATE_THRESHOLD,
    MIN_COMBINED_FOR_BLOCK,
    MIN_EVAL_SAMPLE,
    MIN_PAPER_SAMPLE,
    SCORE_ADJ_MAX,
    SCORE_ADJ_MIN,
    _compute_block,
    _compute_score_adjustment,
    _combined_expectancy,
    build_telegram_message,
    ensure_calibration_schema,
    load_calibration_state,
    run_calibration_pulse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_calibration_schema(conn)
    conn.execute("""
        CREATE TABLE setup_call_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asof_date TEXT,
            setup_family TEXT,
            call_direction TEXT,
            status TEXT DEFAULT 'scored',
            direction_hit INTEGER,
            signed_return_pct REAL
        )
    """)
    conn.execute("""
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setup_family TEXT,
            direction TEXT,
            status TEXT DEFAULT 'closed',
            pnl_pct REAL,
            exit_date TEXT
        )
    """)
    conn.commit()
    return conn


def _seed_eval(conn: sqlite3.Connection, family: str, direction: str, returns: list[float]) -> None:
    for ret in returns:
        conn.execute(
            "INSERT INTO setup_call_evaluations (asof_date, setup_family, call_direction, "
            "status, direction_hit, signed_return_pct) VALUES ('2026-04-01', ?, ?, 'scored', ?, ?)",
            (family, direction, 1 if ret > 0 else 0, ret),
        )
    conn.commit()


def _seed_paper(conn: sqlite3.Connection, family: str, direction: str, returns: list[float]) -> None:
    for ret in returns:
        conn.execute(
            "INSERT INTO paper_trades (setup_family, direction, status, pnl_pct, exit_date) "
            "VALUES (?, ?, 'closed', ?, '2026-04-01')",
            (family, direction, ret),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# _combined_expectancy
# ---------------------------------------------------------------------------

class TestCombinedExpectancy:
    def test_paper_only(self):
        paper = {"sample": 10, "hit_rate_pct": 40.0, "expectancy_pct": -1.2}
        result = _combined_expectancy(None, paper, paper_weight=2.0)
        assert result["paper_sample"] == 10
        assert result["eval_sample"] == 0
        assert result["expectancy_pct"] is not None

    def test_eval_only(self):
        ev = {"sample": 20, "hit_rate_pct": 55.0, "expectancy_pct": 0.8}
        result = _combined_expectancy(ev, None, paper_weight=2.0)
        assert result["eval_sample"] == 20
        assert result["paper_sample"] == 0

    def test_paper_weighted_more(self):
        """Paper stat should dominate when both present (paper_weight=2x)."""
        ev = {"sample": 10, "hit_rate_pct": 60.0, "expectancy_pct": 1.0}
        paper = {"sample": 5, "hit_rate_pct": 30.0, "expectancy_pct": -2.0}
        combined = _combined_expectancy(ev, paper, paper_weight=2.0)
        # With ev=10 and paper effective=10 (5*2), weighted exp should be between -2 and 1
        assert combined["expectancy_pct"] is not None
        assert -2.0 <= combined["expectancy_pct"] <= 1.0

    def test_both_none_returns_zero_sample(self):
        result = _combined_expectancy(None, None, paper_weight=2.0)
        assert result["combined_sample"] == 0
        assert result["expectancy_pct"] is None


# ---------------------------------------------------------------------------
# _compute_score_adjustment
# ---------------------------------------------------------------------------

class TestScoreAdjustment:
    @pytest.mark.parametrize("exp,expected_sign", [
        (-3.0, -1),   # very negative → large negative adj
        (-1.0, -1),   # negative → negative adj
        (0.3, 1),     # slightly positive → positive adj
        (2.5, 1),     # very positive → max positive adj
    ])
    def test_sign_matches_expectancy(self, exp, expected_sign):
        combined = {"expectancy_pct": exp}
        adj = _compute_score_adjustment(combined)
        assert (adj * expected_sign) > 0, f"exp={exp}: adj={adj}"

    def test_clamped_to_max(self):
        combined = {"expectancy_pct": 100.0}
        adj = _compute_score_adjustment(combined)
        assert adj <= SCORE_ADJ_MAX

    def test_none_expectancy_returns_zero(self):
        assert _compute_score_adjustment({"expectancy_pct": None}) == 0.0

    def test_adj_range(self):
        for exp in [-5.0, -2.5, -1.5, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0]:
            adj = _compute_score_adjustment({"expectancy_pct": exp})
            assert SCORE_ADJ_MIN <= adj <= SCORE_ADJ_MAX, f"exp={exp}: adj={adj} out of range"


# ---------------------------------------------------------------------------
# _compute_block
# ---------------------------------------------------------------------------

class TestComputeBlock:
    def test_block_when_below_all_thresholds(self):
        combined = {
            "combined_sample": MIN_COMBINED_FOR_BLOCK,
            "expectancy_pct": BLOCK_EXPECTANCY_THRESHOLD - 0.1,
            "hit_rate_pct": BLOCK_HIT_RATE_THRESHOLD - 1.0,
        }
        assert _compute_block(combined) is True

    def test_no_block_insufficient_sample(self):
        combined = {
            "combined_sample": MIN_COMBINED_FOR_BLOCK - 1,
            "expectancy_pct": -5.0,
            "hit_rate_pct": 20.0,
        }
        assert _compute_block(combined) is False

    def test_no_block_when_only_expectancy_bad(self):
        """Requires BOTH expectancy AND hit rate to be bad."""
        combined = {
            "combined_sample": MIN_COMBINED_FOR_BLOCK,
            "expectancy_pct": BLOCK_EXPECTANCY_THRESHOLD - 0.1,
            "hit_rate_pct": BLOCK_HIT_RATE_THRESHOLD + 5.0,  # hit rate OK
        }
        assert _compute_block(combined) is False

    def test_no_block_at_exact_threshold(self):
        """At exactly the threshold — not below — should not block."""
        combined = {
            "combined_sample": MIN_COMBINED_FOR_BLOCK,
            "expectancy_pct": BLOCK_EXPECTANCY_THRESHOLD,  # exactly at threshold
            "hit_rate_pct": BLOCK_HIT_RATE_THRESHOLD,
        }
        assert _compute_block(combined) is False


# ---------------------------------------------------------------------------
# run_calibration_pulse integration
# ---------------------------------------------------------------------------

class TestRunCalibrationPulse:
    def test_empty_db_returns_ok(self):
        conn = _make_conn()
        result = run_calibration_pulse(conn, trigger="test")
        assert result["ok"] is True
        assert result["families_updated"] == 0
        assert result["changes"] == []

    def test_insufficient_sample_not_written(self):
        conn = _make_conn()
        # Only 3 eval samples — below MIN_EVAL_SAMPLE (15)
        _seed_eval(conn, "bullish_breakout", "long", [-1.0, -1.0, -1.0])
        run_calibration_pulse(conn, trigger="test")
        rows = conn.execute("SELECT * FROM calibration_state").fetchall()
        assert len(rows) == 0

    def test_sufficient_eval_sample_written(self):
        conn = _make_conn()
        returns = [-1.5] * MIN_EVAL_SAMPLE  # all losses → negative expectancy
        _seed_eval(conn, "bullish_continuation", "long", returns)
        run_calibration_pulse(conn, trigger="test")
        row = conn.execute(
            "SELECT score_adjustment, block_new_entries FROM calibration_state "
            "WHERE family='bullish_continuation' AND direction='long'"
        ).fetchone()
        assert row is not None
        assert row[0] < 0  # negative adjustment for losing family

    def test_block_triggered_by_large_sample(self):
        conn = _make_conn()
        # Lots of losses → should trigger block
        eval_returns = [-3.0] * MIN_EVAL_SAMPLE
        paper_returns = [-3.0] * MIN_PAPER_SAMPLE
        _seed_eval(conn, "bullish_reversal", "long", eval_returns)
        _seed_paper(conn, "bullish_reversal", "long", paper_returns)
        run_calibration_pulse(conn, trigger="test")
        row = conn.execute(
            "SELECT block_new_entries FROM calibration_state "
            "WHERE family='bullish_reversal' AND direction='long'"
        ).fetchone()
        # Whether block triggers depends on combined sample vs MIN_COMBINED_FOR_BLOCK (20)
        # With eval=15 + paper=5 → combined=20 = exactly at threshold
        # expectancy is -3.0, hit_rate=0% → should block
        assert row is not None
        assert row[0] == 1  # blocked

    def test_load_calibration_state(self):
        conn = _make_conn()
        _seed_eval(conn, "bearish_continuation", "short", [-2.0] * MIN_EVAL_SAMPLE)
        run_calibration_pulse(conn, trigger="test")
        state = load_calibration_state(conn)
        assert ("bearish_continuation", "short") in state
        entry = state[("bearish_continuation", "short")]
        assert "score_adjustment" in entry
        assert "block_new_entries" in entry

    def test_load_empty_when_no_table(self):
        conn = sqlite3.connect(":memory:")  # no schema
        state = load_calibration_state(conn)
        assert state == {}

    def test_upsert_idempotent(self):
        conn = _make_conn()
        _seed_eval(conn, "bullish_watch", "long", [-1.0] * MIN_EVAL_SAMPLE)
        run_calibration_pulse(conn, trigger="test")
        run_calibration_pulse(conn, trigger="test")  # run twice
        count = conn.execute("SELECT COUNT(*) FROM calibration_state").fetchone()[0]
        assert count == 1  # not doubled


# ---------------------------------------------------------------------------
# build_telegram_message
# ---------------------------------------------------------------------------

class TestTelegramMessage:
    def test_no_changes_message(self):
        msg = build_telegram_message({
            "trigger": "scheduled",
            "ts": "2026-04-10T23:15:00",
            "families_updated": 5,
            "changes": [],
            "blocks_added": [],
            "blocks_lifted": [],
        })
        assert "Calibration Pulse" in msg
        assert "No significant changes" in msg

    def test_blocks_appear_in_message(self):
        msg = build_telegram_message({
            "trigger": "manual",
            "ts": "2026-04-10T23:15:00",
            "families_updated": 3,
            "changes": [
                {"key": "bullish_reversal:long", "old_adj": 0.0, "new_adj": -15.0,
                 "old_block": False, "new_block": True, "expectancy_pct": -2.5,
                 "hit_rate_pct": 35.0, "combined_sample": 25},
            ],
            "blocks_added": ["bullish_reversal:long"],
            "blocks_lifted": [],
        })
        assert "bullish_reversal:long" in msg
        assert "Blocked" in msg or "\u274c" in msg
