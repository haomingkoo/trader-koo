"""Tests for the /api/admin/cohort-stats endpoint helpers.

Focuses on the pure functions ``_wilson_interval``, ``_r_histogram``,
``_family_breakdown``, and ``_summarise_cohort``. The HTTP layer just
glues these together — covered separately in router integration tests.

Wilson CI math is asserted because the dashboard's "Actionable Review"
currently surfaces 5-trade samples as recommendations; this is the
guardrail that flags those as not statistically distinguishable from
zero edge.
"""
from __future__ import annotations

import sqlite3

import pytest

from trader_koo.backend.routers.admin.cohort_stats import (
    _family_breakdown,
    _r_histogram,
    _summarise_cohort,
    _wilson_interval,
)


class TestWilsonInterval:
    def test_zero_trades_returns_zero(self):
        lo, hi = _wilson_interval(wins=0, total=0)
        assert lo == 0.0 and hi == 0.0

    def test_perfect_record_at_n5_still_has_wide_ci(self):
        """5/5 wins is NOT statistically distinguishable from a coin flip.
        The Wilson upper bound at 100% obs is exactly 100%, but the
        lower bound should be well below 50%."""
        lo, hi = _wilson_interval(wins=5, total=5)
        assert hi == 100.0
        assert lo < 60.0, f"5/5 should leave room for noise, got lo={lo}"

    def test_3_of_5_wins_includes_50pct(self):
        """3/5 = 60% looks like an edge until you see the 95% CI
        spans both 50% and below — the dashboard shouldn't claim a
        family "works" off this sample."""
        lo, hi = _wilson_interval(wins=3, total=5)
        assert lo < 50.0 < hi

    def test_15_of_20_wins_excludes_50pct(self):
        """At N=20 with 75% WR, the lower CI should clear 50% — this
        is roughly the activation threshold."""
        lo, hi = _wilson_interval(wins=15, total=20)
        assert lo > 50.0


class TestRHistogram:
    def test_empty_returns_all_zero_bins(self):
        h = _r_histogram([])
        assert sum(h.values()) == 0
        assert set(h.keys()) == {"<-2R", "-2R..-1R", "-1R..0", "0..1R", "1R..2R", ">2R"}

    def test_bins_at_boundaries(self):
        h = _r_histogram([-3.0, -1.5, -0.5, 0.5, 1.5, 2.5])
        assert h["<-2R"] == 1
        assert h["-2R..-1R"] == 1
        assert h["-1R..0"] == 1
        assert h["0..1R"] == 1
        assert h["1R..2R"] == 1
        assert h[">2R"] == 1

    def test_user_data_distribution(self):
        """Mirrors the user's pasted CSV — should show concentration of
        losses near -1R (correct dollar-risk sizing) plus 1 outlier
        winner above 2R (the one target_hit trade)."""
        rs = [-1.03, -1.04, -1.02, -1.05, -1.41, -1.46, -1.13, -1.13,
              0.94, 0.92, 0.91, 0.94, 0.96, 0.73, 1.10, 1.97, 2.56]
        h = _r_histogram(rs)
        # most losses cluster at -1R..-2R
        assert h["-2R..-1R"] >= 6
        # exactly one >2R winner (WMT target hit)
        assert h[">2R"] == 1


class TestFamilyBreakdown:
    def _make_row(self, family: str, pnl_pct: float, r: float):
        return {
            "setup_family": family,
            "pnl_pct": pnl_pct,
            "r_multiple": r,
        }

    def test_aggregates_by_family(self):
        rows = [
            self._make_row("bullish_continuation", -1.65, -1.41),
            self._make_row("bullish_continuation", -1.92, -1.46),
            self._make_row("bullish_continuation", 3.81, 1.0),  # rare winner
            self._make_row("bearish_continuation", 3.10, 0.91),
            self._make_row("bearish_continuation", 4.66, 0.73),
        ]
        out = _family_breakdown(rows)
        by_fam = {r["family"]: r for r in out}

        assert by_fam["bullish_continuation"]["n_trades"] == 3
        assert by_fam["bullish_continuation"]["n_wins"] == 1
        assert by_fam["bullish_continuation"]["win_rate_pct"] == pytest.approx(33.3, abs=0.1)
        # CI should be wide at N=3
        wr_lo, wr_hi = by_fam["bullish_continuation"]["win_rate_ci_95"]
        assert wr_hi - wr_lo > 50.0

    def test_sorted_by_n_trades_desc(self):
        rows = [
            self._make_row("rare", 1.0, 0.5),
            self._make_row("common", 1.0, 0.5),
            self._make_row("common", -1.0, -0.5),
            self._make_row("common", 2.0, 1.0),
        ]
        out = _family_breakdown(rows)
        assert out[0]["family"] == "common"


class TestSummariseCohort:
    """``_summarise_cohort`` consumes sqlite3.Row-like objects. We use
    sqlite3.Connection to produce real Row instances so the indexing
    semantics match production."""

    def _setup_conn(self, trades: list[dict]) -> list[sqlite3.Row]:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE paper_trades (
                bot_version TEXT,
                status TEXT,
                pnl_pct REAL,
                r_multiple REAL,
                setup_family TEXT,
                position_size_pct REAL,
                deployed_capital_pct REAL,
                entry_date TEXT
            )
        """)
        for t in trades:
            conn.execute(
                "INSERT INTO paper_trades VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    t.get("bot_version", "test-cohort"),
                    t.get("status", "closed"),
                    t.get("pnl_pct"),
                    t.get("r_multiple"),
                    t.get("setup_family", "bullish_reversal"),
                    t.get("position_size_pct"),
                    t.get("deployed_capital_pct"),
                    t.get("entry_date", "2026-04-01"),
                ),
            )
        return conn.execute("SELECT * FROM paper_trades").fetchall()

    def test_empty_cohort_returns_note(self):
        result = _summarise_cohort("v1", [])
        assert result["n_closed"] == 0
        assert "no closed trades" in result.get("note", "")

    def test_only_open_trades_no_closed(self):
        rows = self._setup_conn([
            {"status": "open", "pnl_pct": None, "r_multiple": None},
        ])
        result = _summarise_cohort("v1", rows)
        assert result["n_open"] == 1
        assert result["n_closed"] == 0

    def test_basic_stats(self):
        rows = self._setup_conn([
            {"status": "stopped_out", "pnl_pct": -1.0, "r_multiple": -1.0,
             "position_size_pct": 8.0, "entry_date": "2026-04-01"},
            {"status": "target_hit", "pnl_pct": 3.0, "r_multiple": 2.5,
             "position_size_pct": 8.0, "entry_date": "2026-04-05"},
            {"status": "expired", "pnl_pct": 1.5, "r_multiple": 1.0,
             "position_size_pct": 8.0, "entry_date": "2026-04-10"},
        ])
        result = _summarise_cohort("v1", rows)
        assert result["n_closed"] == 3
        assert result["n_wins"] == 2
        assert result["n_losses"] == 1
        assert result["win_rate_pct"] == pytest.approx(66.7, abs=0.1)
        assert result["expectancy_r"] == pytest.approx((-1.0 + 2.5 + 1.0) / 3, abs=0.01)
        assert result["best_trade_pct"] == 3.0
        assert result["worst_trade_pct"] == -1.0
        # Cash-adjusted: each at 8% sizing -> contribs are 0.08 * [-1, 3, 1.5]
        # = -0.08 + 0.24 + 0.12 = 0.28
        assert result["cash_adjusted_portfolio_contrib_pct"] == pytest.approx(0.28, abs=0.01)
        assert result["first_trade_date"] == "2026-04-01"
        assert result["last_trade_date"] == "2026-04-10"

    def test_wilson_ci_warns_on_thin_sample(self):
        """3-trade cohort with 2 wins should NOT have a CI lower bound
        above 50% — caller should treat it as inconclusive."""
        rows = self._setup_conn([
            {"status": "target_hit", "pnl_pct": 2.0, "r_multiple": 1.5},
            {"status": "target_hit", "pnl_pct": 1.0, "r_multiple": 0.8},
            {"status": "stopped_out", "pnl_pct": -1.0, "r_multiple": -1.0},
        ])
        result = _summarise_cohort("v1", rows)
        wr_lo, _ = result["win_rate_ci_95"]
        assert wr_lo < 50.0, "thin sample should not show 'edge' at 95% CI"
