"""Tests for trader_koo.paper_trades module."""
from __future__ import annotations

import datetime as dt
import sqlite3

import pytest

from trader_koo.paper_trades import (
    _compute_pnl,
    _compute_r_multiple,
    _direction_from_row,
    compute_stop_and_target,
    create_paper_trades_from_report,
    evaluate_setup_for_paper_trade,
    ensure_paper_trade_schema,
    list_paper_trades,
    manually_close_trade,
    mark_to_market,
    paper_trade_summary,
    qualify_setup_for_paper_trade,
)


@pytest.fixture()
def conn():
    """In-memory SQLite connection with paper trade + price schema."""
    db = sqlite3.connect(":memory:")
    ensure_paper_trade_schema(db)
    db.execute("""
        CREATE TABLE IF NOT EXISTS price_daily (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume INTEGER,
            UNIQUE(ticker, date)
        )
    """)
    db.commit()
    return db


def _seed_price(
    conn: sqlite3.Connection,
    ticker: str,
    close: float,
    date: str = "2026-03-14",
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO price_daily (ticker, date, close, high, low, open) VALUES (?, ?, ?, ?, ?, ?)",
        (
            ticker, date, close,
            high if high is not None else close,
            low if low is not None else close,
            open_ if open_ is not None else close,
        ),
    )
    conn.commit()


def _make_setup_row(
    ticker: str = "AAPL",
    setup_tier: str = "A",
    score: float = 80.0,
    actionability: str = "higher-probability",
    signal_bias: str = "bullish",
    close: float = 150.0,
    setup_family: str = "Bullish Breakout",
    atr_pct_14: float = 2.5,
    support_level: float = 140.0,
    resistance_level: float = 165.0,
    risk_note: str = "Standard risk controls.",
    debate_agreement_score: float = 80.0,
) -> dict:
    return {
        "ticker": ticker,
        "setup_tier": setup_tier,
        "score": score,
        "actionability": actionability,
        "signal_bias": signal_bias,
        "close": close,
        "setup_family": setup_family,
        "atr_pct_14": atr_pct_14,
        "support_level": support_level,
        "resistance_level": resistance_level,
        "observation": "Test setup",
        "action": "Buy on breakout",
        "risk_note": risk_note,
        "debate_agreement_score": debate_agreement_score,
    }


# ── Direction Detection ──────────────────────────────────────────

class TestDirectionFromRow:
    def test_bullish_family_returns_long(self):
        assert _direction_from_row({"setup_family": "Bullish Breakout"}) == "long"

    def test_bearish_family_returns_short(self):
        assert _direction_from_row({"setup_family": "Bearish Reversal"}) == "short"

    def test_bullish_bias_returns_long(self):
        assert _direction_from_row({"signal_bias": "bullish"}) == "long"

    def test_bearish_bias_returns_short(self):
        assert _direction_from_row({"signal_bias": "bearish"}) == "short"

    def test_neutral_family_and_bias_returns_neutral(self):
        assert _direction_from_row({"setup_family": "Consolidation", "signal_bias": "neutral"}) == "neutral"


# ── Qualification ────────────────────────────────────────────────

class TestQualifySetup:
    def test_qualifying_setup_passes(self):
        row = _make_setup_row()

        assert qualify_setup_for_paper_trade(row) is True

    def test_evaluator_returns_stage_metadata(self):
        decision = evaluate_setup_for_paper_trade(_make_setup_row())

        assert decision["approved"] is True
        assert decision["decision_state"] == "approved"
        assert decision["analyst_stage"] == "pass"
        assert decision["portfolio_decision"] == "approved"

    def test_low_tier_rejects(self):
        row = _make_setup_row(setup_tier="D")

        assert qualify_setup_for_paper_trade(row) is False

    def test_low_score_rejects(self):
        row = _make_setup_row(score=30.0)

        assert qualify_setup_for_paper_trade(row) is False

    def test_non_actionable_rejects(self):
        row = _make_setup_row(actionability="monitoring")

        assert qualify_setup_for_paper_trade(row) is False

    def test_neutral_direction_rejects(self):
        row = _make_setup_row(signal_bias="neutral", setup_family="Consolidation")

        assert qualify_setup_for_paper_trade(row) is False

    def test_zero_price_rejects(self):
        row = _make_setup_row(close=0)

        assert qualify_setup_for_paper_trade(row) is False

    def test_conditional_setup_gets_caution_state(self):
        decision = evaluate_setup_for_paper_trade(
            _make_setup_row(actionability="conditional", risk_note="Watch earnings"),
        )

        assert decision["approved"] is True
        assert decision["decision_state"] == "approved_with_flags"
        assert decision["debate_stage"] == "caution"
        assert decision["risk_stage"] == "caution"
        assert decision["risk_flags"]


# ── Stop / Target Computation ────────────────────────────────────

class TestComputeStopAndTarget:
    def test_long_atr_based_stop(self):
        row = _make_setup_row(close=100.0, atr_pct_14=2.0, support_level=None, resistance_level=None)

        result = compute_stop_and_target(row, "long")

        assert result["stop_loss"] < 100.0
        assert result["target_price"] > 100.0
        assert result["atr_at_entry"] == 2.0

    def test_short_atr_based_stop(self):
        row = _make_setup_row(close=100.0, atr_pct_14=2.0, signal_bias="bearish",
                              setup_family="Bearish Reversal", support_level=None, resistance_level=None)

        result = compute_stop_and_target(row, "short")

        assert result["stop_loss"] > 100.0
        assert result["target_price"] < 100.0

    def test_long_uses_resistance_as_target(self):
        row = _make_setup_row(close=100.0, atr_pct_14=2.0, resistance_level=110.0)

        result = compute_stop_and_target(row, "long")

        assert result["target_price"] == 110.0

    def test_long_support_tightens_stop(self):
        row = _make_setup_row(close=100.0, atr_pct_14=2.0, support_level=98.0)

        result = compute_stop_and_target(row, "long")

        # Support stop = 98 * 0.99 = 97.02, should be within valid range
        assert result["stop_loss"] >= 95.0
        assert result["stop_loss"] < 100.0

    def test_fallback_stop_when_no_atr(self):
        row = _make_setup_row(close=100.0, atr_pct_14=None, support_level=None, resistance_level=None)

        result = compute_stop_and_target(row, "long")

        assert result["stop_loss"] == 97.0  # 3% default stop
        assert result["atr_at_entry"] is None


# ── PnL Computation ──────────────────────────────────────────────

class TestComputePnl:
    def test_long_profit(self):
        assert round(_compute_pnl("long", 100.0, 110.0), 2) == 10.0

    def test_long_loss(self):
        assert round(_compute_pnl("long", 100.0, 95.0), 2) == -5.0

    def test_short_profit(self):
        assert round(_compute_pnl("short", 100.0, 90.0), 2) == 10.0

    def test_short_loss(self):
        assert round(_compute_pnl("short", 100.0, 105.0), 2) == -5.0


class TestComputeRMultiple:
    def test_long_1r_win(self):
        result = _compute_r_multiple("long", 100.0, 105.0, 95.0)

        assert result == 1.0

    def test_short_2r_win(self):
        result = _compute_r_multiple("short", 100.0, 90.0, 105.0)

        assert result == 2.0

    def test_none_stop_uses_default(self):
        result = _compute_r_multiple("long", 100.0, 103.0, None)

        assert result is not None
        assert result == 1.0  # 3% default, 3/3 = 1R


# ── Trade Creation ───────────────────────────────────────────────

class TestCreatePaperTrades:
    def test_creates_qualifying_trade(self, conn):
        rows = [_make_setup_row()]

        inserted = create_paper_trades_from_report(
            conn, setup_rows=rows, report_date="2026-03-14", generated_ts="2026-03-14T22:00:00Z",
        )

        assert inserted == 1
        trade = conn.execute(
            "SELECT ticker, direction, status, decision_state, portfolio_decision FROM paper_trades",
        ).fetchone()
        assert trade == ("AAPL", "long", "open", "approved", "approved")

    def test_stores_decision_metadata_for_flagged_trade(self, conn):
        rows = [_make_setup_row(actionability="conditional", risk_note="Watch earnings date")]

        inserted = create_paper_trades_from_report(
            conn, setup_rows=rows, report_date="2026-03-14", generated_ts="2026-03-14T22:00:00Z",
        )

        assert inserted == 1
        trade = conn.execute(
            "SELECT decision_state, debate_stage, risk_stage, decision_summary, risk_flags "
            "FROM paper_trades WHERE ticker = 'AAPL'",
        ).fetchone()
        assert trade[0] == "approved_with_flags"
        assert trade[1] == "caution"
        assert trade[2] == "caution"
        assert "caution" in trade[3].lower()
        assert "earnings" in str(trade[4]).lower()

    def test_stores_position_plan_metadata(self, conn):
        rows = [_make_setup_row()]

        inserted = create_paper_trades_from_report(
            conn, setup_rows=rows, report_date="2026-03-14", generated_ts="2026-03-14T22:00:00Z",
        )

        assert inserted == 1
        trade = conn.execute(
            "SELECT position_size_pct, risk_budget_pct, expected_r_multiple, entry_plan, exit_plan, sizing_summary "
            "FROM paper_trades WHERE ticker = 'AAPL'",
        ).fetchone()
        assert trade[0] is not None
        assert trade[1] is not None
        assert trade[2] is not None and trade[2] >= 1.5
        assert "enter" in str(trade[3]).lower()
        assert "stop" in str(trade[4]).lower()
        assert "%" in str(trade[5])

    def test_stores_bot_version_and_entry_context(self, conn):
        for idx in range(25):
            date = (dt.date(2026, 2, 18) + dt.timedelta(days=idx)).isoformat()
            _seed_price(conn, "^VIX", 16.0 + (idx % 5), date=date)
        for idx, close in enumerate(range(500, 550), start=1):
            conn.execute(
                "INSERT OR REPLACE INTO price_daily (ticker, date, close) VALUES (?, ?, ?)",
                ("SPY", f"2026-01-{idx:02d}" if idx <= 31 else f"2026-02-{idx-31:02d}", float(close)),
            )
        conn.commit()

        inserted = create_paper_trades_from_report(
            conn,
            setup_rows=[_make_setup_row()],
            report_date="2026-03-14",
            generated_ts="2026-03-14T22:00:00Z",
        )

        assert inserted == 1
        trade = conn.execute(
            "SELECT bot_version, vix_at_entry, vix_percentile_at_entry, regime_state_at_entry "
            "FROM paper_trades WHERE ticker = 'AAPL'",
        ).fetchone()
        assert trade[0] == "v1.0.0"
        assert trade[1] == pytest.approx(20.0)
        assert trade[2] is not None
        assert trade[3] is not None
        version_row = conn.execute(
            "SELECT bot_version, decision_version FROM bot_versions WHERE bot_version = 'v1.0.0'",
        ).fetchone()
        assert version_row == ("v1.0.0", "paper-trade-eval-v1")

    def test_skips_trade_with_poor_reward_to_risk(self, conn):
        rows = [_make_setup_row(resistance_level=151.0)]

        inserted = create_paper_trades_from_report(
            conn, setup_rows=rows, report_date="2026-03-14", generated_ts="2026-03-14T22:00:00Z",
        )

        assert inserted == 0

    def test_skips_non_qualifying_setup(self, conn):
        rows = [_make_setup_row(score=10.0)]

        inserted = create_paper_trades_from_report(
            conn, setup_rows=rows, report_date="2026-03-14", generated_ts="2026-03-14T22:00:00Z",
        )

        assert inserted == 0

    def test_deduplicates_on_same_report_date_ticker_direction(self, conn):
        rows = [_make_setup_row()]

        create_paper_trades_from_report(conn, setup_rows=rows, report_date="2026-03-14", generated_ts="ts1")
        conn.commit()
        inserted2 = create_paper_trades_from_report(conn, setup_rows=rows, report_date="2026-03-14", generated_ts="ts2")

        assert inserted2 == 0
        assert conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 1

    def test_respects_max_open_limit(self, conn):
        import trader_koo.paper_trades as pt
        original = pt.PAPER_TRADE_MAX_OPEN
        pt.PAPER_TRADE_MAX_OPEN = 1
        try:
            rows = [
                _make_setup_row(ticker="AAPL"),
                _make_setup_row(ticker="MSFT"),
            ]

            inserted = create_paper_trades_from_report(
                conn, setup_rows=rows, report_date="2026-03-14", generated_ts="ts",
            )

            assert inserted == 1
        finally:
            pt.PAPER_TRADE_MAX_OPEN = original

    def test_empty_rows_returns_zero(self, conn):
        inserted = create_paper_trades_from_report(
            conn, setup_rows=[], report_date="2026-03-14", generated_ts="ts",
        )

        assert inserted == 0


# ── Mark to Market ───────────────────────────────────────────────

class TestMarkToMarket:
    def _insert_open_trade(self, conn, ticker="AAPL", entry_price=100.0, direction="long",
                           stop_loss=95.0, target_price=110.0, entry_date=None):
        if entry_date is None:
            entry_date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        conn.execute(
            """INSERT INTO paper_trades (report_date, ticker, direction, entry_price, entry_date,
               target_price, stop_loss, status, current_price, unrealized_pnl_pct,
               high_water_mark, low_water_mark, generated_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, 0.0, ?, ?, ?)""",
            (entry_date, ticker, direction, entry_price, entry_date,
             target_price, stop_loss, entry_price, entry_price, entry_price, "ts"),
        )
        conn.commit()

    def test_updates_price_on_open_trade(self, conn):
        self._insert_open_trade(conn, "AAPL", 100.0)
        _seed_price(conn, "AAPL", 105.0)

        result = mark_to_market(conn)
        conn.commit()

        assert result["updated"] == 1
        assert result["closed"] == 0
        trade = conn.execute("SELECT current_price, unrealized_pnl_pct FROM paper_trades WHERE ticker='AAPL'").fetchone()
        assert trade[0] == 105.0
        assert trade[1] == 5.0

    def test_triggers_stop_loss(self, conn):
        self._insert_open_trade(conn, "AAPL", 100.0, stop_loss=95.0)
        _seed_price(conn, "AAPL", 93.0)

        result = mark_to_market(conn)
        conn.commit()

        assert result["closed"] == 1
        status = conn.execute(
            "SELECT status, exit_reason, review_status, review_summary FROM paper_trades WHERE ticker='AAPL'",
        ).fetchone()
        assert status[0] == "stopped_out"
        assert status[1] == "stopped_out"
        assert status[2] == "stopped_out"
        assert "invalidation" in str(status[3]).lower()

    def test_intraday_low_triggers_stop_even_if_close_above(self, conn):
        """Stop should trigger on intraday low, not just close."""
        self._insert_open_trade(conn, "AAPL", 100.0, stop_loss=95.0)
        # Close is above stop, but intraday low hit the stop
        _seed_price(conn, "AAPL", 98.0, high=101.0, low=94.0)

        result = mark_to_market(conn)
        conn.commit()

        assert result["closed"] == 1
        trade = conn.execute(
            "SELECT status, exit_price FROM paper_trades WHERE ticker='AAPL'"
        ).fetchone()
        assert trade[0] == "stopped_out"
        assert trade[1] == 95.0  # filled at stop level, not at the low

    def test_intraday_high_triggers_short_stop(self, conn):
        """Short stop should trigger on intraday high."""
        self._insert_open_trade(conn, "AAPL", 100.0, direction="short", stop_loss=105.0, target_price=90.0)
        # Close is below stop, but intraday high breached it. Open is safe (101).
        _seed_price(conn, "AAPL", 102.0, high=106.0, low=99.0, open_=101.0)

        result = mark_to_market(conn)
        conn.commit()

        assert result["closed"] == 1
        trade = conn.execute(
            "SELECT status, exit_price FROM paper_trades WHERE ticker='AAPL'"
        ).fetchone()
        assert trade[0] == "stopped_out"
        assert trade[1] == 105.0  # filled at stop level

    def test_intraday_low_triggers_short_target(self, conn):
        """Short target should trigger on intraday low."""
        self._insert_open_trade(conn, "AAPL", 100.0, direction="short", target_price=90.0, stop_loss=105.0)
        # Close is above target, but intraday low hit it
        _seed_price(conn, "AAPL", 93.0, high=98.0, low=89.0)

        result = mark_to_market(conn)
        conn.commit()

        assert result["closed"] == 1
        trade = conn.execute(
            "SELECT status, exit_price FROM paper_trades WHERE ticker='AAPL'"
        ).fetchone()
        assert trade[0] == "target_hit"
        assert trade[1] == 90.0  # filled at target level

    def test_open_gaps_through_target_takes_profit(self, conn):
        """If open itself is past the target, take profit with no ambiguity."""
        self._insert_open_trade(conn, "AAPL", 100.0, direction="short", target_price=90.0, stop_loss=105.0)
        # Open gaps below target - guaranteed profit
        _seed_price(conn, "AAPL", 92.0, high=93.0, low=87.0, open_=88.0)

        result = mark_to_market(conn)
        conn.commit()

        assert result["closed"] == 1
        trade = conn.execute(
            "SELECT status, exit_price FROM paper_trades WHERE ticker='AAPL'"
        ).fetchone()
        assert trade[0] == "target_hit"
        assert trade[1] == 90.0  # filled at target level

    def test_open_gaps_through_stop_gets_stopped(self, conn):
        """If open itself is past the stop, stopped out with no ambiguity."""
        self._insert_open_trade(conn, "AAPL", 100.0, stop_loss=95.0)
        # Open gaps below stop - guaranteed stop
        _seed_price(conn, "AAPL", 96.0, high=97.0, low=91.0, open_=93.0)

        result = mark_to_market(conn)
        conn.commit()

        assert result["closed"] == 1
        trade = conn.execute(
            "SELECT status, exit_price FROM paper_trades WHERE ticker='AAPL'"
        ).fetchone()
        assert trade[0] == "stopped_out"
        assert trade[1] == 95.0  # filled at stop level

    def test_both_stop_and_target_hit_intraday_takes_stop(self, conn):
        """If both stop and target hit intraday (not at open), conservative: assume stop."""
        self._insert_open_trade(conn, "AAPL", 100.0, stop_loss=95.0, target_price=110.0)
        # Open is safe, but both extremes breach stop and target
        _seed_price(conn, "AAPL", 102.0, high=112.0, low=93.0, open_=100.0)

        result = mark_to_market(conn)
        conn.commit()

        assert result["closed"] == 1
        trade = conn.execute(
            "SELECT status, exit_price FROM paper_trades WHERE ticker='AAPL'"
        ).fetchone()
        assert trade[0] == "stopped_out"
        assert trade[1] == 95.0  # conservative: assume stop hit first

    def test_triggers_target_hit(self, conn):
        self._insert_open_trade(conn, "AAPL", 100.0, target_price=110.0)
        _seed_price(conn, "AAPL", 112.0)

        result = mark_to_market(conn)
        conn.commit()

        assert result["closed"] == 1
        status = conn.execute("SELECT status, exit_reason FROM paper_trades WHERE ticker='AAPL'").fetchone()
        assert status[0] == "target_hit"

    def test_triggers_expiry(self, conn):
        old_date = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=15)).strftime("%Y-%m-%d")
        self._insert_open_trade(conn, "AAPL", 100.0, entry_date=old_date)
        _seed_price(conn, "AAPL", 102.0)

        result = mark_to_market(conn)
        conn.commit()

        assert result["closed"] == 1
        status = conn.execute("SELECT status, exit_reason FROM paper_trades WHERE ticker='AAPL'").fetchone()
        assert status[0] == "expired"

    def test_creates_portfolio_snapshot(self, conn):
        self._insert_open_trade(conn, "AAPL", 100.0)
        _seed_price(conn, "AAPL", 105.0)

        mark_to_market(conn)
        conn.commit()

        snapshot = conn.execute("SELECT * FROM paper_portfolio_snapshots").fetchone()
        assert snapshot is not None

    def test_no_open_trades_still_snapshots(self, conn):
        result = mark_to_market(conn)
        conn.commit()

        assert result["open_trades"] == 0
        snapshot = conn.execute("SELECT COUNT(*) FROM paper_portfolio_snapshots").fetchone()
        assert snapshot[0] == 1


# ── Manual Close ─────────────────────────────────────────────────

class TestManuallyCloseTrade:
    def test_closes_trade_with_explicit_price(self, conn):
        conn.execute(
            """INSERT INTO paper_trades (report_date, ticker, direction, entry_price, entry_date,
               stop_loss, status, current_price, unrealized_pnl_pct,
               high_water_mark, low_water_mark, generated_ts)
            VALUES ('2026-03-14', 'MSFT', 'long', 300.0, '2026-03-14',
               290.0, 'open', 300.0, 0.0, 300.0, 300.0, 'ts')""",
        )
        conn.commit()
        trade_id = conn.execute("SELECT id FROM paper_trades").fetchone()[0]

        result = manually_close_trade(conn, trade_id=trade_id, exit_price=320.0)

        assert result["pnl_pct"] > 0
        assert result["status"] == "closed"
        assert result["ticker"] == "MSFT"

    def test_raises_on_already_closed(self, conn):
        conn.execute(
            """INSERT INTO paper_trades (report_date, ticker, direction, entry_price, entry_date,
               status, current_price, generated_ts)
            VALUES ('2026-03-14', 'GOOG', 'long', 150.0, '2026-03-14',
               'closed', 150.0, 'ts')""",
        )
        conn.commit()
        trade_id = conn.execute("SELECT id FROM paper_trades").fetchone()[0]

        with pytest.raises(ValueError, match="already closed"):
            manually_close_trade(conn, trade_id=trade_id, exit_price=160.0)

    def test_raises_on_nonexistent_trade(self, conn):
        with pytest.raises(ValueError, match="not found"):
            manually_close_trade(conn, trade_id=9999)


# ── Listing ──────────────────────────────────────────────────────

class TestListPaperTrades:
    def test_filters_by_status(self, conn):
        conn.execute(
            """INSERT INTO paper_trades (report_date, ticker, direction, entry_price, entry_date,
               status, current_price, generated_ts)
            VALUES ('2026-03-14', 'AAPL', 'long', 150.0, '2026-03-14', 'open', 150.0, 'ts')""",
        )
        conn.execute(
            """INSERT INTO paper_trades (report_date, ticker, direction, entry_price, entry_date,
               status, current_price, generated_ts)
            VALUES ('2026-03-14', 'MSFT', 'long', 300.0, '2026-03-14', 'closed', 310.0, 'ts')""",
        )
        conn.commit()

        open_trades = list_paper_trades(conn, status="open")
        all_trades = list_paper_trades(conn, status="all")

        assert len(open_trades) == 1
        assert open_trades[0]["ticker"] == "AAPL"
        assert len(all_trades) == 2

    def test_filters_by_ticker(self, conn):
        conn.execute(
            """INSERT INTO paper_trades (report_date, ticker, direction, entry_price, entry_date,
               status, current_price, generated_ts)
            VALUES ('2026-03-14', 'AAPL', 'long', 150.0, '2026-03-14', 'open', 150.0, 'ts')""",
        )
        conn.commit()

        result = list_paper_trades(conn, ticker="AAPL")

        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"


# ── Summary ──────────────────────────────────────────────────────

class TestPaperTradeSummary:
    def test_empty_portfolio_returns_structure(self, conn):
        result = paper_trade_summary(conn)

        assert result["overall"]["total_trades"] == 0
        assert result["by_direction"] == {}
        assert result["equity_curve"] == []
        assert result["policy"]["bot_version"] == "v1.0.0"
        assert result["policy"]["decision_version"] == "paper-trade-eval-v1"
        assert result["feedback"] == []

    def test_summary_with_closed_trades(self, conn):
        for ticker, pnl, r in [("AAPL", 5.0, 1.0), ("MSFT", -3.0, -0.6), ("GOOG", 8.0, 1.6)]:
            conn.execute(
                """INSERT INTO paper_trades (report_date, ticker, direction, entry_price, entry_date,
                   status, pnl_pct, r_multiple, exit_date, exit_reason, current_price, generated_ts)
                VALUES (?, ?, 'long', 100.0, '2026-03-10',
                   'closed', ?, ?, '2026-03-14', 'manual_close', 100.0, 'ts')""",
                (f"2026-03-{10 + hash(ticker) % 3}", ticker, pnl, r),
            )
        conn.commit()

        result = paper_trade_summary(conn)

        assert result["overall"]["total_trades"] == 3
        assert result["overall"]["wins"] == 2
        assert result["overall"]["losses"] == 1
        assert result["overall"]["win_rate_pct"] == pytest.approx(66.7, abs=0.1)
        assert result["overall"]["expectancy_pct"] == pytest.approx(3.33, abs=0.01)
        assert result["overall"]["profit_factor"] == pytest.approx(4.33, abs=0.01)
        assert result["overall"]["total_pnl_pct"] == 10.0
        assert result["policy"]["min_tier"] == "B"
        assert result["feedback"] == []

    def test_summary_includes_edge_tables(self, conn):
        base_date = dt.date.today() - dt.timedelta(days=20)
        for idx in range(6):
            entry_date = (base_date + dt.timedelta(days=idx)).isoformat()
            conn.execute(
                """INSERT INTO paper_trades (
                    report_date, ticker, direction, entry_price, entry_date, status,
                    pnl_pct, r_multiple, exit_date, exit_reason, current_price, generated_ts,
                    setup_family, regime_state_at_entry, vix_at_entry, bot_version
                ) VALUES (?, ?, 'long', 100.0, ?, 'closed', ?, ?, ?, ?, 100.0, 'ts', ?, ?, ?, ?)""",
                (
                    entry_date,
                    f"T{idx}",
                    entry_date,
                    2.0 if idx < 4 else -1.0,
                    1.0 if idx < 4 else -0.5,
                    entry_date,
                    "target_hit" if idx < 4 else "stopped_out",
                    "bullish_continuation",
                    "bull_normal",
                    17.5,
                    "v1.0.0",
                ),
            )
        conn.commit()

        result = paper_trade_summary(conn)

        assert result["family_edges"]
        assert result["regime_edges"]
        assert result["vix_bucket_edges"]
