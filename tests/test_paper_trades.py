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


def _seed_price(conn: sqlite3.Connection, ticker: str, close: float, date: str = "2026-03-14") -> None:
    conn.execute(
        "INSERT OR REPLACE INTO price_daily (ticker, date, close) VALUES (?, ?, ?)",
        (ticker, date, close),
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
        rows = [_make_setup_row(actionability="conditional", risk_note="High volatility into earnings")]

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
                           stop_loss=95.0, target_price=110.0, entry_date="2026-03-10"):
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
        status = conn.execute("SELECT status, exit_reason FROM paper_trades WHERE ticker='AAPL'").fetchone()
        assert status[0] == "stopped_out"
        assert status[1] == "stopped_out"

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
        assert result["overall"]["total_pnl_pct"] == 10.0
