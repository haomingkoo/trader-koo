"""Tests for trader_koo.notifications.morning_summary.

Validates the enhanced morning summary with:
- Overnight market moves (BTC, ETH, Gold, DXY)
- Open paper trade positions with individual P&L
- Economic calendar events
- Active counter-trade signals
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from trader_koo.notifications.morning_summary import (
    _fear_greed_label,
    _fetch_counter_trade_signals,
    _fetch_crypto_snapshot,
    _fetch_economic_events_today,
    _fetch_index_snapshot,
    _fetch_open_positions,
    _fetch_paper_trade_stats,
    _fmt_money,
    _hours_to_market_open,
    _vix_label,
    generate_morning_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Create a temporary SQLite DB with all required schemas."""
    path = tmp_path / "test_summary.db"
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE price_daily (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            UNIQUE(ticker, date)
        );

        CREATE TABLE crypto_bars (
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            interval TEXT NOT NULL DEFAULT '1m',
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY (symbol, timestamp, interval)
        );

        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            direction TEXT DEFAULT 'long',
            entry_price REAL,
            entry_date TEXT,
            exit_date TEXT,
            status TEXT DEFAULT 'open',
            pnl_pct REAL,
            unrealized_pnl_pct REAL,
            position_size_pct REAL DEFAULT 8.0
        );

        CREATE TABLE hyperliquid_counter_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet_label TEXT NOT NULL,
            coin TEXT NOT NULL,
            counter_side TEXT NOT NULL,
            their_side TEXT NOT NULL,
            their_size REAL,
            their_leverage INTEGER,
            their_notional_usd REAL,
            confidence REAL,
            reasoning TEXT,
            signal_ts TEXT NOT NULL,
            created_ts TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()
    return path


@pytest.fixture()
def conn(db_path: Path) -> sqlite3.Connection:
    """Return a connection to the test DB."""
    c = sqlite3.connect(str(db_path))
    c.row_factory = sqlite3.Row
    return c


@pytest.fixture()
def populated_conn(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Populate the test DB with sample data."""
    # Price data
    conn.executemany(
        "INSERT INTO price_daily (ticker, date, close) VALUES (?, ?, ?)",
        [
            ("SPY", "2026-03-25", 585.20),
            ("SPY", "2026-03-24", 583.00),
            ("QQQ", "2026-03-25", 510.50),
            ("QQQ", "2026-03-24", 508.30),
            ("^VIX", "2026-03-25", 18.5),
            ("^VIX", "2026-03-24", 19.2),
            ("GLD", "2026-03-25", 230.40),
            ("GLD", "2026-03-24", 229.10),
            ("UUP", "2026-03-25", 27.80),
            ("UUP", "2026-03-24", 27.95),
        ],
    )

    # Crypto data (1d bars)
    conn.executemany(
        "INSERT INTO crypto_bars (symbol, timestamp, interval, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("BTC-USD", "2026-03-25T00:00:00", "1d", 87000, 89000, 86500, 88500, 1200),
            ("BTC-USD", "2026-03-24T00:00:00", "1d", 86000, 87500, 85500, 87000, 1100),
            ("ETH-USD", "2026-03-25T00:00:00", "1d", 2050, 2120, 2040, 2100, 5000),
            ("ETH-USD", "2026-03-24T00:00:00", "1d", 2010, 2060, 2000, 2050, 4800),
        ],
    )

    # Paper trades
    conn.executemany(
        "INSERT INTO paper_trades (ticker, direction, entry_price, entry_date, status, pnl_pct, unrealized_pnl_pct, position_size_pct) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("AAPL", "long", 225.00, "2026-03-20", "open", None, 3.5, 8.0),
            ("NVDA", "short", 900.00, "2026-03-22", "open", None, -1.2, 6.0),
            ("MSFT", "long", 420.00, "2026-03-18", "closed", 5.0, None, 8.0),
        ],
    )

    # Counter-trade signals
    now_ts = dt.datetime.now(dt.timezone.utc).isoformat()
    old_ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=48)).isoformat()
    conn.executemany(
        """
        INSERT INTO hyperliquid_counter_signals
            (wallet_label, coin, counter_side, their_side, their_size, their_leverage,
             their_notional_usd, confidence, reasoning, signal_ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("machibro", "BTC", "short", "long", 1.2, 25, 88_000, 82.0, "counter BTC long", now_ts),
            ("machibro", "ETH", "long", "short", 12.0, 15, 2_050, 71.0, "counter ETH short", now_ts),
            ("machibro", "SOL", "short", "long", 100.0, 20, 180, 76.0, "counter SOL long", old_ts),  # older than 24h
        ],
    )

    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------

class TestVixLabel:
    def test_low_vol(self) -> None:
        assert _vix_label(12.0) == "Low vol"

    def test_moderate(self) -> None:
        assert _vix_label(17.0) == "Moderate"

    def test_elevated(self) -> None:
        assert _vix_label(22.0) == "Elevated"

    def test_high(self) -> None:
        assert _vix_label(28.0) == "High"

    def test_extreme(self) -> None:
        assert _vix_label(35.0) == "Extreme"


class TestFearGreedLabel:
    def test_extreme_fear(self) -> None:
        assert _fear_greed_label(15) == "Extreme Fear"

    def test_neutral(self) -> None:
        assert _fear_greed_label(50) == "Neutral"

    def test_extreme_greed(self) -> None:
        assert _fear_greed_label(80) == "Extreme Greed"


class TestFmtMoney:
    def test_millions(self) -> None:
        assert _fmt_money(1_012_340.0) == "$1,012,340"

    def test_smaller(self) -> None:
        assert _fmt_money(999.50) == "$999.50"


class TestHoursToMarketOpen:
    def test_returns_positive_float(self) -> None:
        result = _hours_to_market_open()
        assert isinstance(result, float)
        assert result >= 0


# ---------------------------------------------------------------------------
# Unit tests: data fetchers
# ---------------------------------------------------------------------------

class TestFetchIndexSnapshot:
    def test_returns_snapshot_with_change(self, populated_conn: sqlite3.Connection) -> None:
        result = _fetch_index_snapshot(populated_conn, "SPY")

        assert result is not None
        assert result["ticker"] == "SPY"
        assert result["close"] == 585.20
        assert result["change_pct"] > 0  # 585.20 vs 583.00

    def test_returns_none_for_missing_ticker(self, conn: sqlite3.Connection) -> None:
        result = _fetch_index_snapshot(conn, "NOTEXIST")
        assert result is None


class TestFetchCryptoSnapshot:
    def test_btc_snapshot_from_1d_bars(self, populated_conn: sqlite3.Connection) -> None:
        result = _fetch_crypto_snapshot(populated_conn, "BTC-USD")

        assert result is not None
        assert result["symbol"] == "BTC-USD"
        assert result["close"] == 88500.0
        # 88500/87000 - 1 = ~1.72%
        assert result["change_pct"] > 1.0

    def test_eth_snapshot(self, populated_conn: sqlite3.Connection) -> None:
        result = _fetch_crypto_snapshot(populated_conn, "ETH-USD")

        assert result is not None
        assert result["close"] == 2100.0
        # 2100/2050 - 1 = ~2.44%
        assert result["change_pct"] > 2.0

    def test_returns_none_for_missing_symbol(self, conn: sqlite3.Connection) -> None:
        result = _fetch_crypto_snapshot(conn, "DOGE-USD")
        assert result is None


class TestFetchOpenPositions:
    def test_returns_open_positions_only(self, populated_conn: sqlite3.Connection) -> None:
        positions = _fetch_open_positions(populated_conn)

        assert len(positions) == 2
        tickers = {p["ticker"] for p in positions}
        assert "AAPL" in tickers
        assert "NVDA" in tickers
        assert "MSFT" not in tickers  # closed

    def test_position_fields(self, populated_conn: sqlite3.Connection) -> None:
        positions = _fetch_open_positions(populated_conn)

        aapl = next(p for p in positions if p["ticker"] == "AAPL")
        assert aapl["direction"] == "long"
        assert aapl["entry_price"] == 225.00
        assert aapl["pnl_pct"] == 3.5
        assert aapl["size_pct"] == 8.0

    def test_empty_when_no_open(self, conn: sqlite3.Connection) -> None:
        positions = _fetch_open_positions(conn)
        assert positions == []


class TestFetchCounterTradeSignals:
    def test_returns_recent_signals_only(self, populated_conn: sqlite3.Connection) -> None:
        signals = _fetch_counter_trade_signals(populated_conn)

        # SOL signal is >24h old, should be excluded
        coins = {s["coin"] for s in signals}
        assert "BTC" in coins
        assert "ETH" in coins
        assert "SOL" not in coins

    def test_signal_fields(self, populated_conn: sqlite3.Connection) -> None:
        signals = _fetch_counter_trade_signals(populated_conn)

        btc = next(s for s in signals if s["coin"] == "BTC")
        assert btc["direction"] == "short"
        assert btc["entry_price"] is None
        assert btc["wallet"] == "machibro"

    def test_legacy_schema_still_supported(self, conn: sqlite3.Connection) -> None:
        conn.execute("DROP TABLE hyperliquid_counter_signals")
        conn.execute(
            """
            CREATE TABLE hyperliquid_counter_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL,
                signal_ts TEXT NOT NULL,
                wallet_label TEXT
            )
            """
        )
        now_ts = dt.datetime.now(dt.timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO hyperliquid_counter_signals
                (coin, direction, entry_price, signal_ts, wallet_label)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("BTC", "short", 88000, now_ts, "machibro"),
        )
        conn.commit()

        signals = _fetch_counter_trade_signals(conn)
        assert signals[0]["direction"] == "short"
        assert signals[0]["entry_price"] == 88000
        assert signals[0]["wallet"] == "machibro"

    def test_empty_when_no_signals(self, conn: sqlite3.Connection) -> None:
        # Table doesn't exist yet in bare conn
        signals = _fetch_counter_trade_signals(conn)
        assert signals == []


class TestFetchEconomicEventsToday:
    @patch("trader_koo.notifications.morning_summary.dt")
    def test_returns_high_impact_events(self, mock_dt: MagicMock) -> None:
        # Mock to a known date with FOMC
        mock_dt.datetime = dt.datetime
        mock_dt.timedelta = dt.timedelta
        mock_dt.timezone = dt.timezone

        # Patch the actual fetch function
        with patch(
            "trader_koo.catalyst_data.fetch_economic_calendar",
            return_value=[
                {"date": "2026-03-26", "event": "FOMC Decision", "impact": "high"},
                {"date": "2026-03-26", "event": "Minor report", "impact": "low"},
            ],
        ):
            events = _fetch_economic_events_today()

        # Only high/medium impact events should be included
        assert len(events) == 1
        assert events[0]["event"] == "FOMC Decision"

    def test_handles_missing_module_gracefully(self) -> None:
        with patch(
            "trader_koo.notifications.morning_summary.dt"
        ) as mock_dt:
            mock_dt.datetime = dt.datetime
            mock_dt.timedelta = dt.timedelta
            mock_dt.timezone = dt.timezone
            # If the module import fails, should return empty list
            with patch.dict("sys.modules", {"trader_koo.catalyst_data": None}):
                events = _fetch_economic_events_today()
                assert events == []


class TestFetchPaperTradeStats:
    def test_counts_open_and_closed(self, populated_conn: sqlite3.Connection) -> None:
        stats = _fetch_paper_trade_stats(populated_conn)

        assert stats["open_count"] == 2
        assert isinstance(stats["win_rate_pct"], float)
        assert isinstance(stats["portfolio_value"], float)


# ---------------------------------------------------------------------------
# Integration test: full message generation
# ---------------------------------------------------------------------------

class TestGenerateMorningSummary:
    @patch("trader_koo.notifications.morning_summary._fetch_fear_greed", return_value={"composite_score": 55})
    @patch("trader_koo.notifications.morning_summary._fetch_economic_events_today", return_value=[])
    def test_full_summary_contains_all_sections(
        self,
        mock_econ: MagicMock,
        mock_fg: MagicMock,
        db_path: Path,
        populated_conn: sqlite3.Connection,
        tmp_path: Path,
    ) -> None:
        # Close populated_conn so generate_morning_summary can open its own
        populated_conn.close()

        # Create a minimal report
        report_dir = tmp_path / "reports"
        report_dir.mkdir()

        with patch(
            "trader_koo.backend.services.report_loader.latest_daily_report_json",
            return_value=("2026-03-25", {}),
        ):
            message = generate_morning_summary(db_path, report_dir)

        # Verify key sections are present
        assert "Morning Briefing" in message
        assert "Market Snapshot" in message
        assert "SPY" in message
        assert "QQQ" in message
        assert "VIX" in message

        # Overnight moves
        assert "Overnight Moves" in message
        assert "BTC" in message
        assert "ETH" in message
        assert "Gold" in message
        assert "DXY" in message

        # Paper trades with open positions
        assert "Paper Trades" in message
        assert "Open positions" in message
        assert "AAPL" in message
        assert "NVDA" in message

        # Counter-trade signals
        assert "Counter-Trade" in message

        # Market open time
        assert "US market opens in" in message

    @patch("trader_koo.notifications.morning_summary._fetch_fear_greed", return_value=None)
    @patch("trader_koo.notifications.morning_summary._fetch_economic_events_today", return_value=[])
    def test_summary_with_empty_db(
        self,
        mock_econ: MagicMock,
        mock_fg: MagicMock,
        db_path: Path,
        tmp_path: Path,
    ) -> None:
        report_dir = tmp_path / "reports"
        report_dir.mkdir()

        with patch(
            "trader_koo.backend.services.report_loader.latest_daily_report_json",
            return_value=("2026-03-25", {}),
        ):
            message = generate_morning_summary(db_path, report_dir)

        # Should still produce a valid message without crashing
        assert "Morning Briefing" in message
        assert "Paper Trades" in message
