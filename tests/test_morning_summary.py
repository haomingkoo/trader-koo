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

from trader_koo.paper_trades import PAPER_TRADE_CAMPAIGN_ID
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
    _report_staleness,
    _vix_label,
    generate_morning_summary,
    send_morning_summary,
)
from trader_koo.report.utils import expected_price_date


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Create a temporary SQLite DB with all required schemas."""
    path = tmp_path / "test_summary.db"
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(f"""
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
            position_size_pct REAL DEFAULT 8.0,
            campaign_id TEXT NOT NULL DEFAULT '{PAPER_TRADE_CAMPAIGN_ID}'
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
            action TEXT,
            score REAL,
            reasons_json TEXT,
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
             their_notional_usd, confidence, reasoning, action, score, reasons_json, signal_ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("machibro", "BTC", "short", "long", 1.2, 25, 88_000, 82.0, "[COUNTER] score=8", "COUNTER", 8, '["large notional"]', now_ts),
            ("machibro", "ETH", "long", "short", 12.0, 15, 2_050, 71.0, "[COUNTER] score=7", "COUNTER", 7, '["crowded shorts"]', now_ts),
            ("machibro", "DOGE", "short", "long", 10.0, 5, 200, 30.0, "[SKIP] score=0", "SKIP", 0, '["skip list"]', now_ts),
            ("machibro", "SOL", "short", "long", 100.0, 20, 180, 76.0, "[COUNTER] score=8", "COUNTER", 8, '["old"]', old_ts),  # older than 24h
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
        assert "DOGE" not in coins
        assert "SOL" not in coins

    def test_signal_fields(self, populated_conn: sqlite3.Connection) -> None:
        signals = _fetch_counter_trade_signals(populated_conn)

        btc = next(s for s in signals if s["coin"] == "BTC")
        assert btc["direction"] == "short"
        assert btc["entry_price"] is None
        assert btc["wallet"] == "machibro"
        assert btc["action"] == "COUNTER"
        assert btc["score"] == 8
        assert btc["reasons"] == ["large notional"]

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

    def test_ignores_trades_from_other_campaigns(self, populated_conn: sqlite3.Connection) -> None:
        populated_conn.executemany(
            "INSERT INTO paper_trades (ticker, direction, entry_price, entry_date, exit_date, status, pnl_pct, unrealized_pnl_pct, position_size_pct, campaign_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'paper-v1')",
            [
                ("KO", "long", 60.00, "2026-03-01", "2026-03-10", "closed", -4.0, None, 8.0),
                ("PEP", "long", 170.00, "2026-03-02", None, "open", None, 2.0, 8.0),
            ],
        )
        stats = _fetch_paper_trade_stats(populated_conn)
        positions = _fetch_open_positions(populated_conn)

        assert stats["open_count"] == 2
        assert stats["win_rate_pct"] == 100.0
        assert {p["ticker"] for p in positions} == {"AAPL", "NVDA"}


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
        assert "Hyperliquid Research Signals" in message
        assert "Research only" in message

        # Market open time
        assert "US market opens in" in message

    @patch("trader_koo.notifications.morning_summary._fetch_fear_greed", return_value=None)
    @patch("trader_koo.notifications.morning_summary._fetch_economic_events_today", return_value=[])
    def test_summary_prefers_compact_research_suggestions(
        self,
        mock_econ: MagicMock,
        mock_fg: MagicMock,
        db_path: Path,
        populated_conn: sqlite3.Connection,
        tmp_path: Path,
    ) -> None:
        populated_conn.close()
        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        report = {
            "latest_data": {"price_date": expected_price_date().isoformat()},
            "signals": {
                "suggestions": {
                    "items": [
                        {
                            "ticker": "AAPL",
                            "action": "Paper Long",
                            "conviction": "Medium",
                            "probability_pct": 61.2,
                            "sample_size": 9,
                            "why": ["Strong *setup* [external] confirmation."],
                            "invalidation": "Invalid below support near 180.00.",
                        }
                    ],
                    "note": "Research suggestions only, not financial advice.",
                },
                "setup_quality_top": [
                    {
                        "ticker": "OLD",
                        "setup_tier": "A",
                        "signal_bias": "bullish",
                    }
                ],
            }
        }

        with patch(
            "trader_koo.backend.services.report_loader.latest_daily_report_json",
            return_value=("2026-03-25", report),
        ):
            message = generate_morning_summary(db_path, report_dir)

        assert "Top Research Suggestions" in message
        assert "Research only. Not financial advice." in message
        assert "AAPL" in message
        assert "61% n=9" in message
        assert "Strong setup (external) confirmation." in message
        assert "Invalid below support near 180.00." in message
        assert "Top 1 Setups Today" not in message
        assert "OLD" not in message

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


# ---------------------------------------------------------------------------
# Report staleness guard
# ---------------------------------------------------------------------------

# 13 Nov 2026 08:00 SGT: the incident where 30 Oct setups went out as "today".
_INCIDENT_NOW = dt.datetime(2026, 11, 13, 0, 0, tzinfo=dt.timezone.utc)


def _report_for(price_date: str) -> dict[str, Any]:
    return {
        "generated_ts": f"{price_date}T22:40:00Z",
        "latest_data": {"price_date": price_date},
        "signals": {
            "setup_quality_top": [
                {"ticker": "STALE", "setup_tier": "A", "signal_bias": "bullish"}
            ],
            "suggestions": {
                "items": [
                    {"ticker": "STALE", "action": "Paper Long", "conviction": "High"}
                ]
            },
            "green_barrier_hits": [
                {"ticker": "STALE", "timeframe": "monthly", "value": -100.0, "asof": price_date}
            ],
        },
    }


class TestReportStaleness:
    def test_report_weeks_old_is_stale_with_actual_date(self) -> None:
        notice = _report_staleness(_report_for("2026-10-30"), _INCIDENT_NOW)
        assert notice is not None
        assert "2026-10-30" in notice
        assert "stale" in notice.lower()

    def test_report_for_expected_session_is_fresh(self) -> None:
        # At 00:00 UTC Fri 13 Nov the nightly run for Thu 12 Nov may still be
        # in flight, so Wed 11 Nov is the newest session that is owed.
        assert _report_staleness(_report_for("2026-11-11"), _INCIDENT_NOW) is None
        assert _report_staleness(_report_for("2026-11-12"), _INCIDENT_NOW) is None

    def test_friday_report_is_fresh_on_monday_morning(self) -> None:
        monday_sgt = dt.datetime(2026, 11, 16, 0, 0, tzinfo=dt.timezone.utc)
        assert _report_staleness(_report_for("2026-11-13"), monday_sgt) is None

    def test_one_session_behind_owed_session_is_stale(self) -> None:
        tue_sgt = dt.datetime(2026, 11, 18, 0, 0, tzinfo=dt.timezone.utc)
        # Owed: Mon 16 Nov. A Fri 13 Nov report is a session behind.
        notice = _report_staleness(_report_for("2026-11-13"), tue_sgt)
        assert notice is not None
        assert "2026-11-13" in notice

    def test_missing_report_is_explicit(self) -> None:
        notice = _report_staleness({}, _INCIDENT_NOW)
        assert notice is not None
        assert "no published daily report" in notice.lower()

    def test_report_without_market_date_is_explicit(self) -> None:
        report = _report_for("2026-11-12")
        report["latest_data"] = {}
        notice = _report_staleness(report, _INCIDENT_NOW)
        assert notice is not None
        assert "market date" in notice.lower()

    @patch("trader_koo.notifications.morning_summary._fetch_fear_greed", return_value=None)
    @patch("trader_koo.notifications.morning_summary._fetch_economic_events_today", return_value=[])
    def test_stale_report_setups_are_suppressed_with_banner(
        self,
        mock_econ: MagicMock,
        mock_fg: MagicMock,
        db_path: Path,
        tmp_path: Path,
    ) -> None:
        message = generate_morning_summary(
            db_path, tmp_path, report=_report_for("2026-10-30")
        )
        assert "Stale report" in message
        assert "2026-10-30" in message
        assert "STALE" not in message
        assert "Setups Today" not in message
        assert "Top Research Suggestions" not in message
        assert "Green Barrier Current Conditions" not in message
        # Live sections still render.
        assert "Market Snapshot" in message
        assert "Paper Trades" in message

    @patch("trader_koo.notifications.morning_summary._fetch_fear_greed", return_value=None)
    @patch("trader_koo.notifications.morning_summary._fetch_economic_events_today", return_value=[])
    def test_fresh_report_has_no_stale_banner(
        self,
        mock_econ: MagicMock,
        mock_fg: MagicMock,
        db_path: Path,
        tmp_path: Path,
    ) -> None:
        fresh = expected_price_date().isoformat()
        message = generate_morning_summary(db_path, tmp_path, report=_report_for(fresh))
        assert "Stale report" not in message
        assert "STALE" in message

    def test_send_skips_charts_for_stale_report(self, db_path: Path, tmp_path: Path) -> None:
        with (
            patch("trader_koo.notifications.morning_summary.is_configured", return_value=True),
            patch("trader_koo.notifications.morning_summary.generate_morning_summary", return_value="brief"),
            patch("trader_koo.notifications.morning_summary.send_message", return_value=True) as send_text,
            patch("trader_koo.notifications.morning_summary.send_photo", return_value=True) as send_image,
            patch(
                "trader_koo.backend.services.report_loader.latest_daily_report_json",
                return_value=(tmp_path / "daily_report_latest.json", _report_for("2026-10-30")),
            ),
        ):
            assert send_morning_summary(db_path, tmp_path) is True
        send_text.assert_called_once_with("brief")
        send_image.assert_not_called()
