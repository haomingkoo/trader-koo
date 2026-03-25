"""Tests for the Telegram price alert engine (REST polling approach)."""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from trader_koo.notifications.alert_engine import (
    DEFAULT_COOLDOWN_SEC,
    MAX_REPORT_TICKERS,
    POLL_INTERVAL_SEC,
    AlertEngine,
    _ensure_telegram_alerts_table,
    _is_us_market_hours,
)


# ------------------------------------------------------------------
# _is_us_market_hours
# ------------------------------------------------------------------


class TestIsUsMarketHours:
    """Verify market-hours gate logic."""

    def test_weekday_within_hours(self) -> None:
        # Wednesday 2026-03-18 14:30 UTC = 09:30 ET (EST, UTC-5)
        now = dt.datetime(2026, 3, 18, 14, 30, tzinfo=dt.timezone.utc)

        assert _is_us_market_hours(now) is True

    def test_weekday_before_open(self) -> None:
        # Wednesday 2026-03-18 13:00 UTC = 08:00 ET
        now = dt.datetime(2026, 3, 18, 13, 0, tzinfo=dt.timezone.utc)

        assert _is_us_market_hours(now) is False

    def test_weekday_after_close(self) -> None:
        # Wednesday 2026-03-18 21:30 UTC = 16:30 ET
        now = dt.datetime(2026, 3, 18, 21, 30, tzinfo=dt.timezone.utc)

        assert _is_us_market_hours(now) is False

    def test_weekend_rejected(self) -> None:
        # Saturday 2026-03-21 15:00 UTC = 10:00 ET
        now = dt.datetime(2026, 3, 21, 15, 0, tzinfo=dt.timezone.utc)

        assert _is_us_market_hours(now) is False

    def test_market_close_boundary(self) -> None:
        # Wednesday 2026-03-18 21:00 UTC = 16:00 ET (exactly at close)
        now = dt.datetime(2026, 3, 18, 21, 0, tzinfo=dt.timezone.utc)

        assert _is_us_market_hours(now) is True


# ------------------------------------------------------------------
# _ensure_telegram_alerts_table
# ------------------------------------------------------------------


class TestEnsureTelegramAlertsTable:
    """Verify schema creation is idempotent."""

    def test_creates_table(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))

        _ensure_telegram_alerts_table(conn)

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]

        assert "telegram_alerts" in table_names
        conn.close()

    def test_idempotent(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))

        _ensure_telegram_alerts_table(conn)
        _ensure_telegram_alerts_table(conn)  # should not raise

        conn.close()


# ------------------------------------------------------------------
# AlertEngine — configuration
# ------------------------------------------------------------------


class TestAlertEngineConfig:
    """Verify engine initialization and configuration constants."""

    def test_poll_interval_is_120_seconds(self) -> None:
        assert POLL_INTERVAL_SEC == 120

    def test_max_poll_tickers_is_10(self) -> None:
        assert MAX_REPORT_TICKERS == 10

    def test_cooldown_is_4_hours(self) -> None:
        assert DEFAULT_COOLDOWN_SEC == 4 * 3600

    def test_constructor_accepts_finnhub_api_key(
        self, tmp_path: Path,
    ) -> None:
        engine = AlertEngine(
            db_path=tmp_path / "test.db",
            report_dir=tmp_path / "reports",
            finnhub_api_key="test_key_placeholder",
        )

        assert engine._finnhub_api_key == "test_key_placeholder"

    def test_constructor_falls_back_to_env_var(
        self, tmp_path: Path,
    ) -> None:
        with patch.dict(
            "os.environ", {"FINNHUB_API_KEY": "env_key_placeholder"},
        ):
            engine = AlertEngine(
                db_path=tmp_path / "test.db",
                report_dir=tmp_path / "reports",
            )

            assert engine._finnhub_api_key == "env_key_placeholder"


# ------------------------------------------------------------------
# AlertEngine._check_tick
# ------------------------------------------------------------------


class TestCheckTick:
    """Verify proximity detection and cooldown logic."""

    def _make_engine(
        self,
        tmp_path: Path,
        watchlist: dict[str, list[dict[str, Any]]],
    ) -> AlertEngine:
        engine = AlertEngine(
            db_path=tmp_path / "test.db",
            report_dir=tmp_path / "reports",
            finnhub_api_key="placeholder",
        )
        engine._watchlist = watchlist
        return engine

    @patch("trader_koo.notifications.alert_engine.send_price_alert")
    def test_fires_alert_within_proximity(
        self, mock_send: MagicMock, tmp_path: Path,
    ) -> None:
        mock_send.return_value = True
        engine = self._make_engine(tmp_path, {
            "AAPL": [
                {
                    "level": 200.0,
                    "level_type": "resistance",
                    "setup_tier": "A",
                    "bias": "bullish",
                },
            ],
        })

        # Price is 200.50 - above 200.0 resistance = breakout (within 1%)
        # Telegram only fires for breakouts/breakdowns
        engine._check_tick("AAPL", 200.50)

        mock_send.assert_called_once()
        call_kwargs = mock_send.call_args
        assert call_kwargs[1]["ticker"] == "AAPL"
        assert call_kwargs[1]["alert_type"] == "breakout_above_resistance"

    @patch("trader_koo.notifications.alert_engine.send_price_alert")
    def test_no_alert_outside_proximity(
        self, mock_send: MagicMock, tmp_path: Path,
    ) -> None:
        engine = self._make_engine(tmp_path, {
            "AAPL": [
                {
                    "level": 200.0,
                    "level_type": "support",
                    "setup_tier": "A",
                    "bias": "bullish",
                },
            ],
        })

        # Price is 190.0 — 5% from 200.0 (way outside 1%)
        engine._check_tick("AAPL", 190.0)

        mock_send.assert_not_called()

    @patch("trader_koo.notifications.alert_engine.send_price_alert")
    def test_cooldown_suppresses_duplicate(
        self, mock_send: MagicMock, tmp_path: Path,
    ) -> None:
        mock_send.return_value = True
        engine = self._make_engine(tmp_path, {
            "AAPL": [
                {
                    "level": 200.0,
                    "level_type": "support",
                    "setup_tier": "A",
                    "bias": "bullish",
                },
            ],
        })

        engine._check_tick("AAPL", 199.50)
        engine._check_tick("AAPL", 199.60)

        # Second call should be suppressed by cooldown
        assert mock_send.call_count == 1

    @patch("trader_koo.notifications.alert_engine.send_price_alert")
    def test_breakout_above_resistance(
        self, mock_send: MagicMock, tmp_path: Path,
    ) -> None:
        mock_send.return_value = True
        engine = self._make_engine(tmp_path, {
            "TSLA": [
                {
                    "level": 300.0,
                    "level_type": "resistance",
                    "setup_tier": "B",
                    "bias": "bullish",
                },
            ],
        })

        # Price is 301.0 — 0.33% above 300.0 resistance
        engine._check_tick("TSLA", 301.0)

        mock_send.assert_called_once()
        assert mock_send.call_args[1]["alert_type"] == "breakout_above_resistance"

    @patch("trader_koo.notifications.alert_engine.send_price_alert")
    def test_breakdown_below_support(
        self, mock_send: MagicMock, tmp_path: Path,
    ) -> None:
        mock_send.return_value = True
        engine = self._make_engine(tmp_path, {
            "NVDA": [
                {
                    "level": 100.0,
                    "level_type": "support",
                    "setup_tier": "A",
                    "bias": "bearish",
                },
            ],
        })

        # Price is 99.50 — 0.5% below 100.0 support
        engine._check_tick("NVDA", 99.50)

        mock_send.assert_called_once()
        assert mock_send.call_args[1]["alert_type"] == "breakdown_below_support"


# ------------------------------------------------------------------
# AlertEngine._persist_alert
# ------------------------------------------------------------------


class TestPersistAlert:
    """Verify alert records are written to SQLite."""

    def test_writes_alert_to_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        _ensure_telegram_alerts_table(conn)
        conn.close()

        engine = AlertEngine(
            db_path=db_path,
            report_dir=tmp_path / "reports",
            finnhub_api_key="placeholder",
        )

        engine._persist_alert(
            ticker="AAPL",
            level=200.0,
            price=199.50,
            alert_type="approaching_support",
            setup_tier="A",
            bias="bullish",
        )

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM telegram_alerts"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        row = dict(rows[0])
        assert row["ticker"] == "AAPL"
        assert row["level"] == 200.0
        assert row["price"] == 199.50
        assert row["alert_type"] == "approaching_support"
        assert row["setup_tier"] == "A"
        assert row["bias"] == "bullish"
        assert row["sent_at"]  # non-empty ISO timestamp


# ------------------------------------------------------------------
# AlertEngine._poll_finnhub_quote
# ------------------------------------------------------------------


class TestPollFinnhubQuote:
    """Verify REST quote polling logic (mocking HTTP)."""

    def test_returns_price_on_success(self, tmp_path: Path) -> None:
        engine = AlertEngine(
            db_path=tmp_path / "test.db",
            report_dir=tmp_path / "reports",
            finnhub_api_key="test_key",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "c": 199.50,  # current price
            "d": -1.23,
            "dp": -0.61,
            "h": 201.0,
            "l": 198.0,
            "o": 200.5,
            "pc": 200.73,
        }

        with patch("trader_koo.notifications.alert_engine.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            price = engine._poll_finnhub_quote("AAPL")

        assert price == 199.50

    def test_returns_none_on_api_error(self, tmp_path: Path) -> None:
        engine = AlertEngine(
            db_path=tmp_path / "test.db",
            report_dir=tmp_path / "reports",
            finnhub_api_key="test_key",
        )

        mock_response = MagicMock()
        mock_response.status_code = 429

        with patch("trader_koo.notifications.alert_engine.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            price = engine._poll_finnhub_quote("AAPL")

        assert price is None

    def test_returns_none_when_no_api_key(self, tmp_path: Path) -> None:
        engine = AlertEngine(
            db_path=tmp_path / "test.db",
            report_dir=tmp_path / "reports",
            finnhub_api_key="",
        )

        price = engine._poll_finnhub_quote("AAPL")

        assert price is None

    def test_returns_none_on_zero_price(self, tmp_path: Path) -> None:
        engine = AlertEngine(
            db_path=tmp_path / "test.db",
            report_dir=tmp_path / "reports",
            finnhub_api_key="test_key",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"c": 0, "d": None, "dp": None}

        with patch("trader_koo.notifications.alert_engine.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            price = engine._poll_finnhub_quote("INVALID")

        assert price is None


# ------------------------------------------------------------------
# AlertEngine.get_watchlist_summary
# ------------------------------------------------------------------


class TestWatchlistSummary:
    """Verify summary includes REST polling metadata."""

    def test_summary_includes_polling_config(self, tmp_path: Path) -> None:
        engine = AlertEngine(
            db_path=tmp_path / "test.db",
            report_dir=tmp_path / "reports",
            finnhub_api_key="placeholder",
        )
        engine._watchlist = {
            "AAPL": [{"level": 200.0, "level_type": "support", "setup_tier": "A", "bias": "bullish"}],
            "TSLA": [{"level": 300.0, "level_type": "resistance", "setup_tier": "B", "bias": "neutral"}],
        }

        summary = engine.get_watchlist_summary()

        assert summary["tickers"] == 2
        assert summary["levels"] == 2
        assert summary["max_tickers"] == MAX_REPORT_TICKERS
        assert summary["poll_interval_sec"] == POLL_INTERVAL_SEC
        assert "AAPL" in summary["ticker_list"]
        assert "TSLA" in summary["ticker_list"]


# ------------------------------------------------------------------
# AlertEngine.get_recent_alerts
# ------------------------------------------------------------------


class TestGetRecentAlerts:
    """Verify alert querying from SQLite."""

    def test_returns_alerts_ordered_by_id_desc(
        self, tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        _ensure_telegram_alerts_table(conn)
        for i in range(3):
            conn.execute(
                "INSERT INTO telegram_alerts "
                "(ticker, level, price, alert_type, setup_tier, bias, sent_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"TICK{i}", 100.0 + i, 99.0 + i, "approaching_support", "A", "bullish", "2026-03-20T00:00:00"),
            )
        conn.commit()
        conn.close()

        engine = AlertEngine(
            db_path=db_path,
            report_dir=tmp_path / "reports",
        )

        alerts = engine.get_recent_alerts(limit=2)

        assert len(alerts) == 2
        assert alerts[0]["ticker"] == "TICK2"
        assert alerts[1]["ticker"] == "TICK1"
