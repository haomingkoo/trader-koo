"""Tests for trader_koo.notifications.market_monitor.

Validates schema creation, snapshot archival, spike detection logic,
alert formatting, and the admin endpoint.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trader_koo.notifications.market_monitor import (
    _format_crypto_alert,
    _format_polymarket_alert,
    _format_price,
    _format_volume,
    detect_crypto_spikes,
    detect_polymarket_spikes,
    ensure_polymarket_schema,
    get_recent_spikes,
    send_spike_alerts,
    snapshot_polymarket,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Create a temporary SQLite DB with required schemas."""
    path = tmp_path / "test_monitor.db"
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    ensure_polymarket_schema(conn)
    # Create crypto_bars table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crypto_bars (
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            interval TEXT NOT NULL DEFAULT '1m',
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY (symbol, timestamp, interval)
        )
    """)
    conn.commit()
    conn.close()
    return path


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_ensure_polymarket_schema_creates_table(self, db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='polymarket_snapshots'"
        ).fetchone()
        conn.close()
        assert row is not None

    def test_ensure_polymarket_schema_idempotent(self, db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path))
        ensure_polymarket_schema(conn)
        ensure_polymarket_schema(conn)
        conn.close()

    def test_ensure_polymarket_schema_creates_index(self, db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_poly_snap_slug_ts'"
        ).fetchone()
        conn.close()
        assert row is not None


# ---------------------------------------------------------------------------
# Snapshot tests
# ---------------------------------------------------------------------------

class TestSnapshotPolymarket:
    @patch("trader_koo.ml.external_data.fetch_polymarket_events")
    def test_snapshot_saves_active_markets(
        self, mock_fetch: MagicMock, db_path: Path,
    ) -> None:
        mock_fetch.return_value = [
            {
                "slug": "test-event",
                "title": "Test Event",
                "markets": [
                    {
                        "question": "Will X happen?",
                        "outcomes": ["Yes", "No"],
                        "prices_pct": [65.0, 35.0],
                        "volume": 1_000_000,
                        "active": True,
                    },
                ],
            },
        ]

        count = snapshot_polymarket(db_path)

        assert count == 1
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM polymarket_snapshots").fetchone()
        conn.close()
        assert row["event_slug"] == "test-event"
        assert row["probability"] == 65.0

    @patch("trader_koo.ml.external_data.fetch_polymarket_events")
    def test_snapshot_skips_inactive_markets(
        self, mock_fetch: MagicMock, db_path: Path,
    ) -> None:
        mock_fetch.return_value = [
            {
                "slug": "resolved-event",
                "title": "Resolved Event",
                "markets": [
                    {
                        "question": "Resolved?",
                        "outcomes": ["Yes", "No"],
                        "prices_pct": [100.0, 0.0],
                        "volume": 500_000,
                        "active": False,
                    },
                ],
            },
        ]

        count = snapshot_polymarket(db_path)
        assert count == 0

    @patch("trader_koo.ml.external_data.fetch_polymarket_events")
    def test_snapshot_returns_zero_on_empty(
        self, mock_fetch: MagicMock, db_path: Path,
    ) -> None:
        mock_fetch.return_value = []
        assert snapshot_polymarket(db_path) == 0

    @patch("trader_koo.ml.external_data.fetch_polymarket_events")
    def test_snapshot_handles_fetch_error(
        self, mock_fetch: MagicMock, db_path: Path,
    ) -> None:
        mock_fetch.side_effect = RuntimeError("API down")
        assert snapshot_polymarket(db_path) == 0


# ---------------------------------------------------------------------------
# Polymarket spike detection tests
# ---------------------------------------------------------------------------

class TestDetectPolymarketSpikes:
    def _seed_snapshots(
        self,
        db_path: Path,
        old_prob: float,
        new_prob: float,
        hours_ago: int = 8,
    ) -> None:
        """Insert two snapshot rows: one old and one current."""
        conn = sqlite3.connect(str(db_path))
        now = dt.datetime.now(dt.timezone.utc)
        old_ts = (now - dt.timedelta(hours=hours_ago)).isoformat()
        new_ts = now.isoformat()

        conn.execute(
            """
            INSERT INTO polymarket_snapshots
                (event_slug, event_title, market_question, probability, volume, snapshot_ts)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("test-slug", "Test Title", "Will it happen?", old_prob, 2_000_000, old_ts),
        )
        conn.execute(
            """
            INSERT INTO polymarket_snapshots
                (event_slug, event_title, market_question, probability, volume, snapshot_ts)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("test-slug", "Test Title", "Will it happen?", new_prob, 2_200_000, new_ts),
        )
        conn.commit()
        conn.close()

    def test_detects_upward_spike(self, db_path: Path) -> None:
        self._seed_snapshots(db_path, old_prob=10.0, new_prob=25.0)

        spikes = detect_polymarket_spikes(db_path, lookback_hours=6, threshold_pct=5.0)

        assert len(spikes) == 1
        assert spikes[0]["direction"] == "up"
        assert spikes[0]["change_pct"] == 15.0
        assert spikes[0]["old_prob"] == 10.0
        assert spikes[0]["new_prob"] == 25.0

    def test_detects_downward_spike(self, db_path: Path) -> None:
        self._seed_snapshots(db_path, old_prob=60.0, new_prob=45.0)

        spikes = detect_polymarket_spikes(db_path, lookback_hours=6, threshold_pct=5.0)

        assert len(spikes) == 1
        assert spikes[0]["direction"] == "down"
        assert spikes[0]["change_pct"] == -15.0

    def test_no_spike_below_threshold(self, db_path: Path) -> None:
        self._seed_snapshots(db_path, old_prob=50.0, new_prob=53.0)

        spikes = detect_polymarket_spikes(db_path, lookback_hours=6, threshold_pct=5.0)
        assert len(spikes) == 0

    def test_empty_db_returns_no_spikes(self, db_path: Path) -> None:
        spikes = detect_polymarket_spikes(db_path, lookback_hours=6, threshold_pct=5.0)
        assert spikes == []


# ---------------------------------------------------------------------------
# Crypto spike detection tests
# ---------------------------------------------------------------------------

class TestDetectCryptoSpikes:
    def _seed_crypto_bars(
        self,
        db_path: Path,
        symbol: str,
        old_price: float,
        new_price: float,
        hours_ago: int = 6,
    ) -> None:
        """Insert two crypto bars: one old and one current."""
        conn = sqlite3.connect(str(db_path))
        now = dt.datetime.now(dt.timezone.utc)
        old_ts = (now - dt.timedelta(hours=hours_ago)).isoformat()
        new_ts = now.isoformat()

        for ts, price in [(old_ts, old_price), (new_ts, new_price)]:
            conn.execute(
                """
                INSERT OR IGNORE INTO crypto_bars
                    (symbol, timestamp, interval, open, high, low, close, volume)
                VALUES (?, ?, '1m', ?, ?, ?, ?, ?)
                """,
                (symbol, ts, price, price * 1.01, price * 0.99, price, 1000.0),
            )
        conn.commit()
        conn.close()

    @patch("trader_koo.crypto.binance_oi.fetch_open_interest_history")
    def test_detects_price_spike(
        self, mock_oi: MagicMock, db_path: Path,
    ) -> None:
        mock_oi.return_value = []
        self._seed_crypto_bars(db_path, "BTC-USD", 80000.0, 85000.0)

        spikes = detect_crypto_spikes(db_path, lookback_hours=4)

        assert len(spikes) == 1
        assert spikes[0]["symbol"] == "BTC-USD"
        assert spikes[0]["price_spike"] is True
        assert spikes[0]["price_change_pct"] > 5.0

    @patch("trader_koo.crypto.binance_oi.fetch_open_interest_history")
    def test_no_spike_below_threshold(
        self, mock_oi: MagicMock, db_path: Path,
    ) -> None:
        mock_oi.return_value = []
        self._seed_crypto_bars(db_path, "BTC-USD", 80000.0, 81000.0)

        spikes = detect_crypto_spikes(db_path, lookback_hours=4)
        assert len(spikes) == 0

    def test_empty_db_returns_no_spikes(self, db_path: Path) -> None:
        spikes = detect_crypto_spikes(db_path, lookback_hours=4)
        assert spikes == []

    def test_no_crypto_bars_table(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(path))
        conn.close()

        spikes = detect_crypto_spikes(path, lookback_hours=4)
        assert spikes == []


# ---------------------------------------------------------------------------
# Formatting tests
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_format_volume_millions(self) -> None:
        assert _format_volume(2_200_000) == "$2.2M"

    def test_format_volume_thousands(self) -> None:
        assert _format_volume(500_000) == "$500.0K"

    def test_format_volume_small(self) -> None:
        assert _format_volume(999) == "$999"

    def test_format_volume_none(self) -> None:
        assert _format_volume(None) == "N/A"

    def test_format_volume_zero(self) -> None:
        assert _format_volume(0) == "N/A"

    def test_format_price_large(self) -> None:
        assert _format_price(85200.0) == "$85,200.00"

    def test_format_price_small(self) -> None:
        assert _format_price(0.1234) == "$0.1234"

    def test_format_polymarket_alert_contains_key_info(self) -> None:
        spike = {
            "event_title": "Fed rate cut March 31",
            "question": "Will it happen?",
            "old_prob": 6.0,
            "new_prob": 15.0,
            "change_pct": 9.0,
            "direction": "up",
            "volume": 2_200_000,
            "lookback_hours": 6,
        }
        msg = _format_polymarket_alert(spike)
        assert "Prediction Market Spike" in msg
        assert "Fed rate cut March 31" in msg
        assert "6%" in msg
        assert "15%" in msg
        assert "$2.2M" in msg

    def test_format_crypto_alert_contains_key_info(self) -> None:
        spike = {
            "symbol": "BTC-USD",
            "old_price": 85200.0,
            "new_price": 89800.0,
            "price_change_pct": 5.4,
            "direction": "up",
            "lookback_hours": 4,
            "price_spike": True,
            "oi_spike": True,
            "oi_change_pct": 12.0,
        }
        msg = _format_crypto_alert(spike)
        assert "Crypto Alert" in msg
        assert "BTC-USD" in msg
        assert "$85,200.00" in msg
        assert "$89,800.00" in msg
        assert "OI change" in msg
        assert "Volume surge detected" in msg

    def test_format_crypto_alert_no_oi(self) -> None:
        spike = {
            "symbol": "ETH-USD",
            "old_price": 3000.0,
            "new_price": 3200.0,
            "price_change_pct": 6.7,
            "direction": "up",
            "lookback_hours": 4,
            "price_spike": True,
            "oi_spike": False,
        }
        msg = _format_crypto_alert(spike)
        assert "OI change" not in msg


# ---------------------------------------------------------------------------
# Alert sending tests
# ---------------------------------------------------------------------------

class TestSendSpikeAlerts:
    @patch("trader_koo.notifications.telegram.is_configured")
    @patch("trader_koo.notifications.telegram.send_message")
    def test_sends_alerts_when_spikes_found(
        self,
        mock_send: MagicMock,
        mock_configured: MagicMock,
        db_path: Path,
    ) -> None:
        mock_configured.return_value = True
        mock_send.return_value = True

        # Seed data that will produce a polymarket spike
        conn = sqlite3.connect(str(db_path))
        now = dt.datetime.now(dt.timezone.utc)
        old_ts = (now - dt.timedelta(hours=8)).isoformat()
        new_ts = now.isoformat()
        conn.execute(
            """
            INSERT INTO polymarket_snapshots
                (event_slug, event_title, market_question, probability, volume, snapshot_ts)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("test-slug", "Test Event", "Will it?", 10.0, 1_000_000, old_ts),
        )
        conn.execute(
            """
            INSERT INTO polymarket_snapshots
                (event_slug, event_title, market_question, probability, volume, snapshot_ts)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("test-slug", "Test Event", "Will it?", 25.0, 1_200_000, new_ts),
        )
        conn.commit()
        conn.close()

        result = send_spike_alerts(db_path, Path("/tmp"))
        assert result == 1
        mock_send.assert_called_once()

    @patch("trader_koo.notifications.telegram.is_configured")
    def test_skips_when_telegram_not_configured(
        self, mock_configured: MagicMock, db_path: Path,
    ) -> None:
        mock_configured.return_value = False

        result = send_spike_alerts(db_path, Path("/tmp"))
        assert result == 0


# ---------------------------------------------------------------------------
# Recent spikes query tests
# ---------------------------------------------------------------------------

class TestGetRecentSpikes:
    def test_returns_structure(self, db_path: Path) -> None:
        result = get_recent_spikes(db_path, hours=24)

        assert result["ok"] is True
        assert "polymarket_spikes" in result
        assert "crypto_spikes" in result
        assert "total_spikes" in result
        assert "checked_at" in result
