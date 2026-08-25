from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from unittest.mock import patch

from trader_koo.notifications.options_digest import (
    build_options_digest,
    generate_options_digest,
    send_options_digest,
)


NOW = dt.datetime(2026, 4, 24, 22, 0, tzinfo=dt.timezone.utc)


def _db_with_options(tmp_path: Path) -> Path:
    db_path = tmp_path / "options.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE options_iv (
            snapshot_ts TEXT NOT NULL,
            ticker TEXT NOT NULL,
            expiration TEXT NOT NULL,
            option_type TEXT NOT NULL,
            strike REAL NOT NULL,
            last_price REAL,
            bid REAL,
            ask REAL,
            implied_vol REAL,
            open_interest REAL,
            volume REAL,
            moneyness REAL
        );

        INSERT INTO options_iv VALUES
        ('2026-04-24T21:40:00Z', 'AMD', '2026-05-15', 'call', 100, 4.0, 3.9, 4.1, 0.42, 1200, 300, 1.0),
        ('2026-04-24T21:40:00Z', 'AMD', '2026-05-15', 'put', 95, 3.0, 2.9, 3.1, 0.45, 800, 120, 0.95),
        ('2026-04-23T21:40:00Z', 'OLD', '2026-05-15', 'call', 100, 40.0, 39.0, 41.0, 0.42, 12000, 3000, 1.0);
        """
    )
    conn.commit()
    conn.close()
    return db_path


def test_generate_options_digest_formats_top_proxy_rows(tmp_path: Path):
    db_path = _db_with_options(tmp_path)

    message = generate_options_digest(db_path, limit=3, now_utc=NOW)

    assert "Options Premium Proxy" in message
    assert "Not live signed flow" in message
    assert "AMD" in message
    assert "Call skew" in message
    assert "Vol net $84.0K" in message
    assert "OLD" not in message


def test_send_options_digest_skips_missing_db(tmp_path: Path):
    digest = build_options_digest(tmp_path / "missing.db")
    assert digest.has_data is False

    with (
        patch("trader_koo.notifications.options_digest.is_configured", return_value=True),
        patch("trader_koo.notifications.options_digest.send_message") as send_message,
    ):
        sent = send_options_digest(tmp_path / "missing.db")

    assert sent is False
    send_message.assert_not_called()


def test_send_options_digest_sends_when_configured(tmp_path: Path):
    db_path = _db_with_options(tmp_path)

    with (
        patch("trader_koo.notifications.options_digest.is_configured", return_value=True),
        patch("trader_koo.notifications.options_digest.send_message", return_value=True) as send_message,
    ):
        sent = send_options_digest(db_path, now_utc=NOW)

    assert sent is True
    assert "AMD" in send_message.call_args[0][0]


def test_send_options_digest_skips_stale_snapshot(tmp_path: Path):
    db_path = _db_with_options(tmp_path)

    with (
        patch("trader_koo.notifications.options_digest.is_configured", return_value=True),
        patch("trader_koo.notifications.options_digest.send_message") as send_message,
    ):
        sent = send_options_digest(
            db_path,
            now_utc=NOW + dt.timedelta(days=1),
        )

    assert sent is False
    send_message.assert_not_called()


def test_send_options_digest_sends_each_snapshot_once(tmp_path: Path):
    db_path = _db_with_options(tmp_path)

    with (
        patch("trader_koo.notifications.options_digest.is_configured", return_value=True),
        patch("trader_koo.notifications.options_digest.send_message", return_value=True) as send_message,
    ):
        first = send_options_digest(db_path, now_utc=NOW)
        second = send_options_digest(db_path, now_utc=NOW)

    assert first is True
    assert second is False
    send_message.assert_called_once()


def test_failed_options_digest_send_does_not_consume_snapshot(tmp_path: Path):
    db_path = _db_with_options(tmp_path)

    with (
        patch("trader_koo.notifications.options_digest.is_configured", return_value=True),
        patch(
            "trader_koo.notifications.options_digest.send_message",
            side_effect=[False, True],
        ) as send_message,
    ):
        first = send_options_digest(db_path, now_utc=NOW)
        second = send_options_digest(db_path, now_utc=NOW)

    assert first is False
    assert second is True
    assert send_message.call_count == 2
