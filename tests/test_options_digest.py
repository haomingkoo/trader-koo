from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

from trader_koo.notifications.options_digest import (
    build_options_digest,
    generate_options_digest,
    send_options_digest,
)


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
        ('2026-04-24T21:40:00Z', 'AMD', '2026-05-15', 'put', 95, 3.0, 2.9, 3.1, 0.45, 800, 120, 0.95);
        """
    )
    conn.commit()
    conn.close()
    return db_path


def test_generate_options_digest_formats_top_proxy_rows(tmp_path: Path):
    db_path = _db_with_options(tmp_path)

    message = generate_options_digest(db_path, limit=3)

    assert "Options Premium Proxy" in message
    assert "Not live signed flow" in message
    assert "AMD" in message
    assert "Call skew" in message
    assert "Vol net $84.0K" in message


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
        sent = send_options_digest(db_path)

    assert sent is True
    assert "AMD" in send_message.call_args[0][0]
