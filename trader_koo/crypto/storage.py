"""Persist crypto OHLCV bars to SQLite for history across restarts."""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from typing import Any

from trader_koo.crypto.models import CryptoBar

LOG = logging.getLogger("trader_koo.crypto.storage")


def ensure_crypto_schema(conn: sqlite3.Connection) -> None:
    """Create the crypto_bars table and index if they do not exist."""
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
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_crypto_bars_symbol_ts
        ON crypto_bars (symbol, timestamp DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_crypto_bars_symbol_interval_ts
        ON crypto_bars (symbol, interval, timestamp DESC)
    """)
    conn.commit()
    LOG.info("crypto_bars schema ensured")


def save_bars(conn: sqlite3.Connection, bars: list[CryptoBar]) -> int:
    """Batch insert bars, ignoring duplicates. Returns count inserted."""
    if not bars:
        return 0
    inserted = 0
    for bar in bars:
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO crypto_bars
                    (symbol, timestamp, interval, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bar.symbol,
                    bar.timestamp.isoformat(),
                    bar.interval,
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                ),
            )
            inserted += conn.total_changes  # approximate
        except sqlite3.Error as exc:
            LOG.error("Failed to insert crypto bar %s@%s: %s", bar.symbol, bar.timestamp, exc)
    conn.commit()
    LOG.debug("Saved %d crypto bars (from batch of %d)", inserted, len(bars))
    return inserted


def load_recent_bars(
    conn: sqlite3.Connection,
    symbol: str,
    interval: str = "1m",
    limit: int = 1440,
) -> list[CryptoBar]:
    """Load most recent bars for a symbol from DB, oldest-first."""
    rows = conn.execute(
        """
        SELECT symbol, timestamp, interval, open, high, low, close, volume
        FROM crypto_bars
        WHERE symbol = ? AND interval = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (symbol, interval, limit),
    ).fetchall()
    bars: list[CryptoBar] = []
    for row in rows:
        try:
            ts = dt.datetime.fromisoformat(row[1])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=dt.timezone.utc)
            bars.append(
                CryptoBar(
                    symbol=row[0],
                    timestamp=ts,
                    interval=row[2],
                    open=float(row[3]),
                    high=float(row[4]),
                    low=float(row[5]),
                    close=float(row[6]),
                    volume=float(row[7]),
                )
            )
        except Exception as exc:
            LOG.warning("Skipping malformed crypto bar row: %s", exc)
    # Reverse so oldest is first
    bars.reverse()
    LOG.debug("Loaded %d bars for %s from DB", len(bars), symbol)
    return bars


def prune_old_bars(
    conn: sqlite3.Connection,
    retention_days: int = 30,
) -> int:
    """Delete old bars using interval-aware retention. Returns count deleted."""
    now_utc = dt.datetime.now(dt.timezone.utc)
    retention_by_interval = {
        "1m": retention_days,
        "5m": max(retention_days, 90),
        "15m": max(retention_days, 120),
        "30m": max(retention_days, 180),
        "1h": max(retention_days, 365),
        "2h": max(retention_days, 540),
        "4h": max(retention_days, 730),
        "6h": max(retention_days, 1095),
        "12h": max(retention_days, 1825),
        "1d": max(retention_days, 3650),
        "1w": max(retention_days, 3650),
    }
    deleted = 0
    for interval, days in retention_by_interval.items():
        cutoff = (now_utc - dt.timedelta(days=days)).isoformat()
        cursor = conn.execute(
            "DELETE FROM crypto_bars WHERE interval = ? AND timestamp < ?",
            (interval, cutoff),
        )
        deleted += int(cursor.rowcount or 0)
    conn.commit()
    if deleted > 0:
        LOG.info(
            "Pruned %d crypto bars using interval-aware retention (1m=%dd)",
            deleted,
            retention_days,
        )
    return deleted


def get_crypto_data_status(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return row counts and freshness per symbol/interval."""
    rows = conn.execute(
        """
        SELECT symbol, interval, COUNT(*) AS row_count,
               MAX(timestamp) AS latest_ts, MIN(timestamp) AS oldest_ts
        FROM crypto_bars
        GROUP BY symbol, interval
        ORDER BY symbol, interval
        """
    ).fetchall()
    return [
        {
            "symbol": row[0],
            "interval": row[1],
            "row_count": row[2],
            "latest_ts": row[3],
            "oldest_ts": row[4],
        }
        for row in rows
    ]
