"""Collect full trade history for tracked Hyperliquid wallets.

Pages through user_fills_by_time API, respecting rate limits,
and stores fills in SQLite. Can be resumed - skips already-fetched
time ranges.

Usage:
    python -m trader_koo.hyperliquid.collect_history
    python -m trader_koo.hyperliquid.collect_history --wallet machibro --days 365
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

import requests

from trader_koo.hyperliquid.wallets import get_tracked_wallets

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(os.getenv(
    "TRADER_KOO_DB_PATH",
    str(Path(__file__).resolve().parents[2] / "data" / "trader_koo.db"),
))

API_URL = "https://api.hyperliquid.xyz/info"
FILLS_PER_PAGE = 2000
DELAY_BETWEEN_PAGES = 1.0  # seconds (rate limit safe)
MAX_PAGES_PER_RUN = 50  # ~100K fills max per run


def ensure_fills_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hyperliquid_fills (
            tid TEXT PRIMARY KEY,
            wallet_label TEXT NOT NULL,
            wallet_address TEXT NOT NULL,
            coin TEXT NOT NULL,
            side TEXT NOT NULL,
            size REAL NOT NULL,
            price REAL NOT NULL,
            closed_pnl REAL NOT NULL,
            fee REAL NOT NULL,
            is_liquidation INTEGER NOT NULL DEFAULT 0,
            fill_time_ms INTEGER NOT NULL,
            fill_date TEXT NOT NULL,
            dir TEXT,
            start_position TEXT,
            oid TEXT,
            raw_json TEXT,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_hl_fills_wallet_time "
        "ON hyperliquid_fills(wallet_label, fill_time_ms)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_hl_fills_coin_time "
        "ON hyperliquid_fills(coin, fill_time_ms)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_hl_fills_date "
        "ON hyperliquid_fills(fill_date)"
    )
    conn.commit()


def get_latest_fill_time(conn: sqlite3.Connection, wallet_label: str) -> int | None:
    """Get the newest fill timestamp already stored for this wallet."""
    row = conn.execute(
        "SELECT MAX(fill_time_ms) FROM hyperliquid_fills WHERE wallet_label = ?",
        (wallet_label,),
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def get_oldest_fill_time(conn: sqlite3.Connection, wallet_label: str) -> int | None:
    """Get the oldest fill timestamp already stored for this wallet."""
    row = conn.execute(
        "SELECT MIN(fill_time_ms) FROM hyperliquid_fills WHERE wallet_label = ?",
        (wallet_label,),
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def fetch_fills_page(
    address: str,
    start_ms: int,
    end_ms: int | None = None,
) -> list[dict]:
    """Fetch one page of fills from the API."""
    payload: dict = {
        "type": "userFillsByTime",
        "user": address,
        "startTime": start_ms,
    }
    if end_ms is not None:
        payload["endTime"] = end_ms

    resp = requests.post(API_URL, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def store_fills(
    conn: sqlite3.Connection,
    fills: list[dict],
    wallet_label: str,
    wallet_address: str,
) -> int:
    """Store fills in SQLite, skipping duplicates by tid."""
    inserted = 0
    for f in fills:
        tid = f.get("tid")
        if not tid:
            continue

        fill_time_ms = int(f.get("time", 0))
        fill_date = dt.datetime.fromtimestamp(
            fill_time_ms / 1000, tz=dt.timezone.utc
        ).strftime("%Y-%m-%d")

        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO hyperliquid_fills
                    (tid, wallet_label, wallet_address, coin, side, size, price,
                     closed_pnl, fee, is_liquidation, fill_time_ms, fill_date,
                     dir, start_position, oid, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tid,
                    wallet_label,
                    wallet_address,
                    f.get("coin", ""),
                    f.get("side", ""),
                    float(f.get("sz", 0)),
                    float(f.get("px", 0)),
                    float(f.get("closedPnl", 0)),
                    float(f.get("fee", 0)),
                    1 if f.get("liquidation") else 0,
                    fill_time_ms,
                    fill_date,
                    f.get("dir"),
                    f.get("startPosition"),
                    f.get("oid"),
                    json.dumps(f),
                ),
            )
            if conn.total_changes:
                inserted += 1
        except sqlite3.IntegrityError:
            pass  # duplicate tid

    conn.commit()
    return inserted


def collect_forward(
    conn: sqlite3.Connection,
    wallet_label: str,
    wallet_address: str,
    start_ms: int,
) -> int:
    """Page forward from start_ms, collecting all fills up to now."""
    total_stored = 0
    current_start = start_ms
    end_ms = int(time.time() * 1000)

    for page in range(MAX_PAGES_PER_RUN):
        try:
            fills = fetch_fills_page(wallet_address, current_start, end_ms)
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                LOG.warning("Rate limited at page %d. Stored %d fills so far. Resume later.", page + 1, total_stored)
                break
            raise

        if not fills:
            LOG.info("No more fills after page %d", page + 1)
            break

        stored = store_fills(conn, fills, wallet_label, wallet_address)
        total_stored += stored

        newest_ts = max(int(f.get("time", 0)) for f in fills)
        oldest_ts = min(int(f.get("time", 0)) for f in fills)
        newest_dt = dt.datetime.fromtimestamp(newest_ts / 1000, tz=dt.timezone.utc)
        oldest_dt = dt.datetime.fromtimestamp(oldest_ts / 1000, tz=dt.timezone.utc)

        LOG.info(
            "Page %d: %d fills (%d new) | %s to %s | total stored: %d",
            page + 1, len(fills), stored,
            oldest_dt.strftime("%Y-%m-%d"), newest_dt.strftime("%Y-%m-%d"),
            total_stored,
        )

        if len(fills) < FILLS_PER_PAGE:
            break

        current_start = newest_ts + 1
        time.sleep(DELAY_BETWEEN_PAGES)

    return total_stored


def collect_wallet(
    conn: sqlite3.Connection,
    wallet_label: str,
    wallet_address: str,
    lookback_days: int = 365,
) -> dict:
    """Collect full history for a wallet, resuming from where we left off."""
    ensure_fills_table(conn)

    # Check existing data
    latest = get_latest_fill_time(conn, wallet_label)
    oldest = get_oldest_fill_time(conn, wallet_label)
    existing = conn.execute(
        "SELECT COUNT(*) FROM hyperliquid_fills WHERE wallet_label = ?",
        (wallet_label,),
    ).fetchone()[0]

    LOG.info(
        "Existing data for %s: %d fills%s",
        wallet_label, existing,
        f" ({dt.datetime.fromtimestamp(oldest/1000, tz=dt.timezone.utc).strftime('%Y-%m-%d')} to "
        f"{dt.datetime.fromtimestamp(latest/1000, tz=dt.timezone.utc).strftime('%Y-%m-%d')})"
        if oldest and latest else "",
    )

    if latest:
        # Resume from where we left off (forward from latest)
        LOG.info("Resuming forward from latest fill...")
        new_forward = collect_forward(conn, wallet_label, wallet_address, latest + 1)
    else:
        # Fresh start - go back lookback_days
        start_ms = int((time.time() - lookback_days * 86400) * 1000)
        LOG.info("Fresh collection from %d days ago...", lookback_days)
        new_forward = collect_forward(conn, wallet_label, wallet_address, start_ms)

    final_count = conn.execute(
        "SELECT COUNT(*) FROM hyperliquid_fills WHERE wallet_label = ?",
        (wallet_label,),
    ).fetchone()[0]

    # Summary stats
    stats_row = conn.execute(
        """
        SELECT
            COUNT(*) as fills,
            SUM(closed_pnl) as total_pnl,
            SUM(fee) as total_fees,
            SUM(is_liquidation) as liquidations,
            MIN(fill_date) as first_date,
            MAX(fill_date) as last_date
        FROM hyperliquid_fills WHERE wallet_label = ?
        """,
        (wallet_label,),
    ).fetchone()

    result = {
        "wallet": wallet_label,
        "fills_total": final_count,
        "fills_new": new_forward,
        "total_pnl": round(float(stats_row[1] or 0), 2),
        "total_fees": round(float(stats_row[2] or 0), 2),
        "liquidations": int(stats_row[3] or 0),
        "date_range": f"{stats_row[4]} to {stats_row[5]}",
    }
    LOG.info("Collection complete: %s", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Collect Hyperliquid fill history")
    parser.add_argument("--wallet", default="machibro", help="Wallet label to collect")
    parser.add_argument("--days", type=int, default=365, help="Lookback days for fresh collection")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Database path")
    args = parser.parse_args()

    tracked_wallets = get_tracked_wallets()
    address = tracked_wallets.get(args.wallet)
    if not address:
        LOG.error("Unknown wallet: %s. Known: %s", args.wallet, list(tracked_wallets.keys()))
        return

    conn = sqlite3.connect(args.db)
    try:
        result = collect_wallet(conn, args.wallet, address, lookback_days=args.days)
        print(json.dumps(result, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
