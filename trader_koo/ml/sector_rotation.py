"""Sector rotation signals for ML features.

Computes ONCE per date, then looks up per ticker. No repeated DB queries.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)

SECTOR_ETFS = {
    "XLK": "technology", "XLF": "financials", "XLV": "health_care",
    "XLE": "energy", "XLY": "consumer_disc", "XLP": "consumer_staples",
    "XLI": "industrials", "XLU": "utilities", "XLB": "materials",
    "XLRE": "real_estate", "XLC": "communication", "IGV": "software",
}

# Rough sector mapping for major tickers
TICKER_SECTOR_MAP = {
    "AAPL": "technology", "MSFT": "technology", "NVDA": "technology",
    "AVGO": "technology", "AMD": "technology", "INTC": "technology",
    "GOOGL": "communication", "GOOG": "communication", "META": "communication",
    "NFLX": "communication", "DIS": "communication",
    "AMZN": "consumer_disc", "TSLA": "consumer_disc", "HD": "consumer_disc",
    "MCD": "consumer_disc", "NKE": "consumer_disc", "SBUX": "consumer_disc",
    "BRK-B": "financials", "JPM": "financials", "V": "financials",
    "MA": "financials", "BAC": "financials", "GS": "financials",
    "UNH": "health_care", "JNJ": "health_care", "LLY": "health_care",
    "PFE": "health_care", "ABBV": "health_care", "MRK": "health_care",
    "XOM": "energy", "CVX": "energy", "COP": "energy",
    "PG": "consumer_staples", "KO": "consumer_staples", "PEP": "consumer_staples",
    "WMT": "consumer_staples", "COST": "consumer_staples",
    "CRM": "software", "ADBE": "software", "NOW": "software",
    "CAT": "industrials", "UNP": "industrials", "HON": "industrials",
    "NEE": "utilities", "DUK": "utilities", "SO": "utilities",
}

# Date-level cache — compute once, reuse for all tickers on the same date
_cache_lock = threading.Lock()
_sector_cache: dict[str, dict[str, Any]] = {}  # key = as_of_date


def _compute_sector_data(conn: sqlite3.Connection, as_of_date: str) -> dict[str, Any]:
    """Compute all sector data for a date in ONE batch query. Cached."""
    with _cache_lock:
        if as_of_date in _sector_cache:
            return _sector_cache[as_of_date]

    etf_tickers = list(SECTOR_ETFS.keys())
    placeholders = ",".join("?" * len(etf_tickers))

    # Single query with date lower bound to avoid full table scan
    import pandas as pd
    date_lower = (pd.Timestamp(as_of_date) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")

    rows = conn.execute(
        f"""
        SELECT ticker, date, CAST(close AS REAL) AS close
        FROM price_daily
        WHERE ticker IN ({placeholders})
          AND date >= ? AND date <= ?
          AND close IS NOT NULL AND close > 0
        ORDER BY ticker, date DESC
        """,
        (*etf_tickers, date_lower, as_of_date),
    ).fetchall()

    # Group by ticker, keep last 22 per ticker
    by_ticker: dict[str, list[float]] = {}
    for r in rows:
        tkr = str(r[0])
        closes = by_ticker.setdefault(tkr, [])
        if len(closes) < 22:
            closes.append(float(r[2]))

    sector_ret_5d: dict[str, float] = {}
    sector_ret_21d: dict[str, float] = {}

    for etf, sector in SECTOR_ETFS.items():
        closes = by_ticker.get(etf, [])
        if len(closes) >= 6:
            sector_ret_5d[sector] = (closes[0] / closes[5]) - 1
        if len(closes) >= 22:
            sector_ret_21d[sector] = (closes[0] / closes[21]) - 1

    # Rank sectors
    ranks: dict[str, float] = {}
    if len(sector_ret_5d) >= 3:
        sorted_sectors = sorted(sector_ret_5d.items(), key=lambda x: x[1])
        denom = max(len(sorted_sectors) - 1, 1)
        ranks = {sector: i / denom for i, (sector, _) in enumerate(sorted_sectors)}

    result = {
        "sector_ret_5d": sector_ret_5d,
        "sector_ret_21d": sector_ret_21d,
        "ranks": ranks,
        "leading": max(sector_ret_5d.values()) if sector_ret_5d else np.nan,
        "lagging": min(sector_ret_5d.values()) if sector_ret_5d else np.nan,
        "dispersion": (max(sector_ret_5d.values()) - min(sector_ret_5d.values())) if len(sector_ret_5d) >= 2 else np.nan,
    }

    with _cache_lock:
        # Keep cache bounded
        if len(_sector_cache) > 200:
            _sector_cache.clear()
        _sector_cache[as_of_date] = result

    return result


def compute_sector_features(
    conn: sqlite3.Connection,
    *,
    as_of_date: str,
    ticker: str | None = None,
) -> dict[str, float]:
    """Get sector features — uses cached date-level computation."""
    data = _compute_sector_data(conn, as_of_date)

    result: dict[str, float] = {
        "sector_rank": np.nan,
        "sector_momentum_5d": np.nan,
        "sector_momentum_21d": np.nan,
        "leading_sector_momentum": data["leading"],
        "lagging_sector_momentum": data["lagging"],
        "sector_dispersion": data["dispersion"],
    }

    if ticker:
        sector = TICKER_SECTOR_MAP.get(ticker.upper())
        if sector:
            result["sector_rank"] = data["ranks"].get(sector, np.nan)
            result["sector_momentum_5d"] = data["sector_ret_5d"].get(sector, np.nan)
            result["sector_momentum_21d"] = data["sector_ret_21d"].get(sector, np.nan)

    return result


def clear_cache() -> None:
    """Clear the sector cache (call between training runs if needed)."""
    with _cache_lock:
        _sector_cache.clear()
