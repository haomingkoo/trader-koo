"""Sector rotation signals for ML features.

Computes ONCE per date, then looks up per ticker. No repeated DB queries.

Sector mapping is built dynamically from the finviz_fundamentals table
(covers 500+ tickers) with a hardcoded fallback for tickers not in Finviz.
"""
from __future__ import annotations

import json
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

# Mapping from Finviz "Sector" names to our internal sector keys (matching SECTOR_ETFS values).
_FINVIZ_SECTOR_TO_INTERNAL: dict[str, str] = {
    "Technology": "technology",
    "Financial": "financials",
    "Financial Services": "financials",
    "Financials": "financials",
    "Healthcare": "health_care",
    "Health Care": "health_care",
    "Energy": "energy",
    "Consumer Cyclical": "consumer_disc",
    "Consumer Discretionary": "consumer_disc",
    "Consumer Defensive": "consumer_staples",
    "Consumer Staples": "consumer_staples",
    "Industrials": "industrials",
    "Utilities": "utilities",
    "Basic Materials": "materials",
    "Materials": "materials",
    "Real Estate": "real_estate",
    "Communication Services": "communication",
    "Communication": "communication",
}

# Hardcoded fallback for ~40 major tickers (used when Finviz data is unavailable)
TICKER_SECTOR_MAP: dict[str, str] = {
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

# Sector map cache — built once from DB, reused across all dates
_sector_map_cache: dict[str, str] | None = None


def build_sector_map_from_db(conn: sqlite3.Connection) -> dict[str, str]:
    """Build ticker-to-sector mapping from finviz_fundamentals.

    Queries the latest snapshot per ticker from finviz_fundamentals,
    extracts the "Sector" field from raw_json, and maps it to our
    internal sector keys (matching SECTOR_ETFS values).

    Falls back to the hardcoded TICKER_SECTOR_MAP for any tickers
    not found in Finviz.

    Returns
    -------
    dict[str, str]
        Mapping from ticker (uppercase) to internal sector key.
    """
    global _sector_map_cache
    with _cache_lock:
        if _sector_map_cache is not None:
            return _sector_map_cache

    # Start with the hardcoded fallback
    result: dict[str, str] = dict(TICKER_SECTOR_MAP)

    try:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='finviz_fundamentals' LIMIT 1"
        ).fetchone()
        if not has_table:
            LOG.debug("finviz_fundamentals table not found; using hardcoded sector map")
            with _cache_lock:
                _sector_map_cache = result
            return result

        # Get the latest snapshot timestamp
        snap_row = conn.execute(
            "SELECT snapshot_ts FROM finviz_fundamentals "
            "ORDER BY snapshot_ts DESC LIMIT 1"
        ).fetchone()
        if not snap_row:
            LOG.debug("No finviz_fundamentals snapshots found")
            with _cache_lock:
                _sector_map_cache = result
            return result

        snapshot_ts = snap_row[0]
        rows = conn.execute(
            "SELECT ticker, raw_json FROM finviz_fundamentals "
            "WHERE snapshot_ts = ?",
            (snapshot_ts,),
        ).fetchall()

        mapped_count = 0
        for ticker_val, raw_json_str in rows:
            ticker = str(ticker_val or "").upper().strip()
            if not ticker or not raw_json_str:
                continue
            try:
                raw_obj = json.loads(raw_json_str)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(raw_obj, dict):
                continue

            # Extract sector name from Finviz raw_json
            finviz_sector = str(
                raw_obj.get("Sector") or raw_obj.get("sector") or ""
            ).strip()
            if not finviz_sector or finviz_sector == "-":
                continue

            # Map Finviz sector name to our internal key
            internal_sector = _FINVIZ_SECTOR_TO_INTERNAL.get(finviz_sector)
            if internal_sector:
                result[ticker] = internal_sector
                mapped_count += 1
            else:
                LOG.debug(
                    "Unknown Finviz sector '%s' for %s; skipping",
                    finviz_sector, ticker,
                )

        LOG.info(
            "Sector map: %d tickers from Finviz + %d hardcoded fallback = %d total",
            mapped_count, len(TICKER_SECTOR_MAP), len(result),
        )
    except Exception as exc:
        LOG.warning("build_sector_map_from_db failed (non-fatal): %s", exc)

    with _cache_lock:
        _sector_map_cache = result
    return result


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
        sector_map = build_sector_map_from_db(conn)
        sector = sector_map.get(ticker.upper())
        if sector:
            result["sector_rank"] = data["ranks"].get(sector, np.nan)
            result["sector_momentum_5d"] = data["sector_ret_5d"].get(sector, np.nan)
            result["sector_momentum_21d"] = data["sector_ret_21d"].get(sector, np.nan)

    return result


def clear_cache() -> None:
    """Clear the sector cache (call between training runs if needed)."""
    global _sector_map_cache
    with _cache_lock:
        _sector_cache.clear()
        _sector_map_cache = None
