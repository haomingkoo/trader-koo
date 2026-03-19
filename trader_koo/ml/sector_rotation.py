"""Sector rotation signals for ML features.

Tracks relative performance of SPDR sector ETFs to detect which
sectors are leading/lagging the market — a strong signal for
individual stock momentum.

Sector ETFs (must be in ALWAYS_FETCH or ingested separately):
- XLK: Technology        - XLF: Financials
- XLV: Health Care       - XLE: Energy
- XLY: Consumer Disc.    - XLP: Consumer Staples
- XLI: Industrials       - XLU: Utilities
- XLB: Materials         - XLRE: Real Estate
- XLC: Communication     - IGV: Software (sub-sector)
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

# Sector ETFs we track
SECTOR_ETFS = {
    "XLK": "technology",
    "XLF": "financials",
    "XLV": "health_care",
    "XLE": "energy",
    "XLY": "consumer_disc",
    "XLP": "consumer_staples",
    "XLI": "industrials",
    "XLU": "utilities",
    "XLB": "materials",
    "XLRE": "real_estate",
    "XLC": "communication",
    "IGV": "software",
}

# Rough sector mapping for S&P 500 tickers (top holdings)
TICKER_SECTOR_MAP = {
    "AAPL": "technology", "MSFT": "technology", "NVDA": "technology",
    "GOOGL": "communication", "GOOG": "communication", "META": "communication",
    "AMZN": "consumer_disc", "TSLA": "consumer_disc",
    "BRK-B": "financials", "JPM": "financials", "V": "financials", "MA": "financials",
    "UNH": "health_care", "JNJ": "health_care", "LLY": "health_care",
    "XOM": "energy", "CVX": "energy",
    "PG": "consumer_staples", "KO": "consumer_staples", "PEP": "consumer_staples",
    "AVGO": "technology", "AMD": "technology", "INTC": "technology", "CRM": "software",
}


def compute_sector_features(
    conn: sqlite3.Connection,
    *,
    as_of_date: str,
    ticker: str | None = None,
) -> dict[str, float]:
    """Compute sector rotation features for a given date.

    Returns:
    - sector_rank: relative rank of this ticker's sector (0=worst, 1=best)
    - sector_momentum_5d: ticker's sector ETF 5-day return
    - sector_momentum_21d: ticker's sector ETF 21-day return
    - leading_sector_momentum: best sector's 5d return
    - lagging_sector_momentum: worst sector's 5d return
    - sector_dispersion: spread between best and worst sector
    """
    result: dict[str, float] = {
        "sector_rank": np.nan,
        "sector_momentum_5d": np.nan,
        "sector_momentum_21d": np.nan,
        "leading_sector_momentum": np.nan,
        "lagging_sector_momentum": np.nan,
        "sector_dispersion": np.nan,
    }

    # Fetch sector ETF returns
    sector_returns: dict[str, float] = {}
    sector_returns_21d: dict[str, float] = {}

    for etf, sector in SECTOR_ETFS.items():
        rows = conn.execute(
            """
            SELECT CAST(close AS REAL) AS close
            FROM price_daily
            WHERE ticker = ? AND date <= ? AND close IS NOT NULL
            ORDER BY date DESC LIMIT 22
            """,
            (etf, as_of_date),
        ).fetchall()

        if len(rows) >= 6:
            ret_5d = (float(rows[0][0]) / float(rows[5][0])) - 1
            sector_returns[sector] = ret_5d
        if len(rows) >= 22:
            ret_21d = (float(rows[0][0]) / float(rows[21][0])) - 1
            sector_returns_21d[sector] = ret_21d

    if len(sector_returns) < 3:
        return result

    # Rank sectors by 5d return
    sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1])
    ranks = {sector: i / (len(sorted_sectors) - 1) for i, (sector, _) in enumerate(sorted_sectors)}

    result["leading_sector_momentum"] = sorted_sectors[-1][1]
    result["lagging_sector_momentum"] = sorted_sectors[0][1]
    result["sector_dispersion"] = sorted_sectors[-1][1] - sorted_sectors[0][1]

    # If we know this ticker's sector, add sector-specific features
    if ticker:
        ticker_sector = TICKER_SECTOR_MAP.get(ticker.upper())
        if ticker_sector and ticker_sector in ranks:
            result["sector_rank"] = ranks[ticker_sector]
            result["sector_momentum_5d"] = sector_returns.get(ticker_sector, np.nan)
            result["sector_momentum_21d"] = sector_returns_21d.get(ticker_sector, np.nan)

    return result
