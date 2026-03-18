"""Macro-economic feature extraction for the ML model.

Extracts market-wide regime features from index/macro tickers already
in the database.  These are the same for ALL tickers on a given date
(cross-sectional constants) and capture the macro environment.

Data sources (all from price_daily, already ingested):
- ^VIX: volatility regime, term structure proxy
- ^TNX: 10-year treasury yield (rate environment)
- ^GSPC: S&P 500 (broad market trend)
- ^DJI: Dow Jones (breadth confirmation)
- SPY: S&P 500 ETF (volume, momentum)
- SVIX: Short VIX ETF (vol-of-vol proxy)

Future additions (need to add to ALWAYS_FETCH):
- ^IRX: 13-week T-bill (short-term rates)
- ^TYX: 30-year yield (long end)
- GLD: gold (risk-off proxy)
- TLT: long-term treasury ETF (duration proxy)
- HYG: high-yield bonds (credit risk proxy)
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

MACRO_FEATURE_COLUMNS = [
    # VIX regime (some overlap with per-ticker features, but these are richer)
    "vix_close",
    "vix_ma20",
    "vix_ma50",
    "vix_ma20_ratio",
    "vix_percentile_252d",
    "vix_ret_1d",
    "vix_ret_5d",
    "vix_ret_21d",
    "vix_realized_vol_21d",
    "vix_above_ma20",
    "vix_above_ma50",
    # Treasury / rates
    "tnx_close",
    "tnx_ret_5d",
    "tnx_ret_21d",
    "tnx_ma50",
    "tnx_above_ma50",
    # Broad market
    "sp500_ret_1d",
    "sp500_ret_5d",
    "sp500_ret_21d",
    "sp500_ret_63d",
    "sp500_vol_21d",
    "sp500_dist_ma50_pct",
    "sp500_dist_ma200_pct",
    "sp500_ma50_above_ma200",
    "sp500_breadth_ratio",
    # Volume regime
    "spy_volume_ratio_20d",
    # SVIX (inverse VIX — vol-of-vol proxy)
    "svix_ret_5d",
    "svix_vix_correlation_21d",
]


def _fetch_ticker_series(
    conn: sqlite3.Connection,
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch close + volume series for a single ticker."""
    df = pd.read_sql_query(
        """
        SELECT date, CAST(close AS REAL) AS close, CAST(volume AS REAL) AS volume
        FROM price_daily
        WHERE ticker = ? AND date >= ? AND date <= ?
          AND close IS NOT NULL AND close > 0
        ORDER BY date
        """,
        conn,
        params=[ticker, start_date, end_date],
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


def extract_macro_features(
    conn: sqlite3.Connection,
    *,
    as_of_date: str,
    lookback_days: int = 300,
) -> dict[str, float]:
    """Extract macro-level features for a given date.

    These features are the SAME for all tickers — they describe the
    market environment, not individual stocks.  All backward-looking.

    Returns a dict with keys from MACRO_FEATURE_COLUMNS.
    """
    start = (pd.Timestamp(as_of_date) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    result: dict[str, float] = {col: np.nan for col in MACRO_FEATURE_COLUMNS}

    # --- VIX ---
    vix = _fetch_ticker_series(conn, "^VIX", start, as_of_date)
    if len(vix) >= 20:
        c = vix["close"]
        result["vix_close"] = float(c.iloc[-1])
        result["vix_ma20"] = float(c.rolling(20).mean().iloc[-1])
        result["vix_ma50"] = float(c.rolling(50).mean().iloc[-1]) if len(vix) >= 50 else np.nan
        result["vix_ma20_ratio"] = result["vix_close"] / result["vix_ma20"] if result["vix_ma20"] > 0 else np.nan
        result["vix_percentile_252d"] = float((c <= c.iloc[-1]).tail(252).mean() * 100)
        result["vix_ret_1d"] = float(c.pct_change().iloc[-1]) if len(vix) > 1 else np.nan
        result["vix_ret_5d"] = float(c.iloc[-1] / c.iloc[-6] - 1) if len(vix) > 5 else np.nan
        result["vix_ret_21d"] = float(c.iloc[-1] / c.iloc[-22] - 1) if len(vix) > 21 else np.nan
        vix_log_ret = np.log(c / c.shift(1)).dropna()
        result["vix_realized_vol_21d"] = float(vix_log_ret.rolling(21).std().iloc[-1] * np.sqrt(252)) if len(vix_log_ret) >= 21 else np.nan
        result["vix_above_ma20"] = 1.0 if result["vix_close"] > result["vix_ma20"] else 0.0
        result["vix_above_ma50"] = 1.0 if result.get("vix_ma50") and result["vix_close"] > result["vix_ma50"] else 0.0

    # --- Treasury 10Y (^TNX) ---
    tnx = _fetch_ticker_series(conn, "^TNX", start, as_of_date)
    if len(tnx) >= 5:
        c = tnx["close"]
        result["tnx_close"] = float(c.iloc[-1])
        result["tnx_ret_5d"] = float(c.iloc[-1] / c.iloc[-6] - 1) if len(tnx) > 5 else np.nan
        result["tnx_ret_21d"] = float(c.iloc[-1] / c.iloc[-22] - 1) if len(tnx) > 21 else np.nan
        result["tnx_ma50"] = float(c.rolling(50).mean().iloc[-1]) if len(tnx) >= 50 else np.nan
        result["tnx_above_ma50"] = 1.0 if result.get("tnx_ma50") and result["tnx_close"] > result["tnx_ma50"] else 0.0

    # --- S&P 500 (^GSPC) ---
    sp = _fetch_ticker_series(conn, "^GSPC", start, as_of_date)
    if len(sp) >= 20:
        c = sp["close"]
        result["sp500_ret_1d"] = float(c.pct_change().iloc[-1])
        result["sp500_ret_5d"] = float(c.iloc[-1] / c.iloc[-6] - 1) if len(sp) > 5 else np.nan
        result["sp500_ret_21d"] = float(c.iloc[-1] / c.iloc[-22] - 1) if len(sp) > 21 else np.nan
        result["sp500_ret_63d"] = float(c.iloc[-1] / c.iloc[-64] - 1) if len(sp) > 63 else np.nan
        log_ret = np.log(c / c.shift(1)).dropna()
        result["sp500_vol_21d"] = float(log_ret.rolling(21).std().iloc[-1] * np.sqrt(252)) if len(log_ret) >= 21 else np.nan
        ma50 = c.rolling(50).mean()
        ma200 = c.rolling(200).mean()
        result["sp500_dist_ma50_pct"] = float((c.iloc[-1] - ma50.iloc[-1]) / ma50.iloc[-1] * 100) if len(sp) >= 50 else np.nan
        result["sp500_dist_ma200_pct"] = float((c.iloc[-1] - ma200.iloc[-1]) / ma200.iloc[-1] * 100) if len(sp) >= 200 else np.nan
        result["sp500_ma50_above_ma200"] = 1.0 if len(sp) >= 200 and float(ma50.iloc[-1]) > float(ma200.iloc[-1]) else 0.0

    # --- Breadth (advancing vs declining from latest date) ---
    try:
        breadth = conn.execute(
            """
            WITH latest AS (
                SELECT ticker, CAST(close AS REAL) AS close
                FROM price_daily WHERE date = (SELECT MAX(date) FROM price_daily WHERE date <= ?)
                AND close IS NOT NULL
            ),
            prev AS (
                SELECT ticker, CAST(close AS REAL) AS close
                FROM price_daily WHERE date = (
                    SELECT MAX(date) FROM price_daily
                    WHERE date < (SELECT MAX(date) FROM price_daily WHERE date <= ?)
                )
                AND close IS NOT NULL
            )
            SELECT
                SUM(CASE WHEN l.close > p.close THEN 1 ELSE 0 END) AS adv,
                SUM(CASE WHEN l.close < p.close THEN 1 ELSE 0 END) AS dec
            FROM latest l JOIN prev p ON l.ticker = p.ticker
            """,
            (as_of_date, as_of_date),
        ).fetchone()
        if breadth and breadth[0] is not None and breadth[1] is not None:
            adv = int(breadth[0])
            dec = int(breadth[1])
            total = adv + dec
            result["sp500_breadth_ratio"] = float(adv / total) if total > 0 else np.nan
    except Exception:
        pass

    # --- SPY volume ---
    spy = _fetch_ticker_series(conn, "SPY", start, as_of_date)
    if len(spy) >= 20:
        v = spy["volume"]
        v_ma20 = v.rolling(20).mean()
        result["spy_volume_ratio_20d"] = float(v.iloc[-1] / v_ma20.iloc[-1]) if float(v_ma20.iloc[-1]) > 0 else np.nan

    # --- SVIX (inverse VIX) ---
    svix = _fetch_ticker_series(conn, "SVIX", start, as_of_date)
    if len(svix) >= 5:
        c = svix["close"]
        result["svix_ret_5d"] = float(c.iloc[-1] / c.iloc[-6] - 1) if len(svix) > 5 else np.nan
        # VIX-SVIX correlation (should be strongly negative)
        if len(vix) >= 21 and len(svix) >= 21:
            vix_ret = vix["close"].pct_change().tail(21)
            svix_ret = svix["close"].pct_change().tail(21)
            # Align on dates
            common = vix_ret.index.intersection(svix_ret.index)
            if len(common) >= 10:
                result["svix_vix_correlation_21d"] = float(vix_ret.loc[common].corr(svix_ret.loc[common]))

    return result
