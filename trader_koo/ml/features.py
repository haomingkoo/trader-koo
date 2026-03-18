"""Feature extraction for the swing-trade forecasting model.

Extracts a standardized feature vector per ticker per date from the
price_daily table.  All features are backward-looking — no future data
is used.  Features are designed for 2–10 day holding periods.

Feature categories:
1. Multi-horizon momentum (returns at 1d, 5d, 10d, 21d, 63d)
2. Volatility (realized vol 5d, 21d, ATR percentile)
3. Volume (ratio vs 20-day average)
4. Mean-reversion (distance from MA20, MA50, MA200)
5. Regime context (VIX level, VIX percentile, VIX term structure)
6. Cross-sectional rank (percentile within universe for key features)
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

# Feature columns in the order the model expects them.
FEATURE_COLUMNS = [
    # Momentum
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_21d",
    "ret_63d",
    # Volatility
    "vol_5d",
    "vol_21d",
    "atr_pct_14",
    # Volume
    "volume_ratio_20d",
    # Mean-reversion / trend
    "dist_ma20_pct",
    "dist_ma50_pct",
    "dist_ma200_pct",
    # Regime context
    "vix_level",
    "vix_percentile",
    # Cross-sectional ranks (filled later)
    "rank_ret_5d",
    "rank_ret_21d",
    "rank_vol_21d",
]


def _safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
    """Percentage change without look-ahead.  NaN for insufficient data."""
    shifted = series.shift(periods)
    return (series - shifted) / shifted.abs().replace(0, np.nan)


def _realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Annualized realized volatility from daily log returns."""
    log_ret = np.log1p(returns)
    return log_ret.rolling(window, min_periods=max(3, window // 2)).std() * np.sqrt(252)


def extract_features_for_universe(
    conn: sqlite3.Connection,
    *,
    as_of_date: str,
    lookback_days: int = 300,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Extract feature matrix for all tickers as of a specific date.

    Returns a DataFrame indexed by ticker with columns from FEATURE_COLUMNS.
    All computations use only data on or before *as_of_date* — no leakage.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection to price_daily table.
    as_of_date : str
        The date to compute features for (YYYY-MM-DD).  Only data up to
        and including this date is used.
    lookback_days : int
        How many calendar days of history to load (default 300, enough
        for 200-day MA + momentum).
    tickers : list[str] | None
        If provided, restrict to these tickers.  Otherwise, all tickers
        with data on as_of_date.
    """
    # Load price data — only up to as_of_date (no future data)
    cutoff_start = pd.Timestamp(as_of_date) - pd.Timedelta(days=lookback_days)
    ticker_clause = ""
    params: list[Any] = [cutoff_start.strftime("%Y-%m-%d"), as_of_date]
    if tickers:
        placeholders = ",".join("?" * len(tickers))
        ticker_clause = f"AND ticker IN ({placeholders})"
        params.extend(tickers)

    df = pd.read_sql_query(
        f"""
        SELECT ticker, date,
               CAST(open AS REAL) AS open,
               CAST(high AS REAL) AS high,
               CAST(low AS REAL) AS low,
               CAST(close AS REAL) AS close,
               CAST(volume AS REAL) AS volume
        FROM price_daily
        WHERE date >= ? AND date <= ?
          AND close IS NOT NULL AND close > 0
          {ticker_clause}
        ORDER BY ticker, date
        """,
        conn,
        params=params,
    )

    if df.empty:
        return pd.DataFrame(columns=["ticker"] + FEATURE_COLUMNS)

    df["date"] = pd.to_datetime(df["date"])

    # VIX data for regime features
    vix_df = pd.read_sql_query(
        """
        SELECT date, CAST(close AS REAL) AS vix_close
        FROM price_daily
        WHERE ticker = '^VIX' AND date >= ? AND date <= ?
          AND close IS NOT NULL
        ORDER BY date
        """,
        conn,
        params=[cutoff_start.strftime("%Y-%m-%d"), as_of_date],
    )
    vix_df["date"] = pd.to_datetime(vix_df["date"])
    vix_latest = float(vix_df["vix_close"].iloc[-1]) if not vix_df.empty else np.nan
    vix_pctile = np.nan
    if len(vix_df) >= 20:
        vix_pctile = float((vix_df["vix_close"] <= vix_latest).mean() * 100)

    results: list[dict[str, Any]] = []

    for ticker, group in df.groupby("ticker"):
        grp = group.sort_values("date").reset_index(drop=True)
        if len(grp) < 20:
            continue

        close = grp["close"]
        high = grp["high"]
        low = grp["low"]
        volume = grp["volume"]
        daily_ret = close.pct_change()

        # Only use the latest row (as_of_date or closest prior)
        latest_idx = len(grp) - 1

        # Momentum returns
        ret_1d = _safe_pct_change(close, 1).iloc[latest_idx] if len(grp) > 1 else np.nan
        ret_5d = _safe_pct_change(close, 5).iloc[latest_idx] if len(grp) > 5 else np.nan
        ret_10d = _safe_pct_change(close, 10).iloc[latest_idx] if len(grp) > 10 else np.nan
        ret_21d = _safe_pct_change(close, 21).iloc[latest_idx] if len(grp) > 21 else np.nan
        ret_63d = _safe_pct_change(close, 63).iloc[latest_idx] if len(grp) > 63 else np.nan

        # Volatility
        vol_5d_series = _realized_vol(daily_ret, 5)
        vol_21d_series = _realized_vol(daily_ret, 21)
        vol_5d = float(vol_5d_series.iloc[latest_idx]) if len(grp) > 5 else np.nan
        vol_21d = float(vol_21d_series.iloc[latest_idx]) if len(grp) > 21 else np.nan

        # ATR % (14-day)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(14, min_periods=7).mean()
        current_close = float(close.iloc[latest_idx])
        atr_pct = float(atr_14.iloc[latest_idx] / current_close * 100) if current_close > 0 and len(grp) > 14 else np.nan

        # Volume ratio
        vol_ma20 = volume.rolling(20, min_periods=10).mean()
        current_vol = float(volume.iloc[latest_idx])
        vol_ma20_val = float(vol_ma20.iloc[latest_idx])
        vol_ratio = current_vol / vol_ma20_val if vol_ma20_val > 0 else np.nan

        # Distance from MAs
        ma20 = close.rolling(20, min_periods=10).mean()
        ma50 = close.rolling(50, min_periods=25).mean()
        ma200 = close.rolling(200, min_periods=100).mean()
        dist_ma20 = (current_close - float(ma20.iloc[latest_idx])) / float(ma20.iloc[latest_idx]) * 100 if len(grp) >= 20 else np.nan
        dist_ma50 = (current_close - float(ma50.iloc[latest_idx])) / float(ma50.iloc[latest_idx]) * 100 if len(grp) >= 50 else np.nan
        dist_ma200 = (current_close - float(ma200.iloc[latest_idx])) / float(ma200.iloc[latest_idx]) * 100 if len(grp) >= 200 else np.nan

        results.append({
            "ticker": str(ticker),
            "ret_1d": float(ret_1d) if np.isfinite(ret_1d) else np.nan,
            "ret_5d": float(ret_5d) if np.isfinite(ret_5d) else np.nan,
            "ret_10d": float(ret_10d) if np.isfinite(ret_10d) else np.nan,
            "ret_21d": float(ret_21d) if np.isfinite(ret_21d) else np.nan,
            "ret_63d": float(ret_63d) if np.isfinite(ret_63d) else np.nan,
            "vol_5d": vol_5d,
            "vol_21d": vol_21d,
            "atr_pct_14": atr_pct,
            "volume_ratio_20d": vol_ratio,
            "dist_ma20_pct": dist_ma20,
            "dist_ma50_pct": dist_ma50,
            "dist_ma200_pct": dist_ma200,
            "vix_level": vix_latest,
            "vix_percentile": vix_pctile,
            # Ranks filled below
            "rank_ret_5d": np.nan,
            "rank_ret_21d": np.nan,
            "rank_vol_21d": np.nan,
        })

    if not results:
        return pd.DataFrame(columns=["ticker"] + FEATURE_COLUMNS)

    feat_df = pd.DataFrame(results)

    # Cross-sectional ranks (percentile within universe on this date)
    for col, rank_col in [
        ("ret_5d", "rank_ret_5d"),
        ("ret_21d", "rank_ret_21d"),
        ("vol_21d", "rank_vol_21d"),
    ]:
        valid = feat_df[col].notna()
        if valid.sum() >= 5:
            feat_df.loc[valid, rank_col] = feat_df.loc[valid, col].rank(pct=True)

    return feat_df.set_index("ticker")[FEATURE_COLUMNS].copy()
