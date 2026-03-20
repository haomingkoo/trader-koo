"""Triple-barrier labeling for supervised learning.

Generates labels by looking at what happens AFTER a given date:
- Hit the upper barrier (profit target) → label = +1
- Hit the lower barrier (stop loss) → label = -1
- Time expired without hitting either → label = 0

The barriers are volatility-scaled to adapt to each ticker's behavior.

IMPORTANT: Labels use future data by design (they're the ground truth
we're trying to predict).  Features must NEVER use data past the
as-of date.  The train/test split and walk-forward validation in
trainer.py enforce the temporal boundary.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def _compute_daily_vol(close: pd.Series, window: int = 21) -> pd.Series:
    """Rolling daily volatility (standard deviation of log returns)."""
    return np.log(close / close.shift(1)).rolling(window, min_periods=10).std()


def generate_triple_barrier_labels(
    conn: sqlite3.Connection,
    *,
    entry_dates: list[str],
    tickers: list[str] | None = None,
    max_holding_days: int = 10,
    profit_mult: float = 2.0,
    stop_mult: float = 2.0,
) -> pd.DataFrame:
    """Generate triple-barrier labels for each (ticker, entry_date) pair.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database with price_daily table.
    entry_dates : list[str]
        List of entry dates (YYYY-MM-DD) to label.
    tickers : list[str] | None
        Tickers to label. If None, labels all tickers with data.
    max_holding_days : int
        Vertical barrier — max trading days to hold.
    profit_mult : float
        Upper barrier = entry_price + profit_mult × daily_vol × entry_price.
    stop_mult : float
        Lower barrier = entry_price - stop_mult × daily_vol × entry_price.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, entry_date, label (+1, -1, 0), exit_date,
        exit_reason, return_pct, days_held, barrier_width_pct.
    """
    # Load all needed price data
    min_date = min(entry_dates)
    # Need history before min_date for vol computation + data after for outcomes
    lookback_start = (pd.Timestamp(min_date) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")

    ticker_clause = ""
    params: list[Any] = [lookback_start]
    if tickers:
        placeholders = ",".join("?" * len(tickers))
        ticker_clause = f"AND ticker IN ({placeholders})"
        params.extend(tickers)

    df = pd.read_sql_query(
        f"""
        SELECT ticker, date,
               CAST(high AS REAL) AS high,
               CAST(low AS REAL) AS low,
               CAST(close AS REAL) AS close
        FROM price_daily
        WHERE date >= ? AND close IS NOT NULL AND close > 0
          {ticker_clause}
        ORDER BY ticker, date
        """,
        conn,
        params=params,
    )

    if df.empty:
        return pd.DataFrame(columns=[
            "ticker", "entry_date", "label", "exit_date",
            "exit_reason", "return_pct", "days_held", "barrier_width_pct",
        ])

    df["date"] = pd.to_datetime(df["date"])
    entry_date_set = set(pd.to_datetime(entry_dates))
    results: list[dict[str, Any]] = []

    for ticker, group in df.groupby("ticker"):
        grp = group.sort_values("date").reset_index(drop=True)
        if len(grp) < 30:
            continue

        close = grp["close"]
        high = grp["high"]
        low = grp["low"]
        dates = grp["date"]
        daily_vol = _compute_daily_vol(close)

        for entry_idx in range(len(grp)):
            entry_date = dates.iloc[entry_idx]
            if entry_date not in entry_date_set:
                continue

            vol = daily_vol.iloc[entry_idx]
            if np.isnan(vol) or vol <= 0:
                continue

            entry_price = float(close.iloc[entry_idx])
            upper = entry_price * (1 + profit_mult * vol)
            lower = entry_price * (1 - stop_mult * vol)
            barrier_width = (upper - lower) / entry_price * 100

            # Walk forward through future bars (NO look-ahead in features)
            label = 0
            exit_date = None
            exit_reason = "time_expired"
            exit_price = entry_price
            days_held = 0

            for future_idx in range(entry_idx + 1, min(entry_idx + 1 + max_holding_days, len(grp))):
                future_high = float(high.iloc[future_idx])
                future_low = float(low.iloc[future_idx])
                future_close = float(close.iloc[future_idx])
                days_held = future_idx - entry_idx

                # Check HIGH for profit target (intraday touch counts —
                # limit orders fill at barrier price in real trading)
                if future_high >= upper:
                    label = 1
                    exit_date = dates.iloc[future_idx]
                    exit_reason = "profit_target"
                    exit_price = future_close
                    break
                # Check LOW for stop loss (intraday touch counts)
                elif future_low <= lower:
                    label = -1
                    exit_date = dates.iloc[future_idx]
                    exit_reason = "stop_loss"
                    exit_price = future_close
                    break
            else:
                # Time expired — use last available price
                last_idx = min(entry_idx + max_holding_days, len(grp) - 1)
                if last_idx > entry_idx:
                    exit_price = float(close.iloc[last_idx])
                    exit_date = dates.iloc[last_idx]
                    days_held = last_idx - entry_idx

            return_pct = (exit_price - entry_price) / entry_price * 100

            results.append({
                "ticker": str(ticker),
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "label": label,
                "exit_date": exit_date.strftime("%Y-%m-%d") if exit_date is not None else None,
                "exit_reason": exit_reason,
                "return_pct": round(return_pct, 4),
                "days_held": days_held,
                "barrier_width_pct": round(barrier_width, 4),
            })

    return pd.DataFrame(results)
