"""Feature extraction for the swing-trade forecasting model.

All features are backward-looking — no future data is used.

Feature categories:
1. Multi-horizon momentum (1d, 5d, 10d, 21d, 63d returns)
2. Volatility (realized vol, ATR %, Bollinger bandwidth)
3. Volume (ratio vs 20d MA, on-balance volume trend)
4. Mean-reversion (distance from MA20/50/200)
5. Regime context (VIX level, VIX percentile, VIX MA ratio)
6. Time series (lag autocorrelation, recent trend strength)
7. Seasonality (day of week, month, quarter effects)
8. YOLO patterns (if available in DB)
9. Cross-sectional ranks
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

# Pruned feature set based on actual feature importance from first training runs.
# Dropped: seasonality (zero signal), YOLO (insufficient data), binary MA flags
# (too coarse), redundant macro features. Kept top 25 by importance.
FEATURE_COLUMNS = [
    # === Per-ticker features (what makes THIS stock different) ===
    # Momentum (multi-horizon)
    "ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d",
    # Volatility (top predictive cluster)
    "vol_5d", "vol_21d", "atr_pct_14", "bb_width",
    # Volume
    "volume_ratio_20d", "obv_slope_10d",
    # Trend position (distance from key MAs)
    "dist_ma20_pct", "dist_ma50_pct", "dist_ma200_pct",
    # Time series (autocorrelation = mean-reversion signal)
    "autocorr_lag1", "autocorr_lag5",
    "trend_strength_10d", "mean_reversion_5d",
    # VIX regime (per-ticker duplicate but strongest cluster)
    "vix_level", "vix_percentile", "vix_ma20_ratio", "vix_ret_5d",
    # Cross-sectional rank (where this stock sits vs peers)
    "rank_ret_5d", "rank_ret_21d", "rank_vol_21d", "rank_volume_ratio",
    # === Macro features (same for all tickers — market environment) ===
    # Treasury (rate environment — #6 in importance)
    "macro_tnx_close", "macro_tnx_ret_21d",
    # Broad market
    "macro_sp500_ret_63d", "macro_sp500_breadth_ratio",
    "macro_sp500_vol_21d",
    # Inverse VIX (vol-of-vol proxy)
    "macro_svix_ret_5d",
    # FRED macro (fetched via external_data.py)
    "fred_yield_curve_10y2y",
    "fred_high_yield_oas",
    "fred_fed_funds_rate",
]


def _safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
    shifted = series.shift(periods)
    return (series - shifted) / shifted.abs().replace(0, np.nan)


def _realized_vol(returns: pd.Series, window: int) -> pd.Series:
    log_ret = np.log1p(returns)
    return log_ret.rolling(window, min_periods=max(3, window // 2)).std() * np.sqrt(252)


def _obv_slope(close: pd.Series, volume: pd.Series, window: int = 10) -> float:
    """On-Balance Volume slope over window — trend of money flow."""
    if len(close) < window + 1:
        return np.nan
    direction = np.sign(close.diff())
    obv = (direction * volume).cumsum()
    recent = obv.iloc[-window:]
    if len(recent) < window:
        return np.nan
    x = np.arange(len(recent), dtype=float)
    slope = np.polyfit(x, recent.values, 1)[0]
    return slope / (recent.abs().mean() + 1e-10)  # normalize


def _autocorrelation(series: pd.Series, lag: int) -> float:
    if len(series) < lag + 10:
        return np.nan
    return float(series.autocorr(lag=lag))


def _trend_strength(close: pd.Series, window: int = 10) -> float:
    """Linear regression R² over window — how directional is the trend."""
    if len(close) < window:
        return np.nan
    recent = close.iloc[-window:].values
    x = np.arange(len(recent), dtype=float)
    if np.std(recent) < 1e-10:
        return 0.0
    corr = np.corrcoef(x, recent)[0, 1]
    return float(corr ** 2)


def _mean_reversion_signal(returns: pd.Series, window: int = 5) -> float:
    """Cumulative return over window — extreme values signal reversion."""
    if len(returns) < window:
        return np.nan
    return float(returns.iloc[-window:].sum())


def _get_yolo_data(
    conn: sqlite3.Connection,
    tickers: list[str],
    as_of_date: str,
) -> dict[str, tuple[int, float]]:
    """Get YOLO pattern data per ticker: (has_pattern, max_confidence)."""
    try:
        # Check if yolo_patterns table exists
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='yolo_patterns' LIMIT 1"
        ).fetchone()
        if not has_table:
            return {}

        placeholders = ",".join("?" * len(tickers))
        rows = conn.execute(
            f"""
            SELECT ticker, MAX(confidence) as max_conf, COUNT(*) as pattern_count
            FROM yolo_patterns
            WHERE ticker IN ({placeholders})
              AND detected_date <= ?
              AND detected_date >= date(?, '-30 days')
            GROUP BY ticker
            """,
            (*tickers, as_of_date, as_of_date),
        ).fetchall()
        return {
            str(r[0]): (1, float(r[1]) if r[1] else 0.5)
            for r in rows
        }
    except Exception:
        return {}


def extract_features_for_universe(
    conn: sqlite3.Connection,
    *,
    as_of_date: str,
    lookback_days: int = 300,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Extract feature matrix for all tickers as of a specific date.

    All computations use only data on or before *as_of_date*.
    """
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
    as_of_ts = pd.Timestamp(as_of_date)

    # VIX regime features
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
    vix_ma20 = float(vix_df["vix_close"].rolling(20).mean().iloc[-1]) if len(vix_df) >= 20 else np.nan
    vix_ma20_ratio = vix_latest / vix_ma20 if vix_ma20 and vix_ma20 > 0 else np.nan
    vix_ret_5d = np.nan
    if len(vix_df) >= 6:
        vix_ret_5d = float((vix_df["vix_close"].iloc[-1] - vix_df["vix_close"].iloc[-6]) / vix_df["vix_close"].iloc[-6])

    # YOLO data
    all_tickers = sorted(df["ticker"].unique().tolist())
    yolo_data = _get_yolo_data(conn, all_tickers, as_of_date)

    # Seasonality from as_of_date
    dow = as_of_ts.dayofweek
    month = as_of_ts.month
    is_month_end = 1 if as_of_ts.is_month_end else 0
    is_quarter_end = 1 if as_of_ts.is_quarter_end else 0

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
        n = len(grp)
        i = n - 1  # latest index

        # Momentum
        ret_1d = float(_safe_pct_change(close, 1).iloc[i]) if n > 1 else np.nan
        ret_5d = float(_safe_pct_change(close, 5).iloc[i]) if n > 5 else np.nan
        ret_10d = float(_safe_pct_change(close, 10).iloc[i]) if n > 10 else np.nan
        ret_21d = float(_safe_pct_change(close, 21).iloc[i]) if n > 21 else np.nan
        ret_63d = float(_safe_pct_change(close, 63).iloc[i]) if n > 63 else np.nan

        # Volatility
        vol_5d = float(_realized_vol(daily_ret, 5).iloc[i]) if n > 5 else np.nan
        vol_21d = float(_realized_vol(daily_ret, 21).iloc[i]) if n > 21 else np.nan

        # ATR %
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr_14 = tr.rolling(14, min_periods=7).mean()
        current_close = float(close.iloc[i])
        atr_pct = float(atr_14.iloc[i] / current_close * 100) if current_close > 0 and n > 14 else np.nan

        # Bollinger bandwidth
        ma20 = close.rolling(20, min_periods=10).mean()
        std20 = close.rolling(20, min_periods=10).std()
        bb_width = float(2 * std20.iloc[i] / ma20.iloc[i] * 100) if n >= 20 and float(ma20.iloc[i]) > 0 else np.nan

        # Volume
        vol_ma20 = volume.rolling(20, min_periods=10).mean()
        vol_ratio = float(volume.iloc[i]) / float(vol_ma20.iloc[i]) if float(vol_ma20.iloc[i]) > 0 else np.nan
        obv_slope = _obv_slope(close, volume, 10)

        # Mean-reversion / trend
        ma50 = close.rolling(50, min_periods=25).mean()
        ma200 = close.rolling(200, min_periods=100).mean()
        dist_ma20 = (current_close - float(ma20.iloc[i])) / float(ma20.iloc[i]) * 100 if n >= 20 else np.nan
        dist_ma50 = (current_close - float(ma50.iloc[i])) / float(ma50.iloc[i]) * 100 if n >= 50 else np.nan
        dist_ma200 = (current_close - float(ma200.iloc[i])) / float(ma200.iloc[i]) * 100 if n >= 200 else np.nan
        ma20_above_ma50 = 1.0 if n >= 50 and float(ma20.iloc[i]) > float(ma50.iloc[i]) else 0.0
        ma50_above_ma200 = 1.0 if n >= 200 and float(ma50.iloc[i]) > float(ma200.iloc[i]) else 0.0

        # Time series features
        autocorr_1 = _autocorrelation(daily_ret.dropna(), 1)
        autocorr_5 = _autocorrelation(daily_ret.dropna(), 5)
        trend_str = _trend_strength(close, 10)
        mean_rev = _mean_reversion_signal(daily_ret.dropna(), 5)

        # YOLO
        yolo = yolo_data.get(str(ticker), (0, 0.0))

        row = {
            "ticker": str(ticker),
            "ret_1d": ret_1d if np.isfinite(ret_1d) else np.nan,
            "ret_5d": ret_5d if np.isfinite(ret_5d) else np.nan,
            "ret_10d": ret_10d if np.isfinite(ret_10d) else np.nan,
            "ret_21d": ret_21d if np.isfinite(ret_21d) else np.nan,
            "ret_63d": ret_63d if np.isfinite(ret_63d) else np.nan,
            "vol_5d": vol_5d, "vol_21d": vol_21d,
            "atr_pct_14": atr_pct, "bb_width": bb_width,
            "volume_ratio_20d": vol_ratio, "obv_slope_10d": obv_slope,
            "dist_ma20_pct": dist_ma20, "dist_ma50_pct": dist_ma50, "dist_ma200_pct": dist_ma200,
            "ma20_above_ma50": ma20_above_ma50, "ma50_above_ma200": ma50_above_ma200,
            "vix_level": vix_latest, "vix_percentile": vix_pctile,
            "vix_ma20_ratio": vix_ma20_ratio, "vix_ret_5d": vix_ret_5d,
            "autocorr_lag1": autocorr_1, "autocorr_lag5": autocorr_5,
            "trend_strength_10d": trend_str, "mean_reversion_5d": mean_rev,
            "day_of_week": float(dow), "month": float(month),
            "is_month_end": float(is_month_end), "is_quarter_end": float(is_quarter_end),
            "has_yolo_pattern": float(yolo[0]), "yolo_confidence": yolo[1],
            "rank_ret_5d": np.nan, "rank_ret_21d": np.nan,
            "rank_vol_21d": np.nan, "rank_volume_ratio": np.nan,
        }
        results.append(row)

    if not results:
        return pd.DataFrame(columns=["ticker"] + FEATURE_COLUMNS)

    feat_df = pd.DataFrame(results)

    # Cross-sectional ranks
    for col, rank_col in [
        ("ret_5d", "rank_ret_5d"), ("ret_21d", "rank_ret_21d"),
        ("vol_21d", "rank_vol_21d"), ("volume_ratio_20d", "rank_volume_ratio"),
    ]:
        valid = feat_df[col].notna()
        if valid.sum() >= 5:
            feat_df.loc[valid, rank_col] = feat_df.loc[valid, col].rank(pct=True)

    # Merge macro features (same for all tickers on this date)
    try:
        from trader_koo.ml.macro_features import extract_macro_features, MACRO_FEATURE_COLUMNS

        macro = extract_macro_features(conn, as_of_date=as_of_date, lookback_days=lookback_days)
        for col_name in MACRO_FEATURE_COLUMNS:
            macro_col = f"macro_{col_name}"
            if macro_col in FEATURE_COLUMNS:
                feat_df[macro_col] = macro.get(col_name, np.nan)
    except Exception as exc:
        LOG.warning("Macro feature extraction failed (non-fatal): %s", exc)

    # FRED macro data (yield curve, credit stress, rates)
    try:
        from trader_koo.ml.external_data import get_fred_latest

        feat_df["fred_yield_curve_10y2y"] = get_fred_latest("T10Y2Y")
        feat_df["fred_high_yield_oas"] = get_fred_latest("BAMLH0A0HYM2")
        feat_df["fred_fed_funds_rate"] = get_fred_latest("DFF")
    except Exception as exc:
        LOG.warning("FRED feature extraction failed (non-fatal): %s", exc)

    # Ensure all expected columns exist
    for col in FEATURE_COLUMNS:
        if col not in feat_df.columns:
            feat_df[col] = np.nan

    return feat_df.set_index("ticker")[FEATURE_COLUMNS].copy()
