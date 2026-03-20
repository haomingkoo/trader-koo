"""Feature extraction for the swing-trade forecasting model.

All features are backward-looking — no future data is used.

Feature categories:
1. Multi-horizon momentum (1d, 5d, 10d, 21d, 63d returns)
2. Volatility (realized vol, ATR %, Bollinger bandwidth)
3. Volume (ratio vs 20d MA, OBV trend, volume-confirmed momentum)
4. Range expansion (ATR expansion, gap percentage)
5. Mean-reversion (distance from MA20/50/200)
6. Regime context (VIX level, VIX percentile, VIX MA ratio)
7. Time series (lag autocorrelation, recent trend strength)
8. Seasonality (day of week, month, quarter effects)
9. YOLO patterns (if available in DB)
10. Cross-sectional ranks
11. News sentiment (Finnhub company-news + RSS lexicon scoring)
12. Earnings proximity (days to next earnings, earnings week flag)
13. Polymarket prediction market probabilities
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import sqlite3
import urllib.error
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

# Full feature set (54 features) — kept for reference / ablation comparison.
# Pruned from the original set based on first training runs (dropped seasonality,
# YOLO, binary MA flags, redundant macro features).
FEATURE_COLUMNS_FULL = [
    # === Per-ticker features (what makes THIS stock different) ===
    # Momentum (multi-horizon)
    "ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d",
    # Volatility (top predictive cluster)
    "vol_5d", "vol_21d", "atr_pct_14", "bb_width",
    # Volume
    "volume_ratio_20d", "obv_slope_10d",
    "volume_confirmed_momentum",
    # ATR expansion (breakout vs consolidation)
    "atr_expansion_5d",
    # Gap
    "gap_pct_1d",
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
    # Credit spread velocity (change in HY OAS — leading risk signal)
    "fred_hy_oas_change_5d",
    "fred_hy_oas_change_21d",
    # Commodities + global risk (from macro_features.py)
    "macro_gold_ret_5d", "macro_gold_ret_21d",
    "macro_oil_ret_5d", "macro_oil_ret_21d",
    "macro_em_ret_5d", "macro_smallcap_ret_5d", "macro_usd_ret_5d",
    # Sector rotation (from sector_rotation.py)
    "sector_rank", "sector_momentum_5d", "sector_momentum_21d",
    "leading_sector_momentum", "lagging_sector_momentum", "sector_dispersion",
    # News sentiment (Finnhub company-news + RSS lexicon scoring)
    "news_sentiment_score",
    # Earnings proximity (vol expansion + positioning into catalyst)
    "days_to_next_earnings", "is_earnings_week",
    # NOTE: Polymarket features (polymarket_fed_cut_prob, polymarket_recession_prob,
    # polymarket_macro_sentiment) are intentionally EXCLUDED from training features.
    # They are always NaN during historical training (API returns current prices only),
    # so the model never learns from them and they cause distribution shift at inference.
    # They are still computed and available via extract_features_for_universe() for
    # display purposes but are not part of the model's feature vector.
]

# Slim feature set (~18 features) — top non-redundant features by panel importance.
# Drops correlated pairs (ret_5d/rank_ret_5d, vol_5d/atr_pct_14, overlapping VIX).
# Used as default for training; reduces overfitting risk and training time.
FEATURE_COLUMNS_SLIM = [
    # Per-ticker: momentum (multi-horizon, no rank duplicates)
    "ret_1d",                   # importance: 1595
    "ret_21d",                  # importance: 1558
    "ret_63d",                  # importance: 1334
    # Per-ticker: volatility (one measure, not two correlated ones)
    "atr_pct_14",               # importance: 1884
    "vol_21d",                  # importance: 1557
    # Per-ticker: cross-sectional rank (one vol rank, best-performing)
    "rank_vol_21d",             # importance: 1534
    # Per-ticker: volume acceleration + range expansion
    "volume_confirmed_momentum",  # institutional accumulation/distribution
    "atr_expansion_5d",           # breakout vs consolidation regime
    "gap_pct_1d",                 # overnight gap signal
    # Per-ticker: trend / mean-reversion
    "dist_ma50_pct",            # trend position vs key MA
    "mean_reversion_5d",        # short-term reversion signal
    # Macro: broad market (strongest single macro feature)
    "macro_sp500_ret_63d",      # importance: 2529
    # Macro: rates
    "macro_tnx_ret_21d",        # importance: 1510
    # Macro: credit stress velocity (leading indicator)
    "fred_hy_oas_change_5d",    # new: 5d change in HY OAS
    # Sector rotation (top cluster)
    "sector_dispersion",        # importance: 2217
    "leading_sector_momentum",  # importance: 1402
    "sector_rank",              # per-ticker sector positioning
    # Earnings catalyst
    "days_to_next_earnings",    # vol expansion into catalyst
]

# Default feature set used by the trainer and scorer.
# Switch to FEATURE_COLUMNS_FULL for ablation experiments.
FEATURE_COLUMNS = FEATURE_COLUMNS_SLIM


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
    return slope / max(recent.abs().mean(), 1.0)  # normalize; floor at 1.0 to avoid extreme values for low-volume stocks


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


def _get_news_sentiment_scores(
    tickers: list[str],
    as_of_date: str,
    lookback_days: int = 3,
    conn: sqlite3.Connection | None = None,
) -> dict[str, float]:
    """Fetch per-ticker news sentiment scores, using DB cache when available.

    Priority:
    1. DB cache (news_sentiment_cache table) -- instant, no API calls
    2. Live Finnhub API -- only for cache misses during live scoring

    Returns a dict mapping ticker -> raw sentiment score in [-1, 1].
    Uses only news published on or before *as_of_date* (backward-looking).
    Returns NaN for tickers with no scoreable headlines or on API failure.
    """
    scores: dict[str, float] = {}
    uncached_tickers: list[str] = list(tickers)

    # --- Step 1: Try DB cache first ---
    if conn is not None:
        try:
            from trader_koo.ml.sentiment_cache import lookup_cached_sentiment_batch

            cached = lookup_cached_sentiment_batch(conn, tickers, as_of_date)
            if cached:
                scores.update(cached)
                uncached_tickers = [t for t in tickers if t not in cached]
                LOG.debug(
                    "Sentiment cache hit: %d/%d tickers for %s",
                    len(cached), len(tickers), as_of_date,
                )
        except Exception as exc:
            LOG.debug("Sentiment cache lookup failed (non-fatal): %s", exc)

    if not uncached_tickers:
        return scores

    # --- Step 2: Fall back to live Finnhub API for cache misses ---
    try:
        from trader_koo.news_sentiment import _finnhub_get, _finnhub_key
        from trader_koo.rss_news import _score_headline
    except ImportError:
        LOG.warning("News sentiment modules not available; skipping news features")
        return scores

    api_key = _finnhub_key()
    if not api_key:
        LOG.debug("FINNHUB_API_KEY not set; news_sentiment_score will be NaN for uncached tickers")
        return scores

    date_to = as_of_date
    date_from = (
        pd.Timestamp(as_of_date) - pd.Timedelta(days=lookback_days)
    ).strftime("%Y-%m-%d")

    for ticker in uncached_tickers:
        try:
            articles = _finnhub_get(
                "company-news",
                {"symbol": ticker, "from": date_from, "to": date_to},
                api_key,
            )
            if not isinstance(articles, list):
                scores[ticker] = np.nan
                continue

            # Filter articles to only those published on or before as_of_date
            # Finnhub returns datetime as unix timestamp in "datetime" field
            as_of_ts = pd.Timestamp(as_of_date).timestamp() + 86400  # end of day
            headline_scores: list[float] = []
            for item in articles[:30]:
                if not isinstance(item, dict):
                    continue
                article_ts = item.get("datetime")
                if isinstance(article_ts, int | float) and article_ts > as_of_ts:
                    continue  # skip future articles (safety check)
                headline = str(item.get("headline") or "").strip()
                summary = str(item.get("summary") or "").strip()
                if not headline:
                    continue
                raw, _bullish, _bearish = _score_headline(headline, summary)
                if raw is not None and raw != 0.0:
                    headline_scores.append(raw)

            if headline_scores:
                scores[ticker] = sum(headline_scores) / len(headline_scores)
            else:
                scores[ticker] = np.nan
        except Exception as exc:
            LOG.debug("News sentiment fetch failed for %s: %s", ticker, exc)
            scores[ticker] = np.nan

    return scores


def _parse_finviz_earnings_date(
    raw_json_str: str | None,
    as_of_date: dt.date,
) -> dt.date | None:
    """Extract and parse earnings date from Finviz raw_json.

    Uses the same parsing logic as catalyst_data.parse_earnings_value
    but inlined here to avoid a circular import and keep the feature
    pipeline self-contained.
    """
    if not raw_json_str:
        return None
    try:
        raw_obj = json.loads(raw_json_str)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(raw_obj, dict):
        return None

    # Extract the earnings value from the raw object
    earnings_raw = None
    for key in ("Earnings", "Earnings Date", "earnings", "earningsDate"):
        value = raw_obj.get(key)
        if value not in {None, ""}:
            out = str(value).strip()
            if out and out != "-":
                earnings_raw = out
                break
    if not earnings_raw:
        return None

    # Parse the earnings string into a date
    upper = earnings_raw.upper()
    # Strip session tokens (BMO/AMC) before date parsing
    cleaned = upper
    for token in (
        "BMO", "AMC", "BEFORE OPEN", "BEFORE MARKET OPEN",
        "AFTER CLOSE", "AFTER MARKET CLOSE", "/", "|",
    ):
        cleaned = cleaned.replace(token, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")

    parsed_date: dt.date | None = None
    if cleaned.startswith("TODAY"):
        parsed_date = as_of_date
    elif cleaned.startswith("TOMORROW"):
        parsed_date = as_of_date + dt.timedelta(days=1)
    else:
        fragments = [cleaned]
        if "," in cleaned:
            fragments.append(cleaned.replace(",", ""))
        for frag in fragments:
            for fmt in ("%b %d %Y", "%B %d %Y", "%b %d", "%B %d"):
                try:
                    parsed = dt.datetime.strptime(frag, fmt)
                except ValueError:
                    continue
                year = parsed.year if "%Y" in fmt else as_of_date.year
                parsed_date = dt.date(year, parsed.month, parsed.day)
                # If no year in format and date is far in the past, bump year
                if "%Y" not in fmt and parsed_date < (as_of_date - dt.timedelta(days=45)):
                    parsed_date = dt.date(year + 1, parsed.month, parsed.day)
                break
            if parsed_date is not None:
                break

    return parsed_date


def _get_earnings_proximity(
    conn: sqlite3.Connection,
    tickers: list[str],
    as_of_date: str,
) -> dict[str, float]:
    """Return days-to-next-earnings for each ticker.

    Data sources (checked in order):
    1. Finnhub earnings calendar cached in external_data_cache
    2. Finviz fundamentals "Earnings" field (from raw_json)

    Only earnings dates on or after as_of_date are considered (no leakage).
    Returns {ticker: days_to_next_earnings} with NaN for missing data.
    """
    as_of = dt.date.fromisoformat(as_of_date)
    result: dict[str, float] = {}

    # --- Source 1: Finnhub / Alpha Vantage cached calendar ---
    try:
        has_cache_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='external_data_cache' LIMIT 1"
        ).fetchone()
        if has_cache_table:
            # Find all earnings calendar cache entries (any date range)
            cache_rows = conn.execute(
                "SELECT payload_json FROM external_data_cache "
                "WHERE cache_key LIKE 'finnhub:earnings_calendar:%' "
                "   OR cache_key LIKE 'alpha_vantage:earnings_calendar:%'"
            ).fetchall()
            for (payload_json,) in cache_rows:
                try:
                    events = json.loads(payload_json or "[]")
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(events, list):
                    continue
                for event in events:
                    ticker = str(event.get("ticker") or "").upper().strip()
                    edate_str = str(event.get("earnings_date") or "").strip()
                    if not ticker or not edate_str:
                        continue
                    try:
                        edate = dt.date.fromisoformat(edate_str[:10])
                    except ValueError:
                        continue
                    # Only future or same-day earnings (no leakage)
                    if edate < as_of:
                        continue
                    days = (edate - as_of).days
                    # Keep the closest earnings date per ticker
                    if ticker not in result or days < result[ticker]:
                        result[ticker] = float(days)
    except Exception as exc:
        LOG.debug("Earnings cache lookup failed (non-fatal): %s", exc)

    # --- Source 2: Finviz fundamentals (latest snapshot on or before as_of) ---
    try:
        has_fund_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='finviz_fundamentals' LIMIT 1"
        ).fetchone()
        if has_fund_table:
            # Get the latest snapshot on or before as_of_date
            snap_row = conn.execute(
                "SELECT snapshot_ts FROM finviz_fundamentals "
                "WHERE snapshot_ts <= ? "
                "ORDER BY snapshot_ts DESC LIMIT 1",
                (as_of_date + "T23:59:59Z",),
            ).fetchone()
            if snap_row:
                snapshot_ts = snap_row[0]
                if tickers:
                    placeholders = ",".join("?" * len(tickers))
                    fund_rows = conn.execute(
                        f"SELECT ticker, raw_json FROM finviz_fundamentals "
                        f"WHERE snapshot_ts = ? AND ticker IN ({placeholders})",
                        (snapshot_ts, *tickers),
                    ).fetchall()
                else:
                    fund_rows = conn.execute(
                        "SELECT ticker, raw_json FROM finviz_fundamentals "
                        "WHERE snapshot_ts = ?",
                        (snapshot_ts,),
                    ).fetchall()
                for ticker_val, raw_json in fund_rows:
                    ticker = str(ticker_val or "").upper().strip()
                    if not ticker:
                        continue
                    # Skip if we already have a closer date from the calendar API
                    edate = _parse_finviz_earnings_date(raw_json, as_of)
                    if edate is None or edate < as_of:
                        continue
                    days = (edate - as_of).days
                    if ticker not in result or days < result[ticker]:
                        result[ticker] = float(days)
    except Exception as exc:
        LOG.debug("Finviz earnings lookup failed (non-fatal): %s", exc)

    return result


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
    strict: bool = False,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Extract feature matrix for all tickers as of a specific date.

    All computations use only data on or before *as_of_date*.

    Parameters
    ----------
    strict : bool
        When True (training), unexpected errors propagate so broken
        features are caught early. When False (live scoring), broad
        exception handling prevents API crashes.
    feature_columns : list[str] | None
        Which features to return. Defaults to FEATURE_COLUMNS (slim set).
        Pass FEATURE_COLUMNS_FULL for the full 51-feature set.
    """
    output_cols = feature_columns if feature_columns is not None else FEATURE_COLUMNS
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
        return pd.DataFrame(columns=["ticker"] + output_cols)

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

    # Earnings proximity (days to next earnings report)
    earnings_proximity = _get_earnings_proximity(conn, all_tickers, as_of_date)

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

        # Volume-confirmed momentum: 3d/20d volume ratio signed by 3d return
        vol_3d = volume.rolling(3, min_periods=2).mean()
        vol_ratio_3d_20d = vol_3d / vol_ma20
        ret_3d = close.pct_change(3)
        vol_conf_mom = (
            float(vol_ratio_3d_20d.iloc[i] * np.sign(ret_3d.iloc[i]))
            if n > 20 and np.isfinite(ret_3d.iloc[i]) and float(vol_ma20.iloc[i]) > 0
            else np.nan
        )

        # ATR expansion: 5d ATR / 20d ATR (>1 = expanding range, <1 = contracting)
        atr_5 = tr.rolling(5, min_periods=3).mean()
        atr_20 = tr.rolling(20, min_periods=10).mean()
        atr_exp = (
            float(atr_5.iloc[i] / atr_20.iloc[i])
            if n > 20 and float(atr_20.iloc[i]) > 0
            else np.nan
        )

        # Gap percentage: overnight gap from yesterday's close to today's open
        open_prices = grp["open"]
        gap_pct = (
            float((open_prices.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1])
            if n > 1 and float(close.iloc[i - 1]) > 0
            else np.nan
        )

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
            "vol_5d": vol_5d if np.isfinite(vol_5d) else np.nan,
            "vol_21d": vol_21d if np.isfinite(vol_21d) else np.nan,
            "atr_pct_14": atr_pct if np.isfinite(atr_pct) else np.nan,
            "bb_width": bb_width if np.isfinite(bb_width) else np.nan,
            "volume_ratio_20d": vol_ratio if np.isfinite(vol_ratio) else np.nan,
            "obv_slope_10d": obv_slope if np.isfinite(obv_slope) else np.nan,
            "volume_confirmed_momentum": vol_conf_mom if np.isfinite(vol_conf_mom) else np.nan,
            "atr_expansion_5d": atr_exp if np.isfinite(atr_exp) else np.nan,
            "gap_pct_1d": gap_pct if np.isfinite(gap_pct) else np.nan,
            "dist_ma20_pct": dist_ma20 if np.isfinite(dist_ma20) else np.nan,
            "dist_ma50_pct": dist_ma50 if np.isfinite(dist_ma50) else np.nan,
            "dist_ma200_pct": dist_ma200 if np.isfinite(dist_ma200) else np.nan,
            "ma20_above_ma50": ma20_above_ma50, "ma50_above_ma200": ma50_above_ma200,
            "vix_level": vix_latest, "vix_percentile": vix_pctile,
            "vix_ma20_ratio": vix_ma20_ratio, "vix_ret_5d": vix_ret_5d,
            "autocorr_lag1": autocorr_1, "autocorr_lag5": autocorr_5,
            "trend_strength_10d": trend_str, "mean_reversion_5d": mean_rev,
            "day_of_week": float(dow), "month": float(month),
            "is_month_end": float(is_month_end), "is_quarter_end": float(is_quarter_end),
            "has_yolo_pattern": float(yolo[0]), "yolo_confidence": yolo[1],
            "days_to_next_earnings": earnings_proximity.get(str(ticker), np.nan),
            "is_earnings_week": (
                1.0
                if str(ticker) in earnings_proximity
                and earnings_proximity[str(ticker)] <= 7
                else 0.0
                if str(ticker) in earnings_proximity
                else np.nan
            ),
            "rank_ret_5d": np.nan, "rank_ret_21d": np.nan,
            "rank_vol_21d": np.nan, "rank_volume_ratio": np.nan,
        }
        results.append(row)

    if not results:
        return pd.DataFrame(columns=["ticker"] + output_cols)

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
            if macro_col in FEATURE_COLUMNS_FULL:
                feat_df[macro_col] = macro.get(col_name, np.nan)
    except (sqlite3.OperationalError, KeyError, ValueError) as exc:
        LOG.warning("Macro feature extraction failed: %s", exc)
    except ImportError:
        LOG.warning("macro_features module not available; skipping macro features")
    except Exception as exc:
        if strict:
            raise
        LOG.warning("Macro feature extraction failed (non-fatal): %s", exc)

    # FRED macro data (yield curve, credit stress, rates)
    # NOTE: FRED data is fetched as of as_of_date to prevent data leakage.
    # During training, as_of_date is in the past, so we must NOT use today's values.
    # Tries bulk-prefetched data first (fast O(1) lookup); falls back to per-date
    # API call only if bulk data isn't available.
    try:
        from trader_koo.ml.external_data import fetch_fred_series, lookup_fred_value

        for series_id, col_name in [
            ("T10Y2Y", "fred_yield_curve_10y2y"),
            ("BAMLH0A0HYM2", "fred_high_yield_oas"),
            ("DFF", "fred_fed_funds_rate"),
        ]:
            # Fast path: lookup from bulk-prefetched data
            bulk_value = lookup_fred_value(series_id, as_of_date)
            if bulk_value is not None:
                feat_df[col_name] = bulk_value
                continue

            # Slow path: per-date API call (fallback for non-training use)
            rows = fetch_fred_series(series_id, lookback_days=90, as_of_date=as_of_date)
            value = np.nan
            for r in reversed(rows):
                if r["date"] <= as_of_date:
                    value = r["value"]
                    break
            feat_df[col_name] = value

        # Credit spread velocity: 5d and 21d change in high-yield OAS.
        # Widening OAS = rising credit stress = risk-off signal.
        hy_rows = fetch_fred_series(
            "BAMLH0A0HYM2", lookback_days=90, as_of_date=as_of_date,
        )
        hy_values = [
            r for r in hy_rows if r["date"] <= as_of_date
        ]
        if len(hy_values) >= 6:
            feat_df["fred_hy_oas_change_5d"] = (
                hy_values[-1]["value"] - hy_values[-6]["value"]
            )
        else:
            feat_df["fred_hy_oas_change_5d"] = np.nan
        if len(hy_values) >= 22:
            feat_df["fred_hy_oas_change_21d"] = (
                hy_values[-1]["value"] - hy_values[-22]["value"]
            )
        else:
            feat_df["fred_hy_oas_change_21d"] = np.nan
    except (KeyError, ValueError, urllib.error.URLError) as exc:
        LOG.warning("FRED feature extraction failed: %s", exc)
    except ImportError:
        LOG.warning("external_data module not available; skipping FRED features")
    except Exception as exc:
        if strict:
            raise
        LOG.warning("FRED feature extraction failed (non-fatal): %s", exc)

    # Sector rotation features (cached — one DB query per date, not per ticker)
    try:
        from trader_koo.ml.sector_rotation import compute_sector_features

        # This call caches the date-level sector data (1 batch query)
        market_sector = compute_sector_features(conn, as_of_date=as_of_date)
        for k in ["leading_sector_momentum", "lagging_sector_momentum", "sector_dispersion"]:
            if k in FEATURE_COLUMNS_FULL:
                feat_df[k] = market_sector.get(k, np.nan)

        # Per-ticker lookup from cache — no additional DB queries
        ticker_list = feat_df["ticker"].tolist() if "ticker" in feat_df.columns else feat_df.index.tolist()
        sector_ranks = []
        sector_mom_5d = []
        sector_mom_21d = []
        for tkr in ticker_list:
            s = compute_sector_features(conn, as_of_date=as_of_date, ticker=str(tkr))
            sector_ranks.append(s.get("sector_rank", np.nan))
            sector_mom_5d.append(s.get("sector_momentum_5d", np.nan))
            sector_mom_21d.append(s.get("sector_momentum_21d", np.nan))
        feat_df["sector_rank"] = sector_ranks
        feat_df["sector_momentum_5d"] = sector_mom_5d
        feat_df["sector_momentum_21d"] = sector_mom_21d
    except (sqlite3.OperationalError, KeyError, ValueError) as exc:
        LOG.warning("Sector feature extraction failed: %s", exc)
    except ImportError:
        LOG.warning("sector_rotation module not available; skipping sector features")
    except Exception as exc:
        if strict:
            raise
        LOG.warning("Sector feature extraction failed (non-fatal): %s", exc)

    # News sentiment per ticker (Finnhub company-news + lexicon scoring)
    # NOTE: Uses only articles published on or before as_of_date — no data leakage.
    # Falls back to NaN when FINNHUB_API_KEY is not set or API is unreachable.
    try:
        ticker_list = feat_df["ticker"].tolist() if "ticker" in feat_df.columns else []
        sentiment_scores = _get_news_sentiment_scores(ticker_list, as_of_date, conn=conn)
        if sentiment_scores:
            feat_df["news_sentiment_score"] = feat_df["ticker"].map(
                lambda t: sentiment_scores.get(t, np.nan)
            )
        else:
            feat_df["news_sentiment_score"] = np.nan
    except (KeyError, ValueError, urllib.error.URLError) as exc:
        LOG.warning("News sentiment feature extraction failed: %s", exc)
    except Exception as exc:
        if strict:
            raise
        LOG.warning("News sentiment feature extraction failed (non-fatal): %s", exc)

    # Polymarket prediction market probabilities (date-level, same for all tickers)
    # NOTE: These are LIVE market prices — only meaningful for forward scoring.
    # During historical training (as_of_date in the past), the API still returns
    # current prices, which would be data leakage. We only populate these when
    # as_of_date is today (live scoring); otherwise they stay NaN.
    try:
        today_str = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        is_live_scoring = as_of_date >= today_str

        if is_live_scoring:
            from trader_koo.ml.external_data import get_polymarket_macro_probabilities

            poly_probs = get_polymarket_macro_probabilities()
            for col_name, value in poly_probs.items():
                if col_name in FEATURE_COLUMNS_FULL:
                    feat_df[col_name] = value
        else:
            # Historical date — Polymarket data would be leakage, fill NaN
            for col_name in [
                "polymarket_fed_cut_prob",
                "polymarket_recession_prob",
                "polymarket_macro_sentiment",
            ]:
                feat_df[col_name] = np.nan
    except (KeyError, ValueError, urllib.error.URLError) as exc:
        LOG.warning("Polymarket feature extraction failed: %s", exc)
    except ImportError:
        LOG.warning("external_data module not available; skipping Polymarket features")
    except Exception as exc:
        if strict:
            raise
        LOG.warning("Polymarket feature extraction failed (non-fatal): %s", exc)

    # Ensure all expected columns exist (compute full set, return requested subset)
    for col in FEATURE_COLUMNS_FULL:
        if col not in feat_df.columns:
            feat_df[col] = np.nan

    return feat_df.set_index("ticker")[output_cols].copy()
