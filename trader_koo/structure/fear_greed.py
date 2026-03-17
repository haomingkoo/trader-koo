"""Market sentiment composite gauge.

Computes a 0-100 score from five market indicators:
1. Market Momentum (SPY vs 125-day MA)
2. Market Volatility (VIX level)
3. Stock Price Breadth (advancing vs declining)
4. Stock Price Strength (52-week high/low proximity)
5. Put/Call Ratio (options data)

Scoring zones:
  0-25: Extreme Fear
 25-45: Fear
 45-55: Neutral
 55-75: Greed
 75-100: Extreme Greed
"""
from __future__ import annotations

import datetime as dt
import logging
import math
import sqlite3
from typing import Any

from trader_koo.news_sentiment import get_external_news_sentiment

logger = logging.getLogger(__name__)

_METHODOLOGY_BASIS = [
    "SPY vs 125-day moving average",
    "VIX level",
    "Advancers vs decliners",
    "52-week strength",
    "Put/call ratio",
]

_METHODOLOGY_SUMMARY = (
    "Internal market-data composite built from trend, volatility, breadth, "
    "price strength, and options positioning. No social or news scraping."
)

# ---------------------------------------------------------------------------
# Zone definitions
# ---------------------------------------------------------------------------

_ZONES: list[tuple[int, int, str, str]] = [
    (0, 25, "Extreme Fear", "#ff6b6b"),
    (25, 45, "Fear", "#ff9800"),
    (45, 55, "Neutral", "#fdd835"),
    (55, 75, "Greed", "#4caf50"),
    (75, 100, "Extreme Greed", "#1b5e20"),
]


def _label_for_score(score: int) -> tuple[str, str]:
    """Return (label, hex_color) for a 0-100 score."""
    for lo, hi, label, color in _ZONES:
        if score < hi:
            return label, color
    return "Extreme Greed", "#1b5e20"


def _signal_for_score(score: float) -> str:
    """Return a signal label for a component score."""
    if score < 25:
        return "Extreme Fear"
    if score < 45:
        return "Fear"
    if score < 55:
        return "Neutral"
    if score < 75:
        return "Greed"
    return "Extreme Greed"


def _blended_sentiment(
    internal_score: int | None,
    external_news: dict[str, Any],
) -> tuple[int | None, str | None, str | None, str | None]:
    news_score = external_news.get("score")
    news_available = bool(external_news.get("available"))
    if internal_score is None or not news_available or not isinstance(news_score, int | float):
        return None, None, None, None
    blended_score = round(internal_score * 0.75 + float(news_score) * 0.25)
    blended_label, blended_color = _label_for_score(blended_score)
    blended_summary = "75% internal market composite + 25% external news sentiment"
    return blended_score, blended_label, blended_color, blended_summary


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _fetch_close_series(
    conn: sqlite3.Connection,
    ticker: str,
    limit: int,
) -> list[tuple[str, float]]:
    """Fetch (date, close) pairs, newest first."""
    rows = conn.execute(
        """
        SELECT date, CAST(close AS REAL)
        FROM price_daily
        WHERE ticker = ? AND close IS NOT NULL
        ORDER BY date DESC
        LIMIT ?
        """,
        (ticker, limit),
    ).fetchall()
    return [(str(r[0]), float(r[1])) for r in rows if r[1] is not None]


def _fetch_latest_close(conn: sqlite3.Connection, ticker: str) -> float | None:
    """Fetch latest close for *ticker*."""
    series = _fetch_close_series(conn, ticker, 1)
    return series[0][1] if series else None


def _simple_ma(closes: list[float], window: int) -> float | None:
    """Compute simple moving average from a list (newest first)."""
    if len(closes) < window:
        return None
    return sum(closes[:window]) / window


# ---------------------------------------------------------------------------
# Individual component scorers (each returns 0-100 or None)
# ---------------------------------------------------------------------------

def _score_market_momentum(conn: sqlite3.Connection) -> tuple[float | None, str]:
    """SPY price vs 125-day MA. Above = greed, below = fear."""
    series = _fetch_close_series(conn, "SPY", 130)
    if len(series) < 2:
        return None, "Insufficient SPY data"

    current = series[0][1]
    closes = [c for _, c in series]
    ma_125 = _simple_ma(closes, 125)

    if ma_125 is None or ma_125 == 0:
        return None, "Insufficient data for 125-day MA"

    pct_diff = ((current - ma_125) / ma_125) * 100

    # Map: -10% below MA = 0, +10% above MA = 100 (linear)
    score = max(0, min(100, 50 + pct_diff * 5))

    direction = "above" if pct_diff >= 0 else "below"
    detail = f"SPY {abs(pct_diff):.1f}% {direction} 125-day MA"
    return round(score, 1), detail


def _score_market_volatility(conn: sqlite3.Connection) -> tuple[float | None, str]:
    """VIX level: <15 = extreme greed, >30 = extreme fear."""
    vix = _fetch_latest_close(conn, "^VIX")
    if vix is None:
        return None, "VIX data unavailable"

    # Map: VIX 10 = score 100, VIX 40 = score 0 (inverse linear)
    score = max(0, min(100, ((40 - vix) / 30) * 100))

    if vix < 15:
        detail = f"VIX at {vix:.2f} (below 15)"
    elif vix < 20:
        detail = f"VIX at {vix:.2f} (normal range)"
    elif vix < 25:
        detail = f"VIX at {vix:.2f} (elevated)"
    elif vix < 30:
        detail = f"VIX at {vix:.2f} (above 25)"
    else:
        detail = f"VIX at {vix:.2f} (above 30)"

    return round(score, 1), detail


def _score_stock_breadth(conn: sqlite3.Connection) -> tuple[float | None, str]:
    """Market breadth from tracked tickers: advancing vs declining."""
    try:
        rows = conn.execute(
            """
            SELECT ticker, CAST(close AS REAL) AS close
            FROM price_daily
            WHERE date = (SELECT MAX(date) FROM price_daily)
              AND close IS NOT NULL
            """,
        ).fetchall()

        if len(rows) < 10:
            return None, "Insufficient breadth data"

        # Get previous day's closes
        prev_rows = conn.execute(
            """
            SELECT p.ticker, CAST(p.close AS REAL) AS close
            FROM price_daily p
            WHERE p.date = (
                SELECT MAX(date) FROM price_daily
                WHERE date < (SELECT MAX(date) FROM price_daily)
            )
              AND p.close IS NOT NULL
            """,
        ).fetchall()

        prev_map = {str(r[0]): float(r[1]) for r in prev_rows}

        advancers = 0
        decliners = 0
        for row in rows:
            ticker = str(row[0])
            current = float(row[1])
            prev = prev_map.get(ticker)
            if prev is None or prev == 0:
                continue
            if current > prev:
                advancers += 1
            elif current < prev:
                decliners += 1

        total = advancers + decliners
        if total == 0:
            return None, "No advancing/declining data"

        pct_advancing = (advancers / total) * 100

        # Map: 30% advancing = score 0, 70% advancing = score 100
        score = max(0, min(100, (pct_advancing - 30) * (100 / 40)))

        detail = f"{pct_advancing:.1f}% advancing ({advancers} vs {decliners})"
        return round(score, 1), detail

    except Exception as exc:
        logger.warning("Breadth scoring failed: %s", exc)
        return None, f"Breadth computation error: {exc}"


def _score_stock_strength(conn: sqlite3.Connection) -> tuple[float | None, str]:
    """52-week high vs low proximity across tracked tickers."""
    try:
        rows = conn.execute(
            """
            SELECT
                ticker,
                CAST(close AS REAL) AS close,
                CAST(MAX(high) OVER (
                    PARTITION BY ticker
                    ORDER BY date
                    ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
                ) AS REAL) AS high_52w,
                CAST(MIN(low) OVER (
                    PARTITION BY ticker
                    ORDER BY date
                    ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
                ) AS REAL) AS low_52w
            FROM price_daily
            WHERE date = (SELECT MAX(date) FROM price_daily)
              AND close IS NOT NULL
            """,
        ).fetchall()

        if len(rows) < 10:
            # Fallback: use a simpler query
            rows = conn.execute(
                """
                WITH latest AS (
                    SELECT ticker, CAST(close AS REAL) AS close
                    FROM price_daily
                    WHERE date = (SELECT MAX(date) FROM price_daily)
                      AND close IS NOT NULL
                ),
                yearly AS (
                    SELECT
                        ticker,
                        MAX(CAST(high AS REAL)) AS high_52w,
                        MIN(CAST(low AS REAL)) AS low_52w
                    FROM price_daily
                    WHERE date >= date((SELECT MAX(date) FROM price_daily), '-252 days')
                      AND high IS NOT NULL AND low IS NOT NULL
                    GROUP BY ticker
                )
                SELECT l.ticker, l.close, y.high_52w, y.low_52w
                FROM latest l
                JOIN yearly y ON l.ticker = y.ticker
                """,
            ).fetchall()

        if len(rows) < 10:
            return None, "Insufficient 52-week data"

        near_high = 0
        near_low = 0
        total = 0

        for row in rows:
            close = float(row[1])
            high_52w = float(row[2]) if row[2] else None
            low_52w = float(row[3]) if row[3] else None

            if high_52w is None or low_52w is None or high_52w == 0:
                continue

            total += 1
            pct_from_high = ((high_52w - close) / high_52w) * 100
            pct_from_low = ((close - low_52w) / low_52w) * 100 if low_52w > 0 else 0

            if pct_from_high <= 5:
                near_high += 1
            if pct_from_low <= 5:
                near_low += 1

        if total == 0:
            return None, "No 52-week data available"

        high_pct = (near_high / total) * 100
        low_pct = (near_low / total) * 100

        # Net score: more near highs = greed, more near lows = fear
        # If 30% near high and 5% near low -> greed
        net = high_pct - low_pct
        # Map: -20 = score 0, +20 = score 100
        score = max(0, min(100, 50 + net * 2.5))

        detail = f"{near_high} near 52w high, {near_low} near 52w low (of {total})"
        return round(score, 1), detail

    except Exception as exc:
        logger.warning("Stock strength scoring failed: %s", exc)
        return None, f"Strength computation error: {exc}"


def _score_put_call_ratio(conn: sqlite3.Connection) -> tuple[float | None, str]:
    """Aggregate put/call OI ratio across tracked tickers."""
    try:
        row = conn.execute(
            """
            SELECT
                SUM(CAST(call_oi AS REAL)) AS total_call_oi,
                SUM(CAST(put_oi AS REAL)) AS total_put_oi
            FROM options_summary
            WHERE snapshot_ts = (SELECT MAX(snapshot_ts) FROM options_summary)
              AND call_oi IS NOT NULL AND put_oi IS NOT NULL
            """,
        ).fetchone()

        if not row or row[0] is None or row[1] is None:
            return None, "Options data unavailable"

        total_call = float(row[0])
        total_put = float(row[1])

        if total_call == 0:
            return None, "No call OI data"

        ratio = total_put / total_call

        # Map: ratio 0.5 = score 100 (greed), ratio 1.5 = score 0 (fear)
        score = max(0, min(100, ((1.5 - ratio) / 1.0) * 100))

        detail = f"P/C ratio {ratio:.2f}"
        return round(score, 1), detail

    except Exception as exc:
        logger.warning("Put/call scoring failed: %s", exc)
        return None, f"Put/call computation error: {exc}"


# ---------------------------------------------------------------------------
# Historical score helpers
# ---------------------------------------------------------------------------

def _compute_historical_score(
    conn: sqlite3.Connection,
    target_date: str,
) -> int | None:
    """Compute a rough Fear & Greed score for a specific date.

    Uses SPY price vs 125-day MA and VIX level only (the two most
    reliable indicators for historical lookback).
    """
    try:
        # Get SPY close for target date
        spy_row = conn.execute(
            """
            SELECT CAST(close AS REAL)
            FROM price_daily
            WHERE ticker = 'SPY' AND date <= ?
            ORDER BY date DESC
            LIMIT 1
            """,
            (target_date,),
        ).fetchone()

        # Get SPY 125-day MA as of target date
        spy_closes = conn.execute(
            """
            SELECT CAST(close AS REAL)
            FROM price_daily
            WHERE ticker = 'SPY' AND date <= ?
            ORDER BY date DESC
            LIMIT 125
            """,
            (target_date,),
        ).fetchall()

        vix_row = conn.execute(
            """
            SELECT CAST(close AS REAL)
            FROM price_daily
            WHERE ticker = '^VIX' AND date <= ?
            ORDER BY date DESC
            LIMIT 1
            """,
            (target_date,),
        ).fetchone()

        scores = []

        if spy_row and len(spy_closes) >= 125:
            spy_current = float(spy_row[0])
            spy_ma = sum(float(r[0]) for r in spy_closes) / len(spy_closes)
            if spy_ma > 0:
                pct_diff = ((spy_current - spy_ma) / spy_ma) * 100
                scores.append(max(0, min(100, 50 + pct_diff * 5)))

        if vix_row:
            vix_val = float(vix_row[0])
            scores.append(max(0, min(100, ((40 - vix_val) / 30) * 100)))

        if not scores:
            return None

        return round(sum(scores) / len(scores))

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_fear_greed_index(conn: sqlite3.Connection) -> dict[str, Any]:
    """Compute a 0-100 market sentiment score from multiple market indicators."""
    components: list[dict[str, Any]] = []
    valid_scores: list[float] = []
    external_news = get_external_news_sentiment()

    # 1. Market Momentum
    score, detail = _score_market_momentum(conn)
    components.append({
        "name": "Market Momentum",
        "score": score,
        "signal": _signal_for_score(score) if score is not None else "Unavailable",
        "detail": detail,
    })
    if score is not None:
        valid_scores.append(score)

    # 2. Market Volatility
    score, detail = _score_market_volatility(conn)
    components.append({
        "name": "Market Volatility",
        "score": score,
        "signal": _signal_for_score(score) if score is not None else "Unavailable",
        "detail": detail,
    })
    if score is not None:
        valid_scores.append(score)

    # 3. Stock Price Breadth
    score, detail = _score_stock_breadth(conn)
    components.append({
        "name": "Stock Price Breadth",
        "score": score,
        "signal": _signal_for_score(score) if score is not None else "Unavailable",
        "detail": detail,
    })
    if score is not None:
        valid_scores.append(score)

    # 4. Stock Price Strength
    score, detail = _score_stock_strength(conn)
    components.append({
        "name": "Stock Price Strength",
        "score": score,
        "signal": _signal_for_score(score) if score is not None else "Unavailable",
        "detail": detail,
    })
    if score is not None:
        valid_scores.append(score)

    # 5. Put/Call Ratio
    score, detail = _score_put_call_ratio(conn)
    components.append({
        "name": "Put/Call Ratio",
        "score": score,
        "signal": _signal_for_score(score) if score is not None else "Unavailable",
        "detail": detail,
    })
    if score is not None:
        valid_scores.append(score)

    # Compute overall score
    if not valid_scores:
        blended_score, blended_label, blended_color, blended_summary = _blended_sentiment(
            None,
            external_news,
        )
        return {
            "score": None,
            "label": "Unavailable",
            "color": "#757575",
            "previous_close": None,
            "one_week_ago": None,
            "one_month_ago": None,
            "methodology": "internal_market_composite",
            "summary": _METHODOLOGY_SUMMARY,
            "basis": _METHODOLOGY_BASIS,
            "uses_social_sentiment": False,
            "external_news": external_news,
            "blended_score": blended_score,
            "blended_label": blended_label,
            "blended_color": blended_color,
            "blended_summary": blended_summary,
            "components": components,
        }

    overall = round(sum(valid_scores) / len(valid_scores))
    label, color = _label_for_score(overall)
    blended_score, blended_label, blended_color, blended_summary = _blended_sentiment(
        overall,
        external_news,
    )

    # Historical comparisons
    try:
        latest_date_row = conn.execute(
            "SELECT MAX(date) FROM price_daily",
        ).fetchone()
        latest_date = str(latest_date_row[0]) if latest_date_row and latest_date_row[0] else None
    except Exception:
        latest_date = None

    previous_close = None
    one_week_ago = None
    one_month_ago = None

    if latest_date:
        try:
            base = dt.date.fromisoformat(latest_date)
            prev_date = (base - dt.timedelta(days=1)).isoformat()
            week_date = (base - dt.timedelta(days=7)).isoformat()
            month_date = (base - dt.timedelta(days=30)).isoformat()

            previous_close = _compute_historical_score(conn, prev_date)
            one_week_ago = _compute_historical_score(conn, week_date)
            one_month_ago = _compute_historical_score(conn, month_date)
        except Exception as exc:
            logger.warning("Historical F&G scores failed: %s", exc)

    return {
        "score": overall,
        "label": label,
        "color": color,
        "previous_close": previous_close,
        "one_week_ago": one_week_ago,
        "one_month_ago": one_month_ago,
        "methodology": "internal_market_composite",
        "summary": _METHODOLOGY_SUMMARY,
        "basis": _METHODOLOGY_BASIS,
        "uses_social_sentiment": False,
        "external_news": external_news,
        "blended_score": blended_score,
        "blended_label": blended_label,
        "blended_color": blended_color,
        "blended_summary": blended_summary,
        "components": components,
    }
