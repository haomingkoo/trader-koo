"""Williams %R Green Barrier scan and compact alert-chart rendering."""
from __future__ import annotations

import io
import datetime as dt
import logging
import os
import sqlite3
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from trader_koo.db.price_contract import research_eligible_tickers

GREEN_BARRIER_PERIOD = 14
LOG = logging.getLogger(__name__)


def _configured_threshold() -> float:
    try:
        value = float(os.getenv("TRADER_KOO_GREEN_BARRIER_THRESHOLD", "-98.0"))
    except ValueError:
        LOG.warning("Invalid TRADER_KOO_GREEN_BARRIER_THRESHOLD; using -98.0")
        value = -98.0
    return max(-100.0, min(0.0, value))


GREEN_BARRIER_THRESHOLD = _configured_threshold()


def _configured_max_age_days() -> int:
    try:
        return max(1, int(os.getenv("TRADER_KOO_GREEN_BARRIER_MAX_AGE_DAYS", "7")))
    except ValueError:
        LOG.warning("Invalid TRADER_KOO_GREEN_BARRIER_MAX_AGE_DAYS; using 7")
        return 7


def compute_williams_percent_r(
    frame: pd.DataFrame,
    period: int = GREEN_BARRIER_PERIOD,
) -> pd.Series:
    """Return Williams %R on an OHLC frame using the standard 0 to -100 scale."""
    highs = pd.to_numeric(frame["high"], errors="coerce")
    lows = pd.to_numeric(frame["low"], errors="coerce")
    closes = pd.to_numeric(frame["close"], errors="coerce")
    highest = highs.rolling(period, min_periods=period).max()
    lowest = lows.rolling(period, min_periods=period).min()
    span = (highest - lowest).replace(0, np.nan)
    return -100.0 * (highest - closes) / span


def resample_ohlcv(
    frame: pd.DataFrame,
    timeframe: str,
    *,
    completed_only: bool = False,
    as_of: dt.date | None = None,
) -> pd.DataFrame:
    """Aggregate daily OHLCV rows to weekly or monthly bars."""
    if frame.empty:
        return frame.copy()
    tf = str(timeframe or "monthly").strip().lower()
    if tf not in {"weekly", "monthly"}:
        raise ValueError(f"Unsupported Green Barrier timeframe: {timeframe}")
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date", "open", "high", "low", "close"])
    work["source_date"] = work["date"]
    rule = "W-FRI" if tf == "weekly" else "ME"
    aggregated = (
        work.set_index("date")
        .resample(rule)
        .agg(
            date=("source_date", "last"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index(drop=True)
    )
    if completed_only and not aggregated.empty:
        cutoff = pd.Timestamp(as_of or dt.date.today())
        source_dates = pd.to_datetime(aggregated["date"], errors="coerce")
        if tf == "weekly":
            period_end = source_dates.dt.to_period("W-FRI").dt.end_time.dt.normalize()
        else:
            period_end = source_dates.dt.to_period("M").dt.end_time.dt.normalize()
        aggregated = aggregated.loc[period_end <= cutoff.normalize()].reset_index(drop=True)
    return aggregated


def scan_green_barriers(
    conn: sqlite3.Connection,
    *,
    threshold: float = GREEN_BARRIER_THRESHOLD,
    timeframes: tuple[str, ...] = ("monthly", "weekly"),
    as_of: dt.date | None = None,
    max_age_days: int | None = None,
) -> list[dict[str, Any]]:
    """Find tracked tickers whose latest %R(14) is at or below *threshold*."""
    return scan_green_barrier_snapshot(
        conn,
        threshold=threshold,
        timeframes=timeframes,
        as_of=as_of,
        max_age_days=max_age_days,
    )["hits"]


def scan_green_barrier_snapshot(
    conn: sqlite3.Connection,
    *,
    threshold: float = GREEN_BARRIER_THRESHOLD,
    timeframes: tuple[str, ...] = ("monthly", "weekly"),
    as_of: dt.date | None = None,
    max_age_days: int | None = None,
) -> dict[str, Any]:
    """Return current conditions plus explicit scan-coverage metadata."""
    daily = pd.read_sql_query(
        """
        SELECT ticker, date, open, high, low, close, COALESCE(volume, 0) AS volume
        FROM price_daily
        WHERE open IS NOT NULL AND high IS NOT NULL AND low IS NOT NULL AND close IS NOT NULL
        ORDER BY ticker, date
        """,
        conn,
    )
    reference_date = as_of or dt.date.today()
    allowed_age = _configured_max_age_days() if max_age_days is None else max_age_days
    configured_threshold = max(-100.0, min(0.0, float(threshold)))
    coverage: dict[str, Any] = {
        "scan_asof": reference_date.isoformat(),
        "threshold": round(configured_threshold, 2),
        "max_age_days": allowed_age,
        "source_ticker_count": int(daily["ticker"].nunique()) if not daily.empty else 0,
        "scanned_ticker_count": 0,
        "stale_skipped_count": 0,
        "stale_skipped_tickers": [],
        "invalid_date_skipped_count": 0,
        "insufficient_history_skipped_count": 0,
        "basis_unresolved_skipped_count": 0,
        "basis_unresolved_skipped_tickers": [],
    }
    if daily.empty:
        return {"hits": [], "coverage": coverage}

    hits: list[dict[str, Any]] = []
    stale_tickers: list[str] = []
    eligible_tickers = research_eligible_tickers(conn)
    basis_skipped: list[str] = []
    for ticker, ticker_daily in daily.groupby("ticker", sort=True):
        if str(ticker) not in eligible_tickers:
            basis_skipped.append(str(ticker))
            continue
        latest_source_date = pd.to_datetime(ticker_daily["date"], errors="coerce").max()
        if pd.isna(latest_source_date):
            coverage["invalid_date_skipped_count"] += 1
            continue
        age_days = (reference_date - latest_source_date.date()).days
        if age_days > allowed_age:
            stale_tickers.append(str(ticker))
            continue
        coverage["scanned_ticker_count"] += 1
        for timeframe in timeframes:
            bars = resample_ohlcv(
                ticker_daily,
                timeframe,
                completed_only=True,
                as_of=reference_date,
            )
            if len(bars) < GREEN_BARRIER_PERIOD:
                coverage["insufficient_history_skipped_count"] += 1
                continue
            value = compute_williams_percent_r(bars).iloc[-1]
            if not np.isfinite(value) or float(value) > configured_threshold:
                continue
            hits.append(
                {
                    "ticker": str(ticker),
                    "timeframe": timeframe,
                    "period": GREEN_BARRIER_PERIOD,
                    "value": round(float(value), 2),
                    "threshold": round(configured_threshold, 2),
                    "asof": pd.Timestamp(bars["date"].iloc[-1]).date().isoformat(),
                    "close": round(float(bars["close"].iloc[-1]), 4),
                    "distance_to_barrier": round(float(value) + 100.0, 2),
                    "age_days": max(0, age_days),
                }
            )
    if stale_tickers:
        LOG.warning(
            "Skipped Green Barrier scan for %d ticker(s) older than %d days",
            len(stale_tickers),
            allowed_age,
        )
    coverage["stale_skipped_count"] = len(stale_tickers)
    coverage["stale_skipped_tickers"] = stale_tickers
    coverage["basis_unresolved_skipped_count"] = len(basis_skipped)
    coverage["basis_unresolved_skipped_tickers"] = basis_skipped
    return {
        "hits": sorted(
            hits,
            key=lambda row: (float(row["value"]), row["ticker"], row["timeframe"]),
        ),
        "coverage": coverage,
    }


def build_green_barrier_chart_png(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    timeframe: str,
    as_of: str | dt.date | None = None,
    threshold: float = GREEN_BARRIER_THRESHOLD,
    expected_value: float | None = None,
) -> bytes:
    """Render a Telegram-sized price + Williams %R chart as PNG bytes."""
    symbol = str(ticker or "").strip().upper()
    if symbol not in research_eligible_tickers(conn):
        raise ValueError(f"Price basis is not research eligible for {symbol}")
    query = """
        SELECT date, open, high, low, close, COALESCE(volume, 0) AS volume
        FROM price_daily WHERE ticker = ?
    """
    params: tuple[Any, ...] = (symbol,)
    if as_of is not None:
        cutoff = pd.Timestamp(as_of).date().isoformat()
        query += " AND date <= ?"
        params = (symbol, cutoff)
    query += " ORDER BY date"
    daily = pd.read_sql_query(query, conn, params=params)
    reference_date = pd.Timestamp(as_of).date() if as_of is not None else dt.date.today()
    bars = resample_ohlcv(
        daily,
        timeframe,
        completed_only=True,
        as_of=reference_date,
    )
    if len(bars) < GREEN_BARRIER_PERIOD:
        raise ValueError(f"Not enough {timeframe} bars for {symbol}")
    bars["williams_r"] = compute_williams_percent_r(bars)
    view = bars.tail(60 if timeframe == "monthly" else 104).reset_index(drop=True)

    width, height = 1200, 720
    image = Image.new("RGB", (width, height), (9, 14, 24))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font, label_font, small_font = _load_fonts()
    left, right = 78, width - 42
    price_top, price_bottom = 92, 430
    osc_top, osc_bottom = 500, 660

    latest = view.iloc[-1]
    latest_wr = float(latest["williams_r"])
    if expected_value is not None and abs(latest_wr - float(expected_value)) > 0.02:
        raise ValueError(
            f"Green Barrier snapshot mismatch for {symbol}/{timeframe}: "
            f"report={float(expected_value):.2f}, chart={latest_wr:.2f}"
        )
    chart_threshold = max(-100.0, min(0.0, float(threshold)))
    draw.text((left, 24), f"{symbol}  {timeframe.upper()} GREEN BARRIER CURRENT CONDITION", font=title_font, fill=(240, 246, 255))
    draw.text(
        (left, 60),
        f"Williams %R({GREEN_BARRIER_PERIOD}) {latest_wr:.1f}  |  Close {float(latest['close']):,.2f}  |  As of {pd.Timestamp(latest['date']).date()}",
        font=label_font,
        fill=(123, 231, 190),
    )

    lows = pd.to_numeric(view["low"], errors="coerce")
    highs = pd.to_numeric(view["high"], errors="coerce")
    min_price, max_price = float(lows.min()), float(highs.max())
    price_pad = max((max_price - min_price) * 0.06, max_price * 0.01)
    min_price -= price_pad
    max_price += price_pad
    count = len(view)
    step = (right - left) / max(count, 1)
    body_half = max(2, min(7, int(step * 0.28)))

    draw.rectangle((left, price_top, right, price_bottom), outline=(62, 76, 98, 160), width=1)
    for idx, row in view.iterrows():
        x = int(left + (idx + 0.5) * step)
        o, h, lo, c = (float(row[key]) for key in ("open", "high", "low", "close"))
        color = (56, 211, 159, 235) if c >= o else (255, 107, 107, 235)
        y_high = _scale(h, min_price, max_price, price_bottom, price_top)
        y_low = _scale(lo, min_price, max_price, price_bottom, price_top)
        y_open = _scale(o, min_price, max_price, price_bottom, price_top)
        y_close = _scale(c, min_price, max_price, price_bottom, price_top)
        draw.line((x, y_high, x, y_low), fill=color, width=2)
        draw.rectangle(
            (x - body_half, min(y_open, y_close), x + body_half, max(y_open, y_close) + 1),
            fill=color,
        )

    green_y = _scale(-100, -105, 5, osc_bottom, osc_top)
    trigger_y = _scale(chart_threshold, -105, 5, osc_bottom, osc_top)
    oversold_y = _scale(-80, -105, 5, osc_bottom, osc_top)
    draw.rectangle((left, oversold_y, right, green_y), fill=(56, 211, 159, 28))
    draw.line((left, green_y, right, green_y), fill=(56, 211, 159, 255), width=4)
    if chart_threshold > -100:
        draw.line((left, trigger_y, right, trigger_y), fill=(255, 205, 92, 210), width=2)
    draw.line((left, oversold_y, right, oversold_y), fill=(56, 211, 159, 150), width=2)
    draw.rectangle((left, osc_top, right, osc_bottom), outline=(62, 76, 98, 160), width=1)

    points: list[tuple[int, int]] = []
    for idx, value in enumerate(view["williams_r"]):
        if not np.isfinite(value):
            continue
        x = int(left + (idx + 0.5) * step)
        y = _scale(float(value), -105, 5, osc_bottom, osc_top)
        points.append((x, y))
    if len(points) >= 2:
        draw.line(points, fill=(99, 179, 255, 255), width=3, joint="curve")
    if points:
        x, y = points[-1]
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(56, 211, 159, 255))

    draw.text((14, green_y - 9), "-100", font=small_font, fill=(56, 211, 159))
    draw.text((22, oversold_y - 9), "-80", font=small_font, fill=(123, 231, 190))
    draw.text((right - 228, green_y - 25), "BARRIER  -100", font=label_font, fill=(56, 211, 159))
    if chart_threshold > -100:
        draw.text(
            (left + 10, trigger_y - 22),
            f"CURRENT CONDITION ≤ {chart_threshold:g}",
            font=small_font,
            fill=(255, 205, 92),
        )
    draw.text((left, height - 34), "Research context only — not a buy signal.", font=small_font, fill=(146, 160, 184))

    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _scale(value: float, low: float, high: float, px_low: int, px_high: int) -> int:
    if high <= low:
        return int((px_low + px_high) / 2)
    ratio = (float(value) - low) / (high - low)
    return int(px_low + ratio * (px_high - px_low))


def _load_fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont]:
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return (
                ImageFont.truetype(path, size=28),
                ImageFont.truetype(path, size=18),
                ImageFont.truetype(path, size=14),
            )
        except Exception:
            continue
    default = ImageFont.load_default()
    return default, default, default
