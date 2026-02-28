from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import io
import os
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from trader_koo.features.technical import FeatureConfig, add_basic_features, compute_pivots
from trader_koo.scripts.run_yolo_patterns import render_chart, resample_to_weekly
from trader_koo.structure.levels import LevelConfig, add_fallback_levels, build_levels_from_pivots, select_target_levels


PREVIEW_FEATURE_CFG = FeatureConfig()
PREVIEW_LEVEL_CFG = LevelConfig()
SUPPORT_COLOR = (59, 130, 246)
RESISTANCE_COLOR = (249, 115, 22)
BOX_BULL = (45, 212, 191)
BOX_BEAR = (248, 113, 113)
BOX_NEUTRAL = (250, 204, 21)


def _chart_secret() -> str:
    return str(
        os.getenv("TRADER_KOO_EMAIL_CHART_SECRET")
        or os.getenv("TRADER_KOO_ADMIN_SESSION_SECRET")
        or os.getenv("TRADER_KOO_API_KEY")
        or ""
    ).strip()


def chart_preview_enabled(base_url: str | None) -> bool:
    return bool(base_url and _chart_secret())


def build_chart_preview_url(
    *,
    base_url: str,
    ticker: str,
    timeframe: str = "daily",
    report_ts: str | None = None,
    expires_hours: int = 72,
) -> str | None:
    secret = _chart_secret()
    base = str(base_url or "").strip().rstrip("/")
    symbol = str(ticker or "").strip().upper()
    tf = "weekly" if str(timeframe or "").strip().lower() == "weekly" else "daily"
    if not (secret and base and symbol):
        return None
    exp = int(dt.datetime.now(dt.timezone.utc).timestamp()) + max(1, expires_hours) * 3600
    payload = {
        "ticker": symbol,
        "timeframe": tf,
        "report_ts": str(report_ts or "").strip(),
        "exp": exp,
    }
    sig = _sign_preview_payload(payload, secret)
    payload["sig"] = sig
    return f"{base}/api/email/chart-preview?{urlencode(payload)}"


def verify_chart_preview_signature(
    *,
    ticker: str,
    timeframe: str,
    report_ts: str | None,
    exp: int,
    sig: str,
) -> bool:
    secret = _chart_secret()
    if not secret or not sig:
        return False
    if int(exp) < int(dt.datetime.now(dt.timezone.utc).timestamp()):
        return False
    expected = _sign_preview_payload(
        {
            "ticker": str(ticker or "").strip().upper(),
            "timeframe": "weekly" if str(timeframe or "").strip().lower() == "weekly" else "daily",
            "report_ts": str(report_ts or "").strip(),
            "exp": int(exp),
        },
        secret,
    )
    return hmac.compare_digest(expected, sig)


def _sign_preview_payload(payload: dict[str, Any], secret: str) -> str:
    raw = "|".join(
        [
            str(payload.get("ticker") or "").strip().upper(),
            str(payload.get("timeframe") or "").strip().lower(),
            str(payload.get("report_ts") or "").strip(),
            str(int(payload.get("exp") or 0)),
        ]
    )
    return hmac.new(secret.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256).hexdigest()


def build_email_chart_preview_png(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    timeframe: str = "daily",
) -> bytes:
    symbol = str(ticker or "").strip().upper()
    tf = "weekly" if str(timeframe or "").strip().lower() == "weekly" else "daily"
    daily = _load_price_df(conn, symbol)
    if daily.empty:
        raise ValueError(f"No price data for {symbol}")

    calc_df = daily.tail(520).copy()
    if tf == "weekly":
        calc_df = resample_to_weekly(calc_df)
        view_df = calc_df.tail(110).copy()
    else:
        view_df = calc_df.tail(140).copy()

    if view_df.empty:
        raise ValueError(f"No {tf} chart data for {symbol}")

    model = add_basic_features(calc_df.copy(), PREVIEW_FEATURE_CFG)
    model = compute_pivots(model, left=3, right=3)
    last_close = float(model["close"].iloc[-1])
    levels = build_levels_from_pivots(model, PREVIEW_LEVEL_CFG)
    levels = select_target_levels(levels, last_close, PREVIEW_LEVEL_CFG)
    levels = add_fallback_levels(model, levels, last_close, PREVIEW_LEVEL_CFG)

    img_arr, axes_info = render_chart(view_df, dpi=90, fig_w=10.4, fig_h=5.8)
    image = Image.fromarray(img_arr).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    font, small_font = _load_fonts()
    dates = [str(v) for v in view_df["date"].tolist()]
    yolo = _load_yolo_patterns(conn, symbol, tf, dates)

    _draw_levels(draw, image, axes_info, levels, font, small_font)
    _draw_yolo_boxes(draw, axes_info, dates, yolo, font)
    _draw_header(draw, image, symbol, tf, last_close, font, small_font)

    out = io.BytesIO()
    image.convert("RGB").save(out, format="PNG", optimize=True)
    return out.getvalue()


def _load_price_df(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT date, open, high, low, close, volume
        FROM price_daily
        WHERE ticker = ?
        ORDER BY date
        """,
        conn,
        params=(ticker,),
    )
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    return df.reset_index(drop=True)


def _load_yolo_patterns(
    conn: sqlite3.Connection,
    ticker: str,
    timeframe: str,
    dates: list[str],
) -> list[dict[str, Any]]:
    row = conn.execute(
        "SELECT MAX(as_of_date) FROM yolo_patterns WHERE ticker = ? AND timeframe = ?",
        (ticker, timeframe),
    ).fetchone()
    asof = row[0] if row else None
    if not asof:
        return []
    rows = conn.execute(
        """
        SELECT pattern, confidence, x0_date, x1_date, y0, y1
        FROM yolo_patterns
        WHERE ticker = ? AND timeframe = ? AND as_of_date = ?
        ORDER BY confidence DESC
        LIMIT 4
        """,
        (ticker, timeframe, asof),
    ).fetchall()
    date_set = set(dates)
    out: list[dict[str, Any]] = []
    for r in rows:
        x0 = str(r[2] or "")
        x1 = str(r[3] or "")
        if x0 and x0 not in date_set and x1 and x1 not in date_set:
            continue
        out.append(
            {
                "pattern": str(r[0] or ""),
                "confidence": float(r[1] or 0.0),
                "x0_date": x0,
                "x1_date": x1,
                "y0": float(r[4] or 0.0),
                "y1": float(r[5] or 0.0),
            }
        )
    return out


def _load_fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return (
                ImageFont.truetype(path, size=16),
                ImageFont.truetype(path, size=12),
            )
        except Exception:
            continue
    default = ImageFont.load_default()
    return default, default


def _draw_header(
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    ticker: str,
    timeframe: str,
    last_close: float,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    draw.rounded_rectangle((14, 12, 270, 60), radius=12, fill=(11, 18, 32, 220))
    draw.text((28, 21), f"{ticker} • {timeframe.upper()} view", font=font, fill=(245, 247, 250))
    draw.text((28, 41), f"Close {last_close:.2f}", font=small_font, fill=(182, 194, 210))
    draw.text(
        (image.width - 190, 20),
        "Research only. Not financial advice.",
        font=small_font,
        fill=(214, 220, 229),
    )


def _draw_levels(
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    axes_info: dict[str, Any],
    levels: pd.DataFrame,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    if levels is None or levels.empty:
        return
    right_x0 = int(min(image.width - 170, axes_info["ax_x1"] + 10))
    right_x1 = int(min(image.width - 16, right_x0 + 150))
    axis_left = int(axes_info["ax_x0"])
    axis_right = int(axes_info["ax_x1"])
    shown = levels.sort_values(["type", "dist", "touches"], ascending=[True, True, False]).head(6)
    for _, row in shown.iterrows():
        color = SUPPORT_COLOR if str(row.get("type")) == "support" else RESISTANCE_COLOR
        y = int(_price_to_px_y(float(row.get("level") or 0.0), axes_info))
        fill = (*color, 72 if str(row.get("tier")) == "primary" else 40)
        draw.line((axis_left, y, axis_right, y), fill=(*color, 210), width=2)
        label = f"{str(row.get('type') or '').upper()} {float(row.get('level') or 0.0):.2f}"
        draw.rounded_rectangle((right_x0, y - 14, right_x1, y + 14), radius=8, outline=(*color, 220), fill=fill)
        draw.text((right_x0 + 8, y - 10), label, font=small_font, fill=(245, 247, 250))


def _draw_yolo_boxes(
    draw: ImageDraw.ImageDraw,
    axes_info: dict[str, Any],
    dates: list[str],
    detections: list[dict[str, Any]],
    font: ImageFont.ImageFont,
) -> None:
    if not detections:
        return
    for det in detections[:3]:
        x0 = _date_to_px_x(det.get("x0_date"), dates, axes_info)
        x1 = _date_to_px_x(det.get("x1_date"), dates, axes_info)
        y0 = _price_to_px_y(float(det.get("y0") or 0.0), axes_info)
        y1 = _price_to_px_y(float(det.get("y1") or 0.0), axes_info)
        pattern = str(det.get("pattern") or "")
        pattern_l = pattern.lower()
        if ("bottom" in pattern_l) or ("w_bottom" in pattern_l):
            color = BOX_BULL
        elif ("top" in pattern_l) or ("m_head" in pattern_l):
            color = BOX_BEAR
        else:
            color = BOX_NEUTRAL
        left, right = sorted((int(x0), int(x1)))
        top, bottom = sorted((int(y0), int(y1)))
        draw.rounded_rectangle((left, top, right, bottom), radius=8, outline=(*color, 225), fill=(*color, 38), width=3)
        label = f"{pattern} {float(det.get('confidence') or 0.0) * 100:.0f}%"
        label_w = int(min(260, max(90, len(label) * 8)))
        draw.rounded_rectangle((left + 6, top + 6, left + 6 + label_w, top + 30), radius=8, fill=(15, 23, 42, 210))
        draw.text((left + 12, top + 10), label, font=font, fill=(*color, 255))


def _date_to_px_x(value: Any, dates: list[str], axes_info: dict[str, Any]) -> float:
    target = str(value or "")
    if not dates:
        return float(axes_info["ax_x0"])
    try:
        idx = dates.index(target)
    except ValueError:
        idx = 0
    x_frac = (idx - axes_info["xlim"][0]) / max(axes_info["xlim"][1] - axes_info["xlim"][0], 1)
    return float(axes_info["ax_x0"]) + x_frac * float(axes_info["ax_x1"] - axes_info["ax_x0"])


def _price_to_px_y(price: float, axes_info: dict[str, Any]) -> float:
    frac = (float(price) - axes_info["ylim"][0]) / max(axes_info["ylim"][1] - axes_info["ylim"][0], 1)
    fig_y = axes_info["ax_y0"] + frac * (axes_info["ax_y1"] - axes_info["ax_y0"])
    return float(axes_info["fig_h_px"]) - fig_y
