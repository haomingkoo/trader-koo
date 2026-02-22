#!/usr/bin/env python3
"""Batch YOLOv8 pattern detection — renders each ticker's chart and stores detections in DB.

Initial seed (all tickers, slow ~15-20 min):
    python trader_koo/scripts/run_yolo_patterns.py --db-path /data/trader_koo.db

Daily incremental (only tickers with a new candle, fast ~1-3 min):
    python trader_koo/scripts/run_yolo_patterns.py --db-path /data/trader_koo.db --only-new
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

# Set headless backend BEFORE any matplotlib import
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

LOG = logging.getLogger("yolo_patterns")
if not LOG.handlers:
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

YOLO_MODEL_ID = "foduucom/stockmarket-pattern-detection-yolov8"
YOLO_CLASSES = [
    "Head and shoulders bottom",
    "Head and shoulders top",
    "M_Head",
    "StockLine",
    "Triangle",
    "W_Bottom",
]
DEFAULT_LOOKBACK_DAYS = 180


# ── DB helpers ───────────────────────────────────────────────────────────────

def ensure_yolo_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS yolo_patterns (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker        TEXT    NOT NULL,
            pattern       TEXT    NOT NULL,
            confidence    REAL    NOT NULL,
            x0_date       TEXT    NOT NULL,
            x1_date       TEXT    NOT NULL,
            y0            REAL    NOT NULL,
            y1            REAL    NOT NULL,
            lookback_days INTEGER NOT NULL,
            as_of_date    TEXT,
            detected_ts   TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_yolo_ticker ON yolo_patterns(ticker)")
    conn.commit()
    # Add as_of_date column if upgrading from older schema
    try:
        conn.execute("ALTER TABLE yolo_patterns ADD COLUMN as_of_date TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists


def get_tickers_to_process(
    conn: sqlite3.Connection, only_new: bool
) -> list[tuple[str, str]]:
    """Return list of (ticker, latest_price_date) to process."""
    rows = conn.execute(
        "SELECT ticker, MAX(date) as latest FROM price_daily GROUP BY ticker ORDER BY ticker"
    ).fetchall()
    if not only_new:
        return [(r[0], r[1]) for r in rows]

    # --only-new: skip tickers whose yolo as_of_date matches their latest price date
    existing = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT ticker, MAX(as_of_date) FROM yolo_patterns GROUP BY ticker"
        ).fetchall()
    }
    result = []
    for ticker, latest_date in rows:
        last_detected = existing.get(ticker)
        if last_detected != latest_date:
            result.append((ticker, latest_date))
    return result


def get_price_df(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM price_daily WHERE ticker = ? ORDER BY date",
        conn,
        params=(ticker,),
    )


def save_detections(
    conn: sqlite3.Connection,
    ticker: str,
    detections: list[dict],
    lookback_days: int,
    as_of_date: str,
) -> None:
    conn.execute("DELETE FROM yolo_patterns WHERE ticker = ?", (ticker,))
    for d in detections:
        conn.execute(
            """INSERT INTO yolo_patterns
               (ticker, pattern, confidence, x0_date, x1_date, y0, y1, lookback_days, as_of_date)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, d["pattern"], d["confidence"],
             d["x0_date"], d["x1_date"], d["y0"], d["y1"],
             lookback_days, as_of_date),
        )
    conn.commit()


# ── Chart rendering ──────────────────────────────────────────────────────────

def render_chart(df: pd.DataFrame):
    """Render OHLCV data as a standard candlestick chart (yahoo style, white bg).

    Returns (rgb_ndarray, axes_info_dict). axes_info has coordinate mapping
    needed to convert YOLO pixel bboxes back to date/price.
    """
    import mplfinance as mpf
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    df_plot = df.copy()
    df_plot.index = pd.DatetimeIndex(df_plot["date"])
    df_plot = (
        df_plot[["open", "high", "low", "close", "volume"]]
        .rename(columns={"open": "Open", "high": "High", "low": "Low",
                         "close": "Close", "volume": "Volume"})
        .astype(float)
    )

    style = mpf.make_mpf_style(base_mpf_style="yahoo")
    DPI, FIG_W, FIG_H = 100, 12, 6

    fig, axes = mpf.plot(
        df_plot, type="candle", style=style, volume=True,
        returnfig=True, figsize=(FIG_W, FIG_H),
    )
    ax = axes[0]  # main price axis

    pos = ax.get_position()   # figure-fraction bounds
    xlim = ax.get_xlim()      # bar-index limits
    ylim = ax.get_ylim()      # price limits
    fig_w_px = FIG_W * DPI
    fig_h_px = FIG_H * DPI

    axes_info = {
        "ax_x0": pos.x0 * fig_w_px,
        "ax_x1": pos.x1 * fig_w_px,
        "ax_y0": pos.y0 * fig_h_px,
        "ax_y1": pos.y1 * fig_h_px,
        "xlim": xlim,
        "ylim": ylim,
        "fig_h_px": fig_h_px,
    }

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return np.array(img), axes_info


def px_to_bar_price(x_px: float, y_px: float, ai: dict) -> tuple[float, float]:
    """Convert image pixel (x, y) → (bar_index, price)."""
    x_bar = (
        ai["xlim"][0]
        + (x_px - ai["ax_x0"]) / (ai["ax_x1"] - ai["ax_x0"])
        * (ai["xlim"][1] - ai["xlim"][0])
    )
    fig_y = ai["fig_h_px"] - y_px   # image y=0 is top; figure y=0 is bottom
    frac = (fig_y - ai["ax_y0"]) / (ai["ax_y1"] - ai["ax_y0"])
    price = ai["ylim"][0] + frac * (ai["ylim"][1] - ai["ylim"][0])
    return x_bar, price


def run_inference(model, img_arr, ai: dict, dates: list) -> list[dict]:
    """Run YOLO and return detections in date/price space."""
    n = len(dates)
    price_lo, price_hi = ai["ylim"]

    try:
        results = model(img_arr, verbose=False)
    except Exception as e:
        LOG.warning("YOLO inference error: %s", e)
        return []

    def bar_to_date(idx: float) -> str:
        i = int(round(max(0.0, min(float(n - 1), idx))))
        return str(dates[i])

    detections = []
    if not results or results[0].boxes is None:
        return detections

    for box in results[0].boxes:
        x0, y0, x1, y1 = (float(v) for v in box.xyxy[0].tolist())
        cls = int(box.cls[0])
        conf = round(float(box.conf[0]), 3)

        bar0, pr0 = px_to_bar_price(x0, y0, ai)
        bar1, pr1 = px_to_bar_price(x1, y1, ai)
        p_lo = round(min(pr0, pr1), 2)
        p_hi = round(max(pr0, pr1), 2)

        # Skip boxes that fall in the volume panel (price out of price-axis range)
        if p_hi < price_lo * 0.3 or p_lo > price_hi * 1.5:
            continue

        detections.append({
            "pattern": YOLO_CLASSES[cls] if 0 <= cls < len(YOLO_CLASSES) else f"class_{cls}",
            "confidence": conf,
            "x0_date": bar_to_date(bar0),
            "x1_date": bar_to_date(bar1),
            "y0": p_lo,
            "y1": p_hi,
        })

    return detections


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch YOLO pattern detection")
    parser.add_argument("--db-path", default=os.getenv("TRADER_KOO_DB_PATH", "/data/trader_koo.db"))
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument(
        "--only-new", action="store_true",
        help="Skip tickers whose YOLO patterns are already up-to-date with the latest price candle",
    )
    parser.add_argument("--sleep", type=float, default=0.05,
                        help="Seconds to sleep between tickers (reduces CPU spikes)")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        LOG.error("DB not found at %s", db_path)
        sys.exit(1)

    # Cache YOLO model in persistent volume if available
    if Path("/data").exists():
        os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", "/data/.ultralytics")

    LOG.info("Loading YOLO model: %s", YOLO_MODEL_ID)
    try:
        # PyTorch ≥2.6 defaults weights_only=True, which rejects legacy YOLO weights.
        # Patch torch.load to keep the old behaviour before importing ultralyticsplus.
        import torch as _torch
        _orig_load = _torch.load
        _torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": kw.get("weights_only", False)})

        from ultralyticsplus import YOLO
        model = YOLO(YOLO_MODEL_ID)
        model.overrides.update({
            "conf": 0.25, "iou": 0.45,
            "agnostic_nms": False, "max_det": 1000,
        })
    except Exception as e:
        LOG.error("Failed to load YOLO model: %s", e)
        sys.exit(1)
    LOG.info("YOLO model ready")

    conn = sqlite3.connect(str(db_path))
    try:
        ensure_yolo_table(conn)
        ticker_dates = get_tickers_to_process(conn, only_new=args.only_new)
        LOG.info(
            "Processing %d ticker(s) (lookback=%d days, only_new=%s)",
            len(ticker_dates), args.lookback_days, args.only_new,
        )

        ok = failed = skipped = 0
        for i, (ticker, latest_date) in enumerate(ticker_dates, 1):
            try:
                df = get_price_df(conn, ticker)
                if df.empty or len(df) < 20:
                    skipped += 1
                    continue

                cutoff = pd.Timestamp(df["date"].max()) - pd.DateOffset(days=args.lookback_days)
                df = df[pd.to_datetime(df["date"]) >= cutoff].reset_index(drop=True)
                if len(df) < 20:
                    skipped += 1
                    continue

                img_arr, ai = render_chart(df)
                dates = df["date"].tolist()
                detections = run_inference(model, img_arr, ai, dates)
                save_detections(conn, ticker, detections, args.lookback_days, latest_date)

                LOG.info("[%d/%d] %s (as_of=%s) → %d pattern(s): %s",
                         i, len(ticker_dates), ticker, latest_date,
                         len(detections),
                         ", ".join(d["pattern"] for d in detections) if detections else "none")
                ok += 1
            except Exception:
                LOG.exception("[%d/%d] FAILED on %s", i, len(ticker_dates), ticker)
                failed += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

    finally:
        conn.close()

    LOG.info("Done — ok=%d failed=%d skipped=%d", ok, failed, skipped)


if __name__ == "__main__":
    main()
