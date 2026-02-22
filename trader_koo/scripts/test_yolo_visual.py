#!/usr/bin/env python3
"""Visual test for YOLO pattern detection.

Renders a white-bg candlestick chart, runs the YOLO model, draws detected
bounding boxes directly on the image, then saves annotated PNGs so you can
inspect what the model sees and how well coordinates map back to date/price.

Usage:
    python trader_koo/scripts/test_yolo_visual.py \
        --tickers SPY AAPL NVDA \
        --db-path /data/trader_koo.db \
        --out-dir /tmp/yolo_test \
        --lookback-days 180 \
        --conf 0.25
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

# Allow running from the repo root
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont

from trader_koo.scripts.run_yolo_patterns import (
    DEFAULT_LOOKBACK_DAYS,
    YOLO_MODEL_ID,
    get_price_df,
    render_chart,
    run_inference,
)


# ── colours per class (PIL uses int tuples) ──────────────────────────────────
_STROKE = {
    "Head and shoulders bottom": (56, 211, 159),
    "Head and shoulders top":    (255, 107, 107),
    "M_Head":                    (255, 107, 107),
    "W_Bottom":                  (56, 211, 159),
    "Triangle":                  (248, 194, 78),
    "StockLine":                 (106, 169, 255),
}
_DEFAULT_STROKE = (200, 200, 200)


def stroke_for(pattern: str) -> tuple[int, int, int]:
    return _STROKE.get(pattern, _DEFAULT_STROKE)


def annotate_image(img_arr, detections: list[dict], axes_info: dict,
                   dates: list) -> Image.Image:
    """Draw YOLO bounding boxes + labels onto the raw chart image."""
    img = Image.fromarray(img_arr).copy()
    draw = ImageDraw.Draw(img, "RGBA")

    # Try to get a small font; fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=13)
    except Exception:
        font = ImageFont.load_default()

    ai = axes_info

    for d in detections:
        color = stroke_for(d["pattern"])
        fill = color + (28,)  # semi-transparent fill

        # Map date back to pixel x for annotation position
        def date_to_px_x(date_str: str) -> float:
            try:
                idx = dates.index(date_str)
            except ValueError:
                # approximate: find closest
                idx = min(range(len(dates)), key=lambda i: abs(
                    pd.Timestamp(dates[i]) - pd.Timestamp(date_str)
                ))
            bar_idx = idx
            x_frac = (bar_idx - ai["xlim"][0]) / max(ai["xlim"][1] - ai["xlim"][0], 1)
            return ai["ax_x0"] + x_frac * (ai["ax_x1"] - ai["ax_x0"])

        def price_to_px_y(price: float) -> float:
            frac = (price - ai["ylim"][0]) / max(ai["ylim"][1] - ai["ylim"][0], 1)
            fig_y = ai["ax_y0"] + frac * (ai["ax_y1"] - ai["ax_y0"])
            return ai["fig_h_px"] - fig_y

        x0_px = date_to_px_x(d["x0_date"])
        x1_px = date_to_px_x(d["x1_date"])
        y0_px = price_to_px_y(d["y0"])
        y1_px = price_to_px_y(d["y1"])

        # Draw filled rect + border
        draw.rectangle(
            [min(x0_px, x1_px), min(y0_px, y1_px),
             max(x0_px, x1_px), max(y0_px, y1_px)],
            fill=fill,
            outline=color,
            width=2,
        )

        # Label: pattern name + confidence
        label = f"{d['pattern']}  {d['confidence']*100:.0f}%"
        lx = min(x0_px, x1_px) + 4
        ly = min(y0_px, y1_px) + 4
        # Background box for text readability
        try:
            bbox = font.getbbox(label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = len(label) * 7, 14
        draw.rectangle([lx - 2, ly - 2, lx + tw + 4, ly + th + 4],
                       fill=(20, 20, 20, 200))
        draw.text((lx, ly), label, fill=color, font=font)

    # Also print a summary below the chart
    summary_lines = [f"  {d['pattern']}  {d['confidence']*100:.0f}%  "
                     f"[{d['x0_date']} → {d['x1_date']}]  "
                     f"y0={d['y0']:.2f}  y1={d['y1']:.2f}"
                     for d in detections]
    if not summary_lines:
        summary_lines = ["  (no detections above confidence threshold)"]

    # Expand canvas to fit summary text
    line_h = 18
    total_extra = line_h * (len(summary_lines) + 2)
    new_img = Image.new("RGB", (img.width, img.height + total_extra), (240, 240, 240))
    new_img.paste(img, (0, 0))
    draw2 = ImageDraw.Draw(new_img)
    draw2.text((8, img.height + 6), "Detections:", fill=(50, 50, 50), font=font)
    for i, line in enumerate(summary_lines):
        draw2.text((8, img.height + 6 + (i + 1) * line_h), line, fill=(30, 30, 30), font=font)

    return new_img


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual YOLO pattern test")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "AAPL", "NVDA"],
                        help="Tickers to test (default: SPY AAPL NVDA)")
    parser.add_argument("--db-path",
                        default=os.getenv("TRADER_KOO_DB_PATH", "/data/trader_koo.db"))
    parser.add_argument("--out-dir", default="/tmp/yolo_test",
                        help="Directory to save annotated PNG images")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}")
        sys.exit(1)

    # Cache model in persistent location if available
    if Path("/data").exists():
        os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", "/data/.ultralytics")

    print(f"Loading YOLO model: {YOLO_MODEL_ID}")
    try:
        # PyTorch ≥2.6 defaults weights_only=True, which rejects legacy YOLO weights.
        import torch as _torch
        _orig_load = _torch.load
        _torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": kw.get("weights_only", False)})

        from ultralyticsplus import YOLO
        model = YOLO(YOLO_MODEL_ID)
        model.overrides.update({
            "conf": args.conf,
            "iou": args.iou,
            "agnostic_nms": False,
            "max_det": 1000,
        })
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model: {e}")
        sys.exit(1)
    print("Model ready.\n")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    for ticker in args.tickers:
        ticker = ticker.upper().strip()
        print(f"── {ticker} ──────────────────────────────────────────")
        try:
            df = get_price_df(conn, ticker)
            if df.empty or len(df) < 20:
                print(f"  SKIP: not enough data ({len(df)} rows)")
                continue

            cutoff = pd.Timestamp(df["date"].max()) - pd.DateOffset(days=args.lookback_days)
            df = df[pd.to_datetime(df["date"]) >= cutoff].reset_index(drop=True)
            if len(df) < 20:
                print(f"  SKIP: not enough data after lookback ({len(df)} rows)")
                continue

            print(f"  Rendering chart ({len(df)} bars, {df['date'].min()} → {df['date'].max()})")
            img_arr, ai = render_chart(df)
            dates = df["date"].tolist()

            print(f"  Running YOLO inference...")
            detections = run_inference(model, img_arr, ai, dates)

            if detections:
                print(f"  Detected {len(detections)} pattern(s):")
                for d in detections:
                    print(f"    {d['pattern']:40s}  conf={d['confidence']:.3f}  "
                          f"{d['x0_date']} → {d['x1_date']}  "
                          f"y0={d['y0']:.2f}  y1={d['y1']:.2f}")
            else:
                print("  No patterns detected above threshold.")

            # Save raw chart
            raw_path = out_dir / f"{ticker}_raw.png"
            Image.fromarray(img_arr).save(str(raw_path))
            print(f"  Saved raw chart: {raw_path}")

            # Save annotated chart (boxes drawn in price space)
            annotated = annotate_image(img_arr, detections, ai, dates)
            ann_path = out_dir / f"{ticker}_annotated.png"
            annotated.save(str(ann_path))
            print(f"  Saved annotated: {ann_path}")

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
        print()

    conn.close()
    print(f"\nDone. Images in: {out_dir}")
    print("Open with:  open " + str(out_dir))


if __name__ == "__main__":
    main()
