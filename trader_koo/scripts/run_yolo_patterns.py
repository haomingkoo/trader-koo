#!/usr/bin/env python3
"""Batch YOLOv8 pattern detection — renders each ticker's chart and stores detections in DB.

Runs two passes per ticker:
  1. daily  — 180 days of daily candles  (~124 bars) — short-term patterns
  2. weekly — 730 days resampled to weekly (~104 bars) — long-term patterns

Initial seed (all tickers, ~20-30 min for both passes):
    python trader_koo/scripts/run_yolo_patterns.py --db-path /data/trader_koo.db

Daily incremental (only tickers with a new candle, fast ~2-5 min):
    python trader_koo/scripts/run_yolo_patterns.py --db-path /data/trader_koo.db --only-new

Single timeframe (e.g. daily only):
    python trader_koo/scripts/run_yolo_patterns.py --db-path /data/trader_koo.db --timeframe daily
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import logging
import os
import signal
import sqlite3
import sys
import time
from contextlib import contextmanager
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
DEFAULT_LOOKBACK_DAYS = 180        # daily pass — ~124 trading bars
DEFAULT_WEEKLY_LOOKBACK_DAYS = 730 # weekly pass — ~104 weekly bars
DEFAULT_DPI = int(os.getenv("TRADER_KOO_YOLO_DPI", "80"))
DEFAULT_FIG_W = float(os.getenv("TRADER_KOO_YOLO_FIG_W", "10"))
DEFAULT_FIG_H = float(os.getenv("TRADER_KOO_YOLO_FIG_H", "5"))
DEFAULT_CONF = float(os.getenv("TRADER_KOO_YOLO_CONF", "0.25"))
DEFAULT_IOU = float(os.getenv("TRADER_KOO_YOLO_IOU", "0.45"))
DEFAULT_IMGSZ = int(os.getenv("TRADER_KOO_YOLO_IMGSZ", "640"))
DEFAULT_MAX_SECS_PER_TICKER = float(os.getenv("TRADER_KOO_YOLO_MAX_SECS_PER_TICKER", "180"))
DEFAULT_MODEL_INIT_TIMEOUT_SEC = float(os.getenv("TRADER_KOO_YOLO_MODEL_INIT_TIMEOUT_SEC", "600"))


# ── DB helpers ───────────────────────────────────────────────────────────────

def ensure_yolo_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS yolo_patterns (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker        TEXT    NOT NULL,
            timeframe     TEXT    NOT NULL DEFAULT 'daily',
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_yolo_timeframe_asof ON yolo_patterns(timeframe, as_of_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_yolo_ticker_timeframe_asof ON yolo_patterns(ticker, timeframe, as_of_date)")
    conn.commit()
    # Migrations for older schema versions
    for col, defn in [
        ("as_of_date", "TEXT"),
        ("timeframe",  "TEXT NOT NULL DEFAULT 'daily'"),
    ]:
        try:
            conn.execute(f"ALTER TABLE yolo_patterns ADD COLUMN {col} {defn}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists


def ensure_yolo_run_events_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS yolo_run_events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id        TEXT    NOT NULL,
            timeframe     TEXT    NOT NULL,
            ticker        TEXT    NOT NULL,
            status        TEXT    NOT NULL,
            reason        TEXT,
            elapsed_sec   REAL,
            bars          INTEGER,
            detections    INTEGER,
            as_of_date    TEXT,
            created_ts    TEXT    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_yolo_events_run_id ON yolo_run_events(run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_yolo_events_status ON yolo_run_events(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_yolo_events_ticker ON yolo_run_events(ticker)")
    conn.commit()


def record_yolo_run_event(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    timeframe: str,
    ticker: str,
    status: str,
    reason: str | None,
    elapsed_sec: float | None,
    bars: int | None,
    detections: int | None,
    as_of_date: str | None,
) -> None:
    try:
        conn.execute(
            """
            INSERT INTO yolo_run_events
            (run_id, timeframe, ticker, status, reason, elapsed_sec, bars, detections, as_of_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                timeframe,
                ticker,
                status,
                reason,
                None if elapsed_sec is None else float(elapsed_sec),
                None if bars is None else int(bars),
                None if detections is None else int(detections),
                as_of_date,
            ),
        )
        conn.commit()
    except Exception as exc:
        LOG.warning("Failed to persist yolo_run_event run_id=%s ticker=%s status=%s: %s", run_id, ticker, status, exc)


def get_tickers_to_process(
    conn: sqlite3.Connection, only_new: bool, timeframe: str
) -> list[tuple[str, str]]:
    """Return list of (ticker, latest_price_date) to process for the given timeframe."""
    rows = conn.execute(
        "SELECT ticker, MAX(date) as latest FROM price_daily GROUP BY ticker ORDER BY ticker"
    ).fetchall()
    if not only_new:
        return [(r[0], r[1]) for r in rows]

    # --only-new: skip tickers whose yolo as_of_date (for this timeframe) is up-to-date
    existing = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT ticker, MAX(as_of_date) FROM yolo_patterns WHERE timeframe = ? GROUP BY ticker",
            (timeframe,),
        ).fetchall()
    }
    result = []
    for ticker, latest_date in rows:
        if existing.get(ticker) != latest_date:
            result.append((ticker, latest_date))
    return result


def get_price_df(conn: sqlite3.Connection, ticker: str, start_date: str | None = None) -> pd.DataFrame:
    if start_date:
        sql = """
            SELECT date, open, high, low, close, volume
            FROM price_daily
            WHERE ticker = ? AND date >= ?
            ORDER BY date
        """
        params = (ticker, start_date)
    else:
        sql = """
            SELECT date, open, high, low, close, volume
            FROM price_daily
            WHERE ticker = ?
            ORDER BY date
        """
        params = (ticker,)
    return pd.read_sql_query(sql, conn, params=params)


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly (week-end anchor) for longer-term detection."""
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"])
    df2 = df2.set_index("date")
    weekly = df2.resample("W").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open", "close"])
    weekly = weekly.reset_index()
    weekly["date"] = weekly["date"].dt.strftime("%Y-%m-%d")
    return weekly


def save_detections(
    conn: sqlite3.Connection,
    ticker: str,
    timeframe: str,
    detections: list[dict],
    lookback_days: int,
    as_of_date: str,
) -> None:
    # Preserve historical snapshots. Only replace rows for the same ticker/timeframe/as_of_date.
    conn.execute(
        "DELETE FROM yolo_patterns WHERE ticker = ? AND timeframe = ? AND as_of_date = ?",
        (ticker, timeframe, as_of_date),
    )
    for d in detections:
        conn.execute(
            """INSERT INTO yolo_patterns
               (ticker, timeframe, pattern, confidence, x0_date, x1_date, y0, y1, lookback_days, as_of_date)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, timeframe, d["pattern"], d["confidence"],
             d["x0_date"], d["x1_date"], d["y0"], d["y1"],
             lookback_days, as_of_date),
        )
    conn.commit()


# ── Chart rendering ──────────────────────────────────────────────────────────

def render_chart(
    df: pd.DataFrame,
    *,
    dpi: int = DEFAULT_DPI,
    fig_w: float = DEFAULT_FIG_W,
    fig_h: float = DEFAULT_FIG_H,
):
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
    fig, axes = mpf.plot(
        df_plot, type="candle", style=style, volume=True,
        returnfig=True, figsize=(fig_w, fig_h),
    )
    ax = axes[0]  # main price axis

    pos = ax.get_position()   # figure-fraction bounds
    xlim = ax.get_xlim()      # bar-index limits
    ylim = ax.get_ylim()      # price limits
    fig_w_px = fig_w * dpi
    fig_h_px = fig_h * dpi

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
    fig.savefig(buf, format="png", dpi=dpi)
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


@contextmanager
def _ticker_timeout(seconds: float):
    """Raise TimeoutError if a ticker takes too long on Unix platforms."""
    if seconds <= 0 or not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise TimeoutError(f"ticker exceeded timeout ({seconds:.1f}s)")

    prev_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev_handler)


# ── Per-pass processing ───────────────────────────────────────────────────────

def _run_pass(
    model,
    conn: sqlite3.Connection,
    run_id: str,
    timeframe: str,
    lookback_days: int,
    only_new: bool,
    sleep_s: float,
    dpi: int,
    fig_w: float,
    fig_h: float,
    max_secs_per_ticker: float,
) -> tuple[int, int, int]:
    """Run one full detection pass (daily or weekly). Returns (ok, failed, skipped)."""
    ticker_dates = get_tickers_to_process(conn, only_new=only_new, timeframe=timeframe)
    LOG.info(
        "[%s] run_id=%s Processing %d ticker(s) (lookback=%d days, only_new=%s, dpi=%s, fig=%.1fx%.1f, max_sec_per_ticker=%.1f)",
        timeframe, run_id, len(ticker_dates), lookback_days, only_new, dpi, fig_w, fig_h, max_secs_per_ticker,
    )

    ok = failed = skipped = 0
    total = len(ticker_dates)
    total_elapsed_s = 0.0
    total_render_s = 0.0
    total_infer_s = 0.0
    total_bars = 0
    for i, (ticker, latest_date) in enumerate(ticker_dates, 1):
        t_start = time.perf_counter()
        render_s = 0.0
        infer_s = 0.0
        try:
            with _ticker_timeout(max_secs_per_ticker):
                query_start_date = None
                try:
                    latest_dt = dt.date.fromisoformat(str(latest_date))
                    query_pad = 35 if timeframe == "weekly" else 7
                    query_start_date = (latest_dt - dt.timedelta(days=lookback_days + query_pad)).isoformat()
                except Exception:
                    query_start_date = None

                df = get_price_df(conn, ticker, start_date=query_start_date)
                if df.empty or len(df) < 20:
                    skipped += 1
                    elapsed_s = time.perf_counter() - t_start
                    reason = f"insufficient_rows_before_filter={len(df)}"
                    LOG.info("[%s %d/%d] SKIP %s — %s", timeframe, i, total, ticker, reason)
                    record_yolo_run_event(
                        conn,
                        run_id=run_id,
                        timeframe=timeframe,
                        ticker=ticker,
                        status="skipped",
                        reason=reason,
                        elapsed_sec=elapsed_s,
                        bars=len(df),
                        detections=0,
                        as_of_date=str(latest_date),
                    )
                    continue

                cutoff = pd.Timestamp(df["date"].max()) - pd.Timedelta(days=lookback_days)
                df = df[pd.to_datetime(df["date"]) >= cutoff].reset_index(drop=True)

                if timeframe == "weekly":
                    df = resample_to_weekly(df)

                if len(df) < 10:
                    skipped += 1
                    elapsed_s = time.perf_counter() - t_start
                    reason = f"insufficient_rows_after_filter={len(df)}"
                    LOG.info("[%s %d/%d] SKIP %s — %s", timeframe, i, total, ticker, reason)
                    record_yolo_run_event(
                        conn,
                        run_id=run_id,
                        timeframe=timeframe,
                        ticker=ticker,
                        status="skipped",
                        reason=reason,
                        elapsed_sec=elapsed_s,
                        bars=len(df),
                        detections=0,
                        as_of_date=str(latest_date),
                    )
                    continue

                t_render0 = time.perf_counter()
                img_arr, ai = render_chart(df, dpi=dpi, fig_w=fig_w, fig_h=fig_h)
                render_s = time.perf_counter() - t_render0
                dates = df["date"].tolist()
                t_infer0 = time.perf_counter()
                detections = run_inference(model, img_arr, ai, dates)
                infer_s = time.perf_counter() - t_infer0
                save_detections(conn, ticker, timeframe, detections, lookback_days, latest_date)

                elapsed_s = time.perf_counter() - t_start
                total_elapsed_s += elapsed_s
                total_render_s += render_s
                total_infer_s += infer_s
                total_bars += len(df)

                LOG.info("[%s %d/%d] %s → %d pattern(s): %s",
                         timeframe, i, total, ticker,
                         len(detections),
                         ", ".join(d["pattern"] for d in detections) if detections else "none")
                LOG.info(
                    "[%s %d/%d] %s timings sec=%.2f render=%.2f infer=%.2f bars=%d",
                    timeframe, i, total, ticker, elapsed_s, render_s, infer_s, len(df),
                )
                record_yolo_run_event(
                    conn,
                    run_id=run_id,
                    timeframe=timeframe,
                    ticker=ticker,
                    status="ok",
                    reason="ok",
                    elapsed_sec=elapsed_s,
                    bars=len(df),
                    detections=len(detections),
                    as_of_date=str(latest_date),
                )
                ok += 1
        except TimeoutError:
            failed += 1
            elapsed_s = time.perf_counter() - t_start
            reason = f"timeout>{max_secs_per_ticker:.1f}s"
            LOG.error(
                "[%s %d/%d] TIMEOUT on %s after %.2fs (limit=%.2fs) — skipping ticker",
                timeframe, i, total, ticker, elapsed_s, max_secs_per_ticker,
            )
            record_yolo_run_event(
                conn,
                run_id=run_id,
                timeframe=timeframe,
                ticker=ticker,
                status="timeout",
                reason=reason,
                elapsed_sec=elapsed_s,
                bars=None,
                detections=None,
                as_of_date=str(latest_date),
            )
        except Exception:
            elapsed_s = time.perf_counter() - t_start
            reason = str(sys.exc_info()[1])[:500]
            LOG.exception("[%s %d/%d] FAILED on %s", timeframe, i, total, ticker)
            record_yolo_run_event(
                conn,
                run_id=run_id,
                timeframe=timeframe,
                ticker=ticker,
                status="failed",
                reason=reason,
                elapsed_sec=elapsed_s,
                bars=None,
                detections=None,
                as_of_date=str(latest_date),
            )
            failed += 1

        if sleep_s > 0:
            time.sleep(sleep_s)

    processed = max(1, ok)
    LOG.info(
        "[%s] run_id=%s Done ok=%d failed=%d skipped=%d avg_sec=%.2f avg_render=%.2f avg_infer=%.2f avg_bars=%.1f",
        timeframe,
        run_id,
        ok,
        failed,
        skipped,
        total_elapsed_s / processed,
        total_render_s / processed,
        total_infer_s / processed,
        total_bars / processed,
    )

    return ok, failed, skipped


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch YOLO pattern detection")
    parser.add_argument("--db-path", default=os.getenv("TRADER_KOO_DB_PATH", "/data/trader_koo.db"))
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help="Daily pass: days of daily candles (default 180 ≈ 124 bars)")
    parser.add_argument("--weekly-lookback-days", type=int, default=DEFAULT_WEEKLY_LOOKBACK_DAYS,
                        help="Weekly pass: days to fetch before resampling (default 730 ≈ 104 weekly bars)")
    parser.add_argument(
        "--only-new", action="store_true",
        help="Skip tickers already up-to-date with latest price candle",
    )
    parser.add_argument(
        "--timeframe", choices=["daily", "weekly", "both"], default="both",
        help="Which timeframe pass(es) to run (default: both)",
    )
    parser.add_argument("--sleep", type=float, default=0.05,
                        help="Seconds to sleep between tickers (reduces CPU spikes)")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Rendered chart DPI")
    parser.add_argument("--fig-w", type=float, default=DEFAULT_FIG_W, help="Rendered chart width (inches)")
    parser.add_argument("--fig-h", type=float, default=DEFAULT_FIG_H, help="Rendered chart height (inches)")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="YOLO IoU threshold")
    parser.add_argument(
        "--max-seconds-per-ticker",
        type=float,
        default=DEFAULT_MAX_SECS_PER_TICKER,
        help="Fail-safe timeout per ticker; timeout skips ticker and continues (0 disables)",
    )
    parser.add_argument(
        "--model-init-timeout-sec",
        type=float,
        default=DEFAULT_MODEL_INIT_TIMEOUT_SEC,
        help="Fail-safe timeout for YOLO model init/download (0 disables)",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run id for tracing and event logging (auto-generated if empty)",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        LOG.error("DB not found at %s", db_path)
        sys.exit(1)

    # Cache YOLO model in persistent volume if available
    if Path("/data").exists():
        os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", "/data/.ultralytics")

    if args.model_init_timeout_sec > 0:
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(int(args.model_init_timeout_sec)))
    LOG.info("Loading YOLO model: %s (init_timeout_sec=%.1f)", YOLO_MODEL_ID, args.model_init_timeout_sec)
    try:
        with _ticker_timeout(args.model_init_timeout_sec):
            # Validate OpenCV import early so failures are explicit in logs.
            LOG.info("YOLO init step: importing OpenCV")
            try:
                import cv2 as _cv2
                LOG.info("OpenCV backend: cv2=%s from %s", getattr(_cv2, "__version__", "?"), getattr(_cv2, "__file__", "?"))
            except Exception as cv_exc:
                LOG.error("OpenCV import failed before YOLO init: %s", cv_exc)
                raise

            # PyTorch >= 2.6 defaults weights_only=True, which rejects legacy YOLO weights.
            # Patch torch.load to keep the old behaviour before importing ultralyticsplus.
            LOG.info("YOLO init step: importing torch")
            import torch as _torch
            _orig_load = _torch.load
            _torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": kw.get("weights_only", False)})

            LOG.info("YOLO init step: importing ultralyticsplus.YOLO")
            from ultralyticsplus import YOLO
            LOG.info("YOLO init step: constructing model")
            model = YOLO(YOLO_MODEL_ID)
            model.overrides.update({
                "imgsz": args.imgsz,
                "conf": args.conf,
                "iou": args.iou,
                "agnostic_nms": False, "max_det": 1000,
            })
    except TimeoutError:
        LOG.error(
            "Failed to load YOLO model within %.1fs; init/download timed out",
            args.model_init_timeout_sec,
        )
        sys.exit(1)
    except Exception as e:
        LOG.error("Failed to load YOLO model: %s", e)
        sys.exit(1)
    LOG.info("YOLO model ready")

    conn = sqlite3.connect(str(db_path))
    try:
        run_id = (args.run_id or "").strip() or (
            dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + f"-yolo-{os.getpid()}"
        )
        LOG.info("Starting YOLO run_id=%s", run_id)
        ensure_yolo_table(conn)
        ensure_yolo_run_events_table(conn)

        passes = []
        if args.timeframe in ("daily", "both"):
            passes.append(("daily", args.lookback_days))
        if args.timeframe in ("weekly", "both"):
            passes.append(("weekly", args.weekly_lookback_days))

        total_ok = total_failed = total_skipped = 0
        for timeframe, lookback in passes:
            ok, failed, skipped = _run_pass(
                model, conn,
                run_id=run_id,
                timeframe=timeframe,
                lookback_days=lookback,
                only_new=args.only_new,
                sleep_s=args.sleep,
                dpi=args.dpi,
                fig_w=args.fig_w,
                fig_h=args.fig_h,
                max_secs_per_ticker=args.max_seconds_per_ticker,
            )
            total_ok += ok
            total_failed += failed
            total_skipped += skipped

    finally:
        conn.close()

    LOG.info("Done run_id=%s — ok=%d failed=%d skipped=%d", run_id, total_ok, total_failed, total_skipped)


if __name__ == "__main__":
    main()
