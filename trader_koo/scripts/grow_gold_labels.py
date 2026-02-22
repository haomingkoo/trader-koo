from __future__ import annotations

import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

"""
Incremental gold-label growing loop.

Two modes:

  detect  — Run pattern detection on a batch of tickers, skip anything already
             confirmed in gold_labels.csv, and write a review template CSV for
             the new samples only.

  merge   — Read the filled-in review template, apply decisions, and append
             confirmed labels to gold_labels.csv.

Typical workflow
----------------
# Round 1 — small seed batch
python scripts/grow_gold_labels.py detect \
    --tickers "AAPL,SPY,NVDA" \
    --render-images

# Open data/cv/pending_review_template.csv in Excel / Google Sheets.
# Fill in the "approved" column: TRUE / FALSE
# (optionally fill label_override if you want to rename a pattern)
# Save the file.

python scripts/grow_gold_labels.py merge

# Round 2 — add more tickers (already-approved samples are skipped)
python scripts/grow_gold_labels.py detect \
    --tickers "MSFT,TSLA,QQQ" \
    --render-images

python scripts/grow_gold_labels.py merge

# Scale out to everything
python scripts/grow_gold_labels.py detect --use-all-tickers --render-images
python scripts/grow_gold_labels.py merge
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from trader_koo.cv.proxy_patterns import CVProxyConfig, detect_cv_proxy_patterns
from trader_koo.data.schema import ensure_ohlcv_schema
from trader_koo.features.candle_patterns import CandlePatternConfig, detect_candlestick_patterns
from trader_koo.features.technical import FeatureConfig, add_basic_features, compute_pivots
from trader_koo.structure.hybrid_patterns import HybridPatternConfig, score_hybrid_patterns
from trader_koo.structure.patterns import PatternConfig, detect_patterns

# reuse helpers from build_cv_weak_labels
from trader_koo.scripts.build_cv_weak_labels import (  # type: ignore[import]
    merge_pattern_sources,
    pick_tickers,
)

GOLD_COLS = [
    "sample_id", "ticker", "pattern", "status",
    "consensus_conf", "rule_conf", "hybrid_conf", "cv_conf",
    "decision", "agreement", "label_source", "review_state",
    "review_notes", "reviewer",
    "yolo_x_center", "yolo_y_center", "yolo_w", "yolo_h",
]

REVIEW_TEMPLATE_COLS = [
    "sample_id", "pattern", "approved", "label_override",
    "status_override", "review_notes", "reviewer",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _load_gold(gold_path: Path) -> pd.DataFrame:
    if gold_path.exists():
        df = pd.read_csv(gold_path)
        df["sample_id"] = df["sample_id"].astype(str)
        df["pattern"] = df["pattern"].astype(str)
        return df
    cols = GOLD_COLS
    return pd.DataFrame(columns=cols)


def _load_price(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM price_daily WHERE ticker = ? ORDER BY date",
        conn, params=(ticker,),
    )
    return ensure_ohlcv_schema(df)


def _parse_bool(x: object) -> bool:
    return str(x or "").strip().lower() in {"1", "true", "t", "yes", "y"}


# Pattern colour buckets (for line drawing)
_BULL = {"bull_flag", "ascending_triangle", "inv_head_and_shoulders", "double_bottom", "cup_and_handle"}
_BEAR = {"bear_flag", "descending_triangle", "head_and_shoulders", "double_top"}

def _pattern_color(name: str) -> str:
    n = str(name).lower()
    if n in _BULL:
        return "#38d39f"
    if n in _BEAR:
        return "#ff6b6b"
    return "#f8c24e"   # neutral / wedge / symmetrical


def _render_annotated_png(
    window: pd.DataFrame,
    all_patterns: list[pd.DataFrame],
    out_path: Path,
    width: int = 1280,
    height: int = 720,
    title: str = "",
) -> bool:
    """Render candlestick chart with detected pattern lines overlaid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle as MRect
    except ImportError:
        return False

    n = len(window)
    if n < 2:
        return False

    # date → bar-index map
    date_to_idx = {
        pd.Timestamp(d).strftime("%Y-%m-%d"): i
        for i, d in enumerate(pd.to_datetime(window["date"], errors="coerce"))
    }

    fig_w = max(width / 100.0, 6.0)
    fig_h = max(height / 100.0, 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.set_facecolor("#111722")
    fig.patch.set_facecolor("#111722")

    # ── candlesticks ──────────────────────────────────────────────────────────
    for i, r in enumerate(window.itertuples(index=False)):
        o, h, l, c = float(r.open), float(r.high), float(r.low), float(r.close)
        up = c >= o
        color = "#38d39f" if up else "#ff6b6b"
        ax.vlines(i, l, h, color=color, linewidth=1.0, alpha=0.9)
        body_lo = min(o, c)
        body_h = max(abs(c - o), 1e-6)
        ax.add_patch(MRect((i - 0.3, body_lo), 0.6, body_h, facecolor=color, edgecolor=color, lw=0.7))

    # ── pattern overlays ──────────────────────────────────────────────────────
    drawn: dict[str, int] = {}   # pattern_name → count (for legend dedup)
    for df in all_patterns:
        if df is None or df.empty:
            continue
        for r in df.itertuples(index=False):
            pname = str(getattr(r, "pattern", "") or "")
            if not pname:
                continue

            x0_d = str(getattr(r, "x0_date", "") or getattr(r, "start_date", "") or "")
            x1_d = str(getattr(r, "x1_date", "") or getattr(r, "end_date", "") or "")
            i0 = date_to_idx.get(x0_d, 0)
            i1 = date_to_idx.get(x1_d, n - 1)
            if i1 < i0:
                i0, i1 = i1, i0

            y0  = _safe_float(getattr(r, "y0", None))
            y1  = _safe_float(getattr(r, "y1", None))
            y0b = _safe_float(getattr(r, "y0b", None))
            y1b = _safe_float(getattr(r, "y1b", None))

            color = _pattern_color(pname)
            lbl = pname if pname not in drawn else None
            drawn[pname] = drawn.get(pname, 0) + 1

            if y0 is not None and y1 is not None:
                ax.plot([i0, i1], [y0, y1], color=color, lw=1.6, alpha=0.9,
                        label=lbl, linestyle="-")
            if y0b is not None and y1b is not None:
                ax.plot([i0, i1], [y0b, y1b], color=color, lw=1.6, alpha=0.9,
                        linestyle="--")

            # Key-point marker (head for H&S, 2nd peak for double top/bottom)
            x_mid_d = str(getattr(r, "x_mid_date", "") or "")
            y_mid_v = _safe_float(getattr(r, "y_mid", None))
            if x_mid_d and y_mid_v is not None:
                i_mid = date_to_idx.get(x_mid_d, (i0 + i1) // 2)
                ax.scatter([i_mid], [y_mid_v], s=60, color=color, zorder=5, alpha=0.95, marker="^")

            # Start-point marker at (i0, y0)
            if y0 is not None:
                ax.scatter([i0], [y0], s=40, color=color, zorder=4, alpha=0.8, marker="o")

            # Annotate with pattern name near the midpoint
            mid_x = (i0 + i1) / 2
            mid_y_val = None
            if y0 is not None and y1 is not None:
                mid_y_val = (y0 + y1) / 2
            if mid_y_val is not None:
                ax.annotate(
                    pname.replace("_", " "),
                    xy=(mid_x, mid_y_val),
                    fontsize=7,
                    color=color,
                    alpha=0.85,
                    ha="center",
                    va="bottom",
                    xytext=(0, 6),
                    textcoords="offset points",
                )

    ax.grid(True, color="#24314a", alpha=0.35, lw=0.6)
    ax.set_xlim(-1, n)
    ax.tick_params(colors="#94a7c4", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#24314a")
    if title:
        ax.set_title(title, color="#dce6f7", fontsize=10)

    handles = [h for h in ax.get_legend_handles_labels()[0] if h is not None]
    labels  = [l for l in ax.get_legend_handles_labels()[1] if l]
    if handles:
        ax.legend(handles, labels, fontsize=7, loc="upper left",
                  facecolor="#1a2232", edgecolor="#3a4a5a", labelcolor="#dce6f7")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return True


def _render_clean_png(
    window: pd.DataFrame,
    out_path: Path,
    width: int = 1280,
    height: int = 720,
    title: str = "",
) -> bool:
    """Plain candlestick chart — no overlays, no text. Used for model training."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle as MRect
    except ImportError:
        return False

    n = len(window)
    if n < 2:
        return False

    fig_w = max(width / 100.0, 6.0)
    fig_h = max(height / 100.0, 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.set_facecolor("#111722")
    fig.patch.set_facecolor("#111722")

    for i, r in enumerate(window.itertuples(index=False)):
        o, h, l, c = float(r.open), float(r.high), float(r.low), float(r.close)
        up = c >= o
        color = "#38d39f" if up else "#ff6b6b"
        ax.vlines(i, l, h, color=color, linewidth=1.0, alpha=0.9)
        body_lo = min(o, c)
        body_h = max(abs(c - o), 1e-6)
        ax.add_patch(MRect((i - 0.3, body_lo), 0.6, body_h, facecolor=color, edgecolor=color, lw=0.7))

    ax.grid(True, color="#24314a", alpha=0.35, lw=0.6)
    ax.set_xlim(-1, n)
    ax.axis("off")   # no axes, ticks or labels — pure chart signal

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return True


def _safe_float(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
        return None if (f != f) else f   # reject NaN
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# detect
# --------------------------------------------------------------------------- #

def cmd_detect(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    (out_dir / "images").mkdir(parents=True, exist_ok=True)          # clean, for model training
    (out_dir / "images_review").mkdir(parents=True, exist_ok=True)   # annotated, for human review
    (out_dir / "windows").mkdir(parents=True, exist_ok=True)

    gold_path = Path(args.gold_labels).resolve()
    gold = _load_gold(gold_path)
    already_confirmed: set[str] = set(gold["sample_id"].unique()) if not gold.empty else set()

    conn = sqlite3.connect(str(Path(args.db_path).resolve()))
    try:
        tickers = pick_tickers(conn, args)
        if not tickers:
            raise ValueError("No tickers selected.")
        log.info("Detecting patterns for %d ticker(s): %s", len(tickers), ", ".join(tickers[:10]) + ("…" if len(tickers) > 10 else ""))
        log.info("Already confirmed in gold: %d samples — will skip these", len(already_confirmed))

        fcfg = FeatureConfig()
        pcfg = PatternConfig()
        ccfg = CandlePatternConfig()
        hcfg = HybridPatternConfig()
        cvcfg = CVProxyConfig()

        sample_rows, weak_rows = [], []
        new_count = 0

        for ticker in tickers:
            df = _load_price(conn, ticker)
            if len(df) < max(args.min_bars, args.lookback_bars):
                continue

            window_ends = list(range(args.lookback_bars, len(df) + 1, args.stride))
            if args.max_windows_per_ticker > 0:
                window_ends = window_ends[-args.max_windows_per_ticker:]
            log.info("  %s — %d windows", ticker, len(window_ends))

            for end in window_ends:
                win = df.iloc[end - args.lookback_bars: end].copy().reset_index(drop=True)
                win["date"] = pd.to_datetime(win["date"], errors="coerce")
                win = win.dropna(subset=["date"]).reset_index(drop=True)
                if len(win) < args.lookback_bars:
                    continue

                sample_id = f"{ticker}_{pd.Timestamp(win['date'].iloc[-1]).strftime('%Y%m%d')}_{len(win)}"

                # Skip samples already confirmed in gold
                if sample_id in already_confirmed:
                    continue

                new_count += 1
                win_save = win.copy()
                win_save["date"] = win_save["date"].dt.strftime("%Y-%m-%d")

                win_csv = out_dir / "windows" / f"{sample_id}.csv"
                win_save.to_csv(win_csv, index=False)

                # clean image (no overlays) → used for model training
                img_path = out_dir / "images" / f"{sample_id}.png"
                # annotated image (pattern lines + labels) → used for human review
                img_review_path = out_dir / "images_review" / f"{sample_id}.png"

                model = add_basic_features(win_save, fcfg)
                model = compute_pivots(model, left=3, right=3)
                patt = detect_patterns(model, pcfg)
                candles = detect_candlestick_patterns(model, ccfg)
                hybrid = score_hybrid_patterns(model, patt, candles, hcfg)
                cv_proxy = detect_cv_proxy_patterns(model, cvcfg)

                if args.render_images:
                    try:
                        # Clean chart → model trains on this, no text/lines
                        _render_clean_png(
                            win_save, img_path,
                            width=args.image_width,
                            height=args.image_height,
                            title=f"{ticker}  {win_save['date'].iloc[-1]}",
                        )
                        # Annotated chart → human review only
                        _render_annotated_png(
                            win_save,
                            all_patterns=[cv_proxy, patt, hybrid],
                            out_path=img_review_path,
                            width=args.image_width,
                            height=args.image_height,
                            title=f"{ticker}  {win_save['date'].iloc[-1]}",
                        )
                    except Exception as _render_err:
                        log.warning("Image render failed for %s: %s", sample_id, _render_err)

                merged = merge_pattern_sources(
                    ticker=ticker,
                    sample_id=sample_id,
                    window=win_save,
                    rule_df=patt,
                    hybrid_df=hybrid,
                    cv_df=cv_proxy,
                    auto_accept=args.auto_accept_threshold,
                    review_min=args.review_threshold,
                )
                weak_rows.extend(merged)

                sample_rows.append({
                    "sample_id": sample_id,
                    "ticker": ticker,
                    "end_date": win_save["date"].iloc[-1],
                    "num_bars": len(win_save),
                    "image_path": str(img_path.relative_to(out_dir)) if img_path.exists() else "",
                    "review_image_path": str(img_review_path.relative_to(out_dir)) if img_review_path.exists() else "",
                })

    finally:
        conn.close()

    if not weak_rows:
        log.info("No new samples found — all already confirmed in gold.")
        print(json.dumps({
            "new_samples": 0,
            "already_in_gold": len(already_confirmed),
            "message": "Nothing new to review. All samples already confirmed.",
        }, indent=2))
        return

    log.info("Detection done: %d new samples, %d weak labels", new_count, len(weak_rows))
    weak_df = pd.DataFrame(weak_rows)
    weak_df.to_csv(out_dir / "batch_weak_labels.csv", index=False)

    pd.DataFrame(sample_rows).to_csv(out_dir / "batch_samples.csv", index=False)

    # Write review template — only NEW samples not yet in gold
    template = weak_df[["sample_id", "pattern"]].copy()
    template["approved"] = ""
    template["label_override"] = ""
    template["status_override"] = ""
    template["review_notes"] = ""
    template["reviewer"] = ""
    template_path = out_dir / "pending_review_template.csv"
    template.to_csv(template_path, index=False)

    auto_accepted = weak_df[weak_df["decision"] == "auto_accept"]

    print(json.dumps({
        "tickers_processed": len(tickers) if tickers else 0,
        "new_samples": new_count,
        "already_in_gold": len(already_confirmed),
        "new_weak_labels": len(weak_df),
        "auto_accepted": len(auto_accepted),
        "needs_review": len(weak_df) - len(auto_accepted),
        "pending_review_template": str(template_path),
        "next_step": "Fill 'approved' column (TRUE/FALSE) in pending_review_template.csv, then run: python scripts/grow_gold_labels.py merge",
    }, indent=2))


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #

def cmd_merge(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    gold_path = Path(args.gold_labels).resolve()
    template_path = out_dir / "pending_review_template.csv"
    batch_weak_path = out_dir / "batch_weak_labels.csv"

    if not template_path.exists():
        raise FileNotFoundError(f"No pending review template found at {template_path}. Run detect first.")
    if not batch_weak_path.exists():
        raise FileNotFoundError(f"No batch weak labels found at {batch_weak_path}. Run detect first.")

    decisions = pd.read_csv(template_path)
    weak = pd.read_csv(batch_weak_path)

    for c in ["sample_id", "pattern"]:
        decisions[c] = decisions[c].astype(str)
        weak[c] = weak[c].astype(str)

    decisions["approved_flag"] = decisions["approved"].map(_parse_bool)

    # Auto-accept those flagged by detector + manually approved ones
    auto_ok = weak["decision"].astype(str).eq("auto_accept")
    merged = weak.merge(decisions[["sample_id", "pattern", "approved_flag", "label_override", "review_notes", "reviewer"]],
                        on=["sample_id", "pattern"], how="left")
    merged["approved_flag"] = merged["approved_flag"].fillna(False)

    keep = auto_ok | merged["approved_flag"]
    new_gold = merged[keep].copy()

    if new_gold.empty:
        log.info("No approved labels to add.")
        print(json.dumps({"added_to_gold": 0, "message": "No approved labels to add."}, indent=2))
        return

    log.info("Merging %d approved labels into gold set", len(new_gold))

    # Resolve label overrides (guard against NaN coerced to string 'nan')
    override_val = new_gold["label_override"].astype(str).str.strip()
    has_override = override_val.ne("") & override_val.str.lower().ne("nan")
    new_gold.loc[has_override, "pattern"] = override_val[has_override]

    new_gold["label_source"] = "auto_accept"
    new_gold.loc[merged.loc[keep, "approved_flag"].values, "label_source"] = "human_review"
    new_gold["review_state"] = "confirmed"

    for c in GOLD_COLS:
        if c not in new_gold.columns:
            new_gold[c] = ""

    new_gold_df = new_gold[GOLD_COLS].copy()

    # Load and append to existing gold
    existing = _load_gold(gold_path)
    combined = pd.concat([existing, new_gold_df], ignore_index=True)
    # Deduplicate — keep last (newest review wins)
    combined = combined.drop_duplicates(subset=["sample_id", "pattern"], keep="last")

    gold_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(gold_path, index=False)
    log.info("Gold set updated: %d total confirmed labels → %s", len(combined), gold_path)

    # Archive the used template so detect can start fresh next round
    archive_path = out_dir / f"pending_review_template.done.csv"
    template_path.rename(archive_path)

    rejected = merged[~keep]
    print(json.dumps({
        "added_to_gold": len(new_gold_df),
        "total_gold": len(combined),
        "rejected_this_round": len(rejected),
        "gold_labels_path": str(gold_path),
        "next_step": "Run detect with more tickers to keep growing, or train a model on gold_labels.csv",
    }, indent=2))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Incrementally grow gold labels through detect → review → merge cycles.")
    p.add_argument("--db-path", default="data/trader_koo.db")
    p.add_argument("--out-dir", default="data/cv")
    p.add_argument("--gold-labels", default="data/cv/gold_labels.csv")

    sub = p.add_subparsers(dest="mode", required=True)

    # detect subcommand
    d = sub.add_parser("detect", help="Detect patterns for new tickers and write pending_review_template.csv")
    d.add_argument("--tickers", default="AAPL,SPY,NVDA")
    d.add_argument("--use-all-tickers", action="store_true")
    d.add_argument("--lookback-bars", type=int, default=120)
    d.add_argument("--stride", type=int, default=5)
    d.add_argument("--min-bars", type=int, default=260)
    d.add_argument("--max-windows-per-ticker", type=int, default=50)
    d.add_argument("--render-images", action="store_true")
    d.add_argument("--image-width", type=int, default=1280)
    d.add_argument("--image-height", type=int, default=720)
    d.add_argument("--auto-accept-threshold", type=float, default=0.82)
    d.add_argument("--review-threshold", type=float, default=0.55)

    # merge subcommand
    sub.add_parser("merge", help="Merge approved labels from pending_review_template.csv into gold_labels.csv")

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "detect":
        cmd_detect(args)
    elif args.mode == "merge":
        cmd_merge(args)
