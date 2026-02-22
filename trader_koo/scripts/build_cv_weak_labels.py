from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from trader_koo.cv.compare import HybridCVCompareConfig, compare_hybrid_vs_cv
from trader_koo.cv.proxy_patterns import CVProxyConfig, detect_cv_proxy_patterns
from trader_koo.data.schema import ensure_ohlcv_schema
from trader_koo.features.candle_patterns import CandlePatternConfig, detect_candlestick_patterns
from trader_koo.features.technical import FeatureConfig, add_basic_features, compute_pivots
from trader_koo.structure.hybrid_patterns import HybridPatternConfig, score_hybrid_patterns
from trader_koo.structure.patterns import PatternConfig, detect_patterns


DEFAULT_CLASSES = [
    "bull_flag",
    "bear_flag",
    "rising_wedge",
    "falling_wedge",
    "ascending_triangle",
    "descending_triangle",
    "symmetrical_triangle",
    "double_top",
    "double_bottom",
    "head_and_shoulders",
    "inv_head_and_shoulders",
    "cup_and_handle",
]


def parse_tickers(raw: str) -> list[str]:
    out = []
    for t in (raw or "").split(","):
        x = t.strip().upper()
        if not x:
            continue
        out.append(x)
    return sorted(set(out))


def pick_tickers(conn: sqlite3.Connection, args: argparse.Namespace) -> list[str]:
    if args.use_all_tickers:
        rows = conn.execute("SELECT DISTINCT ticker FROM price_daily ORDER BY ticker").fetchall()
        return [str(r[0]).upper() for r in rows]
    return parse_tickers(args.tickers)


def load_price_df(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
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
    return ensure_ohlcv_schema(df)


def maybe_import_matplotlib() -> tuple[Any, Any] | tuple[None, None]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        return plt, Rectangle
    except Exception:
        return None, None


def render_candles_png(
    window: pd.DataFrame,
    out_path: Path,
    width: int,
    height: int,
    title: str = "",
) -> bool:
    plt, Rectangle = maybe_import_matplotlib()
    if plt is None or Rectangle is None:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(window)
    if n < 2:
        return False

    fig_w = max(width / 100.0, 6.0)
    fig_h = max(height / 100.0, 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.set_facecolor("#111722")
    fig.patch.set_facecolor("#111722")

    x = range(n)
    for i, r in enumerate(window.itertuples(index=False)):
        o = float(r.open)
        h = float(r.high)
        l = float(r.low)
        c = float(r.close)
        up = c >= o
        color = "#38d39f" if up else "#ff6b6b"
        ax.vlines(i, l, h, color=color, linewidth=1.0, alpha=0.95)
        body_low = min(o, c)
        body_h = max(abs(c - o), 1e-6)
        ax.add_patch(Rectangle((i - 0.3, body_low), 0.6, body_h, facecolor=color, edgecolor=color, linewidth=0.7))

    ax.grid(True, color="#24314a", alpha=0.35, linewidth=0.6)
    ax.set_xlim(-1, n)
    ax.tick_params(colors="#94a7c4", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#24314a")
    if title:
        ax.set_title(title, color="#dce6f7", fontsize=10)

    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return True


def class_map_from_labels(all_labels: pd.DataFrame) -> dict[str, int]:
    classes = list(DEFAULT_CLASSES)
    if not all_labels.empty:
        extras = sorted(set(str(x) for x in all_labels["pattern"].dropna().tolist() if str(x) not in classes))
        classes.extend(extras)
    return {name: i for i, name in enumerate(classes)}


def merge_pattern_sources(
    ticker: str,
    sample_id: str,
    window: pd.DataFrame,
    rule_df: pd.DataFrame,
    hybrid_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    auto_accept: float,
    review_min: float,
) -> list[dict[str, Any]]:
    idx_by_date = {pd.Timestamp(d).strftime("%Y-%m-%d"): i for i, d in enumerate(pd.to_datetime(window["date"], errors="coerce"))}
    pmin = float(window["low"].min())
    pmax = float(window["high"].max())
    span = max(pmax - pmin, 1e-9)
    n = len(window)

    agg: dict[str, dict[str, Any]] = {}

    def upd(pattern: str, source: str, conf: float, status: str, row: dict[str, Any]) -> None:
        k = str(pattern)
        if k not in agg:
            agg[k] = {
                "pattern": k,
                "ticker": ticker,
                "sample_id": sample_id,
                "statuses": [],
                "rule_conf": math.nan,
                "hybrid_conf": math.nan,
                "cv_conf": math.nan,
                "geom_row": None,
                "geom_conf": -1.0,
            }
        agg[k]["statuses"].append(str(status or "forming"))
        if source == "rule":
            agg[k]["rule_conf"] = float(conf)
        elif source == "hybrid":
            agg[k]["hybrid_conf"] = float(conf)
        elif source == "cv_proxy":
            agg[k]["cv_conf"] = float(conf)

        if row and all(x in row for x in ["x0_date", "x1_date", "y0", "y1", "y0b", "y1b"]):
            if float(conf) > float(agg[k]["geom_conf"]):
                agg[k]["geom_conf"] = float(conf)
                agg[k]["geom_row"] = row

    for r in rule_df.to_dict(orient="records"):
        upd(r.get("pattern", ""), "rule", float(r.get("confidence", 0.0) or 0.0), str(r.get("status", "forming")), r)
    for r in hybrid_df.to_dict(orient="records"):
        upd(r.get("pattern", ""), "hybrid", float(r.get("hybrid_confidence", 0.0) or 0.0), str(r.get("status", "forming")), r)
    for r in cv_df.to_dict(orient="records"):
        upd(r.get("pattern", ""), "cv_proxy", float(r.get("cv_confidence", 0.0) or 0.0), str(r.get("status", "forming")), r)

    rows: list[dict[str, Any]] = []
    for k, v in agg.items():
        confs = [x for x in [v["rule_conf"], v["hybrid_conf"], v["cv_conf"]] if isinstance(x, float) and not math.isnan(x)]
        if not confs:
            continue

        # Weighted consensus gives hybrid slightly stronger weight.
        wsum = 0.0
        csum = 0.0
        if isinstance(v["rule_conf"], float) and not math.isnan(v["rule_conf"]):
            wsum += 0.30
            csum += 0.30 * v["rule_conf"]
        if isinstance(v["hybrid_conf"], float) and not math.isnan(v["hybrid_conf"]):
            wsum += 0.45
            csum += 0.45 * v["hybrid_conf"]
        if isinstance(v["cv_conf"], float) and not math.isnan(v["cv_conf"]):
            wsum += 0.25
            csum += 0.25 * v["cv_conf"]
        consensus = csum / max(wsum, 1e-9)

        statuses = [s for s in v["statuses"] if s]
        uniq_status = sorted(set(statuses))
        agreement = "aligned" if len(uniq_status) <= 1 else "status_conflict"
        final_status = uniq_status[0] if len(uniq_status) == 1 else (statuses[-1] if statuses else "forming")

        source_count = len(confs)
        if source_count >= 2 and consensus >= auto_accept and agreement == "aligned":
            decision = "auto_accept"
        elif consensus >= review_min:
            decision = "review"
        else:
            decision = "low_conf"

        geom = v["geom_row"] or {}
        x0 = str(geom.get("x0_date") or geom.get("start_date") or "")
        x1 = str(geom.get("x1_date") or geom.get("end_date") or "")
        i0 = idx_by_date.get(x0, max(0, n - 12))
        i1 = idx_by_date.get(x1, n - 1)
        if i1 < i0:
            i0, i1 = i1, i0
        ys = []
        for yc in ["y0", "y1", "y0b", "y1b"]:
            try:
                ys.append(float(geom.get(yc)))
            except Exception:
                pass
        if ys:
            ylo = max(pmin, min(ys))
            yhi = min(pmax, max(ys))
        else:
            ylo, yhi = pmin, pmax

        xc = ((i0 + i1) / 2.0) / max(n - 1, 1)
        bw = max((i1 - i0 + 1) / max(n - 1, 1), 1e-6)
        yc = 1.0 - (((ylo + yhi) / 2.0 - pmin) / span)
        bh = max((yhi - ylo) / span, 1e-6)

        rows.append(
            {
                "sample_id": sample_id,
                "ticker": ticker,
                "pattern": k,
                "status": final_status,
                "rule_conf": round(float(v["rule_conf"]), 2) if isinstance(v["rule_conf"], float) and not math.isnan(v["rule_conf"]) else None,
                "hybrid_conf": round(float(v["hybrid_conf"]), 2) if isinstance(v["hybrid_conf"], float) and not math.isnan(v["hybrid_conf"]) else None,
                "cv_conf": round(float(v["cv_conf"]), 2) if isinstance(v["cv_conf"], float) and not math.isnan(v["cv_conf"]) else None,
                "consensus_conf": round(float(consensus), 2),
                "source_count": source_count,
                "agreement": agreement,
                "decision": decision,
                "x0_idx": int(i0),
                "x1_idx": int(i1),
                "y_min": float(ylo),
                "y_max": float(yhi),
                "yolo_x_center": float(max(0.0, min(1.0, xc))),
                "yolo_y_center": float(max(0.0, min(1.0, yc))),
                "yolo_w": float(max(1e-6, min(1.0, bw))),
                "yolo_h": float(max(1e-6, min(1.0, bh))),
            }
        )
    return rows


def assign_splits(samples: pd.DataFrame, train_ratio: float, val_ratio: float) -> pd.DataFrame:
    out = samples.copy()
    if out.empty:
        out["split"] = []
        return out
    out["end_date"] = pd.to_datetime(out["end_date"], errors="coerce")
    out = out.sort_values("end_date").reset_index(drop=True)
    n = len(out)
    train_cut = int(n * train_ratio)
    val_cut = int(n * (train_ratio + val_ratio))
    split = []
    for i in range(n):
        if i < train_cut:
            split.append("train")
        elif i < val_cut:
            split.append("val")
        else:
            split.append("test")
    out["split"] = split
    out["end_date"] = out["end_date"].dt.strftime("%Y-%m-%d")
    return out


def write_yolo_dataset(
    out_dir: Path,
    samples: pd.DataFrame,
    labels: pd.DataFrame,
    class_map: dict[str, int],
    yolo_min_conf: float,
    yolo_include_review: bool,
) -> None:
    yolo_dir = out_dir / "yolo"
    for split in ["train", "val", "test"]:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    labels_ok = labels.copy()
    if labels_ok.empty:
        labels_ok = pd.DataFrame(columns=["sample_id"])

    keep_decisions = {"auto_accept"}
    if yolo_include_review:
        keep_decisions.add("review")
    labels_ok = labels_ok[
        labels_ok["decision"].isin(keep_decisions) & (pd.to_numeric(labels_ok["consensus_conf"], errors="coerce") >= yolo_min_conf)
    ].copy()

    sample_by_id = {r["sample_id"]: r for r in samples.to_dict(orient="records")}
    for sid, srow in sample_by_id.items():
        split = str(srow.get("split") or "train")
        src_img = out_dir / "images" / f"{sid}.png"
        dst_img = yolo_dir / "images" / split / f"{sid}.png"
        if src_img.exists():
            shutil.copy2(src_img, dst_img)

        txt_path = yolo_dir / "labels" / split / f"{sid}.txt"
        sub = labels_ok[labels_ok["sample_id"] == sid]
        lines = []
        for r in sub.to_dict(orient="records"):
            cls = class_map.get(str(r.get("pattern")))
            if cls is None:
                continue
            lines.append(
                f"{cls} {float(r['yolo_x_center']):.6f} {float(r['yolo_y_center']):.6f} {float(r['yolo_w']):.6f} {float(r['yolo_h']):.6f}"
            )
        txt_path.write_text("\n".join(lines), encoding="utf-8")

    classes = [k for k, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    yaml = {
        "path": str(yolo_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(classes)},
    }
    (yolo_dir / "dataset.yaml").write_text(json.dumps(yaml, indent=2), encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "windows").mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(Path(args.db_path).resolve()))
    try:
        tickers = pick_tickers(conn, args)
        if not tickers:
            raise ValueError("No tickers selected.")

        fcfg = FeatureConfig()
        pcfg = PatternConfig()
        ccfg = CandlePatternConfig()
        hcfg = HybridPatternConfig()
        cvcfg = CVProxyConfig()
        cmpcfg = HybridCVCompareConfig()

        sample_rows: list[dict[str, Any]] = []
        weak_label_rows: list[dict[str, Any]] = []
        compare_rows: list[dict[str, Any]] = []

        for ticker in tickers:
            df = load_price_df(conn, ticker)
            if len(df) < max(args.min_bars, args.lookback_bars):
                continue

            window_ends = list(range(args.lookback_bars, len(df) + 1, args.stride))
            if args.max_windows_per_ticker > 0:
                window_ends = window_ends[-args.max_windows_per_ticker :]

            for end in window_ends:
                win = df.iloc[end - args.lookback_bars : end].copy().reset_index(drop=True)
                if win.empty:
                    continue
                win["date"] = pd.to_datetime(win["date"], errors="coerce")
                win = win.dropna(subset=["date"]).reset_index(drop=True)
                if len(win) < args.lookback_bars:
                    continue

                sample_id = f"{ticker}_{pd.Timestamp(win['date'].iloc[-1]).strftime('%Y%m%d')}_{len(win)}"
                win_csv = out_dir / "windows" / f"{sample_id}.csv"
                img_path = out_dir / "images" / f"{sample_id}.png"

                win_save = win.copy()
                win_save["date"] = win_save["date"].dt.strftime("%Y-%m-%d")
                win_save.to_csv(win_csv, index=False)

                if args.render_images:
                    render_candles_png(win_save, img_path, width=args.image_width, height=args.image_height, title=f"{ticker} {win_save['date'].iloc[-1]}")

                model = add_basic_features(win_save, fcfg)
                model = compute_pivots(model, left=3, right=3)
                patt = detect_patterns(model, pcfg)
                candles = detect_candlestick_patterns(model, ccfg)
                hybrid = score_hybrid_patterns(model, patt, candles, hcfg)
                cv_proxy = detect_cv_proxy_patterns(model, cvcfg)
                comp = compare_hybrid_vs_cv(hybrid, cv_proxy, cmpcfg)

                sample_rows.append(
                    {
                        "sample_id": sample_id,
                        "ticker": ticker,
                        "start_date": pd.Timestamp(win_save["date"].iloc[0]).strftime("%Y-%m-%d"),
                        "end_date": pd.Timestamp(win_save["date"].iloc[-1]).strftime("%Y-%m-%d"),
                        "num_bars": len(win_save),
                        "price_min": float(win_save["low"].min()),
                        "price_max": float(win_save["high"].max()),
                        "rule_pattern_count": int(len(patt)),
                        "hybrid_pattern_count": int(len(hybrid)),
                        "cv_pattern_count": int(len(cv_proxy)),
                        "image_path": str(img_path.relative_to(out_dir)) if img_path.exists() else "",
                        "window_path": str(win_csv.relative_to(out_dir)),
                    }
                )

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
                weak_label_rows.extend(merged)

                if not comp.empty:
                    c = comp.copy()
                    c["sample_id"] = sample_id
                    c["ticker"] = ticker
                    compare_rows.extend(c.to_dict(orient="records"))

        samples_df = pd.DataFrame(sample_rows)
        labels_df = pd.DataFrame(weak_label_rows)
        compare_df = pd.DataFrame(compare_rows)

        if not samples_df.empty:
            samples_df = assign_splits(samples_df, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

        class_map = class_map_from_labels(labels_df)
        class_df = pd.DataFrame([{"class_id": v, "class_name": k} for k, v in sorted(class_map.items(), key=lambda kv: kv[1])])

        review_df = pd.DataFrame()
        if not labels_df.empty:
            review_df = labels_df[
                (labels_df["decision"] == "review")
                | (labels_df["agreement"] != "aligned")
                | (pd.to_numeric(labels_df["source_count"], errors="coerce") < 2)
            ].copy()
            if not compare_df.empty:
                bad = compare_df[compare_df["agreement"] != "aligned"]["sample_id"].dropna().astype(str).unique().tolist()
                if bad:
                    extra = labels_df[labels_df["sample_id"].isin(bad)].copy()
                    review_df = pd.concat([review_df, extra], ignore_index=True)
            if not review_df.empty:
                review_df = review_df.drop_duplicates(subset=["sample_id", "pattern"]).reset_index(drop=True)

        samples_df.to_csv(out_dir / "samples.csv", index=False)
        labels_df.to_csv(out_dir / "weak_labels.csv", index=False)
        compare_df.to_csv(out_dir / "hybrid_cv_compare.csv", index=False)
        review_df.to_csv(out_dir / "review_queue.csv", index=False)
        class_df.to_csv(out_dir / "class_map.csv", index=False)
        (out_dir / "classes.txt").write_text("\n".join(class_df["class_name"].tolist()) + "\n", encoding="utf-8")

        template_cols = ["sample_id", "pattern", "approved", "label_override", "status_override", "review_notes", "reviewer"]
        if not review_df.empty:
            template = review_df[["sample_id", "pattern"]].copy()
            template["approved"] = ""
            template["label_override"] = ""
            template["status_override"] = ""
            template["review_notes"] = ""
            template["reviewer"] = ""
        else:
            template = pd.DataFrame(columns=template_cols)
        template.to_csv(out_dir / "review_decisions_template.csv", index=False)

        if not samples_df.empty:
            write_yolo_dataset(
                out_dir=out_dir,
                samples=samples_df,
                labels=labels_df,
                class_map=class_map,
                yolo_min_conf=args.yolo_min_confidence,
                yolo_include_review=args.yolo_include_review,
            )

        summary = {
            "tickers": len(tickers),
            "samples": int(len(samples_df)),
            "weak_labels": int(len(labels_df)),
            "review_queue": int(len(review_df)),
            "classes": int(len(class_df)),
            "out_dir": str(out_dir),
        }
        print(json.dumps(summary, indent=2))
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build weak-labeled CV dataset with human-review queue.")
    p.add_argument("--db-path", default="data/trader_koo.db")
    p.add_argument("--out-dir", default="data/cv")
    p.add_argument("--tickers", default="SPY,QQQ,IWM,DIA,NVDA,AAPL,MSFT,TSLA")
    p.add_argument("--use-all-tickers", action="store_true")
    p.add_argument("--lookback-bars", type=int, default=120)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--min-bars", type=int, default=260)
    p.add_argument("--max-windows-per-ticker", type=int, default=250)
    p.add_argument("--render-images", action="store_true")
    p.add_argument("--image-width", type=int, default=1280)
    p.add_argument("--image-height", type=int, default=720)
    p.add_argument("--auto-accept-threshold", type=float, default=0.82)
    p.add_argument("--review-threshold", type=float, default=0.55)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--yolo-min-confidence", type=float, default=0.65)
    p.add_argument("--yolo-include-review", action="store_true")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    build(args)
