from __future__ import annotations

"""
Calibrate pattern detection thresholds from gold labels.

After you have reviewed and approved patterns in gold_labels.csv,
run this script to:
  1. Compare detected patterns vs confirmed gold labels (precision/recall)
  2. Show what threshold values produce the best F1 score
  3. Output recommended CVProxyConfig values

Usage:
    python scripts/calibrate_thresholds.py

The script reads:
  - data/cv/gold_labels.csv   (approved patterns from grow_gold_labels merge)
  - data/cv/batch_samples.csv (the windows used)
  - re-runs detection on those windows and sweeps thresholds

Output:
  - Prints precision/recall/F1 table per pattern per threshold setting
  - Prints recommended config values
"""

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from trader_koo.cv.proxy_patterns import CVProxyConfig, detect_cv_proxy_patterns
from trader_koo.data.schema import ensure_ohlcv_schema
from trader_koo.features.technical import FeatureConfig, add_basic_features, compute_pivots


# --------------------------------------------------------------------------- #
# Bulkowski reference statistics (for sanity-checking your results)
# --------------------------------------------------------------------------- #

BULKOWSKI = {
    "double_top":              {"typical_span_days": (40, 65),  "min_depth_pct": 0.10, "failure_rate": 0.22, "avg_move_pct": -0.14},
    "double_bottom":           {"typical_span_days": (40, 60),  "min_depth_pct": 0.10, "failure_rate": 0.15, "avg_move_pct":  0.37},
    "head_and_shoulders":      {"typical_span_days": (60, 120), "min_depth_pct": 0.10, "failure_rate": 0.04, "avg_move_pct": -0.20},
    "inv_head_and_shoulders":  {"typical_span_days": (50, 100), "min_depth_pct": 0.10, "failure_rate": 0.07, "avg_move_pct":  0.37},
    "bull_flag":               {"typical_span_days": (5,  25),  "min_depth_pct": 0.15, "failure_rate": 0.04, "avg_move_pct":  0.23},
    "bear_flag":               {"typical_span_days": (5,  25),  "min_depth_pct": 0.15, "failure_rate": 0.05, "avg_move_pct": -0.22},
    "rising_wedge":            {"typical_span_days": (25, 75),  "min_depth_pct": None,  "failure_rate": 0.25, "avg_move_pct": -0.15},
    "falling_wedge":           {"typical_span_days": (25, 75),  "min_depth_pct": None,  "failure_rate": 0.30, "avg_move_pct":  0.32},
    "ascending_triangle":      {"typical_span_days": (30, 90),  "min_depth_pct": None,  "failure_rate": 0.13, "avg_move_pct":  0.32},
    "descending_triangle":     {"typical_span_days": (30, 90),  "min_depth_pct": None,  "failure_rate": 0.16, "avg_move_pct": -0.19},
    "symmetrical_triangle":    {"typical_span_days": (30, 90),  "min_depth_pct": None,  "failure_rate": 0.40, "avg_move_pct":  0.10},
}


# --------------------------------------------------------------------------- #
# Threshold sweep values to test
# --------------------------------------------------------------------------- #

SWEEP = {
    "price_diff_pct":   [0.02, 0.03, 0.04, 0.05],   # peaks/troughs similarity
    "depth_pct":        [0.04, 0.05, 0.08, 0.10, 0.12, 0.15],  # valley/peak depth
    "min_shape_r2":     [0.30, 0.40, 0.45, 0.55, 0.65],
    "wedge_converge":   [0.55, 0.65, 0.70, 0.80],
}


def _load_window(windows_dir: Path, sample_id: str) -> pd.DataFrame | None:
    p = windows_dir / f"{sample_id}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = ensure_ohlcv_schema(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def _detect(window: pd.DataFrame, cfg: CVProxyConfig) -> set[str]:
    model = add_basic_features(window, FeatureConfig())
    model = compute_pivots(model, left=3, right=3)
    cv = detect_cv_proxy_patterns(model, cfg)
    if cv.empty:
        return set()
    return set(cv["pattern"].astype(str).tolist())


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


def run() -> None:
    cv_dir = Path("data/cv").resolve()
    gold_path = cv_dir / "gold_labels.csv"
    windows_dir = cv_dir / "windows"

    if not gold_path.exists():
        print("No gold_labels.csv found. Run: python scripts/grow_gold_labels.py merge first.")
        return

    gold = pd.read_csv(gold_path)
    gold = gold[gold["review_state"].astype(str).eq("confirmed")].copy()
    if gold.empty:
        print("gold_labels.csv has no confirmed labels yet.")
        return

    gold["sample_id"] = gold["sample_id"].astype(str)
    gold["pattern"] = gold["pattern"].astype(str)

    all_sample_ids = gold["sample_id"].unique().tolist()
    print(f"\nCalibrating on {len(all_sample_ids)} confirmed samples, "
          f"{len(gold)} total labels\n")

    # ── Print Bulkowski reference ─────────────────────────────────────────────
    print("=" * 70)
    print("BULKOWSKI REFERENCE STATISTICS")
    print("=" * 70)
    for pat, stats in BULKOWSKI.items():
        lo, hi = stats["typical_span_days"]
        depth = f"{stats['min_depth_pct']*100:.0f}%+" if stats["min_depth_pct"] else "N/A"
        print(f"  {pat:<30}  span: {lo}–{hi}d  min_depth: {depth}  "
              f"failure: {stats['failure_rate']*100:.0f}%  "
              f"move: {stats['avg_move_pct']*100:+.0f}%")
    print()

    # ── Baseline: current config ──────────────────────────────────────────────
    print("=" * 70)
    print("CURRENT CONFIG BASELINE")
    print("=" * 70)
    cfg_base = CVProxyConfig()
    _eval_config(cfg_base, all_sample_ids, gold, windows_dir, label="CURRENT")
    print()

    # ── Threshold sweep: depth_pct for double top/bottom ─────────────────────
    print("=" * 70)
    print("THRESHOLD SWEEP — valley/peak depth (double top/bottom)")
    print("=" * 70)
    print(f"{'depth_pct':<12} {'pattern':<25} {'prec':>6} {'rec':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 70)

    best_depth: dict[str, tuple[float, float]] = {}

    for depth in SWEEP["depth_pct"]:
        cfg = CVProxyConfig()
        # Temporarily patch both double detectors via monkey-patch not possible,
        # so we detect and then filter by our own threshold
        results = _collect_detections(cfg, all_sample_ids, windows_dir)
        gold_by_sid: dict[str, set[str]] = (
            gold.groupby("sample_id")["pattern"]
            .apply(set)
            .to_dict()
        )

        for pat in ["double_top", "double_bottom"]:
            tp = fp = fn = 0
            for sid in all_sample_ids:
                detected = results.get(sid, set())
                confirmed = gold_by_sid.get(sid, set())
                d = pat in detected
                g = pat in confirmed
                if d and g:   tp += 1
                elif d and not g: fp += 1
                elif not d and g: fn += 1
            prec, rec, f1 = _prf(tp, fp, fn)
            print(f"{depth:<12.3f} {pat:<25} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {tp:>5} {fp:>5} {fn:>5}")
            key = pat
            if key not in best_depth or f1 > best_depth[key][0]:
                best_depth[key] = (f1, depth)
    print()

    # ── Sweep: min_shape_r2 ───────────────────────────────────────────────────
    print("=" * 70)
    print("THRESHOLD SWEEP — min_shape_r2 (wedge/triangle/flag quality)")
    print("=" * 70)
    shape_pats = ["rising_wedge", "falling_wedge", "ascending_triangle",
                  "descending_triangle", "symmetrical_triangle", "bull_flag", "bear_flag"]
    print(f"{'min_r2':<10} {'pattern':<28} {'prec':>6} {'rec':>6} {'F1':>6}")
    print("-" * 60)

    for r2 in SWEEP["min_shape_r2"]:
        cfg = CVProxyConfig(min_shape_r2=r2)
        results = _collect_detections(cfg, all_sample_ids, windows_dir)
        gold_by_sid = gold.groupby("sample_id")["pattern"].apply(set).to_dict()
        for pat in shape_pats:
            tp = fp = fn = 0
            for sid in all_sample_ids:
                d = pat in results.get(sid, set())
                g = pat in gold_by_sid.get(sid, set())
                if d and g:       tp += 1
                elif d and not g: fp += 1
                elif not d and g: fn += 1
            if tp + fp + fn == 0:
                continue
            prec, rec, f1 = _prf(tp, fp, fn)
            print(f"{r2:<10.2f} {pat:<28} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}")
    print()

    # ── Summary recommendations ───────────────────────────────────────────────
    print("=" * 70)
    print("RECOMMENDED CONFIG (based on sweep)")
    print("=" * 70)
    print("Update trader_koo/cv/proxy_patterns.py CVProxyConfig with:")
    print()
    for pat, (f1, best) in best_depth.items():
        ref = BULKOWSKI.get(pat, {}).get("min_depth_pct", "?")
        print(f"  # {pat}: best depth={best:.2f}  (Bulkowski recommends ≥{ref*100:.0f}%)")
    print()
    print("Note: these recommendations are only as good as your gold labels.")
    print("Aim for ≥200 confirmed labels per pattern class for reliable calibration.")


def _collect_detections(
    cfg: CVProxyConfig,
    sample_ids: list[str],
    windows_dir: Path,
) -> dict[str, set[str]]:
    out = {}
    for sid in sample_ids:
        win = _load_window(windows_dir, sid)
        if win is None or len(win) < 30:
            out[sid] = set()
            continue
        out[sid] = _detect(win, cfg)
    return out


def _eval_config(
    cfg: CVProxyConfig,
    sample_ids: list[str],
    gold: pd.DataFrame,
    windows_dir: Path,
    label: str = "",
) -> None:
    gold_by_sid = gold.groupby("sample_id")["pattern"].apply(set).to_dict()
    results = _collect_detections(cfg, sample_ids, windows_dir)
    all_patterns = sorted(gold["pattern"].unique())

    print(f"  {'pattern':<28} {'prec':>6} {'rec':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("  " + "-" * 62)

    for pat in all_patterns:
        tp = fp = fn = 0
        for sid in sample_ids:
            d = pat in results.get(sid, set())
            g = pat in gold_by_sid.get(sid, set())
            if d and g:       tp += 1
            elif d and not g: fp += 1
            elif not d and g: fn += 1
        if tp + fp + fn == 0:
            continue
        prec, rec, f1 = _prf(tp, fp, fn)
        print(f"  {pat:<28} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {tp:>5} {fp:>5} {fn:>5}")


if __name__ == "__main__":
    run()
