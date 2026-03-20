"""Multi-model benchmark pipeline.

Runs every combination of:
- 3 target modes: return_sign, barrier, rank
- 4 models: LightGBM, CatBoost, Logistic Regression, Random Forest
- 3 feature sets: ranked (7), slim (14), full (68)

Same walk-forward folds for all — apples-to-apples comparison.
Results saved to a JSON file for analysis.

Usage:
    /opt/anaconda3/bin/python -m trader_koo.ml.benchmark --db-path trader_koo/data/trader_koo.db
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

LOG = logging.getLogger("trader_koo.ml.benchmark")

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
from trader_koo.ml.features import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_SLIM,
    FEATURE_COLUMNS_FULL,
    extract_features_for_universe,
)

try:
    from trader_koo.ml.features import FEATURE_COLUMNS_RANKED
except ImportError:
    FEATURE_COLUMNS_RANKED = FEATURE_COLUMNS

from trader_koo.ml.trainer import _apply_target_mode, build_dataset

FEATURE_SETS = {
    "ranked_7": FEATURE_COLUMNS_RANKED,
    "slim_14": FEATURE_COLUMNS_SLIM,
}

TARGET_MODES = ["return_sign", "barrier", "rank"]


def _get_models() -> dict[str, Any]:
    """Return dict of model_name -> unfitted model instance."""
    import lightgbm as lgb

    models: dict[str, Any] = {
        "lgbm": lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            num_leaves=15,
            max_depth=3,
            learning_rate=0.01,
            n_estimators=300,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        ),
        "logistic": LogisticRegression(
            max_iter=1000,
            C=0.1,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        ),
    }

    try:
        from catboost import CatBoostClassifier
        models["catboost"] = CatBoostClassifier(
            iterations=300,
            depth=3,
            learning_rate=0.01,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=0,
        )
    except ImportError:
        LOG.info("CatBoost not installed — skipping")

    return models


def run_benchmark(
    conn: sqlite3.Connection,
    *,
    start_date: str = "2020-01-01",
    train_days: int = 180,
    test_days: int = 60,
    step_days: int = 30,
    embargo_days: int = 15,
    sample_frequency: int = 10,
) -> dict[str, Any]:
    """Run full benchmark: all models × all targets × all feature sets.

    Returns a dict with results for every combination.
    """
    print("=" * 70)
    print("MULTI-MODEL BENCHMARK")
    print(f"Start: {start_date}, Train: {train_days}d, Test: {test_days}d")
    print(f"Step: {step_days}d, Embargo: {embargo_days}d, Sample freq: {sample_frequency}")
    print("=" * 70)
    sys.stdout.flush()

    # Build dataset ONCE (labels + features are the same for all models)
    print("\n[1/3] Building dataset...")
    sys.stdout.flush()
    end_date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    dataset = build_dataset(
        conn,
        start_date=start_date,
        end_date=end_date,
        sample_frequency=sample_frequency,
    )
    print(f"  Dataset: {len(dataset)} samples, {dataset['entry_date'].nunique()} dates")
    sys.stdout.flush()

    models = _get_models()
    all_results: list[dict[str, Any]] = []

    # Walk-forward fold boundaries (compute once, reuse for all combos)
    dataset["entry_date_ts"] = pd.to_datetime(dataset["entry_date"])
    min_date = dataset["entry_date_ts"].min()
    max_date = dataset["entry_date_ts"].max()

    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    fold_start = min_date
    while True:
        train_end = fold_start + pd.Timedelta(days=train_days)
        test_start = train_end + pd.Timedelta(days=embargo_days)
        test_end = test_start + pd.Timedelta(days=test_days)
        if test_start > max_date:
            break
        folds.append((fold_start, train_end, test_start, min(test_end, max_date)))
        fold_start += pd.Timedelta(days=step_days)

    print(f"  Walk-forward: {len(folds)} folds")
    sys.stdout.flush()

    combo_count = len(TARGET_MODES) * len(FEATURE_SETS) * len(models)
    print(f"\n[2/3] Running {combo_count} combinations ({len(TARGET_MODES)} targets × {len(FEATURE_SETS)} feature sets × {len(models)} models)")
    sys.stdout.flush()

    combo_idx = 0
    for target_mode in TARGET_MODES:
        # Apply target mode
        ds = _apply_target_mode(dataset.copy(), target_mode)
        if len(ds) < 100:
            print(f"  SKIP target_mode={target_mode} — only {len(ds)} samples after filtering")
            sys.stdout.flush()
            continue

        for feat_name, feat_cols in FEATURE_SETS.items():
            # Check which features actually exist in the dataset
            available_cols = [c for c in feat_cols if c in ds.columns]
            if len(available_cols) < 3:
                print(f"  SKIP {feat_name} — only {len(available_cols)} features available")
                sys.stdout.flush()
                continue

            for model_name, model_template in models.items():
                combo_idx += 1
                t0 = time.time()
                fold_aucs: list[float] = []
                fold_accs: list[float] = []

                for fold_start_ts, train_end_ts, test_start_ts, test_end_ts in folds:
                    train_mask = (ds["entry_date_ts"] >= fold_start_ts) & (ds["entry_date_ts"] <= train_end_ts)
                    test_mask = (ds["entry_date_ts"] >= test_start_ts) & (ds["entry_date_ts"] <= test_end_ts)

                    X_train = ds.loc[train_mask, available_cols].copy()
                    y_train = ds.loc[train_mask, "target"].copy()
                    X_test = ds.loc[test_mask, available_cols].copy()
                    y_test = ds.loc[test_mask, "target"].copy()

                    if len(X_train) < 50 or len(X_test) < 10:
                        continue
                    if y_train.nunique() < 2 or y_test.nunique() < 2:
                        continue

                    # Impute NaN
                    medians = X_train.median()
                    X_train = X_train.fillna(medians)
                    X_test = X_test.fillna(medians)

                    # Train (fresh model each fold)
                    import copy
                    model = copy.deepcopy(model_template)
                    try:
                        if model_name == "lgbm":
                            import lightgbm as lgb
                            # Split train into train/val for early stopping
                            val_split = int(len(X_train) * 0.8)
                            model.fit(
                                X_train.iloc[:val_split], y_train.iloc[:val_split],
                                eval_set=[(X_train.iloc[val_split:], y_train.iloc[val_split:])],
                                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
                            )
                        else:
                            model.fit(X_train, y_train)

                        y_prob = model.predict_proba(X_test)[:, 1]
                        y_pred = model.predict(X_test)
                        auc = float(roc_auc_score(y_test, y_prob))
                        acc = float(accuracy_score(y_test, y_pred))
                        fold_aucs.append(auc)
                        fold_accs.append(acc)
                    except Exception as exc:
                        LOG.debug("Fold failed for %s/%s/%s: %s", model_name, target_mode, feat_name, exc)
                        continue

                elapsed = time.time() - t0
                avg_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.0
                avg_acc = float(np.mean(fold_accs)) if fold_accs else 0.0

                result = {
                    "model": model_name,
                    "target_mode": target_mode,
                    "feature_set": feat_name,
                    "n_features": len(available_cols),
                    "n_folds": len(fold_aucs),
                    "avg_auc": round(avg_auc, 4),
                    "avg_acc": round(avg_acc, 4),
                    "min_auc": round(min(fold_aucs), 4) if fold_aucs else 0.0,
                    "max_auc": round(max(fold_aucs), 4) if fold_aucs else 0.0,
                    "elapsed_sec": round(elapsed, 1),
                }
                all_results.append(result)

                marker = "***" if avg_auc > 0.53 else "  " if avg_auc > 0.50 else " ."
                print(f"  [{combo_idx}/{combo_count}] {model_name:15s} | {target_mode:12s} | {feat_name:10s} | AUC {avg_auc:.4f} {marker} | {len(fold_aucs)} folds | {elapsed:.0f}s")
                sys.stdout.flush()

    # Sort by AUC descending
    all_results.sort(key=lambda r: r["avg_auc"], reverse=True)

    print("\n" + "=" * 70)
    print("[3/3] FINAL LEADERBOARD")
    print("=" * 70)
    print(f"{'Rank':<5} {'Model':<15} {'Target':<12} {'Features':<10} {'AUC':<8} {'Acc':<8} {'Folds':<6}")
    print("-" * 70)
    for i, r in enumerate(all_results):
        print(f"{i+1:<5} {r['model']:<15} {r['target_mode']:<12} {r['feature_set']:<10} {r['avg_auc']:<8.4f} {r['avg_acc']:<8.4f} {r['n_folds']:<6}")
    sys.stdout.flush()

    # Save results
    output_path = Path("data/models/benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config": {
                "start_date": start_date,
                "train_days": train_days,
                "test_days": test_days,
                "step_days": step_days,
                "embargo_days": embargo_days,
                "sample_frequency": sample_frequency,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return {"results": all_results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-model ML benchmark")
    parser.add_argument("--db-path", default="trader_koo/data/trader_koo.db")
    parser.add_argument("--start-date", default="2020-01-01")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    conn = sqlite3.connect(args.db_path)

    # Prefetch FRED data
    try:
        import os
        from trader_koo.ml.external_data import prefetch_fred_series_bulk
        prefetch_fred_series_bulk(
            ["DFF", "T10Y2Y", "BAMLH0A0HYM2"],
            args.start_date,
            dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
        )
        print("FRED data prefetched")
    except Exception as exc:
        print(f"FRED prefetch failed (continuing): {exc}")

    run_benchmark(conn, start_date=args.start_date)
    conn.close()
