"""Walk-forward LightGBM trainer with purged validation.

Trains a binary classifier: will a long entry produce a profitable
outcome (label = +1) vs not (label ∈ {0, -1}).

Key design decisions to prevent data leakage:
1. Walk-forward: train on [t-train_days, t], test on [t+embargo, t+test_days]
2. Purged: embargo gap between train and test windows
3. No future data in features (enforced by features.py)
4. Labels use future data (by design — that's the target)
5. Model saved with metadata: training dates, feature columns, metrics
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from trader_koo.ml.features import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_FULL,
    extract_features_for_universe,
)
from trader_koo.ml.labels import generate_triple_barrier_labels

LOG = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("/data/models")
LOCAL_MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"

# LightGBM hyperparameters — conservative to avoid overfitting
# num_leaves=15, max_depth=3 keep trees shallow; n_estimators=500 with
# early_stopping(50) lets boosting find the right iteration count.
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 15,
    "max_depth": 3,
    "learning_rate": 0.01,
    "n_estimators": 500,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "importance_type": "gain",
}


def _model_dir() -> Path:
    """Return the model storage directory (Railway or local)."""
    if DEFAULT_MODEL_DIR.exists():
        return DEFAULT_MODEL_DIR
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return LOCAL_MODEL_DIR


def _get_trading_dates(conn: sqlite3.Connection, start: str, end: str) -> list[str]:
    """Return sorted list of dates that have price data (trading days)."""
    rows = conn.execute(
        """
        SELECT DISTINCT date FROM price_daily
        WHERE date >= ? AND date <= ? AND ticker = 'SPY'
        ORDER BY date
        """,
        (start, end),
    ).fetchall()
    return [str(r[0]) for r in rows]


def build_dataset(
    conn: sqlite3.Connection,
    *,
    start_date: str,
    end_date: str,
    sample_frequency: int = 5,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build a labeled feature dataset for training.

    For every *sample_frequency*-th trading day between start and end,
    extracts features for the full universe and generates triple-barrier
    labels.

    Parameters
    ----------
    conn : sqlite3.Connection
    start_date, end_date : str
        Date range (YYYY-MM-DD) for the dataset.
    sample_frequency : int
        Sample every Nth trading day (default 5 = weekly).
    feature_columns : list[str] | None
        Which feature set to extract. Defaults to FEATURE_COLUMNS (slim).
        Pass FEATURE_COLUMNS_FULL for the full set.
    """
    trading_dates = _get_trading_dates(conn, start_date, end_date)
    if not trading_dates:
        LOG.warning("No trading dates found between %s and %s", start_date, end_date)
        return pd.DataFrame()

    # Sample dates
    sampled_dates = trading_dates[::sample_frequency]
    LOG.info(
        "Building dataset: %d sampled dates from %s to %s (of %d total)",
        len(sampled_dates), start_date, end_date, len(trading_dates),
    )

    all_features: list[pd.DataFrame] = []
    all_labels: list[pd.DataFrame] = []

    for date in sampled_dates:
        features = extract_features_for_universe(
            conn, as_of_date=date, strict=True,
            feature_columns=feature_columns,
        )
        if features.empty:
            continue
        features["entry_date"] = date
        features = features.reset_index()  # ticker becomes a column
        all_features.append(features)

    if not all_features:
        return pd.DataFrame()

    feat_df = pd.concat(all_features, ignore_index=True)

    # Symmetric barriers: both profit and stop at 2× daily vol.
    # Previously profit_mult=1.5 / stop_mult=2.0 created label imbalance
    # (profit target hit too easily → inflated positive rate → poor AUC).
    unique_dates = sorted(feat_df["entry_date"].unique())
    unique_tickers = sorted(feat_df["ticker"].unique())
    labels_df = generate_triple_barrier_labels(
        conn,
        entry_dates=unique_dates,
        tickers=unique_tickers,
        profit_mult=2.0,
        stop_mult=2.0,
        max_holding_days=10,
    )

    if labels_df.empty:
        LOG.warning("No labels generated")
        return pd.DataFrame()

    # Merge features + labels
    dataset = feat_df.merge(
        labels_df[["ticker", "entry_date", "label", "exit_reason", "return_pct", "days_held"]],
        on=["ticker", "entry_date"],
        how="inner",
    )

    # Target column is set later by _apply_target_mode() based on the
    # chosen target_mode.  Default placeholder so build_dataset() callers
    # that don't go through train_walk_forward() still get a target.
    dataset["target"] = (dataset["return_pct"] > 0).astype(int)

    LOG.info(
        "Dataset built: %d samples, %d tickers, %d dates, target balance: %.1f%% positive",
        len(dataset),
        dataset["ticker"].nunique(),
        dataset["entry_date"].nunique(),
        dataset["target"].mean() * 100,
    )

    return dataset


def _apply_target_mode(dataset: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Set the binary ``target`` column based on the chosen labeling strategy.

    Parameters
    ----------
    dataset : pd.DataFrame
        Must contain ``label``, ``return_pct`` columns from triple-barrier
        labeling.
    mode : str
        ``"return_sign"`` — target = 1 if return_pct > 0, else 0.
        ``"barrier"`` — original triple-barrier (label == +1 means profit
        target was hit before stop or time expiry).
        ``"rank"`` — cross-sectional rank within each entry_date; top
        quintile = 1, bottom quintile = 0, middle 60% dropped.
    """
    if mode == "return_sign":
        dataset["target"] = (dataset["return_pct"] > 0).astype(int)
    elif mode == "barrier":
        dataset["target"] = (dataset["label"] == 1).astype(int)
    elif mode == "rank":
        # Cross-sectional quintile ranking per entry date
        dataset["rank_pct"] = dataset.groupby("entry_date")["return_pct"].rank(
            pct=True,
        )
        # Top 20% = 1, bottom 20% = 0, middle dropped
        dataset["target"] = np.where(
            dataset["rank_pct"] >= 0.8, 1,
            np.where(dataset["rank_pct"] <= 0.2, 0, np.nan),
        )
        dataset = dataset.dropna(subset=["target"]).copy()
        dataset["target"] = dataset["target"].astype(int)
        dataset = dataset.drop(columns=["rank_pct"])
    else:
        raise ValueError(
            f"Unknown target_mode '{mode}'. "
            "Choose from: 'return_sign', 'barrier', 'rank'."
        )
    return dataset


def train_walk_forward(
    conn: sqlite3.Connection,
    *,
    start_date: str = "2025-01-01",
    end_date: str | None = None,
    train_days: int = 180,
    test_days: int = 60,
    step_days: int = 30,
    embargo_days: int = 15,  # Must be >= max_holding_days (10) + buffer
    feature_columns: list[str] | None = None,
    backfill_sentiment: bool = False,
    target_mode: str = "return_sign",
) -> dict[str, Any]:
    """Train LightGBM with walk-forward validation.

    Walks through time:
    1. Train on [fold_start, fold_start + train_days]
    2. Embargo gap of embargo_days
    3. Test on [fold_start + train_days + embargo_days, ... + test_days]
    4. Step forward by step_days
    5. Repeat

    The LAST fold's model is saved as the production model.

    Parameters
    ----------
    feature_columns : list[str] | None
        Which feature set to use. Defaults to FEATURE_COLUMNS (slim).
        Pass FEATURE_COLUMNS_FULL for the full 51-feature set.
    backfill_sentiment : bool
        If True, pre-warm the news_sentiment_cache for all tickers and
        sampled training dates before building the dataset.  Slow (one-time)
        but avoids per-ticker API calls during feature extraction.
    target_mode : str
        How to define the binary target.  One of:
        ``"return_sign"`` (default) — 1 if return_pct > 0, else 0.
        ``"barrier"`` — original triple-barrier (label == +1).
        ``"rank"`` — cross-sectional top/bottom quintile.

    Returns training report with per-fold metrics.
    """
    if end_date is None:
        end_date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    # Optionally backfill sentiment cache before training
    if backfill_sentiment:
        try:
            from trader_koo.ml.sentiment_cache import backfill_sentiment_cache

            # Get all active tickers from the universe
            ticker_rows = conn.execute(
                "SELECT DISTINCT ticker FROM price_daily "
                "WHERE date >= ? AND date <= ? "
                "AND ticker NOT LIKE '^%%'",
                (start_date, end_date),
            ).fetchall()
            all_tickers = sorted(str(r[0]) for r in ticker_rows)
            LOG.info(
                "Backfilling sentiment cache: %d tickers, %s to %s",
                len(all_tickers), start_date, end_date,
            )
            backfill_sentiment_cache(
                conn, all_tickers, start_date, end_date,
                sample_frequency=5,
            )
        except Exception as exc:
            LOG.warning("Sentiment backfill failed (non-fatal): %s", exc)

    # Bulk-prefetch FRED series for the full training date range.
    # These 3 series (DFF, T10Y2Y, BAMLH0A0HYM2) are never revised, so
    # fetching full history in 3 API calls replaces 100s of per-date calls.
    try:
        from trader_koo.ml.external_data import prefetch_fred_series_bulk

        prefetch_fred_series_bulk(
            ["DFF", "T10Y2Y", "BAMLH0A0HYM2"],
            start_date,
            end_date,
        )
    except Exception as exc:
        LOG.warning("FRED bulk prefetch failed (will fall back to per-date): %s", exc)

    LOG.info("Building full dataset from %s to %s", start_date, end_date)
    # Sample frequency must be >= max_holding_days (10) to avoid label overlap.
    # Previously frequency=5 with 10-day holding meant consecutive samples
    # shared 50% of their outcome window, leaking label information.
    use_features = feature_columns if feature_columns is not None else FEATURE_COLUMNS
    max_holding_days = 10
    dataset = build_dataset(
        conn, start_date=start_date, end_date=end_date,
        sample_frequency=max_holding_days,
        feature_columns=use_features,
    )
    if dataset.empty or len(dataset) < 100:
        return {
            "ok": False,
            "error": f"Insufficient data: {len(dataset)} samples (need ≥100)",
            "samples": len(dataset),
        }

    # Apply target labeling strategy before walk-forward splits
    dataset = _apply_target_mode(dataset, target_mode)
    if len(dataset) < 100:
        return {
            "ok": False,
            "error": (
                f"Insufficient data after target_mode='{target_mode}': "
                f"{len(dataset)} samples (need ≥100)"
            ),
            "samples": len(dataset),
        }

    dataset["entry_date_ts"] = pd.to_datetime(dataset["entry_date"])
    feature_cols = [c for c in use_features if c in dataset.columns]

    # Walk-forward folds
    all_dates = sorted(dataset["entry_date_ts"].unique())
    min_date = all_dates[0]
    max_date = all_dates[-1]

    folds: list[dict[str, Any]] = []
    oos_indices: list[int] = []  # indices of out-of-sample test rows
    oos_probs: list[float] = []  # OOS predicted probabilities
    fold_start = min_date
    best_model = None
    best_auc = 0.0

    while True:
        train_end = fold_start + pd.Timedelta(days=train_days)
        test_start = train_end + pd.Timedelta(days=embargo_days)
        test_end = test_start + pd.Timedelta(days=test_days)

        if test_start > max_date:
            break

        train_mask = (dataset["entry_date_ts"] >= fold_start) & (dataset["entry_date_ts"] <= train_end)
        test_mask = (dataset["entry_date_ts"] >= test_start) & (dataset["entry_date_ts"] <= min(test_end, max_date))

        X_train_full = dataset.loc[train_mask, feature_cols].copy()
        y_train_full = dataset.loc[train_mask, "target"].copy()
        X_test = dataset.loc[test_mask, feature_cols].copy()
        y_test = dataset.loc[test_mask, "target"].copy()

        if len(X_train_full) < 50 or len(X_test) < 10:
            fold_start += pd.Timedelta(days=step_days)
            continue

        # Split training into train (80%) + validation (20%) for early stopping.
        # Validation is the LAST 20% of training data (temporal, not random)
        # so we don't use the test set for model selection (that would be leakage).
        val_split = int(len(X_train_full) * 0.8)
        X_train = X_train_full.iloc[:val_split]
        y_train = y_train_full.iloc[:val_split]
        X_val = X_train_full.iloc[val_split:]
        y_val = y_train_full.iloc[val_split:]

        # Handle NaN — save training medians for consistent imputation at scoring
        train_medians = X_train.median()
        X_train = X_train.fillna(train_medians)
        X_val = X_val.fillna(train_medians)
        X_test = X_test.fillna(train_medians)

        # Train with early stopping on VALIDATION set (not test set — no leakage)
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        # Predict (out-of-sample)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Collect OOS predictions for meta-labeling (no in-sample contamination)
        oos_indices.extend(X_test.index.tolist())
        oos_probs.extend(y_prob.tolist())

        # Metrics
        try:
            auc = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            auc = 0.5
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        fold_result = {
            "fold": len(folds) + 1,
            "train_start": fold_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": min(test_end, max_date).strftime("%Y-%m-%d"),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "auc": round(auc, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "target_rate_train": round(float(y_train.mean()), 4),
            "target_rate_test": round(float(y_test.mean()), 4),
        }
        folds.append(fold_result)

        LOG.info(
            "Fold %d: train=%d test=%d AUC=%.3f Acc=%.3f Prec=%.3f",
            fold_result["fold"], len(X_train), len(X_test), auc, acc, prec,
        )

        if auc > best_auc:
            best_auc = auc
            best_model = model

        fold_start += pd.Timedelta(days=step_days)

    if not folds:
        return {"ok": False, "error": "No valid folds produced", "folds": []}

    # Save the last fold's model (most recent data) + training medians for imputation
    model_path = _save_model(model, feature_cols, folds, train_medians=train_medians)

    # Feature importance from the last model
    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # Aggregate metrics across folds
    avg_auc = round(np.mean([f["auc"] for f in folds]), 4)
    avg_acc = round(np.mean([f["accuracy"] for f in folds]), 4)
    avg_prec = round(np.mean([f["precision"] for f in folds]), 4)

    # Meta-labeling: train secondary model using ONLY out-of-sample predictions.
    # Previously this used the full dataset (train+test) which inflated meta-labels
    # because in-sample predictions are more accurate than OOS ones.
    meta_metrics: dict[str, Any] = {"ok": False, "note": "not_attempted"}
    try:
        from trader_koo.ml.meta_label import build_meta_labels_from_oos, train_meta_model

        meta_dataset = build_meta_labels_from_oos(
            dataset, oos_indices, oos_probs, feature_cols,
        )
        meta_model, meta_metrics = train_meta_model(meta_dataset, feature_cols)
        if meta_model is not None:
            # Save meta-model alongside primary
            meta_model_path = _model_dir() / "swing_meta_lgbm_latest.txt"
            meta_model.booster_.save_model(str(meta_model_path))
            meta_metrics["model_path"] = str(meta_model_path)
            LOG.info("Meta-model saved: %s", meta_model_path)
    except Exception as exc:
        LOG.warning("Meta-labeling failed (non-fatal): %s", exc)
        meta_metrics = {"ok": False, "error": str(exc)}

    return {
        "ok": True,
        "model_path": str(model_path),
        "target_mode": target_mode,
        "folds": folds,
        "fold_count": len(folds),
        "total_samples": len(dataset),
        "feature_columns": feature_cols,
        "feature_importance": sorted_importance,
        "aggregate_metrics": {
            "avg_auc": avg_auc,
            "avg_accuracy": avg_acc,
            "avg_precision": avg_prec,
            "best_auc": round(best_auc, 4),
        },
        "meta_labeling": meta_metrics,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def _save_model(
    model: lgb.LGBMClassifier,
    feature_cols: list[str],
    folds: list[dict[str, Any]],
    *,
    train_medians: pd.Series | None = None,
) -> Path:
    """Save model + metadata to disk."""
    model_dir = _model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = model_dir / f"swing_lgbm_{ts}.txt"
    meta_path = model_dir / f"swing_lgbm_{ts}_meta.json"
    latest_path = model_dir / "swing_lgbm_latest.txt"
    latest_meta = model_dir / "swing_lgbm_latest_meta.json"

    model.booster_.save_model(str(model_path))

    # Save training medians for consistent NaN imputation at scoring time
    medians_dict: dict[str, float] = {}
    if train_medians is not None:
        medians_dict = {
            k: float(v) for k, v in train_medians.to_dict().items()
            if pd.notna(v)
        }

    meta = {
        "model_type": "lightgbm",
        "task": "binary_classification",
        "target": "triple_barrier_profit",
        "feature_columns": feature_cols,
        "lgbm_params": LGBM_PARAMS,
        "folds": folds,
        "trained_at": ts,
        "train_medians": medians_dict,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Symlink-like: copy to "latest"
    model.booster_.save_model(str(latest_path))
    latest_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    LOG.info("Model saved: %s", model_path)
    return model_path
