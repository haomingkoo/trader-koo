"""Meta-labeling: secondary model that decides WHEN to bet.

The primary model predicts direction (up/down).
The meta-labeling model predicts whether to ACT on that prediction.

This is the single biggest improvement in financial ML (Lopez de Prado).
Instead of trying to improve directional accuracy, we improve PRECISION
by filtering out low-confidence primary signals.

Pipeline:
1. Primary model predicts direction → "long" or "short" signal
2. Meta-label model predicts: P(primary signal is correct)
3. If P > threshold → trade. If P < threshold → skip.
4. Size position proportional to P (optional).
"""
from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

LOG = logging.getLogger(__name__)

# Meta-labeling uses the same LightGBM but with different tuning:
# higher regularization because we want to avoid false positives
META_LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 15,        # simpler than primary (31)
    "max_depth": 3,          # shallower to prevent overfitting
    "learning_rate": 0.03,   # slower learning
    "n_estimators": 150,
    "min_child_samples": 30, # more conservative
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,        # stronger L1
    "reg_lambda": 2.0,       # stronger L2
    "random_state": 42,
    "verbose": -1,
    "importance_type": "gain",
    "is_unbalance": True,    # handle class imbalance
}


def build_meta_labels(
    dataset: pd.DataFrame,
    primary_model: lgb.LGBMClassifier,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Generate meta-labels from a primary model's predictions.

    For each sample in the dataset:
    1. Get primary model's predicted probability
    2. Get primary model's predicted class (0/1)
    3. Meta-label = 1 if primary prediction was CORRECT, 0 if WRONG

    This creates the training data for the meta-labeling model.
    """
    X = dataset[feature_cols].fillna(dataset[feature_cols].median())
    primary_probs = primary_model.predict_proba(X)[:, 1]
    primary_preds = (primary_probs >= 0.5).astype(int)
    actual = dataset["target"].values

    # Meta-label: was the primary model correct?
    meta_labels = (primary_preds == actual).astype(int)

    result = dataset.copy()
    result["primary_prob"] = primary_probs
    result["primary_pred"] = primary_preds
    result["meta_label"] = meta_labels

    # Add primary model confidence as a feature for the meta model
    result["primary_confidence"] = np.abs(primary_probs - 0.5) * 2  # 0=uncertain, 1=confident

    LOG.info(
        "Meta-labels: %d samples, primary accuracy=%.3f, meta_label_rate=%.3f",
        len(result),
        float(accuracy_score(actual, primary_preds)),
        float(meta_labels.mean()),
    )
    return result


def train_meta_model(
    dataset: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[lgb.LGBMClassifier, dict[str, Any]]:
    """Train a meta-labeling model.

    Uses the original features PLUS primary model's prediction confidence
    to decide whether to act on the primary signal.

    Returns (model, metrics_dict).
    """
    meta_features = feature_cols + ["primary_confidence", "primary_prob"]
    available = [c for c in meta_features if c in dataset.columns]

    X = dataset[available].fillna(dataset[available].median())
    y = dataset["meta_label"]

    # Simple train/test split (the outer walk-forward handles temporal validity)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(X_train) < 50 or len(X_test) < 20:
        LOG.warning("Insufficient data for meta-labeling: train=%d test=%d", len(X_train), len(X_test))
        return None, {"ok": False, "error": "insufficient_data"}

    model = lgb.LGBMClassifier(**META_LGBM_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        auc = 0.5

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))

    # Simulate: what happens if we only trade when meta-model says "go"?
    # Filter test set to where meta says "trade" (pred=1)
    trade_mask = y_pred == 1
    if trade_mask.sum() > 0:
        actual_test = dataset["target"].iloc[split_idx:].values
        filtered_accuracy = float(actual_test[trade_mask].mean())
        trade_rate = float(trade_mask.mean())
    else:
        filtered_accuracy = 0.0
        trade_rate = 0.0

    metrics = {
        "ok": True,
        "meta_auc": round(auc, 4),
        "meta_accuracy": round(acc, 4),
        "meta_precision": round(prec, 4),
        "trade_rate": round(trade_rate, 4),
        "filtered_accuracy": round(filtered_accuracy, 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "meta_features": available,
    }

    LOG.info(
        "Meta-model: AUC=%.3f acc=%.3f prec=%.3f trade_rate=%.1f%% filtered_acc=%.3f",
        auc, acc, prec, trade_rate * 100, filtered_accuracy,
    )

    return model, metrics


def apply_meta_filter(
    primary_probs: np.ndarray,
    meta_model: lgb.LGBMClassifier,
    features: pd.DataFrame,
    feature_cols: list[str],
    *,
    min_meta_prob: float = 0.5,
) -> np.ndarray:
    """Apply meta-model to filter primary predictions.

    Returns adjusted probabilities: if meta-model says "don't trade",
    probability is pushed toward 0.5 (uncertain).
    """
    # Build feature matrix with primary model outputs FIRST, then filter to available
    X = features[feature_cols].copy() if all(c in features.columns for c in feature_cols) else features.copy()
    X["primary_prob"] = primary_probs
    X["primary_confidence"] = np.abs(primary_probs - 0.5) * 2
    X = X.fillna(0.0)

    # Use all columns the meta-model was trained on
    meta_features = feature_cols + ["primary_confidence", "primary_prob"]
    available = [c for c in meta_features if c in X.columns]

    meta_probs = meta_model.predict_proba(X[available])[:, 1]

    # If meta says "don't trade" (low prob), push primary toward 0.5
    adjusted = np.where(
        meta_probs >= min_meta_prob,
        primary_probs,  # keep original signal
        0.5,            # neutralize to "no trade"
    )

    filtered = int((meta_probs >= min_meta_prob).sum())
    LOG.info(
        "Meta-filter: %d/%d signals pass (%.1f%%)",
        filtered, len(primary_probs), filtered / len(primary_probs) * 100,
    )
    return adjusted
