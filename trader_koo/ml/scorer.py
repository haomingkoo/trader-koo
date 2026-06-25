"""Model scoring — load a trained model and score new setups.

Used by the paper trade decision pipeline to get ML-based predictions
for setup quality.  The model is loaded once and cached in memory.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from trader_koo.ml.features import (
    FEATURE_COLUMNS,
    extract_features_for_universe,
)

LOG = logging.getLogger(__name__)

_MODEL_DIRS = [
    Path("/data/models"),
    Path(__file__).resolve().parents[2] / "data" / "models",
]

# Cached model
_cached_model: lgb.Booster | None = None
_cached_meta: dict[str, Any] | None = None


def _target_mode(meta: dict[str, Any] | None) -> str:
    """Return the trained target mode, or ``unknown`` for legacy metadata."""
    if not meta:
        return "unknown"
    mode = meta.get("target_mode")
    if mode in {"return_sign", "barrier", "rank"}:
        return str(mode)
    return "unknown"


def _prediction_label(target_mode: str) -> str:
    """Short label for what the probability means."""
    if target_mode == "barrier":
        return "target_hit_probability"
    if target_mode == "rank":
        return "top_quintile_probability"
    if target_mode == "return_sign":
        return "positive_return_probability"
    return "model_probability"


def _signal_from_probability(prob: float, target_mode: str) -> str:
    """Convert probability to a non-misleading coarse signal label.

    Barrier models only predict whether a long setup hits its profit target.
    A low probability is not evidence for a short, so keep it neutral.
    """
    if prob > 0.55:
        return "bullish"
    if prob < 0.45 and target_mode in {"return_sign", "rank"}:
        return "bearish"
    return "neutral"


def _as_probability(values: Any) -> np.ndarray:
    """Normalize LightGBM outputs to probabilities.

    ``Booster.predict`` for a binary objective returns probabilities by default.
    Older code applied a sigmoid again, which distorted calibration by pulling
    bearish probabilities above 0.5.  Keep in-range outputs as-is and only apply
    sigmoid if a caller/model explicitly returns raw margins outside [0, 1].
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
        return np.clip(arr, 0.0, 1.0)
    return 1.0 / (1.0 + np.exp(-arr))


def _find_latest_model() -> tuple[Path | None, Path | None]:
    """Find the latest model files on disk."""
    for model_dir in _MODEL_DIRS:
        model_path = model_dir / "swing_lgbm_latest.txt"
        meta_path = model_dir / "swing_lgbm_latest_meta.json"
        if model_path.exists():
            return model_path, meta_path if meta_path.exists() else None
    return None, None


def load_model(*, force_reload: bool = False) -> tuple[lgb.Booster | None, dict[str, Any] | None]:
    """Load the latest trained model.  Returns (model, metadata) or (None, None)."""
    global _cached_model, _cached_meta

    if not force_reload and _cached_model is not None:
        return _cached_model, _cached_meta

    model_path, meta_path = _find_latest_model()
    if model_path is None:
        LOG.info("No trained model found in %s", [str(d) for d in _MODEL_DIRS])
        return None, None

    try:
        model = lgb.Booster(model_file=str(model_path))
        meta = None
        if meta_path and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        _cached_model = model
        _cached_meta = meta
        LOG.info("Loaded model from %s", model_path)
        return model, meta
    except Exception as exc:
        LOG.warning("Failed to load model from %s: %s", model_path, exc)
        return None, None


def score_universe(
    conn: sqlite3.Connection,
    *,
    as_of_date: str,
    tickers: list[str] | None = None,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """Score all tickers and return the top N by model probability.

    ``predicted_win_prob`` is kept as the storage/API field for compatibility.
    Use ``prediction_label`` to interpret what the probability means.
    """
    model, meta = load_model()
    if model is None:
        return []

    target_mode = _target_mode(meta)
    prediction_label = _prediction_label(target_mode)
    feature_cols = meta.get("feature_columns", FEATURE_COLUMNS) if meta else FEATURE_COLUMNS
    features = extract_features_for_universe(conn, as_of_date=as_of_date, tickers=tickers)
    if features.empty:
        return []

    # Align columns with what the model expects
    X = features.reindex(columns=feature_cols)
    # Use training medians for imputation (prevents train/serve skew)
    train_medians = meta.get("train_medians", {}) if meta else {}
    if train_medians:
        X = X.fillna({k: v for k, v in train_medians.items() if k in X.columns})
    X = X.fillna(0.0)  # fallback for any remaining NaN

    try:
        probs = _as_probability(model.predict(X.values))
    except Exception as exc:
        LOG.warning("Model prediction failed: %s", exc)
        return []

    results = []
    for i, ticker in enumerate(features.index):
        results.append({
            "ticker": str(ticker),
            "predicted_win_prob": round(float(probs[i]), 4),
            "target_mode": target_mode,
            "prediction_label": prediction_label,
            "as_of_date": as_of_date,
        })

    # Sort by probability descending, return top N
    results.sort(key=lambda x: x["predicted_win_prob"], reverse=True)
    return results[:top_n]


def score_single_ticker(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    as_of_date: str,
) -> dict[str, Any]:
    """Score a single ticker.  Returns prediction + confidence."""
    model, meta = load_model()
    if model is None:
        return {
            "ticker": ticker,
            "model_available": False,
            "predicted_win_prob": None,
            "target_mode": None,
            "prediction_label": None,
            "confidence": None,
        }

    target_mode = _target_mode(meta)
    prediction_label = _prediction_label(target_mode)
    feature_cols = meta.get("feature_columns", FEATURE_COLUMNS) if meta else FEATURE_COLUMNS
    features = extract_features_for_universe(
        conn,
        as_of_date=as_of_date,
        tickers=[ticker],
    )

    if features.empty or ticker not in features.index:
        return {
            "ticker": ticker,
            "model_available": True,
            "predicted_win_prob": None,
            "target_mode": target_mode,
            "prediction_label": prediction_label,
            "confidence": None,
            "note": "Insufficient data for feature extraction",
        }

    X = features.loc[[ticker]].reindex(columns=feature_cols)
    # Use training medians for consistent imputation (same as training time).
    # Previously used fillna(0.0) which caused train/serve skew.
    train_medians = meta.get("train_medians", {}) if meta else {}
    if train_medians:
        X = X.fillna({k: v for k, v in train_medians.items() if k in X.columns})
    X = X.fillna(0.0)  # fallback for any remaining NaN

    try:
        prob = float(_as_probability(model.predict(X.values))[0])
    except Exception as exc:
        return {
            "ticker": ticker,
            "model_available": True,
            "predicted_win_prob": None,
            "target_mode": target_mode,
            "prediction_label": prediction_label,
            "confidence": None,
            "note": f"Prediction error: {exc}",
        }

    # Confidence: distance from 0.5 (random)
    confidence = abs(prob - 0.5) * 2  # 0 = no confidence, 1 = max confidence

    return {
        "ticker": ticker,
        "model_available": True,
        "predicted_win_prob": round(prob, 4),
        "target_mode": target_mode,
        "prediction_label": prediction_label,
        "confidence": round(confidence, 4),
        "signal": _signal_from_probability(prob, target_mode),
    }


def model_status() -> dict[str, Any]:
    """Return model status for the API / UI."""
    model, meta = load_model()
    if model is None:
        return {
            "loaded": False,
            "model_path": None,
            "trained_at": None,
            "feature_count": 0,
            "fold_count": 0,
        }

    return {
        "loaded": True,
        "trained_at": meta.get("trained_at") if meta else None,
        "target_mode": _target_mode(meta),
        "prediction_label": _prediction_label(_target_mode(meta)),
        "target": meta.get("target") if meta else None,
        "feature_columns": meta.get("feature_columns", []) if meta else [],
        "feature_count": len(meta.get("feature_columns", [])) if meta else 0,
        "fold_count": len(meta.get("folds", [])) if meta else 0,
        "model_type": meta.get("model_type", "lightgbm") if meta else "lightgbm",
        "aggregate_metrics": meta.get("folds", [{}])[-1] if meta and meta.get("folds") else {},
    }
