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

from trader_koo.ml.features import FEATURE_COLUMNS, extract_features_for_universe

LOG = logging.getLogger(__name__)

_MODEL_DIRS = [
    Path("/data/models"),
    Path(__file__).resolve().parents[2] / "data" / "models",
]

# Cached model
_cached_model: lgb.Booster | None = None
_cached_meta: dict[str, Any] | None = None


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
    """Score all tickers and return the top N by predicted win probability.

    Returns a list of dicts with ticker, predicted_win_prob, and features.
    """
    model, meta = load_model()
    if model is None:
        return []

    feature_cols = meta.get("feature_columns", FEATURE_COLUMNS) if meta else FEATURE_COLUMNS
    features = extract_features_for_universe(conn, as_of_date=as_of_date, tickers=tickers)
    if features.empty:
        return []

    # Align columns with what the model expects
    X = features.reindex(columns=feature_cols)
    X = X.fillna(X.median())

    try:
        # Booster.predict() returns raw margins, need to convert to probabilities
        raw = model.predict(X.values)
        # Sigmoid to convert log-odds to probability
        probs = 1.0 / (1.0 + np.exp(-raw))
    except Exception as exc:
        LOG.warning("Model prediction failed: %s", exc)
        return []

    results = []
    for i, ticker in enumerate(features.index):
        results.append({
            "ticker": str(ticker),
            "predicted_win_prob": round(float(probs[i]), 4),
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
            "confidence": None,
        }

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
            "confidence": None,
            "note": "Insufficient data for feature extraction",
        }

    X = features.loc[[ticker]].reindex(columns=feature_cols)
    X = X.fillna(X.median())  # consistent with score_universe

    try:
        raw = float(model.predict(X.values)[0])
        prob = 1.0 / (1.0 + np.exp(-raw))  # sigmoid for Booster
    except Exception as exc:
        return {
            "ticker": ticker,
            "model_available": True,
            "predicted_win_prob": None,
            "confidence": None,
            "note": f"Prediction error: {exc}",
        }

    # Confidence: distance from 0.5 (random)
    confidence = abs(prob - 0.5) * 2  # 0 = no confidence, 1 = max confidence

    return {
        "ticker": ticker,
        "model_available": True,
        "predicted_win_prob": round(prob, 4),
        "confidence": round(confidence, 4),
        "signal": "bullish" if prob > 0.55 else ("bearish" if prob < 0.45 else "neutral"),
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
        "feature_columns": meta.get("feature_columns", []) if meta else [],
        "feature_count": len(meta.get("feature_columns", [])) if meta else 0,
        "fold_count": len(meta.get("folds", [])) if meta else 0,
        "model_type": meta.get("model_type", "lightgbm") if meta else "lightgbm",
        "aggregate_metrics": meta.get("folds", [{}])[-1] if meta and meta.get("folds") else {},
    }
