"""SHAP feature importance analysis for model interpretability.

Explains WHY the model makes each prediction — not just feature
importance from training, but per-prediction attribution.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def compute_shap_summary(
    model: Any,
    X: pd.DataFrame,
    *,
    max_samples: int = 500,
) -> dict[str, Any]:
    """Compute SHAP values and return a summary for the API/UI.

    Returns:
        {
            "feature_importance": {feature: mean_abs_shap, ...},
            "top_positive": [{feature, shap_value}, ...],
            "top_negative": [{feature, shap_value}, ...],
            "sample_count": int,
        }
    """
    try:
        import shap
    except ImportError:
        return {"error": "shap not installed", "feature_importance": {}}

    # Subsample for speed
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X.copy()

    X_sample = X_sample.fillna(X_sample.median())

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # For binary classification, shap_values might be a list [class0, class1]
        if isinstance(shap_values, list) and len(shap_values) == 2:
            sv = shap_values[1]  # class 1 (positive/profitable)
        else:
            sv = shap_values

        feature_names = list(X_sample.columns)
        mean_abs_shap = np.abs(sv).mean(axis=0)

        # Sort by importance
        importance = dict(sorted(
            zip(feature_names, mean_abs_shap.tolist()),
            key=lambda x: x[1],
            reverse=True,
        ))

        # Mean SHAP (signed — shows direction of effect)
        mean_shap = sv.mean(axis=0)
        top_positive = [
            {"feature": f, "shap_value": round(float(v), 6)}
            for f, v in sorted(zip(feature_names, mean_shap.tolist()), key=lambda x: x[1], reverse=True)[:5]
            if v > 0
        ]
        top_negative = [
            {"feature": f, "shap_value": round(float(v), 6)}
            for f, v in sorted(zip(feature_names, mean_shap.tolist()), key=lambda x: x[1])[:5]
            if v < 0
        ]

        return {
            "feature_importance": {k: round(float(v), 6) for k, v in importance.items()},
            "top_positive_drivers": top_positive,
            "top_negative_drivers": top_negative,
            "sample_count": len(X_sample),
            "base_value": round(float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value), 6),
        }

    except Exception as exc:
        LOG.warning("SHAP computation failed: %s", exc)
        return {"error": str(exc), "feature_importance": {}}


def explain_single_prediction(
    model: Any,
    X_single: pd.DataFrame,
    X_background: pd.DataFrame,
    *,
    max_background: int = 100,
) -> dict[str, Any]:
    """Explain a single prediction — why did the model say buy/skip for this ticker?"""
    try:
        import shap
    except ImportError:
        return {"error": "shap not installed"}

    bg = X_background.sample(n=min(max_background, len(X_background)), random_state=42)
    bg = bg.fillna(bg.median())
    X_single = X_single.fillna(0.0)

    try:
        explainer = shap.TreeExplainer(model, bg)
        sv = explainer.shap_values(X_single)

        if isinstance(sv, list) and len(sv) == 2:
            sv = sv[1]

        features = list(X_single.columns)
        shap_dict = dict(zip(features, sv[0].tolist()))

        # Sort by absolute contribution
        sorted_contributions = sorted(
            shap_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return {
            "contributions": [
                {"feature": f, "value": round(float(X_single[f].iloc[0]), 4), "shap": round(float(s), 6)}
                for f, s in sorted_contributions[:10]
            ],
            "prediction_shift": round(float(sum(sv[0])), 6),
        }
    except Exception as exc:
        return {"error": str(exc)}
