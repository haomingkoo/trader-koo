"""Concept drift detection for model health monitoring.

Detects when the model's predictions degrade relative to actual outcomes.
Triggers alerts when accuracy drops below thresholds.
"""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)

# Thresholds
MIN_ACCURACY = 0.50     # Below random = model is harmful
WARN_ACCURACY = 0.52    # Below useful = model needs retraining
HEALTHY_ACCURACY = 0.54 # Acceptable performance


def check_model_drift(
    conn: sqlite3.Connection,
    *,
    window_days: int = 30,
) -> dict[str, Any]:
    """Check if the model's predictions are still accurate.

    Compares predicted win probability against actual outcomes
    from closed paper trades that have a `predicted_win_prob` value.

    Returns:
        {
            "status": "healthy" | "warning" | "degraded" | "insufficient_data",
            "accuracy": float | None,
            "sample_count": int,
            "window_days": int,
            "recommendation": str,
        }
    """
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=window_days)).strftime("%Y-%m-%d")

    try:
        # Check if paper_trades table has predicted_win_prob column
        has_col = conn.execute(
            "SELECT 1 FROM pragma_table_info('paper_trades') WHERE name='predicted_win_prob' LIMIT 1"
        ).fetchone()
        if not has_col:
            return {
                "status": "insufficient_data",
                "accuracy": None,
                "sample_count": 0,
                "window_days": window_days,
                "recommendation": "No ML predictions stored yet. Train and deploy model first.",
            }

        rows = conn.execute(
            """
            SELECT
                pnl_pct,
                predicted_win_prob
            FROM paper_trades
            WHERE status != 'open'
              AND pnl_pct IS NOT NULL
              AND predicted_win_prob IS NOT NULL
              AND entry_date >= ?
            """,
            (cutoff,),
        ).fetchall()

        if len(rows) < 10:
            return {
                "status": "insufficient_data",
                "accuracy": None,
                "sample_count": len(rows),
                "window_days": window_days,
                "recommendation": f"Only {len(rows)} trades with ML predictions. Need ≥10 for drift detection.",
            }

        # Compare: model predicted win (prob > 0.5) vs actual win (pnl > 0)
        correct = sum(
            1 for r in rows
            if (float(r[1]) > 0.5) == (float(r[0]) > 0)
        )
        accuracy = correct / len(rows)

        if accuracy < MIN_ACCURACY:
            status = "degraded"
            recommendation = (
                f"Model accuracy {accuracy:.1%} is below random ({MIN_ACCURACY:.0%}). "
                "URGENT: retrain or disable ML predictions."
            )
        elif accuracy < WARN_ACCURACY:
            status = "warning"
            recommendation = (
                f"Model accuracy {accuracy:.1%} is below useful threshold ({WARN_ACCURACY:.0%}). "
                "Consider retraining with recent data."
            )
        else:
            status = "healthy"
            recommendation = f"Model accuracy {accuracy:.1%} is acceptable."

        return {
            "status": status,
            "accuracy": round(accuracy, 4),
            "sample_count": len(rows),
            "window_days": window_days,
            "recommendation": recommendation,
            "correct_predictions": correct,
            "total_predictions": len(rows),
        }

    except Exception as exc:
        LOG.warning("Drift detection failed: %s", exc)
        return {
            "status": "error",
            "accuracy": None,
            "sample_count": 0,
            "window_days": window_days,
            "recommendation": f"Drift detection error: {exc}",
        }


def check_feature_drift(
    current_features: dict[str, float],
    historical_means: dict[str, float],
    historical_stds: dict[str, float],
    *,
    z_threshold: float = 3.0,
) -> list[dict[str, Any]]:
    """Check if current feature values are anomalous vs historical distribution.

    Returns list of features that are >z_threshold standard deviations from mean.
    """
    anomalies: list[dict[str, Any]] = []
    for feature, value in current_features.items():
        if feature not in historical_means or feature not in historical_stds:
            continue
        mean = historical_means[feature]
        std = historical_stds[feature]
        if std <= 0:
            continue
        z_score = abs(value - mean) / std
        if z_score > z_threshold:
            anomalies.append({
                "feature": feature,
                "current_value": round(value, 4),
                "historical_mean": round(mean, 4),
                "historical_std": round(std, 4),
                "z_score": round(z_score, 2),
            })
    return anomalies
