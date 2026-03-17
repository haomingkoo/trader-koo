"""Hidden Markov Model regime detection.

Trains a 3-state Gaussian HMM on daily price features to classify
market regimes: Low Vol (risk-on), Normal (transitional), High Vol (risk-off).
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger(__name__)

REGIME_LABELS = {0: "low_vol", 1: "normal", 2: "high_vol"}
REGIME_COLORS = {"low_vol": "#38d39f", "normal": "#f8c24e", "high_vol": "#ff6b6b"}

# Simple per-ticker model cache: {ticker: (model, scaler, state_order, ts)}
_MODEL_CACHE: dict[str, tuple[GaussianHMM, StandardScaler, np.ndarray, float]] = {}
_CACHE_TTL_SEC = 3600  # 1 hour


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract HMM features from OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: date, open, high, low, close, volume.
        Sorted by date ascending.

    Returns
    -------
    pd.DataFrame
        Feature columns: log_return, realized_vol_20d, return_zscore,
        volume_ratio, high_low_range.  Rows with NaN are dropped.
    """
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")

    log_return = np.log(close / close.shift(1))
    rolling_std = log_return.rolling(window=20).std()
    realized_vol_20d = rolling_std * np.sqrt(252)
    return_zscore = log_return / rolling_std.replace(0, np.nan)
    avg_volume_20 = volume.rolling(window=20).mean()
    volume_ratio = volume / avg_volume_20.replace(0, np.nan)
    high_low_range = (high - low) / close.replace(0, np.nan)

    features = pd.DataFrame(
        {
            "log_return": log_return,
            "realized_vol_20d": realized_vol_20d,
            "return_zscore": return_zscore,
            "volume_ratio": volume_ratio,
            "high_low_range": high_low_range,
        },
        index=df.index,
    )
    return features.dropna()


def _sort_states_by_volatility(
    model: GaussianHMM,
    scaler: StandardScaler,
    vol_feature_idx: int = 1,
) -> np.ndarray:
    """Return an index array that sorts HMM states by mean realized volatility.

    State 0 = lowest mean vol, State 2 = highest.
    """
    # Means are in scaled space; we only need relative ordering
    means = model.means_[:, vol_feature_idx]
    return np.argsort(means)


def fit_hmm(
    features: np.ndarray,
    n_states: int = 3,
    n_iter: int = 100,
) -> tuple[GaussianHMM, StandardScaler, np.ndarray]:
    """Fit a Gaussian HMM on scaled features.

    Returns
    -------
    tuple
        (fitted_model, scaler, state_order)
        state_order maps internal states to sorted-by-vol states.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=42,
        tol=0.01,
    )
    model.fit(scaled)
    state_order = _sort_states_by_volatility(model, scaler)
    return model, scaler, state_order


def predict_regimes(
    df: pd.DataFrame,
    lookback_days: int = 504,
    ticker: str = "",
) -> dict[str, Any] | None:
    """Run HMM regime detection on a price DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns: date, open, high, low, close, volume.
        Sorted by date ascending.
    lookback_days : int
        Number of trading days for HMM training window (default 504 = ~2 years).
    ticker : str
        Used for cache key.

    Returns
    -------
    dict or None
        On success: {regimes, current_state, current_probs,
                      transition_matrix, days_in_current}.
        On failure: None (logged, never silent).
    """
    t0 = time.monotonic()

    if df is None or df.empty:
        LOG.warning("HMM regime: empty DataFrame provided")
        return None

    if len(df) < 60:
        LOG.warning(
            "HMM regime: insufficient data (%d rows, need >= 60)", len(df)
        )
        return None

    # Extract features on full range
    features_df = extract_features(df)
    if features_df.empty or len(features_df) < 40:
        LOG.warning(
            "HMM regime: insufficient valid features (%d rows after NaN drop)",
            len(features_df),
        )
        return None

    feature_cols = [
        "log_return",
        "realized_vol_20d",
        "return_zscore",
        "volume_ratio",
        "high_low_range",
    ]
    feature_matrix = features_df[feature_cols].values

    # Use last lookback_days for training
    train_end = len(feature_matrix)
    train_start = max(0, train_end - lookback_days)
    train_data = feature_matrix[train_start:train_end]

    if len(train_data) < 40:
        LOG.warning(
            "HMM regime: training window too small (%d rows)", len(train_data)
        )
        return None

    # Check cache
    cache_key = ticker.upper().strip() or "__default__"
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        model, scaler, state_order, cached_ts = cached
        if (time.monotonic() - cached_ts) < _CACHE_TTL_SEC:
            LOG.debug("HMM regime: using cached model for %s", cache_key)
        else:
            cached = None

    if cached is None:
        model, scaler, state_order = fit_hmm(train_data, n_states=3)
        _MODEL_CACHE[cache_key] = (model, scaler, state_order, time.monotonic())
        LOG.info(
            "HMM regime: fitted new model for %s (%d training samples)",
            cache_key,
            len(train_data),
        )

    # Predict on full feature range
    scaled_all = scaler.transform(feature_matrix)
    raw_states = model.predict(scaled_all)
    raw_probs = model.predict_proba(scaled_all)

    # Build reverse mapping: internal_state -> sorted_state
    # state_order[i] = internal state that should be mapped to sorted position i
    # We need: for each internal state, what sorted position does it map to?
    inv_order = np.empty_like(state_order)
    for sorted_pos, internal_state in enumerate(state_order):
        inv_order[internal_state] = sorted_pos

    sorted_states = inv_order[raw_states]
    # Reorder probability columns: column j = probability of sorted state j
    sorted_probs = raw_probs[:, state_order]

    # Build transition matrix in sorted state space
    raw_transmat = model.transmat_
    sorted_transmat = raw_transmat[np.ix_(state_order, state_order)]

    # Align with original DataFrame dates
    dates_aligned = df.loc[features_df.index, "date"]
    dates_list = pd.to_datetime(dates_aligned, errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )

    regimes: list[dict[str, Any]] = []
    for i in range(len(features_df)):
        state_int = int(sorted_states[i])
        label = REGIME_LABELS.get(state_int, "normal")
        color = REGIME_COLORS.get(label, "#f8c24e")
        regimes.append(
            {
                "date": dates_list.iloc[i],
                "state": state_int,
                "label": label,
                "color": color,
                "prob_low": round(float(sorted_probs[i, 0]), 4),
                "prob_normal": round(float(sorted_probs[i, 1]), 4),
                "prob_high": round(float(sorted_probs[i, 2]), 4),
            }
        )

    # Current state info
    current_state_int = int(sorted_states[-1])
    current_label = REGIME_LABELS.get(current_state_int, "normal")
    current_probs = {
        "low_vol": round(float(sorted_probs[-1, 0]), 4),
        "normal": round(float(sorted_probs[-1, 1]), 4),
        "high_vol": round(float(sorted_probs[-1, 2]), 4),
    }

    # Days in current state
    days_in_current = 1
    for i in range(len(sorted_states) - 2, -1, -1):
        if int(sorted_states[i]) == current_state_int:
            days_in_current += 1
        else:
            break

    elapsed_ms = (time.monotonic() - t0) * 1000
    LOG.info(
        "HMM regime: %s -> %s (%.0f%% confidence, %d consecutive days, %.0fms)",
        cache_key,
        current_label,
        max(current_probs.values()) * 100,
        days_in_current,
        elapsed_ms,
    )

    return {
        "regimes": regimes,
        "current_state": current_label,
        "current_probs": current_probs,
        "transition_matrix": [
            [round(float(v), 4) for v in row] for row in sorted_transmat
        ],
        "days_in_current": days_in_current,
    }
