from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    atr_length: int = 14
    ma_windows: tuple[int, ...] = field(default_factory=lambda: (20, 50, 100, 200))


def add_basic_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.rolling(cfg.atr_length, min_periods=cfg.atr_length).mean()
    out["atr_pct"] = out["atr"] / out["close"].replace(0, np.nan)
    out["ret_1d"] = out["close"].pct_change(1)
    for w in cfg.ma_windows:
        out[f"ma{w}"] = out["close"].rolling(w, min_periods=w).mean()
    return out


def compute_pivots(df: pd.DataFrame, left: int, right: int) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    low = out["low"].to_numpy(dtype=float)
    high = out["high"].to_numpy(dtype=float)
    pl = np.zeros(n, dtype=bool)
    ph = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        lo_window = low[i - left : i + right + 1]
        hi_window = high[i - left : i + right + 1]
        if np.isfinite(low[i]) and low[i] == np.nanmin(lo_window):
            pl[i] = True
        if np.isfinite(high[i]) and high[i] == np.nanmax(hi_window):
            ph[i] = True
    out["pivot_low"] = pl
    out["pivot_high"] = ph
    return out
