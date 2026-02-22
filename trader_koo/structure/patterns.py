from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PatternConfig:
    flag_lookback_bars: int = 30
    flag_pole_bars: int = 8
    min_pole_return: float = 0.04
    max_pullback_ratio: float = 0.75
    flag_parallel_tol_pct_per_bar: float = 0.003
    wedge_lookback_bars: int = 45
    wedge_end_gap_ratio: float = 0.78
    max_patterns: int = 5


def _empty_patterns() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "pattern",
            "start_date",
            "end_date",
            "status",
            "confidence",
            "notes",
            "x0_date",
            "x1_date",
            "y0",
            "y1",
            "y0b",
            "y1b",
        ]
    )


def _fit_line(y: np.ndarray) -> tuple[float, float] | None:
    if y.size < 2:
        return None
    x = np.arange(y.size, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None
    return float(slope), float(intercept)


def _build_flag(df: pd.DataFrame, bullish: bool, cfg: PatternConfig) -> dict | None:
    if len(df) < cfg.flag_pole_bars + 8:
        return None

    seg = df.tail(cfg.flag_lookback_bars).reset_index(drop=True)
    if len(seg) < cfg.flag_pole_bars + 8:
        return None

    pole = seg.iloc[: cfg.flag_pole_bars].copy()
    cons = seg.iloc[cfg.flag_pole_bars :].copy()
    if cons.empty:
        return None

    pole_start = float(pole["open"].iloc[0])
    pole_end = float(pole["close"].iloc[-1])
    pole_move = (pole_end - pole_start) / max(abs(pole_start), 1e-9)

    cons_high = cons["high"].to_numpy(dtype=float)
    cons_low = cons["low"].to_numpy(dtype=float)
    fit_hi = _fit_line(cons_high)
    fit_lo = _fit_line(cons_low)
    if fit_hi is None or fit_lo is None:
        return None
    s_hi, i_hi = fit_hi
    s_lo, i_lo = fit_lo

    last_close = float(cons["close"].iloc[-1])
    parallel = abs(s_hi - s_lo) / max(abs(last_close), 1e-9) <= cfg.flag_parallel_tol_pct_per_bar
    if not parallel:
        return None

    if bullish:
        if pole_move < cfg.min_pole_return:
            return None
        if not (s_hi < 0 and s_lo < 0):
            return None
        pullback = (pole_end - float(cons["low"].min())) / max(pole_end - pole_start, 1e-9)
        if pullback <= 0 or pullback > cfg.max_pullback_ratio:
            return None
        upper_last = s_hi * (len(cons) - 1) + i_hi
        status = "breakout" if last_close > upper_last else "forming"
        pattern = "bull_flag"
    else:
        if pole_move > -cfg.min_pole_return:
            return None
        if not (s_hi > 0 and s_lo > 0):
            return None
        pullback = (float(cons["high"].max()) - pole_end) / max(pole_start - pole_end, 1e-9)
        if pullback <= 0 or pullback > cfg.max_pullback_ratio:
            return None
        lower_last = s_lo * (len(cons) - 1) + i_lo
        status = "breakdown" if last_close < lower_last else "forming"
        pattern = "bear_flag"

    confidence = 0.45
    confidence += min(abs(pole_move), 0.14) * 2.4
    confidence += 0.12 if status in {"breakout", "breakdown"} else 0.0
    confidence = float(min(max(confidence, 0.0), 0.99))

    x0 = pd.Timestamp(cons["date"].iloc[0]).strftime("%Y-%m-%d")
    x1 = pd.Timestamp(cons["date"].iloc[-1]).strftime("%Y-%m-%d")
    n = len(cons) - 1
    return {
        "pattern": pattern,
        "start_date": x0,
        "end_date": x1,
        "status": status,
        "confidence": confidence,
        "notes": "Pole then parallel consolidation channel",
        "x0_date": x0,
        "x1_date": x1,
        "y0": float(i_hi),
        "y1": float(s_hi * n + i_hi),
        "y0b": float(i_lo),
        "y1b": float(s_lo * n + i_lo),
    }


def _build_wedge(df: pd.DataFrame, cfg: PatternConfig) -> dict | None:
    if len(df) < 20:
        return None
    seg = df.tail(cfg.wedge_lookback_bars).reset_index(drop=True)
    if len(seg) < 20:
        return None

    fit_hi = _fit_line(seg["high"].to_numpy(dtype=float))
    fit_lo = _fit_line(seg["low"].to_numpy(dtype=float))
    if fit_hi is None or fit_lo is None:
        return None
    s_hi, i_hi = fit_hi
    s_lo, i_lo = fit_lo

    start_gap = i_hi - i_lo
    end_gap = (s_hi * (len(seg) - 1) + i_hi) - (s_lo * (len(seg) - 1) + i_lo)
    if start_gap <= 0 or end_gap <= 0:
        return None

    converging = end_gap <= start_gap * cfg.wedge_end_gap_ratio and (s_hi - s_lo) < 0
    if not converging:
        return None

    both_up = s_hi > 0 and s_lo > 0
    both_down = s_hi < 0 and s_lo < 0
    if not (both_up or both_down):
        return None

    last_close = float(seg["close"].iloc[-1])
    upper_last = s_hi * (len(seg) - 1) + i_hi
    lower_last = s_lo * (len(seg) - 1) + i_lo

    if both_up:
        pattern = "rising_wedge"
        status = "breakdown" if last_close < lower_last else "forming"
    else:
        pattern = "falling_wedge"
        status = "breakout" if last_close > upper_last else "forming"

    conv_strength = 1.0 - (end_gap / max(start_gap, 1e-9))
    slope_penalty = min(abs((s_hi + s_lo) / 2.0) / max(abs(last_close), 1e-9), 0.05)
    confidence = float(min(max(0.40 + conv_strength * 0.45 - slope_penalty * 4.0, 0.0), 0.99))

    x0 = pd.Timestamp(seg["date"].iloc[0]).strftime("%Y-%m-%d")
    x1 = pd.Timestamp(seg["date"].iloc[-1]).strftime("%Y-%m-%d")
    n = len(seg) - 1
    return {
        "pattern": pattern,
        "start_date": x0,
        "end_date": x1,
        "status": status,
        "confidence": confidence,
        "notes": "Converging upper/lower boundaries",
        "x0_date": x0,
        "x1_date": x1,
        "y0": float(i_hi),
        "y1": float(s_hi * n + i_hi),
        "y0b": float(i_lo),
        "y1b": float(s_lo * n + i_lo),
    }


def detect_patterns(df: pd.DataFrame, cfg: PatternConfig) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_patterns()
    work = df.sort_values("date").reset_index(drop=True)
    rows: list[dict] = []

    for bullish in (True, False):
        r = _build_flag(work, bullish=bullish, cfg=cfg)
        if r:
            rows.append(r)
    w = _build_wedge(work, cfg=cfg)
    if w:
        rows.append(w)

    if not rows:
        return _empty_patterns()
    out = pd.DataFrame(rows).sort_values("confidence", ascending=False).head(cfg.max_patterns).reset_index(drop=True)
    return out
