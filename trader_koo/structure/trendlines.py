from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TrendlineConfig:
    lookback_bars: int = 252
    min_points: int = 3
    max_lines_per_side: int = 2
    touch_tol_atr: float = 0.70
    max_slope_pct_per_day: float = 0.03
    recency_half_life_days: int = 45


def _empty_lines() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "type",
            "x0_date",
            "x1_date",
            "y0",
            "y1",
            "slope",
            "intercept",
            "touch_count",
            "last_touch_date",
            "score",
        ]
    )


def _fit_line(points: pd.DataFrame, origin: pd.Timestamp) -> tuple[float, float] | None:
    x = (pd.to_datetime(points["date"]) - origin).dt.days.to_numpy(dtype=float)
    y = points["price"].to_numpy(dtype=float)
    if len(x) < 2 or np.unique(x).size < 2:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None
    return float(slope), float(intercept)


def _score_line(
    *,
    points_all: pd.DataFrame,
    slope: float,
    intercept: float,
    origin: pd.Timestamp,
    tol: float,
    last_close: float,
    cfg: TrendlineConfig,
) -> tuple[int, float, pd.Timestamp | None]:
    x = (pd.to_datetime(points_all["date"]) - origin).dt.days.to_numpy(dtype=float)
    y = points_all["price"].to_numpy(dtype=float)
    pred = slope * x + intercept
    err = np.abs(y - pred)
    touch_mask = err <= tol
    touch_count = int(np.sum(touch_mask))
    if touch_count == 0:
        return 0, float("-inf"), None

    touch_dates = pd.to_datetime(points_all.loc[touch_mask, "date"])
    last_touch = touch_dates.max()
    age_days = max(int((pd.to_datetime(points_all["date"]).max() - last_touch).days), 0)
    recency_score = float(0.5 ** (age_days / max(cfg.recency_half_life_days, 1)))
    slope_pct = abs(slope) / max(last_close, 1e-6)
    if slope_pct > cfg.max_slope_pct_per_day:
        return touch_count, float("-inf"), last_touch
    score = float(touch_count + 1.5 * recency_score - 10.0 * slope_pct)
    return touch_count, score, last_touch


def _build_side_lines(
    piv: pd.DataFrame,
    *,
    side: str,
    pivot_col: str,
    price_col: str,
    cfg: TrendlineConfig,
    tol: float,
    last_close: float,
) -> pd.DataFrame:
    points = piv.loc[piv[pivot_col], ["date", price_col]].rename(columns={price_col: "price"}).copy()
    if points.empty or len(points) < cfg.min_points:
        return _empty_lines()

    points = points.tail(cfg.lookback_bars).reset_index(drop=True)
    origin = pd.to_datetime(piv["date"]).min()
    x0_date = pd.Timestamp(piv["date"].min())
    x1_date = pd.Timestamp(piv["date"].max())
    x0 = float((x0_date - origin).days)
    x1 = float((x1_date - origin).days)

    candidates: list[dict] = []
    seen: set[tuple[float, float]] = set()
    for start in range(0, len(points) - cfg.min_points + 1):
        subset = points.iloc[start:].copy()
        fit = _fit_line(subset, origin)
        if fit is None:
            continue
        slope, intercept = fit
        key = (round(slope, 6), round(intercept, 3))
        if key in seen:
            continue
        seen.add(key)
        touch_count, score, last_touch = _score_line(
            points_all=points,
            slope=slope,
            intercept=intercept,
            origin=origin,
            tol=tol,
            last_close=last_close,
            cfg=cfg,
        )
        if not np.isfinite(score):
            continue
        candidates.append(
            {
                "type": side,
                "x0_date": x0_date.strftime("%Y-%m-%d"),
                "x1_date": x1_date.strftime("%Y-%m-%d"),
                "y0": float(slope * x0 + intercept),
                "y1": float(slope * x1 + intercept),
                "slope": float(slope),
                "intercept": float(intercept),
                "touch_count": touch_count,
                "last_touch_date": pd.Timestamp(last_touch).strftime("%Y-%m-%d") if last_touch is not None else None,
                "score": float(score),
            }
        )

    if not candidates:
        return _empty_lines()
    out = pd.DataFrame(candidates).sort_values(["score", "touch_count"], ascending=[False, False])
    return out.head(cfg.max_lines_per_side).reset_index(drop=True)


def detect_trendlines(piv: pd.DataFrame, last_close: float, cfg: TrendlineConfig) -> pd.DataFrame:
    if piv.empty:
        return _empty_lines()
    atr_med = float(np.nanmedian(piv["atr"].to_numpy())) if "atr" in piv.columns else np.nan
    if not np.isfinite(atr_med) or atr_med <= 0:
        atr_med = float(np.nanmedian((piv["high"] - piv["low"]).to_numpy()))
    if not np.isfinite(atr_med) or atr_med <= 0:
        atr_med = 1.0
    tol = max(0.01, cfg.touch_tol_atr * atr_med)

    sup = _build_side_lines(
        piv,
        side="support_line",
        pivot_col="pivot_low",
        price_col="low",
        cfg=cfg,
        tol=tol,
        last_close=last_close,
    )
    res = _build_side_lines(
        piv,
        side="resistance_line",
        pivot_col="pivot_high",
        price_col="high",
        cfg=cfg,
        tol=tol,
        last_close=last_close,
    )
    valid = [x for x in [sup, res] if not x.empty]
    if not valid:
        return _empty_lines()
    return pd.concat(valid, axis=0, ignore_index=True)
