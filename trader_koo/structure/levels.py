from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LevelConfig:
    level_tol_atr: float = 0.60
    zone_half_width_atr: float = 0.35
    min_zone_width: float = 0.05
    min_touches: int = 2
    primary_each_side: int = 2
    secondary_each_side: int = 2
    max_dist_primary_pct: float = 0.20
    max_dist_secondary_pct: float = 0.45
    near_side_tolerance_pct: float = 0.015
    recency_half_life_days: int = 45
    min_recency_score: float = 0.03
    fallback_lookback_bars: int = 126


LEVEL_COLUMNS = [
    "type",
    "level",
    "zone_low",
    "zone_high",
    "touches",
    "last_touch_date",
    "recency_score",
    "dist",
    "tier",
]


def _empty_levels() -> pd.DataFrame:
    return pd.DataFrame(columns=LEVEL_COLUMNS)


def _safe_concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
    if not valid:
        return _empty_levels()
    return pd.concat(valid, axis=0, ignore_index=True)


def estimate_level_tolerance(df: pd.DataFrame, cfg: LevelConfig) -> float:
    atr_med = float(np.nanmedian(df["atr"].to_numpy())) if "atr" in df.columns else np.nan
    if not np.isfinite(atr_med) or atr_med <= 0:
        atr_med = float(np.nanmedian((df["high"] - df["low"]).to_numpy()))
    if not np.isfinite(atr_med) or atr_med <= 0:
        atr_med = 1.0
    return max(cfg.min_zone_width, cfg.level_tol_atr * atr_med)


def _cluster_points(points: pd.DataFrame, tol: float) -> list[pd.DataFrame]:
    if points.empty:
        return []
    p = points.sort_values("price").reset_index(drop=True)
    clusters: list[list[int]] = [[0]]
    running_mean = float(p.loc[0, "price"])
    for i in range(1, len(p)):
        x = float(p.loc[i, "price"])
        if abs(x - running_mean) <= tol:
            clusters[-1].append(i)
            running_mean = float(p.loc[clusters[-1], "price"].mean())
        else:
            clusters.append([i])
            running_mean = x
    return [p.loc[idxs].copy() for idxs in clusters]


def _build_side_levels(
    piv: pd.DataFrame,
    *,
    side: str,
    price_col: str,
    pivot_col: str,
    tol: float,
    cfg: LevelConfig,
) -> pd.DataFrame:
    points = piv.loc[piv[pivot_col], ["date", price_col]].rename(columns={price_col: "price"})
    if points.empty:
        return _empty_levels()

    latest_date = piv["date"].max()
    rows: list[dict] = []
    for cluster in _cluster_points(points, tol=tol):
        touches = len(cluster)
        if touches < cfg.min_touches:
            continue
        level = float(cluster["price"].mean())
        last_touch = pd.to_datetime(cluster["date"]).max()
        age_days = max(int((latest_date - last_touch).days), 0)
        recency_score = float(0.5 ** (age_days / max(cfg.recency_half_life_days, 1)))
        if recency_score < cfg.min_recency_score:
            continue
        zone_half = max(cfg.min_zone_width, tol * cfg.zone_half_width_atr)
        rows.append(
            {
                "type": side,
                "level": level,
                "zone_low": level - zone_half,
                "zone_high": level + zone_half,
                "touches": int(touches),
                "last_touch_date": pd.Timestamp(last_touch).strftime("%Y-%m-%d"),
                "recency_score": recency_score,
                "dist": np.nan,
                "tier": "raw",
            }
        )

    if not rows:
        return _empty_levels()
    return pd.DataFrame(rows).sort_values("level").reset_index(drop=True)


def build_levels_from_pivots(piv: pd.DataFrame, cfg: LevelConfig) -> pd.DataFrame:
    tol = estimate_level_tolerance(piv, cfg)
    sup = _build_side_levels(
        piv,
        side="support",
        price_col="low",
        pivot_col="pivot_low",
        tol=tol,
        cfg=cfg,
    )
    res = _build_side_levels(
        piv,
        side="resistance",
        price_col="high",
        pivot_col="pivot_high",
        tol=tol,
        cfg=cfg,
    )
    out = _safe_concat([sup, res])
    if out.empty:
        return out
    return out.drop_duplicates(subset=["type", "level"]).sort_values("level").reset_index(drop=True)


def select_target_levels(levels: pd.DataFrame, last_close: float, cfg: LevelConfig) -> pd.DataFrame:
    if levels is None or levels.empty:
        return _empty_levels()

    lv = levels.copy()
    lv["dist"] = (lv["level"] - last_close).abs()
    lv = lv[lv["dist"] <= cfg.max_dist_secondary_pct * last_close].copy()
    if lv.empty:
        return _empty_levels()

    def pick(side: str, is_support: bool) -> pd.DataFrame:
        cond = lv["type"] == side
        cond = cond & (lv["level"] <= last_close if is_support else lv["level"] >= last_close)
        pool = lv[cond].sort_values(["dist", "touches", "recency_score"], ascending=[True, False, False])
        # If price is sitting at/near extremes, strict one-sided filtering can return empty.
        # In that case, allow nearby levels around last_close before falling back.
        if pool.empty:
            tol = max(cfg.near_side_tolerance_pct, 0.0)
            near_cond = lv["type"] == side
            if is_support:
                near_cond = near_cond & (lv["level"] <= last_close * (1 + tol))
            else:
                near_cond = near_cond & (lv["level"] >= last_close * (1 - tol))
            pool = lv[near_cond].sort_values(["dist", "touches", "recency_score"], ascending=[True, False, False])
        if pool.empty:
            return _empty_levels()
        primary = pool[pool["dist"] <= cfg.max_dist_primary_pct * last_close].head(cfg.primary_each_side).copy()
        primary["tier"] = "primary"
        secondary = pool.loc[~pool.index.isin(primary.index)].head(cfg.secondary_each_side).copy()
        secondary["tier"] = "secondary"
        return _safe_concat([primary, secondary])

    sup = pick("support", True)
    res = pick("resistance", False)
    out = _safe_concat([sup, res])
    if out.empty:
        return out
    return out.drop_duplicates(subset=["type", "level"]).sort_values("level").reset_index(drop=True)


def add_fallback_levels(
    prices: pd.DataFrame,
    levels: pd.DataFrame,
    last_close: float,
    cfg: LevelConfig,
) -> pd.DataFrame:
    lv = levels.copy() if levels is not None else _empty_levels()
    if prices.empty:
        return lv

    recent = prices.tail(min(len(prices), cfg.fallback_lookback_bars)).copy()
    if recent.empty:
        return lv

    tol = estimate_level_tolerance(prices, cfg)
    zone_half = max(cfg.min_zone_width, tol * cfg.zone_half_width_atr)
    add_rows: list[dict] = []
    has_support = (not lv.empty) and (lv["type"] == "support").any()
    has_resistance = (not lv.empty) and (lv["type"] == "resistance").any()

    if not has_support:
        mask = recent["low"] <= last_close
        if mask.any():
            idx = recent.loc[mask, "low"].idxmax()
        else:
            idx = recent["low"].idxmin()
        level = float(recent.loc[idx, "low"])
        add_rows.append(
            {
                "type": "support",
                "level": level,
                "zone_low": level - zone_half,
                "zone_high": level + zone_half,
                "touches": 1,
                "last_touch_date": pd.Timestamp(recent.loc[idx, "date"]).strftime("%Y-%m-%d"),
                "recency_score": 1.0,
                "dist": abs(level - last_close),
                "tier": "fallback",
            }
        )

    if not has_resistance:
        mask = recent["high"] >= last_close
        if mask.any():
            idx = recent.loc[mask, "high"].idxmin()
        else:
            idx = recent["high"].idxmax()
        level = float(recent.loc[idx, "high"])
        add_rows.append(
            {
                "type": "resistance",
                "level": level,
                "zone_low": level - zone_half,
                "zone_high": level + zone_half,
                "touches": 1,
                "last_touch_date": pd.Timestamp(recent.loc[idx, "date"]).strftime("%Y-%m-%d"),
                "recency_score": 1.0,
                "dist": abs(level - last_close),
                "tier": "fallback",
            }
        )

    if add_rows:
        lv = _safe_concat([lv, pd.DataFrame(add_rows)])

    if lv.empty:
        return lv
    return lv.drop_duplicates(subset=["type", "level"]).sort_values("level").reset_index(drop=True)
