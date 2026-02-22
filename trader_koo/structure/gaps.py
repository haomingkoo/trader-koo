from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class GapConfig:
    gaps_lookback_months: int = 18
    max_gaps: int = 4
    max_dist_pct: float = 0.12
    only_open: bool = True


def detect_gaps(df: pd.DataFrame) -> pd.DataFrame:
    x = df.sort_values("date").reset_index(drop=True).copy()
    if len(x) < 2:
        return pd.DataFrame(
            columns=["date", "type", "gap_low", "gap_high", "mid", "filled", "fill_date"]
        )

    x["prev_high"] = x["high"].shift(1)
    x["prev_low"] = x["low"].shift(1)
    rows: list[dict] = []

    for i in range(1, len(x)):
        row = x.iloc[i]
        gap_type = None
        gap_low = None
        gap_high = None
        if row["low"] > row["prev_high"]:
            gap_type = "bull_gap"
            gap_low = float(row["prev_high"])
            gap_high = float(row["low"])
        elif row["high"] < row["prev_low"]:
            gap_type = "bear_gap"
            gap_low = float(row["high"])
            gap_high = float(row["prev_low"])
        if gap_type is None:
            continue

        future = x.iloc[i + 1 :]
        fill_date = None
        if not future.empty:
            if gap_type == "bull_gap":
                touched = future[future["low"] <= gap_low]
            else:
                touched = future[future["high"] >= gap_high]
            if not touched.empty:
                fill_date = pd.Timestamp(touched.iloc[0]["date"]).strftime("%Y-%m-%d")

        rows.append(
            {
                "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "type": gap_type,
                "gap_low": gap_low,
                "gap_high": gap_high,
                "mid": (gap_low + gap_high) / 2.0,
                "filled": fill_date is not None,
                "fill_date": fill_date,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["date", "type", "gap_low", "gap_high", "mid", "filled", "fill_date"]
        )
    return pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)


def select_gaps_for_display(
    gaps: pd.DataFrame,
    *,
    last_close: float,
    asof: pd.Timestamp,
    cfg: GapConfig,
) -> pd.DataFrame:
    if gaps is None or gaps.empty:
        return pd.DataFrame(
            columns=["date", "type", "gap_low", "gap_high", "mid", "filled", "fill_date", "dist"]
        )

    g = gaps.copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g = g.dropna(subset=["date"]).copy()
    lookback_start = asof - pd.DateOffset(months=max(1, cfg.gaps_lookback_months))
    g = g[g["date"] >= lookback_start]

    if cfg.only_open:
        g = g[~g["filled"]]

    if g.empty:
        return pd.DataFrame(
            columns=["date", "type", "gap_low", "gap_high", "mid", "filled", "fill_date", "dist"]
        )

    g["dist"] = (g["mid"] - last_close).abs()
    g = g[g["dist"] <= cfg.max_dist_pct * last_close]
    if g.empty:
        return pd.DataFrame(
            columns=["date", "type", "gap_low", "gap_high", "mid", "filled", "fill_date", "dist"]
        )

    g = g.sort_values(["dist", "date"], ascending=[True, False]).head(cfg.max_gaps).reset_index(drop=True)
    g["date"] = g["date"].dt.strftime("%Y-%m-%d")
    return g
