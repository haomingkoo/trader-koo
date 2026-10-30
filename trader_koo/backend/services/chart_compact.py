"""Compact wire format for the chart page's quick and regime payloads.

The chart page computes its moving averages client-side from ``close``, plots
only the HMM probability bands, and never reads the per-row indicator columns
or per-day regime labels. These helpers drop what the page does not read and
round float32 noise (86.12999725341797) to cents-safe precision. They only
shape API responses; stored reports and model outputs are left untouched.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

CHART_ROW_KEYS = ("date", "open", "high", "low", "close", "volume")
PRICE_DECIMALS = 4
# Matches predict_regimes' default training window (~2 trading years).
HMM_CHART_DAYS = 504
HMM_DAY_KEYS = ("date", "prob_low", "prob_normal", "prob_high")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def compact_chart_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return date + OHLCV rows with rounded prices, int volume, None for gaps."""
    if frame is None or frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        row: dict[str, Any] = {"date": record.get("date")}
        for key in ("open", "high", "low", "close"):
            number = _finite(record.get(key))
            row[key] = None if number is None else round(number, PRICE_DECIMALS)
        volume = _finite(record.get("volume"))
        row["volume"] = None if volume is None else int(round(volume))
        rows.append(row)
    return rows


def compact_hmm_regime(regime: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep the summary fields and the plotted probabilities for recent days."""
    if not isinstance(regime, dict):
        return None
    days = regime.get("regimes") or []
    out = {key: value for key, value in regime.items() if key != "regimes"}
    out["regimes"] = [
        {key: day.get(key) for key in HMM_DAY_KEYS}
        for day in days[-HMM_CHART_DAYS:]
        if isinstance(day, dict)
    ]
    return out
