from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class HybridPatternConfig:
    recent_candle_days: int = 7
    volume_ma_window: int = 20
    weight_base: float = 0.55
    weight_candle: float = 0.20
    weight_volume: float = 0.15
    weight_breakout: float = 0.10
    max_rows: int = 8


HYBRID_COLUMNS = [
    "pattern",
    "status",
    "base_confidence",
    "candle_score",
    "volume_score",
    "breakout_score",
    "hybrid_confidence",
    "candle_bias",
    "vol_ratio",
    "start_date",
    "end_date",
    "notes",
]


def _empty_hybrid() -> pd.DataFrame:
    return pd.DataFrame(columns=HYBRID_COLUMNS)


def _is_bullish_pattern(name: str) -> bool:
    n = str(name or "").lower()
    return ("bull" in n) or ("falling_wedge" in n)


def _is_bearish_pattern(name: str) -> bool:
    n = str(name or "").lower()
    return ("bear" in n) or ("rising_wedge" in n)


def _to_conf01(v: float | int | None, default: float = 0.5) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return x


def score_hybrid_patterns(
    prices: pd.DataFrame,
    patterns: pd.DataFrame,
    candles: pd.DataFrame,
    cfg: HybridPatternConfig,
) -> pd.DataFrame:
    if patterns is None or patterns.empty:
        return _empty_hybrid()
    if prices is None or prices.empty:
        return _empty_hybrid()

    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if px.empty:
        return _empty_hybrid()

    px["volume"] = pd.to_numeric(px.get("volume"), errors="coerce")
    px["vol_ma"] = px["volume"].rolling(max(2, cfg.volume_ma_window), min_periods=5).mean()

    cdl = candles.copy() if candles is not None else pd.DataFrame(columns=["date", "bias", "pattern"])
    if not cdl.empty:
        cdl["date"] = pd.to_datetime(cdl["date"], errors="coerce")
        cdl = cdl.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    out_rows: list[dict] = []
    for r in patterns.to_dict(orient="records"):
        pattern = str(r.get("pattern") or "")
        status = str(r.get("status") or "forming")
        end_date = pd.to_datetime(r.get("end_date"), errors="coerce")
        start_date = pd.to_datetime(r.get("start_date"), errors="coerce")
        if pd.isna(end_date):
            continue

        side = "neutral"
        if _is_bullish_pattern(pattern):
            side = "bullish"
        elif _is_bearish_pattern(pattern):
            side = "bearish"

        base_conf = _to_conf01(r.get("confidence"), default=0.5)

        # Candlestick alignment near the pattern end date.
        candle_score = 0.5
        candle_bias = "neutral"
        if not cdl.empty:
            recent = cdl[(cdl["date"] <= end_date) & (cdl["date"] >= end_date - pd.Timedelta(days=cfg.recent_candle_days))]
            if not recent.empty:
                bull_n = int((recent["bias"] == "bullish").sum())
                bear_n = int((recent["bias"] == "bearish").sum())
                net = bull_n - bear_n
                if side == "bearish":
                    net = -net
                candle_score = max(0.0, min(1.0, 0.5 + 0.12 * net))
                if net > 0:
                    candle_bias = "aligned"
                elif net < 0:
                    candle_bias = "conflict"
                else:
                    candle_bias = "mixed"

        # Volume regime scoring.
        pos = px[px["date"] <= end_date]
        if pos.empty:
            continue
        pr = pos.iloc[-1]
        vol = float(pr.get("volume") or 0.0)
        vol_ma = float(pr.get("vol_ma") or 0.0)
        vol_ratio = (vol / vol_ma) if vol_ma > 0 else 1.0
        if status in {"breakout", "breakdown"}:
            volume_score = max(0.0, min(1.0, (vol_ratio - 0.8) / 0.9))
            breakout_score = 0.9
        else:
            # Forming patterns are healthier when participation is not overly noisy.
            volume_score = max(0.0, min(1.0, 1.15 - abs(vol_ratio - 0.9)))
            breakout_score = 0.45

        hybrid = (
            cfg.weight_base * base_conf
            + cfg.weight_candle * candle_score
            + cfg.weight_volume * volume_score
            + cfg.weight_breakout * breakout_score
        )
        hybrid = max(0.0, min(0.99, float(hybrid)))

        start_s = pd.Timestamp(start_date).strftime("%Y-%m-%d") if pd.notna(start_date) else ""
        end_s = pd.Timestamp(end_date).strftime("%Y-%m-%d")
        out_rows.append(
            {
                "pattern": pattern,
                "status": status,
                "base_confidence": round(base_conf, 2),
                "candle_score": round(candle_score, 2),
                "volume_score": round(volume_score, 2),
                "breakout_score": round(breakout_score, 2),
                "hybrid_confidence": round(hybrid, 2),
                "candle_bias": candle_bias,
                "vol_ratio": round(vol_ratio, 2),
                "start_date": start_s,
                "end_date": end_s,
                "notes": str(r.get("notes") or ""),
            }
        )

    if not out_rows:
        return _empty_hybrid()
    out = pd.DataFrame(out_rows).sort_values(["hybrid_confidence", "base_confidence"], ascending=[False, False])
    out = out.head(cfg.max_rows).reset_index(drop=True)
    return out[HYBRID_COLUMNS].copy()

