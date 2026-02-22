from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import talib  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    talib = None


@dataclass
class CandlePatternConfig:
    lookback_bars: int = 180
    max_rows: int = 30
    use_talib: bool = True
    min_abs_signal: int = 100


def _bias_of(name: str) -> str:
    n = name.lower()
    if any(x in n for x in ["bull", "hammer", "morning"]):
        return "bullish"
    if any(x in n for x in ["bear", "shooting", "evening"]):
        return "bearish"
    return "neutral"


def _explain(name: str) -> str:
    m = {
        "bullish_engulfing": "Bull candle fully engulfs prior bear body; reversal candidate.",
        "bearish_engulfing": "Bear candle fully engulfs prior bull body; reversal candidate.",
        "doji": "Open and close are very close; indecision, often context-dependent.",
        "hammer": "Long lower wick after decline; buyers rejected lower prices.",
        "shooting_star": "Long upper wick after rise; sellers rejected higher prices.",
        "morning_star": "Three-candle bullish reversal sequence after weakness.",
        "evening_star": "Three-candle bearish reversal sequence after strength.",
        "three_white_soldiers": "Three strong consecutive bull candles; momentum continuation/reversal context.",
        "three_black_crows": "Three strong consecutive bear candles; momentum continuation/reversal context.",
        "three_inside_up": "Bullish three-candle pattern after weakness.",
        "three_inside_down": "Bearish three-candle pattern after strength.",
        "three_outside_up": "Bullish engulfing continuation signal on 3-candle basis.",
        "three_outside_down": "Bearish engulfing continuation signal on 3-candle basis.",
    }
    return m.get(name, "Candlestick pattern candidate.")


def _talib_name_to_slug(name: str) -> str:
    core = name.replace("CDL", "").lower()
    mapping = {
        "3whitesoldiers": "three_white_soldiers",
        "3blackcrows": "three_black_crows",
        "3inside": "three_inside",
        "3outside": "three_outside",
        "3linestrike": "three_line_strike",
        "abandonedbaby": "abandoned_baby",
        "engulfing": "engulfing",
    }
    if core in mapping:
        return mapping[core]
    # insert underscores between alpha blocks and digits minimally
    out = []
    for i, ch in enumerate(core):
        if i > 0 and ch.isdigit() and core[i - 1].isalpha():
            out.append("_")
        if i > 0 and ch.isalpha() and core[i - 1].isdigit():
            out.append("_")
        out.append(ch)
    return "".join(out)


def _detect_talib_patterns(df: pd.DataFrame, cfg: CandlePatternConfig) -> list[dict]:
    if talib is None or not cfg.use_talib:
        return []
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(dtype=float)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    l = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(o).all() or np.isnan(h).all() or np.isnan(l).all() or np.isnan(c).all():
        return []

    date_s = pd.to_datetime(df["date"], errors="coerce")
    out: list[dict] = []
    funcs = [n for n in dir(talib) if n.startswith("CDL")]
    for fname in funcs:
        fn = getattr(talib, fname, None)
        if not callable(fn):
            continue
        try:
            sig = fn(o, h, l, c)
        except Exception:
            continue
        if sig is None:
            continue
        sig_arr = np.asarray(sig, dtype=float)
        for i, v in enumerate(sig_arr):
            if not np.isfinite(v):
                continue
            av = abs(int(v))
            if av < cfg.min_abs_signal:
                continue
            d = date_s.iloc[i]
            if pd.isna(d):
                continue
            slug = _talib_name_to_slug(fname)
            bias = "bullish" if v > 0 else "bearish"
            conf = min(0.95, 0.55 + 0.35 * min(av, 200) / 200.0)
            out.append(
                {
                    "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                    "pattern": slug,
                    "bias": bias,
                    "confidence": round(float(conf), 2),
                    "explanation": f"TA-Lib candlestick signal ({bias}).",
                }
            )
    return out


def detect_candlestick_patterns(df: pd.DataFrame, cfg: CandlePatternConfig) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "pattern", "bias", "confidence", "explanation"])

    x = df.tail(cfg.lookback_bars).reset_index(drop=True).copy()
    rows: list[dict] = []
    for i in range(len(x)):
        r = x.iloc[i]
        o = float(r["open"])
        h = float(r["high"])
        l = float(r["low"])
        c = float(r["close"])
        rng = max(h - l, 1e-9)
        body = abs(c - o)
        upper = h - max(o, c)
        lower = min(o, c) - l
        date = pd.Timestamp(r["date"]).strftime("%Y-%m-%d")

        # Single-candle
        if body <= 0.10 * rng:
            rows.append({"date": date, "pattern": "doji", "confidence": 0.50})
        if lower >= 2.2 * max(body, 1e-9) and upper <= 0.7 * max(body, 1e-9):
            rows.append({"date": date, "pattern": "hammer", "confidence": 0.56})
        if upper >= 2.2 * max(body, 1e-9) and lower <= 0.7 * max(body, 1e-9):
            rows.append({"date": date, "pattern": "shooting_star", "confidence": 0.56})

        # Two-candle
        if i >= 1:
            p = x.iloc[i - 1]
            po, pc = float(p["open"]), float(p["close"])
            prev_bear = pc < po
            prev_bull = pc > po
            curr_bull = c > o
            curr_bear = c < o
            if prev_bear and curr_bull and o <= pc and c >= po:
                rows.append({"date": date, "pattern": "bullish_engulfing", "confidence": 0.64})
            if prev_bull and curr_bear and o >= pc and c <= po:
                rows.append({"date": date, "pattern": "bearish_engulfing", "confidence": 0.64})
            # Three outside up/down via engulfing + confirmation
            if i >= 2:
                pp = x.iloc[i - 2]
                ppo, ppc = float(pp["open"]), float(pp["close"])
                if ppc < ppo and prev_bull and curr_bull and po <= ppc and pc >= ppo and c > pc:
                    rows.append({"date": date, "pattern": "three_outside_up", "confidence": 0.60})
                if ppc > ppo and prev_bear and curr_bear and po >= ppc and pc <= ppo and c < pc:
                    rows.append({"date": date, "pattern": "three_outside_down", "confidence": 0.60})

        # Three-candle
        if i >= 2:
            a = x.iloc[i - 2]
            b = x.iloc[i - 1]
            ao, ac = float(a["open"]), float(a["close"])
            bo, bc = float(b["open"]), float(b["close"])
            a_body = abs(ac - ao)
            b_body = abs(bc - bo)
            mid_a = (ao + ac) / 2.0
            # Morning star (simplified)
            if ac < ao and b_body <= 0.5 * max(a_body, 1e-9) and c > o and c >= mid_a:
                rows.append({"date": date, "pattern": "morning_star", "confidence": 0.60})
            # Evening star (simplified)
            if ac > ao and b_body <= 0.5 * max(a_body, 1e-9) and c < o and c <= mid_a:
                rows.append({"date": date, "pattern": "evening_star", "confidence": 0.60})
            # Three inside up/down (simplified)
            if ac < ao and bc > bo and bc < ao and bc > ac and c > max(ao, bo, bc):
                rows.append({"date": date, "pattern": "three_inside_up", "confidence": 0.58})
            if ac > ao and bc < bo and bc > ao and bc < ac and c < min(ao, bo, bc):
                rows.append({"date": date, "pattern": "three_inside_down", "confidence": 0.58})
            # Three white soldiers / black crows (simplified)
            if (
                float(x.iloc[i - 2]["close"]) > float(x.iloc[i - 2]["open"])
                and float(x.iloc[i - 1]["close"]) > float(x.iloc[i - 1]["open"])
                and c > o
                and float(x.iloc[i - 2]["close"]) < float(x.iloc[i - 1]["close"]) < c
            ):
                rows.append({"date": date, "pattern": "three_white_soldiers", "confidence": 0.62})
            if (
                float(x.iloc[i - 2]["close"]) < float(x.iloc[i - 2]["open"])
                and float(x.iloc[i - 1]["close"]) < float(x.iloc[i - 1]["open"])
                and c < o
                and float(x.iloc[i - 2]["close"]) > float(x.iloc[i - 1]["close"]) > c
            ):
                rows.append({"date": date, "pattern": "three_black_crows", "confidence": 0.62})

    rows.extend(_detect_talib_patterns(x, cfg))

    if not rows:
        return pd.DataFrame(columns=["date", "pattern", "bias", "confidence", "explanation"])

    out = pd.DataFrame(rows).drop_duplicates(subset=["date", "pattern"]).copy()
    if "bias" not in out.columns:
        out["bias"] = out["pattern"].map(_bias_of)
    else:
        out["bias"] = out["bias"].fillna(out["pattern"].map(_bias_of))
    if "explanation" not in out.columns:
        out["explanation"] = out["pattern"].map(_explain)
    else:
        out["explanation"] = out["explanation"].fillna(out["pattern"].map(_explain))
    out["confidence"] = out["confidence"].clip(0.0, 0.99).round(2)
    out = out.sort_values(["date", "confidence"], ascending=[False, False]).head(cfg.max_rows).reset_index(drop=True)
    return out[["date", "pattern", "bias", "confidence", "explanation"]]
