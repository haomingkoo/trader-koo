from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CVProxyConfig:
    lookback_bars: int = 100
    shape_bars: int = 45
    flag_lookback_bars: int = 30
    flag_pole_bars: int = 8
    min_pole_return: float = 0.05      # pole must move at least 5% (was 4%)
    parallel_tol: float = 0.005        # flag channel parallel tolerance (was 0.003)
    wedge_converge_ratio: float = 0.65 # end gap ≤65% of start gap = 35%+ convergence (was 0.80)
    flat_slope_tol: float = 0.0008
    min_shape_r2: float = 0.45         # minimum R² for regression lines to be accepted
    max_patterns: int = 8
    double_lookback_bars: int = 120
    hs_lookback_bars: int = 150
    pivot_window: int = 5


CV_COLUMNS = [
    "pattern",
    "start_date",
    "end_date",
    "status",
    "cv_confidence",
    "method",
    "notes",
    "x0_date",
    "x1_date",
    "y0",
    "y1",
    "y0b",
    "y1b",
    "x_mid_date",
    "y_mid",
]


def _empty_cv() -> pd.DataFrame:
    return pd.DataFrame(columns=CV_COLUMNS)


def _fit_line(y: np.ndarray) -> tuple[float, float] | None:
    if y.size < 2:
        return None
    x = np.arange(y.size, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None
    return float(slope), float(intercept)


def _r2(y: np.ndarray, slope: float, intercept: float) -> float:
    x = np.arange(y.size, dtype=float)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    v = 1.0 - ss_res / ss_tot
    return max(0.0, min(1.0, v))


def _to_norm(seg: pd.DataFrame) -> pd.DataFrame:
    lo = float(seg["low"].min())
    hi = float(seg["high"].max())
    span = max(hi - lo, 1e-9)
    out = seg.copy()
    for c in ["open", "high", "low", "close"]:
        out[c] = (pd.to_numeric(out[c], errors="coerce") - lo) / span
    return out


def _to_date_str(d: object) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def _dates(seg: pd.DataFrame) -> tuple[str, str]:
    return _to_date_str(seg["date"].iloc[0]), _to_date_str(seg["date"].iloc[-1])


def _find_pivot_highs(highs: np.ndarray, window: int = 5) -> list[int]:
    """Return sorted indices of local price maxima."""
    n = len(highs)
    result = []
    for i in range(window, n - window):
        local = highs[i - window: i + window + 1]
        if highs[i] >= local.max() - 1e-9 and highs[i] > highs[i - 1]:
            result.append(i)
    return result


def _find_pivot_lows(lows: np.ndarray, window: int = 5) -> list[int]:
    """Return sorted indices of local price minima."""
    n = len(lows)
    result = []
    for i in range(window, n - window):
        local = lows[i - window: i + window + 1]
        if lows[i] <= local.min() + 1e-9 and lows[i] < lows[i - 1]:
            result.append(i)
    return result


def _fit_pivots(values: np.ndarray, pivot_idxs: list[int]) -> tuple[float, float] | None:
    """Fit a line through pivot points only (not every bar)."""
    if len(pivot_idxs) < 2:
        return None
    x = np.array(pivot_idxs, dtype=float)
    y = values[pivot_idxs]
    slope, intercept = np.polyfit(x, y, 1)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None
    return float(slope), float(intercept)


def _r2_pivots(values: np.ndarray, pivot_idxs: list[int], slope: float, intercept: float) -> float:
    """R² measured at pivot points only."""
    if len(pivot_idxs) < 2:
        return 0.0
    x = np.array(pivot_idxs, dtype=float)
    y = values[pivot_idxs]
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, 1.0 - ss_res / ss_tot))


def _shape_candidates(seg: pd.DataFrame, cfg: CVProxyConfig) -> list[dict]:
    out: list[dict] = []

    # Capture actual price range for denormalization before fitting
    actual_lo = float(seg["low"].min())
    actual_hi = float(seg["high"].max())
    actual_span = max(actual_hi - actual_lo, 1e-9)

    nseg = _to_norm(seg)
    hi = nseg["high"].to_numpy(dtype=float)
    lo = nseg["low"].to_numpy(dtype=float)
    close = nseg["close"].to_numpy(dtype=float)

    # Fit lines through swing highs / swing lows only — more faithful to how
    # traders draw trendlines than fitting every bar
    pivot_hi = _find_pivot_highs(hi, window=max(2, cfg.pivot_window // 2))
    pivot_lo = _find_pivot_lows(lo, window=max(2, cfg.pivot_window // 2))

    fit_hi = _fit_pivots(hi, pivot_hi) if len(pivot_hi) >= 2 else _fit_line(hi)
    fit_lo = _fit_pivots(lo, pivot_lo) if len(pivot_lo) >= 2 else _fit_line(lo)
    if fit_hi is None or fit_lo is None:
        return out
    s_hi, i_hi = fit_hi
    s_lo, i_lo = fit_lo

    r2_hi = _r2_pivots(hi, pivot_hi, s_hi, i_hi) if len(pivot_hi) >= 2 else _r2(hi, s_hi, i_hi)
    r2_lo = _r2_pivots(lo, pivot_lo, s_lo, i_lo) if len(pivot_lo) >= 2 else _r2(lo, s_lo, i_lo)
    quality = 0.5 * (r2_hi + r2_lo)

    # Reject noisy fits — lines must actually track the boundaries well
    if quality < cfg.min_shape_r2:
        return out

    start_gap = i_hi - i_lo
    end_gap = (s_hi * (len(seg) - 1) + i_hi) - (s_lo * (len(seg) - 1) + i_lo)
    if start_gap <= 0 or end_gap <= 0:
        return out
    converge = end_gap / max(start_gap, 1e-9)
    conv_strength = max(0.0, min(1.0, 1.0 - converge))

    x0, x1 = _dates(seg)
    last_close = float(close[-1])
    upper_last = s_hi * (len(seg) - 1) + i_hi
    lower_last = s_lo * (len(seg) - 1) + i_lo

    # Denormalized boundary y-values for chart overlay (actual price space)
    n_last = len(seg) - 1
    y0_act = float(i_hi) * actual_span + actual_lo
    y1_act = float(s_hi * n_last + i_hi) * actual_span + actual_lo
    y0b_act = float(i_lo) * actual_span + actual_lo
    y1b_act = float(s_lo * n_last + i_lo) * actual_span + actual_lo

    def add(pattern: str, status: str, conf: float, note: str) -> None:
        out.append(
            {
                "pattern": pattern,
                "start_date": x0,
                "end_date": x1,
                "status": status,
                "cv_confidence": round(float(max(0.0, min(0.99, conf))), 2),
                "method": "cv_proxy_regression",
                "notes": note,
                "x0_date": x0,
                "x1_date": x1,
                "y0": y0_act,
                "y1": y1_act,
                "y0b": y0b_act,
                "y1b": y1b_act,
                "x_mid_date": None,
                "y_mid": None,
            }
        )

    if converge <= cfg.wedge_converge_ratio:
        if s_hi > 0 and s_lo > 0:
            status = "breakdown" if last_close < lower_last else "forming"
            add("rising_wedge", status, 0.45 + 0.30 * conv_strength + 0.20 * quality, "Converging rising boundaries")
        if s_hi < 0 and s_lo < 0:
            status = "breakout" if last_close > upper_last else "forming"
            add("falling_wedge", status, 0.45 + 0.30 * conv_strength + 0.20 * quality, "Converging falling boundaries")

        if abs(s_hi) <= cfg.flat_slope_tol and s_lo > cfg.flat_slope_tol:
            status = "breakout" if last_close > upper_last else "forming"
            add("ascending_triangle", status, 0.42 + 0.30 * conv_strength + 0.20 * quality, "Flat top + rising lows")
        if abs(s_lo) <= cfg.flat_slope_tol and s_hi < -cfg.flat_slope_tol:
            status = "breakdown" if last_close < lower_last else "forming"
            add("descending_triangle", status, 0.42 + 0.30 * conv_strength + 0.20 * quality, "Falling highs + flat base")
        if s_hi < -cfg.flat_slope_tol and s_lo > cfg.flat_slope_tol:
            if last_close > upper_last:
                status = "breakout"
            elif last_close < lower_last:
                status = "breakdown"
            else:
                status = "forming"
            add("symmetrical_triangle", status, 0.40 + 0.28 * conv_strength + 0.20 * quality, "Converging opposite slopes")

    return out


def _flag_candidates(seg: pd.DataFrame, cfg: CVProxyConfig) -> list[dict]:
    out: list[dict] = []
    if len(seg) < cfg.flag_pole_bars + 8:
        return out

    # Capture actual price range before normalization for denormalization
    actual_lo = float(seg["low"].min())
    actual_hi = float(seg["high"].max())
    actual_span = max(actual_hi - actual_lo, 1e-9)

    nseg = _to_norm(seg)
    pole = nseg.iloc[: cfg.flag_pole_bars].copy()
    cons = nseg.iloc[cfg.flag_pole_bars :].copy()
    if cons.empty:
        return out

    pole_start = float(pole["open"].iloc[0])
    pole_end = float(pole["close"].iloc[-1])
    pole_move = pole_end - pole_start

    hi = cons["high"].to_numpy(dtype=float)
    lo = cons["low"].to_numpy(dtype=float)
    fit_hi = _fit_line(hi)
    fit_lo = _fit_line(lo)
    if fit_hi is None or fit_lo is None:
        return out
    s_hi, i_hi = fit_hi
    s_lo, i_lo = fit_lo
    quality = 0.5 * (_r2(hi, s_hi, i_hi) + _r2(lo, s_lo, i_lo))
    parallel = abs(s_hi - s_lo) <= cfg.parallel_tol
    if not parallel or quality < cfg.min_shape_r2:
        return out

    x0 = _to_date_str(cons["date"].iloc[0])
    x1 = _to_date_str(cons["date"].iloc[-1])
    n = len(cons) - 1
    last_close = float(cons["close"].iloc[-1])
    upper_last = s_hi * n + i_hi
    lower_last = s_lo * n + i_lo
    # quality already computed above

    # Denormalize boundary y-values to actual price space
    y0_act = float(i_hi) * actual_span + actual_lo
    y1_act = float(s_hi * n + i_hi) * actual_span + actual_lo
    y0b_act = float(i_lo) * actual_span + actual_lo
    y1b_act = float(s_lo * n + i_lo) * actual_span + actual_lo

    def add(pattern: str, status: str, conf: float, note: str) -> None:
        out.append(
            {
                "pattern": pattern,
                "start_date": x0,
                "end_date": x1,
                "status": status,
                "cv_confidence": round(float(max(0.0, min(0.99, conf))), 2),
                "method": "cv_proxy_regression",
                "notes": note,
                "x0_date": x0,
                "x1_date": x1,
                "y0": y0_act,
                "y1": y1_act,
                "y0b": y0b_act,
                "y1b": y1b_act,
                "x_mid_date": None,
                "y_mid": None,
            }
        )

    if pole_move >= cfg.min_pole_return and s_hi < 0 and s_lo < 0:
        status = "breakout" if last_close > upper_last else "forming"
        add("bull_flag", status, 0.44 + min(0.25, pole_move * 2.0) + 0.18 * quality, "Strong pole then downward channel")
    if pole_move <= -cfg.min_pole_return and s_hi > 0 and s_lo > 0:
        status = "breakdown" if last_close < lower_last else "forming"
        add("bear_flag", status, 0.44 + min(0.25, abs(pole_move) * 2.0) + 0.18 * quality, "Strong drop then upward channel")
    return out


def _double_top_candidates(seg: pd.DataFrame, cfg: CVProxyConfig) -> list[dict]:
    """Detect M-shaped double top: two peaks at similar heights with meaningful valley between."""
    out: list[dict] = []
    if len(seg) < 30:
        return out

    highs = seg["high"].to_numpy(dtype=float)
    lows = seg["low"].to_numpy(dtype=float)
    closes = seg["close"].to_numpy(dtype=float)
    dates = [_to_date_str(d) for d in seg["date"]]
    n = len(dates)

    peaks = _find_pivot_highs(highs, cfg.pivot_window)
    if len(peaks) < 2:
        return out

    peaks = peaks[-6:]  # consider only recent peaks

    best: dict | None = None
    best_conf = 0.0

    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            p1, p2 = peaks[i], peaks[j]
            p1_price, p2_price = float(highs[p1]), float(highs[p2])

            sep = p2 - p1
            if sep < 10 or sep > 100:
                continue

            avg_peak = (p1_price + p2_price) / 2.0
            price_diff_pct = abs(p1_price - p2_price) / max(avg_peak, 1e-9)
            if price_diff_pct > 0.03:   # peaks must be within 3% of each other
                continue

            # Deepest valley between the two peaks
            between_lows = lows[p1: p2 + 1]
            valley_rel = int(np.argmin(between_lows))
            valley_idx = p1 + valley_rel
            valley_price = float(lows[valley_idx])

            drop_pct = (avg_peak - valley_price) / max(avg_peak, 1e-9)
            if drop_pct < 0.08:   # meaningful valley: at least 8% below peaks (Bulkowski recommends 10%)
                continue

            neckline = valley_price
            last_close = float(closes[-1])

            # Only confirm as double_top when neckline is actually broken — two peaks
            # without a break is just a resistance zone, not a reversal pattern
            if last_close >= neckline:
                continue
            status = "breakdown"

            recency = p2 / n
            conf = (
                0.40
                + (1.0 - price_diff_pct / 0.04) * 0.25
                + min(drop_pct * 3.0, 0.22)
                + recency * 0.08
            )
            conf = float(min(max(conf, 0.0), 0.92))

            if conf > best_conf:
                best_conf = conf
                best = {
                    "pattern": "double_top",
                    "start_date": dates[p1],
                    "end_date": dates[p2],
                    "status": status,
                    "cv_confidence": round(conf, 2),
                    "method": "cv_proxy_pivot",
                    "notes": f"Peaks {p1_price:.2f}/{p2_price:.2f} neckline {neckline:.2f}",
                    # Line 1 (y0→y1): horizontal resistance from p1 to p2 only
                    # Line 2 (y0b→y1b): horizontal neckline from p1 to p2 only
                    "x0_date": dates[p1],
                    "x1_date": dates[p2],
                    "y0": float(avg_peak),
                    "y1": float(avg_peak),
                    "y0b": float(neckline),
                    "y1b": float(neckline),
                    "x_mid_date": dates[p2],
                    "y_mid": float(p2_price),
                }

    if best:
        out.append(best)
    return out


def _double_bottom_candidates(seg: pd.DataFrame, cfg: CVProxyConfig) -> list[dict]:
    """Detect W-shaped double bottom: two troughs at similar depths with meaningful peak between."""
    out: list[dict] = []
    if len(seg) < 30:
        return out

    highs = seg["high"].to_numpy(dtype=float)
    lows = seg["low"].to_numpy(dtype=float)
    closes = seg["close"].to_numpy(dtype=float)
    dates = [_to_date_str(d) for d in seg["date"]]
    n = len(dates)

    troughs = _find_pivot_lows(lows, cfg.pivot_window)
    if len(troughs) < 2:
        return out

    troughs = troughs[-6:]

    best: dict | None = None
    best_conf = 0.0

    for i in range(len(troughs)):
        for j in range(i + 1, len(troughs)):
            t1, t2 = troughs[i], troughs[j]
            t1_price, t2_price = float(lows[t1]), float(lows[t2])

            sep = t2 - t1
            if sep < 10 or sep > 100:
                continue

            avg_trough = (t1_price + t2_price) / 2.0
            price_diff_pct = abs(t1_price - t2_price) / max(avg_trough, 1e-9)
            if price_diff_pct > 0.03:   # troughs must be within 3% of each other
                continue

            # Highest peak between the two troughs (the neckline resistance)
            between_highs = highs[t1: t2 + 1]
            peak_rel = int(np.argmax(between_highs))
            peak_idx = t1 + peak_rel
            peak_price = float(highs[peak_idx])

            rise_pct = (peak_price - avg_trough) / max(avg_trough, 1e-9)
            if rise_pct < 0.08:   # meaningful peak: at least 8% above troughs (Bulkowski recommends 10%)
                continue

            neckline = peak_price
            last_close = float(closes[-1])

            # Only confirm as double_bottom when neckline is actually broken upward
            if last_close <= neckline:
                continue
            status = "breakout"

            recency = t2 / n
            conf = (
                0.40
                + (1.0 - price_diff_pct / 0.04) * 0.25
                + min(rise_pct * 3.0, 0.22)
                + recency * 0.08
            )
            conf = float(min(max(conf, 0.0), 0.92))

            if conf > best_conf:
                best_conf = conf
                best = {
                    "pattern": "double_bottom",
                    "start_date": dates[t1],
                    "end_date": dates[t2],
                    "status": status,
                    "cv_confidence": round(conf, 2),
                    "method": "cv_proxy_pivot",
                    "notes": f"Troughs {t1_price:.2f}/{t2_price:.2f} neckline {neckline:.2f}",
                    # Line 1 (y0→y1): horizontal support from t1 to t2 only
                    # Line 2 (y0b→y1b): horizontal neckline from t1 to t2 only
                    "x0_date": dates[t1],
                    "x1_date": dates[t2],
                    "y0": float(avg_trough),
                    "y1": float(avg_trough),
                    "y0b": float(neckline),
                    "y1b": float(neckline),
                    "x_mid_date": dates[t2],
                    "y_mid": float(t2_price),
                }

    if best:
        out.append(best)
    return out


def _head_shoulders_candidates(seg: pd.DataFrame, cfg: CVProxyConfig) -> list[dict]:
    """Detect head-and-shoulders: three peaks, middle (head) highest, shoulders at similar heights."""
    out: list[dict] = []
    if len(seg) < 40:
        return out

    highs = seg["high"].to_numpy(dtype=float)
    lows = seg["low"].to_numpy(dtype=float)
    closes = seg["close"].to_numpy(dtype=float)
    dates = [_to_date_str(d) for d in seg["date"]]
    n = len(dates)

    peaks = _find_pivot_highs(highs, cfg.pivot_window)
    if len(peaks) < 3:
        return out

    peaks = peaks[-8:]

    best: dict | None = None
    best_conf = 0.0

    for i in range(len(peaks) - 2):
        for j in range(i + 1, len(peaks) - 1):
            for k in range(j + 1, len(peaks)):
                ls_idx, h_idx, rs_idx = peaks[i], peaks[j], peaks[k]
                ls_price = float(highs[ls_idx])
                h_price = float(highs[h_idx])
                rs_price = float(highs[rs_idx])

                if not (h_price > ls_price and h_price > rs_price):
                    continue

                shoulder_diff_pct = abs(ls_price - rs_price) / max(ls_price, rs_price, 1e-9)
                if shoulder_diff_pct > 0.06:   # shoulders within 6%
                    continue

                avg_shoulder = (ls_price + rs_price) / 2.0
                head_dom = (h_price - avg_shoulder) / max(avg_shoulder, 1e-9)
                if head_dom < 0.05:   # head must be clearly higher than shoulders
                    continue

                total_sep = rs_idx - ls_idx
                if total_sep < 20 or total_sep > 120:
                    continue

                # Neckline valleys: left between ls-head, right between head-rs
                lv_rel = int(np.argmin(lows[ls_idx: h_idx + 1]))
                lv_idx = ls_idx + lv_rel
                neckline_l = float(lows[lv_idx])

                rv_rel = int(np.argmin(lows[h_idx: rs_idx + 1]))
                rv_idx = h_idx + rv_rel
                neckline_r = float(lows[rv_idx])

                neckline_slope_pct = abs(neckline_r - neckline_l) / max(neckline_l, 1e-9)
                if neckline_slope_pct > 0.10:
                    continue

                # Extrapolate neckline to current bar
                valley_span = max(rv_idx - lv_idx, 1)
                progress = (n - 1 - lv_idx) / valley_span
                neckline_now = neckline_l + progress * (neckline_r - neckline_l)

                last_close = float(closes[-1])
                status = "breakdown" if last_close < neckline_now else "forming"

                recency = rs_idx / n
                conf = (
                    0.38
                    + (1.0 - shoulder_diff_pct / 0.08) * 0.20
                    + min(head_dom * 4.0, 0.25)
                    + recency * 0.10
                )
                conf = float(min(max(conf, 0.0), 0.92))

                if conf > best_conf:
                    best_conf = conf
                    best = {
                        "pattern": "head_and_shoulders",
                        "start_date": dates[ls_idx],
                        "end_date": dates[rs_idx],
                        "status": status,
                        "cv_confidence": round(conf, 2),
                        "method": "cv_proxy_pivot",
                        "notes": f"LS {ls_price:.2f} H {h_price:.2f} RS {rs_price:.2f}",
                        # Line 1 (y0→y1): neckline between the two valleys only (no extrapolation)
                        # Line 2 (y0b→y1b): shoulder reference (horizontal at avg shoulder height)
                        "x0_date": dates[lv_idx],
                        "x1_date": dates[rv_idx],
                        "y0": float(neckline_l),
                        "y1": float(neckline_r),
                        "y0b": float(avg_shoulder),
                        "y1b": float(avg_shoulder),
                        "x_mid_date": dates[h_idx],
                        "y_mid": float(h_price),
                    }

    if best:
        out.append(best)
    return out


def _inv_head_shoulders_candidates(seg: pd.DataFrame, cfg: CVProxyConfig) -> list[dict]:
    """Detect inverse H&S: three troughs, middle (head) lowest, shoulders at similar depths."""
    out: list[dict] = []
    if len(seg) < 40:
        return out

    highs = seg["high"].to_numpy(dtype=float)
    lows = seg["low"].to_numpy(dtype=float)
    closes = seg["close"].to_numpy(dtype=float)
    dates = [_to_date_str(d) for d in seg["date"]]
    n = len(dates)

    troughs = _find_pivot_lows(lows, cfg.pivot_window)
    if len(troughs) < 3:
        return out

    troughs = troughs[-8:]

    best: dict | None = None
    best_conf = 0.0

    for i in range(len(troughs) - 2):
        for j in range(i + 1, len(troughs) - 1):
            for k in range(j + 1, len(troughs)):
                ls_idx, h_idx, rs_idx = troughs[i], troughs[j], troughs[k]
                ls_price = float(lows[ls_idx])
                h_price = float(lows[h_idx])
                rs_price = float(lows[rs_idx])

                # Head must be the deepest trough
                if not (h_price < ls_price and h_price < rs_price):
                    continue

                shoulder_diff_pct = abs(ls_price - rs_price) / max(ls_price, rs_price, 1e-9)
                if shoulder_diff_pct > 0.08:
                    continue

                avg_shoulder = (ls_price + rs_price) / 2.0
                head_dom = (avg_shoulder - h_price) / max(avg_shoulder, 1e-9)
                if head_dom < 0.03:
                    continue

                total_sep = rs_idx - ls_idx
                if total_sep < 20 or total_sep > 120:
                    continue

                # Neckline peaks: left between ls-head, right between head-rs
                lp_rel = int(np.argmax(highs[ls_idx: h_idx + 1]))
                lp_idx = ls_idx + lp_rel
                neckline_l = float(highs[lp_idx])

                rp_rel = int(np.argmax(highs[h_idx: rs_idx + 1]))
                rp_idx = h_idx + rp_rel
                neckline_r = float(highs[rp_idx])

                neckline_slope_pct = abs(neckline_r - neckline_l) / max(neckline_l, 1e-9)
                if neckline_slope_pct > 0.10:
                    continue

                peak_span = max(rp_idx - lp_idx, 1)
                progress = (n - 1 - lp_idx) / peak_span
                neckline_now = neckline_l + progress * (neckline_r - neckline_l)

                last_close = float(closes[-1])
                status = "breakout" if last_close > neckline_now else "forming"

                recency = rs_idx / n
                conf = (
                    0.38
                    + (1.0 - shoulder_diff_pct / 0.08) * 0.20
                    + min(head_dom * 4.0, 0.25)
                    + recency * 0.10
                )
                conf = float(min(max(conf, 0.0), 0.92))

                if conf > best_conf:
                    best_conf = conf
                    best = {
                        "pattern": "inv_head_and_shoulders",
                        "start_date": dates[ls_idx],
                        "end_date": dates[rs_idx],
                        "status": status,
                        "cv_confidence": round(conf, 2),
                        "method": "cv_proxy_pivot",
                        "notes": f"LS {ls_price:.2f} H {h_price:.2f} RS {rs_price:.2f}",
                        # Line 1 (y0→y1): neckline between the two peaks only (no extrapolation)
                        # Line 2 (y0b→y1b): shoulder reference (horizontal at avg shoulder trough)
                        "x0_date": dates[lp_idx],
                        "x1_date": dates[rp_idx],
                        "y0": float(neckline_l),
                        "y1": float(neckline_r),
                        "y0b": float(avg_shoulder),
                        "y1b": float(avg_shoulder),
                        "x_mid_date": dates[h_idx],
                        "y_mid": float(h_price),
                    }

    if best:
        out.append(best)
    return out


def detect_cv_proxy_patterns(df: pd.DataFrame, cfg: CVProxyConfig) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_cv()
    work = df.sort_values("date").reset_index(drop=True).copy()
    if len(work) < 30:
        return _empty_cv()

    rows: list[dict] = []

    shape_seg = work.tail(min(len(work), cfg.shape_bars)).copy()
    rows.extend(_shape_candidates(shape_seg, cfg))

    flag_seg = work.tail(min(len(work), cfg.flag_lookback_bars)).copy()
    rows.extend(_flag_candidates(flag_seg, cfg))

    double_seg = work.tail(min(len(work), cfg.double_lookback_bars)).copy()
    rows.extend(_double_top_candidates(double_seg, cfg))
    rows.extend(_double_bottom_candidates(double_seg, cfg))

    hs_seg = work.tail(min(len(work), cfg.hs_lookback_bars)).copy()
    rows.extend(_head_shoulders_candidates(hs_seg, cfg))
    rows.extend(_inv_head_shoulders_candidates(hs_seg, cfg))

    if not rows:
        return _empty_cv()

    out = pd.DataFrame(rows)
    # Ensure all schema columns exist (handles mixed dicts from different detectors)
    for col in CV_COLUMNS:
        if col not in out.columns:
            out[col] = None

    out = (
        out.drop_duplicates(subset=["pattern", "start_date", "end_date"])
        .sort_values("cv_confidence", ascending=False)
        .head(cfg.max_patterns)
        .reset_index(drop=True)
    )
    return out[CV_COLUMNS].copy()
