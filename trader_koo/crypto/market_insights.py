"""Cross-asset and market-overview helpers for the crypto dashboard."""
from __future__ import annotations

import datetime as dt
import logging
import math
import sqlite3
from typing import Any

LOG = logging.getLogger("trader_koo.crypto.market_insights")


_DEFAULT_WINDOWS = (5, 10, 20)


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in (None, 0):
        return None
    return round(((current / previous) - 1.0) * 100.0, 2)


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=False))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return round(cov / math.sqrt(var_x * var_y), 3)


def _beta(xs: list[float], ys: list[float]) -> float | None:
    """Return beta of x relative to y."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=False))
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_y <= 0:
        return None
    return round(cov / var_y, 3)


def _fetch_equity_history(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    limit: int = 90,
) -> list[tuple[str, float]]:
    rows = conn.execute(
        """
        SELECT date, CAST(close AS REAL) AS close
        FROM price_daily
        WHERE ticker = ? AND close IS NOT NULL
        ORDER BY date DESC
        LIMIT ?
        """,
        (ticker, limit),
    ).fetchall()
    history = [(str(row[0]), float(row[1])) for row in rows if row[1] is not None]
    history.reverse()
    return history


def _daily_close_map(bars: list[Any]) -> dict[str, float]:
    daily: dict[str, float] = {}
    for bar in sorted(bars, key=lambda item: item.timestamp):
        daily[bar.timestamp.date().isoformat()] = float(bar.close)
    return daily


def _window_metrics(
    aligned_rows: list[dict[str, float | str]],
    *,
    window: int,
) -> dict[str, Any]:
    returns = aligned_rows[-(window + 1):]
    if len(returns) < window + 1:
        return {
            "window_days": window,
            "sample_size": max(len(returns) - 1, 0),
            "correlation": None,
            "beta": None,
            "asset_return_pct": None,
            "benchmark_return_pct": None,
            "relative_performance_pct": None,
            "ratio_zscore": None,
        }

    asset_rets = [
        float(((returns[idx]["asset_close"] / returns[idx - 1]["asset_close"]) - 1.0))
        for idx in range(1, len(returns))
    ]
    benchmark_rets = [
        float(((returns[idx]["benchmark_close"] / returns[idx - 1]["benchmark_close"]) - 1.0))
        for idx in range(1, len(returns))
    ]
    asset_return = _pct_change(
        _safe_float(returns[-1]["asset_close"]),
        _safe_float(returns[0]["asset_close"]),
    )
    benchmark_return = _pct_change(
        _safe_float(returns[-1]["benchmark_close"]),
        _safe_float(returns[0]["benchmark_close"]),
    )
    relative_perf = (
        round(asset_return - benchmark_return, 2)
        if asset_return is not None and benchmark_return is not None
        else None
    )

    ratios = [
        float(row["asset_close"]) / max(float(row["benchmark_close"]), 0.000001)
        for row in returns
    ]
    ratio_mean = sum(ratios) / len(ratios)
    ratio_var = sum((value - ratio_mean) ** 2 for value in ratios) / len(ratios)
    ratio_sd = math.sqrt(ratio_var)
    ratio_zscore = None
    if ratio_sd > 0:
        ratio_zscore = round((ratios[-1] - ratio_mean) / ratio_sd, 2)

    return {
        "window_days": window,
        "sample_size": len(asset_rets),
        "correlation": _correlation(asset_rets, benchmark_rets),
        "beta": _beta(asset_rets, benchmark_rets),
        "asset_return_pct": asset_return,
        "benchmark_return_pct": benchmark_return,
        "relative_performance_pct": relative_perf,
        "ratio_zscore": ratio_zscore,
    }


def _relationship_label(window_20: dict[str, Any]) -> str:
    corr = _safe_float(window_20.get("correlation"))
    relative = _safe_float(window_20.get("relative_performance_pct"))
    if corr is None:
        return "insufficient overlap"
    if corr >= 0.65 and relative is not None and relative >= 8:
        return "btc leading risk-on"
    if corr >= 0.65 and relative is not None and relative <= -8:
        return "equity-led risk-on"
    if corr <= 0.2 and relative is not None and abs(relative) <= 5:
        return "decoupling"
    if corr <= -0.2 and relative is not None and relative > 0:
        return "anti-risk divergence"
    if corr >= 0.45:
        return "risk assets moving together"
    if corr <= -0.2:
        return "inverse correlation"
    if corr <= 0.2:
        return "low correlation"
    return "weak positive correlation"


def build_btc_spy_correlation(
    conn: sqlite3.Connection,
    *,
    asset_symbol: str,
    benchmark_symbol: str,
    asset_bars: list[Any],
    windows: tuple[int, ...] = _DEFAULT_WINDOWS,
) -> dict[str, Any]:
    asset_daily = _daily_close_map(asset_bars)
    benchmark_history = _fetch_equity_history(conn, benchmark_symbol, limit=max(max(windows) * 3, 60))

    aligned_rows: list[dict[str, float | str]] = []
    for date_str, benchmark_close in benchmark_history:
        asset_close = asset_daily.get(date_str)
        if asset_close is None:
            continue
        aligned_rows.append(
            {
                "date": date_str,
                "asset_close": round(asset_close, 6),
                "benchmark_close": round(benchmark_close, 6),
            }
        )

    if len(aligned_rows) < 6:
        return {
            "symbol": asset_symbol,
            "benchmark": benchmark_symbol,
            "ok": False,
            "as_of": aligned_rows[-1]["date"] if aligned_rows else None,
            "sample_size": len(aligned_rows),
            "relationship_label": "insufficient overlap",
            "note": "Need at least 6 aligned daily closes between crypto and benchmark.",
            "windows": {f"{window}d": _window_metrics(aligned_rows, window=window) for window in windows},
            "aligned_history": [],
        }

    first_asset = float(aligned_rows[0]["asset_close"])
    first_benchmark = float(aligned_rows[0]["benchmark_close"])
    rebased_history = [
        {
            "date": str(row["date"]),
            "asset_close": float(row["asset_close"]),
            "benchmark_close": float(row["benchmark_close"]),
            "asset_rebased": round(float(row["asset_close"]) / first_asset * 100.0, 2),
            "benchmark_rebased": round(float(row["benchmark_close"]) / first_benchmark * 100.0, 2),
        }
        for row in aligned_rows[-20:]
        if first_asset > 0 and first_benchmark > 0
    ]

    windows_payload = {f"{window}d": _window_metrics(aligned_rows, window=window) for window in windows}
    window_20 = windows_payload.get("20d") or next(iter(windows_payload.values()))

    latest = aligned_rows[-1]
    previous = aligned_rows[-2] if len(aligned_rows) >= 2 else None

    return {
        "symbol": asset_symbol,
        "benchmark": benchmark_symbol,
        "ok": True,
        "as_of": str(latest["date"]),
        "sample_size": len(aligned_rows),
        "relationship_label": _relationship_label(window_20),
        "note": "Aligned on benchmark trading dates; crypto weekend closes are excluded from the overlap window.",
        "latest": {
            "asset_close": float(latest["asset_close"]),
            "benchmark_close": float(latest["benchmark_close"]),
            "asset_change_1d_pct": _pct_change(
                float(latest["asset_close"]),
                _safe_float(previous["asset_close"]) if previous else None,
            ),
            "benchmark_change_1d_pct": _pct_change(
                float(latest["benchmark_close"]),
                _safe_float(previous["benchmark_close"]) if previous else None,
            ),
        },
        "windows": windows_payload,
        "aligned_history": rebased_history,
    }


def _market_posture(
    *,
    bullish_count: int,
    total_count: int,
    avg_change_pct: float | None,
    avg_momentum: float | None,
) -> str:
    if total_count <= 0:
        return "unavailable"
    breadth = bullish_count / total_count
    if breadth >= 0.6 and (avg_change_pct or 0.0) > 0 and (avg_momentum or 0.0) > 0:
        return "broad crypto risk-on"
    if breadth <= 0.4 and (avg_change_pct or 0.0) < 0 and (avg_momentum or 0.0) < 0:
        return "broad crypto risk-off"
    if breadth >= 0.6:
        return "trend breadth positive"
    if breadth <= 0.4:
        return "trend breadth defensive"
    return "mixed rotation"


def _volatility_regime(avg_atr_pct: float | None, avg_realized_vol: float | None) -> str:
    atr = avg_atr_pct or 0.0
    realized = avg_realized_vol or 0.0
    if atr >= 4.0 or realized >= 4.0:
        return "expansion"
    if atr <= 1.5 and realized <= 1.5:
        return "compression"
    return "balanced"


def build_crypto_market_structure(
    *,
    interval: str,
    summaries: dict[str, dict[str, Any]],
    structures: list[dict[str, Any]],
) -> dict[str, Any]:
    symbol_rows: list[dict[str, Any]] = []
    bullish_count = 0
    at_support_count = 0
    at_resistance_count = 0
    atr_values: list[float] = []
    vol_values: list[float] = []
    momentum_values: list[float] = []
    change_values: list[float] = []

    for payload in structures:
        symbol = str(payload.get("symbol") or "").strip()
        if not symbol:
            continue
        context = payload.get("context") or {}
        summary = summaries.get(symbol) or {}
        change_pct = _safe_float(summary.get("change_pct_24h"))
        atr_pct = _safe_float(context.get("atr_pct"))
        realized_vol = _safe_float(context.get("realized_vol_20"))
        momentum_20 = _safe_float(context.get("momentum_20"))
        level_context = str(context.get("level_context") or "unavailable")
        ma_trend = str(context.get("ma_trend") or "unknown")
        hmm_regime = payload.get("hmm_regime") or {}
        hmm_state = hmm_regime.get("current_state")
        hmm_conf = None
        if isinstance(hmm_regime, dict) and hmm_state:
            current_probs = hmm_regime.get("current_probs") or {}
            hmm_conf = _safe_float(current_probs.get(hmm_state))

        if ma_trend == "bullish":
            bullish_count += 1
        if level_context == "at_support":
            at_support_count += 1
        if level_context == "at_resistance":
            at_resistance_count += 1
        if atr_pct is not None:
            atr_values.append(atr_pct)
        if realized_vol is not None:
            vol_values.append(realized_vol)
        if momentum_20 is not None:
            momentum_values.append(momentum_20)
        if change_pct is not None:
            change_values.append(change_pct)

        symbol_rows.append(
            {
                "symbol": symbol,
                "price": _safe_float(summary.get("price")) or _safe_float(context.get("latest_close")),
                "change_pct_24h": change_pct,
                "ma_trend": ma_trend,
                "level_context": level_context,
                "support_level": _safe_float(context.get("support_level")),
                "resistance_level": _safe_float(context.get("resistance_level")),
                "pct_to_support": _safe_float(context.get("pct_to_support")),
                "pct_to_resistance": _safe_float(context.get("pct_to_resistance")),
                "atr_pct": atr_pct,
                "realized_vol_20": realized_vol,
                "momentum_20": momentum_20,
                "range_position": _safe_float(context.get("range_position")),
                "hmm_state": str(hmm_state) if hmm_state else None,
                "hmm_confidence": round(hmm_conf * 100.0, 1) if hmm_conf is not None else None,
            }
        )

    symbol_rows.sort(key=lambda row: (_safe_float(row.get("change_pct_24h")) or -999.0), reverse=True)

    avg_change_pct = round(sum(change_values) / len(change_values), 2) if change_values else None
    avg_momentum = round(sum(momentum_values) / len(momentum_values), 2) if momentum_values else None
    avg_atr_pct = round(sum(atr_values) / len(atr_values), 2) if atr_values else None
    avg_realized_vol = round(sum(vol_values) / len(vol_values), 2) if vol_values else None
    total_count = len(symbol_rows)

    return {
        "ok": total_count > 0,
        "interval": interval,
        "as_of": None,
        "overview": {
            "tracked_symbols": total_count,
            "bullish_trend_count": bullish_count,
            "bearish_or_mixed_count": max(total_count - bullish_count, 0),
            "at_support_count": at_support_count,
            "at_resistance_count": at_resistance_count,
            "avg_change_pct_24h": avg_change_pct,
            "avg_momentum_20": avg_momentum,
            "avg_atr_pct": avg_atr_pct,
            "avg_realized_vol_20": avg_realized_vol,
            "volatility_regime": _volatility_regime(avg_atr_pct, avg_realized_vol),
            "market_posture": _market_posture(
                bullish_count=bullish_count,
                total_count=total_count,
                avg_change_pct=avg_change_pct,
                avg_momentum=avg_momentum,
            ),
        },
        "leaders": symbol_rows[:2],
        "laggards": list(reversed(symbol_rows[-2:])) if symbol_rows else [],
        "symbols": symbol_rows,
    }


# ── Correlation Regime Tracking ─────────────────────────────────────────────

def ensure_correlation_snapshot_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crypto_correlation_snapshots (
            snapshot_date TEXT NOT NULL,
            asset TEXT NOT NULL,
            benchmark TEXT NOT NULL,
            corr_5d REAL,
            corr_10d REAL,
            corr_20d REAL,
            beta_20d REAL,
            relationship_label TEXT,
            PRIMARY KEY (snapshot_date, asset, benchmark)
        )
    """)
    conn.commit()


def save_correlation_snapshot(
    conn: sqlite3.Connection,
    *,
    asset: str,
    benchmark: str,
    correlation_data: dict[str, Any],
) -> None:
    """Persist today's correlation values for regime change tracking."""
    ensure_correlation_snapshot_table(conn)
    today = dt.date.today().isoformat()
    windows = correlation_data.get("windows", {})
    conn.execute(
        """
        INSERT OR REPLACE INTO crypto_correlation_snapshots
            (snapshot_date, asset, benchmark, corr_5d, corr_10d, corr_20d, beta_20d, relationship_label)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            today,
            asset,
            benchmark,
            _safe_float((windows.get("5d") or {}).get("correlation")),
            _safe_float((windows.get("10d") or {}).get("correlation")),
            _safe_float((windows.get("20d") or {}).get("correlation")),
            _safe_float((windows.get("20d") or {}).get("beta")),
            correlation_data.get("relationship_label"),
        ),
    )
    conn.commit()


def detect_correlation_regime_change(
    conn: sqlite3.Connection,
    *,
    asset: str,
    benchmark: str,
    current_label: str,
    current_corr_20d: float | None,
) -> dict[str, Any] | None:
    """Check if the correlation regime changed from the previous snapshot."""
    ensure_correlation_snapshot_table(conn)
    today = dt.date.today().isoformat()
    prev = conn.execute(
        """
        SELECT snapshot_date, relationship_label, corr_20d
        FROM crypto_correlation_snapshots
        WHERE asset = ? AND benchmark = ? AND snapshot_date < ?
        ORDER BY snapshot_date DESC
        LIMIT 1
        """,
        (asset, benchmark, today),
    ).fetchone()
    if prev is None:
        return None
    prev_label = str(prev[1] or "")
    prev_corr = _safe_float(prev[2])
    if not prev_label or prev_label == current_label:
        return None

    # Significant correlation shift
    corr_delta = None
    if current_corr_20d is not None and prev_corr is not None:
        corr_delta = round(current_corr_20d - prev_corr, 3)

    return {
        "changed": True,
        "from_label": prev_label,
        "to_label": current_label,
        "from_date": str(prev[0]),
        "corr_delta": corr_delta,
        "signal": f"{asset} vs {benchmark}: regime shifted from '{prev_label}' to '{current_label}'",
    }
