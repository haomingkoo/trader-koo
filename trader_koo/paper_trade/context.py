"""Market context capture for paper trades.

Snapshots regime, VIX, and breadth state at trade creation time so
that post-trade analysis can slice performance by market conditions.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any

LOG = logging.getLogger(__name__)


def _value(row: Any, key: str, index: int) -> Any:
    if row is None:
        return None
    try:
        return row[key]
    except Exception:
        try:
            return row[index]
        except Exception:
            return None


def capture_market_context(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return a dict of market-context fields to store alongside a paper trade.

    Reads from the same DB that the ingest pipeline writes to:
    - ``^VIX`` latest close → ``vix_at_entry``
    - VIX percentile (rank vs 252-day history) → ``vix_percentile_at_entry``
    - Simple regime label from VIX + SPY momentum → ``regime_state_at_entry``
    - HMM regime for SPY (if available) → ``hmm_regime_at_entry``
    """
    ctx: dict[str, Any] = {
        "bot_version": None,  # filled by caller from config
        "vix_at_entry": None,
        "vix_percentile_at_entry": None,
        "regime_state_at_entry": None,
        "hmm_regime_at_entry": None,
        "hmm_confidence_at_entry": None,
        "directional_regime_at_entry": None,
        "directional_regime_confidence": None,
    }

    try:
        # VIX latest close
        vix_row = conn.execute(
            """
            SELECT CAST(close AS REAL) AS close
            FROM price_daily
            WHERE ticker = '^VIX' AND close IS NOT NULL
            ORDER BY date DESC LIMIT 1
            """,
        ).fetchone()
        vix_close_raw = _value(vix_row, "close", 0)
        if vix_row and vix_close_raw is not None:
            vix = float(vix_close_raw)
            ctx["vix_at_entry"] = round(vix, 2)

            # VIX percentile (rank within last 252 trading days)
            vix_history = conn.execute(
                """
                SELECT CAST(close AS REAL) AS close
                FROM price_daily
                WHERE ticker = '^VIX' AND close IS NOT NULL
                ORDER BY date DESC LIMIT 252
                """,
            ).fetchall()
            if len(vix_history) >= 20:
                below = sum(1 for r in vix_history if float(_value(r, "close", 0)) <= vix)
                ctx["vix_percentile_at_entry"] = round(below / len(vix_history) * 100, 1)

            # Simple regime from VIX level
            if vix < 15:
                vix_regime = "low_vol"
            elif vix < 20:
                vix_regime = "normal"
            elif vix < 25:
                vix_regime = "elevated"
            elif vix < 30:
                vix_regime = "high"
            else:
                vix_regime = "extreme"
        else:
            vix_regime = "unknown"

        # SPY momentum (above/below 50-day MA)
        spy_rows = conn.execute(
            """
            SELECT CAST(close AS REAL) AS close
            FROM price_daily
            WHERE ticker = 'SPY' AND close IS NOT NULL
            ORDER BY date DESC LIMIT 50
            """,
        ).fetchall()
        if len(spy_rows) >= 50:
            spy_current = float(_value(spy_rows[0], "close", 0))
            spy_ma50 = sum(float(_value(r, "close", 0)) for r in spy_rows) / len(spy_rows)
            trend = "bull" if spy_current > spy_ma50 else "bear"
        elif len(spy_rows) >= 2:
            trend = "bull" if float(_value(spy_rows[0], "close", 0)) > float(_value(spy_rows[-1], "close", 0)) else "bear"
        else:
            trend = "unknown"

        ctx["regime_state_at_entry"] = f"{trend}_{vix_regime}"

        # HMM regime (if the structure module is available and SPY has data)
        try:
            from trader_koo.structure.hmm_regime import predict_regimes

            spy_ohlcv = conn.execute(
                """
                SELECT date, open, high, low, close, volume
                FROM price_daily
                WHERE ticker = 'SPY' AND close IS NOT NULL
                ORDER BY date ASC
                """,
            ).fetchall()
            if len(spy_ohlcv) >= 60:
                import pandas as pd

                spy_df = pd.DataFrame(
                    spy_ohlcv,
                    columns=["date", "open", "high", "low", "close", "volume"],
                )
                result = predict_regimes(spy_df, ticker="SPY")
                if result:
                    ctx["hmm_regime_at_entry"] = str(result.get("current_state", ""))
                    probs = result.get("current_probs") or {}
                    current_label = result.get("current_state", "")
                    ctx["hmm_confidence_at_entry"] = probs.get(current_label)
        except Exception as exc:
            LOG.debug("HMM regime capture skipped: %s", exc)

        # Directional HMM (bullish/chop/bearish) — same as crypto uses
        try:
            from trader_koo.structure.hmm_regime import predict_directional_regimes

            # Reuse spy_ohlcv from above if available, otherwise re-fetch
            if "spy_df" not in dir():
                spy_ohlcv_dir = conn.execute(
                    "SELECT date, open, high, low, close, volume "
                    "FROM price_daily WHERE ticker = 'SPY' AND close IS NOT NULL "
                    "ORDER BY date ASC",
                ).fetchall()
                if len(spy_ohlcv_dir) >= 60:
                    import pandas as pd
                    spy_df = pd.DataFrame(
                        spy_ohlcv_dir,
                        columns=["date", "open", "high", "low", "close", "volume"],
                    )
            if "spy_df" in dir() and spy_df is not None:
                dir_result = predict_directional_regimes(spy_df, ticker="SPY")
                if dir_result:
                    ctx["directional_regime_at_entry"] = str(dir_result.get("current_state", ""))
                    dir_probs = dir_result.get("current_probs") or {}
                    dir_label = dir_result.get("current_state", "")
                    ctx["directional_regime_confidence"] = dir_probs.get(dir_label)
        except Exception as exc:
            LOG.debug("Directional HMM capture skipped: %s", exc)

    except Exception as exc:
        LOG.warning("Market context capture failed (non-fatal): %s", exc)

    return ctx
