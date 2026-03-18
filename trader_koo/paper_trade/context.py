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

            if len(spy_rows) >= 50:
                closes = [float(_value(r, "close", 0)) for r in reversed(spy_rows)]
                regimes = predict_regimes(closes)
                if regimes:
                    latest = regimes[-1]
                    ctx["hmm_regime_at_entry"] = str(latest.get("regime", ""))
                    ctx["hmm_confidence_at_entry"] = latest.get("confidence")
        except Exception as exc:
            LOG.debug("HMM regime capture skipped: %s", exc)

    except Exception as exc:
        LOG.warning("Market context capture failed (non-fatal): %s", exc)

    return ctx
