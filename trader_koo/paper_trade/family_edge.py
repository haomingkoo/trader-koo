"""Rolling family-level edge computation for the paper-trading bot.

Computes win rate, expectancy, and R-multiple per setup family (and
optionally per direction and regime) over a rolling window.  This data
feeds the bot's self-awareness: families with proven negative edge
should be flagged for demotion, and families with sustained positive
edge should get priority.
"""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from typing import Any

LOG = logging.getLogger(__name__)

_DEFAULT_WINDOW_DAYS = 60
_MIN_TRADES_FOR_EDGE = 5


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


def compute_family_edges(
    conn: sqlite3.Connection,
    *,
    window_days: int = _DEFAULT_WINDOW_DAYS,
    min_trades: int = _MIN_TRADES_FOR_EDGE,
    bot_version: str | None = None,
) -> list[dict[str, Any]]:
    """Compute rolling edge metrics per setup_family × direction.

    Returns a list of dicts, one per (family, direction) combination that
    has at least *min_trades* closed trades in the window.

    Each dict contains::

        {
            "setup_family": str,
            "direction": str,            # "long" | "short"
            "trade_count": int,
            "wins": int,
            "losses": int,
            "win_rate_pct": float,
            "avg_pnl_pct": float,        # expectancy
            "avg_r_multiple": float | None,
            "total_pnl_pct": float,
            "best_r": float | None,
            "worst_r": float | None,
            "target_hit_rate_pct": float,
            "stopped_out_rate_pct": float,
            "edge_label": str,           # "positive", "neutral", "negative"
            "window_days": int,
            "bot_version": str | None,
        }
    """
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=window_days)).strftime("%Y-%m-%d")

    version_clause = ""
    params: list[Any] = [cutoff]
    if bot_version:
        version_clause = "AND bot_version = ?"
        params.append(bot_version)

    rows = conn.execute(
        f"""
        SELECT
            setup_family,
            direction,
            COUNT(*) AS trade_count,
            SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN pnl_pct <= 0 THEN 1 ELSE 0 END) AS losses,
            AVG(pnl_pct) AS avg_pnl_pct,
            SUM(pnl_pct) AS total_pnl_pct,
            AVG(r_multiple) AS avg_r_multiple,
            MAX(r_multiple) AS best_r,
            MIN(r_multiple) AS worst_r,
            SUM(CASE WHEN exit_reason = 'target_hit' THEN 1 ELSE 0 END) AS target_hits,
            SUM(CASE WHEN exit_reason = 'stopped_out' THEN 1 ELSE 0 END) AS stopped_outs
        FROM paper_trades
        WHERE status != 'open'
          AND pnl_pct IS NOT NULL
          AND entry_date >= ?
          AND setup_family IS NOT NULL
          {version_clause}
        GROUP BY setup_family, direction
        HAVING COUNT(*) >= ?
        ORDER BY AVG(pnl_pct) DESC
        """,
        (*params, min_trades),
    ).fetchall()

    edges: list[dict[str, Any]] = []
    for r in rows:
        count = int(_value(r, "trade_count", 2) or 0)
        wins = int(_value(r, "wins", 3) or 0)
        avg_pnl = float(_value(r, "avg_pnl_pct", 5) or 0)
        avg_r_raw = _value(r, "avg_r_multiple", 7)
        avg_r = float(avg_r_raw) if avg_r_raw is not None else None

        if avg_pnl > 0.5:
            edge_label = "positive"
        elif avg_pnl < -0.5:
            edge_label = "negative"
        else:
            edge_label = "neutral"

        edges.append({
            "setup_family": _value(r, "setup_family", 0),
            "direction": _value(r, "direction", 1),
            "trade_count": count,
            "wins": wins,
            "losses": int(_value(r, "losses", 4) or 0),
            "win_rate_pct": round(wins / count * 100, 1) if count > 0 else 0,
            "avg_pnl_pct": round(avg_pnl, 2),
            "avg_r_multiple": round(avg_r, 2) if avg_r is not None else None,
            "total_pnl_pct": round(float(_value(r, "total_pnl_pct", 6) or 0), 2),
            "best_r": round(float(_value(r, "best_r", 8)), 2) if _value(r, "best_r", 8) is not None else None,
            "worst_r": round(float(_value(r, "worst_r", 9)), 2) if _value(r, "worst_r", 9) is not None else None,
            "target_hit_rate_pct": round(int(_value(r, "target_hits", 10) or 0) / count * 100, 1) if count > 0 else 0,
            "stopped_out_rate_pct": round(int(_value(r, "stopped_outs", 11) or 0) / count * 100, 1) if count > 0 else 0,
            "edge_label": edge_label,
            "window_days": window_days,
            "bot_version": bot_version,
        })

    return edges


def compute_regime_edges(
    conn: sqlite3.Connection,
    *,
    window_days: int = _DEFAULT_WINDOW_DAYS,
    min_trades: int = _MIN_TRADES_FOR_EDGE,
) -> list[dict[str, Any]]:
    """Compute edge metrics per regime state.

    Slices closed trades by ``regime_state_at_entry`` to show which
    market conditions produce the best/worst outcomes.
    """
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=window_days)).strftime("%Y-%m-%d")

    rows = conn.execute(
        """
        SELECT
            regime_state_at_entry,
            COUNT(*) AS trade_count,
            SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) AS wins,
            AVG(pnl_pct) AS avg_pnl_pct,
            AVG(r_multiple) AS avg_r_multiple
        FROM paper_trades
        WHERE status != 'open'
          AND pnl_pct IS NOT NULL
          AND entry_date >= ?
          AND regime_state_at_entry IS NOT NULL
        GROUP BY regime_state_at_entry
        HAVING COUNT(*) >= ?
        ORDER BY AVG(pnl_pct) DESC
        """,
        (cutoff, min_trades),
    ).fetchall()

    return [
        {
            "regime": _value(r, "regime_state_at_entry", 0),
            "trade_count": int(_value(r, "trade_count", 1) or 0),
            "wins": int(_value(r, "wins", 2) or 0),
            "win_rate_pct": round(int(_value(r, "wins", 2) or 0) / int(_value(r, "trade_count", 1) or 1) * 100, 1),
            "avg_pnl_pct": round(float(_value(r, "avg_pnl_pct", 3) or 0), 2),
            "avg_r_multiple": round(float(_value(r, "avg_r_multiple", 4)), 2) if _value(r, "avg_r_multiple", 4) is not None else None,
            "window_days": window_days,
        }
        for r in rows
    ]


def compute_vix_bucket_edges(
    conn: sqlite3.Connection,
    *,
    window_days: int = _DEFAULT_WINDOW_DAYS,
    min_trades: int = 3,
) -> list[dict[str, Any]]:
    """Compute edge metrics per VIX bucket at entry.

    Buckets: low (<15), normal (15-20), elevated (20-25), high (25-30), extreme (>30).
    """
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=window_days)).strftime("%Y-%m-%d")

    rows = conn.execute(
        """
        SELECT
            CASE
                WHEN vix_at_entry < 15 THEN 'low (<15)'
                WHEN vix_at_entry < 20 THEN 'normal (15-20)'
                WHEN vix_at_entry < 25 THEN 'elevated (20-25)'
                WHEN vix_at_entry < 30 THEN 'high (25-30)'
                ELSE 'extreme (>30)'
            END AS vix_bucket,
            COUNT(*) AS trade_count,
            SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) AS wins,
            AVG(pnl_pct) AS avg_pnl_pct,
            AVG(r_multiple) AS avg_r_multiple
        FROM paper_trades
        WHERE status != 'open'
          AND pnl_pct IS NOT NULL
          AND entry_date >= ?
          AND vix_at_entry IS NOT NULL
        GROUP BY vix_bucket
        HAVING COUNT(*) >= ?
        ORDER BY MIN(vix_at_entry)
        """,
        (cutoff, min_trades),
    ).fetchall()

    return [
        {
            "vix_bucket": _value(r, "vix_bucket", 0),
            "trade_count": int(_value(r, "trade_count", 1) or 0),
            "wins": int(_value(r, "wins", 2) or 0),
            "win_rate_pct": round(int(_value(r, "wins", 2) or 0) / int(_value(r, "trade_count", 1) or 1) * 100, 1),
            "avg_pnl_pct": round(float(_value(r, "avg_pnl_pct", 3) or 0), 2),
            "avg_r_multiple": round(float(_value(r, "avg_r_multiple", 4)), 2) if _value(r, "avg_r_multiple", 4) is not None else None,
            "window_days": window_days,
        }
        for r in rows
    ]


def generate_edge_feedback(
    family_edges: list[dict[str, Any]],
    regime_edges: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Generate actionable feedback items from edge data.

    Returns items like:
    - "bullish_breakout long has -1.2% avg expectancy over 60 days → consider demotion"
    - "bullish_engulfing long has +2.5% avg expectancy → benchmark family"
    """
    feedback: list[dict[str, Any]] = []

    # Top 3 strongest families
    positive = [e for e in family_edges if e["edge_label"] == "positive"]
    for e in positive[:3]:
        feedback.append({
            "kind": "family_strength",
            "severity": "green",
            "title": f"{e['setup_family']} ({e['direction']}) shows edge",
            "detail": (
                f"{e['win_rate_pct']}% win rate, {e['avg_pnl_pct']}% avg PnL, "
                f"{e['avg_r_multiple']}R avg over {e['trade_count']} trades ({e['window_days']}d)"
            ),
            "action": "Benchmark this family. Consider priority allocation.",
        })

    # Bottom 3 weakest families
    negative = [e for e in family_edges if e["edge_label"] == "negative"]
    for e in negative[:3]:
        feedback.append({
            "kind": "family_weakness",
            "severity": "red",
            "title": f"{e['setup_family']} ({e['direction']}) negative edge",
            "detail": (
                f"{e['win_rate_pct']}% win rate, {e['avg_pnl_pct']}% avg PnL, "
                f"{e['avg_r_multiple']}R avg over {e['trade_count']} trades ({e['window_days']}d)"
            ),
            "action": "Consider demoting or watch-only for this family.",
        })

    # Regime feedback
    if regime_edges:
        best_regime = max(regime_edges, key=lambda r: r["avg_pnl_pct"]) if regime_edges else None
        worst_regime = min(regime_edges, key=lambda r: r["avg_pnl_pct"]) if regime_edges else None
        if best_regime and best_regime["avg_pnl_pct"] > 0:
            feedback.append({
                "kind": "regime_strength",
                "severity": "green",
                "title": f"Best regime: {best_regime['regime']}",
                "detail": f"{best_regime['avg_pnl_pct']}% avg PnL over {best_regime['trade_count']} trades",
                "action": "Size up in this regime.",
            })
        if worst_regime and worst_regime["avg_pnl_pct"] < 0:
            feedback.append({
                "kind": "regime_weakness",
                "severity": "amber",
                "title": f"Worst regime: {worst_regime['regime']}",
                "detail": f"{worst_regime['avg_pnl_pct']}% avg PnL over {worst_regime['trade_count']} trades",
                "action": "Reduce exposure or tighten stops in this regime.",
            })

    return feedback
