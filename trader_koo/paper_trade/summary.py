"""Summary and query helpers for paper trades."""

from __future__ import annotations

import datetime as dt
import logging
import math
import sqlite3
from typing import Any

from trader_koo.paper_trade.config import PaperTradeConfig, config_snapshot
from trader_koo.paper_trade.schema import decode_json_list, ensure_paper_trade_schema

LOG = logging.getLogger(__name__)


def update_portfolio_snapshot(conn: sqlite3.Connection) -> None:
    """Compute and persist daily portfolio metrics."""
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    open_count = conn.execute(
        "SELECT COUNT(*) FROM paper_trades WHERE status = 'open'"
    ).fetchone()[0]

    closed_rows = conn.execute(
        "SELECT pnl_pct, r_multiple FROM paper_trades "
        "WHERE status != 'open' AND pnl_pct IS NOT NULL "
        "ORDER BY exit_date, id"
    ).fetchall()

    total_closed = len(closed_rows)
    if total_closed == 0:
        conn.execute(
            """
            INSERT OR REPLACE INTO paper_portfolio_snapshots
                (snapshot_date, open_trades, closed_trades_total, wins, losses,
                 equity_index)
            VALUES (?, ?, 0, 0, 0, 100.0)
            """,
            (today, open_count),
        )
        return

    pnls = [float(r[0]) for r in closed_rows]
    r_mults = [float(r[1]) for r in closed_rows if r[1] is not None]

    wins = sum(1 for p in pnls if p > 0)
    losses = total_closed - wins
    win_rate = round((wins / total_closed) * 100.0, 1) if total_closed > 0 else 0.0
    avg_pnl = round(sum(pnls) / len(pnls), 2)
    avg_r = round(sum(r_mults) / len(r_mults), 2) if r_mults else None
    total_pnl = round(sum(pnls), 2)

    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)
    max_dd = round(max_dd, 2)

    if len(pnls) > 1:
        mean_p = sum(pnls) / len(pnls)
        var_p = sum((pnl - mean_p) ** 2 for pnl in pnls) / (len(pnls) - 1)
        std_p = math.sqrt(var_p) if var_p > 0 else 0
        sharpe = round(mean_p / std_p, 2) if std_p > 0 else None
    else:
        sharpe = None

    gross_win = sum(pnl for pnl in pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else None

    best_trade = round(max(pnls), 2)
    worst_trade = round(min(pnls), 2)

    equity = 100.0
    for pnl in pnls:
        equity *= (1.0 + pnl / 100.0)
    equity = round(equity, 2)

    conn.execute(
        """
        INSERT OR REPLACE INTO paper_portfolio_snapshots
            (snapshot_date, open_trades, closed_trades_total, wins, losses,
             win_rate_pct, avg_pnl_pct, avg_r_multiple, total_pnl_pct,
             max_drawdown_pct, sharpe_ratio, profit_factor, equity_index,
             best_trade_pct, worst_trade_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (today, open_count, total_closed, wins, losses,
         win_rate, avg_pnl, avg_r, total_pnl,
         max_dd, sharpe, profit_factor, equity, best_trade, worst_trade),
    )


def recent_trades(conn: sqlite3.Connection, *, limit: int = 20) -> list[dict[str, Any]]:
    """Return most recent paper trades."""
    rows = conn.execute(
        """
        SELECT id, ticker, direction, entry_price, entry_date,
               target_price, stop_loss, exit_price, exit_date,
               status, pnl_pct, r_multiple, unrealized_pnl_pct,
               setup_family, setup_tier, score, exit_reason,
               decision_state, decision_summary,
               position_size_pct, risk_budget_pct, stop_distance_pct,
               expected_reward_pct, expected_r_multiple,
               entry_plan, exit_plan, sizing_summary,
               review_status, review_summary
        FROM paper_trades
        ORDER BY COALESCE(exit_date, entry_date) DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    keys = [
        "id", "ticker", "direction", "entry_price", "entry_date",
        "target_price", "stop_loss", "exit_price", "exit_date",
        "status", "pnl_pct", "r_multiple", "unrealized_pnl_pct",
        "setup_family", "setup_tier", "score", "exit_reason",
        "decision_state", "decision_summary",
        "position_size_pct", "risk_budget_pct", "stop_distance_pct",
        "expected_reward_pct", "expected_r_multiple",
        "entry_plan", "exit_plan", "sizing_summary",
        "review_status", "review_summary",
    ]
    return [dict(zip(keys, row)) for row in rows]


def _policy_snapshot(config: PaperTradeConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return config_snapshot(config)


def _feedback_items(
    *,
    overall: dict[str, Any],
    by_direction: dict[str, dict[str, Any]],
    by_family: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    expectancy = overall.get("expectancy_pct")
    total = overall.get("total_trades")
    if isinstance(expectancy, (int, float)) and isinstance(total, int) and total >= 5:
        if expectancy < 0:
            items.append(
                {
                    "kind": "risk",
                    "severity": "high",
                    "title": "Overall expectancy is still negative",
                    "detail": f"Closed paper trades are averaging {expectancy:.2f}% across {total} trades.",
                    "action": "Tighten setup-ready criteria and keep position size conservative until expectancy recovers.",
                }
            )
        else:
            items.append(
                {
                    "kind": "edge",
                    "severity": "green",
                    "title": "Overall expectancy is positive",
                    "detail": f"Closed paper trades are averaging +{expectancy:.2f}% across {total} trades.",
                    "action": "Keep the playbook stable and focus on whether the positive edge survives a larger sample.",
                }
            )

    for direction in ("long", "short"):
        stats = by_direction.get(direction)
        if not stats:
            continue
        sample = stats.get("total")
        avg_pnl = stats.get("avg_pnl_pct")
        if not isinstance(sample, int) or sample < 4 or not isinstance(avg_pnl, (int, float)):
            continue
        if avg_pnl < 0:
            items.append(
                {
                    "kind": "direction",
                    "severity": "amber",
                    "title": f"{direction.title()} setups are underperforming",
                    "detail": f"{direction.title()} paper trades are averaging {avg_pnl:.2f}% across {sample} closed trades.",
                    "action": f"Reduce {direction} size or require cleaner confirmation until the sample turns positive.",
                }
            )

    weak_families = [
        (family, stats)
        for family, stats in by_family.items()
        if isinstance(stats.get("total"), int)
        and stats["total"] >= 4
        and isinstance(stats.get("avg_pnl_pct"), (int, float))
        and stats["avg_pnl_pct"] < 0
    ]
    weak_families.sort(key=lambda item: item[1]["avg_pnl_pct"])
    for family, stats in weak_families[:2]:
        items.append(
            {
                "kind": "family",
                "severity": "amber",
                "title": f"{family.replace('_', ' ')} is weak",
                "detail": (
                    f"Expectancy is {stats['avg_pnl_pct']:.2f}% with "
                    f"{stats['win_rate_pct']:.1f}% win rate over {stats['total']} trades."
                ),
                "action": "Downgrade this family to watch-only by default or demand stronger level/trend confirmation.",
            }
        )

    strong_families = [
        (family, stats)
        for family, stats in by_family.items()
        if isinstance(stats.get("total"), int)
        and stats["total"] >= 4
        and isinstance(stats.get("avg_pnl_pct"), (int, float))
        and stats["avg_pnl_pct"] > 0
    ]
    strong_families.sort(key=lambda item: item[1]["avg_pnl_pct"], reverse=True)
    for family, stats in strong_families[:2]:
        items.append(
            {
                "kind": "family",
                "severity": "green",
                "title": f"{family.replace('_', ' ')} is working",
                "detail": (
                    f"Expectancy is +{stats['avg_pnl_pct']:.2f}% with "
                    f"{stats['win_rate_pct']:.1f}% win rate over {stats['total']} trades."
                ),
                "action": "Use this family as the benchmark playbook and compare weaker setups against it.",
            }
        )

    return items[:6]


def paper_trade_summary(
    conn: sqlite3.Connection,
    *,
    window_days: int = 180,
    config: PaperTradeConfig | None = None,
) -> dict[str, Any]:
    """Return comprehensive paper trading performance metrics."""
    ensure_paper_trade_schema(conn)

    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=window_days)).strftime("%Y-%m-%d")

    all_closed = conn.execute(
        """
        SELECT pnl_pct, r_multiple, direction, setup_family, setup_tier, exit_reason
        FROM paper_trades
        WHERE status != 'open' AND pnl_pct IS NOT NULL AND entry_date >= ?
        ORDER BY exit_date
        """,
        (cutoff,),
    ).fetchall()

    open_trades = conn.execute(
        "SELECT COUNT(*) FROM paper_trades WHERE status = 'open'"
    ).fetchone()[0]

    total = len(all_closed)
    if total == 0:
        return {
            "overall": {"total_trades": 0, "open_count": open_trades},
            "by_direction": {},
            "by_family": {},
            "by_tier": {},
            "by_exit_reason": {},
            "equity_curve": [],
            "recent_trades": recent_trades(conn, limit=20),
            "policy": _policy_snapshot(config),
            "feedback": [],
        }

    pnls = [float(row[0]) for row in all_closed]
    r_mults = [float(row[1]) for row in all_closed if row[1] is not None]
    wins = sum(1 for pnl in pnls if pnl > 0)
    win_pnls = [pnl for pnl in pnls if pnl > 0]
    loss_pnls = [pnl for pnl in pnls if pnl < 0]
    avg_win = round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else None
    avg_loss = round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else None
    payoff_ratio = (
        round(abs(avg_win / avg_loss), 2)
        if isinstance(avg_win, (int, float)) and isinstance(avg_loss, (int, float)) and avg_loss != 0
        else None
    )
    gross_win = sum(win_pnls)
    gross_loss = abs(sum(loss_pnls))
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else None
    hit_target_count = sum(1 for row in all_closed if row[5] == "target_hit")
    stopped_out_count = sum(1 for row in all_closed if row[5] == "stopped_out")

    overall = {
        "total_trades": total,
        "open_count": open_trades,
        "wins": wins,
        "losses": total - wins,
        "win_rate_pct": round(wins / total * 100, 1),
        "avg_pnl_pct": round(sum(pnls) / total, 2),
        "expectancy_pct": round(sum(pnls) / total, 2),
        "avg_r_multiple": round(sum(r_mults) / len(r_mults), 2) if r_mults else None,
        "total_pnl_pct": round(sum(pnls), 2),
        "best_trade_pct": round(max(pnls), 2),
        "worst_trade_pct": round(min(pnls), 2),
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        "target_hit_rate_pct": round(hit_target_count / total * 100, 1),
        "stopped_out_rate_pct": round(stopped_out_count / total * 100, 1),
    }

    by_direction: dict[str, dict[str, Any]] = {}
    for direction in ("long", "short"):
        direction_rows = [row for row in all_closed if row[2] == direction]
        if direction_rows:
            direction_pnls = [float(row[0]) for row in direction_rows]
            direction_wins = sum(1 for pnl in direction_pnls if pnl > 0)
            by_direction[direction] = {
                "total": len(direction_rows),
                "wins": direction_wins,
                "win_rate_pct": round(direction_wins / len(direction_rows) * 100, 1),
                "avg_pnl_pct": round(sum(direction_pnls) / len(direction_pnls), 2),
            }

    by_family: dict[str, dict[str, Any]] = {}
    families = set(row[3] for row in all_closed if row[3])
    for family in sorted(families):
        family_rows = [row for row in all_closed if row[3] == family]
        family_pnls = [float(row[0]) for row in family_rows]
        family_wins = sum(1 for pnl in family_pnls if pnl > 0)
        by_family[family] = {
            "total": len(family_rows),
            "wins": family_wins,
            "win_rate_pct": round(family_wins / len(family_rows) * 100, 1),
            "avg_pnl_pct": round(sum(family_pnls) / len(family_pnls), 2),
        }

    by_tier: dict[str, dict[str, Any]] = {}
    tiers = set(row[4] for row in all_closed if row[4])
    for tier in sorted(tiers):
        tier_rows = [row for row in all_closed if row[4] == tier]
        tier_pnls = [float(row[0]) for row in tier_rows]
        tier_wins = sum(1 for pnl in tier_pnls if pnl > 0)
        by_tier[tier] = {
            "total": len(tier_rows),
            "wins": tier_wins,
            "win_rate_pct": round(tier_wins / len(tier_rows) * 100, 1),
            "avg_pnl_pct": round(sum(tier_pnls) / len(tier_pnls), 2),
        }

    by_exit_reason: dict[str, int] = {}
    for row in all_closed:
        reason = row[5] or "unknown"
        by_exit_reason[reason] = by_exit_reason.get(reason, 0) + 1

    eq_rows = conn.execute(
        """
        SELECT snapshot_date, equity_index, open_trades, closed_trades_total
        FROM paper_portfolio_snapshots
        WHERE snapshot_date >= ?
        ORDER BY snapshot_date
        """,
        (cutoff,),
    ).fetchall()
    equity_curve = [
        {
            "date": row[0],
            "equity_index": row[1],
            "open_trades": row[2],
            "closed_total": row[3],
        }
        for row in eq_rows
    ]

    # Family edge, regime edge, VIX bucket analysis
    family_edges: list[dict[str, Any]] = []
    regime_edges: list[dict[str, Any]] = []
    vix_bucket_edges: list[dict[str, Any]] = []
    edge_feedback: list[dict[str, Any]] = []
    try:
        from trader_koo.paper_trade.family_edge import (
            compute_family_edges,
            compute_regime_edges,
            compute_vix_bucket_edges,
            generate_edge_feedback,
        )

        family_edges = compute_family_edges(conn, window_days=window_days)
        regime_edges = compute_regime_edges(conn, window_days=window_days)
        vix_bucket_edges = compute_vix_bucket_edges(conn, window_days=window_days)
        edge_feedback = generate_edge_feedback(family_edges, regime_edges)
    except Exception as exc:
        LOG.warning("Edge computation failed (non-fatal): %s", exc)

    base_feedback = _feedback_items(
        overall=overall,
        by_direction=by_direction,
        by_family=by_family,
    )

    return {
        "overall": overall,
        "by_direction": by_direction,
        "by_family": by_family,
        "by_tier": by_tier,
        "by_exit_reason": by_exit_reason,
        "equity_curve": equity_curve,
        "recent_trades": recent_trades(conn, limit=20),
        "policy": _policy_snapshot(config),
        "feedback": base_feedback + edge_feedback,
        "family_edges": family_edges,
        "regime_edges": regime_edges,
        "vix_bucket_edges": vix_bucket_edges,
    }


def list_paper_trades(
    conn: sqlite3.Connection,
    *,
    status: str = "all",
    ticker: str | None = None,
    direction: str | None = None,
    family: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List paper trades with optional filters."""
    ensure_paper_trade_schema(conn)

    clauses: list[str] = []
    params: list[Any] = []

    if status != "all":
        clauses.append("status = ?")
        params.append(status)
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.upper().strip())
    if direction:
        clauses.append("direction = ?")
        params.append(direction)
    if family:
        clauses.append("setup_family = ?")
        params.append(family)
    if from_date:
        clauses.append("entry_date >= ?")
        params.append(from_date)
    if to_date:
        clauses.append("entry_date <= ?")
        params.append(to_date)

    where = " AND ".join(clauses) if clauses else "1=1"
    params.append(limit)

    rows = conn.execute(
        f"""
        SELECT id, report_date, ticker, direction, entry_price, entry_date,
               target_price, stop_loss, exit_price, exit_date, exit_reason,
               status, current_price, unrealized_pnl_pct, pnl_pct, r_multiple,
               setup_family, setup_tier, score, signal_bias, actionability,
               observation, action_text, risk_note, debate_agreement_score,
               high_water_mark, low_water_mark, decision_version,
               decision_state, analyst_stage, debate_stage, risk_stage,
               portfolio_decision, decision_summary, decision_reasons, risk_flags,
               position_size_pct, risk_budget_pct, stop_distance_pct,
               expected_reward_pct, expected_r_multiple,
               entry_plan, exit_plan, sizing_summary,
               review_status, review_summary
        FROM paper_trades
        WHERE {where}
        ORDER BY entry_date DESC, id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()

    keys = [
        "id", "report_date", "ticker", "direction", "entry_price", "entry_date",
        "target_price", "stop_loss", "exit_price", "exit_date", "exit_reason",
        "status", "current_price", "unrealized_pnl_pct", "pnl_pct", "r_multiple",
        "setup_family", "setup_tier", "score", "signal_bias", "actionability",
        "observation", "action_text", "risk_note", "debate_agreement_score",
        "high_water_mark", "low_water_mark", "decision_version",
        "decision_state", "analyst_stage", "debate_stage", "risk_stage",
        "portfolio_decision", "decision_summary", "decision_reasons", "risk_flags",
        "position_size_pct", "risk_budget_pct", "stop_distance_pct",
        "expected_reward_pct", "expected_r_multiple",
        "entry_plan", "exit_plan", "sizing_summary",
        "review_status", "review_summary",
    ]
    trades: list[dict[str, Any]] = []
    for row in rows:
        trade = dict(zip(keys, row))
        trade["decision_reasons"] = decode_json_list(trade.get("decision_reasons"))
        trade["risk_flags"] = decode_json_list(trade.get("risk_flags"))
        trades.append(trade)
    return trades
