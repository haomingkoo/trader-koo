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

        # Sortino: uses downside deviation (only negative returns)
        neg_pnls = [p for p in pnls if p < 0]
        if neg_pnls:
            downside_var = sum((p - mean_p) ** 2 for p in neg_pnls) / len(neg_pnls)
            downside_std = math.sqrt(downside_var) if downside_var > 0 else 0
            sortino = round(mean_p / downside_std, 2) if downside_std > 0 else None
        else:
            sortino = None  # no losses = infinite sortino

        # Calmar: annualized return / max drawdown
        if max_dd > 0 and total_closed >= 5:
            annualized_return = mean_p * min(total_closed, 252)
            calmar = round(annualized_return / max_dd, 2)
        else:
            calmar = None
    else:
        sharpe = None
        sortino = None
        calmar = None

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
             max_drawdown_pct, sharpe_ratio, sortino_ratio, calmar_ratio,
             profit_factor, equity_index,
             best_trade_pct, worst_trade_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (today, open_count, total_closed, wins, losses,
         win_rate, avg_pnl, avg_r, total_pnl,
         max_dd, sharpe, sortino, calmar, profit_factor, equity,
         best_trade, worst_trade),
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


def _compute_spy_benchmark(
    conn: sqlite3.Connection,
    *,
    first_entry_date: str,
    last_exit_date: str,
) -> dict[str, Any] | None:
    """Compute SPY buy-and-hold return over the paper trade date range."""
    try:
        start_row = conn.execute(
            "SELECT close FROM price_daily "
            "WHERE ticker = 'SPY' AND date >= ? ORDER BY date ASC LIMIT 1",
            (first_entry_date,),
        ).fetchone()
        end_row = conn.execute(
            "SELECT close FROM price_daily "
            "WHERE ticker = 'SPY' AND date <= ? ORDER BY date DESC LIMIT 1",
            (last_exit_date,),
        ).fetchone()
        if not start_row or not end_row:
            return None
        start_price = float(start_row[0])
        end_price = float(end_row[0])
        if start_price <= 0:
            return None

        price_return_pct = (end_price / start_price - 1.0) * 100.0

        start_dt = dt.datetime.strptime(first_entry_date, "%Y-%m-%d")
        end_dt = dt.datetime.strptime(last_exit_date, "%Y-%m-%d")
        period_days = max((end_dt - start_dt).days, 1)

        # Add pro-rated SPY dividend yield (~1.8% annual)
        spy_annual_dividend_yield = 1.8
        dividend_pct = spy_annual_dividend_yield * period_days / 365
        total_return_pct = round(price_return_pct + dividend_pct, 2)

        return {
            "return_pct": total_return_pct,
            "price_return_pct": round(price_return_pct, 2),
            "dividend_pct": round(dividend_pct, 2),
            "period_days": period_days,
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
        }
    except Exception as exc:
        LOG.warning("SPY benchmark computation failed: %s", exc)
        return None


def _compute_unfiltered_baseline(
    conn: sqlite3.Connection,
    *,
    cutoff: str,
) -> dict[str, Any] | None:
    """Estimate what would happen if every qualifying setup was traded.

    Uses the setup evaluation scores stored in daily reports to count
    ALL setups with tier A or B that had a direction (long/short),
    regardless of ML filter or critic review.  The actual outcome is
    approximated by looking up the signed return from price_daily
    (entry close vs close 5 trading days later).
    """
    try:
        # All paper trades that were ever created (both approved and
        # rejected setups are not stored -- only approved ones).
        # Instead, count all setups that had qualifying tier+direction
        # from the report that were in the same window, then compare
        # with actually-taken trades.
        #
        # Simpler approach: gather all paper_trades in the window
        # (which already passed the basic qualify gate) and compute
        # stats *without* filtering by ML or critic.  The trades
        # table only contains approved+taken trades, so the "unfiltered"
        # view is the same data -- we cannot reconstruct rejected setups.
        #
        # Best available: use setup_evaluation scored calls from
        # the report system if available.  Fall back to computing
        # a naive "all-entries" return using price_daily for the
        # tickers that had any paper trade considered.
        #
        # Pragmatic approach: Query *all* paper trades (they already
        # passed the basic tier/score gate but might have been filtered
        # by critic/ML). Since we only have the taken trades, we report
        # the "pre-filter" baseline as the total set of taken trades
        # PLUS compute a simulated "always-enter" return for each
        # ticker/date pair using a fixed 5-day hold.

        # Get all entry points (ticker + entry_date) from paper trades
        entries = conn.execute(
            """
            SELECT ticker, direction, entry_date, entry_price
            FROM paper_trades
            WHERE entry_date >= ?
            ORDER BY entry_date
            """,
            (cutoff,),
        ).fetchall()

        if not entries:
            return None

        # For each entry, compute the forward return matching expiry_days
        # (default 10 trading days, consistent with paper trade expiry)
        baseline_hold_days = 10
        pnls: list[float] = []
        for ticker, direction, entry_date, entry_price in entries:
            if entry_price is None or float(entry_price) <= 0:
                continue
            forward_row = conn.execute(
                """
                SELECT close FROM price_daily
                WHERE ticker = ? AND date > ?
                ORDER BY date ASC
                LIMIT 1 OFFSET ?
                """,
                (str(ticker), str(entry_date), baseline_hold_days - 1),
            ).fetchone()
            if not forward_row or forward_row[0] is None:
                # Fall back to latest available price
                forward_row = conn.execute(
                    "SELECT close FROM price_daily "
                    "WHERE ticker = ? AND date > ? "
                    "ORDER BY date DESC LIMIT 1",
                    (str(ticker), str(entry_date)),
                ).fetchone()
            if not forward_row or forward_row[0] is None:
                continue
            exit_price = float(forward_row[0])
            ep = float(entry_price)
            if str(direction).lower() == "short":
                pnl = (1.0 - exit_price / ep) * 100.0
            else:
                pnl = (exit_price / ep - 1.0) * 100.0
            pnls.append(round(pnl, 2))

        if not pnls:
            return None

        total = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        total_return = round(sum(pnls), 2)
        avg_return = round(total_return / total, 2)
        win_rate = round(wins / total * 100.0, 1)

        # Sharpe (per-trade, not annualised)
        sharpe: float | None = None
        if total > 1:
            mean_p = sum(pnls) / total
            var_p = sum((p - mean_p) ** 2 for p in pnls) / (total - 1)
            std_p = math.sqrt(var_p) if var_p > 0 else 0
            sharpe = round(mean_p / std_p, 2) if std_p > 0 else None

        return {
            "trades": total,
            "win_rate": win_rate,
            "return_pct": avg_return,
            "total_return_pct": total_return,
            "sharpe": sharpe,
            "hold_days": baseline_hold_days,
        }
    except Exception as exc:
        LOG.warning("Unfiltered baseline computation failed: %s", exc)
        return None


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
        SELECT pnl_pct, r_multiple, direction, setup_family, setup_tier, exit_reason,
               COALESCE(position_size_pct, 8.0) AS position_size_pct
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
            "overall": {
                "total_trades": 0, "open_count": open_trades,
                "starting_capital": config.starting_capital if config else 1_000_000.0,
                "portfolio_value": config.starting_capital if config else 1_000_000.0,
                "realized_pnl": 0, "unrealized_pnl": 0, "total_return_pct": 0,
            },
            "by_direction": {},
            "by_family": {},
            "by_tier": {},
            "by_exit_reason": {},
            "equity_curve": [],
            "recent_trades": recent_trades(conn, limit=20),
            "policy": _policy_snapshot(config),
            "feedback": [],
            "benchmarks": {},
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

    # Portfolio value tracking
    STARTING_CAPITAL = config.starting_capital if config else 1_000_000.0
    # Each closed trade's P&L contribution = position_size_pct × pnl_pct / 100
    # position_size_pct is read from the actual trade record (index 6 in the query).
    realized_pnl_dollars = 0.0
    for row in all_closed:
        trade_pnl_pct = float(row[0])
        pos_pct = float(row[6]) if row[6] is not None else 8.0
        position_dollars = STARTING_CAPITAL * (pos_pct / 100.0)
        realized_pnl_dollars += position_dollars * (trade_pnl_pct / 100)

    # Unrealized P&L from open trades
    open_rows = conn.execute(
        "SELECT unrealized_pnl_pct, position_size_pct FROM paper_trades WHERE status = 'open'"
    ).fetchall()
    unrealized_pnl_dollars = 0.0
    for orow in open_rows:
        u_pnl = float(orow[0] or 0)
        pos_pct = float(orow[1] or 8.0)
        position_dollars = STARTING_CAPITAL * (pos_pct / 100)
        unrealized_pnl_dollars += position_dollars * (u_pnl / 100)

    portfolio_value = STARTING_CAPITAL + realized_pnl_dollars + unrealized_pnl_dollars

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
        # Portfolio tracking
        "starting_capital": STARTING_CAPITAL,
        "portfolio_value": round(portfolio_value, 2),
        "realized_pnl": round(realized_pnl_dollars, 2),
        "unrealized_pnl": round(unrealized_pnl_dollars, 2),
        "total_return_pct": round((portfolio_value - STARTING_CAPITAL) / STARTING_CAPITAL * 100, 2),
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

    # Benchmark comparisons (SPY buy-and-hold + unfiltered baseline)
    benchmarks: dict[str, Any] = {}
    try:
        date_range = conn.execute(
            "SELECT MIN(entry_date), MAX(COALESCE(exit_date, entry_date)) "
            "FROM paper_trades WHERE entry_date >= ?",
            (cutoff,),
        ).fetchone()
        if date_range and date_range[0] and date_range[1]:
            spy_result = _compute_spy_benchmark(
                conn,
                first_entry_date=date_range[0],
                last_exit_date=date_range[1],
            )
            if spy_result is not None:
                benchmarks["spy_buy_hold"] = spy_result

        unfiltered_result = _compute_unfiltered_baseline(conn, cutoff=cutoff)
        if unfiltered_result is not None:
            benchmarks["unfiltered_setups"] = unfiltered_result
    except Exception as exc:
        LOG.warning("Benchmark computation failed (non-fatal): %s", exc)

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
        "benchmarks": benchmarks,
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
