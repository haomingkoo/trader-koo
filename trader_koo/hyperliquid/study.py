"""Machibro counter-trade study analysis.

Reconstructs position cycles from fill history and computes
strategy metrics: notional regime buckets, duration analysis,
coin breakdown, and counter-trade edge.

Research only. Not financial advice.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cycle reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_cycles(conn: sqlite3.Connection, wallet: str) -> list[dict[str, Any]]:
    """Reconstruct position open/close cycles from fill history."""
    rows = conn.execute(
        """
        SELECT coin, dir, size, price, closed_pnl, fee, is_liquidation,
               fill_time_ms, fill_date
        FROM hyperliquid_fills
        WHERE wallet_label = ?
        ORDER BY fill_time_ms ASC
        """,
        (wallet,),
    ).fetchall()

    if not rows:
        return []

    # Track net position per coin to detect cycle boundaries
    positions: dict[str, dict[str, Any]] = {}
    cycles: list[dict[str, Any]] = []

    for row in rows:
        coin = row[0]
        direction = str(row[1] or "")
        size = float(row[2] or 0)
        price = float(row[3] or 0)
        pnl = float(row[4] or 0)
        fee = float(row[5] or 0)
        is_liq = bool(row[6])
        fill_ts = int(row[7] or 0)
        fill_date = row[8]

        is_open = "Open" in direction
        is_close = "Close" in direction or "Liquidation" in direction

        if coin not in positions:
            positions[coin] = {
                "net_size": 0.0,
                "entry_notional": 0.0,
                "entry_ts": 0,
                "entry_date": None,
                "direction": None,
                "total_pnl": 0.0,
                "total_fees": 0.0,
                "open_fills": 0,
                "close_fills": 0,
                "was_liquidated": False,
            }

        pos = positions[coin]

        if is_open:
            if pos["net_size"] == 0:
                pos["entry_ts"] = fill_ts
                pos["entry_date"] = fill_date
                pos["direction"] = "Long" if "Long" in direction else "Short"
                pos["total_pnl"] = 0.0
                pos["total_fees"] = 0.0
                pos["open_fills"] = 0
                pos["close_fills"] = 0
                pos["was_liquidated"] = False
                pos["entry_notional"] = 0.0
            pos["net_size"] += size
            pos["entry_notional"] += size * price
            pos["open_fills"] += 1
            pos["total_fees"] += fee

        if is_close:
            pos["net_size"] = max(0, pos["net_size"] - size)
            pos["close_fills"] += 1
            pos["total_pnl"] += pnl
            pos["total_fees"] += fee
            if is_liq:
                pos["was_liquidated"] = True

            # Cycle complete when position returns to zero (or close enough)
            if pos["net_size"] < 0.0001 and pos["entry_ts"] > 0:
                duration_hours = (fill_ts - pos["entry_ts"]) / 3_600_000
                cycles.append({
                    "coin": coin,
                    "direction": pos["direction"],
                    "entry_notional_usd": round(pos["entry_notional"], 2),
                    "closed_pnl": round(pos["total_pnl"], 2),
                    "total_fees": round(pos["total_fees"], 2),
                    "duration_hours": round(max(0, duration_hours), 2),
                    "was_liquidated": pos["was_liquidated"],
                    "open_fills": pos["open_fills"],
                    "close_fills": pos["close_fills"],
                    "cycle_start": pos["entry_date"],
                    "cycle_end": fill_date,
                })
                pos["net_size"] = 0.0
                pos["entry_ts"] = 0

    return cycles


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _bucket_label(notional: float) -> str:
    if notional < 1_000_000:
        return "<$1M"
    if notional < 5_000_000:
        return "$1-5M"
    if notional < 15_000_000:
        return "$5-15M"
    if notional < 25_000_000:
        return "$15-25M"
    return ">$25M"


def _duration_label(hours: float) -> str:
    if hours < 1:
        return "Flash (<1h)"
    if hours < 6:
        return "Short (1-6h)"
    if hours < 24:
        return "Medium (6-24h)"
    if hours < 168:
        return "Long (1-7d)"
    return "Extended (>7d)"


def _bucket_stats(pnls: list[float]) -> dict[str, Any]:
    if not pnls:
        return {"count": 0, "win_rate_pct": 0, "avg_pnl": 0, "total_pnl": 0}
    wins = sum(1 for p in pnls if p > 0)
    return {
        "count": len(pnls),
        "win_rate_pct": round(wins / len(pnls) * 100, 1),
        "avg_pnl": round(sum(pnls) / len(pnls), 2),
        "total_pnl": round(sum(pnls), 2),
    }


def _counter_action(win_rate: float) -> str:
    """Determine counter-trade action based on his win rate."""
    if win_rate < 35:
        return "COUNTER"
    if win_rate < 50:
        return "LEAN_COUNTER"
    if win_rate < 60:
        return "SKIP"
    return "COPY"


def compute_study(
    conn: sqlite3.Connection,
    wallet: str = "machibro",
) -> dict[str, Any]:
    """Full counter-trade study analysis."""
    cycles = _reconstruct_cycles(conn, wallet)

    if not cycles:
        return {"ok": False, "error": "no_cycles", "wallet": wallet}

    # Overall stats
    all_pnls = [c["closed_pnl"] for c in cycles]
    total_pnl = sum(all_pnls)
    wins = sum(1 for p in all_pnls if p > 0)

    overview = {
        "total_cycles": len(cycles),
        "total_pnl": round(total_pnl, 2),
        "win_rate_pct": round(wins / len(cycles) * 100, 1),
        "avg_pnl": round(total_pnl / len(cycles), 2),
        "total_fees": round(sum(c["total_fees"] for c in cycles), 2),
        "liquidation_cycles": sum(1 for c in cycles if c["was_liquidated"]),
        "date_range": {
            "start": min(c["cycle_start"] for c in cycles if c["cycle_start"]),
            "end": max(c["cycle_end"] for c in cycles if c["cycle_end"]),
        },
    }

    # Notional regime buckets
    notional_buckets: dict[str, list[float]] = {}
    for c in cycles:
        label = _bucket_label(c["entry_notional_usd"])
        notional_buckets.setdefault(label, []).append(c["closed_pnl"])

    bucket_order = ["<$1M", "$1-5M", "$5-15M", "$15-25M", ">$25M"]
    notional_analysis = []
    for label in bucket_order:
        pnls = notional_buckets.get(label, [])
        if not pnls:
            continue
        stats = _bucket_stats(pnls)
        action = _counter_action(stats["win_rate_pct"])
        notional_analysis.append({
            "bucket": label,
            **stats,
            "counter_edge_total": round(-stats["total_pnl"], 2),
            "action": action,
        })

    # Duration analysis
    duration_buckets: dict[str, list[float]] = {}
    for c in cycles:
        label = _duration_label(c["duration_hours"])
        duration_buckets.setdefault(label, []).append(c["closed_pnl"])

    duration_order = ["Flash (<1h)", "Short (1-6h)", "Medium (6-24h)", "Long (1-7d)", "Extended (>7d)"]
    duration_analysis = []
    for label in duration_order:
        pnls = duration_buckets.get(label, [])
        if not pnls:
            continue
        stats = _bucket_stats(pnls)
        duration_analysis.append({"duration": label, **stats})

    # Coin breakdown
    coin_buckets: dict[str, list[dict[str, Any]]] = {}
    for c in cycles:
        coin_buckets.setdefault(c["coin"], []).append(c)

    coin_analysis = []
    for coin in sorted(coin_buckets, key=lambda c: len(coin_buckets[c]), reverse=True):
        coin_cycles = coin_buckets[coin]
        pnls = [c["closed_pnl"] for c in coin_cycles]
        stats = _bucket_stats(pnls)
        liqs = sum(1 for c in coin_cycles if c["was_liquidated"])
        coin_analysis.append({
            "coin": coin,
            **stats,
            "liquidations": liqs,
        })

    # Monthly breakdown
    month_buckets: dict[str, list[float]] = {}
    for c in cycles:
        if c["cycle_end"]:
            month = c["cycle_end"][:7]
            month_buckets.setdefault(month, []).append(c["closed_pnl"])

    monthly_analysis = []
    for month in sorted(month_buckets):
        pnls = month_buckets[month]
        stats = _bucket_stats(pnls)
        monthly_analysis.append({"month": month, **stats})

    # Counter-trade strategy summary (computed from actual data)
    rules = []
    for item in notional_analysis:
        rules.append({
            "condition": f"Notional {item['bucket']}",
            "action": item["action"],
            "reason": f"{item['win_rate_pct']}% WR over {item['count']} cycles",
            "counter_edge": item["counter_edge_total"],
        })

    # Determine dominant direction
    directions = [c["direction"] for c in cycles]
    long_pct = round(sum(1 for d in directions if d == "Long") / len(directions) * 100, 1)

    # Top coin by cycle count
    top_coin = coin_analysis[0] if coin_analysis else None
    top_coin_str = (
        f"{top_coin['coin']} is primary coin ({top_coin['count']}/{len(cycles)} cycles) "
        f"with {top_coin['win_rate_pct']}% win rate"
    ) if top_coin else "No dominant coin"

    # Best counter bucket
    best_counter = max(notional_analysis, key=lambda x: x["counter_edge_total"]) if notional_analysis else None
    best_counter_str = (
        f"Best counter-trade bucket: {best_counter['bucket']} "
        f"(his WR={best_counter['win_rate_pct']}%, edge=${best_counter['counter_edge_total']:,.0f})"
    ) if best_counter else "No clear best bucket"

    # Duration insight
    worst_dur = max(duration_analysis, key=lambda x: -x["avg_pnl"]) if duration_analysis else None
    best_dur = max(duration_analysis, key=lambda x: x["avg_pnl"]) if duration_analysis else None

    key_findings = [
        f"{long_pct}% of {len(cycles)} cycles are long positions (perma-long bias)",
        top_coin_str,
        f"{overview['liquidation_cycles']} liquidation cycles out of {overview['total_cycles']} total",
        best_counter_str,
        f"Total PnL: ${total_pnl:,.0f} over {len(cycles)} cycles ({overview['date_range']['start']} to {overview['date_range']['end']})",
    ]
    if worst_dur:
        key_findings.append(
            f"Worst duration: {worst_dur['duration']} ({worst_dur['win_rate_pct']}% WR, avg ${worst_dur['avg_pnl']:,.0f})"
        )
    if best_dur:
        key_findings.append(
            f"Best duration: {best_dur['duration']} ({best_dur['win_rate_pct']}% WR, avg ${best_dur['avg_pnl']:,.0f})"
        )

    strategy = {
        "name": "Notional Regime Counter-Trade",
        "description": (
            f"Counter-trade {wallet} based on position notional size. "
            f"Analysis of {len(cycles)} complete position cycles shows "
            f"a {long_pct}% long bias with overall {overview['win_rate_pct']}% win rate. "
            "Edge exists in specific notional buckets where his win rate drops below 50%."
        ),
        "rules": rules,
        "key_findings": key_findings,
        "disclaimer": (
            "This is a research study for educational purposes only. "
            "Not financial advice. Past performance does not guarantee future results. "
            "Counter-trading carries significant risk of loss. "
            "Cryptocurrency derivatives are highly volatile and leveraged products. "
            "Sample sizes for some buckets are small and may not be statistically significant."
        ),
    }

    return {
        "ok": True,
        "wallet": wallet,
        "overview": overview,
        "notional_analysis": notional_analysis,
        "duration_analysis": duration_analysis,
        "coin_analysis": coin_analysis,
        "monthly_analysis": monthly_analysis,
        "strategy": strategy,
        "cycles": cycles,  # Raw data for charts
    }
