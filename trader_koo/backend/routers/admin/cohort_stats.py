"""Cohort statistics for paper trades.

Slices closed paper trades by ``bot_version`` (which since the diagnostics
work auto-derives from the deployed git SHA) so that performance can be
attributed to specific code versions.

Answers questions like:
  - Did the 04-10 fix wave (commit 4ada5f6) actually improve win rate?
  - Are post-fix trades more cost-consistent than pre-fix?
  - Does YOLO-boosted alpha exist within a single cohort?

Response includes a 95% Wilson confidence interval on win rate so users
can avoid mistaking 5-trade noise for a real edge — the dashboard's
"Actionable Review" doesn't currently do this and that's how 03-18-style
clusters get promoted to "this strategy works".
"""
from __future__ import annotations

import logging
import math
import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, Query

from trader_koo.backend.services.database import DB_PATH, table_exists
from trader_koo.middleware.auth import require_admin_auth

LOG = logging.getLogger("trader_koo.routers.admin.cohort_stats")

router = APIRouter(tags=["admin", "admin-cohort-stats"])


def _wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion.

    More accurate than the normal approximation at small N — important
    here because cohorts can have ~5-10 trades.
    """
    if total <= 0:
        return (0.0, 0.0)
    p = wins / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (round(lo * 100, 1), round(hi * 100, 1))


def _r_histogram(r_multiples: list[float]) -> dict[str, int]:
    """Bin R-multiples into a coarse histogram for sanity-checking outliers."""
    bins = {"<-2R": 0, "-2R..-1R": 0, "-1R..0": 0, "0..1R": 0, "1R..2R": 0, ">2R": 0}
    for r in r_multiples:
        if r < -2.0:
            bins["<-2R"] += 1
        elif r < -1.0:
            bins["-2R..-1R"] += 1
        elif r < 0.0:
            bins["-1R..0"] += 1
        elif r < 1.0:
            bins["0..1R"] += 1
        elif r < 2.0:
            bins["1R..2R"] += 1
        else:
            bins[">2R"] += 1
    return bins


def _family_breakdown(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    """Aggregate by setup_family within a cohort."""
    by_family: dict[str, dict[str, Any]] = {}
    for row in rows:
        fam = row["setup_family"] or "unknown"
        bucket = by_family.setdefault(
            fam,
            {"family": fam, "n_trades": 0, "n_wins": 0, "pnl_sum": 0.0, "r_sum": 0.0},
        )
        bucket["n_trades"] += 1
        if (row["pnl_pct"] or 0) > 0:
            bucket["n_wins"] += 1
        bucket["pnl_sum"] += float(row["pnl_pct"] or 0)
        bucket["r_sum"] += float(row["r_multiple"] or 0)
    out = []
    for bucket in by_family.values():
        n = bucket["n_trades"]
        wins = bucket["n_wins"]
        wr_lo, wr_hi = _wilson_interval(wins, n)
        out.append({
            "family": bucket["family"],
            "n_trades": n,
            "n_wins": wins,
            "win_rate_pct": round(wins / n * 100, 1) if n else 0.0,
            "win_rate_ci_95": [wr_lo, wr_hi],
            "avg_pnl_pct": round(bucket["pnl_sum"] / n, 2) if n else 0.0,
            "avg_r_multiple": round(bucket["r_sum"] / n, 2) if n else 0.0,
        })
    out.sort(key=lambda x: x["n_trades"], reverse=True)
    return out


def _summarise_cohort(version: str, rows: list[sqlite3.Row]) -> dict[str, Any]:
    closed = [r for r in rows if r["status"] in ("closed", "stopped_out", "target_hit", "expired")]
    open_n = sum(1 for r in rows if r["status"] == "open")
    n = len(closed)
    if n == 0:
        return {
            "bot_version": version,
            "n_trades": len(rows),
            "n_closed": 0,
            "n_open": open_n,
            "note": "no closed trades in cohort",
        }

    pnls = [float(r["pnl_pct"] or 0) for r in closed]
    r_mults = [float(r["r_multiple"] or 0) for r in closed]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    wr_lo, wr_hi = _wilson_interval(wins, n)

    # Cash-adjusted return: weight each trade's P&L by the fraction of book
    # that was deployed when it opened. A trade run on 5% of capital that
    # made +2% adds ~0.10pp to portfolio; we surface that here so the
    # cohort's contribution can be compared to SPY on equal footing.
    cash_adjusted_contrib = 0.0
    deployed_count = 0
    for r in closed:
        deployed_pct = r["deployed_capital_pct"]
        pnl = r["pnl_pct"]
        if deployed_pct is not None and pnl is not None:
            # position_size is what THIS trade used of the book; we approximate
            # that as a small fraction (<=tier ceiling). For now use position_size_pct
            # if present, else fall back to deployed_pct as a rough proxy.
            pos_size_pct = r["position_size_pct"]
            if pos_size_pct is not None:
                cash_adjusted_contrib += float(pnl) * (float(pos_size_pct) / 100.0)
                deployed_count += 1

    expectancy_r = sum(r_mults) / n if n else 0.0
    dates = sorted(r["entry_date"] for r in closed if r["entry_date"])

    return {
        "bot_version": version,
        "n_trades": len(rows),
        "n_closed": n,
        "n_open": open_n,
        "n_wins": wins,
        "n_losses": losses,
        "win_rate_pct": round(wins / n * 100, 1),
        "win_rate_ci_95": [wr_lo, wr_hi],
        "avg_pnl_pct": round(sum(pnls) / n, 2),
        "total_pnl_pct": round(sum(pnls), 2),
        "avg_r_multiple": round(sum(r_mults) / n, 2),
        "expectancy_r": round(expectancy_r, 3),
        "best_trade_pct": round(max(pnls), 2),
        "worst_trade_pct": round(min(pnls), 2),
        "r_distribution": _r_histogram(r_mults),
        "family_breakdown": _family_breakdown(closed),
        "cash_adjusted_portfolio_contrib_pct": round(cash_adjusted_contrib, 3),
        "cash_adjusted_n": deployed_count,
        "first_trade_date": dates[0] if dates else None,
        "last_trade_date": dates[-1] if dates else None,
    }


@router.get("/api/admin/cohort-stats")
async def cohort_stats(
    bot_version: str | None = Query(
        default=None,
        description="If set, return stats for only this bot_version (else all cohorts).",
    ),
    min_trades: int = Query(
        default=1,
        ge=0,
        description="Suppress cohorts with fewer than this many closed trades.",
    ),
    _: None = Depends(require_admin_auth),
) -> dict[str, Any]:
    """Return paper-trade performance sliced by ``bot_version``.

    Each cohort gets a Wilson 95% CI on win rate so callers can tell
    real edges apart from N=5 noise. Use ``min_trades=20`` (per the
    feature-activation checklist) to ignore cohorts that haven't earned
    a verdict yet.
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        if not table_exists(conn, "paper_trades"):
            return {
                "ok": False,
                "error": "paper_trades table missing",
                "cohorts": [],
            }

        if bot_version:
            rows = conn.execute(
                "SELECT * FROM paper_trades WHERE bot_version = ? ORDER BY entry_date",
                (bot_version,),
            ).fetchall()
            cohort_rows = {bot_version: list(rows)}
        else:
            all_rows = conn.execute(
                "SELECT * FROM paper_trades ORDER BY entry_date"
            ).fetchall()
            cohort_rows = {}
            for r in all_rows:
                key = r["bot_version"] or "unknown"
                cohort_rows.setdefault(key, []).append(r)

        cohorts: list[dict[str, Any]] = []
        for version, rows in cohort_rows.items():
            summary = _summarise_cohort(version, rows)
            if summary.get("n_closed", 0) >= min_trades or summary["n_trades"] >= min_trades:
                cohorts.append(summary)

        cohorts.sort(key=lambda c: c.get("first_trade_date") or "", reverse=True)

        # Overall (all cohorts combined) for context
        all_closed = [
            r for rows in cohort_rows.values() for r in rows
            if r["status"] in ("closed", "stopped_out", "target_hit", "expired")
        ]
        overall = _summarise_cohort("ALL", all_closed) if all_closed else {}

        return {
            "ok": True,
            "n_cohorts": len(cohorts),
            "min_trades_filter": min_trades,
            "cohorts": cohorts,
            "overall": overall,
            "note": (
                "Wilson 95% CI on win_rate. Cohorts with N<20 are not "
                "statistically distinguishable per feature-activation.md. "
                "cash_adjusted_portfolio_contrib_pct sums each trade's "
                "pnl_pct * position_size_pct/100 — that's what actually "
                "moved the portfolio, vs raw pnl_pct that ignores sizing."
            ),
        }
    finally:
        conn.close()
