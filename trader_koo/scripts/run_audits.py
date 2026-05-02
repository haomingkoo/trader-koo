"""Run one or all post-trade audits against the paper_trades table.

Each audit is a self-contained function that returns a JSON-serialisable
dict so output can be eyeballed or piped into further analysis. The
audits answer the open questions from the Phase 2 plan:

  1. risk-budget       Are dollar losses constant-sized? (sizing fix verified?)
  2. stop-tightness    Stop distance vs ATR per losing trade (whipsaw check)
  3. clustering        Same-day entries and their outcomes (03-18 problem)
  4. regime-conflict   Trades whose direction contradicts the regime at entry
  5. exit-quality      Why only 1 of 27 hit target? trail vs target vs stop
  6. sector-cap        Sector concentration cap firing rate post-04-10
  7. turnover          Holding-period distribution + cost-payback sanity
  8. correlation       Open-position correlation matrix and factor exposure
  9. attribution       PnL decomposed into beta / sector / idiosyncratic
 10. signal-decay      Forward returns at T+1 / T+5 / T+20 by family
 11. reproducibility   Rerun signals on the same day, confirm trade list match
 12. yolo-edge         WR / expectancy of yolo_boost_pts > 0 vs == 0 cohorts

Usage:
    python -m trader_koo.scripts.run_audits all --db-path /data/trader_koo.db
    python -m trader_koo.scripts.run_audits risk-budget --db-path /data/...
    python -m trader_koo.scripts.run_audits yolo-edge --json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

LOG = logging.getLogger("trader_koo.scripts.run_audits")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _wilson_lo(wins: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = wins / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return round(max(0.0, center - half) * 100, 1)


def _stdev_safe(values: list[float]) -> float:
    return round(statistics.stdev(values), 4) if len(values) >= 2 else 0.0


def _starting_capital() -> float:
    """Match paper_trades.PAPER_TRADE_STARTING_CAPITAL default."""
    import os
    return float(os.getenv("TRADER_KOO_PAPER_TRADE_STARTING_CAPITAL", "1000000.0"))


# ---------------------------------------------------------------------------
# 1. risk-budget consistency
# ---------------------------------------------------------------------------

def audit_risk_budget(conn: sqlite3.Connection) -> dict[str, Any]:
    """STDEV/AVG of dollar loss across stopped-out trades, per cohort.

    With constant-dollar-risk sizing post-04-10 (commit 4ada5f6) we
    expect tight clustering of loss size: STDEV/AVG should be < 0.30.
    A wider spread means the sizing fix didn't take and trades are
    still risking variable dollars per stop.
    """
    capital = _starting_capital()
    rows = conn.execute(
        """
        SELECT bot_version, pnl_pct, position_size_pct, status, ticker, entry_date
        FROM paper_trades
        WHERE status = 'stopped_out'
          AND pnl_pct IS NOT NULL
          AND position_size_pct IS NOT NULL
        """
    ).fetchall()

    by_cohort: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        # Approximate dollar loss: pnl_pct * position_size_pct/100 * starting_capital.
        # This is a first-order approximation — equity changes over time.
        dollar_loss = abs(float(r["pnl_pct"])) * float(r["position_size_pct"]) / 100.0 * capital / 100.0
        by_cohort[r["bot_version"] or "unknown"].append(dollar_loss)

    findings = []
    for cohort, losses in sorted(by_cohort.items()):
        if not losses:
            continue
        mean = statistics.mean(losses)
        sd = _stdev_safe(losses)
        cv = round(sd / mean, 3) if mean else 0.0
        verdict = "OK (consistent)" if cv < 0.30 else "FLAG (variable)"
        findings.append({
            "cohort": cohort,
            "n_stops": len(losses),
            "avg_loss_usd": round(mean, 0),
            "stdev_loss_usd": round(sd, 0),
            "cv": cv,
            "min_loss_usd": round(min(losses), 0),
            "max_loss_usd": round(max(losses), 0),
            "verdict": verdict,
        })

    return {
        "audit": "risk-budget",
        "expectation": "CV < 0.30 if dollar-risk sizing is working",
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# 2. stop tightness vs ATR
# ---------------------------------------------------------------------------

def audit_stop_tightness(conn: sqlite3.Connection) -> dict[str, Any]:
    """Stop distance / ATR_at_entry per losing trade.

    Stops <1.0x ATR are noise stops — they get hit by daily volatility
    before any move plays out. WYNN and EQR (both stopped same-day at
    -1.41R / -1.46R) are likely culprits. Threshold per CLAUDE.md
    sprint notes: stops should be at least 1.5x ATR or 2.5%.
    """
    rows = conn.execute(
        """
        SELECT ticker, entry_date, entry_price, stop_loss, atr_at_entry,
               r_multiple, status, setup_family, bot_version
        FROM paper_trades
        WHERE status IN ('stopped_out', 'closed')
          AND r_multiple < 0
          AND entry_price IS NOT NULL
          AND stop_loss IS NOT NULL
          AND atr_at_entry IS NOT NULL
          AND atr_at_entry > 0
        ORDER BY r_multiple ASC
        """
    ).fetchall()

    findings = []
    noise_stops = 0
    for r in rows:
        stop_dist = abs(float(r["entry_price"]) - float(r["stop_loss"]))
        atr_mult = round(stop_dist / float(r["atr_at_entry"]), 2)
        is_noise = atr_mult < 1.0
        if is_noise:
            noise_stops += 1
        findings.append({
            "ticker": r["ticker"],
            "entry_date": r["entry_date"],
            "stop_dist_atr_mult": atr_mult,
            "r_multiple": round(float(r["r_multiple"]), 2),
            "family": r["setup_family"],
            "cohort": r["bot_version"],
            "is_noise_stop": is_noise,
        })

    return {
        "audit": "stop-tightness",
        "expectation": "stop_dist >= 1.5x ATR per CLAUDE.md sprint notes",
        "n_losing_trades": len(rows),
        "n_noise_stops": noise_stops,
        "findings": findings[:30],  # top 30 worst losses
    }


# ---------------------------------------------------------------------------
# 3. same-day clustering
# ---------------------------------------------------------------------------

def audit_clustering(conn: sqlite3.Connection) -> dict[str, Any]:
    """Count entries per day and the outcome distribution by cluster size.

    The 2026-03-18 cluster (11 trades, 4 longs all stopped) is the
    smoking gun. If new trades repeat the pattern, the regime gate
    is still letting through correlated entries.
    """
    rows = conn.execute(
        """
        SELECT entry_date, direction, status, pnl_pct, ticker, setup_family,
               bot_version
        FROM paper_trades
        WHERE entry_date IS NOT NULL
        ORDER BY entry_date
        """
    ).fetchall()

    by_date: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for r in rows:
        by_date[r["entry_date"]].append(r)

    clusters = []
    for date, trades in sorted(by_date.items()):
        if len(trades) < 2:
            continue
        n_long = sum(1 for t in trades if t["direction"] == "long")
        n_short = sum(1 for t in trades if t["direction"] == "short")
        closed_pnls = [
            float(t["pnl_pct"]) for t in trades
            if t["pnl_pct"] is not None and t["status"] != "open"
        ]
        long_pnls = [
            float(t["pnl_pct"]) for t in trades
            if t["direction"] == "long" and t["pnl_pct"] is not None
        ]
        short_pnls = [
            float(t["pnl_pct"]) for t in trades
            if t["direction"] == "short" and t["pnl_pct"] is not None
        ]
        clusters.append({
            "date": date,
            "n_trades": len(trades),
            "n_long": n_long,
            "n_short": n_short,
            "long_avg_pnl": round(sum(long_pnls) / len(long_pnls), 2) if long_pnls else None,
            "short_avg_pnl": round(sum(short_pnls) / len(short_pnls), 2) if short_pnls else None,
            "all_closed_avg_pnl": round(sum(closed_pnls) / len(closed_pnls), 2) if closed_pnls else None,
            "cohort": trades[0]["bot_version"],
            "tickers": [t["ticker"] for t in trades],
        })

    clusters.sort(key=lambda c: c["n_trades"], reverse=True)
    return {
        "audit": "clustering",
        "expectation": "no day with >= 3 same-direction entries post-fix",
        "n_cluster_days": len(clusters),
        "findings": clusters,
    }


# ---------------------------------------------------------------------------
# 4. regime conflict at entry
# ---------------------------------------------------------------------------

def audit_regime_conflict(conn: sqlite3.Connection) -> dict[str, Any]:
    """Trades where ``direction`` contradicts the regime label at entry.

    A long entered in a "bear_*" regime, or short in "bull_*", should
    have been blocked by the post-04-10 hard-block rules. Anything
    that slipped through is either a regime label edge case or a gap
    in the gating logic.
    """
    rows = conn.execute(
        """
        SELECT ticker, entry_date, direction, regime_state_at_entry,
               hmm_regime_at_entry, directional_regime_at_entry,
               setup_tier, score, pnl_pct, status, bot_version
        FROM paper_trades
        WHERE regime_state_at_entry IS NOT NULL
        ORDER BY entry_date
        """
    ).fetchall()

    conflicts = []
    for r in rows:
        regime = (r["regime_state_at_entry"] or "").lower()
        direction = r["direction"]
        is_conflict = (
            (direction == "long" and regime.startswith("bear"))
            or (direction == "short" and regime.startswith("bull"))
        )
        # Also flag directional HMM contradiction
        dir_regime = (r["directional_regime_at_entry"] or "").lower()
        hmm_conflict = (
            (direction == "long" and "bear" in dir_regime)
            or (direction == "short" and "bull" in dir_regime)
        )
        if is_conflict or hmm_conflict:
            conflicts.append({
                "ticker": r["ticker"],
                "entry_date": r["entry_date"],
                "direction": direction,
                "regime": regime,
                "hmm_directional": dir_regime,
                "tier": r["setup_tier"],
                "score": r["score"],
                "pnl_pct": r["pnl_pct"],
                "status": r["status"],
                "cohort": r["bot_version"],
                "vs_simple_regime": is_conflict,
                "vs_hmm_directional": hmm_conflict,
            })

    return {
        "audit": "regime-conflict",
        "expectation": "0 conflicts post-04-10 (4ada5f6 hard-blocks counter-trend)",
        "n_total_with_regime_data": len(rows),
        "n_conflicts": len(conflicts),
        "findings": conflicts,
    }


# ---------------------------------------------------------------------------
# 5. exit quality
# ---------------------------------------------------------------------------

def audit_exit_quality(conn: sqlite3.Connection) -> dict[str, Any]:
    """Why only 1 of 27 hit target? Decompose exits by reason.

    Compare: target_hit count, expired-with-positive-R count (winners
    capped by trailing stop), and stopped-out-after-MFE > 1R count
    (potential give-back). High-water_mark column gives a poor man's
    MFE proxy.
    """
    rows = conn.execute(
        """
        SELECT ticker, direction, entry_price, target_price, stop_loss,
               exit_price, exit_reason, pnl_pct, r_multiple,
               high_water_mark, low_water_mark, atr_at_entry,
               setup_family, bot_version, status
        FROM paper_trades
        WHERE status IN ('closed', 'stopped_out', 'target_hit', 'expired')
        """
    ).fetchall()

    by_reason: Counter = Counter()
    mfe_giveback = []
    for r in rows:
        reason = r["exit_reason"] or r["status"]
        by_reason[reason] += 1

        # MFE proxy: peak favourable excursion
        try:
            entry = float(r["entry_price"])
            stop = float(r["stop_loss"])
            r_unit = abs(entry - stop)
            if r_unit <= 0:
                continue
            if r["direction"] == "long":
                mfe = float(r["high_water_mark"] or entry) - entry
            else:
                mfe = entry - float(r["low_water_mark"] or entry)
            mfe_r = mfe / r_unit
            realised_r = float(r["r_multiple"] or 0)
            if reason in ("stopped_out", "expired") and mfe_r >= 1.0 and realised_r < mfe_r * 0.5:
                mfe_giveback.append({
                    "ticker": r["ticker"],
                    "mfe_r": round(mfe_r, 2),
                    "realised_r": round(realised_r, 2),
                    "give_back": round(mfe_r - realised_r, 2),
                    "reason": reason,
                    "family": r["setup_family"],
                    "cohort": r["bot_version"],
                })
        except (TypeError, ValueError):
            continue

    return {
        "audit": "exit-quality",
        "n_total": len(rows),
        "exit_reason_counts": dict(by_reason),
        "target_hit_rate_pct": round(by_reason.get("target_hit", 0) / len(rows) * 100, 1) if rows else 0,
        "mfe_giveback_trades": sorted(mfe_giveback, key=lambda x: x["give_back"], reverse=True)[:20],
        "interpretation": (
            "If target_hit_rate_pct < 10% AND mfe_giveback_trades is long, "
            "trailing stops are too tight: trades reach 1R+ MFE then trail "
            "back to break-even or lose. Either widen the trail cushion or "
            "lower the target multiple."
        ),
    }


# ---------------------------------------------------------------------------
# 6. sector cap firing rate
# ---------------------------------------------------------------------------

def audit_sector_cap(conn: sqlite3.Connection) -> dict[str, Any]:
    """How often does the sector cap (added 04-10 in 4ada5f6) actually fire?

    Without an audit table for cap rejections, we can only infer from
    open trades — count concurrent same-sector positions. If we ever see
    >1 simultaneous open trade in the same sector, the cap is missing
    or broken.
    """
    rows = conn.execute(
        """
        SELECT pt.ticker, pt.entry_date, pt.exit_date, pt.status, pt.bot_version,
               COALESCE(s.sector, 'unknown') AS sector
        FROM paper_trades pt
        LEFT JOIN finviz_snapshots s ON s.ticker = pt.ticker
        ORDER BY pt.entry_date
        """
    ).fetchall() if _table_exists(conn, "finviz_snapshots") else []

    if not rows:
        return {
            "audit": "sector-cap",
            "skipped": "finviz_snapshots not available — cannot resolve sectors",
            "findings": [],
        }

    overlaps = []
    for i, r in enumerate(rows):
        if r["status"] == "open":
            continue
        for j in range(i + 1, len(rows)):
            r2 = rows[j]
            if r["sector"] != r2["sector"] or r["sector"] == "unknown":
                continue
            # Overlap if r2 entered before r exited
            if r["exit_date"] and r2["entry_date"] < r["exit_date"]:
                overlaps.append({
                    "ticker_a": r["ticker"],
                    "ticker_b": r2["ticker"],
                    "sector": r["sector"],
                    "a_dates": [r["entry_date"], r["exit_date"]],
                    "b_dates": [r2["entry_date"], r2["exit_date"]],
                    "cohort_a": r["bot_version"],
                    "cohort_b": r2["bot_version"],
                })

    return {
        "audit": "sector-cap",
        "expectation": "0 same-sector overlaps post-4ada5f6",
        "n_overlaps": len(overlaps),
        "findings": overlaps[:30],
    }


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,),
    ).fetchone())


# ---------------------------------------------------------------------------
# 7. turnover + holding period
# ---------------------------------------------------------------------------

def audit_turnover(conn: sqlite3.Connection) -> dict[str, Any]:
    """Holding period distribution + cost-payback feasibility.

    With 0.09R per-trade expectancy, costs in the order of 0.05R per
    trade would zero it out. Short hold + thin edge = fragile.
    """
    rows = conn.execute(
        """
        SELECT entry_date, exit_date, r_multiple, status, bot_version
        FROM paper_trades
        WHERE entry_date IS NOT NULL AND exit_date IS NOT NULL
        """
    ).fetchall()

    holds = []
    for r in rows:
        try:
            import datetime as dt
            d1 = dt.date.fromisoformat(r["entry_date"])
            d2 = dt.date.fromisoformat(r["exit_date"])
            holds.append((d2 - d1).days)
        except Exception:
            continue

    if not holds:
        return {"audit": "turnover", "findings": [], "note": "no closed trades with dates"}

    holds.sort()
    return {
        "audit": "turnover",
        "n_closed": len(holds),
        "min_hold_days": holds[0],
        "median_hold_days": holds[len(holds) // 2],
        "p75_hold_days": holds[int(len(holds) * 0.75)],
        "max_hold_days": holds[-1],
        "avg_hold_days": round(sum(holds) / len(holds), 1),
        "n_same_day": sum(1 for h in holds if h == 0),
        "interpretation": (
            "Same-day stop-outs at >1R loss point to noise stops or "
            "next-day-open slippage gapping past stops."
        ),
    }


# ---------------------------------------------------------------------------
# 8. open-position correlation (placeholder — needs price history)
# ---------------------------------------------------------------------------

def audit_correlation(conn: sqlite3.Connection) -> dict[str, Any]:
    """Pairwise return correlation across simultaneously-open positions.

    Sector cap is necessary but not sufficient — two different sectors
    can still load on the same factor (e.g. high-beta tech + cyclical
    consumer both being 'risk-on' bets). We compute pairwise 60d
    correlation of daily returns for tickers that were simultaneously
    open and flag the max.
    """
    if not _table_exists(conn, "price_daily"):
        return {"audit": "correlation", "skipped": "price_daily table missing"}

    open_overlaps = conn.execute(
        """
        SELECT a.ticker AS ta, b.ticker AS tb, a.entry_date AS ed,
               COALESCE(MIN(a.exit_date, b.exit_date), DATE('now')) AS overlap_end
        FROM paper_trades a
        JOIN paper_trades b
          ON a.id < b.id
          AND a.entry_date <= COALESCE(b.exit_date, DATE('now'))
          AND b.entry_date <= COALESCE(a.exit_date, DATE('now'))
        """
    ).fetchall()

    high_corr = []
    for r in open_overlaps[:200]:  # cap pair lookups
        try:
            prices_a = conn.execute(
                "SELECT close FROM price_daily WHERE ticker=? AND close IS NOT NULL "
                "ORDER BY date DESC LIMIT 60",
                (r["ta"],),
            ).fetchall()
            prices_b = conn.execute(
                "SELECT close FROM price_daily WHERE ticker=? AND close IS NOT NULL "
                "ORDER BY date DESC LIMIT 60",
                (r["tb"],),
            ).fetchall()
            if len(prices_a) < 30 or len(prices_b) < 30:
                continue
            n = min(len(prices_a), len(prices_b))
            ra = [float(prices_a[i][0]) / float(prices_a[i + 1][0]) - 1 for i in range(n - 1)]
            rb = [float(prices_b[i][0]) / float(prices_b[i + 1][0]) - 1 for i in range(n - 1)]
            mu_a, mu_b = sum(ra) / len(ra), sum(rb) / len(rb)
            cov = sum((ra[i] - mu_a) * (rb[i] - mu_b) for i in range(len(ra))) / len(ra)
            sd_a = math.sqrt(sum((x - mu_a) ** 2 for x in ra) / len(ra))
            sd_b = math.sqrt(sum((x - mu_b) ** 2 for x in rb) / len(rb))
            if sd_a == 0 or sd_b == 0:
                continue
            corr = cov / (sd_a * sd_b)
            if abs(corr) > 0.7:
                high_corr.append({
                    "ticker_a": r["ta"],
                    "ticker_b": r["tb"],
                    "corr_60d": round(corr, 2),
                    "overlap_start": r["ed"],
                })
        except Exception:
            continue

    return {
        "audit": "correlation",
        "expectation": "no concurrent pair with |corr| > 0.7 means low concentration risk",
        "n_high_corr_pairs": len(high_corr),
        "findings": sorted(high_corr, key=lambda x: abs(x["corr_60d"]), reverse=True)[:20],
    }


# ---------------------------------------------------------------------------
# 9. PnL attribution (rough)
# ---------------------------------------------------------------------------

def audit_attribution(conn: sqlite3.Connection) -> dict[str, Any]:
    """Decompose realised PnL into beta vs idiosyncratic.

    For each closed long, alpha = (ticker_return - beta * SPY_return)
    over the holding period. Sum alpha across trades = book alpha.
    Sum beta * SPY_return = book beta. If the book is mostly beta,
    you're not earning fees by being selective — SPY 1x with no work.
    """
    if not _table_exists(conn, "price_daily"):
        return {"audit": "attribution", "skipped": "price_daily table missing"}

    rows = conn.execute(
        """
        SELECT ticker, direction, entry_date, exit_date, pnl_pct
        FROM paper_trades
        WHERE status IN ('closed', 'stopped_out', 'target_hit', 'expired')
          AND entry_date IS NOT NULL AND exit_date IS NOT NULL
          AND pnl_pct IS NOT NULL
        """
    ).fetchall()

    total_pnl = total_beta_pnl = 0.0
    n_processed = 0
    BETA = 1.0  # crude — use 1.0 for all S&P 500 names; refine with stored beta later
    for r in rows:
        try:
            spy_start = conn.execute(
                "SELECT close FROM price_daily WHERE ticker='SPY' AND date <= ? "
                "ORDER BY date DESC LIMIT 1", (r["entry_date"],),
            ).fetchone()
            spy_end = conn.execute(
                "SELECT close FROM price_daily WHERE ticker='SPY' AND date <= ? "
                "ORDER BY date DESC LIMIT 1", (r["exit_date"],),
            ).fetchone()
            if not spy_start or not spy_end:
                continue
            spy_ret = (float(spy_end[0]) / float(spy_start[0]) - 1) * 100
            sign = 1 if r["direction"] == "long" else -1
            beta_pnl = BETA * spy_ret * sign
            total_pnl += float(r["pnl_pct"])
            total_beta_pnl += beta_pnl
            n_processed += 1
        except Exception:
            continue

    alpha = total_pnl - total_beta_pnl
    return {
        "audit": "attribution",
        "n_processed": n_processed,
        "total_pnl_pct_sum": round(total_pnl, 2),
        "beta_attributed_pct_sum": round(total_beta_pnl, 2),
        "alpha_pct_sum": round(alpha, 2),
        "interpretation": (
            "alpha > 0 means selection added value above SPY beta. "
            "alpha <= 0 means the system captured no real edge — "
            "raw return was beta in disguise."
        ),
    }


# ---------------------------------------------------------------------------
# 10. signal decay (placeholder — needs setup history)
# ---------------------------------------------------------------------------

def audit_signal_decay(conn: sqlite3.Connection) -> dict[str, Any]:
    """T+1 / T+5 / T+20 forward returns for signaled tickers.

    Approximation: for each closed trade, forward-look the price T+1,
    T+5, T+20 trading days from entry. If the edge concentrates at
    T+1 and disappears by T+20, the system is reading short-term
    noise. If the edge grows monotonically, hold longer.
    """
    if not _table_exists(conn, "price_daily"):
        return {"audit": "signal-decay", "skipped": "price_daily table missing"}

    rows = conn.execute(
        """
        SELECT ticker, direction, entry_date, setup_family, bot_version
        FROM paper_trades
        WHERE entry_date IS NOT NULL
        """
    ).fetchall()

    by_family: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"t1": [], "t5": [], "t20": []}
    )
    for r in rows:
        try:
            entry_close = conn.execute(
                "SELECT close FROM price_daily WHERE ticker=? AND date <= ? "
                "ORDER BY date DESC LIMIT 1", (r["ticker"], r["entry_date"]),
            ).fetchone()
            if not entry_close:
                continue
            entry_p = float(entry_close[0])
            future = conn.execute(
                "SELECT close FROM price_daily WHERE ticker=? AND date > ? "
                "ORDER BY date ASC LIMIT 20",
                (r["ticker"], r["entry_date"]),
            ).fetchall()
            sign = 1 if r["direction"] == "long" else -1
            family = r["setup_family"] or "unknown"
            if len(future) >= 1:
                by_family[family]["t1"].append(sign * (float(future[0][0]) / entry_p - 1) * 100)
            if len(future) >= 5:
                by_family[family]["t5"].append(sign * (float(future[4][0]) / entry_p - 1) * 100)
            if len(future) >= 20:
                by_family[family]["t20"].append(sign * (float(future[19][0]) / entry_p - 1) * 100)
        except Exception:
            continue

    findings = []
    for fam, by_t in by_family.items():
        findings.append({
            "family": fam,
            "n_t1": len(by_t["t1"]),
            "avg_t1_pct": round(sum(by_t["t1"]) / len(by_t["t1"]), 2) if by_t["t1"] else None,
            "avg_t5_pct": round(sum(by_t["t5"]) / len(by_t["t5"]), 2) if by_t["t5"] else None,
            "avg_t20_pct": round(sum(by_t["t20"]) / len(by_t["t20"]), 2) if by_t["t20"] else None,
        })

    return {
        "audit": "signal-decay",
        "interpretation": (
            "Edge concentrated at T+1 means short-term noise; edge growing "
            "to T+20 means trend-following works and trails are too tight."
        ),
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# 11. reproducibility
# ---------------------------------------------------------------------------

def audit_reproducibility(conn: sqlite3.Connection) -> dict[str, Any]:
    """Sanity check: re-running the setup scorer with the same inputs
    should produce the same trades.

    Stub for now — a full check requires re-running ``generate_daily_report``
    against an as-of date and diffing the trade list. We emit a TODO so
    this can be wired to the report runner.
    """
    return {
        "audit": "reproducibility",
        "status": "TODO",
        "next_step": (
            "Wire to generate_daily_report.py with a fixed --as-of date; "
            "diff the resulting paper_trades against the original. "
            "Bit-identical = no hidden randomness or time-dependent inputs."
        ),
    }


# ---------------------------------------------------------------------------
# 12. YOLO boost edge
# ---------------------------------------------------------------------------

def audit_yolo_edge(conn: sqlite3.Connection) -> dict[str, Any]:
    """Compare WR / expectancy of YOLO-boosted vs unboosted trades.

    yolo_boost_pts > 0 means the score that qualified the trade
    received some chart-pattern uplift. If WR/expectancy are no better
    than yolo_boost_pts == 0, kill YOLO — the ~30 min/wk of compute
    isn't earning its keep.
    """
    rows = conn.execute(
        """
        SELECT yolo_boost_pts, pnl_pct, r_multiple, status
        FROM paper_trades
        WHERE status IN ('closed', 'stopped_out', 'target_hit', 'expired')
          AND r_multiple IS NOT NULL
        """
    ).fetchall()

    boosted = [r for r in rows if (r["yolo_boost_pts"] or 0) > 0]
    unboosted = [r for r in rows if (r["yolo_boost_pts"] or 0) == 0]

    def _stats(group: list[sqlite3.Row], label: str) -> dict[str, Any]:
        if not group:
            return {"label": label, "n": 0}
        wins = sum(1 for r in group if (r["pnl_pct"] or 0) > 0)
        rs = [float(r["r_multiple"] or 0) for r in group]
        return {
            "label": label,
            "n": len(group),
            "n_wins": wins,
            "win_rate_pct": round(wins / len(group) * 100, 1),
            "win_rate_ci_lo_95": _wilson_lo(wins, len(group)),
            "expectancy_r": round(sum(rs) / len(rs), 3) if rs else 0.0,
            "avg_pnl_pct": round(sum(float(r["pnl_pct"] or 0) for r in group) / len(group), 2),
        }

    return {
        "audit": "yolo-edge",
        "boosted": _stats(boosted, "yolo_boost_pts > 0"),
        "unboosted": _stats(unboosted, "yolo_boost_pts == 0"),
        "kill_criterion": (
            "If after >= 15 trades each, boosted expectancy_r is "
            "<= unboosted expectancy_r AND boosted CI lower bound is "
            "<= unboosted CI lower bound, kill YOLO."
        ),
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

AUDITS: dict[str, Callable[[sqlite3.Connection], dict[str, Any]]] = {
    "risk-budget": audit_risk_budget,
    "stop-tightness": audit_stop_tightness,
    "clustering": audit_clustering,
    "regime-conflict": audit_regime_conflict,
    "exit-quality": audit_exit_quality,
    "sector-cap": audit_sector_cap,
    "turnover": audit_turnover,
    "correlation": audit_correlation,
    "attribution": audit_attribution,
    "signal-decay": audit_signal_decay,
    "reproducibility": audit_reproducibility,
    "yolo-edge": audit_yolo_edge,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audit",
        choices=["all", *AUDITS.keys()],
        help="Which audit to run (or 'all').",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("/data/trader_koo.db"),
        help="SQLite DB path (default: /data/trader_koo.db)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON instead of pretty text.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.db_path.exists():
        raise SystemExit(f"DB not found: {args.db_path}")

    targets = list(AUDITS.keys()) if args.audit == "all" else [args.audit]
    conn = _connect(args.db_path)
    results: dict[str, dict[str, Any]] = {}
    try:
        for name in targets:
            try:
                results[name] = AUDITS[name](conn)
            except Exception as exc:  # noqa: BLE001
                results[name] = {"audit": name, "error": str(exc)}
    finally:
        conn.close()

    if args.json:
        json.dump(results, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        for name, payload in results.items():
            print(f"\n=== {name.upper()} ===")
            print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
