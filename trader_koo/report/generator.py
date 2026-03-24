"""Main report orchestrator: fetch_signals and fetch_report_payload."""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from trader_koo.catalyst_data import build_earnings_calendar_payload
from trader_koo.llm_narrative import llm_enabled, llm_status
from trader_koo.paper_trades import (
    PAPER_TRADE_ENABLED,
    create_paper_trades_from_report,
    mark_to_market,
)
from trader_koo.report.market_context import (
    _build_regime_context,
    _fetch_technical_context,
    _fetch_volatility_inputs,
    _report_warnings as _mc_report_warnings,
)
from trader_koo.report.pattern_analysis import (
    _summarize_yolo_lifecycle,
    _yolo_snapshot_matches,
    fetch_yolo_delta,
    fetch_yolo_pattern_persistence,
)
from trader_koo.report.setup_scoring import (
    DEBATE_ENGINE_ENABLED,
    SETUP_EVAL_ENABLED,
    SETUP_EVAL_HIT_THRESHOLD_PCT,
    SETUP_EVAL_MIN_SAMPLE,
    SETUP_EVAL_TRACK_LIMIT,
    SETUP_EVAL_WINDOW_DAYS,
    _apply_agreement_tier_adjustment,
    _apply_debate_guardrails,
    _apply_debate_payload,
    _apply_llm_narrative_overrides,
    _apply_setup_eval_fields,
    _describe_setup,
    _persist_setup_call_candidates,
    _refresh_setup_eval_surfaces,
    _score_open_setup_call_outcomes,
    _score_setup_from_confluence,
    _setup_cluster_rows,
    _summarize_setup_call_evaluations,
    _yolo_age_factor,
    _yolo_pattern_bias,
    _yolo_recency_label,
    build_no_trade_conditions,
    build_tonight_key_changes,
    ensure_setup_call_eval_schema,
)
from trader_koo.report.utils import (
    _clamp,
    _normalize_report_kind,
    _setup_tier,
    days_since_date,
    hours_since,
    market_calendar_context,
    row_to_dict,
    table_exists,
    tail_text,
)

LOG = logging.getLogger(__name__)

# Accumulates warnings from except blocks during report generation so the
# frontend can surface which sections degraded.  Reset at the start of each
# fetch_report_payload() call.
_report_warnings: list[str] = []


def _apply_ensemble_adjustment(row: dict[str, Any]) -> None:
    """Adjust score and tier based on technical ensemble agreement.

    If the 5-strategy ensemble agrees with confluence bias, boost score.
    If it disagrees, penalize. High agreement (>70%) = bigger effect.
    """
    ensemble_bias = row.get("ensemble_bias")
    confluence_bias = row.get("signal_bias")
    agreement = row.get("ensemble_agreement_pct", 0)
    net_score = row.get("ensemble_net_score", 0)

    if not ensemble_bias or not confluence_bias:
        return

    score = float(row.get("score") or 0)
    tier = str(row.get("setup_tier") or "D")

    if ensemble_bias == confluence_bias:
        # Agreement: boost score proportionally to ensemble confidence
        boost = min(8.0, abs(net_score) * 0.1)
        if agreement >= 80:
            boost *= 1.5
        score = min(100.0, score + boost)
        row["ensemble_effect"] = f"+{boost:.1f} (ensemble agrees)"
    elif ensemble_bias != "neutral" and confluence_bias != "neutral" and ensemble_bias != confluence_bias:
        # Disagreement: penalize and potentially downgrade tier
        penalty = min(10.0, abs(net_score) * 0.12)
        if agreement >= 80:
            penalty *= 1.5
        score = max(0.0, score - penalty)
        row["ensemble_effect"] = f"-{penalty:.1f} (ensemble disagrees)"
        # Strong disagreement with high agreement: downgrade tier
        if penalty >= 8.0 and tier in {"A", "B"}:
            tier = "B" if tier == "A" else "C"
    else:
        row["ensemble_effect"] = "neutral"

    row["score"] = round(score, 1)
    row["confluence_score"] = round(score, 1)
    row["setup_tier"] = tier


def fetch_signals(conn: sqlite3.Connection) -> dict[str, Any]:
    """Market signals for the daily report: 52W extremes, top YOLO patterns, candle signals."""
    signals: dict[str, Any] = {
        "near_52w_high": [],
        "near_52w_low": [],
        "movers_up_today": [],
        "movers_down_today": [],
        "large_moves_today": [],
        "market_breadth": {},
        "volatility_context": {},
        "regime_context": {},
        "yolo_top_today": [],
        "candle_patterns_today": [],
        "sector_heatmap": [],
        "setup_quality_top": [],
        "setup_quality_lookup": {},
        "watchlist_candidates": [],
        "setup_evaluation": {},
        "earnings_catalysts": {},
        "tonight_key_changes": [],
    }
    movers_all: list[dict[str, Any]] = []
    fundamentals_map: dict[str, dict[str, Any]] = {}
    yolo_by_ticker: dict[str, dict[str, Any]] = {}
    vol_by_ticker: dict[str, dict[str, float | None]] = {}
    vol_ctx: dict[str, Any] = {}
    technical_by_ticker: dict[str, dict[str, Any]] = {}
    yolo_history_rows: list[dict[str, Any]] = []
    yolo_asof_dates_desc: dict[str, list[str]] = {"daily": [], "weekly": []}

    try:
        vol_by_ticker, vol_ctx = _fetch_volatility_inputs(conn)
        signals["volatility_context"] = vol_ctx
    except Exception as exc:
        LOG.error("Volatility inputs fetch failed: %s", exc, exc_info=True)
        _report_warnings.append("volatility_inputs_failed")
    try:
        signals["regime_context"] = _build_regime_context(conn)
    except Exception as exc:
        LOG.error("Regime context build failed: %s", exc, exc_info=True)
        _report_warnings.append("regime_context_failed")
        signals["regime_context"] = {}
    # Cache VIX metrics in nightly report (avoids live recompute on every page load)
    try:
        from trader_koo.structure.vix_metrics import compute_vix_metrics
        signals["vix_metrics"] = compute_vix_metrics(conn)
    except Exception as exc:
        LOG.debug("VIX metrics cache in report skipped: %s", exc)
        signals["vix_metrics"] = None
    # Drain warnings accumulated by market_context functions.
    _report_warnings.extend(_mc_report_warnings)
    _mc_report_warnings.clear()
    try:
        technical_by_ticker = _fetch_technical_context(conn)
    except Exception as exc:
        LOG.error("Technical context fetch failed: %s", exc, exc_info=True)
        _report_warnings.append("technical_context_failed")
        technical_by_ticker = {}
    # Drain any remaining market_context warnings.
    _report_warnings.extend(_mc_report_warnings)
    _mc_report_warnings.clear()

    # ── 52W high / low proximity ─────────────────────────────────────────────
    try:
        rows = conn.execute("""
            WITH latest AS (
                SELECT ticker, MAX(date) AS max_date FROM price_daily GROUP BY ticker
            ),
            latest_close AS (
                SELECT p.ticker, CAST(p.close AS REAL) AS close, p.date
                FROM price_daily p
                JOIN latest l ON p.ticker = l.ticker AND p.date = l.max_date
            ),
            range_52w AS (
                SELECT p.ticker,
                       MAX(CAST(p.high AS REAL)) AS high_52w,
                       MIN(CAST(p.low  AS REAL)) AS low_52w
                FROM price_daily p
                JOIN latest l ON p.ticker = l.ticker
                WHERE p.date >= date(l.max_date, '-365 days')
                GROUP BY p.ticker
            )
            SELECT lc.ticker, lc.close, lc.date, r.high_52w, r.low_52w
            FROM latest_close lc
            JOIN range_52w r ON lc.ticker = r.ticker
            WHERE lc.close > 0 AND r.high_52w > 0 AND r.low_52w > 0
        """).fetchall()
        near_high: list[dict] = []
        near_low: list[dict] = []
        for row in rows:
            ticker, close, _, high_52w, low_52w = row
            close, high_52w, low_52w = float(close), float(high_52w), float(low_52w)
            pct_from_high = (high_52w - close) / high_52w * 100
            pct_from_low  = (close - low_52w) / low_52w * 100
            if pct_from_high <= 3.0:
                near_high.append({"ticker": ticker, "close": round(close, 2),
                                   "high_52w": round(high_52w, 2),
                                   "pct_from_high": round(pct_from_high, 2)})
            if pct_from_low <= 3.0:
                near_low.append({"ticker": ticker, "close": round(close, 2),
                                  "low_52w": round(low_52w, 2),
                                  "pct_from_low": round(pct_from_low, 2)})
        signals["near_52w_high"] = sorted(near_high, key=lambda x: x["pct_from_high"])
        signals["near_52w_low"]  = sorted(near_low,  key=lambda x: x["pct_from_low"])
    except Exception as exc:
        LOG.warning("52-week high/low proximity failed: %s", exc)
        _report_warnings.append("52w_proximity_failed")

    # ── Large moves vs prior close + breadth snapshot ───────────────────────
    try:
        threshold_raw = os.getenv("TRADER_KOO_REPORT_LARGE_MOVE_PCT", "2.5").strip()
        try:
            large_move_threshold = float(threshold_raw)
        except ValueError:
            large_move_threshold = 2.5
        large_move_threshold = max(0.5, min(50.0, large_move_threshold))

        rows = conn.execute(
            """
            WITH ranked AS (
                SELECT
                    ticker,
                    date,
                    CAST(close AS REAL) AS close,
                    CAST(volume AS REAL) AS volume,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
                FROM price_daily
            ),
            latest AS (
                SELECT ticker, date, close, volume
                FROM ranked
                WHERE rn = 1
            ),
            prev AS (
                SELECT ticker, date AS prev_date, close AS prev_close
                FROM ranked
                WHERE rn = 2
            ),
            range_52w AS (
                SELECT
                    p.ticker,
                    MAX(CAST(p.high AS REAL)) AS high_52w,
                    MIN(CAST(p.low  AS REAL)) AS low_52w
                FROM price_daily p
                JOIN latest l ON l.ticker = p.ticker
                WHERE p.date >= date(l.date, '-365 days')
                GROUP BY p.ticker
            )
            SELECT
                l.ticker,
                l.date AS latest_date,
                l.close,
                p.prev_date,
                p.prev_close,
                l.volume,
                r.high_52w,
                r.low_52w
            FROM latest l
            JOIN prev p ON p.ticker = l.ticker
            LEFT JOIN range_52w r ON r.ticker = l.ticker
            WHERE l.close > 0 AND p.prev_close > 0
            """
        ).fetchall()

        movers: list[dict[str, Any]] = []
        advancers = 0
        decliners = 0
        unchanged = 0
        pct_changes: list[float] = []
        up_threshold = 0.05
        down_threshold = -0.05

        for row in rows:
            ticker = str(row[0])
            latest_date = row[1]
            close = float(row[2])
            prev_close = float(row[4])
            volume = int(float(row[5] or 0))
            high_52w = float(row[6]) if row[6] is not None else None
            low_52w = float(row[7]) if row[7] is not None else None

            pct_change = ((close - prev_close) / prev_close) * 100.0
            pct_changes.append(pct_change)
            if pct_change > up_threshold:
                advancers += 1
            elif pct_change < down_threshold:
                decliners += 1
            else:
                unchanged += 1

            pct_from_high: float | None = None
            pct_from_low: float | None = None
            if high_52w and high_52w > 0:
                pct_from_high = ((high_52w - close) / high_52w) * 100.0
            if low_52w and low_52w > 0:
                pct_from_low = ((close - low_52w) / low_52w) * 100.0

            movers.append(
                {
                    "ticker": ticker,
                    "date": latest_date,
                    "close": round(close, 2),
                    "prev_close": round(prev_close, 2),
                    "pct_change": round(pct_change, 2),
                    "volume": volume,
                    "high_52w": round(high_52w, 2) if high_52w is not None else None,
                    "low_52w": round(low_52w, 2) if low_52w is not None else None,
                    "pct_from_high": round(pct_from_high, 2) if pct_from_high is not None else None,
                    "pct_from_low": round(pct_from_low, 2) if pct_from_low is not None else None,
                    "near_52w_high": bool(pct_from_high is not None and pct_from_high <= 3.0),
                    "near_52w_low": bool(pct_from_low is not None and pct_from_low <= 3.0),
                }
            )

        signals["movers_up_today"] = sorted(
            [m for m in movers if float(m["pct_change"]) > 0.0],
            key=lambda x: float(x["pct_change"]),
            reverse=True,
        )[:20]
        signals["movers_down_today"] = sorted(
            [m for m in movers if float(m["pct_change"]) < 0.0],
            key=lambda x: float(x["pct_change"]),
        )[:20]
        signals["large_moves_today"] = sorted(
            [m for m in movers if abs(float(m["pct_change"])) >= large_move_threshold],
            key=lambda x: abs(float(x["pct_change"])),
            reverse=True,
        )[:40]

        total = advancers + decliners + unchanged
        avg_pct = (sum(pct_changes) / len(pct_changes)) if pct_changes else None
        median_pct: float | None = None
        if pct_changes:
            sorted_changes = sorted(pct_changes)
            n = len(sorted_changes)
            if n % 2 == 1:
                median_pct = sorted_changes[n // 2]
            else:
                median_pct = (sorted_changes[(n // 2) - 1] + sorted_changes[n // 2]) / 2.0
        signals["market_breadth"] = {
            "total_tickers": total,
            "advancers": advancers,
            "decliners": decliners,
            "unchanged": unchanged,
            "pct_advancing": round((advancers / total) * 100.0, 2) if total > 0 else None,
            "avg_pct_change": round(avg_pct, 2) if avg_pct is not None else None,
            "median_pct_change": round(median_pct, 2) if median_pct is not None else None,
            "large_move_threshold_pct": round(large_move_threshold, 2),
            "large_move_count": len(signals["large_moves_today"]),
        }
        movers_all = movers
    except Exception as exc:
        LOG.warning("Large moves / market breadth computation failed: %s", exc)
        _report_warnings.append("market_breadth_failed")

    # ── Top YOLO patterns from today's run ───────────────────────────────────
    try:
        row = conn.execute("SELECT MAX(as_of_date) FROM yolo_patterns").fetchone()
        latest_asof = row[0] if row else None
        if latest_asof:
            hist_rows = conn.execute(
                """
                SELECT ticker, timeframe, pattern, x0_date, x1_date, as_of_date
                FROM yolo_patterns
                WHERE as_of_date IS NOT NULL
                ORDER BY as_of_date DESC
                """
            ).fetchall()
            yolo_history_rows = [
                {
                    "ticker": str(r[0] or ""),
                    "timeframe": str(r[1] or ""),
                    "pattern": str(r[2] or ""),
                    "x0_date": r[3],
                    "x1_date": r[4],
                    "as_of_date": r[5],
                }
                for r in hist_rows
            ]
            for timeframe_key in ("daily", "weekly"):
                dates = {
                    str(row.get("as_of_date") or "")
                    for row in yolo_history_rows
                    if str(row.get("timeframe") or "").strip().lower() == timeframe_key
                    and str(row.get("as_of_date") or "").strip()
                }
                yolo_asof_dates_desc[timeframe_key] = sorted(dates, reverse=True)
            asof_date: dt.date | None = None
            try:
                asof_date = dt.date.fromisoformat(str(latest_asof))
            except (TypeError, ValueError) as exc:
                LOG.warning("Failed to parse YOLO asof_date %r: %s", latest_asof, exc)
                asof_date = None
            yolo_rows = conn.execute("""
                SELECT ticker, timeframe, pattern,
                       CAST(confidence AS REAL) AS confidence,
                       x0_date, x1_date
                FROM yolo_patterns
                WHERE as_of_date = ?
                ORDER BY confidence DESC
                LIMIT 30
            """, (latest_asof,)).fetchall()
            yolo_top_today: list[dict[str, Any]] = []
            for r in yolo_rows:
                x1_date = r[5]
                age_days: int | None = None
                if asof_date is not None and x1_date and len(str(x1_date)) >= 10:
                    try:
                        x1_dt = dt.date.fromisoformat(str(x1_date)[:10])
                        age_days = max(0, (asof_date - x1_dt).days)
                    except (TypeError, ValueError) as exc:
                        LOG.warning("Failed to parse YOLO x1_date %r: %s", x1_date, exc)
                        age_days = None
                yolo_top_today.append(
                    {
                        "ticker": r[0],
                        "timeframe": r[1],
                        "pattern": r[2],
                        "confidence": round(float(r[3]), 3),
                        "x0_date": r[4],
                        "x1_date": x1_date,
                        "as_of_date": latest_asof,
                        "age_days": age_days,
                        "recency": _yolo_recency_label(age_days, r[1]),
                    }
                )
            for item in yolo_top_today:
                item.update(
                    _summarize_yolo_lifecycle(
                        item,
                        yolo_history_rows,
                        yolo_asof_dates_desc.get(str(item.get("timeframe") or "").strip().lower(), []),
                    )
                )
            yolo_top_today.sort(
                key=lambda item: (
                    _yolo_age_factor(item.get("age_days"), item.get("timeframe")),
                    int(item.get("current_streak") or 0),
                    1 if str(item.get("timeframe") or "").strip().lower() == "daily" else 0,
                    float(item.get("confidence") or 0.0),
                    -(int(item.get("age_days")) if isinstance(item.get("age_days"), int) else 9999),
                ),
                reverse=True,
            )
            signals["yolo_top_today"] = yolo_top_today

            # Best YOLO signal per ticker for setup scoring.
            full_rows = conn.execute(
                """
                SELECT ticker, timeframe, pattern,
                       CAST(confidence AS REAL) AS confidence,
                       x0_date, x1_date
                FROM yolo_patterns
                WHERE as_of_date = ?
                ORDER BY confidence DESC
                """,
                (latest_asof,),
            ).fetchall()
            for r in full_rows:
                ticker = str(r[0])
                conf = float(r[3] or 0.0)
                x1_date = r[5]
                age_days: int | None = None
                if asof_date is not None and x1_date and len(str(x1_date)) >= 10:
                    try:
                        x1_dt = dt.date.fromisoformat(str(x1_date)[:10])
                        age_days = max(0, (asof_date - x1_dt).days)
                    except (TypeError, ValueError) as exc:
                        LOG.warning("Failed to parse YOLO x1_date %r for scoring: %s", x1_date, exc)
                        age_days = None
                candidate = {
                    "ticker": ticker,
                    "timeframe": r[1],
                    "pattern": r[2],
                    "confidence": round(conf, 3),
                    "x0_date": r[4],
                    "x1_date": x1_date,
                    "as_of_date": latest_asof,
                    "age_days": age_days,
                }
                candidate.update(
                    _summarize_yolo_lifecycle(
                        candidate,
                        yolo_history_rows,
                        yolo_asof_dates_desc.get(str(candidate.get("timeframe") or "").strip().lower(), []),
                    )
                )
                prev = yolo_by_ticker.get(ticker)
                if prev is None:
                    yolo_by_ticker[ticker] = candidate
                    continue
                prev_daily = str(prev.get("timeframe") or "") == "daily"
                cand_daily = str(candidate.get("timeframe") or "") == "daily"
                if cand_daily and not prev_daily:
                    yolo_by_ticker[ticker] = candidate
                    continue
                prev_rank = (
                    _yolo_age_factor(prev.get("age_days"), prev.get("timeframe")),
                    int(prev.get("current_streak") or 0),
                    float(prev.get("confidence") or 0.0),
                )
                cand_rank = (
                    _yolo_age_factor(candidate.get("age_days"), candidate.get("timeframe")),
                    int(candidate.get("current_streak") or 0),
                    float(candidate.get("confidence") or 0.0),
                )
                if cand_daily == prev_daily and cand_rank > prev_rank:
                    yolo_by_ticker[ticker] = candidate
    except Exception as exc:
        LOG.error("YOLO patterns section failed: %s", exc, exc_info=True)
        _report_warnings.append("yolo_patterns_failed")

    # ── Fundamentals snapshot map (discount/PEG + sector/industry metadata) ─
    try:
        snap_row = conn.execute("SELECT MAX(snapshot_ts) FROM finviz_fundamentals").fetchone()
        latest_snap = snap_row[0] if snap_row else None
        if latest_snap:
            fund_rows = conn.execute(
                """
                SELECT ticker, discount_pct, peg, raw_json
                FROM finviz_fundamentals
                WHERE snapshot_ts = ?
                """,
                (latest_snap,),
            ).fetchall()
            for r in fund_rows:
                ticker = str(r[0])
                discount = float(r[1]) if r[1] is not None else None
                peg = float(r[2]) if r[2] is not None else None
                sector = None
                industry = None
                raw = r[3]
                if raw:
                    try:
                        raw_obj = json.loads(str(raw))
                        if isinstance(raw_obj, dict):
                            sector = raw_obj.get("Sector") or raw_obj.get("sector")
                            industry = raw_obj.get("Industry") or raw_obj.get("industry")
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        LOG.warning("Failed to parse fundamentals JSON for %s: %s", ticker, exc)
                fundamentals_map[ticker] = {
                    "discount_pct": round(discount, 2) if discount is not None else None,
                    "peg": round(peg, 2) if peg is not None else None,
                    "sector": str(sector).strip() if sector else "Unknown",
                    "industry": str(industry).strip() if industry else None,
                }
    except Exception as exc:
        LOG.error("Fundamentals snapshot fetch failed: %s", exc, exc_info=True)
        _report_warnings.append("fundamentals_snapshot_failed")

    # ── Sector heatmap + setup quality scoring ───────────────────────────────
    try:
        sector_buckets: dict[str, dict[str, Any]] = {}
        setup_rows: list[dict[str, Any]] = []
        for m in movers_all:
            ticker = str(m.get("ticker") or "").upper()
            if not ticker:
                continue
            pct_change = float(m.get("pct_change") or 0.0)
            near_high = bool(m.get("near_52w_high"))
            near_low = bool(m.get("near_52w_low"))
            fund = fundamentals_map.get(ticker, {})
            sector = str(fund.get("sector") or "Unknown").strip() or "Unknown"

            bucket = sector_buckets.setdefault(
                sector,
                {
                    "sector": sector,
                    "tickers": 0,
                    "advancers": 0,
                    "decliners": 0,
                    "unchanged": 0,
                    "near_high_count": 0,
                    "near_low_count": 0,
                    "_changes": [],
                },
            )
            bucket["tickers"] += 1
            bucket["_changes"].append(pct_change)
            if pct_change > 0.05:
                bucket["advancers"] += 1
            elif pct_change < -0.05:
                bucket["decliners"] += 1
            else:
                bucket["unchanged"] += 1
            if near_high:
                bucket["near_high_count"] += 1
            if near_low:
                bucket["near_low_count"] += 1

            # Setup quality score: valuation + momentum + AI signal freshness.
            score = 50.0
            discount = fund.get("discount_pct")
            discount_component = 0.0
            if isinstance(discount, (int, float)):
                discount_component = _clamp(float(discount) * 0.8, -20.0, 20.0)
                score += discount_component

            peg = fund.get("peg")
            peg_component = 0.0
            if isinstance(peg, (int, float)) and float(peg) > 0:
                peg_v = float(peg)
                if peg_v <= 0.8:
                    peg_component = 15.0
                elif peg_v <= 1.5:
                    peg_component = 10.0
                elif peg_v <= 2.5:
                    peg_component = 4.0
                elif peg_v <= 4.0:
                    peg_component = -4.0
                else:
                    peg_component = -8.0
                score += peg_component

            momentum_component = _clamp(pct_change * 1.5, -12.0, 12.0)
            score += momentum_component

            proximity_component = 0.0
            if near_high and pct_change > 0:
                proximity_component += 5.0
            if near_low and pct_change < 0:
                proximity_component -= 5.0
            score += proximity_component

            yolo = yolo_by_ticker.get(ticker)
            vol = vol_by_ticker.get(ticker, {})
            tech = technical_by_ticker.get(ticker, {})
            yolo_component = 0.0
            volatility_component = 0.0
            yolo_pattern = None
            yolo_confidence = None
            yolo_age_days = None
            yolo_timeframe = None
            yolo_first_seen_asof = None
            yolo_last_seen_asof = None
            yolo_snapshots_seen = None
            yolo_current_streak = None
            yolo_first_seen_days_ago = None
            signal_bias = "neutral"
            atr_pct_14 = vol.get("atr_pct_14")
            realized_vol_20 = vol.get("realized_vol_20")
            bb_width_20 = vol.get("bb_width_20")
            if yolo:
                yolo_pattern = yolo.get("pattern")
                yolo_confidence = yolo.get("confidence")
                yolo_age_days = yolo.get("age_days")
                yolo_timeframe = yolo.get("timeframe")
                yolo_first_seen_asof = yolo.get("first_seen_asof")
                yolo_last_seen_asof = yolo.get("last_seen_asof")
                yolo_snapshots_seen = yolo.get("snapshots_seen")
                yolo_current_streak = yolo.get("current_streak")
                yolo_first_seen_days_ago = yolo.get("first_seen_days_ago")
                signal_bias = _yolo_pattern_bias(yolo_pattern)
                conf = float(yolo_confidence or 0.0)
                yolo_component += _clamp(conf * 20.0, 0.0, 18.0)
                if str(yolo_timeframe) == "daily":
                    yolo_component += 2.0
                if isinstance(yolo_age_days, int):
                    if yolo_age_days <= 10:
                        yolo_component += 3.0
                    elif yolo_age_days <= 30:
                        yolo_component += 1.0
                score += yolo_component

            if isinstance(atr_pct_14, (int, float)):
                atr_v = float(atr_pct_14)
                if atr_v < 1.0:
                    volatility_component -= 2.0
                elif atr_v <= 4.5:
                    volatility_component += 5.0
                elif atr_v <= 7.0:
                    volatility_component += 2.0
                elif atr_v <= 10.0:
                    volatility_component -= 2.0
                else:
                    volatility_component -= 6.0

            if isinstance(bb_width_20, (int, float)):
                bb_v = float(bb_width_20)
                if bb_v <= 6.0 and abs(pct_change) >= 1.5:
                    volatility_component += 3.0
                elif bb_v >= 18.0:
                    volatility_component -= 3.0

            if isinstance(realized_vol_20, (int, float)):
                rv_v = float(realized_vol_20)
                if rv_v > 60.0:
                    volatility_component -= 3.0
                elif rv_v >= 20.0:
                    volatility_component += 2.0
                elif rv_v < 12.0:
                    volatility_component -= 1.0

            vix_pctile = vol_ctx.get("vix_percentile_1y")
            if isinstance(vix_pctile, (int, float)):
                vix_p = float(vix_pctile)
                if vix_p >= 90.0:
                    volatility_component -= 4.0
                elif vix_p >= 75.0:
                    volatility_component -= 2.0
                elif vix_p <= 35.0:
                    volatility_component += 1.0

            volatility_component = _clamp(volatility_component, -15.0, 15.0)
            score += volatility_component

            row = {
                "ticker": ticker,
                "score": round(_clamp(score, 0.0, 100.0), 1),
                "confluence_score": round(_clamp(score, 0.0, 100.0), 1),
                "setup_tier": _setup_tier(round(_clamp(score, 0.0, 100.0), 1)),
                "sector": sector,
                "pct_change": round(pct_change, 2),
                "discount_pct": discount,
                "peg": peg,
                "atr_pct_14": atr_pct_14,
                "realized_vol_20": realized_vol_20,
                "bb_width_20": bb_width_20,
                "avg_volume_20": tech.get("avg_volume_20"),
                "volume_ratio_20": tech.get("volume_ratio_20"),
                "recent_range_pct_10": tech.get("recent_range_pct_10"),
                "recent_range_pct_20": tech.get("recent_range_pct_20"),
                "near_52w_high": near_high,
                "near_52w_low": near_low,
                "yolo_pattern": yolo_pattern,
                "yolo_confidence": yolo_confidence,
                "yolo_age_days": yolo_age_days,
                "yolo_timeframe": yolo_timeframe,
                "yolo_first_seen_asof": yolo_first_seen_asof,
                "yolo_last_seen_asof": yolo_last_seen_asof,
                "yolo_snapshots_seen": yolo_snapshots_seen,
                "yolo_current_streak": yolo_current_streak,
                "yolo_first_seen_days_ago": yolo_first_seen_days_ago,
                "signal_bias": signal_bias,
                "close": tech.get("close"),
                "ma20": tech.get("ma20"),
                "ma50": tech.get("ma50"),
                "pct_vs_ma20": tech.get("pct_vs_ma20"),
                "pct_vs_ma50": tech.get("pct_vs_ma50"),
                "pct_from_20d_high": tech.get("pct_from_20d_high"),
                "pct_from_20d_low": tech.get("pct_from_20d_low"),
                "trend_state": tech.get("trend_state") or "mixed",
                "level_context": tech.get("level_context") or "mid_range",
                "support_level": tech.get("support_level"),
                "support_zone_low": tech.get("support_zone_low"),
                "support_zone_high": tech.get("support_zone_high"),
                "support_tier": tech.get("support_tier"),
                "support_touches": tech.get("support_touches"),
                "resistance_level": tech.get("resistance_level"),
                "resistance_zone_low": tech.get("resistance_zone_low"),
                "resistance_zone_high": tech.get("resistance_zone_high"),
                "resistance_tier": tech.get("resistance_tier"),
                "resistance_touches": tech.get("resistance_touches"),
                "pct_to_support": tech.get("pct_to_support"),
                "pct_to_resistance": tech.get("pct_to_resistance"),
                "range_position": tech.get("range_position"),
                "stretch_state": tech.get("stretch_state") or "normal",
                "breakout_state": tech.get("breakout_state") or "none",
                "structure_state": tech.get("structure_state") or "normal",
                "candle_pattern": None,
                "candle_bias": "neutral",
                "candle_confidence": None,
                "observation": "",
                "actionability": "watch-only",
                "action": "",
                "risk_note": "",
                "technical_read": "",
                "components": {
                    "discount": round(discount_component, 2),
                    "peg": round(peg_component, 2),
                    "momentum": round(momentum_component, 2),
                    "proximity": round(proximity_component, 2),
                    "volatility": round(volatility_component, 2),
                    "yolo": round(yolo_component, 2),
                },
            }
            row.update(_score_setup_from_confluence(row))

            # Enrich with 5-strategy technical ensemble
            try:
                from trader_koo.analysis.technical_ensemble import compute_technical_ensemble

                ticker_ohlcv = conn.execute(
                    "SELECT date, open, high, low, close, volume FROM price_daily "
                    "WHERE ticker = ? AND close IS NOT NULL ORDER BY date ASC",
                    (row.get("ticker"),),
                ).fetchall()
                if len(ticker_ohlcv) >= 64:
                    import pandas as pd

                    ticker_df = pd.DataFrame(
                        ticker_ohlcv,
                        columns=["date", "open", "high", "low", "close", "volume"],
                    )
                    ensemble = compute_technical_ensemble(ticker_df)
                    row["ensemble_bias"] = ensemble["aggregate"]["bias"]
                    row["ensemble_net_score"] = ensemble["aggregate"]["net_score"]
                    row["ensemble_agreement_pct"] = ensemble["aggregate"]["agreement_pct"]
                    row["ensemble_strategies"] = ensemble["strategies"]
            except Exception as exc:
                LOG.debug("Ensemble enrichment skipped for %s: %s", row.get("ticker"), exc)

            # Adjust score/tier based on ensemble agreement with confluence bias
            _apply_ensemble_adjustment(row)

            setup_rows.append(row)

        sector_rows: list[dict[str, Any]] = []
        for _, bucket in sector_buckets.items():
            changes = [float(x) for x in bucket.pop("_changes", [])]
            if not changes:
                continue
            changes_sorted = sorted(changes)
            n = len(changes_sorted)
            if n % 2 == 1:
                median_change = changes_sorted[n // 2]
            else:
                median_change = (changes_sorted[(n // 2) - 1] + changes_sorted[n // 2]) / 2.0
            tickers = int(bucket.get("tickers") or 0)
            advancers = int(bucket.get("advancers") or 0)
            bucket["avg_pct_change"] = round(sum(changes) / len(changes), 2)
            bucket["median_pct_change"] = round(median_change, 2)
            bucket["pct_advancing"] = round((advancers / tickers) * 100.0, 2) if tickers > 0 else None
            sector_rows.append(bucket)

        sector_rows.sort(
            key=lambda r: (
                float(r.get("avg_pct_change") or 0.0),
                float(r.get("pct_advancing") or 0.0),
                int(r.get("tickers") or 0),
            ),
            reverse=True,
        )
        signals["sector_heatmap"] = sector_rows

        setup_rows.sort(
            key=lambda r: (
                float(r.get("score") or 0.0),
                float(r.get("pct_change") or 0.0),
                float(r.get("discount_pct") or -999.0),
            ),
            reverse=True,
        )
        _apply_debate_payload(setup_rows)
        # Apply agreement score tier adjustment after debate payload, before guardrails
        for row in setup_rows:
            if isinstance(row, dict):
                _apply_agreement_tier_adjustment(row)
        _apply_debate_guardrails(setup_rows)
        _apply_llm_narrative_overrides(setup_rows, source="daily_report")
        signals["setup_quality_top"] = setup_rows[:40]
        signals["watchlist_candidates"] = [
            {
                "ticker": r.get("ticker"),
                "score": r.get("score"),
                "setup_tier": r.get("setup_tier"),
                "pct_change": r.get("pct_change"),
                "yolo_pattern": r.get("yolo_pattern"),
                "yolo_confidence": r.get("yolo_confidence"),
            }
            for r in setup_rows[:20]
        ]
    except Exception as e:
        LOG.error("Setup quality scoring failed: %s", e, exc_info=True)
        _report_warnings.append("setup_quality_scoring_failed")

    # ── Candle patterns on latest close date ─────────────────────────────────
    try:
        import pandas as pd
        from trader_koo.features.candle_patterns import (
            CandlePatternConfig,
            detect_candlestick_patterns,
        )

        row = conn.execute("SELECT MAX(date) FROM price_daily").fetchone()
        latest_date = row[0] if row else None
        if latest_date:
            raw_rows = conn.execute("""
                SELECT ticker, date, open, high, low, close
                FROM (
                    SELECT ticker, date, open, high, low, close,
                           ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
                    FROM price_daily
                )
                WHERE rn <= 10
                ORDER BY ticker, date
            """).fetchall()

            ticker_rows: dict = defaultdict(list)
            for r in raw_rows:
                ticker_rows[r[0]].append({
                    "date": r[1], "open": r[2], "high": r[3], "low": r[4], "close": r[5],
                })

            cfg = CandlePatternConfig(lookback_bars=10, use_talib=False)
            candle_signals: list[dict] = []
            candle_by_ticker: dict[str, dict[str, Any]] = {}
            for ticker, candles in ticker_rows.items():
                df = pd.DataFrame(candles)
                try:
                    result = detect_candlestick_patterns(df, cfg)
                    if not result.empty:
                        on_latest = result[result["date"] == latest_date]
                        for _, r in on_latest.iterrows():
                            candle_signals.append({
                                "ticker": ticker,
                                "pattern": str(r["pattern"]),
                                "bias": str(r.get("bias", "neutral")),
                                "confidence": round(float(r.get("confidence", 0.5)), 2),
                            })
                except Exception as exc:
                    LOG.warning("Candle pattern detection failed for %s: %s", ticker, exc)

            candle_signals.sort(key=lambda x: x["confidence"], reverse=True)
            signals["candle_patterns_today"] = candle_signals[:60]
            for row in candle_signals:
                ticker = str(row.get("ticker") or "").upper()
                prev = candle_by_ticker.get(ticker)
                if prev is None or float(row.get("confidence") or 0.0) > float(prev.get("confidence") or 0.0):
                    candle_by_ticker[ticker] = row

            if isinstance(setup_rows, list):
                for row in setup_rows:
                    ticker = str(row.get("ticker") or "").upper()
                    candle = candle_by_ticker.get(ticker) or {}
                    if candle:
                        row["candle_pattern"] = candle.get("pattern")
                        row["candle_bias"] = candle.get("bias") or "neutral"
                        row["candle_confidence"] = candle.get("confidence")
                    row.update(_score_setup_from_confluence(row))
                    readout = _describe_setup(row)
                    row.update(readout)

                _apply_debate_payload(setup_rows)
                # Apply agreement score tier adjustment after debate payload, before guardrails
                for row in setup_rows:
                    if isinstance(row, dict):
                        _apply_agreement_tier_adjustment(row)
                _apply_debate_guardrails(setup_rows)

                setup_rows.sort(
                    key=lambda r: (
                        float(r.get("score") or 0.0),
                        float(r.get("confirmation_count") or 0.0),
                        -float(r.get("contradiction_count") or 0.0),
                        float(r.get("pct_change") or 0.0),
                    ),
                    reverse=True,
                )

                _apply_llm_narrative_overrides(setup_rows, source="daily_report")
                signals["setup_quality_top"] = setup_rows[:40]
                signals["watchlist_candidates"] = [
                    {
                        "ticker": r.get("ticker"),
                        "score": r.get("score"),
                        "setup_tier": r.get("setup_tier"),
                        "pct_change": r.get("pct_change"),
                        "yolo_pattern": r.get("yolo_pattern"),
                        "yolo_confidence": r.get("yolo_confidence"),
                        "signal_bias": r.get("signal_bias"),
                        "observation": r.get("observation"),
                        "actionability": r.get("actionability"),
                        "action": r.get("action"),
                        "risk_note": r.get("risk_note"),
                        "technical_read": r.get("technical_read"),
                        "candle_pattern": r.get("candle_pattern"),
                        "candle_bias": r.get("candle_bias"),
                        "candle_confidence": r.get("candle_confidence"),
                    }
                    for r in setup_rows[:20]
                ]
                signals["setup_quality_lookup"] = {
                    str(row.get("ticker") or "").upper(): {
                        key: row.get(key)
                        for key in (
                            "ticker",
                            "score",
                            "setup_tier",
                            "setup_family",
                            "sector",
                            "pct_change",
                            "discount_pct",
                            "peg",
                            "atr_pct_14",
                            "realized_vol_20",
                            "bb_width_20",
                            "signal_bias",
                            "actionability",
                            "observation",
                            "action",
                            "risk_note",
                            "technical_read",
                            "confirmation_count",
                            "contradiction_count",
                            "valuation_bias",
                            "candle_pattern",
                            "candle_bias",
                            "candle_confidence",
                            "yolo_pattern",
                            "yolo_bias",
                            "yolo_confidence",
                            "yolo_age_days",
                            "yolo_timeframe",
                            "yolo_recency",
                            "yolo_direction_conflict",
                            "yolo_conflict_strength",
                            "debate_v1",
                            "debate_consensus_state",
                            "debate_consensus_bias",
                            "debate_agreement_score",
                            "debate_disagreement_count",
                            "debate_safety_adjustment",
                            "trend_state",
                            "level_context",
                            "support_level",
                            "resistance_level",
                            "pct_to_support",
                            "pct_to_resistance",
                            "trend_state",
                            "level_context",
                            "candle_pattern",
                            "candle_bias",
                            "candle_confidence",
                        )
                    }
                    for row in setup_rows
                    if isinstance(row, dict) and row.get("ticker")
                }
    except Exception as exc:
        LOG.error("Candle patterns / setup scoring section failed: %s", exc, exc_info=True)
        _report_warnings.append("candle_patterns_section_failed")

    # ── Pre-compute HMM regime for all tracked tickers ────────────────────
    try:
        import pandas as pd
        from trader_koo.structure.hmm_regime import predict_regimes as hmm_predict_regimes

        lookup_tickers = list((signals.get("setup_quality_lookup") or {}).keys())
        if not lookup_tickers:
            lookup_tickers = [
                str(r[0] or "").upper()
                for r in conn.execute("SELECT DISTINCT ticker FROM price_daily").fetchall()
                if str(r[0] or "").strip()
            ]
        hmm_cache: dict[str, dict[str, Any]] = {}
        hmm_t0 = dt.datetime.now(dt.timezone.utc)
        for ticker_sym in lookup_tickers:
            try:
                rows = conn.execute(
                    "SELECT date, open, high, low, close, volume FROM price_daily WHERE ticker = ? ORDER BY date",
                    (ticker_sym,),
                ).fetchall()
                if len(rows) < 140:
                    continue
                df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
                result = hmm_predict_regimes(df, ticker=ticker_sym)
                if result is not None:
                    hmm_cache[ticker_sym] = {
                        "current_state": result.get("current_state"),
                        "current_probs": result.get("current_probs"),
                        "transition_risk_pct": result.get("transition_risk_pct"),
                        "days_in_current": result.get("days_in_current"),
                        "regimes": result.get("regimes", [])[-120:],
                    }
            except Exception:
                pass
        hmm_elapsed = (dt.datetime.now(dt.timezone.utc) - hmm_t0).total_seconds()
        signals["hmm_regime_by_ticker"] = hmm_cache
        LOG.info("HMM regime pre-computed for %d/%d tickers in %.1fs", len(hmm_cache), len(lookup_tickers), hmm_elapsed)
    except Exception as exc:
        LOG.error("HMM regime pre-computation failed: %s", exc, exc_info=True)
        _report_warnings.append("hmm_regime_precompute_failed")
        signals["hmm_regime_by_ticker"] = {}

    # ── Pre-fetch FRED economic calendar (90 days forward) ────────────────
    try:
        from trader_koo.catalyst_data import fetch_economic_calendar
        today = dt.datetime.now(dt.timezone.utc).date()
        econ_events = fetch_economic_calendar(
            today.isoformat(),
            (today + dt.timedelta(days=90)).isoformat(),
            use_fred=True,
        )
        signals["economic_events"] = econ_events
        LOG.info("Economic calendar cached: %d events over 90 days", len(econ_events))
    except Exception as exc:
        LOG.error("Economic calendar pre-fetch failed: %s", exc)
        _report_warnings.append("economic_calendar_prefetch_failed")
        signals["economic_events"] = []

    return signals


def fetch_report_payload(
    db_path: Path,
    run_log: Path,
    tail_lines: int,
    report_kind: str = "daily",
) -> dict[str, Any]:
    global _report_warnings
    _report_warnings = []
    now = dt.datetime.now(dt.timezone.utc)
    report_kind_norm = _normalize_report_kind(report_kind)
    payload: dict[str, Any] = {
        "generated_ts": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "meta": {
            "report_kind": report_kind_norm,
            "llm": llm_status(),
        },
        "db_path": str(db_path),
        "db_exists": db_path.exists(),
        "ok": False,
        "warnings": [],
        "counts": {},
        "freshness": {},
        "market_session": market_calendar_context(now),
        "risk_filters": {},
        "latest_data": {},
        "latest_ingest_run": {},
        "yolo": {
            "table_exists": False,
            "summary": {},
            "timeframes": [],
            "delta": {},
            "delta_daily": {},
            "delta_weekly": {},
            "persistence": {},
        },
        "cron_log_path": str(run_log),
        "cron_log_tail": tail_text(run_log, lines=tail_lines),
    }
    if not db_path.exists():
        payload["warnings"].append("database_missing")
        return payload

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        counts = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM price_daily) AS price_rows,
                (SELECT COUNT(*) FROM finviz_fundamentals) AS fundamentals_rows,
                (SELECT COUNT(*) FROM options_iv) AS options_rows,
                (SELECT COUNT(*) FROM yolo_patterns) AS yolo_rows,
                (SELECT COUNT(DISTINCT ticker) FROM price_daily) AS tracked_tickers,
                (SELECT MAX(date) FROM price_daily) AS latest_price_date,
                (SELECT MAX(snapshot_ts) FROM finviz_fundamentals) AS latest_fund_snapshot,
                (SELECT MAX(snapshot_ts) FROM options_iv) AS latest_opt_snapshot,
                (SELECT MAX(detected_ts) FROM yolo_patterns) AS latest_yolo_ts
            """
        ).fetchone()
        payload["counts"] = {
            "tracked_tickers": int((counts["tracked_tickers"] or 0) if counts else 0),
            "price_rows": int((counts["price_rows"] or 0) if counts else 0),
            "fundamentals_rows": int((counts["fundamentals_rows"] or 0) if counts else 0),
            "options_rows": int((counts["options_rows"] or 0) if counts else 0),
            "yolo_rows": int((counts["yolo_rows"] or 0) if counts else 0),
        }
        payload["latest_data"] = {
            "price_date": counts["latest_price_date"] if counts else None,
            "fund_snapshot": counts["latest_fund_snapshot"] if counts else None,
            "options_snapshot": counts["latest_opt_snapshot"] if counts else None,
            "yolo_detected_ts": counts["latest_yolo_ts"] if counts else None,
        }
        price_age = days_since_date(counts["latest_price_date"], now) if counts else None
        fund_age = hours_since(counts["latest_fund_snapshot"], now) if counts else None
        opt_age = hours_since(counts["latest_opt_snapshot"], now) if counts else None
        yolo_age = hours_since(counts["latest_yolo_ts"], now) if counts else None
        payload["freshness"] = {
            "price_age_days": None if price_age is None else round(price_age, 2),
            "fund_age_hours": None if fund_age is None else round(fund_age, 2),
            "opt_age_hours": None if opt_age is None else round(opt_age, 2),
            "yolo_age_hours": None if yolo_age is None else round(yolo_age, 2),
        }

        if table_exists(conn, "ingest_runs"):
            latest_run = conn.execute(
                """
                SELECT run_id, started_ts, finished_ts, status, tickers_total, tickers_ok, tickers_failed, error_message
                FROM ingest_runs
                ORDER BY started_ts DESC
                LIMIT 1
                """
            ).fetchone()
            payload["latest_ingest_run"] = row_to_dict(latest_run)
            if latest_run and str(latest_run["status"] or "").lower() == "failed":
                payload["warnings"].append("latest_ingest_run_failed")
        else:
            payload["warnings"].append("ingest_runs_missing")

        if table_exists(conn, "yolo_patterns"):
            payload["yolo"]["table_exists"] = True
            yolo_summary = conn.execute(
                """
                SELECT
                    COUNT(*) AS rows_total,
                    COUNT(DISTINCT ticker) AS tickers_with_patterns,
                    MAX(detected_ts) AS latest_detected_ts,
                    MAX(as_of_date) AS latest_asof_date
                FROM yolo_patterns
                """
            ).fetchone()
            payload["yolo"]["summary"] = row_to_dict(yolo_summary)
            tf_rows = conn.execute(
                """
                SELECT
                    timeframe,
                    COUNT(*) AS rows_total,
                    COUNT(DISTINCT ticker) AS tickers_with_patterns,
                    AVG(confidence) AS avg_confidence,
                    MAX(detected_ts) AS latest_detected_ts,
                    MAX(as_of_date) AS latest_asof_date
                FROM yolo_patterns
                GROUP BY timeframe
                ORDER BY timeframe
                """
            ).fetchall()
            payload["yolo"]["timeframes"] = [dict(r) for r in tf_rows]
        else:
            payload["warnings"].append("yolo_patterns_missing")

        # Market signals (52W extremes, movers, sector/quality overlays, AI/candles)
        payload["signals"] = fetch_signals(conn)
        if SETUP_EVAL_ENABLED:
            eval_summary: dict[str, Any] = {"enabled": True, "scored_calls": 0, "open_calls": 0}
            try:
                ensure_setup_call_eval_schema(conn)
                signals_ref = payload.get("signals") if isinstance(payload.get("signals"), dict) else {}
                setup_rows_ref = signals_ref.get("setup_quality_top") if isinstance(signals_ref.get("setup_quality_top"), list) else []
                asof_for_eval = str((payload.get("latest_data") or {}).get("price_date") or "").strip()
                inserted_calls = 0
                if asof_for_eval and setup_rows_ref:
                    inserted_calls = _persist_setup_call_candidates(
                        conn,
                        generated_ts=str(payload.get("generated_ts") or ""),
                        report_kind=report_kind_norm,
                        asof_date=asof_for_eval,
                        setup_rows=setup_rows_ref,
                    )
                scored_this_run = _score_open_setup_call_outcomes(conn)
                conn.commit()
                summary, reliability_lookup = _summarize_setup_call_evaluations(
                    conn,
                    window_days=SETUP_EVAL_WINDOW_DAYS,
                    min_sample=SETUP_EVAL_MIN_SAMPLE,
                )
                calibration = _apply_setup_eval_fields(
                    setup_rows_ref,
                    reliability_lookup=reliability_lookup,
                    min_sample=SETUP_EVAL_MIN_SAMPLE,
                    hit_threshold_pct=SETUP_EVAL_HIT_THRESHOLD_PCT,
                )
                setup_rows_ref.sort(
                    key=lambda r: (
                        float(r.get("score") or 0.0),
                        float(r.get("confirmation_count") or 0.0),
                        -float(r.get("contradiction_count") or 0.0),
                        float(r.get("pct_change") or 0.0),
                    ),
                    reverse=True,
                )
                _refresh_setup_eval_surfaces(signals_ref)
                eval_summary = summary or {"enabled": True}
                eval_summary["tracked_this_run"] = int(min(len(setup_rows_ref), SETUP_EVAL_TRACK_LIMIT))
                eval_summary["inserted_calls"] = int(inserted_calls)
                eval_summary["scored_this_run"] = int(scored_this_run)
                eval_summary["calibration"] = calibration
            except Exception as exc:
                LOG.error("Setup evaluation failed: %s", exc, exc_info=True)
                _report_warnings.append("setup_evaluation_failed")
                conn.rollback()
                eval_summary = {
                    "enabled": True,
                    "error": "setup_evaluation_failed",
                }
            payload["signals"]["setup_evaluation"] = eval_summary
        else:
            payload["signals"]["setup_evaluation"] = {"enabled": False, "reason": "disabled_by_env"}

        # ── Paper Trade Integration ─────────────────────────────────
        paper_trade_result: dict[str, Any] = {"enabled": PAPER_TRADE_ENABLED}
        if PAPER_TRADE_ENABLED:
            try:
                asof_date_pt = str((payload.get("latest_data") or {}).get("price_date") or "").strip()
                setup_rows_pt = (
                    (payload.get("signals") or {}).get("setup_quality_top")
                    if isinstance((payload.get("signals") or {}).get("setup_quality_top"), list)
                    else []
                )
                generated_ts_pt = str(payload.get("generated_ts") or "")
                if asof_date_pt and setup_rows_pt:
                    inserted_pt = create_paper_trades_from_report(
                        conn,
                        setup_rows=setup_rows_pt,
                        report_date=asof_date_pt,
                        generated_ts=generated_ts_pt,
                    )
                    conn.commit()
                    paper_trade_result["inserted"] = inserted_pt
                else:
                    paper_trade_result["inserted"] = 0
                    paper_trade_result["skipped_reason"] = "no_date_or_setups"

                mtm_result = mark_to_market(conn)
                conn.commit()
                paper_trade_result["mtm"] = mtm_result
            except Exception as exc:
                conn.rollback()
                paper_trade_result["error"] = "paper_trade_failed"
                LOG.exception("Paper trade integration failed")
                _report_warnings.append("paper_trade_failed")
        payload["paper_trades"] = paper_trade_result
        market_date_raw = str((payload.get("market_session") or {}).get("market_date") or "").strip()
        market_date_obj: dt.date | None = None
        try:
            if market_date_raw:
                market_date_obj = dt.date.fromisoformat(market_date_raw)
        except ValueError:
            market_date_obj = None
        if market_date_obj is not None:
            try:
                payload["signals"]["earnings_catalysts"] = build_earnings_calendar_payload(
                    conn,
                    market_date=market_date_obj,
                    days=14 if report_kind_norm == "daily" else 21,
                    limit=120,
                    tickers=None,
                    setup_map=(payload.get("signals") or {}).get("setup_quality_lookup") or {},
                )
            except Exception as exc:
                LOG.warning("Earnings catalysts build failed: %s", exc)
                _report_warnings.append("earnings_catalysts_failed")
                payload["signals"]["earnings_catalysts"] = {}

        # YOLO deltas and persistence by timeframe.
        delta_daily = fetch_yolo_delta(conn, timeframe="daily")
        delta_weekly = fetch_yolo_delta(conn, timeframe="weekly")
        payload["yolo"]["delta_daily"] = delta_daily
        payload["yolo"]["delta_weekly"] = delta_weekly
        payload["yolo"]["delta"] = delta_weekly if report_kind_norm == "weekly" else delta_daily
        payload["yolo"]["persistence"] = {
            "daily": fetch_yolo_pattern_persistence(conn, timeframe="daily", lookback_asof=20, top_n=25),
            "weekly": fetch_yolo_pattern_persistence(conn, timeframe="weekly", lookback_asof=20, top_n=25),
        }
        try:
            payload["signals"]["tonight_key_changes"] = build_tonight_key_changes(
                payload["signals"], payload["yolo"]["delta"]
            )
        except Exception as exc:
            LOG.warning("Tonight key changes build failed: %s", exc)
            _report_warnings.append("tonight_key_changes_failed")
            payload["signals"]["tonight_key_changes"] = []

        # Refresh LLM status after signal narrative pass so this snapshot reflects current runtime state.
        try:
            payload["meta"]["llm"] = llm_status()
        except Exception as exc:
            LOG.warning("LLM status refresh failed: %s", exc)
            payload["meta"]["llm"] = {}

        # Basic health guardrails.
        price_age = payload["freshness"]["price_age_days"]
        fund_age  = payload["freshness"]["fund_age_hours"]
        yolo_age  = payload["freshness"]["yolo_age_hours"]
        if price_age is None:
            payload["warnings"].append("price_data_missing")
        elif isinstance(price_age, (int, float)) and price_age > 3:
            payload["warnings"].append("price_data_stale")
        if fund_age is None:
            payload["warnings"].append("fundamentals_missing")
        elif isinstance(fund_age, (int, float)) and fund_age > 48:
            payload["warnings"].append("fundamentals_stale")
        if yolo_age is None:
            payload["warnings"].append("yolo_data_missing")
        elif isinstance(yolo_age, (int, float)) and yolo_age > 30:
            payload["warnings"].append("yolo_data_stale")

        llm_meta = payload.get("meta", {}).get("llm", {}) if isinstance(payload.get("meta"), dict) else {}
        llm_health = llm_meta.get("health") if isinstance(llm_meta, dict) and isinstance(llm_meta.get("health"), dict) else {}
        if llm_enabled():
            if isinstance(llm_meta, dict) and not llm_meta.get("ready"):
                payload["warnings"].append("llm_not_ready")
            if isinstance(llm_meta, dict) and llm_meta.get("runtime_disabled"):
                payload["warnings"].append("llm_runtime_cooldown")
            if llm_health.get("degraded"):
                payload["warnings"].append("llm_degraded")

        payload["risk_filters"] = build_no_trade_conditions(payload)
        payload["generation_warnings"] = list(_report_warnings)
        payload["ok"] = len(payload["warnings"]) == 0
        return payload
    finally:
        conn.close()

