"""Main report orchestrator: fetch_signals and fetch_report_payload."""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from trader_koo.catalyst_data import build_earnings_calendar_payload
from trader_koo.db.price_contract import research_eligible_tickers, research_price_contract
from trader_koo.paper_trade.schema import ensure_paper_trade_schema
from trader_koo.paper_trades import (
    PAPER_TRADE_ENABLED,
    mark_to_market,
)
from trader_koo.report.freshness import assess_data_freshness
from trader_koo.report.market_context import (
    _build_regime_context,
    _fetch_technical_context,
    _fetch_volatility_inputs,
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
    SETUP_EVAL_MIN_SAMPLE,
    SETUP_NEWS_ENABLED,
    SETUP_EVAL_TRACK_LIMIT,
    SETUP_EVAL_WINDOW_DAYS,
    _score_open_setup_call_outcomes,
    _setup_cluster_rows,
    _summarize_setup_call_evaluations,
    _yolo_age_factor,
    _yolo_recency_label,
    annotate_earnings_proximity,
    build_no_trade_conditions,
    build_tonight_key_changes,
    ensure_setup_call_eval_schema,
)
from trader_koo.report.setup_ranking import calibrate_setups, rank_setups, rerank_with_candles, score_setup
from trader_koo.report.utils import (
    _normalize_report_kind,
    days_since_date,
    hours_since,
    market_calendar_context,
    nyse_holidays_for_year,
    row_to_dict,
    table_exists,
    tail_text,
)

LOG = logging.getLogger(__name__)

# Movers and breadth only use tickers priced on the latest session. If fewer
# than this share of recently active tickers are covered, flag it in the report.
MIN_LATEST_SESSION_COVERAGE = 0.8
RECENT_ACTIVITY_DAYS = 30


def _warn_if_latest_session_thin(conn: sqlite3.Connection, covered: int, report_warnings: list[str]) -> None:
    """Warn when the latest session prices far fewer tickers than were recently active."""
    universe = conn.execute(
        "SELECT COUNT(DISTINCT ticker) FROM price_daily "
        "WHERE date >= date((SELECT MAX(date) FROM price_daily), ?)",
        (f"-{RECENT_ACTIVITY_DAYS} days",),
    ).fetchone()[0]
    if universe and covered < universe * MIN_LATEST_SESSION_COVERAGE:
        LOG.warning("Latest session covers %d of %d recently active tickers", covered, universe)
        report_warnings.append("market_breadth_latest_session_thin")


def _green_barrier_coverage_requires_warning(coverage: dict[str, Any]) -> bool:
    """Return whether missing coverage reflects stale or invalid market data."""
    return (
        int(coverage.get("stale_skipped_count") or 0) > 0
        or int(coverage.get("invalid_date_skipped_count") or 0) > 0
    )


def _trading_days_between(start_date: dt.date, end_date: dt.date) -> int:
    """Count NYSE trading days in the open interval (start_date, end_date]."""
    if end_date <= start_date:
        return 0

    holidays: set[dt.date] = set()
    for year in range(start_date.year, end_date.year + 1):
        holidays.update(nyse_holidays_for_year(year).keys())

    days = 0
    cursor = start_date + dt.timedelta(days=1)
    while cursor <= end_date:
        if cursor.weekday() < 5 and cursor not in holidays:
            days += 1
        cursor += dt.timedelta(days=1)
    return days


def _report_stale_price_guard_error(
    *,
    latest_run: dict[str, Any] | None,
    latest_price_date: str | None,
    market_date: str | None,
) -> str | None:
    """Return a blocking stale-price message when report generation should abort.

    Blocks in two cases:
    1. Ingest explicitly failed AND prices are stale (original behavior)
    2. Prices are >1 trading day stale regardless of ingest status (new guard)
       This catches partial ingest failures and weekend/holiday edge cases.
    """
    if not latest_price_date:
        if isinstance(latest_run, dict) and str(latest_run.get("status") or "").strip().lower() == "failed":
            return (
                "Latest ingest run failed and price_daily has no latest price date. "
                "Aborting report generation to avoid publishing a report with missing price data."
            )
        return None

    try:
        price_date = dt.date.fromisoformat(str(latest_price_date).strip()[:10])
    except ValueError:
        return (
            f"Latest price date is invalid ({latest_price_date!r}). "
            "Aborting report generation."
        )

    try:
        session_date = dt.date.fromisoformat(str(market_date or "").strip()[:10])
    except ValueError:
        session_date = dt.datetime.now(dt.timezone.utc).date()

    trading_days_behind = _trading_days_between(price_date, session_date)
    if trading_days_behind <= 0:
        return None

    run_status = ""
    started_ts = "unknown"
    finished_ts = "unknown"
    if isinstance(latest_run, dict):
        run_status = str(latest_run.get("status") or "").strip().lower()
        started_ts = str(latest_run.get("started_ts") or "").strip() or "unknown"
        finished_ts = str(latest_run.get("finished_ts") or "").strip() or "unknown"

    # Block if prices are >1 trading day stale (regardless of ingest status)
    if trading_days_behind > 1:
        return (
            f"Price data is {trading_days_behind} trading days stale "
            f"(price_date={price_date.isoformat()}, market_date={session_date.isoformat()}, "
            f"ingest_status={run_status or 'unknown'}, started_ts={started_ts}, "
            f"finished_ts={finished_ts}). Aborting report generation."
        )

    # Block if exactly 1 day stale AND ingest failed
    if run_status == "failed":
        return (
            "Latest ingest run failed and price data is stale "
            f"(price_date={price_date.isoformat()}, market_date={session_date.isoformat()}, "
            f"trading_days_behind={trading_days_behind}, started_ts={started_ts}, "
            f"finished_ts={finished_ts}). Aborting report generation."
        )

    return None


def _fetch_signals_snapshot(conn: sqlite3.Connection, report_warnings: list[str]) -> dict[str, Any]:
    """Market signals for the daily report: 52W extremes, top YOLO patterns, candle signals.

    Degraded sections append their warning codes to ``report_warnings``.
    """
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
        "setup_quality_all": [],
        "setup_quality_lookup": {},
        "report_decisions": [],
        "scanned_universe": [],
        "watchlist_candidates": [],
        "setup_evaluation": {},
        "earnings_catalysts": {},
        "tonight_key_changes": [],
        "green_barrier_hits": [],
        "green_barrier_coverage": {},
    }
    try:
        universe_row = conn.execute(
            """
            SELECT run_id
            FROM ingest_runs
            WHERE status = 'ok'
              AND tickers_failed = 0
              AND json_extract(args_json, '$.use_sp500') = 1
            ORDER BY started_ts DESC
            LIMIT 1
            """
        ).fetchone()
        current_universe = (
            [
                str(row[0])
                for row in conn.execute(
                    """
                    SELECT ticker
                    FROM ingest_ticker_status
                    WHERE run_id = ? AND status = 'ok'
                    ORDER BY ticker
                    """,
                    (universe_row[0],),
                )
            ]
            if universe_row is not None
            else []
        )
    except sqlite3.OperationalError:
        current_universe = []
    current_universe_set = set(current_universe)
    eligible_universe = current_universe_set & research_eligible_tickers(conn)
    excluded_universe = sorted(current_universe_set - eligible_universe)
    price_contract = (
        research_price_contract(conn, sorted(eligible_universe))
        if eligible_universe
        else {
            "eligible": False,
            "basis": "unknown",
            "version": "unknown",
            "status": "unresolved",
            "reason": "no_research_eligible_tickers",
        }
    )
    price_contract.update({
        "requested_universe_count": len(current_universe_set),
        "eligible_universe_count": len(eligible_universe),
        "excluded_tickers": excluded_universe,
    })
    signals["price_contract"] = price_contract
    if not price_contract["eligible"]:
        report_warnings.append("price_basis_unresolved")
        LOG.error(
            "Report signals excluded: price basis is not research eligible (%s)",
            price_contract["reason"],
        )
        return signals
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
        report_warnings.append("volatility_inputs_failed")
    regime_warnings: list[str] = []
    try:
        signals["regime_context"] = _build_regime_context(conn, regime_warnings)
    except Exception as exc:
        LOG.error("Regime context build failed: %s", exc, exc_info=True)
        report_warnings.append("regime_context_failed")
        signals["regime_context"] = {}
    # Cache VIX metrics in nightly report (avoids live recompute on every page load)
    try:
        from trader_koo.structure.vix_metrics import compute_vix_metrics
        signals["vix_metrics"] = compute_vix_metrics(conn)
    except Exception as exc:
        LOG.debug("VIX metrics cache in report skipped: %s", exc)
        signals["vix_metrics"] = None
    try:
        from trader_koo.analysis.green_barrier import scan_green_barrier_snapshot

        green_barrier = scan_green_barrier_snapshot(conn)
        signals["green_barrier_hits"] = green_barrier["hits"]
        signals["green_barrier_coverage"] = green_barrier["coverage"]
        coverage = green_barrier["coverage"]
        if _green_barrier_coverage_requires_warning(coverage):
            report_warnings.append("green_barrier_incomplete_coverage")
    except Exception as exc:
        LOG.error("Green Barrier scan failed: %s", exc, exc_info=True)
        report_warnings.append("green_barrier_scan_failed")
    # Regime warnings go after the Green Barrier ones, matching earlier reports.
    report_warnings.extend(regime_warnings)
    try:
        technical_by_ticker = _fetch_technical_context(conn)
    except Exception as exc:
        LOG.error("Technical context fetch failed: %s", exc, exc_info=True)
        report_warnings.append("technical_context_failed")
        technical_by_ticker = {}

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
              AND lc.date = (SELECT MAX(date) FROM price_daily)
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
        report_warnings.append("52w_proximity_failed")

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
              AND l.date = (SELECT MAX(date) FROM price_daily)
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
        median_pct: float | None = statistics.median(pct_changes) if pct_changes else None
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
        _warn_if_latest_session_thin(conn, len(movers), report_warnings)
    except Exception as exc:
        LOG.warning("Large moves / market breadth computation failed: %s", exc)
        report_warnings.append("market_breadth_failed")

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
        report_warnings.append("yolo_patterns_failed")

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
        report_warnings.append("fundamentals_snapshot_failed")

    # ── Sector heatmap + setup quality scoring ───────────────────────────────
    try:
        sector_buckets: dict[str, dict[str, Any]] = {}
        setup_rows: list[dict[str, Any]] = []
        for m in movers_all:
            ticker = str(m.get("ticker") or "").upper()
            if not ticker or ticker not in eligible_universe:
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

            setup_rows.append(
                score_setup(
                    conn,
                    ticker=ticker,
                    pct_change=pct_change,
                    near_high=near_high,
                    near_low=near_low,
                    fund=fund,
                    sector=sector,
                    yolo=yolo_by_ticker.get(ticker),
                    vol=vol_by_ticker.get(ticker, {}),
                    tech=technical_by_ticker.get(ticker, {}),
                    vol_ctx=vol_ctx,
                )
            )

        sector_rows: list[dict[str, Any]] = []
        for _, bucket in sector_buckets.items():
            changes = [float(x) for x in bucket.pop("_changes", [])]
            if not changes:
                continue
            median_change = statistics.median(changes)
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

        rank_setups(conn, setup_rows, signals)
    except Exception as e:
        LOG.error("Setup quality scoring failed: %s", e, exc_info=True)
        report_warnings.append("setup_quality_scoring_failed")

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
                rerank_with_candles(
                    conn,
                    setup_rows,
                    candle_by_ticker,
                    signals,
                    as_of_date=str(latest_date) if latest_date else None,
                )
    except Exception as exc:
        LOG.error("Candle patterns / setup scoring section failed: %s", exc, exc_info=True)
        report_warnings.append("candle_patterns_section_failed")

    # Per-ticker HMMs remain available on demand in the chart workspace. They
    # are intentionally absent here because they do not affect report ranking
    # and cold model fits delayed publication by tens of minutes.
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
        report_warnings.append("economic_calendar_prefetch_failed")
        signals["economic_events"] = []

    return signals


def fetch_signals(conn: sqlite3.Connection, report_warnings: list[str]) -> dict[str, Any]:
    """Build analytical signals from one caller-preserving SQLite snapshot."""
    owns_snapshot = not conn.in_transaction
    if owns_snapshot:
        conn.execute("BEGIN")
    try:
        return _fetch_signals_snapshot(conn, report_warnings)
    finally:
        if owns_snapshot and conn.in_transaction:
            conn.rollback()


def _price_contract_excluded(signals: Any) -> set[str]:
    contract = signals.get("price_contract") if isinstance(signals, dict) else None
    excluded = contract.get("excluded_tickers") if isinstance(contract, dict) else None
    return {str(t) for t in excluded or []}


def fetch_report_payload(
    db_path: Path,
    run_log: Path,
    tail_lines: int,
    report_kind: str = "daily",
    report_run_id: str | None = None,
) -> dict[str, Any]:
    report_warnings: list[str] = []
    now = dt.datetime.now(dt.timezone.utc)
    report_kind_norm = _normalize_report_kind(report_kind)
    payload: dict[str, Any] = {
        "generated_ts": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "meta": {
            "report_kind": report_kind_norm,
            "report_run": {
                "run_id": report_run_id,
                "state": "started" if report_run_id else "unlinked_legacy",
                "lineage": "linked" if report_run_id else "unlinked legacy",
            },
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
        # Startup-style schema ownership must happen before report evaluation
        # opens any business transaction. Later paper calls are then no-ops.
        ensure_paper_trade_schema(conn)
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
                (SELECT MAX(detected_ts) FROM yolo_patterns) AS latest_yolo_ts,
                (SELECT MAX(as_of_date) FROM yolo_patterns WHERE timeframe = 'daily') AS latest_yolo_daily_as_of
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
            "yolo_as_of_date": counts["latest_yolo_daily_as_of"] if counts else None,
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

        stale_price_error = _report_stale_price_guard_error(
            latest_run=payload["latest_ingest_run"],
            latest_price_date=payload["latest_data"].get("price_date"),
            market_date=(payload.get("market_session") or {}).get("market_date"),
        )
        if stale_price_error:
            raise RuntimeError(stale_price_error)

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
        payload["signals"] = fetch_signals(conn, report_warnings)
        price_basis = payload["signals"].get("price_contract")
        if isinstance(price_basis, dict):
            payload["meta"]["price_basis"] = price_basis
        for warning in (
            "price_basis_unresolved",
            "green_barrier_incomplete_coverage",
            "green_barrier_scan_failed",
            "market_breadth_latest_session_thin",
        ):
            if warning in report_warnings:
                payload["warnings"].append(warning)
        if SETUP_EVAL_ENABLED:
            eval_summary: dict[str, Any] = {"enabled": True, "scored_calls": 0, "open_calls": 0}
            try:
                ensure_setup_call_eval_schema(conn)
                signals_ref = payload.get("signals") if isinstance(payload.get("signals"), dict) else {}
                setup_rows_ref = signals_ref.get("setup_quality_top") if isinstance(signals_ref.get("setup_quality_top"), list) else []
                scored_this_run = _score_open_setup_call_outcomes(conn)
                conn.commit()
                summary, reliability_lookup = _summarize_setup_call_evaluations(
                    conn,
                    window_days=SETUP_EVAL_WINDOW_DAYS,
                    min_sample=SETUP_EVAL_MIN_SAMPLE,
                )
                try:
                    from trader_koo.report.calibration_pulse import load_calibration_state
                    _calib_state = load_calibration_state(conn)
                except Exception as exc:
                    LOG.error("calibration_state unavailable; setup scores use inline reliability: %s", exc)
                    payload.setdefault("warnings", []).append(f"calibration_state unavailable: {exc}")
                    _calib_state = {}
                # The hit-rate bar is the baseline hit rate; SETUP_EVAL_HIT_THRESHOLD_PCT is a return threshold.
                calibration = calibrate_setups(
                    signals_ref,
                    reliability_lookup=reliability_lookup,
                    min_sample=SETUP_EVAL_MIN_SAMPLE,
                    baseline_hit_rate_pct=(summary.get("overall") or {}).get("hit_rate_pct") if isinstance(summary, dict) else None,
                    calibration_state=_calib_state,
                )
                eval_summary = summary or {"enabled": True}
                eval_summary["tracked_this_run"] = int(min(len(setup_rows_ref), SETUP_EVAL_TRACK_LIMIT))
                eval_summary["inserted_calls"] = 0
                if report_run_id:
                    eval_summary["admission_state"] = "pending_verified_publication"
                else:
                    eval_summary["admission_state"] = "requires_report_run_publication"
                eval_summary["scored_this_run"] = int(scored_this_run)
                eval_summary["calibration"] = calibration
            except Exception as exc:
                LOG.error("Setup evaluation failed: %s", exc, exc_info=True)
                report_warnings.append("setup_evaluation_failed")
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
                paper_trade_result["inserted"] = 0
                paper_trade_result["admission_state"] = (
                    "pending_verified_publication"
                    if report_run_id
                    else "requires_report_run_publication"
                )
                mtm_result = mark_to_market(conn)
                conn.commit()
                paper_trade_result["mtm"] = mtm_result
            except Exception as exc:
                conn.rollback()
                paper_trade_result["error"] = "paper_trade_failed"
                LOG.exception("Paper trade integration failed")
                report_warnings.append("paper_trade_failed")
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
                report_warnings.append("earnings_catalysts_failed")
                payload["signals"]["earnings_catalysts"] = {}

        # ── Annotate setups with earnings proximity ─────────────────────
        try:
            setup_rows_for_earnings = (
                (payload.get("signals") or {}).get("setup_quality_top")
                if isinstance(
                    (payload.get("signals") or {}).get("setup_quality_top"), list
                )
                else []
            )
            earnings_cat = (payload.get("signals") or {}).get(
                "earnings_catalysts"
            ) or {}
            flagged_count = annotate_earnings_proximity(
                setup_rows_for_earnings, earnings_cat
            )
            if flagged_count:
                LOG.info(
                    "Earnings proximity: %d setups flagged within 5 trading days",
                    flagged_count,
                )
        except Exception as exc:
            LOG.warning("Earnings proximity annotation failed: %s", exc)
            report_warnings.append("earnings_proximity_failed")

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
            report_warnings.append("tonight_key_changes_failed")
            payload["signals"]["tonight_key_changes"] = []

        # Basic health guardrails.
        freshness = assess_data_freshness(
            conn, now, yolo_excluded_tickers=_price_contract_excluded(payload["signals"])
        )
        if freshness.price_missing:
            payload["warnings"].append("price_data_missing")
        elif freshness.price_stale:
            payload["warnings"].append("price_data_stale")
        if freshness.fund_missing:
            payload["warnings"].append("fundamentals_missing")
        elif freshness.fund_stale:
            payload["warnings"].append("fundamentals_stale")
        if freshness.yolo_missing:
            payload["warnings"].append("yolo_data_missing")
        elif freshness.yolo_stale:
            payload["warnings"].append("yolo_data_stale")

        payload["risk_filters"] = build_no_trade_conditions(payload)
        signals = payload.get("signals") if isinstance(payload.get("signals"), dict) else {}
        all_rows = signals.get("setup_quality_all") if isinstance(signals, dict) else []
        all_rows = all_rows if isinstance(all_rows, list) else []
        selected_rows = signals.get("setup_quality_top") if isinstance(signals, dict) else []
        selected_rows = selected_rows if isinstance(selected_rows, list) else []
        selected_tickers = {
            str(row.get("ticker") or "").upper().strip()
            for row in selected_rows
            if isinstance(row, dict) and str(row.get("ticker") or "").strip()
        }
        ordered_rows = [dict(row) for row in selected_rows if isinstance(row, dict)]
        ordered_rows.extend(
            dict(row)
            for row in all_rows
            if isinstance(row, dict)
            and str(row.get("ticker") or "").upper().strip() not in selected_tickers
        )
        decisions: list[dict[str, Any]] = []
        for rank, row in enumerate(ordered_rows, start=1):
            ticker = str(row.get("ticker") or "").upper().strip()
            if not ticker:
                continue
            accepted = ticker in selected_tickers
            decisions.append(
                {
                    "ticker": ticker,
                    "selected_rank": rank,
                    "decision": "accepted" if accepted else "rejected",
                    "reason_codes": (
                        list(row.get("selection_reason_codes") or [])
                        or (["selected_report_cohort"] if accepted else ["outside_report_selection_limit"])
                    ),
                    "inputs": row,
                }
            )
        if isinstance(signals, dict):
            signals["report_decisions"] = decisions
            signals["scanned_universe"] = [row["ticker"] for row in decisions]
        payload["generation_warnings"] = list(report_warnings)
        for warning in report_warnings:
            if warning not in payload["warnings"]:
                payload["warnings"].append(warning)
        payload["ok"] = len(payload["warnings"]) == 0
        return payload
    finally:
        conn.close()
