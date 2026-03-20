"""Report serialization: Markdown output, JSON snapshots, pruning."""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)


def _md_line(k: str, v: Any) -> str:
    if v is None or v == "":
        return f"- **{k}**: -"
    return f"- **{k}**: {v}"


def to_markdown(report: dict[str, Any]) -> str:
    counts = report.get("counts", {})
    latest = report.get("latest_data", {})
    fresh = report.get("freshness", {})
    run = report.get("latest_ingest_run", {})
    yolo = report.get("yolo", {})
    yolo_summary = yolo.get("summary", {})
    warn = report.get("warnings", [])
    email = report.get("email", {}) if isinstance(report.get("email"), dict) else {}
    llm_meta = ((report.get("meta") or {}).get("llm") or {}) if isinstance(report.get("meta"), dict) else {}
    llm_health = llm_meta.get("health") if isinstance(llm_meta.get("health"), dict) else {}
    llm_alert = report.get("llm_alert", {}) if isinstance(report.get("llm_alert"), dict) else {}

    lines: list[str] = []
    lines.append("# Trader Koo Daily Report")
    lines.append("")
    lines.append(_md_line("Generated (UTC)", report.get("generated_ts")))
    lines.append(_md_line("DB Path", report.get("db_path")))
    lines.append(_md_line("Overall OK", report.get("ok")))
    lines.append("")
    lines.append("## Counts")
    for k in ["tracked_tickers", "price_rows", "fundamentals_rows", "options_rows", "yolo_rows"]:
        lines.append(_md_line(k, counts.get(k)))
    lines.append("")
    lines.append("## Freshness")
    for k in ["price_age_days", "fund_age_hours", "opt_age_hours", "yolo_age_hours"]:
        lines.append(_md_line(k, fresh.get(k)))
    lines.append("")
    lines.append("## Latest Data")
    for k in ["price_date", "fund_snapshot", "options_snapshot", "yolo_detected_ts"]:
        lines.append(_md_line(k, latest.get(k)))
    session = report.get("market_session", {})
    if session:
        lines.append("")
        lines.append("## Market Session Context")
        for k in [
            "market_tz",
            "as_of_market_ts",
            "market_date",
            "is_holiday",
            "holiday_name",
            "is_early_close",
            "early_close_name",
        ]:
            lines.append(_md_line(k, session.get(k)))
        if isinstance(session.get("next_holiday"), dict):
            nh = session["next_holiday"]
            lines.append(_md_line("next_holiday", f"{nh.get('date')} {nh.get('name')}"))
        if isinstance(session.get("next_early_close"), dict):
            ne = session["next_early_close"]
            lines.append(_md_line("next_early_close", f"{ne.get('date')} {ne.get('name')}"))
    lines.append("")
    lines.append("## Latest Ingest Run")
    if run:
        for k in ["run_id", "started_ts", "finished_ts", "status", "tickers_total", "tickers_ok", "tickers_failed", "error_message"]:
            lines.append(_md_line(k, run.get(k)))
    else:
        lines.append("- No ingest run found")
    lines.append("")
    lines.append("## Email Delivery")
    if email.get("attempted"):
        lines.append(_md_line("attempted", "yes"))
        lines.append(_md_line("sent", "yes" if email.get("sent") else "no"))
        lines.append(_md_line("to", email.get("to")))
        if email.get("error"):
            lines.append(_md_line("error", email.get("error")))
    else:
        lines.append("- Not attempted (auto-email disabled for this run).")
    lines.append("")
    lines.append("## LLM Health")
    if llm_meta:
        lines.append(_md_line("enabled", llm_meta.get("enabled")))
        lines.append(_md_line("ready", llm_meta.get("ready")))
        lines.append(_md_line("runtime_disabled", llm_meta.get("runtime_disabled")))
        lines.append(_md_line("runtime_disabled_remaining_sec", llm_meta.get("runtime_disabled_remaining_sec")))
        if llm_health:
            lines.append(_md_line("degraded", llm_health.get("degraded")))
            lines.append(_md_line("consecutive_failures", llm_health.get("consecutive_failures")))
            lines.append(_md_line("last_success_ts", llm_health.get("last_success_ts")))
            lines.append(_md_line("last_failure_ts", llm_health.get("last_failure_ts")))
            lines.append(_md_line("last_failure_reason", llm_health.get("last_failure_reason")))
    else:
        lines.append("- LLM status unavailable")
    lines.append("")
    lines.append("## LLM Alert")
    if llm_alert:
        lines.append(_md_line("attempted", llm_alert.get("attempted")))
        lines.append(_md_line("reason", llm_alert.get("reason")))
        lines.append(_md_line("sent_count", llm_alert.get("sent_count")))
        lines.append(_md_line("failed_count", llm_alert.get("failed_count")))
        if llm_alert.get("error"):
            lines.append(_md_line("error", llm_alert.get("error")))
    else:
        lines.append("- no llm alert metadata")
    lines.append("")
    lines.append("## YOLO Summary")
    lines.append(_md_line("table_exists", yolo.get("table_exists")))
    for k in ["rows_total", "tickers_with_patterns", "latest_detected_ts", "latest_asof_date"]:
        lines.append(_md_line(k, yolo_summary.get(k)))
    tf_rows = yolo.get("timeframes", [])
    if tf_rows:
        lines.append("")
        lines.append("| timeframe | rows_total | tickers_with_patterns | avg_confidence | latest_detected_ts | latest_asof_date |")
        lines.append("|---|---:|---:|---:|---|---|")
        for r in tf_rows:
            avg_conf = r.get("avg_confidence")
            avg_conf = round(float(avg_conf), 4) if isinstance(avg_conf, (int, float)) else "-"
            lines.append(
                f"| {r.get('timeframe', '-')} | {r.get('rows_total', '-')} | {r.get('tickers_with_patterns', '-')} | {avg_conf} | {r.get('latest_detected_ts', '-')} | {r.get('latest_asof_date', '-')} |"
            )
    persistence = yolo.get("persistence", {})
    if isinstance(persistence, dict):
        for tf_key in ("daily", "weekly"):
            block = persistence.get(tf_key, {})
            rows = block.get("rows", []) if isinstance(block, dict) else []
            if rows:
                lines.append("")
                lines.append(f"## YOLO Pattern Persistence ({tf_key.title()}, active on latest as-of)")
                lines.append(
                    _md_line(
                        "window",
                        f"{block.get('lookback_asof', '-')} as-of snapshots ending {block.get('latest_asof', '-')}",
                    )
                )
                lines.append(
                    "| ticker | pattern | streak | seen_in_lookback | coverage_pct | latest_confidence | avg_confidence_window | first_seen_asof | last_seen_asof |"
                )
                lines.append("|---|---|---:|---:|---:|---:|---:|---|---|")
                for p in rows[:15]:
                    lines.append(
                        f"| {p.get('ticker')} | {p.get('pattern')} | {p.get('streak')} | {p.get('seen_in_lookback')} | "
                        f"{p.get('coverage_pct')} | {p.get('latest_confidence')} | {p.get('avg_confidence_window')} | "
                        f"{p.get('first_seen_asof', '-')} | {p.get('last_seen_asof', '-')} |"
                    )

    def _render_delta_section(title: str, delta: dict[str, Any]) -> None:
        lines.append("")
        lines.append(title)
        lines.append(_md_line("comparing", f"{delta.get('prev_asof', '?')} → {delta.get('today_asof', '?')}"))
        lines.append(_md_line("new_patterns", delta.get("new_count", 0)))
        lines.append(_md_line("lost_patterns", delta.get("lost_count", 0)))

        new_pats = delta.get("new_patterns", [])
        if new_pats:
            lines.append("")
            lines.append("### New Patterns (appeared today)")
            lines.append("| ticker | timeframe | pattern | confidence | x0_date | x1_date |")
            lines.append("|---|---|---|---:|---|---|")
            for p in new_pats[:80]:
                lines.append(
                    f"| {p['ticker']} | {p['timeframe']} | {p['pattern']} | {p['confidence']} | {p.get('x0_date', '-')} | {p.get('x1_date', '-')} |"
                )

        lost_pats = delta.get("lost_patterns", [])
        if lost_pats:
            lines.append("")
            lines.append("### Lost Patterns (gone today — invalidated or completed)")
            lines.append("| ticker | timeframe | pattern | confidence | x0_date | x1_date |")
            lines.append("|---|---|---|---:|---|---|")
            for p in lost_pats[:80]:
                lines.append(
                    f"| {p['ticker']} | {p['timeframe']} | {p['pattern']} | {p['confidence']} | {p.get('x0_date', '-')} | {p.get('x1_date', '-')} |"
                )

    # ── YOLO delta (new / lost patterns) ─────────────────────────────────────
    delta_daily = yolo.get("delta_daily", {}) if isinstance(yolo.get("delta_daily"), dict) else {}
    delta_weekly = yolo.get("delta_weekly", {}) if isinstance(yolo.get("delta_weekly"), dict) else {}
    delta = yolo.get("delta", {}) if isinstance(yolo.get("delta"), dict) else {}
    if delta_daily:
        _render_delta_section("## YOLO Pattern Delta (Daily)", delta_daily)
    if delta_weekly:
        _render_delta_section("## YOLO Pattern Delta (Weekly)", delta_weekly)
    if delta and not delta_daily and not delta_weekly:
        _render_delta_section("## YOLO Pattern Delta", delta)

    signals = report.get("signals", {})
    breadth = signals.get("market_breadth", {})
    if breadth:
        lines.append("")
        lines.append("## Market Breadth")
        for k in [
            "total_tickers",
            "advancers",
            "decliners",
            "unchanged",
            "pct_advancing",
            "avg_pct_change",
            "median_pct_change",
            "large_move_threshold_pct",
            "large_move_count",
        ]:
            lines.append(_md_line(k, breadth.get(k)))

    vol_ctx = signals.get("volatility_context", {})
    if vol_ctx:
        lines.append("")
        lines.append("## Volatility Context")
        lines.append(_md_line("vix_close", vol_ctx.get("vix_close")))
        lines.append(_md_line("vix_percentile_1y", vol_ctx.get("vix_percentile_1y")))
        lines.append(_md_line("vix_points", vol_ctx.get("vix_points")))

    regime_ctx = signals.get("regime_context", {})
    if isinstance(regime_ctx, dict) and (
        regime_ctx.get("summary") or regime_ctx.get("vix") or regime_ctx.get("participation")
    ):
        lines.append("")
        lines.append("## Regime Context")
        lines.append(_md_line("context_only", regime_ctx.get("context_only")))
        lines.append(_md_line("asof_date", regime_ctx.get("asof_date")))
        lines.append(_md_line("summary", regime_ctx.get("summary")))
        vix_block = regime_ctx.get("vix", {})
        if isinstance(vix_block, dict) and vix_block:
            lines.append("")
            lines.append("### VIX Structure")
            for key in [
                "close",
                "change_pct_1d",
                "ma20",
                "ma50",
                "ma100",
                "pct_vs_ma20",
                "pct_vs_ma50",
                "pct_vs_ma100",
                "ma_state",
                "ma_cross_state",
                "bb_width_20",
                "bb_width_pctile_lookback",
                "compression_state",
                "breakout_state",
                "risk_state",
                "term_structure_ratio",
                "term_structure_state",
                "vix3m_close",
            ]:
                lines.append(_md_line(key, vix_block.get(key)))
        health_block = regime_ctx.get("health", {})
        if isinstance(health_block, dict) and health_block:
            lines.append("")
            lines.append("### Market Health")
            for key in ["score", "state", "confidence"]:
                lines.append(_md_line(key, health_block.get(key)))
            drivers = [str(v).strip() for v in (health_block.get("drivers") or []) if str(v or "").strip()]
            warnings = [str(v).strip() for v in (health_block.get("warnings") or []) if str(v or "").strip()]
            lines.append(_md_line("drivers", "; ".join(drivers) if drivers else "-"))
            lines.append(_md_line("warnings", "; ".join(warnings) if warnings else "-"))
        tf_rows = regime_ctx.get("timeframes", [])
        if isinstance(tf_rows, list) and tf_rows:
            lines.append("")
            lines.append("### VIX Multi-Timeframe")
            lines.append("| timeframe | lookback_days | change_pct | range_low | range_high | range_position_pct | structure | location |")
            lines.append("|---|---:|---:|---:|---:|---:|---|---|")
            for row in tf_rows:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| {row.get('timeframe', '-')} | {row.get('lookback_days', '-')} | {row.get('change_pct', '-')} | "
                    f"{row.get('range_low', '-')} | {row.get('range_high', '-')} | {row.get('range_position_pct', '-')} | "
                    f"{row.get('structure', '-')} | {row.get('location', '-')} |"
                )
        level_rows = regime_ctx.get("levels", [])
        if isinstance(level_rows, list) and level_rows:
            lines.append("")
            lines.append("### VIX Key Levels")
            lines.append("| type | level | zone_low | zone_high | tier | source | touches | distance_pct | last_touch_date |")
            lines.append("|---|---:|---:|---:|---|---|---:|---:|---|")
            for row in level_rows:
                if not isinstance(row, dict):
                    continue
                # Requirement 10.6: Include source in reports
                source = row.get('source', 'unknown')
                lines.append(
                    f"| {row.get('type', '-')} | {row.get('level', '-')} | {row.get('zone_low', '-')} | {row.get('zone_high', '-')} | "
                    f"{row.get('tier', '-')} | {source} | {row.get('touches', '-')} | {row.get('distance_pct', '-')} | {row.get('last_touch_date', '-')} |"
                )
        participation = regime_ctx.get("participation", [])
        if isinstance(participation, list) and participation:
            lines.append("")
            lines.append("### Participation")
            lines.append("| symbol | window_days | up_days | down_days | up_volume_share_pct | down_volume_share_pct | up_down_volume_ratio | heavy_up_days | heavy_down_days | bias |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            for row in participation:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| {row.get('symbol', '-')} | {row.get('window_days', '-')} | {row.get('up_days', '-')} | {row.get('down_days', '-')} | "
                    f"{row.get('up_volume_share_pct', '-')} | {row.get('down_volume_share_pct', '-')} | {row.get('up_down_volume_ratio', '-')} | "
                    f"{row.get('heavy_up_days', '-')} | {row.get('heavy_down_days', '-')} | {row.get('bias', '-')} |"
                )
        overall = regime_ctx.get("overall", {})
        if isinstance(overall, dict) and overall:
            lines.append("")
            lines.append("### Regime Rollup")
            for key in [
                "participation_bias",
                "accumulation_symbols",
                "distribution_symbols",
                "total_symbols",
                "avg_up_volume_share_pct",
            ]:
                lines.append(_md_line(key, overall.get(key)))

    movers_up = signals.get("movers_up_today", [])[:10]
    if movers_up:
        lines.append("")
        lines.append("## Top Gainers Today")
        lines.append("| ticker | pct_change | close | prev_close | near_52w_high |")
        lines.append("|---|---:|---:|---:|---|")
        for m in movers_up:
            lines.append(
                f"| {m.get('ticker')} | {m.get('pct_change')}% | {m.get('close')} | {m.get('prev_close')} | {m.get('near_52w_high')} |"
            )

    movers_down = signals.get("movers_down_today", [])[:10]
    if movers_down:
        lines.append("")
        lines.append("## Top Losers Today")
        lines.append("| ticker | pct_change | close | prev_close | near_52w_low |")
        lines.append("|---|---:|---:|---:|---|")
        for m in movers_down:
            lines.append(
                f"| {m.get('ticker')} | {m.get('pct_change')}% | {m.get('close')} | {m.get('prev_close')} | {m.get('near_52w_low')} |"
            )

    earnings = signals.get("earnings_catalysts", {})
    if isinstance(earnings, dict) and earnings.get("rows"):
        lines.append("")
        lines.append("## Earnings Catalysts")
        summary = earnings.get("summary", {}) if isinstance(earnings.get("summary"), dict) else {}
        provider_status = earnings.get("provider_status", {}) if isinstance(earnings.get("provider_status"), dict) else {}
        lines.append(_md_line("provider", earnings.get("provider")))
        lines.append(_md_line("window_days", summary.get("window_days")))
        lines.append(_md_line("total_events", summary.get("total_events")))
        lines.append(_md_line("high_risk", summary.get("high_risk")))
        if provider_status.get("detail"):
            lines.append(_md_line("provider_detail", provider_status.get("detail")))
        lines.append("")
        lines.append("| date | session | ticker | score | bias | earnings_risk | action |")
        lines.append("|---|---|---|---:|---|---|---|")
        for row in earnings.get("rows", [])[:12]:
            lines.append(
                f"| {row.get('earnings_date')} | {row.get('earnings_session')} | {row.get('ticker')} | "
                f"{row.get('score')} | {row.get('signal_bias')} | {row.get('earnings_risk')} | {row.get('action')} |"
            )

    key_changes = signals.get("tonight_key_changes", [])[:5]
    if key_changes:
        lines.append("")
        lines.append("## Tonight's 5 Key Changes")
        for idx, change in enumerate(key_changes, start=1):
            lines.append(
                f"{idx}. **{change.get('title', 'Change')}** - {change.get('detail', '-')}"
            )

    setup_rows = signals.get("setup_quality_top", [])[:12]
    if setup_rows:
        lines.append("")
        lines.append("## Confluence Score (Top Candidates)")
        lines.append("| ticker | score | tier | bias | reliability_signal | validity | historical_reliability | observation | reasonable_action | risk_note |")
        lines.append("|---|---:|---|---|---|---|---|---|---|---|")
        for r in setup_rows:
            lines.append(
                f"| {r.get('ticker')} | {r.get('score')} | {r.get('setup_tier') or '-'} | "
                f"{r.get('signal_bias') or '-'} | {r.get('reliability_signal') or '-'} | "
                f"{r.get('validity_label') or '-'} | {r.get('reliability_label') or '-'} | "
                f"{r.get('observation') or '-'} | {r.get('action') or '-'} | {r.get('risk_note') or '-'} |"
            )

    setup_eval = signals.get("setup_evaluation", {})
    if isinstance(setup_eval, dict) and setup_eval.get("enabled"):
        lines.append("")
        lines.append("## Setup Evaluation Backtest")
        lines.append(_md_line("window_days", setup_eval.get("window_days")))
        lines.append(_md_line("min_sample", setup_eval.get("min_sample")))
        lines.append(_md_line("hit_threshold_pct", setup_eval.get("hit_threshold_pct")))
        lines.append(_md_line("tracked_this_run", setup_eval.get("tracked_this_run")))
        lines.append(_md_line("inserted_calls", setup_eval.get("inserted_calls")))
        lines.append(_md_line("scored_this_run", setup_eval.get("scored_this_run")))
        lines.append(_md_line("scored_calls", setup_eval.get("scored_calls")))
        lines.append(_md_line("open_calls", setup_eval.get("open_calls")))
        lines.append(_md_line("latest_scored_asof", setup_eval.get("latest_scored_asof")))
        calibration = setup_eval.get("calibration", {}) if isinstance(setup_eval.get("calibration"), dict) else {}
        if calibration:
            lines.append(_md_line("calibration_adjusted_calls", calibration.get("adjusted_calls")))
            lines.append(_md_line("calibration_avg_adjustment", calibration.get("avg_adjustment")))
            lines.append(_md_line("calibration_max_positive_adjustment", calibration.get("max_positive_adjustment")))
            lines.append(_md_line("calibration_max_negative_adjustment", calibration.get("max_negative_adjustment")))
        overall_eval = setup_eval.get("overall", {}) if isinstance(setup_eval.get("overall"), dict) else {}
        if overall_eval:
            lines.append(_md_line("overall_hit_rate_pct", overall_eval.get("hit_rate_pct")))
            lines.append(_md_line("overall_avg_signed_return_pct", overall_eval.get("avg_signed_return_pct")))
            lines.append(_md_line("overall_median_signed_return_pct", overall_eval.get("median_signed_return_pct")))
            lines.append(_md_line("overall_expectancy_pct", overall_eval.get("expectancy_pct")))
            lines.append(_md_line("overall_profit_factor", overall_eval.get("profit_factor")))
            lines.append(_md_line("overall_avg_validity_days", overall_eval.get("avg_validity_days")))
        by_dir = setup_eval.get("by_direction", [])
        if isinstance(by_dir, list) and by_dir:
            lines.append("")
            lines.append("| direction | calls | hit_rate_pct | avg_signed_return_pct | expectancy_pct | profit_factor |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for row in by_dir:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| {row.get('direction') or row.get('label') or '-'} | {row.get('calls')} | "
                    f"{row.get('hit_rate_pct')} | {row.get('avg_signed_return_pct')} | {row.get('expectancy_pct')} | {row.get('profit_factor')} |"
                )
        by_validity = setup_eval.get("by_validity_days", [])
        if isinstance(by_validity, list) and by_validity:
            lines.append("")
            lines.append("| validity_days | calls | hit_rate_pct | avg_signed_return_pct | expectancy_pct |")
            lines.append("|---:|---:|---:|---:|---:|")
            for row in by_validity:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"| {row.get('validity_days') or '-'} | {row.get('calls')} | "
                    f"{row.get('hit_rate_pct')} | {row.get('avg_signed_return_pct')} | {row.get('expectancy_pct')} |"
                )
        improvement_actions = setup_eval.get("improvement_actions", [])
        if isinstance(improvement_actions, list) and improvement_actions:
            lines.append("")
            lines.append("### Setup Improvement Actions")
            for action in improvement_actions:
                if not isinstance(action, dict):
                    continue
                lines.append(
                    f"- [{action.get('priority', 'info')}] "
                    f"{action.get('scope', 'global')}: "
                    f"{action.get('reason', '-')}"
                )
                recommendation = str(action.get("recommendation") or "").strip()
                if recommendation:
                    lines.append(f"  - recommendation: {recommendation}")

    sector_rows = signals.get("sector_heatmap", [])[:12]
    if sector_rows:
        lines.append("")
        lines.append("## Sector Heatmap")
        lines.append("| sector | avg_pct_change | pct_advancing | tickers | near_high | near_low |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in sector_rows:
            lines.append(
                f"| {r.get('sector')} | {r.get('avg_pct_change')} | {r.get('pct_advancing')} | "
                f"{r.get('tickers')} | {r.get('near_high_count')} | {r.get('near_low_count')} |"
            )

    risk_filters = report.get("risk_filters", {}) if isinstance(report.get("risk_filters"), dict) else {}
    lines.append("")
    lines.append("## No-Trade Conditions")
    if risk_filters:
        lines.append(_md_line("trade_mode", risk_filters.get("trade_mode")))
        lines.append(_md_line("hard_blocks", risk_filters.get("hard_blocks")))
        lines.append(_md_line("soft_flags", risk_filters.get("soft_flags")))
        conditions = risk_filters.get("conditions", [])
        if conditions:
            for cond in conditions:
                lines.append(
                    f"- [{cond.get('severity', 'soft')}] {cond.get('code', 'condition')}: {cond.get('reason', '-')}"
                )
        else:
            lines.append("- none")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("## Warnings")
    if warn:
        for w in warn:
            lines.append(f"- {w}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Cron Log Tail")
    tail = report.get("cron_log_tail", [])
    if tail:
        lines.append("```text")
        lines.extend(tail[-80:])
        lines.append("```")
    else:
        lines.append("- no log lines")
    lines.append("")
    return "\n".join(lines)


def _parse_report_snapshot_ts(path: Path) -> dt.datetime | None:
    stem = path.stem
    prefix = "daily_report_"
    if not stem.startswith(prefix):
        return None
    ts = stem[len(prefix):]
    if ts == "latest":
        return None
    try:
        return dt.datetime.strptime(ts, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def _prune_report_snapshots(out_dir: Path) -> dict[str, int]:
    keep_files_raw = os.getenv("TRADER_KOO_REPORT_KEEP_FILES", "21").strip()
    max_age_days_raw = os.getenv("TRADER_KOO_REPORT_MAX_AGE_DAYS", "45").strip()
    try:
        keep_files = max(3, int(keep_files_raw))
    except ValueError:
        keep_files = 21
    try:
        max_age_days = max(7, int(max_age_days_raw))
    except ValueError:
        max_age_days = 45

    snapshots: list[tuple[dt.datetime, list[Path]]] = []
    grouped: dict[str, list[Path]] = {}
    for path in out_dir.glob("daily_report_*"):
        if path.name in {"daily_report_latest.json", "daily_report_latest.md"}:
            continue
        ts = _parse_report_snapshot_ts(path)
        if ts is None:
            continue
        grouped.setdefault(ts.isoformat(), []).append(path)
    for ts_key, paths in grouped.items():
        try:
            ts = dt.datetime.fromisoformat(ts_key)
        except ValueError:
            continue
        snapshots.append((ts, sorted(paths)))
    snapshots.sort(key=lambda item: item[0], reverse=True)

    now_utc = dt.datetime.now(dt.timezone.utc)
    deleted_files = 0
    deleted_snapshots = 0
    retained_snapshots = 0
    for idx, (snap_ts, paths) in enumerate(snapshots):
        age_days = max(0.0, (now_utc - snap_ts).total_seconds() / 86400.0)
        should_delete = idx >= keep_files or age_days > float(max_age_days)
        if should_delete:
            removed_any = False
            for path in paths:
                try:
                    path.unlink(missing_ok=True)
                    deleted_files += 1
                    removed_any = True
                except OSError:
                    continue
            if removed_any:
                deleted_snapshots += 1
        else:
            retained_snapshots += 1
    return {
        "retained_snapshots": retained_snapshots,
        "deleted_snapshots": deleted_snapshots,
        "deleted_files": deleted_files,
    }


def write_reports(report: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"daily_report_{ts}.json"
    md_path = out_dir / f"daily_report_{ts}.md"
    latest_json = out_dir / "daily_report_latest.json"
    latest_md = out_dir / "daily_report_latest.md"

    json_text = json.dumps(report, indent=2)
    md_text = to_markdown(report)
    json_path.write_text(json_text + "\n", encoding="utf-8")
    md_path.write_text(md_text + "\n", encoding="utf-8")
    latest_json.write_text(json_text + "\n", encoding="utf-8")
    latest_md.write_text(md_text + "\n", encoding="utf-8")
    prune_info = _prune_report_snapshots(out_dir)

    return {
        "json_path": str(json_path),
        "md_path": str(md_path),
        "latest_json": str(latest_json),
        "latest_md": str(latest_md),
        "retained_snapshots": prune_info["retained_snapshots"],
        "pruned_snapshots": prune_info["deleted_snapshots"],
        "pruned_files": prune_info["deleted_files"],
    }
