#!/usr/bin/env python3
"""Thin CLI wrapper for daily report generation.

All logic has been moved to the ``trader_koo.report`` package.
This module re-exports every public symbol so that existing
``from trader_koo.scripts.generate_daily_report import …`` imports
continue to work without changes.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-export everything from the report package so callers that import from
# ``trader_koo.scripts.generate_daily_report`` keep working.
# ---------------------------------------------------------------------------

from trader_koo.report.utils import (  # noqa: F401, E402
    MARKET_CLOSE_HOUR,
    MARKET_TZ,
    MARKET_TZ_NAME,
    TRUTHY_VALUES,
    _as_bool,
    _clamp,
    _fmt_pct_short,
    _median,
    _normalize_report_kind,
    _parse_iso_date,
    _percentile_rank,
    _round_or_none,
    _setup_tier,
    _stdev,
    _to_float,
    days_since_date,
    hours_since,
    market_calendar_context,
    nyse_early_closes_for_year,
    nyse_holidays_for_year,
    parse_iso_utc,
    row_to_dict,
    table_exists,
    tail_text,
)

from trader_koo.report.email_dispatch import (  # noqa: F401, E402
    _email_transport,
    _resend_cfg,
    _send_resend_email,
    _smtp_cfg,
    send_llm_failure_alert_email,
    send_report_email,
)

from trader_koo.report.pattern_analysis import (  # noqa: F401, E402
    _summarize_yolo_lifecycle,
    _yolo_match_tolerance_days,
    _yolo_seen_streak,
    _yolo_snapshot_matches,
    fetch_yolo_delta,
    fetch_yolo_pattern_persistence,
)

from trader_koo.report.market_context import (  # noqa: F401, E402
    REPORT_FEATURE_CFG,
    REPORT_LEVEL_CFG,
    _build_regime_context,
    _build_regime_llm_commentary,
    _fetch_symbol_ohlcv,
    _fetch_technical_context,
    _fetch_volatility_inputs,
)

from trader_koo.report.setup_scoring import (  # noqa: F401, E402
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
    _cap_tier,
    _describe_setup,
    _downgrade_tier,
    _fundamental_context,
    _persist_setup_call_candidates,
    _refresh_setup_eval_surfaces,
    _score_open_setup_call_outcomes,
    _score_setup_from_confluence,
    _setup_call_direction,
    _setup_cluster_rows,
    _setup_eval_bucket,
    _setup_eval_score_adjustment,
    _setup_validity_days,
    _summarize_setup_call_evaluations,
    _tier_rank,
    _yolo_age_factor,
    _yolo_pattern_bias,
    _yolo_recency_label,
    build_no_trade_conditions,
    build_tonight_key_changes,
    ensure_setup_call_eval_schema,
)

from trader_koo.report.generator import (  # noqa: F401, E402
    fetch_report_payload,
    fetch_signals,
)

from trader_koo.report.serializer import (  # noqa: F401, E402
    _md_line,
    _parse_report_snapshot_ts,
    _prune_report_snapshots,
    to_markdown,
    write_reports,
)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate daily run report from trader_koo DB/logs.")
    p.add_argument("--db-path", default=os.getenv("TRADER_KOO_DB_PATH", "/data/trader_koo.db"))
    p.add_argument("--out-dir", default=os.getenv("TRADER_KOO_REPORT_DIR", "/data/reports"))
    p.add_argument("--run-log", default=os.getenv("TRADER_KOO_RUN_LOG_PATH", "/data/logs/cron_daily.log"))
    p.add_argument("--tail-lines", type=int, default=80)
    p.add_argument(
        "--report-kind",
        choices=["daily", "weekly"],
        default=_normalize_report_kind(os.getenv("TRADER_KOO_REPORT_KIND", "daily")),
        help="Report cadence label used for email subject/body and YOLO delta focus.",
    )
    p.add_argument(
        "--send-email", action="store_true",
        default=_as_bool(os.getenv("TRADER_KOO_AUTO_EMAIL", "")),
        help="Send report email after generating (requires TRADER_KOO_SMTP_* env vars)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    db_path = Path(args.db_path).resolve()
    report = fetch_report_payload(
        db_path=db_path,
        run_log=Path(args.run_log).resolve(),
        tail_lines=max(0, int(args.tail_lines)),
        report_kind=args.report_kind,
    )

    email_meta: dict[str, Any] = {
        "attempted": bool(args.send_email),
        "sent": False,
        "to": None,
    }
    if args.send_email:
        try:
            transport = _email_transport()
            smtp_cfg = _smtp_cfg()
            resend_cfg = _resend_cfg()
            email_meta["to"] = resend_cfg.get("to_email") if transport == "resend" else smtp_cfg.get("to_email")
            if transport == "resend":
                print(
                    "[EMAIL] attempt "
                    "transport=resend "
                    f"to={resend_cfg.get('to_email') or '-'} "
                    f"has_api_key={1 if resend_cfg.get('api_key') else 0}"
                )
            else:
                print(
                    "[EMAIL] attempt "
                    "transport=smtp "
                    f"host={smtp_cfg.get('host') or '-'} "
                    f"port={smtp_cfg.get('port')} "
                    f"security={smtp_cfg.get('security')} "
                    f"to={smtp_cfg.get('to_email') or '-'}"
                )
            md_text = to_markdown(report)
            email_summary = send_report_email(
                report,
                md_text,
                db_path=db_path,
            )
            email_meta["sent"] = bool(email_summary.get("sent_count"))
            email_meta["sent_count"] = int(email_summary.get("sent_count") or 0)
            email_meta["failed_count"] = int(email_summary.get("failed_count") or 0)
            email_meta["skipped_duplicate_count"] = int(email_summary.get("skipped_duplicate_count") or 0)
            email_meta["sample_recipients"] = email_summary.get("sample_recipients") or []
            print(
                "[EMAIL] sent "
                f"transport={transport} sent={email_meta['sent_count']} "
                f"failed={email_meta['failed_count']} "
                f"skipped_duplicate={email_meta['skipped_duplicate_count']}"
            )
        except Exception as exc:
            email_meta["error"] = str(exc)
            LOG.error("Email dispatch failed: %s", exc)
    llm_alert_meta: dict[str, Any] = {"attempted": False, "reason": "not_checked"}
    try:
        llm_alert_meta = send_llm_failure_alert_email(
            report,
            db_path=db_path,
        )
        if llm_alert_meta.get("attempted"):
            print(
                "[LLM-ALERT] sent "
                f"transport={llm_alert_meta.get('transport') or '-'} "
                f"sent={int(llm_alert_meta.get('sent_count') or 0)} "
                f"failed={int(llm_alert_meta.get('failed_count') or 0)}"
            )
    except Exception as exc:
        llm_alert_meta = {
            "attempted": True,
            "reason": "dispatch_error",
            "sent_count": 0,
            "failed_count": 0,
            "error": str(exc),
        }
        LOG.error("LLM failure alert email failed: %s", exc)

    report["llm_alert"] = llm_alert_meta
    report["email"] = email_meta
    if email_meta["attempted"] and not email_meta["sent"]:
        warnings = report.get("warnings")
        if not isinstance(warnings, list):
            warnings = []
            report["warnings"] = warnings
        if "report_email_failed" not in warnings:
            warnings.append("report_email_failed")
    if llm_alert_meta.get("attempted") and int(llm_alert_meta.get("sent_count") or 0) == 0 and llm_alert_meta.get("error"):
        warnings = report.get("warnings")
        if not isinstance(warnings, list):
            warnings = []
            report["warnings"] = warnings
        if "llm_alert_send_failed" not in warnings:
            warnings.append("llm_alert_send_failed")
    if str(llm_alert_meta.get("reason") or "") in {
        "missing_alert_recipients",
        "resend_not_configured",
        "smtp_not_configured",
        "smtp_password_missing",
    }:
        warnings = report.get("warnings")
        if not isinstance(warnings, list):
            warnings = []
            report["warnings"] = warnings
        if "llm_alert_not_configured" not in warnings:
            warnings.append("llm_alert_not_configured")
    report["ok"] = len(report.get("warnings", [])) == 0

    out_paths = write_reports(report, Path(args.out_dir).resolve())

    print(
        json.dumps(
            {
                "ok": report.get("ok", False),
                "warnings": report.get("warnings", []),
                "generated_ts": report.get("generated_ts"),
                **out_paths,
                "email_attempted": email_meta.get("attempted", False),
                "email_sent": email_meta.get("sent", False),
                "email_error": email_meta.get("error"),
                "llm_alert_attempted": llm_alert_meta.get("attempted", False),
                "llm_alert_reason": llm_alert_meta.get("reason"),
                "llm_alert_sent_count": int(llm_alert_meta.get("sent_count") or 0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
