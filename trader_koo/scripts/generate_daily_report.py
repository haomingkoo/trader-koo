#!/usr/bin/env python3
"""Thin CLI wrapper for daily report generation.

All logic lives in the ``trader_koo.report`` package; this module only
wires up the command-line entry point.
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
# Imports used by the CLI entry point below.
# ---------------------------------------------------------------------------

from trader_koo.report.utils import _as_bool, _normalize_report_kind
from trader_koo.report.email_dispatch import (
    _email_transport,
    _resend_cfg,
    _smtp_cfg,
    send_llm_failure_alert_email,
    send_report_email,
)
from trader_koo.report.generator import fetch_report_payload
from trader_koo.report.serializer import to_markdown, write_reports


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
