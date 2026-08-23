#!/usr/bin/env python3
"""Thin CLI wrapper for daily report generation.

All logic lives in the ``trader_koo.report`` package; this module only
wires up the command-line entry point.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sqlite3
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from trader_koo.report.email_dispatch import (
    _email_transport,
    _resend_cfg,
    _smtp_cfg,
    send_llm_failure_alert_email,
    send_report_email,
)
from trader_koo.report.generator import fetch_report_payload
from trader_koo.report.runs import (
    admit_published_report,
    complete_report_run,
    fail_report_run,
    publish_report_run,
    sha256_file,
    start_report_run,
)
from trader_koo.report.serializer import to_markdown, write_reports
from trader_koo.report.utils import _as_bool, _normalize_report_kind

LOG = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def effective_report_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Capture the effective non-secret policy, including module defaults."""
    from trader_koo import paper_trades
    from trader_koo.paper_trade import critic
    from trader_koo.paper_trade.config import config_snapshot
    from trader_koo.report import calibration_pulse, setup_scoring

    def constants(module: Any) -> dict[str, Any]:
        return {
            name: _json_safe(value)
            for name, value in vars(module).items()
            if name.lstrip("_").isupper()
            and name not in {"LOG"}
            and not callable(value)
        }

    secret_fragments = ("SECRET", "TOKEN", "PASSWORD", "API_KEY", "PRIVATE", "CREDENTIAL")
    env_overrides = {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith("TRADER_KOO_")
        and not any(fragment in name.upper() for fragment in secret_fragments)
    }
    return {
        "cli": _json_safe(vars(args)),
        "environment_overrides": env_overrides,
        "setup_scoring": constants(setup_scoring),
        "calibration": constants(calibration_pulse),
        "paper_trade": _json_safe(config_snapshot(paper_trades._build_config())),
        "paper_trade_module": constants(paper_trades),
        "critic": constants(critic),
    }

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
    out_dir = Path(args.out_dir).resolve()
    run_log = Path(args.run_log).resolve()
    lifecycle_conn = sqlite3.connect(str(db_path))
    run_id = start_report_run(
        lifecycle_conn,
        report_kind=args.report_kind,
        configuration=effective_report_configuration(args),
    )
    try:
        report = fetch_report_payload(
            db_path=db_path,
            run_log=run_log,
            tail_lines=max(0, int(args.tail_lines)),
            report_kind=args.report_kind,
            report_run_id=run_id,
        )

        # Delivery is intentionally represented as pending in the immutable
        # artifact. External messages are attempted only after hash-verified
        # publication and transactional downstream admission succeed.
        email_meta: dict[str, Any] = {
            "attempted": bool(args.send_email),
            "sent": False,
            "to": None,
            "state": "pending_after_publication" if args.send_email else "disabled",
        }
        llm_alert_meta: dict[str, Any] = {
            "attempted": False,
            "reason": "pending_after_publication",
        }
        report["llm_alert"] = llm_alert_meta
        report["email"] = email_meta
        warnings = report.get("warnings")
        if not isinstance(warnings, list):
            warnings = []
            report["warnings"] = warnings
        report["ok"] = report.get("ok") is True and len(warnings) == 0

        completed_ts = (
            dt.datetime.now(dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        report_run_meta = (report.get("meta") or {}).get("report_run")
        if isinstance(report_run_meta, dict):
            report_run_meta.update({"state": "completed", "completed_ts": completed_ts})
        out_paths = write_reports(
            report,
            out_dir,
            run_id=run_id,
            publish_latest=False,
        )
        artifact_path = Path(str(out_paths["json_path"]))
        complete_report_run(
            lifecycle_conn,
            run_id=run_id,
            report=report,
            artifact_path=artifact_path,
            markdown_path=Path(str(out_paths["md_path"])),
            content_hash=sha256_file(artifact_path),
            completed_ts=completed_ts,
        )
        publication = publish_report_run(lifecycle_conn, run_id=run_id, report_dir=out_dir)
        admission = admit_published_report(
            lifecycle_conn,
            run_id=run_id,
            report_dir=out_dir,
        )

        if args.send_email:
            try:
                transport = _email_transport()
                smtp_cfg = _smtp_cfg()
                resend_cfg = _resend_cfg()
                email_meta["to"] = (
                    resend_cfg.get("to_email")
                    if transport == "resend"
                    else smtp_cfg.get("to_email")
                )
                email_summary = send_report_email(
                    report,
                    Path(str(out_paths["md_path"])).read_text(encoding="utf-8"),
                    db_path=db_path,
                )
                email_meta.update(
                    {
                        "state": "sent" if email_summary.get("sent_count") else "failed",
                        "sent": bool(email_summary.get("sent_count")),
                        "sent_count": int(email_summary.get("sent_count") or 0),
                        "failed_count": int(email_summary.get("failed_count") or 0),
                        "skipped_duplicate_count": int(
                            email_summary.get("skipped_duplicate_count") or 0
                        ),
                        "sample_recipients": email_summary.get("sample_recipients") or [],
                    }
                )
            except Exception as exc:
                email_meta.update({"state": "failed", "error": str(exc)})
                LOG.error("Email dispatch failed after publication: %s", exc)

        try:
            llm_alert_meta = send_llm_failure_alert_email(report, db_path=db_path)
        except Exception as exc:
            llm_alert_meta = {
                "attempted": True,
                "reason": "dispatch_error",
                "sent_count": 0,
                "failed_count": 0,
                "error": str(exc),
            }
            LOG.error("LLM failure alert email failed after publication: %s", exc)

        print(
            json.dumps(
                {
                    "ok": report.get("ok", False),
                    "warnings": warnings,
                    "generated_ts": report.get("generated_ts"),
                    "report_run": publication,
                    "admission": admission,
                    **out_paths,
                    "latest_json": str(out_dir / "daily_report_latest.json"),
                    "latest_md": str(out_dir / "daily_report_latest.md"),
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
    except Exception as exc:
        try:
            fail_report_run(lifecycle_conn, run_id=run_id, error=str(exc))
        except ValueError:
            # Completed/published runs are immutable; never mask the original
            # post-publication admission or delivery failure.
            pass
        raise
    finally:
        lifecycle_conn.close()


if __name__ == "__main__":
    main()
