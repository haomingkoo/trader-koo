"""Email dispatch and SMTP/Resend health."""
from __future__ import annotations

import datetime as dt
import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from trader_koo.backend.services.database import DB_PATH
from trader_koo.backend.services.report_loader import latest_daily_report_json
from trader_koo.email_delivery import (
    Attachment,
    OutgoingEmail,
    load_email_config,
    transport_for,
)
from trader_koo.email_subscribers import email_max_recipients
from trader_koo.report_email import (
    build_report_email_bodies,
    build_report_email_subject,
    report_email_app_url,
)

from trader_koo.backend.routers.admin._shared import LOG, REPORT_DIR

router = APIRouter(tags=["admin", "admin-email"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/api/admin/smtp-health")
def smtp_health() -> dict[str, Any]:
    """Return email delivery config health (without secrets)."""
    config = load_email_config()
    smtp = config.smtp
    resend = config.resend
    auto_email = str(
        os.getenv("TRADER_KOO_AUTO_EMAIL", "")
    ).strip().lower() in {"1", "true", "yes"}
    missing = config.missing()
    if auto_email and not config.recipients():
        missing.append("TRADER_KOO_REPORT_EMAIL_TO")
    return {
        "ok": len(missing) == 0,
        "auto_email_enabled": auto_email,
        "transport": config.transport,
        "email_max_recipients": email_max_recipients(),
        "missing": missing,
        "smtp": {
            "host": smtp.host,
            "port": smtp.port,
            "security": smtp.security,
            "timeout_sec": smtp.timeout_sec,
            "from_email": smtp.from_email,
            "default_to": config.default_to,
            "has_user": bool(smtp.user),
            "has_password": bool(smtp.password),
        },
        "resend": {
            "has_api_key": bool(resend.api_key),
            "from_email": resend.from_email,
            "default_to": config.default_to,
            "timeout_sec": resend.timeout_sec,
        },
    }


@router.post("/api/admin/email-latest-report")
def email_latest_report(
    to: str | None = Query(default=None),
    include_markdown: bool = Query(default=True),
    attach_json: bool = Query(default=True),
) -> dict[str, Any]:
    """Send the latest daily report by email via configured transport."""
    config = load_email_config()
    transport = config.transport
    recipients = config.recipients(to)
    missing = config.missing()
    if not recipients:
        missing.append("TRADER_KOO_REPORT_EMAIL_TO (or use ?to=...)")
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing email config: {', '.join(missing)}",
        )
    report_dir = REPORT_DIR
    latest_path, latest_payload = latest_daily_report_json(report_dir)
    if latest_payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"No report found in {report_dir}",
        )
    latest_md_path = report_dir / "daily_report_latest.md"
    md_text = ""
    if latest_md_path.exists():
        try:
            md_text = latest_md_path.read_text(encoding="utf-8")
        except Exception as exc:
            LOG.warning("Failed to read markdown report: %s", exc)
            md_text = ""
    generated = str(
        latest_payload.get("generated_ts")
        or latest_payload.get("generated_at_utc")
        or latest_payload.get("snapshot_ts")
        or ""
    ).strip()
    if not generated:
        generated = (
            dt.datetime.now(dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        )
    subject = build_report_email_subject(latest_payload)
    text_body, html_body = build_report_email_bodies(
        latest_payload,
        md_text if include_markdown else "",
        app_url=report_email_app_url(),
    )
    attachments: list[Attachment] = []
    if attach_json:
        filename = (
            latest_path.name
            if latest_path is not None
            else "daily_report_latest.json"
        )
        json_bytes = json.dumps(latest_payload, indent=2).encode("utf-8")
        attachments.append(Attachment(json_bytes, "application", "json", filename))
    if include_markdown and md_text:
        attachments.append(
            Attachment(
                md_text.encode("utf-8"),
                "text",
                "markdown",
                "daily_report_latest.md",
            )
        )
    email = OutgoingEmail(
        subject=subject,
        text=text_body,
        html=html_body,
        attachments=tuple(attachments),
    )
    sender = transport_for(config)
    try:
        for recipient in recipients:
            sender.send(email, recipient)
    except Exception as exc:
        LOG.exception(
            "Failed to send daily report email (transport=%s)", transport
        )
        raise HTTPException(
            status_code=500, detail=f"Email send failed: {exc}"
        ) from exc
    return {
        "ok": True,
        "transport": transport,
        "to": ", ".join(recipients),
        "subject": subject,
        "report_file": str(latest_path) if latest_path else None,
        "smtp_host": config.smtp.host,
        "smtp_port": config.smtp.port,
        "smtp_security": config.smtp.security,
    }
