"""Nightly report email: recipients, de-duplication and delivery records.

Transport config and sending live in ``trader_koo.email_delivery``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from trader_koo.email_delivery import (
    Attachment,
    EmailTransport,
    OutgoingEmail,
    load_email_config,
    transport_for,
)
from trader_koo.email_subscribers import (
    already_sent_generated_report,
    record_generated_report_delivery,
)
from trader_koo.report_email import (
    build_report_email_bodies,
    build_report_email_subject,
    report_email_app_url,
)

LOG = logging.getLogger(__name__)


def send_report_email(
    report: dict[str, Any],
    md_text: str,
    *,
    db_path: Path | None = None,
    force: bool = False,
    email_transport: EmailTransport | None = None,
) -> dict[str, Any]:
    """Send daily report email to env recipients with anti-duplication safeguards."""
    config = load_email_config()
    transport = config.transport
    sender = transport_for(config) if email_transport is None else email_transport
    missing = config.missing()
    if not db_path:
        default_db = Path(os.getenv("TRADER_KOO_DB_PATH", str((Path(__file__).resolve().parents[1] / "data" / "trader_koo.db"))))
        db_path = default_db.resolve()

    recipient_rows = [{"email": email} for email in config.recipients()]
    if not recipient_rows:
        missing.append("TRADER_KOO_REPORT_EMAIL_TO")
    if missing:
        raise RuntimeError(f"Missing email config for {transport}: {', '.join(missing)}")

    generated = str(report.get("generated_ts") or "").strip()
    generated_key = generated or "unknown"
    subject = build_report_email_subject(report)
    app_url = report_email_app_url()
    sent = 0
    failed = 0
    skipped_duplicate = 0
    failures: list[str] = []

    for row in recipient_rows:
        recipient = str(row.get("email") or "").strip().lower()
        if not recipient:
            continue
        if generated and not force:
            try:
                if already_sent_generated_report(db_path, email=recipient, generated_ts=generated):
                    skipped_duplicate += 1
                    continue
            except Exception as exc:
                LOG.warning("Duplicate-send check failed for %s: %s", recipient, exc)
        text_body, html_body = build_report_email_bodies(
            report,
            md_text,
            app_url=app_url,
        )
        try:
            sender.send(
                OutgoingEmail(
                    subject=subject,
                    text=text_body,
                    html=html_body,
                    attachments=(
                        Attachment(
                            md_text.encode(),
                            "text",
                            "markdown",
                            f"daily_report_{generated_key[:10]}.md",
                        ),
                    ),
                ),
                recipient,
            )
            sent += 1
            try:
                if generated:
                    record_generated_report_delivery(
                        db_path,
                        email=recipient,
                        generated_ts=generated,
                        transport=transport,
                        status="sent",
                    )
            except Exception as exc_rec:
                LOG.warning("Failed to record successful delivery for %s: %s", recipient, exc_rec)
        except Exception as exc:
            failed += 1
            failures.append(f"{recipient}: {exc}")
            try:
                if generated:
                    record_generated_report_delivery(
                        db_path,
                        email=recipient,
                        generated_ts=generated,
                        transport=transport,
                        status="failed",
                        error=str(exc),
                    )
            except Exception as exc_rec:
                LOG.warning("Failed to record failed delivery for %s: %s", recipient, exc_rec)
    if sent == 0 and failed > 0:
        raise RuntimeError(f"Email send failed for all recipients: {failures[0]}")
    if sent == 0 and skipped_duplicate > 0:
        raise RuntimeError("Email skipped: latest generated report was already sent to all recipients.")
    return {
        "transport": transport,
        "recipients_total": len(recipient_rows),
        "sent_count": sent,
        "failed_count": failed,
        "skipped_duplicate_count": skipped_duplicate,
        "failed_recipients": failures[:10],
        "sample_recipients": [str(r.get("email") or "") for r in recipient_rows[:8]],
    }
