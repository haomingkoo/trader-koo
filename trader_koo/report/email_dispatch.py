"""Email sending: SMTP and Resend transports for report and LLM alert emails."""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import smtplib
import ssl
import urllib.error
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from urllib.parse import quote

from trader_koo.email_subscribers import (
    already_sent_generated_report,
    email_max_recipients,
    ensure_subscriber_schema,
    parse_recipients,
    record_generated_report_delivery,
    resolve_report_recipients,
)
from trader_koo.llm_health import (
    llm_alert_enabled,
    mark_llm_alert_sent,
    should_send_llm_alert,
)
from trader_koo.report_email import (
    build_report_email_bodies,
    build_report_email_subject,
    report_email_app_url,
)

LOG = logging.getLogger(__name__)


def _smtp_cfg() -> dict[str, Any]:
    port_raw = os.getenv("TRADER_KOO_SMTP_PORT", "587").strip()
    try:
        port = int(port_raw)
    except ValueError:
        port = 587
    timeout_raw = os.getenv("TRADER_KOO_SMTP_TIMEOUT_SEC", "30").strip()
    try:
        timeout_sec = max(5, int(timeout_raw))
    except ValueError:
        timeout_sec = 30
    security = os.getenv("TRADER_KOO_SMTP_SECURITY", "starttls").strip().lower()
    if security not in {"starttls", "ssl", "none"}:
        security = "starttls"
    return {
        "host": os.getenv("TRADER_KOO_SMTP_HOST", "").strip(),
        "port": port,
        "user": os.getenv("TRADER_KOO_SMTP_USER", "").strip(),
        "password": os.getenv("TRADER_KOO_SMTP_PASS", ""),
        "from_email": os.getenv("TRADER_KOO_SMTP_FROM", "").strip(),
        "to_email": os.getenv("TRADER_KOO_REPORT_EMAIL_TO", "").strip(),
        "timeout_sec": timeout_sec,
        "security": security,
    }


def _resend_cfg() -> dict[str, Any]:
    timeout_raw = os.getenv("TRADER_KOO_RESEND_TIMEOUT_SEC", os.getenv("TRADER_KOO_SMTP_TIMEOUT_SEC", "30")).strip()
    try:
        timeout_sec = max(5, int(timeout_raw))
    except ValueError:
        timeout_sec = 30
    return {
        "api_key": os.getenv("TRADER_KOO_RESEND_API_KEY", "").strip(),
        "from_email": os.getenv("TRADER_KOO_RESEND_FROM", os.getenv("TRADER_KOO_SMTP_FROM", "")).strip(),
        "to_email": os.getenv("TRADER_KOO_REPORT_EMAIL_TO", "").strip(),
        "timeout_sec": timeout_sec,
    }


def _email_transport() -> str:
    raw = os.getenv("TRADER_KOO_EMAIL_TRANSPORT", "auto").strip().lower()
    if raw not in {"auto", "smtp", "resend"}:
        raw = "auto"
    if raw == "auto":
        return "resend" if _resend_cfg().get("api_key") else "smtp"
    return raw


def _send_resend_email(
    *,
    subject: str,
    text: str,
    html_body: str,
    resend: dict[str, Any],
    recipient: str,
) -> None:
    user_agent = os.getenv("TRADER_KOO_EMAIL_USER_AGENT", "trader-koo/1.0")
    payload = {
        "from": resend["from_email"],
        "to": [recipient],
        "subject": subject,
        "text": text,
        "html": html_body,
    }
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {resend['api_key']}",
            "Content-Type": "application/json",
            "User-Agent": user_agent,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=int(resend["timeout_sec"])) as resp:
            status = int(getattr(resp, "status", 200))
            body = resp.read().decode("utf-8", errors="replace")
        if status >= 300:
            raise RuntimeError(f"Resend API failed status={status} body={body[:500]}")
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Resend API HTTP {exc.code}: {err_body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Resend connect failed: {exc.reason}") from exc


def send_report_email(
    report: dict[str, Any],
    md_text: str,
    *,
    db_path: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Send daily report email to resolved recipients with anti-duplication safeguards."""
    transport = _email_transport()
    smtp = _smtp_cfg()
    resend = _resend_cfg()
    missing = []
    if transport == "resend":
        if not resend["api_key"]:
            missing.append("TRADER_KOO_RESEND_API_KEY")
        if not resend["from_email"]:
            missing.append("TRADER_KOO_RESEND_FROM (or TRADER_KOO_SMTP_FROM)")
    else:
        if not smtp["host"]:
            missing.append("TRADER_KOO_SMTP_HOST")
        if not smtp["from_email"]:
            missing.append("TRADER_KOO_SMTP_FROM")
    fallback_to = resend["to_email"] if transport == "resend" else smtp["to_email"]
    fallback_recipients = fallback_to or os.getenv("TRADER_KOO_REPORT_EMAIL_TO", "")
    if not db_path:
        default_db = Path(os.getenv("TRADER_KOO_DB_PATH", str((Path(__file__).resolve().parents[1] / "data" / "trader_koo.db"))))
        db_path = default_db.resolve()

    try:
        ensure_subscriber_schema(db_path)
        recipient_rows = resolve_report_recipients(
            db_path,
            fallback_raw=fallback_recipients,
            max_recipients=email_max_recipients(),
        )
    except Exception as exc:
        LOG.warning("Subscriber resolution failed, using env fallback: %s", exc)
        recipient_rows = [{"email": email, "source": "env", "unsubscribe_token": None} for email in parse_recipients(fallback_recipients)]
    if not recipient_rows:
        missing.append("TRADER_KOO_REPORT_EMAIL_TO or active subscribers")
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

    host, port, timeout_sec = smtp["host"], int(smtp["port"]), int(smtp["timeout_sec"])
    user, password, security = smtp["user"], smtp["password"], smtp["security"]
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
        unsub_token = str(row.get("unsubscribe_token") or "").strip()
        manage_url = None
        if app_url and unsub_token:
            manage_url = f"{app_url}/api/email/unsubscribe?token={quote(unsub_token)}"
        text_body, html_body = build_report_email_bodies(
            report,
            md_text,
            app_url=app_url,
            manage_url=manage_url,
        )
        try:
            if transport == "resend":
                _send_resend_email(
                    subject=subject,
                    text=text_body,
                    html_body=html_body,
                    resend=resend,
                    recipient=recipient,
                )
            else:
                msg = EmailMessage()
                msg["Subject"] = subject
                msg["From"] = smtp["from_email"]
                msg["To"] = recipient
                msg.set_content(text_body)
                msg.add_alternative(html_body, subtype="html")
                msg.add_attachment(
                    md_text.encode(),
                    maintype="text",
                    subtype="markdown",
                    filename=f"daily_report_{generated_key[:10]}.md",
                )
                if security == "ssl":
                    with smtplib.SMTP_SSL(host, port, timeout=timeout_sec, context=ssl.create_default_context()) as server:
                        if user:
                            server.login(user, password)
                        server.send_message(msg)
                else:
                    with smtplib.SMTP(host, port, timeout=timeout_sec) as server:
                        server.ehlo()
                        if security == "starttls":
                            server.starttls(context=ssl.create_default_context())
                            server.ehlo()
                        if user:
                            server.login(user, password)
                        server.send_message(msg)
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


def send_llm_failure_alert_email(
    report: dict[str, Any],
    *,
    db_path: Path,
) -> dict[str, Any]:
    """Send degraded-LLM alert email to operator recipients (not subscribers)."""
    if not llm_alert_enabled():
        return {"attempted": False, "reason": "alerts_disabled"}

    llm_meta = ((report.get("meta") or {}).get("llm") or {}) if isinstance(report.get("meta"), dict) else {}
    health = llm_meta.get("health") if isinstance(llm_meta.get("health"), dict) else {}
    if not health.get("degraded"):
        return {"attempted": False, "reason": "not_degraded"}

    should_alert, reason = should_send_llm_alert(db_path)
    if not should_alert:
        return {"attempted": False, "reason": reason}

    transport = _email_transport()
    smtp = _smtp_cfg()
    resend = _resend_cfg()
    fallback_to = resend["to_email"] if transport == "resend" else smtp["to_email"]
    alert_raw = (
        str(os.getenv("TRADER_KOO_LLM_FAIL_ALERT_TO", "") or "").strip()
        or str(os.getenv("TRADER_KOO_LLM_ALERT_TO", "") or "").strip()  # legacy alias
        or str(fallback_to or "").strip()
        or str(os.getenv("TRADER_KOO_REPORT_EMAIL_TO", "") or "").strip()
    )
    recipients = parse_recipients(alert_raw)
    if not recipients:
        return {"attempted": False, "reason": "missing_alert_recipients"}

    if transport == "resend":
        if not resend.get("api_key") or not resend.get("from_email"):
            return {"attempted": False, "reason": "resend_not_configured"}
    else:
        if not smtp.get("host") or not smtp.get("from_email"):
            return {"attempted": False, "reason": "smtp_not_configured"}
        if smtp.get("user") and not smtp.get("password"):
            return {"attempted": False, "reason": "smtp_password_missing"}

    generated = str(report.get("generated_ts") or "").strip() or dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    report_kind = str(((report.get("meta") or {}).get("report_kind")) or "daily").strip().lower()
    consecutive = int(health.get("consecutive_failures") or 0)
    last_failure_ts = str(health.get("last_failure_ts") or "-")
    last_reason = str(health.get("last_failure_reason") or "unknown")
    last_error = str(health.get("last_error_class") or "unknown")
    runtime_disabled = bool(llm_meta.get("runtime_disabled"))
    disable_remain = int(llm_meta.get("runtime_disabled_remaining_sec") or 0)
    disable_hint = f"{disable_remain}s remaining" if runtime_disabled and disable_remain > 0 else "not in cooldown"

    subject = (
        f"[trader_koo] LLM ALERT | {generated[:10]} | {report_kind.upper()} degraded "
        f"(fails={consecutive})"
    )
    text_body = (
        "trader_koo LLM degradation alert\n\n"
        f"Generated: {generated}\n"
        f"Report kind: {report_kind}\n"
        f"Consecutive failures: {consecutive}\n"
        f"Last failure ts: {last_failure_ts}\n"
        f"Last failure reason: {last_reason}\n"
        f"Last error class: {last_error}\n"
        f"Runtime cooldown: {disable_hint}\n\n"
        "Fallback is active: rule-based narratives are being used for continuity.\n"
        "Please check Azure OpenAI credentials/deployment/network and recent API logs.\n"
    )
    html_body = (
        "<div style='font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;color:#0f172a;'>"
        "<h2 style='margin:0 0 10px;color:#b42318;'>trader_koo LLM Alert</h2>"
        "<p style='line-height:1.6;margin:0 0 12px;'>LLM narrative generation is degraded. "
        "Rule-based fallback is active so reports continue to run.</p>"
        "<table style='border-collapse:collapse;'>"
        f"<tr><td style='padding:4px 10px 4px 0;color:#475569;'>Generated</td><td style='padding:4px 0;font-weight:700;'>{generated}</td></tr>"
        f"<tr><td style='padding:4px 10px 4px 0;color:#475569;'>Report kind</td><td style='padding:4px 0;'>{report_kind}</td></tr>"
        f"<tr><td style='padding:4px 10px 4px 0;color:#475569;'>Consecutive failures</td><td style='padding:4px 0;font-weight:700;color:#b42318;'>{consecutive}</td></tr>"
        f"<tr><td style='padding:4px 10px 4px 0;color:#475569;'>Last failure</td><td style='padding:4px 0;'>{last_failure_ts}</td></tr>"
        f"<tr><td style='padding:4px 10px 4px 0;color:#475569;'>Reason</td><td style='padding:4px 0;'>{last_reason}</td></tr>"
        f"<tr><td style='padding:4px 10px 4px 0;color:#475569;'>Error class</td><td style='padding:4px 0;'>{last_error}</td></tr>"
        f"<tr><td style='padding:4px 10px 4px 0;color:#475569;'>Cooldown</td><td style='padding:4px 0;'>{disable_hint}</td></tr>"
        "</table>"
        "<p style='line-height:1.6;margin:12px 0 0;color:#475569;'>"
        "Check `/api/admin/llm-health` and application logs for details."
        "</p>"
        "</div>"
    )

    sent = 0
    failed = 0
    failures: list[str] = []
    for recipient in recipients:
        try:
            if transport == "resend":
                _send_resend_email(
                    subject=subject,
                    text=text_body,
                    html_body=html_body,
                    resend=resend,
                    recipient=recipient,
                )
            else:
                msg = EmailMessage()
                msg["Subject"] = subject
                msg["From"] = smtp["from_email"]
                msg["To"] = recipient
                msg.set_content(text_body)
                msg.add_alternative(html_body, subtype="html")
                host, port, timeout_sec = smtp["host"], int(smtp["port"]), int(smtp["timeout_sec"])
                user, password, security = smtp["user"], smtp["password"], smtp["security"]
                if security == "ssl":
                    with smtplib.SMTP_SSL(host, port, timeout=timeout_sec, context=ssl.create_default_context()) as server:
                        if user:
                            server.login(user, password)
                        server.send_message(msg)
                else:
                    with smtplib.SMTP(host, port, timeout=timeout_sec) as server:
                        server.ehlo()
                        if security == "starttls":
                            server.starttls(context=ssl.create_default_context())
                            server.ehlo()
                        if user:
                            server.login(user, password)
                        server.send_message(msg)
            sent += 1
        except Exception as exc:
            failed += 1
            failures.append(f"{recipient}: {exc}")

    if sent > 0:
        mark_llm_alert_sent(
            db_path,
            detail={
                "generated_ts": generated,
                "report_kind": report_kind,
                "sent_count": sent,
                "failed_count": failed,
                "reason": reason,
            },
        )
    if sent == 0 and failed > 0:
        raise RuntimeError(f"LLM alert email failed for all recipients: {failures[0]}")
    return {
        "attempted": True,
        "reason": reason,
        "transport": transport,
        "recipients_total": len(recipients),
        "sent_count": sent,
        "failed_count": failed,
        "failed_recipients": failures[:10],
    }
