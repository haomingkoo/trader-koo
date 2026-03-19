"""Email dispatch, subscriber management, SMTP/Resend health."""
from __future__ import annotations

import datetime as dt
import json
import os
import smtplib
import ssl
import urllib.error
import urllib.parse
import urllib.request
from email.message import EmailMessage
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from trader_koo.backend.services.database import DB_PATH
from trader_koo.backend.services.report_loader import latest_daily_report_json
from trader_koo.email_subscribers import (
    email_max_recipients,
    email_subscribe_enabled,
    parse_recipients,
    subscriber_counts,
)
from trader_koo.llm_health import (
    llm_alert_cooldown_min,
    llm_alert_enabled,
    llm_degraded_threshold,
    llm_health_summary,
)
from trader_koo.middleware.auth import require_admin_auth
from trader_koo.report_email import (
    build_report_email_bodies,
    build_report_email_subject,
    report_email_app_url,
)

from trader_koo.backend.routers.admin._shared import LOG, REPORT_DIR

router = APIRouter(tags=["admin", "admin-email"])


# ---------------------------------------------------------------------------
# Email helpers
# ---------------------------------------------------------------------------


def _smtp_settings() -> dict[str, Any]:
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
    security = os.getenv(
        "TRADER_KOO_SMTP_SECURITY", "starttls"
    ).strip().lower()
    if security not in {"starttls", "ssl", "none"}:
        security = "starttls"
    return {
        "host": os.getenv("TRADER_KOO_SMTP_HOST", "").strip(),
        "port": port,
        "user": os.getenv("TRADER_KOO_SMTP_USER", "").strip(),
        "password": os.getenv("TRADER_KOO_SMTP_PASS", ""),
        "from_email": os.getenv("TRADER_KOO_SMTP_FROM", "").strip(),
        "default_to": os.getenv(
            "TRADER_KOO_REPORT_EMAIL_TO", ""
        ).strip(),
        "timeout_sec": timeout_sec,
        "security": security,
    }


def _resend_settings() -> dict[str, Any]:
    timeout_raw = os.getenv(
        "TRADER_KOO_RESEND_TIMEOUT_SEC",
        os.getenv("TRADER_KOO_SMTP_TIMEOUT_SEC", "30"),
    ).strip()
    try:
        timeout_sec = max(5, int(timeout_raw))
    except ValueError:
        timeout_sec = 30
    return {
        "api_key": os.getenv("TRADER_KOO_RESEND_API_KEY", "").strip(),
        "from_email": os.getenv(
            "TRADER_KOO_RESEND_FROM",
            os.getenv("TRADER_KOO_SMTP_FROM", ""),
        ).strip(),
        "default_to": os.getenv(
            "TRADER_KOO_REPORT_EMAIL_TO", ""
        ).strip(),
        "timeout_sec": timeout_sec,
    }


def _email_transport() -> str:
    raw = os.getenv(
        "TRADER_KOO_EMAIL_TRANSPORT", "auto"
    ).strip().lower()
    if raw not in {"auto", "smtp", "resend"}:
        raw = "auto"
    if raw == "auto":
        resend = _resend_settings()
        return "resend" if resend.get("api_key") else "smtp"
    return raw


def _send_resend_email(
    subject: str,
    text: str,
    recipient: str,
    resend: dict[str, Any],
    *,
    html_body: str | None = None,
) -> None:
    user_agent = os.getenv(
        "TRADER_KOO_EMAIL_USER_AGENT", "trader-koo/1.0"
    )
    payload = {
        "from": resend["from_email"],
        "to": [recipient],
        "subject": subject,
        "text": text,
    }
    if html_body:
        payload["html"] = html_body
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
        with urllib.request.urlopen(
            req, timeout=int(resend["timeout_sec"])
        ) as resp:
            status_code = int(getattr(resp, "status", 200))
            body = resp.read().decode("utf-8", errors="replace")
        if status_code >= 300:
            raise RuntimeError(
                f"Resend API failed status={status_code} body={body[:500]}"
            )
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Resend API HTTP {exc.code}: {err_body[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Resend connect failed: {exc.reason}"
        ) from exc


def _send_smtp_email(
    message: EmailMessage, smtp: dict[str, Any]
) -> None:
    host = smtp["host"]
    port = int(smtp["port"])
    timeout_sec = int(smtp["timeout_sec"])
    security = str(smtp["security"])
    user = str(smtp.get("user") or "")
    password = str(smtp.get("password") or "")
    if security == "ssl":
        with smtplib.SMTP_SSL(
            host, port, timeout=timeout_sec, context=ssl.create_default_context()
        ) as server:
            if user:
                server.login(user, password)
            server.send_message(message)
        return
    with smtplib.SMTP(host, port, timeout=timeout_sec) as server:
        server.ehlo()
        if security == "starttls":
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
        if user:
            server.login(user, password)
        server.send_message(message)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/api/admin/smtp-health")
@require_admin_auth
def smtp_health() -> dict[str, Any]:
    """Return email delivery config health (without secrets)."""
    smtp = _smtp_settings()
    resend = _resend_settings()
    transport = _email_transport()
    subs_enabled = email_subscribe_enabled()
    subs = (
        subscriber_counts(DB_PATH)
        if subs_enabled
        else {
            "active": 0,
            "pending": 0,
            "unsubscribed": 0,
            "total": 0,
        }
    )
    auto_email = str(
        os.getenv("TRADER_KOO_AUTO_EMAIL", "")
    ).strip().lower() in {"1", "true", "yes"}
    llm_alert_to = (
        str(os.getenv("TRADER_KOO_LLM_FAIL_ALERT_TO", "") or "").strip()
        or str(os.getenv("TRADER_KOO_LLM_ALERT_TO", "") or "").strip()
    )
    llm_alert_recipient_count = (
        len(parse_recipients(llm_alert_to)) if llm_alert_to else 0
    )
    llm_health: dict[str, Any] = {}
    try:
        llm_health = llm_health_summary(DB_PATH, recent_limit=10)
    except Exception as exc:
        LOG.warning(
            "Failed to load LLM health summary for smtp-health: %s", exc
        )
        llm_health = {"error": str(exc)}
    missing: list[str] = []
    if transport == "resend":
        if not resend["api_key"]:
            missing.append("TRADER_KOO_RESEND_API_KEY")
        if not resend["from_email"]:
            missing.append(
                "TRADER_KOO_RESEND_FROM (or TRADER_KOO_SMTP_FROM)"
            )
        if auto_email and not resend["default_to"]:
            missing.append("TRADER_KOO_REPORT_EMAIL_TO")
    else:
        if not smtp["host"]:
            missing.append("TRADER_KOO_SMTP_HOST")
        if not smtp["from_email"]:
            missing.append("TRADER_KOO_SMTP_FROM")
        if auto_email and not smtp["default_to"]:
            missing.append("TRADER_KOO_REPORT_EMAIL_TO")
        if smtp["user"] and not smtp["password"]:
            missing.append("TRADER_KOO_SMTP_PASS")
    return {
        "ok": len(missing) == 0,
        "auto_email_enabled": auto_email,
        "transport": transport,
        "subscriber_registry_enabled": subs_enabled,
        "subscriber_counts": subs,
        "email_max_recipients": email_max_recipients(),
        "missing": missing,
        "smtp": {
            "host": smtp["host"],
            "port": smtp["port"],
            "security": smtp["security"],
            "timeout_sec": smtp["timeout_sec"],
            "from_email": smtp["from_email"],
            "default_to": smtp["default_to"],
            "has_user": bool(smtp["user"]),
            "has_password": bool(smtp["password"]),
        },
        "resend": {
            "has_api_key": bool(resend["api_key"]),
            "from_email": resend["from_email"],
            "default_to": resend["default_to"],
            "timeout_sec": resend["timeout_sec"],
        },
        "llm_alert": {
            "enabled": llm_alert_enabled(),
            "cooldown_min": llm_alert_cooldown_min(),
            "degraded_threshold": llm_degraded_threshold(),
            "has_override_recipients": bool(llm_alert_to),
            "override_recipient_count": llm_alert_recipient_count,
            "health": llm_health,
        },
    }


@router.post("/api/admin/email-latest-report")
@require_admin_auth
def email_latest_report(
    to: str | None = Query(default=None),
    include_markdown: bool = Query(default=True),
    attach_json: bool = Query(default=True),
) -> dict[str, Any]:
    """Send the latest daily report by email via configured transport."""
    smtp = _smtp_settings()
    resend = _resend_settings()
    transport = _email_transport()
    default_to = (
        resend["default_to"] if transport == "resend" else smtp["default_to"]
    )
    recipient = (to or default_to or "").strip()
    missing: list[str] = []
    if transport == "resend":
        if not resend["api_key"]:
            missing.append("TRADER_KOO_RESEND_API_KEY")
        if not resend["from_email"]:
            missing.append(
                "TRADER_KOO_RESEND_FROM (or TRADER_KOO_SMTP_FROM)"
            )
    else:
        if not smtp["host"]:
            missing.append("TRADER_KOO_SMTP_HOST")
        if not smtp["from_email"]:
            missing.append("TRADER_KOO_SMTP_FROM")
        if smtp["user"] and not smtp["password"]:
            missing.append("TRADER_KOO_SMTP_PASS")
    if not recipient:
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
    from_header = (
        resend["from_email"] if transport == "resend" else smtp["from_email"]
    )
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = from_header
    message["To"] = recipient
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")
    if attach_json:
        filename = (
            latest_path.name
            if latest_path is not None
            else "daily_report_latest.json"
        )
        json_bytes = json.dumps(latest_payload, indent=2).encode("utf-8")
        message.add_attachment(
            json_bytes,
            maintype="application",
            subtype="json",
            filename=filename,
        )
    if include_markdown and md_text:
        message.add_attachment(
            md_text.encode("utf-8"),
            maintype="text",
            subtype="markdown",
            filename="daily_report_latest.md",
        )
    try:
        if transport == "resend":
            _send_resend_email(
                subject=subject,
                text=text_body,
                recipient=recipient,
                resend=resend,
                html_body=html_body,
            )
        else:
            _send_smtp_email(message, smtp)
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
        "to": recipient,
        "subject": subject,
        "report_file": str(latest_path) if latest_path else None,
        "smtp_host": smtp["host"],
        "smtp_port": smtp["port"],
        "smtp_security": smtp["security"],
    }
