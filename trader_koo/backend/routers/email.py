"""Email endpoints: subscribe, confirm, unsubscribe, chart-preview."""
from __future__ import annotations

import html
import json
import logging
import os
import smtplib
import ssl
import urllib.error
import urllib.parse
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse

from trader_koo.backend.services.database import DB_PATH, get_conn
from trader_koo.email_chart_preview import (
    build_email_chart_preview_png,
    verify_chart_preview_signature,
)
from trader_koo.email_subscribers import (
    check_request_rate_limit,
    confirm_subscriber_token,
    email_subscribe_enabled,
    find_subscriber_by_email,
    is_valid_email,
    mark_verification_sent,
    record_subscriber_event,
    subscribe_email_daily_limit,
    subscribe_ip_hourly_limit,
    subscribe_resend_cooldown_min,
    unsubscribe_subscriber_token,
    upsert_subscriber_pending,
)
from trader_koo.report_email import report_email_app_url

router = APIRouter()

LOG = logging.getLogger("trader_koo.routers.email")


# ---------------------------------------------------------------------------
# Env-derived helpers
# ---------------------------------------------------------------------------

def _clean_optional_url(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw or raw == "*":
        return None
    if raw.startswith(("http://", "https://")):
        return raw.rstrip("/")
    return raw


STATUS_APP_URL = _clean_optional_url(os.getenv("TRADER_KOO_APP_URL")) or _clean_optional_url(
    os.getenv("TRADER_KOO_ALLOWED_ORIGIN")
)
STATUS_BASE_URL = _clean_optional_url(os.getenv("TRADER_KOO_BASE_URL"))


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "-"


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
    security = os.getenv("TRADER_KOO_SMTP_SECURITY", "starttls").strip().lower()
    if security not in {"starttls", "ssl", "none"}:
        security = "starttls"
    return {
        "host": os.getenv("TRADER_KOO_SMTP_HOST", "").strip(),
        "port": port,
        "user": os.getenv("TRADER_KOO_SMTP_USER", "").strip(),
        "password": os.getenv("TRADER_KOO_SMTP_PASS", ""),
        "from_email": os.getenv("TRADER_KOO_SMTP_FROM", "").strip(),
        "default_to": os.getenv("TRADER_KOO_REPORT_EMAIL_TO", "").strip(),
        "timeout_sec": timeout_sec,
        "security": security,
    }


def _resend_settings() -> dict[str, Any]:
    timeout_raw = os.getenv(
        "TRADER_KOO_RESEND_TIMEOUT_SEC", os.getenv("TRADER_KOO_SMTP_TIMEOUT_SEC", "30")
    ).strip()
    try:
        timeout_sec = max(5, int(timeout_raw))
    except ValueError:
        timeout_sec = 30
    return {
        "api_key": os.getenv("TRADER_KOO_RESEND_API_KEY", "").strip(),
        "from_email": os.getenv("TRADER_KOO_RESEND_FROM", os.getenv("TRADER_KOO_SMTP_FROM", "")).strip(),
        "default_to": os.getenv("TRADER_KOO_REPORT_EMAIL_TO", "").strip(),
        "timeout_sec": timeout_sec,
    }


def _email_transport() -> str:
    raw = os.getenv("TRADER_KOO_EMAIL_TRANSPORT", "auto").strip().lower()
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
    user_agent = os.getenv("TRADER_KOO_EMAIL_USER_AGENT", "trader-koo/1.0")
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


def _send_smtp_email(message: EmailMessage, smtp: dict[str, Any]) -> None:
    host = smtp["host"]
    port = int(smtp["port"])
    timeout_sec = int(smtp["timeout_sec"])
    security = str(smtp["security"])
    user = str(smtp.get("user") or "")
    password = str(smtp.get("password") or "")
    if security == "ssl":
        with smtplib.SMTP_SSL(host, port, timeout=timeout_sec, context=ssl.create_default_context()) as server:
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


def _send_transactional_email(
    *,
    recipient: str,
    subject: str,
    text_body: str,
    html_body: str,
) -> str:
    transport = _email_transport()
    smtp = _smtp_settings()
    resend = _resend_settings()
    if transport == "resend":
        if not resend["api_key"]:
            raise RuntimeError("Missing TRADER_KOO_RESEND_API_KEY")
        if not resend["from_email"]:
            raise RuntimeError("Missing TRADER_KOO_RESEND_FROM (or TRADER_KOO_SMTP_FROM)")
        _send_resend_email(
            subject=subject,
            text=text_body,
            recipient=recipient,
            resend=resend,
            html_body=html_body,
        )
        return "resend"
    if not smtp["host"]:
        raise RuntimeError("Missing TRADER_KOO_SMTP_HOST")
    if not smtp["from_email"]:
        raise RuntimeError("Missing TRADER_KOO_SMTP_FROM")
    if smtp["user"] and not smtp["password"]:
        raise RuntimeError("Missing TRADER_KOO_SMTP_PASS")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp["from_email"]
    msg["To"] = recipient
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")
    _send_smtp_email(msg, smtp)
    return "smtp"


def _subscription_base_url(request: Request | None = None) -> str:
    explicit = report_email_app_url() or STATUS_APP_URL or STATUS_BASE_URL
    if explicit:
        return str(explicit).rstrip("/")
    if request is not None:
        return str(request.base_url).rstrip("/")
    return "http://localhost:8000"


def _subscription_result_page(title: str, message: str, *, ok: bool = True) -> HTMLResponse:
    accent = "#0f9d58" if ok else "#d93025"
    safe_title = html.escape(title)
    safe_message = html.escape(message)
    page = f"""\
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{safe_title}</title>
  </head>
  <body style="margin:0;padding:22px;background:#eef3f8;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#0f172a;">
    <div style="max-width:560px;margin:0 auto;border:1px solid #dde7f0;border-radius:16px;background:#ffffff;padding:20px;">
      <div style="font-size:20px;line-height:26px;font-weight:700;color:{accent};">{safe_title}</div>
      <div style="margin-top:10px;font-size:14px;line-height:22px;color:#334155;">{safe_message}</div>
      <div style="margin-top:14px;font-size:12px;line-height:18px;color:#64748b;">
        trader_koo alerts are research only and not financial advice.
      </div>
    </div>
  </body>
</html>
"""
    return HTMLResponse(content=page)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/api/email/subscribe")
async def email_subscribe(request: Request) -> dict[str, Any]:
    if not email_subscribe_enabled():
        raise HTTPException(status_code=404, detail="Email subscriptions are disabled.")
    payload: dict[str, Any] = {}
    try:
        payload = await request.json()
    except Exception:
        # Allow empty body -- email can come from query params
        payload = {}
    email = str(payload.get("email") or request.query_params.get("email") or "").strip().lower()
    if not is_valid_email(email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    source_ip = _client_ip(request)
    allowed, reason = check_request_rate_limit(
        DB_PATH,
        event_type="subscribe_request",
        source_ip=source_ip,
        email=email,
        ip_hourly_limit=subscribe_ip_hourly_limit(),
        email_daily_limit=subscribe_email_daily_limit(),
    )
    if not allowed:
        record_subscriber_event(
            DB_PATH,
            event_type="subscribe_rate_limited",
            email=email,
            source_ip=source_ip,
            details=reason or "",
        )
        raise HTTPException(status_code=429, detail=reason or "Too many requests. Try again later.")

    upserted = upsert_subscriber_pending(
        DB_PATH,
        email=email,
        source_ip=source_ip,
        source_user_agent=request.headers.get("user-agent"),
        resend_cooldown_min=subscribe_resend_cooldown_min(),
    )
    if not upserted.get("ok"):
        raise HTTPException(status_code=400, detail=upserted.get("detail") or "Unable to register email.")
    state = str(upserted.get("state") or "pending")
    should_send = bool(upserted.get("should_send_verification"))
    confirm_token = str(upserted.get("confirm_token") or "")
    unsubscribe_token = str(upserted.get("unsubscribe_token") or "")
    base_url = _subscription_base_url(request)
    confirm_url = f"{base_url}/api/email/confirm?token={urllib.parse.quote(confirm_token)}"
    unsubscribe_url = f"{base_url}/api/email/unsubscribe?token={urllib.parse.quote(unsubscribe_token)}"
    transport_used = None
    if should_send:
        subject = "[trader_koo] Confirm your email subscription"
        text_body = (
            "trader_koo email confirmation\n\n"
            f"Click to confirm alerts: {confirm_url}\n"
            f"If this was not you, ignore this email or unsubscribe: {unsubscribe_url}\n\n"
            "This dashboard is research only and not financial advice."
        )
        html_body = (
            "<div style='font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;color:#0f172a;'>"
            "<h2 style='margin:0 0 10px;'>Confirm trader_koo Alerts</h2>"
            "<p style='line-height:1.6;'>You requested daily trader_koo report alerts.</p>"
            f"<p><a href='{confirm_url}' style='display:inline-block;padding:10px 14px;border-radius:999px;"
            "background:#1677ff;color:#fff;text-decoration:none;font-weight:700;'>Confirm subscription</a></p>"
            f"<p style='font-size:13px;color:#475569;'>Not you? <a href='{unsubscribe_url}'>Unsubscribe</a>"
            " or ignore this message.</p>"
            "<p style='font-size:12px;color:#64748b;'>Research only. Not financial advice.</p>"
            "</div>"
        )
        try:
            transport_used = _send_transactional_email(
                recipient=email,
                subject=subject,
                text_body=text_body,
                html_body=html_body,
            )
            mark_verification_sent(DB_PATH, email=email)
        except Exception as exc:
            record_subscriber_event(
                DB_PATH,
                event_type="subscribe_send_failed",
                email=email,
                source_ip=source_ip,
                details=str(exc),
            )
            raise HTTPException(status_code=500, detail=f"Unable to send confirmation email: {exc}") from exc

    detail = (
        "Confirmation email sent. Check inbox/spam and click confirm."
        if should_send
        else "Subscription request already pending recently. Check your inbox for the latest confirmation email."
    )
    if state == "already_active":
        detail = "This email is already subscribed."
    return {
        "ok": True,
        "email": email,
        "state": state,
        "verification_sent": should_send,
        "transport": transport_used,
        "detail": detail,
    }


@router.post("/api/email/unsubscribe-request")
async def email_unsubscribe_request(request: Request) -> dict[str, Any]:
    if not email_subscribe_enabled():
        raise HTTPException(status_code=404, detail="Email subscriptions are disabled.")
    payload: dict[str, Any] = {}
    try:
        payload = await request.json()
    except Exception:
        # Allow empty body -- email can come from query params
        payload = {}
    email = str(payload.get("email") or request.query_params.get("email") or "").strip().lower()
    if not is_valid_email(email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    source_ip = _client_ip(request)
    allowed, reason = check_request_rate_limit(
        DB_PATH,
        event_type="unsubscribe_request",
        source_ip=source_ip,
        email=email,
        ip_hourly_limit=subscribe_ip_hourly_limit(),
        email_daily_limit=subscribe_email_daily_limit(),
    )
    if not allowed:
        raise HTTPException(status_code=429, detail=reason or "Too many requests. Try again later.")
    record_subscriber_event(
        DB_PATH,
        event_type="unsubscribe_request",
        email=email,
        source_ip=source_ip,
    )
    row = find_subscriber_by_email(DB_PATH, email=email)
    if row and str(row.get("unsubscribe_token") or "").strip():
        base_url = _subscription_base_url(request)
        unsub_url = (
            f"{base_url}/api/email/unsubscribe?token="
            f"{urllib.parse.quote(str(row.get('unsubscribe_token') or '').strip())}"
        )
        text_body = (
            "trader_koo unsubscribe request\n\n"
            f"Click to unsubscribe: {unsub_url}\n\n"
            "If this was not you, ignore this email."
        )
        html_body = (
            "<div style='font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;color:#0f172a;'>"
            "<h2 style='margin:0 0 10px;'>Manage trader_koo Alerts</h2>"
            "<p style='line-height:1.6;'>You requested an unsubscribe link.</p>"
            f"<p><a href='{unsub_url}' style='display:inline-block;padding:10px 14px;border-radius:999px;"
            "background:#1677ff;color:#fff;text-decoration:none;font-weight:700;'>Unsubscribe</a></p>"
            "<p style='font-size:12px;color:#64748b;'>If this was not you, ignore this message.</p>"
            "</div>"
        )
        try:
            _send_transactional_email(
                recipient=email,
                subject="[trader_koo] Unsubscribe link",
                text_body=text_body,
                html_body=html_body,
            )
        except Exception as exc:
            LOG.warning("Unsubscribe request email send failed for %s: %s", email, exc)
    return {
        "ok": True,
        "detail": "If the address exists, an unsubscribe email has been sent.",
    }


@router.get("/api/email/confirm")
def email_confirm(token: str = Query(..., min_length=16, max_length=512)) -> HTMLResponse:
    row = confirm_subscriber_token(DB_PATH, token=token)
    if row is None:
        return _subscription_result_page(
            "Confirmation Link Invalid",
            "This link is invalid or expired. Request a new confirmation from the dashboard.",
            ok=False,
        )
    if row.get("error") == "token_expired":
        return _subscription_result_page(
            "Confirmation Link Expired",
            row.get("detail", "This confirmation link has expired. Please request a new one."),
            ok=False,
        )
    return _subscription_result_page(
        "Subscription Confirmed",
        f"{row['email']} is now subscribed to trader_koo report emails.",
        ok=True,
    )


@router.get("/api/email/unsubscribe")
def email_unsubscribe(token: str = Query(..., min_length=16, max_length=512)) -> HTMLResponse:
    row = unsubscribe_subscriber_token(DB_PATH, token=token)
    if row is None:
        return _subscription_result_page(
            "Unsubscribe Link Invalid",
            "This link is invalid or expired. Request a new unsubscribe link from the dashboard.",
            ok=False,
        )
    return _subscription_result_page(
        "Unsubscribed",
        f"{row['email']} has been removed from trader_koo report emails.",
        ok=True,
    )


@router.get("/api/email/chart-preview")
def email_chart_preview(
    ticker: str = Query(..., min_length=1, max_length=16),
    timeframe: str = Query(default="daily"),
    report_ts: str | None = Query(default=None),
    exp: int = Query(...),
    sig: str = Query(..., min_length=32, max_length=128),
) -> Response:
    if not verify_chart_preview_signature(
        ticker=ticker,
        timeframe=timeframe,
        report_ts=report_ts,
        exp=exp,
        sig=sig,
    ):
        raise HTTPException(status_code=403, detail="Invalid chart preview signature")
    conn = get_conn()
    try:
        png = build_email_chart_preview_png(
            conn,
            ticker=ticker,
            timeframe=timeframe,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        conn.close()
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )
