"""Email endpoints: chart-preview, plus the transport helpers shared with admin."""
from __future__ import annotations

import json
import os
import smtplib
import ssl
import urllib.error
import urllib.request
from email.message import EmailMessage
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response

from trader_koo.backend.services.database import get_conn
from trader_koo.email_chart_preview import (
    build_email_chart_preview_png,
    verify_chart_preview_signature,
)

router = APIRouter()

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
