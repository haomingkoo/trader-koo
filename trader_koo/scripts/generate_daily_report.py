#!/usr/bin/env python3
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import math
import os
import smtplib
import sqlite3
import ssl
import urllib.error
import urllib.request
from urllib.parse import quote
from collections import defaultdict
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from trader_koo.catalyst_data import build_earnings_calendar_payload
from trader_koo.debate_engine import build_setup_debate
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
from trader_koo.llm_narrative import llm_enabled, llm_max_setups, llm_status, maybe_rewrite_setup_copy
from trader_koo.report_email import (
    build_report_email_bodies,
    build_report_email_subject,
    report_email_app_url,
)
from trader_koo.features.technical import FeatureConfig, add_basic_features, compute_pivots
from trader_koo.structure.levels import (
    LevelConfig,
    add_fallback_levels,
    build_levels_from_pivots,
    select_target_levels,
)
from trader_koo.structure.vix_analysis import (
    calculate_term_structure,
    calculate_vix_percentile,
    get_percentile_color,
    should_show_volatility_warning,
)
from trader_koo.structure.vix_patterns import (
    VIXTrapReclaimConfig,
    detect_vix_trap_reclaim_patterns,
    get_pattern_glossary,
)


MARKET_TZ_NAME = os.getenv("TRADER_KOO_MARKET_TZ", "America/New_York")
try:
    MARKET_TZ = ZoneInfo(MARKET_TZ_NAME)
except Exception:
    MARKET_TZ = dt.timezone.utc
MARKET_CLOSE_HOUR = min(23, max(0, int(os.getenv("TRADER_KOO_MARKET_CLOSE_HOUR", "16"))))
TRUTHY_VALUES = {"1", "true", "yes", "on"}
REPORT_FEATURE_CFG = FeatureConfig()
REPORT_LEVEL_CFG = LevelConfig()


def _as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in TRUTHY_VALUES


SETUP_EVAL_ENABLED = _as_bool(os.getenv("TRADER_KOO_SETUP_EVAL_ENABLED", "1"))
SETUP_EVAL_TRACK_LIMIT = max(5, int(os.getenv("TRADER_KOO_SETUP_EVAL_TRACK_LIMIT", "40")))
SETUP_EVAL_WINDOW_DAYS = max(30, int(os.getenv("TRADER_KOO_SETUP_EVAL_WINDOW_DAYS", "180")))
SETUP_EVAL_MIN_SAMPLE = max(3, int(os.getenv("TRADER_KOO_SETUP_EVAL_MIN_SAMPLE", "5")))
SETUP_EVAL_HIT_THRESHOLD_PCT = float(os.getenv("TRADER_KOO_SETUP_EVAL_HIT_THRESHOLD_PCT", "0.3"))
DEBATE_ENGINE_ENABLED = _as_bool(os.getenv("TRADER_KOO_DEBATE_ENABLED", "1"))


def _normalize_report_kind(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw == "weekly":
        return "weekly"
    return "daily"


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def parse_iso_utc(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        ts = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def hours_since(value: str | None, now: dt.datetime) -> float | None:
    ts = parse_iso_utc(value)
    if ts is None:
        return None
    return (now - ts).total_seconds() / 3600.0


def days_since_date(value: str | None, now: dt.datetime) -> float | None:
    if not value:
        return None
    try:
        market_date = dt.date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None
    market_close = dt.datetime.combine(market_date, dt.time(hour=MARKET_CLOSE_HOUR), tzinfo=MARKET_TZ)
    now_market = now.astimezone(MARKET_TZ)
    age_days = (now_market - market_close).total_seconds() / 86400.0
    return max(0.0, age_days)


def _observed_holiday(day: dt.date) -> dt.date:
    # NYSE-style weekend observation for fixed-date holidays.
    if day.weekday() == 5:  # Saturday -> Friday
        return day - dt.timedelta(days=1)
    if day.weekday() == 6:  # Sunday -> Monday
        return day + dt.timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    first = dt.date(year, month, 1)
    delta = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=delta + (n - 1) * 7)


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    last_day = calendar.monthrange(year, month)[1]
    d = dt.date(year, month, last_day)
    delta = (d.weekday() - weekday) % 7
    return d - dt.timedelta(days=delta)


def _easter_sunday(year: int) -> dt.date:
    # Anonymous Gregorian algorithm.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def nyse_holidays_for_year(year: int) -> dict[dt.date, str]:
    out: dict[dt.date, str] = {}
    out[_observed_holiday(dt.date(year, 1, 1))] = "New Year's Day"
    # Include observed New Year of next year if it lands in this year (e.g. Dec 31).
    next_new_year_obs = _observed_holiday(dt.date(year + 1, 1, 1))
    if next_new_year_obs.year == year:
        out[next_new_year_obs] = "New Year's Day (Observed)"
    out[_nth_weekday(year, 1, 0, 3)] = "Martin Luther King Jr. Day"  # 3rd Monday Jan
    out[_nth_weekday(year, 2, 0, 3)] = "Washington's Birthday"  # 3rd Monday Feb
    out[_easter_sunday(year) - dt.timedelta(days=2)] = "Good Friday"
    out[_last_weekday(year, 5, 0)] = "Memorial Day"  # last Monday May
    out[_observed_holiday(dt.date(year, 6, 19))] = "Juneteenth National Independence Day"
    out[_observed_holiday(dt.date(year, 7, 4))] = "Independence Day"
    out[_nth_weekday(year, 9, 0, 1)] = "Labor Day"
    out[_nth_weekday(year, 11, 3, 4)] = "Thanksgiving Day"  # 4th Thursday Nov
    out[_observed_holiday(dt.date(year, 12, 25))] = "Christmas Day"
    return out


def nyse_early_closes_for_year(year: int) -> dict[dt.date, str]:
    out: dict[dt.date, str] = {}
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    friday_after_thanksgiving = thanksgiving + dt.timedelta(days=1)
    if friday_after_thanksgiving.weekday() < 5:
        out[friday_after_thanksgiving] = "Day after Thanksgiving (1:00 PM ET close)"

    christmas_eve = dt.date(year, 12, 24)
    if christmas_eve.weekday() < 5 and christmas_eve not in nyse_holidays_for_year(year):
        out[christmas_eve] = "Christmas Eve (1:00 PM ET close)"

    july3 = dt.date(year, 7, 3)
    if july3.weekday() < 5 and july3 not in nyse_holidays_for_year(year):
        out[july3] = "Pre-Independence Day (1:00 PM ET close)"
    return out


def market_calendar_context(now_utc: dt.datetime) -> dict[str, Any]:
    now_et = now_utc.astimezone(MARKET_TZ)
    today = now_et.date()
    years = [today.year - 1, today.year, today.year + 1]
    holidays: dict[dt.date, str] = {}
    early_closes: dict[dt.date, str] = {}
    for year in years:
        holidays.update(nyse_holidays_for_year(year))
        early_closes.update(nyse_early_closes_for_year(year))

    today_holiday = holidays.get(today)
    today_early_close = early_closes.get(today)
    next_holiday = None
    next_early_close = None
    for day in sorted(holidays):
        if day >= today:
            next_holiday = {"date": day.isoformat(), "name": holidays[day]}
            break
    for day in sorted(early_closes):
        if day >= today:
            next_early_close = {"date": day.isoformat(), "name": early_closes[day]}
            break

    return {
        "market_tz": MARKET_TZ_NAME,
        "as_of_market_ts": now_et.replace(microsecond=0).isoformat(),
        "market_date": today.isoformat(),
        "is_holiday": bool(today_holiday),
        "holiday_name": today_holiday,
        "is_early_close": bool(today_early_close),
        "early_close_name": today_early_close,
        "next_holiday": next_holiday,
        "next_early_close": next_early_close,
    }


def tail_text(path: Path, lines: int = 80, max_bytes: int = 96_000) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, max_bytes)
            f.seek(max(0, size - read_size))
            data = f.read().decode("utf-8", errors="replace")
        return data.splitlines()[-lines:]
    except Exception:
        return []


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any]:
    if row is None:
        return {}
    return dict(row)


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
    except Exception:
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
            except Exception:
                pass
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
            except Exception:
                pass
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
            except Exception:
                pass
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


def _parse_iso_date(value: Any) -> dt.date | None:
    if not value:
        return None
    raw = str(value).strip()
    if len(raw) < 10:
        return None
    try:
        return dt.date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _yolo_match_tolerance_days(timeframe: Any) -> int:
    tf = str(timeframe or "").strip().lower()
    return 35 if tf == "weekly" else 14


def _yolo_snapshot_matches(anchor: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if str(anchor.get("ticker") or "").upper() != str(candidate.get("ticker") or "").upper():
        return False
    if str(anchor.get("timeframe") or "").strip().lower() != str(candidate.get("timeframe") or "").strip().lower():
        return False
    if str(anchor.get("pattern") or "").strip() != str(candidate.get("pattern") or "").strip():
        return False

    tolerance = _yolo_match_tolerance_days(anchor.get("timeframe"))
    anchor_x0 = _parse_iso_date(anchor.get("x0_date"))
    anchor_x1 = _parse_iso_date(anchor.get("x1_date"))
    cand_x0 = _parse_iso_date(candidate.get("x0_date"))
    cand_x1 = _parse_iso_date(candidate.get("x1_date"))

    x0_match = (
        anchor_x0 is not None
        and cand_x0 is not None
        and abs((anchor_x0 - cand_x0).days) <= tolerance
    )
    x1_match = (
        anchor_x1 is not None
        and cand_x1 is not None
        and abs((anchor_x1 - cand_x1).days) <= tolerance
    )
    return x0_match or x1_match


def _yolo_seen_streak(seen_asofs: set[str], asof_dates_desc: list[str], latest_asof: str | None = None) -> int:
    if not seen_asofs or not asof_dates_desc:
        return 0
    target_start = str(latest_asof or asof_dates_desc[0] or "")
    started = False
    streak = 0
    for asof in asof_dates_desc:
        if not started:
            if asof != target_start:
                continue
            started = True
        if asof in seen_asofs:
            streak += 1
        else:
            break
    return streak


def _summarize_yolo_lifecycle(
    anchor: dict[str, Any],
    history_rows: list[dict[str, Any]],
    asof_dates_desc: list[str],
) -> dict[str, Any]:
    current_asof = str(anchor.get("as_of_date") or "").strip() or None
    seen_asofs: set[str] = set()
    for row in history_rows:
        asof = str(row.get("as_of_date") or "").strip()
        if not asof:
            continue
        if _yolo_snapshot_matches(anchor, row):
            seen_asofs.add(asof)

    if not seen_asofs:
        return {
            "first_seen_asof": current_asof,
            "last_seen_asof": current_asof,
            "snapshots_seen": 1 if current_asof else 0,
            "current_streak": 1 if current_asof else 0,
            "first_seen_days_ago": 0 if current_asof else None,
        }

    seen_sorted = sorted(seen_asofs)
    first_seen = seen_sorted[0]
    last_seen = seen_sorted[-1]
    first_dt = _parse_iso_date(first_seen)
    current_dt = _parse_iso_date(current_asof)
    first_seen_days_ago = None
    if first_dt is not None and current_dt is not None:
        first_seen_days_ago = max(0, (current_dt - first_dt).days)

    return {
        "first_seen_asof": first_seen,
        "last_seen_asof": last_seen,
        "snapshots_seen": len(seen_asofs),
        "current_streak": _yolo_seen_streak(seen_asofs, asof_dates_desc, current_asof),
        "first_seen_days_ago": first_seen_days_ago,
    }


def fetch_yolo_delta(
    conn: sqlite3.Connection,
    timeframe: str | None = None,
    x0_tolerance_days: int = 14,
) -> dict[str, Any]:
    """Compare YOLO detections between latest two as_of dates (optionally per timeframe)."""
    tf = str(timeframe or "").strip().lower()
    if tf not in {"daily", "weekly"}:
        tf = ""
    delta: dict[str, Any] = {
        "timeframe": tf or "all",
        "today_asof": None,
        "prev_asof": None,
        "history_retained": 0,
        "new_patterns": [],
        "lost_patterns": [],
        "new_count": 0,
        "lost_count": 0,
    }
    try:
        if tf:
            dates = conn.execute(
                """
                SELECT DISTINCT as_of_date
                FROM yolo_patterns
                WHERE as_of_date IS NOT NULL
                  AND timeframe = ?
                ORDER BY as_of_date DESC
                LIMIT 2
                """,
                (tf,),
            ).fetchall()
        else:
            dates = conn.execute(
                """
                SELECT DISTINCT as_of_date
                FROM yolo_patterns
                WHERE as_of_date IS NOT NULL
                ORDER BY as_of_date DESC
                LIMIT 2
                """
            ).fetchall()
        delta["history_retained"] = len(dates)
        if not dates:
            return delta
        today_asof = dates[0][0]
        delta["today_asof"] = today_asof
        if len(dates) < 2:
            return delta
        prev_asof = dates[1][0]
        delta["prev_asof"] = prev_asof

        def load_patterns(asof: str) -> list[dict]:
            if tf:
                rows = conn.execute(
                    """
                    SELECT ticker, timeframe, pattern, confidence, x0_date, x1_date
                    FROM yolo_patterns
                    WHERE as_of_date = ?
                      AND timeframe = ?
                    ORDER BY confidence DESC
                    """,
                    (asof, tf),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT ticker, timeframe, pattern, confidence, x0_date, x1_date
                    FROM yolo_patterns
                    WHERE as_of_date = ?
                    ORDER BY confidence DESC
                    """,
                    (asof,),
                ).fetchall()
            return [
                {"ticker": r[0], "timeframe": r[1], "pattern": r[2],
                 "confidence": round(float(r[3]), 3), "x0_date": r[4], "x1_date": r[5]}
                for r in rows
            ]

        today_pats = load_patterns(today_asof)
        prev_pats = load_patterns(prev_asof)

        def parse_date(s: str | None) -> dt.date | None:
            if not s:
                return None
            try:
                return dt.date.fromisoformat(str(s)[:10])
            except ValueError:
                return None

        tolerance = dt.timedelta(days=x0_tolerance_days)

        def find_match(pat: dict, candidates: list[dict]) -> bool:
            d = parse_date(pat["x0_date"])
            for c in candidates:
                if c["ticker"] != pat["ticker"]:
                    continue
                if c["timeframe"] != pat["timeframe"]:
                    continue
                if c["pattern"] != pat["pattern"]:
                    continue
                cd = parse_date(c["x0_date"])
                if d is None or cd is None:
                    continue
                if abs((d - cd).days) <= tolerance.days:
                    return True
            return False

        delta["new_patterns"] = [p for p in today_pats if not find_match(p, prev_pats)]
        delta["lost_patterns"] = [p for p in prev_pats if not find_match(p, today_pats)]
        delta["new_count"] = len(delta["new_patterns"])
        delta["lost_count"] = len(delta["lost_patterns"])
    except Exception:
        pass
    return delta


def fetch_yolo_pattern_persistence(
    conn: sqlite3.Connection,
    timeframe: str,
    lookback_asof: int = 20,
    top_n: int = 30,
) -> dict[str, Any]:
    tf = str(timeframe or "").strip().lower()
    if tf not in {"daily", "weekly"}:
        return {"timeframe": tf, "latest_asof": None, "lookback_asof": 0, "rows": []}

    dates_rows = conn.execute(
        """
        SELECT DISTINCT as_of_date
        FROM yolo_patterns
        WHERE timeframe = ? AND as_of_date IS NOT NULL
        ORDER BY as_of_date DESC
        LIMIT ?
        """,
        (tf, int(max(2, lookback_asof))),
    ).fetchall()
    asof_dates = [str(r[0]) for r in dates_rows if r and r[0]]
    if not asof_dates:
        return {"timeframe": tf, "latest_asof": None, "lookback_asof": 0, "rows": []}

    latest_asof = asof_dates[0]
    placeholders = ",".join(["?"] * len(asof_dates))
    rows = conn.execute(
        f"""
        SELECT
            as_of_date,
            ticker,
            pattern,
            AVG(CAST(confidence AS REAL)) AS confidence
        FROM yolo_patterns
        WHERE timeframe = ?
          AND as_of_date IN ({placeholders})
        GROUP BY as_of_date, ticker, pattern
        ORDER BY as_of_date DESC
        """,
        [tf, *asof_dates],
    ).fetchall()

    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        asof = str(row[0])
        ticker = str(row[1])
        pattern = str(row[2])
        conf = float(row[3] or 0.0)
        key = (ticker, pattern)
        state = by_key.get(key)
        if state is None:
            state = {
                "ticker": ticker,
                "timeframe": tf,
                "pattern": pattern,
                "asof_dates": set(),
                "latest_confidence": None,
                "avg_confidence_window": 0.0,
                "seen_count": 0,
            }
            by_key[key] = state
        state["asof_dates"].add(asof)
        state["avg_confidence_window"] += conf
        state["seen_count"] += 1
        if asof == latest_asof and state["latest_confidence"] is None:
            state["latest_confidence"] = round(conf, 3)

    out_rows: list[dict[str, Any]] = []
    for state in by_key.values():
        if latest_asof not in state["asof_dates"]:
            continue
        streak = 0
        for asof in asof_dates:
            if asof in state["asof_dates"]:
                streak += 1
            else:
                break
        seen = int(state["seen_count"])
        avg_conf = (
            round(float(state["avg_confidence_window"]) / float(seen), 3)
            if seen > 0
            else None
        )
        coverage_pct = round((100.0 * seen) / len(asof_dates), 2) if asof_dates else 0.0
        out_rows.append(
            {
                "ticker": state["ticker"],
                "timeframe": state["timeframe"],
                "pattern": state["pattern"],
                "latest_confidence": state["latest_confidence"],
                "avg_confidence_window": avg_conf,
                "streak": streak,
                "seen_in_lookback": seen,
                "lookback_asof": len(asof_dates),
                "coverage_pct": coverage_pct,
                "first_seen_asof": min(state["asof_dates"]),
                "last_seen_asof": max(state["asof_dates"]),
                "latest_asof": latest_asof,
            }
        )

    out_rows.sort(
        key=lambda x: (
            int(x.get("streak") or 0),
            int(x.get("seen_in_lookback") or 0),
            float(x.get("latest_confidence") or 0.0),
        ),
        reverse=True,
    )
    return {
        "timeframe": tf,
        "latest_asof": latest_asof,
        "lookback_asof": len(asof_dates),
        "rows": out_rows[: max(1, int(top_n))],
    }


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _setup_tier(score: float) -> str:
    if score >= 80.0:
        return "A"
    if score >= 65.0:
        return "B"
    if score >= 50.0:
        return "C"
    return "D"


def _stdev(values: list[float]) -> float | None:
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(max(var, 0.0))


def _fetch_volatility_inputs(conn: sqlite3.Connection) -> tuple[dict[str, dict[str, float | None]], dict[str, Any]]:
    """Build per-ticker volatility features and a market volatility context."""
    by_ticker: dict[str, dict[str, float | None]] = {}
    market_ctx: dict[str, Any] = {}
    if not table_exists(conn, "price_daily"):
        return by_ticker, market_ctx

    rows = conn.execute(
        """
        SELECT ticker, date, CAST(high AS REAL), CAST(low AS REAL), CAST(close AS REAL)
        FROM (
            SELECT ticker, date, high, low, close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM price_daily
        )
        WHERE rn <= 40
        ORDER BY ticker, date
        """
    ).fetchall()
    bucket: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for r in rows:
        ticker = str(r[0] or "").upper().strip()
        if not ticker:
            continue
        high = r[2]
        low = r[3]
        close = r[4]
        if high is None or low is None or close is None:
            continue
        try:
            h = float(high)
            l = float(low)
            c = float(close)
        except (TypeError, ValueError):
            continue
        if not (h > 0 and l > 0 and c > 0):
            continue
        bucket[ticker].append((h, l, c))

    for ticker, bars in bucket.items():
        if len(bars) < 3:
            continue
        highs = [b[0] for b in bars]
        lows = [b[1] for b in bars]
        closes = [b[2] for b in bars]

        returns: list[float] = []
        trs: list[float] = []
        for i in range(1, len(closes)):
            prev_close = closes[i - 1]
            close = closes[i]
            if prev_close > 0:
                returns.append((close / prev_close) - 1.0)
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - prev_close),
                abs(lows[i] - prev_close),
            )
            trs.append(tr)

        atr_pct_14: float | None = None
        if len(trs) >= 14 and closes[-1] > 0:
            atr_14 = sum(trs[-14:]) / 14.0
            atr_pct_14 = (atr_14 / closes[-1]) * 100.0

        realized_vol_20: float | None = None
        if len(returns) >= 20:
            ret_sd = _stdev(returns[-20:])
            if ret_sd is not None:
                realized_vol_20 = ret_sd * math.sqrt(252.0) * 100.0

        bb_width_20: float | None = None
        if len(closes) >= 20:
            win = closes[-20:]
            mean_20 = sum(win) / 20.0
            sd_20 = _stdev(win)
            if mean_20 > 0 and sd_20 is not None:
                bb_width_20 = ((4.0 * sd_20) / mean_20) * 100.0

        by_ticker[ticker] = {
            "atr_pct_14": round(atr_pct_14, 2) if atr_pct_14 is not None else None,
            "realized_vol_20": round(realized_vol_20, 2) if realized_vol_20 is not None else None,
            "bb_width_20": round(bb_width_20, 2) if bb_width_20 is not None else None,
        }

    vix_rows = conn.execute(
        """
        SELECT CAST(close AS REAL)
        FROM price_daily
        WHERE ticker = '^VIX' AND close IS NOT NULL
        ORDER BY date DESC
        LIMIT 252
        """
    ).fetchall()
    vix_vals = []
    for r in vix_rows:
        try:
            v = float(r[0])
        except (TypeError, ValueError):
            continue
        if v > 0:
            vix_vals.append(v)
    if vix_vals:
        vix_now = vix_vals[0]
        rank_le = sum(1 for v in vix_vals if v <= vix_now)
        vix_pctile = (rank_le / len(vix_vals)) * 100.0
        market_ctx = {
            "vix_close": round(vix_now, 2),
            "vix_percentile_1y": round(vix_pctile, 1),
            "vix_points": len(vix_vals),
        }

    return by_ticker, market_ctx


def _fetch_symbol_ohlcv(
    conn: sqlite3.Connection, ticker: str, limit: int = 120
) -> list[dict[str, float | str]]:
    rows = conn.execute(
        """
        SELECT date, CAST(open AS REAL), CAST(high AS REAL), CAST(low AS REAL), CAST(close AS REAL), CAST(volume AS REAL)
        FROM price_daily
        WHERE ticker = ? AND close IS NOT NULL
        ORDER BY date DESC
        LIMIT ?
        """,
        (ticker, int(max(10, limit))),
    ).fetchall()
    out: list[dict[str, float | str]] = []
    for r in reversed(rows):
        try:
            close_v = float(r[4])
        except (TypeError, ValueError):
            continue
        if close_v <= 0:
            continue
        try:
            open_v = float(r[1])
            high_v = float(r[2])
            low_v = float(r[3])
            vol_v = float(r[5] or 0.0)
        except (TypeError, ValueError):
            continue
        out.append(
            {
                "date": str(r[0]),
                "open": open_v,
                "high": high_v,
                "low": low_v,
                "close": close_v,
                "volume": vol_v,
            }
        )
    return out


def _percentile_rank(values: list[float], current: float | None) -> float | None:
    if current is None or not values:
        return None
    rank_le = sum(1 for v in values if v <= current)
    return (rank_le / len(values)) * 100.0


def _build_regime_llm_commentary(regime: dict[str, Any]) -> dict[str, Any]:
    safe_regime = regime if isinstance(regime, dict) else {}
    vix = safe_regime.get("vix") if isinstance(safe_regime.get("vix"), dict) else {}
    health = safe_regime.get("health") if isinstance(safe_regime.get("health"), dict) else {}
    summary = str(safe_regime.get("summary") or "").strip() or "Regime context is unavailable for this snapshot."
    health_state = str(health.get("state") or "neutral").strip().lower()
    warnings = [str(w).strip() for w in (health.get("warnings") or []) if str(w).strip()]
    risk_note = warnings[0] if warnings else "context_only_signal"
    asof_date = str(safe_regime.get("asof_date") or "").strip() or None
    setup_score = health.get("score")
    try:
        setup_score = float(setup_score) if setup_score is not None else None
    except (TypeError, ValueError):
        setup_score = None
    if isinstance(setup_score, (int, float)):
        if setup_score >= 85:
            setup_tier = "A"
        elif setup_score >= 70:
            setup_tier = "B"
        elif setup_score >= 55:
            setup_tier = "C"
        else:
            setup_tier = "D"
    else:
        setup_tier = "C"

    if health_state == "risk_on":
        action = "Risk backdrop is supportive, but only act on confirmed setups with defined invalidation levels."
        signal_bias = "bullish"
    elif health_state == "risk_off":
        action = "Risk backdrop is defensive. Reduce gross exposure and wait for stronger confirmation before adding risk."
        signal_bias = "bearish"
    else:
        action = "Backdrop is mixed. Keep size moderate and prioritize cleaner structures with confirmation."
        signal_bias = "neutral"

    row: dict[str, Any] = {
        "ticker": "^VIX",
        "asof": asof_date,
        "score": setup_score,
        "setup_tier": setup_tier,
        "setup_family": "market_regime",
        "signal_bias": signal_bias,
        "trend_state": vix.get("ma_state"),
        "breakout_state": vix.get("breakout_state"),
        "structure_state": vix.get("compression_state"),
        "observation": summary,
        "action": action,
        "risk_note": risk_note,
    }
    source = "rule"
    try:
        llm_overrides = maybe_rewrite_setup_copy(row, source="regime_context")
    except Exception:
        llm_overrides = {}
    if llm_overrides:
        row.update(llm_overrides)
        source = "llm"
    return {
        "source": source,
        "observation": str(row.get("observation") or summary),
        "action": str(row.get("action") or action),
        "risk_note": str(row.get("risk_note") or risk_note or "context_only_signal"),
    }


def _build_regime_context(conn: sqlite3.Connection) -> dict[str, Any]:
    regime: dict[str, Any] = {
        "context_only": True,
        "asof_date": None,
        "summary": "",
        "vix": {},
        "ma_matrix": [],
        "comparison": {},
        "participation": [],
        "overall": {},
        "health": {},
        "timeframes": [],
        "levels": [],
    }
    if not table_exists(conn, "price_daily"):
        return regime

    # --- VIX structure context ---
    vix_rows = _fetch_symbol_ohlcv(conn, "^VIX", limit=280)
    vix_latest = None
    if len(vix_rows) >= 55:
        closes = [float(r["close"]) for r in vix_rows]
        latest = closes[-1]
        prev = closes[-2]
        vix_latest = latest
        ma20 = sum(closes[-20:]) / 20.0
        ma50 = sum(closes[-50:]) / 50.0
        ma100 = sum(closes[-100:]) / 100.0 if len(closes) >= 100 else None
        pct_vs_ma20 = ((latest / ma20) - 1.0) * 100.0 if ma20 > 0 else None
        pct_vs_ma50 = ((latest / ma50) - 1.0) * 100.0 if ma50 > 0 else None
        pct_vs_ma100 = ((latest / ma100) - 1.0) * 100.0 if isinstance(ma100, (int, float)) and ma100 > 0 else None

        if latest >= ma20 and latest >= ma50:
            ma_state = "above_ma20_ma50"
        elif latest >= ma20 and latest < ma50:
            ma_state = "above_ma20_below_ma50"
        elif latest < ma20 and latest >= ma50:
            ma_state = "below_ma20_above_ma50"
        else:
            ma_state = "below_ma20_ma50"

        ma_cross_state = "ma20_above_ma50" if ma20 > ma50 else "ma20_below_ma50"
        ma20_vs_ma50 = ((ma20 / ma50) - 1.0) * 100.0 if ma50 > 0 else None
        ma50_vs_ma100 = ((ma50 / ma100) - 1.0) * 100.0 if isinstance(ma100, (int, float)) and ma100 > 0 else None

        bb_width_series: list[float] = []
        for idx in range(19, len(closes)):
            win = closes[idx - 19 : idx + 1]
            mean_20 = sum(win) / 20.0
            sd_20 = _stdev(win)
            if mean_20 > 0 and sd_20 is not None:
                bb_width_series.append(((4.0 * sd_20) / mean_20) * 100.0)
        bb_width_20 = bb_width_series[-1] if bb_width_series else None
        bb_width_pctile = _percentile_rank(bb_width_series, bb_width_20)
        if isinstance(bb_width_pctile, (int, float)):
            if bb_width_pctile <= 30.0:
                compression_state = "compression"
            elif bb_width_pctile >= 70.0:
                compression_state = "expansion"
            else:
                compression_state = "normal"
        else:
            compression_state = "normal"

        # 10-day range break context (excluding latest bar for reference range).
        prev_window = closes[-11:-1] if len(closes) >= 11 else closes[:-1]
        breakout_state = "inside_range"
        if prev_window:
            hi_prev = max(prev_window)
            lo_prev = min(prev_window)
            if latest >= (hi_prev * 1.002):
                breakout_state = "range_breakout_up"
            elif latest <= (lo_prev * 0.998):
                breakout_state = "range_breakdown_down"

        if ma_state == "above_ma20_ma50" and breakout_state == "range_breakout_up":
            risk_state = "risk_off_pressure_rising"
        elif ma_state == "below_ma20_ma50" and breakout_state == "range_breakdown_down":
            risk_state = "risk_on_relief"
        elif compression_state == "compression":
            risk_state = "coiled_regime"
        else:
            risk_state = "mixed_regime"

        # Calculate term structure with fallback (Requirements 9.1-9.6)
        term_structure = calculate_term_structure(conn)
        
        # Extract values for backward compatibility
        vix3m_latest = term_structure.vix_3m or term_structure.vix_6m
        term_structure_ratio = None
        term_structure_state = "unavailable"
        
        if term_structure.source != "unavailable":
            if term_structure.vix_3m:
                term_structure_ratio = term_structure.vix_3m / latest
            elif term_structure.vix_6m:
                term_structure_ratio = term_structure.vix_6m / latest
            
            if term_structure_ratio:
                if term_structure_ratio >= 1.03:
                    term_structure_state = "contango"
                elif term_structure_ratio <= 0.97:
                    term_structure_state = "backwardation"
                else:
                    term_structure_state = "flat"

        # Multi-timeframe VIX structure using rolling windows on daily bars.
        tf_rows: list[dict[str, Any]] = []
        for label, bars in (("1W", 5), ("2W", 10), ("1M", 21), ("3M", 63)):
            if len(closes) < (bars + 1):
                continue
            start = closes[-(bars + 1)]
            if start <= 0:
                continue
            change_pct = ((latest / start) - 1.0) * 100.0
            win = closes[-bars:]
            hi = max(win)
            lo = min(win)
            range_pos = ((latest - lo) / max(hi - lo, 1e-9)) * 100.0
            if change_pct >= 8.0:
                structure = "vol_expansion"
            elif change_pct <= -8.0:
                structure = "vol_compression"
            else:
                structure = "rangebound"
            if range_pos >= 80.0:
                location = "near_range_high"
            elif range_pos <= 20.0:
                location = "near_range_low"
            else:
                location = "mid_range"
            tf_rows.append(
                {
                    "timeframe": label,
                    "lookback_days": bars,
                    "change_pct": round(change_pct, 2),
                    "range_high": round(hi, 2),
                    "range_low": round(lo, 2),
                    "range_position_pct": round(range_pos, 1),
                    "structure": structure,
                    "location": location,
                }
            )
        regime["timeframes"] = tf_rows

        # VIX support/resistance using the same level engine as chart mode.
        levels_out: list[dict[str, Any]] = []
        level_source = "shared_level_engine"
        try:
            model = pd.DataFrame(vix_rows).copy()
            if len(model) >= REPORT_FEATURE_CFG.min_bars:
                model = add_basic_features(model, REPORT_FEATURE_CFG)
                model = compute_pivots(model, REPORT_FEATURE_CFG)
                levels_raw = build_levels_from_pivots(model, REPORT_LEVEL_CFG)
                levels = select_target_levels(levels_raw, float(latest), REPORT_LEVEL_CFG)
                levels = add_fallback_levels(model, levels, float(latest), REPORT_LEVEL_CFG)
                if levels is not None and not levels.empty:
                    level_pool = levels.copy()

                    def _ordered_side(side: str) -> pd.DataFrame:
                        pool = level_pool[level_pool["type"] == side].copy()
                        if pool.empty:
                            return pool
                        if side == "support":
                            near = pool[pool["level"] <= latest].sort_values("level", ascending=False)
                            far = pool[pool["level"] > latest].sort_values("level", ascending=True)
                        else:
                            near = pool[pool["level"] >= latest].sort_values("level", ascending=True)
                            far = pool[pool["level"] < latest].sort_values("level", ascending=False)
                        return pd.concat([near, far], ignore_index=True).head(3)

                    selected = pd.concat(
                        [_ordered_side("support"), _ordered_side("resistance")],
                        ignore_index=True,
                    )
                    for _, lv in selected.iterrows():
                        level_v = float(lv.get("level") or 0.0)
                        if level_v <= 0:
                            continue
                        side = str(lv.get("type") or "")
                        if side == "support":
                            dist_pct = ((latest - level_v) / level_v) * 100.0
                        else:
                            dist_pct = ((level_v - latest) / level_v) * 100.0
                        levels_out.append(
                            {
                                "type": side,
                                "level": round(level_v, 2),
                                "zone_low": round(float(lv.get("zone_low") or level_v), 2),
                                "zone_high": round(float(lv.get("zone_high") or level_v), 2),
                                "tier": str(lv.get("tier") or ""),
                                "touches": int(lv.get("touches") or 0),
                                "distance_pct": round(dist_pct, 2),
                                "last_touch_date": str(lv.get("last_touch_date") or ""),
                                "source": str(lv.get("source") or "pivot_cluster"),  # Requirement 10.1, 10.3
                            }
                        )
        except Exception:
            levels_out = []
        if not levels_out:
            level_source = "rolling_window_fallback"
            closes_for_levels = closes
            dates_for_levels = [str(r.get("date") or "") for r in vix_rows]
            candidates: list[tuple[str, float, str]] = []
            for bars, tier in ((21, "primary"), (63, "secondary"), (126, "secondary")):
                if len(closes_for_levels) < bars:
                    continue
                win = closes_for_levels[-bars:]
                candidates.append(("support", float(min(win)), tier))
                candidates.append(("resistance", float(max(win)), tier))
            dedup: dict[tuple[str, int], tuple[str, float, str]] = {}
            for side, lv, tier in candidates:
                key = (side, int(round(lv * 100)))
                dedup[key] = (side, lv, tier)

            def _ordered_fallback(side: str) -> list[tuple[str, float, str]]:
                pool = [item for item in dedup.values() if item[0] == side]
                if side == "support":
                    near = sorted([p for p in pool if p[1] <= latest], key=lambda x: x[1], reverse=True)
                    far = sorted([p for p in pool if p[1] > latest], key=lambda x: x[1])
                else:
                    near = sorted([p for p in pool if p[1] >= latest], key=lambda x: x[1])
                    far = sorted([p for p in pool if p[1] < latest], key=lambda x: x[1], reverse=True)
                return (near + far)[:3]

            for side, level_v, tier in _ordered_fallback("support") + _ordered_fallback("resistance"):
                if level_v <= 0:
                    continue
                zone_pad = max(level_v * 0.003, 0.05)
                zone_low = level_v - zone_pad
                zone_high = level_v + zone_pad
                if side == "support":
                    dist_pct = ((latest - level_v) / level_v) * 100.0
                else:
                    dist_pct = ((level_v - latest) / level_v) * 100.0
                last_touch_idx = None
                for idx in range(len(closes_for_levels) - 1, -1, -1):
                    cv = closes_for_levels[idx]
                    if abs(cv - level_v) / max(level_v, 1e-9) <= 0.004:
                        last_touch_idx = idx
                        break
                levels_out.append(
                    {
                        "type": side,
                        "level": round(level_v, 2),
                        "zone_low": round(zone_low, 2),
                        "zone_high": round(zone_high, 2),
                        "tier": tier,
                        "touches": 1 if last_touch_idx is not None else 0,
                        "distance_pct": round(dist_pct, 2),
                        "last_touch_date": dates_for_levels[last_touch_idx] if last_touch_idx is not None else "",
                        "source": "fallback",  # Requirement 10.1, 10.3
                    }
                )
        regime["levels"] = levels_out

        # Detect VIX trap/reclaim patterns
        trap_reclaim_patterns: list[dict[str, Any]] = []
        try:
            vix_df = pd.DataFrame(vix_rows)
            if not vix_df.empty and levels_out:
                patterns = detect_vix_trap_reclaim_patterns(
                    vix_df, levels_out, VIXTrapReclaimConfig()
                )
                for pattern in patterns:
                    trap_reclaim_patterns.append(
                        {
                            "pattern_type": pattern.pattern_type,
                            "date": pattern.date,
                            "price": round(pattern.price, 2),
                            "level": round(pattern.level, 2),
                            "confidence": round(pattern.confidence, 2),
                            "explanation": pattern.explanation,
                            "bars_to_reversal": pattern.bars_to_reversal,
                        }
                    )
        except Exception as e:
            # Log but don't fail if pattern detection has issues
            pass

        regime["trap_reclaim_patterns"] = trap_reclaim_patterns

        # Calculate VIX percentile (Requirement 11.1)
        vix_percentile = calculate_vix_percentile(conn, window_days=252)
        vix_percentile_color = get_percentile_color(vix_percentile)
        vix_percentile_warning = should_show_volatility_warning(vix_percentile)

        regime["asof_date"] = str(vix_rows[-1]["date"])
        regime["vix"] = {
            "ticker": "^VIX",
            "close": round(latest, 2),
            "change_pct_1d": round(((latest / prev) - 1.0) * 100.0, 2) if prev > 0 else None,
            "ma20": round(ma20, 2),
            "ma50": round(ma50, 2),
            "ma100": round(ma100, 2) if isinstance(ma100, (int, float)) else None,
            "pct_vs_ma20": round(pct_vs_ma20, 2) if pct_vs_ma20 is not None else None,
            "pct_vs_ma50": round(pct_vs_ma50, 2) if pct_vs_ma50 is not None else None,
            "pct_vs_ma100": round(pct_vs_ma100, 2) if pct_vs_ma100 is not None else None,
            "ma_state": ma_state,
            "ma_cross_state": ma_cross_state,
            "bb_width_20": round(bb_width_20, 2) if isinstance(bb_width_20, (int, float)) else None,
            "bb_width_pctile_lookback": round(bb_width_pctile, 1)
            if isinstance(bb_width_pctile, (int, float))
            else None,
            "compression_state": compression_state,
            "breakout_state": breakout_state,
            "risk_state": risk_state,
            "term_structure_ratio": round(term_structure_ratio, 3)
            if isinstance(term_structure_ratio, (int, float))
            else None,
            "term_structure_state": term_structure_state,
            "term_structure_source": term_structure.source,  # Requirement 9.3
            "term_structure_timestamp": term_structure.timestamp.isoformat(),  # Requirement 9.6
            "vix3m_close": round(vix3m_latest, 2) if isinstance(vix3m_latest, (int, float)) else None,
            "level_source": level_source,
            "percentile": round(vix_percentile, 1) if vix_percentile is not None else None,  # Requirement 11.6
            "percentile_color": vix_percentile_color,  # Requirement 11.3
            "percentile_warning": vix_percentile_warning,  # Requirement 11.5
        }
        regime["ma_matrix"] = [
            {
                "metric": "Close vs MA20",
                "value_pct": round(pct_vs_ma20, 2) if isinstance(pct_vs_ma20, (int, float)) else None,
                "state": "above_ma20" if isinstance(pct_vs_ma20, (int, float)) and pct_vs_ma20 >= 0 else "below_ma20",
                "risk_read": "risk_off_pressure" if isinstance(pct_vs_ma20, (int, float)) and pct_vs_ma20 >= 0 else "risk_on_relief",
            },
            {
                "metric": "Close vs MA50",
                "value_pct": round(pct_vs_ma50, 2) if isinstance(pct_vs_ma50, (int, float)) else None,
                "state": "above_ma50" if isinstance(pct_vs_ma50, (int, float)) and pct_vs_ma50 >= 0 else "below_ma50",
                "risk_read": "risk_off_pressure" if isinstance(pct_vs_ma50, (int, float)) and pct_vs_ma50 >= 0 else "risk_on_relief",
            },
            {
                "metric": "Close vs MA100",
                "value_pct": round(pct_vs_ma100, 2) if isinstance(pct_vs_ma100, (int, float)) else None,
                "state": "above_ma100" if isinstance(pct_vs_ma100, (int, float)) and pct_vs_ma100 >= 0 else "below_ma100",
                "risk_read": "risk_off_pressure" if isinstance(pct_vs_ma100, (int, float)) and pct_vs_ma100 >= 0 else "risk_on_relief",
            },
            {
                "metric": "MA20 vs MA50",
                "value_pct": round(ma20_vs_ma50, 2) if isinstance(ma20_vs_ma50, (int, float)) else None,
                "state": "ma20_above_ma50" if ma20 > ma50 else "ma20_below_ma50",
                "risk_read": "short_term_stress_up" if ma20 > ma50 else "short_term_stress_down",
            },
            {
                "metric": "MA50 vs MA100",
                "value_pct": round(ma50_vs_ma100, 2) if isinstance(ma50_vs_ma100, (int, float)) else None,
                "state": "ma50_above_ma100"
                if isinstance(ma50_vs_ma100, (int, float)) and ma50_vs_ma100 >= 0
                else "ma50_below_ma100",
                "risk_read": "intermediate_stress_up"
                if isinstance(ma50_vs_ma100, (int, float)) and ma50_vs_ma100 >= 0
                else "intermediate_stress_down",
            },
        ]

        # Benchmark comparison window for regime context: VIX + SPX/DJI/NDX.
        benchmark_candidates = {
            "spx": ("^GSPC", ".SPX", "SPX"),
            "dji": ("^DJI", ".DJI", "DJI"),
            "ndx": ("^NDX", ".NDX", "NDX", "QQQ"),
        }
        benchmark_maps: dict[str, dict[str, float]] = {}
        benchmark_symbol_used: dict[str, str] = {"vix": "^VIX"}
        for key, symbols in benchmark_candidates.items():
            rows_best: list[dict[str, float | str]] = []
            symbol_best = None
            for symbol in symbols:
                rows = _fetch_symbol_ohlcv(conn, symbol, limit=160)
                if len(rows) > len(rows_best):
                    rows_best = rows
                    symbol_best = symbol
            if rows_best:
                benchmark_maps[key] = {str(r["date"]): float(r["close"]) for r in rows_best if float(r["close"]) > 0}
                benchmark_symbol_used[key] = str(symbol_best or "")

        vix_map = {str(r["date"]): float(r["close"]) for r in vix_rows if float(r["close"]) > 0}
        date_pool = sorted(vix_map.keys())[-90:]
        comparison_rows: list[dict[str, Any]] = []
        if date_pool:
            bases: dict[str, float] = {}
            for name in ("vix", "spx", "dji", "ndx"):
                source_map = vix_map if name == "vix" else benchmark_maps.get(name, {})
                base_v = next((source_map.get(d) for d in date_pool if isinstance(source_map.get(d), (int, float))), None)
                if isinstance(base_v, (int, float)) and base_v > 0:
                    bases[name] = float(base_v)
            for d in date_pool:
                row_cmp: dict[str, Any] = {"date": d}
                for name in ("vix", "spx", "dji", "ndx"):
                    source_map = vix_map if name == "vix" else benchmark_maps.get(name, {})
                    raw_v = source_map.get(d)
                    row_cmp[f"{name}_close"] = round(float(raw_v), 2) if isinstance(raw_v, (int, float)) else None
                    base_v = bases.get(name)
                    row_cmp[f"{name}_idx"] = (
                        round((float(raw_v) / base_v) * 100.0, 2)
                        if isinstance(raw_v, (int, float)) and isinstance(base_v, (int, float)) and base_v > 0
                        else None
                    )
                comparison_rows.append(row_cmp)
        regime["comparison"] = {
            "window_days": len(comparison_rows),
            "symbols": benchmark_symbol_used,
            "series": comparison_rows,
        }

    # --- Participation / distribution context for core market proxies ---
    participation_rows: list[dict[str, Any]] = []
    for symbol in ("SPY", "QQQ"):
        sym_rows = _fetch_symbol_ohlcv(conn, symbol, limit=45)
        window = 20
        if len(sym_rows) < window + 1:
            continue
        closes = [float(r["close"]) for r in sym_rows[-(window + 1) :]]
        vols = [float(r["volume"]) for r in sym_rows[-(window + 1) :]]
        up_days = 0
        down_days = 0
        up_vol = 0.0
        down_vol = 0.0
        day_changes: list[tuple[float, float]] = []
        for idx in range(1, len(closes)):
            prev_c = closes[idx - 1]
            cur_c = closes[idx]
            vol = vols[idx]
            if prev_c <= 0:
                continue
            ret = ((cur_c / prev_c) - 1.0) * 100.0
            day_changes.append((ret, vol))
            if ret > 0:
                up_days += 1
                up_vol += vol
            elif ret < 0:
                down_days += 1
                down_vol += vol

        total_signed_vol = up_vol + down_vol
        up_share = (up_vol / total_signed_vol) * 100.0 if total_signed_vol > 0 else None
        down_share = (down_vol / total_signed_vol) * 100.0 if total_signed_vol > 0 else None
        vol_ratio = (up_vol / down_vol) if down_vol > 0 else None

        avg_vol = sum(v for _, v in day_changes) / len(day_changes) if day_changes else 0.0
        heavy_up_days = sum(1 for ret, vol in day_changes if ret > 0 and vol >= avg_vol)
        heavy_down_days = sum(1 for ret, vol in day_changes if ret < 0 and vol >= avg_vol)

        if isinstance(vol_ratio, (int, float)) and vol_ratio >= 1.15 and heavy_up_days >= heavy_down_days:
            bias = "accumulation"
        elif isinstance(vol_ratio, (int, float)) and vol_ratio <= 0.85 and heavy_down_days >= heavy_up_days:
            bias = "distribution"
        else:
            bias = "balanced"

        participation_rows.append(
            {
                "symbol": symbol,
                "window_days": window,
                "up_days": up_days,
                "down_days": down_days,
                "up_volume_share_pct": round(up_share, 2) if up_share is not None else None,
                "down_volume_share_pct": round(down_share, 2) if down_share is not None else None,
                "up_down_volume_ratio": round(vol_ratio, 3) if isinstance(vol_ratio, (int, float)) else None,
                "heavy_up_days": int(heavy_up_days),
                "heavy_down_days": int(heavy_down_days),
                "bias": bias,
            }
        )

    regime["participation"] = participation_rows
    if participation_rows:
        accum = sum(1 for row in participation_rows if row.get("bias") == "accumulation")
        dist = sum(1 for row in participation_rows if row.get("bias") == "distribution")
        avg_up_share = [
            float(row.get("up_volume_share_pct"))
            for row in participation_rows
            if isinstance(row.get("up_volume_share_pct"), (int, float))
        ]
        if accum > dist:
            overall_participation = "accumulation_bias"
        elif dist > accum:
            overall_participation = "distribution_bias"
        else:
            overall_participation = "balanced_bias"
        regime["overall"] = {
            "participation_bias": overall_participation,
            "accumulation_symbols": accum,
            "distribution_symbols": dist,
            "total_symbols": len(participation_rows),
            "avg_up_volume_share_pct": round(sum(avg_up_share) / len(avg_up_share), 2) if avg_up_share else None,
        }
    else:
        regime["overall"] = {"participation_bias": "unknown"}

    # --- Overall market health score (0-100, context-only) ---
    vix_info = regime.get("vix", {}) if isinstance(regime.get("vix"), dict) else {}
    participation_bias = str((regime.get("overall") or {}).get("participation_bias") or "unknown")
    health_score = 50.0
    drivers: list[str] = []
    warnings: list[str] = []

    # VIX percentile as primary factor (Requirement 11.4)
    vix_percentile = vix_info.get("percentile")
    if isinstance(vix_percentile, (int, float)):
        if vix_percentile < 30:
            # Low volatility percentile = risk-on
            health_score += 20.0
            drivers.append(f"VIX percentile at {vix_percentile:.1f}% (historically low volatility)")
        elif vix_percentile < 50:
            # Below median = moderately risk-on
            health_score += 10.0
            drivers.append(f"VIX percentile at {vix_percentile:.1f}% (below median volatility)")
        elif vix_percentile < 70:
            # Above median = moderately risk-off
            health_score -= 10.0
            warnings.append(f"VIX percentile at {vix_percentile:.1f}% (above median volatility)")
        else:
            # High volatility percentile = risk-off
            health_score -= 20.0
            warnings.append(f"VIX percentile at {vix_percentile:.1f}% (historically high volatility)")
        
        # Additional warning for elevated volatility (Requirement 11.5)
        if vix_info.get("percentile_warning"):
            warnings.append("⚠️ ELEVATED VOLATILITY: VIX percentile exceeds 80%")

    ma_state = str(vix_info.get("ma_state") or "")
    if ma_state == "below_ma20_ma50":
        health_score += 15.0
        drivers.append("VIX below MA20 and MA50 (risk-on backdrop)")
    elif ma_state == "above_ma20_ma50":
        health_score -= 15.0
        warnings.append("VIX above MA20 and MA50 (risk-off pressure)")
    elif ma_state == "below_ma20_above_ma50":
        health_score += 6.0
        drivers.append("VIX below MA20 while still above MA50")
    elif ma_state == "above_ma20_below_ma50":
        health_score -= 6.0
        warnings.append("VIX above MA20 while below MA50 (short-term stress)")

    breakout_state = str(vix_info.get("breakout_state") or "")
    if breakout_state == "range_breakdown_down":
        health_score += 10.0
        drivers.append("VIX broke down from its recent range")
    elif breakout_state == "range_breakout_up":
        health_score -= 10.0
        warnings.append("VIX broke up from its recent range")

    compression_state = str(vix_info.get("compression_state") or "")
    if compression_state == "compression":
        health_score += 4.0
        drivers.append("VIX volatility is compressed (coiled regime)")
    elif compression_state == "expansion":
        health_score -= 4.0
        warnings.append("VIX volatility is expanding")

    change_1d = vix_info.get("change_pct_1d")
    if isinstance(change_1d, (int, float)):
        if change_1d <= -2.0:
            health_score += 6.0
            drivers.append("VIX fell sharply vs prior session")
        elif change_1d >= 2.0:
            health_score -= 6.0
            warnings.append("VIX rose sharply vs prior session")

    term_state = str(vix_info.get("term_structure_state") or "")
    if term_state == "contango":
        health_score += 8.0
        drivers.append("VIX term structure is in contango")
    elif term_state == "backwardation":
        health_score -= 8.0
        warnings.append("VIX term structure is in backwardation")

    if participation_bias == "accumulation_bias":
        health_score += 12.0
        drivers.append("SPY/QQQ volume profile shows accumulation")
    elif participation_bias == "distribution_bias":
        health_score -= 12.0
        warnings.append("SPY/QQQ volume profile shows distribution")

    health_score = _clamp(health_score, 0.0, 100.0)
    if health_score >= 65.0:
        health_state = "risk_on"
    elif health_score <= 35.0:
        health_state = "risk_off"
    else:
        health_state = "neutral"

    regime["health"] = {
        "score": round(health_score, 1),
        "state": health_state,
        "confidence": round(abs(health_score - 50.0) * 2.0, 1),  # distance from neutral center
        "drivers": drivers[:6],
        "warnings": warnings[:6],
    }

    if isinstance(vix_latest, (int, float)):
        summary_prefix = (
            f"Market health {regime['health']['score']}/100 ({health_state.replace('_', ' ')})"
        )
        vix_risk = str(vix_info.get("risk_state") or "mixed_regime").replace("_", " ")
        participation_state = participation_bias.replace("_", " ")
        regime["summary"] = (
            f"{summary_prefix}. VIX regime: {vix_risk}; participation: {participation_state}. "
            "Use this as backdrop for sizing and confirmation, not as a standalone trigger."
        )
    else:
        regime["summary"] = "Regime context unavailable (insufficient VIX history)."
    regime["llm_commentary"] = _build_regime_llm_commentary(regime)
    return regime


def _fetch_technical_context(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Build report interpretation context using the same level engine as the dashboard."""
    by_ticker: dict[str, dict[str, Any]] = {}
    if not table_exists(conn, "price_daily"):
        return by_ticker

    rows = conn.execute(
        """
        SELECT ticker, date, CAST(open AS REAL), CAST(high AS REAL), CAST(low AS REAL), CAST(close AS REAL), CAST(volume AS REAL)
        FROM (
            SELECT ticker, date, open, high, low, close, volume,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM price_daily
        )
        WHERE rn <= 320
        ORDER BY ticker, date
        """
    ).fetchall()
    bucket: dict[str, list[tuple[str, float, float, float, float, float]]] = defaultdict(list)
    for r in rows:
        ticker = str(r[0] or "").upper().strip()
        if not ticker:
            continue
        try:
            open_v = float(r[2])
            high_v = float(r[3])
            low_v = float(r[4])
            close_v = float(r[5])
            volume_v = float(r[6] or 0.0)
        except (TypeError, ValueError):
            continue
        if min(open_v, high_v, low_v, close_v) <= 0:
            continue
        bucket[ticker].append((str(r[1]), open_v, high_v, low_v, close_v, volume_v))

    for ticker, bars in bucket.items():
        if len(bars) < 60:
            continue
        closes = [b[4] for b in bars]
        highs = [b[2] for b in bars]
        lows = [b[3] for b in bars]
        volumes = [b[5] for b in bars]
        close_now = closes[-1]
        prev_close = closes[-2] if len(closes) >= 2 else close_now
        high_now = highs[-1]
        low_now = lows[-1]
        ma20 = sum(closes[-20:]) / 20.0 if len(closes) >= 20 else None
        ma50 = sum(closes[-50:]) / 50.0 if len(closes) >= 50 else None
        ma100 = sum(closes[-100:]) / 100.0 if len(closes) >= 100 else None
        ma200 = sum(closes[-200:]) / 200.0 if len(closes) >= 200 else None
        prev_ma20 = sum(closes[-21:-1]) / 20.0 if len(closes) >= 21 else None
        prev_ma50 = sum(closes[-51:-1]) / 50.0 if len(closes) >= 51 else None
        prev_ma200 = sum(closes[-201:-1]) / 200.0 if len(closes) >= 201 else None
        recent_high_20 = max(highs[-20:]) if len(highs) >= 20 else None
        recent_low_20 = min(lows[-20:]) if len(lows) >= 20 else None
        recent_high_prev_20 = max(highs[-21:-1]) if len(highs) >= 21 else None
        recent_low_prev_20 = min(lows[-21:-1]) if len(lows) >= 21 else None
        avg_volume_20 = (sum(volumes[-20:]) / 20.0) if len(volumes) >= 20 else None
        volume_ratio_20 = (volumes[-1] / avg_volume_20) if avg_volume_20 and avg_volume_20 > 0 else None
        recent_range_pct_10 = (
            ((max(highs[-10:]) - min(lows[-10:])) / close_now) * 100.0
            if len(highs) >= 10 and close_now > 0
            else None
        )
        recent_range_pct_20 = (
            ((max(highs[-20:]) - min(lows[-20:])) / close_now) * 100.0
            if len(highs) >= 20 and close_now > 0
            else None
        )

        pct_vs_ma20 = ((close_now / ma20) - 1.0) * 100.0 if ma20 and ma20 > 0 else None
        pct_vs_ma50 = ((close_now / ma50) - 1.0) * 100.0 if ma50 and ma50 > 0 else None
        pct_from_20d_high = (
            ((recent_high_20 - close_now) / recent_high_20) * 100.0
            if recent_high_20 and recent_high_20 > 0
            else None
        )
        pct_from_20d_low = (
            ((close_now - recent_low_20) / recent_low_20) * 100.0
            if recent_low_20 and recent_low_20 > 0
            else None
        )

        trend_state = "mixed"
        if (
            ma20 is not None
            and ma50 is not None
            and close_now > ma20 > ma50
        ):
            trend_state = "uptrend"
        elif (
            ma20 is not None
            and ma50 is not None
            and close_now < ma20 < ma50
        ):
            trend_state = "downtrend"

        ma_signal = None
        if (
            prev_ma20 is not None
            and prev_ma50 is not None
            and ma20 is not None
            and ma50 is not None
        ):
            if prev_ma20 >= prev_ma50 and ma20 < ma50:
                ma_signal = "bearish_20_50_cross"
            elif prev_ma20 <= prev_ma50 and ma20 > ma50:
                ma_signal = "bullish_20_50_cross"
            elif ma20 < ma50:
                ma_signal = "20_below_50"
            elif ma20 > ma50:
                ma_signal = "20_above_50"

        ma_major_signal = None
        if (
            prev_ma50 is not None
            and prev_ma200 is not None
            and ma50 is not None
            and ma200 is not None
        ):
            if prev_ma50 >= prev_ma200 and ma50 < ma200:
                ma_major_signal = "death_cross"
            elif prev_ma50 <= prev_ma200 and ma50 > ma200:
                ma_major_signal = "golden_cross"
            elif ma50 < ma200:
                ma_major_signal = "50_below_200"
            elif ma50 > ma200:
                ma_major_signal = "50_above_200"

        ma_reclaim_state = None
        if prev_ma20 is not None and ma20 is not None:
            if prev_close <= prev_ma20 and close_now > ma20:
                ma_reclaim_state = "reclaimed_ma20"
            elif prev_close >= prev_ma20 and close_now < ma20:
                ma_reclaim_state = "lost_ma20"
        if prev_ma50 is not None and ma50 is not None:
            if prev_close <= prev_ma50 and close_now > ma50:
                ma_reclaim_state = "reclaimed_ma50"
            elif prev_close >= prev_ma50 and close_now < ma50:
                ma_reclaim_state = "lost_ma50"

        recent_gap_state = None
        recent_gap_days = None
        if len(bars) >= 2:
            start_idx = max(1, len(bars) - 4)
            for idx in range(len(bars) - 1, start_idx - 1, -1):
                prev_high = highs[idx - 1]
                prev_low = lows[idx - 1]
                bar_high = highs[idx]
                bar_low = lows[idx]
                if bar_low > prev_high:
                    recent_gap_state = "bull_gap"
                    recent_gap_days = len(bars) - 1 - idx
                    break
                if bar_high < prev_low:
                    recent_gap_state = "bear_gap"
                    recent_gap_days = len(bars) - 1 - idx
                    break

        level_context = "mid_range"
        if isinstance(pct_from_20d_low, (int, float)) and pct_from_20d_low <= 2.5:
            level_context = "at_support"
        elif isinstance(pct_from_20d_high, (int, float)) and pct_from_20d_high <= 2.5:
            level_context = "at_resistance"

        stretch_state = "normal"
        if isinstance(pct_vs_ma20, (int, float)):
            if pct_vs_ma20 >= 8.0:
                stretch_state = "extended_up"
            elif pct_vs_ma20 <= -8.0:
                stretch_state = "extended_down"

        support_level = None
        support_zone_low = None
        support_zone_high = None
        support_tier = None
        support_touches = None
        resistance_level = None
        resistance_zone_low = None
        resistance_zone_high = None
        resistance_tier = None
        resistance_touches = None
        pct_to_support = None
        pct_to_resistance = None
        range_position = None
        breakout_state = "none"
        level_event = "none"
        structure_state = "normal"

        try:
            frame = pd.DataFrame(
                bars,
                columns=["date", "open", "high", "low", "close", "volume"],
            )
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.dropna(subset=["date"]).reset_index(drop=True)
            if len(frame) >= 60:
                model = add_basic_features(frame.copy(), REPORT_FEATURE_CFG)
                model = compute_pivots(model, left=3, right=3)
                levels_raw = build_levels_from_pivots(model, REPORT_LEVEL_CFG)
                levels = select_target_levels(levels_raw, float(close_now), REPORT_LEVEL_CFG)
                levels = add_fallback_levels(model, levels, float(close_now), REPORT_LEVEL_CFG)

                def _pick_level(side: str) -> dict[str, Any] | None:
                    if levels is None or levels.empty:
                        return None
                    pool = levels[levels["type"] == side].copy()
                    if pool.empty:
                        return None
                    if side == "support":
                        preferred = pool[pool["level"] <= close_now].sort_values("level", ascending=False)
                    else:
                        preferred = pool[pool["level"] >= close_now].sort_values("level", ascending=True)
                    if preferred.empty:
                        preferred = pool.sort_values(
                            ["dist", "touches", "recency_score"],
                            ascending=[True, False, False],
                        )
                    return preferred.iloc[0].to_dict() if not preferred.empty else None

                support = _pick_level("support")
                resistance = _pick_level("resistance")
                if support:
                    support_level = round(float(support.get("level") or 0.0), 2)
                    support_zone_low = round(float(support.get("zone_low") or 0.0), 2)
                    support_zone_high = round(float(support.get("zone_high") or 0.0), 2)
                    support_tier = str(support.get("tier") or "")
                    support_touches = int(support.get("touches") or 0)
                    if support_level > 0:
                        pct_to_support = round(((close_now - support_level) / support_level) * 100.0, 2)
                if resistance:
                    resistance_level = round(float(resistance.get("level") or 0.0), 2)
                    resistance_zone_low = round(float(resistance.get("zone_low") or 0.0), 2)
                    resistance_zone_high = round(float(resistance.get("zone_high") or 0.0), 2)
                    resistance_tier = str(resistance.get("tier") or "")
                    resistance_touches = int(resistance.get("touches") or 0)
                    if resistance_level > 0:
                        pct_to_resistance = round(((resistance_level - close_now) / resistance_level) * 100.0, 2)
                if (
                    isinstance(support_level, (int, float))
                    and isinstance(resistance_level, (int, float))
                    and resistance_level > support_level
                ):
                    range_position = round(
                        (close_now - support_level) / max(resistance_level - support_level, 0.01),
                        3,
                    )
                if (
                    isinstance(support_zone_low, (int, float))
                    and close_now < float(support_zone_low)
                ):
                    level_context = "below_support"
                elif (
                    isinstance(resistance_zone_high, (int, float))
                    and close_now > float(resistance_zone_high)
                ):
                    level_context = "above_resistance"
                elif (
                    isinstance(support_zone_high, (int, float))
                    and close_now <= float(support_zone_high)
                ):
                    level_context = "at_support"
                elif (
                    isinstance(resistance_zone_low, (int, float))
                    and close_now >= float(resistance_zone_low)
                ):
                    level_context = "at_resistance"
                elif isinstance(range_position, (int, float)):
                    if float(range_position) <= 0.35:
                        level_context = "closer_support"
                    elif float(range_position) >= 0.65:
                        level_context = "closer_resistance"
                    else:
                        level_context = "mid_range"

                if isinstance(resistance_zone_high, (int, float)):
                    rzh = float(resistance_zone_high)
                    if close_now > rzh:
                        breakout_state = "breakout_up"
                    elif high_now > rzh and close_now <= rzh:
                        breakout_state = "failed_breakout_up"
                if isinstance(support_zone_low, (int, float)):
                    szl = float(support_zone_low)
                    if close_now < szl:
                        breakout_state = "breakout_down"
                    elif low_now < szl and close_now >= szl and breakout_state == "none":
                        breakout_state = "failed_breakdown_down"
                if breakout_state == "breakout_up":
                    level_event = "resistance_breakout"
                elif breakout_state == "breakout_down":
                    level_event = "support_breakdown"
                elif breakout_state == "failed_breakout_up":
                    level_event = "resistance_reject"
                elif breakout_state == "failed_breakdown_down":
                    level_event = "support_reclaim"
        except Exception:
            pass

        if (
            isinstance(recent_range_pct_10, (int, float))
            and isinstance(recent_range_pct_20, (int, float))
            and recent_range_pct_10 <= 7.0
            and recent_range_pct_20 <= 12.0
        ):
            if (
                (isinstance(range_position, (int, float)) and float(range_position) >= 0.58)
                or (isinstance(resistance_touches, int) and resistance_touches >= 2)
            ):
                structure_state = "tight_consolidation_high"
            elif (
                (isinstance(range_position, (int, float)) and float(range_position) <= 0.42)
                or (isinstance(support_touches, int) and support_touches >= 2)
            ):
                structure_state = "tight_consolidation_low"
            else:
                structure_state = "tight_consolidation_mid"

        if isinstance(pct_vs_ma20, (int, float)) and trend_state == "uptrend" and float(pct_vs_ma20) >= 7.5:
            if breakout_state == "breakout_up" or (
                isinstance(pct_from_20d_high, (int, float)) and float(pct_from_20d_high) <= 1.5
            ):
                structure_state = "parabolic_up"
        elif isinstance(pct_vs_ma20, (int, float)) and trend_state == "downtrend" and float(pct_vs_ma20) <= -7.5:
            if breakout_state == "breakout_down" or (
                isinstance(pct_from_20d_low, (int, float)) and float(pct_from_20d_low) <= 1.5
            ):
                structure_state = "parabolic_down"

        by_ticker[ticker] = {
            "close": round(close_now, 2),
            "ma20": round(ma20, 2) if ma20 is not None else None,
            "ma50": round(ma50, 2) if ma50 is not None else None,
            "ma100": round(ma100, 2) if ma100 is not None else None,
            "ma200": round(ma200, 2) if ma200 is not None else None,
            "avg_volume_20": round(avg_volume_20, 2) if avg_volume_20 is not None else None,
            "volume_ratio_20": round(volume_ratio_20, 2) if volume_ratio_20 is not None else None,
            "pct_vs_ma20": round(pct_vs_ma20, 2) if pct_vs_ma20 is not None else None,
            "pct_vs_ma50": round(pct_vs_ma50, 2) if pct_vs_ma50 is not None else None,
            "recent_high_20": round(recent_high_20, 2) if recent_high_20 is not None else None,
            "recent_low_20": round(recent_low_20, 2) if recent_low_20 is not None else None,
            "recent_range_pct_10": round(recent_range_pct_10, 2) if recent_range_pct_10 is not None else None,
            "recent_range_pct_20": round(recent_range_pct_20, 2) if recent_range_pct_20 is not None else None,
            "pct_from_20d_high": round(pct_from_20d_high, 2) if pct_from_20d_high is not None else None,
            "pct_from_20d_low": round(pct_from_20d_low, 2) if pct_from_20d_low is not None else None,
            "trend_state": trend_state,
            "ma_signal": ma_signal,
            "ma_major_signal": ma_major_signal,
            "ma_reclaim_state": ma_reclaim_state,
            "level_context": level_context,
            "stretch_state": stretch_state,
            "breakout_state": breakout_state,
            "level_event": level_event,
            "structure_state": structure_state,
            "recent_gap_state": recent_gap_state,
            "recent_gap_days": recent_gap_days,
            "support_level": support_level,
            "support_zone_low": support_zone_low,
            "support_zone_high": support_zone_high,
            "support_tier": support_tier,
            "support_touches": support_touches,
            "resistance_level": resistance_level,
            "resistance_zone_low": resistance_zone_low,
            "resistance_zone_high": resistance_zone_high,
            "resistance_tier": resistance_tier,
            "resistance_touches": resistance_touches,
            "pct_to_support": pct_to_support,
            "pct_to_resistance": pct_to_resistance,
            "range_position": range_position,
        }
    return by_ticker


def _yolo_pattern_bias(pattern: Any) -> str:
    name = str(pattern or "").strip().lower()
    if not name:
        return "neutral"
    if ("bottom" in name) or ("w_bottom" in name):
        return "bullish"
    if ("top" in name) or ("m_head" in name):
        return "bearish"
    if "triangle" in name:
        return "neutral"
    return "neutral"


def _yolo_age_factor(age_days: Any, timeframe: Any) -> float:
    tf = str(timeframe or "").strip().lower()
    if not isinstance(age_days, (int, float)):
        return 0.0
    age = int(max(0, float(age_days)))
    if tf == "weekly":
        if age <= 14:
            return 1.0
        if age <= 35:
            return 0.8
        if age <= 70:
            return 0.45
        if age <= 120:
            return 0.18
        return 0.0
    if age <= 5:
        return 1.0
    if age <= 12:
        return 0.8
    if age <= 25:
        return 0.5
    if age <= 45:
        return 0.2
    return 0.0


def _yolo_recency_label(age_days: Any, timeframe: Any) -> str:
    tf = str(timeframe or "").strip().lower()
    if not isinstance(age_days, (int, float)):
        return "unknown"
    age = int(max(0, float(age_days)))
    if tf == "weekly":
        if age <= 14:
            return "fresh"
        if age <= 35:
            return "recent"
        if age <= 70:
            return "aging"
        return "stale"
    if age <= 5:
        return "fresh"
    if age <= 12:
        return "recent"
    if age <= 25:
        return "aging"
    return "stale"


def _fmt_pct_short(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    num = float(value)
    sign = "+" if num > 0 else ""
    return f"{sign}{num:.1f}%"


def _fundamental_context(discount: Any, peg: Any) -> dict[str, Any]:
    long_points = 0
    short_points = 0
    long_notes: list[str] = []
    short_notes: list[str] = []
    if isinstance(discount, (int, float)):
        d = float(discount)
        if d >= 25.0:
            long_points += 3
            long_notes.append("deep discount")
        elif d >= 12.0:
            long_points += 2
            long_notes.append("discounted")
        elif d >= 5.0:
            long_points += 1
            long_notes.append("small discount")
        elif d <= 0.0:
            short_points += 2
            short_notes.append("no discount")
        elif d <= 3.0:
            short_points += 1
            short_notes.append("thin valuation cushion")
    if isinstance(peg, (int, float)) and float(peg) > 0:
        p = float(peg)
        if p <= 0.8:
            long_points += 3
            long_notes.append("low PEG")
        elif p <= 1.5:
            long_points += 2
            long_notes.append("reasonable PEG")
        elif p <= 2.5:
            long_points += 1
        elif p >= 5.0:
            short_points += 3
            short_notes.append("high PEG")
        elif p >= 3.0:
            short_points += 2
            short_notes.append("rich PEG")
    bias = "neutral"
    if long_points >= short_points + 2:
        bias = "bullish"
    elif short_points >= long_points + 2:
        bias = "bearish"
    return {
        "bias": bias,
        "long_points": long_points,
        "short_points": short_points,
        "long_notes": long_notes,
        "short_notes": short_notes,
    }


def _score_setup_from_confluence(row: dict[str, Any]) -> dict[str, Any]:
    yolo_bias = _yolo_pattern_bias(row.get("yolo_pattern"))
    yolo_age_days = row.get("yolo_age_days")
    yolo_timeframe = str(row.get("yolo_timeframe") or "daily")
    candle_bias = str(row.get("candle_bias") or "neutral")
    trend = str(row.get("trend_state") or "mixed")
    level = str(row.get("level_context") or "mid_range")
    stretch = str(row.get("stretch_state") or "normal")
    breakout_state = str(row.get("breakout_state") or "none")
    level_event = str(row.get("level_event") or "none")
    structure_state = str(row.get("structure_state") or "normal")
    ma_signal = str(row.get("ma_signal") or "")
    ma_major_signal = str(row.get("ma_major_signal") or "")
    ma_reclaim_state = str(row.get("ma_reclaim_state") or "")
    recent_gap_state = str(row.get("recent_gap_state") or "")
    recent_gap_days = row.get("recent_gap_days")
    pct_change = float(row.get("pct_change") or 0.0)
    yolo_conf = float(row.get("yolo_confidence") or 0.0)
    candle_conf = float(row.get("candle_confidence") or 0.0)
    fund = _fundamental_context(row.get("discount_pct"), row.get("peg"))
    valuation_bias = str(fund.get("bias") or "neutral")
    near_support = bool(
        row.get("near_52w_low")
        or level in {"at_support", "closer_support"}
        or (isinstance(row.get("pct_to_support"), (int, float)) and float(row.get("pct_to_support")) <= 1.5)
        or (isinstance(row.get("pct_from_20d_low"), (int, float)) and float(row.get("pct_from_20d_low")) <= 2.5)
    )
    near_resistance = bool(
        row.get("near_52w_high")
        or level in {"at_resistance", "closer_resistance"}
        or (isinstance(row.get("pct_to_resistance"), (int, float)) and float(row.get("pct_to_resistance")) <= 1.5)
        or (isinstance(row.get("pct_from_20d_high"), (int, float)) and float(row.get("pct_from_20d_high")) <= 2.5)
    )

    bull_score = 0.0
    bear_score = 0.0
    confirmations_bull = 0
    confirmations_bear = 0
    contradictions_bull = 0
    contradictions_bear = 0

    yolo_age_factor = _yolo_age_factor(yolo_age_days, yolo_timeframe)
    yolo_recency = _yolo_recency_label(yolo_age_days, yolo_timeframe)
    yolo_direction_conflict = False
    yolo_conflict_strength = "none"
    fresh_bear_gap = recent_gap_state == "bear_gap" and isinstance(recent_gap_days, (int, float)) and float(recent_gap_days) <= 2
    fresh_bull_gap = recent_gap_state == "bull_gap" and isinstance(recent_gap_days, (int, float)) and float(recent_gap_days) <= 2
    below_short_mas = bool(
        isinstance(row.get("pct_vs_ma20"), (int, float))
        and isinstance(row.get("pct_vs_ma50"), (int, float))
        and float(row.get("pct_vs_ma20")) < 0.0
        and float(row.get("pct_vs_ma50")) < 0.0
    )
    above_short_mas = bool(
        isinstance(row.get("pct_vs_ma20"), (int, float))
        and isinstance(row.get("pct_vs_ma50"), (int, float))
        and float(row.get("pct_vs_ma20")) > 0.0
        and float(row.get("pct_vs_ma50")) > 0.0
    )

    if valuation_bias == "bullish":
        bull_score += 10.0 + (float(fund.get("long_points") or 0) * 2.0)
        confirmations_bull += 1
        contradictions_bear += 1
    elif valuation_bias == "bearish":
        bear_score += 10.0 + (float(fund.get("short_points") or 0) * 2.0)
        confirmations_bear += 1
        contradictions_bull += 1

    if yolo_bias == "bullish" and yolo_age_factor > 0.0:
        boost = (8.0 + min(8.0, yolo_conf * 10.0)) * yolo_age_factor
        bull_score += boost
        if yolo_age_factor >= 0.5:
            confirmations_bull += 1
        contradictions_bear += 1
    elif yolo_bias == "bearish" and yolo_age_factor > 0.0:
        boost = (8.0 + min(8.0, yolo_conf * 10.0)) * yolo_age_factor
        bear_score += boost
        if yolo_age_factor >= 0.5:
            confirmations_bear += 1
        contradictions_bull += 1

    if candle_bias == "bullish":
        bull_score += 3.0 + min(4.0, candle_conf * 3.0)
        confirmations_bull += 1
        contradictions_bear += 1
    elif candle_bias == "bearish":
        bear_score += 3.0 + min(4.0, candle_conf * 3.0)
        confirmations_bear += 1
        contradictions_bull += 1

    if near_support:
        bull_score += 10.0
        confirmations_bull += 1
        contradictions_bear += 1
    if near_resistance:
        bear_score += 10.0
        confirmations_bear += 1
        contradictions_bull += 1

    if trend == "uptrend":
        bull_score += 4.0
        confirmations_bull += 1
        contradictions_bear += 1
    elif trend == "downtrend":
        bear_score += 4.0
        confirmations_bear += 1
        contradictions_bull += 1

    if level_event == "resistance_breakout":
        bull_score += 8.0
        confirmations_bull += 1
        contradictions_bear += 1
    elif level_event == "support_reclaim":
        bull_score += 6.0
        confirmations_bull += 1
        contradictions_bear += 1
    elif level_event == "support_breakdown":
        bear_score += 8.0
        confirmations_bear += 1
        contradictions_bull += 1
    elif level_event == "resistance_reject":
        bear_score += 6.0
        confirmations_bear += 1
        contradictions_bull += 1

    if ma_reclaim_state == "reclaimed_ma20":
        bull_score += 4.0
        confirmations_bull += 1
        contradictions_bear += 1
    elif ma_reclaim_state == "reclaimed_ma50":
        bull_score += 6.0
        confirmations_bull += 1
        contradictions_bear += 1
    elif ma_reclaim_state == "lost_ma20":
        bear_score += 4.0
        confirmations_bear += 1
        contradictions_bull += 1
    elif ma_reclaim_state == "lost_ma50":
        bear_score += 6.0
        confirmations_bear += 1
        contradictions_bull += 1

    if recent_gap_state == "bull_gap":
        if fresh_bull_gap:
            bull_score += 6.0
            confirmations_bull += 1
            contradictions_bear += 1
        else:
            bull_score += 2.0
    elif recent_gap_state == "bear_gap":
        if fresh_bear_gap:
            bear_score += 6.0
            confirmations_bear += 1
            contradictions_bull += 1
        else:
            bear_score += 2.0

    if fresh_bear_gap:
        bear_score += 4.0
        contradictions_bull += 1
        if below_short_mas:
            bear_score += 6.0
            contradictions_bull += 1
            bull_score -= 3.0
        if isinstance(pct_change, (int, float)) and float(pct_change) <= -3.0:
            bear_score += 4.0
            confirmations_bear += 1
        if ma_signal == "bearish_20_50_cross":
            bear_score += 4.0
            confirmations_bear += 1
        if level_event == "support_breakdown":
            bear_score += 4.0
        if breakout_state == "failed_breakdown_down":
            bull_score += 2.0
    elif fresh_bull_gap:
        bull_score += 4.0
        contradictions_bear += 1
        if above_short_mas:
            bull_score += 6.0
            contradictions_bear += 1
            bear_score -= 3.0
        if isinstance(pct_change, (int, float)) and float(pct_change) >= 3.0:
            bull_score += 4.0
            confirmations_bull += 1
        if ma_signal == "bullish_20_50_cross":
            bull_score += 4.0
            confirmations_bull += 1
        if level_event == "resistance_breakout":
            bull_score += 4.0

    if ma_major_signal == "death_cross":
        bear_score += 5.0
        confirmations_bear += 1
        contradictions_bull += 1
    elif ma_major_signal == "golden_cross":
        bull_score += 5.0
        confirmations_bull += 1
        contradictions_bear += 1

    if breakout_state == "breakout_up":
        bull_score += 8.0
        confirmations_bull += 1
        contradictions_bear += 1
    elif breakout_state == "breakout_down":
        bear_score += 8.0
        confirmations_bear += 1
        contradictions_bull += 1
    elif breakout_state == "failed_breakout_up":
        bear_score += 6.0
        confirmations_bear += 1
        contradictions_bull += 1
    elif breakout_state == "failed_breakdown_down":
        bull_score += 6.0
        confirmations_bull += 1
        contradictions_bear += 1

    if structure_state == "tight_consolidation_high":
        bull_score += 5.0 if trend != "downtrend" else 2.0
        confirmations_bull += 1 if trend != "downtrend" else 0
    elif structure_state == "tight_consolidation_low":
        bear_score += 5.0 if trend != "uptrend" else 2.0
        confirmations_bear += 1 if trend != "uptrend" else 0
    elif structure_state == "tight_consolidation_mid":
        bull_score += 1.5
        bear_score += 1.5
    elif structure_state == "parabolic_up":
        bull_score += 3.5
        confirmations_bull += 1
        contradictions_bull += 1
        contradictions_bear += 1
    elif structure_state == "parabolic_down":
        bear_score += 3.5
        confirmations_bear += 1
        contradictions_bear += 1
        contradictions_bull += 1

    if 0.5 <= pct_change <= 4.0:
        bull_score += 3.0
    elif pct_change > 5.0:
        bull_score -= 4.0
        contradictions_bull += 1
    if -4.0 <= pct_change <= -0.5:
        bear_score += 3.0
    elif pct_change < -5.0:
        bear_score -= 4.0
        contradictions_bear += 1

    if stretch == "extended_up":
        bull_score -= 5.0
        contradictions_bull += 1
        bear_score += 2.0
    elif stretch == "extended_down":
        bear_score -= 5.0
        contradictions_bear += 1
        bull_score += 2.0

    pct_vs_ma20 = row.get("pct_vs_ma20")
    if isinstance(pct_vs_ma20, (int, float)):
        if float(pct_vs_ma20) <= -3.0:
            bear_score += 3.0
            contradictions_bull += 1
        elif float(pct_vs_ma20) >= 3.0:
            bull_score += 3.0
            contradictions_bear += 1
    pct_vs_ma50 = row.get("pct_vs_ma50")
    if isinstance(pct_vs_ma50, (int, float)):
        if float(pct_vs_ma50) <= -3.0:
            bear_score += 2.0
        elif float(pct_vs_ma50) >= 3.0:
            bull_score += 2.0

    rv20 = row.get("realized_vol_20")
    atr = row.get("atr_pct_14")
    if isinstance(rv20, (int, float)):
        if 18.0 <= float(rv20) <= 45.0:
            bull_score += 1.5
            bear_score += 1.5
        elif float(rv20) >= 65.0:
            bull_score -= 2.0
            bear_score -= 2.0
    if isinstance(atr, (int, float)):
        if float(atr) >= 9.0:
            bull_score -= 2.0
            bear_score -= 2.0

    if yolo_age_factor > 0.0 and yolo_age_factor < 0.3:
        if yolo_bias == "bullish":
            bull_score -= 2.0
        elif yolo_bias == "bearish":
            bear_score -= 2.0

    if bull_score >= bear_score + 5.0:
        bias = "bullish"
    elif bear_score >= bull_score + 5.0:
        bias = "bearish"
    else:
        bias = "neutral"
    score_margin = abs(bull_score - bear_score)
    if bias in {"bullish", "bearish"} and yolo_bias in {"bullish", "bearish"} and bias != yolo_bias:
        yolo_direction_conflict = True
        if yolo_recency == "fresh":
            yolo_conflict_strength = "fresh"
            # Fresh opposite YOLO should neutralize directional claims unless the edge is very large.
            if score_margin < 12.0:
                bias = "neutral"
        elif yolo_recency == "recent":
            yolo_conflict_strength = "recent"
            if score_margin < 8.0:
                bias = "neutral"
        elif yolo_recency == "aging":
            yolo_conflict_strength = "aging"
        elif yolo_recency == "stale":
            yolo_conflict_strength = "stale"

    family = "neutral_watch"
    confirmations = 0
    contradictions = 0
    score = 42.0
    if bias == "bullish":
        confirmations = confirmations_bull
        contradictions = contradictions_bull
        if breakout_state in {"breakout_up", "failed_breakdown_down"} or (
            trend == "uptrend" and structure_state in {"tight_consolidation_high", "parabolic_up"}
        ):
            family = "bullish_continuation"
            score = 36.0 + bull_score
        elif near_support or level == "below_support" or row.get("near_52w_low"):
            family = "bullish_reversal"
            score = 35.0 + bull_score
        elif trend == "uptrend" and not near_resistance:
            family = "bullish_continuation"
            score = 33.0 + bull_score
        else:
            family = "bullish_watch"
            score = 28.0 + bull_score
    elif bias == "bearish":
        confirmations = confirmations_bear
        contradictions = contradictions_bear
        if breakout_state in {"breakout_down", "failed_breakout_up"} or (
            trend == "downtrend" and structure_state in {"tight_consolidation_low", "parabolic_down"}
        ):
            family = "bearish_continuation"
            score = 36.0 + bear_score
        elif near_resistance or level == "above_resistance" or row.get("near_52w_high"):
            family = "bearish_reversal"
            score = 35.0 + bear_score
        elif trend == "downtrend" and not near_support:
            family = "bearish_continuation"
            score = 33.0 + bear_score
        else:
            family = "bearish_watch"
            score = 28.0 + bear_score
    else:
        confirmations = max(confirmations_bull, confirmations_bear)
        contradictions = max(contradictions_bull, contradictions_bear)
        score = 25.0 + max(bull_score, bear_score) - abs(bull_score - bear_score)

    if yolo_direction_conflict:
        if yolo_conflict_strength == "fresh":
            score -= 10.0
            contradictions += 2
        elif yolo_conflict_strength == "recent":
            score -= 7.0
            contradictions += 1
        elif yolo_conflict_strength == "aging":
            score -= 4.0
            contradictions += 1
        elif yolo_conflict_strength == "stale":
            score -= 2.0
    if yolo_recency == "stale" and yolo_bias in {"bullish", "bearish"}:
        score -= 3.0
    if (
        valuation_bias == "bullish"
        and fresh_bear_gap
        and candle_bias != "bullish"
    ):
        score -= 6.0
        contradictions += 1
    if (
        valuation_bias == "bearish"
        and fresh_bull_gap
        and candle_bias != "bearish"
    ):
        score -= 6.0
        contradictions += 1

    score -= contradictions * 5.0
    if confirmations == 0:
        score -= 12.0
    elif confirmations == 1:
        score -= 6.0
    score = round(_clamp(score, 0.0, 100.0), 1)

    if bias == "neutral":
        tier = "C" if score >= 60.0 else "D"
    elif confirmations >= 4 and contradictions == 0 and score >= 78.0 and "watch" not in family:
        tier = "A"
    elif confirmations >= 3 and score >= 68.0:
        tier = "B"
    elif score >= 55.0:
        tier = "C"
    else:
        tier = "D"

    if yolo_direction_conflict:
        if bias == "bullish" and family in {"bullish_reversal", "bullish_continuation"}:
            family = "bullish_watch"
        elif bias == "bearish" and family in {"bearish_reversal", "bearish_continuation"}:
            family = "bearish_watch"
        if yolo_conflict_strength == "fresh" and tier in {"A", "B"}:
            tier = "C"
        elif yolo_conflict_strength == "recent" and tier == "A":
            tier = "B"

    return {
        "signal_bias": bias,
        "setup_family": family,
        "score": score,
        "confluence_score": score,
        "setup_tier": tier,
        "confirmation_count": confirmations,
        "contradiction_count": contradictions,
        "valuation_bias": valuation_bias,
        "valuation_notes": ", ".join((fund.get("long_notes") if bias != "bearish" else fund.get("short_notes")) or []),
        "bull_score": round(bull_score, 1),
        "bear_score": round(bear_score, 1),
        "yolo_age_factor": round(yolo_age_factor, 2) if yolo_age_factor else 0.0,
        "yolo_recency": yolo_recency,
        "yolo_bias": yolo_bias or "neutral",
        "yolo_direction_conflict": yolo_direction_conflict,
        "yolo_conflict_strength": yolo_conflict_strength,
        "score_margin": round(score_margin, 1),
    }


def _describe_setup(row: dict[str, Any]) -> dict[str, str]:
    bias = str(row.get("signal_bias") or "neutral")
    family = str(row.get("setup_family") or "neutral_watch")
    trend = str(row.get("trend_state") or "mixed")
    ma_signal = str(row.get("ma_signal") or "")
    ma_major_signal = str(row.get("ma_major_signal") or "")
    ma_reclaim_state = str(row.get("ma_reclaim_state") or "")
    level = str(row.get("level_context") or "mid_range")
    level_event = str(row.get("level_event") or "none")
    stretch = str(row.get("stretch_state") or "normal")
    breakout_state = str(row.get("breakout_state") or "none")
    structure_state = str(row.get("structure_state") or "normal")
    recent_gap_state = str(row.get("recent_gap_state") or "")
    recent_gap_days = row.get("recent_gap_days")
    candle_bias = str(row.get("candle_bias") or "neutral")
    valuation_bias = str(row.get("valuation_bias") or "neutral")
    valuation_notes = str(row.get("valuation_notes") or "").strip()
    pct_from_high = row.get("pct_from_20d_high")
    pct_from_low = row.get("pct_from_20d_low")
    pct_vs_ma20 = row.get("pct_vs_ma20")
    pct_change = row.get("pct_change")
    pattern = str(row.get("yolo_pattern") or "").strip()
    yolo_bias = str(row.get("yolo_bias") or _yolo_pattern_bias(pattern) or "neutral")
    yolo_direction_conflict = bool(row.get("yolo_direction_conflict"))
    if not yolo_direction_conflict and bias in {"bullish", "bearish"} and yolo_bias in {"bullish", "bearish"}:
        yolo_direction_conflict = yolo_bias != bias
    yolo_conflict_strength = str(row.get("yolo_conflict_strength") or "none").strip().lower()
    yolo_age_days = row.get("yolo_age_days")
    yolo_timeframe = str(row.get("yolo_timeframe") or "daily")
    yolo_recency = str(row.get("yolo_recency") or _yolo_recency_label(yolo_age_days, yolo_timeframe))
    yolo_first_seen_asof = row.get("yolo_first_seen_asof")
    yolo_last_seen_asof = row.get("yolo_last_seen_asof")
    yolo_snapshots_seen = row.get("yolo_snapshots_seen")
    yolo_current_streak = row.get("yolo_current_streak")
    support_level = row.get("support_level")
    resistance_level = row.get("resistance_level")
    pct_to_support = row.get("pct_to_support")
    pct_to_resistance = row.get("pct_to_resistance")
    volume_ratio_20 = row.get("volume_ratio_20")

    level_label = {
        "below_support": "trading below nearby support",
        "at_support": "sitting inside real support",
        "closer_support": "trading closer to support than resistance",
        "at_resistance": "pressing into real resistance",
        "closer_resistance": "trading closer to resistance than support",
        "above_resistance": "trading above nearby resistance",
        "mid_range": "trading in the middle of its recent range",
    }.get(level, "trading in the middle of its recent range")
    trend_label = {
        "uptrend": "in an uptrend",
        "downtrend": "in a downtrend",
        "mixed": "in mixed trend structure",
    }.get(trend, "in mixed trend structure")
    stretch_label = {
        "extended_up": "already stretched above trend",
        "extended_down": "already stretched below trend",
        "normal": "not overly stretched",
    }.get(stretch, "not overly stretched")

    family_label = {
        "bullish_reversal": "bullish reversal candidate",
        "bullish_continuation": "bullish continuation candidate",
        "bullish_watch": "bullish watchlist candidate",
        "bearish_reversal": "bearish reversal candidate",
        "bearish_continuation": "bearish continuation candidate",
        "bearish_watch": "bearish watchlist candidate",
        "neutral_watch": "mixed / unconfirmed candidate",
    }.get(family, "mixed / unconfirmed candidate")
    family_short_label = {
        "bullish_reversal": "Bullish reversal",
        "bullish_continuation": "Bullish continuation",
        "bullish_watch": "Bullish watch",
        "bearish_reversal": "Bearish reversal",
        "bearish_continuation": "Bearish continuation",
        "bearish_watch": "Bearish watch",
        "neutral_watch": "Neutral watch",
    }.get(family, "Neutral watch")
    bias_short_label = {
        "bullish": "bullish bias",
        "bearish": "bearish bias",
        "neutral": "neutral bias",
    }.get(bias, "neutral bias")
    trend_short_label = {
        "uptrend": "uptrend",
        "downtrend": "downtrend",
        "mixed": "mixed trend",
    }.get(trend, "mixed trend")
    ma_signal_label = {
        "bearish_20_50_cross": "recent bearish 20/50 cross",
        "bullish_20_50_cross": "recent bullish 20/50 cross",
        "20_below_50": "20 below 50",
        "20_above_50": "20 above 50",
    }.get(ma_signal)
    ma_major_signal_label = {
        "death_cross": "classic death cross (50 below 200)",
        "golden_cross": "classic golden cross (50 above 200)",
        "50_below_200": "50 below 200",
        "50_above_200": "50 above 200",
    }.get(ma_major_signal)
    ma_reclaim_label = {
        "reclaimed_ma20": "reclaimed the 20-day average",
        "reclaimed_ma50": "reclaimed the 50-day average",
        "lost_ma20": "lost the 20-day average",
        "lost_ma50": "lost the 50-day average",
    }.get(ma_reclaim_state)

    observation_parts: list[str] = []
    observation_parts.append(family_label)
    if valuation_notes:
        observation_parts.append(valuation_notes)
    if ma_major_signal == "death_cross":
        observation_parts.append("classic 50/200 death cross is in place")
    elif ma_major_signal == "golden_cross":
        observation_parts.append("classic 50/200 golden cross is in place")
    elif ma_signal == "bearish_20_50_cross":
        observation_parts.append("recent bearish 20/50 crossover")
    elif ma_signal == "bullish_20_50_cross":
        observation_parts.append("recent bullish 20/50 crossover")
    if ma_reclaim_label:
        observation_parts.append(ma_reclaim_label)
    if level_event == "resistance_breakout":
        observation_parts.append("resistance got blown through")
    elif level_event == "support_breakdown":
        observation_parts.append("support got blown through")
    elif level_event == "resistance_reject":
        observation_parts.append("price rejected at resistance")
    elif level_event == "support_reclaim":
        observation_parts.append("price reclaimed broken support")
    elif breakout_state == "breakout_up":
        observation_parts.append("price is already through resistance")
    elif breakout_state == "breakout_down":
        observation_parts.append("price is already below support")
    elif breakout_state == "failed_breakout_up":
        observation_parts.append("recent breakout attempt failed back under resistance")
    elif breakout_state == "failed_breakdown_down":
        observation_parts.append("recent breakdown attempt failed back above support")
    if structure_state == "tight_consolidation_high":
        observation_parts.append("tight consolidation is building just under resistance")
    elif structure_state == "tight_consolidation_low":
        observation_parts.append("tight consolidation is building just above support")
    elif structure_state == "tight_consolidation_mid":
        observation_parts.append("price is compressing in a tight range")
    elif structure_state == "parabolic_up":
        observation_parts.append("move is becoming parabolic / extended")
    elif structure_state == "parabolic_down":
        observation_parts.append("selloff is becoming climactic / stretched")
    if recent_gap_state == "bear_gap":
        if isinstance(recent_gap_days, (int, float)) and float(recent_gap_days) <= 2:
            observation_parts.append("fresh bearish gap is still influencing price")
        else:
            observation_parts.append("older bearish gap is still overhead")
    elif recent_gap_state == "bull_gap":
        if isinstance(recent_gap_days, (int, float)) and float(recent_gap_days) <= 2:
            observation_parts.append("fresh bullish gap is supporting the move")
        else:
            observation_parts.append("older bullish gap remains in play")
    if pattern:
        if yolo_recency == "fresh":
            observation_parts.append(f"fresh YOLO: {pattern}")
        elif yolo_recency == "recent":
            observation_parts.append(f"recent YOLO: {pattern}")
        elif yolo_recency == "aging":
            observation_parts.append(f"older YOLO context: {pattern}")
        else:
            observation_parts.append(f"stale YOLO context: {pattern}")
    else:
        observation_parts.append("no decisive YOLO")
    if yolo_direction_conflict and pattern:
        conflict_prefix = (
            "fresh" if yolo_conflict_strength == "fresh"
            else ("recent" if yolo_conflict_strength == "recent" else "older")
        )
        observation_parts.append(f"{conflict_prefix} YOLO disagrees with this direction ({yolo_bias} {pattern})")
    if isinstance(yolo_snapshots_seen, int) and yolo_snapshots_seen > 1:
        if isinstance(yolo_current_streak, int) and yolo_current_streak > 1:
            observation_parts.append(f"YOLO has persisted {yolo_current_streak} snapshots")
        else:
            observation_parts.append(f"YOLO has appeared in {yolo_snapshots_seen} retained snapshots")
    if candle_bias == "bullish":
        observation_parts.append("bullish candle confirmation")
    elif candle_bias == "bearish":
        observation_parts.append("bearish candle confirmation")
    else:
        observation_parts.append("no candle confirmation")
    observation_parts.append(trend_label)
    observation_parts.append(level_label)
    observation_parts.append(stretch_label)
    if candle_bias == "bullish":
        observation_parts.append("latest candle bias is supportive")
    elif candle_bias == "bearish":
        observation_parts.append("latest candle bias is conflicting")

    actionability = "watch-only"
    action = "No strong edge yet. Keep it on watch, do not force a trade."

    near_support = (
        level in {"at_support", "closer_support"}
        or (isinstance(pct_to_support, (int, float)) and float(pct_to_support) <= 1.5)
        or (isinstance(pct_from_low, (int, float)) and float(pct_from_low) <= 2.5)
    )
    near_resistance = (
        level in {"at_resistance", "closer_resistance"}
        or (isinstance(pct_to_resistance, (int, float)) and float(pct_to_resistance) <= 1.5)
        or (isinstance(pct_from_high, (int, float)) and float(pct_from_high) <= 2.5)
    )
    large_day = isinstance(pct_change, (int, float)) and abs(float(pct_change)) >= 5.0

    if bias == "bullish":
        if family == "bullish_reversal":
            actionability = "conditional"
            action = "Long only if support holds and the reversal actually confirms. Do not buy a falling knife into earnings or resistance."
        elif family == "bullish_continuation":
            actionability = "conditional"
            action = "Long only on clean continuation through resistance or a disciplined retest. Avoid late chase entries."
        if structure_state == "parabolic_up":
            actionability = "wait"
            action = "Uptrend is still intact, but the move is getting parabolic. Wait for a hold above resistance or a cleaner retest instead of chasing."
        elif breakout_state == "breakout_up":
            actionability = "higher-probability" if trend == "uptrend" else "conditional"
            action = "Breakout / continuation watch. Best entry is a clean hold above resistance or a disciplined retest, not a momentum chase."
        elif structure_state == "tight_consolidation_high":
            actionability = "conditional"
            action = "Compression under resistance. Higher-probability entry is a decisive close through the level or a clean retest after the break."
        if recent_gap_state == "bear_gap" and isinstance(recent_gap_days, (int, float)) and float(recent_gap_days) <= 2:
            actionability = "wait"
            action = "Recent bearish gap is still in control. Wait for price to repair the gap or reclaim short-term structure before treating this as a long."
        if stretch == "extended_up" or large_day:
            actionability = "wait"
            action = "Bullish idea, but do not chase strength. Prefer a pullback or breakout retest."
        elif level == "above_resistance" and trend == "uptrend":
            actionability = "conditional"
            action = "Breakout is already through resistance. Best follow-through entry is a clean hold or retest, not an emotional chase."
        elif family == "bullish_continuation" and trend == "uptrend" and near_resistance:
            actionability = "higher-probability"
            action = "Trend continuation watch. Best if price closes through resistance or retests it cleanly."
        elif family == "bullish_reversal" and near_support and candle_bias != "bearish":
            actionability = "higher-probability"
            action = "Reversal watch at support. Actionable only if support holds and the next candles confirm."
        elif level == "below_support":
            actionability = "wait"
            action = "Bullish pattern is fighting broken support. Stand aside until price reclaims the level."
        elif trend == "downtrend":
            actionability = "conditional"
            action = "Counter-trend bounce only. Wait for trend repair before treating it as a core long."
        else:
            actionability = "conditional"
            action = "Bullish setup is present, but wait for confirmation instead of buying the first signal."
    elif bias == "bearish":
        if family == "bearish_reversal":
            actionability = "conditional"
            action = "Short only if resistance rejection confirms or support breaks. Do not force a short after an already extended flush."
        elif family == "bearish_continuation":
            actionability = "conditional"
            action = "Short only on failed bounces, rejected retests, or clean continuation below support."
        if structure_state == "parabolic_down":
            actionability = "wait"
            action = "Downtrend is still intact, but the move is already stretched lower. Prefer a failed bounce or retest rather than chasing the flush."
        elif breakout_state == "breakout_down":
            actionability = "higher-probability" if trend == "downtrend" else "conditional"
            action = "Breakdown / continuation watch. Best entry is a failed reclaim of broken support or fresh downside follow-through."
        elif structure_state == "tight_consolidation_low":
            actionability = "conditional"
            action = "Compression above support. Higher-probability short only comes if support gives way with follow-through."
        if recent_gap_state == "bull_gap" and isinstance(recent_gap_days, (int, float)) and float(recent_gap_days) <= 2:
            actionability = "wait"
            action = "Recent bullish gap is still defending the move. Wait for the gap to fail before leaning bearish."
        if stretch == "extended_down" or large_day:
            actionability = "wait"
            action = "Avoid chasing the flush. Better setup is a failed bounce or a clean support break."
        elif family == "bearish_continuation" and level == "below_support" and trend == "downtrend":
            actionability = "higher-probability"
            action = "Bearish continuation is already below support. Best entry is failed reclaim or fresh breakdown follow-through."
        elif family == "bearish_reversal" and near_support:
            actionability = "higher-probability"
            action = "Support-failure watch. It is only actionable if support actually breaks with follow-through."
        elif family == "bearish_continuation" and trend == "downtrend" and (near_resistance or candle_bias == "bearish"):
            actionability = "higher-probability"
            action = "Bearish continuation watch. Best entry is rejection near resistance or failed rally."
        elif level == "above_resistance":
            actionability = "wait"
            action = "Bearish idea is fighting price above resistance. Wait for failure back under the level before acting."
        elif trend == "uptrend":
            actionability = "conditional"
            action = "Bearish warning against the trend. Avoid new longs until structure improves."
        else:
            actionability = "conditional"
            action = "Bearish risk is present, but wait for breakdown confirmation rather than front-running it."
    else:
        if near_support and candle_bias == "bullish":
            actionability = "conditional"
            action = "Possible support bounce watch, but there is no strong AI edge yet."
        elif near_resistance and candle_bias == "bearish":
            actionability = "conditional"
            action = "Possible rejection watch near resistance, but the signal is still weak."

    stale_yolo = yolo_recency == "stale"
    aging_yolo = yolo_recency in {"aging", "stale"}
    if stale_yolo and pattern:
        if actionability == "higher-probability":
            actionability = "conditional"
        action = "Old YOLO structure only. Use current candle and level confirmation; do not treat the old box as a fresh trigger."
    elif aging_yolo and pattern and actionability == "higher-probability":
        actionability = "conditional"
        action = "Pattern context is older. Keep it on watch and require fresh confirmation from current price action."
    if yolo_direction_conflict and pattern:
        if yolo_conflict_strength in {"fresh", "recent"}:
            actionability = "wait"
        elif actionability == "higher-probability":
            actionability = "conditional"
        action = (
            f"Directional conflict: latest {yolo_bias} YOLO context disagrees with the current {bias} read. "
            "Treat this as watch-only until candles and levels resolve in one direction."
        )

    risk_notes: list[str] = []
    if stretch == "extended_up":
        risk_notes.append("extended above trend")
    elif stretch == "extended_down":
        risk_notes.append("washed out below trend")
    if isinstance(row.get("realized_vol_20"), (int, float)) and float(row["realized_vol_20"]) >= 60.0:
        risk_notes.append("high volatility")
    if isinstance(row.get("peg"), (int, float)) and float(row["peg"]) >= 2.5:
        risk_notes.append("rich PEG")
    if isinstance(row.get("discount_pct"), (int, float)) and float(row["discount_pct"]) <= 5.0:
        risk_notes.append("little valuation cushion")
    if stale_yolo:
        risk_notes.append("stale YOLO context")
    elif aging_yolo and pattern:
        risk_notes.append("older YOLO context")
    if yolo_direction_conflict:
        if yolo_conflict_strength in {"fresh", "recent"}:
            risk_notes.append("fresh opposite YOLO signal")
        else:
            risk_notes.append("YOLO direction conflict")
    if ma_major_signal == "death_cross":
        risk_notes.append("death cross regime")
    elif ma_signal == "bearish_20_50_cross":
        risk_notes.append("bearish 20/50 crossover")

    if valuation_bias != "neutral" and valuation_bias != bias:
        observation_parts.append("valuation does not fully agree with the direction")
    if bias == "neutral":
        actionability = "watch-only"
        action = "Evidence is mixed. Keep it on watch until price, pattern, and level context line up in one direction."

    observation = ". ".join(part[0].upper() + part[1:] if idx == 0 else part for idx, part in enumerate(observation_parts)) + "."
    level_bits = []
    if isinstance(support_level, (int, float)):
        level_bits.append(f"support {support_level}")
    if isinstance(resistance_level, (int, float)):
        level_bits.append(f"resistance {resistance_level}")
    if isinstance(yolo_age_days, (int, float)) and pattern:
        level_bits.append(f"YOLO age {int(float(yolo_age_days))}d")
    if yolo_direction_conflict and yolo_bias in {"bullish", "bearish"} and pattern:
        level_bits.append(f"YOLO conflict ({yolo_bias})")
    if isinstance(yolo_snapshots_seen, int) and yolo_snapshots_seen > 0 and yolo_first_seen_asof:
        if (
            isinstance(yolo_age_days, (int, float))
            and int(float(yolo_age_days)) > 0
            and int(yolo_snapshots_seen) <= 1
        ):
            level_bits.append(f"retained history starts {yolo_first_seen_asof}")
        elif isinstance(yolo_current_streak, int) and yolo_current_streak > 1:
            level_bits.append(f"YOLO {yolo_current_streak}x since {yolo_first_seen_asof}")
        else:
            level_bits.append(f"YOLO first seen {yolo_first_seen_asof}")
    if ma_signal_label:
        level_bits.append(f"MA {ma_signal_label}")
    if ma_major_signal_label:
        level_bits.append(f"MA {ma_major_signal_label}")
    if ma_reclaim_label:
        level_bits.append(ma_reclaim_label)
    if recent_gap_state:
        recent_gap_label = "fresh" if isinstance(recent_gap_days, (int, float)) and float(recent_gap_days) <= 2 else "older"
        level_bits.append(f"{recent_gap_label} {recent_gap_state.replace('_', ' ')}")
    if breakout_state != "none":
        level_bits.append(breakout_state.replace("_", " "))
    if structure_state != "normal":
        level_bits.append(structure_state.replace("_", " "))
    if isinstance(volume_ratio_20, (int, float)):
        level_bits.append(f"vol {float(volume_ratio_20):.2f}x")
    level_suffix = f" | {' / '.join(level_bits)}" if level_bits else ""
    if near_support and near_resistance:
        location_short = "between key levels"
    elif near_support:
        location_short = "near support"
    elif near_resistance:
        location_short = "near resistance"
    else:
        location_short = "mid-range"
    return {
        "signal_bias": bias,
        "observation": observation,
        "actionability": actionability,
        "action": action,
        "risk_note": ", ".join(risk_notes[:3]) if risk_notes else "none",
        "yolo_direction_conflict": yolo_direction_conflict,
        "yolo_conflict_strength": yolo_conflict_strength,
        "technical_read": (
            f"{family_short_label} | {bias_short_label} | {trend_short_label} | "
            f"{location_short} | "
            f"vs MA20 {_fmt_pct_short(pct_vs_ma20)}{level_suffix}"
        ),
    }


def _apply_llm_narrative_overrides(
    setup_rows: list[dict[str, Any]],
    *,
    source: str,
) -> None:
    if not setup_rows:
        return
    if not llm_enabled():
        for row in setup_rows:
            if isinstance(row, dict):
                row.setdefault("narrative_source", "rule")
        return
    max_rows = llm_max_setups()
    if max_rows <= 0:
        for row in setup_rows:
            if isinstance(row, dict):
                row.setdefault("narrative_source", "rule")
        return
    for idx, row in enumerate(setup_rows):
        if not isinstance(row, dict):
            continue
        row.setdefault("narrative_source", "rule")
        if idx >= max_rows:
            continue
        overrides = maybe_rewrite_setup_copy(row, source=source)
        if not overrides:
            continue
        row.update(overrides)
        row["narrative_source"] = "llm"


def _apply_debate_payload(setup_rows: list[dict[str, Any]]) -> None:
    if not DEBATE_ENGINE_ENABLED:
        for row in setup_rows:
            if isinstance(row, dict):
                row.setdefault("debate_v1", {"version": "v1", "mode": "disabled"})
        return
    for row in setup_rows:
        if not isinstance(row, dict):
            continue
        try:
            row["debate_v1"] = build_setup_debate(row)
        except Exception:
            row.setdefault("debate_v1", {"version": "v1", "mode": "error"})


def _tier_rank(tier: Any) -> int:
    raw = str(tier or "").strip().upper()
    return {"D": 1, "C": 2, "B": 3, "A": 4}.get(raw, 1)


def _cap_tier(current_tier: Any, max_tier: str) -> str:
    cur = str(current_tier or "").strip().upper() or "D"
    cap = str(max_tier or "").strip().upper() or "D"
    return cap if _tier_rank(cur) > _tier_rank(cap) else cur


def _apply_debate_guardrails(setup_rows: list[dict[str, Any]]) -> None:
    for row in setup_rows:
        if not isinstance(row, dict):
            continue
        debate = row.get("debate_v1")
        if not isinstance(debate, dict):
            continue
        consensus = debate.get("consensus")
        if not isinstance(consensus, dict):
            continue

        state = str(consensus.get("consensus_state") or "watch").strip().lower()
        bias = str(consensus.get("consensus_bias") or "neutral").strip().lower()
        agreement = _to_float(consensus.get("agreement_score")) or 0.0
        disagreement_count = int(_to_float(consensus.get("disagreement_count")) or 0)
        safety_adj = consensus.get("safety_adjustment")
        if not isinstance(safety_adj, list):
            safety_adj = []

        row["debate_consensus_state"] = state
        row["debate_consensus_bias"] = bias
        row["debate_agreement_score"] = round(agreement, 1)
        row["debate_disagreement_count"] = disagreement_count
        row["debate_safety_adjustment"] = safety_adj

        current_actionability = str(row.get("actionability") or "watch-only").strip().lower()

        if state == "watch":
            row["actionability"] = "watch-only"
            row["setup_tier"] = _cap_tier(row.get("setup_tier"), "C")
            row["action"] = (
                "Watch-only. Multi-angle evidence is mixed; wait for clearer confirmation "
                "from trend, levels, and participation."
            )
        elif state == "conditional":
            if current_actionability in {"higher-probability", "setup_ready", "ready"}:
                row["actionability"] = "conditional"
            row["setup_tier"] = _cap_tier(row.get("setup_tier"), "B")
            if current_actionability in {"higher-probability", "setup_ready", "ready"}:
                row["action"] = (
                    "Conditional setup. Debate disagreement requires confirmation before acting."
                )
        elif state == "ready":
            # Let the debate lift "watch-only" to conditional when agreement is strong,
            # but do not auto-promote to higher-probability here.
            if current_actionability in {"watch-only", "watch", "wait"} and agreement >= 70.0:
                row["actionability"] = "conditional"

        if disagreement_count >= 2 and str(row.get("actionability") or "").strip().lower() in {
            "higher-probability",
            "setup_ready",
            "ready",
        }:
            row["actionability"] = "conditional"

        if bias in {"bullish", "bearish"} and str(row.get("signal_bias") or "neutral").lower() == "neutral":
            row["signal_bias"] = bias

        if safety_adj and str(row.get("actionability") or "").strip().lower() in {"higher-probability", "setup_ready", "ready"}:
            row["actionability"] = "conditional"
            row["action"] = "Conditional setup. Safety guardrails are active; wait for confirmation."

        existing_risk = str(row.get("risk_note") or "").strip()
        debate_risk = f"debate={state} ({agreement:.0f}% agreement)"
        if existing_risk and existing_risk.lower() not in {"none", "-"}:
            if debate_risk not in existing_risk:
                row["risk_note"] = f"{existing_risk}; {debate_risk}"
        else:
            row["risk_note"] = debate_risk


def ensure_setup_call_eval_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS setup_call_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asof_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            report_kind TEXT NOT NULL DEFAULT 'daily',
            generated_ts TEXT,
            call_direction TEXT NOT NULL,
            validity_days INTEGER NOT NULL DEFAULT 5,
            valid_target_date TEXT,
            setup_family TEXT,
            setup_tier TEXT,
            signal_bias TEXT,
            actionability TEXT,
            score REAL,
            close_asof REAL NOT NULL,
            yolo_pattern TEXT,
            yolo_recency TEXT,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'open',
            evaluated_date TEXT,
            close_evaluated REAL,
            raw_return_pct REAL,
            signed_return_pct REAL,
            direction_hit INTEGER,
            UNIQUE(asof_date, ticker, report_kind)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_setup_call_eval_status ON setup_call_evaluations(status, asof_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_setup_call_eval_family ON setup_call_evaluations(setup_family, call_direction, asof_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_setup_call_eval_tier ON setup_call_evaluations(setup_tier, asof_date)"
    )


def _setup_call_direction(row: dict[str, Any]) -> str:
    family = str(row.get("setup_family") or "").strip().lower()
    bias = str(row.get("signal_bias") or "").strip().lower()
    if family.startswith("bullish") or bias == "bullish":
        return "long"
    if family.startswith("bearish") or bias == "bearish":
        return "short"
    return "neutral"


def _setup_validity_days(row: dict[str, Any]) -> int:
    family = str(row.get("setup_family") or "").strip().lower()
    actionability = str(row.get("actionability") or "").strip().lower()
    yolo_recency = str(row.get("yolo_recency") or "").strip().lower()
    if "continuation" in family:
        days = 5
    elif "reversal" in family:
        days = 7
    elif "watch" in family:
        days = 4
    else:
        days = 3

    if actionability == "higher-probability":
        days += 1
    elif actionability in {"wait", "watch-only"}:
        days = max(3, days - 1)

    if yolo_recency == "stale":
        days = max(2, days - 2)
    elif yolo_recency == "aging":
        days = max(3, days - 1)
    return int(max(2, min(12, days)))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    if n % 2 == 1:
        return ordered[n // 2]
    return (ordered[(n // 2) - 1] + ordered[n // 2]) / 2.0


def _round_or_none(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _persist_setup_call_candidates(
    conn: sqlite3.Connection,
    *,
    generated_ts: str | None,
    report_kind: str,
    asof_date: str | None,
    setup_rows: list[dict[str, Any]],
) -> int:
    if not asof_date:
        return 0
    inserted = 0
    for row in (setup_rows or [])[:SETUP_EVAL_TRACK_LIMIT]:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        direction = _setup_call_direction(row)
        if direction not in {"long", "short"}:
            continue
        close_asof = row.get("close")
        if not isinstance(close_asof, (int, float)) or float(close_asof) <= 0:
            close_row = conn.execute(
                "SELECT CAST(close AS REAL) FROM price_daily WHERE ticker = ? AND date = ? LIMIT 1",
                (ticker, asof_date),
            ).fetchone()
            close_asof = float(close_row[0]) if close_row and close_row[0] is not None else None
        if not isinstance(close_asof, (int, float)) or float(close_asof) <= 0:
            continue
        validity_days = _setup_validity_days(row)
        before_changes = conn.total_changes
        conn.execute(
            """
            INSERT INTO setup_call_evaluations (
                asof_date,
                ticker,
                report_kind,
                generated_ts,
                call_direction,
                validity_days,
                setup_family,
                setup_tier,
                signal_bias,
                actionability,
                score,
                close_asof,
                yolo_pattern,
                yolo_recency,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
            ON CONFLICT(asof_date, ticker, report_kind) DO NOTHING
            """,
            (
                asof_date,
                ticker,
                report_kind,
                generated_ts,
                direction,
                int(validity_days),
                row.get("setup_family"),
                row.get("setup_tier"),
                row.get("signal_bias"),
                row.get("actionability"),
                row.get("score"),
                float(close_asof),
                row.get("yolo_pattern"),
                row.get("yolo_recency"),
            ),
        )
        if conn.total_changes > before_changes:
            inserted += 1
    return inserted


def _score_open_setup_call_outcomes(conn: sqlite3.Connection) -> int:
    open_rows = conn.execute(
        """
        SELECT id, ticker, asof_date, call_direction, validity_days, close_asof
        FROM setup_call_evaluations
        WHERE status = 'open'
        ORDER BY asof_date ASC, id ASC
        """
    ).fetchall()
    scored = 0
    for row in open_rows:
        call_id = int(row[0])
        ticker = str(row[1] or "").upper().strip()
        asof_date = str(row[2] or "").strip()
        call_direction = str(row[3] or "").strip().lower()
        validity_days = int(row[4] or 0)
        close_asof = float(row[5] or 0.0)
        if call_direction not in {"long", "short"} or validity_days <= 0 or close_asof <= 0.0:
            conn.execute(
                """
                UPDATE setup_call_evaluations
                SET status = 'invalid',
                    updated_ts = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (call_id,),
            )
            continue

        future_rows = conn.execute(
            """
            SELECT date, CAST(close AS REAL) AS close
            FROM price_daily
            WHERE ticker = ? AND date > ?
            ORDER BY date ASC
            LIMIT ?
            """,
            (ticker, asof_date, int(validity_days)),
        ).fetchall()
        if len(future_rows) < validity_days:
            continue
        eval_date = str(future_rows[-1][0] or "").strip()
        eval_close = float(future_rows[-1][1] or 0.0)
        if not eval_date or eval_close <= 0.0:
            continue
        raw_return_pct = ((eval_close / close_asof) - 1.0) * 100.0
        signed_return_pct = raw_return_pct if call_direction == "long" else (-raw_return_pct)
        hit = 1 if signed_return_pct >= float(SETUP_EVAL_HIT_THRESHOLD_PCT) else 0
        conn.execute(
            """
            UPDATE setup_call_evaluations
            SET status = 'scored',
                valid_target_date = ?,
                evaluated_date = ?,
                close_evaluated = ?,
                raw_return_pct = ?,
                signed_return_pct = ?,
                direction_hit = ?,
                updated_ts = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                eval_date,
                eval_date,
                round(eval_close, 6),
                round(raw_return_pct, 6),
                round(signed_return_pct, 6),
                int(hit),
                call_id,
            ),
        )
        scored += 1
    return scored


def _setup_eval_bucket(
    label: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    calls = len(rows)
    wins = [row for row in rows if int(row.get("direction_hit") or 0) == 1]
    losses = [row for row in rows if int(row.get("direction_hit") or 0) != 1]
    hits = len(wins)
    losses_count = len(losses)
    signed_vals = [float(row.get("signed_return_pct")) for row in rows if isinstance(row.get("signed_return_pct"), (int, float))]
    win_vals = [float(row.get("signed_return_pct")) for row in wins if isinstance(row.get("signed_return_pct"), (int, float))]
    loss_vals = [float(row.get("signed_return_pct")) for row in losses if isinstance(row.get("signed_return_pct"), (int, float))]
    hit_rate = (hits / calls) if calls else None
    loss_rate = (losses_count / calls) if calls else None
    avg_win = (sum(win_vals) / len(win_vals)) if win_vals else None
    avg_loss = (sum(loss_vals) / len(loss_vals)) if loss_vals else None
    expectancy: float | None = None
    if hit_rate is not None and loss_rate is not None:
        expectancy = (hit_rate * float(avg_win or 0.0)) + (loss_rate * float(avg_loss or 0.0))
    pos_sum = sum(v for v in signed_vals if v > 0.0)
    neg_abs_sum = abs(sum(v for v in signed_vals if v < 0.0))
    profit_factor: float | None = None
    if neg_abs_sum > 0.0:
        profit_factor = pos_sum / neg_abs_sum
    elif pos_sum > 0.0 and calls > 0:
        profit_factor = 9.99
    return {
        "label": label,
        "calls": calls,
        "hits": hits,
        "losses": losses_count,
        "hit_rate_pct": _round_or_none((hit_rate * 100.0) if hit_rate is not None else None),
        "loss_rate_pct": _round_or_none((loss_rate * 100.0) if loss_rate is not None else None),
        "avg_signed_return_pct": _round_or_none((sum(signed_vals) / len(signed_vals)) if signed_vals else None),
        "median_signed_return_pct": _round_or_none(_median(signed_vals)),
        "avg_win_return_pct": _round_or_none(avg_win),
        "avg_loss_return_pct": _round_or_none(avg_loss),
        "expectancy_pct": _round_or_none(expectancy),
        "profit_factor": _round_or_none(profit_factor),
    }


def _build_setup_eval_improvement_actions(
    *,
    scored_calls: int,
    min_sample: int,
    hit_threshold_pct: float,
    overall: dict[str, Any],
    by_direction: list[dict[str, Any]],
    by_family: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if scored_calls < max(5, int(min_sample) * 3):
        actions.append(
            {
                "priority": "info",
                "scope": "dataset",
                "reason": (
                    f"Backtest sample is still small ({scored_calls} scored calls). "
                    f"Target at least {max(5, int(min_sample) * 3)} for stable family tuning."
                ),
                "recommendation": "Keep current score weights conservative and collect more outcomes before major threshold changes.",
            }
        )
        return actions

    overall_expectancy = _to_float(overall.get("expectancy_pct"))
    overall_hit = _to_float(overall.get("hit_rate_pct"))
    if overall_expectancy is not None and overall_expectancy <= 0.0:
        actions.append(
            {
                "priority": "high",
                "scope": "global",
                "reason": (
                    f"Overall expectancy is non-positive ({round(overall_expectancy, 2)}%)."
                ),
                "recommendation": (
                    "Tighten setup-ready criteria: require stronger level confirmation and downgrade stale-context candidates to watch."
                ),
            }
        )
    if overall_hit is not None and overall_hit < float(hit_threshold_pct):
        actions.append(
            {
                "priority": "high",
                "scope": "global",
                "reason": (
                    f"Overall hit rate ({round(overall_hit, 2)}%) is below target threshold ({round(hit_threshold_pct, 2)}%)."
                ),
                "recommendation": (
                    "Increase entry confirmation requirements (fresh YOLO + candle/level alignment) before labeling a setup as actionable."
                ),
            }
        )

    dir_map = {
        str(row.get("direction") or "").strip().lower(): row
        for row in by_direction
        if isinstance(row, dict)
    }
    long_row = dir_map.get("long") or {}
    short_row = dir_map.get("short") or {}
    long_calls = int(long_row.get("calls") or 0)
    short_calls = int(short_row.get("calls") or 0)
    long_expectancy = _to_float(long_row.get("expectancy_pct"))
    short_expectancy = _to_float(short_row.get("expectancy_pct"))
    if long_calls >= int(min_sample) and long_expectancy is not None and long_expectancy < 0.0:
        actions.append(
            {
                "priority": "medium",
                "scope": "long_setups",
                "reason": f"Long calls have negative expectancy ({round(long_expectancy, 2)}%) over {long_calls} samples.",
                "recommendation": "Reduce long setup score bonus until trend/level confirmation improves.",
            }
        )
    if short_calls >= int(min_sample) and short_expectancy is not None and short_expectancy < 0.0:
        actions.append(
            {
                "priority": "medium",
                "scope": "short_setups",
                "reason": f"Short calls have negative expectancy ({round(short_expectancy, 2)}%) over {short_calls} samples.",
                "recommendation": "Reduce short setup score bonus until breakdown confirmation improves.",
            }
        )

    candidates = [row for row in by_family if isinstance(row, dict) and int(row.get("calls") or 0) >= int(min_sample)]
    weak = [
        row for row in candidates
        if (
            (_to_float(row.get("hit_rate_pct")) is not None and float(_to_float(row.get("hit_rate_pct")) or 0.0) < float(hit_threshold_pct))
            or (_to_float(row.get("expectancy_pct")) is not None and float(_to_float(row.get("expectancy_pct")) or 0.0) < 0.0)
        )
    ]
    weak.sort(
        key=lambda row: (
            float(_to_float(row.get("expectancy_pct")) or 0.0),
            float(_to_float(row.get("hit_rate_pct")) or 0.0),
            -int(row.get("calls") or 0),
        )
    )
    if weak:
        top_weak = weak[0]
        family = str(top_weak.get("setup_family") or "-")
        direction = str(top_weak.get("call_direction") or "-")
        actions.append(
            {
                "priority": "medium",
                "scope": "family",
                "reason": (
                    f"Weak family detected: {family}:{direction} "
                    f"(hit {top_weak.get('hit_rate_pct')}%, expectancy {top_weak.get('expectancy_pct')}%, calls {top_weak.get('calls')})."
                ),
                "recommendation": (
                    f"Demote {family}:{direction} by default (or require stronger confirmation) until its backtest edge recovers."
                ),
            }
        )

    strong = [
        row for row in candidates
        if (
            (_to_float(row.get("hit_rate_pct")) is not None and float(_to_float(row.get("hit_rate_pct")) or 0.0) >= float(hit_threshold_pct) + 5.0)
            and (_to_float(row.get("expectancy_pct")) is not None and float(_to_float(row.get("expectancy_pct")) or 0.0) > 0.0)
        )
    ]
    strong.sort(
        key=lambda row: (
            float(_to_float(row.get("expectancy_pct")) or 0.0),
            float(_to_float(row.get("hit_rate_pct")) or 0.0),
            int(row.get("calls") or 0),
        ),
        reverse=True,
    )
    if strong:
        top_strong = strong[0]
        family = str(top_strong.get("setup_family") or "-")
        direction = str(top_strong.get("call_direction") or "-")
        actions.append(
            {
                "priority": "low",
                "scope": "family",
                "reason": (
                    f"Strong family detected: {family}:{direction} "
                    f"(hit {top_strong.get('hit_rate_pct')}%, expectancy {top_strong.get('expectancy_pct')}%, calls {top_strong.get('calls')})."
                ),
                "recommendation": (
                    f"Use {family}:{direction} as a baseline and compare weaker families against this quality bar."
                ),
            }
        )

    return actions[:5]


def _summarize_setup_call_evaluations(
    conn: sqlite3.Connection,
    *,
    window_days: int = SETUP_EVAL_WINDOW_DAYS,
    min_sample: int = SETUP_EVAL_MIN_SAMPLE,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    if not table_exists(conn, "setup_call_evaluations"):
        return {}, {}
    cutoff_date = (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=max(30, int(window_days)))).isoformat()
    open_calls = int(
        (
            conn.execute("SELECT COUNT(*) FROM setup_call_evaluations WHERE status = 'open'").fetchone() or [0]
        )[0]
        or 0
    )
    scored_rows = [
        dict(r)
        for r in conn.execute(
            """
            SELECT
                ticker,
                asof_date,
                setup_family,
                setup_tier,
                call_direction,
                validity_days,
                direction_hit,
                signed_return_pct
            FROM setup_call_evaluations
            WHERE status = 'scored'
              AND asof_date >= ?
            ORDER BY asof_date DESC, id DESC
            """,
            (cutoff_date,),
        ).fetchall()
    ]

    overall = _setup_eval_bucket("overall", scored_rows)
    validity_vals = [int(row.get("validity_days") or 0) for row in scored_rows if int(row.get("validity_days") or 0) > 0]
    overall["avg_validity_days"] = _round_or_none((sum(validity_vals) / len(validity_vals)) if validity_vals else None)

    by_direction: list[dict[str, Any]] = []
    for direction in ("long", "short"):
        group = [row for row in scored_rows if str(row.get("call_direction") or "").strip().lower() == direction]
        bucket = _setup_eval_bucket(direction, group)
        bucket["direction"] = direction
        by_direction.append(bucket)

    by_validity: list[dict[str, Any]] = []
    validity_groups: dict[int, list[dict[str, Any]]] = {}
    for row in scored_rows:
        validity = int(row.get("validity_days") or 0)
        if validity <= 0:
            continue
        validity_groups.setdefault(validity, []).append(row)
    for validity, group in sorted(validity_groups.items(), key=lambda kv: kv[0]):
        bucket = _setup_eval_bucket(f"{validity}d", group)
        bucket["validity_days"] = int(validity)
        by_validity.append(bucket)

    by_tier: list[dict[str, Any]] = []
    for tier in ("A", "B", "C", "D"):
        group = [row for row in scored_rows if str(row.get("setup_tier") or "").strip().upper() == tier]
        bucket = _setup_eval_bucket(tier, group)
        bucket["setup_tier"] = tier
        by_tier.append(bucket)

    family_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in scored_rows:
        family = str(row.get("setup_family") or "").strip().lower()
        direction = str(row.get("call_direction") or "").strip().lower()
        if not family or direction not in {"long", "short"}:
            continue
        family_map.setdefault((family, direction), []).append(row)

    by_family: list[dict[str, Any]] = []
    reliability_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group in family_map.items():
        family, direction = key
        bucket = _setup_eval_bucket(f"{family}:{direction}", group)
        bucket["setup_family"] = family
        bucket["call_direction"] = direction
        by_family.append(bucket)
        if int(bucket.get("calls") or 0) >= max(1, int(min_sample)):
            reliability_lookup[key] = {
                "hit_rate_pct": bucket.get("hit_rate_pct"),
                "calls": int(bucket.get("calls") or 0),
                "avg_signed_return_pct": bucket.get("avg_signed_return_pct"),
            }
    by_family.sort(
        key=lambda item: (
            int(item.get("calls") or 0),
            float(item.get("hit_rate_pct") or 0.0),
            float(item.get("avg_signed_return_pct") or 0.0),
        ),
        reverse=True,
    )

    top_long = [row for row in by_family if str(row.get("call_direction") or "").lower() == "long"][:8]
    top_short = [row for row in by_family if str(row.get("call_direction") or "").lower() == "short"][:8]
    weak_families = sorted(
        [
            row for row in by_family
            if int(row.get("calls") or 0) >= int(max(1, min_sample))
        ],
        key=lambda row: (
            float(_to_float(row.get("expectancy_pct")) or 0.0),
            float(_to_float(row.get("hit_rate_pct")) or 0.0),
            -int(row.get("calls") or 0),
        ),
    )[:8]

    newest_scored = next(
        (str(row.get("asof_date") or "").strip() for row in scored_rows if str(row.get("asof_date") or "").strip()),
        None,
    )
    summary = {
        "enabled": True,
        "window_days": int(max(30, int(window_days))),
        "min_sample": int(max(1, int(min_sample))),
        "hit_threshold_pct": float(SETUP_EVAL_HIT_THRESHOLD_PCT),
        "scored_calls": int(overall.get("calls") or 0),
        "open_calls": int(open_calls),
        "latest_scored_asof": newest_scored,
        "overall": overall,
        "by_direction": by_direction,
        "by_validity_days": by_validity,
        "by_tier": by_tier,
        "by_family": by_family[:12],
        "top_long_families": top_long,
        "top_short_families": top_short,
        "weak_families": weak_families,
    }
    summary["improvement_actions"] = _build_setup_eval_improvement_actions(
        scored_calls=int(summary.get("scored_calls") or 0),
        min_sample=int(summary.get("min_sample") or 1),
        hit_threshold_pct=float(summary.get("hit_threshold_pct") or 0.0),
        overall=overall,
        by_direction=by_direction,
        by_family=by_family,
    )
    return summary, reliability_lookup


def _setup_eval_score_adjustment(
    stat: dict[str, Any] | None,
    *,
    min_sample: int,
    hit_threshold_pct: float,
) -> float:
    if not stat:
        return 0.0
    calls = int(stat.get("calls") or 0)
    if calls < max(1, int(min_sample)):
        return 0.0
    hit_rate = float(stat.get("hit_rate_pct") or 0.0)
    avg_signed_return = float(stat.get("avg_signed_return_pct") or 0.0)

    # Confidence grows with sample size but is bounded so old data cannot dominate.
    sample_scale = _clamp(calls / float(max(1, int(min_sample) * 4)), 0.35, 1.0)
    hit_component = _clamp((hit_rate - float(hit_threshold_pct)) * 0.28, -8.0, 8.0)
    return_component = _clamp(avg_signed_return * 1.6, -6.0, 6.0)
    adjustment = (hit_component + return_component) * sample_scale
    return round(_clamp(adjustment, -10.0, 10.0), 1)


def _apply_setup_eval_fields(
    setup_rows: list[dict[str, Any]],
    *,
    reliability_lookup: dict[tuple[str, str], dict[str, Any]],
    min_sample: int,
    hit_threshold_pct: float,
) -> dict[str, Any]:
    adjusted_calls = 0
    adjustments: list[float] = []
    for row in setup_rows or []:
        if not isinstance(row, dict):
            continue
        call_direction = _setup_call_direction(row)
        validity_days = _setup_validity_days(row)
        family = str(row.get("setup_family") or "").strip().lower()
        stat = reliability_lookup.get((family, call_direction))
        row["call_direction"] = call_direction
        row["validity_days"] = int(validity_days)
        row["validity_label"] = f"{int(validity_days)} trading day{'s' if int(validity_days) != 1 else ''}"
        row["historical_reliability_pct"] = stat.get("hit_rate_pct") if stat else None
        row["historical_sample_size"] = int(stat.get("calls") or 0) if stat else 0
        row["historical_avg_signed_return_pct"] = stat.get("avg_signed_return_pct") if stat else None
        if stat:
            row["reliability_label"] = (
                f"{_round_or_none(stat.get('hit_rate_pct'))}% hit rate "
                f"({int(stat.get('calls') or 0)} calls)"
            )
        else:
            row["reliability_label"] = "insufficient history"
        raw_score = float(row.get("score") or 0.0)
        adjustment = _setup_eval_score_adjustment(
            stat,
            min_sample=min_sample,
            hit_threshold_pct=hit_threshold_pct,
        )
        adjusted_score = round(_clamp(raw_score + adjustment, 0.0, 100.0), 1)
        row["setup_score_raw"] = round(raw_score, 1)
        row["setup_score_adjustment"] = adjustment
        row["setup_score_adjusted"] = adjusted_score
        if adjustment >= 2.0:
            row["reliability_signal"] = "tailwind"
        elif adjustment <= -2.0:
            row["reliability_signal"] = "headwind"
        else:
            row["reliability_signal"] = "neutral"
        if adjustment != 0.0:
            adjusted_calls += 1
            adjustments.append(adjustment)
            row["score"] = adjusted_score
            row["confluence_score"] = adjusted_score
            row["setup_tier"] = _setup_tier(adjusted_score)
    return {
        "adjusted_calls": int(adjusted_calls),
        "avg_adjustment": _round_or_none((sum(adjustments) / len(adjustments)) if adjustments else 0.0),
        "max_positive_adjustment": _round_or_none(max(adjustments) if adjustments else 0.0),
        "max_negative_adjustment": _round_or_none(min(adjustments) if adjustments else 0.0),
    }


def _refresh_setup_eval_surfaces(signals: dict[str, Any]) -> None:
    setup_rows = (signals.get("setup_quality_top") or []) if isinstance(signals, dict) else []
    by_ticker = {
        str(row.get("ticker") or "").upper(): row
        for row in setup_rows
        if isinstance(row, dict) and row.get("ticker")
    }
    eval_keys = (
        "call_direction",
        "validity_days",
        "validity_label",
        "historical_reliability_pct",
        "historical_sample_size",
        "historical_avg_signed_return_pct",
        "reliability_label",
        "reliability_signal",
        "setup_score_raw",
        "setup_score_adjustment",
        "setup_score_adjusted",
        "score",
        "confluence_score",
        "setup_tier",
        "setup_family",
        "signal_bias",
        "actionability",
        "observation",
        "action",
        "risk_note",
        "technical_read",
        "narrative_source",
    )
    watchlist = signals.get("watchlist_candidates")
    if isinstance(watchlist, list):
        for row in watchlist:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker") or "").upper()
            src = by_ticker.get(ticker)
            if not src:
                continue
            for key in eval_keys:
                row[key] = src.get(key)
    setup_lookup = signals.get("setup_quality_lookup")
    if isinstance(setup_lookup, dict):
        for ticker, payload in list(setup_lookup.items()):
            if not isinstance(payload, dict):
                continue
            src = by_ticker.get(str(ticker or "").upper())
            if not src:
                continue
            for key in eval_keys:
                payload[key] = src.get(key)


def _setup_cluster_rows(rows: list[dict[str, Any]], *, score_window: float = 3.0, scan_limit: int = 8) -> list[dict[str, Any]]:
    if not rows:
        return []
    best = rows[0]
    best_score = float(best.get("score") or 0.0)
    best_tier = str(best.get("setup_tier") or "").strip().upper()
    best_family = str(best.get("setup_family") or "").strip().lower()
    cluster: list[dict[str, Any]] = [best]
    for row in rows[1:scan_limit]:
        row_tier = str(row.get("setup_tier") or "").strip().upper()
        row_family = str(row.get("setup_family") or "").strip().lower()
        row_score = float(row.get("score") or 0.0)
        if row_tier != best_tier:
            continue
        if best_family and row_family != best_family:
            continue
        if (best_score - row_score) > score_window:
            continue
        cluster.append(row)
    return cluster


def build_tonight_key_changes(signals: dict[str, Any], yolo_delta: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []

    breadth = signals.get("market_breadth") or {}
    if breadth:
        adv = breadth.get("advancers", 0)
        dec = breadth.get("decliners", 0)
        pct = breadth.get("pct_advancing")
        avg_move = breadth.get("avg_pct_change")
        regime = "risk-on" if isinstance(pct, (int, float)) and pct >= 55 else "mixed"
        if isinstance(pct, (int, float)) and pct <= 45:
            regime = "risk-off"
        changes.append(
            {
                "slug": "breadth",
                "title": "Breadth Regime",
                "detail": f"{adv} advancing vs {dec} declining ({pct}% advancers, avg move {avg_move}%).",
                "tone": "positive" if regime == "risk-on" else ("negative" if regime == "risk-off" else "neutral"),
            }
        )

    movers_up = signals.get("movers_up_today") or []
    movers_down = signals.get("movers_down_today") or []
    if movers_up or movers_down:
        top_up = movers_up[0] if movers_up else None
        top_down = movers_down[0] if movers_down else None
        if top_up and top_down:
            detail = (
                f"Leader: {top_up.get('ticker')} {top_up.get('pct_change')}% | "
                f"Laggard: {top_down.get('ticker')} {top_down.get('pct_change')}%."
            )
        elif top_up:
            detail = f"Strongest upside move: {top_up.get('ticker')} {top_up.get('pct_change')}%."
        else:
            detail = f"Weakest move: {top_down.get('ticker')} {top_down.get('pct_change')}%."
        changes.append(
            {
                "slug": "movers",
                "title": "Largest Price Moves",
                "detail": detail,
                "tone": "neutral",
            }
        )

    new_count = int(yolo_delta.get("new_count") or 0)
    lost_count = int(yolo_delta.get("lost_count") or 0)
    if new_count or lost_count:
        new_patterns = yolo_delta.get("new_patterns") or []
        lost_patterns = yolo_delta.get("lost_patterns") or []
        top_new = new_patterns[0] if new_patterns else None
        top_lost = lost_patterns[0] if lost_patterns else None
        yolo_parts = [f"+{new_count} new", f"-{lost_count} lost"]
        if top_new:
            yolo_parts.append(
                f"new highlight: {top_new.get('ticker')} {top_new.get('pattern')} ({top_new.get('confidence')})"
            )
        if top_lost:
            yolo_parts.append(
                f"invalidated/completed: {top_lost.get('ticker')} {top_lost.get('pattern')} ({top_lost.get('confidence')})"
            )
        changes.append(
            {
                "slug": "yolo_delta",
                "title": "YOLO Pattern Churn",
                "detail": " • ".join(yolo_parts),
                "tone": "positive" if new_count > lost_count else ("negative" if lost_count > new_count else "neutral"),
            }
        )

    sector_rows = signals.get("sector_heatmap") or []
    if sector_rows:
        top_sector = sector_rows[0]
        bottom_sector = sector_rows[-1]
        changes.append(
            {
                "slug": "sectors",
                "title": "Sector Rotation",
                "detail": (
                    f"Leader: {top_sector.get('sector')} ({top_sector.get('avg_pct_change')}%) | "
                    f"Laggard: {bottom_sector.get('sector')} ({bottom_sector.get('avg_pct_change')}%)."
                ),
                "tone": "neutral",
            }
        )

    setup_rows = signals.get("setup_quality_top") or []
    if setup_rows:
        best = setup_rows[0]
        setup_cluster = _setup_cluster_rows(setup_rows)
        cluster_family = str(best.get("setup_family") or "").replace("_", " ").strip()
        cluster_label = ", ".join(
            f"{row.get('ticker')} {row.get('score')} ({row.get('setup_tier')})"
            for row in setup_cluster[:3]
        )
        changes.append(
            {
                "slug": "setup",
                "title": "Top Setup Cluster" if len(setup_cluster) > 1 else "Top Setup Candidate",
                "detail": (
                    (
                        f"Leaders: {cluster_label}. "
                        + (f"Shared setup family: {cluster_family}. " if cluster_family else "")
                        if len(setup_cluster) > 1
                        else ""
                    )
                    + f"Highest-rated: {best.get('ticker')} scored {best.get('score')} ({best.get('setup_tier')}) "
                    f"with move {best.get('pct_change')}%, discount {best.get('discount_pct')}%, "
                    f"PEG {best.get('peg')}, ATR {best.get('atr_pct_14') if best.get('atr_pct_14') is not None else '-'}%. "
                    f"Read: {best.get('observation') or 'no technical read yet'} "
                    f"Action: {best.get('action') or 'watch only'}"
                ),
                "tone": "positive",
            }
        )

    while len(changes) < 5:
        changes.append(
            {
                "slug": f"placeholder_{len(changes) + 1}",
                "title": "Signal",
                "detail": "No material change detected for this slot.",
                "tone": "neutral",
            }
        )
    return changes[:5]


def build_no_trade_conditions(report: dict[str, Any]) -> dict[str, Any]:
    warnings = {str(w) for w in (report.get("warnings") or [])}
    signals = report.get("signals") or {}
    breadth = signals.get("market_breadth") or {}
    yolo = report.get("yolo") or {}
    delta_daily = yolo.get("delta_daily") or yolo.get("delta") or {}
    session = report.get("market_session") or {}

    conditions: list[dict[str, Any]] = []

    def add_condition(code: str, severity: str, reason: str) -> None:
        conditions.append({"code": code, "severity": severity, "reason": reason})

    if "price_data_stale" in warnings or "price_data_missing" in warnings:
        add_condition("stale_price_data", "hard", "Price data is stale/missing.")
    if "latest_ingest_run_failed" in warnings:
        add_condition("ingest_failed", "hard", "Latest ingest run failed.")
    if "yolo_data_missing" in warnings:
        add_condition("yolo_missing", "hard", "YOLO detections are missing.")
    if "yolo_data_stale" in warnings:
        add_condition("yolo_stale", "soft", "YOLO detections are stale.")
    if bool(session.get("is_holiday")):
        add_condition(
            "market_holiday",
            "hard",
            f"US market holiday: {session.get('holiday_name') or 'closed'}.",
        )
    if bool(session.get("is_early_close")):
        add_condition(
            "early_close_session",
            "soft",
            f"Early close session: {session.get('early_close_name') or 'shortened day'}.",
        )

    pct_adv = breadth.get("pct_advancing")
    avg_move = breadth.get("avg_pct_change")
    large_moves = breadth.get("large_move_count")
    if (
        isinstance(pct_adv, (int, float))
        and isinstance(avg_move, (int, float))
        and isinstance(large_moves, int)
        and 45.0 <= float(pct_adv) <= 55.0
        and abs(float(avg_move)) <= 0.35
        and int(large_moves) <= 15
    ):
        add_condition(
            "chop_regime",
            "soft",
            (
                "Breadth is mixed and average move is muted "
                f"(adv={pct_adv}%, avg={avg_move}%, large_moves={large_moves})."
            ),
        )

    new_daily = int(delta_daily.get("new_count") or 0)
    lost_daily = int(delta_daily.get("lost_count") or 0)
    if lost_daily >= max(150, new_daily * 8):
        add_condition(
            "high_pattern_churn",
            "soft",
            f"Daily YOLO churn is high (+{new_daily} new / -{lost_daily} lost).",
        )

    hard_count = sum(1 for c in conditions if c.get("severity") == "hard")
    soft_count = sum(1 for c in conditions if c.get("severity") == "soft")
    trade_mode = "blocked" if hard_count > 0 else ("caution" if soft_count > 0 else "normal")
    return {
        "trade_mode": trade_mode,
        "hard_blocks": hard_count,
        "soft_flags": soft_count,
        "conditions": conditions,
    }


def fetch_signals(conn: sqlite3.Connection) -> dict[str, Any]:
    """Market signals for the daily report: 52W extremes, top YOLO patterns, candle signals."""
    signals: dict[str, Any] = {
        "near_52w_high": [],
        "near_52w_low": [],
        "movers_up_today": [],
        "movers_down_today": [],
        "large_moves_today": [],
        "market_breadth": {},
        "volatility_context": {},
        "regime_context": {},
        "yolo_top_today": [],
        "candle_patterns_today": [],
        "sector_heatmap": [],
        "setup_quality_top": [],
        "setup_quality_lookup": {},
        "watchlist_candidates": [],
        "setup_evaluation": {},
        "earnings_catalysts": {},
        "tonight_key_changes": [],
    }
    movers_all: list[dict[str, Any]] = []
    fundamentals_map: dict[str, dict[str, Any]] = {}
    yolo_by_ticker: dict[str, dict[str, Any]] = {}
    vol_by_ticker: dict[str, dict[str, float | None]] = {}
    vol_ctx: dict[str, Any] = {}
    technical_by_ticker: dict[str, dict[str, Any]] = {}
    yolo_history_rows: list[dict[str, Any]] = []
    yolo_asof_dates_desc: dict[str, list[str]] = {"daily": [], "weekly": []}

    try:
        vol_by_ticker, vol_ctx = _fetch_volatility_inputs(conn)
        signals["volatility_context"] = vol_ctx
    except Exception:
        pass
    try:
        signals["regime_context"] = _build_regime_context(conn)
    except Exception:
        signals["regime_context"] = {}
    try:
        technical_by_ticker = _fetch_technical_context(conn)
    except Exception:
        technical_by_ticker = {}

    # ── 52W high / low proximity ─────────────────────────────────────────────
    try:
        rows = conn.execute("""
            WITH latest AS (
                SELECT ticker, MAX(date) AS max_date FROM price_daily GROUP BY ticker
            ),
            latest_close AS (
                SELECT p.ticker, CAST(p.close AS REAL) AS close, p.date
                FROM price_daily p
                JOIN latest l ON p.ticker = l.ticker AND p.date = l.max_date
            ),
            range_52w AS (
                SELECT p.ticker,
                       MAX(CAST(p.high AS REAL)) AS high_52w,
                       MIN(CAST(p.low  AS REAL)) AS low_52w
                FROM price_daily p
                JOIN latest l ON p.ticker = l.ticker
                WHERE p.date >= date(l.max_date, '-365 days')
                GROUP BY p.ticker
            )
            SELECT lc.ticker, lc.close, lc.date, r.high_52w, r.low_52w
            FROM latest_close lc
            JOIN range_52w r ON lc.ticker = r.ticker
            WHERE lc.close > 0 AND r.high_52w > 0 AND r.low_52w > 0
        """).fetchall()
        near_high: list[dict] = []
        near_low: list[dict] = []
        for row in rows:
            ticker, close, _, high_52w, low_52w = row
            close, high_52w, low_52w = float(close), float(high_52w), float(low_52w)
            pct_from_high = (high_52w - close) / high_52w * 100
            pct_from_low  = (close - low_52w) / low_52w * 100
            if pct_from_high <= 3.0:
                near_high.append({"ticker": ticker, "close": round(close, 2),
                                   "high_52w": round(high_52w, 2),
                                   "pct_from_high": round(pct_from_high, 2)})
            if pct_from_low <= 3.0:
                near_low.append({"ticker": ticker, "close": round(close, 2),
                                  "low_52w": round(low_52w, 2),
                                  "pct_from_low": round(pct_from_low, 2)})
        signals["near_52w_high"] = sorted(near_high, key=lambda x: x["pct_from_high"])
        signals["near_52w_low"]  = sorted(near_low,  key=lambda x: x["pct_from_low"])
    except Exception:
        pass

    # ── Large moves vs prior close + breadth snapshot ───────────────────────
    try:
        threshold_raw = os.getenv("TRADER_KOO_REPORT_LARGE_MOVE_PCT", "2.5").strip()
        try:
            large_move_threshold = float(threshold_raw)
        except ValueError:
            large_move_threshold = 2.5
        large_move_threshold = max(0.5, min(50.0, large_move_threshold))

        rows = conn.execute(
            """
            WITH ranked AS (
                SELECT
                    ticker,
                    date,
                    CAST(close AS REAL) AS close,
                    CAST(volume AS REAL) AS volume,
                    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
                FROM price_daily
            ),
            latest AS (
                SELECT ticker, date, close, volume
                FROM ranked
                WHERE rn = 1
            ),
            prev AS (
                SELECT ticker, date AS prev_date, close AS prev_close
                FROM ranked
                WHERE rn = 2
            ),
            range_52w AS (
                SELECT
                    p.ticker,
                    MAX(CAST(p.high AS REAL)) AS high_52w,
                    MIN(CAST(p.low  AS REAL)) AS low_52w
                FROM price_daily p
                JOIN latest l ON l.ticker = p.ticker
                WHERE p.date >= date(l.date, '-365 days')
                GROUP BY p.ticker
            )
            SELECT
                l.ticker,
                l.date AS latest_date,
                l.close,
                p.prev_date,
                p.prev_close,
                l.volume,
                r.high_52w,
                r.low_52w
            FROM latest l
            JOIN prev p ON p.ticker = l.ticker
            LEFT JOIN range_52w r ON r.ticker = l.ticker
            WHERE l.close > 0 AND p.prev_close > 0
            """
        ).fetchall()

        movers: list[dict[str, Any]] = []
        advancers = 0
        decliners = 0
        unchanged = 0
        pct_changes: list[float] = []
        up_threshold = 0.05
        down_threshold = -0.05

        for row in rows:
            ticker = str(row[0])
            latest_date = row[1]
            close = float(row[2])
            prev_close = float(row[4])
            volume = int(float(row[5] or 0))
            high_52w = float(row[6]) if row[6] is not None else None
            low_52w = float(row[7]) if row[7] is not None else None

            pct_change = ((close - prev_close) / prev_close) * 100.0
            pct_changes.append(pct_change)
            if pct_change > up_threshold:
                advancers += 1
            elif pct_change < down_threshold:
                decliners += 1
            else:
                unchanged += 1

            pct_from_high: float | None = None
            pct_from_low: float | None = None
            if high_52w and high_52w > 0:
                pct_from_high = ((high_52w - close) / high_52w) * 100.0
            if low_52w and low_52w > 0:
                pct_from_low = ((close - low_52w) / low_52w) * 100.0

            movers.append(
                {
                    "ticker": ticker,
                    "date": latest_date,
                    "close": round(close, 2),
                    "prev_close": round(prev_close, 2),
                    "pct_change": round(pct_change, 2),
                    "volume": volume,
                    "high_52w": round(high_52w, 2) if high_52w is not None else None,
                    "low_52w": round(low_52w, 2) if low_52w is not None else None,
                    "pct_from_high": round(pct_from_high, 2) if pct_from_high is not None else None,
                    "pct_from_low": round(pct_from_low, 2) if pct_from_low is not None else None,
                    "near_52w_high": bool(pct_from_high is not None and pct_from_high <= 3.0),
                    "near_52w_low": bool(pct_from_low is not None and pct_from_low <= 3.0),
                }
            )

        signals["movers_up_today"] = sorted(
            [m for m in movers if float(m["pct_change"]) > 0.0],
            key=lambda x: float(x["pct_change"]),
            reverse=True,
        )[:20]
        signals["movers_down_today"] = sorted(
            [m for m in movers if float(m["pct_change"]) < 0.0],
            key=lambda x: float(x["pct_change"]),
        )[:20]
        signals["large_moves_today"] = sorted(
            [m for m in movers if abs(float(m["pct_change"])) >= large_move_threshold],
            key=lambda x: abs(float(x["pct_change"])),
            reverse=True,
        )[:40]

        total = advancers + decliners + unchanged
        avg_pct = (sum(pct_changes) / len(pct_changes)) if pct_changes else None
        median_pct: float | None = None
        if pct_changes:
            sorted_changes = sorted(pct_changes)
            n = len(sorted_changes)
            if n % 2 == 1:
                median_pct = sorted_changes[n // 2]
            else:
                median_pct = (sorted_changes[(n // 2) - 1] + sorted_changes[n // 2]) / 2.0
        signals["market_breadth"] = {
            "total_tickers": total,
            "advancers": advancers,
            "decliners": decliners,
            "unchanged": unchanged,
            "pct_advancing": round((advancers / total) * 100.0, 2) if total > 0 else None,
            "avg_pct_change": round(avg_pct, 2) if avg_pct is not None else None,
            "median_pct_change": round(median_pct, 2) if median_pct is not None else None,
            "large_move_threshold_pct": round(large_move_threshold, 2),
            "large_move_count": len(signals["large_moves_today"]),
        }
        movers_all = movers
    except Exception:
        pass

    # ── Top YOLO patterns from today's run ───────────────────────────────────
    try:
        row = conn.execute("SELECT MAX(as_of_date) FROM yolo_patterns").fetchone()
        latest_asof = row[0] if row else None
        if latest_asof:
            hist_rows = conn.execute(
                """
                SELECT ticker, timeframe, pattern, x0_date, x1_date, as_of_date
                FROM yolo_patterns
                WHERE as_of_date IS NOT NULL
                ORDER BY as_of_date DESC
                """
            ).fetchall()
            yolo_history_rows = [
                {
                    "ticker": str(r[0] or ""),
                    "timeframe": str(r[1] or ""),
                    "pattern": str(r[2] or ""),
                    "x0_date": r[3],
                    "x1_date": r[4],
                    "as_of_date": r[5],
                }
                for r in hist_rows
            ]
            for timeframe_key in ("daily", "weekly"):
                dates = {
                    str(row.get("as_of_date") or "")
                    for row in yolo_history_rows
                    if str(row.get("timeframe") or "").strip().lower() == timeframe_key
                    and str(row.get("as_of_date") or "").strip()
                }
                yolo_asof_dates_desc[timeframe_key] = sorted(dates, reverse=True)
            asof_date: dt.date | None = None
            try:
                asof_date = dt.date.fromisoformat(str(latest_asof))
            except Exception:
                asof_date = None
            yolo_rows = conn.execute("""
                SELECT ticker, timeframe, pattern,
                       CAST(confidence AS REAL) AS confidence,
                       x0_date, x1_date
                FROM yolo_patterns
                WHERE as_of_date = ?
                ORDER BY confidence DESC
                LIMIT 30
            """, (latest_asof,)).fetchall()
            yolo_top_today: list[dict[str, Any]] = []
            for r in yolo_rows:
                x1_date = r[5]
                age_days: int | None = None
                if asof_date is not None and x1_date and len(str(x1_date)) >= 10:
                    try:
                        x1_dt = dt.date.fromisoformat(str(x1_date)[:10])
                        age_days = max(0, (asof_date - x1_dt).days)
                    except Exception:
                        age_days = None
                yolo_top_today.append(
                    {
                        "ticker": r[0],
                        "timeframe": r[1],
                        "pattern": r[2],
                        "confidence": round(float(r[3]), 3),
                        "x0_date": r[4],
                        "x1_date": x1_date,
                        "as_of_date": latest_asof,
                        "age_days": age_days,
                        "recency": _yolo_recency_label(age_days, r[1]),
                    }
                )
            for item in yolo_top_today:
                item.update(
                    _summarize_yolo_lifecycle(
                        item,
                        yolo_history_rows,
                        yolo_asof_dates_desc.get(str(item.get("timeframe") or "").strip().lower(), []),
                    )
                )
            yolo_top_today.sort(
                key=lambda item: (
                    _yolo_age_factor(item.get("age_days"), item.get("timeframe")),
                    int(item.get("current_streak") or 0),
                    1 if str(item.get("timeframe") or "").strip().lower() == "daily" else 0,
                    float(item.get("confidence") or 0.0),
                    -(int(item.get("age_days")) if isinstance(item.get("age_days"), int) else 9999),
                ),
                reverse=True,
            )
            signals["yolo_top_today"] = yolo_top_today

            # Best YOLO signal per ticker for setup scoring.
            full_rows = conn.execute(
                """
                SELECT ticker, timeframe, pattern,
                       CAST(confidence AS REAL) AS confidence,
                       x0_date, x1_date
                FROM yolo_patterns
                WHERE as_of_date = ?
                ORDER BY confidence DESC
                """,
                (latest_asof,),
            ).fetchall()
            for r in full_rows:
                ticker = str(r[0])
                conf = float(r[3] or 0.0)
                x1_date = r[5]
                age_days: int | None = None
                if asof_date is not None and x1_date and len(str(x1_date)) >= 10:
                    try:
                        x1_dt = dt.date.fromisoformat(str(x1_date)[:10])
                        age_days = max(0, (asof_date - x1_dt).days)
                    except Exception:
                        age_days = None
                candidate = {
                    "ticker": ticker,
                    "timeframe": r[1],
                    "pattern": r[2],
                    "confidence": round(conf, 3),
                    "x0_date": r[4],
                    "x1_date": x1_date,
                    "as_of_date": latest_asof,
                    "age_days": age_days,
                }
                candidate.update(
                    _summarize_yolo_lifecycle(
                        candidate,
                        yolo_history_rows,
                        yolo_asof_dates_desc.get(str(candidate.get("timeframe") or "").strip().lower(), []),
                    )
                )
                prev = yolo_by_ticker.get(ticker)
                if prev is None:
                    yolo_by_ticker[ticker] = candidate
                    continue
                prev_daily = str(prev.get("timeframe") or "") == "daily"
                cand_daily = str(candidate.get("timeframe") or "") == "daily"
                if cand_daily and not prev_daily:
                    yolo_by_ticker[ticker] = candidate
                    continue
                prev_rank = (
                    _yolo_age_factor(prev.get("age_days"), prev.get("timeframe")),
                    int(prev.get("current_streak") or 0),
                    float(prev.get("confidence") or 0.0),
                )
                cand_rank = (
                    _yolo_age_factor(candidate.get("age_days"), candidate.get("timeframe")),
                    int(candidate.get("current_streak") or 0),
                    float(candidate.get("confidence") or 0.0),
                )
                if cand_daily == prev_daily and cand_rank > prev_rank:
                    yolo_by_ticker[ticker] = candidate
    except Exception:
        pass

    # ── Fundamentals snapshot map (discount/PEG + sector/industry metadata) ─
    try:
        snap_row = conn.execute("SELECT MAX(snapshot_ts) FROM finviz_fundamentals").fetchone()
        latest_snap = snap_row[0] if snap_row else None
        if latest_snap:
            fund_rows = conn.execute(
                """
                SELECT ticker, discount_pct, peg, raw_json
                FROM finviz_fundamentals
                WHERE snapshot_ts = ?
                """,
                (latest_snap,),
            ).fetchall()
            for r in fund_rows:
                ticker = str(r[0])
                discount = float(r[1]) if r[1] is not None else None
                peg = float(r[2]) if r[2] is not None else None
                sector = None
                industry = None
                raw = r[3]
                if raw:
                    try:
                        raw_obj = json.loads(str(raw))
                        if isinstance(raw_obj, dict):
                            sector = raw_obj.get("Sector") or raw_obj.get("sector")
                            industry = raw_obj.get("Industry") or raw_obj.get("industry")
                    except Exception:
                        pass
                fundamentals_map[ticker] = {
                    "discount_pct": round(discount, 2) if discount is not None else None,
                    "peg": round(peg, 2) if peg is not None else None,
                    "sector": str(sector).strip() if sector else "Unknown",
                    "industry": str(industry).strip() if industry else None,
                }
    except Exception:
        pass

    # ── Sector heatmap + setup quality scoring ───────────────────────────────
    try:
        sector_buckets: dict[str, dict[str, Any]] = {}
        setup_rows: list[dict[str, Any]] = []
        for m in movers_all:
            ticker = str(m.get("ticker") or "").upper()
            if not ticker:
                continue
            pct_change = float(m.get("pct_change") or 0.0)
            near_high = bool(m.get("near_52w_high"))
            near_low = bool(m.get("near_52w_low"))
            fund = fundamentals_map.get(ticker, {})
            sector = str(fund.get("sector") or "Unknown").strip() or "Unknown"

            bucket = sector_buckets.setdefault(
                sector,
                {
                    "sector": sector,
                    "tickers": 0,
                    "advancers": 0,
                    "decliners": 0,
                    "unchanged": 0,
                    "near_high_count": 0,
                    "near_low_count": 0,
                    "_changes": [],
                },
            )
            bucket["tickers"] += 1
            bucket["_changes"].append(pct_change)
            if pct_change > 0.05:
                bucket["advancers"] += 1
            elif pct_change < -0.05:
                bucket["decliners"] += 1
            else:
                bucket["unchanged"] += 1
            if near_high:
                bucket["near_high_count"] += 1
            if near_low:
                bucket["near_low_count"] += 1

            # Setup quality score: valuation + momentum + AI signal freshness.
            score = 50.0
            discount = fund.get("discount_pct")
            discount_component = 0.0
            if isinstance(discount, (int, float)):
                discount_component = _clamp(float(discount) * 0.8, -20.0, 20.0)
                score += discount_component

            peg = fund.get("peg")
            peg_component = 0.0
            if isinstance(peg, (int, float)) and float(peg) > 0:
                peg_v = float(peg)
                if peg_v <= 0.8:
                    peg_component = 15.0
                elif peg_v <= 1.5:
                    peg_component = 10.0
                elif peg_v <= 2.5:
                    peg_component = 4.0
                elif peg_v <= 4.0:
                    peg_component = -4.0
                else:
                    peg_component = -8.0
                score += peg_component

            momentum_component = _clamp(pct_change * 1.5, -12.0, 12.0)
            score += momentum_component

            proximity_component = 0.0
            if near_high and pct_change > 0:
                proximity_component += 5.0
            if near_low and pct_change < 0:
                proximity_component -= 5.0
            score += proximity_component

            yolo = yolo_by_ticker.get(ticker)
            vol = vol_by_ticker.get(ticker, {})
            tech = technical_by_ticker.get(ticker, {})
            yolo_component = 0.0
            volatility_component = 0.0
            yolo_pattern = None
            yolo_confidence = None
            yolo_age_days = None
            yolo_timeframe = None
            yolo_first_seen_asof = None
            yolo_last_seen_asof = None
            yolo_snapshots_seen = None
            yolo_current_streak = None
            yolo_first_seen_days_ago = None
            signal_bias = "neutral"
            atr_pct_14 = vol.get("atr_pct_14")
            realized_vol_20 = vol.get("realized_vol_20")
            bb_width_20 = vol.get("bb_width_20")
            if yolo:
                yolo_pattern = yolo.get("pattern")
                yolo_confidence = yolo.get("confidence")
                yolo_age_days = yolo.get("age_days")
                yolo_timeframe = yolo.get("timeframe")
                yolo_first_seen_asof = yolo.get("first_seen_asof")
                yolo_last_seen_asof = yolo.get("last_seen_asof")
                yolo_snapshots_seen = yolo.get("snapshots_seen")
                yolo_current_streak = yolo.get("current_streak")
                yolo_first_seen_days_ago = yolo.get("first_seen_days_ago")
                signal_bias = _yolo_pattern_bias(yolo_pattern)
                conf = float(yolo_confidence or 0.0)
                yolo_component += _clamp(conf * 20.0, 0.0, 18.0)
                if str(yolo_timeframe) == "daily":
                    yolo_component += 2.0
                if isinstance(yolo_age_days, int):
                    if yolo_age_days <= 10:
                        yolo_component += 3.0
                    elif yolo_age_days <= 30:
                        yolo_component += 1.0
                score += yolo_component

            if isinstance(atr_pct_14, (int, float)):
                atr_v = float(atr_pct_14)
                if atr_v < 1.0:
                    volatility_component -= 2.0
                elif atr_v <= 4.5:
                    volatility_component += 5.0
                elif atr_v <= 7.0:
                    volatility_component += 2.0
                elif atr_v <= 10.0:
                    volatility_component -= 2.0
                else:
                    volatility_component -= 6.0

            if isinstance(bb_width_20, (int, float)):
                bb_v = float(bb_width_20)
                if bb_v <= 6.0 and abs(pct_change) >= 1.5:
                    volatility_component += 3.0
                elif bb_v >= 18.0:
                    volatility_component -= 3.0

            if isinstance(realized_vol_20, (int, float)):
                rv_v = float(realized_vol_20)
                if rv_v > 60.0:
                    volatility_component -= 3.0
                elif rv_v >= 20.0:
                    volatility_component += 2.0
                elif rv_v < 12.0:
                    volatility_component -= 1.0

            vix_pctile = vol_ctx.get("vix_percentile_1y")
            if isinstance(vix_pctile, (int, float)):
                vix_p = float(vix_pctile)
                if vix_p >= 90.0:
                    volatility_component -= 4.0
                elif vix_p >= 75.0:
                    volatility_component -= 2.0
                elif vix_p <= 35.0:
                    volatility_component += 1.0

            volatility_component = _clamp(volatility_component, -15.0, 15.0)
            score += volatility_component

            row = {
                "ticker": ticker,
                "score": round(_clamp(score, 0.0, 100.0), 1),
                "confluence_score": round(_clamp(score, 0.0, 100.0), 1),
                "setup_tier": _setup_tier(round(_clamp(score, 0.0, 100.0), 1)),
                "sector": sector,
                "pct_change": round(pct_change, 2),
                "discount_pct": discount,
                "peg": peg,
                "atr_pct_14": atr_pct_14,
                "realized_vol_20": realized_vol_20,
                "bb_width_20": bb_width_20,
                "avg_volume_20": tech.get("avg_volume_20"),
                "volume_ratio_20": tech.get("volume_ratio_20"),
                "recent_range_pct_10": tech.get("recent_range_pct_10"),
                "recent_range_pct_20": tech.get("recent_range_pct_20"),
                "near_52w_high": near_high,
                "near_52w_low": near_low,
                "yolo_pattern": yolo_pattern,
                "yolo_confidence": yolo_confidence,
                "yolo_age_days": yolo_age_days,
                "yolo_timeframe": yolo_timeframe,
                "yolo_first_seen_asof": yolo_first_seen_asof,
                "yolo_last_seen_asof": yolo_last_seen_asof,
                "yolo_snapshots_seen": yolo_snapshots_seen,
                "yolo_current_streak": yolo_current_streak,
                "yolo_first_seen_days_ago": yolo_first_seen_days_ago,
                "signal_bias": signal_bias,
                "close": tech.get("close"),
                "ma20": tech.get("ma20"),
                "ma50": tech.get("ma50"),
                "pct_vs_ma20": tech.get("pct_vs_ma20"),
                "pct_vs_ma50": tech.get("pct_vs_ma50"),
                "pct_from_20d_high": tech.get("pct_from_20d_high"),
                "pct_from_20d_low": tech.get("pct_from_20d_low"),
                "trend_state": tech.get("trend_state") or "mixed",
                "level_context": tech.get("level_context") or "mid_range",
                "support_level": tech.get("support_level"),
                "support_zone_low": tech.get("support_zone_low"),
                "support_zone_high": tech.get("support_zone_high"),
                "support_tier": tech.get("support_tier"),
                "support_touches": tech.get("support_touches"),
                "resistance_level": tech.get("resistance_level"),
                "resistance_zone_low": tech.get("resistance_zone_low"),
                "resistance_zone_high": tech.get("resistance_zone_high"),
                "resistance_tier": tech.get("resistance_tier"),
                "resistance_touches": tech.get("resistance_touches"),
                "pct_to_support": tech.get("pct_to_support"),
                "pct_to_resistance": tech.get("pct_to_resistance"),
                "range_position": tech.get("range_position"),
                "stretch_state": tech.get("stretch_state") or "normal",
                "breakout_state": tech.get("breakout_state") or "none",
                "structure_state": tech.get("structure_state") or "normal",
                "candle_pattern": None,
                "candle_bias": "neutral",
                "candle_confidence": None,
                "observation": "",
                "actionability": "watch-only",
                "action": "",
                "risk_note": "",
                "technical_read": "",
                "components": {
                    "discount": round(discount_component, 2),
                    "peg": round(peg_component, 2),
                    "momentum": round(momentum_component, 2),
                    "proximity": round(proximity_component, 2),
                    "volatility": round(volatility_component, 2),
                    "yolo": round(yolo_component, 2),
                },
            }
            row.update(_score_setup_from_confluence(row))
            setup_rows.append(row)

        sector_rows: list[dict[str, Any]] = []
        for _, bucket in sector_buckets.items():
            changes = [float(x) for x in bucket.pop("_changes", [])]
            if not changes:
                continue
            changes_sorted = sorted(changes)
            n = len(changes_sorted)
            if n % 2 == 1:
                median_change = changes_sorted[n // 2]
            else:
                median_change = (changes_sorted[(n // 2) - 1] + changes_sorted[n // 2]) / 2.0
            tickers = int(bucket.get("tickers") or 0)
            advancers = int(bucket.get("advancers") or 0)
            bucket["avg_pct_change"] = round(sum(changes) / len(changes), 2)
            bucket["median_pct_change"] = round(median_change, 2)
            bucket["pct_advancing"] = round((advancers / tickers) * 100.0, 2) if tickers > 0 else None
            sector_rows.append(bucket)

        sector_rows.sort(
            key=lambda r: (
                float(r.get("avg_pct_change") or 0.0),
                float(r.get("pct_advancing") or 0.0),
                int(r.get("tickers") or 0),
            ),
            reverse=True,
        )
        signals["sector_heatmap"] = sector_rows

        setup_rows.sort(
            key=lambda r: (
                float(r.get("score") or 0.0),
                float(r.get("pct_change") or 0.0),
                float(r.get("discount_pct") or -999.0),
            ),
            reverse=True,
        )
        _apply_debate_payload(setup_rows)
        _apply_debate_guardrails(setup_rows)
        _apply_llm_narrative_overrides(setup_rows, source="daily_report")
        signals["setup_quality_top"] = setup_rows[:40]
        signals["watchlist_candidates"] = [
            {
                "ticker": r.get("ticker"),
                "score": r.get("score"),
                "setup_tier": r.get("setup_tier"),
                "pct_change": r.get("pct_change"),
                "yolo_pattern": r.get("yolo_pattern"),
                "yolo_confidence": r.get("yolo_confidence"),
            }
            for r in setup_rows[:20]
        ]
    except Exception:
        pass

    # ── Candle patterns on latest close date ─────────────────────────────────
    try:
        import pandas as pd
        from trader_koo.features.candle_patterns import (
            CandlePatternConfig,
            detect_candlestick_patterns,
        )

        row = conn.execute("SELECT MAX(date) FROM price_daily").fetchone()
        latest_date = row[0] if row else None
        if latest_date:
            raw_rows = conn.execute("""
                SELECT ticker, date, open, high, low, close
                FROM (
                    SELECT ticker, date, open, high, low, close,
                           ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
                    FROM price_daily
                )
                WHERE rn <= 10
                ORDER BY ticker, date
            """).fetchall()

            ticker_rows: dict = defaultdict(list)
            for r in raw_rows:
                ticker_rows[r[0]].append({
                    "date": r[1], "open": r[2], "high": r[3], "low": r[4], "close": r[5],
                })

            cfg = CandlePatternConfig(lookback_bars=10, use_talib=False)
            candle_signals: list[dict] = []
            candle_by_ticker: dict[str, dict[str, Any]] = {}
            for ticker, candles in ticker_rows.items():
                df = pd.DataFrame(candles)
                try:
                    result = detect_candlestick_patterns(df, cfg)
                    if not result.empty:
                        on_latest = result[result["date"] == latest_date]
                        for _, r in on_latest.iterrows():
                            candle_signals.append({
                                "ticker": ticker,
                                "pattern": str(r["pattern"]),
                                "bias": str(r.get("bias", "neutral")),
                                "confidence": round(float(r.get("confidence", 0.5)), 2),
                            })
                except Exception:
                    pass

            candle_signals.sort(key=lambda x: x["confidence"], reverse=True)
            signals["candle_patterns_today"] = candle_signals[:60]
            for row in candle_signals:
                ticker = str(row.get("ticker") or "").upper()
                prev = candle_by_ticker.get(ticker)
                if prev is None or float(row.get("confidence") or 0.0) > float(prev.get("confidence") or 0.0):
                    candle_by_ticker[ticker] = row

            if isinstance(setup_rows, list):
                for row in setup_rows:
                    ticker = str(row.get("ticker") or "").upper()
                    candle = candle_by_ticker.get(ticker) or {}
                    if candle:
                        row["candle_pattern"] = candle.get("pattern")
                        row["candle_bias"] = candle.get("bias") or "neutral"
                        row["candle_confidence"] = candle.get("confidence")
                    row.update(_score_setup_from_confluence(row))
                    readout = _describe_setup(row)
                    row.update(readout)

                _apply_debate_payload(setup_rows)
                _apply_debate_guardrails(setup_rows)

                setup_rows.sort(
                    key=lambda r: (
                        float(r.get("score") or 0.0),
                        float(r.get("confirmation_count") or 0.0),
                        -float(r.get("contradiction_count") or 0.0),
                        float(r.get("pct_change") or 0.0),
                    ),
                    reverse=True,
                )

                _apply_llm_narrative_overrides(setup_rows, source="daily_report")
                signals["setup_quality_top"] = setup_rows[:40]
                signals["watchlist_candidates"] = [
                    {
                        "ticker": r.get("ticker"),
                        "score": r.get("score"),
                        "setup_tier": r.get("setup_tier"),
                        "pct_change": r.get("pct_change"),
                        "yolo_pattern": r.get("yolo_pattern"),
                        "yolo_confidence": r.get("yolo_confidence"),
                        "signal_bias": r.get("signal_bias"),
                        "observation": r.get("observation"),
                        "actionability": r.get("actionability"),
                        "action": r.get("action"),
                        "risk_note": r.get("risk_note"),
                        "technical_read": r.get("technical_read"),
                        "candle_pattern": r.get("candle_pattern"),
                        "candle_bias": r.get("candle_bias"),
                        "candle_confidence": r.get("candle_confidence"),
                    }
                    for r in setup_rows[:20]
                ]
                signals["setup_quality_lookup"] = {
                    str(row.get("ticker") or "").upper(): {
                        key: row.get(key)
                        for key in (
                            "ticker",
                            "score",
                            "setup_tier",
                            "setup_family",
                            "sector",
                            "pct_change",
                            "discount_pct",
                            "peg",
                            "atr_pct_14",
                            "realized_vol_20",
                            "bb_width_20",
                            "signal_bias",
                            "actionability",
                            "observation",
                            "action",
                            "risk_note",
                            "technical_read",
                            "confirmation_count",
                            "contradiction_count",
                            "valuation_bias",
                            "candle_pattern",
                            "candle_bias",
                            "candle_confidence",
                            "yolo_pattern",
                            "yolo_bias",
                            "yolo_confidence",
                            "yolo_age_days",
                            "yolo_timeframe",
                            "yolo_recency",
                            "yolo_direction_conflict",
                            "yolo_conflict_strength",
                            "debate_v1",
                            "debate_consensus_state",
                            "debate_consensus_bias",
                            "debate_agreement_score",
                            "debate_disagreement_count",
                            "debate_safety_adjustment",
                            "trend_state",
                            "level_context",
                            "support_level",
                            "resistance_level",
                            "pct_to_support",
                            "pct_to_resistance",
                            "trend_state",
                            "level_context",
                            "candle_pattern",
                            "candle_bias",
                            "candle_confidence",
                        )
                    }
                    for row in setup_rows
                    if isinstance(row, dict) and row.get("ticker")
                }
    except Exception:
        pass

    return signals


def fetch_report_payload(
    db_path: Path,
    run_log: Path,
    tail_lines: int,
    report_kind: str = "daily",
) -> dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc)
    report_kind_norm = _normalize_report_kind(report_kind)
    payload: dict[str, Any] = {
        "generated_ts": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "meta": {
            "report_kind": report_kind_norm,
            "llm": llm_status(),
        },
        "db_path": str(db_path),
        "db_exists": db_path.exists(),
        "ok": False,
        "warnings": [],
        "counts": {},
        "freshness": {},
        "market_session": market_calendar_context(now),
        "risk_filters": {},
        "latest_data": {},
        "latest_ingest_run": {},
        "yolo": {
            "table_exists": False,
            "summary": {},
            "timeframes": [],
            "delta": {},
            "delta_daily": {},
            "delta_weekly": {},
            "persistence": {},
        },
        "cron_log_path": str(run_log),
        "cron_log_tail": tail_text(run_log, lines=tail_lines),
    }
    if not db_path.exists():
        payload["warnings"].append("database_missing")
        return payload

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        counts = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM price_daily) AS price_rows,
                (SELECT COUNT(*) FROM finviz_fundamentals) AS fundamentals_rows,
                (SELECT COUNT(*) FROM options_iv) AS options_rows,
                (SELECT COUNT(*) FROM yolo_patterns) AS yolo_rows,
                (SELECT COUNT(DISTINCT ticker) FROM price_daily) AS tracked_tickers,
                (SELECT MAX(date) FROM price_daily) AS latest_price_date,
                (SELECT MAX(snapshot_ts) FROM finviz_fundamentals) AS latest_fund_snapshot,
                (SELECT MAX(snapshot_ts) FROM options_iv) AS latest_opt_snapshot,
                (SELECT MAX(detected_ts) FROM yolo_patterns) AS latest_yolo_ts
            """
        ).fetchone()
        payload["counts"] = {
            "tracked_tickers": int((counts["tracked_tickers"] or 0) if counts else 0),
            "price_rows": int((counts["price_rows"] or 0) if counts else 0),
            "fundamentals_rows": int((counts["fundamentals_rows"] or 0) if counts else 0),
            "options_rows": int((counts["options_rows"] or 0) if counts else 0),
            "yolo_rows": int((counts["yolo_rows"] or 0) if counts else 0),
        }
        payload["latest_data"] = {
            "price_date": counts["latest_price_date"] if counts else None,
            "fund_snapshot": counts["latest_fund_snapshot"] if counts else None,
            "options_snapshot": counts["latest_opt_snapshot"] if counts else None,
            "yolo_detected_ts": counts["latest_yolo_ts"] if counts else None,
        }
        price_age = days_since_date(counts["latest_price_date"], now) if counts else None
        fund_age = hours_since(counts["latest_fund_snapshot"], now) if counts else None
        opt_age = hours_since(counts["latest_opt_snapshot"], now) if counts else None
        yolo_age = hours_since(counts["latest_yolo_ts"], now) if counts else None
        payload["freshness"] = {
            "price_age_days": None if price_age is None else round(price_age, 2),
            "fund_age_hours": None if fund_age is None else round(fund_age, 2),
            "opt_age_hours": None if opt_age is None else round(opt_age, 2),
            "yolo_age_hours": None if yolo_age is None else round(yolo_age, 2),
        }

        if table_exists(conn, "ingest_runs"):
            latest_run = conn.execute(
                """
                SELECT run_id, started_ts, finished_ts, status, tickers_total, tickers_ok, tickers_failed, error_message
                FROM ingest_runs
                ORDER BY started_ts DESC
                LIMIT 1
                """
            ).fetchone()
            payload["latest_ingest_run"] = row_to_dict(latest_run)
            if latest_run and str(latest_run["status"] or "").lower() == "failed":
                payload["warnings"].append("latest_ingest_run_failed")
        else:
            payload["warnings"].append("ingest_runs_missing")

        if table_exists(conn, "yolo_patterns"):
            payload["yolo"]["table_exists"] = True
            yolo_summary = conn.execute(
                """
                SELECT
                    COUNT(*) AS rows_total,
                    COUNT(DISTINCT ticker) AS tickers_with_patterns,
                    MAX(detected_ts) AS latest_detected_ts,
                    MAX(as_of_date) AS latest_asof_date
                FROM yolo_patterns
                """
            ).fetchone()
            payload["yolo"]["summary"] = row_to_dict(yolo_summary)
            tf_rows = conn.execute(
                """
                SELECT
                    timeframe,
                    COUNT(*) AS rows_total,
                    COUNT(DISTINCT ticker) AS tickers_with_patterns,
                    AVG(confidence) AS avg_confidence,
                    MAX(detected_ts) AS latest_detected_ts,
                    MAX(as_of_date) AS latest_asof_date
                FROM yolo_patterns
                GROUP BY timeframe
                ORDER BY timeframe
                """
            ).fetchall()
            payload["yolo"]["timeframes"] = [dict(r) for r in tf_rows]
        else:
            payload["warnings"].append("yolo_patterns_missing")

        # Market signals (52W extremes, movers, sector/quality overlays, AI/candles)
        payload["signals"] = fetch_signals(conn)
        if SETUP_EVAL_ENABLED:
            eval_summary: dict[str, Any] = {"enabled": True, "scored_calls": 0, "open_calls": 0}
            try:
                ensure_setup_call_eval_schema(conn)
                signals_ref = payload.get("signals") if isinstance(payload.get("signals"), dict) else {}
                setup_rows_ref = signals_ref.get("setup_quality_top") if isinstance(signals_ref.get("setup_quality_top"), list) else []
                asof_for_eval = str((payload.get("latest_data") or {}).get("price_date") or "").strip()
                inserted_calls = 0
                if asof_for_eval and setup_rows_ref:
                    inserted_calls = _persist_setup_call_candidates(
                        conn,
                        generated_ts=str(payload.get("generated_ts") or ""),
                        report_kind=report_kind_norm,
                        asof_date=asof_for_eval,
                        setup_rows=setup_rows_ref,
                    )
                scored_this_run = _score_open_setup_call_outcomes(conn)
                conn.commit()
                summary, reliability_lookup = _summarize_setup_call_evaluations(
                    conn,
                    window_days=SETUP_EVAL_WINDOW_DAYS,
                    min_sample=SETUP_EVAL_MIN_SAMPLE,
                )
                calibration = _apply_setup_eval_fields(
                    setup_rows_ref,
                    reliability_lookup=reliability_lookup,
                    min_sample=SETUP_EVAL_MIN_SAMPLE,
                    hit_threshold_pct=SETUP_EVAL_HIT_THRESHOLD_PCT,
                )
                setup_rows_ref.sort(
                    key=lambda r: (
                        float(r.get("score") or 0.0),
                        float(r.get("confirmation_count") or 0.0),
                        -float(r.get("contradiction_count") or 0.0),
                        float(r.get("pct_change") or 0.0),
                    ),
                    reverse=True,
                )
                _refresh_setup_eval_surfaces(signals_ref)
                eval_summary = summary or {"enabled": True}
                eval_summary["tracked_this_run"] = int(min(len(setup_rows_ref), SETUP_EVAL_TRACK_LIMIT))
                eval_summary["inserted_calls"] = int(inserted_calls)
                eval_summary["scored_this_run"] = int(scored_this_run)
                eval_summary["calibration"] = calibration
            except Exception:
                conn.rollback()
                eval_summary = {
                    "enabled": True,
                    "error": "setup_evaluation_failed",
                }
            payload["signals"]["setup_evaluation"] = eval_summary
        else:
            payload["signals"]["setup_evaluation"] = {"enabled": False, "reason": "disabled_by_env"}
        market_date_raw = str((payload.get("market_session") or {}).get("market_date") or "").strip()
        market_date_obj: dt.date | None = None
        try:
            if market_date_raw:
                market_date_obj = dt.date.fromisoformat(market_date_raw)
        except ValueError:
            market_date_obj = None
        if market_date_obj is not None:
            try:
                payload["signals"]["earnings_catalysts"] = build_earnings_calendar_payload(
                    conn,
                    market_date=market_date_obj,
                    days=14 if report_kind_norm == "daily" else 21,
                    limit=120,
                    tickers=None,
                    setup_map=(payload.get("signals") or {}).get("setup_quality_lookup") or {},
                )
            except Exception:
                payload["signals"]["earnings_catalysts"] = {}

        # YOLO deltas and persistence by timeframe.
        delta_daily = fetch_yolo_delta(conn, timeframe="daily")
        delta_weekly = fetch_yolo_delta(conn, timeframe="weekly")
        payload["yolo"]["delta_daily"] = delta_daily
        payload["yolo"]["delta_weekly"] = delta_weekly
        payload["yolo"]["delta"] = delta_weekly if report_kind_norm == "weekly" else delta_daily
        payload["yolo"]["persistence"] = {
            "daily": fetch_yolo_pattern_persistence(conn, timeframe="daily", lookback_asof=20, top_n=25),
            "weekly": fetch_yolo_pattern_persistence(conn, timeframe="weekly", lookback_asof=20, top_n=25),
        }
        try:
            payload["signals"]["tonight_key_changes"] = build_tonight_key_changes(
                payload["signals"], payload["yolo"]["delta"]
            )
        except Exception:
            payload["signals"]["tonight_key_changes"] = []

        # Refresh LLM status after signal narrative pass so this snapshot reflects current runtime state.
        try:
            payload["meta"]["llm"] = llm_status()
        except Exception:
            payload["meta"]["llm"] = {}

        # Basic health guardrails.
        price_age = payload["freshness"]["price_age_days"]
        fund_age  = payload["freshness"]["fund_age_hours"]
        yolo_age  = payload["freshness"]["yolo_age_hours"]
        if price_age is None:
            payload["warnings"].append("price_data_missing")
        elif isinstance(price_age, (int, float)) and price_age > 3:
            payload["warnings"].append("price_data_stale")
        if fund_age is None:
            payload["warnings"].append("fundamentals_missing")
        elif isinstance(fund_age, (int, float)) and fund_age > 48:
            payload["warnings"].append("fundamentals_stale")
        if yolo_age is None:
            payload["warnings"].append("yolo_data_missing")
        elif isinstance(yolo_age, (int, float)) and yolo_age > 30:
            payload["warnings"].append("yolo_data_stale")

        llm_meta = payload.get("meta", {}).get("llm", {}) if isinstance(payload.get("meta"), dict) else {}
        llm_health = llm_meta.get("health") if isinstance(llm_meta, dict) and isinstance(llm_meta.get("health"), dict) else {}
        if llm_enabled():
            if isinstance(llm_meta, dict) and not llm_meta.get("ready"):
                payload["warnings"].append("llm_not_ready")
            if isinstance(llm_meta, dict) and llm_meta.get("runtime_disabled"):
                payload["warnings"].append("llm_runtime_cooldown")
            if llm_health.get("degraded"):
                payload["warnings"].append("llm_degraded")

        payload["risk_filters"] = build_no_trade_conditions(payload)
        payload["ok"] = len(payload["warnings"]) == 0
        return payload
    finally:
        conn.close()


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
            print(f"[EMAIL] failed {exc}")
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
        print(f"[LLM-ALERT] failed {exc}")

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
