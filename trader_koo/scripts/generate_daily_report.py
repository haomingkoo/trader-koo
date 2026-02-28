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
from collections import defaultdict
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from trader_koo.report_email import (
    build_report_email_bodies,
    build_report_email_subject,
    report_email_app_url,
)


MARKET_TZ_NAME = os.getenv("TRADER_KOO_MARKET_TZ", "America/New_York")
try:
    MARKET_TZ = ZoneInfo(MARKET_TZ_NAME)
except Exception:
    MARKET_TZ = dt.timezone.utc
MARKET_CLOSE_HOUR = min(23, max(0, int(os.getenv("TRADER_KOO_MARKET_CLOSE_HOUR", "16"))))
TRUTHY_VALUES = {"1", "true", "yes", "on"}


def _as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in TRUTHY_VALUES


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


def _send_resend_email(subject: str, text: str, html_body: str, resend: dict[str, Any]) -> None:
    user_agent = os.getenv("TRADER_KOO_EMAIL_USER_AGENT", "trader-koo/1.0")
    payload = {
        "from": resend["from_email"],
        "to": [resend["to_email"]],
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


def send_report_email(report: dict[str, Any], md_text: str) -> None:
    """Send the daily report email. Raises on missing config or delivery failure."""
    transport = _email_transport()
    smtp = _smtp_cfg()
    resend = _resend_cfg()
    missing = []
    if transport == "resend":
        if not resend["api_key"]:
            missing.append("TRADER_KOO_RESEND_API_KEY")
        if not resend["from_email"]:
            missing.append("TRADER_KOO_RESEND_FROM (or TRADER_KOO_SMTP_FROM)")
        if not resend["to_email"]:
            missing.append("TRADER_KOO_REPORT_EMAIL_TO")
    else:
        if not smtp["host"]:
            missing.append("TRADER_KOO_SMTP_HOST")
        if not smtp["from_email"]:
            missing.append("TRADER_KOO_SMTP_FROM")
        if not smtp["to_email"]:
            missing.append("TRADER_KOO_REPORT_EMAIL_TO")
    if missing:
        raise RuntimeError(f"Missing email env vars for {transport}: {', '.join(missing)}")

    generated = report.get("generated_ts", "unknown")
    subject = build_report_email_subject(report)
    text_body, html_body = build_report_email_bodies(
        report,
        md_text,
        app_url=report_email_app_url(),
    )

    if transport == "resend":
        _send_resend_email(subject=subject, text=text_body, html_body=html_body, resend=resend)
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp["from_email"]
    msg["To"] = smtp["to_email"]
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")
    msg.add_attachment(md_text.encode(), maintype="text", subtype="markdown",
                       filename=f"daily_report_{generated[:10]}.md")

    host, port, timeout_sec = smtp["host"], int(smtp["port"]), int(smtp["timeout_sec"])
    user, password, security = smtp["user"], smtp["password"], smtp["security"]

    if security == "ssl":
        with smtplib.SMTP_SSL(host, port, timeout=timeout_sec, context=ssl.create_default_context()) as server:
            if user:
                server.login(user, password)
            server.send_message(msg)
        return

    with smtplib.SMTP(host, port, timeout=timeout_sec) as server:
        server.ehlo()
        if security == "starttls":
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
        if user:
            server.login(user, password)
        server.send_message(msg)


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
        if len(dates) < 2:
            return delta
        today_asof = dates[0][0]
        prev_asof = dates[1][0]
        delta["today_asof"] = today_asof
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


def _fetch_technical_context(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Build lightweight trend/level context for per-ticker report interpretation."""
    by_ticker: dict[str, dict[str, Any]] = {}
    if not table_exists(conn, "price_daily"):
        return by_ticker

    rows = conn.execute(
        """
        SELECT ticker, date, CAST(open AS REAL), CAST(high AS REAL), CAST(low AS REAL), CAST(close AS REAL)
        FROM (
            SELECT ticker, date, open, high, low, close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM price_daily
        )
        WHERE rn <= 80
        ORDER BY ticker, date
        """
    ).fetchall()
    bucket: dict[str, list[tuple[str, float, float, float, float]]] = defaultdict(list)
    for r in rows:
        ticker = str(r[0] or "").upper().strip()
        if not ticker:
            continue
        try:
            open_v = float(r[2])
            high_v = float(r[3])
            low_v = float(r[4])
            close_v = float(r[5])
        except (TypeError, ValueError):
            continue
        if min(open_v, high_v, low_v, close_v) <= 0:
            continue
        bucket[ticker].append((str(r[1]), open_v, high_v, low_v, close_v))

    for ticker, bars in bucket.items():
        if len(bars) < 20:
            continue
        closes = [b[4] for b in bars]
        highs = [b[2] for b in bars]
        lows = [b[3] for b in bars]
        close_now = closes[-1]
        ma20 = sum(closes[-20:]) / 20.0 if len(closes) >= 20 else None
        ma50 = sum(closes[-50:]) / 50.0 if len(closes) >= 50 else None
        recent_high_20 = max(highs[-20:]) if len(highs) >= 20 else None
        recent_low_20 = min(lows[-20:]) if len(lows) >= 20 else None

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

        by_ticker[ticker] = {
            "close": round(close_now, 2),
            "ma20": round(ma20, 2) if ma20 is not None else None,
            "ma50": round(ma50, 2) if ma50 is not None else None,
            "pct_vs_ma20": round(pct_vs_ma20, 2) if pct_vs_ma20 is not None else None,
            "pct_vs_ma50": round(pct_vs_ma50, 2) if pct_vs_ma50 is not None else None,
            "recent_high_20": round(recent_high_20, 2) if recent_high_20 is not None else None,
            "recent_low_20": round(recent_low_20, 2) if recent_low_20 is not None else None,
            "pct_from_20d_high": round(pct_from_20d_high, 2) if pct_from_20d_high is not None else None,
            "pct_from_20d_low": round(pct_from_20d_low, 2) if pct_from_20d_low is not None else None,
            "trend_state": trend_state,
            "level_context": level_context,
            "stretch_state": stretch_state,
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


def _fmt_pct_short(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    num = float(value)
    sign = "+" if num > 0 else ""
    return f"{sign}{num:.1f}%"


def _describe_setup(row: dict[str, Any]) -> dict[str, str]:
    bias = str(row.get("signal_bias") or "neutral")
    trend = str(row.get("trend_state") or "mixed")
    level = str(row.get("level_context") or "mid_range")
    stretch = str(row.get("stretch_state") or "normal")
    candle_bias = str(row.get("candle_bias") or "neutral")
    pct_from_high = row.get("pct_from_20d_high")
    pct_from_low = row.get("pct_from_20d_low")
    pct_vs_ma20 = row.get("pct_vs_ma20")
    pct_change = row.get("pct_change")
    pattern = str(row.get("yolo_pattern") or "").strip()

    level_label = {
        "at_support": "sitting near short-term support",
        "at_resistance": "pressing into short-term resistance",
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

    observation_parts: list[str] = []
    if pattern:
        if bias == "bullish":
            observation_parts.append(f"bullish YOLO ({pattern})")
        elif bias == "bearish":
            observation_parts.append(f"bearish YOLO ({pattern})")
        else:
            observation_parts.append(f"neutral YOLO ({pattern})")
    else:
        observation_parts.append("no decisive YOLO pattern")
    observation_parts.append(trend_label)
    observation_parts.append(level_label)
    observation_parts.append(stretch_label)
    if candle_bias == "bullish":
        observation_parts.append("latest candle bias is supportive")
    elif candle_bias == "bearish":
        observation_parts.append("latest candle bias is conflicting")

    actionability = "watch-only"
    action = "No strong edge yet. Keep it on watch, do not force a trade."

    near_support = isinstance(pct_from_low, (int, float)) and float(pct_from_low) <= 2.5
    near_resistance = isinstance(pct_from_high, (int, float)) and float(pct_from_high) <= 2.5
    large_day = isinstance(pct_change, (int, float)) and abs(float(pct_change)) >= 5.0

    if bias == "bullish":
        if stretch == "extended_up" or large_day:
            actionability = "wait"
            action = "Bullish idea, but do not chase strength. Prefer a pullback or breakout retest."
        elif trend == "uptrend" and near_resistance:
            actionability = "higher-probability"
            action = "Trend continuation watch. Best if price closes through resistance or retests it cleanly."
        elif near_support and candle_bias != "bearish":
            actionability = "higher-probability"
            action = "Reversal watch at support. Actionable only if support holds and the next candles confirm."
        elif trend == "downtrend":
            actionability = "conditional"
            action = "Counter-trend bounce only. Wait for trend repair before treating it as a core long."
        else:
            actionability = "conditional"
            action = "Bullish setup is present, but wait for confirmation instead of buying the first signal."
    elif bias == "bearish":
        if stretch == "extended_down" or large_day:
            actionability = "wait"
            action = "Avoid chasing the flush. Better setup is a failed bounce or a clean support break."
        elif near_support:
            actionability = "higher-probability"
            action = "Support-failure watch. It is only actionable if support actually breaks with follow-through."
        elif trend == "downtrend" and (near_resistance or candle_bias == "bearish"):
            actionability = "higher-probability"
            action = "Bearish continuation watch. Best entry is rejection near resistance or failed rally."
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

    observation = ". ".join(part[0].upper() + part[1:] if idx == 0 else part for idx, part in enumerate(observation_parts)) + "."
    return {
        "signal_bias": bias,
        "observation": observation,
        "actionability": actionability,
        "action": action,
        "risk_note": ", ".join(risk_notes[:3]) if risk_notes else "none",
        "technical_read": (
            f"{bias} | {trend} | "
            f"{'support' if near_support else ('resistance' if near_resistance else 'mid-range')} | "
            f"vs MA20 {_fmt_pct_short(pct_vs_ma20)}"
        ),
    }


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
        changes.append(
            {
                "slug": "setup",
                "title": "Top Setup Candidate",
                "detail": (
                    f"{best.get('ticker')} scored {best.get('score')} ({best.get('setup_tier')}) "
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
        "yolo_top_today": [],
        "candle_patterns_today": [],
        "sector_heatmap": [],
        "setup_quality_top": [],
        "watchlist_candidates": [],
        "tonight_key_changes": [],
    }
    movers_all: list[dict[str, Any]] = []
    fundamentals_map: dict[str, dict[str, Any]] = {}
    yolo_by_ticker: dict[str, dict[str, Any]] = {}
    vol_by_ticker: dict[str, dict[str, float | None]] = {}
    vol_ctx: dict[str, Any] = {}
    technical_by_ticker: dict[str, dict[str, Any]] = {}

    try:
        vol_by_ticker, vol_ctx = _fetch_volatility_inputs(conn)
        signals["volatility_context"] = vol_ctx
    except Exception:
        pass
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
                    }
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
                prev = yolo_by_ticker.get(ticker)
                if prev is None:
                    yolo_by_ticker[ticker] = candidate
                    continue
                prev_daily = str(prev.get("timeframe") or "") == "daily"
                cand_daily = str(candidate.get("timeframe") or "") == "daily"
                if cand_daily and not prev_daily:
                    yolo_by_ticker[ticker] = candidate
                    continue
                if cand_daily == prev_daily and conf > float(prev.get("confidence") or 0.0):
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
            signal_bias = "neutral"
            atr_pct_14 = vol.get("atr_pct_14")
            realized_vol_20 = vol.get("realized_vol_20")
            bb_width_20 = vol.get("bb_width_20")
            if yolo:
                yolo_pattern = yolo.get("pattern")
                yolo_confidence = yolo.get("confidence")
                yolo_age_days = yolo.get("age_days")
                yolo_timeframe = yolo.get("timeframe")
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

            final_score = round(_clamp(score, 0.0, 100.0), 1)
            setup_rows.append(
                {
                    "ticker": ticker,
                    "score": final_score,
                    "confluence_score": final_score,
                    "setup_tier": _setup_tier(final_score),
                    "sector": sector,
                    "pct_change": round(pct_change, 2),
                    "discount_pct": discount,
                    "peg": peg,
                    "atr_pct_14": atr_pct_14,
                    "realized_vol_20": realized_vol_20,
                    "bb_width_20": bb_width_20,
                    "near_52w_high": near_high,
                    "near_52w_low": near_low,
                    "yolo_pattern": yolo_pattern,
                    "yolo_confidence": yolo_confidence,
                    "yolo_age_days": yolo_age_days,
                    "yolo_timeframe": yolo_timeframe,
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
                    "stretch_state": tech.get("stretch_state") or "normal",
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
            )

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

            setup_quality_rows = signals.get("setup_quality_top")
            if isinstance(setup_quality_rows, list):
                for row in setup_quality_rows:
                    ticker = str(row.get("ticker") or "").upper()
                    candle = candle_by_ticker.get(ticker) or {}
                    if candle:
                        row["candle_pattern"] = candle.get("pattern")
                        row["candle_bias"] = candle.get("bias") or "neutral"
                        row["candle_confidence"] = candle.get("confidence")
                    readout = _describe_setup(row)
                    row.update(readout)

            watchlist_rows = signals.get("watchlist_candidates")
            if isinstance(watchlist_rows, list):
                setup_map = {
                    str(row.get("ticker") or "").upper(): row
                    for row in (setup_quality_rows or [])
                }
                for row in watchlist_rows:
                    src = setup_map.get(str(row.get("ticker") or "").upper())
                    if not src:
                        continue
                    for key in (
                        "signal_bias",
                        "observation",
                        "actionability",
                        "action",
                        "risk_note",
                        "technical_read",
                        "candle_pattern",
                        "candle_bias",
                        "candle_confidence",
                    ):
                        row[key] = src.get(key)
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
                    "| ticker | pattern | streak | seen_in_lookback | coverage_pct | latest_confidence | avg_confidence_window |"
                )
                lines.append("|---|---|---:|---:|---:|---:|---:|")
                for p in rows[:15]:
                    lines.append(
                        f"| {p.get('ticker')} | {p.get('pattern')} | {p.get('streak')} | {p.get('seen_in_lookback')} | "
                        f"{p.get('coverage_pct')} | {p.get('latest_confidence')} | {p.get('avg_confidence_window')} |"
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
        lines.append("| ticker | score | bias | observation | reasonable_action | risk_note |")
        lines.append("|---|---:|---|---|---|---|")
        for r in setup_rows:
            lines.append(
                f"| {r.get('ticker')} | {r.get('score')} | {r.get('signal_bias') or '-'} | "
                f"{r.get('observation') or '-'} | {r.get('action') or '-'} | {r.get('risk_note') or '-'} |"
            )

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


def write_reports(report: dict[str, Any], out_dir: Path) -> dict[str, str]:
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

    return {
        "json_path": str(json_path),
        "md_path": str(md_path),
        "latest_json": str(latest_json),
        "latest_md": str(latest_md),
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
    report = fetch_report_payload(
        db_path=Path(args.db_path).resolve(),
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
            send_report_email(report, md_text)
            email_meta["sent"] = True
            print(f"[EMAIL] sent ok transport={transport}")
        except Exception as exc:
            email_meta["error"] = str(exc)
            print(f"[EMAIL] failed {exc}")
    report["email"] = email_meta
    if email_meta["attempted"] and not email_meta["sent"]:
        warnings = report.get("warnings")
        if not isinstance(warnings, list):
            warnings = []
            report["warnings"] = warnings
        if "report_email_failed" not in warnings:
            warnings.append("report_email_failed")
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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
