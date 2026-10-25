from __future__ import annotations

import datetime as dt
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

EMAIL_RE = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,63}$", re.IGNORECASE)


def email_max_recipients() -> int:
    raw = str(os.getenv("TRADER_KOO_EMAIL_MAX_RECIPIENTS", "200") or "").strip()
    try:
        return max(1, min(5000, int(raw)))
    except ValueError:
        return 200


def canonical_email(value: Any) -> str:
    return str(value or "").strip().lower()


def is_valid_email(value: Any) -> bool:
    email = canonical_email(value)
    if not email or len(email) > 254:
        return False
    return EMAIL_RE.match(email) is not None


def parse_recipients(raw: Any) -> list[str]:
    text = str(raw or "")
    parts = re.split(r"[,\s;]+", text)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        email = canonical_email(part)
        if not is_valid_email(email):
            continue
        if email in seen:
            continue
        seen.add(email)
        out.append(email)
    return out


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def _iso(ts: dt.datetime | None = None) -> str:
    return (ts or _utc_now()).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_subscriber_schema(db_path: Path) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS email_report_deliveries (
                email TEXT NOT NULL,
                generated_ts TEXT NOT NULL,
                sent_ts TEXT NOT NULL,
                transport TEXT,
                status TEXT NOT NULL,
                error TEXT,
                PRIMARY KEY (email, generated_ts)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_email_report_deliveries_generated ON email_report_deliveries(generated_ts)"
        )
        conn.commit()
    finally:
        conn.close()


def already_sent_generated_report(
    db_path: Path,
    *,
    email: str,
    generated_ts: str,
) -> bool:
    normalized = canonical_email(email)
    ts = str(generated_ts or "").strip()
    if not normalized or not ts:
        return False
    ensure_subscriber_schema(db_path)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT 1
            FROM email_report_deliveries
            WHERE email = ? AND generated_ts = ? AND status = 'sent'
            LIMIT 1
            """,
            (normalized, ts),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def record_generated_report_delivery(
    db_path: Path,
    *,
    email: str,
    generated_ts: str,
    transport: str,
    status: str,
    error: str | None = None,
    now_utc: dt.datetime | None = None,
) -> None:
    normalized = canonical_email(email)
    ts = str(generated_ts or "").strip()
    if not normalized or not ts:
        return
    ensure_subscriber_schema(db_path)
    now = now_utc or _utc_now()
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO email_report_deliveries (email, generated_ts, sent_ts, transport, status, error)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(email, generated_ts) DO UPDATE SET
                sent_ts = excluded.sent_ts,
                transport = excluded.transport,
                status = excluded.status,
                error = excluded.error
            """,
            (
                normalized,
                ts,
                _iso(now),
                str(transport or "").strip()[:32] or None,
                str(status or "").strip()[:32],
                str(error or "").strip()[:400] or None,
            ),
        )
        conn.commit()
    finally:
        conn.close()
