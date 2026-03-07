from __future__ import annotations

import datetime as dt
import os
import re
import secrets
import sqlite3
from pathlib import Path
from typing import Any

TRUTHY_VALUES = {"1", "true", "yes", "on"}
EMAIL_RE = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,63}$", re.IGNORECASE)


def _as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in TRUTHY_VALUES


def email_subscribe_enabled() -> bool:
    return _as_bool(os.getenv("TRADER_KOO_EMAIL_SUBSCRIBE_ENABLED", "1"))


def email_max_recipients() -> int:
    raw = str(os.getenv("TRADER_KOO_EMAIL_MAX_RECIPIENTS", "200") or "").strip()
    try:
        return max(1, min(5000, int(raw)))
    except ValueError:
        return 200


def subscribe_ip_hourly_limit() -> int:
    raw = str(os.getenv("TRADER_KOO_EMAIL_SUBSCRIBE_IP_HOURLY_LIMIT", "12") or "").strip()
    try:
        return max(1, min(200, int(raw)))
    except ValueError:
        return 12


def subscribe_email_daily_limit() -> int:
    raw = str(os.getenv("TRADER_KOO_EMAIL_SUBSCRIBE_EMAIL_DAILY_LIMIT", "6") or "").strip()
    try:
        return max(1, min(100, int(raw)))
    except ValueError:
        return 6


def subscribe_resend_cooldown_min() -> int:
    raw = str(os.getenv("TRADER_KOO_EMAIL_SUBSCRIBE_RESEND_COOLDOWN_MIN", "15") or "").strip()
    try:
        return max(1, min(24 * 60, int(raw)))
    except ValueError:
        return 15


def email_token_expiry_hours() -> int:
    """Get email token expiry hours from environment (default: 168 = 7 days)."""
    raw = str(os.getenv("EMAIL_TOKEN_EXPIRY_HOURS", "168") or "").strip()
    try:
        return max(1, min(8760, int(raw)))  # 1 hour to 1 year
    except ValueError:
        return 168


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


def _parse_iso(value: Any) -> dt.datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


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
            CREATE TABLE IF NOT EXISTS email_subscribers (
                email TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                confirm_token TEXT,
                unsubscribe_token TEXT NOT NULL,
                source_ip TEXT,
                source_user_agent TEXT,
                subscribed_ts TEXT NOT NULL,
                confirmed_ts TEXT,
                unsubscribed_ts TEXT,
                verify_sent_count INTEGER NOT NULL DEFAULT 0,
                last_verify_sent_ts TEXT,
                last_email_sent_ts TEXT,
                last_email_error TEXT,
                updated_ts TEXT NOT NULL,
                confirm_token_created_ts TEXT,
                confirm_token_expires_ts TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS email_subscriber_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT,
                event_type TEXT NOT NULL,
                source_ip TEXT,
                details TEXT,
                created_ts TEXT NOT NULL
            )
            """
        )
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
            "CREATE INDEX IF NOT EXISTS idx_email_subscribers_status ON email_subscribers(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_email_subscribers_confirm_token ON email_subscribers(confirm_token)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_email_subscribers_unsub_token ON email_subscribers(unsubscribe_token)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_email_subscriber_events_type_ts ON email_subscriber_events(event_type, created_ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_email_subscriber_events_ip_ts ON email_subscriber_events(source_ip, created_ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_email_report_deliveries_generated ON email_report_deliveries(generated_ts)"
        )
        conn.commit()
    finally:
        conn.close()


def record_subscriber_event(
    db_path: Path,
    *,
    event_type: str,
    email: str | None = None,
    source_ip: str | None = None,
    details: str | None = None,
    created_ts: dt.datetime | None = None,
) -> None:
    ensure_subscriber_schema(db_path)
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO email_subscriber_events (email, event_type, source_ip, details, created_ts)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                canonical_email(email) if email else None,
                str(event_type or "").strip()[:64],
                str(source_ip or "").strip()[:96] or None,
                str(details or "").strip()[:400] or None,
                _iso(created_ts),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _count_recent_events(
    conn: sqlite3.Connection,
    *,
    event_type: str,
    since_ts: dt.datetime,
    source_ip: str | None = None,
    email: str | None = None,
) -> int:
    clauses = ["event_type = ?", "created_ts >= ?"]
    params: list[Any] = [event_type, _iso(since_ts)]
    if source_ip:
        clauses.append("source_ip = ?")
        params.append(str(source_ip).strip()[:96])
    if email:
        clauses.append("email = ?")
        params.append(canonical_email(email))
    row = conn.execute(
        f"SELECT COUNT(*) AS c FROM email_subscriber_events WHERE {' AND '.join(clauses)}",
        tuple(params),
    ).fetchone()
    return int((row["c"] if row else 0) or 0)


def check_request_rate_limit(
    db_path: Path,
    *,
    event_type: str,
    source_ip: str,
    email: str,
    now_utc: dt.datetime | None = None,
    ip_hourly_limit: int | None = None,
    email_daily_limit: int | None = None,
) -> tuple[bool, str | None]:
    ensure_subscriber_schema(db_path)
    now = now_utc or _utc_now()
    ip_limit = int(ip_hourly_limit if ip_hourly_limit is not None else subscribe_ip_hourly_limit())
    mail_limit = int(email_daily_limit if email_daily_limit is not None else subscribe_email_daily_limit())
    conn = _connect(db_path)
    try:
        ip_count = _count_recent_events(
            conn,
            event_type=event_type,
            since_ts=now - dt.timedelta(hours=1),
            source_ip=source_ip,
        )
        if ip_count >= ip_limit:
            return (False, f"Too many requests from this IP. Try again in about 1 hour.")
        email_count = _count_recent_events(
            conn,
            event_type=event_type,
            since_ts=now - dt.timedelta(days=1),
            email=email,
        )
        if email_count >= mail_limit:
            return (False, f"Too many requests for this email. Try again tomorrow.")
        return (True, None)
    finally:
        conn.close()


def _new_token() -> str:
    return secrets.token_urlsafe(32)


def _calculate_token_expiry(
    created_ts: dt.datetime,
    expiry_hours: int = 168  # 7 days default
) -> dt.datetime:
    """Calculate token expiration timestamp.
    
    Args:
        created_ts: Token creation timestamp.
        expiry_hours: Hours until expiration (default: 168 = 7 days).
        
    Returns:
        Expiration timestamp.
    """
    return created_ts + dt.timedelta(hours=expiry_hours)


def _is_token_expired(
    expires_ts: dt.datetime | None,
    now_utc: dt.datetime | None = None
) -> bool:
    """Check if a token has expired.
    
    Args:
        expires_ts: Token expiration timestamp.
        now_utc: Current time (defaults to now).
        
    Returns:
        True if token is expired, False otherwise.
    """
    if expires_ts is None:
        return False
    now = now_utc or _utc_now()
    return now >= expires_ts


def upsert_subscriber_pending(
    db_path: Path,
    *,
    email: str,
    source_ip: str | None,
    source_user_agent: str | None,
    now_utc: dt.datetime | None = None,
    resend_cooldown_min: int | None = None,
    token_expiry_hours: int | None = None,
) -> dict[str, Any]:
    ensure_subscriber_schema(db_path)
    now = now_utc or _utc_now()
    cooldown = int(resend_cooldown_min if resend_cooldown_min is not None else subscribe_resend_cooldown_min())
    expiry_hours = int(token_expiry_hours if token_expiry_hours is not None else email_token_expiry_hours())
    normalized = canonical_email(email)
    if not is_valid_email(normalized):
        return {
            "ok": False,
            "error": "invalid_email",
            "detail": "Invalid email format.",
        }

    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM email_subscribers WHERE email = ?",
            (normalized,),
        ).fetchone()
        existing = dict(row) if row else {}
        status = str(existing.get("status") or "").strip().lower()
        unsubscribe_token = str(existing.get("unsubscribe_token") or "").strip() or _new_token()
        confirm_token = _new_token()
        confirm_token_created_ts = now
        confirm_token_expires_ts = _calculate_token_expiry(now, expiry_hours)
        should_send = True
        state = "pending"

        if status == "active":
            should_send = False
            state = "already_active"
        elif status == "pending":
            last_sent = _parse_iso(existing.get("last_verify_sent_ts"))
            if last_sent is not None and (now - last_sent).total_seconds() < (cooldown * 60):
                should_send = False
                state = "pending_recently_sent"

        subscribed_ts = str(existing.get("subscribed_ts") or _iso(now))
        confirmed_ts = str(existing.get("confirmed_ts") or "") or None
        unsubscribed_ts = None
        if state == "already_active":
            confirm_token = str(existing.get("confirm_token") or "") or None
            if not confirm_token:
                confirm_token = _new_token()
                confirm_token_created_ts = now
                confirm_token_expires_ts = _calculate_token_expiry(now, expiry_hours)
                status = "pending"
                state = "pending"
                should_send = True
        else:
            status = "pending"
            confirmed_ts = None

        conn.execute(
            """
            INSERT INTO email_subscribers (
                email,
                status,
                confirm_token,
                unsubscribe_token,
                source_ip,
                source_user_agent,
                subscribed_ts,
                confirmed_ts,
                unsubscribed_ts,
                updated_ts,
                confirm_token_created_ts,
                confirm_token_expires_ts
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                status = excluded.status,
                confirm_token = excluded.confirm_token,
                unsubscribe_token = excluded.unsubscribe_token,
                source_ip = COALESCE(excluded.source_ip, email_subscribers.source_ip),
                source_user_agent = COALESCE(excluded.source_user_agent, email_subscribers.source_user_agent),
                subscribed_ts = COALESCE(email_subscribers.subscribed_ts, excluded.subscribed_ts),
                confirmed_ts = excluded.confirmed_ts,
                unsubscribed_ts = excluded.unsubscribed_ts,
                updated_ts = excluded.updated_ts,
                confirm_token_created_ts = excluded.confirm_token_created_ts,
                confirm_token_expires_ts = excluded.confirm_token_expires_ts
            """,
            (
                normalized,
                status,
                confirm_token,
                unsubscribe_token,
                str(source_ip or "").strip()[:96] or None,
                str(source_user_agent or "").strip()[:255] or None,
                subscribed_ts,
                confirmed_ts,
                unsubscribed_ts,
                _iso(now),
                _iso(confirm_token_created_ts),
                _iso(confirm_token_expires_ts),
            ),
        )
        conn.execute(
            """
            INSERT INTO email_subscriber_events (email, event_type, source_ip, details, created_ts)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                normalized,
                "subscribe_request",
                str(source_ip or "").strip()[:96] or None,
                state,
                _iso(now),
            ),
        )
        conn.commit()
        return {
            "ok": True,
            "email": normalized,
            "state": state,
            "should_send_verification": bool(should_send),
            "confirm_token": confirm_token,
            "unsubscribe_token": unsubscribe_token,
            "confirm_token_expires_ts": _iso(confirm_token_expires_ts),
        }
    finally:
        conn.close()


def mark_verification_sent(
    db_path: Path,
    *,
    email: str,
    now_utc: dt.datetime | None = None,
) -> None:
    ensure_subscriber_schema(db_path)
    now = now_utc or _utc_now()
    normalized = canonical_email(email)
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            UPDATE email_subscribers
            SET
                verify_sent_count = COALESCE(verify_sent_count, 0) + 1,
                last_verify_sent_ts = ?,
                updated_ts = ?
            WHERE email = ?
            """,
            (_iso(now), _iso(now), normalized),
        )
        conn.execute(
            """
            INSERT INTO email_subscriber_events (email, event_type, details, created_ts)
            VALUES (?, 'subscribe_verification_sent', NULL, ?)
            """,
            (normalized, _iso(now)),
        )
        conn.commit()
    finally:
        conn.close()


def confirm_subscriber_token(
    db_path: Path,
    *,
    token: str,
    now_utc: dt.datetime | None = None,
) -> dict[str, Any] | None:
    ensure_subscriber_schema(db_path)
    token_clean = str(token or "").strip()
    if not token_clean:
        return None
    now = now_utc or _utc_now()
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT email, status, confirm_token_expires_ts FROM email_subscribers WHERE confirm_token = ? LIMIT 1",
            (token_clean,),
        ).fetchone()
        if row is None:
            return None
        
        email = canonical_email(row["email"])
        expires_ts = _parse_iso(row["confirm_token_expires_ts"])
        
        # Check if token has expired
        if _is_token_expired(expires_ts, now):
            return {
                "error": "token_expired",
                "detail": "This confirmation link has expired. Please request a new one.",
                "email": email,
            }
        
        conn.execute(
            """
            UPDATE email_subscribers
            SET
                status = 'active',
                confirm_token = NULL,
                confirmed_ts = COALESCE(confirmed_ts, ?),
                unsubscribed_ts = NULL,
                updated_ts = ?,
                confirm_token_created_ts = NULL,
                confirm_token_expires_ts = NULL
            WHERE email = ?
            """,
            (_iso(now), _iso(now), email),
        )
        conn.execute(
            """
            INSERT INTO email_subscriber_events (email, event_type, details, created_ts)
            VALUES (?, 'subscribe_confirmed', NULL, ?)
            """,
            (email, _iso(now)),
        )
        conn.commit()
        return {"email": email, "status": "active"}
    finally:
        conn.close()


def find_subscriber_by_email(db_path: Path, *, email: str) -> dict[str, Any] | None:
    ensure_subscriber_schema(db_path)
    normalized = canonical_email(email)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM email_subscribers WHERE email = ? LIMIT 1",
            (normalized,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        conn.close()


def unsubscribe_subscriber_token(
    db_path: Path,
    *,
    token: str,
    now_utc: dt.datetime | None = None,
) -> dict[str, Any] | None:
    ensure_subscriber_schema(db_path)
    token_clean = str(token or "").strip()
    if not token_clean:
        return None
    now = now_utc or _utc_now()
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT email FROM email_subscribers WHERE unsubscribe_token = ? LIMIT 1",
            (token_clean,),
        ).fetchone()
        if row is None:
            return None
        email = canonical_email(row["email"])
        conn.execute(
            """
            UPDATE email_subscribers
            SET
                status = 'unsubscribed',
                unsubscribed_ts = ?,
                updated_ts = ?
            WHERE email = ?
            """,
            (_iso(now), _iso(now), email),
        )
        conn.execute(
            """
            INSERT INTO email_subscriber_events (email, event_type, details, created_ts)
            VALUES (?, 'unsubscribed', NULL, ?)
            """,
            (email, _iso(now)),
        )
        conn.commit()
        return {"email": email, "status": "unsubscribed"}
    finally:
        conn.close()


def list_active_subscribers(db_path: Path, *, limit: int = 5000) -> list[dict[str, Any]]:
    ensure_subscriber_schema(db_path)
    cap = max(1, min(10_000, int(limit)))
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT email, unsubscribe_token, confirmed_ts, subscribed_ts
            FROM email_subscribers
            WHERE status = 'active'
            ORDER BY COALESCE(confirmed_ts, subscribed_ts) DESC, email ASC
            LIMIT ?
            """,
            (cap,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def resolve_report_recipients(
    db_path: Path,
    *,
    fallback_raw: str,
    max_recipients: int | None = None,
) -> list[dict[str, Any]]:
    recipients: dict[str, dict[str, Any]] = {}
    for email in parse_recipients(fallback_raw):
        recipients[email] = {
            "email": email,
            "source": "env",
            "unsubscribe_token": None,
        }
    if email_subscribe_enabled():
        for row in list_active_subscribers(db_path, limit=10_000):
            email = canonical_email(row.get("email"))
            if not is_valid_email(email):
                continue
            recipients[email] = {
                "email": email,
                "source": "subscriber",
                "unsubscribe_token": str(row.get("unsubscribe_token") or "").strip() or None,
            }
    out = sorted(
        recipients.values(),
        key=lambda item: (0 if item.get("source") == "subscriber" else 1, str(item.get("email") or "")),
    )
    limit = int(max_recipients if max_recipients is not None else email_max_recipients())
    return out[: max(1, limit)]


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
        conn.execute(
            """
            UPDATE email_subscribers
            SET
                last_email_sent_ts = ?,
                last_email_error = ?,
                updated_ts = ?
            WHERE email = ?
            """,
            (
                _iso(now),
                str(error or "").strip()[:400] or None,
                _iso(now),
                normalized,
            ),
        )
        conn.execute(
            """
            INSERT INTO email_subscriber_events (email, event_type, details, created_ts)
            VALUES (?, ?, ?, ?)
            """,
            (
                normalized,
                "report_email_sent" if status == "sent" else "report_email_failed",
                ts,
                _iso(now),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def subscriber_counts(db_path: Path) -> dict[str, int]:
    ensure_subscriber_schema(db_path)
    conn = _connect(db_path)
    try:
        if not _table_exists(conn, "email_subscribers"):
            return {"active": 0, "pending": 0, "unsubscribed": 0, "total": 0}
        row = conn.execute(
            """
            SELECT
                SUM(CASE WHEN status='active' THEN 1 ELSE 0 END) AS active,
                SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN status='unsubscribed' THEN 1 ELSE 0 END) AS unsubscribed,
                COUNT(*) AS total
            FROM email_subscribers
            """
        ).fetchone()
        if row is None:
            return {"active": 0, "pending": 0, "unsubscribed": 0, "total": 0}
        return {
            "active": int(row["active"] or 0),
            "pending": int(row["pending"] or 0),
            "unsubscribed": int(row["unsubscribed"] or 0),
            "total": int(row["total"] or 0),
        }
    finally:
        conn.close()

