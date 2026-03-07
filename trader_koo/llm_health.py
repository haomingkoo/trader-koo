from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from pathlib import Path
from typing import Any


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
        ts = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def _as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def llm_alert_enabled() -> bool:
    return _as_bool(os.getenv("TRADER_KOO_LLM_FAIL_ALERT_ENABLED", "1"))


def llm_alert_cooldown_min() -> int:
    raw = str(os.getenv("TRADER_KOO_LLM_FAIL_ALERT_COOLDOWN_MIN", "180") or "").strip()
    try:
        return max(5, min(24 * 60, int(raw)))
    except ValueError:
        return 180


def llm_failure_event_cooldown_sec() -> int:
    raw = str(os.getenv("TRADER_KOO_LLM_FAILURE_EVENT_COOLDOWN_SEC", "300") or "").strip()
    try:
        return max(30, min(24 * 3600, int(raw)))
    except ValueError:
        return 300


def llm_degraded_threshold() -> int:
    raw = str(os.getenv("TRADER_KOO_LLM_DEGRADED_FAILS", "3") or "").strip()
    try:
        return max(1, min(50, int(raw)))
    except ValueError:
        return 3


def llm_disable_seconds_on_fail() -> int:
    raw = str(os.getenv("TRADER_KOO_LLM_DISABLE_SEC_ON_FAIL", "300") or "").strip()
    try:
        return max(0, min(24 * 3600, int(raw)))
    except ValueError:
        return 300


def llm_health_max_events() -> int:
    raw = str(os.getenv("TRADER_KOO_LLM_HEALTH_MAX_EVENTS", "5000") or "").strip()
    try:
        return max(200, min(200_000, int(raw)))
    except ValueError:
        return 5000


def llm_usage_max_rows() -> int:
    raw = str(os.getenv("TRADER_KOO_LLM_USAGE_MAX_ROWS", "20000") or "").strip()
    try:
        return max(500, min(2_000_000, int(raw)))
    except ValueError:
        return 20000


def _env_float(name: str, default: float = 0.0) -> float:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def llm_cost_input_per_1m() -> float:
    return max(0.0, _env_float("TRADER_KOO_LLM_COST_INPUT_PER_1M", 0.0))


def llm_cost_output_per_1m() -> float:
    return max(0.0, _env_float("TRADER_KOO_LLM_COST_OUTPUT_PER_1M", 0.0))


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_llm_health_schema(db_path: Path) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_health_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_ts TEXT NOT NULL,
                outcome TEXT NOT NULL,
                source TEXT,
                ticker TEXT,
                reason TEXT,
                error_class TEXT,
                details TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_health_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_ts TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_health_events_ts ON llm_health_events(event_ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_health_events_outcome_ts ON llm_health_events(outcome, event_ts)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_ts TEXT NOT NULL,
                source TEXT,
                ticker TEXT,
                model TEXT,
                deployment TEXT,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                meta_json TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_token_usage_ts ON llm_token_usage(event_ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_token_usage_source_ts ON llm_token_usage(source, event_ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_token_usage_ticker_ts ON llm_token_usage(ticker, event_ts)"
        )
        conn.commit()
    finally:
        conn.close()


def _state_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM llm_health_state WHERE key = ? LIMIT 1",
        (str(key or "").strip(),),
    ).fetchone()
    if row is None:
        return None
    return str(row["value"] or "")


def _state_set(conn: sqlite3.Connection, key: str, value: Any, now: dt.datetime) -> None:
    conn.execute(
        """
        INSERT INTO llm_health_state (key, value, updated_ts)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_ts = excluded.updated_ts
        """,
        (str(key or "").strip(), str(value), _iso(now)),
    )


def _record_event(
    conn: sqlite3.Connection,
    *,
    now: dt.datetime,
    outcome: str,
    source: str | None = None,
    ticker: str | None = None,
    reason: str | None = None,
    error_class: str | None = None,
    details: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO llm_health_events (
            event_ts, outcome, source, ticker, reason, error_class, details
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _iso(now),
            str(outcome or "").strip()[:32],
            str(source or "").strip()[:64] or None,
            str(ticker or "").strip().upper()[:16] or None,
            str(reason or "").strip()[:120] or None,
            str(error_class or "").strip()[:80] or None,
            str(details or "").strip()[:500] or None,
        ),
    )


def _prune_events(conn: sqlite3.Connection) -> None:
    max_events = llm_health_max_events()
    row = conn.execute("SELECT COUNT(*) AS c FROM llm_health_events").fetchone()
    total = int((row["c"] if row else 0) or 0)
    if total <= max_events:
        return
    to_delete = total - max_events
    conn.execute(
        """
        DELETE FROM llm_health_events
        WHERE id IN (
            SELECT id FROM llm_health_events
            ORDER BY event_ts ASC, id ASC
            LIMIT ?
        )
        """,
        (to_delete,),
    )


def _prune_token_usage_rows(conn: sqlite3.Connection) -> None:
    max_rows = llm_usage_max_rows()
    row = conn.execute("SELECT COUNT(*) AS c FROM llm_token_usage").fetchone()
    total = int((row["c"] if row else 0) or 0)
    if total <= max_rows:
        return
    to_delete = total - max_rows
    conn.execute(
        """
        DELETE FROM llm_token_usage
        WHERE id IN (
            SELECT id FROM llm_token_usage
            ORDER BY event_ts ASC, id ASC
            LIMIT ?
        )
        """,
        (to_delete,),
    )


def note_llm_success(
    db_path: Path,
    *,
    source: str | None = None,
    ticker: str | None = None,
) -> None:
    ensure_llm_health_schema(db_path)
    now = _utc_now()
    conn = _connect(db_path)
    try:
        raw_consec = _state_get(conn, "consecutive_failures") or "0"
        try:
            consecutive = max(0, int(raw_consec))
        except ValueError:
            consecutive = 0
        _state_set(conn, "consecutive_failures", "0", now)
        _state_set(conn, "last_success_ts", _iso(now), now)
        _record_event(
            conn,
            now=now,
            outcome="success",
            source=source,
            ticker=ticker,
            reason="recovered_after_failures" if consecutive > 0 else "ok",
        )
        _prune_events(conn)
        conn.commit()
    finally:
        conn.close()


def note_llm_failure(
    db_path: Path,
    *,
    source: str | None = None,
    ticker: str | None = None,
    reason: str | None = None,
    error_class: str | None = None,
    details: str | None = None,
) -> None:
    ensure_llm_health_schema(db_path)
    now = _utc_now()
    cooldown_sec = llm_failure_event_cooldown_sec()
    conn = _connect(db_path)
    try:
        raw_consec = _state_get(conn, "consecutive_failures") or "0"
        try:
            consecutive = max(0, int(raw_consec))
        except ValueError:
            consecutive = 0
        consecutive += 1
        _state_set(conn, "consecutive_failures", str(consecutive), now)
        _state_set(conn, "last_failure_ts", _iso(now), now)
        if reason:
            _state_set(conn, "last_failure_reason", str(reason)[:120], now)
        if error_class:
            _state_set(conn, "last_error_class", str(error_class)[:80], now)
        if details:
            _state_set(conn, "last_error_details", str(details)[:500], now)

        should_record = True
        last_event_ts = _parse_iso(_state_get(conn, "last_failure_event_ts"))
        last_event_key = _state_get(conn, "last_failure_event_key") or ""
        event_key = "|".join(
            [
                str(source or "").strip().lower(),
                str(error_class or "").strip().lower(),
                str(reason or "").strip().lower(),
            ]
        )
        if last_event_ts is not None:
            age_sec = (now - last_event_ts).total_seconds()
            if age_sec < cooldown_sec and event_key == last_event_key:
                should_record = False
        if should_record:
            _record_event(
                conn,
                now=now,
                outcome="failure",
                source=source,
                ticker=ticker,
                reason=reason,
                error_class=error_class,
                details=details,
            )
            _state_set(conn, "last_failure_event_ts", _iso(now), now)
            _state_set(conn, "last_failure_event_key", event_key, now)
            _prune_events(conn)
        conn.commit()
    finally:
        conn.close()


def note_llm_token_usage(
    db_path: Path,
    *,
    source: str | None = None,
    ticker: str | None = None,
    model: str | None = None,
    deployment: str | None = None,
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    total_tokens: Any = None,
    meta: dict[str, Any] | None = None,
) -> None:
    ensure_llm_health_schema(db_path)
    now = _utc_now()
    try:
        prompt = max(0, int(float(prompt_tokens or 0)))
    except Exception:
        prompt = 0
    try:
        completion = max(0, int(float(completion_tokens or 0)))
    except Exception:
        completion = 0
    try:
        total = max(0, int(float(total_tokens or 0)))
    except Exception:
        total = 0
    if total <= 0:
        total = prompt + completion
    if total <= 0 and not (model or deployment):
        return

    meta_json = ""
    if isinstance(meta, dict) and meta:
        try:
            meta_json = json.dumps(meta, ensure_ascii=True, separators=(",", ":"))[:1000]
        except Exception:
            meta_json = ""

    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO llm_token_usage (
                event_ts, source, ticker, model, deployment,
                prompt_tokens, completion_tokens, total_tokens, meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _iso(now),
                str(source or "").strip()[:64] or None,
                str(ticker or "").strip().upper()[:16] or None,
                str(model or "").strip()[:120] or None,
                str(deployment or "").strip()[:120] or None,
                int(prompt),
                int(completion),
                int(total),
                meta_json or None,
            ),
        )
        _prune_token_usage_rows(conn)
        conn.commit()
    finally:
        conn.close()


def llm_health_summary(
    db_path: Path,
    *,
    recent_limit: int = 25,
) -> dict[str, Any]:
    ensure_llm_health_schema(db_path)
    limit = max(1, min(200, int(recent_limit)))
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT
                outcome,
                COUNT(*) AS count
            FROM llm_health_events
            GROUP BY outcome
            """
        ).fetchall()
        counts = {str(r["outcome"] or ""): int(r["count"] or 0) for r in rows}
        recent_rows = conn.execute(
            """
            SELECT event_ts, outcome, source, ticker, reason, error_class, details
            FROM llm_health_events
            ORDER BY event_ts DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        recent = [dict(r) for r in recent_rows]

        raw_consec = _state_get(conn, "consecutive_failures") or "0"
        try:
            consecutive = max(0, int(raw_consec))
        except ValueError:
            consecutive = 0

        last_success_ts = _state_get(conn, "last_success_ts")
        last_failure_ts = _state_get(conn, "last_failure_ts")
        last_failure_reason = _state_get(conn, "last_failure_reason")
        last_error_class = _state_get(conn, "last_error_class")
        last_error_details = _state_get(conn, "last_error_details")
        threshold = llm_degraded_threshold()
        degraded = consecutive >= threshold and bool(last_failure_ts)

        return {
            "degraded": degraded,
            "degraded_threshold": threshold,
            "consecutive_failures": consecutive,
            "last_success_ts": last_success_ts,
            "last_failure_ts": last_failure_ts,
            "last_failure_reason": last_failure_reason,
            "last_error_class": last_error_class,
            "last_error_details": last_error_details,
            "counts": {
                "success": int(counts.get("success") or 0),
                "failure": int(counts.get("failure") or 0),
                "other": sum(v for k, v in counts.items() if k not in {"success", "failure"}),
                "total": sum(counts.values()),
            },
            "recent_events": recent,
        }
    finally:
        conn.close()


def llm_token_usage_summary(
    db_path: Path,
    *,
    days: int = 30,
    limit: int = 50,
) -> dict[str, Any]:
    ensure_llm_health_schema(db_path)
    window_days = max(1, min(3650, int(days)))
    row_limit = max(1, min(500, int(limit)))
    cutoff = _iso(_utc_now() - dt.timedelta(days=window_days))
    input_rate = llm_cost_input_per_1m()
    output_rate = llm_cost_output_per_1m()
    estimator_enabled = input_rate > 0 or output_rate > 0

    def _estimate_usd(prompt: int, completion: int) -> float | None:
        if not estimator_enabled:
            return None
        usd = (float(prompt) / 1_000_000.0) * input_rate + (float(completion) / 1_000_000.0) * output_rate
        return round(usd, 6)

    conn = _connect(db_path)
    try:
        total_row = conn.execute(
            """
            SELECT
                COUNT(*) AS requests,
                COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                COALESCE(SUM(total_tokens), 0) AS total_tokens
            FROM llm_token_usage
            WHERE event_ts >= ?
            """,
            (cutoff,),
        ).fetchone()
        requests = int((total_row["requests"] if total_row else 0) or 0)
        prompt_tokens = int((total_row["prompt_tokens"] if total_row else 0) or 0)
        completion_tokens = int((total_row["completion_tokens"] if total_row else 0) or 0)
        total_tokens = int((total_row["total_tokens"] if total_row else 0) or 0)

        recent_rows = conn.execute(
            """
            SELECT
                event_ts, source, ticker, model, deployment,
                prompt_tokens, completion_tokens, total_tokens
            FROM llm_token_usage
            ORDER BY event_ts DESC, id DESC
            LIMIT ?
            """,
            (row_limit,),
        ).fetchall()
        recent = [dict(r) for r in recent_rows]
        for row in recent:
            row["estimated_cost_usd"] = _estimate_usd(
                int(row.get("prompt_tokens") or 0),
                int(row.get("completion_tokens") or 0),
            )

        daily_rows = conn.execute(
            """
            SELECT
                substr(event_ts, 1, 10) AS day_utc,
                COUNT(*) AS requests,
                COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                COALESCE(SUM(total_tokens), 0) AS total_tokens
            FROM llm_token_usage
            WHERE event_ts >= ?
            GROUP BY substr(event_ts, 1, 10)
            ORDER BY day_utc DESC
            LIMIT ?
            """,
            (cutoff, row_limit),
        ).fetchall()
        daily = [dict(r) for r in daily_rows]
        for row in daily:
            row["estimated_cost_usd"] = _estimate_usd(
                int(row.get("prompt_tokens") or 0),
                int(row.get("completion_tokens") or 0),
            )

        source_rows = conn.execute(
            """
            SELECT
                COALESCE(source, 'unknown') AS source,
                COUNT(*) AS requests,
                COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                COALESCE(SUM(total_tokens), 0) AS total_tokens
            FROM llm_token_usage
            WHERE event_ts >= ?
            GROUP BY COALESCE(source, 'unknown')
            ORDER BY total_tokens DESC
            LIMIT ?
            """,
            (cutoff, row_limit),
        ).fetchall()
        by_source = [dict(r) for r in source_rows]
        for row in by_source:
            row["estimated_cost_usd"] = _estimate_usd(
                int(row.get("prompt_tokens") or 0),
                int(row.get("completion_tokens") or 0),
            )

        ticker_rows = conn.execute(
            """
            SELECT
                ticker,
                COUNT(*) AS requests,
                COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                COALESCE(SUM(total_tokens), 0) AS total_tokens
            FROM llm_token_usage
            WHERE event_ts >= ? AND ticker IS NOT NULL AND ticker != ''
            GROUP BY ticker
            ORDER BY total_tokens DESC
            LIMIT ?
            """,
            (cutoff, row_limit),
        ).fetchall()
        top_tickers = [dict(r) for r in ticker_rows]
        for row in top_tickers:
            row["estimated_cost_usd"] = _estimate_usd(
                int(row.get("prompt_tokens") or 0),
                int(row.get("completion_tokens") or 0),
            )

        avg_tokens = round(float(total_tokens) / float(requests), 2) if requests > 0 else 0.0

        return {
            "window_days": window_days,
            "totals": {
                "requests": requests,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "avg_tokens_per_request": avg_tokens,
                "estimated_cost_usd": _estimate_usd(prompt_tokens, completion_tokens),
            },
            "pricing": {
                "input_per_1m": input_rate,
                "output_per_1m": output_rate,
                "estimation_enabled": estimator_enabled,
            },
            "by_source": by_source,
            "top_tickers": top_tickers,
            "daily": daily,
            "recent": recent,
        }
    finally:
        conn.close()


def should_send_llm_alert(db_path: Path) -> tuple[bool, str]:
    ensure_llm_health_schema(db_path)
    if not llm_alert_enabled():
        return (False, "alerts_disabled")
    now = _utc_now()
    cooldown = llm_alert_cooldown_min()
    conn = _connect(db_path)
    try:
        raw_consec = _state_get(conn, "consecutive_failures") or "0"
        try:
            consecutive = max(0, int(raw_consec))
        except ValueError:
            consecutive = 0
        if consecutive < llm_degraded_threshold():
            return (False, "not_degraded")
        last_alert_ts = _parse_iso(_state_get(conn, "last_alert_ts"))
        if last_alert_ts is not None:
            age_min = (now - last_alert_ts).total_seconds() / 60.0
            if age_min < cooldown:
                return (False, f"cooldown_active:{int(cooldown - age_min)}m")
        return (True, "degraded_and_alert_due")
    finally:
        conn.close()


def mark_llm_alert_sent(
    db_path: Path,
    *,
    detail: dict[str, Any] | None = None,
) -> None:
    ensure_llm_health_schema(db_path)
    now = _utc_now()
    conn = _connect(db_path)
    try:
        _state_set(conn, "last_alert_ts", _iso(now), now)
        if detail:
            _state_set(conn, "last_alert_payload", json.dumps(detail, ensure_ascii=True)[:2000], now)
        _record_event(
            conn,
            now=now,
            outcome="alert",
            reason="llm_degraded_notification_sent",
        )
        _prune_events(conn)
        conn.commit()
    finally:
        conn.close()
