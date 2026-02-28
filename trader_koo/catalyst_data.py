from __future__ import annotations

import csv
import datetime as dt
import io
import json
import logging
import os
import re
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

LOG = logging.getLogger("trader_koo.catalyst")

SESSION_ORDER = {"BMO": 0, "TBD": 1, "AMC": 2}
SESSION_LABELS = {
    "BMO": "Premarket",
    "TBD": "TBD",
    "AMC": "After Hours",
}
CACHE_PROVIDER = "alpha_vantage"
CACHE_TTL_HOURS = max(1, int(os.getenv("TRADER_KOO_EARNINGS_CACHE_HOURS", "6")))
ALPHA_VANTAGE_TIMEOUT_SEC = max(5, int(os.getenv("TRADER_KOO_ALPHA_VANTAGE_TIMEOUT_SEC", "20")))


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def ensure_external_data_cache_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS external_data_cache (
            cache_key   TEXT PRIMARY KEY,
            provider    TEXT NOT NULL,
            fetched_ts  TEXT NOT NULL,
            expires_ts  TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_external_data_cache_provider ON external_data_cache(provider, fetched_ts DESC)"
    )
    conn.commit()


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso_utc(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    raw = str(value).strip()
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


def _cache_load(conn: sqlite3.Connection, cache_key: str) -> dict[str, Any] | None:
    ensure_external_data_cache_table(conn)
    row = conn.execute(
        "SELECT provider, fetched_ts, expires_ts, payload_json FROM external_data_cache WHERE cache_key = ?",
        (cache_key,),
    ).fetchone()
    if row is None:
        return None
    payload = []
    try:
        payload = json.loads(str(row[3] or "[]"))
    except Exception:
        payload = []
    fetched_ts = _parse_iso_utc(str(row[1] or ""))
    expires_ts = _parse_iso_utc(str(row[2] or ""))
    now = _now_utc()
    age_hours = None
    if fetched_ts is not None:
        age_hours = round((now - fetched_ts).total_seconds() / 3600.0, 2)
    return {
        "provider": str(row[0] or CACHE_PROVIDER),
        "fetched_ts": str(row[1] or "") or None,
        "expires_ts": str(row[2] or "") or None,
        "payload": payload if isinstance(payload, list) else [],
        "is_fresh": bool(expires_ts is not None and expires_ts > now),
        "age_hours": age_hours,
    }


def _cache_store(conn: sqlite3.Connection, cache_key: str, provider: str, payload: list[dict[str, Any]], ttl_hours: int) -> dict[str, Any]:
    ensure_external_data_cache_table(conn)
    now = _now_utc()
    expires = now + dt.timedelta(hours=max(1, ttl_hours))
    conn.execute(
        """
        INSERT INTO external_data_cache(cache_key, provider, fetched_ts, expires_ts, payload_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET
            provider = excluded.provider,
            fetched_ts = excluded.fetched_ts,
            expires_ts = excluded.expires_ts,
            payload_json = excluded.payload_json
        """,
        (
            cache_key,
            provider,
            _iso_utc(now),
            _iso_utc(expires),
            json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        ),
    )
    conn.commit()
    return {
        "provider": provider,
        "fetched_ts": _iso_utc(now),
        "expires_ts": _iso_utc(expires),
        "payload": payload,
        "is_fresh": True,
        "age_hours": 0.0,
    }


def _alpha_vantage_key() -> str:
    return str(os.getenv("TRADER_KOO_ALPHA_VANTAGE_KEY", "") or "").strip()


def _alpha_vantage_horizon(days: int) -> str:
    if days <= 95:
        return "3month"
    if days <= 185:
        return "6month"
    return "12month"


def _normalize_session(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if raw in {"BMO", "PREMARKET", "BEFORE OPEN", "BEFORE MARKET OPEN"}:
        return "BMO"
    if raw in {"AMC", "AFTER CLOSE", "AFTER MARKET CLOSE", "AFTER HOURS"}:
        return "AMC"
    return "TBD"


def extract_earnings_value(raw_obj: dict[str, Any]) -> str | None:
    for key in ("Earnings", "Earnings Date", "earnings", "earningsDate"):
        value = raw_obj.get(key)
        if value not in {None, ""}:
            out = str(value).strip()
            if out and out != "-":
                return out
    return None


def parse_earnings_value(raw_value: Any, market_date: dt.date) -> dict[str, Any]:
    raw = str(raw_value or "").strip()
    if not raw or raw == "-":
        return {}
    upper = raw.upper()
    session = "TBD"
    if "BMO" in upper or "BEFORE OPEN" in upper or "BEFORE MARKET OPEN" in upper:
        session = "BMO"
    elif "AMC" in upper or "AFTER CLOSE" in upper or "AFTER MARKET CLOSE" in upper:
        session = "AMC"

    cleaned = upper
    for token in (
        "BMO",
        "AMC",
        "BEFORE OPEN",
        "BEFORE MARKET OPEN",
        "AFTER CLOSE",
        "AFTER MARKET CLOSE",
        "/",
        "|",
    ):
        cleaned = cleaned.replace(token, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")

    parsed_date: dt.date | None = None
    if cleaned.startswith("TODAY"):
        parsed_date = market_date
    elif cleaned.startswith("TOMORROW"):
        parsed_date = market_date + dt.timedelta(days=1)
    else:
        fragments = [cleaned]
        if "," in cleaned:
            fragments.append(cleaned.replace(",", ""))
        for frag in fragments:
            for fmt in ("%b %d %Y", "%B %d %Y", "%b %d", "%B %d"):
                try:
                    parsed = dt.datetime.strptime(frag, fmt)
                except ValueError:
                    continue
                year = parsed.year if "%Y" in fmt else market_date.year
                parsed_date = dt.date(year, parsed.month, parsed.day)
                if "%Y" not in fmt and parsed_date < (market_date - dt.timedelta(days=45)):
                    parsed_date = dt.date(year + 1, parsed.month, parsed.day)
                break
            if parsed_date is not None:
                break

    days_until = (parsed_date - market_date).days if parsed_date is not None else None
    return {
        "earnings_raw": raw,
        "earnings_date": parsed_date.isoformat() if parsed_date is not None else None,
        "earnings_session": session,
        "days_until": days_until,
    }


def _select_fund_snapshot(conn: sqlite3.Connection, min_complete_tickers: int = 400) -> tuple[str | None, int]:
    latest = conn.execute(
        """
        SELECT snapshot_ts, COUNT(DISTINCT ticker) AS c
        FROM finviz_fundamentals
        GROUP BY snapshot_ts
        ORDER BY snapshot_ts DESC
        LIMIT 1
        """
    ).fetchone()
    if latest is None:
        return None, 0
    latest_snap = latest[0]
    latest_count = int(latest[1] or 0)
    if latest_count >= min_complete_tickers:
        return latest_snap, latest_count
    latest_complete = conn.execute(
        """
        SELECT snapshot_ts, COUNT(DISTINCT ticker) AS c
        FROM finviz_fundamentals
        GROUP BY snapshot_ts
        HAVING COUNT(DISTINCT ticker) >= ?
        ORDER BY snapshot_ts DESC
        LIMIT 1
        """,
        (min_complete_tickers,),
    ).fetchone()
    if latest_complete is None:
        return latest_snap, latest_count
    return str(latest_complete[0]), int(latest_complete[1] or 0)


def _load_fundamentals_snapshot_map(
    conn: sqlite3.Connection,
    *,
    market_date: dt.date,
    snapshot_ts: str | None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not snapshot_ts:
        return out
    rows = conn.execute(
        """
        SELECT ticker, price, discount_pct, peg, raw_json
        FROM finviz_fundamentals
        WHERE snapshot_ts = ?
        """,
        (snapshot_ts,),
    ).fetchall()
    for row in rows:
        ticker = str(row[0] or "").upper().strip()
        if not ticker:
            continue
        raw_obj: dict[str, Any] = {}
        raw_json = row[4]
        if raw_json:
            try:
                parsed = json.loads(str(raw_json))
                if isinstance(parsed, dict):
                    raw_obj = parsed
            except Exception:
                raw_obj = {}
        earnings_raw = extract_earnings_value(raw_obj)
        parsed_earnings = parse_earnings_value(earnings_raw, market_date) if earnings_raw else {}
        out[ticker] = {
            "ticker": ticker,
            "price": _to_float(row[1]),
            "discount_pct": _to_float(row[2]),
            "peg": _to_float(row[3]),
            "sector": raw_obj.get("Sector") or raw_obj.get("sector") or "Unknown",
            "industry": raw_obj.get("Industry") or raw_obj.get("industry"),
            "earnings_raw": earnings_raw,
            "earnings_date": parsed_earnings.get("earnings_date"),
            "earnings_session": parsed_earnings.get("earnings_session") or "TBD",
        }
    return out


def _fetch_alpha_vantage_calendar_rows(api_key: str, horizon: str) -> list[dict[str, Any]]:
    qs = urllib.parse.urlencode(
        {
            "function": "EARNINGS_CALENDAR",
            "horizon": horizon,
            "apikey": api_key,
        }
    )
    url = f"https://www.alphavantage.co/query?{qs}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "trader_koo/1.0",
            "Accept": "text/csv, application/json;q=0.9, */*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=ALPHA_VANTAGE_TIMEOUT_SEC) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        content_type = str(resp.headers.get("Content-Type", ""))
    text = body.strip()
    if not text:
        return []
    if "json" in content_type.lower() or text.startswith("{"):
        try:
            payload = json.loads(text)
        except Exception as exc:
            raise RuntimeError(f"Alpha Vantage returned invalid JSON: {exc}") from exc
        message = None
        if isinstance(payload, dict):
            for key in ("Error Message", "Information", "Note", "message"):
                if payload.get(key):
                    message = str(payload[key])
                    break
        raise RuntimeError(message or "Alpha Vantage earnings calendar request failed")

    reader = csv.DictReader(io.StringIO(text))
    out: list[dict[str, Any]] = []
    for row in reader:
        symbol = str(row.get("symbol") or row.get("ticker") or "").upper().strip()
        report_date = str(row.get("reportDate") or row.get("date") or "").strip()
        if not symbol or not report_date:
            continue
        out.append(
            {
                "ticker": symbol,
                "company_name": str(row.get("name") or "").strip() or None,
                "earnings_date": report_date,
                "earnings_session": "TBD",
                "fiscal_date_ending": str(row.get("fiscalDateEnding") or "").strip() or None,
                "estimate_eps": _to_float(row.get("estimate")),
                "currency": str(row.get("currency") or "").strip() or None,
                "source": "alpha_vantage",
            }
        )
    return out


def _get_primary_calendar_rows(conn: sqlite3.Connection, days: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    api_key = _alpha_vantage_key()
    if not api_key:
        return [], {
            "provider": None,
            "enabled": False,
            "detail": "TRADER_KOO_ALPHA_VANTAGE_KEY not configured",
            "used_cache": False,
            "stale": False,
        }

    horizon = _alpha_vantage_horizon(days)
    cache_key = f"{CACHE_PROVIDER}:earnings_calendar:{horizon}"
    cached = _cache_load(conn, cache_key)
    if cached and cached.get("is_fresh"):
        return list(cached.get("payload") or []), {
            "provider": CACHE_PROVIDER,
            "enabled": True,
            "used_cache": True,
            "stale": False,
            "age_hours": cached.get("age_hours"),
            "fetched_ts": cached.get("fetched_ts"),
            "horizon": horizon,
            "detail": "Using cached Alpha Vantage earnings calendar",
        }

    try:
        rows = _fetch_alpha_vantage_calendar_rows(api_key, horizon)
        cached = _cache_store(conn, cache_key, CACHE_PROVIDER, rows, CACHE_TTL_HOURS)
        return rows, {
            "provider": CACHE_PROVIDER,
            "enabled": True,
            "used_cache": False,
            "stale": False,
            "age_hours": cached.get("age_hours"),
            "fetched_ts": cached.get("fetched_ts"),
            "horizon": horizon,
            "detail": "Fetched Alpha Vantage earnings calendar",
        }
    except Exception as exc:
        LOG.warning("Alpha Vantage earnings calendar fetch failed: %s", exc)
        if cached:
            return list(cached.get("payload") or []), {
                "provider": CACHE_PROVIDER,
                "enabled": True,
                "used_cache": True,
                "stale": True,
                "age_hours": cached.get("age_hours"),
                "fetched_ts": cached.get("fetched_ts"),
                "horizon": horizon,
                "detail": f"Using stale Alpha Vantage cache after fetch failure: {exc}",
            }
        return [], {
            "provider": CACHE_PROVIDER,
            "enabled": True,
            "used_cache": False,
            "stale": False,
            "horizon": horizon,
            "detail": f"Alpha Vantage fetch failed: {exc}",
        }


def _load_latest_yolo_map(conn: sqlite3.Connection, tickers: set[str] | None = None) -> dict[str, dict[str, Any]]:
    if not table_exists(conn, "yolo_patterns"):
        return {}
    row = conn.execute("SELECT MAX(as_of_date) FROM yolo_patterns").fetchone()
    latest_asof = row[0] if row else None
    if not latest_asof:
        return {}
    sql = (
        "SELECT ticker, timeframe, pattern, CAST(confidence AS REAL) AS confidence, x0_date, x1_date "
        "FROM yolo_patterns WHERE as_of_date = ? ORDER BY confidence DESC"
    )
    params: tuple[Any, ...] = (latest_asof,)
    rows = conn.execute(sql, params).fetchall()
    asof_date: dt.date | None = None
    try:
        asof_date = dt.date.fromisoformat(str(latest_asof))
    except Exception:
        asof_date = None
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        ticker = str(row[0] or "").upper().strip()
        if not ticker:
            continue
        if tickers and ticker not in tickers:
            continue
        conf = float(row[3] or 0.0)
        x1_date = row[5]
        age_days = None
        if asof_date is not None and x1_date:
            try:
                age_days = max(0, (asof_date - dt.date.fromisoformat(str(x1_date)[:10])).days)
            except Exception:
                age_days = None
        candidate = {
            "timeframe": row[1],
            "pattern": row[2],
            "confidence": round(conf, 3),
            "x0_date": row[4],
            "x1_date": x1_date,
            "age_days": age_days,
            "as_of_date": latest_asof,
        }
        prev = out.get(ticker)
        if prev is None:
            out[ticker] = candidate
            continue
        prev_daily = str(prev.get("timeframe") or "") == "daily"
        cand_daily = str(candidate.get("timeframe") or "") == "daily"
        if cand_daily and not prev_daily:
            out[ticker] = candidate
            continue
        if cand_daily == prev_daily and conf > float(prev.get("confidence") or 0.0):
            out[ticker] = candidate
    return out


def _fallback_catalyst_score(row: dict[str, Any]) -> float | None:
    if isinstance(row.get("score"), (int, float)):
        return round(float(row["score"]), 1)
    score = 50.0
    discount = row.get("discount_pct")
    if isinstance(discount, (int, float)):
        score += max(-15.0, min(15.0, float(discount) * 0.6))
    peg = row.get("peg")
    if isinstance(peg, (int, float)) and float(peg) > 0:
        peg_v = float(peg)
        if peg_v <= 0.8:
            score += 12.0
        elif peg_v <= 1.5:
            score += 8.0
        elif peg_v <= 2.5:
            score += 3.0
        elif peg_v > 4.0:
            score -= 6.0
    yolo_conf = row.get("yolo_confidence")
    if isinstance(yolo_conf, (int, float)):
        score += min(12.0, float(yolo_conf) * 12.0)
    risk = str(row.get("earnings_risk") or "normal")
    if risk == "high":
        score -= 10.0
    elif risk == "elevated":
        score -= 5.0
    return round(max(0.0, min(100.0, score)), 1)


def _derive_earnings_risk(row: dict[str, Any]) -> tuple[str, str]:
    points = 0
    days_until = row.get("days_until")
    if isinstance(days_until, int):
        if days_until <= 1:
            points += 3
        elif days_until <= 3:
            points += 2
        elif days_until <= 7:
            points += 1
    realized_vol_20 = row.get("realized_vol_20")
    if isinstance(realized_vol_20, (int, float)):
        if float(realized_vol_20) >= 60.0:
            points += 2
        elif float(realized_vol_20) >= 35.0:
            points += 1
    atr_pct_14 = row.get("atr_pct_14")
    if isinstance(atr_pct_14, (int, float)) and float(atr_pct_14) >= 5.0:
        points += 1
    bb_width_20 = row.get("bb_width_20")
    if isinstance(bb_width_20, (int, float)) and float(bb_width_20) >= 20.0:
        points += 1
    pct_change = row.get("pct_change")
    if isinstance(pct_change, (int, float)) and abs(float(pct_change)) >= 4.0:
        points += 1
    if points >= 5:
        return "high", "Binary earnings risk is elevated; gap-through levels is plausible."
    if points >= 3:
        return "elevated", "Event risk is meaningful; reduce size or wait for post-print confirmation."
    return "normal", "Catalyst risk is present but not unusually stretched for this setup."


def _default_catalyst_action(row: dict[str, Any]) -> str:
    days_until = row.get("days_until")
    bias = str(row.get("signal_bias") or "neutral")
    risk = str(row.get("earnings_risk") or "normal")
    if isinstance(days_until, int) and days_until <= 1:
        return "Wait for the print or the first post-earnings reaction. Do not force a pre-event chase."
    if risk == "high":
        return "Plan levels ahead of the event, but let the earnings reaction prove the setup first."
    if bias == "bullish":
        return "Bullish idea only works if price respects support / reclaim levels into the event window."
    if bias == "bearish":
        return "Bearish idea is actionable only on failed bounces or confirmed support breaks after the event."
    return "Keep it on watch and refresh the chart closer to the catalyst; avoid blind positioning."


def _display_date(day: dt.date) -> str:
    return f"{day.strftime('%a')} {day.strftime('%b')} {day.day}"


def build_earnings_calendar_payload(
    conn: sqlite3.Connection,
    *,
    market_date: dt.date,
    days: int = 21,
    limit: int = 250,
    tickers: set[str] | None = None,
    setup_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    requested = {str(t or "").strip().upper() for t in (tickers or set()) if str(t or "").strip()}
    snapshot_ts, universe_count = _select_fund_snapshot(conn, min_complete_tickers=400)
    fundamentals_map = _load_fundamentals_snapshot_map(conn, market_date=market_date, snapshot_ts=snapshot_ts)
    primary_rows, source_meta = _get_primary_calendar_rows(conn, days)
    yolo_map = _load_latest_yolo_map(conn, tickers=requested or None)
    max_date = market_date + dt.timedelta(days=days)
    setup_map = {str(k or "").upper(): v for k, v in (setup_map or {}).items() if str(k or "").strip()}

    merged: dict[tuple[str, str], dict[str, Any]] = {}

    def _merge_row(seed: dict[str, Any]) -> None:
        ticker = str(seed.get("ticker") or "").upper().strip()
        earnings_date = str(seed.get("earnings_date") or "").strip()
        if not ticker or not earnings_date:
            return
        try:
            event_date = dt.date.fromisoformat(earnings_date[:10])
        except ValueError:
            return
        if event_date < market_date or event_date > max_date:
            return
        if requested and ticker not in requested:
            return
        key = (ticker, event_date.isoformat())
        row = dict(merged.get(key) or {})
        fund = fundamentals_map.get(ticker) or {}
        setup = setup_map.get(ticker) or {}
        yolo = yolo_map.get(ticker) or {}
        session = _normalize_session(seed.get("earnings_session") or fund.get("earnings_session"))
        row.update(
            {
                "ticker": ticker,
                "company_name": seed.get("company_name") or fund.get("company_name"),
                "earnings_date": event_date.isoformat(),
                "display_date": _display_date(event_date),
                "earnings_session": session,
                "earnings_session_label": SESSION_LABELS.get(session, "TBD"),
                "days_until": (event_date - market_date).days,
                "source": seed.get("source") or row.get("source") or "fundamentals_snapshot",
                "provider": seed.get("source") or row.get("provider") or source_meta.get("provider") or "fundamentals_snapshot",
                "fiscal_date_ending": seed.get("fiscal_date_ending") or row.get("fiscal_date_ending"),
                "estimate_eps": seed.get("estimate_eps") if seed.get("estimate_eps") is not None else row.get("estimate_eps"),
                "currency": seed.get("currency") or row.get("currency"),
                "price": fund.get("price"),
                "discount_pct": fund.get("discount_pct"),
                "peg": fund.get("peg"),
                "sector": fund.get("sector") or "Unknown",
                "industry": fund.get("industry"),
                "score": setup.get("score"),
                "setup_tier": setup.get("setup_tier"),
                "signal_bias": setup.get("signal_bias") or ("bearish" if "top" in str(yolo.get("pattern") or "").lower() or "m_head" in str(yolo.get("pattern") or "").lower() else ("bullish" if "bottom" in str(yolo.get("pattern") or "").lower() or "w_bottom" in str(yolo.get("pattern") or "").lower() else "neutral")),
                "actionability": setup.get("actionability") or "watch-only",
                "observation": setup.get("observation") or "Upcoming earnings catalyst; refresh the chart and levels before acting.",
                "action": setup.get("action"),
                "technical_read": setup.get("technical_read"),
                "risk_note": setup.get("risk_note"),
                "yolo_pattern": setup.get("yolo_pattern") or yolo.get("pattern"),
                "yolo_confidence": setup.get("yolo_confidence") if setup.get("yolo_confidence") is not None else yolo.get("confidence"),
                "yolo_timeframe": setup.get("yolo_timeframe") or yolo.get("timeframe"),
                "atr_pct_14": setup.get("atr_pct_14"),
                "realized_vol_20": setup.get("realized_vol_20"),
                "bb_width_20": setup.get("bb_width_20"),
                "pct_change": setup.get("pct_change"),
            }
        )
        risk, risk_note = _derive_earnings_risk(row)
        row["earnings_risk"] = risk
        row["earnings_risk_note"] = risk_note
        if not row.get("action"):
            row["action"] = _default_catalyst_action(row)
        row["score"] = _fallback_catalyst_score(row)
        merged[key] = row

    for row in primary_rows:
        _merge_row(row)
    for fund in fundamentals_map.values():
        if fund.get("earnings_date"):
            _merge_row(
                {
                    "ticker": fund.get("ticker"),
                    "earnings_date": fund.get("earnings_date"),
                    "earnings_session": fund.get("earnings_session"),
                    "source": "fundamentals_snapshot",
                }
            )

    rows = sorted(
        merged.values(),
        key=lambda r: (
            str(r.get("earnings_date") or ""),
            SESSION_ORDER.get(str(r.get("earnings_session") or "TBD"), 1),
            -(float(r.get("score") or 0.0)),
            str(r.get("ticker") or ""),
        ),
    )
    rows = rows[:limit]

    groups: list[dict[str, Any]] = []
    high_risk = 0
    elevated_risk = 0
    by_session = {"BMO": 0, "TBD": 0, "AMC": 0}
    current_day: dt.date | None = None
    current_rows: list[dict[str, Any]] = []
    for row in rows:
        session = str(row.get("earnings_session") or "TBD")
        by_session[session] = by_session.get(session, 0) + 1
        risk = str(row.get("earnings_risk") or "normal")
        if risk == "high":
            high_risk += 1
        elif risk == "elevated":
            elevated_risk += 1
        try:
            row_day = dt.date.fromisoformat(str(row.get("earnings_date") or "")[:10])
        except ValueError:
            continue
        if current_day is None:
            current_day = row_day
        if row_day != current_day:
            groups.append(_build_group(current_day, current_rows))
            current_day = row_day
            current_rows = []
        current_rows.append(row)
    if current_day is not None:
        groups.append(_build_group(current_day, current_rows))

    provider_label = source_meta.get("provider") or "fundamentals_snapshot"
    detail = None if rows else f"No upcoming earnings found in the next {days} days."
    return {
        "ok": True,
        "market_date": market_date.isoformat(),
        "snapshot_ts": snapshot_ts,
        "universe_count": universe_count,
        "requested_tickers": sorted(requested),
        "count": len(rows),
        "rows": rows,
        "groups": groups,
        "detail": detail,
        "provider": provider_label,
        "provider_status": source_meta,
        "summary": {
            "window_days": days,
            "total_events": len(rows),
            "high_risk": high_risk,
            "elevated_risk": elevated_risk,
            "by_session": by_session,
            "scored_rows": sum(1 for row in rows if isinstance(row.get("score"), (int, float))),
        },
    }


def _build_group(day: dt.date, rows: list[dict[str, Any]]) -> dict[str, Any]:
    session_rows = {code: [] for code in ("BMO", "TBD", "AMC")}
    for row in rows:
        code = _normalize_session(row.get("earnings_session"))
        session_rows.setdefault(code, []).append(row)
    sessions = []
    for code in ("BMO", "TBD", "AMC"):
        entries = session_rows.get(code) or []
        sessions.append(
            {
                "code": code,
                "label": SESSION_LABELS.get(code, code),
                "count": len(entries),
                "rows": entries,
            }
        )
    return {
        "date": day.isoformat(),
        "display_date": _display_date(day),
        "count": len(rows),
        "sessions": sessions,
    }


def get_ticker_earnings_markers(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    market_date: dt.date,
    forward_days: int = 120,
    max_markers: int = 3,
) -> list[dict[str, Any]]:
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return []
    payload = build_earnings_calendar_payload(
        conn,
        market_date=market_date,
        days=forward_days,
        limit=max_markers,
        tickers={symbol},
        setup_map=None,
    )
    markers: list[dict[str, Any]] = []
    for row in payload.get("rows", [])[:max_markers]:
        session = _normalize_session(row.get("earnings_session"))
        markers.append(
            {
                "ticker": symbol,
                "date": row.get("earnings_date"),
                "session": session,
                "session_label": SESSION_LABELS.get(session, "TBD"),
                "days_until": row.get("days_until"),
                "label": f"Earnings {SESSION_LABELS.get(session, 'TBD')}",
                "earnings_risk": row.get("earnings_risk"),
                "source": row.get("source"),
            }
        )
    return markers


def _to_float(value: Any) -> float | None:
    try:
        if value in {None, "", "-"}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
