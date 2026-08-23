"""Bounded, auditable restatement of explicitly proposed price rows."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import sqlite3
from typing import Any, Iterable

from trader_koo.db.price_contract import record_price_series_revision

PRICE_FIELDS = ("open", "high", "low", "close", "volume")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def ensure_price_repair_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS price_repair_runs (
               run_id TEXT PRIMARY KEY,
               plan_sha256 TEXT NOT NULL UNIQUE,
               adjustment_version TEXT NOT NULL,
               reason TEXT NOT NULL,
               provider_evidence_json TEXT NOT NULL,
               planned_change_count INTEGER NOT NULL,
               created_ts TEXT NOT NULL
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS price_corrections (
               correction_id TEXT PRIMARY KEY,
               run_id TEXT NOT NULL REFERENCES price_repair_runs(run_id),
               ticker TEXT NOT NULL,
               price_date TEXT NOT NULL,
               original_json TEXT NOT NULL,
               proposed_json TEXT NOT NULL,
               action_json TEXT NOT NULL,
               source_row_sha256 TEXT NOT NULL,
               adjustment_version TEXT NOT NULL,
               reason TEXT NOT NULL,
               applied_ts TEXT NOT NULL,
               UNIQUE(run_id,ticker,price_date)
           )"""
    )
    for table in ("price_repair_runs", "price_corrections"):
        conn.execute(
            f"""CREATE TRIGGER IF NOT EXISTS {table}_no_update
                BEFORE UPDATE ON {table}
                BEGIN SELECT RAISE(ABORT,'{table} is append-only'); END"""
        )
        conn.execute(
            f"""CREATE TRIGGER IF NOT EXISTS {table}_no_delete
                BEFORE DELETE ON {table}
                BEGIN SELECT RAISE(ABORT,'{table} is append-only'); END"""
        )


def _row(conn: sqlite3.Connection, ticker: str, date: str) -> dict[str, Any] | None:
    raw = conn.execute(
        """SELECT open,high,low,close,volume,data_source,fetch_timestamp,
                  adjustment_basis,adjustment_version,basis_status,unresolved_reason
           FROM price_daily WHERE ticker=? AND date=?""",
        (ticker, date),
    ).fetchone()
    if raw is None:
        return None
    keys = (*PRICE_FIELDS, "data_source", "fetch_timestamp", "adjustment_basis",
            "adjustment_version", "basis_status", "unresolved_reason")
    return dict(zip(keys, raw))


def _dependents(conn: sqlite3.Connection, ticker: str, date: str) -> dict[str, int]:
    result: dict[str, int] = {}
    checks = {
        "setup_calls": ("ticker", "report_date"),
        "paper_trades": ("ticker", "entry_date"),
        "yolo_patterns": ("ticker", "date"),
    }
    for table, (ticker_column, date_column) in checks.items():
        columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
        if {ticker_column, date_column}.issubset(columns):
            result[table] = int(conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {ticker_column}=? AND {date_column}>=?",
                (ticker, date),
            ).fetchone()[0])
    return result


def plan_price_repair(
    conn: sqlite3.Connection,
    proposals: Iterable[dict[str, Any]],
    *,
    adjustment_version: str,
    reason: str,
    provider_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Dry-run exact proposed rows; no database state is changed."""
    if not adjustment_version.strip() or not reason.strip():
        raise ValueError("adjustment_version and reason are required")
    if (
        provider_evidence.get("vendor_action_ledger_checked") is not True
        or provider_evidence.get("full_history_verified") is not True
    ):
        raise ValueError("full-history provider action evidence is required")
    changes: list[dict[str, Any]] = []
    unchanged: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in proposals:
        ticker = str(raw.get("ticker") or "").strip().upper()
        date = str(raw.get("date") or "").strip()
        identity = (ticker, date)
        action = raw.get("action")
        if not ticker or not date or identity in seen:
            unresolved.append({"ticker": ticker, "date": date, "reason": "invalid_or_duplicate_identity"})
            continue
        seen.add(identity)
        if (
            not isinstance(action, dict)
            or action.get("action_type") not in {"split", "reverse_split", "rebase"}
            or not action.get("action_date")
            or not isinstance(action.get("factor"), (int, float))
        ):
            unresolved.append({"ticker": ticker, "date": date, "reason": "declared_action_required"})
            continue
        proposed: dict[str, float | None] = {}
        try:
            for field in PRICE_FIELDS:
                value = raw.get(field)
                proposed[field] = None if value is None else float(value)
            if any(
                value is not None and not math.isfinite(value)
                for value in proposed.values()
            ) or any(float(proposed[field] or 0) <= 0 for field in ("open", "high", "low", "close")):
                raise ValueError
        except (TypeError, ValueError):
            unresolved.append({"ticker": ticker, "date": date, "reason": "invalid_proposed_ohlcv"})
            continue
        original = _row(conn, ticker, date)
        if original is None:
            unresolved.append({"ticker": ticker, "date": date, "reason": "source_row_missing"})
            continue
        if all(original[field] == proposed[field] for field in PRICE_FIELDS):
            unchanged.append({"ticker": ticker, "date": date})
            continue
        changes.append({
            "ticker": ticker,
            "date": date,
            "original": original,
            "proposed": proposed,
            "action": action,
            "source_row_sha256": _hash(original),
            "dependent_artifacts": _dependents(conn, ticker, date),
        })
    material = {
        "schema_version": "price-repair-v1",
        "adjustment_version": adjustment_version,
        "reason": reason,
        "provider_evidence": provider_evidence,
        "changes": changes,
        "unchanged": unchanged,
        "unresolved": unresolved,
    }
    return {**material, "plan_sha256": _hash(material), "apply_eligible": bool(changes) and not unresolved}


def apply_price_repair(conn: sqlite3.Connection, plan: dict[str, Any]) -> dict[str, Any]:
    """Apply one hash-bound plan, preserving every original row."""
    material = {key: value for key, value in plan.items() if key not in {"plan_sha256", "apply_eligible"}}
    if plan.get("plan_sha256") != _hash(material):
        raise ValueError("price repair plan hash mismatch")
    if not plan.get("apply_eligible") or plan.get("unresolved"):
        raise ValueError("price repair plan is not apply eligible")
    ensure_price_repair_schema(conn)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    run_id = _hash({"plan_sha256": plan["plan_sha256"], "kind": "price_repair"})
    conn.execute(
        """INSERT OR IGNORE INTO price_repair_runs
               (run_id,plan_sha256,adjustment_version,reason,provider_evidence_json,
                planned_change_count,created_ts)
           VALUES (?,?,?,?,?,?,?)""",
        (run_id, plan["plan_sha256"], plan["adjustment_version"], plan["reason"],
         _json(plan["provider_evidence"]), len(plan["changes"]), now),
    )
    changed = 0
    already_applied = 0
    tickers: set[str] = set()
    for item in plan["changes"]:
        current = _row(conn, item["ticker"], item["date"])
        if current is None:
            raise ValueError(f"source row disappeared for {item['ticker']} {item['date']}")
        if all(current[field] == item["proposed"][field] for field in PRICE_FIELDS):
            already_applied += 1
            continue
        if _hash(current) != item["source_row_sha256"]:
            raise ValueError(f"source row drifted for {item['ticker']} {item['date']}")
        correction_id = _hash({"run_id": run_id, "ticker": item["ticker"], "date": item["date"]})
        conn.execute(
            """INSERT INTO price_corrections
                   (correction_id,run_id,ticker,price_date,original_json,proposed_json,
                    action_json,source_row_sha256,adjustment_version,reason,applied_ts)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (correction_id, run_id, item["ticker"], item["date"], _json(item["original"]),
             _json(item["proposed"]), _json(item["action"]), item["source_row_sha256"],
             plan["adjustment_version"], plan["reason"], now),
        )
        values = [item["proposed"][field] for field in PRICE_FIELDS]
        conn.execute(
            """UPDATE price_daily SET open=?,high=?,low=?,close=?,volume=?,
                      adjustment_version=?,basis_status='verified',unresolved_reason=NULL
               WHERE ticker=? AND date=?""",
            (*values, plan["adjustment_version"], item["ticker"], item["date"]),
        )
        changed += 1
        tickers.add(item["ticker"])
    for ticker in sorted(tickers):
        conn.execute(
            """UPDATE price_daily SET adjustment_version=?,basis_status='verified',
                      unresolved_reason=NULL WHERE ticker=?""",
            (plan["adjustment_version"], ticker),
        )
        for item in plan["changes"]:
            if item["ticker"] != ticker:
                continue
            action = item["action"]
            provider = str(plan["provider_evidence"].get("provider") or "repair_plan")
            evidence = {
                **action,
                "provider": provider,
                "repair_run_id": run_id,
                "full_history_verified": True,
            }
            conn.execute(
                """INSERT INTO price_corporate_actions
                       (ticker,action_date,action_type,provider,value,applied_to_prices,
                        adjustment_version,fetch_timestamp,evidence_json)
                   VALUES (?,?,?,?,?,1,?,?,?)
                   ON CONFLICT(ticker,action_date,action_type,provider) DO UPDATE SET
                       value=excluded.value,
                       applied_to_prices=1,
                       adjustment_version=excluded.adjustment_version,
                       fetch_timestamp=excluded.fetch_timestamp,
                       evidence_json=excluded.evidence_json""",
                (ticker, action["action_date"], action["action_type"], provider,
                 float(action["factor"]), plan["adjustment_version"], now, _json(evidence)),
            )
        record_price_series_revision(
            conn,
            ticker,
            evidence={
                **plan["provider_evidence"],
                "repair_run_id": run_id,
                "repair_plan_sha256": plan["plan_sha256"],
                "normalization_actions": [
                    item["action"] for item in plan["changes"] if item["ticker"] == ticker
                ],
            },
            fetch_timestamp=now,
        )
    return {
        "run_id": run_id,
        "plan_sha256": plan["plan_sha256"],
        "changed_rows": changed,
        "already_applied_rows": already_applied,
        "tickers": sorted(tickers),
    }
