"""Qlib-style validation registry for model/backtest experiments.

This module intentionally tracks experiments separately from model files. A
saved LightGBM model is not automatically a deployable/champion model; a run
must pass explicit out-of-sample gates before it is eligible for promotion.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from typing import Any

MIN_FOLD_COUNT = 5
MIN_AVG_AUC = 0.55
MIN_ALPHA_VS_SPY_PCT = 2.0
MIN_PROFIT_FACTOR = 1.20
MIN_TOTAL_TRADES = 50
MAX_DRAWDOWN_PCT = -15.0


_DDL = """
CREATE TABLE IF NOT EXISTS ml_validation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    created_ts TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    target_mode TEXT,
    start_date TEXT,
    end_date TEXT,
    fold_count INTEGER,
    total_samples INTEGER,
    avg_auc REAL,
    best_auc REAL,
    avg_accuracy REAL,
    avg_precision REAL,
    model_path TEXT,
    artifact_path TEXT,
    backtest_return_pct REAL,
    spy_return_pct REAL,
    alpha_vs_spy_pct REAL,
    rule_baseline_return_pct REAL,
    alpha_vs_rule_baseline_pct REAL,
    max_drawdown_pct REAL,
    total_trades INTEGER,
    profit_factor REAL,
    champion_eligible INTEGER NOT NULL DEFAULT 0,
    promotion_status TEXT NOT NULL,
    eligibility_reasons_json TEXT,
    details_json TEXT
)
"""


def ensure_validation_registry_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_DDL)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ml_validation_runs_created "
        "ON ml_validation_runs(created_ts)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ml_validation_runs_status "
        "ON ml_validation_runs(promotion_status, created_ts)"
    )
    conn.commit()


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _training_metrics(result: dict[str, Any]) -> dict[str, Any]:
    training = result.get("training") if isinstance(result.get("training"), dict) else result
    agg = training.get("aggregate_metrics") if isinstance(training.get("aggregate_metrics"), dict) else {}
    return {
        "target_mode": training.get("target_mode") or result.get("target_mode"),
        "fold_count": _to_int(training.get("fold_count")),
        "total_samples": _to_int(training.get("total_samples")),
        "avg_auc": _to_float(agg.get("avg_auc")),
        "best_auc": _to_float(agg.get("best_auc")),
        "avg_accuracy": _to_float(agg.get("avg_accuracy")),
        "avg_precision": _to_float(agg.get("avg_precision")),
        "model_path": training.get("model_path") or result.get("model_path"),
    }


def _backtest_metrics(result: dict[str, Any]) -> dict[str, Any]:
    backtest = result.get("backtest") if isinstance(result.get("backtest"), dict) else {}
    summary = backtest.get("summary") if isinstance(backtest.get("summary"), dict) else {}
    rule = result.get("rule_baseline") if isinstance(result.get("rule_baseline"), dict) else {}
    rule_summary = rule.get("summary") if isinstance(rule.get("summary"), dict) else {}
    model_return = _to_float(summary.get("total_return_pct"))
    rule_return = (
        _to_float(rule.get("return_pct"))
        if rule.get("return_pct") is not None
        else _to_float(rule_summary.get("return_pct") or rule_summary.get("total_return_pct"))
    )
    return {
        "backtest_return_pct": model_return,
        "spy_return_pct": _to_float(summary.get("spy_return_pct")),
        "alpha_vs_spy_pct": _to_float(summary.get("alpha_pct")),
        "rule_baseline_return_pct": rule_return,
        "alpha_vs_rule_baseline_pct": (
            round(model_return - rule_return, 2)
            if model_return is not None and rule_return is not None
            else None
        ),
        "max_drawdown_pct": _to_float(summary.get("max_drawdown_pct")),
        "total_trades": _to_int(summary.get("total_trades")),
        "profit_factor": _to_float(summary.get("profit_factor")),
    }


def extract_validation_metrics(result: dict[str, Any]) -> dict[str, Any]:
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    metrics = {
        "status": "ok" if result.get("ok") else "failed",
        "start_date": result.get("start_date") or config.get("start_date"),
        "end_date": result.get("end_date") or config.get("end_date"),
    }
    metrics.update(_training_metrics(result))
    metrics.update(_backtest_metrics(result))
    return metrics


def evaluate_champion_eligibility(result: dict[str, Any]) -> dict[str, Any]:
    metrics = extract_validation_metrics(result)
    reasons: list[str] = []

    if metrics["status"] != "ok":
        reasons.append("run_failed")
    if (metrics.get("fold_count") or 0) < MIN_FOLD_COUNT:
        reasons.append(f"fold_count_below_{MIN_FOLD_COUNT}")
    if metrics.get("avg_auc") is None or float(metrics["avg_auc"]) < MIN_AVG_AUC:
        reasons.append(f"avg_auc_below_{MIN_AVG_AUC:.2f}")
    if metrics.get("alpha_vs_spy_pct") is None:
        reasons.append("missing_spy_backtest")
    elif float(metrics["alpha_vs_spy_pct"]) < MIN_ALPHA_VS_SPY_PCT:
        reasons.append(f"alpha_vs_spy_below_{MIN_ALPHA_VS_SPY_PCT:.1f}pp")
    if metrics.get("rule_baseline_return_pct") is None:
        reasons.append("missing_rule_baseline")
    elif (
        metrics.get("alpha_vs_rule_baseline_pct") is None
        or float(metrics["alpha_vs_rule_baseline_pct"]) <= 0
    ):
        reasons.append("does_not_beat_rule_baseline")
    if (metrics.get("total_trades") or 0) < MIN_TOTAL_TRADES:
        reasons.append(f"trade_count_below_{MIN_TOTAL_TRADES}")
    if metrics.get("profit_factor") is None or float(metrics["profit_factor"]) < MIN_PROFIT_FACTOR:
        reasons.append(f"profit_factor_below_{MIN_PROFIT_FACTOR:.2f}")
    if metrics.get("max_drawdown_pct") is None:
        reasons.append("missing_drawdown")
    elif float(metrics["max_drawdown_pct"]) <= MAX_DRAWDOWN_PCT:
        reasons.append(f"drawdown_worse_than_{abs(MAX_DRAWDOWN_PCT):.0f}pct")

    eligible = len(reasons) == 0
    return {
        "champion_eligible": eligible,
        "promotion_status": "eligible" if eligible else "blocked",
        "eligibility_reasons": reasons,
        "metrics": metrics,
    }


def record_validation_run(
    conn: sqlite3.Connection,
    result: dict[str, Any],
    *,
    source: str,
    artifact_path: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Persist a validation result and return the inserted registry row."""
    ensure_validation_registry_schema(conn)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    final_run_id = run_id or f"{source}-{now}"
    evaluation = evaluate_champion_eligibility(result)
    metrics = evaluation["metrics"]
    reasons = evaluation["eligibility_reasons"]

    conn.execute(
        """
        INSERT INTO ml_validation_runs (
            run_id, created_ts, source, status, target_mode, start_date, end_date,
            fold_count, total_samples, avg_auc, best_auc, avg_accuracy, avg_precision,
            model_path, artifact_path, backtest_return_pct, spy_return_pct,
            alpha_vs_spy_pct, rule_baseline_return_pct, alpha_vs_rule_baseline_pct,
            max_drawdown_pct, total_trades, profit_factor, champion_eligible,
            promotion_status, eligibility_reasons_json, details_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            status = excluded.status,
            target_mode = excluded.target_mode,
            start_date = excluded.start_date,
            end_date = excluded.end_date,
            fold_count = excluded.fold_count,
            total_samples = excluded.total_samples,
            avg_auc = excluded.avg_auc,
            best_auc = excluded.best_auc,
            avg_accuracy = excluded.avg_accuracy,
            avg_precision = excluded.avg_precision,
            model_path = excluded.model_path,
            artifact_path = excluded.artifact_path,
            backtest_return_pct = excluded.backtest_return_pct,
            spy_return_pct = excluded.spy_return_pct,
            alpha_vs_spy_pct = excluded.alpha_vs_spy_pct,
            rule_baseline_return_pct = excluded.rule_baseline_return_pct,
            alpha_vs_rule_baseline_pct = excluded.alpha_vs_rule_baseline_pct,
            max_drawdown_pct = excluded.max_drawdown_pct,
            total_trades = excluded.total_trades,
            profit_factor = excluded.profit_factor,
            champion_eligible = excluded.champion_eligible,
            promotion_status = excluded.promotion_status,
            eligibility_reasons_json = excluded.eligibility_reasons_json,
            details_json = excluded.details_json
        """,
        (
            final_run_id,
            now,
            source,
            metrics.get("status"),
            metrics.get("target_mode"),
            metrics.get("start_date"),
            metrics.get("end_date"),
            metrics.get("fold_count"),
            metrics.get("total_samples"),
            metrics.get("avg_auc"),
            metrics.get("best_auc"),
            metrics.get("avg_accuracy"),
            metrics.get("avg_precision"),
            metrics.get("model_path"),
            artifact_path,
            metrics.get("backtest_return_pct"),
            metrics.get("spy_return_pct"),
            metrics.get("alpha_vs_spy_pct"),
            metrics.get("rule_baseline_return_pct"),
            metrics.get("alpha_vs_rule_baseline_pct"),
            metrics.get("max_drawdown_pct"),
            metrics.get("total_trades"),
            metrics.get("profit_factor"),
            int(bool(evaluation["champion_eligible"])),
            evaluation["promotion_status"],
            json.dumps(reasons),
            json.dumps(result, default=str),
        ),
    )
    conn.commit()
    return {
        "run_id": final_run_id,
        "source": source,
        **metrics,
        "artifact_path": artifact_path,
        "champion_eligible": evaluation["champion_eligible"],
        "promotion_status": evaluation["promotion_status"],
        "eligibility_reasons": reasons,
    }


def list_validation_runs(conn: sqlite3.Connection, *, limit: int = 20) -> list[dict[str, Any]]:
    ensure_validation_registry_schema(conn)
    cur = conn.execute(
        """
        SELECT *
        FROM ml_validation_runs
        ORDER BY created_ts DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description or []]
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row) if isinstance(row, sqlite3.Row) else dict(zip(columns, row))
        raw = item.get("eligibility_reasons_json")
        try:
            item["eligibility_reasons"] = json.loads(raw or "[]")
        except Exception:
            item["eligibility_reasons"] = []
        item["champion_eligible"] = bool(item.get("champion_eligible"))
        out.append(item)
    return out


def _compact_run(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the fields an operator needs for a model/version card."""
    if not row:
        return None
    reasons = row.get("eligibility_reasons")
    if not isinstance(reasons, list):
        try:
            reasons = json.loads(row.get("eligibility_reasons_json") or "[]")
        except Exception:
            reasons = []
    return {
        "run_id": row.get("run_id"),
        "created_ts": row.get("created_ts"),
        "source": row.get("source"),
        "status": row.get("status"),
        "target_mode": row.get("target_mode"),
        "model_path": row.get("model_path"),
        "artifact_path": row.get("artifact_path"),
        "promotion_status": row.get("promotion_status"),
        "champion_eligible": bool(row.get("champion_eligible")),
        "eligibility_reasons": reasons,
        "metrics": {
            "avg_auc": row.get("avg_auc"),
            "alpha_vs_spy_pct": row.get("alpha_vs_spy_pct"),
            "rule_baseline_return_pct": row.get("rule_baseline_return_pct"),
            "alpha_vs_rule_baseline_pct": row.get("alpha_vs_rule_baseline_pct"),
            "max_drawdown_pct": row.get("max_drawdown_pct"),
            "total_trades": row.get("total_trades"),
            "profit_factor": row.get("profit_factor"),
        },
    }


def _promotion_gates() -> dict[str, Any]:
    return {
        "min_fold_count": MIN_FOLD_COUNT,
        "min_avg_auc": MIN_AVG_AUC,
        "min_alpha_vs_spy_pct": MIN_ALPHA_VS_SPY_PCT,
        "min_profit_factor": MIN_PROFIT_FACTOR,
        "min_total_trades": MIN_TOTAL_TRADES,
        "max_drawdown_pct": MAX_DRAWDOWN_PCT,
        "requires_rule_baseline": True,
    }


def champion_status(conn: sqlite3.Connection) -> dict[str, Any]:
    ensure_validation_registry_schema(conn)
    latest = list_validation_runs(conn, limit=1)
    cur = conn.execute(
        """
        SELECT *
        FROM ml_validation_runs
        WHERE champion_eligible = 1
        ORDER BY created_ts DESC, id DESC
        LIMIT 1
        """
    )
    eligible = cur.fetchone()
    columns = [desc[0] for desc in cur.description or []]
    eligible_item = None
    if eligible:
        eligible_item = dict(eligible) if isinstance(eligible, sqlite3.Row) else dict(zip(columns, eligible))
        if eligible_item is not None:
            try:
                eligible_item["eligibility_reasons"] = json.loads(
                    eligible_item.get("eligibility_reasons_json") or "[]"
                )
            except Exception:
                eligible_item["eligibility_reasons"] = []
            eligible_item["champion_eligible"] = True
    return {
        "ok": True,
        "latest_run": latest[0] if latest else None,
        "latest_eligible_run": eligible_item,
        "promotion_gates": _promotion_gates(),
    }


def model_version_card(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return a compact operator-facing model version summary."""
    status = champion_status(conn)
    latest = status.get("latest_run")
    eligible = status.get("latest_eligible_run")

    if latest is None:
        deployment_state = "no_validation_runs"
        summary = "No validation run has been recorded yet."
    elif latest.get("champion_eligible"):
        deployment_state = "latest_candidate_eligible"
        summary = "Latest candidate passed the promotion gates."
    elif eligible:
        deployment_state = "using_previous_eligible"
        summary = "Latest candidate is blocked; keep using the latest eligible run."
    else:
        deployment_state = "no_eligible_champion"
        summary = "Latest candidate is blocked and no eligible run is recorded yet."

    return {
        "ok": True,
        "deployment_state": deployment_state,
        "summary": summary,
        "candidate": _compact_run(latest),
        "latest_eligible": _compact_run(eligible),
        "promotion_gates": _promotion_gates(),
    }
