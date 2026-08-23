"""Append-only, redacted observability for real LLM calls.

Deterministic rules never call this module and therefore never appear as agent
spans. Raw prompts, credentials, headers, and unrestricted model output are
not persisted or returned by the read API.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import sqlite3
import statistics
import uuid
from pathlib import Path
from typing import Any

from trader_koo.llm_health import (
    llm_cost_input_per_1m,
    llm_cost_output_per_1m,
    llm_health_summary,
)

PROMPT_RETENTION = "hash_only"
TRACE_RETENTION_DAYS = 365
_SENSITIVE = ("api_key", "authorization", "credential", "password", "secret", "token")


def _iso(value: dt.datetime | None = None) -> str:
    ts = value or dt.datetime.now(dt.timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc).isoformat()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if any(part in str(key).lower() for part in _SENSITIVE)
                else _redact(item)
            )
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return value[:2000]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value)[:2000]


def redacted_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(_redact(value)).encode()).hexdigest()


def ensure_observability_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS llm_run_graphs (
            run_id TEXT PRIMARY KEY,
            graph_version TEXT NOT NULL,
            graph_kind TEXT NOT NULL,
            report_run_id TEXT,
            campaign_id TEXT,
            ticker TEXT,
            started_ts TEXT NOT NULL,
            ended_ts TEXT NOT NULL,
            terminal_status TEXT NOT NULL,
            disagreement INTEGER NOT NULL DEFAULT 0,
            adjudicator_role TEXT,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS llm_call_traces (
            trace_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES llm_run_graphs(run_id),
            span_id TEXT NOT NULL UNIQUE,
            parent_span_id TEXT,
            report_run_id TEXT,
            campaign_id TEXT,
            ticker TEXT,
            role TEXT NOT NULL,
            stage TEXT NOT NULL,
            source TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT,
            deployment TEXT,
            prompt_template_version TEXT NOT NULL,
            evaluator_version TEXT,
            evaluation_result_json TEXT,
            cache_identity_sha256 TEXT,
            redacted_input_sha256 TEXT NOT NULL,
            redacted_output_sha256 TEXT NOT NULL,
            started_ts TEXT NOT NULL,
            ended_ts TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            estimated_cost_usd REAL,
            retry_count INTEGER NOT NULL DEFAULT 0,
            validator_result TEXT NOT NULL,
            fallback_reason TEXT,
            terminal_status TEXT NOT NULL,
            message_artifact_sha256 TEXT,
            retention_class TEXT NOT NULL DEFAULT 'hash_only',
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_llm_call_traces_started
            ON llm_call_traces(started_ts DESC);
        CREATE INDEX IF NOT EXISTS idx_llm_call_traces_run
            ON llm_call_traces(run_id,span_id);
        CREATE TABLE IF NOT EXISTS llm_contributions (
            contribution_id TEXT PRIMARY KEY,
            trace_id TEXT NOT NULL REFERENCES llm_call_traces(trace_id),
            decision_scope TEXT NOT NULL,
            deterministic_pre_sha256 TEXT NOT NULL,
            proposed_change_json TEXT,
            proposed_change_sha256 TEXT NOT NULL,
            final_adjudicated_sha256 TEXT NOT NULL,
            changed_fields_json TEXT NOT NULL,
            content_changed INTEGER,
            decision_changed INTEGER NOT NULL,
            decision_contract_changed INTEGER,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS llm_outcome_links (
            link_id TEXT PRIMARY KEY,
            trace_id TEXT NOT NULL REFERENCES llm_call_traces(trace_id),
            paper_trade_id INTEGER NOT NULL,
            outcome_sha256 TEXT NOT NULL,
            analysis_label TEXT NOT NULL DEFAULT 'observational_non_causal',
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)
    trace_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(llm_call_traces)")}
    for name in ("evaluator_version", "evaluation_result_json", "cache_identity_sha256"):
        if name not in trace_columns:
            conn.execute(f"ALTER TABLE llm_call_traces ADD COLUMN {name} TEXT")
    contribution_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(llm_contributions)")
    }
    for name in ("content_changed", "decision_contract_changed"):
        if name not in contribution_columns:
            conn.execute(f"ALTER TABLE llm_contributions ADD COLUMN {name} INTEGER")
    for table in (
        "llm_run_graphs", "llm_call_traces", "llm_contributions",
        "llm_outcome_links",
    ):
        conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {table}_immutable_update
            BEFORE UPDATE ON {table}
            BEGIN SELECT RAISE(ABORT,'LLM observability records are append-only'); END
        """)
        conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {table}_immutable_delete
            BEFORE DELETE ON {table}
            BEGIN SELECT RAISE(ABORT,'LLM observability records are append-only'); END
        """)


def _tokens(usage: dict[str, Any], key: str) -> int:
    try:
        return max(0, int(float(usage.get(key) or 0)))
    except (TypeError, ValueError):
        return 0


def _cost(prompt_tokens: int, completion_tokens: int) -> float | None:
    input_rate = llm_cost_input_per_1m()
    output_rate = llm_cost_output_per_1m()
    if input_rate <= 0 and output_rate <= 0:
        return None
    return round(
        prompt_tokens / 1_000_000 * input_rate
        + completion_tokens / 1_000_000 * output_rate,
        8,
    )


def record_llm_call(
    db_path: Path,
    *,
    source: str,
    role: str,
    stage: str,
    provider: str,
    model: str | None,
    deployment: str | None,
    prompt_template_version: str,
    input_payload: dict[str, Any],
    proposed_output: dict[str, Any] | None,
    deterministic_pre: dict[str, Any],
    final_adjudicated: dict[str, Any],
    started_at: dt.datetime,
    ended_at: dt.datetime | None = None,
    usage: dict[str, Any] | None = None,
    validator_result: str,
    fallback_reason: str | None,
    terminal_status: str,
    retry_count: int = 0,
    report_run_id: str | None = None,
    campaign_id: str | None = None,
    ticker: str | None = None,
    parent_span_id: str | None = None,
    disagreement: bool = False,
    adjudicator_role: str | None = "deterministic_validator",
    decision_scope: str = "observation_narrative_only",
    evaluation_result: dict[str, Any] | None = None,
    cache_identity_sha256: str | None = None,
) -> dict[str, str]:
    """Append one completed real-LLM span and its bounded contribution."""
    db_path = Path(db_path)
    ended = ended_at or dt.datetime.now(dt.timezone.utc)
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=dt.timezone.utc)
    if ended.tzinfo is None:
        ended = ended.replace(tzinfo=dt.timezone.utc)
    run_id, trace_id, span_id = (str(uuid.uuid4()) for _ in range(3))
    usage = usage if isinstance(usage, dict) else {}
    prompt_tokens = _tokens(usage, "prompt_tokens")
    completion_tokens = _tokens(usage, "completion_tokens")
    total_tokens = _tokens(usage, "total_tokens") or prompt_tokens + completion_tokens
    proposed = _redact(proposed_output or {})
    pre = _redact(deterministic_pre)
    final = _redact(final_adjudicated)
    changed_fields = sorted(
        key for key in set(pre) | set(final) if pre.get(key) != final.get(key)
    ) if isinstance(pre, dict) and isinstance(final, dict) else []
    decision_fields = {"action", "risk_note", "intent", "signal_bias", "actionability"}
    decision_changed = any(field in decision_fields for field in changed_fields)
    input_hash = redacted_hash(input_payload)
    output_hash = redacted_hash(final)
    proposed_hash = redacted_hash(proposed)
    evaluation = _redact(evaluation_result or {})
    artifact_hash = redacted_hash({
        "input": input_hash, "output": output_hash,
        "prompt_template_version": prompt_template_version,
    })
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_observability_schema(conn)
        conn.execute(
            """INSERT INTO llm_run_graphs (
                   run_id,graph_version,graph_kind,report_run_id,campaign_id,ticker,
                   started_ts,ended_ts,terminal_status,disagreement,adjudicator_role
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id, "llm-run-graph-v1",
                "cached_llm_result" if terminal_status == "cache_hit" else "single_llm_call",
                report_run_id,
                campaign_id, str(ticker or "").upper() or None, _iso(started_at),
                _iso(ended), terminal_status, int(disagreement), adjudicator_role,
            ),
        )
        conn.execute(
            """INSERT INTO llm_call_traces (
                   trace_id,run_id,span_id,parent_span_id,report_run_id,campaign_id,
                   ticker,role,stage,source,provider,model,deployment,
                   prompt_template_version,evaluator_version,evaluation_result_json,
                   cache_identity_sha256,redacted_input_sha256,
                   redacted_output_sha256,started_ts,ended_ts,latency_ms,
                   prompt_tokens,completion_tokens,total_tokens,estimated_cost_usd,
                   retry_count,validator_result,fallback_reason,terminal_status,
                   message_artifact_sha256,retention_class
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                trace_id, run_id, span_id, parent_span_id, report_run_id,
                campaign_id, str(ticker or "").upper() or None, role, stage,
                source, provider, model, deployment, prompt_template_version,
                str(evaluation.get("version") or "") or None,
                _canonical(evaluation) if evaluation else None,
                cache_identity_sha256, input_hash, output_hash, _iso(started_at), _iso(ended),
                max(0.0, (ended - started_at).total_seconds() * 1000),
                prompt_tokens, completion_tokens, total_tokens,
                _cost(prompt_tokens, completion_tokens), max(0, retry_count),
                validator_result, fallback_reason, terminal_status,
                artifact_hash, PROMPT_RETENTION,
            ),
        )
        conn.execute(
            """INSERT INTO llm_contributions (
                   contribution_id,trace_id,decision_scope,deterministic_pre_sha256,
                   proposed_change_json,proposed_change_sha256,
                   final_adjudicated_sha256,changed_fields_json,content_changed,
                   decision_changed,decision_contract_changed,
                   created_ts
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)""",
            (
                str(uuid.uuid4()), trace_id, decision_scope, redacted_hash(pre),
                _canonical(proposed)[:8000], proposed_hash, redacted_hash(final),
                _canonical(changed_fields), int(bool(changed_fields)),
                int(bool(changed_fields)), int(decision_changed),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return {"run_id": run_id, "trace_id": trace_id, "span_id": span_id}


def record_outcome_link(
    db_path: Path, *, trace_id: str, paper_trade_id: int, outcome: dict[str, Any]
) -> str:
    """Append an explicitly observational, non-causal outcome linkage."""
    db_path = Path(db_path)
    link_id = str(uuid.uuid4())
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_observability_schema(conn)
        conn.execute(
            """INSERT INTO llm_outcome_links (
                   link_id,trace_id,paper_trade_id,outcome_sha256,analysis_label
               ) VALUES (?,?,?,?, 'observational_non_causal')""",
            (link_id, trace_id, int(paper_trade_id), redacted_hash(outcome)),
        )
        conn.commit()
    finally:
        conn.close()
    return link_id


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def observability_summary(db_path: Path, *, limit: int = 50) -> dict[str, Any]:
    """Return sanitized aggregates and drill-downs; never raw prompts."""
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        ensure_observability_schema(conn)
        rows = conn.execute(
            """SELECT t.*,c.decision_scope,c.changed_fields_json,
                      COALESCE(c.content_changed,c.decision_changed) AS content_changed,
                      c.decision_contract_changed AS decision_changed
               FROM llm_call_traces t
               LEFT JOIN llm_contributions c ON c.trace_id=t.trace_id
               ORDER BY t.started_ts DESC,t.trace_id DESC"""
        ).fetchall()
        graphs = conn.execute(
            "SELECT COUNT(*) AS total,COALESCE(SUM(disagreement),0) AS disagreements FROM llm_run_graphs"
        ).fetchone()
    finally:
        conn.close()
    total = len(rows)
    statuses = {name: sum(row["terminal_status"] == name for row in rows) for name in (
        "success", "cache_hit", "fallback", "error", "unresolved",
    )}
    latencies = [float(row["latency_ms"]) for row in rows]
    traces = []
    for row in rows[:max(1, min(500, int(limit)))]:
        trace = {key: row[key] for key in (
            "trace_id", "run_id", "span_id", "parent_span_id", "report_run_id",
            "campaign_id", "ticker", "role", "stage", "source", "provider",
            "model", "deployment", "prompt_template_version",
            "evaluator_version", "evaluation_result_json", "cache_identity_sha256",
            "redacted_input_sha256", "redacted_output_sha256", "started_ts",
            "ended_ts", "latency_ms", "prompt_tokens", "completion_tokens",
            "total_tokens", "estimated_cost_usd", "retry_count",
            "validator_result", "fallback_reason", "terminal_status",
            "message_artifact_sha256", "retention_class", "decision_scope",
            "changed_fields_json", "content_changed", "decision_changed",
        )}
        try:
            trace["evaluation_result"] = json.loads(
                str(trace.pop("evaluation_result_json") or "null")
            )
        except json.JSONDecodeError:
            trace["evaluation_result"] = None
        traces.append(trace)
    def rate(count: int) -> float | None:
        return round(count / total * 100, 4) if total else None
    decision_rows = [row for row in rows if row["decision_changed"] is not None]
    decision_change_rate = (
        round(sum(bool(row["decision_changed"]) for row in decision_rows) / len(decision_rows) * 100, 4)
        if decision_rows else None
    )
    legacy_raw = llm_health_summary(db_path, recent_limit=10)
    legacy = {
        key: legacy_raw.get(key) for key in (
            "degraded", "degraded_threshold", "consecutive_failures",
            "last_success_ts", "last_failure_ts", "counts",
        )
    }
    return {
        "schema_version": "llm-observability-v2",
        "retention": {
            "prompt_storage": PROMPT_RETENTION,
            "trace_retention_days": TRACE_RETENTION_DAYS,
            "credentials_stored": False,
        },
        "aggregate": {
            "traces": total,
            "success_rate_pct": rate(statuses["success"]),
            "cache_hit_rate_pct": rate(statuses["cache_hit"]),
            "fallback_rate_pct": rate(statuses["fallback"]),
            "error_rate_pct": rate(statuses["error"]),
            "unresolved_traces": statuses["unresolved"],
            "p50_latency_ms": statistics.median(latencies) if latencies else None,
            "p95_latency_ms": _percentile(latencies, .95),
            "prompt_tokens": sum(int(row["prompt_tokens"]) for row in rows),
            "completion_tokens": sum(int(row["completion_tokens"]) for row in rows),
            "total_tokens": sum(int(row["total_tokens"]) for row in rows),
            "estimated_cost_usd": (
                round(sum(float(row["estimated_cost_usd"] or 0) for row in rows), 8)
                if any(row["estimated_cost_usd"] is not None for row in rows) else None
            ),
            "validator_failures": sum(row["validator_result"] != "passed" for row in rows),
            "decision_change_rate_pct": decision_change_rate,
            "decision_change_coverage": len(decision_rows),
            "run_graphs": int(graphs["total"] or 0) if graphs else 0,
            "disagreements": int(graphs["disagreements"] or 0) if graphs else 0,
        },
        "traces": traces,
        "legacy_health_counters": {"label": "legacy", **legacy},
    }


def observability_trace(db_path: Path, trace_id: str) -> dict[str, Any] | None:
    """Return one sanitized trace, graph, contribution, and outcome hashes."""
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        ensure_observability_schema(conn)
        row = conn.execute(
            """SELECT t.*,c.decision_scope,c.deterministic_pre_sha256,
                      c.proposed_change_json,c.proposed_change_sha256,
                      c.final_adjudicated_sha256,c.changed_fields_json,
                      COALESCE(c.content_changed,c.decision_changed) AS content_changed,
                      c.decision_contract_changed AS decision_changed
               FROM llm_call_traces t
               LEFT JOIN llm_contributions c ON c.trace_id=t.trace_id
               WHERE t.trace_id=?""",
            (trace_id,),
        ).fetchone()
        if row is None:
            return None
        graph = conn.execute(
            "SELECT * FROM llm_run_graphs WHERE run_id=?", (row["run_id"],)
        ).fetchone()
        outcomes = conn.execute(
            """SELECT paper_trade_id,outcome_sha256,analysis_label,created_ts
               FROM llm_outcome_links WHERE trace_id=? ORDER BY created_ts""",
            (trace_id,),
        ).fetchall()
    finally:
        conn.close()
    trace = dict(row)
    for key in ("changed_fields_json", "proposed_change_json", "evaluation_result_json"):
        try:
            trace[key.removesuffix("_json")] = json.loads(str(trace.pop(key) or "null"))
        except json.JSONDecodeError:
            trace[key.removesuffix("_json")] = None
    return {
        "trace": trace,
        "run_graph": dict(graph) if graph is not None else None,
        "outcomes": [dict(item) for item in outcomes],
        "causal_interpretation": "observational_non_causal",
    }
