"""Small real-API fixture used by the Paper Trades browser contract test."""
from __future__ import annotations

import os
import sqlite3
import hashlib
import datetime as dt
from pathlib import Path

from fastapi import Depends, FastAPI

DB_PATH = Path(os.environ["TRADER_KOO_DB_PATH"])
if DB_PATH.exists():
    DB_PATH.unlink()

from trader_koo.backend.routers.paper_trades import router
from trader_koo.llm.observability import observability_summary, record_llm_call
from trader_koo.middleware.auth import AdminAuthConfig, AdminAuthenticator, require_admin
from trader_koo.paper_trades import create_paper_trades_from_report, ensure_paper_trade_schema
from trader_koo.report.runs import complete_report_run, publish_report_run, sha256_file
from trader_koo.report.serializer import write_reports

conn = sqlite3.connect(DB_PATH)
ensure_paper_trade_schema(conn)
run_id = "browser-real-api-report"
config_json = "{}"
conn.execute(
    """INSERT INTO report_runs
       (run_id,report_kind,status,started_ts,config_json,config_hash,code_version)
       VALUES (?,'daily','started','2026-08-21T21:00:00Z',?,?,?)""",
    (run_id, config_json, hashlib.sha256(config_json.encode()).hexdigest(), "a" * 40),
)
conn.commit()
decision = {
    "ticker": "REJECT", "selected_rank": 1, "decision": "accepted",
    "reason_codes": ["selected_report_cohort"], "inputs": {},
}
report = {
    "generated_ts": "2026-08-21T22:00:00Z",
    "meta": {"report_kind": "daily"},
    "latest_data": {},
    "signals": {"report_decisions": [decision], "scanned_universe": ["REJECT"]},
    "counts": {}, "risk_filters": {}, "warnings": [], "ok": True,
}
paths = write_reports(report, DB_PATH.parent, run_id=run_id, publish_latest=False)
artifact = Path(paths["json_path"])
complete_report_run(
    conn, run_id=run_id, report=report, artifact_path=artifact,
    markdown_path=Path(paths["md_path"]), content_hash=sha256_file(artifact),
    completed_ts="2026-08-21T22:01:00Z",
)
publish_report_run(conn, run_id=run_id, report_dir=DB_PATH.parent)
create_paper_trades_from_report(
    conn,
    setup_rows=[{
        "ticker": "REJECT",
        "setup_tier": "F",
        "score": 10.0,
        "actionability": "watch",
        "signal_bias": "bullish",
        "close": 100.0,
    }],
    report_date="2026-08-21",
    generated_ts="2026-08-21T22:00:00Z",
    report_run_id=run_id,
)
conn.commit()
conn.close()

trace_start = dt.datetime(2026, 8, 21, 22, 5, tzinfo=dt.timezone.utc)
record_llm_call(
    DB_PATH, source="chart_commentary", role="narrative_rewriter",
    stage="setup_copy_rewrite", provider="azure_openai", model="gpt-fixture",
    deployment="fixture-deployment", prompt_template_version="setup-rewrite-v2",
    input_payload={"ticker": "REJECT"},
    proposed_output={"observation": "Candidate rejected by deterministic tier rule."},
    deterministic_pre={"observation": "Tier F candidate."},
    final_adjudicated={"observation": "Candidate rejected by deterministic tier rule."},
    started_at=trace_start, ended_at=trace_start + dt.timedelta(milliseconds=42),
    usage={"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    validator_result="passed", fallback_reason=None, terminal_status="success",
    ticker="REJECT",
    evaluation_result={
        "version": "setup-grounding-v2", "passed": True, "errors": [],
        "semantic_outcome": "rephrased", "prose_quality_scored": False,
        "decision_scope": "narrative_only",
    },
    cache_identity_sha256="a" * 64,
)

app = FastAPI()
app.state.admin_authenticator = AdminAuthenticator(
    AdminAuthConfig(api_key="e2e-agent-key")
)
app.include_router(router)


@app.get("/api/admin/agent-observability", dependencies=[Depends(require_admin)])
def agent_observability_fixture():
    return {"ok": True, **observability_summary(DB_PATH, limit=100)}
