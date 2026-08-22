"""Small real-API fixture used by the Paper Trades browser contract test."""
from __future__ import annotations

import os
import sqlite3
import hashlib
from pathlib import Path

from fastapi import FastAPI

DB_PATH = Path(os.environ["TRADER_KOO_DB_PATH"])
if DB_PATH.exists():
    DB_PATH.unlink()

from trader_koo.backend.routers.paper_trades import router
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

app = FastAPI()
app.include_router(router)
