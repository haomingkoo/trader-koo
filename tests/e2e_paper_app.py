"""Small real-API fixture used by the Paper Trades browser contract test."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from fastapi import FastAPI

DB_PATH = Path(os.environ["TRADER_KOO_DB_PATH"])
if DB_PATH.exists():
    DB_PATH.unlink()

from trader_koo.backend.routers.paper_trades import router
from trader_koo.paper_trades import create_paper_trades_from_report, ensure_paper_trade_schema

conn = sqlite3.connect(DB_PATH)
ensure_paper_trade_schema(conn)
conn.execute(
    """CREATE TABLE report_runs (
           run_id TEXT PRIMARY KEY, status TEXT NOT NULL,
           is_generation_canonical INTEGER NOT NULL
       )"""
)
conn.execute(
    "INSERT INTO report_runs VALUES ('browser-real-api-report','published',1)"
)
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
    report_run_id="browser-real-api-report",
)
conn.commit()
conn.close()

app = FastAPI()
app.include_router(router)
