"""Publish and admit one canonical report through the production application seam."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import time
from pathlib import Path

from trader_koo.db.price_contract import (
    ensure_price_series_revision_schema,
    record_price_series_revision,
    research_price_contract,
)
from trader_koo.backend.services.maintenance import acquire_lease
from trader_koo.paper_trade.chronology import next_scheduled_session_after
from trader_koo.report.runs import (
    admit_published_report,
    complete_report_run,
    publish_report_run,
    sha256_file,
    start_report_run,
)
from trader_koo.report.serializer import write_reports


def _stamp(offset_seconds: int) -> str:
    return (
        dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=offset_seconds)
    ).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _seed_prices(conn: sqlite3.Connection, report_date: str, next_date: str) -> str:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS price_daily (
               ticker TEXT NOT NULL,date TEXT NOT NULL,open REAL,high REAL,low REAL,
               close REAL,volume REAL,data_source TEXT,fetch_timestamp TEXT,
               adjustment_basis TEXT,adjustment_version TEXT,basis_status TEXT,
               unresolved_reason TEXT,UNIQUE(ticker,date)
           )"""
    )
    rows = [
        ("JRN", report_date, 99.0, 101.0, 98.0, 100.0, 2_000_000),
        ("JRN", next_date, 101.0, 104.0, 100.0, 103.0, 2_100_000),
        ("SPY", report_date, 500.0, 502.0, 498.0, 501.0, 10_000_000),
        ("SPY", next_date, 502.0, 505.0, 501.0, 504.0, 10_100_000),
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO price_daily
           (ticker,date,open,high,low,close,volume,data_source,fetch_timestamp,
            adjustment_basis,adjustment_version,basis_status,unresolved_reason)
           VALUES (?,?,?,?,?,?,?,'copied-db-fixture',?,
                   'split_adjusted_price_only','journey-v1','verified',NULL)""",
        [(*row, _stamp(-10)) for row in rows],
    )
    ensure_price_series_revision_schema(conn)
    for ticker in ("JRN", "SPY"):
        record_price_series_revision(
            conn, ticker,
            evidence={"provider": "copied-db-fixture", "vendor_action_ledger_checked": True,
                      "vendor_action_ledger": []},
            fetch_timestamp=_stamp(-10),
        )
    conn.commit()
    contract = research_price_contract(conn, ["JRN", "SPY"])
    if not contract.get("eligible"):
        raise RuntimeError("copied fixture price contract is not eligible")
    return str(contract["revision"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--lease-ready", type=Path)
    parser.add_argument("--lease-release", type=Path)
    parser.add_argument("--lease-only", action="store_true")
    args = parser.parse_args()

    lease = acquire_lease(args.db_path, exclusive=False)
    if args.lease_ready or args.lease_release:
        if not args.lease_ready or not args.lease_release:
            parser.error("--lease-ready and --lease-release are required together")
        args.lease_ready.write_text("ready\n")
        while not args.lease_release.exists():
            time.sleep(0.01)
    if args.lease_only:
        lease.close()
        return
    now = dt.datetime.now(dt.timezone.utc)
    report_date = now.date().isoformat()
    next_date = next_scheduled_session_after(report_date)
    conn = sqlite3.connect(args.db_path)
    revision = _seed_prices(conn, report_date, next_date)
    started_ts, generated_ts, completed_ts = _stamp(-3), _stamp(-2), _stamp(-1)
    run_id = start_report_run(
        conn, report_kind="daily", configuration={"journey": "copied-database"},
        code_version="a" * 40, started_ts=started_ts,
    )
    candidate = {
        "ticker": "JRN", "setup_tier": "A", "score": 82.0,
        "actionability": "higher-probability", "signal_bias": "bullish",
        "close": 100.0, "setup_family": "Bullish Breakout", "atr_pct_14": 2.0,
        "support_level": 95.0, "resistance_level": 99.0,
        "debate_agreement_score": 85.0, "risk_note": "Copied-database fixture only.",
    }
    decisions = ([{
        "ticker": "JRN", "selected_rank": 1, "decision": "accepted",
        "reason_codes": ["selected_report_cohort"], "inputs": candidate,
    }] if args.candidate else [])
    report = {
        "ok": True, "generated_ts": generated_ts, "warnings": [],
        "meta": {
            "report_kind": "daily",
            "price_basis": {"revision": revision},
        },
        "latest_data": {"price_date": report_date},
        "signals": {
            "report_decisions": decisions,
            "scanned_universe": [item["ticker"] for item in decisions],
        },
        "counts": {}, "risk_filters": {}, "market_session": {},
    }
    paths = write_reports(report, args.report_dir, run_id=run_id, publish_latest=False)
    artifact, markdown = Path(paths["json_path"]), Path(paths["md_path"])
    complete_report_run(
        conn, run_id=run_id, report=report, artifact_path=artifact,
        markdown_path=markdown, content_hash=sha256_file(artifact),
        completed_ts=completed_ts,
    )
    publication = publish_report_run(conn, run_id=run_id, report_dir=args.report_dir)
    admitted = admit_published_report(conn, run_id=run_id, report_dir=args.report_dir)
    print(json.dumps({
        "run_id": run_id, "publication": publication, "admitted": admitted,
    }))
    conn.close()
    lease.close()


if __name__ == "__main__":
    main()
