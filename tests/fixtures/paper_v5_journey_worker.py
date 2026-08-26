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


def _price_contract(
    conn: sqlite3.Connection,
    report_date: str,
    next_date: str,
) -> dict[str, object]:
    expected_dates = conn.execute(
        "SELECT date FROM price_daily WHERE ticker='JRN' ORDER BY date"
    ).fetchall()
    if expected_dates != [(report_date,), (next_date,)]:
        raise RuntimeError("copied fixture JRN price preparation is missing")
    spy_open = conn.execute(
        "SELECT open FROM price_daily WHERE ticker='SPY' AND date=?",
        (next_date,),
    ).fetchone()
    if spy_open is None or spy_open[0] is None or float(spy_open[0]) <= 0:
        raise RuntimeError("copied fixture SPY next-session open is missing")
    contract = research_price_contract(conn, ["JRN", "SPY"])
    if not contract.get("eligible"):
        raise RuntimeError("copied fixture price contract is not eligible")
    return contract


def _prepare_prices(
    conn: sqlite3.Connection,
    report_date: str,
    next_date: str,
) -> dict[str, object]:
    if conn.execute(
        "SELECT 1 FROM price_daily WHERE ticker='JRN' LIMIT 1"
    ).fetchone() is not None:
        raise RuntimeError("copied fixture ticker JRN already exists")
    spy_contract = research_price_contract(conn, ["SPY"])
    if not spy_contract.get("eligible"):
        raise RuntimeError("copied fixture requires an eligible existing SPY series")
    if conn.execute(
        "SELECT 1 FROM price_daily WHERE ticker='SPY' AND date=?",
        (next_date,),
    ).fetchone() is not None:
        raise RuntimeError("copied fixture SPY append date already exists")
    basis = str(spy_contract["basis"])
    version = str(spy_contract["version"])
    rows = [
        ("JRN", report_date, 99.0, 101.0, 98.0, 100.0, 2_000_000),
        ("JRN", next_date, 101.0, 104.0, 100.0, 103.0, 2_100_000),
        ("SPY", next_date, 502.0, 505.0, 501.0, 504.0, 10_100_000),
    ]
    conn.executemany(
        """INSERT INTO price_daily
           (ticker,date,open,high,low,close,volume,data_source,fetch_timestamp,
            adjustment_basis,adjustment_version,basis_status,unresolved_reason)
           VALUES (?,?,?,?,?,?,?,'copied-db-fixture',?,?,?,'verified',NULL)""",
        [(*row, _stamp(-10), basis, version) for row in rows],
    )
    ensure_price_series_revision_schema(conn)
    record_price_series_revision(
        conn,
        "JRN",
        evidence={
            "provider": "copied-db-fixture",
            "vendor_action_ledger_checked": True,
            "vendor_action_ledger": [],
        },
        fetch_timestamp=_stamp(-10),
    )
    spy_evidence_row = conn.execute(
        "SELECT evidence_json FROM price_series_revisions WHERE ticker='SPY'"
    ).fetchone()
    try:
        spy_evidence = json.loads(str(spy_evidence_row[0]))
    except (TypeError, ValueError, IndexError) as exc:
        raise RuntimeError("copied fixture SPY revision evidence is malformed") from exc
    if not isinstance(spy_evidence, dict):
        raise RuntimeError("copied fixture SPY revision evidence is malformed")
    spy_evidence["non_production_rehearsal_append"] = {
        "date": next_date,
        "provider": "copied-db-fixture",
    }
    record_price_series_revision(
        conn,
        "SPY",
        evidence=spy_evidence,
        fetch_timestamp=_stamp(-10),
    )
    conn.commit()
    contract = research_price_contract(conn, ["JRN"])
    combined = research_price_contract(conn, ["JRN", "SPY"])
    if not contract.get("eligible") or not combined.get("eligible"):
        raise RuntimeError("copied fixture price preparation is not eligible")
    return {
        "candidate_revision": contract["revision"],
        "execution_revision": combined["revision"],
        "spy_appended_date": next_date,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--lease-ready", type=Path)
    parser.add_argument("--lease-release", type=Path)
    parser.add_argument("--lease-only", action="store_true")
    parser.add_argument("--prepare-prices", action="store_true")
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
    if args.prepare_prices:
        print(json.dumps(_prepare_prices(conn, report_date, next_date)))
        conn.close()
        lease.close()
        return
    price_contract = _price_contract(conn, report_date, next_date)
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
            "price_basis": price_contract,
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
