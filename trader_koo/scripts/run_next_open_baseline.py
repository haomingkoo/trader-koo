"""Run the locked next-open research baseline against a selected SQLite DB."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from trader_koo.research.next_open_baseline import (
    BaselineConfig,
    run_next_open_baseline,
    write_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--report-dir", required=True, type=Path,
        help="Explicit directory containing the immutable report artifacts referenced by the DB",
    )
    parser.add_argument("--diagnostic-holding-sessions", type=int)
    parser.add_argument("--no-consume-heldout", action="store_true")
    args = parser.parse_args()
    holding = args.diagnostic_holding_sessions or 10
    if args.diagnostic_holding_sessions is not None and holding == 10:
        parser.error("omit --diagnostic-holding-sessions for the locked primary endpoint")
    conn = sqlite3.connect(args.db)
    try:
        artifact = run_next_open_baseline(
            conn,
            config=BaselineConfig(holding_sessions=holding),
            consume_heldout=(
                not args.no_consume_heldout
                and args.diagnostic_holding_sessions is None
            ),
            report_dir=args.report_dir,
        )
    finally:
        conn.close()
    file_hash = write_artifact(args.output, artifact)
    print(json.dumps({
        "ok": True,
        "artifact": str(args.output),
        "artifact_file_sha256": file_hash,
        "evidence_state": artifact["evidence_state"],
        "causal_valid": artifact["causal_valid"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
