"""Run the frozen challenger tournament against an explicit SQLite snapshot."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from trader_koo.research.challenger_tournament import run_challenger_tournament
from trader_koo.research.next_open_baseline import write_artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--consume-heldout",
        action="store_true",
        help="Read the sealed 20%% once, only after validation selects a challenger",
    )
    args = parser.parse_args()

    with sqlite3.connect(args.db) as conn:
        artifact = run_challenger_tournament(
            conn, consume_heldout=args.consume_heldout
        )
    file_hash = write_artifact(args.output, artifact)
    print(json.dumps({
        "ok": True,
        "artifact": str(args.output),
        "artifact_file_sha256": file_hash,
        "status": artifact["status"],
        "selected_challenger": artifact["selected_challenger"],
        "sealed_heldout_accessed": artifact["sealed_heldout"]["accessed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
