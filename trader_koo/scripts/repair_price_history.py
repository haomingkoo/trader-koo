"""Dry-run or apply one explicit, provider-evidenced price restatement."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from trader_koo.db.price_repairs import apply_price_repair, plan_price_repair


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    return parser


def run(*, db_path: Path, input_path: Path, apply: bool) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("proposals"), list):
        raise ValueError("input must contain a proposals array")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE" if apply else "BEGIN")
        plan = plan_price_repair(
            conn,
            payload["proposals"],
            adjustment_version=str(payload.get("adjustment_version") or ""),
            reason=str(payload.get("reason") or ""),
            provider_evidence=payload.get("provider_evidence") or {},
        )
        if not apply:
            conn.rollback()
            return {"mode": "dry_run", "plan": plan}
        result = apply_price_repair(conn, plan)
        conn.commit()
        return {"mode": "applied", "plan": plan, "result": result}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(
        run(db_path=args.db_path, input_path=args.input, apply=args.apply),
        sort_keys=True,
        separators=(",", ":"),
    ))


if __name__ == "__main__":
    main()
