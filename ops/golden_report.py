"""Golden-report harness: snapshot report, email and paper decisions, then compare two snapshots.

A snapshot runs the real report generator on a copy of a pinned database with a
frozen clock, LLM disabled and every socket connect blocked. Two snapshots of the
same code on the same inputs are byte-identical, so any diff between a base ref
and a branch is a behaviour change.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sqlite3
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

JSON_FILES = ("report.json", "paper.json", "policy.json", "network.json")
TEXT_FILES = ("email.txt", "email.html")
# Fields that differ between runs of identical inputs.
VOLATILE_KEYS = frozenset({
    "generated_ts", "db_path", "cron_log_path", "run_id", "report_run_id",
    "duration_ms", "duration_s", "duration_seconds", "elapsed_ms", "elapsed_s",
})


def normalize(payload: Any) -> Any:
    """Drop volatile fields and sort scalar lists, which have no defined order."""
    if isinstance(payload, dict):
        return {k: normalize(v) for k, v in payload.items() if k not in VOLATILE_KEYS}
    if isinstance(payload, list):
        items = [normalize(v) for v in payload]
        if all(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in items):
            return sorted(items, key=lambda v: (type(v).__name__, v))
        return items
    return payload


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=1, default=str)


def _registered_hash(db_path: Path) -> str | None:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT policy_hash FROM paper_campaigns WHERE status='active' ORDER BY campaign_id LIMIT 1"
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()
    return (row[0] or None) if row else None


def _setup_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    signals = report.get("signals") or {}
    rows = signals.get("setup_quality_all") or signals.get("setup_quality_top") or []
    return sorted((r for r in rows if isinstance(r, dict)), key=lambda r: str(r.get("ticker")))


def snapshot(db_path: Path, out_dir: Path, now: datetime) -> Path:
    import time_machine

    from trader_koo.paper_trade.campaign import canonical_hash
    from trader_koo.paper_trade.config import config_snapshot
    from trader_koo.paper_trades import _build_config, evaluate_setup_for_paper_trade
    from trader_koo.report.generator import fetch_report_payload
    from trader_koo.report_email import build_report_email_bodies

    out_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[str] = []

    def blocked_connect(self: socket.socket, address: Any) -> None:
        attempts.append(str(address))
        raise OSError(f"golden_report: network blocked ({address})")

    saved_env = os.environ.get("TRADER_KOO_LLM_ENABLED")
    os.environ["TRADER_KOO_LLM_ENABLED"] = "0"
    real_connect = socket.socket.connect
    socket.socket.connect = blocked_connect  # type: ignore[method-assign]
    try:
        with tempfile.TemporaryDirectory() as tmp, time_machine.travel(now, tick=False):
            work_db = Path(tmp) / "snapshot.db"
            if db_path.exists():
                shutil.copy(db_path, work_db)
            try:
                raw = fetch_report_payload(work_db, Path(tmp) / "run.log", tail_lines=0)
                if not raw.get("ok") and not raw.get("error"):
                    raw["error"] = "report not ok: " + (", ".join(map(str, raw.get("warnings") or [])) or "unknown")
            except Exception as exc:  # recorded as output so a change in failure mode shows as a diff
                raw = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            report = normalize(raw)

            email_txt, email_html = build_report_email_bodies(raw, "")
            paper = [
                {"ticker": row.get("ticker"), "decision": normalize(evaluate_setup_for_paper_trade(row))}
                for row in _setup_rows(raw)
            ]
            policy = {
                "default_hash": canonical_hash(config_snapshot(_build_config())),
                "registered_hash": _registered_hash(work_db) if work_db.exists() else None,
            }
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]
        if saved_env is None:
            os.environ.pop("TRADER_KOO_LLM_ENABLED", None)
        else:
            os.environ["TRADER_KOO_LLM_ENABLED"] = saved_env

    (out_dir / "report.json").write_text(_dump(report))
    (out_dir / "email.txt").write_text(email_txt)
    (out_dir / "email.html").write_text(email_html)
    (out_dir / "paper.json").write_text(_dump(paper))
    (out_dir / "policy.json").write_text(_dump(policy))
    (out_dir / "network.json").write_text(_dump(attempts))
    return out_dir


def _diff(a: Any, b: Any, path: str) -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in a:
                out.append(f"{sub} added")
            elif key not in b:
                out.append(f"{sub} removed")
            else:
                out += _diff(a[key], b[key], sub)
        return out
    if isinstance(a, list) and isinstance(b, list):
        out = []
        for i in range(max(len(a), len(b))):
            sub = f"{path}[{i}]"
            if i >= len(a):
                out.append(f"{sub} added")
            elif i >= len(b):
                out.append(f"{sub} removed")
            else:
                out += _diff(a[i], b[i], sub)
        return out
    return [] if a == b else [f"{path} {json.dumps(a)} -> {json.dumps(b)}"]


def compare(base_dir: Path, head_dir: Path) -> list[str]:
    """One line per differing file and JSON path; empty when the snapshots match."""
    lines: list[str] = []
    for name in sorted(JSON_FILES + TEXT_FILES):
        base, head = base_dir / name, head_dir / name
        if not base.exists() and not head.exists():
            continue
        if not base.exists():
            lines.append(f"{name}: only in head")
        elif not head.exists():
            lines.append(f"{name}: only in base")
        elif name in TEXT_FILES:
            if base.read_text() != head.read_text():
                lines.append(f"{name}: text differs")
        else:
            lines += [f"{name}: {d}" for d in _diff(json.loads(base.read_text()), json.loads(head.read_text()), "")]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ops.golden_report")
    sub = parser.add_subparsers(dest="cmd", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--db", type=Path, required=True)
    snap.add_argument("--out", type=Path, required=True)
    snap.add_argument("--now", required=True, help="ISO8601 instant, e.g. 2026-10-23T22:00:00+00:00")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("base", type=Path)
    cmp_.add_argument("head", type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "snapshot":
        now = datetime.fromisoformat(args.now.replace("Z", "+00:00"))
        if now.tzinfo is None:
            parser.error("--now must carry a UTC offset")
        print(snapshot(args.db, args.out, now))
        return 0
    diffs = compare(args.base, args.head)
    print("\n".join(diffs) if diffs else "golden: identical")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
