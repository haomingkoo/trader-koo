"""One-command, reproducible build of the research database.

    make research-db RESEARCH_DB=~/dev/.research/trader_koo_research.db SOURCE_DB=/path/to/export.db

Stages run one after another in a single process tree, so exactly one writer
touches the file at a time (the SEC stage once died silently with "database is
locked" while another writer was running). The build stops at the first stage
that exits non-zero or whose output shows a database failure, then runs the
end-of-build data checks, which also fail the build.

Per-symbol "No earnings dates found" lines from yfinance are logged as ERROR by
that library and are normal for ETFs; they are not treated as failures. The
patterns below are.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ops.research_db_checks import CONTEXT_SYMBOLS, render, run_checks

PYTHON = sys.executable
PREDECESSORS = ("XOM=34088",)
FAILURE_PATTERN = re.compile(
    r"database is locked|database table is locked|Traceback \(most recent call last\)|"
    r"sqlite3\.\w*Error|disk I/O error|database disk image is malformed",
    re.IGNORECASE,
)
EARNINGS_SOURCES = ("both", "finnhub", "yfinance")


@dataclass
class Stage:
    name: str
    argv: list[str]
    after: str | None = None  # name of a post-run verifier, see VERIFIERS


def scan_line(line: str) -> str | None:
    """Return the log line (trimmed) if it shows a database or crash failure."""
    return line.strip() if FAILURE_PATTERN.search(line) else None


def _module(mod: str, *args: str) -> list[str]:
    return [PYTHON, "-m", f"trader_koo.scripts.{mod}", *args]


def build_stages(db: Path, source: Path, tickers: list[str] | None, earnings_source: str,
                 report_dir: Path) -> list[Stage]:
    subset = ["--tickers", ",".join(tickers)] if tickers else []
    # --tickers restricts the run to that list, so the context symbols must be named in it too.
    price_subset = ["--tickers", ",".join([*tickers, *CONTEXT_SYMBOLS])] if tickers else []
    price = _module(
        "backfill_price_history", "--db-path", str(db), "--source-db", str(source),
        "--context-symbols", ",".join(CONTEXT_SYMBOLS),
        "--report-json", str(report_dir / "backfill_report.json"), *price_subset,
    )
    membership = _module("load_sp500_membership", "--db-path", str(db), "--prices-db", str(db),
                         "--report-dir", str(report_dir / "membership"))
    sec = _module("load_sec_fundamentals", "--db-path", str(db), *subset,
                  *[a for p in PREDECESSORS for a in ("--predecessor", p)])
    earnings = _module("backfill_earnings_history", "--db-path", str(db), "--source", earnings_source,
                       "--max-tickers", "5000", "--tickers", ",".join(tickers) if tickers else _ALL_MEMBERS)
    return [Stage("price backfill", price, "backfill_report"), Stage("membership load", membership),
            Stage("SEC fundamentals load", sec), Stage("earnings backfill", earnings)]


_ALL_MEMBERS = "@members"  # replaced with the membership tickers once the membership stage has run


def member_tickers(db: Path) -> list[str]:
    import sqlite3
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return [r[0] for r in conn.execute(
            "SELECT DISTINCT m.ticker FROM sp500_membership m JOIN "
            "(SELECT DISTINCT ticker FROM price_daily) p ON p.ticker = m.ticker ORDER BY 1")]
    finally:
        conn.close()


def verify_backfill_report(report_dir: Path) -> str | None:
    path = report_dir / "backfill_report.json"
    if not path.exists():
        return f"price backfill wrote no report at {path}"
    counts = json.loads(path.read_text()).get("status_counts", {})
    errors = counts.get("error", 0)
    return f"price backfill reported {errors} ticker(s) with status 'error'; see {path}" if errors else None


def run_stage(stage: Stage, log, env: dict[str, str]) -> str | None:
    """Run one stage, tee its output to ``log`` and stdout. Returns a failure reason or None."""
    print(f"\n== {stage.name}", flush=True)
    log.write(f"\n== {stage.name}\n$ {' '.join(stage.argv)}\n")
    proc = subprocess.Popen(stage.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env, bufsize=1)
    hit = None
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        log.write(line)
        hit = hit or scan_line(line)
    code = proc.wait()
    log.flush()
    if code != 0:
        return f"{stage.name} exited with status {code}"
    if hit:
        return f"{stage.name} logged a database/crash failure: {hit!r}"
    return None


def run_stages(stages: Iterable[Stage], log, report_dir: Path, db: Path, env: dict[str, str]) -> str | None:
    for stage in stages:
        if _ALL_MEMBERS in stage.argv:
            members = member_tickers(db)
            if not members:
                return f"{stage.name}: no membership tickers with prices to run on"
            stage.argv = [",".join(members) if a == _ALL_MEMBERS else a for a in stage.argv]
        failure = run_stage(stage, log, env)
        if failure is None and stage.after == "backfill_report":
            failure = verify_backfill_report(report_dir)
        if failure:
            return failure
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db-path", required=True, type=Path)
    p.add_argument("--source-db", required=True, type=Path, help="prod export or golden copy, opened read-only")
    p.add_argument("--tickers", help="comma separated subset (test builds); omit for the full universe")
    p.add_argument("--earnings-source", choices=EARNINGS_SOURCES, default="both")
    args = p.parse_args(argv)
    db, source = args.db_path.expanduser().resolve(), args.source_db.expanduser().resolve()
    if not source.exists():
        p.error(f"--source-db does not exist: {source}")
    if source == db:
        p.error("--source-db and --db-path must differ")
    if args.earnings_source in ("both", "finnhub") and not os.environ.get("FINNHUB_API_KEY", "").strip():
        p.error("FINNHUB_API_KEY is not set: set it, or pass --earnings-source yfinance explicitly "
                "(the earnings script would otherwise skip Finnhub silently)")
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else None
    db.parent.mkdir(parents=True, exist_ok=True)
    report_dir = db.parent / f"{db.stem}_build"
    report_dir.mkdir(exist_ok=True)

    with open(f"{db}.build.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"another build holds {lock.name}; refusing to write concurrently", file=sys.stderr)
            return 2
        with open(report_dir / "build.log", "a") as log:
            stages = build_stages(db, source, tickers, args.earnings_source, report_dir)
            failure = run_stages(stages, log, report_dir, db, os.environ.copy())
            if failure:
                print(f"\nBUILD FAILED: {failure}", file=sys.stderr)
                return 1
            checks = run_checks(db, tickers)
            text = render(checks)
            print(text)
            log.write(text + "\n")
            if checks.failed:
                print("\nBUILD FAILED: data checks violated", file=sys.stderr)
                return 1
    print(f"\nBUILD OK: {db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
