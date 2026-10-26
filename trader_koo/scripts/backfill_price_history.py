"""CLI for the 2010+ price backfill into a separate research database.

Reads S&P 500 membership from the pinned fja05680/sp500 file (the same dataset
the point-in-time ``sp500_membership`` work uses), seeds the research file from
``--source-db`` (read-only), and backfills earlier dates through the daily
ingest's fetch, normalisation and write functions. Core logic and safety rules
live in ``price_backfill``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from trader_koo.price_backfill import config as cfg
from trader_koo.price_backfill.core import (
    DEFAULT_FLOOR,
    DEFAULT_WARMUP_DAYS,
    Spell,
    add_context_symbols,
    load_membership,
    run_backfill,
    _day,
)
from trader_koo.price_backfill.store import seed_research_db
from trader_koo.scripts.update_market_db import ensure_schema, fetch_price_daily

LOG = logging.getLogger("trader_koo.backfill")

# Shared with the point-in-time membership work: same dataset, same pin.
MEMBERSHIP_SOURCE_SHA = "a2430f2af0c79ddf0748e91de11bdeb1616ab5a7"
MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/"
    f"{MEMBERSHIP_SOURCE_SHA}/sp500_ticker_start_end.csv"
)
MEMBERSHIP_SHA256 = "b2f4fe2f2e4dce8eaf0eb4fbd3946a32a20c1ddfe69687dfca3b6b59a0b71f9f"

# --- live yfinance helpers --------------------------------------------------------

def default_fetch(ticker: str, start: str, end: Optional[str]) -> tuple[pd.DataFrame, str]:
    return fetch_price_daily(ticker, start, end, auto_adjust=False)


def default_actions(ticker: str) -> list:
    from trader_koo.db.sources import get_data_source_manager

    return get_data_source_manager().fetch_ticker_actions(ticker)


def default_probe_alive(ticker: str) -> bool:
    import yfinance as yf

    return not yf.Ticker(ticker).history(period="5d").empty


def probe_coverage(
    membership: dict[str, list[Spell]],
    *,
    floor: str,
    pace_sec: float,
    only: Optional[set[str]] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """Classify each symbol with a ~2 week history call at the start of its first spell."""
    import yfinance as yf

    rows: list[dict] = []
    for ticker, spells in membership.items():
        if only and ticker not in only:
            continue
        if limit is not None and len(rows) >= limit:
            break
        first = max(_day(floor), _day(spells[0].start))
        departed = all(s.end is not None for s in spells)
        try:
            hist = yf.Ticker(ticker).history(
                start=first.isoformat(), end=(first + dt.timedelta(days=cfg.PROBE_DAYS)).isoformat(),
                auto_adjust=False,
            )
            if not hist.empty:
                status = "served"
            elif default_probe_alive(ticker):
                status = "recycled_or_gap_symbol" if departed else "current_member_no_bars_at_start"
            else:
                status = "not_served"
        except Exception as exc:
            status = f"error:{type(exc).__name__}"
        rows.append({"ticker": ticker, "departed": departed, "status": status})
        time.sleep(pace_sec)
    return rows


def ensure_membership_file(path: Optional[Path]) -> Path:
    if path is None:
        path = Path("/tmp") / f"sp500_ticker_start_end_{MEMBERSHIP_SOURCE_SHA[:cfg.SHA_PREFIX_CHARS]}.csv"
    if not path.exists():
        response = requests.get(MEMBERSHIP_URL, timeout=cfg.HTTP_TIMEOUT_SEC, headers={"User-Agent": "trader-koo-backfill"})
        response.raise_for_status()
        path.write_bytes(response.content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MEMBERSHIP_SHA256:
        raise SystemExit(f"{path}: sha256 {digest} does not match the pinned membership file")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--db-path", required=True, type=Path,
        help="Research SQLite file to create or extend (e.g. ~/dev/.research/trader_koo_research.db). No default.",
    )
    p.add_argument(
        "--source-db", type=Path, default=None,
        help="Database whose price rows seed the research file (opened read-only). Existing research rows win.",
    )
    p.add_argument("--membership-csv", type=Path, default=None, help="Pinned membership CSV; downloaded if absent.")
    p.add_argument("--start", default=DEFAULT_FLOOR)
    p.add_argument("--warmup-days", type=int, default=DEFAULT_WARMUP_DAYS)
    p.add_argument("--pace-sec", type=float, default=cfg.PACE_SEC)
    p.add_argument("--retries", type=int, default=cfg.RETRIES)
    p.add_argument("--backoff-sec", type=float, default=cfg.BACKOFF_SEC)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tickers", default="", help="Comma separated subset (price_daily format, e.g. BRK-B).")
    p.add_argument(
        "--context-symbols", default="",
        help="Comma separated non-member symbols to backfill from --start (e.g. SPY,QQQ,^VIX,^IRX). "
             "Same fetch path and never modifies existing rows; combine with --tickers to run only these.",
    )
    p.add_argument("--state", type=Path, default=None, help="Resume file. Default: <db>.backfill_state.json")
    p.add_argument("--retry-terminal", action="store_true", help="Retry tickers already settled in the state file.")
    p.add_argument("--coverage-only", action="store_true", help="Probe yfinance for every symbol; write nothing.")
    p.add_argument("--report-json", type=Path, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dropped_spells: list[tuple[str, str, str]] = []
    membership = load_membership(
        ensure_membership_file(args.membership_csv), floor=args.start, dropped_before_floor=dropped_spells)
    context = [t.strip() for t in args.context_symbols.split(",") if t.strip()]
    context_already_members: list[str] = []
    membership = add_context_symbols(
        membership, context, floor=args.start, already_members=context_already_members)
    only = {t.strip().upper() for t in args.tickers.split(",") if t.strip()} or None
    report: dict = {
        "membership_source": MEMBERSHIP_URL,
        "symbols": len(membership),
        "spells_ended_before_floor_dropped": len(dropped_spells),
        "spells_ended_before_floor_dropped_list": dropped_spells,
        "context_symbols_already_members": context_already_members,
    }

    if args.coverage_only:
        rows = probe_coverage(membership, floor=args.start, pace_sec=args.pace_sec, only=only, limit=args.limit)
        report["coverage"] = rows
        report["coverage_counts"] = dict(
            Counter(f"{'departed' if r['departed'] else 'current'}:{r['status']}" for r in rows)
        )
    else:
        db_path = args.db_path.expanduser()
        if args.source_db and args.source_db.expanduser().resolve() == db_path.resolve():
            raise SystemExit("--db-path must differ from --source-db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        size_before = db_path.stat().st_size if db_path.exists() else 0
        conn = sqlite3.connect(db_path)
        conn.execute(f"PRAGMA busy_timeout={cfg.BUSY_TIMEOUT_MS}")
        ensure_schema(conn)
        if args.source_db:
            report["seeded_rows"] = seed_research_db(conn, args.source_db.expanduser())
        started = time.monotonic()
        results = run_backfill(
            conn, membership, fetch_fn=default_fetch, actions_fn=default_actions,
            probe_alive_fn=default_probe_alive, floor=args.start, warmup_days=args.warmup_days,
            retries=args.retries, backoff_sec=args.backoff_sec, pace_sec=args.pace_sec,
            state_path=args.state or Path(str(db_path) + ".backfill_state.json"),
            limit=args.limit, only=only, retry_terminal=args.retry_terminal,
        )
        conn.close()
        report.update(
            results=results,
            status_counts=dict(Counter(r["status"] for r in results)),
            rows_added=sum(r["rows_added"] for r in results),
            elapsed_sec=round(time.monotonic() - started, 1),
            db_bytes_before=size_before,
            db_bytes_after=db_path.stat().st_size,
        )
    text = json.dumps(report, indent=1, sort_keys=True, default=str)
    if args.report_json:
        args.report_json.write_text(text)
    print(json.dumps({k: v for k, v in report.items() if k not in ("results", "coverage", "spells_ended_before_floor_dropped_list")}, indent=1, default=str))
    return 1 if report.get("status_counts", {}).get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
