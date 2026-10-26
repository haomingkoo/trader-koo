#!/usr/bin/env python3
"""Load SEC EDGAR companyfacts into sec_fundamentals (resumable, rate-limited).

Never defaults to the production database: --db-path is required. Re-running
skips finished CIKs; --refresh refetches them to pick up newer filings.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path

from trader_koo.fundamentals import config as cfg
from trader_koo.fundamentals.loader import add_predecessor_cik, load_universe, store_cik_map, universe_tickers
from trader_koo.fundamentals.report import coverage_report
from trader_koo.fundamentals.schema import ensure_sec_schema
from trader_koo.fundamentals.sec_client import SecClient

LOG = logging.getLogger("trader_koo.fundamentals.cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db-path", required=True, help="SQLite file that receives sec_* tables")
    p.add_argument("--universe-db", help="DB with finviz_fundamentals (default: --db-path)")
    p.add_argument("--tickers", help="comma-separated override of the universe")
    p.add_argument("--limit", type=int, help="load at most this many universe tickers")
    p.add_argument(
        "--predecessor", action="append", default=[], metavar="TICKER=CIK",
        help="also load a predecessor CIK for TICKER, e.g. XOM=34088 (repeatable)",
    )
    p.add_argument("--refresh", action="store_true", help="refetch CIKs already loaded")
    p.add_argument("--report", action="store_true", help="print coverage JSON and exit")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")
    db_path = Path(args.db_path)
    conn = sqlite3.connect(db_path)
    ensure_sec_schema(conn)
    universe = sqlite3.connect(f"file:{args.universe_db or db_path}?mode=ro", uri=True)
    if args.report:
        print(json.dumps(coverage_report(conn, universe), indent=2))
        return 0
    client = SecClient()
    status, payload = client.get_json(cfg.SEC_TICKERS_URL)
    if payload is None:
        raise SystemExit(f"SEC ticker map returned HTTP {status}; refusing to continue without it")
    LOG.info("mapped %d tickers to CIKs", store_cik_map(conn, payload))
    for item in args.predecessor:
        ticker, _, cik = item.partition("=")
        add_predecessor_cik(conn, ticker, int(cik))
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = universe_tickers(universe)
    if args.limit:
        tickers = tickers[: args.limit]
    counts = load_universe(conn, client, tickers, refresh=args.refresh)
    LOG.info("done: %s", counts)
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
