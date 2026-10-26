#!/usr/bin/env python3
"""One-time earnings-history backfill into earnings_events.

Sources: Finnhub earnings calendar queried per symbol (needs FINNHUB_API_KEY,
paced under the free tier's 60 calls/min) and yfinance ``get_earnings_dates``.
Every row records its source; fields a source does not supply stay NULL.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sqlite3
from pathlib import Path

from trader_koo.catalyst_data import _fetch_finnhub_calendar_rows
from trader_koo.collection.config import (
    BACKFILL_DEFAULT_YEARS,
    BACKFILL_LOOKAHEAD_DAYS,
    UNIVERSE_DEFAULT_MAX_TICKERS,
)
from trader_koo.collection.earnings_history import (
    backfill_earnings_history,
    ensure_earnings_events_schema,
    fetch_yfinance_history,
)
from trader_koo.collection.options_universe import load_ingest_universe
from trader_koo.config import DEFAULT_DB_PATH, env_path
from trader_koo.options_research import normalize_options_tickers

LOG = logging.getLogger("trader_koo.earnings_backfill")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db-path", default=str(env_path("TRADER_KOO_DB_PATH", DEFAULT_DB_PATH)))
    p.add_argument("--tickers", default="", help="comma list; default is the latest ingest universe")
    p.add_argument("--max-tickers", type=int, default=UNIVERSE_DEFAULT_MAX_TICKERS)
    p.add_argument("--years", type=int, default=BACKFILL_DEFAULT_YEARS, help="Finnhub lookback in years")
    p.add_argument("--source", choices=["both", "finnhub", "yfinance"], default="both")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    conn = sqlite3.connect(str(Path(args.db_path)))
    conn.row_factory = sqlite3.Row
    try:
        ensure_earnings_events_schema(conn)
        conn.commit()
        tickers = normalize_options_tickers(args.tickers) or load_ingest_universe(conn)
        truncated = tickers[args.max_tickers:]
        tickers = tickers[: args.max_tickers]
        if not tickers:
            raise SystemExit("no tickers: pass --tickers or run after an ingest")

        finnhub_fetch = None
        sources_skipped: dict[str, str] = {}
        if args.source in ("both", "finnhub"):
            key = os.getenv("FINNHUB_API_KEY", "").strip()
            if key:
                today = dt.date.today()
                start = today.replace(year=today.year - args.years).isoformat()
                end = (today + dt.timedelta(days=BACKFILL_LOOKAHEAD_DAYS)).isoformat()
                finnhub_fetch = lambda sym: _fetch_finnhub_calendar_rows(key, start, end, symbol=sym.replace("-", "."))  # noqa: E731
            elif args.source == "finnhub":
                raise SystemExit("FINNHUB_API_KEY is not set")
            else:
                LOG.warning("FINNHUB_API_KEY not set; skipping Finnhub")
                sources_skipped["finnhub"] = "FINNHUB_API_KEY not set"
        yfinance_fetch = fetch_yfinance_history if args.source in ("both", "yfinance") else None

        summary = backfill_earnings_history(
            conn, tickers, finnhub_fetch=finnhub_fetch, yfinance_fetch=yfinance_fetch
        )
        summary["sources_skipped"] = sources_skipped
        summary["tickers_beyond_max_not_run"] = truncated
        print(json.dumps(summary, sort_keys=True))
        if summary["failures"]:
            raise SystemExit(f"{len(summary['failures'])} ticker/source fetches failed; see the summary above")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
