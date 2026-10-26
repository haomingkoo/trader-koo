#!/usr/bin/env python3
"""Store daily option metrics (options_daily_metrics) for the whole ingest universe.

Runs after the US close and stops before the nightly report window, or earlier
if a pipeline run becomes active. Nothing here changes trading or report logic.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sqlite3
from pathlib import Path

from trader_koo.backend.services.pipeline import read_active_pipeline_run
from trader_koo.collection.config import get_options_universe_config
from trader_koo.collection.options_universe import (
    fetch_rate,
    load_ingest_universe,
    snapshot_universe,
    universe_window_open,
)
from trader_koo.config import DEFAULT_DB_PATH, env_path, env_str
from trader_koo.scripts.snapshot_options_iv import setup_logging

LOG = logging.getLogger("trader_koo.options_snapshot")


def build_parser() -> argparse.ArgumentParser:
    uni = get_options_universe_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db-path", default=str(env_path("TRADER_KOO_DB_PATH", DEFAULT_DB_PATH)))
    p.add_argument("--max-tickers", type=int, default=uni.max_tickers)
    p.add_argument("--workers", type=int, default=uni.workers)
    p.add_argument("--sleep", type=float, default=uni.sleep_sec)
    p.add_argument("--max-expiries", type=int, default=uni.nearest_expiries)
    p.add_argument("--target-days", default=",".join(str(d) for d in uni.target_days))
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--ignore-window",
        action="store_true",
        help="manual runs only: skip the post-close / pre-report time guard",
    )
    p.add_argument("--log-level", default=env_str("TRADER_KOO_LOG_LEVEL", "INFO", allow_blank=False))
    p.add_argument("--log-file", default=str(uni.log_path))
    return p


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.log_level, args.log_file)
    db_path = Path(args.db_path)
    target_days = tuple(int(d) for d in args.target_days.split(",") if d.strip())

    def should_stop() -> bool:
        if not args.ignore_window and not universe_window_open(dt.datetime.now(dt.timezone.utc)):
            LOG.warning("Options universe snapshot stopping: outside post-close window")
            return True
        active = read_active_pipeline_run(db_path)
        if active is not None:
            LOG.warning("Options universe snapshot stopping: pipeline run %s active", active.get("run_id"))
            return True
        return False

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        tickers = load_ingest_universe(conn, max_tickers=args.max_tickers)
        if not tickers:
            raise SystemExit("ingest universe unavailable; refusing to guess a ticker list")
        LOG.info(
            "Starting options universe snapshot tickers=%d workers=%d expiries=nearest%d+%s",
            len(tickers), args.workers, args.max_expiries, list(target_days),
        )
        try:
            rate = fetch_rate()
        except Exception as exc:  # skew needs a rate; without one it is stored as NULL
            LOG.warning("%s rate unavailable (%s); put_skew_25d will be NULL", "^IRX", exc)
            rate = None
        summary = snapshot_universe(
            conn,
            tickers,
            nearest=args.max_expiries,
            target_days=target_days,
            force=args.force,
            sleep_sec=args.sleep,
            workers=args.workers,
            rate=rate,
            should_stop=should_stop,
        )
        LOG.info(
            "Options universe snapshot done rows=%d skipped=%d quality=%s errors=%d stopped_early=%s",
            summary["rows_written"], summary["skipped_existing"], summary["quality"],
            len(summary["errors"]), summary["stopped_early"],
        )
        print(json.dumps(summary, sort_keys=True))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
