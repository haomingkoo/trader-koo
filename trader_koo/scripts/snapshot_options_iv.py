#!/usr/bin/env python3
"""Snapshot bounded Yahoo/yfinance option-chain context into options_iv."""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from logging.handlers import RotatingFileHandler
from pathlib import Path

from trader_koo.config import DEFAULT_DB_PATH, env_path, env_str, get_options_config
from trader_koo.options_research import (
    load_options_snapshot_tickers,
    snapshot_options_iv,
)

LOG = logging.getLogger("trader_koo.options_snapshot")


def setup_logging(level: str, log_file: str | None) -> None:
    LOG.handlers.clear()
    LOG.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    LOG.addHandler(stream)
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(str(path), maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(formatter)
        LOG.addHandler(handler)


def build_parser() -> argparse.ArgumentParser:
    options_config = get_options_config()
    snapshot_config = options_config.snapshot
    p = argparse.ArgumentParser(description="Snapshot yfinance option-chain rows into options_iv.")
    p.add_argument(
        "--db-path",
        default=str(env_path("TRADER_KOO_DB_PATH", DEFAULT_DB_PATH)),
    )
    p.add_argument("--tickers", default=snapshot_config.tickers)
    p.add_argument(
        "--latest-report",
        default=str(snapshot_config.latest_report_path),
    )
    p.add_argument("--max-tickers", type=int, default=snapshot_config.max_tickers)
    p.add_argument("--max-expiries", type=int, default=snapshot_config.max_expiries)
    p.add_argument("--min-moneyness", type=float, default=snapshot_config.min_moneyness)
    p.add_argument("--max-moneyness", type=float, default=snapshot_config.max_moneyness)
    p.add_argument(
        "--min-interval-hours",
        type=float,
        default=snapshot_config.min_interval_hours,
    )
    p.add_argument("--sleep", type=float, default=snapshot_config.sleep_sec)
    p.add_argument("--force", action="store_true")
    p.add_argument("--log-level", default=env_str("TRADER_KOO_LOG_LEVEL", "INFO", allow_blank=False))
    p.add_argument(
        "--log-file",
        default=str(snapshot_config.log_path),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.log_level, args.log_file)

    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        tickers = load_options_snapshot_tickers(
            conn,
            explicit_tickers=args.tickers,
            latest_report_path=args.latest_report,
            max_tickers=args.max_tickers,
        )
        LOG.info(
            "Starting options snapshot tickers=%d max_expiries=%d moneyness=%.2f-%.2f min_interval_hours=%.1f",
            len(tickers),
            args.max_expiries,
            args.min_moneyness,
            args.max_moneyness,
            args.min_interval_hours,
        )
        summary = snapshot_options_iv(
            conn,
            tickers,
            max_expiries=args.max_expiries,
            min_moneyness=args.min_moneyness,
            max_moneyness=args.max_moneyness,
            min_interval_hours=args.min_interval_hours,
            force=args.force,
            sleep_sec=args.sleep,
        )
        LOG.info(
            "Options snapshot done refreshed=%d skipped_recent=%d empty=%d failed=%d rows=%d",
            summary["tickers_refreshed"],
            summary["tickers_skipped_recent"],
            summary["tickers_empty"],
            summary["tickers_failed"],
            summary["rows_inserted"],
        )
        print(json.dumps(summary, sort_keys=True))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
