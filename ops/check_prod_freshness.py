"""Check production reachability and price freshness from outside the container.

Every in-app monitor dies with the process, so a stopped container went
unnoticed for 47 hours. Run from GitHub Actions on a schedule. Exit 1 on any
problem so the failed run notifies the owner. Standard library only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trader_koo.report.utils import expected_price_date  # noqa: E402

# Covers GitHub's delayed cron and a slow nightly run on top of INGEST_DUE_LAG.
EXTRA_GRACE = dt.timedelta(hours=2)
FETCH_ATTEMPTS = 3
FETCH_TIMEOUT_SEC = 20
RETRY_SLEEP_SEC = 10
# The site firewall answers 403 to the default Python-urllib User-Agent.
USER_AGENT = "trader-koo-freshness-monitor/1.0 (+https://github.com/haomingkoo/trader-koo)"


def fetch_json(url: str, timeout: int, attempts: int) -> dict[str, Any]:
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return json.load(resp)
        except (OSError, ValueError):
            if attempt == attempts:
                raise
            time.sleep(RETRY_SLEEP_SEC)
    raise AssertionError("unreachable")


def evaluate(health: dict[str, Any], status: dict[str, Any], now_utc: dt.datetime) -> list[str]:
    problems: list[str] = []
    if not (health.get("ok") and health.get("db_exists")):
        problems.append(f"/api/health not ok: {health}")
    price_date = (status.get("latest_data") or {}).get("price_date")
    if not price_date:
        problems.append("/api/status reports no latest_data.price_date")
        return problems
    expected = expected_price_date(now_utc - EXTRA_GRACE)
    if str(price_date)[:10] < expected.isoformat():
        problems.append(f"price_date {price_date} is behind expected {expected.isoformat()}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args(argv)
    base = args.base_url.strip().rstrip("/")
    if not base:
        print("FAIL: base URL is empty; set the TRADER_KOO_PRODUCTION_URL repository variable")
        return 1
    try:
        health = fetch_json(f"{base}/api/health", FETCH_TIMEOUT_SEC, FETCH_ATTEMPTS)
        status = fetch_json(f"{base}/api/status", FETCH_TIMEOUT_SEC, FETCH_ATTEMPTS)
    except (OSError, ValueError) as exc:
        print(f"FAIL: service unreachable or returned invalid JSON: {type(exc).__name__}")
        return 1
    problems = evaluate(health, status, dt.datetime.now(dt.timezone.utc))
    for problem in problems:
        print(f"FAIL: {problem}")
    if not problems:
        print(f"OK: price_date={status['latest_data']['price_date']}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
