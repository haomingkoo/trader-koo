"""Load point-in-time S&P 500 membership and cross-check it against Wikipedia.

Source: fja05680/sp500 ``sp500_ticker_start_end.csv`` (MIT), fetched at a pinned
commit. Cross-checks use Wikipedia's current constituents and its change log
(now on "Historical components of the S&P 500"). Disagreements are written to a
report and never repaired.

    python -m trader_koo.scripts.load_sp500_membership --check-only \\
        --report-dir /tmp/membership --prices-db golden.db
    python -m trader_koo.scripts.load_sp500_membership --db-path /path/to/copy.db

``--check-only`` builds the table in memory and writes nothing to any database.
Without it the table is written to ``--db-path``, which is required (there is
no default). The table is meant for the separate research database on the
workstation, not production: the production volume is near capacity, so do not
point ``--db-path`` at it. The table is additive and nothing in the app creates
or reads it.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import sqlite3
import urllib.request
from pathlib import Path
from typing import Any

from trader_koo.db import membership as mem
from trader_koo.db import membership_check as chk

PINNED_SHA = "a2430f2af0c79ddf0748e91de11bdeb1616ab5a7"
CSV_URL = "https://raw.githubusercontent.com/fja05680/sp500/{sha}/sp500_ticker_start_end.csv"
WIKI_CURRENT_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
USER_AGENT = "trader-koo-research (haoming@kooexperience.com)"


def _year(iso: str) -> str:
    return iso.split("-")[0]


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=chk.FETCH_TIMEOUT_SEC) as response:  # noqa: S310 - fixed https URLs
        return response.read().decode("utf-8")


def read_source_csv(text: str) -> list[tuple[str, str, str | None]]:
    return [
        (r["ticker"], r["start_date"], r["end_date"] or None)
        for r in csv.DictReader(io.StringIO(text))
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _attach_prices(conn: sqlite3.Connection, prices_db: Path | None) -> str | None:
    if prices_db is not None:
        conn.execute("ATTACH DATABASE ? AS px", (f"{prices_db.resolve().as_uri()}?mode=ro",))
        return "px.price_daily"
    has = conn.execute("SELECT 1 FROM sqlite_master WHERE name='price_daily'").fetchone()
    return "price_daily" if has else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    as_of = dt.datetime.now(dt.timezone.utc).date()
    sha = args.sha
    source_csv = Path(args.source_csv).read_text() if args.source_csv else _fetch(CSV_URL.format(sha=sha))
    current_html = Path(args.wiki_current).read_text() if args.wiki_current else _fetch(WIKI_CURRENT_URL)
    changes_html = Path(args.wiki_changes).read_text() if args.wiki_changes else _fetch(WIKI_CHANGES_URL)

    intervals = mem.build_intervals(read_source_csv(source_csv))
    target = ":memory:" if args.check_only else Path(args.db_path).resolve().as_uri()
    conn = sqlite3.connect(target, uri=True)
    mem.replace_membership(conn, intervals, mem.SOURCE_NAME, sha)

    since = f"{chk.MEMBERSHIP_FIRST_YEAR}-01-01"
    counts = chk.month_end_counts(conn, as_of)
    current = chk.compare_current(conn, as_of, chk.parse_wiki_constituents(current_html))
    problems = chk.compare_events(conn, chk.parse_wiki_changes(changes_html), since)

    report = Path(args.report_dir)
    report.mkdir(parents=True, exist_ok=True)
    _write_csv(report / "month_end_counts.csv", counts, ["date", "count", "ok"])
    _write_csv(report / "disagreements.csv", problems, ["issue", "kind", "ticker", "date", "other_date"])
    summary: dict[str, Any] = {
        "source_version": sha,
        "as_of": as_of.isoformat(),
        "intervals": len(intervals),
        "month_ends_checked": len(counts),
        "month_ends_out_of_range": [c for c in counts if not c["ok"]],
        "count_by_year": {
            y: {"min": min(c["count"] for c in counts if _year(c["date"]) == y),
                "max": max(c["count"] for c in counts if _year(c["date"]) == y)}
            for y in sorted({_year(c["date"]) for c in counts})
        },
        "current_vs_wikipedia": current,
        "disagreement_counts": {
            k: sum(1 for p in problems if p["issue"] == k)
            for k in ("date_mismatch", "wiki_only", "db_only")
        },
    }
    price_table = _attach_prices(conn, Path(args.prices_db) if args.prices_db else None)
    if price_table:
        until = as_of.isoformat()
        summary["coverage_by_year"] = chk.coverage_by_year(conn, since, until, price_table)
        gap = chk.survivorship_gap(conn, since, until, price_table)
        _write_csv(report / "survivorship_gap.csv", gap, ["ticker", "member_days", "last_end"])
        summary["survivorship_gap_tickers"] = len(gap)
    (report / "summary.json").write_text(json.dumps(summary, indent=2))
    conn.close()
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--db-path", type=Path, help="database to write (required unless --check-only)")
    parser.add_argument("--check-only", action="store_true", help="build in memory, write no database")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--prices-db", type=Path, help="read-only database holding price_daily")
    parser.add_argument("--sha", default=PINNED_SHA, help="fja05680/sp500 commit to fetch")
    parser.add_argument("--source-csv", help="local copy of the source CSV (offline/testing)")
    parser.add_argument("--wiki-current", help="local HTML of the constituents page")
    parser.add_argument("--wiki-changes", help="local HTML of the changes page")
    args = parser.parse_args(argv)
    if not args.check_only and args.db_path is None:
        parser.error("--db-path is required unless --check-only")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
