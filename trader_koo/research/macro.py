"""Market and macro series for the research database, stored with the date each value became known.

FRED / ALFRED: every vintage of every observation (``realtime_start`` is the first day FRED showed
that value), so a signal on day t reads the value as it was published by t, not the later revision.
CBOE volatility indices: daily closes, never revised; a close is known at the end of its own day.

The FRED key is read from the FRED_API_KEY environment variable and never logged.

Usage: FRED_API_KEY=... python -m trader_koo.research.macro --db <research db>
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import sqlite3
import sys
import urllib.parse
import urllib.request

# Source definitions (the published series ids and endpoints), not tunables.
FRED_SERIES = {
    "M2SL": "M2 money stock, monthly, seasonally adjusted",
    "CPIAUCSL": "CPI all urban consumers, monthly, seasonally adjusted",
    "DTWEXBGS": "Nominal broad U.S. dollar index, daily",
    "T10Y2Y": "10-year minus 2-year Treasury constant maturity, daily",
    "DTB3": "3-month Treasury bill secondary market rate, daily, percent",
}
# Market series FRED never revises and ALFRED refuses in full (too many vintages): the current file is
# loaded and each value is taken as known from the next calendar day, when FRED posts it.
FRED_LATEST_SERIES = {"T10Y2Y", "DTB3"}
CBOE_SERIES = ("VIX3M", "VIX9D", "SKEW", "VVIX")
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
FRED_PAGE = 100000          # FRED's maximum rows per request
HTTP_TIMEOUT_SEC = 60
USER_AGENT = "Mozilla/5.0 trader-koo research haomingkoo@gmail.com"
ALL_TIME = ("1776-07-04", "9999-12-31")  # ALFRED's documented full real-time range
MISSING = "."                # FRED's marker for no value
SOURCE_COL = 5               # position of "source" in a macro_series row

SCHEMA = """
CREATE TABLE IF NOT EXISTS macro_series (
    series TEXT NOT NULL,
    obs_date TEXT NOT NULL,
    value REAL NOT NULL,
    known_from TEXT NOT NULL,
    known_until TEXT NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (series, obs_date, known_from)
);
CREATE INDEX IF NOT EXISTS idx_macro_series_known ON macro_series (series, known_from);
CREATE TABLE IF NOT EXISTS macro_loads (
    series TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    rows INTEGER NOT NULL,
    first_obs TEXT,
    last_obs TEXT,
    loaded_ts TEXT NOT NULL
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:
        return resp.read()


def _next_day(day: str) -> str:
    return (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()


def fred_rows(series: str, key: str, fetch=_get) -> list[tuple]:
    """Every vintage of ``series``: (series, obs_date, value, known_from, known_until, 'alfred')."""
    latest = series in FRED_LATEST_SERIES
    out, offset = [], 0
    while True:
        params = {"series_id": series, "api_key": key, "file_type": "json", "limit": FRED_PAGE, "offset": offset}
        if not latest:
            params.update(realtime_start=ALL_TIME[0], realtime_end=ALL_TIME[1])
        query = urllib.parse.urlencode(params)
        page = json.loads(fetch(f"{FRED_URL}?{query}"))
        obs = page["observations"]
        out += [(series, o["date"], float(o["value"]),
                 _next_day(o["date"]) if latest else o["realtime_start"],
                 ALL_TIME[1] if latest else o["realtime_end"], "fred_latest" if latest else "alfred")
                for o in obs if o["value"] != MISSING]
        offset += len(obs)
        if not obs or offset >= int(page["count"]):
            return out


def cboe_rows(name: str, fetch=_get) -> list[tuple]:
    """Daily closes of a CBOE index; each is known from its own date."""
    reader = csv.DictReader(io.StringIO(fetch(CBOE_URL.format(name=name)).decode("utf-8")))
    col = "CLOSE" if "CLOSE" in (reader.fieldnames or []) else name
    out = []
    for r in reader:
        day = dt.datetime.strptime(r["DATE"], "%m/%d/%Y").date().isoformat()
        if r[col]:
            out.append((name, day, float(r[col]), day, ALL_TIME[1], "cboe"))
    return out


def store(conn: sqlite3.Connection, series: str, rows: list[tuple]) -> None:
    if not rows:
        raise ValueError(f"{series}: no rows returned; refusing to record an empty load")
    with conn:
        conn.execute("DELETE FROM macro_series WHERE series = ?", (series,))
        conn.executemany("INSERT INTO macro_series VALUES (?,?,?,?,?,?)", rows)
        conn.execute("INSERT OR REPLACE INTO macro_loads VALUES (?,?,?,?,?,?)",
                     (series, rows[0][SOURCE_COL], len(rows), min(r[1] for r in rows), max(r[1] for r in rows),
                      dt.datetime.now(dt.timezone.utc).isoformat()))


def as_known(conn: sqlite3.Connection, series: str, day: str) -> list[tuple[str, float]]:
    """(obs_date, value) of ``series`` as published by the end of ``day``, oldest first."""
    return conn.execute(
        """SELECT obs_date, value FROM macro_series
           WHERE series = ? AND known_from <= ? AND known_until >= ? ORDER BY obs_date""",
        (series, day, day)).fetchall()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="trader_koo.research.macro")
    p.add_argument("--db", required=True)
    args = p.parse_args(argv)
    key = os.environ.get("FRED_API_KEY")
    if not key:
        print("FRED_API_KEY is not set", file=sys.stderr)
        return 2
    conn = sqlite3.connect(args.db)
    ensure_schema(conn)
    for s in FRED_SERIES:
        rows = fred_rows(s, key)
        store(conn, s, rows)
        print(f"{s}: {len(rows)} vintage rows")
    for s in CBOE_SERIES:
        rows = cboe_rows(s)
        store(conn, s, rows)
        print(f"{s}: {len(rows)} daily closes")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
