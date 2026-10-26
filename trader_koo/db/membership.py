"""Point-in-time S&P 500 membership: schema, ticker normalisation, queries.

Research bench only. Nothing here changes trading, admission or any paper_* /
report_* table. Intervals are half-open: a ticker is a member on ``d`` when
``start_date <= d < end_date`` (``end_date`` NULL means still a member). The
source data records the index-change effective date as the end date, so on that
day the old name is already out and its replacement already in.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable

SOURCE_NAME = "fja05680/sp500"

# Tickers are stored in the form price_daily uses (yfinance): BRK.B -> BRK-B.
# yfinance serves a renamed company's whole history under its current symbol,
# so a rename is folded into the current symbol and the touching intervals are
# merged. Every entry is a continuous company, not a recycled symbol. Symbols
# reused by a different company (e.g. AAL 1996 vs AAL 2015) stay separate rows.
TICKER_RENAMES = {
    "FB": "META",
    "CDAY": "DAY",
    "WLTW": "WTW",
    "ANTM": "ELV",
    "PKI": "RVTY",
    "ABC": "COR",
    "FI": "FISV",
    "HCP": "DOC",
    "PEAK": "DOC",
    "TMK": "GL",
    "UTX": "RTX",
    "DISCA": "WBD",
    "DISCK": "WBD",
    "VIAC": "PSKY",
    "PARA": "PSKY",
    "BK": "BNY",
    "MMC": "MRSH",
    "BHGE": "BKR",
    "HRS": "LHX",
    "KORS": "CPRI",
    "SYMC": "GEN",
    "NLOK": "GEN",
    "BLL": "BALL",
    "COG": "CTRA",
    "LB": "BBWI",
    "JEC": "J",
    "CTL": "LUMN",
    "RE": "EG",
    "FLT": "CPAY",
}

Interval = tuple[str, str, str | None]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sp500_membership (
    ticker TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT,
    source TEXT NOT NULL,
    source_version TEXT NOT NULL,
    PRIMARY KEY (ticker, start_date)
)
"""
_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_sp500_membership_dates "
    "ON sp500_membership(start_date, end_date)"
)


def normalize_ticker(ticker: str) -> str:
    symbol = ticker.strip().upper().replace(".", "-")
    return TICKER_RENAMES.get(symbol, symbol)


def build_intervals(rows: Iterable[tuple[str, str, str | None]]) -> list[Interval]:
    """Normalise tickers, then merge intervals that touch or overlap."""
    by_ticker: dict[str, list[tuple[str, str | None]]] = {}
    for ticker, start, end in rows:
        by_ticker.setdefault(normalize_ticker(ticker), []).append((start, end or None))
    out: list[Interval] = []
    for ticker in sorted(by_ticker):
        merged: list[list[str | None]] = []
        for start, end in sorted(by_ticker[ticker], key=lambda r: r[0]):
            last = merged[-1] if merged else None
            if last is not None and (last[1] is None or start <= last[1]):
                last[1] = None if last[1] is None or end is None else max(last[1], end)
            else:
                merged.append([start, end])
        out.extend((ticker, str(s), e) for s, e in merged)
    return out


def ensure_membership_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_SCHEMA)
    conn.execute(_INDEX)


def replace_membership(
    conn: sqlite3.Connection, intervals: Iterable[Interval], source: str, source_version: str
) -> int:
    """Replace every row of ``source`` in one transaction; safe to repeat."""
    ensure_membership_schema(conn)
    rows = [(t, s, e, source, source_version) for t, s, e in intervals]
    with conn:
        conn.execute("DELETE FROM sp500_membership WHERE source = ?", (source,))
        conn.executemany("INSERT INTO sp500_membership VALUES (?, ?, ?, ?, ?)", rows)
    return len(rows)


def members_on(conn: sqlite3.Connection, date: str) -> set[str]:
    rows = conn.execute(
        "SELECT ticker FROM sp500_membership "
        "WHERE start_date <= ? AND (end_date IS NULL OR ? < end_date)",
        (date, date),
    )
    return {r[0] for r in rows}


def is_member(conn: sqlite3.Connection, ticker: str, date: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sp500_membership WHERE ticker = ? AND start_date <= ? "
        "AND (end_date IS NULL OR ? < end_date) LIMIT 1",
        (normalize_ticker(ticker), date, date),
    ).fetchone()
    return row is not None
