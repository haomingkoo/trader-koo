"""Resumable per-CIK loader for SEC companyfacts into sec_fundamentals."""
from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from collections.abc import Iterable, Iterator
from typing import Any

from trader_koo.fundamentals import config as cfg
from trader_koo.fundamentals.sec_client import SecClient, SecFetchError

LOG = logging.getLogger("trader_koo.fundamentals.loader")

_INSERT = """INSERT OR IGNORE INTO sec_fundamentals
    (cik, taxonomy, tag, unit, period_start, period_end, fiscal_year, fiscal_period,
     value, form, accession, filed_date, frame)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sec_ticker(ticker: str) -> str:
    """Map a Yahoo-style symbol to SEC's spelling (BRK-B stays BRK-B)."""
    return ticker.strip().upper().replace(".", "-")


def fact_rows(cik: int, payload: dict[str, Any]) -> Iterator[tuple]:
    """Yield raw reported facts for the stored tag set; nothing is derived."""
    for taxonomy, tag in sorted(cfg.STORED_TAGS):
        node = payload.get("facts", {}).get(taxonomy, {}).get(tag)
        if not node:
            continue
        for unit, facts in node.get("units", {}).items():
            for f in facts:
                yield (
                    cik, taxonomy, tag, unit, f.get("start") or "", f["end"],
                    f.get("fy"), f.get("fp"), f["val"], f.get("form"),
                    f["accn"], f["filed"], f.get("frame"),
                )


def store_cik_map(conn: sqlite3.Connection, tickers_payload: dict[str, Any]) -> int:
    ts = _now()
    rows = [
        (sec_ticker(v["ticker"]), int(v["cik_str"]), v.get("title"), ts)
        for v in tickers_payload.values()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO sec_cik_map (ticker, cik, title, mapped_ts) VALUES (?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def add_predecessor_cik(conn: sqlite3.Connection, ticker: str, cik: int) -> None:
    """Attach a predecessor CIK to a ticker (e.g. after a holding-company reorganisation)."""
    conn.execute(
        "INSERT OR REPLACE INTO sec_cik_map (ticker, cik, title, mapped_ts) VALUES (?,?,'predecessor',?)",
        (sec_ticker(ticker), cik, _now()),
    )
    conn.commit()


def universe_tickers(universe_conn: sqlite3.Connection) -> list[str]:
    """Current ingest universe (finviz snapshot tickers) plus departed index members."""
    tickers = {r[0] for r in universe_conn.execute("SELECT DISTINCT ticker FROM finviz_fundamentals")}
    has_membership = universe_conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sp500_membership'"
    ).fetchone()
    if has_membership:
        tickers |= {r[0] for r in universe_conn.execute("SELECT DISTINCT ticker FROM sp500_membership")}
    return sorted(sec_ticker(t) for t in tickers if t and not t.startswith("^"))


def fetch_cik(conn: sqlite3.Connection, client: SecClient, cik: int) -> str:
    """Fetch and store one company. Returns the recorded status."""
    try:
        status, payload = client.get_json(cfg.SEC_FACTS_URL.format(cik=cik))
    except SecFetchError as exc:
        conn.execute(
            "INSERT OR REPLACE INTO sec_fetch_status (cik, status, n_facts, fetched_ts, error)"
            " VALUES (?, 'error', 0, ?, ?)",
            (cik, _now(), str(exc)),
        )
        conn.commit()
        LOG.error("cik %s failed: %s", cik, exc)
        return "error"
    if payload is None:
        conn.execute(
            "INSERT OR REPLACE INTO sec_fetch_status (cik, status, http_status, n_facts, fetched_ts)"
            " VALUES (?, 'no_facts', ?, 0, ?)",
            (cik, status, _now()),
        )
        conn.commit()
        return "no_facts"
    rows = list(fact_rows(cik, payload))
    tags = sorted({r[2] for r in rows})
    with conn:
        conn.executemany(_INSERT, rows)
        conn.execute(
            "INSERT OR REPLACE INTO sec_fetch_status"
            " (cik, status, http_status, n_facts, tags_json, fetched_ts) VALUES (?, 'ok', ?, ?, ?, ?)",
            (cik, status, len(rows), json.dumps(tags), _now()),
        )
    return "ok"


def load_universe(
    conn: sqlite3.Connection,
    client: SecClient,
    tickers: Iterable[str],
    *,
    refresh: bool = False,
) -> dict[str, int]:
    """Load every mapped ticker's CIK once. Finished CIKs are skipped unless refresh."""
    counts = {"ok": 0, "no_facts": 0, "error": 0, "skipped": 0, "unmapped": 0}
    done = set() if refresh else {
        r[0] for r in conn.execute("SELECT cik FROM sec_fetch_status WHERE status IN ('ok','no_facts')")
    }
    seen: set[int] = set()
    for ticker in tickers:
        ciks = [r[0] for r in conn.execute("SELECT cik FROM sec_cik_map WHERE ticker=?", (sec_ticker(ticker),))]
        if not ciks:
            counts["unmapped"] += 1
        for cik in ciks:
            if cik in seen or cik in done:
                counts["skipped"] += 1
                continue
            seen.add(cik)
            counts[fetch_cik(conn, client, cik)] += 1
    return counts
