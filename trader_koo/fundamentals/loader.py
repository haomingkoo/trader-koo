"""Resumable per-CIK loader for SEC companyfacts into sec_fundamentals."""
from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from collections.abc import Iterable, Iterator
from typing import Any, NamedTuple

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


class FetchResult(NamedTuple):
    status: str
    duplicates_ignored: int


def fact_rows(cik: int, payload: dict[str, Any]) -> Iterator[tuple]:
    """Yield raw reported facts for the stored tag set; nothing is derived.

    A payload without a ``facts`` object is malformed and raises; a company that
    reports none of the stored tags yields no rows.
    """
    if "facts" not in payload:
        raise ValueError(f"cik {cik}: companyfacts payload has no 'facts' object")
    for taxonomy, tag in sorted(cfg.STORED_TAGS):
        node = payload["facts"].get(taxonomy, {}).get(tag)
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
    """Current ingest universe (finviz snapshot tickers) plus departed index members.

    Raises LookupError when the database has no ``sp500_membership`` table: without
    it the departed members would be missing from the universe without notice.
    """
    tickers = {r[0] for r in universe_conn.execute("SELECT DISTINCT ticker FROM finviz_fundamentals")}
    has_membership = universe_conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sp500_membership'"
    ).fetchone()
    if not has_membership:
        raise LookupError("universe database has no sp500_membership table; departed members would be missing")
    tickers |= {r[0] for r in universe_conn.execute("SELECT DISTINCT ticker FROM sp500_membership")}
    return sorted(sec_ticker(t) for t in tickers if t and not t.startswith("^"))


def fetch_cik(conn: sqlite3.Connection, client: SecClient, cik: int) -> FetchResult:
    """Fetch and store one company. Returns the recorded status and the duplicate count.

    ``empty`` is returned (and ``ok`` stored with n_facts 0) when the company
    reports none of the stored tags. Facts already stored under the same primary
    key are not overwritten; their number is logged and counted by ``load_universe``.
    """
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
        return FetchResult("error", 0)
    if payload is None:
        conn.execute(
            "INSERT OR REPLACE INTO sec_fetch_status (cik, status, http_status, n_facts, fetched_ts)"
            " VALUES (?, 'no_facts', ?, 0, ?)",
            (cik, status, _now()),
        )
        conn.commit()
        return FetchResult("no_facts", 0)
    try:
        rows = list(fact_rows(cik, payload))
    except ValueError as exc:
        conn.execute(
            "INSERT OR REPLACE INTO sec_fetch_status (cik, status, http_status, n_facts, fetched_ts, error)"
            " VALUES (?, 'error', ?, 0, ?, ?)",
            (cik, status, _now(), str(exc)),
        )
        conn.commit()
        LOG.error("cik %s malformed payload: %s", cik, exc)
        return FetchResult("error", 0)
    tags = sorted({r[2] for r in rows})
    before = conn.total_changes
    with conn:
        conn.executemany(_INSERT, rows)
        inserted = conn.total_changes - before
        conn.execute(
            "INSERT OR REPLACE INTO sec_fetch_status"
            " (cik, status, http_status, n_facts, tags_json, fetched_ts) VALUES (?, 'ok', ?, ?, ?, ?)",
            (cik, status, len(rows), json.dumps(tags), _now()),
        )
    return FetchResult("ok" if rows else "empty", len(rows) - inserted)


def load_universe(
    conn: sqlite3.Connection,
    client: SecClient,
    tickers: Iterable[str],
    *,
    refresh: bool = False,
) -> dict[str, int]:
    """Load every mapped ticker's CIK once. Finished CIKs are skipped unless refresh."""
    counts = {"ok": 0, "empty": 0, "no_facts": 0, "error": 0, "skipped": 0, "unmapped": 0, "duplicate_facts_ignored": 0}
    done = set() if refresh else {
        r[0] for r in conn.execute("SELECT cik FROM sec_fetch_status WHERE status IN ('ok','no_facts')")
    }
    seen: set[int] = set()
    for ticker in tickers:
        ciks = [r[0] for r in conn.execute("SELECT cik FROM sec_cik_map WHERE ticker=?", (sec_ticker(ticker),))]
        if not ciks:
            counts["unmapped"] += 1
            LOG.warning("no CIK mapped for %s", ticker)
        for cik in ciks:
            if cik in seen or cik in done:
                counts["skipped"] += 1
                continue
            seen.add(cik)
            result = fetch_cik(conn, client, cik)
            counts[result.status] += 1
            counts["duplicate_facts_ignored"] += result.duplicates_ignored
    return counts
