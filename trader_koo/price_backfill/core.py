"""Core of the price-history backfill (see backfill_price_history.py for the CLI).

Backfill daily prices before each ticker's earliest stored row (research only).

Universe: every ticker that was an S&P 500 member at any time since the floor
date, from the fja05680/sp500 membership file (pinned commit below). Prices and
corporate actions come from the same fetch, normalisation, ledger-reconciliation
and write functions as the daily ingest in ``update_market_db``, so the rows
carry the same raw (split-adjusted, not dividend-adjusted) basis.

Safety rules, each covered by tests:
- Only dates before the ticker's current earliest row are requested or written.
- A write that would alter any existing price or action row is rolled back.
- Departed tickers are fetched only through their last membership spell, so a
  symbol later reused by another company never gets the wrong company's bars.
  A departed symbol with no bars in its spell that still trades today is
  skipped and reported as a recycled symbol.
- Progress is kept in a state file, so an interrupted run resumes.

Target: a separate research database (``--db-path``), never the production
database. ``--source-db`` seeds it with the current price rows, then the
backfill adds the earlier dates. Production storage is not touched.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import pandas as pd

from trader_koo.price_backfill import config as cfg
from trader_koo.price_backfill.store import load_state, save_state
from trader_koo.db.sources import PriceFetchError
from trader_koo.scripts.update_market_db import (
    ensure_schema,
    fetch_price_daily,
    mark_full_history_actions_verified,
    reconcile_vendor_action_ledger,
    write_price_daily,
)

LOG = logging.getLogger("trader_koo.backfill")

DEFAULT_FLOOR = "2010-01-01"
DEFAULT_WARMUP_DAYS = cfg.WARMUP_DAYS
# Calendar days kept after a departed ticker's last spell day (delisting settlement).
DEPARTURE_BUFFER_DAYS = cfg.DEPARTURE_BUFFER_DAYS
# The floor is often a market holiday; history this close to it counts as reaching it.
FLOOR_TOLERANCE_DAYS = cfg.FLOOR_TOLERANCE_DAYS
TERMINAL_STATUSES = {"ok", "up_to_date", "no_data", "skipped_recycled_symbol",
                     "skipped_would_modify_existing_rows", "skipped_history_starts_after_spell"}

FetchFn = Callable[[str, str, Optional[str]], tuple[pd.DataFrame, str]]
ActionsFn = Callable[[str], list]
ProbeFn = Callable[[str], bool]


class Spell(NamedTuple):
    start: str
    end: Optional[str]


def to_db_ticker(symbol: str) -> str:
    return symbol.strip().upper().replace(".", "-")


def load_membership(path: Path, *, floor: str) -> dict[str, list[Spell]]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    membership: dict[str, list[Spell]] = {}
    for row in frame.itertuples(index=False):
        end = row.end_date or None
        if end is not None and end < floor:
            continue
        ticker = to_db_ticker(row.ticker)
        ticker = cfg.SYMBOL_RENAMES.get(ticker, ticker)
        membership.setdefault(ticker, []).append(Spell(row.start_date, end))
    return {ticker: sorted(spells) for ticker, spells in sorted(membership.items())}


def _day(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def usable_spells(spells: list[Spell], *, same_company: bool = False) -> list[Spell]:
    """Drop spells before a long membership gap unless the symbol is a known same-company re-add.

    After a gap of a year or more the symbol may belong to another company, and
    Yahoo serves only the current holder's bars. Attaching them to the earlier
    member would put the wrong company's prices on the member's dates.
    """
    if same_company:
        return list(spells)
    kept = [spells[-1]]
    for earlier in reversed(spells[:-1]):
        gap = (_day(kept[0].start) - _day(earlier.end or kept[0].start)).days
        if gap > cfg.GAP_DAYS:
            break
        kept.insert(0, earlier)
    return kept


def plan_window(
    spells: list[Spell],
    earliest: Optional[str],
    *,
    floor: str,
    warmup_days: int,
    same_company: bool = False,
) -> Optional[tuple[str, Optional[str]]]:
    """Return the half-open ``[start, end)`` range to fetch, or None if nothing is missing."""
    relevant = [
        s for s in usable_spells(spells, same_company=same_company)
        if s.end is None or s.end >= floor
    ]
    if not relevant:
        return None
    start = max(_day(floor), _day(min(s.start for s in relevant)) - dt.timedelta(days=warmup_days))
    end: Optional[dt.date] = None
    if all(s.end is not None for s in relevant):
        last_end = _day(max(s.end for s in relevant if s.end))
        end = last_end + dt.timedelta(days=DEPARTURE_BUFFER_DAYS + 1)
    if earliest is not None:
        earliest_day = _day(earliest)
        if earliest_day <= start + dt.timedelta(days=FLOOR_TOLERANCE_DAYS):
            return None
        end = earliest_day if end is None else min(end, earliest_day)
    if end is not None and end <= start:
        return None
    return start.isoformat(), end.isoformat() if end else None


def _in_a_spell(date: str, spells: list[Spell]) -> bool:
    for spell in spells:
        limit = (
            "9999-12-31" if spell.end is None
            else (_day(spell.end) + dt.timedelta(days=DEPARTURE_BUFFER_DAYS)).isoformat()
        )
        if spell.start <= date <= limit:
            return True
    return False


def _existing_snapshot(conn: sqlite3.Connection, ticker: str, earliest: Optional[str]) -> tuple:
    if earliest is None:
        return ((), ())
    prices = conn.execute(
        "SELECT * FROM price_daily WHERE ticker=? AND date>=? ORDER BY date", (ticker, earliest)
    ).fetchall()
    actions = conn.execute(
        "SELECT * FROM price_corporate_actions WHERE ticker=? AND action_date>=? "
        "ORDER BY action_date, action_type, provider",
        (ticker, earliest),
    ).fetchall()
    return tuple(prices), tuple(actions)


def _save_yolo(conn: sqlite3.Connection, ticker: str) -> tuple[list[str], list[tuple]]:
    """record_price_series_revision clears a ticker's chart patterns on a new start date.

    Older history does not change recent bars, so the patterns stay valid.
    """
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='yolo_patterns'"
    ).fetchone()
    if not exists:
        return [], []
    columns = [row[1] for row in conn.execute("PRAGMA table_info(yolo_patterns)")]
    rows = conn.execute("SELECT * FROM yolo_patterns WHERE ticker=?", (ticker,)).fetchall()
    return columns, rows


def _restore_yolo(conn: sqlite3.Connection, ticker: str, saved: tuple[list[str], list[tuple]]) -> None:
    columns, rows = saved
    if not columns:
        return
    conn.execute("DELETE FROM yolo_patterns WHERE ticker=?", (ticker,))
    marks = ",".join("?" * len(columns))
    conn.executemany(
        f"INSERT INTO yolo_patterns ({','.join(columns)}) VALUES ({marks})", rows
    )


def _fetch_with_retry(
    ticker: str,
    start: str,
    end: Optional[str],
    *,
    fetch_fn: FetchFn,
    actions_fn: ActionsFn,
    retries: int,
    backoff_sec: float,
    sleep_fn: Callable[[float], None],
) -> tuple[Optional[pd.DataFrame], str, list]:
    """Return (frame, source, ledger). frame is None for a clean empty response."""
    attempt = 0
    while True:
        try:
            frame, source = fetch_fn(ticker, start, end)
            return frame, source, actions_fn(ticker)
        except PriceFetchError as exc:
            if "Empty response" in str(exc):
                return None, "", []
            error = exc
        except Exception as exc:  # network, timeout, vendor errors: retry then report
            error = exc
        if attempt >= retries:
            raise error
        delay = backoff_sec * (2 ** attempt)
        LOG.warning("%s: fetch failed (%s); retry %d/%d in %.0fs", ticker, error, attempt + 1, retries, delay)
        sleep_fn(delay)
        attempt += 1


def backfill_ticker(
    conn: sqlite3.Connection,
    ticker: str,
    spells: list[Spell],
    *,
    fetch_fn: FetchFn,
    actions_fn: ActionsFn,
    probe_alive_fn: ProbeFn,
    floor: str = DEFAULT_FLOOR,
    warmup_days: int = DEFAULT_WARMUP_DAYS,
    retries: int = cfg.RETRIES,
    backoff_sec: float = cfg.BACKOFF_SEC,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict:
    started = time.monotonic()
    result: dict = {"ticker": ticker, "status": "", "rows_added": 0, "detail": ""}

    def done(status: str, detail: str = "") -> dict:
        result.update(status=status, detail=detail, elapsed_sec=round(time.monotonic() - started, 2))
        return result

    earliest = conn.execute(
        "SELECT MIN(date) FROM price_daily WHERE ticker=?", (ticker,)
    ).fetchone()[0]
    result["earliest_before"] = earliest
    same_company = ticker in cfg.SAME_COMPANY_READDS
    spells_used = usable_spells(spells, same_company=same_company)
    if len(spells_used) < len(spells):
        result["dropped_spells"] = [list(s) for s in spells[: len(spells) - len(spells_used)]]
    spells = spells_used
    window = plan_window(
        spells, earliest, floor=floor, warmup_days=warmup_days, same_company=True
    )
    if window is None:
        result["earliest_after"] = earliest
        return done("up_to_date")
    start, end = window
    result["window"] = [start, end]
    departed = all(s.end is not None for s in spells)

    try:
        frame, source, ledger = _fetch_with_retry(
            ticker, start, end, fetch_fn=fetch_fn, actions_fn=actions_fn,
            retries=retries, backoff_sec=backoff_sec, sleep_fn=sleep_fn,
        )
    except Exception as exc:
        return done("error", f"{type(exc).__name__}: {exc}"[:cfg.DETAIL_MAX_CHARS])

    if frame is not None and departed:
        dates = [d for d in frame["date"] if _in_a_spell(d, spells)]
        if not dates:
            frame = None
    if frame is None:
        if not departed:
            return done("error", "empty_window_for_current_member")
        if probe_alive_fn(ticker):
            return done("skipped_recycled_symbol", "symbol trades today but has no bars in its S&P 500 spell")
        return done("no_data", "yfinance serves no bars for this symbol")

    # A member must have bars when it joined. A series that starts years later is
    # another holder of the symbol (or a vendor gap), so fail closed and list it.
    must_start_by = _day(max(spells[0].start, floor)) + dt.timedelta(days=cfg.LATE_START_DAYS)
    if _day(str(frame["date"].min())) > must_start_by:
        return done(
            "skipped_history_starts_after_spell",
            f"first bar {frame['date'].min()} is after the spell start {spells[0].start}",
        )

    # The request already ended at the earliest stored date; keep that true even if a vendor ignores `end`.
    attrs = dict(frame.attrs)
    if earliest is not None:
        frame = frame[frame["date"] < earliest]
    frame = frame.reset_index(drop=True)
    frame.attrs.update(attrs)
    if frame.empty:
        return done("up_to_date")
    for key in ("corporate_actions", "unresolved_discontinuities"):
        frame.attrs[key] = [
            item for item in frame.attrs.get(key, [])
            if earliest is None or str(item.get("action_date") or item.get("date") or "") < earliest
        ]

    rows_before = conn.execute("SELECT COUNT(*) FROM price_daily WHERE ticker=?", (ticker,)).fetchone()[0]
    snapshot = _existing_snapshot(conn, ticker, earliest)
    saved_yolo = _save_yolo(conn, ticker)
    conn.execute("SAVEPOINT backfill")
    try:
        reconcile_vendor_action_ledger(frame, ledger, managed_start=start, managed_end=end)
        mark_full_history_actions_verified(frame)
        write_price_daily(conn, ticker, frame, data_source=source)
        _restore_yolo(conn, ticker, saved_yolo)
        if _existing_snapshot(conn, ticker, earliest) != snapshot:
            conn.execute("ROLLBACK TO backfill")
            conn.execute("RELEASE backfill")
            conn.commit()
            return done(
                "skipped_would_modify_existing_rows",
                "existing rows are not verified on the same basis, or the join looks discontinuous",
            )
        conn.execute("RELEASE backfill")
        conn.commit()
    except Exception as exc:
        conn.execute("ROLLBACK TO backfill")
        conn.execute("RELEASE backfill")
        conn.commit()
        return done("error", f"write failed: {type(exc).__name__}: {exc}"[:cfg.DETAIL_MAX_CHARS])

    rows_after = conn.execute("SELECT COUNT(*) FROM price_daily WHERE ticker=?", (ticker,)).fetchone()[0]
    result["rows_added"] = rows_after - rows_before
    result["earliest_after"] = conn.execute(
        "SELECT MIN(date) FROM price_daily WHERE ticker=?", (ticker,)
    ).fetchone()[0]
    result["actions_added"] = len(frame.attrs.get("corporate_actions", []))
    return done("ok")


def run_backfill(
    conn: sqlite3.Connection,
    membership: dict[str, list[Spell]],
    *,
    fetch_fn: FetchFn,
    actions_fn: ActionsFn,
    probe_alive_fn: ProbeFn,
    floor: str = DEFAULT_FLOOR,
    warmup_days: int = DEFAULT_WARMUP_DAYS,
    retries: int = cfg.RETRIES,
    backoff_sec: float = cfg.BACKOFF_SEC,
    pace_sec: float = cfg.PACE_SEC,
    sleep_fn: Callable[[float], None] = time.sleep,
    state_path: Optional[Path] = None,
    limit: Optional[int] = None,
    only: Optional[set[str]] = None,
    retry_terminal: bool = False,
) -> list[dict]:
    state = load_state(state_path)
    results: list[dict] = []
    attempted = 0
    for ticker, spells in membership.items():
        if only and ticker not in only:
            continue
        prior = state.get(ticker)
        if prior and prior["status"] in TERMINAL_STATUSES and not retry_terminal:
            continue
        if limit is not None and attempted >= limit:
            break
        attempted += 1
        out = backfill_ticker(
            conn, ticker, spells, fetch_fn=fetch_fn, actions_fn=actions_fn,
            probe_alive_fn=probe_alive_fn, floor=floor, warmup_days=warmup_days,
            retries=retries, backoff_sec=backoff_sec, sleep_fn=sleep_fn,
        )
        LOG.info(
            "%s status=%s rows_added=%s earliest=%s elapsed=%ss %s",
            ticker, out["status"], out["rows_added"], out.get("earliest_after"),
            out["elapsed_sec"], out["detail"],
        )
        results.append(out)
        state[ticker] = {k: out.get(k) for k in ("status", "rows_added", "earliest_after", "detail")}
        save_state(state_path, state)
        if out["status"] != "up_to_date" and pace_sec:
            sleep_fn(pace_sec)
    return results


