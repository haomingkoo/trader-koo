"""Daily option-metrics collection for the whole ingest universe.

Chains are fetched in memory and reduced to one ``options_daily_metrics`` row
per ticker per trading day (see ``options_metrics`` for definitions). Raw chains
are not stored: the production volume cannot hold them. The raw 40-ticker
rotation into ``options_iv`` is unchanged.
"""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

from trader_koo.collection.config import (
    HISTORY_PERIOD,
    RATE_SYMBOL,
    UNIVERSE_NEAREST_EXPIRIES,
    UNIVERSE_TARGET_DAYS,
    UNIVERSE_WINDOW_END_UTC,
    UNIVERSE_WINDOW_START_UTC,
)
from trader_koo.collection.options_metrics import Chain, compute_daily_metrics
from trader_koo.options_research import (
    _latest_close_from_history,
    normalize_options_tickers,
)

LOG = logging.getLogger(__name__)
_NY = ZoneInfo("America/New_York")
METRIC_COLUMNS = (
    "spot", "rate", "iv_30d", "iv_60d", "iv_90d", "put_skew_25d", "cp_iv_spread",
    "call_volume", "put_volume", "call_oi", "put_oi", "pc_volume_ratio", "pc_oi_ratio",
    "n_expiries", "n_strikes", "quality",
)


def ensure_options_daily_metrics_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS options_daily_metrics (
            asof_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            snapshot_ts TEXT NOT NULL,
            spot REAL,
            rate REAL,
            iv_30d REAL,
            iv_60d REAL,
            iv_90d REAL,
            put_skew_25d REAL,
            cp_iv_spread REAL,
            call_volume REAL,
            put_volume REAL,
            call_oi REAL,
            put_oi REAL,
            pc_volume_ratio REAL,
            pc_oi_ratio REAL,
            n_expiries INTEGER,
            n_strikes INTEGER,
            quality TEXT NOT NULL,
            source TEXT NOT NULL,
            PRIMARY KEY (asof_date, ticker)
        )
        """
    )


def select_metric_expiries(
    expiries: Iterable[str],
    today: dt.date,
    *,
    nearest: int = UNIVERSE_NEAREST_EXPIRIES,
    target_days: Iterable[int] = UNIVERSE_TARGET_DAYS,
) -> list[str]:
    """The ``nearest`` expiries plus the expiries bracketing each target maturity."""
    dated = sorted(
        (days, str(exp))
        for exp in expiries
        for days in [_days_until(exp, today)]
        if days >= 0  # an expiry already past is not an option to fetch
    )
    chosen = {exp for _, exp in dated[: max(0, int(nearest))]}
    for target in target_days:
        below = [e for d, e in dated if d <= target]
        above = [e for d, e in dated if d >= target]
        chosen.update(below[-1:] + above[:1])
    return sorted(chosen)


def _days_until(expiry: str, today: dt.date) -> int:
    """Calendar days to ``expiry``; an unparseable date raises, which records the ticker as an error."""
    return (dt.date.fromisoformat(str(expiry)) - today).days


@lru_cache(maxsize=1)
def _xnys():
    return xcals.get_calendar("XNYS")


def is_xnys_session(day: dt.date) -> bool:
    """True when ``day`` is an NYSE session (holidays and weekends are not).

    The calendar only covers a bounded range; outside it the library raises
    ValueError, which is passed on: the calendar package needs updating, and
    answering "not a session" would stop the job silently.
    """
    return bool(_xnys().is_session(day.isoformat()))


def universe_session_date(now: dt.datetime) -> dt.date | None:
    """US session date a run at ``now`` belongs to, or None outside the window.

    The window runs from 23:45 UTC on an XNYS-session evening to 03:00 UTC the
    next morning; that evening's UTC date is the session date.
    """
    now = now.astimezone(dt.timezone.utc)
    if now.time() >= UNIVERSE_WINDOW_START_UTC:
        anchor = now.date()
    elif now.time() < UNIVERSE_WINDOW_END_UTC:
        anchor = now.date() - dt.timedelta(days=1)
    else:
        return None
    return anchor if is_xnys_session(anchor) else None


def latest_session_date(now: dt.datetime) -> dt.date:
    """Most recent XNYS session that has already closed (early closes honoured)."""
    cal = _xnys()
    session = cal.date_to_session(now.astimezone(_NY).date().isoformat(), direction="previous")
    if cal.session_close(session).to_pydatetime() > now:
        session = cal.previous_session(session)
    return session.date()


def load_ingest_universe(conn: sqlite3.Connection, *, max_tickers: int | None = None) -> list[str]:
    """Tickers of the latest ingest run; empty (fail closed) when the ledger tables are missing.

    Any other database error propagates.
    """
    try:
        row = conn.execute("SELECT run_id FROM ingest_runs ORDER BY started_ts DESC LIMIT 1").fetchone()
        if not row or not row[0]:
            return []
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM ingest_ticker_status WHERE run_id = ? ORDER BY ticker",
            (str(row[0]),),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" not in str(exc):
            raise
        LOG.warning("ingest ledger missing (%s); the ingest universe is empty", exc)
        return []
    tickers = normalize_options_tickers([r[0] for r in rows if r[0]])
    return tickers[:max_tickers] if max_tickers else tickers


def fetch_rate(factory: Callable[[str], Any] | None = None) -> float | None:
    """13-week T-bill rate as a decimal from ^IRX, or None if unavailable."""
    if factory is None:
        import yfinance as yf

        factory = yf.Ticker
    close = _latest_close_from_history(factory(RATE_SYMBOL).history(period=HISTORY_PERIOD))
    return None if close is None else close / 100


def fetch_chains(tk: Any, day: dt.date, nearest: int, target_days: tuple[int, ...]) -> list[Chain]:
    chains: list[Chain] = []
    for exp in select_metric_expiries(getattr(tk, "options", None) or [], day, nearest=nearest, target_days=target_days):
        chain = tk.option_chain(exp)
        legs = [
            [] if frame is None else frame.to_dict("records")
            for frame in (getattr(chain, "calls", None), getattr(chain, "puts", None))
        ]
        chains.append((dt.date.fromisoformat(exp), legs[0], legs[1]))
    return chains


def write_metrics(
    conn: sqlite3.Connection, asof: dt.date, ticker: str, snapshot_ts: str, metrics: dict[str, Any]
) -> None:
    ensure_options_daily_metrics_schema(conn)
    cols = ("asof_date", "ticker", "snapshot_ts", *METRIC_COLUMNS, "source")
    conn.execute(
        f"INSERT OR REPLACE INTO options_daily_metrics ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' for _ in cols)})",
        (asof.isoformat(), ticker, snapshot_ts, *(metrics.get(c) for c in METRIC_COLUMNS), "yfinance"),
    )


def snapshot_universe(
    conn: sqlite3.Connection,
    tickers: Iterable[str],
    *,
    snapshot_ts: str | None = None,
    nearest: int = UNIVERSE_NEAREST_EXPIRIES,
    target_days: tuple[int, ...] = UNIVERSE_TARGET_DAYS,
    force: bool = False,
    sleep_sec: float = 0.0,
    workers: int = 1,
    should_stop: Callable[[], bool] | None = None,
    ticker_factory: Callable[[str], Any] | None = None,
    rate: float | None = None,
    today: dt.date | None = None,
) -> dict[str, Any]:
    """Compute and store one metrics row per ticker for ``today`` (US/Eastern date).

    Fetches run on up to ``workers`` threads; all database access stays on the
    caller. ``should_stop`` is polled before each batch. A ticker whose spot or
    chain cannot be fetched is recorded in ``errors`` and gets no row.
    """
    ensure_options_daily_metrics_schema(conn)
    snap = snapshot_ts or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    day = today or dt.datetime.now(_NY).date()
    selected = normalize_options_tickers(tickers)
    summary: dict[str, Any] = {
        "asof_date": day.isoformat(),
        "tickers_total": len(selected),
        "rows_written": 0,
        "skipped_existing": 0,
        "quality": {},
        "rate": rate,
        "stopped_early": False,
        "errors": {},
    }
    factory = ticker_factory
    if factory is None:
        import yfinance as yf

        factory = yf.Ticker

    def work(symbol: str) -> dict[str, Any]:
        tk = factory(symbol)
        spot = _latest_close_from_history(tk.history(period=HISTORY_PERIOD))
        chains = fetch_chains(tk, day, nearest, target_days)
        return compute_daily_metrics(spot, rate, chains, day, target_days)

    batch_size = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        for start in range(0, len(selected), batch_size):
            if should_stop is not None and should_stop():
                summary["stopped_early"] = True
                break
            due = []
            for ticker in selected[start : start + batch_size]:
                exists = conn.execute(
                    "SELECT 1 FROM options_daily_metrics WHERE asof_date = ? AND ticker = ?",
                    (day.isoformat(), ticker),
                ).fetchone()
                if exists and not force:
                    summary["skipped_existing"] += 1
                else:
                    due.append(ticker)
            futures = [(t, pool.submit(work, t)) for t in due]
            for ticker, future in futures:
                try:
                    metrics = future.result()
                    write_metrics(conn, day, ticker, snap, metrics)
                    conn.commit()
                except Exception as exc:  # recorded per ticker, surfaced in the summary
                    conn.rollback()
                    summary["errors"][ticker] = str(exc)
                    continue
                summary["rows_written"] += 1
                q = metrics["quality"]
                summary["quality"][q] = summary["quality"].get(q, 0) + 1
            if sleep_sec > 0 and due:
                time.sleep(float(sleep_sec))
    summary["not_attempted"] = (
        summary["tickers_total"] - summary["rows_written"] - summary["skipped_existing"] - len(summary["errors"])
    )
    return summary
