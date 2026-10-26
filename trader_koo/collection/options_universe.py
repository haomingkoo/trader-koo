"""Daily option-metrics collection for the whole ingest universe.

Chains are fetched in memory and reduced to one ``options_daily_metrics`` row
per ticker per trading day (see ``options_metrics`` for definitions). Raw chains
are not stored: the production volume cannot hold them. The raw 40-ticker
rotation into ``options_iv`` is unchanged.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

from trader_koo.collection.config import (
    RATE_SYMBOL,
    UNIVERSE_CLOSE_HOUR_ET,
    UNIVERSE_NEAREST_EXPIRIES,
    UNIVERSE_TARGET_DAYS,
    UNIVERSE_WINDOW_END_UTC,
    UNIVERSE_WINDOW_START_UTC,
    WEEKEND_DAYS,
)
from trader_koo.collection.options_metrics import Chain, compute_daily_metrics
from trader_koo.options_research import (
    _latest_close_from_history,
    normalize_options_tickers,
)

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
        if days is not None and days >= 0
    )
    chosen = {exp for _, exp in dated[: max(0, int(nearest))]}
    for target in target_days:
        below = [e for d, e in dated if d <= target]
        above = [e for d, e in dated if d >= target]
        chosen.update(below[-1:] + above[:1])
    return sorted(chosen)


def _days_until(expiry: str, today: dt.date) -> int | None:
    try:
        return (dt.date.fromisoformat(str(expiry)) - today).days
    except ValueError:
        return None


def universe_session_date(now: dt.datetime) -> dt.date | None:
    """US session date a run at ``now`` belongs to, or None outside the window.

    The window runs from 23:45 UTC on a Mon-Fri evening to 03:00 UTC the next
    morning; that evening's UTC date is the session date.
    """
    now = now.astimezone(dt.timezone.utc)
    if now.time() >= UNIVERSE_WINDOW_START_UTC:
        anchor = now.date()
    elif now.time() < UNIVERSE_WINDOW_END_UTC:
        anchor = now.date() - dt.timedelta(days=1)
    else:
        return None
    return None if anchor.weekday() in WEEKEND_DAYS else anchor


def latest_session_date(now: dt.datetime) -> dt.date:
    """Most recent weekday whose US close (16:00 ET) has passed; for manual runs."""
    et = now.astimezone(_NY)
    day = et.date() if et.hour >= UNIVERSE_CLOSE_HOUR_ET else et.date() - dt.timedelta(days=1)
    while day.weekday() in WEEKEND_DAYS:
        day -= dt.timedelta(days=1)
    return day


def load_ingest_universe(conn: sqlite3.Connection, *, max_tickers: int | None = None) -> list[str]:
    """Tickers of the latest ingest run; empty (fail closed) when the ledger is missing."""
    try:
        row = conn.execute("SELECT run_id FROM ingest_runs ORDER BY started_ts DESC LIMIT 1").fetchone()
        if not row or not row[0]:
            return []
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM ingest_ticker_status WHERE run_id = ? ORDER BY ticker",
            (str(row[0]),),
        ).fetchall()
    except sqlite3.Error:
        return []
    tickers = normalize_options_tickers([r[0] for r in rows if r[0]])
    return tickers[:max_tickers] if max_tickers else tickers


def fetch_rate(factory: Callable[[str], Any] | None = None) -> float | None:
    """13-week T-bill rate as a decimal from ^IRX, or None if unavailable."""
    if factory is None:
        import yfinance as yf

        factory = yf.Ticker
    close = _latest_close_from_history(factory(RATE_SYMBOL).history(period="5d"))
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
        spot = _latest_close_from_history(tk.history(period="5d"))
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
    return summary
