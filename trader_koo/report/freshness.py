"""Data freshness verdict shared by the daily report and ``/api/status``.

Each caller maps the verdict to its own warning strings; the thresholds and
the YOLO coverage rule live only here.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass

from trader_koo.report.utils import days_since_date, hours_since, table_exists

PRICE_STALE_AFTER_DAYS = 3.0
FUNDAMENTALS_STALE_AFTER_HOURS = 48.0
# Share of tickers holding the latest price bar that daily YOLO must have processed.
YOLO_MIN_DAILY_COVERAGE = 0.90
ISO_DATE_LEN = len("YYYY-MM-DD")


@dataclass(frozen=True)
class FreshnessVerdict:
    price_date: str | None
    price_age_days: float | None
    price_missing: bool
    price_stale: bool
    fund_snapshot: str | None
    fund_age_hours: float | None
    fund_missing: bool
    fund_stale: bool
    yolo_as_of_date: str | None
    # None when there is no price date to measure coverage against.
    yolo_coverage_ok: bool | None
    yolo_missing: bool
    yolo_stale: bool


def assess_data_freshness(
    conn: sqlite3.Connection,
    now: dt.datetime,
    *,
    yolo_excluded_tickers: Iterable[str] = (),
) -> FreshnessVerdict:
    """Judge price, fundamentals and daily YOLO freshness as of *now*.

    ``yolo_excluded_tickers`` are tickers YOLO legitimately does not process
    (not research eligible); they are left out of the coverage denominator.
    """
    row = conn.execute(
        "SELECT (SELECT MAX(date) FROM price_daily), (SELECT MAX(snapshot_ts) FROM finviz_fundamentals)"
    ).fetchone()
    price_date = str(row[0])[:ISO_DATE_LEN] if row and row[0] else None
    fund_snapshot = str(row[1]) if row and row[1] else None
    price_age = days_since_date(price_date, now)
    fund_age = hours_since(fund_snapshot, now)

    yolo_table = table_exists(conn, "yolo_patterns")
    yolo_as_of = None
    if yolo_table:
        yolo_row = conn.execute(
            "SELECT MAX(as_of_date) FROM yolo_patterns WHERE timeframe = 'daily'"
        ).fetchone()
        yolo_as_of = str(yolo_row[0])[:ISO_DATE_LEN] if yolo_row and yolo_row[0] else None
    coverage_ok: bool | None = None
    if price_date:
        coverage_ok = yolo_table and _yolo_daily_coverage_ok(
            conn, price_date, set(yolo_excluded_tickers)
        )

    # YOLO only writes rows for tickers with a new candle, so detected_ts age
    # grows over weekends. Current means daily patterns reach the latest bar
    # date and cover most of the tickers that have that bar.
    return FreshnessVerdict(
        price_date=price_date,
        price_age_days=price_age,
        price_missing=price_age is None,
        price_stale=price_age is not None and price_age > PRICE_STALE_AFTER_DAYS,
        fund_snapshot=fund_snapshot,
        fund_age_hours=fund_age,
        fund_missing=fund_age is None,
        fund_stale=fund_age is not None and fund_age > FUNDAMENTALS_STALE_AFTER_HOURS,
        yolo_as_of_date=yolo_as_of,
        yolo_coverage_ok=coverage_ok,
        yolo_missing=yolo_as_of is None,
        yolo_stale=(
            yolo_as_of is not None
            and price_date is not None
            and (yolo_as_of < price_date or not coverage_ok)
        ),
    )


def unsealed_price_tickers(conn: sqlite3.Connection) -> set[str]:
    """Tickers without a persisted verified revision seal, which YOLO skips.

    Reads only the persisted seal status, never re-hashing price history, so
    ``/api/status`` stays cheap.
    """
    if not table_exists(conn, "price_series_revisions"):
        return {str(r[0]) for r in conn.execute("SELECT DISTINCT ticker FROM price_daily")}
    return {
        str(r[0])
        for r in conn.execute(
            """SELECT p.ticker
                 FROM (SELECT DISTINCT ticker FROM price_daily) p
                 LEFT JOIN price_series_revisions r
                   ON r.ticker = p.ticker AND r.status = 'verified'
                WHERE r.ticker IS NULL"""
        )
    }


def _yolo_daily_coverage_ok(
    conn: sqlite3.Connection, price_date: str, excluded: set[str]
) -> bool:
    """True when daily YOLO processed enough tickers that hold the latest price bar.

    A ticker counts as processed when it has a daily pattern row or an ok/skipped
    run event for that as-of date. Zero-detection tickers only leave an event.
    """
    expected = {
        str(r[0])
        for r in conn.execute(
            "SELECT ticker FROM price_daily GROUP BY ticker HAVING MAX(date) = ?",
            (price_date,),
        )
    } - excluded
    if not expected:
        return True
    covered = {
        str(r[0])
        for r in conn.execute(
            "SELECT ticker FROM yolo_patterns WHERE timeframe = 'daily' AND as_of_date = ?",
            (price_date,),
        )
    }
    try:
        covered |= {
            str(r[0])
            for r in conn.execute(
                "SELECT ticker FROM yolo_run_events WHERE timeframe = 'daily' "
                "AND as_of_date = ? AND status IN ('ok', 'skipped')",
                (price_date,),
            )
        }
    except sqlite3.OperationalError:
        pass  # events table absent: pattern rows are the only coverage evidence
    return len(expected & covered) / len(expected) >= YOLO_MIN_DAILY_COVERAGE
