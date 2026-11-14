"""One freshness verdict shared by the daily report and /api/status."""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from unittest.mock import patch

from tests.conftest import _create_test_db
from trader_koo.backend.routers import system
from trader_koo.backend.services.pipeline import invalidate_status_cache
from trader_koo.db.price_contract import record_price_series_revision
from trader_koo.notifications.market_monitor import ensure_polymarket_schema
from trader_koo.report.freshness import assess_data_freshness
from trader_koo.scripts.run_yolo_patterns import ensure_yolo_run_events_table, ensure_yolo_table

# Friday 17:00 New York, one hour after the close.
NOW = dt.datetime(2026, 10, 30, 21, 0, tzinfo=dt.timezone.utc)
LATEST_SESSION = "2026-10-30"


def _ts(at: dt.datetime) -> str:
    return at.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _db(
    *,
    price_date: str = LATEST_SESSION,
    fund_age_hours: float | None = 1.0,
    tickers: tuple[str, ...] = ("AAA",),
    yolo_tickers: tuple[str, ...] | None = None,
    sealed: bool = True,
    yolo_tables: bool = True,
    polymarket_age_minutes: float | None = 5.0,
) -> sqlite3.Connection:
    conn = _create_test_db()
    ensure_polymarket_schema(conn)
    if polymarket_age_minutes is not None:
        conn.execute(
            "INSERT INTO polymarket_snapshots "
            "(event_slug,event_title,market_question,probability,volume,snapshot_ts) "
            "VALUES ('fed','Fed?','Cut?',40.0,1000.0,?)",
            ((NOW - dt.timedelta(minutes=polymarket_age_minutes)).replace(microsecond=0).isoformat(),),
        )
    for ticker in tickers:
        conn.execute(
            "INSERT INTO price_daily (ticker,date,open,high,low,close,volume,data_source) "
            "VALUES (?,?,100,101,99,100,1000000,'test')",
            (ticker, price_date),
        )
        if sealed:
            record_price_series_revision(
                conn,
                ticker,
                evidence={"provider": "fixture", "vendor_action_ledger_checked": True},
                fetch_timestamp=_ts(NOW),
            )
    if fund_age_hours is not None:
        conn.execute(
            "INSERT INTO finviz_fundamentals (ticker,snapshot_ts,price) VALUES ('AAA',?,100)",
            (_ts(NOW - dt.timedelta(hours=fund_age_hours)),),
        )
    if yolo_tables:
        ensure_yolo_table(conn)
        ensure_yolo_run_events_table(conn)
        for ticker in tickers if yolo_tickers is None else yolo_tickers:
            conn.execute(
                "INSERT INTO yolo_patterns (ticker,timeframe,pattern,confidence,x0_date,x1_date,"
                "y0,y1,lookback_days,as_of_date,detected_ts) "
                "VALUES (?,'daily','double_bottom',0.9,?,?,99,101,120,?,?)",
                (ticker, price_date, price_date, price_date, _ts(NOW)),
            )
    conn.commit()
    return conn


def _status(conn: sqlite3.Connection, tmp_path: Path) -> dict:
    db_path = tmp_path / "status.db"
    db_path.touch()
    invalidate_status_cache()
    with (
        patch.object(system, "get_conn", return_value=conn),
        patch.object(system, "DB_PATH", db_path),
        patch.object(system, "set_cached_status"),
        patch.object(system, "resolve_published_report", return_value=None),
        patch.object(system, "is_report_fresh", return_value=True),
        patch.object(system, "pipeline_status_snapshot", return_value={"active": False}),
        patch.object(system, "post_ingest_resume_candidate", return_value=None),
    ):
        return system._compute_status(NOW)


def test_current_sources_are_fresh() -> None:
    verdict = assess_data_freshness(_db(), NOW)

    assert verdict.price_date == LATEST_SESSION
    assert verdict.price_age_days is not None and verdict.price_age_days < 1
    assert (verdict.price_missing, verdict.price_stale) == (False, False)
    assert (verdict.fund_missing, verdict.fund_stale) == (False, False)
    assert verdict.yolo_as_of_date == LATEST_SESSION
    assert verdict.yolo_coverage_ok is True
    assert (verdict.yolo_missing, verdict.yolo_stale) == (False, False)


def test_four_day_old_prices_are_stale() -> None:
    verdict = assess_data_freshness(_db(price_date="2026-10-26"), NOW)

    assert verdict.price_age_days is not None and verdict.price_age_days > 4
    assert verdict.price_stale is True
    assert verdict.price_missing is False


def test_fundamentals_boundary_is_48_hours() -> None:
    assert assess_data_freshness(_db(fund_age_hours=47), NOW).fund_stale is False
    assert assess_data_freshness(_db(fund_age_hours=49), NOW).fund_stale is True


def test_missing_sources_are_missing_not_stale() -> None:
    conn = _db(tickers=(), fund_age_hours=None, yolo_tables=False)
    verdict = assess_data_freshness(conn, NOW)

    assert (verdict.price_missing, verdict.price_stale) == (True, False)
    assert (verdict.fund_missing, verdict.fund_stale) == (True, False)
    assert (verdict.yolo_missing, verdict.yolo_stale) == (True, False)


def test_partial_yolo_coverage_is_stale_unless_ticker_is_excluded() -> None:
    conn = _db(tickers=("AAA", "BBB"), yolo_tickers=("AAA",))

    verdict = assess_data_freshness(conn, NOW)
    assert verdict.yolo_coverage_ok is False
    assert verdict.yolo_stale is True

    excused = assess_data_freshness(conn, NOW, yolo_excluded_tickers={"BBB"})
    assert excused.yolo_coverage_ok is True
    assert excused.yolo_stale is False


def test_ok_run_event_counts_as_yolo_coverage() -> None:
    conn = _db(tickers=("AAA", "BBB"), yolo_tickers=("AAA",))
    conn.execute(
        "INSERT INTO yolo_run_events (run_id,timeframe,ticker,status,as_of_date) "
        "VALUES ('r1','daily','BBB','ok',?)",
        (LATEST_SESSION,),
    )

    assert assess_data_freshness(conn, NOW).yolo_stale is False


def test_status_keeps_its_price_warning_string(tmp_path: Path) -> None:
    payload = _status(_db(price_date="2026-10-26"), tmp_path)

    assert "price_daily stale" in payload["operational_warnings"]
    assert payload["ok"] is False


def test_status_flags_missing_yolo_coverage(tmp_path: Path) -> None:
    payload = _status(_db(tickers=("AAA", "BBB"), yolo_tickers=("AAA",)), tmp_path)

    assert payload["operational_warnings"] == ["yolo_patterns stale"]
    assert payload["ok"] is False


def test_status_ignores_unsealed_tickers_for_yolo_coverage(tmp_path: Path) -> None:
    conn = _db(tickers=("AAA",))
    conn.execute(
        "INSERT INTO price_daily (ticker,date,open,high,low,close,volume,data_source) "
        "VALUES ('UNSEALED',?,100,101,99,100,1000000,'test')",
        (LATEST_SESSION,),
    )
    conn.commit()

    payload = _status(conn, tmp_path)

    assert "yolo_patterns stale" not in payload["operational_warnings"]


def test_status_flags_absent_yolo_table(tmp_path: Path) -> None:
    payload = _status(_db(yolo_tables=False), tmp_path)

    assert "yolo_patterns stale" in payload["operational_warnings"]


def test_status_with_fresh_sources_has_no_freshness_warnings(tmp_path: Path) -> None:
    payload = _status(_db(), tmp_path)

    assert not any(
        warning in payload["operational_warnings"]
        for warning in ("price_daily stale", "finviz_fundamentals stale", "yolo_patterns stale")
    )
    assert payload["freshness"]["price_age_days"] < 1
    assert payload["freshness"]["fund_age_hours"] == 1.0


def test_status_warns_when_polymarket_archive_is_older_than_30_minutes(tmp_path: Path) -> None:
    payload = _status(_db(polymarket_age_minutes=45), tmp_path)

    assert payload["operational_warnings"] == ["polymarket_snapshots stale"]
    assert "polymarket_snapshots stale" in payload["warnings"]
    assert payload["ok"] is False
    assert payload["freshness"]["polymarket_snapshot_age_minutes"] == 45.0
    assert payload["latest_data"]["polymarket_snapshot"] == "2026-10-30T20:15:00+00:00"


def test_status_warns_when_polymarket_archive_is_empty(tmp_path: Path) -> None:
    payload = _status(_db(polymarket_age_minutes=None), tmp_path)

    assert payload["operational_warnings"] == ["polymarket_snapshots stale"]
    assert payload["freshness"]["polymarket_snapshot_age_minutes"] is None


def test_status_accepts_a_recent_polymarket_archive(tmp_path: Path) -> None:
    payload = _status(_db(polymarket_age_minutes=25), tmp_path)

    assert "polymarket_snapshots stale" not in payload["operational_warnings"]
    assert payload["freshness"]["polymarket_snapshot_age_minutes"] == 25.0
