from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from trader_koo.backend.routers import system
from trader_koo.backend.services.pipeline import invalidate_status_cache
from trader_koo.llm_health import llm_health_summary_readonly


def _status_dependencies():
    return (
        patch.object(system, "resolve_published_report", return_value=None),
        patch.object(system, "pipeline_status_snapshot", return_value={}),
        patch.object(system, "post_ingest_resume_candidate", return_value=None),
        patch.object(system, "llm_status_readonly", return_value={"enabled": False}),
    )


def _seed_ingest_cohort(
    conn: sqlite3.Connection,
    tickers: list[str],
    *,
    run_id: str = "current",
    started_ts: str = "2026-08-25T22:00:00Z",
    use_sp500: bool = True,
) -> None:
    conn.execute(
        """INSERT INTO ingest_runs
           (run_id,started_ts,finished_ts,status,tickers_total,tickers_ok,tickers_failed,args_json)
           VALUES (?,?,?,'ok',?,?,0,?)""",
        (
            run_id,
            started_ts,
            started_ts,
            len(tickers),
            len(tickers),
            json.dumps({"use_sp500": use_sp500, "skip_price": False}),
        ),
    )
    conn.executemany(
        "INSERT INTO ingest_ticker_status (run_id,ticker,status) VALUES (?,?,'ok')",
        [(run_id, ticker) for ticker in tickers],
    )
    conn.commit()


def test_status_never_revalidates_full_price_history(seeded_conn, tmp_path: Path) -> None:
    _seed_ingest_cohort(seeded_conn, ["SPY"])
    statements: list[str] = []
    seeded_conn.set_trace_callback(statements.append)
    seeded_conn.execute("PRAGMA query_only=ON")
    invalidate_status_cache()
    db_path = tmp_path / "status.db"
    db_path.touch()

    dependencies = _status_dependencies()
    with (
        patch.object(system, "get_conn", return_value=seeded_conn),
        patch.object(system, "DB_PATH", db_path),
        patch("trader_koo.db.price_contract._series_material", side_effect=AssertionError("deep hash")),
        dependencies[0],
        dependencies[1],
        dependencies[2],
        dependencies[3],
    ):
        payload = system.status()

    normalized = [" ".join(statement.lower().split()) for statement in statements]
    assert payload["price_basis"]["verified_tickers"] == 1
    assert payload["price_basis"]["verification_mode"] == "persisted_revision_identity_join"
    assert not any("from price_daily where ticker=" in statement for statement in normalized)
    assert not any(
        "from price_daily group by adjustment_basis" in statement
        for statement in normalized
    )


def test_status_caches_at_completion_time(seeded_conn, tmp_path: Path) -> None:
    started = dt.datetime(2026, 8, 26, 0, 0, tzinfo=dt.timezone.utc)
    completed = started + dt.timedelta(seconds=25)

    class FakeDateTime(dt.datetime):
        values = iter((started, completed))

        @classmethod
        def now(cls, tz=None):
            return next(cls.values)

    clock = SimpleNamespace(datetime=FakeDateTime, timedelta=dt.timedelta, timezone=dt.timezone)
    db_path = tmp_path / "status.db"
    db_path.touch()
    dependencies = _status_dependencies()
    with (
        patch.object(system, "dt", clock),
        patch.object(system, "get_conn", return_value=seeded_conn),
        patch.object(system, "DB_PATH", db_path),
        patch.object(system, "get_cached_status", return_value=None),
        patch.object(system, "set_cached_status") as cache,
        dependencies[0],
        dependencies[1],
        dependencies[2],
        dependencies[3],
    ):
        system.status()

    assert cache.call_args.args[0] == completed


def test_retained_historical_symbol_does_not_inflate_current_cohort(
    seeded_conn,
    tmp_path: Path,
) -> None:
    _seed_ingest_cohort(seeded_conn, ["SPY"])
    seeded_conn.execute(
        """
        INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
        VALUES ('UNRELATED', '2026-08-25', 10, 11, 9, 10.5, 1000)
        """
    )
    seeded_conn.commit()
    db_path = tmp_path / "status.db"
    db_path.touch()
    invalidate_status_cache()
    report = {"generated_ts": "2026-08-26T00:00:00Z"}

    with (
        patch.object(system, "get_conn", return_value=seeded_conn),
        patch.object(system, "DB_PATH", db_path),
        patch.object(system, "get_cached_status", return_value=None),
        patch.object(system, "set_cached_status"),
        patch.object(system, "days_since", return_value=0.0),
        patch.object(system, "hours_since", return_value=0.0),
        patch.object(system, "resolve_published_report", return_value=(tmp_path, report)),
        patch.object(system, "is_report_fresh", return_value=True),
        patch.object(
            system,
            "pipeline_status_snapshot",
            return_value={"active": False, "stage": "idle"},
        ),
        patch.object(system, "post_ingest_resume_candidate", return_value=None),
        patch.object(system, "llm_status_readonly", return_value={"enabled": False}),
    ):
        payload = system.status()

    assert payload["ok"] is True
    assert payload["research_ready"] is True
    assert payload["operational_warnings"] == []
    assert payload["research_warnings"] == []
    assert payload["operational_warning_count"] == 0
    assert payload["research_warning_count"] == 0
    assert payload["price_basis"]["cohort_tickers"] == 1
    assert payload["price_basis"]["unresolved_tickers"] == 0
    assert payload["price_basis"]["missing_revision_tickers"] == 0
    assert payload["price_basis"]["retained_history"]["unresolved_tickers"] == 1
    assert payload["price_basis"]["retained_history"]["missing_revision_tickers"] == 1


def test_current_cohort_unresolved_symbol_keeps_research_fail_closed(
    seeded_conn,
    tmp_path: Path,
) -> None:
    seeded_conn.execute(
        """INSERT INTO price_daily (ticker,date,open,high,low,close,volume)
           VALUES ('CURRENT','2026-08-25',10,11,9,10.5,1000)"""
    )
    _seed_ingest_cohort(seeded_conn, ["SPY", "CURRENT"])
    db_path = tmp_path / "status.db"
    db_path.touch()
    invalidate_status_cache()
    report = {"generated_ts": "2026-08-26T00:00:00Z"}

    with (
        patch.object(system, "get_conn", return_value=seeded_conn),
        patch.object(system, "DB_PATH", db_path),
        patch.object(system, "get_cached_status", return_value=None),
        patch.object(system, "set_cached_status"),
        patch.object(system, "days_since", return_value=0.0),
        patch.object(system, "hours_since", return_value=0.0),
        patch.object(system, "resolve_published_report", return_value=(tmp_path, report)),
        patch.object(system, "is_report_fresh", return_value=True),
        patch.object(system, "pipeline_status_snapshot", return_value={"active": False}),
        patch.object(system, "post_ingest_resume_candidate", return_value=None),
        patch.object(system, "llm_status_readonly", return_value={"enabled": False}),
    ):
        payload = system.status()

    assert payload["research_ready"] is False
    assert payload["research_warnings"] == ["price basis unresolved"]
    assert payload["price_basis"]["cohort_tickers"] == 2
    assert payload["price_basis"]["verified_tickers"] == 1
    assert payload["price_basis"]["unresolved_tickers"] == 1
    assert payload["price_basis"]["missing_revision_tickers"] == 1


def test_newer_targeted_backfill_cannot_replace_canonical_price_cohort(
    seeded_conn,
) -> None:
    seeded_conn.execute(
        """INSERT INTO price_daily (ticker,date,open,high,low,close,volume)
           VALUES ('BAD','2026-08-25',10,11,9,10.5,1000)"""
    )
    _seed_ingest_cohort(
        seeded_conn,
        ["SPY", "BAD"],
        run_id="canonical",
        started_ts="2026-08-25T22:00:00Z",
    )
    _seed_ingest_cohort(
        seeded_conn,
        ["SPY"],
        run_id="targeted",
        started_ts="2026-08-26T01:00:00Z",
        use_sp500=False,
    )

    summary = system._persisted_price_revision_summary(seeded_conn, tracked_tickers=2)

    assert summary["cohort_run_id"] == "canonical"
    assert summary["cohort_tickers"] == 2
    assert summary["unresolved_tickers"] == 1


def test_orphan_revision_seal_does_not_inflate_retained_history(seeded_conn) -> None:
    seeded_conn.execute(
        """INSERT INTO price_series_revisions
           (ticker,managed_start,managed_end,row_count,adjustment_basis,
            adjustment_version,price_sha256,action_sha256,evidence_sha256,
            revision_sha256,status,evidence_json,fetch_timestamp)
           VALUES ('ORPHAN','2026-08-25','2026-08-25',1,'split_adjusted','v1',
                   'p','a','e','r','verified','{}','2026-08-25T22:00:00Z')"""
    )
    _seed_ingest_cohort(seeded_conn, ["SPY"])

    summary = system._persisted_price_revision_summary(seeded_conn, tracked_tickers=1)

    retained = summary["retained_history"]
    assert retained["ticker_count"] == 1
    assert retained["verified_tickers"] == 1
    assert retained["revision_tickers"] == 1
    assert retained["missing_revision_tickers"] == 0


def test_missing_successful_ingest_cohort_is_explicitly_unavailable(
    seeded_conn,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "status.db"
    db_path.touch()
    invalidate_status_cache()
    report = {"generated_ts": "2026-08-26T00:00:00Z"}

    with (
        patch.object(system, "get_conn", return_value=seeded_conn),
        patch.object(system, "DB_PATH", db_path),
        patch.object(system, "get_cached_status", return_value=None),
        patch.object(system, "set_cached_status"),
        patch.object(system, "days_since", return_value=0.0),
        patch.object(system, "hours_since", return_value=0.0),
        patch.object(system, "resolve_published_report", return_value=(tmp_path, report)),
        patch.object(system, "is_report_fresh", return_value=True),
        patch.object(system, "pipeline_status_snapshot", return_value={"active": False}),
        patch.object(system, "post_ingest_resume_candidate", return_value=None),
        patch.object(system, "llm_status_readonly", return_value={"enabled": False}),
    ):
        payload = system.status()

    assert payload["research_ready"] is False
    assert payload["research_warnings"] == ["current price cohort unavailable"]
    assert payload["price_basis"]["cohort_available"] is False
    assert payload["price_basis"]["cohort_tickers"] == 0
    assert payload["price_basis"]["retained_history"]["ticker_count"] == 1


def test_operational_failure_does_not_change_research_readiness(
    seeded_conn,
    tmp_path: Path,
) -> None:
    _seed_ingest_cohort(seeded_conn, ["SPY"])
    db_path = tmp_path / "status.db"
    db_path.touch()
    invalidate_status_cache()
    report = {"generated_ts": "2026-08-26T00:00:00Z"}

    with (
        patch.object(system, "get_conn", return_value=seeded_conn),
        patch.object(system, "DB_PATH", db_path),
        patch.object(system, "get_cached_status", return_value=None),
        patch.object(system, "set_cached_status"),
        patch.object(system, "days_since", return_value=0.0),
        patch.object(system, "hours_since", return_value=0.0),
        patch.object(system, "resolve_published_report", return_value=(tmp_path, report)),
        patch.object(system, "is_report_fresh", return_value=False),
        patch.object(
            system,
            "pipeline_status_snapshot",
            return_value={"active": False, "stage": "idle"},
        ),
        patch.object(system, "post_ingest_resume_candidate", return_value=None),
        patch.object(system, "llm_status_readonly", return_value={"enabled": False}),
    ):
        payload = system.status()

    assert payload["ok"] is False
    assert payload["research_ready"] is True
    assert payload["operational_warnings"] == ["daily_report stale"]
    assert payload["research_warnings"] == []
    assert payload["operational_warning_count"] == 1
    assert payload["research_warning_count"] == 0


def test_persisted_revision_summary_reads_only_price_identities(seeded_conn) -> None:
    _seed_ingest_cohort(seeded_conn, ["SPY"])
    statements: list[str] = []
    seeded_conn.set_trace_callback(statements.append)

    summary = system._persisted_price_revision_summary(seeded_conn, tracked_tickers=2)

    assert summary["verified_tickers"] == 1
    assert summary["cohort_available"] is True
    assert summary["cohort_tickers"] == 1
    assert summary["unresolved_tickers"] == 0
    assert summary["missing_revision_tickers"] == 0
    assert summary["retained_history"]["unresolved_tickers"] == 1
    assert summary["retained_history"]["missing_revision_tickers"] == 1
    price_queries = [statement.lower() for statement in statements if "price_daily" in statement.lower()]
    assert len(price_queries) == 1
    assert "select distinct ticker from price_daily" in price_queries[0]
    assert all(column not in price_queries[0] for column in (" open", " close", " high", " low", " volume"))


def test_llm_health_readonly_does_not_initialize_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "health.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE llm_health_events (
            id INTEGER PRIMARY KEY,
            event_ts TEXT NOT NULL,
            outcome TEXT NOT NULL,
            source TEXT,
            ticker TEXT,
            reason TEXT,
            error_class TEXT,
            details TEXT
        );
        CREATE TABLE llm_health_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_ts TEXT NOT NULL
        );
        INSERT INTO llm_health_events (event_ts, outcome)
        VALUES ('2026-08-26T00:00:00Z', 'success');
        """
    )
    conn.commit()
    conn.close()
    before = db_path.read_bytes()

    with patch(
        "trader_koo.llm_health.ensure_llm_health_schema",
        side_effect=AssertionError("DDL attempted"),
    ):
        summary = llm_health_summary_readonly(db_path, recent_limit=1)

    assert summary["counts"]["success"] == 1
    assert db_path.read_bytes() == before
