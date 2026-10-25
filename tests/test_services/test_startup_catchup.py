"""Startup catch-up after a container outage: pure decision + queueing."""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trader_koo.backend.services.pipeline import (
    STARTUP_CATCHUP_DELAY,
    ensure_pipeline_runs_schema,
    queue_startup_catchup,
    read_active_pipeline_run,
    reserve_pipeline_run,
    startup_catchup_reason,
    startup_work_queued,
)
from trader_koo.report.utils import expected_price_date

UTC = dt.timezone.utc


def _utc(y: int, m: int, d: int, h: int = 0, mi: int = 0) -> dt.datetime:
    return dt.datetime(y, m, d, h, mi, tzinfo=UTC)


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        # Sunday: the last completed session is Friday.
        (_utc(2026, 10, 25, 12), dt.date(2026, 10, 23)),
        # Wednesday 23:00Z is before the 4h ingest-due lag clears the close.
        (_utc(2026, 10, 21, 23), dt.date(2026, 10, 20)),
        # Thursday 01:00Z: Wednesday's 22:00Z ingest is now due.
        (_utc(2026, 10, 22, 1), dt.date(2026, 10, 21)),
        # Thanksgiving 2026-11-26: nothing new is due, Wednesday stays expected.
        (_utc(2026, 11, 26, 15), dt.date(2026, 11, 25)),
        # Naive input is read as UTC.
        (dt.datetime(2026, 10, 25, 12), dt.date(2026, 10, 23)),
    ],
)
def test_expected_price_date_follows_nyse_calendar(now, expected):
    assert expected_price_date(now) == expected


@pytest.mark.parametrize(
    ("case", "latest_price", "now", "already_queued", "should_queue"),
    [
        ("stale after outage", "2026-10-20", _utc(2026, 10, 23, 8, 44), False, True),
        ("current data", "2026-10-22", _utc(2026, 10, 23, 8, 44), False, False),
        ("weekend current", "2026-10-23", _utc(2026, 10, 25, 12), False, False),
        ("weekend stale", "2026-10-22", _utc(2026, 10, 25, 12), False, True),
        ("holiday current", "2026-11-25", _utc(2026, 11, 26, 15), False, False),
        ("day after holiday stale", "2026-11-24", _utc(2026, 11, 27, 15), False, True),
        ("cron slot not yet due", "2026-10-21", _utc(2026, 10, 22, 21), False, False),
        ("resume already queued", "2026-10-20", _utc(2026, 10, 23, 8, 44), True, False),
        ("no price data", None, _utc(2026, 10, 23, 8, 44), False, False),
    ],
)
def test_startup_catchup_reason(case, latest_price, now, already_queued, should_queue):
    reason = startup_catchup_reason(
        latest_price_date=latest_price,
        now_utc=now,
        already_queued=already_queued,
    )

    assert (reason is not None) is should_queue, case


def _db_with_price(tmp_path: Path, latest: str | None) -> Path:
    db = tmp_path / "t.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE price_daily (date TEXT)")
        if latest:
            conn.execute("INSERT INTO price_daily VALUES (?)", (latest,))
    return db


def test_queue_startup_catchup_schedules_delayed_full_run(tmp_path):
    db = _db_with_price(tmp_path, "2026-10-20")
    scheduler = MagicMock()
    now = _utc(2026, 10, 23, 8, 44)

    result = queue_startup_catchup(scheduler, db_path=db, now_utc=now)

    assert result["scheduled"] is True
    kwargs = scheduler.add_job.call_args.kwargs
    assert kwargs["trigger"] == "date"
    assert kwargs["run_date"] == now + STARTUP_CATCHUP_DELAY
    assert kwargs["kwargs"] == {"mode": "full", "source": "startup_catchup"}


def test_queue_startup_catchup_noop_when_current(tmp_path):
    db = _db_with_price(tmp_path, "2026-10-22")
    scheduler = MagicMock()

    result = queue_startup_catchup(
        scheduler, db_path=db, now_utc=_utc(2026, 10, 23, 8, 44)
    )

    assert result["scheduled"] is False
    scheduler.add_job.assert_not_called()


def test_queue_startup_catchup_skips_when_prior_block_queued(tmp_path):
    db = _db_with_price(tmp_path, "2026-10-20")
    scheduler = MagicMock()

    result = queue_startup_catchup(
        scheduler, db_path=db, now_utc=_utc(2026, 10, 23, 8, 44), already_queued=True
    )

    assert result["scheduled"] is False
    scheduler.add_job.assert_not_called()


def test_queue_startup_catchup_missing_price_table_fails_closed_and_logs(tmp_path, monkeypatch):
    db = tmp_path / "empty.db"
    sqlite3.connect(db).close()
    scheduler = MagicMock()
    logged: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "trader_koo.backend.services.pipeline._append_run_log",
        lambda tag, message, **kw: logged.append((tag, message)),
    )

    result = queue_startup_catchup(
        scheduler, db_path=db, now_utc=_utc(2026, 10, 23, 8, 44)
    )

    assert result["scheduled"] is False
    scheduler.add_job.assert_not_called()
    assert len(logged) == 1
    assert logged[0][0] == "CATCHUP"
    assert "skipped" in logged[0][1]


def test_queue_startup_catchup_no_price_rows_fails_closed_and_logs(tmp_path, monkeypatch):
    db = _db_with_price(tmp_path, None)
    scheduler = MagicMock()
    logged: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "trader_koo.backend.services.pipeline._append_run_log",
        lambda tag, message, **kw: logged.append((tag, message)),
    )

    result = queue_startup_catchup(
        scheduler, db_path=db, now_utc=_utc(2026, 10, 23, 8, 44)
    )

    assert result["scheduled"] is False
    scheduler.add_job.assert_not_called()
    assert len(logged) == 1
    assert logged[0][0] == "CATCHUP"
    assert "no price data" in logged[0][1]


def _insert_run(db: Path, run_id: str, started: str, status: str) -> None:
    with sqlite3.connect(db) as conn:
        ensure_pipeline_runs_schema(conn)
        conn.execute(
            "INSERT INTO pipeline_runs (run_id, started_ts, mode, source, status, stage)"
            " VALUES (?, ?, 'full', 'manual', ?, ?)",
            (run_id, started, status, status),
        )


def test_queue_startup_catchup_job_can_reserve_after_stale_rows(tmp_path):
    # An old queued row behind a newer running row survives startup recovery,
    # which only handles the newest row. The catch-up job reserves its own run,
    # so those leftovers must be cleared or the job logs SKIP and does nothing.
    db = _db_with_price(tmp_path, "2026-10-20")
    _insert_run(db, "old_queued", "2026-10-21T22:00:00Z", "queued")
    _insert_run(db, "new_running", "2026-10-22T22:00:00Z", "running")
    scheduler = MagicMock()

    result = queue_startup_catchup(
        scheduler, db_path=db, now_utc=_utc(2026, 10, 23, 8, 44)
    )

    assert result["scheduled"] is True
    active = reserve_pipeline_run(
        run_id="catchup_run", mode="full", source="startup_catchup", db_path=db
    )
    assert active is None


def test_queue_startup_catchup_leaves_finished_rows_alone(tmp_path):
    db = _db_with_price(tmp_path, "2026-10-20")
    _insert_run(db, "done", "2026-10-21T22:00:00Z", "ok")

    queue_startup_catchup(
        MagicMock(), db_path=db, now_utc=_utc(2026, 10, 23, 8, 44)
    )

    with sqlite3.connect(db) as conn:
        status = conn.execute(
            "SELECT status FROM pipeline_runs WHERE run_id = 'done'"
        ).fetchone()[0]
    assert status == "ok"


def test_queue_startup_catchup_keeps_active_rows_when_not_needed(tmp_path):
    db = _db_with_price(tmp_path, "2026-10-22")
    _insert_run(db, "queued_row", "2026-10-22T22:00:00Z", "queued")

    queue_startup_catchup(
        MagicMock(), db_path=db, now_utc=_utc(2026, 10, 23, 8, 44)
    )

    assert read_active_pipeline_run(db_path=db)["run_id"] == "queued_row"


def test_queue_startup_catchup_keeps_active_rows_when_prior_block_queued(tmp_path):
    db = _db_with_price(tmp_path, "2026-10-20")
    _insert_run(db, "requeued", "2026-10-22T22:00:00Z", "queued")

    queue_startup_catchup(
        MagicMock(),
        db_path=db,
        now_utc=_utc(2026, 10, 23, 8, 44),
        already_queued=True,
    )

    assert read_active_pipeline_run(db_path=db)["run_id"] == "requeued"


@pytest.mark.parametrize(
    ("case", "reserved", "interrupted_resume_queued", "post_ingest", "expected"),
    [
        ("nothing queued", {"scheduled": False}, False, {"scheduled": False}, False),
        # An interrupted row that needs no resume leaves no job behind.
        ("interrupted row without a queued resume", {}, False, {}, False),
        ("reserved requeued", {"scheduled": True}, False, {}, True),
        ("interrupted resume queued", {}, True, {}, True),
        ("post-ingest resume scheduled", {}, False, {"scheduled": True}, True),
    ],
)
def test_startup_work_queued(case, reserved, interrupted_resume_queued, post_ingest, expected):
    result = startup_work_queued(
        reserved=reserved,
        interrupted_resume_queued=interrupted_resume_queued,
        post_ingest=post_ingest,
    )

    assert result is expected, case
