"""reconcile_stale_running_runs(): which orphaned ingest runs fail, and what they record.

The clock is frozen through the module's ``dt`` so stale-limit boundaries are exact.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from trader_koo.backend.services import pipeline

NOW = dt.datetime(2026, 11, 2, 12, 0, 0, 654321, tzinfo=dt.timezone.utc)
LIMIT_SEC = pipeline.INGEST_RUNNING_STALE_MIN * 60


class _FrozenDateTime(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


def _ts(seconds_ago: float) -> str:
    # Microseconds kept: NOW has them, so a whole-second stamp would add 0.65 s of age.
    return (NOW - dt.timedelta(seconds=seconds_ago)).isoformat().replace("+00:00", "Z")


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    db = tmp_path / "trader_koo.db"
    with sqlite3.connect(db) as conn:
        # No primary key, as in the oldest deployed schema, so a duplicated run id is possible.
        conn.execute(
            "CREATE TABLE ingest_runs (run_id TEXT, started_ts TEXT, finished_ts TEXT, status TEXT, "
            "tickers_total INTEGER, tickers_ok INTEGER, tickers_failed INTEGER, error_message TEXT)"
        )
    log: list[tuple[str, str]] = []
    invalidated: list[bool] = []
    monkeypatch.setattr(pipeline, "DB_PATH", db)
    monkeypatch.setattr(pipeline, "dt", SimpleNamespace(datetime=_FrozenDateTime, timezone=dt.timezone))
    monkeypatch.setattr(pipeline, "_append_run_log", lambda tag, msg: log.append((tag, msg)))
    monkeypatch.setattr(pipeline, "invalidate_status_cache", lambda: invalidated.append(True))
    return SimpleNamespace(db=db, log=log, invalidated=invalidated)


def _insert(db: Path, *rows: tuple) -> None:
    with sqlite3.connect(db) as conn:
        conn.executemany(
            "INSERT INTO ingest_runs (run_id, started_ts, status, tickers_total, tickers_ok, tickers_failed, "
            "error_message) VALUES (?,?,?,?,?,?,?)",
            rows,
        )


def _ticker_status(db: Path, *rows: tuple[str, str, str]) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE ingest_ticker_status (run_id TEXT, ticker TEXT, status TEXT)")
        conn.executemany("INSERT INTO ingest_ticker_status VALUES (?,?,?)", rows)


def _rows(db: Path) -> dict[str, tuple]:
    with sqlite3.connect(db) as conn:
        return {
            r[0]: r[1:]
            for r in conn.execute("SELECT run_id, status, tickers_ok, tickers_failed, finished_ts, error_message "
                                  "FROM ingest_runs")
        }


def test_runs_fail_only_past_the_stale_limit(env) -> None:
    _insert(
        env.db,
        ("at_limit", _ts(LIMIT_SEC), "running", 1, 0, 0, None),
        ("past_limit", _ts(LIMIT_SEC + 1), "running", 1, 0, 0, None),
    )

    result = pipeline.reconcile_stale_running_runs()

    assert result == {"checked": 2, "reconciled": 1, "run_ids": ["past_limit"]}
    assert _rows(env.db)["at_limit"][0] == "running"


def test_failed_run_records_the_time_and_reason(env) -> None:
    age_sec = LIMIT_SEC + 15 * 60 + 30  # 15.5 minutes past the limit
    _insert(
        env.db,
        ("silent", _ts(age_sec), "running", 4, 1, 0, ""),
        ("explained", _ts(age_sec), "running", 4, 1, 0, "disk full"),
    )

    pipeline.reconcile_stale_running_runs()

    age_min = f"{age_sec / 60:.1f}"
    rows = _rows(env.db)
    assert rows["silent"] == (
        "failed", 1, 3, "2026-11-02T12:00:00Z",
        f"run_orphaned_after_restart: auto-failed after stale running "
        f"({age_min}m > {pipeline.INGEST_RUNNING_STALE_MIN}m)",
    )
    assert rows["explained"][4] == "disk full"
    assert env.log == [
        ("RECOVER", f"run_id=silent stale-running auto-failed age_min={age_min} total=4 processed=1"),
        ("RECOVER", f"run_id=explained stale-running auto-failed age_min={age_min} total=4 processed=1"),
    ]
    assert env.invalidated == [True]


def test_unparsable_start_counts_as_stale_and_does_not_stop_the_pass(env) -> None:
    # ORDER BY started_ts puts the digit-led fresh run first and the garbage one last.
    _insert(
        env.db,
        ("fresh", _ts(60), "running", 1, 0, 0, None),
        ("garbage", "not-a-timestamp", "running", 1, 0, 0, None),
    )

    result = pipeline.reconcile_stale_running_runs()

    assert result == {"checked": 2, "reconciled": 1, "run_ids": ["garbage"]}


def test_run_already_failed_earlier_in_the_pass_is_not_counted_twice(env) -> None:
    old = _ts(LIMIT_SEC + 600)
    _insert(
        env.db,
        ("dup", old, "running", 1, 0, 0, None),
        ("dup", old, "running", 1, 0, 0, None),
        ("other", _ts(LIMIT_SEC + 300), "running", 1, 0, 0, None),
    )

    result = pipeline.reconcile_stale_running_runs()

    assert result == {"checked": 3, "reconciled": 2, "run_ids": ["dup", "other"]}


def test_nothing_stale_leaves_the_status_cache_alone(env) -> None:
    _insert(env.db, ("fresh", _ts(60), "running", 1, 0, 0, None))

    assert pipeline.reconcile_stale_running_runs() == {"checked": 1, "reconciled": 0, "run_ids": []}
    assert env.invalidated == []
    assert env.log == []


@pytest.mark.parametrize(
    ("total", "ok", "failed", "expected", "processed"),
    [
        (10, 4, 2, (4, 6), 6),  # unresolved tickers count as failed
        (10, None, None, (0, 10), 0),
        (None, 1, 3, (1, 3), 4),  # unknown total: only the recorded failures
        (None, None, 3, (0, 3), 3),
        (None, 2, None, (2, 0), 2),
    ],
)
def test_counts_without_per_ticker_status(env, total, ok, failed, expected, processed) -> None:
    _insert(env.db, ("r", _ts(LIMIT_SEC + 60), "running", total, ok, failed, None))

    pipeline.reconcile_stale_running_runs()

    assert _rows(env.db)["r"][1:3] == expected
    assert env.log[0][1].endswith(f"total={total or 0} processed={processed}")


@pytest.mark.parametrize(
    ("total", "statuses", "expected", "processed"),
    [
        (5, ["ok", "ok", "failed"], (2, 3), 3),
        (3, ["ok", "ok", "ok", "failed", "failed"], (3, 0), 5),  # over-counted: failures clamp to the rest
        (1, ["ok", "failed"], (1, 0), 2),
        (None, ["ok", "failed", "failed"], (1, 2), 3),  # no total to clamp against
        (5, [], (0, 5), 0),  # the run recorded no ticker status
        (None, [], (0, 0), 0),
    ],
)
def test_per_ticker_status_overrides_the_run_counts(env, total, statuses, expected, processed) -> None:
    _insert(env.db, ("r", _ts(LIMIT_SEC + 60), "running", total, 3, 1, None))
    _ticker_status(env.db, *[("r", f"T{i}", s) for i, s in enumerate(statuses)], ("other", "X", "ok"))

    pipeline.reconcile_stale_running_runs()

    assert _rows(env.db)["r"][1:3] == expected
    assert env.log[0][1].endswith(f"total={total or 0} processed={processed}")


def test_ok_count_above_the_total_never_goes_negative(env) -> None:
    _insert(env.db, ("r", _ts(LIMIT_SEC + 60), "running", 3, 0, 0, None))
    _ticker_status(env.db, *[("r", f"T{i}", "ok") for i in range(4)])

    pipeline.reconcile_stale_running_runs()

    assert _rows(env.db)["r"][1:3] == (4, 0)
