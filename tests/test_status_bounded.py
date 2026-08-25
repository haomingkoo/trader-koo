from __future__ import annotations

import datetime as dt
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


def test_status_never_revalidates_full_price_history(seeded_conn, tmp_path: Path) -> None:
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
    assert payload["price_basis"]["verification_mode"] == "persisted_revision_seal"
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


def test_persisted_revision_summary_does_not_read_price_rows(seeded_conn) -> None:
    statements: list[str] = []
    seeded_conn.set_trace_callback(statements.append)

    summary = system._persisted_price_revision_summary(seeded_conn, tracked_tickers=2)

    assert summary["verified_tickers"] == 1
    assert summary["unresolved_tickers"] == 1
    assert summary["missing_revision_tickers"] == 1
    assert not any("price_daily" in statement.lower() for statement in statements)


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
