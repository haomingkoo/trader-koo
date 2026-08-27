"""Unit tests for trader_koo.backend.services.scheduler."""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from trader_koo.backend.services.scheduler import (
    _run_memory_cleanup,
    _run_daily_update_unlocked,
    _run_preopen_report_watchdog,
    _normalize_update_mode,
    create_scheduler,
)
from trader_koo.backend.services.pipeline import (
    ensure_pipeline_runs_schema,
    reserve_pipeline_run,
)
from trader_koo.middleware.auth import AdminAuthConfig, AdminAuthenticator


class TestCreateScheduler:
    def test_returns_background_scheduler(self):
        from apscheduler.schedulers.background import BackgroundScheduler

        scheduler = create_scheduler()

        assert isinstance(scheduler, BackgroundScheduler)

    def test_has_daily_update_job(self):
        scheduler = create_scheduler()
        job = scheduler.get_job("daily_update")

        assert job is not None

    def test_has_preopen_report_watchdog(self):
        scheduler = create_scheduler()
        job = scheduler.get_job("preopen_report_watchdog")

        assert job is not None
        assert str(job.trigger) == "cron[day_of_week='mon-fri', hour='12', minute='0']"

    def test_has_weekly_yolo_job(self):
        scheduler = create_scheduler()
        job = scheduler.get_job("weekly_yolo")

        assert job is not None

    def test_has_options_iv_snapshot_job_by_default(self):
        scheduler = create_scheduler()
        job = scheduler.get_job("options_iv_snapshot")

        assert job is not None

    def test_scheduler_not_running_on_create(self):
        scheduler = create_scheduler()

        assert not scheduler.running

    def test_default_fast_monitor_intervals(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")

        scheduler = create_scheduler()

        assert (
            scheduler.get_job("polymarket_snapshot").trigger.interval.total_seconds()
            == 5 * 60
        )
        assert scheduler.get_job("spike_alerts").trigger.interval.total_seconds() == 5 * 60
        assert scheduler.get_job("macro_alert").trigger.interval.total_seconds() == 10 * 60
        assert (
            scheduler.get_job("hyperliquid_poll").trigger.interval.total_seconds()
            == 5 * 60
        )
        assert (
            scheduler.get_job("site_health_check").trigger.interval.total_seconds()
            == 10 * 60
        )
        assert (
            scheduler.get_job("crypto_health_check").trigger.interval.total_seconds()
            == 15 * 60
        )
        assert (
            scheduler.get_job("derivatives_snapshot").trigger.interval.total_seconds()
            == 15 * 60
        )

    def test_options_iv_snapshot_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("TRADER_KOO_OPTIONS_SNAPSHOT_ENABLED", "0")

        scheduler = create_scheduler()

        assert scheduler.get_job("options_iv_snapshot") is None

    def test_memory_cleanup_uses_authenticator_interface(self):
        authenticator = AdminAuthenticator(
            AdminAuthConfig(api_key="x" * 32, failure_window_sec=10, block_sec=10)
        )
        authenticator._failures["expired"] = {"updated_ts": 0.0}
        app = SimpleNamespace(
            state=SimpleNamespace(
                admin_authenticator=authenticator,
                rate_limiter=None,
            )
        )

        _run_memory_cleanup(app)

        assert authenticator._failures == {}

    def test_monitor_intervals_can_be_overridden(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setenv("TRADER_KOO_POLYMARKET_SNAPSHOT_MINUTES", "7")
        monkeypatch.setenv("TRADER_KOO_SPIKE_ALERT_MINUTES", "11")
        monkeypatch.setenv("TRADER_KOO_HYPERLIQUID_POLL_MINUTES", "13")

        scheduler = create_scheduler()

        assert (
            scheduler.get_job("polymarket_snapshot").trigger.interval.total_seconds()
            == 7 * 60
        )
        assert scheduler.get_job("spike_alerts").trigger.interval.total_seconds() == 11 * 60
        assert (
            scheduler.get_job("hyperliquid_poll").trigger.interval.total_seconds()
            == 13 * 60
        )


class TestNormalizeUpdateMode:
    def test_full_maps_to_full(self):
        assert _normalize_update_mode("full") == "full"

    def test_all_maps_to_full(self):
        assert _normalize_update_mode("all") == "full"

    def test_yolo_maps_to_yolo(self):
        assert _normalize_update_mode("yolo") == "yolo"

    def test_yolo_report_maps_to_yolo(self):
        assert _normalize_update_mode("yolo_report") == "yolo"

    def test_yolo_plus_report_maps_to_yolo(self):
        assert _normalize_update_mode("yolo+report") == "yolo"

    def test_report_maps_to_report(self):
        assert _normalize_update_mode("report") == "report"

    def test_report_only_maps_to_report(self):
        assert _normalize_update_mode("report_only") == "report"

    def test_email_maps_to_report(self):
        assert _normalize_update_mode("email") == "report"

    def test_none_defaults_to_full(self):
        assert _normalize_update_mode(None) == "full"

    def test_unknown_mode_returns_none(self):
        assert _normalize_update_mode("garbage_mode") is None

    def test_case_insensitive(self):
        assert _normalize_update_mode("FULL") == "full"
        assert _normalize_update_mode("Yolo") == "yolo"
        assert _normalize_update_mode("REPORT") == "report"

    def test_whitespace_stripped(self):
        assert _normalize_update_mode("  full  ") == "full"


def test_preopen_watchdog_does_not_duplicate_current_report(tmp_path, monkeypatch):
    db_path = tmp_path / "scheduler.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE price_daily (date TEXT)")
        conn.execute("INSERT INTO price_daily VALUES ('2026-08-26')")
        conn.execute(
            """CREATE TABLE report_runs (
                run_id TEXT, status TEXT, publication_verified INTEGER,
                is_generation_canonical INTEGER, published_ts TEXT,
                source_timestamps_json TEXT
            )"""
        )
        conn.execute(
            "INSERT INTO report_runs VALUES (?, ?, ?, ?, ?, ?)",
            ("run", "published", 1, 1, "2026-08-26T22:30:00Z", '{"price_date":"2026-08-26"}'),
        )
    monkeypatch.setattr("trader_koo.backend.services.scheduler.DB_PATH", db_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "trader_koo.backend.services.scheduler._run_daily_update",
        lambda mode, source: calls.append((mode, source)),
    )

    _run_preopen_report_watchdog()

    assert calls == []


def test_preopen_watchdog_recovers_newer_ingested_price_date(tmp_path, monkeypatch):
    db_path = tmp_path / "scheduler.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE price_daily (date TEXT)")
        conn.execute("INSERT INTO price_daily VALUES ('2026-08-26')")
        conn.execute(
            """CREATE TABLE report_runs (
                run_id TEXT, status TEXT, publication_verified INTEGER,
                is_generation_canonical INTEGER, published_ts TEXT,
                source_timestamps_json TEXT
            )"""
        )
        conn.execute(
            "INSERT INTO report_runs VALUES (?, ?, ?, ?, ?, ?)",
            ("old", "published", 1, 1, "2026-08-25T22:30:00Z", '{"price_date":"2026-08-25"}'),
        )
        conn.execute(
            """CREATE TABLE ingest_runs (
                run_id TEXT, started_ts TEXT, finished_ts TEXT, status TEXT,
                tickers_total INTEGER, tickers_ok INTEGER, tickers_failed INTEGER,
                error_message TEXT
            )"""
        )
        conn.execute(
            "INSERT INTO ingest_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("ingest", "2026-08-26T22:00:00Z", "2026-08-26T22:20:00Z", "ok", 540, 540, 0, None),
        )
    monkeypatch.setattr("trader_koo.backend.services.scheduler.DB_PATH", db_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "trader_koo.backend.services.scheduler._run_daily_update",
        lambda mode, source: calls.append((mode, source)),
    )

    _run_preopen_report_watchdog()

    assert calls == [("report", "preopen_watchdog")]


def test_preopen_watchdog_reruns_failed_ingest_before_report(tmp_path, monkeypatch):
    db_path = tmp_path / "scheduler.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE price_daily (date TEXT)")
        conn.execute("INSERT INTO price_daily VALUES ('2026-08-26')")
        conn.execute(
            """CREATE TABLE report_runs (
                run_id TEXT, status TEXT, publication_verified INTEGER,
                is_generation_canonical INTEGER, published_ts TEXT,
                source_timestamps_json TEXT
            )"""
        )
        conn.execute(
            "INSERT INTO report_runs VALUES (?, ?, ?, ?, ?, ?)",
            ("old", "published", 1, 1, "2026-08-25T22:30:00Z", '{"price_date":"2026-08-25"}'),
        )
        conn.execute(
            """CREATE TABLE ingest_runs (
                run_id TEXT, started_ts TEXT, finished_ts TEXT, status TEXT,
                tickers_total INTEGER, tickers_ok INTEGER, tickers_failed INTEGER,
                error_message TEXT
            )"""
        )
        conn.execute(
            "INSERT INTO ingest_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("ingest", "2026-08-26T22:00:00Z", "2026-08-26T22:20:00Z", "failed", 540, 538, 2, "2 failed"),
        )
    monkeypatch.setattr("trader_koo.backend.services.scheduler.DB_PATH", db_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "trader_koo.backend.services.scheduler._run_daily_update",
        lambda mode, source: calls.append((mode, source)),
    )

    _run_preopen_report_watchdog()

    assert calls == [("full", "preopen_watchdog")]


def test_worker_transitions_reserved_record_before_subprocess(tmp_path, monkeypatch):
    db_path = tmp_path / "scheduler.db"
    conn = sqlite3.connect(db_path)
    ensure_pipeline_runs_schema(conn)
    conn.close()
    reserve_pipeline_run(
        run_id="pipe_reserved",
        mode="report",
        source="admin",
        db_path=db_path,
    )

    def completed_subprocess(*args, **kwargs):
        verify = sqlite3.connect(db_path)
        row = verify.execute(
            "SELECT status, stage FROM pipeline_runs WHERE run_id = 'pipe_reserved'"
        ).fetchone()
        verify.close()
        assert row == ("running", "report")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("trader_koo.backend.services.scheduler.DB_PATH", db_path)
    monkeypatch.setattr(
        "trader_koo.backend.services.scheduler.subprocess.run",
        completed_subprocess,
    )

    _run_daily_update_unlocked(
        "report",
        "admin",
        pipeline_run_id="pipe_reserved",
    )

    verify = sqlite3.connect(db_path)
    rows = verify.execute(
        "SELECT run_id, status, report_ok FROM pipeline_runs"
    ).fetchall()
    verify.close()
    assert rows == [("pipe_reserved", "ok", 1)]
