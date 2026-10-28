"""Unit tests for trader_koo.backend.services.scheduler."""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from trader_koo.backend.services import ops_jobs
from trader_koo.backend.services import scheduler as scheduler_module
from trader_koo.backend.services.scheduler import (
    _run_daily_backup,
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

    _run_preopen_report_watchdog(now_utc=dt.datetime(2026, 8, 27, 12, tzinfo=dt.timezone.utc))

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

    _run_preopen_report_watchdog(now_utc=dt.datetime(2026, 8, 27, 12, tzinfo=dt.timezone.utc))

    assert calls == [("full", "preopen_watchdog")]


def _watchdog_db(tmp_path, *, price_date: str, ingest_status: str = "ok"):
    db_path = tmp_path / "scheduler.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE price_daily (date TEXT)")
        conn.execute("INSERT INTO price_daily VALUES (?)", (price_date,))
        conn.execute(
            """CREATE TABLE report_runs (
                run_id TEXT, status TEXT, publication_verified INTEGER,
                is_generation_canonical INTEGER, published_ts TEXT,
                source_timestamps_json TEXT
            )"""
        )
        conn.execute(
            "INSERT INTO report_runs VALUES ('old','published',1,1,'2026-10-15T22:30:00Z','{\"price_date\":\"2026-10-14\"}')"
        )
        conn.execute(
            """CREATE TABLE ingest_runs (
                run_id TEXT, started_ts TEXT, finished_ts TEXT, status TEXT,
                tickers_total INTEGER, tickers_ok INTEGER, tickers_failed INTEGER,
                error_message TEXT
            )"""
        )
        conn.execute(
            "INSERT INTO ingest_runs VALUES ('i','2026-10-20T22:00:00Z','2026-10-20T22:20:00Z',?,538,538,0,NULL)",
            (ingest_status,),
        )
    return db_path


@pytest.mark.parametrize(
    ("price_date", "now", "mode"),
    [
        # Oct 23 12:00Z after the outage: ingest says ok but prices are 3 sessions old.
        ("2026-10-20", dt.datetime(2026, 10, 23, 12, tzinfo=dt.timezone.utc), "full"),
        # Prices are current (Thu close) and only the publication is stale.
        ("2026-10-22", dt.datetime(2026, 10, 23, 12, tzinfo=dt.timezone.utc), "report"),
        # Saturday: Friday is the newest expected session, so Friday is current.
        ("2026-10-23", dt.datetime(2026, 10, 24, 12, tzinfo=dt.timezone.utc), "report"),
    ],
)
def test_preopen_watchdog_mode_follows_price_freshness(tmp_path, monkeypatch, price_date, now, mode):
    db_path = _watchdog_db(tmp_path, price_date=price_date)
    monkeypatch.setattr("trader_koo.backend.services.scheduler.DB_PATH", db_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "trader_koo.backend.services.scheduler._run_daily_update",
        lambda mode, source: calls.append((mode, source)),
    )

    _run_preopen_report_watchdog(now_utc=now)

    assert calls == [(mode, "preopen_watchdog")]


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
        "trader_koo.backend.services.scheduler._run_tracked_child",
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


# ---------------------------------------------------------------------------
# Failure alerts and the daily backup
# ---------------------------------------------------------------------------


@pytest.fixture()
def telegram(monkeypatch):
    """Record Telegram sends; reset the per-kind alert cooldowns."""
    from trader_koo.notifications import telegram as telegram_module

    monkeypatch.setattr(ops_jobs, "_last_ops_alert_ts", {})
    sent: list[tuple[str, str]] = []
    state = SimpleNamespace(configured=True, result=True, raises=None, sent=sent)

    def fake_send(text, parse_mode="Markdown", **kwargs):
        if state.raises is not None:
            raise state.raises
        sent.append((text, parse_mode))
        return state.result

    monkeypatch.setattr(telegram_module, "is_configured", lambda: state.configured)
    monkeypatch.setattr(telegram_module, "send_message", fake_send)
    return state


def _failing_pipeline(tmp_path, monkeypatch, run_id: str, returncode: int = 3):
    db_path = tmp_path / "scheduler.db"
    conn = sqlite3.connect(db_path)
    ensure_pipeline_runs_schema(conn)
    conn.close()
    reserve_pipeline_run(run_id=run_id, mode="report", source="scheduler", db_path=db_path)
    monkeypatch.setattr(scheduler_module, "DB_PATH", db_path)
    monkeypatch.setattr(
        scheduler_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=returncode),
    )
    _run_daily_update_unlocked("report", "scheduler", pipeline_run_id=run_id)
    with sqlite3.connect(db_path) as verify:
        return verify.execute(
            "SELECT status, error_message FROM pipeline_runs WHERE run_id = ?", (run_id,)
        ).fetchone()


def test_failed_pipeline_sends_one_telegram_alert(tmp_path, monkeypatch, telegram):
    row = _failing_pipeline(tmp_path, monkeypatch, "pipe_fail_1")

    assert row == ("failed", "daily_update.sh rc=3")
    assert len(telegram.sent) == 1
    text, parse_mode = telegram.sent[0]
    assert parse_mode == "HTML"
    assert "pipe_fail_1" in text and "rc=3" in text and "mode=report" in text


def test_successful_pipeline_sends_no_alert(tmp_path, monkeypatch, telegram):
    _failing_pipeline(tmp_path, monkeypatch, "pipe_ok", returncode=0)

    assert telegram.sent == []


def test_repeated_pipeline_failures_respect_the_cooldown(tmp_path, monkeypatch, telegram):
    _failing_pipeline(tmp_path, monkeypatch, "pipe_fail_a")
    _failing_pipeline(tmp_path, monkeypatch, "pipe_fail_b")
    assert len(telegram.sent) == 1

    ops_jobs._last_ops_alert_ts["pipeline_failed"] -= (
        ops_jobs.OPS_ALERT_COOLDOWN_SEC + 1
    )
    _failing_pipeline(tmp_path, monkeypatch, "pipe_fail_c")

    assert len(telegram.sent) == 2
    assert "pipe_fail_c" in telegram.sent[1][0]


def test_unsent_alert_does_not_start_the_cooldown(tmp_path, monkeypatch, telegram):
    telegram.result = False
    _failing_pipeline(tmp_path, monkeypatch, "pipe_fail_unsent")
    telegram.result = True
    _failing_pipeline(tmp_path, monkeypatch, "pipe_fail_retry")

    assert [("pipe_fail_retry" in text) for text, _ in telegram.sent] == [False, True]


def test_alert_failure_does_not_mask_the_pipeline_failure(tmp_path, monkeypatch, telegram, caplog):
    telegram.raises = RuntimeError("telegram down")

    with caplog.at_level("ERROR", logger="trader_koo.services.scheduler"):
        row = _failing_pipeline(tmp_path, monkeypatch, "pipe_fail_quiet")

    assert row == ("failed", "daily_update.sh rc=3")
    assert any("daily_update.sh failed" in r.getMessage() for r in caplog.records)


def test_unconfigured_telegram_logs_that_the_alert_was_not_sent(
    tmp_path, monkeypatch, telegram, caplog
):
    telegram.configured = False

    with caplog.at_level("WARNING", logger="trader_koo.services.ops_jobs"):
        _failing_pipeline(tmp_path, monkeypatch, "pipe_fail_unconfigured")

    assert telegram.sent == []
    assert any("not configured" in r.getMessage() for r in caplog.records)


def _backup_env(tmp_path, monkeypatch, *, blocked: bool = False):
    from trader_koo.backend.services import maintenance
    from trader_koo.scripts import backup_db

    db_path = tmp_path / "live.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE evidence(id INTEGER PRIMARY KEY)")
    monkeypatch.setattr(scheduler_module, "DB_PATH", db_path)
    monkeypatch.setattr(maintenance, "writers_blocked", lambda path: blocked)
    calls: list[Path] = []
    real_backup = backup_db.backup_database

    def recording_backup(path, backup_dir=tmp_path / "backups", *args, **kwargs):
        calls.append(path)
        return real_backup(path, backup_dir, *args, **kwargs)

    monkeypatch.setattr(backup_db, "backup_database", recording_backup)
    return calls


def test_daily_backup_succeeds_without_alert(tmp_path, monkeypatch, telegram):
    calls = _backup_env(tmp_path, monkeypatch)

    _run_daily_backup()

    assert len(calls) == 1
    assert len(list((tmp_path / "backups").glob("*.db.gz"))) == 1
    assert telegram.sent == []


def test_failed_backup_sends_an_alert(tmp_path, monkeypatch, telegram):
    from trader_koo.scripts import backup_db

    _backup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(backup_db, "_quick_check", lambda path: ["row 9 missing from index"])

    _run_daily_backup()

    assert len(telegram.sent) == 1
    assert "Backup failed" in telegram.sent[0][0]
    assert "row 9 missing from index" in telegram.sent[0][0]
    assert list((tmp_path / "backups").glob("*.db.gz")) == []


def test_backup_and_pipeline_alerts_have_separate_cooldowns(tmp_path, monkeypatch, telegram):
    from trader_koo.scripts import backup_db

    _failing_pipeline(tmp_path, monkeypatch, "pipe_fail_first")
    _backup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(backup_db, "_quick_check", lambda path: ["bad page"])

    _run_daily_backup()

    assert len(telegram.sent) == 2


def test_missing_database_backup_sends_an_alert(tmp_path, monkeypatch, telegram):
    monkeypatch.setattr(scheduler_module, "DB_PATH", tmp_path / "absent.db")

    _run_daily_backup()

    assert len(telegram.sent) == 1
    assert "not found" in telegram.sent[0][0]


def test_backup_skips_while_maintenance_blocks_writers(tmp_path, monkeypatch, telegram):
    calls = _backup_env(tmp_path, monkeypatch, blocked=True)

    _run_daily_backup()

    assert calls == []
    assert telegram.sent == []


def test_storage_pressure_alerts_once_per_cooldown(tmp_path, monkeypatch, telegram):
    from trader_koo.scripts import cleanup_storage

    monkeypatch.setattr(scheduler_module, "DB_PATH", tmp_path / "live.db")
    monkeypatch.setattr(scheduler_module, "_append_run_log", lambda tag, msg: None)
    monkeypatch.setattr(cleanup_storage, "run_cleanup", lambda path, dry_run: {})
    monkeypatch.setattr(cleanup_storage, "storage_health_alert", lambda path: "volume 91% full")

    scheduler_module._run_storage_cleanup()
    scheduler_module._run_storage_cleanup()

    assert len(telegram.sent) == 1
    assert "Storage pressure" in telegram.sent[0][0]
    assert "volume 91% full" in telegram.sent[0][0]


def test_backup_is_daily_after_the_nightly_pipeline():
    scheduler = create_scheduler()

    assert scheduler.get_job("weekly_backup") is None
    job = scheduler.get_job("daily_backup")
    assert job is not None
    assert str(job.trigger) == "cron[hour='3', minute='0']"
    pipeline = str(scheduler.get_job("daily_update").trigger)
    assert pipeline == "cron[day_of_week='mon-fri', hour='22', minute='0']"
    cleanup = str(scheduler.get_job("storage_cleanup").trigger)
    assert cleanup == "cron[hour='4', minute='0']"
