from __future__ import annotations

import datetime as dt
import sqlite3
from types import SimpleNamespace


def _sched(monkeypatch, tmp_path):
    from trader_koo.backend.services import options_universe_job as job
    from trader_koo.backend.services import scheduler

    monkeypatch.setattr(scheduler, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(scheduler, "RUN_LOG_PATH", tmp_path / "run.log")
    monkeypatch.setattr(scheduler, "LOG_DIR", tmp_path)
    ran: list[list[str]] = []
    monkeypatch.setattr(
        scheduler, "_run_tracked_child",
        lambda cmd, **kw: ran.append(cmd) or SimpleNamespace(returncode=0),
    )
    return scheduler, job, ran


IN_WINDOW = dt.datetime(2026, 10, 26, 23, 45, tzinfo=dt.timezone.utc)  # Monday evening


def test_job_skips_outside_window(monkeypatch, tmp_path):
    scheduler, job, ran = _sched(monkeypatch, tmp_path)
    inside_nightly = dt.datetime(2026, 10, 26, 22, 10, tzinfo=dt.timezone.utc)
    assert job.run_options_universe_snapshot(now_utc=inside_nightly) == "outside_window"
    assert ran == []


def _reserve(scheduler):
    from trader_koo.backend.services.pipeline import reserve_pipeline_run

    sqlite3.connect(scheduler.DB_PATH).close()
    reserve_pipeline_run(run_id="r1", mode="full", source="test", db_path=scheduler.DB_PATH)


def test_job_waits_for_active_pipeline_run_then_runs(monkeypatch, tmp_path):
    from trader_koo.backend.services.pipeline import finish_pipeline_run

    scheduler, job, ran = _sched(monkeypatch, tmp_path)
    _reserve(scheduler)
    sleeps: list[float] = []

    def sleep(sec):
        sleeps.append(sec)
        if len(sleeps) == 2:  # nightly update finishes during the second wait
            finish_pipeline_run("r1", status="completed", db_path=scheduler.DB_PATH)

    assert job.run_options_universe_snapshot(now_utc=IN_WINDOW, sleep=sleep) == "ok"
    assert len(sleeps) == 2 and len(ran) == 1


def test_job_gives_up_at_window_end_if_pipeline_never_finishes(monkeypatch, tmp_path):
    scheduler, job, ran = _sched(monkeypatch, tmp_path)
    _reserve(scheduler)
    clock_now = {"t": IN_WINDOW}

    def sleep(sec):
        clock_now["t"] += dt.timedelta(hours=1)

    result = job.run_options_universe_snapshot(clock=lambda: clock_now["t"], sleep=sleep)
    assert result == "gave_up" and ran == []
    assert "gave up for the day" in (tmp_path / "run.log").read_text()


def test_job_runs_script_with_configured_caps(monkeypatch, tmp_path):
    scheduler, job, ran = _sched(monkeypatch, tmp_path)
    monkeypatch.setenv("TRADER_KOO_OPTIONS_UNIVERSE_MAX_TICKERS", "123")
    assert job.run_options_universe_snapshot(now_utc=IN_WINDOW) == "ok"
    cmd = ran[0]
    assert cmd[cmd.index("--max-tickers") + 1] == "123"
    assert cmd[cmd.index("--workers") + 1] == "2"
    assert cmd[cmd.index("--target-days") + 1] == "30,60,90"


def test_job_is_registered_after_the_nightly_window():
    from trader_koo.backend.services.scheduler import create_scheduler

    job = create_scheduler().get_job("options_iv_universe_snapshot")
    assert job is not None
    assert str(job.trigger.timezone) == "UTC"
    fields = {f.name: str(f) for f in job.trigger.fields}
    assert fields["hour"] == "23" and fields["minute"] == "45"
    assert fields["day_of_week"] == "mon-fri"
