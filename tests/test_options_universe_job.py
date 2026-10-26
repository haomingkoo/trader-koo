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
        scheduler.subprocess, "run",
        lambda cmd, **kw: ran.append(cmd) or SimpleNamespace(returncode=0),
    )
    return scheduler, job, ran


IN_WINDOW = dt.datetime(2026, 10, 26, 20, 15, tzinfo=dt.timezone.utc)


def test_job_skips_outside_window(monkeypatch, tmp_path):
    scheduler, job, ran = _sched(monkeypatch, tmp_path)
    late = dt.datetime(2026, 10, 26, 21, 40, tzinfo=dt.timezone.utc)
    assert job.run_options_universe_snapshot(now_utc=late) == "outside_window"
    assert ran == []


def test_job_skips_while_pipeline_run_is_active(monkeypatch, tmp_path):
    from trader_koo.backend.services.pipeline import reserve_pipeline_run

    scheduler, job, ran = _sched(monkeypatch, tmp_path)
    sqlite3.connect(scheduler.DB_PATH).close()
    reserve_pipeline_run(run_id="r1", mode="full", source="test", db_path=scheduler.DB_PATH)
    assert job.run_options_universe_snapshot(now_utc=IN_WINDOW) == "pipeline_active"
    assert ran == []


def test_job_runs_script_with_configured_caps(monkeypatch, tmp_path):
    scheduler, job, ran = _sched(monkeypatch, tmp_path)
    monkeypatch.setenv("TRADER_KOO_OPTIONS_UNIVERSE_MAX_TICKERS", "123")
    assert job.run_options_universe_snapshot(now_utc=IN_WINDOW) == "ok"
    cmd = ran[0]
    assert cmd[cmd.index("--max-tickers") + 1] == "123"
    assert cmd[cmd.index("--workers") + 1] == "2"
    assert cmd[cmd.index("--target-days") + 1] == "30,60,90"


def test_job_is_registered_after_close_in_new_york_time():
    from trader_koo.backend.services.scheduler import create_scheduler

    job = create_scheduler().get_job("options_iv_universe_snapshot")
    assert job is not None
    assert str(job.trigger.timezone) == "America/New_York"
    fields = {f.name: str(f) for f in job.trigger.fields}
    assert fields["hour"] == "16" and fields["minute"] == "10"
    assert fields["day_of_week"] == "mon-fri"
