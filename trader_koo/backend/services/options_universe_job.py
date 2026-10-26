"""Scheduler job: archive option chains for the whole ingest universe after the close."""
from __future__ import annotations

import datetime as dt
import logging
import subprocess
import sys

from trader_koo.backend.services import scheduler as sched
from trader_koo.backend.services.maintenance import (
    MaintenanceError,
    inherited_writer_lease_fds,
    writers_blocked,
)
from trader_koo.backend.services.pipeline import read_active_pipeline_run
from trader_koo.collection.config import (
    UNIVERSE_CRON_HOUR_ET,
    UNIVERSE_CRON_MINUTE_ET,
    get_options_universe_config,
)
from trader_koo.collection.options_universe import universe_window_open

LOG = logging.getLogger(__name__)
JOB_ID = "options_iv_universe_snapshot"


def run_options_universe_snapshot(now_utc: dt.datetime | None = None) -> str:
    """Run the universe snapshot script; return why it did not run, or ok/failed.

    This never reserves a pipeline run (that would make the 22:00 daily_update
    skip). It refuses to start, and the script stops, when the post-close window
    is closed or a pipeline run is active.
    """
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    if not universe_window_open(now):
        sched._append_run_log("OPTIONS_IV", "Universe snapshot skipped: outside post-close window")
        return "outside_window"
    try:
        blocked = writers_blocked(sched.DB_PATH)
    except MaintenanceError as exc:
        blocked = True
        LOG.error("options universe snapshot fail-closed on maintenance state: %s", exc.code)
    if blocked:
        sched._append_run_log("OPTIONS_IV", "Universe snapshot skipped: maintenance intent")
        return "maintenance"
    active = read_active_pipeline_run(sched.DB_PATH)
    if active is not None:
        sched._append_run_log(
            "OPTIONS_IV", f"Universe snapshot skipped: pipeline run active run_id={active.get('run_id')}"
        )
        return "pipeline_active"

    cfg = get_options_universe_config()
    cmd = [
        sys.executable,
        str(sched.SCRIPTS_DIR / "snapshot_options_universe.py"),
        "--db-path", str(sched.DB_PATH),
        "--max-tickers", str(cfg.max_tickers),
        "--workers", str(cfg.workers),
        "--sleep", str(cfg.sleep_sec),
        "--max-expiries", str(cfg.nearest_expiries),
        "--target-days", ",".join(str(d) for d in cfg.target_days),
        "--log-file", str(cfg.log_path),
    ]
    sched._append_run_log("OPTIONS_IV", "Universe snapshot started")
    started = dt.datetime.now(dt.timezone.utc)
    with sched.RUN_LOG_PATH.open("a", encoding="utf-8") as run_log:
        result = subprocess.run(
            cmd,
            stdout=run_log,
            stderr=subprocess.STDOUT,
            text=True,
            pass_fds=inherited_writer_lease_fds(),
        )
    elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    sched._append_run_log(
        "OPTIONS_IV", f"Universe snapshot finished rc={result.returncode} sec={elapsed:.1f}"
    )
    return "ok" if result.returncode == 0 else "failed"


def register_options_universe_job(scheduler) -> None:
    from apscheduler.triggers.cron import CronTrigger

    if not get_options_universe_config().enabled:
        LOG.info("Options IV universe snapshot disabled by TRADER_KOO_OPTIONS_UNIVERSE_ENABLED")
        return
    scheduler.add_job(
        run_options_universe_snapshot,
        CronTrigger(
            hour=UNIVERSE_CRON_HOUR_ET,
            minute=UNIVERSE_CRON_MINUTE_ET,
            day_of_week="mon-fri",
            timezone="America/New_York",
        ),
        id=JOB_ID,
        replace_existing=True,
    )
    LOG.info("Options IV universe snapshot registered: Mon-Fri after the US close (America/New_York)")
