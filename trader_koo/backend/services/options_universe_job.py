"""Scheduler job: archive option chains for the whole ingest universe after the nightly window."""
from __future__ import annotations

import datetime as dt
import logging
import subprocess
import sys
import time
from typing import Callable

from trader_koo.backend.services import scheduler as sched
from trader_koo.backend.services.maintenance import (
    MaintenanceError,
    inherited_writer_lease_fds,
    writers_blocked,
)
from trader_koo.backend.services.pipeline import read_active_pipeline_run
from trader_koo.collection.config import (
    UNIVERSE_CRON_HOUR_UTC,
    UNIVERSE_CRON_MINUTE_UTC,
    UNIVERSE_WAIT_POLL_SEC,
    get_options_universe_config,
)
from trader_koo.collection.options_universe import universe_session_date

LOG = logging.getLogger(__name__)
JOB_ID = "options_iv_universe_snapshot"


def run_options_universe_snapshot(
    now_utc: dt.datetime | None = None,
    *,
    clock: Callable[[], dt.datetime] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Run the universe snapshot script; return why it did not run, or ok/failed.

    Starts at 23:45 UTC. If a pipeline run (the nightly daily_update) is still
    active it waits, re-checking every few minutes, and gives up for the day at
    03:00 UTC. It never reserves a pipeline run, so it cannot make a daily_update
    skip.
    """
    clock = clock or (lambda: now_utc or dt.datetime.now(dt.timezone.utc))
    if universe_session_date(clock()) is None:
        sched._append_run_log("OPTIONS_IV", "Universe snapshot skipped: outside 23:45-03:00 UTC window")
        return "outside_window"
    try:
        blocked = writers_blocked(sched.DB_PATH)
    except MaintenanceError as exc:
        blocked = True
        LOG.error("options universe snapshot fail-closed on maintenance state: %s", exc.code)
    if blocked:
        sched._append_run_log("OPTIONS_IV", "Universe snapshot skipped: maintenance intent")
        return "maintenance"
    waited = False
    while (active := read_active_pipeline_run(sched.DB_PATH)) is not None:
        if not waited:
            sched._append_run_log(
                "OPTIONS_IV",
                f"Universe snapshot waiting for pipeline run run_id={active.get('run_id')}",
            )
            waited = True
        sleep(UNIVERSE_WAIT_POLL_SEC)
        if universe_session_date(clock()) is None:
            sched._append_run_log(
                "OPTIONS_IV", "Universe snapshot gave up for the day: pipeline still active at window end"
            )
            return "gave_up"

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
            hour=UNIVERSE_CRON_HOUR_UTC,
            minute=UNIVERSE_CRON_MINUTE_UTC,
            day_of_week="mon-fri",
            timezone="UTC",
        ),
        id=JOB_ID,
        replace_existing=True,
    )
    LOG.info("Options IV universe snapshot registered: Mon-Fri 23:45 UTC (session of that day)")
