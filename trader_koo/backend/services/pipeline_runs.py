"""Pipeline runs: request a ``daily_update`` run, and recover after a restart.

Callers use two entry points:

- ``request_run(scheduler, mode=..., source=...)`` mints a run id, reserves
  it in ``pipeline_runs``, queues the job, and closes the reservation again
  if the scheduler rejects the job.
- ``recover_after_restart(scheduler)`` runs startup recovery in its fixed
  order: reconcile stale ingest runs, requeue a surviving reservation, resume
  an interrupted run, the legacy post-ingest resume, then the outage catch-up.

The scheduler is passed in, so tests can hand over an in-memory fake. Jobs
name the runner by its APScheduler textual reference, which keeps this module
from importing the scheduler module (the scheduler imports this one).
"""
from __future__ import annotations

import datetime as dt
import logging
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from trader_koo.backend.services import pipeline as pipeline_status
from trader_koo.backend.services.pipeline import (
    _abandon_stale_pipeline_runs,
    _latest_price_date,
    determine_resume_mode,
    finish_pipeline_run,
    post_ingest_resume_candidate,
    read_active_pipeline_run,
    read_interrupted_pipeline_run,
    reserve_pipeline_run,
    startup_catchup_reason,
    startup_work_queued,
)
from trader_koo.backend.services.run_log import append_run_log as _append_run_log

if TYPE_CHECKING:
    from apscheduler.schedulers.background import BackgroundScheduler

LOG = logging.getLogger("trader_koo.services.pipeline_runs")

DAILY_UPDATE_JOB = "trader_koo.backend.services.scheduler:_run_daily_update"
_ID_SUFFIX_BYTES = 3  # six hex characters
_RESUME_STATUS_LOG_LINES = 120


def _stamp(now: dt.datetime) -> str:
    return now.strftime("%Y%m%dT%H%M%S")


def mint_run_id(now: dt.datetime | None = None) -> str:
    """Return a new run id: ``pipe_<UTC yyyymmddThhmmss>_<6 hex>``."""
    return f"pipe_{_stamp(now or dt.datetime.now(dt.timezone.utc))}_{secrets.token_hex(_ID_SUFFIX_BYTES)}"


def _job_id(prefix: str, now: dt.datetime) -> str:
    return f"{prefix}_{_stamp(now)}_{secrets.token_hex(_ID_SUFFIX_BYTES)}"


# ---------------------------------------------------------------------------
# Requesting a run
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunRequest:
    """Outcome of ``request_run``.

    ``status`` is ``queued`` (reserved and on the scheduler), ``active``
    (another run is queued or running; nothing changed), ``unpersisted``
    (the reservation could not be written; nothing queued) or ``rejected``
    (the scheduler refused the job; the reservation is closed as failed).
    """

    status: Literal["queued", "active", "unpersisted", "rejected"]
    mode: str
    source: str
    run_id: str | None = None
    job_id: str | None = None
    active_run: dict[str, Any] | None = None


def request_run(
    scheduler: BackgroundScheduler,
    *,
    mode: str,
    source: str,
    job_prefix: str = "manual_daily_update",
    db_path: Path | None = None,
    now: dt.datetime | None = None,
) -> RunRequest:
    """Reserve a run and queue it to start now, undoing the reservation on failure."""
    now = now or dt.datetime.now(dt.timezone.utc)
    run_id = mint_run_id(now)
    try:
        active = reserve_pipeline_run(run_id=run_id, mode=mode, source=source, db_path=db_path)
    except Exception as exc:
        LOG.error("Failed to persist pipeline reservation source=%s: %s", source, exc)
        return RunRequest("unpersisted", mode, source)
    if active is not None:
        return RunRequest("active", mode, source, active_run=active)
    job_id = _job_id(job_prefix, now)
    try:
        scheduler.add_job(
            DAILY_UPDATE_JOB,
            trigger="date",
            run_date=now,
            id=job_id,
            kwargs={"mode": mode, "source": source, "run_id": run_id},
        )
    except Exception as exc:
        finish_pipeline_run(
            run_id,
            status="failed",
            error_message="scheduler_enqueue_failed",
            db_path=db_path,
        )
        LOG.error("Failed to enqueue reserved pipeline run_id=%s: %s", run_id, exc)
        return RunRequest("rejected", mode, source, run_id=run_id)
    return RunRequest("queued", mode, source, run_id=run_id, job_id=job_id)


# ---------------------------------------------------------------------------
# Restart recovery steps
# ---------------------------------------------------------------------------

def queue_reserved_pipeline_run(
    scheduler: BackgroundScheduler,
    *,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Requeue a durable manual reservation that survived an app restart."""
    reserved = read_active_pipeline_run(db_path=db_path)
    if reserved is None or reserved.get("status") != "queued":
        return {"scheduled": False}
    run_id = str(reserved["run_id"])
    mode = str(reserved["mode"])
    source = str(reserved["source"])
    job_id = f"resume_reserved_{run_id}"
    scheduler.add_job(
        DAILY_UPDATE_JOB,
        trigger="date",
        run_date=dt.datetime.now(dt.timezone.utc),
        id=job_id,
        kwargs={"mode": mode, "source": source, "run_id": run_id},
    )
    LOG.warning("Requeued reserved pipeline run_id=%s mode=%s source=%s", run_id, mode, source)
    return {"scheduled": True, "job_id": job_id, "run_id": run_id, "mode": mode, "source": source}


def resume_interrupted_run(
    scheduler: BackgroundScheduler,
    *,
    db_path: Path | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Close a run killed by the restart and request a run for its unfinished stages."""
    interrupted = read_interrupted_pipeline_run(db_path=db_path)
    if not interrupted:
        return {"found": False, "queued": False}
    interrupted_id = str(interrupted.get("run_id") or "")
    interrupted_mode = str(interrupted.get("mode") or "")
    interrupted_stage = str(interrupted.get("stage") or "")
    out: dict[str, Any] = {"found": True, "queued": False, "interrupted_run_id": interrupted_id}
    LOG.warning(
        "Startup detected interrupted pipeline run_id=%s mode=%s stage=%s",
        interrupted_id,
        interrupted_mode,
        interrupted_stage,
    )
    try:
        finish_pipeline_run(
            interrupted_id,
            status="interrupted",
            error_message=f"killed_by_restart at stage {interrupted_stage or 'unknown'}",
            db_path=db_path,
        )
    except Exception as exc:
        LOG.warning("Failed to mark interrupted pipeline run: %s", exc)

    resume_mode = determine_resume_mode(interrupted)
    out["resume_mode"] = resume_mode
    if resume_mode and not pipeline_status.AUTO_RESUME_INTERRUPTED_PIPELINE:
        LOG.warning(
            "Startup auto-resume disabled; not queueing interrupted pipeline "
            "run_id=%s mode=%s stage=%s resume_mode=%s",
            interrupted_id,
            interrupted_mode,
            interrupted_stage,
            resume_mode,
        )
        return out
    if not resume_mode:
        LOG.info(
            "Startup: interrupted pipeline run_id=%s does not need resume (stage=%s)",
            interrupted_id,
            interrupted_stage,
        )
        return out

    request = request_run(
        scheduler,
        mode=resume_mode,
        source=f"startup_resume:interrupted:{interrupted_id}",
        job_prefix="resume_pipeline",
        db_path=db_path,
        now=now,
    )
    out.update(queued=request.status == "queued", run_id=request.run_id, job_id=request.job_id)
    if request.status == "queued":
        LOG.warning(
            "Startup queued pipeline resume job_id=%s run_id=%s mode=%s "
            "from interrupted run_id=%s stage=%s",
            request.job_id,
            request.run_id,
            resume_mode,
            interrupted_id,
            interrupted_stage,
        )
    else:
        LOG.warning(
            "Startup did not queue pipeline resume for interrupted run_id=%s: %s",
            interrupted_id,
            request.status,
        )
    return out


def queue_post_ingest_resume(
    scheduler: BackgroundScheduler,
    source: str = "startup",
) -> dict[str, Any]:
    """Conditionally schedule a post-ingest resume job on *scheduler*.

    The job reserves its own run when it starts, as the cron job does.
    """
    now_utc = dt.datetime.now(dt.timezone.utc)
    pipeline = pipeline_status.pipeline_status_snapshot(log_lines=_RESUME_STATUS_LOG_LINES)
    candidate = post_ingest_resume_candidate(
        latest_run=pipeline.get("latest_run"),
        pipeline_active=bool(pipeline.get("active")),
        now_utc=now_utc,
    )
    if candidate is None:
        return {"scheduled": False}
    if not candidate.get("eligible", True):
        LOG.warning(
            "Post-ingest resume not queued: mode=%s reason=%s run_id=%s retries=%s/%s",
            candidate.get("mode"),
            candidate.get("reason"),
            ((candidate.get("latest_run") or {}).get("run_id") or "-"),
            candidate.get("resume_attempts"),
            candidate.get("max_retries"),
        )
        return {"scheduled": False, **candidate}

    job_id = _job_id("resume_daily_update", now_utc)
    scheduler.add_job(
        DAILY_UPDATE_JOB,
        trigger="date",
        run_date=now_utc,
        id=job_id,
        kwargs={"mode": str(candidate.get("mode") or "yolo"), "source": source},
    )
    _append_run_log(
        "RESUME",
        (
            f"queued mode={candidate.get('mode')} source={source} "
            f"reason={candidate.get('reason')} run_id={((candidate.get('latest_run') or {}).get('run_id') or '-')}"
        ),
    )
    LOG.warning(
        "Queued post-ingest resume mode=%s source=%s reason=%s run_id=%s",
        candidate.get("mode"),
        source,
        candidate.get("reason"),
        ((candidate.get("latest_run") or {}).get("run_id") or "-"),
    )
    return {"scheduled": True, "job_id": job_id, **candidate}


def queue_startup_catchup(
    scheduler: BackgroundScheduler,
    *,
    already_queued: bool = False,
    db_path: Path | None = None,
    now_utc: dt.datetime | None = None,
) -> dict[str, Any]:
    """Queue one delayed full run when stored prices lag the last due session.

    The scheduler keeps no jobstore, so a container that was down at 22:00 UTC
    forgets the missed run. This is the only place that notices at boot.
    Callers pass already_queued=True only when a job was really queued.
    When the catch-up is needed, leftover queued or running pipeline_runs rows
    are marked interrupted first. The job reserves its own run and would
    otherwise log SKIP and do nothing. Skips are written to the run log.
    """
    path = db_path or pipeline_status.DB_PATH
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    latest = _latest_price_date(path)
    if latest is None:
        return {"scheduled": False}
    reason = startup_catchup_reason(
        latest_price_date=latest,
        now_utc=now,
        already_queued=already_queued,
    )
    if reason is None:
        return {"scheduled": False}
    abandoned = _abandon_stale_pipeline_runs(path)
    if abandoned:
        _append_run_log("CATCHUP", f"marked {abandoned} stale pipeline run(s) interrupted")
    job_id = f"startup_catchup_{_stamp(now)}"
    scheduler.add_job(
        DAILY_UPDATE_JOB,
        trigger="date",
        run_date=now + pipeline_status.STARTUP_CATCHUP_DELAY,
        id=job_id,
        kwargs={"mode": "full", "source": "startup_catchup"},
    )
    _append_run_log("CATCHUP", f"queued mode=full source=startup_catchup {reason}")
    LOG.warning("Queued startup catch-up job_id=%s %s", job_id, reason)
    return {"scheduled": True, "job_id": job_id, "reason": reason}


# ---------------------------------------------------------------------------
# Restart recovery
# ---------------------------------------------------------------------------

def recover_after_restart(
    scheduler: BackgroundScheduler,
    *,
    db_path: Path | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Run every startup recovery step in order and return each step's result.

    Order matters: stale ingest rows are failed before the status snapshot
    reads them, a surviving reservation is requeued before an interrupted run
    tries to reserve, and the catch-up runs last so it is skipped whenever an
    earlier step really queued a job.
    """
    reconcile = pipeline_status.reconcile_stale_running_runs()
    if reconcile.get("reconciled"):
        LOG.warning(
            "Startup recovered %s stale ingest run(s): %s",
            reconcile.get("reconciled"),
            ",".join(reconcile.get("run_ids", [])),
        )

    reserved = queue_reserved_pipeline_run(scheduler, db_path=db_path)
    if reserved.get("scheduled"):
        LOG.warning(
            "Startup requeued reserved pipeline run_id=%s job_id=%s mode=%s",
            reserved.get("run_id"),
            reserved.get("job_id"),
            reserved.get("mode"),
        )

    interrupted = resume_interrupted_run(scheduler, db_path=db_path, now=now)

    post_ingest = queue_post_ingest_resume(scheduler, source="startup_resume")
    if post_ingest.get("scheduled"):
        LOG.warning(
            "Startup queued post-ingest resume job_id=%s mode=%s reason=%s",
            post_ingest.get("job_id"),
            post_ingest.get("mode"),
            post_ingest.get("reason"),
        )

    catchup: dict[str, Any] = {"scheduled": False}
    if pipeline_status.AUTO_STARTUP_CATCHUP:
        catchup = queue_startup_catchup(
            scheduler,
            already_queued=startup_work_queued(
                reserved=reserved,
                interrupted_resume_queued=bool(interrupted.get("queued")),
                post_ingest=post_ingest,
            ),
            db_path=db_path,
            now_utc=now,
        )
        if catchup.get("scheduled"):
            LOG.warning(
                "Startup queued outage catch-up job_id=%s %s",
                catchup.get("job_id"),
                catchup.get("reason"),
            )
    return {
        "reconcile": reconcile,
        "reserved": reserved,
        "interrupted": interrupted,
        "post_ingest": post_ingest,
        "catchup": catchup,
    }
