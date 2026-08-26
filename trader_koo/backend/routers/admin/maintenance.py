"""Authenticated request/status boundary for offline database maintenance."""
from __future__ import annotations

import hashlib
import os
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from trader_koo.backend.services.database import DB_PATH
from trader_koo.backend.services.maintenance import MaintenanceError, request_maintenance, status

router = APIRouter(tags=["admin", "admin-maintenance"])


class MaintenanceRequest(BaseModel):
    reason: str = Field(min_length=8, max_length=500)
    timeout_sec: int = Field(default=60, ge=1, le=600)
    idempotency_key: str = Field(min_length=8, max_length=120)
    purpose: Literal["recovery", "copied-rehearsal", "production-migration"] = "recovery"
    approval_ref: str | None = Field(default=None, max_length=240)
    expected_release_sha: str | None = Field(default=None, max_length=40)
    expected_write_gate: Literal["0"] | None = None


@router.post("/api/admin/maintenance/request")
def request_database_maintenance(request: Request, body: MaintenanceRequest) -> dict[str, Any]:
    boot_id = str(getattr(request.app.state, "boot_id", ""))
    if not boot_id or getattr(request.app.state, "maintenance_mode", False):
        raise HTTPException(status_code=409, detail="maintenance_restart_in_progress")
    if body.purpose == "copied-rehearsal" and (
        os.getenv("TRADER_KOO_GIT_SHA")
        or os.getenv("RAILWAY_ENVIRONMENT_NAME", "").lower() == "production"
    ):
        raise HTTPException(status_code=409, detail="copied_rehearsal_production_refused")
    run_id = "maint_" + hashlib.sha256(body.idempotency_key.encode("utf-8")).hexdigest()[:20]
    try:
        result = request_maintenance(DB_PATH, run_id=run_id, boot_id=boot_id,
                                     reason=body.reason, timeout_sec=body.timeout_sec,
                                     purpose=body.purpose,
                                     approval_ref=body.approval_ref,
                                     expected_release_sha=body.expected_release_sha,
                                     expected_write_gate=body.expected_write_gate)
    except MaintenanceError as exc:
        raise HTTPException(status_code=409, detail=exc.code) from exc
    scheduler = getattr(request.app.state, "scheduler", None)
    scheduler_warning = None
    if scheduler is not None and getattr(scheduler, "running", False):
        try:
            scheduler.pause()
        except Exception:
            # Intent is already durable and must never be rolled back. The
            # process lease keeps backup blocked until this instance restarts.
            scheduler_warning = "scheduler_pause_failed_restart_required"
    return {
        "ok": True,
        "maintenance": result,
        "scheduler_warning": scheduler_warning,
        "next": "restart into maintenance-only mode, then run quiesce-backup",
    }


@router.get("/api/admin/maintenance/status")
def database_maintenance_status() -> dict[str, Any]:
    return {"ok": True, "maintenance": status(DB_PATH)}
