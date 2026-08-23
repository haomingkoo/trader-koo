"""Admin backup endpoints — list and download SQLite backups."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from trader_koo.scripts.backup_db import (
    DEFAULT_BACKUP_DIR,
    backup_path_by_name,
    backup_database,
    latest_backup_path,
    list_backups,
)
from trader_koo.backend.services.database import DB_PATH

router = APIRouter(tags=["admin", "admin-backups"])


@router.post("/api/admin/backups")
def admin_create_backup() -> dict[str, Any]:
    """Create a consistent online SQLite backup before a release migration."""
    result = backup_database(DB_PATH, DEFAULT_BACKUP_DIR)
    return {"ok": True, **result}


@router.get("/api/admin/backups")
def admin_list_backups() -> dict[str, Any]:
    """List available database backups with size and timestamp."""
    backups = list_backups(DEFAULT_BACKUP_DIR)
    total_bytes = sum(b["size_bytes"] for b in backups)
    return {
        "ok": True,
        "backup_dir": str(DEFAULT_BACKUP_DIR),
        "count": len(backups),
        "total_size_bytes": total_bytes,
        "backups": backups,
    }


@router.get("/api/admin/backups/latest")
def admin_download_latest_backup() -> Any:
    """Download the most recent backup file."""
    path = latest_backup_path(DEFAULT_BACKUP_DIR)
    if path is None or not path.exists():
        return {"ok": False, "detail": "No backups available"}
    return FileResponse(
        str(path),
        media_type="application/gzip",
        filename=path.name,
    )


@router.get("/api/admin/backups/{backup_name}")
def admin_download_named_backup(backup_name: str) -> Any:
    """Download the exact immutable backup returned by the create endpoint."""
    path = backup_path_by_name(backup_name, DEFAULT_BACKUP_DIR)
    if path is None:
        raise HTTPException(status_code=404, detail="Backup not found")
    return FileResponse(
        str(path),
        media_type="application/gzip",
        filename=path.name,
    )
