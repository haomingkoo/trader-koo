"""Admin backup endpoints — list and download SQLite backups."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import FileResponse

from trader_koo.middleware.auth import require_admin_auth
from trader_koo.scripts.backup_db import (
    DEFAULT_BACKUP_DIR,
    latest_backup_path,
    list_backups,
)

router = APIRouter(tags=["admin", "admin-backups"])


@router.get("/api/admin/backups")
@require_admin_auth
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
@require_admin_auth
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
