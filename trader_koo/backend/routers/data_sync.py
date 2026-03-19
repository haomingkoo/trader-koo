"""Data sync endpoints: export tables and upload model files.

Provides admin-only endpoints for:
- Exporting SQLite tables as CSV (for local ML training)
- Downloading a full SQLite dump of selected tables
- Uploading trained model files to the persistent volume

All endpoints require X-API-Key authentication via the standard admin middleware.
"""
from __future__ import annotations

import io
import logging
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, UploadFile, File
from fastapi.responses import StreamingResponse

from trader_koo.backend.services.database import DB_PATH, get_conn, table_exists
from trader_koo.middleware.auth import require_admin_auth

router = APIRouter()
LOG = logging.getLogger("trader_koo.routers.data_sync")

# Tables that are safe to export (whitelist approach)
EXPORTABLE_TABLES = {
    "yolo_patterns",
    "price_daily",
    "paper_trades",
    "paper_portfolio_snapshots",
    "options_iv",
    "fundamentals",
}

MODEL_DIR = Path(os.getenv("TRADER_KOO_MODEL_DIR", "/data/models"))
LOCAL_MODEL_DIR = Path(__file__).resolve().parents[3] / "data" / "models"

# 50 MB upload limit for model files
MAX_MODEL_SIZE = 50 * 1024 * 1024


def _model_dir() -> Path:
    """Return writable model directory (Railway or local fallback)."""
    if MODEL_DIR.exists():
        return MODEL_DIR
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return LOCAL_MODEL_DIR


# ---------------------------------------------------------------------------
# Export: CSV
# ---------------------------------------------------------------------------

@router.get("/api/admin/export/csv/{table_name}")
@require_admin_auth
def export_table_csv(
    request: Request,
    table_name: str,
    limit: int = Query(default=0, ge=0, description="Row limit (0 = all)"),
    where: str = Query(default="", description="Optional WHERE clause, e.g. ticker='AAPL'"),
) -> StreamingResponse:
    """Export a single table as CSV.

    Only tables in the EXPORTABLE_TABLES whitelist are allowed.
    The optional `where` parameter accepts simple column filters.
    """
    if table_name not in EXPORTABLE_TABLES:
        raise HTTPException(
            status_code=400,
            detail=f"Table '{table_name}' not exportable. Allowed: {sorted(EXPORTABLE_TABLES)}",
        )

    conn = get_conn()
    try:
        if not table_exists(conn, table_name):
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' does not exist")

        query = f"SELECT * FROM {table_name}"  # noqa: S608 — table_name is whitelisted
        params: list[str] = []

        if where:
            # Basic safety: reject anything that looks like SQL injection
            dangerous = {"drop", "delete", "update", "insert", "alter", "create", "--", ";"}
            if any(tok in where.lower() for tok in dangerous):
                raise HTTPException(status_code=400, detail="Invalid WHERE clause")
            query += f" WHERE {where}"

        if limit > 0:
            query += f" LIMIT {limit}"

        cursor = conn.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        # Build CSV in memory
        buf = io.StringIO()
        buf.write(",".join(columns) + "\n")
        for row in rows:
            line = ",".join(
                _csv_escape(str(val)) if val is not None else ""
                for val in row
            )
            buf.write(line + "\n")

        buf.seek(0)
        LOG.info("Exported %d rows from %s (where=%r, limit=%d)", len(rows), table_name, where, limit)

        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{table_name}.csv"'},
        )
    finally:
        conn.close()


def _csv_escape(value: str) -> str:
    """Escape a CSV field if it contains commas, quotes, or newlines."""
    if "," in value or '"' in value or "\n" in value:
        return '"' + value.replace('"', '""') + '"'
    return value


# ---------------------------------------------------------------------------
# Export: SQLite dump (selected tables only)
# ---------------------------------------------------------------------------

@router.get("/api/admin/export/sqlite")
@require_admin_auth
def export_sqlite_dump(
    request: Request,
    tables: str = Query(
        default="yolo_patterns,price_daily,paper_trades",
        description="Comma-separated table names to include",
    ),
) -> StreamingResponse:
    """Export selected tables as a standalone SQLite database file.

    Creates a temporary SQLite file, copies the requested tables into it,
    and streams it back. This is the fastest way to get a training-ready DB.
    """
    requested = [t.strip() for t in tables.split(",") if t.strip()]
    invalid = [t for t in requested if t not in EXPORTABLE_TABLES]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Tables not exportable: {invalid}. Allowed: {sorted(EXPORTABLE_TABLES)}",
        )

    conn = get_conn()
    try:
        # Create a temp file for the export DB
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp_path = tmp.name
        tmp.close()

        export_conn = sqlite3.connect(tmp_path)
        try:
            conn.execute("ATTACH DATABASE ? AS export_db", (tmp_path,))

            for tbl in requested:
                if not table_exists(conn, tbl):
                    LOG.warning("Skipping non-existent table: %s", tbl)
                    continue

                # Get the CREATE TABLE statement
                create_sql = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (tbl,),
                ).fetchone()
                if not create_sql or not create_sql[0]:
                    continue

                # Create table in export DB and copy data
                export_conn.execute(create_sql[0])
                rows = conn.execute(f"SELECT * FROM {tbl}").fetchall()  # noqa: S608
                if rows:
                    cols = [desc[0] for desc in conn.execute(f"SELECT * FROM {tbl} LIMIT 0").description]  # noqa: S608
                    placeholders = ",".join("?" for _ in cols)
                    export_conn.executemany(
                        f"INSERT INTO {tbl} VALUES ({placeholders})",  # noqa: S608
                        rows,
                    )

            export_conn.commit()
            row_counts = {}
            for tbl in requested:
                try:
                    cnt = export_conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]  # noqa: S608
                    row_counts[tbl] = cnt
                except Exception:
                    row_counts[tbl] = 0

            LOG.info("SQLite export complete: %s", row_counts)

        finally:
            export_conn.close()
            try:
                conn.execute("DETACH DATABASE export_db")
            except Exception:
                pass

        # Stream the file
        def file_iter():
            with open(tmp_path, "rb") as f:
                while chunk := f.read(64 * 1024):
                    yield chunk
            os.unlink(tmp_path)

        return StreamingResponse(
            file_iter(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="trader_koo_export.db"'},
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Export: table metadata (for sync script to know what's available)
# ---------------------------------------------------------------------------

@router.get("/api/admin/export/info")
@require_admin_auth
def export_info(request: Request) -> dict[str, Any]:
    """Return metadata about exportable tables: row counts and column names."""
    conn = get_conn()
    try:
        info: dict[str, Any] = {}
        for tbl in sorted(EXPORTABLE_TABLES):
            if not table_exists(conn, tbl):
                continue
            count = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]  # noqa: S608
            cols = [
                row[1] for row in conn.execute(f"PRAGMA table_info({tbl})").fetchall()
            ]
            info[tbl] = {"row_count": count, "columns": cols}
        return {"tables": info, "db_size_bytes": DB_PATH.stat().st_size if DB_PATH.exists() else 0}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Upload: model files
# ---------------------------------------------------------------------------

@router.post("/api/admin/upload/model")
@require_admin_auth
async def upload_model(
    request: Request,
    file: UploadFile = File(...),
    filename: str = Query(
        default="",
        description="Override filename (default: use uploaded name)",
    ),
) -> dict[str, Any]:
    """Upload a trained model file to the persistent volume.

    Accepted file extensions: .txt (LightGBM), .json (metadata).
    Files are saved to /data/models/ on Railway.
    """
    allowed_extensions = {".txt", ".json", ".pkl", ".joblib"}
    original_name = file.filename or "model.txt"
    target_name = filename.strip() or original_name

    ext = Path(target_name).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File extension '{ext}' not allowed. Accepted: {sorted(allowed_extensions)}",
        )

    content = await file.read()
    if len(content) > MAX_MODEL_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)} bytes). Max: {MAX_MODEL_SIZE} bytes",
        )

    dest_dir = _model_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / target_name

    dest_path.write_bytes(content)
    LOG.info("Model uploaded: %s (%d bytes)", dest_path, len(content))

    return {
        "ok": True,
        "path": str(dest_path),
        "filename": target_name,
        "size_bytes": len(content),
    }


@router.get("/api/admin/models/list")
@require_admin_auth
def list_models(request: Request) -> dict[str, Any]:
    """List all model files on the persistent volume."""
    dest_dir = _model_dir()
    if not dest_dir.exists():
        return {"models": [], "model_dir": str(dest_dir)}

    files = []
    for p in sorted(dest_dir.iterdir()):
        if p.is_file():
            files.append({
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "modified": p.stat().st_mtime,
            })
    return {"models": files, "model_dir": str(dest_dir)}
