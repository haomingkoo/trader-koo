#!/usr/bin/env python3
"""Download production data from Railway for local ML training.

Usage:
    # Download all default tables as SQLite (recommended):
    python -m trader_koo.scripts.sync_prod_data

    # Download specific tables:
    python -m trader_koo.scripts.sync_prod_data --tables yolo_patterns,paper_trades

    # Download as CSV files instead:
    python -m trader_koo.scripts.sync_prod_data --format csv

    # Upload a trained model to production:
    python -m trader_koo.scripts.sync_prod_data --upload-model data/models/swing_lgbm_latest.txt

    # Show what's available on prod:
    python -m trader_koo.scripts.sync_prod_data --info

Environment variables:
    TRADER_KOO_PROD_URL   - Production API base URL (default: https://trader.kooexperience.com)
    TRADER_KOO_API_KEY    - Admin API key (required)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROD_URL = os.getenv("TRADER_KOO_PROD_URL", "https://trader.kooexperience.com").rstrip("/")
API_KEY = os.getenv("TRADER_KOO_API_KEY", "")

LOCAL_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LOCAL_DB_PATH = LOCAL_DATA_DIR / "trader_koo_training.db"
LOCAL_CSV_DIR = LOCAL_DATA_DIR / "exports"

DEFAULT_TABLES = "yolo_patterns,price_daily,paper_trades"


def _headers() -> dict[str, str]:
    if not API_KEY:
        print("ERROR: TRADER_KOO_API_KEY environment variable is not set.")
        print("Set it with: export TRADER_KOO_API_KEY='your-key-here'")
        sys.exit(1)
    return {"X-API-Key": API_KEY}


def _api_get(path: str) -> urllib.request.Request:
    """Build an authenticated GET request."""
    url = f"{PROD_URL}{path}"
    req = urllib.request.Request(url, headers=_headers())
    return req


def _api_request(path: str, *, method: str = "GET") -> bytes:
    """Make an authenticated API request and return response bytes."""
    req = _api_get(path)
    req.method = method
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"ERROR: {exc.code} {exc.reason}")
        print(f"  URL: {PROD_URL}{path}")
        print(f"  Response: {body[:500]}")
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"ERROR: Cannot reach {PROD_URL}: {exc.reason}")
        sys.exit(1)


def cmd_info() -> None:
    """Show exportable tables and row counts from production."""
    data = json.loads(_api_request("/api/admin/export/info"))
    db_size_mb = data.get("db_size_bytes", 0) / (1024 * 1024)
    print(f"\nProduction DB size: {db_size_mb:.1f} MB")
    print(f"{'Table':<30} {'Rows':>10}  Columns")
    print("-" * 70)
    for table, info in sorted(data.get("tables", {}).items()):
        cols = ", ".join(info["columns"][:5])
        if len(info["columns"]) > 5:
            cols += f" ... (+{len(info['columns']) - 5} more)"
        print(f"{table:<30} {info['row_count']:>10}  {cols}")
    print()


def cmd_download_sqlite(tables: str) -> None:
    """Download selected tables as a SQLite database."""
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    path = f"/api/admin/export/sqlite?tables={tables}"
    print(f"Downloading tables: {tables}")
    print(f"  From: {PROD_URL}")

    start = time.time()
    data = _api_request(path)
    elapsed = time.time() - start

    LOCAL_DB_PATH.write_bytes(data)
    size_mb = len(data) / (1024 * 1024)
    print(f"  Saved: {LOCAL_DB_PATH} ({size_mb:.1f} MB, {elapsed:.1f}s)")

    # Verify the downloaded DB
    import sqlite3

    conn = sqlite3.connect(str(LOCAL_DB_PATH))
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    downloaded_tables = [row[0] for row in cursor.fetchall()]
    for tbl in downloaded_tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]  # noqa: S608
        print(f"  {tbl}: {count:,} rows")
    conn.close()
    print("Done.\n")


def cmd_download_csv(tables: str) -> None:
    """Download selected tables as individual CSV files."""
    LOCAL_CSV_DIR.mkdir(parents=True, exist_ok=True)

    for table in tables.split(","):
        table = table.strip()
        if not table:
            continue

        path = f"/api/admin/export/csv/{table}"
        print(f"Downloading {table}...", end=" ", flush=True)

        start = time.time()
        data = _api_request(path)
        elapsed = time.time() - start

        dest = LOCAL_CSV_DIR / f"{table}.csv"
        dest.write_bytes(data)

        lines = data.count(b"\n") - 1  # subtract header
        size_kb = len(data) / 1024
        print(f"{lines:,} rows, {size_kb:.0f} KB, {elapsed:.1f}s -> {dest}")

    print("Done.\n")


def cmd_upload_model(model_path: str) -> None:
    """Upload a trained model file to production."""
    file_path = Path(model_path)
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"Uploading {file_path.name} ({size_mb:.2f} MB) to {PROD_URL}")

    # Build multipart form data manually (no requests dependency)
    boundary = "----TraderKooModelUpload"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'.encode()
    )
    body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
    body.extend(file_path.read_bytes())
    body.extend(f"\r\n--{boundary}--\r\n".encode())

    url = f"{PROD_URL}/api/admin/upload/model"
    headers = _headers()
    headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

    req = urllib.request.Request(url, data=bytes(body), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            print(f"  Uploaded to: {result.get('path')}")
            print(f"  Size: {result.get('size_bytes', 0):,} bytes")
            print("Done.\n")
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        print(f"ERROR: {exc.code} {exc.reason}")
        print(f"  Response: {body_text[:500]}")
        sys.exit(1)


def cmd_list_models() -> None:
    """List model files on production."""
    data = json.loads(_api_request("/api/admin/models/list"))
    models = data.get("models", [])
    print(f"\nModels in {data.get('model_dir', '?')}:")
    if not models:
        print("  (none)")
    for m in models:
        size_kb = m["size_bytes"] / 1024
        print(f"  {m['name']:<40} {size_kb:>8.1f} KB")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync data between production Railway and local machine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tables",
        default=DEFAULT_TABLES,
        help=f"Comma-separated tables to download (default: {DEFAULT_TABLES})",
    )
    parser.add_argument(
        "--format",
        choices=["sqlite", "csv"],
        default="sqlite",
        help="Download format (default: sqlite)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show production table info and exit",
    )
    parser.add_argument(
        "--upload-model",
        metavar="PATH",
        help="Upload a trained model file to production",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List model files on production",
    )

    args = parser.parse_args()

    if args.info:
        cmd_info()
        return

    if args.list_models:
        cmd_list_models()
        return

    if args.upload_model:
        cmd_upload_model(args.upload_model)
        return

    if args.format == "sqlite":
        cmd_download_sqlite(args.tables)
    else:
        cmd_download_csv(args.tables)


if __name__ == "__main__":
    main()
