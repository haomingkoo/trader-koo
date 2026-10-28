"""Boot preflight for start.sh: refuse to start on a missing volume or bad DB.

Prints one word on stdout for start.sh to act on:

``bootstrap``
    The DB file is absent and an empty bootstrap is allowed (local dev, or
    production with ``TRADER_KOO_ALLOW_EMPTY_BOOTSTRAP=1`` on a mounted volume).
``ready``
    The DB opened read-only and passed ``PRAGMA quick_check(1)``.
``maintenance``
    The maintenance sidecar has an active run (or is unreadable, or its
    ``.required`` marker exists without it), so the app will boot
    maintenance-only. The live DB is not opened: it may be corrupt or moved
    aside mid-restore, and the operator must be able to SSH in to recover it.

Any other condition prints the reason on stderr and exits 1. The preflight
never writes database contents. Opening a WAL database read-only can still
create its ``-shm`` sidecar when that file is missing.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Callable, Mapping

from trader_koo.backend.services.maintenance import ACTIVE_STATES, _required_marker, state_path

ALLOW_EMPTY_ENV = "TRADER_KOO_ALLOW_EMPTY_BOOTSTRAP"
DB_BUSY_TIMEOUT_SEC = 30.0
MAINTENANCE_STATE_TIMEOUT_SEC = 5.0  # matches maintenance._connect


class PreflightError(RuntimeError):
    """The service must not start."""


def _log(message: str) -> None:
    print(f"[start_preflight] {message}", file=sys.stderr)


def is_production(environ: Mapping[str, str]) -> bool:
    return bool(environ.get("RAILWAY_ENVIRONMENT") or environ.get("RAILWAY_VOLUME_MOUNT_PATH"))


def _verify_volume(db_path: Path, environ: Mapping[str, str], ismount: Callable[[str], bool]) -> None:
    mount = environ.get("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    if not mount:
        raise PreflightError(
            "production boot but no volume is attached (RAILWAY_VOLUME_MOUNT_PATH is unset)"
        )
    mount_real = os.path.realpath(mount)
    if not ismount(mount_real):
        raise PreflightError(f"volume path {mount_real} is not a mount point")
    db_dir_real = os.path.realpath(db_path.parent)
    if os.path.commonpath([mount_real, db_dir_real]) != mount_real:
        raise PreflightError(f"DB directory {db_dir_real} is outside the volume {mount_real}")
    if os.path.isdir(db_dir_real) and os.stat(db_dir_real).st_dev != os.stat(mount_real).st_dev:
        raise PreflightError(f"DB directory {db_dir_real} is on a different device from the volume")


def _maintenance_reason(db_path: Path) -> str | None:
    """Mirror main.lifespan's maintenance-boot decision without creating the sidecar."""
    path = state_path(db_path)
    if path.is_symlink():
        return f"maintenance state {path} is a symlink"
    if not path.exists():
        if _required_marker(db_path).exists():
            return f"maintenance marker exists but state {path} is missing"
        return None
    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=MAINTENANCE_STATE_TIMEOUT_SEC)
        try:
            if not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'maintenance_runs'"
            ).fetchone():
                return None
            row = conn.execute(
                "SELECT run_id, state FROM maintenance_runs ORDER BY requested_ts DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return f"maintenance state {path} is unreadable ({exc})"
    if row is not None and row[1] in ACTIVE_STATES:
        return f"maintenance run {row[0]} is {row[1]}"
    return None


def _inspect_db(db_path: Path) -> int | None:
    """Return the price_daily row count, or None when the table is absent."""
    try:
        conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=DB_BUSY_TIMEOUT_SEC)
        try:
            check = conn.execute("PRAGMA quick_check(1)").fetchone()
            if check != ("ok",):
                raise PreflightError(f"PRAGMA quick_check failed on {db_path}: {check}")
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'price_daily'"
            ).fetchone()
            if not has_table:
                return None
            return int(conn.execute("SELECT COUNT(*) FROM price_daily").fetchone()[0])
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise PreflightError(f"cannot read DB {db_path}: {exc}") from exc


def preflight(
    db_path: Path,
    *,
    environ: Mapping[str, str] = os.environ,
    ismount: Callable[[str], bool] = os.path.ismount,
) -> str:
    production = is_production(environ)
    allow_empty = environ.get(ALLOW_EMPTY_ENV, "").strip() == "1"
    if production:
        _verify_volume(db_path, environ, ismount)

    maintenance = _maintenance_reason(db_path)
    if maintenance:
        _log(f"{maintenance}; skipping live DB checks for a maintenance-only boot")
        return "maintenance"

    if not db_path.exists():
        if production and not allow_empty:
            raise PreflightError(
                f"DB not found at {db_path} on the production volume; refusing to boot an "
                f"empty database. Restore the volume, or set {ALLOW_EMPTY_ENV}=1 for a "
                "deliberate first bootstrap."
            )
        _log(f"DB not found at {db_path}; empty bootstrap allowed")
        return "bootstrap"
    if not db_path.is_file():
        raise PreflightError(f"DB path {db_path} is not a regular file")

    started = time.monotonic()
    rows = _inspect_db(db_path)
    elapsed = time.monotonic() - started
    if rows is None:
        problem = f"DB at {db_path} has no price_daily table"
    elif rows == 0:
        problem = f"DB at {db_path} has 0 price_daily rows"
    else:
        _log(f"DB at {db_path} passed quick_check in {elapsed:.1f}s with {rows} price_daily rows")
        return "ready"
    if production and not allow_empty:
        raise PreflightError(f"{problem}; set {ALLOW_EMPTY_ENV}=1 only while a bootstrap seed is expected")
    _log(f"WARNING: {problem}; continuing (no seed is started for an existing DB)")
    return "ready"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-path", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        print(preflight(args.db_path))
    except PreflightError as exc:
        _log(f"FATAL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
