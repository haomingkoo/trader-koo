"""Offline operator helper for paper-schema maintenance recovery.

Run only after the authenticated admin request has persisted maintenance intent
and the application has restarted in maintenance-only mode. No command here
enables writes or activates a campaign. `migrate-copy` runs the reviewed v4 to
v5 migration only after the recorded complete decision and verified backup.
"""
from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from trader_koo.backend.services.database import DB_PATH
from trader_koo.backend.services.maintenance import (
    decide, migrate_verified_copy, quiesce_backup, restore_backup, status,
    verify_resolution,
)
from trader_koo.scripts.backup_db import DEFAULT_BACKUP_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=(
            "status", "quiesce-backup", "decide", "migrate-copy",
            "restore-backup", "verify-resolution",
        ),
    )
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--run-id")
    parser.add_argument("--boot-id", default=secrets.token_hex(16))
    parser.add_argument("--decision", choices=("restore", "complete"))
    parser.add_argument("--reason")
    args = parser.parse_args()
    if args.action == "status":
        result = status(args.db_path, args.run_id)
    elif not args.run_id:
        parser.error("--run-id is required")
    elif args.action == "quiesce-backup":
        result = quiesce_backup(
            args.db_path, run_id=args.run_id, boot_id=args.boot_id,
            backup_dir=args.backup_dir,
        )
    elif args.action == "decide":
        if not args.decision or not args.reason:
            parser.error("--decision and --reason are required")
        result = decide(args.db_path, run_id=args.run_id, decision=args.decision, reason=args.reason)
    elif args.action == "restore-backup":
        result = restore_backup(args.db_path, run_id=args.run_id)
    elif args.action == "migrate-copy":
        result = migrate_verified_copy(args.db_path, run_id=args.run_id)
    else:
        result = verify_resolution(args.db_path, run_id=args.run_id)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
