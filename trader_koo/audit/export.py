"""Export audit logs to the configured local retention directory."""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import os
from pathlib import Path
from typing import Any

from trader_koo.audit.logger import AuditLogger

EXPORT_LIMIT = 1_000_000


def export_logs_to_local(
    logger: AuditLogger,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    export_format: str = "jsonl",
    export_dir: Path | None = None,
) -> dict[str, Any]:
    """Write matching audit logs as JSON Lines or CSV."""
    logs = logger.query_logs(
        start_date=start_date,
        end_date=end_date,
        limit=EXPORT_LIMIT,
    )
    if not logs:
        return {
            "success": True,
            "message": "No logs to export",
            "records_exported": 0,
        }

    if export_format == "jsonl":
        export_data = "\n".join(json.dumps(log) for log in logs)
    elif export_format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(logs[0]))
        writer.writeheader()
        for log in logs:
            row = log.copy()
            if isinstance(row.get("details"), dict):
                row["details"] = json.dumps(row["details"])
            writer.writerow(row)
        export_data = output.getvalue()
    else:
        raise ValueError(f"Unsupported export format: {export_format}")

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"audit_logs_{timestamp}.{export_format}"
    target_dir = export_dir or Path(
        os.getenv("AUDIT_EXPORT_PATH", "./audit_exports")
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    location = target_dir / filename
    location.write_text(export_data, encoding="utf-8")

    return {
        "success": True,
        "location": str(location),
        "filename": filename,
        "records_exported": len(logs),
        "export_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "start_date": start_date,
        "end_date": end_date,
        "format": export_format,
    }
