"""
External export functionality for audit logs.

Exports audit logs to the local filesystem for long-term retention and
compliance.
"""

import json
import os
import sqlite3
import datetime as dt
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

from trader_koo.audit.logger import AuditLogger


StorageType = Literal["local"]


class AuditExporter:
    """
    Export audit logs to external storage for long-term retention.
    """

    def __init__(
        self,
        storage_type: StorageType = "local",
        storage_config: dict[str, Any] | None = None,
    ):
        """
        Initialize audit exporter.

        Args:
            storage_type: Type of storage (local)
            storage_config: Storage-specific configuration
        """
        self.storage_type = storage_type
        self.storage_config = storage_config or {}

    def export_logs(
        self,
        logger: AuditLogger,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        export_format: str = "jsonl",
    ) -> dict[str, Any]:
        """
        Export audit logs to external storage.

        Args:
            logger: AuditLogger instance
            start_date: Export logs after this date
            end_date: Export logs before this date
            export_format: Format for export (jsonl or csv)

        Returns:
            Export result with location and metadata
        """
        # Query logs to export
        logs = logger.query_logs(
            start_date=start_date,
            end_date=end_date,
            limit=1000000,  # Large limit for full export
        )

        if not logs:
            return {
                "success": True,
                "message": "No logs to export",
                "records_exported": 0,
            }

        # Generate filename
        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"audit_logs_{timestamp}.{export_format}"

        # Prepare export data
        if export_format == "jsonl":
            export_data = "\n".join(json.dumps(log) for log in logs)
        elif export_format == "csv":
            export_data = self._to_csv(logs)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

        # Export to storage
        if self.storage_type == "local":
            location = self._export_to_local(filename, export_data)
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")

        return {
            "success": True,
            "location": location,
            "filename": filename,
            "records_exported": len(logs),
            "export_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "start_date": start_date,
            "end_date": end_date,
            "format": export_format,
        }

    def _to_csv(self, logs: list[dict[str, Any]]) -> str:
        """Convert logs to CSV format."""
        import csv
        import io

        output = io.StringIO()

        if logs:
            fieldnames = list(logs[0].keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            for log in logs:
                row = log.copy()
                # Convert dict fields to JSON strings
                if isinstance(row.get("details"), dict):
                    row["details"] = json.dumps(row["details"])
                writer.writerow(row)

        return output.getvalue()

    def _export_to_local(self, filename: str, data: str) -> str:
        """Export to local filesystem."""
        export_dir = Path(self.storage_config.get("path", "./audit_exports"))
        export_dir.mkdir(parents=True, exist_ok=True)

        filepath = export_dir / filename
        filepath.write_text(data, encoding="utf-8")

        return str(filepath)


def schedule_daily_export(
    logger: AuditLogger,
    exporter: AuditExporter,
    retention_days: int = 90,
) -> dict[str, Any]:
    """
    Export logs older than retention period to external storage.

    This should be called daily to archive old logs before they are deleted.

    Args:
        logger: AuditLogger instance
        exporter: AuditExporter instance
        retention_days: Days to retain in database

    Returns:
        Export result
    """
    # Export logs that are about to be deleted (older than retention - 1 day)
    cutoff_date = dt.datetime.now(dt.timezone.utc) - timedelta(days=retention_days - 1)
    start_date = (cutoff_date - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = cutoff_date.strftime("%Y-%m-%d")

    return exporter.export_logs(
        logger,
        start_date=start_date,
        end_date=end_date,
        export_format="jsonl",
    )


def get_exporter_from_env() -> AuditExporter:
    """
    Create AuditExporter from environment variables.

    Environment variables:
        AUDIT_EXPORT_PATH: Local path for local storage
    """
    config = {
        "path": os.getenv("AUDIT_EXPORT_PATH", "./audit_exports"),
    }

    return AuditExporter(storage_type="local", storage_config=config)
