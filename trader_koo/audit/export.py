"""
External export functionality for audit logs.

Supports exporting audit logs to S3, Azure Blob Storage, or local filesystem
for long-term retention and compliance.
"""

import json
import os
import sqlite3
import datetime as dt
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

from trader_koo.audit.logger import AuditLogger


StorageType = Literal["s3", "azure_blob", "local"]


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
            storage_type: Type of storage (s3, azure_blob, local)
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
        elif self.storage_type == "s3":
            location = self._export_to_s3(filename, export_data)
        elif self.storage_type == "azure_blob":
            location = self._export_to_azure(filename, export_data)
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

    def _export_to_s3(self, filename: str, data: str) -> str:
        """Export to AWS S3."""
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for S3 export. Install with: pip install boto3")

        bucket = self.storage_config.get("bucket")
        if not bucket:
            raise ValueError("S3 bucket not configured")

        prefix = self.storage_config.get("prefix", "audit_logs")
        key = f"{prefix}/{filename}"

        s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.storage_config.get("access_key_id"),
            aws_secret_access_key=self.storage_config.get("secret_access_key"),
            region_name=self.storage_config.get("region", "us-east-1"),
        )

        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data.encode("utf-8"),
            ContentType="application/json" if filename.endswith(".jsonl") else "text/csv",
        )

        return f"s3://{bucket}/{key}"

    def _export_to_azure(self, filename: str, data: str) -> str:
        """Export to Azure Blob Storage."""
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError:
            raise ImportError(
                "azure-storage-blob is required for Azure export. "
                "Install with: pip install azure-storage-blob"
            )

        connection_string = self.storage_config.get("connection_string")
        if not connection_string:
            raise ValueError("Azure connection string not configured")

        container = self.storage_config.get("container", "audit-logs")
        prefix = self.storage_config.get("prefix", "")
        blob_name = f"{prefix}/{filename}" if prefix else filename

        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(
            container=container,
            blob=blob_name,
        )

        blob_client.upload_blob(data.encode("utf-8"), overwrite=True)

        return f"azure://{container}/{blob_name}"


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
        AUDIT_EXPORT_STORAGE: Storage type (s3, azure_blob, local)
        AUDIT_EXPORT_PATH: Local path for local storage
        AUDIT_EXPORT_S3_BUCKET: S3 bucket name
        AUDIT_EXPORT_S3_PREFIX: S3 key prefix
        AUDIT_EXPORT_S3_REGION: AWS region
        AWS_ACCESS_KEY_ID: AWS access key
        AWS_SECRET_ACCESS_KEY: AWS secret key
        AUDIT_EXPORT_AZURE_CONNECTION_STRING: Azure connection string
        AUDIT_EXPORT_AZURE_CONTAINER: Azure container name
    """
    storage_type = os.getenv("AUDIT_EXPORT_STORAGE", "local")

    if storage_type == "s3":
        config = {
            "bucket": os.getenv("AUDIT_EXPORT_S3_BUCKET"),
            "prefix": os.getenv("AUDIT_EXPORT_S3_PREFIX", "audit_logs"),
            "region": os.getenv("AUDIT_EXPORT_S3_REGION", "us-east-1"),
            "access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        }
    elif storage_type == "azure_blob":
        config = {
            "connection_string": os.getenv("AUDIT_EXPORT_AZURE_CONNECTION_STRING"),
            "container": os.getenv("AUDIT_EXPORT_AZURE_CONTAINER", "audit-logs"),
            "prefix": os.getenv("AUDIT_EXPORT_AZURE_PREFIX", ""),
        }
    else:  # local
        config = {
            "path": os.getenv("AUDIT_EXPORT_PATH", "./audit_exports"),
        }

    return AuditExporter(storage_type=storage_type, storage_config=config)
