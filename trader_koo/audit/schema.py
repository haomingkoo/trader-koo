"""
Audit logging database schema and initialization.

This module defines the audit_logs table with partitioning support,
indexes for efficient querying, and retention policy management.
"""

import datetime as dt
import sqlite3
from datetime import timedelta
from typing import Any


def ensure_audit_schema(conn: sqlite3.Connection) -> None:
    """
    Create audit_logs table with indexes if it doesn't exist.

    Note: SQLite doesn't support table partitioning like PostgreSQL,
    but we implement logical partitioning through date-based queries
    and retention policies.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            event_type TEXT NOT NULL,
            user_id TEXT,
            username TEXT,
            resource TEXT,
            action TEXT,
            details TEXT,
            ip_address TEXT,
            user_agent TEXT,
            correlation_id TEXT,
            status_code INTEGER,
            request_method TEXT,
            request_path TEXT,
            response_time_ms REAL
        )
        """
    )

    # Create indexes for efficient querying
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp
        ON audit_logs(timestamp)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id
        ON audit_logs(user_id)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type
        ON audit_logs(event_type)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_audit_logs_correlation_id
        ON audit_logs(correlation_id)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_audit_logs_resource
        ON audit_logs(resource)
        """
    )

    conn.commit()


def apply_retention_policy(conn: sqlite3.Connection, retention_days: int = 90) -> int:
    """
    Delete audit logs older than retention_days.

    Args:
        conn: Database connection
        retention_days: Number of days to retain logs (default: 90)

    Returns:
        Number of rows deleted
    """
    cutoff_date = dt.datetime.now(dt.timezone.utc) - timedelta(days=retention_days)
    cutoff_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")

    cursor = conn.execute(
        """
        DELETE FROM audit_logs
        WHERE timestamp < ?
        """,
        (cutoff_str,)
    )

    deleted_count = cursor.rowcount
    conn.commit()

    return deleted_count


def get_audit_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """
    Get statistics about audit logs.

    Returns:
        Dictionary with total count, oldest entry, newest entry, size by event type
    """
    cursor = conn.execute(
        """
        SELECT
            COUNT(*) as total_count,
            MIN(timestamp) as oldest_entry,
            MAX(timestamp) as newest_entry
        FROM audit_logs
        """
    )

    row = cursor.fetchone()
    stats = {
        "total_count": row[0] if row else 0,
        "oldest_entry": row[1] if row else None,
        "newest_entry": row[2] if row else None,
    }

    # Get count by event type
    cursor = conn.execute(
        """
        SELECT event_type, COUNT(*) as count
        FROM audit_logs
        GROUP BY event_type
        ORDER BY count DESC
        """
    )

    stats["by_event_type"] = {row[0]: row[1] for row in cursor.fetchall()}

    return stats
