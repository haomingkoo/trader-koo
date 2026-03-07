"""
API endpoints for audit log querying and management.
"""

from datetime import datetime, timedelta
from typing import Any

from trader_koo.audit.logger import AuditLogger


def query_audit_logs(
    logger: AuditLogger,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    user_id: str | None = None,
    event_type: str | None = None,
    resource: str | None = None,
    action: str | None = None,
    page: int = 1,
    per_page: int = 50,
) -> dict[str, Any]:
    """
    Query audit logs with filtering and pagination.
    
    Args:
        logger: AuditLogger instance
        start_date: Filter logs after this date (ISO format)
        end_date: Filter logs before this date (ISO format)
        user_id: Filter by user ID
        event_type: Filter by event type
        resource: Filter by resource
        action: Filter by action
        page: Page number (1-indexed)
        per_page: Results per page
        
    Returns:
        Dictionary with logs and pagination info
    """
    # Validate pagination
    page = max(1, page)
    per_page = min(max(1, per_page), 200)  # Cap at 200
    offset = (page - 1) * per_page
    
    # Query logs
    logs = logger.query_logs(
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        event_type=event_type,
        resource=resource,
        action=action,
        limit=per_page,
        offset=offset,
    )
    
    # Get total count
    total = logger.count_logs(
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        event_type=event_type,
    )
    
    total_pages = (total + per_page - 1) // per_page
    
    return {
        "logs": logs,
        "pagination": {
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        },
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "user_id": user_id,
            "event_type": event_type,
            "resource": resource,
            "action": action,
        },
    }


def export_audit_logs(
    logger: AuditLogger,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    user_id: str | None = None,
    event_type: str | None = None,
    format: str = "json",
) -> dict[str, Any] | str:
    """
    Export audit logs in specified format.
    
    Args:
        logger: AuditLogger instance
        start_date: Filter logs after this date
        end_date: Filter logs before this date
        user_id: Filter by user ID
        event_type: Filter by event type
        format: Export format (json or csv)
        
    Returns:
        Exported data in requested format
    """
    # Query all matching logs (no pagination for export)
    logs = logger.query_logs(
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        event_type=event_type,
        limit=100000,  # Large limit for export
        offset=0,
    )
    
    if format == "csv":
        # Convert to CSV format
        import csv
        import io
        
        output = io.StringIO()
        
        if logs:
            fieldnames = list(logs[0].keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for log in logs:
                # Convert details dict to string for CSV
                row = log.copy()
                if isinstance(row.get("details"), dict):
                    import json
                    row["details"] = json.dumps(row["details"])
                writer.writerow(row)
        
        return output.getvalue()
    
    # Default to JSON
    return {
        "logs": logs,
        "export_timestamp": datetime.utcnow().isoformat(),
        "filters": {
            "start_date": start_date,
            "end_date": end_date,
            "user_id": user_id,
            "event_type": event_type,
        },
        "total_records": len(logs),
    }


def get_audit_summary(logger: AuditLogger, days: int = 7) -> dict[str, Any]:
    """
    Get summary statistics for audit logs.
    
    Args:
        logger: AuditLogger instance
        days: Number of days to include in summary
        
    Returns:
        Summary statistics
    """
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    # Get all logs for the period
    logs = logger.query_logs(
        start_date=start_date,
        limit=100000,
    )
    
    # Calculate statistics
    event_type_counts: dict[str, int] = {}
    user_activity: dict[str, int] = {}
    resource_access: dict[str, int] = {}
    hourly_activity: dict[int, int] = {}
    
    for log in logs:
        # Count by event type
        event_type = log.get("event_type", "unknown")
        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        
        # Count by user
        user_id = log.get("user_id")
        if user_id:
            user_activity[user_id] = user_activity.get(user_id, 0) + 1
        
        # Count by resource
        resource = log.get("resource")
        if resource:
            resource_access[resource] = resource_access.get(resource, 0) + 1
        
        # Count by hour
        timestamp = log.get("timestamp", "")
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                hour = dt.hour
                hourly_activity[hour] = hourly_activity.get(hour, 0) + 1
            except (ValueError, AttributeError):
                pass
    
    return {
        "period_days": days,
        "start_date": start_date,
        "total_events": len(logs),
        "event_type_counts": event_type_counts,
        "top_users": sorted(
            user_activity.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10],
        "top_resources": sorted(
            resource_access.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10],
        "hourly_activity": hourly_activity,
    }
