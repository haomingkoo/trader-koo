"""
Audit logging service for tracking system events and user actions.

This module provides immutable append-only audit logging for:
- Admin API requests
- Authentication attempts (success and failure)
- Data modifications
"""

import json
import sqlite3
import uuid
from datetime import datetime
from enum import Enum
from typing import Any


class AuditEventType(Enum):
    """Types of audit events."""
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    API_REQUEST = "api_request"
    DATA_MODIFICATION = "data_modification"
    ADMIN_ACTION = "admin_action"
    EXPORT = "export"
    WEBHOOK_TRIGGER = "webhook_trigger"
    CONFIG_CHANGE = "config_change"
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"


class AuditLogger:
    """
    Immutable append-only audit logger.
    
    All log entries are write-only - no updates or deletes allowed
    except through retention policy cleanup.
    """
    
    def __init__(self, conn: sqlite3.Connection):
        """
        Initialize audit logger with database connection.
        
        Args:
            conn: SQLite database connection
        """
        self.conn = conn
    
    def log_event(
        self,
        event_type: AuditEventType | str,
        *,
        user_id: str | None = None,
        username: str | None = None,
        resource: str | None = None,
        action: str | None = None,
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        correlation_id: str | None = None,
        status_code: int | None = None,
        request_method: str | None = None,
        request_path: str | None = None,
        response_time_ms: float | None = None,
    ) -> str:
        """
        Log an audit event (append-only, immutable).
        
        Args:
            event_type: Type of event (from AuditEventType enum or string)
            user_id: User identifier (if applicable)
            username: Username (if applicable)
            resource: Resource being accessed/modified
            action: Action being performed
            details: Additional event details (will be JSON serialized)
            ip_address: Client IP address
            user_agent: Client user agent string
            correlation_id: Request correlation ID for tracing
            status_code: HTTP status code (for API requests)
            request_method: HTTP method (for API requests)
            request_path: Request path (for API requests)
            response_time_ms: Response time in milliseconds
            
        Returns:
            Correlation ID for this event
        """
        # Convert enum to string if needed
        if isinstance(event_type, AuditEventType):
            event_type = event_type.value
        
        # Generate correlation ID if not provided
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        
        # Serialize details to JSON
        details_json = json.dumps(details) if details else None
        
        # Insert audit log entry (immutable - no UPDATE or DELETE)
        self.conn.execute(
            """
            INSERT INTO audit_logs (
                timestamp,
                event_type,
                user_id,
                username,
                resource,
                action,
                details,
                ip_address,
                user_agent,
                correlation_id,
                status_code,
                request_method,
                request_path,
                response_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
                event_type,
                user_id,
                username,
                resource,
                action,
                details_json,
                ip_address,
                user_agent,
                correlation_id,
                status_code,
                request_method,
                request_path,
                response_time_ms,
            ),
        )
        
        self.conn.commit()
        
        return correlation_id
    
    def log_auth_success(
        self,
        user_id: str,
        username: str,
        ip_address: str,
        user_agent: str | None = None,
        auth_method: str = "api_key",
    ) -> str:
        """Log successful authentication attempt."""
        return self.log_event(
            AuditEventType.AUTH_SUCCESS,
            user_id=user_id,
            username=username,
            action=f"login_{auth_method}",
            ip_address=ip_address,
            user_agent=user_agent,
            details={"auth_method": auth_method},
        )
    
    def log_auth_failure(
        self,
        username: str | None,
        ip_address: str,
        user_agent: str | None = None,
        reason: str = "invalid_credentials",
    ) -> str:
        """Log failed authentication attempt."""
        return self.log_event(
            AuditEventType.AUTH_FAILURE,
            username=username,
            action="login_failed",
            ip_address=ip_address,
            user_agent=user_agent,
            details={"reason": reason},
        )
    
    def log_api_request(
        self,
        request_method: str,
        request_path: str,
        status_code: int,
        response_time_ms: float,
        *,
        user_id: str | None = None,
        username: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        correlation_id: str | None = None,
        request_params: dict[str, Any] | None = None,
    ) -> str:
        """Log API request (especially admin API requests)."""
        return self.log_event(
            AuditEventType.API_REQUEST,
            user_id=user_id,
            username=username,
            resource=request_path,
            action=request_method,
            ip_address=ip_address,
            user_agent=user_agent,
            correlation_id=correlation_id,
            status_code=status_code,
            request_method=request_method,
            request_path=request_path,
            response_time_ms=response_time_ms,
            details={"params": request_params} if request_params else None,
        )
    
    def log_data_modification(
        self,
        table: str,
        operation: str,
        record_id: str | None = None,
        *,
        user_id: str | None = None,
        username: str | None = None,
        before_values: dict[str, Any] | None = None,
        after_values: dict[str, Any] | None = None,
        ip_address: str | None = None,
    ) -> str:
        """Log data modification (INSERT, UPDATE, DELETE)."""
        return self.log_event(
            AuditEventType.DATA_MODIFICATION,
            user_id=user_id,
            username=username,
            resource=table,
            action=operation,
            ip_address=ip_address,
            details={
                "record_id": record_id,
                "before": before_values,
                "after": after_values,
            },
        )
    
    def log_admin_action(
        self,
        action: str,
        resource: str,
        user_id: str,
        username: str,
        *,
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
    ) -> str:
        """Log administrative action."""
        return self.log_event(
            AuditEventType.ADMIN_ACTION,
            user_id=user_id,
            username=username,
            resource=resource,
            action=action,
            ip_address=ip_address,
            details=details,
        )
    
    def query_logs(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        user_id: str | None = None,
        event_type: str | None = None,
        resource: str | None = None,
        action: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Query audit logs with filtering.
        
        Args:
            start_date: Filter logs after this date (ISO format)
            end_date: Filter logs before this date (ISO format)
            user_id: Filter by user ID
            event_type: Filter by event type
            resource: Filter by resource
            action: Filter by action
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of audit log entries
        """
        query = "SELECT * FROM audit_logs WHERE 1=1"
        params: list[Any] = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        if resource:
            query += " AND resource = ?"
            params.append(resource)
        
        if action:
            query += " AND action = ?"
            params.append(action)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor = self.conn.execute(query, params)
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            entry = dict(zip(columns, row))
            
            # Parse JSON details if present
            if entry.get("details"):
                try:
                    entry["details"] = json.loads(entry["details"])
                except json.JSONDecodeError:
                    pass
            
            results.append(entry)
        
        return results
    
    def count_logs(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        user_id: str | None = None,
        event_type: str | None = None,
    ) -> int:
        """Count audit logs matching filters."""
        query = "SELECT COUNT(*) FROM audit_logs WHERE 1=1"
        params: list[Any] = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        cursor = self.conn.execute(query, params)
        return cursor.fetchone()[0]
