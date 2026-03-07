"""
Unit tests for audit logging functionality.

Tests cover:
- Log creation (Requirement 15.1, 15.2, 15.3)
- Immutability (Requirement 15.4)
- Querying (Requirement 15.6)
- Retention policy (Requirement 15.7)
"""

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from trader_koo.audit import (
    AuditEventType,
    AuditLogger,
    apply_retention_policy,
    ensure_audit_schema,
    get_audit_stats,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    
    conn = sqlite3.connect(str(db_path))
    ensure_audit_schema(conn)
    
    yield conn
    
    conn.close()
    db_path.unlink()


@pytest.fixture
def audit_logger(temp_db):
    """Create an AuditLogger instance."""
    return AuditLogger(temp_db)


class TestAuditLogCreation:
    """Test audit log creation and basic functionality."""
    
    def test_log_event_creates_entry(self, audit_logger):
        """Test that log_event creates an audit log entry."""
        correlation_id = audit_logger.log_event(
            AuditEventType.API_REQUEST,
            user_id="user123",
            username="testuser",
            resource="/api/admin/test",
            action="GET",
            ip_address="192.168.1.1",
        )
        
        assert correlation_id is not None
        
        # Verify entry was created
        logs = audit_logger.query_logs(limit=1)
        assert len(logs) == 1
        assert logs[0]["event_type"] == "api_request"
        assert logs[0]["user_id"] == "user123"
        assert logs[0]["username"] == "testuser"
        assert logs[0]["resource"] == "/api/admin/test"
        assert logs[0]["action"] == "GET"
        assert logs[0]["ip_address"] == "192.168.1.1"
    
    def test_log_auth_success(self, audit_logger):
        """Test logging successful authentication."""
        correlation_id = audit_logger.log_auth_success(
            user_id="user123",
            username="testuser",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            auth_method="api_key",
        )
        
        assert correlation_id is not None
        
        logs = audit_logger.query_logs(event_type="auth_success", limit=1)
        assert len(logs) == 1
        assert logs[0]["event_type"] == "auth_success"
        assert logs[0]["user_id"] == "user123"
        assert logs[0]["action"] == "login_api_key"
    
    def test_log_auth_failure(self, audit_logger):
        """Test logging failed authentication."""
        correlation_id = audit_logger.log_auth_failure(
            username="testuser",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            reason="invalid_credentials",
        )
        
        assert correlation_id is not None
        
        logs = audit_logger.query_logs(event_type="auth_failure", limit=1)
        assert len(logs) == 1
        assert logs[0]["event_type"] == "auth_failure"
        assert logs[0]["username"] == "testuser"
        assert logs[0]["action"] == "login_failed"
        
        # Check details
        details = logs[0]["details"]
        assert details["reason"] == "invalid_credentials"
    
    def test_log_api_request(self, audit_logger):
        """Test logging API requests."""
        correlation_id = audit_logger.log_api_request(
            request_method="POST",
            request_path="/api/admin/users",
            status_code=201,
            response_time_ms=45.2,
            user_id="admin",
            username="admin",
            ip_address="192.168.1.1",
            request_params={"role": "analyst"},
        )
        
        assert correlation_id is not None
        
        logs = audit_logger.query_logs(event_type="api_request", limit=1)
        assert len(logs) == 1
        assert logs[0]["request_method"] == "POST"
        assert logs[0]["request_path"] == "/api/admin/users"
        assert logs[0]["status_code"] == 201
        assert logs[0]["response_time_ms"] == 45.2
        assert logs[0]["details"]["params"]["role"] == "analyst"
    
    def test_log_data_modification(self, audit_logger):
        """Test logging data modifications."""
        correlation_id = audit_logger.log_data_modification(
            table="users",
            operation="UPDATE",
            record_id="user123",
            user_id="admin",
            username="admin",
            before_values={"role": "viewer"},
            after_values={"role": "analyst"},
            ip_address="192.168.1.1",
        )
        
        assert correlation_id is not None
        
        logs = audit_logger.query_logs(event_type="data_modification", limit=1)
        assert len(logs) == 1
        assert logs[0]["resource"] == "users"
        assert logs[0]["action"] == "UPDATE"
        assert logs[0]["details"]["record_id"] == "user123"
        assert logs[0]["details"]["before"]["role"] == "viewer"
        assert logs[0]["details"]["after"]["role"] == "analyst"
    
    def test_log_admin_action(self, audit_logger):
        """Test logging administrative actions."""
        correlation_id = audit_logger.log_admin_action(
            action="create_user",
            resource="users",
            user_id="admin",
            username="admin",
            details={"new_user_id": "user456", "role": "analyst"},
            ip_address="192.168.1.1",
        )
        
        assert correlation_id is not None
        
        logs = audit_logger.query_logs(event_type="admin_action", limit=1)
        assert len(logs) == 1
        assert logs[0]["action"] == "create_user"
        assert logs[0]["resource"] == "users"
        assert logs[0]["details"]["new_user_id"] == "user456"


class TestAuditLogImmutability:
    """Test that audit logs are immutable (append-only)."""
    
    def test_no_update_method(self, audit_logger):
        """Verify AuditLogger has no update method."""
        assert not hasattr(audit_logger, "update_log")
        assert not hasattr(audit_logger, "modify_log")
    
    def test_no_delete_method(self, audit_logger):
        """Verify AuditLogger has no delete method (except retention policy)."""
        assert not hasattr(audit_logger, "delete_log")
        assert not hasattr(audit_logger, "remove_log")
    
    def test_direct_update_not_supported(self, audit_logger, temp_db):
        """Test that direct SQL updates are not part of the API."""
        # Create a log entry
        audit_logger.log_event(
            AuditEventType.API_REQUEST,
            user_id="user123",
            resource="/api/test",
            action="GET",
        )
        
        # Get the entry
        logs = audit_logger.query_logs(limit=1)
        original_timestamp = logs[0]["timestamp"]
        log_id = logs[0]["id"]
        
        # Attempt direct update (this is what we're preventing in the API)
        # This test verifies the schema allows it but the API doesn't expose it
        temp_db.execute(
            "UPDATE audit_logs SET action = ? WHERE id = ?",
            ("MODIFIED", log_id)
        )
        temp_db.commit()
        
        # Verify the update happened at DB level (to prove immutability is API-level)
        cursor = temp_db.execute("SELECT action FROM audit_logs WHERE id = ?", (log_id,))
        result = cursor.fetchone()
        assert result[0] == "MODIFIED"
        
        # But the AuditLogger API doesn't provide this functionality
        # This test documents that immutability is enforced by API design


class TestAuditLogQuerying:
    """Test audit log querying functionality."""
    
    def test_query_by_date_range(self, audit_logger):
        """Test filtering logs by date range."""
        # Create logs with different timestamps
        now = datetime.utcnow()
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        
        audit_logger.log_event(
            AuditEventType.API_REQUEST,
            resource="/api/test1",
            action="GET",
        )
        
        # Query with date range
        logs = audit_logger.query_logs(
            start_date=yesterday,
            end_date=tomorrow,
        )
        
        assert len(logs) >= 1
    
    def test_query_by_user(self, audit_logger):
        """Test filtering logs by user ID."""
        audit_logger.log_event(
            AuditEventType.API_REQUEST,
            user_id="user123",
            resource="/api/test1",
        )
        
        audit_logger.log_event(
            AuditEventType.API_REQUEST,
            user_id="user456",
            resource="/api/test2",
        )
        
        # Query for specific user
        logs = audit_logger.query_logs(user_id="user123")
        
        assert len(logs) == 1
        assert logs[0]["user_id"] == "user123"
    
    def test_query_by_event_type(self, audit_logger):
        """Test filtering logs by event type."""
        audit_logger.log_auth_success(
            user_id="user123",
            username="testuser",
            ip_address="192.168.1.1",
        )
        
        audit_logger.log_api_request(
            request_method="GET",
            request_path="/api/test",
            status_code=200,
            response_time_ms=10.0,
        )
        
        # Query for auth events only
        logs = audit_logger.query_logs(event_type="auth_success")
        
        assert len(logs) == 1
        assert logs[0]["event_type"] == "auth_success"
    
    def test_query_by_resource(self, audit_logger):
        """Test filtering logs by resource."""
        audit_logger.log_event(
            AuditEventType.API_REQUEST,
            resource="/api/admin/users",
            action="GET",
        )
        
        audit_logger.log_event(
            AuditEventType.API_REQUEST,
            resource="/api/admin/logs",
            action="GET",
        )
        
        # Query for specific resource
        logs = audit_logger.query_logs(resource="/api/admin/users")
        
        assert len(logs) == 1
        assert logs[0]["resource"] == "/api/admin/users"
    
    def test_query_pagination(self, audit_logger):
        """Test pagination of query results."""
        # Create multiple log entries
        for i in range(10):
            audit_logger.log_event(
                AuditEventType.API_REQUEST,
                resource=f"/api/test{i}",
                action="GET",
            )
        
        # Query with limit
        logs = audit_logger.query_logs(limit=5)
        assert len(logs) == 5
        
        # Query with offset
        logs_page2 = audit_logger.query_logs(limit=5, offset=5)
        assert len(logs_page2) == 5
        
        # Verify different results
        assert logs[0]["id"] != logs_page2[0]["id"]
    
    def test_count_logs(self, audit_logger):
        """Test counting logs with filters."""
        # Create logs
        for i in range(5):
            audit_logger.log_event(
                AuditEventType.API_REQUEST,
                user_id="user123",
                resource=f"/api/test{i}",
            )
        
        for i in range(3):
            audit_logger.log_event(
                AuditEventType.API_REQUEST,
                user_id="user456",
                resource=f"/api/test{i}",
            )
        
        # Count all logs
        total = audit_logger.count_logs()
        assert total == 8
        
        # Count for specific user
        user_count = audit_logger.count_logs(user_id="user123")
        assert user_count == 5


class TestAuditLogRetention:
    """Test audit log retention policy."""
    
    def test_retention_policy_deletes_old_logs(self, temp_db):
        """Test that retention policy deletes logs older than specified days."""
        # Create old log entry by directly inserting with old timestamp
        old_date = (datetime.utcnow() - timedelta(days=100)).strftime("%Y-%m-%d %H:%M:%S")
        temp_db.execute(
            """
            INSERT INTO audit_logs (timestamp, event_type, resource, action)
            VALUES (?, ?, ?, ?)
            """,
            (old_date, "api_request", "/api/test", "GET")
        )
        temp_db.commit()
        
        # Create recent log entry
        recent_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        temp_db.execute(
            """
            INSERT INTO audit_logs (timestamp, event_type, resource, action)
            VALUES (?, ?, ?, ?)
            """,
            (recent_date, "api_request", "/api/test", "GET")
        )
        temp_db.commit()
        
        # Apply retention policy (90 days)
        deleted = apply_retention_policy(temp_db, retention_days=90)
        
        assert deleted == 1
        
        # Verify only recent log remains
        cursor = temp_db.execute("SELECT COUNT(*) FROM audit_logs")
        count = cursor.fetchone()[0]
        assert count == 1
    
    def test_retention_policy_minimum_days(self, temp_db):
        """Test retention policy with minimum retention period."""
        # Create log from 50 days ago
        old_date = (datetime.utcnow() - timedelta(days=50)).strftime("%Y-%m-%d %H:%M:%S")
        temp_db.execute(
            """
            INSERT INTO audit_logs (timestamp, event_type, resource, action)
            VALUES (?, ?, ?, ?)
            """,
            (old_date, "api_request", "/api/test", "GET")
        )
        temp_db.commit()
        
        # Apply 90-day retention (should not delete)
        deleted = apply_retention_policy(temp_db, retention_days=90)
        assert deleted == 0
        
        # Apply 30-day retention (should delete)
        deleted = apply_retention_policy(temp_db, retention_days=30)
        assert deleted == 1


class TestAuditLogStats:
    """Test audit log statistics."""
    
    def test_get_audit_stats(self, audit_logger, temp_db):
        """Test getting audit log statistics."""
        # Create various log entries
        audit_logger.log_auth_success(
            user_id="user123",
            username="testuser",
            ip_address="192.168.1.1",
        )
        
        audit_logger.log_api_request(
            request_method="GET",
            request_path="/api/test",
            status_code=200,
            response_time_ms=10.0,
        )
        
        audit_logger.log_api_request(
            request_method="POST",
            request_path="/api/test",
            status_code=201,
            response_time_ms=20.0,
        )
        
        # Get stats
        stats = get_audit_stats(temp_db)
        
        assert stats["total_count"] == 3
        assert stats["oldest_entry"] is not None
        assert stats["newest_entry"] is not None
        assert "by_event_type" in stats
        assert stats["by_event_type"]["auth_success"] == 1
        assert stats["by_event_type"]["api_request"] == 2


class TestAuditLogDetails:
    """Test audit log detail handling."""
    
    def test_details_json_serialization(self, audit_logger):
        """Test that details are properly JSON serialized."""
        details = {
            "key1": "value1",
            "key2": 123,
            "key3": ["a", "b", "c"],
            "key4": {"nested": "object"},
        }
        
        audit_logger.log_event(
            AuditEventType.ADMIN_ACTION,
            action="test",
            details=details,
        )
        
        logs = audit_logger.query_logs(limit=1)
        retrieved_details = logs[0]["details"]
        
        assert retrieved_details == details
    
    def test_none_details(self, audit_logger):
        """Test handling of None details."""
        audit_logger.log_event(
            AuditEventType.API_REQUEST,
            resource="/api/test",
            details=None,
        )
        
        logs = audit_logger.query_logs(limit=1)
        assert logs[0]["details"] is None


class TestAuditLogCorrelation:
    """Test correlation ID functionality."""
    
    def test_correlation_id_generated(self, audit_logger):
        """Test that correlation ID is generated if not provided."""
        correlation_id = audit_logger.log_event(
            AuditEventType.API_REQUEST,
            resource="/api/test",
        )
        
        assert correlation_id is not None
        
        logs = audit_logger.query_logs(limit=1)
        assert logs[0]["correlation_id"] == correlation_id
    
    def test_correlation_id_provided(self, audit_logger):
        """Test using provided correlation ID."""
        custom_id = "custom-correlation-id-123"
        
        correlation_id = audit_logger.log_event(
            AuditEventType.API_REQUEST,
            resource="/api/test",
            correlation_id=custom_id,
        )
        
        assert correlation_id == custom_id
        
        logs = audit_logger.query_logs(limit=1)
        assert logs[0]["correlation_id"] == custom_id
