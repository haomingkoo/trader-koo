"""
Integration tests for audit logging with FastAPI endpoints.

Tests the complete audit logging flow including:
- Authentication logging
- Admin API request logging
- Query endpoints
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trader_koo.audit import ensure_audit_schema


@pytest.fixture
def temp_db_path():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    
    # Initialize schema
    conn = sqlite3.connect(str(db_path))
    ensure_audit_schema(conn)
    conn.close()
    
    yield db_path
    
    db_path.unlink()


@pytest.fixture
def test_app(temp_db_path, monkeypatch):
    """Create a test FastAPI app with audit logging."""
    # Set environment variables for testing
    monkeypatch.setenv("TRADER_KOO_DB_PATH", str(temp_db_path))
    monkeypatch.setenv("TRADER_KOO_API_KEY", "test-api-key-12345678901234567890")
    monkeypatch.setenv("ADMIN_STRICT_API_KEY", "0")  # Disable strict mode for testing
    
    # Ensure schema is initialized before importing app
    conn = sqlite3.connect(str(temp_db_path))
    ensure_audit_schema(conn)
    conn.close()
    
    # Import app after setting env vars
    from trader_koo.backend.main import app
    
    return app


class TestAuditLoggingIntegration:
    """Integration tests for audit logging."""
    
    def test_admin_endpoint_logs_request(self, test_app, temp_db_path):
        """Test that admin API requests are logged."""
        client = TestClient(test_app)
        
        # Make an admin API request
        response = client.get(
            "/api/admin/routes",
            headers={"X-API-Key": "test-api-key-12345678901234567890"}
        )
        
        assert response.status_code == 200
        
        # Check that the request was logged
        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute(
            """
            SELECT event_type, resource, action, status_code
            FROM audit_logs
            WHERE event_type = 'api_request'
            ORDER BY timestamp DESC
            LIMIT 1
            """
        )
        
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[0] == "api_request"
        assert row[1] == "/api/admin/routes"
        assert row[2] == "GET"
        assert row[3] == 200
    
    def test_failed_auth_logs_failure(self, test_app, temp_db_path):
        """Test that failed authentication attempts are logged."""
        client = TestClient(test_app)
        
        # Make request with invalid API key
        response = client.get(
            "/api/admin/routes",
            headers={"X-API-Key": "invalid-key"}
        )
        
        assert response.status_code == 401
        
        # Check that the failure was logged
        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute(
            """
            SELECT event_type, action, details
            FROM audit_logs
            WHERE event_type = 'auth_failure'
            ORDER BY timestamp DESC
            LIMIT 1
            """
        )
        
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[0] == "auth_failure"
        assert row[1] == "login_failed"
    
    def test_successful_auth_logs_success(self, test_app, temp_db_path):
        """Test that successful authentication is logged."""
        client = TestClient(test_app)
        
        # Make request with valid API key
        response = client.get(
            "/api/admin/routes",
            headers={"X-API-Key": "test-api-key-12345678901234567890"}
        )
        
        assert response.status_code == 200
        
        # Check that the success was logged
        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute(
            """
            SELECT event_type, action, username
            FROM audit_logs
            WHERE event_type = 'auth_success'
            ORDER BY timestamp DESC
            LIMIT 1
            """
        )
        
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[0] == "auth_success"
        assert row[1] == "login_api_key"
        assert row[2] == "admin"
    
    def test_audit_logs_query_endpoint(self, test_app, temp_db_path):
        """Test the audit logs query endpoint."""
        client = TestClient(test_app)
        
        # Make some requests to generate logs
        client.get(
            "/api/admin/routes",
            headers={"X-API-Key": "test-api-key-12345678901234567890"}
        )
        
        # Query audit logs
        response = client.get(
            "/api/admin/audit-logs",
            headers={"X-API-Key": "test-api-key-12345678901234567890"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["ok"] is True
        assert "logs" in data
        assert "pagination" in data
        assert len(data["logs"]) > 0
    
    def test_audit_logs_stats_endpoint(self, test_app, temp_db_path):
        """Test the audit logs stats endpoint."""
        client = TestClient(test_app)
        
        # Make some requests to generate logs
        client.get(
            "/api/admin/routes",
            headers={"X-API-Key": "test-api-key-12345678901234567890"}
        )
        
        # Get stats
        response = client.get(
            "/api/admin/audit-logs/stats",
            headers={"X-API-Key": "test-api-key-12345678901234567890"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["ok"] is True
        assert "total_count" in data
        assert data["total_count"] > 0
        assert "by_event_type" in data
    
    def test_audit_logs_export_json(self, test_app, temp_db_path):
        """Test exporting audit logs as JSON."""
        client = TestClient(test_app)
        
        # Make some requests to generate logs
        client.get(
            "/api/admin/routes",
            headers={"X-API-Key": "test-api-key-12345678901234567890"}
        )
        
        # Export logs
        response = client.get(
            "/api/admin/audit-logs/export?format=json",
            headers={"X-API-Key": "test-api-key-12345678901234567890"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "logs" in data
        assert "export_timestamp" in data
        assert len(data["logs"]) > 0
    
    def test_audit_logs_export_csv(self, test_app, temp_db_path):
        """Test exporting audit logs as CSV."""
        client = TestClient(test_app)
        
        # Make some requests to generate logs
        client.get(
            "/api/admin/routes",
            headers={"X-API-Key": "test-api-key-12345678901234567890"}
        )
        
        # Export logs as CSV
        response = client.get(
            "/api/admin/audit-logs/export?format=csv",
            headers={"X-API-Key": "test-api-key-12345678901234567890"}
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        
        # Verify CSV content
        csv_content = response.text
        assert "event_type" in csv_content  # Header
        assert "api_request" in csv_content or "auth_success" in csv_content
    
    def test_audit_logs_summary_endpoint(self, test_app, temp_db_path):
        """Test the audit logs summary endpoint."""
        client = TestClient(test_app)
        
        # Make some requests to generate logs
        for _ in range(3):
            client.get(
                "/api/admin/routes",
                headers={"X-API-Key": "test-api-key-12345678901234567890"}
            )
        
        # Get summary
        response = client.get(
            "/api/admin/audit-logs/summary?days=7",
            headers={"X-API-Key": "test-api-key-12345678901234567890"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["ok"] is True
        assert "total_events" in data
        assert data["total_events"] > 0
        assert "event_type_counts" in data
        assert "top_users" in data
    
    def test_audit_logs_retention_dry_run(self, test_app, temp_db_path):
        """Test retention policy dry run."""
        client = TestClient(test_app)
        
        # Make some requests to generate logs
        client.get(
            "/api/admin/routes",
            headers={"X-API-Key": "test-api-key-12345678901234567890"}
        )
        
        # Test dry run
        response = client.post(
            "/api/admin/audit-logs/retention?retention_days=90&dry_run=true",
            headers={"X-API-Key": "test-api-key-12345678901234567890"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["ok"] is True
        assert data["dry_run"] is True
        assert "would_delete" in data
        assert data["retention_days"] == 90
    
    def test_correlation_id_in_response(self, test_app, temp_db_path):
        """Test that correlation ID is added to response headers."""
        client = TestClient(test_app)
        
        response = client.get(
            "/api/admin/routes",
            headers={"X-API-Key": "test-api-key-12345678901234567890"}
        )
        
        assert response.status_code == 200
        assert "X-Correlation-ID" in response.headers
        
        correlation_id = response.headers["X-Correlation-ID"]
        assert len(correlation_id) > 0
        
        # Verify the correlation ID is in the audit log
        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute(
            """
            SELECT correlation_id
            FROM audit_logs
            WHERE correlation_id = ?
            LIMIT 1
            """,
            (correlation_id,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[0] == correlation_id
