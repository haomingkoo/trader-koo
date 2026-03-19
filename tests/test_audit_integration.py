"""
Integration tests for audit logging with FastAPI endpoints.

Tests the complete audit logging flow including:
- Authentication logging
- Admin API request logging
- Query endpoints

Uses the same test-app-from-routers pattern as conftest.py to avoid
importing the monolithic main.py (which triggers the full startup chain).
"""

import os
import secrets
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from trader_koo.audit import ensure_audit_schema
from trader_koo.audit.logger import AuditLogger
from trader_koo.audit.middleware import AuditMiddleware, log_auth_attempt
from trader_koo.backend.routers.admin import router as admin_router


def _create_audit_test_db(db_path: Path) -> None:
    """Initialize a database file with audit + minimal schema."""
    conn = sqlite3.connect(str(db_path))
    ensure_audit_schema(conn)
    # Create the minimal tables the admin router may query
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ingest_runs (
            run_id TEXT PRIMARY KEY,
            started_ts TEXT,
            finished_ts TEXT,
            status TEXT DEFAULT 'running',
            tickers_total INTEGER DEFAULT 0,
            tickers_ok INTEGER DEFAULT 0,
            tickers_failed INTEGER DEFAULT 0,
            error_message TEXT
        );
        CREATE TABLE IF NOT EXISTS price_daily (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume INTEGER, data_source TEXT, fetch_timestamp TEXT,
            UNIQUE(ticker, date)
        );
    """)
    conn.close()


@pytest.fixture
def temp_db_path():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    _create_audit_test_db(db_path)
    yield db_path
    db_path.unlink()


class _TestAdminAuthMiddleware(BaseHTTPMiddleware):
    """Lightweight auth middleware that mirrors the real admin auth logic in main.py.

    Checks for X-API-Key on /api/admin/* routes, sets ``request.state.admin_identity``,
    and logs auth events to the audit_logs table so integration tests can verify them.
    """

    def __init__(self, app: FastAPI, *, db_path: Path, api_key: str):
        super().__init__(app)
        self.db_path = db_path
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        path = request.url.path
        if path.startswith("/api/admin"):
            provided = request.headers.get("X-API-Key", "")
            client_ip = request.client.host if request.client else "127.0.0.1"
            user_agent = request.headers.get("user-agent", "")

            if not secrets.compare_digest(provided, self.api_key):
                # Log auth failure
                try:
                    conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
                    ensure_audit_schema(conn)
                    logger = AuditLogger(conn)
                    log_auth_attempt(
                        logger,
                        success=False,
                        username=None,
                        ip_address=client_ip,
                        user_agent=user_agent,
                        auth_method="api_key",
                        reason="invalid_api_key",
                    )
                    conn.close()
                except Exception:
                    pass
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)

            # Valid key — set admin identity
            request.state.admin_identity = {
                "username": "admin",
                "mode": "api_key",
                "user_id": "admin",
            }

            # Log auth success
            try:
                conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
                ensure_audit_schema(conn)
                logger = AuditLogger(conn)
                log_auth_attempt(
                    logger,
                    success=True,
                    username="admin",
                    ip_address=client_ip,
                    user_agent=user_agent,
                    auth_method="api_key",
                )
                conn.close()
            except Exception:
                pass

        return await call_next(request)


@pytest.fixture
def test_app(temp_db_path, monkeypatch):
    """Create a lightweight test FastAPI app with auth + audit middleware + admin router.

    This avoids importing trader_koo.backend.main (which triggers the full
    startup chain with scheduler, crypto feeds, etc.).
    """
    api_key = "test-api-key-12345678901234567890"
    monkeypatch.setenv("TRADER_KOO_API_KEY", api_key)
    monkeypatch.setenv("ADMIN_STRICT_API_KEY", "0")

    # Re-initialise schema (env vars may have changed)
    conn = sqlite3.connect(str(temp_db_path))
    ensure_audit_schema(conn)
    conn.close()

    app = FastAPI(title="audit-test")

    # Wire up audit middleware (for correlation IDs + request logging)
    app.add_middleware(AuditMiddleware, db_path=temp_db_path)
    # Wire up auth middleware (for API key checking + auth event logging)
    app.add_middleware(_TestAdminAuthMiddleware, db_path=temp_db_path, api_key=api_key)

    # Include only the admin router (all audit endpoints live here)
    app.include_router(admin_router)

    # Patch DB_PATH and get_conn so admin router reads/writes the temp DB
    def _fake_get_conn(db_path=None):
        c = sqlite3.connect(str(temp_db_path), check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    patches = [
        patch("trader_koo.backend.services.database.DB_PATH", temp_db_path),
        patch("trader_koo.backend.services.database.get_conn", _fake_get_conn),
        # Patch DB_PATH / get_conn in each admin sub-module that imports them
        patch("trader_koo.backend.routers.admin._shared.DB_PATH", temp_db_path),
        patch("trader_koo.backend.routers.admin.data.DB_PATH", temp_db_path),
        patch("trader_koo.backend.routers.admin.data.get_conn", _fake_get_conn),
        patch("trader_koo.backend.routers.admin.pipeline.DB_PATH", temp_db_path),
        patch("trader_koo.backend.routers.admin.pipeline.get_conn", _fake_get_conn),
        patch("trader_koo.backend.routers.admin.email_admin.DB_PATH", temp_db_path),
        patch("trader_koo.backend.routers.admin.ml.get_conn", _fake_get_conn),
        patch("trader_koo.backend.routers.admin.system.DB_PATH", temp_db_path),
        patch("trader_koo.backend.routers.admin.system.get_conn", _fake_get_conn),
    ]
    for p in patches:
        p.start()

    yield app

    for p in patches:
        p.stop()


class TestAuditLoggingIntegration:
    """Integration tests for audit logging."""

    def test_admin_endpoint_logs_request(self, test_app, temp_db_path):
        """Test that admin API requests are logged."""
        client = TestClient(test_app)

        response = client.get(
            "/api/admin/routes",
            headers={"X-API-Key": "test-api-key-12345678901234567890"},
        )

        assert response.status_code == 200

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

        response = client.get(
            "/api/admin/routes",
            headers={"X-API-Key": "invalid-key"},
        )

        assert response.status_code == 401

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

        response = client.get(
            "/api/admin/routes",
            headers={"X-API-Key": "test-api-key-12345678901234567890"},
        )

        assert response.status_code == 200

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

        # Generate some logs
        client.get(
            "/api/admin/routes",
            headers={"X-API-Key": "test-api-key-12345678901234567890"},
        )

        response = client.get(
            "/api/admin/audit-logs",
            headers={"X-API-Key": "test-api-key-12345678901234567890"},
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

        client.get(
            "/api/admin/routes",
            headers={"X-API-Key": "test-api-key-12345678901234567890"},
        )

        response = client.get(
            "/api/admin/audit-logs/stats",
            headers={"X-API-Key": "test-api-key-12345678901234567890"},
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

        client.get(
            "/api/admin/routes",
            headers={"X-API-Key": "test-api-key-12345678901234567890"},
        )

        response = client.get(
            "/api/admin/audit-logs/export?format=json",
            headers={"X-API-Key": "test-api-key-12345678901234567890"},
        )

        assert response.status_code == 200
        data = response.json()

        assert "logs" in data
        assert "export_timestamp" in data
        assert len(data["logs"]) > 0

    def test_audit_logs_export_csv(self, test_app, temp_db_path):
        """Test exporting audit logs as CSV."""
        client = TestClient(test_app)

        client.get(
            "/api/admin/routes",
            headers={"X-API-Key": "test-api-key-12345678901234567890"},
        )

        response = client.get(
            "/api/admin/audit-logs/export?format=csv",
            headers={"X-API-Key": "test-api-key-12345678901234567890"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"

        csv_content = response.text
        assert "event_type" in csv_content
        assert "api_request" in csv_content or "auth_success" in csv_content

    def test_audit_logs_summary_endpoint(self, test_app, temp_db_path):
        """Test the audit logs summary endpoint."""
        client = TestClient(test_app)

        for _ in range(3):
            client.get(
                "/api/admin/routes",
                headers={"X-API-Key": "test-api-key-12345678901234567890"},
            )

        response = client.get(
            "/api/admin/audit-logs/summary?days=7",
            headers={"X-API-Key": "test-api-key-12345678901234567890"},
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

        client.get(
            "/api/admin/routes",
            headers={"X-API-Key": "test-api-key-12345678901234567890"},
        )

        response = client.post(
            "/api/admin/audit-logs/retention?retention_days=90&dry_run=true",
            headers={"X-API-Key": "test-api-key-12345678901234567890"},
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
            headers={"X-API-Key": "test-api-key-12345678901234567890"},
        )

        assert response.status_code == 200
        assert "X-Correlation-ID" in response.headers

        correlation_id = response.headers["X-Correlation-ID"]
        assert len(correlation_id) > 0

        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute(
            """
            SELECT correlation_id
            FROM audit_logs
            WHERE correlation_id = ?
            LIMIT 1
            """,
            (correlation_id,),
        )

        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == correlation_id
