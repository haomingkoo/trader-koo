"""HTTP contract of the admin backup endpoints that dark-deploy depends on."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trader_koo.backend.routers.admin import backups as backups_router
from trader_koo.backend.routers.admin import router as production_admin_router
from trader_koo.middleware.auth import AdminAuthConfig, AdminAuthenticator
from trader_koo.scripts import backup_db

API_KEY = "k" * 32
AUTH = {"X-API-Key": API_KEY}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "live.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE evidence(id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO evidence(value) VALUES ('preserved')")
        conn.commit()
    monkeypatch.setattr(backups_router, "DB_PATH", db_path)
    monkeypatch.setattr(backups_router, "DEFAULT_BACKUP_DIR", tmp_path / "backups")

    app = FastAPI()
    app.state.admin_authenticator = AdminAuthenticator(AdminAuthConfig(api_key=API_KEY))
    app.include_router(production_admin_router)
    with TestClient(app) as test_client:
        yield test_client


def test_create_backup_returns_500_when_quick_check_fails(client, monkeypatch) -> None:
    monkeypatch.setattr(backup_db, "_quick_check", lambda path: ["row 3 missing from index"])

    response = client.post("/api/admin/backups", headers=AUTH)

    assert response.status_code == 500
    assert response.json()["detail"] == "backup_quick_check_failed"
