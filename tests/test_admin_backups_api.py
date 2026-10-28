"""HTTP contract of the admin backup endpoints that dark-deploy depends on."""
from __future__ import annotations

import gzip
import hashlib
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


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/admin/backups"),
        ("get", "/api/admin/backups"),
        ("get", "/api/admin/backups/latest"),
        ("get", "/api/admin/backups/trader_koo_backup_20261026T234300346462Z.db.gz"),
    ],
)
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": "k" * 31}])
def test_backup_endpoints_reject_missing_or_wrong_key(client, method, path, headers) -> None:
    response = getattr(client, method)(path, headers=headers)

    assert response.status_code == 401
    assert not (backups_router.DEFAULT_BACKUP_DIR).exists()


def test_create_then_download_returns_the_same_bytes_and_hash(client) -> None:
    created = client.post("/api/admin/backups", headers=AUTH)

    assert created.status_code == 200
    body = created.json()
    assert body["ok"] is True
    assert body["backup_name"].startswith(backup_db.BACKUP_PREFIX)
    assert len(body["sha256"]) == 64
    assert body["dest_size_bytes"] > 0

    downloaded = client.get(f"/api/admin/backups/{body['backup_name']}", headers=AUTH)

    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "application/gzip"
    assert len(downloaded.content) == body["dest_size_bytes"]
    assert hashlib.sha256(downloaded.content).hexdigest() == body["sha256"]
    on_disk = backups_router.DEFAULT_BACKUP_DIR / body["backup_name"]
    assert downloaded.content == on_disk.read_bytes()
    restored = on_disk.parent.parent / "restored.db"
    restored.write_bytes(gzip.decompress(downloaded.content))
    with sqlite3.connect(restored) as conn:
        assert conn.execute("SELECT value FROM evidence").fetchone()[0] == "preserved"

    listing = client.get("/api/admin/backups", headers=AUTH).json()
    assert [item["name"] for item in listing["backups"]] == [body["backup_name"]]


def test_unknown_backup_name_is_404(client) -> None:
    client.post("/api/admin/backups", headers=AUTH)

    response = client.get(
        "/api/admin/backups/trader_koo_backup_19990101T000000000000Z.db.gz", headers=AUTH
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Backup not found"


@pytest.mark.parametrize(
    "name",
    [
        "..%2Flive.db",
        "..%2F..%2Fetc%2Fpasswd",
        "%2E%2E%2Ftrader_koo_backup_x.db.gz",
        "trader_koo_backup_..%2F..%2Flive.db.gz",
        "live.db",
    ],
)
def test_traversal_and_foreign_names_are_rejected(client, name) -> None:
    response = client.get(f"/api/admin/backups/{name}", headers=AUTH)

    assert response.status_code == 404
    assert b"preserved" not in response.content


def test_symlinked_backup_name_is_rejected(client) -> None:
    created = client.post("/api/admin/backups", headers=AUTH).json()
    backup_dir = backups_router.DEFAULT_BACKUP_DIR
    link = backup_dir / "trader_koo_backup_20200101T000000000000Z.db.gz"
    link.symlink_to(backup_dir / created["backup_name"])

    response = client.get(f"/api/admin/backups/{link.name}", headers=AUTH)

    assert response.status_code == 404


def test_create_backup_returns_500_when_quick_check_fails(client, monkeypatch) -> None:
    monkeypatch.setattr(backup_db, "_quick_check", lambda path: ["row 3 missing from index"])

    response = client.post("/api/admin/backups", headers=AUTH)

    assert response.status_code == 500
    assert response.json()["detail"] == "backup_quick_check_failed"
