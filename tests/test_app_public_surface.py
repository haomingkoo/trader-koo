"""The production app's public surface, probed in a clean subprocess.

``trader_koo.backend.main`` reads its environment at import time, so each
probe imports it fresh with temporary data paths.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


_PROBE = """
import json
from fastapi.testclient import TestClient
from trader_koo.backend.main import app

paths = sorted(getattr(r, "path", "") for r in app.routes)
client = TestClient(app)
release = client.get("/api/release")
print(json.dumps({
    "openapi_url": app.openapi_url,
    "docs_url": app.docs_url,
    "paths": paths,
    "openapi_status": client.get("/openapi.json").status_code,
    "release_status": release.status_code,
    "release_headers": {k.lower(): v for k, v in release.headers.items()},
}))
"""


def _probe_main_app(tmp_path: Path, docs_enabled: str | None) -> dict:
    env = {
        key: value
        for key, value in os.environ.items()
        if key != "TRADER_KOO_DOCS_ENABLED"
    }
    env.update(
        TRADER_KOO_DB_PATH=str(tmp_path / "trader_koo.db"),
        TRADER_KOO_LOGOS_DIR=str(tmp_path / "logos"),
        TRADER_KOO_REPORT_DIR=str(tmp_path / "reports"),
        PYTHONPATH=str(REPO_ROOT),
    )
    if docs_enabled is not None:
        env["TRADER_KOO_DOCS_ENABLED"] = docs_enabled
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_main_app_hides_openapi_schema_by_default(tmp_path):
    probe = _probe_main_app(tmp_path, None)

    assert probe["openapi_url"] is None
    assert probe["docs_url"] is None
    assert "/openapi.json" not in probe["paths"]
    assert "/docs" not in probe["paths"]
    assert probe["openapi_status"] == 404
    assert probe["release_status"] == 200
    assert probe["release_headers"]["x-frame-options"] == "DENY"
    assert probe["release_headers"]["content-security-policy"] == "frame-ancestors 'none'"
    assert probe["release_headers"]["x-content-type-options"] == "nosniff"


def test_main_app_serves_docs_only_when_enabled(tmp_path):
    probe = _probe_main_app(tmp_path, "1")

    assert probe["openapi_url"] == "/openapi.json"
    assert {"/openapi.json", "/docs", "/redoc"} <= set(probe["paths"])
    assert probe["openapi_status"] == 200
