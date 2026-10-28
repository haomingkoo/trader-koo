"""Run start.sh with stubbed interpreter and server binaries."""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

if sys.platform == "win32":  # pragma: no cover
    pytest.skip("start.sh is a bash entrypoint", allow_module_level=True)


def _stub(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)
    return path


@pytest.fixture()
def harness(tmp_path: Path):
    calls = tmp_path / "calls"
    calls.mkdir()
    uvicorn = _stub(tmp_path / "uvicorn", f'printf "%s\\n" "$@" > "{calls}/uvicorn"\n')
    python = _stub(
        tmp_path / "python",
        "if [ \"${1:-}\" = trader_koo/scripts/update_market_db.py ]; then\n"
        f'  printf "%s\\n" "$@" > "{calls}/seed"; exit 0\n'
        "fi\n"
        f'exec "{sys.executable}" "$@"\n',
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("RAILWAY_")
        and key not in {"TRADER_KOO_ALLOW_EMPTY_BOOTSTRAP", "TRADER_KOO_MAINTENANCE_STATE_PATH"}
    }
    env.update(
        TRADER_KOO_START_PYTHON=str(python),
        TRADER_KOO_START_UVICORN=str(uvicorn),
        TRADER_KOO_LOG_DIR=str(tmp_path / "logs"),
        TRADER_KOO_DB_PATH=str(tmp_path / "data" / "trader_koo.db"),
        PORT="8123",
    )

    def run(**overrides: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(REPO / "start.sh")],
            cwd=REPO,
            env={**env, **overrides},
            capture_output=True,
            text=True,
            timeout=60,
        )

    return run, calls, Path(env["TRADER_KOO_DB_PATH"])


def _wait_for(path: Path, timeout: float = 10.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text():
            return path.read_text()
        time.sleep(0.05)
    raise AssertionError(f"{path.name} was never written")


def _seed_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE price_daily (ticker TEXT, date TEXT, close REAL)")
    conn.execute("INSERT INTO price_daily VALUES ('SPY', 'd1', 1.0)")
    conn.commit()
    conn.close()


def test_local_missing_db_bootstraps_schema_seed_and_server(harness):
    run, calls, db_path = harness

    result = run()

    assert result.returncode == 0, result.stderr
    conn = sqlite3.connect(db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "price_daily" in tables
    assert str(db_path) in _wait_for(calls / "seed").splitlines()
    args = (calls / "uvicorn").read_text().splitlines()
    assert args[:1] == ["trader_koo.backend.main:app"]
    assert ["--port", "8123"] == args[args.index("--port"):args.index("--port") + 2]


def test_existing_db_starts_server_without_seed(harness):
    run, calls, db_path = harness
    _seed_db(db_path)

    result = run()

    assert result.returncode == 0, result.stderr
    assert (calls / "uvicorn").exists()
    assert not (calls / "seed").exists()
    assert "1 price_daily rows" in result.stdout + result.stderr


def test_uvicorn_gets_bounded_graceful_shutdown(harness):
    run, calls, db_path = harness
    _seed_db(db_path)

    result = run()

    assert result.returncode == 0, result.stderr
    args = (calls / "uvicorn").read_text().splitlines()
    i = args.index("--timeout-graceful-shutdown")
    assert args[i + 1] == "90"


def test_production_missing_volume_db_refuses_to_boot(harness, tmp_path):
    run, calls, db_path = harness

    result = run(
        RAILWAY_ENVIRONMENT="production",
        RAILWAY_VOLUME_MOUNT_PATH=str(tmp_path / "data"),
    )

    assert result.returncode != 0
    assert not db_path.exists()
    assert not db_path.parent.exists()
    assert not (calls / "uvicorn").exists()
    assert not (calls / "seed").exists()


def _request_maintenance(db_path: Path) -> None:
    from trader_koo.backend.services.maintenance import request_maintenance

    db_path.parent.mkdir(parents=True, exist_ok=True)
    request_maintenance(db_path, run_id="maint-test", boot_id="boot", reason="test", timeout_sec=1)


def test_corrupt_db_under_maintenance_boots_maintenance_only(harness):
    run, calls, db_path = harness
    _request_maintenance(db_path)
    db_path.write_bytes(b"not a database" * 512)

    result = run()

    assert result.returncode == 0, result.stderr
    assert (calls / "uvicorn").exists()
    assert not (calls / "seed").exists()
    assert db_path.read_bytes() == b"not a database" * 512
