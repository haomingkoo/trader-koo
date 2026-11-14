"""Prove a production install can run the pipeline's runtime code.

Run it in an environment built only the way the Railway build builds it
(requirements.txt, then the Python steps of build.sh), never with
requirements-dev.txt. A dev dependency can supply a transitive package that
production lacks: from 31 Oct to 14 Nov 2026 YOLO failed every ticker with
"No module named 'six'" while every test passed.

Checks, all fatal:
  imports  every module under trader_koo, plus the third-party YOLO stack that
           run_yolo_patterns.py imports lazily inside functions;
  yolo     run_yolo_patterns.py exactly as daily_update.sh runs it, on one
           sealed synthetic ticker in a throwaway database, with the real
           Hugging Face model. It must exit 0 with that ticker marked ok.

build.sh runs ``--imports-only`` so a broken image fails its Railway build.
CI runs both checks. Standard library only until the imports it checks.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import os
import pkgutil
import sqlite3
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Imported lazily by run_yolo_patterns.py, so walking trader_koo never loads them.
YOLO_STACK = (
    "cv2",
    "torch",
    "torchvision",
    "ultralytics",
    "huggingface_hub",
    "mplfinance",
    "matplotlib.pyplot",
    "PIL.Image",
)
SMOKE_TICKER = "SMOKE"
SMOKE_BARS = 200


def _isolate_runtime_paths(workdir: Path) -> None:
    """Point every path the app writes at import time into *workdir*."""
    for name, sub in (
        ("TRADER_KOO_DB_PATH", "trader_koo.db"),
        ("TRADER_KOO_LOG_DIR", "logs"),
        ("TRADER_KOO_REPORT_DIR", "reports"),
        ("TRADER_KOO_LOGOS_DIR", "logos"),
    ):
        os.environ[name] = str(workdir / sub)
    os.environ.setdefault("MPLBACKEND", "Agg")


def check_imports() -> list[str]:
    """Import the YOLO stack and every trader_koo module; return the failures."""
    import trader_koo

    names = list(YOLO_STACK)
    names += [
        info.name
        for info in pkgutil.walk_packages(trader_koo.__path__, prefix="trader_koo.")
    ]
    failures: list[str] = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # any import-time error is a broken install
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    print(f"imports: {len(names) - len(failures)}/{len(names)} ok")
    return failures


def _seed_smoke_db(db_path: Path) -> str:
    """One sealed, research-eligible synthetic ticker in the production schema."""
    from trader_koo.db.price_contract import record_price_series_revision
    from trader_koo.scripts.update_market_db import ensure_schema

    last = dt.datetime.now(dt.timezone.utc).date()
    rows = []
    for i in range(SMOKE_BARS):
        day = last - dt.timedelta(days=SMOKE_BARS - 1 - i)
        close = 100.0 + 10.0 * ((i % 40) - 20) / 20.0 + i * 0.05
        rows.append((
            SMOKE_TICKER, day.isoformat(), close - 0.5, close + 1.0, close - 1.0, close,
            1_000_000 + i, "runtime_smoke", "split_adjusted_price_only", "smoke-v1", "verified",
        ))
    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema(conn)
        conn.executemany(
            """INSERT INTO price_daily
               (ticker, date, open, high, low, close, volume, data_source,
                adjustment_basis, adjustment_version, basis_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        record_price_series_revision(
            conn,
            SMOKE_TICKER,
            evidence={"provider": "runtime_smoke", "vendor_action_ledger_checked": True},
            fetch_timestamp=dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        )
        conn.commit()
    finally:
        conn.close()
    return last.isoformat()


def check_yolo(workdir: Path) -> list[str]:
    """Run run_yolo_patterns.py the way daily_update.sh does; return the failures."""
    db_path = workdir / "yolo-smoke.db"
    as_of = _seed_smoke_db(db_path)
    cmd = [
        sys.executable,
        str(ROOT / "trader_koo" / "scripts" / "run_yolo_patterns.py"),
        "--db-path", str(db_path),
        "--timeframe", "daily",
        "--only-new",
        "--sleep", "0",
        "--run-id", "runtime-smoke",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    sys.stdout.write(result.stdout[-4000:])
    sys.stdout.write(result.stderr[-4000:])
    failures: list[str] = []
    if result.returncode != 0:
        failures.append(f"run_yolo_patterns.py exited rc={result.returncode}")
    conn = sqlite3.connect(str(db_path))
    try:
        events = conn.execute(
            "SELECT ticker, status, reason, as_of_date FROM yolo_run_events WHERE run_id = 'runtime-smoke'"
        ).fetchall()
    except sqlite3.OperationalError as exc:
        events = []
        failures.append(f"yolo_run_events unreadable: {exc}")
    finally:
        conn.close()
    if events != [(SMOKE_TICKER, "ok", "ok", as_of)]:
        failures.append(f"expected one ok event for {SMOKE_TICKER} as of {as_of}, got {events}")
    print(f"yolo: rc={result.returncode} events={events}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--imports-only", action="store_true", help="skip the YOLO run (used by build.sh)")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="trader-koo-smoke-") as tmp:
        workdir = Path(tmp)
        _isolate_runtime_paths(workdir)
        try:
            failures = check_imports()
            if not failures and not args.imports_only:
                failures += check_yolo(workdir)
        except Exception:
            traceback.print_exc()
            failures = ["runtime smoke crashed (traceback above)"]
    for failure in failures:
        print(f"FAIL: {failure}")
    if not failures:
        print("OK: production runtime smoke passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
