"""Golden-report harness: deterministic snapshots of report, email and paper decisions."""
from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ops.golden_report import compare, normalize, snapshot
from trader_koo.paper_trade.campaign import canonical_hash
from trader_koo.paper_trade.config import config_snapshot
from trader_koo.paper_trades import _build_config

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)
SNAPSHOT_FILES = ("report.json", "email.txt", "email.html", "paper.json", "policy.json")


def _local_db() -> Path | None:
    # The price databases are untracked, so a worktree finds them in the main checkout.
    roots = [REPO, *[p for p in REPO.parents if (p / "trader_koo" / "data").exists()][:1]]
    candidates = [
        root / rel
        for root in roots
        for rel in ("trader_koo/data/trader_koo.db", "data/prod_paper_price.db")
    ]
    existing = [p for p in candidates if p.is_file()]
    return max(existing, key=lambda p: p.stat().st_mtime) if existing else None


@pytest.fixture
def golden_db(tmp_path):
    src = _local_db()
    if src is None:
        pytest.skip("no local price database available")
    dest = tmp_path / "golden.db"
    shutil.copy(src, dest)
    return dest


@pytest.fixture
def empty_db(tmp_path):
    dest = tmp_path / "empty.db"
    sqlite3.connect(dest).close()
    return dest


def write_snapshot(out: Path, **files: object) -> None:
    out.mkdir(parents=True)
    for name, value in files.items():
        (out / f"{name}.json").write_text(json.dumps(value, sort_keys=True, indent=1))


def test_snapshot_is_deterministic(tmp_path, golden_db):
    a = snapshot(golden_db, tmp_path / "a", NOW)
    b = snapshot(golden_db, tmp_path / "b", NOW)
    for name in SNAPSHOT_FILES:
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_snapshot_blocks_network(tmp_path, golden_db):
    out = snapshot(golden_db, tmp_path / "s", NOW)
    assert json.loads((out / "network.json").read_text()) == []


def test_policy_json_has_default_and_registered_hash(tmp_path, golden_db):
    policy = json.loads((snapshot(golden_db, tmp_path / "p", NOW) / "policy.json").read_text())
    assert policy["default_hash"] == canonical_hash(config_snapshot(_build_config()))
    assert "registered_hash" in policy


def test_generator_failure_is_recorded(tmp_path, empty_db):
    report = json.loads((snapshot(empty_db, tmp_path / "f", NOW) / "report.json").read_text())
    assert report["ok"] is False and report["error"]


def test_normalize_drops_volatile_fields_and_sorts_unordered_lists():
    payload = {
        "generated_ts": "x",
        "db_path": "/tmp/a",
        "meta": {"report_run": {"run_id": "r1"}, "duration_ms": 5},
        "tags": ["b", "a"],
    }
    assert normalize(payload) == {"meta": {"report_run": {}}, "tags": ["a", "b"]}


def test_compare_identical_is_empty(tmp_path, golden_db):
    a = snapshot(golden_db, tmp_path / "a", NOW)
    b = snapshot(golden_db, tmp_path / "b", NOW)
    assert compare(a, b) == []


def test_compare_reports_json_path(tmp_path):
    write_snapshot(tmp_path / "a", report={"setups": [{"ticker": "AAPL", "score": 70}]})
    write_snapshot(tmp_path / "b", report={"setups": [{"ticker": "AAPL", "score": 71}]})
    assert compare(tmp_path / "a", tmp_path / "b") == ["report.json: setups[0].score 70 -> 71"]


def test_compare_reports_missing_file_and_text_diff(tmp_path):
    write_snapshot(tmp_path / "a", report={"x": 1})
    write_snapshot(tmp_path / "b", report={"x": 1}, paper={"y": 1})
    (tmp_path / "a" / "email.txt").write_text("hello")
    (tmp_path / "b" / "email.txt").write_text("hullo")
    assert compare(tmp_path / "a", tmp_path / "b") == ["email.txt: text differs", "paper.json: only in head"]
