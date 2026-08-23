from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from trader_koo.backend.services.report_loader import (
    latest_daily_report_json,
    report_json_for_generated_ts,
)
from trader_koo.report.calibration_pulse import _eval_stats
from trader_koo.report.runs import (
    admit_published_report,
    complete_report_run,
    ensure_report_run_schema,
    fail_report_run,
    publish_report_run,
    reconcile_report_publication,
    resolve_published_report,
    sha256_file,
    start_report_run,
)
from trader_koo.report.serializer import write_reports
from trader_koo.report.setup_scoring import (
    ensure_setup_call_eval_schema,
)


TEST_SHA = "a" * 40


def test_empty_configured_registry_keeps_pre_migration_report_readable(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    legacy_path = report_dir / "daily_report_20260822T120000Z.json"
    legacy_path.write_text(
        json.dumps({"generated_ts": "2026-08-22T12:00:00Z", "ok": True}) + "\n"
    )
    conn = sqlite3.connect(":memory:")
    ensure_report_run_schema(conn)

    path, payload = latest_daily_report_json(report_dir, registry_conn=conn)

    assert path == legacy_path
    assert payload is not None
    assert payload["report_run"] == {
        "run_id": None,
        "state": "unlinked_legacy",
        "lineage": "unlinked legacy",
    }


def _report(*tickers: str, accepted: int = 1) -> dict:
    rows = [
        {
            "ticker": ticker,
            "score": 90 - index,
            "confirmation_count": 3,
            "contradiction_count": 0,
            "signal_bias": "bullish",
            "setup_family": "bullish_continuation",
            "setup_tier": "A",
            "actionability": "higher-probability",
            "close": 100.0 + index,
        }
        for index, ticker in enumerate(tickers)
    ]
    decisions = [
        {
            "ticker": row["ticker"],
            "selected_rank": index + 1,
            "decision": "accepted" if index < accepted else "rejected",
            "reason_codes": (
                ["selected_report_cohort"]
                if index < accepted
                else ["outside_report_selection_limit"]
            ),
            "inputs": dict(row),
        }
        for index, row in enumerate(rows)
    ]
    return {
        "generated_ts": "2026-08-22T12:00:00Z",
        "meta": {"report_kind": "daily"},
        "latest_data": {
            "price_date": "2026-08-21",
            "fund_snapshot": "2026-08-22T01:00:00Z",
            "options_snapshot": "2026-08-22T02:00:00Z",
            "yolo_detected_ts": "2026-08-22T03:00:00Z",
        },
        "signals": {
            "setup_quality_top": rows[:accepted],
            "setup_quality_all": rows,
            "setup_quality_lookup": {row["ticker"]: row for row in rows},
            "report_decisions": decisions,
            "scanned_universe": [row["ticker"] for row in rows],
        },
        "counts": {"tracked_tickers": len(rows)},
        "risk_filters": {"trade_mode": "normal"},
        "warnings": [],
        "ok": True,
    }


def _complete_and_publish(
    conn: sqlite3.Connection,
    report_dir: Path,
    report: dict,
) -> str:
    run_id = start_report_run(
        conn,
        report_kind="daily",
        configuration={"selection_limit": 40},
        code_version=TEST_SHA,
        started_ts="2026-08-22T11:59:00Z",
    )
    paths = write_reports(report, report_dir, run_id=run_id, publish_latest=False)
    artifact = Path(paths["json_path"])
    complete_report_run(
        conn,
        run_id=run_id,
        report=report,
        artifact_path=artifact,
        markdown_path=Path(paths["md_path"]),
        content_hash=sha256_file(artifact),
    )
    publish_report_run(conn, run_id=run_id, report_dir=report_dir)
    return run_id


def _complete_only(
    conn: sqlite3.Connection,
    report_dir: Path,
    report: dict,
) -> str:
    run_id = start_report_run(
        conn,
        report_kind="daily",
        configuration={"selection_limit": 40},
        code_version=TEST_SHA,
        started_ts="2026-08-22T11:59:00Z",
    )
    paths = write_reports(report, report_dir, run_id=run_id, publish_latest=False)
    artifact = Path(paths["json_path"])
    complete_report_run(
        conn,
        run_id=run_id,
        report=report,
        artifact_path=artifact,
        markdown_path=Path(paths["md_path"]),
        content_hash=sha256_file(artifact),
    )
    return run_id


def test_published_run_owns_immutable_accepted_and_rejected_decisions(tmp_path: Path):
    conn = sqlite3.connect(tmp_path / "report.db")
    run_id = _complete_and_publish(conn, tmp_path / "reports", _report("AAA", "BBB", accepted=1))

    decisions = conn.execute(
        "SELECT ticker, selected_rank, decision, reason_codes_json "
        "FROM report_run_decisions WHERE run_id = ? ORDER BY selected_rank",
        (run_id,),
    ).fetchall()
    assert decisions == [
        ("AAA", 1, "accepted", '["selected_report_cohort"]'),
        ("BBB", 2, "rejected", '["outside_report_selection_limit"]'),
    ]
    with pytest.raises(sqlite3.IntegrityError, match="immutable|state transition"):
        conn.execute(
            "UPDATE report_runs SET decisions_json = '[]' WHERE run_id = ?",
            (run_id,),
        )
    for sql, value in (
        ("UPDATE report_runs SET published_ts = ? WHERE run_id = ?", "2099-01-01T00:00:00Z"),
        ("UPDATE report_runs SET error_message = ? WHERE run_id = ?", "fabricated"),
    ):
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(sql, (value, run_id))
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("DELETE FROM report_runs WHERE run_id = ?", (run_id,))
    with pytest.raises(sqlite3.IntegrityError, match="started parent"):
        conn.execute(
            """
            INSERT INTO report_run_decisions
            (run_id, ticker, selected_rank, decision, reason_codes_json, inputs_json)
            VALUES (?, 'MUTATED', 99, 'accepted', '[]', '{}')
            """,
            (run_id,),
        )


def test_sql_lifecycle_requires_terminal_evidence_and_freezes_failures(tmp_path: Path):
    conn = sqlite3.connect(tmp_path / "report.db")
    ensure_report_run_schema(conn)
    with pytest.raises(sqlite3.IntegrityError, match="begin in started"):
        conn.execute(
            """INSERT INTO report_runs (
                   run_id, report_kind, status, started_ts, config_json,
                   config_hash, code_version
               ) VALUES ('forged', 'daily', 'published', '2026-08-22T00:00:00Z',
                         '{}', 'hash', ?)""",
            (TEST_SHA,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="valid evidence"):
        conn.execute(
            """INSERT INTO report_runs (
                   run_id, report_kind, status, started_ts, config_json,
                   config_hash, code_version
               ) VALUES ('blank-forged', 'daily', 'started', ' ', '', '', '')"""
        )
    # SQLite has no built-in SHA-256. Persistent triggers validate shape;
    # the authoritative resolver validates hash equality without a UDF.
    conn.execute(
        """INSERT INTO report_runs (
               run_id, report_kind, status, started_ts, config_json,
               config_hash, code_version
           ) VALUES ('mismatch-forged', 'daily', 'started',
                     '2026-08-22T00:00:00Z', '{}', ?, ?)""",
        ("e" * 64, TEST_SHA),
    )

    run_id = start_report_run(
        conn, report_kind="daily", configuration={}, code_version=TEST_SHA,
        started_ts="2026-08-22T00:00:00Z",
    )
    with pytest.raises(sqlite3.IntegrityError, match="started report identity is immutable"):
        conn.execute(
            "UPDATE report_runs SET started_ts='2026-08-21T23:59:59Z' WHERE run_id=?",
            (run_id,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="complete evidence"):
        conn.execute(
            """UPDATE report_runs
               SET status='completed', completed_ts='2026-08-22T01:00:00Z',
                   generated_ts='2026-08-21T23:59:59Z',
                   generation_key='daily:2026-08-21T23:59:59Z',
                   scanned_universe_json='[]', ranked_candidates_json='[]',
                   decisions_json='[]', inputs_json='{}', source_timestamps_json='{}',
                   content_hash=?, markdown_hash=?, artifact_path='/tmp/report.json',
                   markdown_path='/tmp/report.md'
               WHERE run_id=?""",
            ("c" * 64, "d" * 64, run_id),
        )
    with pytest.raises(sqlite3.IntegrityError, match="complete evidence"):
        conn.execute(
            "UPDATE report_runs SET status='completed' WHERE run_id=?",
            (run_id,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="complete evidence"):
        conn.execute(
            "UPDATE report_runs SET status='failed' WHERE run_id=?",
            (run_id,),
        )

    conn.execute(
        """UPDATE report_runs
           SET status='completed', completed_ts='2026-08-22T01:00:00Z',
               generated_ts='2026-08-22T00:30:00Z',
               generation_key='daily:2026-08-22T00:30:00Z',
               scanned_universe_json='[]', ranked_candidates_json='[]',
               decisions_json='[]', inputs_json='{}', source_timestamps_json='{}',
               content_hash=?, markdown_hash=?, artifact_path='/tmp/report.json',
               markdown_path='/tmp/report.md'
           WHERE run_id=?""",
        ("c" * 64, "d" * 64, run_id),
    )
    with pytest.raises(sqlite3.IntegrityError, match="complete evidence"):
        conn.execute(
            "UPDATE report_runs SET status='published' WHERE run_id=?",
            (run_id,),
        )

    failed_id = start_report_run(
        conn, report_kind="daily", configuration={}, code_version=TEST_SHA
    )
    fail_report_run(conn, run_id=failed_id, error="source failed")
    with pytest.raises(sqlite3.IntegrityError, match="failed report run is immutable"):
        conn.execute(
            "UPDATE report_runs SET error_message='rewritten' WHERE run_id=?",
            (failed_id,),
        )

    config_json = '{"policy":"exact"}'
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    conn.execute(
        """INSERT INTO report_runs (
               run_id, report_kind, status, started_ts, config_json,
               config_hash, code_version
           ) VALUES ('forged', 'daily', 'started', '2026-08-22T00:00:00Z',
                     ?, ?, ?)""",
        (config_json, config_hash, TEST_SHA),
    )
    conn.execute(
        """UPDATE report_runs
           SET status='completed', completed_ts='2026-08-22T01:00:00Z',
               generated_ts='2026-08-22T00:30:00Z',
               generation_key='daily:2026-08-22T00:30:00Z',
               scanned_universe_json='[]', ranked_candidates_json='[]',
               decisions_json='[]', inputs_json='{}', source_timestamps_json='{}',
               content_hash=?, markdown_hash=?, artifact_path='/missing/report.json',
               markdown_path='/missing/report.md'
           WHERE run_id='forged'""",
        ("d" * 64, "e" * 64),
    )
    conn.execute(
        """UPDATE report_runs
           SET status='published', published_ts='2026-08-22T01:01:00Z',
               publication_verified=1 WHERE run_id='forged'"""
    )
    conn.execute(
        "UPDATE report_runs SET is_generation_canonical=1 WHERE run_id='forged'"
    )
    conn.commit()
    with pytest.raises(ValueError, match="path|hash"):
        resolve_published_report(
            conn, report_dir=tmp_path / "reports", run_id="forged"
        )
    with pytest.raises(ValueError, match="path|hash"):
        admit_published_report(
            conn, run_id="forged", report_dir=tmp_path / "reports"
        )


def test_legacy_admission_ledger_receives_insert_validation(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "legacy-admission.db")
    ensure_report_run_schema(conn)
    run_id = start_report_run(
        conn, report_kind="daily", configuration={}, code_version=TEST_SHA
    )
    conn.execute("DROP TABLE report_admission_attempts")
    conn.execute(
        """CREATE TABLE report_admission_attempts (
               attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
               run_id TEXT NOT NULL,
               status TEXT NOT NULL,
               error_code TEXT,
               error_message TEXT,
               attempted_ts TEXT NOT NULL
           )"""
    )
    conn.execute(
        """INSERT INTO report_admission_attempts
           (run_id,status,error_code,error_message,attempted_ts)
           VALUES (?,'succeeded',NULL,NULL,'2026-08-22T00:00:00Z')""",
        (run_id,),
    )
    conn.execute(
        """INSERT INTO report_admission_attempts
           (run_id,status,error_code,error_message,attempted_ts)
           VALUES (?,'failed','admission_lineage_failed','ValueError','2026-08-22T00:00:01Z')""",
        (run_id,),
    )
    conn.commit()
    conn.execute(
        "DELETE FROM report_schema_migrations WHERE migration='admission-ledger-contract-v5'"
    )
    conn.execute(
        "INSERT OR IGNORE INTO report_schema_migrations(migration,applied_ts) "
        "VALUES ('admission-ledger-contract-v2','2026-08-21T00:00:00Z')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO report_schema_migrations(migration,applied_ts) "
        "VALUES ('admission-ledger-contract-v3','2026-08-21T00:00:00Z')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO report_schema_migrations(migration,applied_ts) "
        "VALUES ('admission-ledger-contract-v4','2026-08-21T00:00:00Z')"
    )

    ensure_report_run_schema(conn)

    with pytest.raises(sqlite3.IntegrityError, match="invalid report admission"):
        conn.execute(
            """INSERT INTO report_admission_attempts
               (run_id,status,error_code,error_message,attempted_ts)
               VALUES (?,'succeeded','forged','forged','2026-08-22T00:00:00Z')""",
            (run_id,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="invalid report admission"):
        conn.execute(
            """INSERT INTO report_admission_attempts
               (run_id,status,error_code,error_message,attempted_ts)
               VALUES (?,'succeeded',NULL,NULL,'2026-99-99T00:00:00Z')""",
            (run_id,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="invalid report admission"):
        conn.execute(
            """INSERT INTO report_admission_attempts
               (run_id,status,error_code,error_message,attempted_ts)
               VALUES (?,'succeeded',NULL,NULL,'2026-08-22T24:00:00Z')""",
            (run_id,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="invalid report admission"):
        conn.execute(
            """INSERT INTO report_admission_attempts
               (run_id,status,error_code,error_message,attempted_ts)
               VALUES ('unknown-run','succeeded',NULL,NULL,'2026-08-22T00:00:00Z')"""
        )
    with pytest.raises(sqlite3.IntegrityError, match="invalid report admission"):
        conn.execute(
            """INSERT INTO report_admission_attempts
               (run_id,status,error_code,error_message,attempted_ts)
               VALUES (?,'failed','admission_finalize_failed','','2026-08-22T00:00:00Z')""",
            (run_id,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="invalid report admission"):
        conn.execute(
            """INSERT INTO report_admission_attempts
               (run_id,status,error_code,error_message,attempted_ts)
               VALUES (?,'failed',NULL,'ValueError','2026-08-22T00:00:00Z')""",
            (run_id,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="invalid report admission"):
        conn.execute(
            """INSERT INTO report_admission_attempts
               (run_id,status,error_code,error_message,attempted_ts)
               VALUES (?,'failed','forged','ValueError','2026-08-22T00:00:00Z')""",
            (run_id,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="invalid report admission"):
        conn.execute(
            """INSERT INTO report_admission_attempts
               (run_id,status,error_code,error_message,attempted_ts)
               VALUES (?,'failed','admission_lineage_failed','ValueError','2026-08-22T00:00:02Z')""",
            (run_id,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="invalid report admission"):
        conn.execute(
            """INSERT INTO report_admission_attempts
               (run_id,status,error_code,error_message,attempted_ts)
               VALUES (?,'failed','admission_finalize_failed','Value.Error',
                       '2026-08-22T00:00:03Z')""",
            (run_id,),
        )
    conn.execute(
        """INSERT INTO report_admission_attempts
           (run_id,status,error_code,error_message,attempted_ts)
           VALUES (?,'failed','admission_finalize_failed','ValueError','2026-08-22T00:00:00Z')""",
        (run_id,),
    )


def test_fresh_admission_table_check_rejects_dotted_exception_name(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "fresh-admission-check.db")
    ensure_report_run_schema(conn)
    run_id = start_report_run(
        conn, report_kind="daily", configuration={}, code_version=TEST_SHA
    )
    conn.execute("DROP TRIGGER report_admission_attempts_valid_insert")

    conn.execute(
        """INSERT INTO report_admission_attempts
           (run_id,status,error_code,error_message,attempted_ts)
           VALUES (?,'failed','admission_finalize_failed','ValueError',
                   '2026-08-22T00:00:00Z')""",
        (run_id,),
    )
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        conn.execute(
            """INSERT INTO report_admission_attempts
               (run_id,status,error_code,error_message,attempted_ts)
               VALUES (?,'failed','admission_finalize_failed','Value.Error',
                       '2026-08-22T00:00:01Z')""",
            (run_id,),
        )


@pytest.mark.parametrize(
    ("case", "status", "error_code", "error_message", "attempted_ts", "null_run"),
    [
        ("status", None, None, None, "2026-08-22T00:00:00Z", False),
        ("error-code", "failed", None, "ValueError", "2026-08-22T00:00:00Z", False),
        ("error-message", "failed", "admission_finalize_failed", None,
         "2026-08-22T00:00:00Z", False),
        ("error-message-whitespace", "failed", "admission_finalize_failed", "\t",
         "2026-08-22T00:00:00Z", False),
        ("error-message-dotted", "failed", "admission_finalize_failed", "Value.Error",
         "2026-08-22T00:00:00Z", False),
        ("timestamp", "succeeded", None, None, None, False),
        ("timestamp-year-zero", "succeeded", None, None,
         "0000-08-22T00:00:00Z", False),
        ("run-id", "succeeded", None, None, "2026-08-22T00:00:00Z", True),
    ],
)
def test_legacy_admission_scan_rejects_null_contract_fields(
    tmp_path: Path,
    case: str,
    status: str | None,
    error_code: str | None,
    error_message: str | None,
    attempted_ts: str | None,
    null_run: bool,
) -> None:
    conn = sqlite3.connect(tmp_path / f"invalid-legacy-admission-{case}.db")
    ensure_report_run_schema(conn)
    run_id = start_report_run(
        conn, report_kind="daily", configuration={}, code_version=TEST_SHA
    )
    conn.execute("DROP TABLE report_admission_attempts")
    conn.execute(
        """CREATE TABLE report_admission_attempts (
               attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
               run_id TEXT,
               status TEXT,
               error_code TEXT,
               error_message TEXT,
               attempted_ts TEXT
           )"""
    )
    conn.execute(
        """INSERT INTO report_admission_attempts
           (run_id,status,error_code,error_message,attempted_ts)
           VALUES (?,?,?,?,?)""",
        (None if null_run else run_id, status, error_code, error_message, attempted_ts),
    )
    conn.execute(
        "DELETE FROM report_schema_migrations WHERE migration='admission-ledger-contract-v5'"
    )
    conn.execute(
        "INSERT OR IGNORE INTO report_schema_migrations(migration,applied_ts) "
        "VALUES ('admission-ledger-contract-v2','2026-08-21T00:00:00Z')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO report_schema_migrations(migration,applied_ts) "
        "VALUES ('admission-ledger-contract-v3','2026-08-21T00:00:00Z')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO report_schema_migrations(migration,applied_ts) "
        "VALUES ('admission-ledger-contract-v4','2026-08-21T00:00:00Z')"
    )
    conn.commit()

    with pytest.raises(RuntimeError, match="legacy report admission attempts"):
        ensure_report_run_schema(conn)


def test_legacy_terminal_run_is_preserved_but_not_trusted_as_verified_lineage():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE report_runs (
               run_id TEXT PRIMARY KEY,
               report_kind TEXT NOT NULL,
               status TEXT NOT NULL,
               started_ts TEXT NOT NULL,
               completed_ts TEXT,
               failed_ts TEXT,
               published_ts TEXT,
               generated_ts TEXT,
               scanned_universe_json TEXT,
               ranked_candidates_json TEXT,
               decisions_json TEXT,
               inputs_json TEXT,
               source_timestamps_json TEXT,
               config_json TEXT NOT NULL,
               config_hash TEXT NOT NULL,
               code_version TEXT NOT NULL,
               content_hash TEXT,
               artifact_path TEXT,
               markdown_path TEXT,
               error_message TEXT
           )"""
    )
    conn.execute(
        """INSERT INTO report_runs (
               run_id, report_kind, status, started_ts, published_ts,
               config_json, config_hash, code_version
           ) VALUES ('legacy-published', 'daily', 'published', ' ', ' ', '', '', '')"""
    )

    ensure_report_run_schema(conn)

    assert conn.execute(
        "SELECT status, publication_verified FROM report_runs "
        "WHERE run_id='legacy-published'"
    ).fetchone() == ("published", 0)
    ensure_setup_call_eval_schema(conn)
    with pytest.raises(sqlite3.IntegrityError, match="canonical published"):
        conn.execute(
            """INSERT INTO setup_call_evaluations (
                   asof_date, ticker, report_kind, report_run_id,
                   call_direction, close_asof
               ) VALUES ('2026-08-22', 'LEGACY-RUN', 'daily',
                         'legacy-published', 'long', 100)"""
        )


def test_retry_has_a_separate_cohort_and_setup_calls_never_union(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import trader_koo.paper_trades as paper_trades

    monkeypatch.setattr(paper_trades, "PAPER_TRADE_ENABLED", False)
    conn = sqlite3.connect(tmp_path / "report.db")
    first = _complete_and_publish(conn, tmp_path / "reports", _report("AAA", "BBB", accepted=1))
    assert admit_published_report(
        conn, run_id=first, report_dir=tmp_path / "reports"
    )["setup_calls"] == 1
    second = _complete_and_publish(conn, tmp_path / "reports", _report("AAA", "CCC", accepted=1))
    assert admit_published_report(
        conn, run_id=second, report_dir=tmp_path / "reports"
    )["setup_calls"] == 1
    assert conn.execute(
        "SELECT status,error_code,error_message FROM report_admission_attempts "
        "WHERE run_id=? ORDER BY attempt_id DESC LIMIT 1",
        (second,),
    ).fetchone() == ("succeeded", None, None)

    cohorts = {
        run_id: json.loads(snapshot)
        for run_id, snapshot in conn.execute(
            "SELECT run_id, scanned_universe_json FROM report_runs ORDER BY started_ts, run_id"
        )
    }
    assert cohorts[first] == ["AAA", "BBB"]
    assert cohorts[second] == ["AAA", "CCC"]
    canonical = {
        row[0]: (row[1], row[2])
        for row in conn.execute(
        "SELECT run_id, is_generation_canonical, superseded_by_run_id "
        "FROM report_runs"
        ).fetchall()
    }
    assert canonical == {first: (0, second), second: (1, None)}

    assert conn.execute(
        "SELECT COUNT(DISTINCT report_run_id) FROM setup_call_evaluations WHERE ticker = 'AAA'"
    ).fetchone()[0] == 2

    conn.execute(
        "UPDATE setup_call_evaluations SET status='scored', signed_return_pct=99, direction_hit=1 "
        "WHERE report_run_id = ?",
        (first,),
    )
    conn.execute(
        "UPDATE setup_call_evaluations SET status='scored', signed_return_pct=-5, direction_hit=0 "
        "WHERE report_run_id = ?",
        (second,),
    )
    stats = _eval_stats(conn, window_days=365)
    assert stats[("bullish_continuation", "long")]["sample"] == 2
    assert stats[("bullish_continuation", "long")]["expectancy_pct"] == 47

    third = _complete_and_publish(
        conn, tmp_path / "reports", _report("AAA", "DDD", accepted=1)
    )
    historical = resolve_published_report(
        conn, report_dir=tmp_path / "reports", run_id=first
    )
    assert historical is not None
    assert historical[1]["report_run"]["canonical_generation"] is False
    assert conn.execute(
        "SELECT superseded_by_run_id FROM report_runs WHERE run_id=?", (second,)
    ).fetchone() == (third,)


def test_setup_call_schema_migration_keeps_legacy_unlinked_and_removes_cohort_union(tmp_path: Path):
    conn = sqlite3.connect(tmp_path / "report.db")
    ensure_setup_call_eval_schema(conn)
    create_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'setup_call_evaluations'"
    ).fetchone()[0]
    conn.execute("DROP TABLE setup_call_evaluations")
    legacy_sql = create_sql.replace(
        "report_run_id TEXT REFERENCES report_runs(run_id),", ""
    ).replace(
        "UNIQUE(report_run_id, ticker)", "UNIQUE(asof_date, ticker, report_kind)"
    )
    conn.execute(legacy_sql)
    conn.execute(
        """INSERT INTO setup_call_evaluations (
               asof_date, ticker, report_kind, call_direction, close_asof
           ) VALUES ('2026-08-21', 'OLD', 'daily', 'long', 100.0)"""
    )

    ensure_setup_call_eval_schema(conn)

    legacy = conn.execute(
        "SELECT report_run_id FROM setup_call_evaluations WHERE ticker = 'OLD'"
    ).fetchone()
    assert legacy == (None,)
    unique_columns = [
        [
            str(column[2])
            for column in conn.execute(f"PRAGMA index_info({index[1]})").fetchall()
        ]
        for index in conn.execute("PRAGMA index_list(setup_call_evaluations)").fetchall()
        if int(index[2] or 0) == 1
    ]
    assert ["asof_date", "ticker", "report_kind"] not in unique_columns
    assert ["report_run_id", "ticker"] in unique_columns
    with pytest.raises(sqlite3.IntegrityError, match="canonical published"):
        conn.execute(
            """INSERT INTO setup_call_evaluations (
                   asof_date, ticker, report_kind, call_direction, close_asof
               ) VALUES ('2026-08-22', 'NEW-UNLINKED', 'daily', 'long', 100.0)"""
        )
    with pytest.raises(sqlite3.IntegrityError, match="lineage is immutable"):
        conn.execute(
            "UPDATE setup_call_evaluations SET report_run_id='forged' "
            "WHERE ticker='OLD'"
        )
    assert conn.execute(
        "SELECT report_run_id FROM setup_call_evaluations WHERE ticker='OLD'"
    ).fetchone() == (None,)


def test_paper_trade_lineage_guard_preserves_preexisting_legacy_rows():
    from trader_koo.paper_trade.schema import ensure_paper_trade_schema

    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE paper_trades (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               report_date TEXT NOT NULL,
               ticker TEXT NOT NULL,
               direction TEXT NOT NULL,
               entry_price REAL NOT NULL,
               entry_date TEXT NOT NULL,
               status TEXT NOT NULL DEFAULT 'open',
               setup_family TEXT
           )"""
    )
    conn.execute(
        """INSERT INTO paper_trades (
               report_date, ticker, direction, entry_price, entry_date, status
               ) VALUES ('2026-08-21', 'LEGACY', 'long', 100.0, '2026-08-21', 'open')"""
    )
    conn.commit()
    ensure_paper_trade_schema(conn)

    assert conn.execute(
        "SELECT report_run_id FROM paper_trades WHERE ticker = 'LEGACY'"
    ).fetchone() == (None,)
    with pytest.raises(sqlite3.IntegrityError, match="canonical published"):
        conn.execute(
            """INSERT INTO paper_trades (
                   report_date, ticker, direction, entry_price, entry_date, status,
                   campaign_id
               ) VALUES ('2026-08-22', 'NEW-UNLINKED', 'long', 100.0,
                         '2026-08-22', 'open', 'paper-v2')"""
        )
    with pytest.raises(sqlite3.IntegrityError, match="lineage is immutable"):
        conn.execute(
            "UPDATE paper_trades SET report_run_id='forged' WHERE ticker='LEGACY'"
        )
    assert conn.execute(
        "SELECT report_run_id FROM paper_trades WHERE ticker='LEGACY'"
    ).fetchone() == (None,)


def test_hash_mismatch_refuses_publication_and_keeps_previous_canonical(tmp_path: Path):
    conn = sqlite3.connect(tmp_path / "report.db")
    report_dir = tmp_path / "reports"
    published_id = _complete_and_publish(conn, report_dir, _report("AAA"))

    retry_id = start_report_run(
        conn,
        report_kind="daily",
        configuration={},
        code_version=TEST_SHA,
        started_ts="2026-08-22T11:59:00Z",
    )
    retry = _report("BBB")
    paths = write_reports(retry, report_dir, run_id=retry_id, publish_latest=False)
    artifact = Path(paths["json_path"])
    complete_report_run(
        conn,
        run_id=retry_id,
        report=retry,
        artifact_path=artifact,
        markdown_path=Path(paths["md_path"]),
        content_hash=sha256_file(artifact),
    )
    artifact.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        publish_report_run(conn, run_id=retry_id, report_dir=report_dir)

    _, canonical = latest_daily_report_json(report_dir, registry_conn=conn)
    assert canonical is not None
    assert canonical["report_run"]["run_id"] == published_id
    assert conn.execute("SELECT status FROM report_runs WHERE run_id = ?", (retry_id,)).fetchone()[0] == "completed"


def test_json_snapshot_and_markdown_are_both_sealed(tmp_path: Path):
    conn = sqlite3.connect(tmp_path / "report.db")
    report_dir = tmp_path / "reports"
    report = _report("SEALED")
    run_id = start_report_run(
        conn, report_kind="daily", configuration={}, code_version=TEST_SHA
    )
    paths = write_reports(report, report_dir, run_id=run_id, publish_latest=False)
    artifact = Path(paths["json_path"])
    report["signals"]["report_decisions"][0]["decision"] = "rejected"
    with pytest.raises(ValueError, match="differs from immutable JSON"):
        complete_report_run(
            conn,
            run_id=run_id,
            report=report,
            artifact_path=artifact,
            markdown_path=Path(paths["md_path"]),
            content_hash=sha256_file(artifact),
        )

    report = _report("MARKDOWN")
    run_id = _complete_and_publish(conn, report_dir, report)
    markdown = Path(conn.execute(
        "SELECT markdown_path FROM report_runs WHERE run_id=?", (run_id,)
    ).fetchone()[0])
    markdown.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Markdown artifact hash mismatch"):
        resolve_published_report(conn, report_dir=report_dir, run_id=run_id)


def test_exact_generated_timestamp_miss_never_returns_latest(tmp_path: Path):
    conn = sqlite3.connect(tmp_path / "report.db")
    report_dir = tmp_path / "reports"
    _complete_and_publish(conn, report_dir, _report("LATEST"))
    assert report_json_for_generated_ts(
        report_dir,
        "2026-08-22T12:00:01Z",
        registry_conn=conn,
    ) == (None, None)


def test_require_current_rejects_an_older_daily_generation(tmp_path: Path):
    conn = sqlite3.connect(tmp_path / "report.db")
    report_dir = tmp_path / "reports"
    older = _complete_and_publish(conn, report_dir, _report("OLDER"))
    newer_report = _report("NEWER")
    newer_report["generated_ts"] = "2026-08-22T13:00:00Z"
    newer = _complete_and_publish(conn, report_dir, newer_report)

    assert resolve_published_report(
        conn, report_dir=report_dir, run_id=older, require_current=True
    ) is None
    assert resolve_published_report(conn, report_dir=report_dir, run_id=older) is not None
    current = resolve_published_report(
        conn, report_dir=report_dir, run_id=newer, require_current=True
    )
    assert current is not None
    assert current[1]["report_run"]["run_id"] == newer


def test_resolver_rejects_symlinked_run_artifact(tmp_path: Path):
    conn = sqlite3.connect(tmp_path / "report.db")
    report_dir = tmp_path / "reports"
    run_id = _complete_and_publish(conn, report_dir, _report("SYMLINK"))
    artifact = Path(conn.execute(
        "SELECT artifact_path FROM report_runs WHERE run_id=?", (run_id,)
    ).fetchone()[0])
    target = tmp_path / "saved-report.json"
    target.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(target)

    with pytest.raises(ValueError, match="path does not match"):
        resolve_published_report(conn, report_dir=report_dir, run_id=run_id)


def test_reconciliation_copies_the_bytes_that_were_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from trader_koo.report import runs

    conn = sqlite3.connect(tmp_path / "report.db")
    report_dir = tmp_path / "reports"
    run_id = _complete_and_publish(conn, report_dir, _report("SEALED-BYTES"))
    artifact = Path(conn.execute(
        "SELECT artifact_path FROM report_runs WHERE run_id=?", (run_id,)
    ).fetchone()[0])
    expected = artifact.read_bytes()
    real_write = runs._atomic_write_if_changed
    mutated = False

    def mutate_after_verification(path: Path, data: bytes) -> None:
        nonlocal mutated
        if path.name == runs.LATEST_MANIFEST and not mutated:
            mutated = True
            artifact.write_text("{}\n", encoding="utf-8")
        real_write(path, data)

    monkeypatch.setattr(runs, "_atomic_write_if_changed", mutate_after_verification)
    reconcile_report_publication(conn, report_dir=report_dir)

    assert mutated
    assert (report_dir / "daily_report_latest.json").read_bytes() == expected


def test_schema_ensure_does_not_commit_caller_transaction(tmp_path: Path):
    db_path = tmp_path / "report.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE caller_work(value TEXT)")
    conn.commit()
    conn.execute("INSERT INTO caller_work VALUES ('uncommitted')")
    ensure_report_run_schema(conn)
    observer = sqlite3.connect(db_path)
    assert observer.execute("SELECT COUNT(*) FROM caller_work").fetchone() == (0,)
    observer.close()
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM caller_work").fetchone() == (0,)


def test_concurrent_schema_ensure_serializes_trigger_replacement(tmp_path: Path):
    db_path = tmp_path / "concurrent-report.db"
    barrier = threading.Barrier(4)

    def ensure() -> None:
        conn = sqlite3.connect(db_path, timeout=10)
        barrier.wait()
        ensure_report_run_schema(conn)
        conn.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _index: ensure(), range(4)))

    conn = sqlite3.connect(db_path)
    triggers = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        )
    }
    assert "report_runs_valid_transition" in triggers
    assert "report_runs_snapshot_immutable" in triggers
    conn.close()


def test_resolver_is_read_only_inside_backtest_snapshot(tmp_path: Path):
    db_path = tmp_path / "report.db"
    conn = sqlite3.connect(db_path)
    report_dir = tmp_path / "reports"
    run_id = _complete_and_publish(conn, report_dir, _report("SNAPSHOT"))
    conn.execute("CREATE TABLE caller_work(value TEXT)")
    conn.commit()
    schema_version = conn.execute("PRAGMA schema_version").fetchone()[0]
    conn.execute("BEGIN")
    conn.execute("INSERT INTO caller_work VALUES ('uncommitted')")
    resolved = resolve_published_report(
        conn, report_dir=report_dir, run_id=run_id
    )
    assert resolved is not None
    assert conn.in_transaction
    assert conn.execute("PRAGMA schema_version").fetchone()[0] == schema_version
    observer = sqlite3.connect(db_path)
    assert observer.execute("SELECT COUNT(*) FROM caller_work").fetchone() == (0,)
    observer.close()
    conn.rollback()


@pytest.mark.parametrize(
    ("assignment", "match"),
    [
        ("generation_key='daily:2099-01-01T00:00:00Z'", "generation key"),
        ("artifact_path='/tmp/unrelated.json'", "path"),
        ("started_ts='2099-01-01T00:00:00Z'", "timestamps are reversed"),
        ("config_hash='ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff'", "configuration hash"),
    ],
)
def test_resolver_rejects_forged_registry_fields(
    tmp_path: Path,
    assignment: str,
    match: str,
):
    conn = sqlite3.connect(tmp_path / "report.db")
    report_dir = tmp_path / "reports"
    run_id = _complete_and_publish(conn, report_dir, _report("FORGE"))
    # Simulate an out-of-band database owner bypassing the immutability guard.
    conn.execute("DROP TRIGGER report_runs_snapshot_immutable")
    conn.execute(f"UPDATE report_runs SET {assignment} WHERE run_id=?", (run_id,))
    with pytest.raises(ValueError, match=match):
        resolve_published_report(conn, report_dir=report_dir, run_id=run_id)


def test_persistent_pointer_trigger_needs_no_connection_udf(tmp_path: Path):
    db_path = tmp_path / "report.db"
    conn = sqlite3.connect(db_path)
    run_id = _complete_and_publish(conn, tmp_path / "reports", _report("POINTER"))
    conn.close()
    fresh = sqlite3.connect(db_path)
    with pytest.raises(sqlite3.IntegrityError, match="canonical report transition"):
        fresh.execute(
            "UPDATE report_runs SET is_generation_canonical=0 WHERE run_id=?",
            (run_id,),
        )
    fresh.close()


def test_null_legacy_lineage_is_excluded_from_calibration(tmp_path: Path):
    conn = sqlite3.connect(tmp_path / "report.db")
    run_id = _complete_and_publish(conn, tmp_path / "reports", _report("LINKED"))
    ensure_setup_call_eval_schema(conn)
    # Simulate a pre-guard row retained by migration, then reinstall the guard.
    conn.execute("DROP TRIGGER setup_call_evaluations_require_canonical_run")
    conn.execute(
        """
        INSERT INTO setup_call_evaluations (
            asof_date, ticker, report_kind, report_run_id, call_direction,
            setup_family, close_asof, status, signed_return_pct, direction_hit
        ) VALUES ('2026-08-21', 'LEGACY', 'daily', NULL, 'long',
                  'bullish_continuation', 100, 'scored', 99.0, 1)
        """
    )
    ensure_setup_call_eval_schema(conn)
    rows = [
        ("2026-08-21", "LINKED", run_id, -3.0),
    ]
    for asof, ticker, lineage, ret in rows:
        conn.execute(
            """
            INSERT INTO setup_call_evaluations (
                asof_date, ticker, report_kind, report_run_id, call_direction,
                setup_family, close_asof, status, signed_return_pct, direction_hit
            ) VALUES (?, ?, 'daily', ?, 'long', 'bullish_continuation', 100,
                      'scored', ?, ?)
            """,
            (asof, ticker, lineage, ret, int(ret > 0)),
        )
    stats = _eval_stats(conn, window_days=365)
    assert stats[("bullish_continuation", "long")] == {
        "sample": 1,
        "hit_rate_pct": 0.0,
        "expectancy_pct": -3.0,
    }
    artifact = Path(conn.execute(
        "SELECT artifact_path FROM report_runs WHERE run_id=?", (run_id,)
    ).fetchone()[0])
    artifact.unlink()
    assert _eval_stats(conn, window_days=365) == {}


def test_publication_crash_keeps_previous_manifest_and_is_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from trader_koo.report import runs

    conn = sqlite3.connect(tmp_path / "report.db")
    report_dir = tmp_path / "reports"
    first = _complete_and_publish(conn, report_dir, _report("FIRST"))
    retry = _report("SECOND")
    retry_id = start_report_run(
        conn, report_kind="daily", configuration={}, code_version=TEST_SHA,
        started_ts="2026-08-22T11:59:00Z",
    )
    paths = write_reports(retry, report_dir, run_id=retry_id, publish_latest=False)
    artifact = Path(paths["json_path"])
    complete_report_run(
        conn,
        run_id=retry_id,
        report=retry,
        artifact_path=artifact,
        markdown_path=Path(paths["md_path"]),
        content_hash=sha256_file(artifact),
    )
    real_atomic_write = runs._atomic_write
    failed = False

    def crash_after_db(path: Path, data: bytes) -> None:
        nonlocal failed
        if not failed and path.name == "daily_report_latest.manifest.json":
            failed = True
            raise OSError("simulated publication crash")
        real_atomic_write(path, data)

    monkeypatch.setattr(runs, "_atomic_write", crash_after_db)
    with pytest.raises(OSError, match="simulated publication crash"):
        publish_report_run(conn, run_id=retry_id, report_dir=report_dir)
    stale_manifest = json.loads(
        (report_dir / "daily_report_latest.manifest.json").read_text()
    )
    assert stale_manifest["run_id"] == first
    assert conn.execute(
        "SELECT status, is_generation_canonical FROM report_runs WHERE run_id=?",
        (retry_id,),
    ).fetchone() == ("published", 1)

    monkeypatch.setattr(runs, "_atomic_write", real_atomic_write)
    monkeypatch.setenv("TRADER_KOO_REPORT_DIR", str(report_dir))
    monkeypatch.setenv("TRADER_KOO_DB_PATH", str(tmp_path / "report.db"))
    _, recovered = latest_daily_report_json(report_dir)
    assert recovered is not None
    assert recovered["report_run"]["run_id"] == retry_id
    assert conn.execute(
        "SELECT is_generation_canonical,superseded_by_run_id FROM report_runs WHERE run_id=?",
        (first,),
    ).fetchone() == (0, retry_id)


def test_retention_never_prunes_registered_or_prepublication_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TRADER_KOO_REPORT_KEEP_FILES", "3")
    monkeypatch.setenv("TRADER_KOO_REPORT_MAX_AGE_DAYS", "99999")
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    for day in (1, 2, 3):
        stem = f"daily_report_2099010{day}T000000Z"
        (report_dir / f"{stem}.json").write_text("{}\n")
        (report_dir / f"{stem}.md").write_text("legacy\n")

    conn = sqlite3.connect(tmp_path / "report.db")
    first = _complete_and_publish(conn, report_dir, _report("FIRST"))
    first_paths = conn.execute(
        "SELECT artifact_path, markdown_path FROM report_runs WHERE run_id=?",
        (first,),
    ).fetchone()
    assert all(Path(path).is_file() for path in first_paths)

    second = _complete_and_publish(conn, report_dir, _report("SECOND"))
    second_paths = conn.execute(
        "SELECT artifact_path, markdown_path FROM report_runs WHERE run_id=?",
        (second,),
    ).fetchone()
    for path in (*first_paths, *second_paths):
        assert Path(path).is_file()
    assert not Path(first_paths[0]).with_name(
        f"{Path(first_paths[0]).name}.published.json"
    ).exists()
    _, payload = latest_daily_report_json(report_dir, registry_conn=conn)
    assert payload is not None
    assert payload["report_run"]["run_id"] == second


def test_concurrent_publications_leave_registry_and_manifest_on_same_run(tmp_path: Path):
    db_path = tmp_path / "report.db"
    report_dir = tmp_path / "reports"
    conn = sqlite3.connect(db_path, timeout=10)
    first = _complete_only(conn, report_dir, _report("FIRST"))
    second = _complete_only(conn, report_dir, _report("SECOND"))
    conn.close()

    def publish(run_id: str) -> None:
        worker = sqlite3.connect(db_path, timeout=10)
        try:
            publish_report_run(worker, run_id=run_id, report_dir=report_dir)
        finally:
            worker.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(publish, (first, second)))

    verify = sqlite3.connect(db_path)
    db_canonical = verify.execute(
        "SELECT run_id FROM report_runs "
        "WHERE status='published' AND is_generation_canonical=1"
    ).fetchone()[0]
    manifest = json.loads((report_dir / "daily_report_latest.manifest.json").read_text())
    assert manifest["run_id"] == db_canonical
    _, payload = latest_daily_report_json(report_dir, registry_conn=verify)
    assert payload is not None
    assert payload["report_run"]["run_id"] == db_canonical
    verify.close()


def test_reader_reconciliation_cannot_overwrite_a_newer_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from trader_koo.report import runs

    db_path = tmp_path / "report.db"
    report_dir = tmp_path / "reports"
    conn = sqlite3.connect(db_path, timeout=10)
    _complete_and_publish(conn, report_dir, _report("FIRST"))
    second_report = _report("SECOND")
    second_report["generated_ts"] = "2026-08-22T13:00:00Z"
    second = _complete_only(conn, report_dir, second_report)
    conn.close()

    reader_resolved = threading.Event()
    let_reader_finish = threading.Event()
    errors: list[BaseException] = []
    real_resolve = runs._resolve_current_publication

    def pause_reader(*args, **kwargs):
        result = real_resolve(*args, **kwargs)
        if threading.current_thread().name == "reader":
            reader_resolved.set()
            assert let_reader_finish.wait(timeout=10)
        return result

    monkeypatch.setattr(runs, "_resolve_current_publication", pause_reader)

    def reconcile() -> None:
        worker = sqlite3.connect(db_path, timeout=10)
        try:
            reconcile_report_publication(worker, report_dir=report_dir)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            worker.close()

    def publish() -> None:
        worker = sqlite3.connect(db_path, timeout=10)
        try:
            publish_report_run(worker, run_id=second, report_dir=report_dir)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            worker.close()

    reader = threading.Thread(target=reconcile, name="reader")
    publisher = threading.Thread(target=publish, name="publisher")
    reader.start()
    assert reader_resolved.wait(timeout=10)
    publisher.start()
    let_reader_finish.set()
    reader.join(timeout=10)
    publisher.join(timeout=10)

    assert not errors
    assert not reader.is_alive() and not publisher.is_alive()
    verify = sqlite3.connect(db_path)
    assert resolve_published_report(
        verify, report_dir=report_dir, run_id=second, require_current=True
    ) is not None
    manifest = json.loads((report_dir / "daily_report_latest.manifest.json").read_text())
    assert manifest["run_id"] == second
    verify.close()


@pytest.mark.parametrize(
    ("ok", "warnings", "generation_warnings"),
    [
        (False, ["setup_quality_scoring_failed"], []),
        (True, [], ["setup_quality_scoring_failed"]),
    ],
)
def test_partial_quality_report_cannot_complete_or_admit(
    tmp_path: Path,
    ok: bool,
    warnings: list[str],
    generation_warnings: list[str],
):
    conn = sqlite3.connect(tmp_path / "report.db")
    report = _report("PARTIAL")
    report["ok"] = ok
    report["warnings"] = warnings
    report["generation_warnings"] = generation_warnings
    run_id = start_report_run(
        conn, report_kind="daily", configuration={}, code_version=TEST_SHA
    )
    paths = write_reports(report, tmp_path / "reports", run_id=run_id, publish_latest=False)
    artifact = Path(paths["json_path"])
    with pytest.raises(ValueError, match="degraded|partial-quality"):
        complete_report_run(
            conn,
            run_id=run_id,
            report=report,
            artifact_path=artifact,
            markdown_path=Path(paths["md_path"]),
            content_hash=sha256_file(artifact),
        )
    assert conn.execute(
        "SELECT status FROM report_runs WHERE run_id = ?", (run_id,)
    ).fetchone() == ("started",)
    assert conn.execute(
        "SELECT COUNT(*) FROM report_run_decisions WHERE run_id = ?", (run_id,)
    ).fetchone() == (0,)


def test_admission_uses_immutable_inputs_and_rolls_back_as_one_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import trader_koo.paper_trades as paper_trades
    import trader_koo.report.setup_scoring as setup_scoring

    conn = sqlite3.connect(tmp_path / "report.db")
    report = _report("AAA")
    run_id = _complete_and_publish(conn, tmp_path / "reports", report)
    monkeypatch.setattr(setup_scoring, "SETUP_EVAL_ENABLED", True)
    monkeypatch.setattr(paper_trades, "PAPER_TRADE_ENABLED", True)

    def fail_after_setup_calls(*args, **kwargs):
        raise RuntimeError("paper admission failed")

    monkeypatch.setattr(paper_trades, "create_paper_trades_from_report", fail_after_setup_calls)
    report["signals"]["setup_quality_top"][0]["ticker"] = "MUTATED"
    with pytest.raises(RuntimeError, match="paper admission failed"):
        admit_published_report(conn, run_id=run_id, report_dir=tmp_path / "reports")
    assert conn.execute("SELECT COUNT(*) FROM setup_call_evaluations").fetchone()[0] == 0
    assert conn.execute(
        "SELECT status,error_code,error_message FROM report_admission_attempts "
        "WHERE run_id=?",
        (run_id,),
    ).fetchone() == (
        "failed", "admission_paper_trade_persistence_failed", "RuntimeError"
    )


def test_admission_records_setup_persistence_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trader_koo.report.setup_scoring as setup_scoring

    conn = sqlite3.connect(tmp_path / "setup-failure.db")
    run_id = _complete_and_publish(conn, tmp_path / "reports", _report("SETUP"))
    monkeypatch.setattr(setup_scoring, "SETUP_EVAL_ENABLED", True)
    monkeypatch.setattr(
        setup_scoring, "_persist_setup_call_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("setup failed")),
    )

    with pytest.raises(RuntimeError, match="setup failed"):
        admit_published_report(conn, run_id=run_id, report_dir=tmp_path / "reports")

    assert conn.execute(
        "SELECT error_code,error_message FROM report_admission_attempts WHERE run_id=?",
        (run_id,),
    ).fetchone() == ("admission_setup_persistence_failed", "RuntimeError")


def test_admission_records_finalize_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import trader_koo.report.runs as runs

    conn = sqlite3.connect(tmp_path / "finalize-failure.db")
    run_id = _complete_and_publish(conn, tmp_path / "reports", _report("FINAL"))
    real_record = runs._record_admission_attempt

    def fail_success_attempt(*args, **kwargs):
        if kwargs["status"] == "succeeded":
            raise RuntimeError("finalize failed")
        return real_record(*args, **kwargs)

    monkeypatch.setattr(runs, "_record_admission_attempt", fail_success_attempt)
    with pytest.raises(RuntimeError, match="finalize failed"):
        admit_published_report(conn, run_id=run_id, report_dir=tmp_path / "reports")

    assert conn.execute(
        "SELECT error_code,error_message FROM report_admission_attempts WHERE run_id=?",
        (run_id,),
    ).fetchone() == ("admission_finalize_failed", "RuntimeError")


def test_admission_rejects_caller_transaction_before_audit(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "caller-transaction.db")
    conn.execute("CREATE TABLE caller_work(value TEXT)")
    conn.execute("INSERT INTO caller_work VALUES ('uncommitted')")

    with pytest.raises(RuntimeError, match="clean transaction boundary"):
        admit_published_report(conn, run_id="unknown", report_dir=tmp_path / "reports")

    assert conn.execute("SELECT COUNT(*) FROM caller_work").fetchone()[0] == 1
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='report_admission_attempts'"
    ).fetchone() is None


def test_email_dispatch_happens_only_after_publication_and_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from trader_koo.scripts import generate_daily_report as command

    db_path = tmp_path / "report.db"
    report_dir = tmp_path / "reports"
    events: list[str] = []
    real_publish = command.publish_report_run

    def publish_then_record(*args, **kwargs):
        result = real_publish(*args, **kwargs)
        events.append("published")
        return result

    def admit_then_record(*args, **kwargs):
        events.append("admitted")
        return {"setup_calls": 0, "paper_trades": 0}

    def send_after_publish(*args, **kwargs):
        assert events == ["published", "admitted"]
        events.append("email")
        return {"sent_count": 1, "failed_count": 0, "skipped_duplicate_count": 0}

    monkeypatch.setattr(command, "fetch_report_payload", lambda **kwargs: _report("AAA"))
    real_start_report_run = start_report_run
    monkeypatch.setattr(
        command,
        "start_report_run",
        lambda conn, **kwargs: real_start_report_run(
            conn, **kwargs, started_ts="2026-08-22T11:59:00Z"
        ),
    )
    monkeypatch.setattr(command, "publish_report_run", publish_then_record)
    monkeypatch.setattr(command, "admit_published_report", admit_then_record)
    monkeypatch.setattr(command, "send_report_email", send_after_publish)
    monkeypatch.setattr(
        command,
        "send_llm_failure_alert_email",
        lambda *args, **kwargs: {"attempted": False, "reason": "not_needed"},
    )
    monkeypatch.setattr(command, "_email_transport", lambda: "smtp")
    monkeypatch.setattr(command, "_smtp_cfg", lambda: {"to_email": "reader@example.com"})
    monkeypatch.setattr(command, "_resend_cfg", lambda: {})
    monkeypatch.setattr(
        command,
        "effective_report_configuration",
        lambda args: {"test": "exact-config"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_daily_report",
            "--db-path",
            str(db_path),
            "--out-dir",
            str(report_dir),
            "--run-log",
            str(tmp_path / "run.log"),
            "--send-email",
        ],
    )

    command.main()
    assert events == ["published", "admitted", "email"]


def test_generator_false_payload_fails_run_even_without_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from trader_koo.scripts import generate_daily_report as command

    report = _report("PARTIAL")
    report["ok"] = False
    report["warnings"] = []
    report["generation_warnings"] = []
    db_path = tmp_path / "report.db"
    called: list[str] = []
    monkeypatch.setattr(command, "fetch_report_payload", lambda **kwargs: report)
    monkeypatch.setattr(
        command,
        "publish_report_run",
        lambda *args, **kwargs: called.append("published"),
    )
    monkeypatch.setattr(
        command,
        "admit_published_report",
        lambda *args, **kwargs: called.append("admitted"),
    )
    monkeypatch.setattr(
        command,
        "effective_report_configuration",
        lambda args: {"test": "partial"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_daily_report",
            "--db-path",
            str(db_path),
            "--out-dir",
            str(tmp_path / "reports"),
            "--run-log",
            str(tmp_path / "run.log"),
        ],
    )

    with pytest.raises(ValueError, match="degraded"):
        command.main()

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT status FROM report_runs").fetchone() == ("failed",)
    assert conn.execute("SELECT COUNT(*) FROM report_run_decisions").fetchone() == (0,)
    assert called == []


def test_effective_config_captures_defaults_and_redacts_secrets(
    monkeypatch: pytest.MonkeyPatch,
):
    import argparse
    from trader_koo.scripts.generate_daily_report import effective_report_configuration

    monkeypatch.setenv("TRADER_KOO_SETUP_EVAL_TRACK_LIMIT", "17")
    monkeypatch.setenv("TRADER_KOO_SMTP_PASSWORD", "do-not-store")
    args = argparse.Namespace(
        db_path="/data/test.db",
        out_dir="/data/reports",
        run_log="/data/run.log",
        tail_lines=80,
        report_kind="daily",
        send_email=True,
    )
    config = effective_report_configuration(args)
    assert "SETUP_EVAL_TRACK_LIMIT" in config["setup_scoring"]
    assert config["environment_overrides"]["TRADER_KOO_SETUP_EVAL_TRACK_LIMIT"] == "17"
    assert "TRADER_KOO_SMTP_PASSWORD" not in config["environment_overrides"]
    assert config["paper_trade"]["critic_fail_open"] is False
    assert config["paper_trade"]["execution"] == {
        "entry_slippage_bps": 10.0,
        "exit_slippage_bps": 10.0,
        "commission_per_trade": 5.0,
        "short_borrow_annual_pct": 3.0,
        "max_adv_pct": 15.0,
    }
    assert "CONVICTION_A_HIGH" in config["critic"]


def test_new_setup_calls_and_paper_trades_require_report_lineage(tmp_path: Path):
    from trader_koo.paper_trades import create_paper_trades_from_report
    from trader_koo.paper_trade.schema import ensure_paper_trade_schema
    from trader_koo.report.setup_scoring import _persist_setup_call_candidates

    conn = sqlite3.connect(tmp_path / "report.db")
    ensure_setup_call_eval_schema(conn)
    ensure_paper_trade_schema(conn)
    setup = _report("AAA")["signals"]["report_decisions"][0]["inputs"]
    with pytest.raises(ValueError, match="report-run lineage"):
        _persist_setup_call_candidates(
            conn,
            generated_ts="2026-08-22T12:00:00Z",
            report_kind="daily",
            asof_date="2026-08-21",
            setup_rows=[setup],
        )
    with pytest.raises(ValueError, match="report-run lineage"):
        create_paper_trades_from_report(
            conn,
            setup_rows=[setup],
            report_date="2026-08-21",
            generated_ts="2026-08-22T12:00:00Z",
        )
    assert conn.execute("SELECT COUNT(*) FROM setup_call_evaluations").fetchone() == (0,)
    assert conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone() == (0,)

    def insert_setup(run_id: str | None, ticker: str) -> None:
        conn.execute(
            """INSERT INTO setup_call_evaluations (
                   asof_date, ticker, report_kind, generated_ts, report_run_id,
                   call_direction, close_asof
               ) VALUES ('2026-08-21', ?, 'daily', '2026-08-22T12:00:00Z',
                         ?, 'long', 100.0)""",
            (ticker, run_id),
        )

    def insert_trade(run_id: str | None, ticker: str, report_date: str) -> None:
        conn.execute(
            """INSERT INTO paper_trades (
                   report_date, generated_ts, report_run_id, ticker, direction,
                   entry_price, entry_date, status
               ) VALUES (?, '2026-08-22T12:00:00Z', ?, ?, 'long', 100.0,
                         ?, 'open')""",
            (report_date, run_id, ticker, report_date),
        )

    for insert in (
        lambda: insert_setup(None, "NULLCALL"),
        lambda: insert_trade(None, "NULLTRADE", "2026-08-21"),
    ):
        with pytest.raises(sqlite3.IntegrityError, match="canonical published"):
            insert()

    started_id = start_report_run(
        conn, report_kind="daily", configuration={}, code_version=TEST_SHA
    )
    for insert in (
        lambda: insert_setup(started_id, "STARTEDCALL"),
        lambda: insert_trade(started_id, "STARTEDTRADE", "2026-08-22"),
    ):
        with pytest.raises(sqlite3.IntegrityError, match="canonical published"):
            insert()

    canonical_id = _complete_and_publish(conn, tmp_path / "reports", _report("CANONICAL"))
    insert_setup(canonical_id, "CANONICAL")
    insert_trade(canonical_id, "CANONICAL", "2026-08-23")

    for insert in (
        lambda: insert_setup(canonical_id, "UNRELATEDCALL"),
        lambda: insert_trade(canonical_id, "UNRELATEDTRADE", "2026-08-23"),
    ):
        with pytest.raises(sqlite3.IntegrityError, match="accepted decision"):
            insert()

    for table, ticker in (
        ("setup_call_evaluations", "CANONICAL"),
        ("paper_trades", "CANONICAL"),
    ):
        for replacement in (None, started_id, "does-not-exist"):
            with pytest.raises(sqlite3.IntegrityError, match="lineage is immutable"):
                conn.execute(
                    f"UPDATE {table} SET report_run_id=? WHERE ticker=?",
                    (replacement, ticker),
                )
        assert conn.execute(
            f"SELECT report_run_id FROM {table} WHERE ticker=?", (ticker,)
        ).fetchone() == (canonical_id,)

    _complete_and_publish(conn, tmp_path / "reports", _report("RETRY"))
    for insert in (
        lambda: insert_setup(canonical_id, "SUPERSEDEDCALL"),
        lambda: insert_trade(canonical_id, "SUPERSEDEDTRADE", "2026-08-24"),
    ):
        with pytest.raises(sqlite3.IntegrityError, match="canonical published"):
            insert()


def test_code_version_supports_railway_and_fails_closed_when_unknown(
    monkeypatch: pytest.MonkeyPatch,
):
    from trader_koo.report import runs

    for name in (
        "TRADER_KOO_GIT_SHA",
        "RAILWAY_GIT_COMMIT_SHA",
        "GITHUB_SHA",
        "SOURCE_VERSION",
        "VERCEL_GIT_COMMIT_SHA",
    ):
        monkeypatch.delenv(name, raising=False)
    railway_sha = "b" * 40
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", railway_sha)
    assert runs.current_code_version() == railway_sha
    monkeypatch.delenv("RAILWAY_GIT_COMMIT_SHA")

    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "unknown")
    with pytest.raises(RuntimeError, match="full 40- or 64-character Git commit SHA"):
        runs.current_code_version()
    monkeypatch.delenv("RAILWAY_GIT_COMMIT_SHA")
    with pytest.raises(RuntimeError, match="full 40- or 64-character Git commit SHA"):
        start_report_run(
            sqlite3.connect(":memory:"),
            report_kind="daily",
            configuration={},
            code_version="test-sha",
        )

    def unavailable(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(runs.subprocess, "run", unavailable)
    with pytest.raises(RuntimeError, match="exact deployed code version"):
        runs.current_code_version()
