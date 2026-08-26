from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Callable

import pytest

from trader_koo.paper_trade.schema import require_contracted_paper_schema
from trader_koo.paper_trade.schema_v5_migration import (
    _logical_database_hash,
    migrate_paper_schema_v4_to_v5,
)
from trader_koo.paper_trade.schema_v5_verifier import (
    FROZEN_SEMANTIC_FINGERPRINT,
    PaperSchemaV5VerificationError,
    verify_paper_schema_v5,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"


def _connect_fixture(name: str = "paper_schema_v5_target.sql") -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript((FIXTURES / name).read_text(encoding="utf-8"))
    return conn


def _connect_target_with_replacement(old: str, new: str) -> sqlite3.Connection:
    sql = (FIXTURES / "paper_schema_v5_target.sql").read_text(encoding="utf-8")
    assert old in sql
    conn = sqlite3.connect(":memory:")
    conn.executescript(sql.replace(old, new, 1))
    return conn


def test_exact_v5_fixture_returns_the_recomputed_frozen_fingerprint_read_only() -> None:
    conn = _connect_fixture()
    before = {
        "logical": _logical_database_hash(conn),
        "changes": conn.total_changes,
        "schema_version": conn.execute("PRAGMA schema_version").fetchone()[0],
        "data_version": conn.execute("PRAGMA data_version").fetchone()[0],
    }

    result = verify_paper_schema_v5(conn)

    assert result == {
        "status": "verified",
        "schema_version": 5,
        "contract_id": "paper-schema-contract-v5",
        "schema_fingerprint": FROZEN_SEMANTIC_FINGERPRINT,
        "read_only": True,
    }
    assert _logical_database_hash(conn) == before["logical"]
    assert conn.total_changes == before["changes"]
    assert conn.execute("PRAGMA schema_version").fetchone()[0] == before["schema_version"]
    assert conn.execute("PRAGMA data_version").fetchone()[0] == before["data_version"]
    assert not conn.in_transaction


def test_unrelated_external_write_is_not_reported_as_verifier_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PRAGMA data_version tracks other connections, not verifier writes."""
    from trader_koo.paper_trade import schema_v5_verifier as verifier

    db_path = tmp_path / "live.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        (FIXTURES / "paper_schema_v5_target.sql").read_text(encoding="utf-8")
    )
    conn.execute("CREATE TABLE runtime_heartbeat(id INTEGER PRIMARY KEY)")
    conn.commit()
    changes_before = conn.total_changes
    original = verifier._identity_diagnostics

    def write_from_other_connection(*args, **kwargs):
        diagnostics = original(*args, **kwargs)
        writer = sqlite3.connect(db_path)
        writer.execute("INSERT INTO runtime_heartbeat DEFAULT VALUES")
        writer.commit()
        writer.close()
        return diagnostics

    monkeypatch.setattr(verifier, "_identity_diagnostics", write_from_other_connection)

    result = verify_paper_schema_v5(conn)

    assert result["status"] == "verified"
    assert conn.total_changes == changes_before
    assert not conn.in_transaction
    conn.close()


def test_migrated_production_like_fixture_passes_exact_verification() -> None:
    conn = _connect_fixture("paper_schema_v4_legacy_production_like.sql")
    migrate_paper_schema_v4_to_v5(conn)
    before = _logical_database_hash(conn)

    result = verify_paper_schema_v5(conn)

    assert result["schema_fingerprint"] == FROZEN_SEMANTIC_FINGERPRINT
    assert _logical_database_hash(conn) == before
    assert conn.execute(
        "SELECT COUNT(*) FROM paper_trades WHERE campaign_id='paper-v1'"
    ).fetchone() == (42,)


def test_migrated_deployed_v4_fixture_passes_exact_verification() -> None:
    conn = _connect_fixture("paper_schema_v4_deployed.sql")
    migrate_paper_schema_v4_to_v5(conn)
    before = _logical_database_hash(conn)

    result = verify_paper_schema_v5(conn)

    assert result["schema_fingerprint"] == FROZEN_SEMANTIC_FINGERPRINT
    assert _logical_database_hash(conn) == before
    assert conn.execute(
        "SELECT id,campaign_id FROM paper_trades"
    ).fetchone() == (101, "paper-v1")
    assert conn.execute(
        "SELECT id,campaign_id FROM paper_portfolio_snapshots"
    ).fetchone() == (201, "paper-v1")


def test_preexisting_transaction_is_rejected_without_altering_it() -> None:
    conn = _connect_fixture()
    conn.execute("BEGIN")
    before = _logical_database_hash(conn)

    with pytest.raises(PaperSchemaV5VerificationError) as raised:
        verify_paper_schema_v5(conn)

    assert raised.value.diagnostics == ({"code": "transaction_already_active"},)
    assert conn.in_transaction
    assert _logical_database_hash(conn) == before
    conn.rollback()


def _extra_table(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE paper_unexpected(id INTEGER PRIMARY KEY)")


def _missing_index(conn: sqlite3.Connection) -> None:
    conn.execute("DROP INDEX idx_paper_trades_status")


def _wrong_index_order(conn: sqlite3.Connection) -> None:
    conn.execute("DROP INDEX idx_paper_trades_status")
    conn.execute(
        "CREATE INDEX idx_paper_trades_status "
        "ON paper_trades(status,campaign_id,entry_date)"
    )


def _quoted_literal_trigger_drift(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='trigger' "
        "AND name='paper_v1_trades_no_insert'"
    ).fetchone()
    conn.execute("DROP TRIGGER paper_v1_trades_no_insert")
    changed = str(row[0]).replace(
        "paper campaign v1 is immutable", "paper  campaign v1 is immutable",
    )
    conn.execute(changed)


def _extra_trigger(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TRIGGER unexpected_paper_trigger BEFORE INSERT ON paper_trades "
        "BEGIN SELECT RAISE(ABORT,'unexpected'); END"
    )


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (_extra_table, "unexpected_table"),
        (_missing_index, "missing_index"),
        (_wrong_index_order, "index_sql_mismatch"),
        (_quoted_literal_trigger_drift, "trigger_sql_mismatch"),
        (_extra_trigger, "unexpected_trigger"),
    ],
)
def test_object_drift_is_rejected_deterministically(
    mutate: Callable[[sqlite3.Connection], None],
    expected_code: str,
) -> None:
    messages: list[str] = []
    for _ in range(2):
        conn = _connect_fixture()
        mutate(conn)
        before = _logical_database_hash(conn)

        with pytest.raises(PaperSchemaV5VerificationError) as raised:
            verify_paper_schema_v5(conn)

        messages.append(str(raised.value))
        assert expected_code in {item["code"] for item in raised.value.diagnostics}
        assert _logical_database_hash(conn) == before
    assert messages[0] == messages[1]


@pytest.mark.parametrize(
    ("old", "new", "expected_code"),
    [
        (
            "notes TEXT DEFAULT ''",
            "notes TEXT DEFAULT 'changed'",
            "default_mismatch",
        ),
        (
            "report_run_id TEXT REFERENCES report_runs(run_id)",
            "report_run_id TEXT REFERENCES paper_campaigns(campaign_id)",
            "ordered_foreign_keys_mismatch",
        ),
        (
            "id INTEGER PRIMARY KEY AUTOINCREMENT,\n            report_date TEXT NOT NULL",
            "report_date TEXT NOT NULL,\n            id INTEGER PRIMARY KEY AUTOINCREMENT",
            "ordered_columns_mismatch",
        ),
    ],
)
def test_column_default_and_foreign_key_drift_is_exact(
    old: str,
    new: str,
    expected_code: str,
) -> None:
    conn = _connect_target_with_replacement(old, new)

    with pytest.raises(PaperSchemaV5VerificationError) as raised:
        verify_paper_schema_v5(conn)

    assert expected_code in {item["code"] for item in raised.value.diagnostics}


def test_all_nine_logical_relations_are_scanned_for_orphans() -> None:
    contract = json.loads((ROOT / "paper-schema-contract-v5.json").read_text())
    assert len(contract["foreign_keys"]["intentionally_absent"]) == 9
    conn = _connect_fixture()
    conn.execute(
        "INSERT INTO paper_trade_reflections "
        "(id,trade_id,ticker,direction) VALUES (1,999,'ORPHAN','long')"
    )
    conn.commit()
    before = _logical_database_hash(conn)

    with pytest.raises(PaperSchemaV5VerificationError) as raised:
        verify_paper_schema_v5(conn)

    assert {
        "code": "orphan_logical_relation",
        "table": "paper_trade_reflections",
        "column": "trade_id",
        "parent_table": "paper_trades",
        "row_count": 1,
    } in raised.value.diagnostics
    assert _logical_database_hash(conn) == before


def _insert_campaign(conn: sqlite3.Connection, campaign_id: str) -> None:
    conn.execute(
        """INSERT INTO paper_campaigns (
             campaign_id,label,policy_version,policy_hash,status,starting_capital,
             zero_admission_streak_limit,replay_live_parity,created_ts,updated_ts
           ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            campaign_id, "Malformed campaign", "test", "", "draft", 100000.0,
            3, "not_measured", "2026-08-26T00:00:00Z",
            "2026-08-26T00:00:00Z",
        ),
    )


def _insert_canonical_trade(conn: sqlite3.Connection, campaign_id: str) -> None:
    trigger_rows = conn.execute(
        "SELECT name,sql FROM sqlite_master WHERE type='trigger' "
        "AND tbl_name IN ('report_runs','report_run_decisions','paper_trades') "
        "ORDER BY name"
    ).fetchall()
    for name, _sql in trigger_rows:
        conn.execute(f'DROP TRIGGER "{name}"')
    conn.execute(
        """INSERT INTO report_runs (
             run_id,report_kind,status,started_ts,completed_ts,published_ts,
             generated_ts,scanned_universe_json,ranked_candidates_json,
             decisions_json,inputs_json,source_timestamps_json,config_json,
             config_hash,code_version,content_hash,markdown_hash,artifact_path,
             markdown_path,generation_key,is_generation_canonical,
             publication_verified
           ) VALUES (
             'blank-run','daily','published','2026-08-26T00:00:00Z',
             '2026-08-26T00:01:00Z','2026-08-26T00:02:00Z',
             '2026-08-26T00:01:00Z','["BLANK"]','["BLANK"]','[]','{}','{}','{}',
             ?,?,?,?,?,?,'daily:2026-08-26T00:01:00Z',1,1)""",
        (
            "a" * 64, "b" * 40, "c" * 64, "d" * 64,
            "/tmp/report.json", "/tmp/report.md",
        ),
    )
    conn.execute(
        """INSERT INTO report_run_decisions
             (run_id,ticker,selected_rank,decision,reason_codes_json,inputs_json)
           VALUES ('blank-run','BLANK',1,'accepted','[]','{}')"""
    )
    conn.execute(
        """INSERT INTO paper_trades (
             report_date,generated_ts,report_run_id,campaign_id,ticker,direction,
             entry_price,entry_date,status
           ) VALUES (
             '2026-08-26','2026-08-26T00:01:00Z','blank-run',?,'BLANK','long',
             100.0,'2026-08-26','open')""",
        (campaign_id,),
    )
    for _name, sql in trigger_rows:
        conn.execute(str(sql))
    conn.commit()


def test_blank_snapshot_campaign_is_rejected_even_with_blank_parent() -> None:
    conn = _connect_fixture()
    _insert_campaign(conn, "")
    conn.execute(
        "INSERT INTO paper_portfolio_snapshots (snapshot_date,campaign_id) "
        "VALUES ('2026-08-26','')"
    )
    conn.commit()
    before = _logical_database_hash(conn)

    with pytest.raises(PaperSchemaV5VerificationError) as raised:
        verify_paper_schema_v5(conn)

    assert "missing_snapshot_campaign" in {
        item["code"] for item in raised.value.diagnostics
    }
    assert _logical_database_hash(conn) == before


def test_whitespace_trade_campaign_is_rejected_even_with_matching_parent() -> None:
    conn = _connect_fixture()
    _insert_campaign(conn, "   ")
    _insert_canonical_trade(conn, "   ")
    before = _logical_database_hash(conn)

    with pytest.raises(PaperSchemaV5VerificationError) as raised:
        verify_paper_schema_v5(conn)

    assert "missing_trade_campaign" in {
        item["code"] for item in raised.value.diagnostics
    }
    assert _logical_database_hash(conn) == before


def test_unknown_snapshot_campaign_has_explicit_collision_diagnostic() -> None:
    conn = _connect_fixture()
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute(
        "INSERT INTO paper_portfolio_snapshots (snapshot_date,campaign_id) "
        "VALUES ('2026-08-26','unknown-campaign')"
    )
    conn.commit()
    conn.execute("PRAGMA foreign_keys=ON")

    with pytest.raises(PaperSchemaV5VerificationError) as raised:
        verify_paper_schema_v5(conn)

    assert "unknown_snapshot_campaign" in {
        item["code"] for item in raised.value.diagnostics
    }


def test_unknown_trade_campaign_has_explicit_collision_diagnostic() -> None:
    conn = _connect_fixture()
    conn.execute("PRAGMA foreign_keys=OFF")
    _insert_canonical_trade(conn, "unknown-campaign")
    conn.execute("PRAGMA foreign_keys=ON")

    with pytest.raises(PaperSchemaV5VerificationError) as raised:
        verify_paper_schema_v5(conn)

    assert "unknown_trade_campaign" in {
        item["code"] for item in raised.value.diagnostics
    }


def test_temp_table_cannot_shadow_invalid_main_campaign_data() -> None:
    conn = _connect_fixture()
    _insert_campaign(conn, "")
    conn.execute(
        "INSERT INTO paper_portfolio_snapshots (snapshot_date,campaign_id) "
        "VALUES ('2026-08-26','')"
    )
    conn.commit()
    table_sql = conn.execute(
        "SELECT sql FROM main.sqlite_master WHERE type='table' "
        "AND name='paper_portfolio_snapshots'"
    ).fetchone()[0]
    conn.execute(str(table_sql).replace("CREATE TABLE", "CREATE TEMP TABLE", 1))

    with pytest.raises(PaperSchemaV5VerificationError) as raised:
        verify_paper_schema_v5(conn)

    assert raised.value.diagnostics == ({
        "code": "temp_schema_overlap",
        "object_type": "table",
        "name": "paper_portfolio_snapshots",
        "table": "paper_portfolio_snapshots",
    },)


def test_previous_fingerprint_is_rejected_and_activation_stays_closed() -> None:
    conn = _connect_fixture()
    conn.execute(
        "UPDATE paper_trade_schema_meta SET schema_fingerprint=? WHERE id=1",
        ("82bbb2248f0b4a1d3f0cb2f5c5af60322e2f17a1f401ce5036c18578812d1191",),
    )
    conn.commit()

    with pytest.raises(PaperSchemaV5VerificationError) as raised:
        verify_paper_schema_v5(conn)

    assert "v5_identity_mismatch" in {
        item["code"] for item in raised.value.diagnostics
    }
    with pytest.raises(ValueError, match="activation interlock"):
        require_contracted_paper_schema(conn)


def test_minimal_same_name_malformed_case_fails_without_raw_sqlite_errors() -> None:
    manifest = json.loads((
        FIXTURES / "paper_schema_v5_contract_cases.json"
    ).read_text())
    case = next(
        item for item in manifest["cases"] if item["id"] == "same_name_malformed_object"
    )
    messages: list[str] = []
    for _ in range(2):
        conn = sqlite3.connect(":memory:")
        conn.executescript(";".join(case["setup_sql"]) + ";")
        with pytest.raises(PaperSchemaV5VerificationError) as raised:
            verify_paper_schema_v5(conn)
        messages.append(str(raised.value))
        assert "table_sql_mismatch" in {
            item["code"] for item in raised.value.diagnostics
        }
    assert messages[0] == messages[1]


def test_semantic_fingerprint_is_computed_only_after_database_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _connect_fixture()
    conn.execute("DROP INDEX idx_paper_trades_status")

    def forbidden(_contract: dict) -> str:
        raise AssertionError("fingerprint must not run for rejected schema")

    monkeypatch.setattr(
        "trader_koo.paper_trade.schema_v5_verifier._semantic_fingerprint",
        forbidden,
    )
    with pytest.raises(PaperSchemaV5VerificationError):
        verify_paper_schema_v5(conn)


def test_edited_contract_cannot_redefine_the_frozen_fingerprint(tmp_path: Path) -> None:
    contract = json.loads((ROOT / "paper-schema-contract-v5.json").read_text())
    contract["legacy_reads"][0]["required"] = False
    contract_path = tmp_path / "paper-schema-contract-v5.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    for fixture in contract["fixtures"]["databases"]:
        target = tmp_path / str(fixture["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / str(fixture["path"]), target)
    cases_target = tmp_path / str(contract["fixtures"]["path"])
    cases_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / str(contract["fixtures"]["path"]), cases_target)
    conn = _connect_fixture()

    with pytest.raises(PaperSchemaV5VerificationError) as raised:
        verify_paper_schema_v5(conn, contract_path=contract_path)

    assert raised.value.diagnostics[0]["code"] == "semantic_fingerprint_mismatch"
