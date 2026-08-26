from __future__ import annotations

import hashlib
import json
import sqlite3
from copy import deepcopy
from pathlib import Path

import pytest

from trader_koo.paper_trade.schema import (
    PAPER_TRADE_SCHEMA_VERSION,
    ensure_paper_trade_schema,
    require_contracted_paper_schema,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "paper-schema-contract-v5.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint(contract: dict) -> str:
    payload = {
        key: deepcopy(contract[key])
        for key in contract["fingerprint"]["payload_sections"]
    }
    for redaction in contract["fingerprint_policy"]["canonical_redactions"]:
        if redaction["section"] != "fixtures":
            raise AssertionError("unsupported contract fingerprint redaction")
        database = next(
            item for item in payload["fixtures"]["databases"]
            if item["id"] == redaction["database_id"]
        )
        database[redaction["field"]] = redaction["replacement"]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_contract_is_frozen_exhaustive_and_has_no_runtime_effect() -> None:
    contract = _load(CONTRACT_PATH)
    assert contract["contract_id"] == "paper-schema-contract-v5"
    assert contract["contract_state"] == "frozen"
    assert contract["source_schema_version"] == PAPER_TRADE_SCHEMA_VERSION == 4
    assert contract["target_schema_version"] == 5
    assert contract["runtime_effect"] == "none"
    assert contract["release_interlocks"] == {
        "pr1_changes_runtime": False,
        "automatic_v4_to_v5_migration_allowed": False,
        "activation_requires_exact_v5_verifier": True,
        "write_enablement_is_separate_transition": True,
        "human_activation_is_separate_transition": True,
    }

    table_names = {item["name"] for item in contract["tables"]}
    assert table_names == {
        "bot_versions", "paper_campaign_approvals", "paper_campaign_audit",
        "paper_campaign_experiments", "paper_campaign_preregistrations",
        "paper_campaigns", "paper_candidate_decisions", "paper_decision_sets",
        "paper_order_events", "paper_pending_orders",
        "paper_portfolio_snapshots", "paper_shadow_decision_sets",
        "paper_shadow_decisions", "paper_shadow_outcomes",
        "paper_shadow_policies", "paper_trade_annotations",
        "paper_trade_events", "paper_trade_reflections",
        "paper_trade_schema_meta", "paper_trades", "report_admission_attempts",
        "report_run_decisions", "report_runs", "report_schema_migrations",
        "schema_migrations",
    }
    assert len(contract["indexes"]) == 18
    assert len({item["name"] for item in contract["indexes"]}) == 18
    assert len(contract["triggers"]) == 51
    assert len({item["name"] for item in contract["triggers"]}) == 51
    assert len(contract["foreign_keys"]["required"]) == 18
    assert len(contract["foreign_keys"]["intentionally_absent"]) == 9
    assert len(contract["defaults"]) == 46
    assert len({(item["table"], item["column"]) for item in contract["defaults"]}) == 46
    assert contract["migration_ids"] == {
        "required_source": [
            "paper_campaign_v1_backfill_20260822",
            "paper_campaign_v2_inactive_governed_20260823",
        ],
        "required_report_dependency": ["admission-ledger-contract-v5"],
        "target": "paper_schema_contract_v5_20260826",
    }


def test_contract_fingerprint_and_fixture_are_hash_bound() -> None:
    contract = _load(CONTRACT_PATH)
    assert _fingerprint(contract) == contract["fingerprint"]["expected_sha256"]

    fixture_path = ROOT / contract["fixtures"]["path"]
    assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == (
        contract["fixtures"]["sha256"]
    )
    fixture = _load(fixture_path)
    assert fixture["fixture_set"] == "paper-schema-v5-contract-cases-v1"
    assert len(fixture["cases"]) == 13
    assert {case["id"] for case in fixture["cases"]} >= {
        "fresh_v4", "production_like_legacy_v4", "interrupted_rebuild",
        "same_name_malformed_object", "exact_v5",
    }
    for database in contract["fixtures"]["databases"]:
        sql_path = ROOT / database["path"]
        assert hashlib.sha256(sql_path.read_bytes()).hexdigest() == database["sha256"]
        conn = sqlite3.connect(":memory:")
        conn.executescript(sql_path.read_text(encoding="utf-8"))
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchone() == (25,)

    legacy = sqlite3.connect(":memory:")
    legacy.executescript((
        ROOT / "tests/fixtures/paper_schema_v4_legacy_production_like.sql"
    ).read_text(encoding="utf-8"))
    assert legacy.execute(
        "SELECT COUNT(*) FROM paper_trades WHERE campaign_id='paper-v1'"
    ).fetchone() == (42,)

    target = sqlite3.connect(":memory:")
    target.executescript((
        ROOT / "tests/fixtures/paper_schema_v5_target.sql"
    ).read_text(encoding="utf-8"))
    assert target.execute(
        "SELECT schema_version,contract_id,schema_fingerprint "
        "FROM paper_trade_schema_meta WHERE id=1"
    ).fetchone() == (
        5,
        contract["contract_id"],
        contract["fingerprint"]["expected_sha256"],
    )
    assert set(target.execute(
        "SELECT migration_id FROM schema_migrations"
    )) == {
        (migration_id,)
        for migration_id in (
            *contract["migration_ids"]["required_source"],
            contract["migration_ids"]["target"],
        )
    }
    assert set(target.execute(
        "SELECT migration FROM report_schema_migrations"
    )) == {
        (migration_id,)
        for migration_id in contract["migration_ids"]["required_report_dependency"]
    }
    assert target.execute(
        "SELECT campaign_id,status FROM paper_campaigns ORDER BY campaign_id"
    ).fetchall() == [("paper-v1", "frozen"), ("paper-v2", "draft")]

    for case in fixture["cases"]:
        if setup_sql := case.get("setup_sql"):
            conn = sqlite3.connect(":memory:")
            conn.executescript(";".join(setup_sql) + ";")


def test_contract_exhausts_the_current_v4_object_inventory() -> None:
    contract = _load(CONTRACT_PATH)
    conn = sqlite3.connect(":memory:")
    ensure_paper_trade_schema(conn)

    def in_scope(name: str) -> bool:
        return name.startswith(("paper_", "idx_paper", "report_", "idx_report")) or name in {
            "bot_versions", "idx_bot_versions_status", "schema_migrations",
        }

    governed_tables = {item["name"] for item in contract["tables"]}
    objects = list(conn.execute(
        "SELECT type,name,tbl_name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
    ))
    for kind, section in (
        ("table", "tables"), ("index", "indexes"), ("trigger", "triggers"),
    ):
        actual = {
            str(name) for object_type, name, table in objects
            if object_type == kind and str(table) in governed_tables
        }
        assert actual == {item["name"] for item in contract[section]}

    defaults = {
        (str(table), str(row[1])): row[4]
        for table, in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        if in_scope(str(table))
        for row in conn.execute(f'PRAGMA table_info("{table}")')
        if row[4] is not None
    }
    contracted_defaults = {
        (item["table"], item["column"]): item for item in contract["defaults"]
    }
    assert defaults.keys() == contracted_defaults.keys()
    for key, value in defaults.items():
        item = contracted_defaults[key]
        assert value in item["accepted_sql"] + item.get("changed_from", [])

    actual_fks = {
        (str(table), str(row[3]), str(row[2]), str(row[4]))
        for table, in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        if in_scope(str(table))
        for row in conn.execute(f'PRAGMA foreign_key_list("{table}")')
    }
    required_fks = {
        (item["table"], item["column"], item["parent_table"], item["parent_column"])
        for item in contract["foreign_keys"]["required"]
    }
    assert required_fks - actual_fks == {
        ("paper_trades", "campaign_id", "paper_campaigns", "campaign_id"),
        (
            "paper_portfolio_snapshots", "campaign_id",
            "paper_campaigns", "campaign_id",
        ),
    }


def test_target_fixture_freezes_exact_table_and_trigger_definitions() -> None:
    contract = _load(CONTRACT_PATH)
    conn = sqlite3.connect(":memory:")
    conn.executescript((
        ROOT / "tests/fixtures/paper_schema_v5_target.sql"
    ).read_text(encoding="utf-8"))

    def sql_hash(value: str) -> str:
        normalized = " ".join(value.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    for section, kind in (("tables", "table"), ("triggers", "trigger")):
        actual = dict(conn.execute(
            "SELECT name,sql FROM sqlite_master WHERE type=?", (kind,)
        ))
        for item in contract[section]:
            assert sql_hash(actual[item["name"]]) == item["normalized_sql_sha256"]

    target_indexes = {
        str(row[0]) for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
        )
    }
    assert target_indexes == {
        item["name"] for item in contract["indexes"]
        if item["disposition"] != "removed"
    }
    target_fks = {
        (str(table), str(row[3]), str(row[2]), str(row[4]))
        for table, in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        )
        for row in conn.execute(f'PRAGMA foreign_key_list("{table}")')
    }
    assert target_fks == {
        (item["table"], item["column"], item["parent_table"], item["parent_column"])
        for item in contract["foreign_keys"]["required"]
    }

    target_defaults = {
        (str(table), str(row[1])): row[4]
        for table, in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        )
        for row in conn.execute(f'PRAGMA table_info("{table}")')
        if row[4] is not None
    }
    contracted_defaults = {
        (item["table"], item["column"]): item["accepted_sql"]
        for item in contract["defaults"]
    }
    assert target_defaults.keys() <= contracted_defaults.keys()
    assert all(
        value in contracted_defaults[key]
        for key, value in target_defaults.items()
    )


def test_contract_names_every_removed_or_changed_compatibility_shape() -> None:
    contract = _load(CONTRACT_PATH)
    indexes = {item["name"]: item for item in contract["indexes"]}
    assert indexes["idx_paper_trades_legacy_compat"]["disposition"] == "removed"
    assert indexes["idx_paper_portfolio_legacy_compat"]["disposition"] == "removed"
    for name in (
        "idx_paper_trades_status", "idx_paper_trades_ticker",
        "idx_paper_trades_family",
    ):
        assert indexes[name]["disposition"] == "changed"
        assert indexes[name]["columns"][0] == "campaign_id"

    defaults = {
        (item["table"], item["column"]): item
        for item in contract["defaults"]
    }
    assert defaults[("paper_trades", "campaign_id")]["accepted_sql"] == [None]
    assert defaults[("paper_portfolio_snapshots", "campaign_id")][
        "accepted_sql"
    ] == [None]


def test_retained_trigger_definitions_are_exactly_frozen() -> None:
    contract = _load(CONTRACT_PATH)
    conn = sqlite3.connect(":memory:")
    ensure_paper_trade_schema(conn)
    actual = dict(conn.execute(
        "SELECT name,sql FROM sqlite_master WHERE type='trigger'"
    ))

    def sql_hash(value: str) -> str:
        normalized = " ".join(value.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    for trigger in contract["triggers"]:
        assert len(trigger["normalized_sql_sha256"]) == 64
        if trigger["disposition"] in {
            "retained_exact", "dependency_retained_exact",
        }:
            assert sql_hash(actual[trigger["name"]]) == trigger[
                "normalized_sql_sha256"
            ]

    changed = next(
        item for item in contract["triggers"]
        if item["name"] == "paper_v1_trades_no_update"
    )
    assert sql_hash(actual[changed["name"]]) != changed["normalized_sql_sha256"]


def test_activation_requires_the_exact_v5_contract() -> None:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        (ROOT / "tests/fixtures/paper_schema_v5_target.sql").read_text(encoding="utf-8")
    )
    assert require_contracted_paper_schema(conn)["contract_id"] == (
        "paper-schema-contract-v5"
    )
