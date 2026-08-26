"""Read-only exact verifier for the frozen paper-schema v5 contract."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any

from trader_koo.paper_trade.schema_v5_migration import (
    DEFAULT_CONTRACT_PATH,
    SCHEMA_META_ROW_ID,
    TARGET_SCHEMA_VERSION,
    _collision_diagnostics,
    _strict_v2_accounting_diagnostics,
)


FROZEN_CONTRACT_ID = "paper-schema-contract-v5"
FROZEN_SEMANTIC_FINGERPRINT = (
    "82bbb2248f0b4a1d3f0cb2f5c5af60322e2f17a1f401ce5036c18578812d1191"
)


class PaperSchemaV5VerificationError(RuntimeError):
    """Exact, stable diagnostics for a rejected v5 database."""

    def __init__(self, diagnostics: list[dict[str, Any]]) -> None:
        self.diagnostics = tuple(diagnostics)
        super().__init__(json.dumps(
            diagnostics, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ))


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _normalize_sql(value: str | None) -> str:
    return " ".join(str(value or "").lower().split())


def _canonical_sql(value: str | None) -> str:
    """Normalize spacing/case outside quotes while preserving quoted bytes."""
    source = str(value or "")
    output: list[str] = []
    quote: str | None = None
    pending_space = False
    index = 0
    while index < len(source):
        character = source[index]
        if quote is not None:
            output.append(character)
            if character == quote:
                if index + 1 < len(source) and source[index + 1] == quote:
                    output.append(source[index + 1])
                    index += 1
                else:
                    quote = None
            index += 1
            continue
        if character in {"'", '"', "`", "["}:
            if pending_space and output:
                output.append(" ")
            pending_space = False
            quote = "]" if character == "[" else character
            output.append(character)
        elif character.isspace():
            pending_space = True
        else:
            if pending_space and output:
                output.append(" ")
            pending_space = False
            output.append(character.lower())
        index += 1
    return "".join(output).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({_quote(table)})")]


def _has_columns(conn: sqlite3.Connection, table: str, columns: tuple[str, ...]) -> bool:
    actual = set(_table_columns(conn, table))
    return all(column in actual for column in columns)


def _load_contract(contract_path: Path) -> tuple[dict[str, Any], Path]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("contract_id") != FROZEN_CONTRACT_ID
        or contract.get("target_schema_version") != TARGET_SCHEMA_VERSION
        or contract.get("fingerprint", {}).get("expected_sha256")
        != FROZEN_SEMANTIC_FINGERPRINT
    ):
        raise PaperSchemaV5VerificationError([{"code": "invalid_contract_identity"}])
    root = contract_path.parent
    target_path: Path | None = None
    for fixture in contract["fixtures"]["databases"]:
        path = root / str(fixture["path"])
        if _sha256(path) != fixture["sha256"]:
            raise PaperSchemaV5VerificationError([{
                "code": "fixture_hash_mismatch", "fixture": fixture["id"],
            }])
        if fixture["id"] == "exact_v5_target":
            target_path = path
    cases_path = root / str(contract["fixtures"]["path"])
    if _sha256(cases_path) != contract["fixtures"]["sha256"]:
        raise PaperSchemaV5VerificationError([{
            "code": "fixture_hash_mismatch", "fixture": "contract_cases",
        }])
    if target_path is None:
        raise PaperSchemaV5VerificationError([{
            "code": "target_fixture_missing",
        }])
    return contract, target_path


def _target_connection(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(path.read_text(encoding="utf-8"))
    return conn


def _governed_table_name(name: str) -> bool:
    return name.startswith(("paper_", "report_")) or name in {
        "bot_versions", "schema_migrations",
    }


def _temp_shadow_diagnostics(
    conn: sqlite3.Connection,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    governed_tables = {str(item["name"]) for item in contract["tables"]}
    governed_objects = governed_tables | {
        str(item["name"]) for item in contract["indexes"]
    } | {
        str(item["name"]) for item in contract["triggers"]
    }
    return [
        {
            "code": "temp_schema_overlap",
            "object_type": str(object_type),
            "name": str(name),
            "table": str(table),
        }
        for object_type, name, table in conn.execute(
            "SELECT type,name,tbl_name FROM sqlite_temp_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name,tbl_name"
        )
        if str(name) in governed_objects or str(table) in governed_tables
    ]


def _column_shape(conn: sqlite3.Connection, table: str) -> list[list[Any]]:
    return [
        [
            int(row[0]), str(row[1]), str(row[2]), int(row[3]), row[4],
            int(row[5]), int(row[6]),
        ]
        for row in conn.execute(f"PRAGMA table_xinfo({_quote(table)})")
    ]


def _foreign_key_shape(conn: sqlite3.Connection, table: str) -> list[list[Any]]:
    return [
        [
            int(row[0]), int(row[1]), str(row[2]), str(row[3]), str(row[4]),
            str(row[5]), str(row[6]), str(row[7]),
        ]
        for row in conn.execute(f"PRAGMA foreign_key_list({_quote(table)})")
    ]


def _index_shape(conn: sqlite3.Connection, index: str) -> list[list[Any]]:
    return [
        [
            int(row[0]), int(row[1]), row[2], int(row[3]), str(row[4]), int(row[5]),
        ]
        for row in conn.execute(f"PRAGMA index_xinfo({_quote(index)})")
    ]


def _object_diagnostics(
    conn: sqlite3.Connection,
    target: sqlite3.Connection,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    tables = {str(item["name"]): item for item in contract["tables"]}
    actual_tables = {
        str(name): str(sql or "")
        for name, sql in conn.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        )
    }
    governed_actual = {
        name for name in actual_tables if _governed_table_name(name)
    }
    for name in sorted(tables.keys() - governed_actual):
        diagnostics.append({"code": "missing_table", "table": name})
    for name in sorted(governed_actual - tables.keys()):
        diagnostics.append({"code": "unexpected_table", "table": name})
    for name in sorted(tables.keys() & governed_actual):
        target_sql = target.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()[0]
        actual_hash = hashlib.sha256(
            _normalize_sql(actual_tables[name]).encode()
        ).hexdigest()
        if (
            actual_hash != tables[name]["normalized_sql_sha256"]
            or _canonical_sql(actual_tables[name]) != _canonical_sql(str(target_sql))
        ):
            diagnostics.append({"code": "table_sql_mismatch", "table": name})
        expected_columns = _column_shape(target, name)
        actual_columns = _column_shape(conn, name)
        if actual_columns != expected_columns:
            diagnostics.append({
                "code": "ordered_columns_mismatch", "table": name,
                "expected": expected_columns, "actual": actual_columns,
            })
        expected_fks = _foreign_key_shape(target, name)
        actual_fks = _foreign_key_shape(conn, name)
        if actual_fks != expected_fks:
            diagnostics.append({
                "code": "ordered_foreign_keys_mismatch", "table": name,
                "expected": expected_fks, "actual": actual_fks,
            })

    actual_indexes = {
        str(name): (str(table), str(sql or ""))
        for name, table, sql in conn.execute(
            "SELECT name,tbl_name,sql FROM sqlite_master "
            "WHERE type='index' AND sql IS NOT NULL"
        )
    }
    target_indexes = {
        str(name): (str(table), str(sql or ""))
        for name, table, sql in target.execute(
            "SELECT name,tbl_name,sql FROM sqlite_master "
            "WHERE type='index' AND sql IS NOT NULL"
        )
    }
    contracted_indexes = {str(item["name"]): item for item in contract["indexes"]}
    expected_index_names = {
        name for name, item in contracted_indexes.items()
        if item["disposition"] != "removed"
    }
    for name, item in sorted(contracted_indexes.items()):
        if item["disposition"] == "removed":
            if name in actual_indexes:
                diagnostics.append({"code": "removed_index_present", "index": name})
            continue
        if name not in actual_indexes:
            diagnostics.append({"code": "missing_index", "index": name})
            continue
        actual_table, actual_sql = actual_indexes[name]
        expected_table, expected_sql = target_indexes[name]
        if actual_table != expected_table or _canonical_sql(actual_sql) != _canonical_sql(
            expected_sql
        ):
            diagnostics.append({"code": "index_sql_mismatch", "index": name})
        expected_shape = _index_shape(target, name)
        actual_shape = _index_shape(conn, name)
        if actual_shape != expected_shape:
            diagnostics.append({
                "code": "ordered_index_columns_mismatch", "index": name,
                "expected": expected_shape, "actual": actual_shape,
            })
    governed_indexes = {
        name for name, (table, _sql) in actual_indexes.items() if table in tables
    }
    for name in sorted(governed_indexes - expected_index_names):
        if name not in {
            item["name"] for item in contract["indexes"]
            if item["disposition"] == "removed"
        }:
            diagnostics.append({"code": "unexpected_index", "index": name})

    actual_triggers = {
        str(name): (str(table), str(sql or ""))
        for name, table, sql in conn.execute(
            "SELECT name,tbl_name,sql FROM sqlite_master WHERE type='trigger'"
        )
    }
    contracted_triggers = {
        str(item["name"]): item for item in contract["triggers"]
    }
    target_triggers = {
        str(name): str(sql or "")
        for name, sql in target.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='trigger'"
        )
    }
    for name, item in sorted(contracted_triggers.items()):
        if name not in actual_triggers:
            diagnostics.append({"code": "missing_trigger", "trigger": name})
            continue
        table, sql = actual_triggers[name]
        actual_hash = hashlib.sha256(_normalize_sql(sql).encode()).hexdigest()
        if (
            table != item["table"]
            or actual_hash != item["normalized_sql_sha256"]
            or _canonical_sql(sql) != _canonical_sql(target_triggers[name])
        ):
            diagnostics.append({"code": "trigger_sql_mismatch", "trigger": name})
    governed_triggers = {
        name for name, (table, _sql) in actual_triggers.items() if table in tables
    }
    for name in sorted(governed_triggers - contracted_triggers.keys()):
        diagnostics.append({"code": "unexpected_trigger", "trigger": name})

    for item in contract["defaults"]:
        table = str(item["table"])
        column = str(item["column"])
        if table not in actual_tables:
            continue
        actual = {
            str(row[1]): row[4]
            for row in conn.execute(f"PRAGMA table_info({_quote(table)})")
        }
        if column not in actual:
            diagnostics.append({
                "code": "missing_default_column", "table": table, "column": column,
            })
        elif actual[column] not in item["accepted_sql"]:
            diagnostics.append({
                "code": "default_mismatch", "table": table, "column": column,
                "expected": item["accepted_sql"], "actual": actual[column],
            })
    return diagnostics


def _logical_orphan_diagnostics(
    conn: sqlite3.Connection,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    primary_keys = {
        str(item["name"]): [str(column) for column in item["primary_key"]]
        for item in contract["tables"]
    }
    for relation in contract["foreign_keys"]["intentionally_absent"]:
        table = str(relation["table"])
        column = str(relation["column"])
        parent = str(relation["parent_table"])
        parent_key = primary_keys[parent]
        if len(parent_key) != 1:
            diagnostics.append({
                "code": "ambiguous_logical_parent_key", "table": parent,
            })
            continue
        parent_column = parent_key[0]
        if not _has_columns(conn, table, (column,)) or not _has_columns(
            conn, parent, (parent_column,),
        ):
            continue
        count = int(conn.execute(
            f"SELECT COUNT(*) FROM {_quote(table)} c "
            f"WHERE c.{_quote(column)} IS NOT NULL AND NOT EXISTS ("
            f"SELECT 1 FROM {_quote(parent)} p "
            f"WHERE p.{_quote(parent_column)}=c.{_quote(column)})"
        ).fetchone()[0])
        if count:
            diagnostics.append({
                "code": "orphan_logical_relation",
                "table": table,
                "column": column,
                "parent_table": parent,
                "row_count": count,
            })
    return diagnostics


def _data_diagnostics(
    conn: sqlite3.Connection,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    integrity = [str(row[0]) for row in conn.execute("PRAGMA integrity_check")]
    if integrity != ["ok"]:
        diagnostics.append({"code": "sqlite_integrity", "result": integrity})
    foreign_keys = [list(row) for row in conn.execute("PRAGMA foreign_key_check")]
    if foreign_keys:
        diagnostics.append({
            "code": "declared_foreign_keys", "row_count": len(foreign_keys),
            "first_break": foreign_keys[0],
        })
    diagnostics.extend(_collision_diagnostics(conn))
    diagnostics.extend(_logical_orphan_diagnostics(conn, contract))

    if _has_columns(conn, "paper_campaigns", ("campaign_id", "starting_capital")):
        row = conn.execute(
            "SELECT starting_capital FROM paper_campaigns WHERE campaign_id='paper-v2'"
        ).fetchone()
        if row is None:
            diagnostics.append({"code": "missing_paper_v2_campaign"})
        elif _has_columns(
            conn, "paper_trades",
            (
                "campaign_id", "accounting_status", "entry_price", "quantity",
                "entry_notional", "entry_commission", "exit_commission",
                "borrow_cost", "realized_pnl_usd",
            ),
        ):
            diagnostics.extend(_strict_v2_accounting_diagnostics(
                conn, Decimal(str(row[0])),
            ))
    return diagnostics


def _identity_diagnostics(
    conn: sqlite3.Connection,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    if not _has_columns(
        conn, "paper_trade_schema_meta",
        ("id", "schema_version", "contract_id", "schema_fingerprint"),
    ):
        diagnostics.append({"code": "missing_v5_identity"})
    else:
        rows = [tuple(row) for row in conn.execute(
            "SELECT id,schema_version,contract_id,schema_fingerprint "
            "FROM paper_trade_schema_meta ORDER BY id"
        )]
        expected = [(
            SCHEMA_META_ROW_ID,
            TARGET_SCHEMA_VERSION,
            contract["contract_id"],
            contract["fingerprint"]["expected_sha256"],
        )]
        if rows != expected:
            diagnostics.append({
                "code": "v5_identity_mismatch", "expected": expected, "actual": rows,
            })
    required_schema = [
        *contract["migration_ids"]["required_source"],
        contract["migration_ids"]["target"],
    ]
    if _has_columns(conn, "schema_migrations", ("migration_id",)):
        for migration_id in required_schema:
            count = int(conn.execute(
                "SELECT COUNT(*) FROM schema_migrations WHERE migration_id=?",
                (migration_id,),
            ).fetchone()[0])
            if count != 1:
                diagnostics.append({
                    "code": "schema_migration_identity_mismatch",
                    "migration_id": migration_id, "count": count,
                })
    for migration_id in contract["migration_ids"]["required_report_dependency"]:
        count = (
            int(conn.execute(
                "SELECT COUNT(*) FROM report_schema_migrations WHERE migration=?",
                (migration_id,),
            ).fetchone()[0])
            if _has_columns(conn, "report_schema_migrations", ("migration",)) else 0
        )
        if count != 1:
            diagnostics.append({
                "code": "report_migration_identity_mismatch",
                "migration_id": migration_id, "count": count,
            })
    return diagnostics


def _legacy_read_diagnostics(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    probes = {
        "legacy_trade_projection": (
            "SELECT report_date,ticker,direction,status FROM paper_trades LIMIT 1"
        ),
        "legacy_snapshot_projection": (
            "SELECT snapshot_date,open_trades,equity_index "
            "FROM paper_portfolio_snapshots LIMIT 1"
        ),
        "frozen_v1_campaign_projection": (
            "SELECT campaign_id,status,pnl_pct FROM paper_trades "
            "WHERE campaign_id='paper-v1' LIMIT 1"
        ),
        "legacy_snapshot_extension_projection": (
            "SELECT snapshot_ts,total_unrealized_pnl_pct "
            "FROM paper_portfolio_snapshots LIMIT 1"
        ),
    }
    for read_id, query in probes.items():
        try:
            conn.execute(query).fetchall()
        except sqlite3.DatabaseError:
            diagnostics.append({"code": "legacy_read_failed", "read": read_id})
    return diagnostics


def _semantic_fingerprint(contract: dict[str, Any]) -> str:
    payload = {
        key: deepcopy(contract[key])
        for key in contract["fingerprint"]["payload_sections"]
    }
    for redaction in contract["fingerprint_policy"]["canonical_redactions"]:
        if redaction["section"] != "fixtures":
            raise PaperSchemaV5VerificationError([{
                "code": "unsupported_fingerprint_redaction",
            }])
        database = next(
            item for item in payload["fixtures"]["databases"]
            if item["id"] == redaction["database_id"]
        )
        database[redaction["field"]] = redaction["replacement"]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sort_diagnostics(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        diagnostics,
        key=lambda item: (
            str(item.get("code", "")),
            str(item.get("table", "")),
            str(item.get("index", "")),
            str(item.get("trigger", "")),
            str(item.get("column", "")),
            json.dumps(item, sort_keys=True, separators=(",", ":")),
        ),
    )


def verify_paper_schema_v5(
    conn: sqlite3.Connection,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    """Verify exact v5 semantics without changing the connection or database."""
    if conn.in_transaction:
        raise PaperSchemaV5VerificationError([{
            "code": "transaction_already_active",
        }])
    transaction_before = conn.in_transaction
    changes_before = conn.total_changes
    schema_version_before = int(conn.execute("PRAGMA schema_version").fetchone()[0])
    data_version_before = int(conn.execute("PRAGMA data_version").fetchone()[0])
    contract, target_path = _load_contract(Path(contract_path))
    temp_diagnostics = _temp_shadow_diagnostics(conn, contract)
    if temp_diagnostics:
        raise PaperSchemaV5VerificationError(_sort_diagnostics(temp_diagnostics))
    target = _target_connection(target_path)
    try:
        object_diagnostics = _object_diagnostics(conn, target, contract)
        diagnostics = list(object_diagnostics)
        diagnostics.extend(_identity_diagnostics(conn, contract))
        if not object_diagnostics:
            diagnostics.extend(_data_diagnostics(conn, contract))
            diagnostics.extend(_legacy_read_diagnostics(conn))
    finally:
        target.close()
    if (
        conn.in_transaction != transaction_before
        or conn.total_changes != changes_before
        or int(conn.execute("PRAGMA schema_version").fetchone()[0])
        != schema_version_before
        or int(conn.execute("PRAGMA data_version").fetchone()[0]) != data_version_before
    ):
        diagnostics.append({"code": "verifier_changed_connection_state"})
    if diagnostics:
        raise PaperSchemaV5VerificationError(_sort_diagnostics(diagnostics))

    fingerprint = _semantic_fingerprint(contract)
    if fingerprint != FROZEN_SEMANTIC_FINGERPRINT:
        raise PaperSchemaV5VerificationError([{
            "code": "semantic_fingerprint_mismatch",
            "expected": FROZEN_SEMANTIC_FINGERPRINT,
            "actual": fingerprint,
        }])
    return {
        "status": "verified",
        "schema_version": TARGET_SCHEMA_VERSION,
        "contract_id": contract["contract_id"],
        "schema_fingerprint": fingerprint,
        "read_only": True,
    }
