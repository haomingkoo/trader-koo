"""Explicit, offline paper-schema v4 to v5 maintenance migration."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "paper-schema-contract-v5.json"
SOURCE_SCHEMA_VERSION = 4
TARGET_SCHEMA_VERSION = 5
SCHEMA_META_ROW_ID = 1


class PaperSchemaV5MigrationError(RuntimeError):
    """Fail-closed migration error with stable machine-readable diagnostics."""

    def __init__(self, diagnostics: list[dict[str, Any]]) -> None:
        self.diagnostics = tuple(diagnostics)
        super().__init__(json.dumps(
            diagnostics, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ))


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sql_text(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_sql(value: str | None) -> str:
    return " ".join(str(value or "").lower().split())


def _governed_table_name(name: str) -> bool:
    return name.startswith(("paper_", "report_")) or name in {
        "bot_versions", "schema_migrations",
    }


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({_quote(table)})")]


def _has_columns(conn: sqlite3.Connection, table: str, columns: tuple[str, ...]) -> bool:
    actual = set(_table_columns(conn, table))
    return all(column in actual for column in columns)


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"blob_hex": value.hex()}
    return value


def _digest_json(digest: Any, value: Any) -> None:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode()
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _rows_hash(
    conn: sqlite3.Connection,
    table: str,
    columns: list[str],
    order_by: list[str],
    where: str = "",
) -> str:
    projection = ",".join(_quote(column) for column in columns)
    ordering = ",".join(_quote(column) for column in order_by)
    digest = hashlib.sha256()
    for row in conn.execute(
        f"SELECT {projection} FROM {_quote(table)} {where} ORDER BY {ordering}"
    ):
        _digest_json(digest, [_json_value(value) for value in row])
    return digest.hexdigest()


def _table_state(
    conn: sqlite3.Connection,
    contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    for item in contract["tables"]:
        table = str(item["name"])
        if table == "paper_trade_schema_meta":
            continue
        columns = _table_columns(conn, table)
        primary_key = [str(column) for column in item["primary_key"]]
        source_primary_key = primary_key
        if table == "paper_portfolio_snapshots" and "id" not in columns:
            source_primary_key = ["snapshot_date"]
        state[table] = {
            "columns": columns,
            "count": int(conn.execute(
                f"SELECT COUNT(*) FROM {_quote(table)}"
            ).fetchone()[0]),
            "content_sha256": _rows_hash(
                conn, table, columns, source_primary_key,
            ),
            "primary_key_sha256": _rows_hash(
                conn, table, source_primary_key, source_primary_key,
            ),
            "source_primary_key": source_primary_key,
        }
    return state


def _logical_database_hash(conn: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for kind, name, table, sql in conn.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
    ):
        _digest_json(
            digest, [str(kind), str(name), str(table), _normalize_sql(sql)],
        )
    for table, in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ):
        name = str(table)
        columns = _table_columns(conn, name)
        _digest_json(digest, [name, columns])
        for row in conn.execute(
            f"SELECT {','.join(_quote(column) for column in columns)} "
            f"FROM {_quote(name)} ORDER BY rowid"
        ):
            _digest_json(digest, [_json_value(value) for value in row])
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'"
    ).fetchone():
        for row in conn.execute("SELECT name,seq FROM sqlite_sequence ORDER BY name"):
            _digest_json(digest, list(row))
    return digest.hexdigest()


def _schema_signature(
    conn: sqlite3.Connection,
    governed_only: bool = False,
) -> list[list[str]]:
    return [
        [str(kind), str(name), str(table), _normalize_sql(sql)]
        for kind, name, table, sql in conn.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name",
        )
        if not governed_only or _governed_table_name(str(table))
    ]


def _fixture_connection(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(path.read_text(encoding="utf-8"))
    return conn


def _first_group(
    conn: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
) -> tuple[int, list[Any]]:
    grouped = ",".join(_quote(column) for column in columns)
    count = int(conn.execute(
        f"SELECT COUNT(*) FROM (SELECT 1 FROM {_quote(table)} "
        f"GROUP BY {grouped} HAVING COUNT(*)>1)"
    ).fetchone()[0])
    if not count:
        return 0, []
    row = conn.execute(
        f"SELECT {grouped} FROM {_quote(table)} GROUP BY {grouped} "
        f"HAVING COUNT(*)>1 ORDER BY {grouped} LIMIT 1"
    ).fetchone()
    return count, [_json_value(value) for value in row]


def _count_and_first_id(
    conn: sqlite3.Connection,
    table: str,
    predicate: str,
) -> tuple[int, Any]:
    count = int(conn.execute(
        f"SELECT COUNT(*) FROM {_quote(table)} WHERE {predicate}"
    ).fetchone()[0])
    if not count:
        return 0, None
    columns = set(_table_columns(conn, table))
    key = "id" if "id" in columns else sorted(columns)[0]
    first = conn.execute(
        f"SELECT {_quote(key)} FROM {_quote(table)} WHERE {predicate} "
        f"ORDER BY {_quote(key)} LIMIT 1"
    ).fetchone()[0]
    return count, _json_value(first)


def _collision_diagnostics(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}

    def duplicate(code: str, table: str, columns: tuple[str, ...]) -> None:
        if not _has_columns(conn, table, columns):
            return
        count, first = _first_group(conn, table, columns)
        if count:
            found[code] = {
                "code": code, "table": table,
                "group_count": count, "first_key": first,
            }

    def rows(code: str, table: str, columns: tuple[str, ...], predicate: str) -> None:
        if not _has_columns(conn, table, columns):
            return
        count, first = _count_and_first_id(conn, table, predicate)
        if count:
            found[code] = {
                "code": code, "table": table,
                "row_count": count, "first_row_key": first,
            }

    duplicate(
        "duplicate_campaign_trade_key", "paper_trades",
        ("campaign_id", "report_date", "ticker", "direction"),
    )
    duplicate(
        "duplicate_campaign_snapshot_key", "paper_portfolio_snapshots",
        ("campaign_id", "snapshot_date"),
    )
    rows(
        "missing_trade_campaign", "paper_trades", ("campaign_id",),
        "campaign_id IS NULL OR TRIM(campaign_id)=''",
    )
    if _has_columns(conn, "paper_campaigns", ("campaign_id",)):
        rows(
            "unknown_trade_campaign", "paper_trades", ("campaign_id",),
            "campaign_id IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM paper_campaigns c WHERE c.campaign_id=paper_trades.campaign_id)",
        )
    rows(
        "missing_snapshot_campaign", "paper_portfolio_snapshots", ("campaign_id",),
        "campaign_id IS NULL OR TRIM(campaign_id)=''",
    )
    if _has_columns(conn, "paper_campaigns", ("campaign_id",)):
        rows(
            "unknown_snapshot_campaign", "paper_portfolio_snapshots", ("campaign_id",),
            "campaign_id IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM paper_campaigns c "
            "WHERE c.campaign_id=paper_portfolio_snapshots.campaign_id)",
        )
    if _has_columns(conn, "report_runs", ("run_id",)):
        rows(
            "orphan_trade_report_run", "paper_trades", ("report_run_id",),
            "report_run_id IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM report_runs r WHERE r.run_id=paper_trades.report_run_id)",
        )
    if (
        _has_columns(conn, "paper_trades", ("campaign_id", "report_run_id", "ticker"))
        and _has_columns(
            conn, "report_runs",
            ("run_id", "status", "publication_verified", "is_generation_canonical"),
        )
        and _has_columns(conn, "report_run_decisions", ("run_id", "ticker", "decision"))
    ):
        rows(
            "invalid_v2_trade_lineage", "paper_trades",
            ("campaign_id", "report_run_id", "ticker"),
            "campaign_id='paper-v2' AND NOT EXISTS ("
            "SELECT 1 FROM report_runs r JOIN report_run_decisions d ON d.run_id=r.run_id "
            "WHERE r.run_id=paper_trades.report_run_id AND r.status='published' "
            "AND r.publication_verified=1 AND r.is_generation_canonical=1 "
            "AND d.ticker=paper_trades.ticker AND d.decision='accepted')",
        )
    rows(
        "invalid_legacy_order_hash", "paper_pending_orders",
        ("order_hash", "status", "resolved_ts"),
        "order_hash IS NULL OR NOT ((length(order_hash)=64 "
        "AND lower(order_hash) NOT GLOB '*[^0-9a-f]*') "
        "OR (order_hash='legacy-unsealed' AND status='cancelled' "
        "AND resolved_ts IS NOT NULL))",
    )
    return list(found.values())


def _logical_orphans(
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
            raise PaperSchemaV5MigrationError([{
                "code": "ambiguous_logical_parent_key", "table": parent,
            }])
        parent_column = parent_key[0]
        if not _has_columns(conn, table, (column,)) or not _has_columns(
            conn, parent, (parent_column,),
        ):
            continue
        predicate = (
            f"{_quote(column)} IS NOT NULL AND NOT EXISTS ("
            f"SELECT 1 FROM {_quote(parent)} p "
            f"WHERE p.{_quote(parent_column)}={_quote(table)}.{_quote(column)})"
        )
        count, first = _count_and_first_id(conn, table, predicate)
        if count:
            diagnostics.append({
                "code": "orphan_logical_relation",
                "table": table,
                "column": column,
                "parent_table": parent,
                "row_count": count,
                "first_row_key": first,
            })
    return diagnostics


def _strict_v2_accounting_diagnostics(
    conn: sqlite3.Connection,
    starting_capital: Decimal,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT id,direction,status,entry_price,current_price,exit_price,
                  quantity,entry_notional,entry_commission,exit_commission,
                  borrow_cost,realized_pnl_usd,accounting_status
           FROM paper_trades WHERE campaign_id='paper-v2' ORDER BY id"""
    ).fetchall()
    unreconciled = [int(row[0]) for row in rows if str(row[12]) != "reconciled"]
    if unreconciled:
        return [{
            "code": "accounting",
            "reason": "unreconciled_v2_trade",
            "row_count": len(unreconciled),
            "first_trade_id": unreconciled[0],
        }]

    cash = starting_capital
    realized = Decimal(0)
    unrealized = Decimal(0)
    position_values: list[Decimal] = []
    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        (
            trade_id, direction, status, entry_price, current_price, exit_price,
            quantity, entry_notional, entry_commission, exit_commission,
            borrow_cost, realized_pnl, _accounting_status,
        ) = row
        if any(value is None for value in (
            entry_price, quantity, entry_notional, entry_commission,
        )):
            diagnostics.append({
                "code": "accounting", "reason": "missing_entry_accounting",
                "trade_id": int(trade_id),
            })
            continue
        entry = Decimal(str(entry_price))
        qty = Decimal(str(quantity))
        reserve = Decimal(str(entry_notional))
        entry_fee = Decimal(str(entry_commission))
        expected_notional = entry * qty
        if reserve != expected_notional:
            diagnostics.append({
                "code": "accounting",
                "reason": "entry_notional_mismatch",
                "trade_id": int(trade_id),
                "expected_entry_notional": str(expected_notional),
                "actual_entry_notional": str(reserve),
                "arithmetic": "exact_decimal_from_persisted_values",
            })
        cash -= reserve + entry_fee
        if str(status) != "open":
            if any(value is None for value in (
                exit_price, exit_commission, borrow_cost, realized_pnl,
            )):
                diagnostics.append({
                    "code": "accounting", "reason": "missing_close_accounting",
                    "trade_id": int(trade_id),
                })
                continue
            exit_value = Decimal(str(exit_price))
            exit_fee = Decimal(str(exit_commission))
            borrow = Decimal(str(borrow_cost))
            gross = (
                (exit_value - entry) * qty
                if str(direction) == "long"
                else (entry - exit_value) * qty
            )
            actual_realized = Decimal(str(realized_pnl))
            expected_realized = gross - entry_fee - exit_fee - borrow
            if actual_realized != expected_realized:
                diagnostics.append({
                    "code": "accounting",
                    "reason": "realized_pnl_mismatch",
                    "trade_id": int(trade_id),
                    "expected_realized_pnl_usd": str(expected_realized),
                    "actual_realized_pnl_usd": str(actual_realized),
                    "arithmetic": "exact_decimal_from_persisted_values",
                })
            cash += reserve + gross - exit_fee - borrow
            realized += actual_realized
            continue
        if current_price is None:
            diagnostics.append({
                "code": "accounting", "reason": "missing_open_mark",
                "trade_id": int(trade_id),
            })
            continue
        mark = Decimal(str(current_price))
        gross = (
            (mark - entry) * qty
            if str(direction) == "long"
            else (entry - mark) * qty
        )
        unrealized += gross - entry_fee
        position_values.append(
            mark * qty if str(direction) == "long" else reserve + gross
        )
    if diagnostics:
        return diagnostics
    equity = cash + sum(position_values, Decimal(0))
    delta = equity - (starting_capital + realized + unrealized)
    if delta != 0:
        diagnostics.append({
            "code": "accounting",
            "reason": "equity_pnl_invariant",
            "delta_usd": str(delta),
            "arithmetic": "exact_decimal_from_persisted_values",
        })
    return diagnostics


def _load_contract(contract_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("contract_id") != "paper-schema-contract-v5"
        or contract.get("source_schema_version") != SOURCE_SCHEMA_VERSION
        or contract.get("target_schema_version") != TARGET_SCHEMA_VERSION
    ):
        raise PaperSchemaV5MigrationError([{"code": "invalid_contract_identity"}])
    root = contract_path.parent
    fixtures: dict[str, Path] = {}
    for item in contract["fixtures"]["databases"]:
        path = root / str(item["path"])
        if _sha256(path) != item["sha256"]:
            raise PaperSchemaV5MigrationError([{
                "code": "fixture_hash_mismatch", "fixture": item["id"],
            }])
        fixtures[str(item["id"])] = path
    cases_path = root / str(contract["fixtures"]["path"])
    if _sha256(cases_path) != contract["fixtures"]["sha256"]:
        raise PaperSchemaV5MigrationError([{
            "code": "fixture_hash_mismatch", "fixture": "contract_cases",
        }])
    fixtures["contract_cases"] = cases_path
    return contract, fixtures


def _source_contract_diagnostics(
    conn: sqlite3.Connection,
    contract: dict[str, Any],
    fixtures: dict[str, Path],
) -> list[dict[str, Any]]:
    diagnostics = _collision_diagnostics(conn)
    diagnostics.extend(_logical_orphans(conn, contract))
    integrity = [str(row[0]) for row in conn.execute("PRAGMA integrity_check")]
    if integrity != ["ok"]:
        diagnostics.append({"code": "sqlite_integrity", "result": integrity})
    foreign_keys = [list(row) for row in conn.execute("PRAGMA foreign_key_check")]
    if foreign_keys:
        diagnostics.append({
            "code": "declared_foreign_keys", "row_count": len(foreign_keys),
            "first_break": foreign_keys[0],
        })
    source_signatures = []
    for fixture_id in ("fresh_v4", "production_like_legacy_v4"):
        fixture_conn = _fixture_connection(fixtures[fixture_id])
        try:
            source_signatures.append(_schema_signature(fixture_conn, governed_only=True))
        finally:
            fixture_conn.close()
    if _schema_signature(conn, governed_only=True) not in source_signatures:
        diagnostics.append({"code": "v4_contract_drift"})
    required_schema = set(contract["migration_ids"]["required_source"])
    actual_schema = {
        str(row[0]) for row in conn.execute("SELECT migration_id FROM schema_migrations")
    } if _has_columns(conn, "schema_migrations", ("migration_id",)) else set()
    if missing := sorted(required_schema - actual_schema):
        diagnostics.append({
            "code": "v4_contract_drift", "missing_schema_migrations": missing,
        })
    target_migration = str(contract["migration_ids"]["target"])
    if target_migration in actual_schema:
        diagnostics.append({
            "code": "v4_contract_drift",
            "reason": "target_migration_without_v5_identity",
            "migration_id": target_migration,
        })
    required_report = set(contract["migration_ids"]["required_report_dependency"])
    actual_report = {
        str(row[0])
        for row in conn.execute("SELECT migration FROM report_schema_migrations")
    } if _has_columns(conn, "report_schema_migrations", ("migration",)) else set()
    if missing := sorted(required_report - actual_report):
        diagnostics.append({
            "code": "v4_contract_drift", "missing_report_migrations": missing,
        })
    order = {
        str(item["code"]): position
        for position, item in enumerate(contract["collision_rules"])
    }
    return sorted(
        diagnostics,
        key=lambda item: (
            order.get(str(item["code"]), len(order)),
            str(item.get("table", "")), str(item.get("column", "")),
            json.dumps(item, sort_keys=True, separators=(",", ":")),
        ),
    )


def _copy_rebuilt_table(
    conn: sqlite3.Connection,
    target_conn: sqlite3.Connection,
    table: str,
) -> None:
    source = f"{table}__v4_source"
    target_sql = target_conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if not target_sql:
        raise PaperSchemaV5MigrationError([{
            "code": "target_fixture_missing_table", "table": table,
        }])
    conn.execute(str(target_sql[0]))
    source_columns = _table_columns(conn, source)
    target_columns = _table_columns(conn, table)
    copied = [column for column in target_columns if column in source_columns]
    target_projection = list(copied)
    source_projection = [_quote(column) for column in copied]
    # Accepted legacy rows predate these target timestamps. Letting SQLite
    # evaluate CURRENT_TIMESTAMP makes copied rehearsal and live migration
    # cohorts differ. Derive every introduced timestamp from retained row data.
    introduced_timestamps = {
        "paper_trades": {
            "created_ts": "report_date || 'T00:00:00Z'",
            "updated_ts": "report_date || 'T00:00:00Z'",
        },
        "paper_portfolio_snapshots": {
            "created_ts": "COALESCE(snapshot_ts,snapshot_date || 'T00:00:00Z')",
        },
    }
    for column, expression in introduced_timestamps.get(table, {}).items():
        if column not in source_columns:
            target_projection.append(column)
            source_projection.append(expression)
    targets = ",".join(_quote(column) for column in target_projection)
    sources = ",".join(source_projection)
    ordering = (
        "snapshot_date,campaign_id"
        if table == "paper_portfolio_snapshots" and "id" not in source_columns
        else "rowid"
    )
    conn.execute(
        f"INSERT INTO {_quote(table)} ({targets}) "
        f"SELECT {sources} FROM {_quote(source)} ORDER BY {ordering}"
    )


def _restore_changed_sequences(
    conn: sqlite3.Connection,
    before: dict[str, int],
    snapshot_had_id: bool,
) -> None:
    for table in ("paper_trades", "paper_portfolio_snapshots"):
        should_preserve = table == "paper_trades" or snapshot_had_id
        if not should_preserve:
            continue
        if table in before:
            updated = conn.execute(
                "UPDATE sqlite_sequence SET seq=? WHERE name=?",
                (before[table], table),
            ).rowcount
            if not updated:
                conn.execute(
                    "INSERT INTO sqlite_sequence(name,seq) VALUES (?,?)",
                    (table, before[table]),
                )
        else:
            conn.execute("DELETE FROM sqlite_sequence WHERE name=?", (table,))


def _validate_preservation(
    conn: sqlite3.Connection,
    contract: dict[str, Any],
    before: dict[str, dict[str, Any]],
    sequences_before: dict[str, int],
    snapshot_had_id: bool,
) -> None:
    diagnostics: list[dict[str, Any]] = []
    for table, state in before.items():
        count = int(conn.execute(
            f"SELECT COUNT(*) FROM {_quote(table)}"
        ).fetchone()[0])
        expected_count = state["count"]
        if table == "schema_migrations":
            expected_count += 1
        if count != expected_count:
            diagnostics.append({
                "code": "row_count_mismatch", "table": table,
                "before": state["count"], "after": count,
            })
            continue
        columns = list(state["columns"])
        order_by = list(state["source_primary_key"])
        where = (
            "WHERE migration_id!=" + _sql_text(str(contract["migration_ids"]["target"]))
            if table == "schema_migrations" else ""
        )
        if _rows_hash(
            conn, table, columns, order_by, where=where,
        ) != state["content_sha256"]:
            diagnostics.append({"code": "content_hash_mismatch", "table": table})
        if _rows_hash(
            conn, table, order_by, order_by, where=where,
        ) != state["primary_key_sha256"]:
            diagnostics.append({"code": "primary_key_mismatch", "table": table})
    sequences_after = {
        str(name): int(value)
        for name, value in conn.execute("SELECT name,seq FROM sqlite_sequence")
    }
    for table, value in sequences_before.items():
        if sequences_after.get(table, -1) < value:
            diagnostics.append({
                "code": "sequence_moved_backwards", "table": table,
                "before": value, "after": sequences_after.get(table),
            })
        if table not in {"paper_trades", "paper_portfolio_snapshots"} and (
            sequences_after.get(table) != value
        ):
            diagnostics.append({"code": "sequence_changed", "table": table})
    if snapshot_had_id and sequences_after.get("paper_portfolio_snapshots") != (
        sequences_before.get("paper_portfolio_snapshots")
    ):
        diagnostics.append({
            "code": "sequence_changed", "table": "paper_portfolio_snapshots",
        })
    if sequences_after.get("paper_trades") != sequences_before.get("paper_trades"):
        diagnostics.append({"code": "sequence_changed", "table": "paper_trades"})
    diagnostics.extend(_collision_diagnostics(conn))
    diagnostics.extend(_logical_orphans(conn, contract))
    if diagnostics:
        raise PaperSchemaV5MigrationError(diagnostics)


def migrate_paper_schema_v4_to_v5(
    conn: sqlite3.Connection,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    inject_failure_at: str | None = None,
    expected_source_logical_sha256: str | None = None,
    target_applied_ts: str | None = None,
) -> dict[str, Any]:
    """Run the explicit maintenance migration; never called by startup."""
    if conn.in_transaction:
        raise PaperSchemaV5MigrationError([{"code": "transaction_already_active"}])
    contract, fixtures = _load_contract(Path(contract_path))
    case_manifest = json.loads(fixtures["contract_cases"].read_text(encoding="utf-8"))
    interrupted = next(
        item for item in case_manifest["cases"] if item["id"] == "interrupted_rebuild"
    )
    fault_points = tuple(str(item) for item in interrupted["fault_points"])
    if inject_failure_at is not None and inject_failure_at not in fault_points:
        raise PaperSchemaV5MigrationError([{
            "code": "unknown_failure_point", "failure_point": inject_failure_at,
        }])

    meta_columns = set(_table_columns(conn, "paper_trade_schema_meta"))
    if {"schema_version", "contract_id", "schema_fingerprint"} <= meta_columns:
        row = conn.execute(
            "SELECT schema_version,contract_id,schema_fingerprint "
            "FROM paper_trade_schema_meta WHERE id=1"
        ).fetchone()
        expected = (
            TARGET_SCHEMA_VERSION,
            contract["contract_id"],
            contract["fingerprint"]["expected_sha256"],
        )
        if row is not None and tuple(row) == expected and conn.execute(
            "SELECT 1 FROM schema_migrations WHERE migration_id=?",
            (contract["migration_ids"]["target"],),
        ).fetchone():
            if expected_source_logical_sha256 is not None:
                raise PaperSchemaV5MigrationError([{
                    "code": "target_identity_requires_recovery_plan",
                }])
            return {
                "status": "already_v5_identity_only",
                "contract_id": contract["contract_id"],
                "schema_fingerprint": expected[2],
            }

    foreign_keys = int(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    legacy_alter = int(conn.execute("PRAGMA legacy_alter_table").fetchone()[0])
    logical_before: str | None = None
    target_conn: sqlite3.Connection | None = None

    def inject(point: str) -> None:
        if inject_failure_at == point:
            raise PaperSchemaV5MigrationError([{
                "code": "injected_failure", "failure_point": point,
            }])

    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("PRAGMA legacy_alter_table=ON")
        try:
            conn.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            raise PaperSchemaV5MigrationError([{
                "code": "exclusive_transaction_unavailable",
            }]) from exc
        diagnostics = _source_contract_diagnostics(conn, contract, fixtures)
        if diagnostics:
            raise PaperSchemaV5MigrationError(diagnostics)

        before_state = _table_state(conn, contract)
        logical_before = _logical_database_hash(conn)
        if (
            expected_source_logical_sha256 is not None
            and logical_before != expected_source_logical_sha256
        ):
            raise PaperSchemaV5MigrationError([{
                "code": "source_cohort_mismatch",
                "expected": expected_source_logical_sha256,
                "actual": logical_before,
            }])
        sequences_before = {
            str(name): int(value)
            for name, value in conn.execute("SELECT name,seq FROM sqlite_sequence")
        }
        snapshot_had_id = "id" in _table_columns(
            conn, "paper_portfolio_snapshots",
        )
        target_conn = _fixture_connection(fixtures["exact_v5_target"])
        for table in (
            "paper_trades", "paper_portfolio_snapshots", "paper_trade_schema_meta",
        ):
            conn.execute(
                f"ALTER TABLE {_quote(table)} RENAME TO {_quote(table + '__v4_source')}"
            )
        inject("after_rename")

        _copy_rebuilt_table(conn, target_conn, "paper_trades")
        _copy_rebuilt_table(conn, target_conn, "paper_portfolio_snapshots")
        meta_sql = target_conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='paper_trade_schema_meta'"
        ).fetchone()[0]
        conn.execute(str(meta_sql))
        conn.execute(
            "INSERT INTO paper_trade_schema_meta "
            "(id,schema_version,contract_id,schema_fingerprint) VALUES (?,?,?,?)",
            (
                SCHEMA_META_ROW_ID,
                TARGET_SCHEMA_VERSION,
                contract["contract_id"],
                contract["fingerprint"]["expected_sha256"],
            ),
        )
        inject("after_copy")

        for table in (
            "paper_trades", "paper_portfolio_snapshots", "paper_trade_schema_meta",
        ):
            conn.execute(f"DROP TABLE {_quote(table + '__v4_source')}")
        for table in ("paper_trades", "paper_portfolio_snapshots"):
            for sql, in target_conn.execute(
                "SELECT sql FROM sqlite_master WHERE type IN ('index','trigger') "
                "AND tbl_name=? AND sql IS NOT NULL ORDER BY type,name", (table,),
            ):
                conn.execute(str(sql))
        if target_applied_ts is None:
            conn.execute(
                "INSERT INTO schema_migrations(migration_id) VALUES (?)",
                (contract["migration_ids"]["target"],),
            )
        else:
            conn.execute(
                "INSERT INTO schema_migrations(migration_id,applied_ts) VALUES (?,?)",
                (contract["migration_ids"]["target"], target_applied_ts),
            )
        _restore_changed_sequences(
            conn, sequences_before, snapshot_had_id=snapshot_had_id,
        )
        _validate_preservation(
            conn, contract, before_state, sequences_before, snapshot_had_id,
        )

        integrity = [str(row[0]) for row in conn.execute("PRAGMA integrity_check")]
        foreign_key_breaks = [list(row) for row in conn.execute("PRAGMA foreign_key_check")]
        if integrity != ["ok"] or foreign_key_breaks:
            raise PaperSchemaV5MigrationError([{
                "code": "post_migration_integrity",
                "integrity": integrity,
                "foreign_key_break_count": len(foreign_key_breaks),
                "first_foreign_key_break": (
                    foreign_key_breaks[0] if foreign_key_breaks else None
                ),
            }])
        starting_capital = Decimal(str(conn.execute(
            "SELECT starting_capital FROM paper_campaigns WHERE campaign_id='paper-v2'"
        ).fetchone()[0]))
        if accounting_diagnostics := _strict_v2_accounting_diagnostics(
            conn, starting_capital,
        ):
            raise PaperSchemaV5MigrationError(accounting_diagnostics)
        for query in (
            "SELECT report_date,ticker,direction,status FROM paper_trades LIMIT 1",
            "SELECT snapshot_date,open_trades,equity_index "
            "FROM paper_portfolio_snapshots LIMIT 1",
            "SELECT snapshot_ts,total_unrealized_pnl_pct "
            "FROM paper_portfolio_snapshots LIMIT 1",
            "SELECT campaign_id,status,pnl_pct FROM paper_trades "
            "WHERE campaign_id='paper-v1' LIMIT 1",
        ):
            conn.execute(query).fetchall()
        changed_tables = {
            item["name"]: item["normalized_sql_sha256"]
            for item in contract["tables"] if item["disposition"] == "changed"
        }
        for table, expected_hash in changed_tables.items():
            sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()[0]
            actual_hash = hashlib.sha256(_normalize_sql(str(sql)).encode()).hexdigest()
            if actual_hash != expected_hash:
                raise PaperSchemaV5MigrationError([{
                    "code": "changed_table_contract_mismatch", "table": table,
                }])
        inject("after_validation")
        inject("before_commit")
        conn.commit()
    except Exception as exc:
        conn.rollback()
        if logical_before is not None and _logical_database_hash(conn) != logical_before:
            raise PaperSchemaV5MigrationError([{
                "code": "rollback_not_logically_identical",
            }]) from exc
        raise
    finally:
        if target_conn is not None:
            target_conn.close()
        conn.execute(f"PRAGMA legacy_alter_table={legacy_alter}")
        conn.execute(f"PRAGMA foreign_keys={foreign_keys}")

    return {
        "status": "migrated",
        "contract_id": contract["contract_id"],
        "schema_fingerprint": contract["fingerprint"]["expected_sha256"],
        "source_logical_sha256": logical_before,
        "retained_table_counts": {
            table: state["count"] for table, state in sorted(before_state.items())
        },
        "integrity_check": "ok",
        "foreign_key_break_count": 0,
        "accounting_break_count": 0,
        "accounting_arithmetic": "exact_decimal_from_persisted_values",
    }
