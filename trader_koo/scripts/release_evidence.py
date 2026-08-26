"""Build fail-closed migration and deterministic replay evidence for a release."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sqlite3
import subprocess
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from trader_koo.db.price_contract import ensure_price_series_revision_schema
from trader_koo.db.price_repairs import ensure_price_repair_schema
from trader_koo.paper_trade.portfolio_accounting import reconcile_portfolio
from trader_koo.paper_trade.errors import (
    ADMISSION_LEDGER_MIGRATION,
    AdmissionLedgerContractError,
)
from trader_koo.paper_trade.replay import replay_campaign
from trader_koo.paper_trade.schema import (
    PAPER_TRADE_SCHEMA_VERSION,
    PaperSchemaInitializationError,
    ensure_paper_trade_schema,
)
from trader_koo.paper_trades import _build_config
from trader_koo.report.runs import ensure_report_run_schema
from trader_koo.research.next_open_baseline import (
    BaselineConfig,
    ExecutionDecision,
    SessionPrice,
    canonical_json_bytes,
    simulate_portfolio,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _validate_database_manifest(payload: dict[str, Any]) -> None:
    """Enforce relationships JSON Schema cannot express."""
    contract = payload.get("report_admission_contract")
    if not isinstance(contract, dict) or not isinstance(contract.get("passed"), bool):
        raise RuntimeError("release database manifest lacks admission contract status")
    if contract["passed"] is False:
        sample = contract.get("affected_attempt_sample")
        if not isinstance(sample, list):
            raise RuntimeError("release database failure sample is malformed")
        invalid_count = int(contract.get("invalid_row_count", -1))
        reported_count = int(contract.get("reported_attempt_count", -1))
        limit = int(contract.get("diagnostic_limit", -1))
        if (
            reported_count != len(sample)
            or not 0 < reported_count <= limit
            or reported_count > invalid_count
            or bool(contract.get("truncated")) != (invalid_count > reported_count)
        ):
            raise RuntimeError("release database failure counts are inconsistent")


def _validate_database_manifest_schema(payload: dict[str, Any]) -> None:
    schema_path = Path(__file__).resolve().parents[2] / "release-database-copy-v4.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(payload)


def _publish_database_manifest(output_dir: Path, payload: dict[str, Any]) -> None:
    try:
        _validate_database_manifest(payload)
        _validate_database_manifest_schema(payload)
    except Exception as exc:
        _write_json(output_dir / "database-migration-generation-error.json", {
            "schema": "release-database-generation-error-v1",
            "error_code": "release_manifest_validation_failed",
            "target_schema": "release-database-copy-v4",
        })
        raise RuntimeError(
            "release database manifest validation failed; primary manifest not published"
        ) from exc
    _write_json(output_dir / "database-migration-manifest.json", payload)


def _code_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _snapshot(source: Path, target: Path) -> None:
    if source.suffix == ".gz":
        with gzip.open(source, "rb") as compressed, target.open("wb") as output:
            shutil.copyfileobj(compressed, output)
        return
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as source_conn:
        with sqlite3.connect(target) as target_conn:
            source_conn.backup(target_conn)


def _normalize_sql(value: str | None) -> str:
    return " ".join(str(value or "").lower().split())


def _schema_contract(conn: sqlite3.Connection) -> dict[str, Any]:
    required_indexes = {
        "idx_paper_trades_campaign_unique": (
            "paper_trades", ("campaign_id", "report_date", "ticker", "direction")
        ),
        "idx_paper_trades_legacy_compat": (
            "paper_trades", ("report_date", "ticker", "direction")
        ),
        "idx_paper_portfolio_campaign_date": (
            "paper_portfolio_snapshots", ("campaign_id", "snapshot_date")
        ),
        "idx_paper_portfolio_legacy_compat": (
            "paper_portfolio_snapshots", ("snapshot_date",)
        ),
    }
    malformed_indexes: list[str] = []
    for name, (table, columns) in required_indexes.items():
        index_rows = {
            str(row[1]): row
            for row in conn.execute(f"PRAGMA index_list({table})")
        }
        row = index_rows.get(name)
        actual_columns = tuple(
            str(item[2]) for item in conn.execute(f'PRAGMA index_info("{name}")')
        )
        if row is None or int(row[2]) != 1 or int(row[4]) != 0 or actual_columns != columns:
            malformed_indexes.append(name)
    required_triggers = {
        "paper_trades_require_canonical_run": """
            CREATE TRIGGER paper_trades_require_canonical_run
            BEFORE INSERT ON paper_trades
            WHEN NOT EXISTS (
                SELECT 1 FROM report_runs r
                JOIN report_run_decisions d ON d.run_id=r.run_id
                WHERE r.run_id=NEW.report_run_id
                  AND r.status='published' AND r.publication_verified=1
                  AND r.is_generation_canonical=1
                  AND d.ticker=NEW.ticker AND d.decision='accepted'
            )
            BEGIN
                SELECT RAISE(ABORT, 'paper trades require a canonical published report run with an accepted decision');
            END
        """,
        "paper_trades_immutable_lineage": """
            CREATE TRIGGER paper_trades_immutable_lineage
            BEFORE UPDATE OF report_run_id ON paper_trades
            WHEN NEW.report_run_id IS NOT OLD.report_run_id
            BEGIN
                SELECT RAISE(ABORT, 'paper trade report lineage is immutable');
            END
        """,
        "paper_v1_trades_no_insert": """
            CREATE TRIGGER paper_v1_trades_no_insert
            BEFORE INSERT ON paper_trades WHEN NEW.campaign_id = 'paper-v1'
            BEGIN SELECT RAISE(ABORT, 'paper campaign v1 is immutable'); END
        """,
        "paper_v1_trades_no_update": """
            CREATE TRIGGER paper_v1_trades_no_update
            BEFORE UPDATE ON paper_trades
            WHEN OLD.campaign_id = 'paper-v1'
              AND NEW.report_run_id IS OLD.report_run_id
            BEGIN SELECT RAISE(ABORT, 'paper campaign v1 is immutable'); END
        """,
        "paper_v1_trades_no_delete": """
            CREATE TRIGGER paper_v1_trades_no_delete
            BEFORE DELETE ON paper_trades WHEN OLD.campaign_id = 'paper-v1'
            BEGIN SELECT RAISE(ABORT, 'paper campaign v1 is immutable'); END
        """,
    }
    actual_triggers = {
        str(row[0]): _normalize_sql(str(row[1] or ""))
        for row in conn.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='trigger'"
        )
    }
    malformed_triggers = sorted(
        name for name, sql in required_triggers.items()
        if actual_triggers.get(name) != _normalize_sql(sql)
    )
    required_foreign_keys = {
        ("paper_trade_events", "trade_id", "paper_trades", "id"),
        ("paper_trade_annotations", "trade_id", "paper_trades", "id"),
        ("paper_pending_orders", "campaign_id", "paper_campaigns", "campaign_id"),
        ("paper_order_events", "order_id", "paper_pending_orders", "order_id"),
        ("paper_campaign_audit", "campaign_id", "paper_campaigns", "campaign_id"),
        ("paper_campaign_preregistrations", "campaign_id", "paper_campaigns", "campaign_id"),
        ("paper_campaign_experiments", "campaign_id", "paper_campaigns", "campaign_id"),
        ("paper_campaign_experiments", "preregistration_id", "paper_campaign_preregistrations", "preregistration_id"),
        ("paper_campaign_approvals", "campaign_id", "paper_campaigns", "campaign_id"),
        ("paper_campaign_approvals", "experiment_id", "paper_campaign_experiments", "experiment_id"),
    }
    actual_foreign_keys: set[tuple[str, str, str, str]] = set()
    for table in {item[0] for item in required_foreign_keys}:
        actual_foreign_keys.update(
            (table, str(row[3]), str(row[2]), str(row[4]))
            for row in conn.execute(f"PRAGMA foreign_key_list({table})")
        )
    missing_foreign_keys = sorted(required_foreign_keys - actual_foreign_keys)
    paper_trade_foreign_keys: dict[int, list[tuple[str, str, str, str, str, str]]] = {}
    for row in conn.execute("PRAGMA foreign_key_list(paper_trades)"):
        paper_trade_foreign_keys.setdefault(int(row[0]), []).append(
            (
                str(row[3]), str(row[2]), str(row[4]),
                str(row[5]), str(row[6]), str(row[7]),
            )
        )
    lineage_constraints = [
        (constraint_id, mappings)
        for constraint_id, mappings in sorted(paper_trade_foreign_keys.items())
        if any(item[0] == "report_run_id" for item in mappings)
    ]
    expected_lineage = [(
        "report_run_id", "report_runs", "run_id", "NO ACTION", "NO ACTION", "NONE"
    )]
    malformed_lineage_constraints = [
        {"id": constraint_id, "mappings": [list(item) for item in mappings]}
        for constraint_id, mappings in lineage_constraints
        if len(lineage_constraints) != 1 or mappings != expected_lineage
    ]
    malformed_optional_foreign_keys = (
        [{
            "table": "paper_trades",
            "column": "report_run_id",
            "constraints": malformed_lineage_constraints,
        }]
        if malformed_lineage_constraints else []
    )
    trade_columns = {
        str(row[1]): str(row[4] or "")
        for row in conn.execute("PRAGMA table_info(paper_trades)")
    }
    portfolio_columns = {
        str(row[1]): str(row[4] or "")
        for row in conn.execute("PRAGMA table_info(paper_portfolio_snapshots)")
    }
    foreign_key_breaks = [list(row) for row in conn.execute("PRAGMA foreign_key_check")]
    legacy_read_probe = True
    try:
        conn.execute(
            "SELECT report_date,ticker,direction,status FROM paper_trades LIMIT 1"
        ).fetchall()
        conn.execute(
            "SELECT snapshot_date,open_trades,equity_index "
            "FROM paper_portfolio_snapshots LIMIT 1"
        ).fetchall()
    except sqlite3.DatabaseError:
        legacy_read_probe = False
    defaults = {
        "paper_trades.campaign_id": trade_columns.get("campaign_id"),
        "paper_portfolio_snapshots.campaign_id": portfolio_columns.get("campaign_id"),
    }
    allowed_defaults = {"'paper-v1'", '\"paper-v1\"', "'paper-v2'", '\"paper-v2\"'}
    defaults_compatible = all(value in allowed_defaults for value in defaults.values())
    return {
        "contract": "paper-schema-expand-v4",
        "malformed_indexes": sorted(malformed_indexes),
        "malformed_triggers": malformed_triggers,
        "missing_foreign_keys": [list(item) for item in missing_foreign_keys],
        "malformed_optional_foreign_keys": malformed_optional_foreign_keys,
        "campaign_defaults": defaults,
        "campaign_defaults_compatible": defaults_compatible,
        "foreign_key_breaks": foreign_key_breaks,
        "legacy_read_probe": legacy_read_probe,
        "previous_image_paper_write_mode": "disabled_during_rollback_window",
        "passed": (
            not malformed_indexes
            and not malformed_triggers
            and not missing_foreign_keys
            and not malformed_optional_foreign_keys
            and defaults_compatible
            and not foreign_key_breaks
            and legacy_read_probe
        ),
    }


def migrate_copy(source: Path, output_dir: Path) -> dict[str, Any]:
    copy_path = output_dir / "database-copy.db"
    source_artifact_hash = _sha256(source)
    _snapshot(source, copy_path)
    source_snapshot_hash = _sha256(copy_path)
    contract_failure: AdmissionLedgerContractError | None = None
    paper_schema_diagnostics: list[dict[str, Any]] = []
    with closing(sqlite3.connect(copy_path)) as conn:
        before_counts = dict(conn.execute(
            "SELECT type,COUNT(*) FROM sqlite_master GROUP BY type"
        ).fetchall())
        # Paper schema v4 may already be current and short-circuit. Report
        # admission lineage has its own versioned migration and must always be
        # verified explicitly on the copied database.
        try:
            ensure_report_run_schema(conn, verify_admission_contract=True)
        except AdmissionLedgerContractError as exc:
            contract_failure = exc
        if contract_failure is None:
            try:
                ensure_paper_trade_schema(conn)
            except PaperSchemaInitializationError as exc:
                # The copied-database contract below records the structural
                # failure, while this field preserves the initializer's exact
                # phase diagnostic instead of silently falling through.
                paper_schema_diagnostics = [dict(item) for item in exc.diagnostics]
            ensure_price_series_revision_schema(conn)
            ensure_price_repair_schema(conn)
            integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
            schema_version = int(conn.execute(
                "SELECT schema_version FROM paper_trade_schema_meta WHERE id=1"
            ).fetchone()[0])
            schema_contract = _schema_contract(conn)
            account = (
                reconcile_portfolio(
                    conn, campaign_id="paper-v2",
                    starting_capital=_build_config().starting_capital,
                )
                if schema_contract["passed"] else None
            )
            after_counts = dict(conn.execute(
                "SELECT type,COUNT(*) FROM sqlite_master GROUP BY type"
            ).fetchall())
            conn.commit()
    if contract_failure is not None:
        failure_manifest = {
            "schema": "release-database-copy-v4",
            "source_artifact": source.name,
            "source_artifact_sha256": source_artifact_hash,
            "source_snapshot_sha256": source_snapshot_hash,
            "migrated_copy_sha256": _sha256(copy_path),
            "report_admission_contract": {
                "passed": False,
                "violation": "legacy_rows_invalid",
                "target_migration": ADMISSION_LEDGER_MIGRATION,
                "invalid_row_count": contract_failure.invalid_count,
                "affected_attempt_sample": contract_failure.attempts,
                "reported_attempt_count": len(contract_failure.attempts),
                "diagnostic_limit": 20,
                "truncated": contract_failure.invalid_count > len(contract_failure.attempts),
                "ordering": "attempt_id_ascending",
            },
            "passed": False,
        }
        _publish_database_manifest(output_dir, failure_manifest)
        raise RuntimeError("database copy report-admission contract failed") from contract_failure
    manifest = {
        "schema": "release-database-copy-v4",
        "source_artifact": source.name,
        "source_artifact_sha256": source_artifact_hash,
        "source_snapshot_sha256": source_snapshot_hash,
        "migrated_copy_sha256": _sha256(copy_path),
        "integrity_check": integrity,
        "paper_trade_schema_version": schema_version,
        "expected_paper_trade_schema_version": PAPER_TRADE_SCHEMA_VERSION,
        "report_admission_contract": {
            "passed": True,
            "migration": ADMISSION_LEDGER_MIGRATION,
        },
        "accounting_invariants": (
            {
                "breaks": account["accounting_breaks"],
                "legacy_unreconciled_count": account["legacy_unreconciled_count"],
                "equity": account["equity"],
                "cash": account["cash"],
            }
            if account is not None else None
        ),
        "schema_contract": schema_contract,
        "sqlite_objects_before": before_counts,
        "sqlite_objects_after": after_counts,
    }
    if paper_schema_diagnostics:
        manifest["paper_schema_initialization_diagnostics"] = paper_schema_diagnostics
    manifest["passed"] = (
        integrity == "ok"
        and schema_version == PAPER_TRADE_SCHEMA_VERSION
        and account is not None
        and not account["accounting_breaks"]
        and schema_contract["passed"]
        and not paper_schema_diagnostics
    )
    _publish_database_manifest(output_dir, manifest)
    if not manifest["passed"]:
        raise RuntimeError("database copy migration or accounting invariant failed")
    return manifest


def replay_smoke(output_dir: Path, code_sha: str) -> dict[str, Any]:
    config = replace(_build_config(), max_open=2, expiry_days=2)
    candidate = {
        "ticker": "FIXTURE", "setup_tier": "A", "score": 85.0,
        "actionability": "higher-probability", "signal_bias": "bullish",
        "close": 100.0, "setup_family": "release-smoke", "atr_pct_14": 2.0,
        "support_level": 95.0, "resistance_level": 105.0,
        "debate_agreement_score": 90.0,
        "critic_outcome": {"approved": True},
    }
    runs = [{
        "report_run_id": "release-smoke", "report_date": "2026-08-20",
        "published_ts": "2026-08-20T12:00:00Z",
        "candidates": [candidate],
    }]
    prices = [
        {"ticker": "SPY", "date": "2026-08-20", "open": 100, "high": 101,
         "low": 99, "close": 100, "volume": 1_000_000},
        {"ticker": "SPY", "date": "2026-08-21", "open": 100, "high": 101,
         "low": 99, "close": 100, "volume": 1_000_000},
        {"ticker": "SPY", "date": "2026-08-24", "open": 100, "high": 101,
         "low": 99, "close": 100, "volume": 1_000_000},
        {"ticker": "FIXTURE", "date": "2026-08-20", "open": 99, "high": 101,
         "low": 98, "close": 100, "volume": 1_000_000},
        {"ticker": "FIXTURE", "date": "2026-08-21", "open": 100, "high": 101,
         "low": 99, "close": 100, "volume": 1_000_000},
        {"ticker": "FIXTURE", "date": "2026-08-24", "open": 100, "high": 108,
         "low": 99, "close": 107, "volume": 1_000_000},
    ]
    first = replay_campaign(
        candidate_runs=runs, price_rows=prices, spy_rows=[], config=config,
        _include_splits=False,
    )
    expected = {
        item["execution_key"]: {
            "disposition": item["disposition"], "inputs_hash": item["inputs_hash"],
        }
        for item in first["decisions"]
    }
    replay = replay_campaign(
        candidate_runs=runs, price_rows=prices, spy_rows=[], config=config,
        expected_execution=expected, _include_splits=False,
    )
    decisions = []
    for raw in replay["execution_ledger"]["decisions"]:
        payload = dict(raw)
        payload["metadata"] = tuple(tuple(item) for item in payload["metadata"])
        decisions.append(ExecutionDecision(**payload))
    baseline = simulate_portfolio(
        decisions,
        [SessionPrice(**row) for row in replay["execution_ledger"]["market_data"]["prices"]],
        replay["execution_ledger"]["market_data"]["sessions"],
        BaselineConfig(**replay["execution_ledger"]["config"]),
    )
    ledger_match = canonical_json_bytes(baseline.ledger) == canonical_json_bytes(
        replay["execution_ledger"]
    )
    manifest = {
        "schema": "release-replay-smoke-v1",
        "code_sha": code_sha,
        "evidence_state": "synthetic_contract_fixture",
        "return_basis": "next_open_whole_share_net_of_configured_costs",
        "benchmark_basis": "none_synthetic_parity_fixture",
        "costs": replay["execution_ledger"]["config"],
        "data_sha256": replay["dataset_hash"],
        "config_sha256": replay["policy_hash"],
        "execution_ledger_sha256": replay["execution_ledger_hash"],
        "decisions": replay["decisions"],
        "fills": replay["trades"],
        "equity": replay["equity_curve"],
        "replay_live_parity": replay["replay_live_parity"],
        "canonical_ledger_parity": "matched" if ledger_match else "diverged",
    }
    manifest["passed"] = (
        replay["replay_live_parity"] == "matched" and ledger_match
    )
    _write_json(output_dir / "replay-smoke-manifest.json", manifest)
    _write_json(output_dir / "backtest-execution-ledger.json", replay["execution_ledger"])
    if not manifest["passed"]:
        raise RuntimeError("live/replay canonical ledger parity failed")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.source_db.is_file():
        raise FileNotFoundError(args.source_db)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    database = migrate_copy(args.source_db.resolve(), args.output_dir.resolve())
    replay = replay_smoke(args.output_dir.resolve(), _code_sha())
    print(json.dumps({"database": database["passed"], "replay": replay["passed"]}))


if __name__ == "__main__":
    main()
