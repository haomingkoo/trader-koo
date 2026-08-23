"""Build fail-closed migration and deterministic replay evidence for a release."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sqlite3
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

from trader_koo.db.price_contract import ensure_price_series_revision_schema
from trader_koo.db.price_repairs import ensure_price_repair_schema
from trader_koo.paper_trade.portfolio_accounting import reconcile_portfolio
from trader_koo.paper_trade.replay import replay_campaign
from trader_koo.paper_trade.schema import PAPER_TRADE_SCHEMA_VERSION, ensure_paper_trade_schema
from trader_koo.paper_trades import _build_config
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


def migrate_copy(source: Path, output_dir: Path) -> dict[str, Any]:
    copy_path = output_dir / "database-copy.db"
    source_artifact_hash = _sha256(source)
    _snapshot(source, copy_path)
    source_snapshot_hash = _sha256(copy_path)
    with sqlite3.connect(copy_path) as conn:
        before_counts = dict(conn.execute(
            "SELECT type,COUNT(*) FROM sqlite_master GROUP BY type"
        ).fetchall())
        ensure_paper_trade_schema(conn)
        ensure_price_series_revision_schema(conn)
        ensure_price_repair_schema(conn)
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        schema_version = int(conn.execute(
            "SELECT schema_version FROM paper_trade_schema_meta WHERE id=1"
        ).fetchone()[0])
        account = reconcile_portfolio(
            conn, campaign_id="paper-v2", starting_capital=_build_config().starting_capital,
        )
        after_counts = dict(conn.execute(
            "SELECT type,COUNT(*) FROM sqlite_master GROUP BY type"
        ).fetchall())
        conn.commit()
    manifest = {
        "schema": "release-database-copy-v1",
        "source_artifact": source.name,
        "source_artifact_sha256": source_artifact_hash,
        "source_snapshot_sha256": source_snapshot_hash,
        "migrated_copy_sha256": _sha256(copy_path),
        "integrity_check": integrity,
        "paper_trade_schema_version": schema_version,
        "expected_paper_trade_schema_version": PAPER_TRADE_SCHEMA_VERSION,
        "accounting_invariants": {
            "breaks": account["accounting_breaks"],
            "legacy_unreconciled_count": account["legacy_unreconciled_count"],
            "equity": account["equity"],
            "cash": account["cash"],
        },
        "sqlite_objects_before": before_counts,
        "sqlite_objects_after": after_counts,
    }
    manifest["passed"] = (
        integrity == "ok"
        and schema_version == PAPER_TRADE_SCHEMA_VERSION
        and not account["accounting_breaks"]
    )
    _write_json(output_dir / "database-migration-manifest.json", manifest)
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
