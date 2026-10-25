"""Release gate on a copied database that already completed the v5 contraction."""
from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from trader_koo.paper_trade.schema_v5_migration import migrate_paper_schema_v4_to_v5
from trader_koo.paper_trade.schema_v5_verifier import FROZEN_SEMANTIC_FINGERPRINT
from trader_koo.scripts.release_evidence import _schema_contract, migrate_copy


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"
LEGACY_V4 = "paper_schema_v4_legacy_production_like.sql"


def _validator() -> Draft202012Validator:
    schema = json.loads((ROOT / "release-database-copy-v5.schema.json").read_text())
    return Draft202012Validator(schema)


def _v5_source(path: Path) -> Path:
    with sqlite3.connect(path) as conn:
        conn.executescript((FIXTURES / LEGACY_V4).read_text(encoding="utf-8"))
        conn.commit()
        assert migrate_paper_schema_v4_to_v5(conn)["status"] == "migrated"
    return path


def _evidence_dir(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir()
    return path


def _manifest(output_dir: Path) -> dict:
    return json.loads((output_dir / "database-migration-manifest.json").read_text())


def test_v4_expand_contract_reports_the_v5_removals_seen_in_production(
    tmp_path: Path,
) -> None:
    source = _v5_source(tmp_path / "v5.db")

    with sqlite3.connect(source) as conn:
        contract = _schema_contract(conn)

    # Dark deploy 6fae457c: these are the v5 contract's intended removals.
    assert contract["malformed_indexes"] == [
        "idx_paper_portfolio_legacy_compat", "idx_paper_trades_legacy_compat",
    ]
    assert contract["malformed_triggers"] == ["paper_v1_trades_no_update"]
    assert contract["campaign_defaults"] == {
        "paper_trades.campaign_id": "",
        "paper_portfolio_snapshots.campaign_id": "",
    }
    assert contract["passed"] is False


def test_release_copy_of_exact_v5_database_passes_phase_aware_gate(
    tmp_path: Path,
) -> None:
    source = _v5_source(tmp_path / "v5.db")
    output_dir = _evidence_dir(tmp_path, "evidence")

    manifest = migrate_copy(source, output_dir)

    _validator().validate(manifest)
    assert manifest["schema"] == "release-database-copy-v5"
    assert manifest["passed"] is True
    assert manifest["paper_trade_schema_version"] == 5
    assert manifest["expected_paper_trade_schema_version"] == 5
    assert manifest["schema_contract"] == {
        "contract": "paper-schema-contract-v5",
        "contract_id": "paper-schema-contract-v5",
        "schema_fingerprint": FROZEN_SEMANTIC_FINGERPRINT,
        "diagnostics": [],
        "passed": True,
    }
    assert manifest["accounting_invariants"]["breaks"] == []
    assert "paper_schema_initialization_diagnostics" not in manifest


def test_release_copy_rejects_v5_with_reintroduced_legacy_index(
    tmp_path: Path,
) -> None:
    source = _v5_source(tmp_path / "v5.db")
    with sqlite3.connect(source) as conn:
        conn.execute(
            "CREATE UNIQUE INDEX idx_paper_trades_legacy_compat "
            "ON paper_trades(report_date,ticker,direction)"
        )
        conn.commit()
    output_dir = _evidence_dir(tmp_path, "evidence")

    with pytest.raises(RuntimeError, match="migration or accounting invariant"):
        migrate_copy(source, output_dir)

    manifest = _manifest(output_dir)
    _validator().validate(manifest)
    assert manifest["passed"] is False
    assert manifest["expected_paper_trade_schema_version"] == 5
    assert manifest["schema_contract"]["passed"] is False
    assert manifest["schema_contract"]["diagnostics"]
    assert manifest["accounting_invariants"] is None


def test_release_copy_rejects_v4_database_claiming_v5(tmp_path: Path) -> None:
    source = tmp_path / "fake-v5.db"
    with sqlite3.connect(source) as conn:
        conn.executescript((FIXTURES / LEGACY_V4).read_text(encoding="utf-8"))
        conn.execute("UPDATE paper_trade_schema_meta SET schema_version=5 WHERE id=1")
        conn.commit()
    output_dir = _evidence_dir(tmp_path, "evidence")

    with pytest.raises(RuntimeError, match="migration or accounting invariant"):
        migrate_copy(source, output_dir)

    manifest = _manifest(output_dir)
    _validator().validate(manifest)
    assert manifest["passed"] is False
    assert manifest["paper_schema_initialization_diagnostics"]
    assert manifest["schema_contract"]["contract"] == "paper-schema-contract-v5"
    assert manifest["schema_contract"]["passed"] is False


def test_v5_manifest_schema_rejects_contradictory_success(tmp_path: Path) -> None:
    manifest = migrate_copy(
        _v5_source(tmp_path / "v5.db"), _evidence_dir(tmp_path, "evidence"),
    )
    validator = _validator()

    wrong_fingerprint = deepcopy(manifest)
    wrong_fingerprint["schema_contract"]["schema_fingerprint"] = "0" * 64
    expand_contract_on_v5 = deepcopy(manifest)
    with sqlite3.connect(tmp_path / "evidence/database-copy.db") as conn:
        expand_contract_on_v5["schema_contract"] = _schema_contract(conn)
    version_mismatch = deepcopy(manifest)
    version_mismatch["paper_trade_schema_version"] = 4
    hidden_diagnostics = deepcopy(manifest)
    hidden_diagnostics["schema_contract"]["diagnostics"] = [{"code": "missing_index"}]

    for contradictory in (
        wrong_fingerprint, expand_contract_on_v5, version_mismatch, hidden_diagnostics,
    ):
        with pytest.raises(ValidationError):
            validator.validate(contradictory)
