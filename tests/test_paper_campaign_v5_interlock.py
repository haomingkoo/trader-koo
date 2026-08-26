"""Adversarial activation checks for the exact-v5 transaction boundary."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

import pytest

from trader_koo.paper_trade.campaign import transition_campaign
from trader_koo.paper_trade.trading import fill_pending_paper_orders, mark_to_market
from trader_koo.paper_trades import _build_config
from trader_koo.paper_trade.schema_v5_verifier import (
    PaperSchemaV5VerificationError,
    verify_paper_schema_v5,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _load(path: Path, fixture: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript((FIXTURES / fixture).read_text())
    return conn


def _eligible_v5(path: Path) -> sqlite3.Connection:
    conn = _load(path, "paper_schema_v5_target.sql")
    policy_version = "paper-campaign-v2.0"
    policy_hash = "a" * 64
    conn.execute(
        "UPDATE paper_campaigns SET policy_version=?,policy_hash=?,"
        "replay_live_parity='matched' WHERE campaign_id='paper-v2'",
        (policy_version, policy_hash),
    )
    config_json = "{}"
    conn.execute(
        """INSERT INTO report_runs
           (run_id,report_kind,status,started_ts,config_json,config_hash,code_version)
           VALUES ('activation-evidence','daily','started',?,?,?,?)""",
        (
            "2026-08-25T21:59:00Z", config_json,
            hashlib.sha256(config_json.encode()).hexdigest(), "a" * 40,
        ),
    )
    conn.execute(
        """UPDATE report_runs SET status='completed',completed_ts=?,generated_ts=?,
           scanned_universe_json='[]',ranked_candidates_json='[]',decisions_json='[]',
           inputs_json='{}',source_timestamps_json='{}',content_hash=?,markdown_hash=?,
           artifact_path='/copy/report.json',markdown_path='/copy/report.md',generation_key=?
           WHERE run_id='activation-evidence'""",
        (
            "2026-08-25T22:00:00Z", "2026-08-25T22:00:00Z",
            "e" * 64, "f" * 64, "daily:2026-08-25T22:00:00Z",
        ),
    )
    conn.execute(
        """UPDATE report_runs SET status='published',published_ts=?,
           publication_verified=1 WHERE run_id='activation-evidence'""",
        ("2026-08-25T22:00:01Z",),
    )
    conn.execute(
        """INSERT INTO paper_decision_sets
           (report_run_id,campaign_id,report_date,generated_ts,policy_version,
            candidate_count,request_hash,candidates_hash,policy_hash,context_hash,
            report_complete,is_canonical,status)
           VALUES ('activation-evidence','paper-v2','2026-08-25',?, ?,0,?,?,?,?,1,1,'sealed')""",
        (
            "2026-08-25T22:00:00Z", policy_version,
            "b" * 64, "c" * 64, policy_hash, "d" * 64,
        ),
    )
    conn.commit()
    verify_paper_schema_v5(conn)
    return conn


def _activate(conn: sqlite3.Connection) -> None:
    transition_campaign(
        conn, campaign_id="paper-v2", action="activate", actor="operator",
        reason="approved copied-database rehearsal", idempotency_key="activate-v5",
    )


def _assert_unmodified(conn: sqlite3.Connection) -> None:
    assert conn.execute(
        "SELECT status FROM paper_campaigns WHERE campaign_id='paper-v2'"
    ).fetchone() == ("draft",)
    assert conn.execute("SELECT COUNT(*) FROM paper_campaign_audit").fetchone() == (0,)


@pytest.mark.parametrize(
    "fixture,mutate",
    [
        ("paper_schema_v4_fresh.sql", None),
        (
            "paper_schema_v5_target.sql",
            lambda conn: conn.execute("DROP INDEX idx_paper_trades_report_run"),
        ),
    ],
)
def test_activation_rejects_v4_and_malformed_v5_without_mutation(
    tmp_path: Path, fixture: str, mutate,
) -> None:
    conn = _load(tmp_path / "paper.db", fixture)
    if mutate is not None:
        mutate(conn)
        conn.commit()

    with pytest.raises(ValueError, match="activation interlock"):
        _activate(conn)

    _assert_unmodified(conn)


def test_activation_verifies_data_on_locked_transaction_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "paper.db"
    conn = _eligible_v5(path)
    from trader_koo.paper_trade import schema

    original = schema._paper_schema_path_pin

    def inject_orphan(active: sqlite3.Connection):
        pin = original(active)
        attacker = sqlite3.connect(path)
        attacker.execute(
            """INSERT INTO paper_trade_reflections
               (trade_id,ticker,direction) VALUES (999999,'ORPHAN','long')"""
        )
        attacker.commit()
        attacker.close()
        return pin

    monkeypatch.setattr(schema, "_paper_schema_path_pin", inject_orphan)
    with pytest.raises(ValueError, match="activation interlock"):
        _activate(conn)

    _assert_unmodified(conn)


def test_activation_rejects_path_replacement_without_writing_either_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "paper.db"
    moved = tmp_path / "paper-original.db"
    replacement = tmp_path / "paper-replacement.db"
    conn = _eligible_v5(path)
    replacement_conn = _eligible_v5(replacement)
    replacement_conn.close()
    from trader_koo.paper_trade import schema

    original = schema._paper_schema_path_pin

    def replace_after_pin(active: sqlite3.Connection):
        pin = original(active)
        os.replace(path, moved)
        os.replace(replacement, path)
        return pin

    monkeypatch.setattr(schema, "_paper_schema_path_pin", replace_after_pin)
    with pytest.raises(ValueError, match="path changed"):
        _activate(conn)

    _assert_unmodified(conn)
    old = sqlite3.connect(moved)
    new = sqlite3.connect(path)
    try:
        _assert_unmodified(old)
        _assert_unmodified(new)
    finally:
        old.close()
        new.close()


def test_activation_rejects_unlinked_path_with_stable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "paper.db"
    conn = _eligible_v5(path)
    from trader_koo.paper_trade import schema

    original = schema._paper_schema_path_pin

    def unlink_after_pin(active: sqlite3.Connection):
        pin = original(active)
        path.unlink()
        return pin

    monkeypatch.setattr(schema, "_paper_schema_path_pin", unlink_after_pin)
    with pytest.raises(ValueError) as raised:
        _activate(conn)

    assert str(raised.value) == "activation interlock: paper-schema path is unavailable"
    _assert_unmodified(conn)


def test_v5_trading_facades_require_explicit_schema_ready_boundary(tmp_path: Path) -> None:
    conn = _eligible_v5(tmp_path / "paper.db")
    assert mark_to_market(conn, config=_build_config())["open_trades"] == 0

    conn.execute("BEGIN")
    with pytest.raises(RuntimeError, match="clean transaction boundary"):
        mark_to_market(conn, config=_build_config())
    with pytest.raises(RuntimeError, match="schema verification before transaction"):
        fill_pending_paper_orders(conn, config=_build_config())
    conn.rollback()


def test_v5_trading_facade_rejects_unverified_schema(tmp_path: Path) -> None:
    conn = _eligible_v5(tmp_path / "paper.db")
    conn.execute("DROP INDEX idx_paper_trades_report_run")
    conn.commit()

    with pytest.raises(PaperSchemaV5VerificationError, match="missing_index"):
        mark_to_market(conn, config=_build_config())
