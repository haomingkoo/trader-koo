"""Schema helpers for paper trades."""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

# Track which on-disk database files have already had their full schema
# ensured, so the ~45 PRAGMA table_info probes run at most once per file.
# In-memory databases are never cached (each connection is a distinct DB).
_ensured_db_paths: set[str] = set()
_ensured_db_paths_lock = threading.Lock()


def _resolve_main_db_path(conn: sqlite3.Connection) -> str:
    """Return the file path of the connection's 'main' database ('' if in-memory)."""
    for row in conn.execute("PRAGMA database_list").fetchall():
        if str(row[1]) == "main":
            return str(row[2] or "")
    return ""


def _ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    ddl: str,
) -> None:
    columns = {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name in columns:
        return
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {ddl}")


def _rebuild_unique_key(conn: sqlite3.Connection, table: str, old: str, new: str) -> None:
    """Replace one legacy table-level UNIQUE clause without rewriting its columns."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    sql = str(row[0] or "") if row else ""
    if old not in sql:
        return
    legacy = f"{table}__campaign_migration"
    columns = [str(item[1]) for item in conn.execute(f"PRAGMA table_info({table})")]
    conn.execute(f"ALTER TABLE {table} RENAME TO {legacy}")
    create_sql = sql.replace(f"CREATE TABLE {table}", f"CREATE TABLE {table}", 1).replace(old, new, 1)
    if table == "paper_trades":
        create_sql = create_sql.replace(
            "campaign_id TEXT", "campaign_id TEXT NOT NULL DEFAULT 'paper-v2'", 1
        )
    conn.execute(create_sql)
    joined = ",".join(f'"{column}"' for column in columns)
    conn.execute(f"INSERT INTO {table} ({joined}) SELECT {joined} FROM {legacy}")
    conn.execute(f"DROP TABLE {legacy}")


def _widen_candidate_dispositions(conn: sqlite3.Connection) -> None:
    """Migrate the legacy CHECK so a sealed signal may await its next open."""
    table = "paper_candidate_decisions"
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    sql = str(row[0] or "") if row else ""
    old = "disposition IN ('rejected', 'admitted', 'duplicate')"
    if old not in sql:
        return
    legacy = f"{table}__pending_migration"
    columns = [str(item[1]) for item in conn.execute(f"PRAGMA table_info({table})")]
    conn.execute(f"ALTER TABLE {table} RENAME TO {legacy}")
    conn.execute(sql.replace(old, "disposition IN ('rejected', 'pending', 'admitted', 'duplicate')"))
    joined = ",".join(f'"{column}"' for column in columns)
    conn.execute(f"INSERT INTO {table} ({joined}) SELECT {joined} FROM {legacy}")
    conn.execute(f"DROP TABLE {legacy}")


def decode_json_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    try:
        payload = json.loads(str(raw))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def ensure_paper_trade_schema(conn: sqlite3.Connection) -> None:
    """Create paper_trades and paper_portfolio_snapshots tables."""
    from trader_koo.report.runs import ensure_report_run_schema

    ensure_report_run_schema(conn)
    db_path = _resolve_main_db_path(conn)
    # In-memory DBs (path '' or ':memory:') are never cached: each connection
    # is a distinct database, so the full ensure must always run.
    is_memory = db_path in ("", ":memory:")
    if not is_memory:
        with _ensured_db_paths_lock:
            if db_path in _ensured_db_paths:
                return

    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date TEXT NOT NULL,
            generated_ts TEXT,
            report_run_id TEXT REFERENCES report_runs(run_id),
            ticker TEXT NOT NULL,
            direction TEXT NOT NULL CHECK (direction IN ('long', 'short')),
            entry_price REAL NOT NULL,
            entry_date TEXT NOT NULL,
            target_price REAL,
            stop_loss REAL,
            atr_at_entry REAL,
            exit_price REAL,
            exit_date TEXT,
            exit_reason TEXT,
            status TEXT NOT NULL DEFAULT 'open'
                CHECK (status IN ('open', 'closed', 'stopped_out', 'target_hit', 'expired')),
            current_price REAL,
            unrealized_pnl_pct REAL,
            last_mtm_date TEXT,
            high_water_mark REAL,
            low_water_mark REAL,
            pnl_pct REAL,
            r_multiple REAL,
            setup_family TEXT,
            setup_tier TEXT,
            score REAL,
            signal_bias TEXT,
            actionability TEXT,
            observation TEXT,
            action_text TEXT,
            risk_note TEXT,
            yolo_pattern TEXT,
            yolo_recency TEXT,
            debate_agreement_score REAL,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(report_date, ticker, direction)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_trades_status "
        "ON paper_trades(status, entry_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_trades_ticker "
        "ON paper_trades(ticker, status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_trades_family "
        "ON paper_trades(setup_family, direction, status)"
    )
    _ensure_column(conn, "paper_trades", "decision_version", "decision_version TEXT")
    _ensure_column(conn, "paper_trades", "decision_state", "decision_state TEXT")
    _ensure_column(conn, "paper_trades", "analyst_stage", "analyst_stage TEXT")
    _ensure_column(conn, "paper_trades", "debate_stage", "debate_stage TEXT")
    _ensure_column(conn, "paper_trades", "risk_stage", "risk_stage TEXT")
    _ensure_column(conn, "paper_trades", "portfolio_decision", "portfolio_decision TEXT")
    _ensure_column(conn, "paper_trades", "decision_summary", "decision_summary TEXT")
    _ensure_column(conn, "paper_trades", "decision_reasons", "decision_reasons TEXT")
    _ensure_column(conn, "paper_trades", "risk_flags", "risk_flags TEXT")
    _ensure_column(conn, "paper_trades", "position_size_pct", "position_size_pct REAL")
    _ensure_column(conn, "paper_trades", "risk_budget_pct", "risk_budget_pct REAL")
    _ensure_column(conn, "paper_trades", "stop_distance_pct", "stop_distance_pct REAL")
    _ensure_column(conn, "paper_trades", "expected_reward_pct", "expected_reward_pct REAL")
    _ensure_column(conn, "paper_trades", "expected_r_multiple", "expected_r_multiple REAL")
    _ensure_column(conn, "paper_trades", "entry_plan", "entry_plan TEXT")
    _ensure_column(conn, "paper_trades", "exit_plan", "exit_plan TEXT")
    _ensure_column(conn, "paper_trades", "sizing_summary", "sizing_summary TEXT")
    _ensure_column(conn, "paper_trades", "review_status", "review_status TEXT")
    _ensure_column(conn, "paper_trades", "review_summary", "review_summary TEXT")
    _ensure_column(conn, "paper_trades", "entry_reason", "entry_reason TEXT")
    _ensure_column(conn, "paper_trades", "entry_evidence", "entry_evidence TEXT")
    _ensure_column(conn, "paper_trades", "entry_risks", "entry_risks TEXT")
    _ensure_column(conn, "paper_trades", "bot_version", "bot_version TEXT")
    _ensure_column(conn, "paper_trades", "vix_at_entry", "vix_at_entry REAL")
    _ensure_column(conn, "paper_trades", "vix_percentile_at_entry", "vix_percentile_at_entry REAL")
    _ensure_column(conn, "paper_trades", "regime_state_at_entry", "regime_state_at_entry TEXT")
    _ensure_column(conn, "paper_trades", "hmm_regime_at_entry", "hmm_regime_at_entry TEXT")
    _ensure_column(conn, "paper_trades", "hmm_confidence_at_entry", "hmm_confidence_at_entry REAL")
    _ensure_column(conn, "paper_trades", "ml_predicted_win_prob", "ml_predicted_win_prob REAL")
    _ensure_column(conn, "paper_trades", "ml_confidence", "ml_confidence REAL")
    _ensure_column(conn, "paper_trades", "ml_signal", "ml_signal TEXT")
    _ensure_column(conn, "paper_trades", "notes", "notes TEXT DEFAULT ''")
    _ensure_column(conn, "paper_trades", "report_run_id", "report_run_id TEXT")
    _ensure_column(conn, "paper_trades", "directional_regime_at_entry", "directional_regime_at_entry TEXT")
    _ensure_column(conn, "paper_trades", "directional_regime_confidence", "directional_regime_confidence REAL")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_trades_report_run ON paper_trades(report_run_id)"
    )
    _ensure_column(conn, "paper_trades", "campaign_id", "campaign_id TEXT")
    _ensure_column(conn, "paper_trades", "policy_version", "policy_version TEXT")

    conn.execute(
        "UPDATE paper_trades SET campaign_id='paper-v1' WHERE campaign_id IS NULL"
    )
    _rebuild_unique_key(
        conn,
        "paper_trades",
        "UNIQUE(report_date, ticker, direction)",
        "UNIQUE(campaign_id, report_date, ticker, direction)",
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(campaign_id, status, entry_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_ticker ON paper_trades(campaign_id, ticker, status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_trades_family ON paper_trades(campaign_id, setup_family, direction, status)")
    # Rebuilding the table removes its triggers. Install the authoritative
    # lineage guards only after the campaign-aware unique-key migration.
    conn.execute("DROP TRIGGER IF EXISTS paper_trades_require_canonical_run")
    conn.execute("DROP TRIGGER IF EXISTS paper_trades_immutable_lineage")
    conn.execute("""
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
    """)
    conn.execute("""
        CREATE TRIGGER paper_trades_immutable_lineage
        BEFORE UPDATE OF report_run_id ON paper_trades
        WHEN NEW.report_run_id IS NOT OLD.report_run_id
        BEGIN
            SELECT RAISE(ABORT, 'paper trade report lineage is immutable');
        END
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_campaigns (
            campaign_id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            policy_version TEXT NOT NULL,
            policy_hash TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL CHECK (status IN ('frozen', 'active', 'draft')),
            starting_capital REAL NOT NULL,
            zero_admission_streak_limit INTEGER NOT NULL DEFAULT 3,
            replay_live_parity TEXT NOT NULL DEFAULT 'not_measured'
                CHECK (replay_live_parity IN ('not_measured', 'matched', 'diverged')),
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    _ensure_column(
        conn,
        "paper_campaigns",
        "replay_live_parity",
        "replay_live_parity TEXT NOT NULL DEFAULT 'not_measured'",
    )
    _ensure_column(conn, "paper_campaigns", "policy_hash", "policy_hash TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "paper_campaigns", "updated_ts", "updated_ts TEXT")
    conn.execute("""
        INSERT OR IGNORE INTO paper_campaigns (
            campaign_id, label, policy_version, status, starting_capital,
            zero_admission_streak_limit
        ) VALUES ('paper-v1', 'Paper Campaign v1', 'paper-trade-eval-v1', 'frozen', 1000000.0, 3)
    """)
    conn.execute("""
        INSERT OR IGNORE INTO paper_campaigns (
            campaign_id, label, policy_version, status, starting_capital,
            zero_admission_streak_limit
        ) VALUES ('paper-v2', 'Paper Campaign v2', 'paper-campaign-v2.0', 'draft', 1000000.0, 3)
    """)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_paper_campaign "
        "ON paper_campaigns(status) WHERE status='active'"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_campaign_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id TEXT NOT NULL REFERENCES paper_campaigns(campaign_id),
            action TEXT NOT NULL CHECK (action IN ('activate', 'rollback')),
            actor TEXT NOT NULL,
            reason TEXT NOT NULL,
            idempotency_key TEXT NOT NULL UNIQUE,
            request_hash TEXT NOT NULL DEFAULT '',
            from_status TEXT NOT NULL,
            to_status TEXT NOT NULL,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    _ensure_column(conn, "paper_campaign_audit", "request_hash", "request_hash TEXT NOT NULL DEFAULT ''")
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_campaign_audit_no_update
        BEFORE UPDATE ON paper_campaign_audit
        BEGIN SELECT RAISE(ABORT, 'paper campaign audit is immutable'); END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_campaign_audit_no_delete
        BEFORE DELETE ON paper_campaign_audit
        BEGIN SELECT RAISE(ABORT, 'paper campaign audit is immutable'); END
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            migration_id TEXT PRIMARY KEY,
            applied_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    lifecycle_migration = "paper_campaign_v2_inactive_governed_20260823"
    if not conn.execute(
        "SELECT 1 FROM schema_migrations WHERE migration_id=?", (lifecycle_migration,)
    ).fetchone():
        conn.execute("UPDATE paper_campaigns SET status='draft' WHERE campaign_id='paper-v2'")
        conn.execute("INSERT INTO schema_migrations (migration_id) VALUES (?)", (lifecycle_migration,))
    migration_id = "paper_campaign_v1_backfill_20260822"
    migrated = conn.execute(
        "SELECT 1 FROM schema_migrations WHERE migration_id = ?", (migration_id,)
    ).fetchone()
    if not migrated:
        conn.execute(
            """UPDATE paper_trades
               SET campaign_id = 'paper-v1',
                   policy_version = COALESCE(policy_version, decision_version, 'paper-trade-eval-v1')
               WHERE campaign_id IS NULL"""
        )
        conn.execute(
            "INSERT INTO schema_migrations (migration_id) VALUES (?)", (migration_id,)
        )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_candidate_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_run_id TEXT NOT NULL,
            report_date TEXT NOT NULL,
            generated_ts TEXT NOT NULL,
            campaign_id TEXT NOT NULL,
            policy_version TEXT NOT NULL,
            ticker TEXT NOT NULL,
            candidate_rank INTEGER NOT NULL,
            rank_inputs_json TEXT NOT NULL,
            eligibility_passed INTEGER NOT NULL CHECK (eligibility_passed IN (0, 1)),
            final_gate TEXT NOT NULL,
            reason_code TEXT NOT NULL,
            reasons_json TEXT NOT NULL,
            inputs_hash TEXT NOT NULL,
            policy_hash TEXT NOT NULL,
            context_hash TEXT NOT NULL,
            disposition TEXT NOT NULL CHECK (
                disposition IN ('rejected', 'pending', 'admitted', 'duplicate')
            ),
            tradeability TEXT NOT NULL DEFAULT 'not_actionable',
            inputs_json TEXT NOT NULL DEFAULT '{}',
            stop_loss REAL,
            target_price REAL,
            expected_r_multiple REAL,
            critic_outcome_json TEXT,
            sizing_json TEXT,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(report_run_id, campaign_id, candidate_rank)
        )
    """)
    _ensure_column(conn, "paper_candidate_decisions", "policy_hash", "policy_hash TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "paper_candidate_decisions", "context_hash", "context_hash TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "paper_candidate_decisions", "tradeability", "tradeability TEXT NOT NULL DEFAULT 'not_actionable'")
    _ensure_column(conn, "paper_candidate_decisions", "inputs_json", "inputs_json TEXT NOT NULL DEFAULT '{}'")
    _widen_candidate_dispositions(conn)
    _rebuild_unique_key(
        conn,
        "paper_candidate_decisions",
        "UNIQUE(report_run_id, campaign_id, ticker)",
        "UNIQUE(report_run_id, campaign_id, candidate_rank)",
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_candidate_decisions_campaign_report "
        "ON paper_candidate_decisions(campaign_id, report_date, report_run_id, candidate_rank)"
    )
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_candidate_decisions_no_update
        BEFORE UPDATE ON paper_candidate_decisions
        BEGIN SELECT RAISE(ABORT, 'paper candidate decisions are immutable'); END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_candidate_decisions_no_delete
        BEFORE DELETE ON paper_candidate_decisions
        BEGIN SELECT RAISE(ABORT, 'paper candidate decisions are immutable'); END
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_decision_sets (
            report_run_id TEXT NOT NULL,
            campaign_id TEXT NOT NULL,
            report_date TEXT NOT NULL,
            generated_ts TEXT NOT NULL,
            policy_version TEXT NOT NULL,
            candidate_count INTEGER NOT NULL,
            request_hash TEXT NOT NULL,
            candidates_hash TEXT NOT NULL,
            policy_hash TEXT NOT NULL,
            context_hash TEXT NOT NULL,
            report_complete INTEGER NOT NULL CHECK (report_complete IN (0,1)),
            is_canonical INTEGER NOT NULL CHECK (is_canonical IN (0,1)),
            status TEXT NOT NULL CHECK (status='sealed'),
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (report_run_id, campaign_id)
        )
    """)
    _ensure_column(conn, "paper_decision_sets", "request_hash", "request_hash TEXT NOT NULL DEFAULT ''")
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_decision_sets_no_update
        BEFORE UPDATE ON paper_decision_sets
        BEGIN SELECT RAISE(ABORT, 'paper decision sets are immutable'); END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_decision_sets_no_delete
        BEFORE DELETE ON paper_decision_sets
        BEGIN SELECT RAISE(ABORT, 'paper decision sets are immutable'); END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_candidate_decisions_no_insert_after_seal
        BEFORE INSERT ON paper_candidate_decisions
        WHEN EXISTS (
            SELECT 1 FROM paper_decision_sets
            WHERE report_run_id=NEW.report_run_id
              AND campaign_id=NEW.campaign_id
              AND status='sealed'
        )
        BEGIN SELECT RAISE(ABORT, 'sealed paper decision set is not appendable'); END
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_pending_orders (
            order_id TEXT PRIMARY KEY,
            report_run_id TEXT NOT NULL,
            report_date TEXT NOT NULL,
            generated_ts TEXT NOT NULL,
            campaign_id TEXT NOT NULL REFERENCES paper_campaigns(campaign_id),
            policy_version TEXT NOT NULL,
            candidate_rank INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            direction TEXT NOT NULL CHECK (direction IN ('long','short')),
            candidate_json TEXT NOT NULL,
            critic_json TEXT NOT NULL,
            market_context_json TEXT NOT NULL,
            avg_daily_volume REAL,
            order_hash TEXT NOT NULL CHECK (
                length(order_hash)=64 AND lower(order_hash) NOT GLOB '*[^0-9a-f]*'
            ),
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','filled','rejected','cancelled')),
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            resolved_ts TEXT,
            UNIQUE(report_run_id,campaign_id,candidate_rank)
        )
    """)
    _ensure_column(conn, "paper_pending_orders", "order_hash", "order_hash TEXT")
    # Pre-seal pending rows cannot be trusted after this migration. Preserve
    # them as cancelled audit history; only newly hashed rows may execute.
    conn.execute(
        """UPDATE paper_pending_orders
           SET status='cancelled',resolved_ts=COALESCE(resolved_ts,CURRENT_TIMESTAMP),
               order_hash=COALESCE(order_hash,'legacy-unsealed')
           WHERE order_hash IS NULL"""
    )
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_pending_orders_valid_insert
        BEFORE INSERT ON paper_pending_orders
        WHEN NEW.status!='pending' OR NEW.resolved_ts IS NOT NULL
          OR NEW.order_hash IS NULL
          OR length(NEW.order_hash)!=64
          OR lower(NEW.order_hash) GLOB '*[^0-9a-f]*'
        BEGIN SELECT RAISE(ABORT, 'pending order requires a sealed immutable payload'); END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_pending_orders_immutable_payload
        BEFORE UPDATE ON paper_pending_orders
        WHEN NEW.order_id IS NOT OLD.order_id
          OR NEW.report_run_id IS NOT OLD.report_run_id
          OR NEW.report_date IS NOT OLD.report_date
          OR NEW.generated_ts IS NOT OLD.generated_ts
          OR NEW.campaign_id IS NOT OLD.campaign_id
          OR NEW.policy_version IS NOT OLD.policy_version
          OR NEW.candidate_rank IS NOT OLD.candidate_rank
          OR NEW.ticker IS NOT OLD.ticker
          OR NEW.direction IS NOT OLD.direction
          OR NEW.candidate_json IS NOT OLD.candidate_json
          OR NEW.critic_json IS NOT OLD.critic_json
          OR NEW.market_context_json IS NOT OLD.market_context_json
          OR NEW.avg_daily_volume IS NOT OLD.avg_daily_volume
          OR NEW.order_hash IS NOT OLD.order_hash
          OR NEW.created_ts IS NOT OLD.created_ts
        BEGIN SELECT RAISE(ABORT, 'pending order payload is immutable'); END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_pending_orders_terminal_transition
        BEFORE UPDATE OF status,resolved_ts ON paper_pending_orders
        WHEN OLD.status!='pending' OR NEW.status NOT IN ('filled','rejected','cancelled')
          OR NEW.resolved_ts IS NULL
        BEGIN SELECT RAISE(ABORT, 'pending order has an invalid terminal transition'); END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_pending_orders_no_delete
        BEFORE DELETE ON paper_pending_orders
        BEGIN SELECT RAISE(ABORT, 'pending orders are immutable audit facts'); END
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_order_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT NOT NULL REFERENCES paper_pending_orders(order_id),
            event_type TEXT NOT NULL CHECK (event_type IN ('created','filled','rejected','cancelled')),
            event_date TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(order_id,event_type)
        )
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_order_events_no_update
        BEFORE UPDATE ON paper_order_events
        BEGIN SELECT RAISE(ABORT, 'paper order events are immutable'); END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_order_events_no_delete
        BEFORE DELETE ON paper_order_events
        BEGIN SELECT RAISE(ABORT, 'paper order events are immutable'); END
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_campaign_preregistrations (
            preregistration_id TEXT PRIMARY KEY,
            campaign_id TEXT NOT NULL REFERENCES paper_campaigns(campaign_id),
            policy_version TEXT NOT NULL,
            policy_hash TEXT NOT NULL,
            dataset_hash TEXT NOT NULL,
            gates_json TEXT NOT NULL,
            artifact_hash TEXT NOT NULL UNIQUE,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_campaign_experiments (
            experiment_id TEXT PRIMARY KEY,
            preregistration_id TEXT NOT NULL REFERENCES paper_campaign_preregistrations(preregistration_id),
            campaign_id TEXT NOT NULL REFERENCES paper_campaigns(campaign_id),
            policy_version TEXT NOT NULL,
            policy_hash TEXT NOT NULL,
            dataset_hash TEXT NOT NULL,
            preregistration_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            parity_status TEXT NOT NULL CHECK (parity_status IN ('matched','diverged')),
            risk_gate_passed INTEGER NOT NULL CHECK (risk_gate_passed IN (0,1)),
            active_return_gate_passed INTEGER NOT NULL CHECK (active_return_gate_passed IN (0,1)),
            eligible INTEGER NOT NULL CHECK (eligible IN (0,1)),
            evidence_hash TEXT NOT NULL UNIQUE,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    _ensure_column(
        conn, "paper_campaign_experiments", "preregistration_id",
        "preregistration_id TEXT REFERENCES paper_campaign_preregistrations(preregistration_id)",
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_campaign_approvals (
            approval_id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL REFERENCES paper_campaign_experiments(experiment_id),
            campaign_id TEXT NOT NULL REFERENCES paper_campaigns(campaign_id),
            actor TEXT NOT NULL,
            reason TEXT NOT NULL,
            experiment_evidence_hash TEXT NOT NULL,
            artifact_json TEXT NOT NULL,
            artifact_hash TEXT NOT NULL UNIQUE,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    _ensure_column(
        conn, "paper_campaign_approvals", "experiment_evidence_hash",
        "experiment_evidence_hash TEXT",
    )
    for table, label in (
        ("paper_campaign_preregistrations", "paper campaign preregistrations"),
        ("paper_campaign_experiments", "paper campaign experiments"),
        ("paper_campaign_approvals", "paper campaign approvals"),
    ):
        conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {table}_no_update
            BEFORE UPDATE ON {table}
            BEGIN SELECT RAISE(ABORT, '{label} are immutable'); END
        """)
        conn.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {table}_no_delete
            BEFORE DELETE ON {table}
            BEGIN SELECT RAISE(ABORT, '{label} are immutable'); END
        """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_v1_trades_no_insert
        BEFORE INSERT ON paper_trades WHEN NEW.campaign_id = 'paper-v1'
        BEGIN SELECT RAISE(ABORT, 'paper campaign v1 is immutable'); END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_v1_trades_no_update
        BEFORE UPDATE ON paper_trades
        WHEN OLD.campaign_id = 'paper-v1'
          AND NEW.report_run_id IS OLD.report_run_id
        BEGIN SELECT RAISE(ABORT, 'paper campaign v1 is immutable'); END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_v1_trades_no_delete
        BEFORE DELETE ON paper_trades WHEN OLD.campaign_id = 'paper-v1'
        BEGIN SELECT RAISE(ABORT, 'paper campaign v1 is immutable'); END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_v1_campaign_no_update
        BEFORE UPDATE ON paper_campaigns WHEN OLD.campaign_id='paper-v1'
        BEGIN SELECT RAISE(ABORT, 'paper campaign v1 metadata is immutable'); END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paper_v1_campaign_no_delete
        BEFORE DELETE ON paper_campaigns WHEN OLD.campaign_id='paper-v1'
        BEGIN SELECT RAISE(ABORT, 'paper campaign v1 metadata is immutable'); END
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_trade_annotations (
            trade_id INTEGER PRIMARY KEY REFERENCES paper_trades(id),
            notes TEXT NOT NULL DEFAULT '',
            actor TEXT NOT NULL DEFAULT 'user',
            updated_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS bot_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_version TEXT NOT NULL UNIQUE,
            decision_version TEXT,
            strategy_kind TEXT NOT NULL DEFAULT 'paper_rules',
            status TEXT NOT NULL DEFAULT 'active',
            config_json TEXT,
            notes TEXT,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_bot_versions_status "
        "ON bot_versions(status, created_ts)"
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL,
            campaign_id TEXT NOT NULL DEFAULT 'paper-v1',
            open_trades INTEGER NOT NULL DEFAULT 0,
            closed_trades_total INTEGER NOT NULL DEFAULT 0,
            wins INTEGER NOT NULL DEFAULT 0,
            losses INTEGER NOT NULL DEFAULT 0,
            win_rate_pct REAL,
            avg_pnl_pct REAL,
            avg_r_multiple REAL,
            total_pnl_pct REAL,
            max_drawdown_pct REAL,
            sharpe_ratio REAL,
            profit_factor REAL,
            equity_index REAL NOT NULL DEFAULT 100.0,
            best_trade_pct REAL,
            worst_trade_pct REAL,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(campaign_id, snapshot_date)
        )
    """)
    _ensure_column(conn, "paper_portfolio_snapshots", "campaign_id", "campaign_id TEXT NOT NULL DEFAULT 'paper-v1'")
    _rebuild_unique_key(
        conn,
        "paper_portfolio_snapshots",
        "snapshot_date TEXT NOT NULL UNIQUE",
        "snapshot_date TEXT NOT NULL",
    )
    _rebuild_unique_key(
        conn,
        "paper_portfolio_snapshots",
        "snapshot_date TEXT PRIMARY KEY",
        "snapshot_date TEXT NOT NULL",
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_portfolio_campaign_date "
        "ON paper_portfolio_snapshots(campaign_id, snapshot_date)"
    )
    _ensure_column(conn, "paper_portfolio_snapshots", "sortino_ratio", "sortino_ratio REAL")
    _ensure_column(conn, "paper_portfolio_snapshots", "calmar_ratio", "calmar_ratio REAL")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_portfolio_date "
        "ON paper_portfolio_snapshots(snapshot_date)"
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_trade_reflections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER NOT NULL UNIQUE,
            ticker TEXT NOT NULL,
            direction TEXT NOT NULL,
            setup_family TEXT,
            entry_date TEXT,
            exit_date TEXT,
            exit_reason TEXT,
            pnl_pct REAL,
            r_multiple REAL,
            spy_return_pct REAL,
            alpha_vs_spy_pct REAL,
            lesson_summary TEXT,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_reflections_trade "
        "ON paper_trade_reflections(trade_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_paper_reflections_ticker "
        "ON paper_trade_reflections(ticker, exit_date)"
    )
    conn.commit()

    if not is_memory:
        with _ensured_db_paths_lock:
            _ensured_db_paths.add(db_path)


def register_bot_version(
    conn: sqlite3.Connection,
    *,
    bot_version: str,
    decision_version: str | None,
    config_json: str | None = None,
    notes: str | None = None,
    schema_ready: bool = False,
) -> None:
    if not schema_ready:
        ensure_paper_trade_schema(conn)
    if not bot_version:
        return
    conn.execute(
        """
        INSERT INTO bot_versions (
            bot_version, decision_version, strategy_kind, status, config_json, notes
        ) VALUES (?, ?, 'paper_rules', 'active', ?, ?)
        ON CONFLICT(bot_version) DO UPDATE SET
            decision_version = excluded.decision_version,
            config_json = COALESCE(excluded.config_json, bot_versions.config_json),
            notes = COALESCE(excluded.notes, bot_versions.notes)
        """,
        (bot_version, decision_version, config_json, notes),
    )
