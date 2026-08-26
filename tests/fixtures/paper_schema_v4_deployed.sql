BEGIN TRANSACTION;
CREATE TABLE bot_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bot_version TEXT NOT NULL UNIQUE,
            decision_version TEXT,
            strategy_kind TEXT NOT NULL DEFAULT 'paper_rules',
            status TEXT NOT NULL DEFAULT 'active',
            config_json TEXT,
            notes TEXT,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
CREATE TABLE paper_campaign_approvals (
            approval_id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL REFERENCES paper_campaign_experiments(experiment_id),
            campaign_id TEXT NOT NULL REFERENCES paper_campaigns(campaign_id),
            actor TEXT NOT NULL,
            reason TEXT NOT NULL,
            experiment_evidence_hash TEXT NOT NULL,
            artifact_json TEXT NOT NULL,
            artifact_hash TEXT NOT NULL UNIQUE,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
CREATE TABLE paper_campaign_audit (
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
        );
CREATE TABLE paper_campaign_experiments (
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
        );
CREATE TABLE paper_campaign_preregistrations (
            preregistration_id TEXT PRIMARY KEY,
            campaign_id TEXT NOT NULL REFERENCES paper_campaigns(campaign_id),
            policy_version TEXT NOT NULL,
            policy_hash TEXT NOT NULL,
            dataset_hash TEXT NOT NULL,
            gates_json TEXT NOT NULL,
            artifact_hash TEXT NOT NULL UNIQUE,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
CREATE TABLE paper_campaigns (
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
        );
INSERT INTO "paper_campaigns" VALUES('paper-v1','Paper Campaign v1','paper-trade-eval-v1','','frozen',1000000.0,3,'not_measured','2026-08-26 00:29:18','2026-08-26 00:29:18');
INSERT INTO "paper_campaigns" VALUES('paper-v2','Paper Campaign v2','paper-campaign-v2.0','','draft',1000000.0,3,'not_measured','2026-08-26 00:29:18','2026-08-26 00:29:18');
CREATE TABLE paper_candidate_decisions (
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
        );
CREATE TABLE paper_decision_sets (
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
        );
CREATE TABLE paper_order_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT NOT NULL REFERENCES paper_pending_orders(order_id),
            event_type TEXT NOT NULL CHECK (event_type IN ('created','filled','rejected','cancelled')),
            event_date TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(order_id,event_type)
        );
CREATE TABLE paper_pending_orders (
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
        );
CREATE TABLE paper_portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL UNIQUE,
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
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, sortino_ratio REAL, calmar_ratio REAL, campaign_id TEXT NOT NULL DEFAULT 'paper-v1', starting_capital REAL, cash REAL, equity REAL, realized_pnl_usd REAL, unrealized_pnl_usd REAL, gross_exposure_usd REAL, gross_exposure_pct REAL, high_water_equity REAL, drawdown_pct REAL, session_pnl_usd REAL, legacy_unreconciled_count INTEGER NOT NULL DEFAULT 0, accounting_breaks_json TEXT NOT NULL DEFAULT '[]'
        );
INSERT INTO paper_portfolio_snapshots (
    id,snapshot_date,open_trades,closed_trades_total,wins,losses,equity_index,
    campaign_id,starting_capital,cash,equity,legacy_unreconciled_count
) VALUES (
    201,'2026-08-25',1,0,0,0,100.0,
    'paper-v1',1000000.0,990000.0,1000000.0,1
);
CREATE TABLE paper_shadow_decision_sets (
            report_run_id TEXT NOT NULL,
            policy_id TEXT NOT NULL,
            report_date TEXT NOT NULL,
            generated_ts TEXT NOT NULL,
            candidate_count INTEGER NOT NULL,
            accepted_count INTEGER NOT NULL,
            decisions_hash TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status='sealed'),
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (report_run_id, policy_id)
        );
CREATE TABLE paper_shadow_decisions (
            decision_id TEXT PRIMARY KEY,
            report_run_id TEXT NOT NULL,
            report_date TEXT NOT NULL,
            generated_ts TEXT NOT NULL,
            policy_id TEXT NOT NULL REFERENCES paper_shadow_policies(policy_id),
            policy_version TEXT NOT NULL,
            candidate_rank INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            disposition TEXT NOT NULL CHECK (disposition IN ('accepted','rejected')),
            gate TEXT NOT NULL,
            reason_code TEXT NOT NULL,
            reasons_json TEXT NOT NULL,
            feature_snapshot_json TEXT NOT NULL,
            feature_snapshot_hash TEXT NOT NULL,
            source_timestamps_json TEXT NOT NULL,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(report_run_id, policy_id, candidate_rank)
        );
CREATE TABLE paper_shadow_outcomes (
            outcome_id TEXT PRIMARY KEY,
            decision_id TEXT NOT NULL UNIQUE REFERENCES paper_shadow_decisions(decision_id),
            intended_entry_date TEXT NOT NULL,
            entry_date TEXT,
            exit_date TEXT,
            status TEXT NOT NULL CHECK (status IN ('pending','resolved','invalid')),
            result_json TEXT NOT NULL,
            result_hash TEXT NOT NULL,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
CREATE TABLE paper_shadow_policies (
            policy_id TEXT PRIMARY KEY,
            start_ts TEXT NOT NULL,
            specification_json TEXT NOT NULL,
            specification_hash TEXT NOT NULL
        );
CREATE TABLE paper_trade_annotations (
            trade_id INTEGER PRIMARY KEY REFERENCES paper_trades(id),
            notes TEXT NOT NULL DEFAULT '',
            actor TEXT NOT NULL DEFAULT 'user',
            updated_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
CREATE TABLE paper_trade_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER NOT NULL REFERENCES paper_trades(id),
            event_type TEXT NOT NULL CHECK (
                event_type IN ('fill','mark','management','close','corporate_action')
            ),
            event_date TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            created_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(trade_id,event_type,event_date,payload_hash)
        );
CREATE TABLE paper_trade_reflections (
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
        );
CREATE TABLE paper_trade_schema_meta (
               id INTEGER PRIMARY KEY CHECK (id=1),
               schema_version INTEGER NOT NULL
           );
INSERT INTO "paper_trade_schema_meta" VALUES(1,4);
CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date TEXT NOT NULL,
            generated_ts TEXT,
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
            decision_version TEXT,
            decision_state TEXT,
            analyst_stage TEXT,
            debate_stage TEXT,
            risk_stage TEXT,
            portfolio_decision TEXT,
            decision_summary TEXT,
            decision_reasons TEXT,
            risk_flags TEXT,
            position_size_pct REAL,
            risk_budget_pct REAL,
            stop_distance_pct REAL,
            expected_reward_pct REAL,
            expected_r_multiple REAL,
            entry_plan TEXT,
            exit_plan TEXT,
            sizing_summary TEXT,
            review_status TEXT,
            review_summary TEXT,
            bot_version TEXT,
            vix_at_entry REAL,
            vix_percentile_at_entry REAL,
            regime_state_at_entry TEXT,
            hmm_regime_at_entry TEXT,
            hmm_confidence_at_entry REAL,
            ml_predicted_win_prob REAL,
            ml_confidence REAL,
            ml_signal TEXT,
            notes TEXT DEFAULT '',
            directional_regime_at_entry TEXT,
            directional_regime_confidence REAL,
            entry_reason TEXT,
            entry_evidence TEXT,
            entry_risks TEXT,
            quantity REAL,
            entry_notional REAL,
            entry_commission REAL,
            exit_commission REAL,
            borrow_cost REAL,
            realized_pnl_usd REAL,
            accounting_status TEXT NOT NULL DEFAULT 'legacy_unreconciled',
            report_run_id TEXT,
            campaign_id TEXT NOT NULL DEFAULT 'paper-v1',
            policy_version TEXT,
            UNIQUE(report_date, ticker, direction)
        );
INSERT INTO paper_trades (
    id,report_date,generated_ts,ticker,direction,entry_price,entry_date,status,
    current_price,quantity,entry_notional,entry_commission,accounting_status,
    campaign_id,policy_version
) VALUES (
    101,'2026-08-22','2026-08-22T22:00:00Z','FIXTURE','long',100.0,
    '2026-08-25','open',101.0,100.0,10000.0,10.0,
    'legacy_unreconciled','paper-v1','paper-trade-eval-v1'
);
CREATE TABLE report_admission_attempts (
            attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES report_runs(run_id),
            status TEXT NOT NULL CHECK (status IN ('succeeded','failed')),
            error_code TEXT,
            error_message TEXT,
            attempted_ts TEXT NOT NULL
                CHECK (
                    attempted_ts GLOB '????-??-??T??:??:??Z'
                    AND attempted_ts NOT GLOB '*[^0-9TZ:-]*'
                    AND strftime('%Y-%m-%dT%H:%M:%SZ',attempted_ts) IS NOT NULL
                    AND strftime('%Y-%m-%dT%H:%M:%SZ',attempted_ts)=attempted_ts
                    AND date(substr(attempted_ts,1,10),'+0 days')=substr(attempted_ts,1,10)
                    AND substr(attempted_ts,1,4) BETWEEN '0001' AND '9999'
                    AND substr(attempted_ts,12,2) BETWEEN '00' AND '23'
                    AND substr(attempted_ts,15,2) BETWEEN '00' AND '59'
                    AND substr(attempted_ts,18,2) BETWEEN '00' AND '59'
                ),
            CHECK (
                (status='succeeded' AND error_code IS NULL AND error_message IS NULL)
                OR
                (status='failed' AND COALESCE(error_code,'') IN ('admission_finalize_failed','admission_paper_trade_persistence_failed','admission_setup_persistence_failed','report_not_current_publication','report_not_verified_published','report_publication_lineage_invalid')
                 AND COALESCE(error_message,'') GLOB '[A-Za-z_]*'
                 AND COALESCE(error_message,'') NOT GLOB '*[^A-Za-z0-9_]*')
            )
        );
CREATE TABLE report_run_decisions (
            run_id TEXT NOT NULL REFERENCES report_runs(run_id),
            ticker TEXT NOT NULL,
            selected_rank INTEGER NOT NULL,
            decision TEXT NOT NULL CHECK (decision IN ('accepted','rejected')),
            reason_codes_json TEXT NOT NULL,
            inputs_json TEXT NOT NULL,
            PRIMARY KEY (run_id, ticker)
        );
CREATE TABLE report_runs (
            run_id TEXT PRIMARY KEY,
            report_kind TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('started','completed','failed','published')),
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
            markdown_hash TEXT,
            artifact_path TEXT,
            markdown_path TEXT,
            error_message TEXT,
            generation_key TEXT,
            is_generation_canonical INTEGER NOT NULL DEFAULT 0,
            publication_verified INTEGER NOT NULL DEFAULT 0,
            superseded_by_run_id TEXT REFERENCES report_runs(run_id)
        );
CREATE TABLE report_schema_migrations (
               migration TEXT PRIMARY KEY,
               applied_ts TEXT NOT NULL
           );
INSERT INTO "report_schema_migrations" VALUES('admission-ledger-contract-v5','2026-08-26T00:29:18Z');
CREATE TABLE schema_migrations (
            migration_id TEXT PRIMARY KEY,
            applied_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
INSERT INTO "schema_migrations" VALUES('paper_campaign_v2_inactive_governed_20260823','2026-08-26 00:29:18');
INSERT INTO "schema_migrations" VALUES('paper_campaign_v1_backfill_20260822','2026-08-26 00:29:18');
CREATE INDEX idx_report_admission_attempts_run ON report_admission_attempts(run_id,attempt_id);
CREATE TRIGGER report_admission_attempts_valid_insert
           BEFORE INSERT ON report_admission_attempts
           WHEN NEW.attempted_ts IS NULL
             OR NEW.status IS NULL
             OR NEW.run_id IS NULL
             OR NEW.attempted_ts NOT GLOB '????-??-??T??:??:??Z'
             OR NEW.attempted_ts GLOB '*[^0-9TZ:-]*'
             OR strftime('%Y-%m-%dT%H:%M:%SZ',NEW.attempted_ts) IS NULL
             OR strftime('%Y-%m-%dT%H:%M:%SZ',NEW.attempted_ts)!=NEW.attempted_ts
             OR date(substr(NEW.attempted_ts,1,10),'+0 days')!=substr(NEW.attempted_ts,1,10)
             OR substr(NEW.attempted_ts,1,4) NOT BETWEEN '0001' AND '9999'
             OR substr(NEW.attempted_ts,12,2) NOT BETWEEN '00' AND '23'
             OR substr(NEW.attempted_ts,15,2) NOT BETWEEN '00' AND '59'
             OR substr(NEW.attempted_ts,18,2) NOT BETWEEN '00' AND '59'
             OR NOT EXISTS (SELECT 1 FROM report_runs WHERE run_id=NEW.run_id)
             OR COALESCE(NOT (
                 (NEW.status='succeeded' AND NEW.error_code IS NULL
                  AND NEW.error_message IS NULL)
                 OR
                 (NEW.status='failed' AND COALESCE(NEW.error_code,'') IN ('admission_finalize_failed','admission_paper_trade_persistence_failed','admission_setup_persistence_failed','report_not_current_publication','report_not_verified_published','report_publication_lineage_invalid')
                  AND COALESCE(NEW.error_message,'') GLOB '[A-Za-z_]*'
                  AND COALESCE(NEW.error_message,'') NOT GLOB '*[^A-Za-z0-9_]*')
             ),1)
           BEGIN SELECT RAISE(ABORT,'invalid report admission attempt'); END;
CREATE TRIGGER report_admission_attempts_no_update
           BEFORE UPDATE ON report_admission_attempts
           BEGIN SELECT RAISE(ABORT,'report admission attempts are immutable'); END;
CREATE TRIGGER report_admission_attempts_no_delete
           BEFORE DELETE ON report_admission_attempts
           BEGIN SELECT RAISE(ABORT,'report admission attempts are immutable'); END;
CREATE INDEX idx_report_runs_published ON report_runs(status,generated_ts,published_ts DESC,run_id DESC);
CREATE UNIQUE INDEX idx_report_runs_canonical_generation ON report_runs(generation_key) WHERE is_generation_canonical=1 AND generation_key IS NOT NULL;
CREATE TRIGGER report_runs_started_insert_only
            BEFORE INSERT ON report_runs
            WHEN NEW.status!='started' OR TRIM(NEW.run_id)='' OR TRIM(NEW.report_kind)=''
              OR NEW.started_ts NOT GLOB '????-??-??T??:??:??Z'
              OR strftime('%Y-%m-%dT%H:%M:%SZ',NEW.started_ts)!=NEW.started_ts
              OR json_valid(NEW.config_json)!=1 OR json_type(NEW.config_json)!='object'
              OR length(NEW.config_hash)!=64 OR lower(NEW.config_hash) GLOB '*[^0-9a-f]*'
              OR length(NEW.code_version) NOT IN (40,64) OR lower(NEW.code_version) GLOB '*[^0-9a-f]*'
              OR NEW.is_generation_canonical!=0 OR NEW.publication_verified!=0
            BEGIN SELECT RAISE(ABORT,'report runs must begin in started state with valid evidence'); END;
CREATE TRIGGER report_runs_terminal_evidence
            BEFORE UPDATE ON report_runs
            WHEN NEW.status IS NOT OLD.status AND (
              (NEW.status='completed' AND (
                NEW.completed_ts NOT GLOB '????-??-??T??:??:??Z'
                OR strftime('%Y-%m-%dT%H:%M:%SZ',NEW.completed_ts)!=NEW.completed_ts
                OR julianday(NEW.completed_ts)<julianday(NEW.started_ts)
                OR NEW.generated_ts NOT GLOB '????-??-??T??:??:??Z'
                OR strftime('%Y-%m-%dT%H:%M:%SZ',NEW.generated_ts)!=NEW.generated_ts
                OR julianday(NEW.generated_ts)<julianday(NEW.started_ts)
                OR julianday(NEW.generated_ts)>julianday(NEW.completed_ts)
                OR NEW.generation_key!=(NEW.report_kind||':'||NEW.generated_ts)
                OR json_type(NEW.scanned_universe_json)!='array'
                OR json_type(NEW.ranked_candidates_json)!='array'
                OR json_type(NEW.decisions_json)!='array'
                OR json_type(NEW.inputs_json)!='object'
                OR json_type(NEW.source_timestamps_json)!='object'
                OR length(NEW.content_hash)!=64 OR lower(NEW.content_hash) GLOB '*[^0-9a-f]*'
                OR length(NEW.markdown_hash)!=64 OR lower(NEW.markdown_hash) GLOB '*[^0-9a-f]*'
                OR TRIM(COALESCE(NEW.artifact_path,''))='' OR TRIM(COALESCE(NEW.markdown_path,''))=''
              )) OR
              (NEW.status='failed' AND (
                NEW.failed_ts NOT GLOB '????-??-??T??:??:??Z'
                OR strftime('%Y-%m-%dT%H:%M:%SZ',NEW.failed_ts)!=NEW.failed_ts
                OR julianday(NEW.failed_ts)<julianday(NEW.started_ts)
                OR TRIM(COALESCE(NEW.error_message,''))=''
              )) OR
              (NEW.status='published' AND (
                NEW.published_ts NOT GLOB '????-??-??T??:??:??Z'
                OR strftime('%Y-%m-%dT%H:%M:%SZ',NEW.published_ts)!=NEW.published_ts
                OR julianday(NEW.published_ts)<julianday(NEW.completed_ts)
                OR NEW.publication_verified!=1 OR NEW.is_generation_canonical!=0
              ))
            ) BEGIN SELECT RAISE(ABORT,'terminal report run requires complete evidence'); END;
CREATE TRIGGER report_runs_valid_transition
            BEFORE UPDATE ON report_runs
            WHEN NEW.status IS NOT OLD.status AND NOT (
              (OLD.status='started' AND NEW.status IN ('completed','failed')) OR
              (OLD.status='completed' AND NEW.status='published')
            ) BEGIN SELECT RAISE(ABORT,'invalid report run state transition'); END;
CREATE TRIGGER report_runs_failed_immutable
            BEFORE UPDATE ON report_runs WHEN OLD.status='failed'
            BEGIN SELECT RAISE(ABORT,'failed report run is immutable'); END;
CREATE TRIGGER report_runs_started_identity_immutable
            BEFORE UPDATE ON report_runs
            WHEN OLD.status='started' AND (
              NEW.run_id IS NOT OLD.run_id OR NEW.report_kind IS NOT OLD.report_kind
              OR NEW.started_ts IS NOT OLD.started_ts
              OR NEW.config_json IS NOT OLD.config_json
              OR NEW.config_hash IS NOT OLD.config_hash
              OR NEW.code_version IS NOT OLD.code_version
            ) BEGIN SELECT RAISE(ABORT,'started report identity is immutable'); END;
CREATE TRIGGER report_runs_snapshot_immutable
            BEFORE UPDATE ON report_runs
            WHEN OLD.status IN ('completed','published') AND (
              NEW.run_id IS NOT OLD.run_id OR NEW.report_kind IS NOT OLD.report_kind
              OR NEW.started_ts IS NOT OLD.started_ts OR NEW.completed_ts IS NOT OLD.completed_ts
              OR NEW.failed_ts IS NOT OLD.failed_ts OR NEW.generated_ts IS NOT OLD.generated_ts
              OR NEW.scanned_universe_json IS NOT OLD.scanned_universe_json
              OR NEW.ranked_candidates_json IS NOT OLD.ranked_candidates_json
              OR NEW.decisions_json IS NOT OLD.decisions_json OR NEW.inputs_json IS NOT OLD.inputs_json
              OR NEW.source_timestamps_json IS NOT OLD.source_timestamps_json
              OR NEW.config_json IS NOT OLD.config_json OR NEW.config_hash IS NOT OLD.config_hash
              OR NEW.code_version IS NOT OLD.code_version OR NEW.content_hash IS NOT OLD.content_hash
              OR NEW.markdown_hash IS NOT OLD.markdown_hash OR NEW.artifact_path IS NOT OLD.artifact_path
              OR NEW.markdown_path IS NOT OLD.markdown_path OR NEW.error_message IS NOT OLD.error_message
              OR NEW.generation_key IS NOT OLD.generation_key
              OR (OLD.status='completed' AND NEW.is_generation_canonical IS NOT OLD.is_generation_canonical)
              OR (OLD.status='completed' AND NEW.superseded_by_run_id IS NOT OLD.superseded_by_run_id)
              OR (OLD.status='published' AND NEW.published_ts IS NOT OLD.published_ts)
              OR (OLD.status='published' AND NEW.publication_verified IS NOT OLD.publication_verified)
            ) BEGIN SELECT RAISE(ABORT,'completed report snapshot is immutable'); END;
CREATE TRIGGER report_runs_pointer_transition
            BEFORE UPDATE ON report_runs
            WHEN OLD.status='published' AND (
              (NEW.is_generation_canonical IS NOT OLD.is_generation_canonical OR
               NEW.superseded_by_run_id IS NOT OLD.superseded_by_run_id) AND NOT (
                (OLD.is_generation_canonical=0 AND NEW.is_generation_canonical=1
                 AND OLD.superseded_by_run_id IS NULL AND NEW.superseded_by_run_id IS NULL
                 AND NOT EXISTS (SELECT 1 FROM report_runs r WHERE r.generation_key=OLD.generation_key
                                 AND r.is_generation_canonical=1 AND r.run_id!=OLD.run_id))
                OR
                (OLD.is_generation_canonical=1 AND NEW.is_generation_canonical=0
                 AND OLD.superseded_by_run_id IS NULL AND NEW.superseded_by_run_id IS NOT NULL
                 AND EXISTS (SELECT 1 FROM report_runs r WHERE r.run_id=NEW.superseded_by_run_id
                             AND r.generation_key=OLD.generation_key AND r.status='published'
                             AND r.publication_verified=1))
              ))
            BEGIN SELECT RAISE(ABORT,'invalid canonical report transition'); END;
CREATE TRIGGER report_runs_immutable_delete
            BEFORE DELETE ON report_runs BEGIN SELECT RAISE(ABORT,'report runs are immutable'); END;
CREATE TRIGGER report_run_decisions_parent_started
            BEFORE INSERT ON report_run_decisions
            WHEN COALESCE((SELECT status FROM report_runs WHERE run_id=NEW.run_id),'')!='started'
            BEGIN SELECT RAISE(ABORT,'report decisions require a started parent run'); END;
CREATE TRIGGER report_run_decisions_immutable_update
            BEFORE UPDATE ON report_run_decisions BEGIN SELECT RAISE(ABORT,'report decisions are immutable'); END;
CREATE TRIGGER report_run_decisions_immutable_delete
            BEFORE DELETE ON report_run_decisions BEGIN SELECT RAISE(ABORT,'report decisions are immutable'); END;
CREATE INDEX idx_paper_trades_status ON paper_trades(status, entry_date);
CREATE INDEX idx_paper_trades_ticker ON paper_trades(ticker, status);
CREATE INDEX idx_paper_trades_family ON paper_trades(setup_family, direction, status);
CREATE INDEX idx_paper_trades_report_run ON paper_trades(report_run_id);
CREATE UNIQUE INDEX idx_paper_trades_campaign_unique ON paper_trades(campaign_id,report_date,ticker,direction);
CREATE UNIQUE INDEX idx_paper_trades_legacy_compat ON paper_trades(report_date,ticker,direction);
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
        END;
CREATE TRIGGER paper_trades_immutable_lineage
        BEFORE UPDATE OF report_run_id ON paper_trades
        WHEN NEW.report_run_id IS NOT OLD.report_run_id
        BEGIN
            SELECT RAISE(ABORT, 'paper trade report lineage is immutable');
        END;
CREATE UNIQUE INDEX idx_one_active_paper_campaign ON paper_campaigns(status) WHERE status='active';
CREATE TRIGGER paper_campaign_audit_no_update
        BEFORE UPDATE ON paper_campaign_audit
        BEGIN SELECT RAISE(ABORT, 'paper campaign audit is immutable'); END;
CREATE TRIGGER paper_campaign_audit_no_delete
        BEFORE DELETE ON paper_campaign_audit
        BEGIN SELECT RAISE(ABORT, 'paper campaign audit is immutable'); END;
CREATE INDEX idx_candidate_decisions_campaign_report ON paper_candidate_decisions(campaign_id, report_date, report_run_id, candidate_rank);
CREATE TRIGGER paper_candidate_decisions_no_update
        BEFORE UPDATE ON paper_candidate_decisions
        BEGIN SELECT RAISE(ABORT, 'paper candidate decisions are immutable'); END;
CREATE TRIGGER paper_candidate_decisions_no_delete
        BEFORE DELETE ON paper_candidate_decisions
        BEGIN SELECT RAISE(ABORT, 'paper candidate decisions are immutable'); END;
CREATE TRIGGER paper_decision_sets_no_update
        BEFORE UPDATE ON paper_decision_sets
        BEGIN SELECT RAISE(ABORT, 'paper decision sets are immutable'); END;
CREATE TRIGGER paper_decision_sets_no_delete
        BEFORE DELETE ON paper_decision_sets
        BEGIN SELECT RAISE(ABORT, 'paper decision sets are immutable'); END;
CREATE TRIGGER paper_candidate_decisions_no_insert_after_seal
        BEFORE INSERT ON paper_candidate_decisions
        WHEN EXISTS (
            SELECT 1 FROM paper_decision_sets
            WHERE report_run_id=NEW.report_run_id
              AND campaign_id=NEW.campaign_id
              AND status='sealed'
        )
        BEGIN SELECT RAISE(ABORT, 'sealed paper decision set is not appendable'); END;
CREATE TRIGGER paper_pending_orders_valid_insert
        BEFORE INSERT ON paper_pending_orders
        WHEN NEW.status!='pending' OR NEW.resolved_ts IS NOT NULL
          OR NEW.order_hash IS NULL
          OR length(NEW.order_hash)!=64
          OR lower(NEW.order_hash) GLOB '*[^0-9a-f]*'
        BEGIN SELECT RAISE(ABORT, 'pending order requires a sealed immutable payload'); END;
CREATE TRIGGER paper_pending_orders_immutable_payload
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
        BEGIN SELECT RAISE(ABORT, 'pending order payload is immutable'); END;
CREATE TRIGGER paper_pending_orders_terminal_transition
        BEFORE UPDATE OF status,resolved_ts ON paper_pending_orders
        WHEN OLD.status!='pending' OR NEW.status NOT IN ('filled','rejected','cancelled')
          OR NEW.resolved_ts IS NULL
        BEGIN SELECT RAISE(ABORT, 'pending order has an invalid terminal transition'); END;
CREATE TRIGGER paper_pending_orders_no_delete
        BEFORE DELETE ON paper_pending_orders
        BEGIN SELECT RAISE(ABORT, 'pending orders are immutable audit facts'); END;
CREATE TRIGGER paper_order_events_no_update
        BEFORE UPDATE ON paper_order_events
        BEGIN SELECT RAISE(ABORT, 'paper order events are immutable'); END;
CREATE TRIGGER paper_order_events_no_delete
        BEFORE DELETE ON paper_order_events
        BEGIN SELECT RAISE(ABORT, 'paper order events are immutable'); END;
CREATE INDEX idx_paper_trade_events_timeline ON paper_trade_events(trade_id,event_date,id);
CREATE TRIGGER paper_trade_events_no_update
        BEFORE UPDATE ON paper_trade_events
        BEGIN SELECT RAISE(ABORT, 'paper trade events are append-only'); END;
CREATE TRIGGER paper_trade_events_no_delete
        BEFORE DELETE ON paper_trade_events
        BEGIN SELECT RAISE(ABORT, 'paper trade events are append-only'); END;
CREATE TRIGGER paper_campaign_preregistrations_no_update
            BEFORE UPDATE ON paper_campaign_preregistrations
            BEGIN SELECT RAISE(ABORT, 'paper campaign preregistrations are immutable'); END;
CREATE TRIGGER paper_campaign_preregistrations_no_delete
            BEFORE DELETE ON paper_campaign_preregistrations
            BEGIN SELECT RAISE(ABORT, 'paper campaign preregistrations are immutable'); END;
CREATE TRIGGER paper_campaign_experiments_no_update
            BEFORE UPDATE ON paper_campaign_experiments
            BEGIN SELECT RAISE(ABORT, 'paper campaign experiments are immutable'); END;
CREATE TRIGGER paper_campaign_experiments_no_delete
            BEFORE DELETE ON paper_campaign_experiments
            BEGIN SELECT RAISE(ABORT, 'paper campaign experiments are immutable'); END;
CREATE TRIGGER paper_campaign_approvals_no_update
            BEFORE UPDATE ON paper_campaign_approvals
            BEGIN SELECT RAISE(ABORT, 'paper campaign approvals are immutable'); END;
CREATE TRIGGER paper_campaign_approvals_no_delete
            BEFORE DELETE ON paper_campaign_approvals
            BEGIN SELECT RAISE(ABORT, 'paper campaign approvals are immutable'); END;
CREATE TRIGGER paper_v1_trades_no_insert
        BEFORE INSERT ON paper_trades WHEN NEW.campaign_id = 'paper-v1'
        BEGIN SELECT RAISE(ABORT, 'paper campaign v1 is immutable'); END;
CREATE TRIGGER paper_v1_trades_no_update
        BEFORE UPDATE ON paper_trades
        WHEN OLD.campaign_id = 'paper-v1'
          AND NEW.report_run_id IS OLD.report_run_id
        BEGIN SELECT RAISE(ABORT, 'paper campaign v1 is immutable'); END;
CREATE TRIGGER paper_v1_trades_no_delete
        BEFORE DELETE ON paper_trades WHEN OLD.campaign_id = 'paper-v1'
        BEGIN SELECT RAISE(ABORT, 'paper campaign v1 is immutable'); END;
CREATE TRIGGER paper_v1_campaign_no_update
        BEFORE UPDATE ON paper_campaigns WHEN OLD.campaign_id='paper-v1'
        BEGIN SELECT RAISE(ABORT, 'paper campaign v1 metadata is immutable'); END;
CREATE TRIGGER paper_v1_campaign_no_delete
        BEFORE DELETE ON paper_campaigns WHEN OLD.campaign_id='paper-v1'
        BEGIN SELECT RAISE(ABORT, 'paper campaign v1 metadata is immutable'); END;
CREATE INDEX idx_bot_versions_status ON bot_versions(status, created_ts);
CREATE UNIQUE INDEX idx_paper_portfolio_legacy_compat ON paper_portfolio_snapshots(snapshot_date);
CREATE UNIQUE INDEX idx_paper_portfolio_campaign_date ON paper_portfolio_snapshots(campaign_id, snapshot_date);
CREATE INDEX idx_paper_portfolio_date ON paper_portfolio_snapshots(snapshot_date);
CREATE INDEX idx_paper_reflections_trade ON paper_trade_reflections(trade_id);
CREATE INDEX idx_paper_reflections_ticker ON paper_trade_reflections(ticker, exit_date);
CREATE TRIGGER paper_shadow_policies_no_update
        BEFORE UPDATE ON paper_shadow_policies
        BEGIN SELECT RAISE(ABORT,'shadow policies are immutable'); END;
CREATE TRIGGER paper_shadow_policies_no_delete
        BEFORE DELETE ON paper_shadow_policies
        BEGIN SELECT RAISE(ABORT,'shadow policies are immutable'); END;
CREATE TRIGGER paper_shadow_decisions_no_update
        BEFORE UPDATE ON paper_shadow_decisions
        BEGIN SELECT RAISE(ABORT,'shadow decisions are immutable'); END;
CREATE TRIGGER paper_shadow_decisions_no_delete
        BEFORE DELETE ON paper_shadow_decisions
        BEGIN SELECT RAISE(ABORT,'shadow decisions are immutable'); END;
CREATE TRIGGER paper_shadow_sets_no_update
        BEFORE UPDATE ON paper_shadow_decision_sets
        BEGIN SELECT RAISE(ABORT,'shadow decision sets are immutable'); END;
CREATE TRIGGER paper_shadow_sets_no_delete
        BEFORE DELETE ON paper_shadow_decision_sets
        BEGIN SELECT RAISE(ABORT,'shadow decision sets are immutable'); END;
CREATE TRIGGER paper_shadow_decisions_no_insert_after_seal
        BEFORE INSERT ON paper_shadow_decisions
        WHEN EXISTS (
            SELECT 1 FROM paper_shadow_decision_sets
            WHERE report_run_id=NEW.report_run_id AND policy_id=NEW.policy_id
        )
        BEGIN SELECT RAISE(ABORT,'sealed shadow decision set is not appendable'); END;
CREATE TRIGGER paper_shadow_outcomes_no_update
        BEFORE UPDATE ON paper_shadow_outcomes
        BEGIN SELECT RAISE(ABORT,'shadow outcomes are immutable'); END;
CREATE TRIGGER paper_shadow_outcomes_no_delete
        BEFORE DELETE ON paper_shadow_outcomes
        BEGIN SELECT RAISE(ABORT,'shadow outcomes are immutable'); END;
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('paper_trades',101);
INSERT INTO "sqlite_sequence" VALUES('paper_portfolio_snapshots',201);
COMMIT;
