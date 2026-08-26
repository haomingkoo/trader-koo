"""Immutable report runs with one verified-publication seam.

SQLite owns publication history. JSON and Markdown are immutable evidence;
``daily_report_latest.*`` and its manifest are compatibility copies rebuilt
from that history. Callers must use :func:`resolve_published_report` rather
than infer trust from status flags or files.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trader_koo.db.report_price_contract import PriceContractError
from trader_koo.paper_trade.errors import (
    ADMISSION_ERROR_CODES,
    ADMISSION_LEDGER_MIGRATION,
    AdmissionLedgerContractError,
    LEGACY_ADMISSION_ERROR_CODES,
    ReportLineageError,
)

LOG = logging.getLogger(__name__)

LATEST_MANIFEST = "daily_report_latest.manifest.json"
PUBLICATION_LOCK = ".daily_report_publication.lock"
_GIT_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$", re.IGNORECASE)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc(value: Any, *, field: str) -> dt.datetime:
    text = str(value or "")
    if not _UTC_RE.fullmatch(text):
        raise ValueError(f"{field} must be a precise UTC timestamp")
    try:
        return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError as exc:
        raise ValueError(f"{field} must be a precise UTC timestamp") from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_code_version() -> str:
    for name in (
        "TRADER_KOO_GIT_SHA",
        "RAILWAY_GIT_COMMIT_SHA",
        "GITHUB_SHA",
        "SOURCE_VERSION",
        "VERCEL_GIT_COMMIT_SHA",
    ):
        if configured := str(os.getenv(name) or "").strip():
            return _validated_code_version(configured)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, timeout=3,
        )
        if version := result.stdout.strip():
            return _validated_code_version(version)
    except (OSError, subprocess.SubprocessError):
        pass
    raise RuntimeError("report publication requires an exact deployed code version")


def _validated_code_version(value: str) -> str:
    version = str(value or "").strip().lower()
    if not _GIT_SHA_RE.fullmatch(version):
        raise RuntimeError("report publication requires a full 40- or 64-character Git commit SHA")
    return version


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    if column not in {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def ensure_report_run_schema(
    conn: sqlite3.Connection, *, verify_admission_contract: bool = False
) -> None:
    """Install/migrate the registry atomically without committing caller work."""
    caller_transaction = conn.in_transaction
    if not caller_transaction:
        conn.execute("BEGIN IMMEDIATE")
    try:
        _ensure_report_run_schema(
            conn, verify_admission_contract=verify_admission_contract
        )
    except Exception:
        if not caller_transaction:
            conn.rollback()
        raise
    if not caller_transaction:
        conn.commit()


def _ensure_report_run_schema(
    conn: sqlite3.Connection, *, verify_admission_contract: bool = False
) -> None:
    allowed_admission_codes = ",".join(
        f"'{code}'" for code in sorted(ADMISSION_ERROR_CODES)
    )
    allowed_historical_admission_codes = ",".join(
        f"'{code}'"
        for code in sorted(ADMISSION_ERROR_CODES | LEGACY_ADMISSION_ERROR_CODES)
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS report_runs (
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
        )"""
    )
    # Older installations and lightweight API fixtures may already have a
    # three-column ``report_runs`` registry.  Add the complete evidence shape
    # before creating indexes or triggers that reference it.
    for column, ddl in (
        ("report_kind", "report_kind TEXT"),
        ("started_ts", "started_ts TEXT"),
        ("completed_ts", "completed_ts TEXT"),
        ("failed_ts", "failed_ts TEXT"),
        ("published_ts", "published_ts TEXT"),
        ("generated_ts", "generated_ts TEXT"),
        ("scanned_universe_json", "scanned_universe_json TEXT"),
        ("ranked_candidates_json", "ranked_candidates_json TEXT"),
        ("decisions_json", "decisions_json TEXT"),
        ("inputs_json", "inputs_json TEXT"),
        ("source_timestamps_json", "source_timestamps_json TEXT"),
        ("config_json", "config_json TEXT"),
        ("config_hash", "config_hash TEXT"),
        ("code_version", "code_version TEXT"),
        ("content_hash", "content_hash TEXT"),
        ("artifact_path", "artifact_path TEXT"),
        ("markdown_path", "markdown_path TEXT"),
        ("error_message", "error_message TEXT"),
        ("generation_key", "generation_key TEXT"),
        ("is_generation_canonical", "is_generation_canonical INTEGER NOT NULL DEFAULT 0"),
        ("publication_verified", "publication_verified INTEGER NOT NULL DEFAULT 0"),
        ("superseded_by_run_id", "superseded_by_run_id TEXT REFERENCES report_runs(run_id)"),
        ("markdown_hash", "markdown_hash TEXT"),
    ):
        _ensure_column(conn, "report_runs", column, ddl)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS report_run_decisions (
            run_id TEXT NOT NULL REFERENCES report_runs(run_id),
            ticker TEXT NOT NULL,
            selected_rank INTEGER NOT NULL,
            decision TEXT NOT NULL CHECK (decision IN ('accepted','rejected')),
            reason_codes_json TEXT NOT NULL,
            inputs_json TEXT NOT NULL,
            PRIMARY KEY (run_id, ticker)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS report_admission_attempts (
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
                (status='failed' AND COALESCE(error_code,'') IN ({allowed_admission_codes})
                 AND COALESCE(error_message,'') GLOB '[A-Za-z_]*'
                 AND COALESCE(error_message,'') NOT GLOB '*[^A-Za-z0-9_]*')
            )
        )""".format(allowed_admission_codes=allowed_admission_codes)
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_report_admission_attempts_run "
        "ON report_admission_attempts(run_id,attempt_id)"
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS report_schema_migrations (
               migration TEXT PRIMARY KEY,
               applied_ts TEXT NOT NULL
           )"""
    )
    # v5 rescans v4 ledgers so exception-class metadata uses one exact contract
    # in SQLite validation and Python diagnostics, including whitespace.
    admission_contract_migration = ADMISSION_LEDGER_MIGRATION
    needs_admission_scan = verify_admission_contract or conn.execute(
        "SELECT 1 FROM report_schema_migrations WHERE migration=?",
        (admission_contract_migration,),
    ).fetchone() is None
    invalid_admission_predicate = """attempted_ts NOT GLOB '????-??-??T??:??:??Z'
              OR attempted_ts IS NULL
              OR status IS NULL
              OR run_id IS NULL
              OR attempted_ts GLOB '*[^0-9TZ:-]*'
              OR strftime('%Y-%m-%dT%H:%M:%SZ',attempted_ts) IS NULL
              OR strftime('%Y-%m-%dT%H:%M:%SZ',attempted_ts)!=attempted_ts
              OR date(substr(attempted_ts,1,10),'+0 days')!=substr(attempted_ts,1,10)
              OR substr(attempted_ts,1,4) NOT BETWEEN '0001' AND '9999'
              OR substr(attempted_ts,12,2) NOT BETWEEN '00' AND '23'
              OR substr(attempted_ts,15,2) NOT BETWEEN '00' AND '59'
              OR substr(attempted_ts,18,2) NOT BETWEEN '00' AND '59'
              OR NOT EXISTS (
                  SELECT 1 FROM report_runs WHERE run_id=report_admission_attempts.run_id
              )
              OR COALESCE(NOT (
                  (status='succeeded' AND error_code IS NULL AND error_message IS NULL)
                  OR
                  (status='failed' AND COALESCE(error_code,'') IN ({allowed_historical_admission_codes})
                   AND COALESCE(error_message,'') GLOB '[A-Za-z_]*'
                   AND COALESCE(error_message,'') NOT GLOB '*[^A-Za-z0-9_]*')
              ),1)""".format(
        allowed_historical_admission_codes=allowed_historical_admission_codes
    )
    invalid_attempts = conn.execute(
        f"SELECT COUNT(*) FROM report_admission_attempts WHERE {invalid_admission_predicate}"
    ).fetchone()[0] if needs_admission_scan else 0
    if invalid_attempts:
        invalid_rows = conn.execute(
            f"""SELECT attempt_id,run_id,status,error_code,error_message,
                      attempted_ts,
                      EXISTS(SELECT 1 FROM report_runs r
                             WHERE r.run_id=report_admission_attempts.run_id)
               FROM report_admission_attempts
               WHERE {invalid_admission_predicate}
               ORDER BY attempt_id LIMIT 20"""
        ).fetchall()
        diagnostics: list[dict[str, object]] = []
        historical_codes = ADMISSION_ERROR_CODES | LEGACY_ADMISSION_ERROR_CODES
        for attempt_id, run_id, status, code, message, attempted_ts, known_run in invalid_rows:
            violations: list[str] = []
            if run_id is None:
                violations.append("run_id_missing")
            elif not known_run:
                violations.append("run_id_unknown")
            try:
                _utc(attempted_ts, field="attempted_ts")
            except ValueError:
                violations.append("attempted_ts_invalid")
            if status not in {"succeeded", "failed"}:
                violations.append("status_invalid")
            elif status == "succeeded" and (code is not None or message is not None):
                violations.append("success_error_metadata_present")
            elif status == "failed" and (
                code not in historical_codes
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(message or "")) is None
            ):
                violations.append("failure_error_metadata_invalid")
            diagnostics.append({
                "attempt_id": int(attempt_id),
                "violations": violations or ["row_contract_invalid"],
            })
        raise AdmissionLedgerContractError(int(invalid_attempts), diagnostics)
    # Reinstall the versioned validator once per ensure so a legacy trigger
    # cannot retain weaker rules. The full ledger scan above runs only once.
    conn.execute("DROP TRIGGER IF EXISTS report_admission_attempts_valid_insert")
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS report_admission_attempts_valid_insert
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
                 (NEW.status='failed' AND COALESCE(NEW.error_code,'') IN ({allowed_admission_codes})
                  AND COALESCE(NEW.error_message,'') GLOB '[A-Za-z_]*'
                  AND COALESCE(NEW.error_message,'') NOT GLOB '*[^A-Za-z0-9_]*')
             ),1)
           BEGIN SELECT RAISE(ABORT,'invalid report admission attempt'); END""".format(
               allowed_admission_codes=allowed_admission_codes
           )
    )
    conn.execute(
        "INSERT OR IGNORE INTO report_schema_migrations(migration,applied_ts) VALUES (?,?)",
        (admission_contract_migration, _utc_now()),
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS report_admission_attempts_no_update
           BEFORE UPDATE ON report_admission_attempts
           BEGIN SELECT RAISE(ABORT,'report admission attempts are immutable'); END"""
    )
    conn.execute(
        """CREATE TRIGGER IF NOT EXISTS report_admission_attempts_no_delete
           BEFORE DELETE ON report_admission_attempts
           BEGIN SELECT RAISE(ABORT,'report admission attempts are immutable'); END"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_report_runs_published "
        "ON report_runs(status,generated_ts,published_ts DESC,run_id DESC)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_report_runs_canonical_generation "
        "ON report_runs(generation_key) WHERE is_generation_canonical=1 AND generation_key IS NOT NULL"
    )
    conn.execute("DROP VIEW IF EXISTS report_publication_ownership")

    triggers = {
        "report_runs_started_insert_only": """
            BEFORE INSERT ON report_runs
            WHEN NEW.status!='started' OR TRIM(NEW.run_id)='' OR TRIM(NEW.report_kind)=''
              OR NEW.started_ts NOT GLOB '????-??-??T??:??:??Z'
              OR strftime('%Y-%m-%dT%H:%M:%SZ',NEW.started_ts)!=NEW.started_ts
              OR json_valid(NEW.config_json)!=1 OR json_type(NEW.config_json)!='object'
              OR length(NEW.config_hash)!=64 OR lower(NEW.config_hash) GLOB '*[^0-9a-f]*'
              OR length(NEW.code_version) NOT IN (40,64) OR lower(NEW.code_version) GLOB '*[^0-9a-f]*'
              OR NEW.is_generation_canonical!=0 OR NEW.publication_verified!=0
            BEGIN SELECT RAISE(ABORT,'report runs must begin in started state with valid evidence'); END
        """,
        "report_runs_terminal_evidence": """
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
            ) BEGIN SELECT RAISE(ABORT,'terminal report run requires complete evidence'); END
        """,
        "report_runs_valid_transition": """
            BEFORE UPDATE ON report_runs
            WHEN NEW.status IS NOT OLD.status AND NOT (
              (OLD.status='started' AND NEW.status IN ('completed','failed')) OR
              (OLD.status='completed' AND NEW.status='published')
            ) BEGIN SELECT RAISE(ABORT,'invalid report run state transition'); END
        """,
        "report_runs_failed_immutable": """
            BEFORE UPDATE ON report_runs WHEN OLD.status='failed'
            BEGIN SELECT RAISE(ABORT,'failed report run is immutable'); END
        """,
        "report_runs_started_identity_immutable": """
            BEFORE UPDATE ON report_runs
            WHEN OLD.status='started' AND (
              NEW.run_id IS NOT OLD.run_id OR NEW.report_kind IS NOT OLD.report_kind
              OR NEW.started_ts IS NOT OLD.started_ts
              OR NEW.config_json IS NOT OLD.config_json
              OR NEW.config_hash IS NOT OLD.config_hash
              OR NEW.code_version IS NOT OLD.code_version
            ) BEGIN SELECT RAISE(ABORT,'started report identity is immutable'); END
        """,
        "report_runs_snapshot_immutable": """
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
            ) BEGIN SELECT RAISE(ABORT,'completed report snapshot is immutable'); END
        """,
        "report_runs_pointer_transition": """
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
            BEGIN SELECT RAISE(ABORT,'invalid canonical report transition'); END
        """,
        "report_runs_immutable_delete": """
            BEFORE DELETE ON report_runs BEGIN SELECT RAISE(ABORT,'report runs are immutable'); END
        """,
        "report_run_decisions_parent_started": """
            BEFORE INSERT ON report_run_decisions
            WHEN COALESCE((SELECT status FROM report_runs WHERE run_id=NEW.run_id),'')!='started'
            BEGIN SELECT RAISE(ABORT,'report decisions require a started parent run'); END
        """,
        "report_run_decisions_immutable_update": """
            BEFORE UPDATE ON report_run_decisions BEGIN SELECT RAISE(ABORT,'report decisions are immutable'); END
        """,
        "report_run_decisions_immutable_delete": """
            BEFORE DELETE ON report_run_decisions BEGIN SELECT RAISE(ABORT,'report decisions are immutable'); END
        """,
    }
    for name in (
        "report_runs_immutable_snapshot", "report_runs_valid_transition",
        "report_run_decisions_immutable_update", "report_run_decisions_immutable_delete",
        "report_run_decisions_parent_started", "report_runs_immutable_delete",
        "report_runs_started_insert_only", "report_runs_terminal_evidence",
        "report_runs_failed_immutable", "report_runs_snapshot_immutable",
        "report_runs_pointer_transition", "report_runs_started_identity_immutable",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    for name, body in triggers.items():
        try:
            conn.execute(f"CREATE TRIGGER {name} {body}")
        except sqlite3.OperationalError as exc:
            raise sqlite3.OperationalError(f"cannot create {name}: {exc}") from exc


def start_report_run(
    conn: sqlite3.Connection,
    *,
    report_kind: str,
    configuration: dict[str, Any],
    code_version: str | None = None,
    started_ts: str | None = None,
) -> str:
    ensure_report_run_schema(conn)
    started = started_ts or _utc_now()
    _utc(started, field="started_ts")
    run_id = str(uuid.uuid4())
    config_json = _canonical_json(configuration)
    conn.execute(
        """INSERT INTO report_runs
           (run_id,report_kind,status,started_ts,config_json,config_hash,code_version)
           VALUES (?,?,'started',?,?,?,?)""",
        (run_id, str(report_kind).strip(), started, config_json,
         hashlib.sha256(config_json.encode()).hexdigest(),
         _validated_code_version(code_version) if code_version else current_code_version()),
    )
    conn.commit()
    return run_id


def _decision_snapshot(report: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    signals = report.get("signals") if isinstance(report.get("signals"), dict) else {}
    rows, universe = signals.get("report_decisions"), signals.get("scanned_universe")
    if not isinstance(rows, list) or not isinstance(universe, list):
        raise ValueError("report is missing its exact decision snapshot")
    decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("report decision snapshot contains a non-object row")
        ticker = str(row.get("ticker") or "").upper().strip()
        decision = str(row.get("decision") or "").lower().strip()
        reasons, inputs = row.get("reason_codes"), row.get("inputs")
        if (not ticker or ticker in seen or decision not in {"accepted", "rejected"}
                or not isinstance(reasons, list) or not reasons or not isinstance(inputs, dict)):
            raise ValueError(f"invalid exact report decision snapshot for {ticker or '<missing>'}")
        seen.add(ticker)
        decisions.append({
            "ticker": ticker, "selected_rank": int(row.get("selected_rank") or 0),
            "decision": decision, "reason_codes": [str(code) for code in reasons],
            "inputs": dict(inputs),
        })
    normalized = [str(ticker).upper().strip() for ticker in universe if str(ticker).strip()]
    if normalized != [item["ticker"] for item in decisions]:
        raise ValueError("scanned universe does not match exact report decisions")
    return normalized, decisions


def _load_artifact(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("report JSON artifact is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("report JSON artifact must be an object")
    return payload


def _load_artifact_bytes(data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("report JSON artifact is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("report JSON artifact must be an object")
    return payload


def _assert_publishable_report(report: dict[str, Any]) -> None:
    if report.get("ok") is not True:
        raise ValueError("degraded report cannot be completed or published (ok must be true)")
    if any(isinstance(report.get(key), list) and report.get(key) for key in ("warnings", "generation_warnings")):
        raise ValueError("partial-quality report warnings prevent publication")


def complete_report_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    report: dict[str, Any],
    artifact_path: Path,
    markdown_path: Path,
    content_hash: str,
    completed_ts: str | None = None,
) -> None:
    ensure_report_run_schema(conn)
    artifact_report = _load_artifact(artifact_path)
    _assert_publishable_report(artifact_report)
    if _canonical_json(report) != _canonical_json(artifact_report):
        raise ValueError("report payload differs from immutable JSON artifact")
    if sha256_file(artifact_path) != str(content_hash).lower():
        raise ValueError("report artifact hash mismatch before completion")
    markdown_hash = sha256_file(markdown_path)
    universe, decisions = _decision_snapshot(artifact_report)
    generated_ts = str(artifact_report.get("generated_ts") or "")
    generated = _utc(generated_ts, field="generated_ts")
    completed_text = completed_ts or _utc_now()
    completed = _utc(completed_text, field="completed_ts")
    if generated > completed:
        raise ValueError("generated_ts cannot be after completed_ts")
    meta = artifact_report.get("meta") if isinstance(artifact_report.get("meta"), dict) else {}
    report_kind = str(meta.get("report_kind") or "daily")
    run_meta = meta.get("report_run") if isinstance(meta.get("report_run"), dict) else {}
    if str(run_meta.get("run_id") or "") != run_id:
        raise ValueError("report artifact has the wrong run identity")
    row = conn.execute(
        "SELECT status,report_kind,started_ts FROM report_runs WHERE run_id=?",
        (run_id,),
    ).fetchone()
    if row is None or row[0] != "started" or str(row[1]) != report_kind:
        raise ValueError(f"report run {run_id} is not the matching started run")
    if generated < _utc(row[2], field="started_ts"):
        raise ValueError("generated_ts cannot be before started_ts")
    ranked = [{"ticker": item["ticker"], "selected_rank": item["selected_rank"]} for item in decisions]
    latest_data = artifact_report.get("latest_data") if isinstance(artifact_report.get("latest_data"), dict) else {}
    inputs = {
        "report_kind": report_kind, "market_session": artifact_report.get("market_session") or {},
        "counts": artifact_report.get("counts") or {}, "risk_filters": artifact_report.get("risk_filters") or {},
        "price_basis": meta.get("price_basis"),
    }
    conn.execute("BEGIN")
    try:
        conn.executemany(
            """INSERT INTO report_run_decisions
               (run_id,ticker,selected_rank,decision,reason_codes_json,inputs_json)
               VALUES (?,?,?,?,?,?)""",
            [(run_id, item["ticker"], item["selected_rank"], item["decision"],
              _canonical_json(item["reason_codes"]), _canonical_json(item["inputs"])) for item in decisions],
        )
        changed = conn.execute(
            """UPDATE report_runs SET status='completed',completed_ts=?,generated_ts=?,generation_key=?,
               scanned_universe_json=?,ranked_candidates_json=?,decisions_json=?,inputs_json=?,
               source_timestamps_json=?,content_hash=?,markdown_hash=?,artifact_path=?,markdown_path=?
               WHERE run_id=? AND status='started'""",
            (completed_text, generated_ts, f"{report_kind}:{generated_ts}", _canonical_json(universe),
             _canonical_json(ranked), _canonical_json(decisions), _canonical_json(inputs),
             _canonical_json(latest_data), str(content_hash).lower(), markdown_hash,
             str(artifact_path.resolve()), str(markdown_path.resolve()), run_id),
        ).rowcount
        if changed != 1:
            raise RuntimeError(f"report run {run_id} completion race")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def fail_report_run(conn: sqlite3.Connection, *, run_id: str, error: str) -> None:
    ensure_report_run_schema(conn)
    changed = conn.execute(
        "UPDATE report_runs SET status='failed',failed_ts=?,error_message=? WHERE run_id=? AND status='started'",
        (_utc_now(), str(error)[:4000], run_id),
    ).rowcount
    if changed != 1:
        raise ValueError(f"report run {run_id} is not in started state")
    conn.commit()


def _row_dict(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    columns = [str(item[1]) for item in conn.execute("PRAGMA table_info(report_runs)")]
    row = conn.execute("SELECT * FROM report_runs WHERE run_id=?", (run_id,)).fetchone()
    return dict(zip(columns, row, strict=True)) if row is not None else None


@dataclass(frozen=True)
class _VerifiedPublication:
    artifact: Path
    markdown: Path
    payload: dict[str, Any]
    json_bytes: bytes
    markdown_bytes: bytes
    row: dict[str, Any]


def _verify_artifacts(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    report_dir: Path,
) -> _VerifiedPublication:
    """Verify identity, chronology, both files, and the exact DB snapshot."""
    if row.get("status") not in {"completed", "published"}:
        raise ValueError("report run does not own completed evidence")
    started = _utc(row.get("started_ts"), field="started_ts")
    completed = _utc(row.get("completed_ts"), field="completed_ts")
    generated = _utc(row.get("generated_ts"), field="generated_ts")
    if generated < started or generated > completed:
        raise ValueError("report run timestamps are reversed")
    if row.get("status") == "published":
        published = _utc(row.get("published_ts"), field="published_ts")
        if published < completed or int(row.get("publication_verified") or 0) != 1:
            raise ValueError("report publication chronology is invalid")
        canonical = int(row.get("is_generation_canonical") or 0) == 1
        superseded_by = row.get("superseded_by_run_id")
        if canonical and superseded_by is not None:
            raise ValueError("canonical report cannot be superseded")
        if not canonical:
            seen = {str(row.get("run_id"))}
            while superseded_by is not None:
                if str(superseded_by) in seen:
                    raise ValueError("superseded report chain is cyclic")
                seen.add(str(superseded_by))
                child = conn.execute(
                    """SELECT status,publication_verified,generation_key,
                              is_generation_canonical,superseded_by_run_id
                       FROM report_runs WHERE run_id=?""",
                    (superseded_by,),
                ).fetchone()
                if child is None or child[:3] != (
                    "published", 1, row.get("generation_key")
                ):
                    raise ValueError("superseded report has an inconsistent successor")
                if int(child[3] or 0) == 1:
                    if child[4] is not None:
                        raise ValueError("canonical report cannot be superseded")
                    break
                superseded_by = child[4]
            else:
                raise ValueError("superseded report has no canonical successor")
    report_kind = str(row.get("report_kind") or "")
    if row.get("generation_key") != f"{report_kind}:{row.get('generated_ts')}":
        raise ValueError("report generation key is invalid")
    config_json = str(row.get("config_json") or "")
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise ValueError("report configuration is invalid") from exc
    if not isinstance(config, dict) or hashlib.sha256(config_json.encode()).hexdigest() != row.get("config_hash"):
        raise ValueError("report configuration hash mismatch")
    if not _GIT_SHA_RE.fullmatch(str(row.get("code_version") or "")):
        raise ValueError("report code version is invalid")

    report_dir = report_dir.resolve()
    artifact = Path(str(row.get("artifact_path") or ""))
    markdown = Path(str(row.get("markdown_path") or ""))
    stamp = generated.strftime("%Y%m%dT%H%M%SZ")
    expected_stem = f"daily_report_{stamp}_{row.get('run_id')}"
    if artifact.is_symlink() or artifact.resolve().parent != report_dir or artifact.name != f"{expected_stem}.json":
        raise ValueError("report artifact path does not match its run identity")
    if markdown.is_symlink() or markdown.resolve().parent != report_dir or markdown.name != f"{expected_stem}.md":
        raise ValueError("report markdown path does not match its run identity")
    try:
        json_bytes = artifact.read_bytes()
        markdown_bytes = markdown.read_bytes()
    except OSError as exc:
        raise ValueError("report artifact is unreadable") from exc
    for data, expected, label in (
        (json_bytes, row.get("content_hash"), "JSON"),
        (markdown_bytes, row.get("markdown_hash"), "Markdown"),
    ):
        actual = hashlib.sha256(data).hexdigest()
        if not _HASH_RE.fullmatch(str(expected or "")) or actual != expected:
            raise ValueError(f"report {label} artifact hash mismatch")

    payload = _load_artifact_bytes(json_bytes)
    _assert_publishable_report(payload)
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    run_meta = meta.get("report_run") if isinstance(meta.get("report_run"), dict) else {}
    if (payload.get("generated_ts") != row.get("generated_ts")
            or str(meta.get("report_kind") or "daily") != report_kind
            or str(run_meta.get("run_id") or "") != str(row.get("run_id") or "")):
        raise ValueError("report artifact identity does not match its registry row")
    universe, decisions = _decision_snapshot(payload)
    ranked = [{"ticker": item["ticker"], "selected_rank": item["selected_rank"]} for item in decisions]
    inputs = {
        "report_kind": report_kind,
        "market_session": payload.get("market_session") or {},
        "counts": payload.get("counts") or {},
        "risk_filters": payload.get("risk_filters") or {},
        "price_basis": meta.get("price_basis"),
    }
    latest_data = payload.get("latest_data") if isinstance(payload.get("latest_data"), dict) else {}
    for column, expected in {
        "scanned_universe_json": universe,
        "ranked_candidates_json": ranked,
        "decisions_json": decisions,
        "inputs_json": inputs,
        "source_timestamps_json": latest_data,
    }.items():
        try:
            stored = json.loads(str(row.get(column) or ""))
        except json.JSONDecodeError as exc:
            raise ValueError(f"stored {column} is invalid") from exc
        if stored != expected:
            raise ValueError("report artifact and stored decision snapshot differ")
    stored_decisions = [
        {"ticker": item[0], "selected_rank": item[1], "decision": item[2],
         "reason_codes": json.loads(item[3]), "inputs": json.loads(item[4])}
        for item in conn.execute(
            """SELECT ticker,selected_rank,decision,reason_codes_json,inputs_json
               FROM report_run_decisions WHERE run_id=? ORDER BY selected_rank,ticker""",
            (row.get("run_id"),),
        )
    ]
    if stored_decisions != decisions:
        raise ValueError("report artifact and stored decision rows differ")
    return _VerifiedPublication(
        artifact=artifact,
        markdown=markdown,
        payload=payload,
        json_bytes=json_bytes,
        markdown_bytes=markdown_bytes,
        row=row,
    )


def _publication_meta(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row.get("run_id"), "state": "published",
        "started_ts": row.get("started_ts"), "completed_ts": row.get("completed_ts"),
        "published_ts": row.get("published_ts"), "generated_ts": row.get("generated_ts"),
        "content_hash": row.get("content_hash"),
        "markdown_hash": row.get("markdown_hash"), "config_hash": row.get("config_hash"),
        "code_version": row.get("code_version"), "generation_key": row.get("generation_key"),
        "canonical_generation": bool(row.get("is_generation_canonical")),
        "superseded_by_run_id": row.get("superseded_by_run_id"),
        "publication_verified": True, "lineage": "linked",
    }


def resolve_published_report(
    conn: sqlite3.Connection,
    *,
    report_dir: Path,
    run_id: str | None = None,
    generated_ts: str | None = None,
    require_current: bool = False,
) -> tuple[Path, dict[str, Any]] | None:
    """Resolve one verified publication; exact selectors never fall back."""
    required = {
        "run_id", "status", "generated_ts", "publication_verified",
        "is_generation_canonical", "content_hash", "markdown_hash",
    }
    columns = [str(item[1]) for item in conn.execute("PRAGMA table_info(report_runs)")]
    if not required.issubset(columns) or conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='report_run_decisions'"
    ).fetchone() is None:
        return None
    clauses = ["status='published'", "publication_verified=1"]
    params: list[Any] = []
    if run_id is not None:
        clauses.append("run_id=?")
        params.append(str(run_id))
    if generated_ts is not None:
        try:
            exact = _utc(generated_ts, field="generated_ts").strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None
        clauses.extend(["generated_ts=?", "is_generation_canonical=1"])
        params.append(exact)
    if require_current:
        clauses.append("is_generation_canonical=1")
        clauses.append(
            """NOT EXISTS (
                SELECT 1 FROM report_runs newer
                WHERE newer.status='published' AND newer.publication_verified=1
                  AND (newer.published_ts>report_runs.published_ts OR
                       (newer.published_ts=report_runs.published_ts AND newer.run_id>report_runs.run_id))
            )"""
        )
    row = conn.execute(
        f"SELECT * FROM report_runs WHERE {' AND '.join(clauses)} "
        "ORDER BY published_ts DESC,run_id DESC LIMIT 1",
        params,
    ).fetchone()
    if row is None:
        return None
    record = dict(zip(columns, row, strict=True))
    verified = _verify_artifacts(conn, record, report_dir)
    linked = dict(verified.payload)
    linked["report_run"] = _publication_meta(record)
    return verified.artifact, linked


def verified_report_run_ids(
    conn: sqlite3.Connection,
    run_ids: list[str] | set[str] | tuple[str, ...],
) -> set[str]:
    """Return only run IDs whose exact historical artifacts still verify."""
    verified: set[str] = set()
    for run_id in sorted({str(value) for value in run_ids if str(value).strip()}):
        row = conn.execute(
            "SELECT artifact_path FROM report_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if row is None or not str(row[0] or "").strip():
            continue
        try:
            resolved = resolve_published_report(
                conn,
                report_dir=Path(str(row[0])).parent,
                run_id=run_id,
            )
        except ValueError:
            continue
        if resolved is not None:
            verified.add(run_id)
    return verified


def _resolve_current_publication(
    conn: sqlite3.Connection,
    *,
    report_dir: Path,
) -> _VerifiedPublication | None:
    columns = [str(item[1]) for item in conn.execute("PRAGMA table_info(report_runs)")]
    if "publication_verified" not in columns:
        return None
    row = conn.execute(
        """SELECT * FROM report_runs
           WHERE status='published' AND publication_verified=1
           ORDER BY published_ts DESC,run_id DESC LIMIT 1"""
    ).fetchone()
    if row is None:
        return None
    return _verify_artifacts(
        conn,
        dict(zip(columns, row, strict=True)),
        report_dir,
    )


def _atomic_write(path: Path, data: bytes) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def _atomic_write_if_changed(path: Path, data: bytes) -> None:
    try:
        if path.read_bytes() == data:
            return
    except OSError:
        pass
    _atomic_write(path, data)


@contextmanager
def _publication_lock(report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / PUBLICATION_LOCK).open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _reconcile_report_publication_locked(
    conn: sqlite3.Connection,
    *,
    report_dir: Path,
) -> dict[str, Any] | None:
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='report_runs'").fetchone() is None:
        return None
    verified = _resolve_current_publication(conn, report_dir=report_dir)
    if verified is None:
        return None
    meta = _publication_meta(verified.row)
    manifest = {
        **meta,
        "artifact_file": verified.artifact.name,
        "markdown_file": verified.markdown.name,
    }
    _atomic_write_if_changed(report_dir / LATEST_MANIFEST, (_canonical_json(manifest) + "\n").encode())
    _atomic_write_if_changed(report_dir / "daily_report_latest.json", verified.json_bytes)
    _atomic_write_if_changed(report_dir / "daily_report_latest.md", verified.markdown_bytes)
    return manifest


def reconcile_report_publication(conn: sqlite3.Connection, *, report_dir: Path) -> dict[str, Any] | None:
    """Serialize recovery writes with publication and copy only verified bytes."""
    with _publication_lock(report_dir):
        return _reconcile_report_publication_locked(conn, report_dir=report_dir)


def publish_report_run(conn: sqlite3.Connection, *, run_id: str, report_dir: Path) -> dict[str, Any]:
    """Verify one completed run, register ownership, then refresh compatibility files."""
    with _publication_lock(report_dir):
        ensure_report_run_schema(conn)
        row = _row_dict(conn, run_id)
        if row is None or row.get("status") not in {"completed", "published"}:
            raise ValueError(f"report run {run_id} is not completed")
        _verify_artifacts(conn, row, report_dir)
        if row.get("status") == "published" and not int(row.get("is_generation_canonical") or 0):
            raise ValueError(f"report run {run_id} was superseded by a canonical retry")
        if row.get("status") == "completed":
            conn.execute("BEGIN IMMEDIATE")
            try:
                published_ts = _utc_now()
                prior = conn.execute("SELECT MAX(published_ts) FROM report_runs WHERE published_ts IS NOT NULL").fetchone()[0]
                if prior and published_ts <= prior:
                    published_ts = (_utc(prior, field="published_ts") + dt.timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
                changed = conn.execute(
                    """UPDATE report_runs SET status='published',published_ts=?,publication_verified=1
                       WHERE run_id=? AND status='completed'""",
                    (published_ts, run_id),
                ).rowcount
                if changed != 1:
                    raise RuntimeError(f"report run {run_id} publication race")
                conn.execute(
                    """UPDATE report_runs SET is_generation_canonical=0,superseded_by_run_id=?
                       WHERE generation_key=(SELECT generation_key FROM report_runs WHERE run_id=?)
                         AND is_generation_canonical=1 AND run_id!=?""",
                    (run_id, run_id, run_id),
                )
                conn.execute(
                    "UPDATE report_runs SET is_generation_canonical=1 WHERE run_id=? AND is_generation_canonical=0",
                    (run_id,),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        manifest = _reconcile_report_publication_locked(conn, report_dir=report_dir)
        if manifest is None:
            raise RuntimeError("published report could not be resolved")
        return manifest if manifest.get("run_id") == run_id else _publication_meta(_row_dict(conn, run_id) or {})


def _record_admission_attempt(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    status: str,
    error: Exception | None = None,
    error_code: str | None = None,
) -> None:
    if status == "failed" and error_code not in ADMISSION_ERROR_CODES:
        raise ValueError("failed admission attempt requires a recognized error code")
    conn.execute(
        """INSERT INTO report_admission_attempts
           (run_id,status,error_code,error_message,attempted_ts)
           VALUES (?,?,?,?,?)""",
        (
            run_id,
            status,
            error_code if error else None,
            (
                error.code
                if isinstance(error, PriceContractError)
                else type(error).__name__
            ) if error else None,
            _utc_now(),
        ),
    )


def admit_published_report(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    report_dir: Path,
) -> dict[str, int]:
    """Admit only the current verified artifact's immutable accepted decisions."""
    if conn.in_transaction:
        raise RuntimeError("report admission requires a clean transaction boundary")
    del report_dir
    from trader_koo.paper_trades import PAPER_TRADE_ENABLED, create_paper_trades_from_report
    from trader_koo.paper_trade.schema import ensure_paper_trade_schema
    from trader_koo.paper_trade.trading import _require_published_canonical_report
    from trader_koo.report.setup_scoring import (
        SETUP_EVAL_ENABLED, _persist_setup_call_candidates, ensure_setup_call_eval_schema,
    )

    ensure_report_run_schema(conn)
    ensure_setup_call_eval_schema(conn)
    ensure_paper_trade_schema(conn)
    calls = trades = 0
    failure_code = "admission_setup_persistence_failed"
    try:
        conn.execute("BEGIN IMMEDIATE")
        lineage = _require_published_canonical_report(conn, run_id)
        # Artifact-derived preparation belongs to setup persistence. Once
        # lineage is valid, every non-lineage failure has a documented phase.
        failure_code = "admission_setup_persistence_failed"
        resolved = lineage["resolved"]
        row = conn.execute(
            "SELECT generated_ts,report_kind,source_timestamps_json "
            "FROM report_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        setups = [
            json.loads(item[0]) for item in conn.execute(
                "SELECT inputs_json FROM report_run_decisions "
                "WHERE run_id=? AND decision='accepted' "
                "ORDER BY selected_rank,ticker",
                (run_id,),
            )
        ]
        source_timestamps = json.loads(str(row[2] or "{}"))
        asof_date = str(source_timestamps.get("price_date") or "").strip()
        linked_payload = resolved[1]
        linked_meta = (
            linked_payload.get("meta")
            if isinstance(linked_payload.get("meta"), dict) else {}
        )
        price_basis = (
            linked_meta.get("price_basis")
            if isinstance(linked_meta.get("price_basis"), dict) else None
        )
        if setups and price_basis is None:
            raise PriceContractError(
                "price_contract_missing",
                "canonical report price contract is missing",
            )
        expected_price_contract = price_basis
        if SETUP_EVAL_ENABLED and asof_date and setups:
            calls = _persist_setup_call_candidates(
                conn, generated_ts=str(row[0]), report_kind=str(row[1]), asof_date=asof_date,
                setup_rows=setups, report_run_id=run_id,
                expected_price_contract=expected_price_contract,
            )
        failure_code = "admission_paper_trade_persistence_failed"
        if PAPER_TRADE_ENABLED and asof_date and setups:
            trades = create_paper_trades_from_report(
                conn, setup_rows=setups, report_date=asof_date, generated_ts=str(row[0]),
                report_run_id=run_id, schema_ready=True,
                expected_price_contract=expected_price_contract,
            )
        failure_code = "admission_finalize_failed"
        _record_admission_attempt(conn, run_id=run_id, status="succeeded")
        conn.commit()
    except Exception as exc:
        conn.rollback()
        try:
            _record_admission_attempt(
                conn,
                run_id=run_id,
                status="failed",
                error=exc,
                error_code=(exc.code if isinstance(exc, ReportLineageError) else failure_code),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            LOG.exception(
                "Could not persist failed admission attempt for report run %s",
                run_id,
            )
        raise
    return {"setup_calls": int(calls), "paper_trades": int(trades)}
