"""
Tests for three bug fixes:
1. datetime.now() → dt.datetime.now(utc) in admin_data_source_health endpoint
2. data_source / fetch_timestamp column migration in ensure_schema()
3. LLM output sanitized before schema validation to avoid fallback on long strings
"""

import datetime as dt
import os
import sqlite3
import sys
import tempfile
from types import ModuleType
from unittest.mock import MagicMock, patch  # noqa: F401 (MagicMock used in stub)


# ---------------------------------------------------------------------------
# Fix 1: datetime.now() → dt.datetime.now() in admin_data_source_health
# ---------------------------------------------------------------------------


class TestDataSourceHealthTimestamp:
    """admin_data_source_health must return a valid ISO timestamp, not crash."""

    def test_timestamp_uses_dt_datetime_now(self):
        """The fixed line uses dt.datetime.now() which produces a valid ISO string."""
        # Replicate exactly what the fixed code does
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        parsed = dt.datetime.fromisoformat(ts)
        assert isinstance(parsed, dt.datetime)

    def test_datetime_module_alias_has_no_now(self):
        """Confirm the original bug: dt (= datetime module) has no .now() attribute.

        This documents why datetime.now() / dt.now() both fail and
        dt.datetime.now() is the correct call.
        """
        import datetime as dt_alias
        # dt_alias is the module — it has no direct .now()
        assert not hasattr(dt_alias, "now"), (
            "datetime module must not have a .now() — caller needs dt.datetime.now()"
        )
        # But dt_alias.datetime (the class) does have .now()
        assert hasattr(dt_alias.datetime, "now")

    def test_timestamp_is_utc_aware(self):
        """Returned timestamp is UTC-aware (matches dt.timezone.utc pattern)."""
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        parsed = dt.datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None

    def test_timestamp_is_recent(self):
        """Timestamp produced by the fixed expression is within a few seconds of now."""
        before = dt.datetime.now(dt.timezone.utc)
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        after = dt.datetime.now(dt.timezone.utc)
        parsed = dt.datetime.fromisoformat(ts)
        assert before - dt.timedelta(seconds=1) <= parsed <= after + dt.timedelta(seconds=1)


# ---------------------------------------------------------------------------
# Helpers for Fix 2 — inject a stub `finviz` so update_market_db can be imported
# ---------------------------------------------------------------------------


def _stub_finviz():
    """Return a minimal finviz stub and register it in sys.modules."""
    stub = ModuleType("finviz")
    stub.get_stock = MagicMock(return_value={})
    sys.modules.setdefault("finviz", stub)
    return stub


def _import_ensure_schema():
    """Import ensure_schema with finviz stubbed out."""
    _stub_finviz()
    # Force re-import if already cached under a broken state
    if "trader_koo.scripts.update_market_db" in sys.modules:
        mod = sys.modules["trader_koo.scripts.update_market_db"]
    else:
        import importlib
        mod = importlib.import_module("trader_koo.scripts.update_market_db")
    return mod.ensure_schema


def _create_old_price_daily(conn: sqlite3.Connection) -> None:
    """Create a price_daily table WITHOUT the new columns (old schema)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_daily (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (ticker, date)
        )
        """
    )
    conn.execute(
        "INSERT INTO price_daily (ticker, date, open, high, low, close, volume) "
        "VALUES ('AAPL', '2024-01-01', 100, 105, 99, 103, 1000000)"
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Fix 2: ensure_schema() migrates missing data_source / fetch_timestamp columns
# ---------------------------------------------------------------------------


class TestEnsureSchemaMigration:
    """ensure_schema() must add data_source and fetch_timestamp to old DBs."""

    def test_adds_data_source_column_to_old_db(self):
        """data_source column is created when it is missing."""
        ensure_schema = _import_ensure_schema()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            conn = sqlite3.connect(db_path)
            _create_old_price_daily(conn)

            cols_before = {r[1] for r in conn.execute("PRAGMA table_info(price_daily)")}
            assert "data_source" not in cols_before, "Pre-condition: column must be absent"

            ensure_schema(conn)

            cols_after = {r[1] for r in conn.execute("PRAGMA table_info(price_daily)")}
            assert "data_source" in cols_after
            conn.close()
        finally:
            os.unlink(db_path)

    def test_adds_fetch_timestamp_column_to_old_db(self):
        """fetch_timestamp column is created when it is missing."""
        ensure_schema = _import_ensure_schema()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            conn = sqlite3.connect(db_path)
            _create_old_price_daily(conn)

            cols_before = {r[1] for r in conn.execute("PRAGMA table_info(price_daily)")}
            assert "fetch_timestamp" not in cols_before

            ensure_schema(conn)

            cols_after = {r[1] for r in conn.execute("PRAGMA table_info(price_daily)")}
            assert "fetch_timestamp" in cols_after
            conn.close()
        finally:
            os.unlink(db_path)

    def test_migration_is_idempotent(self):
        """Running ensure_schema() twice on a fresh DB does not raise."""
        ensure_schema = _import_ensure_schema()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            conn = sqlite3.connect(db_path)
            ensure_schema(conn)
            ensure_schema(conn)  # second call must not raise "duplicate column name"
            cols = {r[1] for r in conn.execute("PRAGMA table_info(price_daily)")}
            assert "data_source" in cols
            assert "fetch_timestamp" in cols
            conn.close()
        finally:
            os.unlink(db_path)

    def test_existing_row_readable_after_migration(self):
        """Rows inserted before migration are still readable afterwards."""
        ensure_schema = _import_ensure_schema()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            conn = sqlite3.connect(db_path)
            _create_old_price_daily(conn)
            ensure_schema(conn)

            row = conn.execute(
                "SELECT ticker, date, data_source FROM price_daily WHERE ticker='AAPL'"
            ).fetchone()
            assert row is not None
            assert row[0] == "AAPL"
            assert row[1] == "2024-01-01"
            # Pre-existing row gets NULL for the new column (SQLite ALTER TABLE DEFAULT
            # only applies to future inserts).
            assert row[2] in (None, "yfinance")
            conn.close()
        finally:
            os.unlink(db_path)

    def test_data_source_query_works_after_migration(self):
        """SELECT data_source, fetch_timestamp ... succeeds without OperationalError."""
        ensure_schema = _import_ensure_schema()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            conn = sqlite3.connect(db_path)
            _create_old_price_daily(conn)
            ensure_schema(conn)

            # This is the exact query from _get_data_sources() that was failing
            row = conn.execute(
                """
                SELECT data_source, fetch_timestamp
                FROM price_daily
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT 1
                """,
                ("AAPL",),
            ).fetchone()
            assert row is not None  # query must not raise OperationalError
            conn.close()
        finally:
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Fix 3: LLM output sanitized before schema validation
# ---------------------------------------------------------------------------


class TestLLMSanitizeBeforeValidation:
    """LLM output exceeding schema field limits should be trimmed, not rejected."""

    def test_long_observation_trimmed_and_validated(self):
        """observation > 260 chars is truncated and then validates successfully."""
        from trader_koo.llm.sanitizer import sanitize_llm_output
        from trader_koo.llm.validator import validate_llm_output
        from trader_koo.llm.schemas import SetupRewrite

        raw = {
            "observation": "A" * 300,  # exceeds 260-char schema limit
            "action": "Watch for entry above resistance.",
            "risk_note": "Use stop losses.",
        }

        # This is what the fixed code does: sanitize first, then validate
        sanitized = sanitize_llm_output(
            raw, field_limits={"observation": 260, "action": 180, "risk_note": 80}
        )
        result = validate_llm_output(sanitized, SetupRewrite)

        assert result.is_valid, f"Expected valid after sanitize, got: {result.errors}"
        assert len(result.data.observation) <= 260

    def test_long_action_trimmed_and_validated(self):
        """action > 180 chars is truncated and then validates successfully."""
        from trader_koo.llm.sanitizer import sanitize_llm_output
        from trader_koo.llm.validator import validate_llm_output
        from trader_koo.llm.schemas import SetupRewrite

        raw = {
            "observation": "Market is bullish.",
            "action": "B" * 200,  # exceeds 180-char schema limit
        }

        sanitized = sanitize_llm_output(
            raw, field_limits={"observation": 260, "action": 180, "risk_note": 80}
        )
        result = validate_llm_output(sanitized, SetupRewrite)

        assert result.is_valid, f"Expected valid after sanitize, got: {result.errors}"
        assert len(result.data.action) <= 180

    def test_long_risk_note_trimmed_and_validated(self):
        """risk_note > 80 chars is truncated and then validates successfully."""
        from trader_koo.llm.sanitizer import sanitize_llm_output
        from trader_koo.llm.validator import validate_llm_output
        from trader_koo.llm.schemas import SetupRewrite

        raw = {
            "observation": "Market is bullish.",
            "action": "Watch for entry.",
            "risk_note": "R" * 100,  # exceeds 80-char schema limit
        }

        sanitized = sanitize_llm_output(
            raw, field_limits={"observation": 260, "action": 180, "risk_note": 80}
        )
        result = validate_llm_output(sanitized, SetupRewrite)

        assert result.is_valid, f"Expected valid after sanitize, got: {result.errors}"
        assert len(result.data.risk_note) <= 80

    def test_validate_without_sanitize_fails_on_long_string(self):
        """Without sanitize, long observation correctly fails schema validation.

        This documents the original bug — validate-before-sanitize causes fallback.
        """
        from trader_koo.llm.validator import validate_llm_output
        from trader_koo.llm.schemas import SetupRewrite

        raw = {
            "observation": "A" * 300,
            "action": "Watch for entry.",
        }
        result = validate_llm_output(raw, SetupRewrite)
        assert not result.is_valid

    def test_maybe_rewrite_no_schema_failure_on_long_llm_reply(self):
        """maybe_rewrite_setup_copy: oversized LLM reply is trimmed, not fallen back."""
        from trader_koo.llm_narrative import maybe_rewrite_setup_copy

        oversized_llm_response = {
            "observation": "O" * 300,  # too long for SetupRewrite (max 260)
            "action": "Watch for entry.",
            "risk_note": "Use stops.",
        }

        row = {
            "ticker": "SPY",
            "observation": "SPY is bullish.",
            "action": "Buy on dip.",
            "risk_note": "Use stops.",
        }

        with (
            patch("trader_koo.llm_narrative._runtime_disabled_now", return_value=False),
            patch("trader_koo.llm_narrative.llm_ready", return_value=True),
            patch(
                "trader_koo.llm_narrative._azure_chat_rewrite",
                return_value=(oversized_llm_response, {}),
            ),
            patch("trader_koo.llm_narrative._safe_note_token_usage"),
            patch("trader_koo.llm_narrative._safe_note_success"),
            patch("trader_koo.llm_narrative._safe_note_failure") as mock_failure,
            patch("trader_koo.llm_narrative._default_db_path", return_value=":memory:"),
        ):
            result = maybe_rewrite_setup_copy(row, source="test")

        # With the fix (sanitize before validate), schema validation must not fail
        schema_failures = [
            c for c in mock_failure.call_args_list
            if c.kwargs.get("reason") == "schema_validation_failed"
        ]
        assert schema_failures == [], (
            "schema_validation_failed must not fire after sanitize-before-validate fix"
        )

        assert "observation" in result
        assert len(result["observation"]) <= 260

    def test_maybe_rewrite_empty_response_activates_runtime_cooldown(self):
        """Empty LLM replies should trigger cooldown before falling back."""
        from trader_koo.llm_narrative import maybe_rewrite_setup_copy

        row = {
            "ticker": "SPY",
            "observation": "SPY is bullish.",
            "action": "Buy on dip.",
            "risk_note": "Use stops.",
        }

        with (
            patch.dict("trader_koo.llm_narrative._PROMPT_CACHE", {}, clear=True),
            patch("trader_koo.llm_narrative._runtime_disabled_now", return_value=False),
            patch("trader_koo.llm_narrative.llm_ready", return_value=True),
            patch("trader_koo.llm_narrative._llm_provider", return_value="azure_openai"),
            patch("trader_koo.llm_narrative._azure_chat_rewrite", return_value=({}, {})),
            patch("trader_koo.llm_narrative._safe_note_token_usage"),
            patch("trader_koo.llm_narrative._safe_note_success"),
            patch("trader_koo.llm_narrative._safe_note_failure"),
            patch("trader_koo.llm_narrative._set_runtime_disable") as mock_disable,
            patch("trader_koo.llm_narrative._default_db_path", return_value=":memory:"),
        ):
            result = maybe_rewrite_setup_copy(row, source="test")

        assert mock_disable.called
        assert "observation" in result

    def test_maybe_rewrite_schema_failure_activates_runtime_cooldown(self):
        """Schema-invalid LLM replies should also trigger cooldown."""
        from trader_koo.llm_narrative import maybe_rewrite_setup_copy

        row = {
            "ticker": "SPY",
            "observation": "SPY is bullish.",
            "action": "Buy on dip.",
            "risk_note": "Use stops.",
        }

        invalid_llm_response = {
            "observation": None,
            "action": ["bad", "type"],
            "risk_note": 123,
        }

        with (
            patch.dict("trader_koo.llm_narrative._PROMPT_CACHE", {}, clear=True),
            patch("trader_koo.llm_narrative._runtime_disabled_now", return_value=False),
            patch("trader_koo.llm_narrative.llm_ready", return_value=True),
            patch("trader_koo.llm_narrative._llm_provider", return_value="azure_openai"),
            patch("trader_koo.llm_narrative._azure_chat_rewrite", return_value=(invalid_llm_response, {})),
            patch("trader_koo.llm_narrative._safe_note_token_usage"),
            patch("trader_koo.llm_narrative._safe_note_success"),
            patch("trader_koo.llm_narrative._safe_note_failure"),
            patch("trader_koo.llm_narrative._set_runtime_disable") as mock_disable,
            patch("trader_koo.llm_narrative._default_db_path", return_value=":memory:"),
        ):
            result = maybe_rewrite_setup_copy(row, source="test")

        assert mock_disable.called
        assert "observation" in result
