from __future__ import annotations

import datetime as dt
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trader_koo.backend.routers.admin import agents
from trader_koo.backend.routers.admin import router as admin_router
from trader_koo.llm import observability
from trader_koo.llm.observability import (
    observability_summary,
    observability_trace,
    record_llm_call,
    record_outcome_link,
)
from trader_koo.middleware.auth import AdminAuthConfig, AdminAuthenticator


def _record(tmp_path, *, status: str = "success", fallback: str | None = None):
    db_path = tmp_path / "observability.db"
    started = dt.datetime(2026, 8, 23, 1, 2, 3, tzinfo=dt.timezone.utc)
    identifiers = record_llm_call(
        db_path,
        source="chart_commentary", role="narrative_rewriter",
        stage="setup_copy_rewrite", provider="azure_openai",
        model="gpt-fixture", deployment="fixture-deployment",
        prompt_template_version="setup-rewrite-v1",
        input_payload={"ticker": "AAA", "api_key": "must-not-persist"},
        proposed_output={"observation": "proposed"},
        deterministic_pre={"observation": "before"},
        final_adjudicated={"observation": "after" if status == "success" else "before"},
        started_at=started,
        ended_at=started + dt.timedelta(milliseconds=125),
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        validator_result="passed" if status == "success" else "failed",
        fallback_reason=fallback, terminal_status=status, ticker="AAA",
    )
    return db_path, identifiers


def test_trace_is_complete_redacted_and_append_only(tmp_path) -> None:
    db_path, identifiers = _record(tmp_path)
    raw = db_path.read_bytes()

    assert b"must-not-persist" not in raw
    detail = observability_trace(db_path, identifiers["trace_id"])
    assert detail is not None
    assert detail["trace"]["provider"] == "azure_openai"
    assert detail["trace"]["model"] == "gpt-fixture"
    assert detail["trace"]["prompt_template_version"] == "setup-rewrite-v1"
    assert detail["trace"]["latency_ms"] == pytest.approx(125)
    assert detail["trace"]["decision_scope"] == "narrative_only"
    assert detail["trace"]["decision_changed"] == 1
    assert detail["run_graph"]["graph_kind"] == "single_llm_call"
    conn = sqlite3.connect(db_path)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(
            "UPDATE llm_call_traces SET terminal_status='forged' WHERE trace_id=?",
            (identifiers["trace_id"],),
        )
    conn.close()


def test_aggregate_reconciles_success_fallback_cost_and_legacy(tmp_path) -> None:
    db_path, _ = _record(tmp_path)
    _record(tmp_path, status="fallback", fallback="schema_validation_failed")
    summary = observability_summary(db_path)

    assert summary["aggregate"]["traces"] == 2
    assert summary["aggregate"]["success_rate_pct"] == 50
    assert summary["aggregate"]["fallback_rate_pct"] == 50
    assert summary["aggregate"]["validator_failures"] == 1
    assert summary["aggregate"]["total_tokens"] == 30
    assert summary["retention"]["credentials_stored"] is False
    assert summary["legacy_health_counters"]["label"] == "legacy"


def test_aggregate_whitelists_legacy_health_fields(tmp_path, monkeypatch) -> None:
    db_path, _ = _record(tmp_path)
    monkeypatch.setattr(observability, "llm_health_summary", lambda *_args, **_kwargs: {
        "degraded": True,
        "degraded_threshold": 3,
        "consecutive_failures": 1,
        "last_success_ts": None,
        "last_failure_ts": "2026-08-23T01:02:03+00:00",
        "counts": {"failure": 1},
        "last_error_details": "https://example.invalid?api_key=must-not-leak",
        "recent_events": [{"details": "must-not-leak"}],
    })

    summary = observability_summary(db_path)
    serialized = str(summary["legacy_health_counters"])

    assert "last_error_details" not in summary["legacy_health_counters"]
    assert "recent_events" not in summary["legacy_health_counters"]
    assert "must-not-leak" not in serialized


def test_outcome_link_is_explicitly_observational_and_non_causal(tmp_path) -> None:
    db_path, identifiers = _record(tmp_path)
    record_outcome_link(
        db_path, trace_id=identifiers["trace_id"], paper_trade_id=7,
        outcome={"exit_reason": "target_hit", "net_pnl": 12.5},
    )
    detail = observability_trace(db_path, identifiers["trace_id"])

    assert detail is not None
    assert detail["outcomes"][0]["paper_trade_id"] == 7
    assert detail["outcomes"][0]["analysis_label"] == "observational_non_causal"
    assert detail["causal_interpretation"] == "observational_non_causal"


def test_observability_api_is_authenticated_and_redacted(tmp_path, monkeypatch) -> None:
    db_path, _ = _record(tmp_path)
    monkeypatch.setattr(agents, "DB_PATH", db_path)
    app = FastAPI()
    app.state.admin_authenticator = AdminAuthenticator(
        AdminAuthConfig(api_key="x" * 32)
    )
    app.include_router(admin_router)
    client = TestClient(app)

    assert client.get("/api/admin/agent-observability").status_code == 401
    response = client.get(
        "/api/admin/agent-observability", headers={"X-API-Key": "x" * 32}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["aggregate"]["traces"] == 1
    serialized = response.text.lower()
    assert "api_key" not in serialized
    assert "must-not-persist" not in serialized


def test_real_narrative_call_records_trace_and_contribution(tmp_path, monkeypatch) -> None:
    from trader_koo import llm_narrative

    db_path = tmp_path / "narrative.db"
    monkeypatch.setattr(llm_narrative, "_default_db_path", lambda: db_path)
    monkeypatch.setattr(llm_narrative, "_runtime_disabled_now", lambda: False)
    monkeypatch.setattr(llm_narrative, "llm_ready", lambda: True)
    monkeypatch.setattr(llm_narrative, "_llm_provider", lambda: "azure_openai")
    monkeypatch.setattr(llm_narrative, "_azure_cfg", lambda: {
        "endpoint": "https://redacted.invalid", "api_key": "secret",
        "deployment": "fixture", "api_version": "fixture-v1",
    })
    monkeypatch.setattr(llm_narrative, "_azure_chat_rewrite", lambda _context: ({
        "observation": "Validated rewrite.",
        "action": "Original action.",
        "risk_note": "Original risk.",
        "intent": {
            "signal_bias": "unspecified",
            "actionability": "unspecified",
            "decision_delta": "none",
        },
    }, {
        "model": "gpt-fixture", "deployment": "fixture",
        "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
    }))
    monkeypatch.setattr(llm_narrative, "_safe_note_success", lambda *args, **kwargs: None)
    monkeypatch.setattr(llm_narrative, "_safe_note_token_usage", lambda *args, **kwargs: None)
    llm_narrative._PROMPT_CACHE.clear()

    result = llm_narrative.maybe_rewrite_setup_copy({
        "ticker": "AAA", "observation": "Original observation.",
        "action": "Original action.", "risk_note": "Original risk.",
    }, source="test")
    summary = observability_summary(db_path)

    assert result["observation"] == "Validated rewrite."
    assert summary["aggregate"]["traces"] == 1
    assert summary["traces"][0]["terminal_status"] == "success"
    assert summary["traces"][0]["decision_scope"] == "observation_narrative_only"
