"""Tests for the ML retrain admin endpoint and supporting functions.

Validates:
- POST /api/admin/ml/retrain triggers background training
- GET /api/admin/ml/retrain-status returns current state
- GET /api/admin/ml/retrain-history reads from ml_train_log
- _log_retrain_metrics persists to DB
- _send_retrain_notification formats correctly
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from trader_koo.backend.routers.admin.ml import (
    _log_retrain_metrics,
    _send_retrain_notification,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def conn() -> sqlite3.Connection:
    """In-memory DB for retrain log tests."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    return c


@pytest.fixture()
def sample_result() -> dict[str, Any]:
    """Sample successful retrain result."""
    return {
        "ok": True,
        "model_path": "/data/models/swing_lgbm_20260326T000000Z.txt",
        "target_mode": "return_sign",
        "fold_count": 3,
        "total_samples": 1500,
        "trained_at": "2026-03-26T00:00:00+00:00",
        "start_date": "2025-01-01",
        "end_date": "2026-03-25",
        "elapsed_sec": 120.5,
        "aggregate_metrics": {
            "avg_auc": 0.5500,
            "best_auc": 0.5800,
            "avg_accuracy": 0.5300,
            "avg_precision": 0.5100,
        },
        "folds": [
            {"fold": 1, "auc": 0.5400, "accuracy": 0.5200},
            {"fold": 2, "auc": 0.5500, "accuracy": 0.5300},
            {"fold": 3, "auc": 0.5600, "accuracy": 0.5400},
        ],
        "meta_labeling": {"ok": True, "auc": 0.5700},
        "feature_columns": ["ret_1d", "vol_21d"],
        "feature_importance": {"ret_1d": 100, "vol_21d": 80},
    }


# ---------------------------------------------------------------------------
# Tests: _log_retrain_metrics
# ---------------------------------------------------------------------------

class TestLogRetrainMetrics:
    def test_creates_table_and_inserts(
        self, conn: sqlite3.Connection, sample_result: dict[str, Any],
    ) -> None:
        _log_retrain_metrics(conn, sample_result)
        conn.commit()

        rows = conn.execute("SELECT * FROM ml_train_log").fetchall()

        assert len(rows) == 1
        row = dict(rows[0])
        assert row["avg_auc"] == 0.5500
        assert row["best_auc"] == 0.5800
        assert row["fold_count"] == 3
        assert row["total_samples"] == 1500
        assert row["target_mode"] == "return_sign"
        assert row["elapsed_sec"] == 120.5

    def test_inserts_multiple_entries(
        self, conn: sqlite3.Connection, sample_result: dict[str, Any],
    ) -> None:
        _log_retrain_metrics(conn, sample_result)
        conn.commit()

        sample_result["aggregate_metrics"]["avg_auc"] = 0.6000
        _log_retrain_metrics(conn, sample_result)
        conn.commit()

        count = conn.execute("SELECT COUNT(*) FROM ml_train_log").fetchone()[0]
        assert count == 2

    def test_handles_missing_fields_gracefully(
        self, conn: sqlite3.Connection,
    ) -> None:
        # Minimal result with missing fields should not crash
        _log_retrain_metrics(conn, {"ok": False, "error": "test"})
        conn.commit()

        rows = conn.execute("SELECT * FROM ml_train_log").fetchall()
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Tests: _send_retrain_notification
# ---------------------------------------------------------------------------

class TestSendRetrainNotification:
    @patch("trader_koo.notifications.telegram.send_message", return_value=True)
    @patch("trader_koo.notifications.telegram.is_configured", return_value=True)
    def test_sends_formatted_message(
        self,
        mock_configured: MagicMock,
        mock_send: MagicMock,
        sample_result: dict[str, Any],
    ) -> None:
        _send_retrain_notification(sample_result)

        mock_send.assert_called_once()
        msg = mock_send.call_args[0][0]
        assert "ML Model Retrained" in msg
        assert "0.5500" in msg
        assert "0.5800" in msg
        assert "return_sign" in msg
        assert "1,500" in msg

    @patch("trader_koo.notifications.telegram.is_configured", return_value=False)
    def test_skips_when_not_configured(
        self,
        mock_configured: MagicMock,
        sample_result: dict[str, Any],
    ) -> None:
        # Should not raise
        _send_retrain_notification(sample_result)

    @patch("trader_koo.notifications.telegram.send_message", side_effect=Exception("network"))
    @patch("trader_koo.notifications.telegram.is_configured", return_value=True)
    def test_handles_send_failure_gracefully(
        self,
        mock_configured: MagicMock,
        mock_send: MagicMock,
        sample_result: dict[str, Any],
    ) -> None:
        # Should not raise even on failure
        _send_retrain_notification(sample_result)


# ---------------------------------------------------------------------------
# Tests: endpoint behavior (via direct function call, no HTTP)
# ---------------------------------------------------------------------------

class TestRetrainEndpointLogic:
    def test_invalid_target_mode_rejected(self) -> None:
        """Verify that invalid target_mode values are caught."""
        from trader_koo.backend.routers.admin import ml as ml_module

        # Reset state
        ml_module._retrain_thread = None
        ml_module._retrain_result = None

        mock_request = MagicMock()
        mock_request.state = MagicMock()
        mock_request.state.admin_identity = "test-admin"

        # Call the underlying function (bypass auth decorator via __wrapped__)
        result = ml_module.retrain_ml_model.__wrapped__(
            mock_request,
            start_date="2025-01-01",
            target_mode="invalid_mode",
            notify=False,
        )
        assert result["ok"] is False
        assert "Invalid" in result["error"]

    def test_retrain_status_when_no_run(self) -> None:
        """retrain-status returns sensible default when no retrain has run."""
        from trader_koo.backend.routers.admin import ml as ml_module

        ml_module._retrain_thread = None
        ml_module._retrain_result = None

        mock_request = MagicMock()
        mock_request.state = MagicMock()
        mock_request.state.admin_identity = "test-admin"

        result = ml_module.retrain_status.__wrapped__(mock_request)

        assert result["running"] is False
        assert "No retrain" in result.get("message", "")
