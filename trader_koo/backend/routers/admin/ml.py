"""ML training, scoring, SHAP analysis, drift detection, backtest, paper trades."""
from __future__ import annotations

import datetime as dt
import threading
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from trader_koo.backend.services.database import get_conn
from trader_koo.middleware.auth import require_admin_auth
from trader_koo.paper_trades import manually_close_trade, mark_to_market

from trader_koo.backend.routers.admin._shared import LOG

router = APIRouter(tags=["admin", "admin-ml"])

_ml_train_thread: threading.Thread | None = None
_ml_train_result: dict[str, Any] | None = None

_backtest_thread: threading.Thread | None = None
_backtest_result: dict[str, Any] | None = None


@router.post("/api/admin/train-ml-model")
@require_admin_auth
def train_ml_model(
    request: Request,
    start_date: str = Query(default="2025-06-01"),
    end_date: str = Query(default=None),
    target_mode: str = Query(default="return_sign"),
) -> dict[str, Any]:
    """Train the swing-trade LightGBM model in a background thread.

    The model trains asynchronously because it can take several minutes.
    Use GET /api/admin/ml-model-status to check progress and results.

    target_mode: "return_sign" (default), "barrier", or "rank".
    """
    global _ml_train_thread, _ml_train_result

    if target_mode not in {"return_sign", "barrier", "rank"}:
        return {
            "ok": False,
            "error": (
                f"Invalid target_mode '{target_mode}'. "
                "Choose from: 'return_sign', 'barrier', 'rank'."
            ),
        }

    if _ml_train_thread and _ml_train_thread.is_alive():
        return {
            "ok": False,
            "message": (
                "Training already in progress "
                "— check /api/admin/ml-model-status"
            ),
        }

    def _run_training() -> None:
        global _ml_train_result
        try:
            from trader_koo.ml.trainer import train_walk_forward

            conn = get_conn()
            try:
                _ml_train_result = train_walk_forward(
                    conn,
                    start_date=start_date,
                    end_date=end_date,
                    target_mode=target_mode,
                )
            finally:
                conn.close()
        except Exception as exc:
            LOG.exception("ML model training failed: %s", exc)
            _ml_train_result = {"ok": False, "error": str(exc)}

    _ml_train_result = {
        "ok": False,
        "status": "training",
        "message": "Training started...",
    }
    _ml_train_thread = threading.Thread(target=_run_training, daemon=True)
    _ml_train_thread.start()

    return {
        "ok": True,
        "message": (
            f"Training started in background "
            f"(start_date={start_date}, target_mode={target_mode}). "
            "Check /api/admin/ml-model-status for results."
        ),
        "start_date": start_date,
        "end_date": end_date,
        "target_mode": target_mode,
    }


@router.get("/api/admin/ml-model-status")
@require_admin_auth
def ml_model_status(request: Request) -> dict[str, Any]:
    """Return the current ML model status, training progress, and metrics."""
    training_active = bool(_ml_train_thread and _ml_train_thread.is_alive())
    try:
        from trader_koo.ml.scorer import model_status

        status = model_status()
        status["training_active"] = training_active
        if _ml_train_result:
            status["last_train_result"] = _ml_train_result
        return status
    except Exception as exc:
        return {
            "loaded": False,
            "training_active": training_active,
            "error": str(exc),
        }


@router.get("/api/admin/ml-score-universe")
@require_admin_auth
def ml_score_universe(
    request: Request,
    date: str = Query(default=None),
    top_n: int = Query(default=20),
) -> dict[str, Any]:
    """Score the full universe and return top N tickers by predicted win probability."""
    try:
        from trader_koo.ml.scorer import score_universe

        if date is None:
            date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        conn = get_conn()
        try:
            scores = score_universe(conn, as_of_date=date, top_n=top_n)
            return {"ok": True, "date": date, "top_n": top_n, "scores": scores}
        finally:
            conn.close()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@router.get("/api/admin/macro-snapshot")
@require_admin_auth
def macro_snapshot(request: Request) -> dict[str, Any]:
    """Return current macro data snapshot (FRED + Polymarket)."""
    try:
        from trader_koo.ml.external_data import get_macro_snapshot

        return {"ok": True, **get_macro_snapshot()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@router.get("/api/admin/ml-shap-analysis")
@require_admin_auth
def ml_shap_analysis(request: Request) -> dict[str, Any]:
    """Run SHAP analysis on the current model to explain feature importance."""
    try:
        from trader_koo.ml.scorer import load_model
        from trader_koo.ml.features import (
            FEATURE_COLUMNS,
            extract_features_for_universe,
        )
        from trader_koo.ml.shap_analysis import compute_shap_summary

        model, meta = load_model()
        if model is None:
            return {"ok": False, "error": "No model loaded"}

        feature_cols = (
            meta.get("feature_columns", FEATURE_COLUMNS)
            if meta
            else FEATURE_COLUMNS
        )
        conn = get_conn()
        try:
            today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
            features = extract_features_for_universe(conn, as_of_date=today)
            if features.empty:
                return {"ok": False, "error": "No feature data available"}
            X = features.reindex(columns=feature_cols)
            result = compute_shap_summary(model, X)
            return {"ok": True, **result}
        finally:
            conn.close()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@router.get("/api/admin/ml-drift-check")
@require_admin_auth
def ml_drift_check(
    request: Request,
    window_days: int = Query(default=30),
) -> dict[str, Any]:
    """Check model drift -- is the model still accurate on recent trades?"""
    try:
        from trader_koo.ml.drift_detection import check_model_drift

        conn = get_conn()
        try:
            return check_model_drift(conn, window_days=window_days)
        finally:
            conn.close()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@router.post("/api/admin/run-backtest")
@require_admin_auth
def run_backtest_endpoint(
    request: Request,
    start_date: str = Query(default="2025-06-01"),
    end_date: str = Query(default=None),
    max_positions: int = Query(default=5),
    min_win_prob: float = Query(default=0.55),
    short_threshold: float = Query(default=0.45),
) -> dict[str, Any]:
    """Run a walk-forward backtest in background. Check /api/admin/backtest-result."""
    global _backtest_thread, _backtest_result

    if _backtest_thread and _backtest_thread.is_alive():
        return {
            "ok": False,
            "message": (
                "Backtest already running "
                "— check /api/admin/backtest-result"
            ),
        }

    def _run() -> None:
        global _backtest_result
        try:
            from trader_koo.ml.backtest import run_backtest

            conn = get_conn()
            try:
                _backtest_result = run_backtest(
                    conn,
                    start_date=start_date,
                    end_date=end_date,
                    max_positions=max_positions,
                    min_win_prob=min_win_prob,
                    short_threshold=short_threshold,
                )
            finally:
                conn.close()
        except Exception as exc:
            LOG.exception("Backtest failed: %s", exc)
            _backtest_result = {"ok": False, "error": str(exc)}

    _backtest_result = {
        "ok": False,
        "status": "running",
        "message": "Backtest started...",
    }
    _backtest_thread = threading.Thread(target=_run, daemon=True)
    _backtest_thread.start()

    return {
        "ok": True,
        "message": (
            "Backtest started in background. "
            "Check /api/admin/backtest-result."
        ),
    }


@router.get("/api/admin/backtest-result")
@require_admin_auth
def get_backtest_result(request: Request) -> dict[str, Any]:
    """Return the latest backtest result."""
    running = bool(_backtest_thread and _backtest_thread.is_alive())
    if _backtest_result:
        return {**_backtest_result, "running": running}
    return {
        "ok": False,
        "running": running,
        "message": "No backtest has been run yet",
    }


# ── Paper trade admin ────────────────────────────────────


@router.post("/api/admin/paper-trades/close")
@require_admin_auth
def admin_close_paper_trade(
    request: Request,
    trade_id: int = Query(..., ge=1),
    exit_price: float | None = Query(default=None),
    exit_reason: str = Query(default="manual_close"),
) -> dict[str, Any]:
    """Manually close an open paper trade."""
    conn = get_conn()
    try:
        result = manually_close_trade(
            conn,
            trade_id=trade_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
        )
        return {"ok": True, **result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        conn.close()


@router.post("/api/admin/paper-trades/mtm")
@require_admin_auth
def admin_trigger_mtm(request: Request) -> dict[str, Any]:
    """Trigger mark-to-market on all open paper trades."""
    conn = get_conn()
    try:
        result = mark_to_market(conn)
        conn.commit()
        return {"ok": True, **result}
    finally:
        conn.close()
