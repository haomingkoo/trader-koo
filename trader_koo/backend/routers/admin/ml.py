"""ML training, scoring, SHAP analysis, drift detection, backtest, paper trades."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from trader_koo.backend.services.database import DB_PATH, get_conn
from trader_koo.middleware.auth import require_admin_auth
from trader_koo.paper_trades import manually_close_trade, mark_to_market

from trader_koo.backend.routers.admin._shared import LOG

router = APIRouter(tags=["admin", "admin-ml"])

_ml_train_thread: threading.Thread | None = None
_ml_train_result: dict[str, Any] | None = None

_backtest_thread: threading.Thread | None = None
_backtest_result: dict[str, Any] | None = None

_retrain_thread: threading.Thread | None = None
_retrain_result: dict[str, Any] | None = None


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


# ── ML retrain from prod DB data ─────────────────────────


def _run_retrain_pipeline(
    start_date: str,
    target_mode: str,
    notify: bool,
) -> None:
    """Background worker: extract features from prod DB, train walk-forward,
    save model, log metrics to ``ml_train_log`` table, optionally notify via
    Telegram.
    """
    global _retrain_result
    started = dt.datetime.now(dt.timezone.utc)
    try:
        from trader_koo.ml.trainer import train_walk_forward

        conn = get_conn()
        try:
            # Determine end_date from the latest data in the DB
            row = conn.execute(
                "SELECT MAX(date) AS max_date FROM price_daily "
                "WHERE ticker = 'SPY' AND close IS NOT NULL"
            ).fetchone()
            end_date = str(row[0]) if row and row[0] else started.strftime("%Y-%m-%d")

            LOG.info(
                "Retrain pipeline: start=%s end=%s target_mode=%s",
                start_date, end_date, target_mode,
            )

            result = train_walk_forward(
                conn,
                start_date=start_date,
                end_date=end_date,
                target_mode=target_mode,
            )

            elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
            result["elapsed_sec"] = round(elapsed, 1)
            result["start_date"] = start_date
            result["end_date"] = end_date

            # Log metrics to DB table for historical tracking
            _log_retrain_metrics(conn, result)
            conn.commit()

            _retrain_result = result

            # Telegram notification
            if notify and result.get("ok"):
                _send_retrain_notification(result)

        finally:
            conn.close()

    except Exception as exc:
        elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
        LOG.exception("Retrain pipeline failed after %.1fs: %s", elapsed, exc)
        _retrain_result = {
            "ok": False,
            "error": str(exc),
            "elapsed_sec": round(elapsed, 1),
        }


def _log_retrain_metrics(
    conn: sqlite3.Connection,
    result: dict[str, Any],
) -> None:
    """Persist retrain metrics to the ``ml_train_log`` table."""
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_train_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trained_at TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                target_mode TEXT,
                fold_count INTEGER,
                total_samples INTEGER,
                avg_auc REAL,
                best_auc REAL,
                avg_accuracy REAL,
                avg_precision REAL,
                model_path TEXT,
                elapsed_sec REAL,
                details_json TEXT
            )
        """)
        agg = result.get("aggregate_metrics") or {}
        conn.execute(
            """
            INSERT INTO ml_train_log
                (trained_at, start_date, end_date, target_mode,
                 fold_count, total_samples, avg_auc, best_auc,
                 avg_accuracy, avg_precision, model_path, elapsed_sec,
                 details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.get("trained_at", dt.datetime.now(dt.timezone.utc).isoformat()),
                result.get("start_date"),
                result.get("end_date"),
                result.get("target_mode"),
                result.get("fold_count"),
                result.get("total_samples"),
                agg.get("avg_auc"),
                agg.get("best_auc"),
                agg.get("avg_accuracy"),
                agg.get("avg_precision"),
                result.get("model_path"),
                result.get("elapsed_sec"),
                json.dumps(result.get("folds", []), default=str),
            ),
        )
    except Exception as exc:
        LOG.warning("Failed to log retrain metrics: %s", exc)


def _send_retrain_notification(result: dict[str, Any]) -> None:
    """Send a Telegram message with retrain results."""
    try:
        from trader_koo.notifications.telegram import is_configured, send_message

        if not is_configured():
            return

        agg = result.get("aggregate_metrics") or {}
        meta = result.get("meta_labeling") or {}
        fold_count = result.get("fold_count", 0)
        samples = result.get("total_samples", 0)
        elapsed = result.get("elapsed_sec", 0)

        lines = [
            "\U0001f916 *ML Model Retrained*",
            "",
            f"Folds: {fold_count} | Samples: {samples:,}",
            f"Avg AUC: {agg.get('avg_auc', 0):.4f} | Best: {agg.get('best_auc', 0):.4f}",
            f"Avg Precision: {agg.get('avg_precision', 0):.4f}",
            f"Target mode: {result.get('target_mode', 'unknown')}",
            f"Range: {result.get('start_date')} to {result.get('end_date')}",
            f"Time: {elapsed:.0f}s",
        ]

        if meta.get("ok"):
            lines.append(f"Meta-label AUC: {meta.get('auc', 0):.4f}")

        model_path = result.get("model_path", "")
        if model_path:
            lines.append(f"Model: `{Path(model_path).name}`")

        send_message("\n".join(lines))
    except Exception as exc:
        LOG.warning("Failed to send retrain notification: %s", exc)


@router.post("/api/admin/ml/retrain")
@require_admin_auth
def retrain_ml_model(
    request: Request,
    start_date: str = Query(default="2025-01-01"),
    target_mode: str = Query(default="return_sign"),
    notify: bool = Query(default=True),
) -> dict[str, Any]:
    """Retrain the LightGBM model using current prod DB data.

    Extracts features from the live database, runs walk-forward training,
    saves the model, logs metrics to ``ml_train_log``, and optionally sends
    a Telegram notification with results.

    The end_date is auto-detected from the latest price data in the DB.

    Parameters
    ----------
    start_date : str
        Training start date (default: 2025-01-01).
    target_mode : str
        Binary target strategy: "return_sign", "barrier", or "rank".
    notify : bool
        Send Telegram notification on completion (default: True).
    """
    global _retrain_thread, _retrain_result

    if target_mode not in {"return_sign", "barrier", "rank"}:
        return {
            "ok": False,
            "error": (
                f"Invalid target_mode '{target_mode}'. "
                "Choose from: 'return_sign', 'barrier', 'rank'."
            ),
        }

    if _retrain_thread and _retrain_thread.is_alive():
        return {
            "ok": False,
            "message": (
                "Retrain already in progress. "
                "Check GET /api/admin/ml/retrain-status"
            ),
        }

    # Also block if legacy training is running
    if _ml_train_thread and _ml_train_thread.is_alive():
        return {
            "ok": False,
            "message": (
                "Training already in progress via /api/admin/train-ml-model. "
                "Check /api/admin/ml-model-status"
            ),
        }

    _retrain_result = {
        "ok": False,
        "status": "training",
        "message": "Retrain pipeline started...",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    _retrain_thread = threading.Thread(
        target=_run_retrain_pipeline,
        args=(start_date, target_mode, notify),
        daemon=True,
    )
    _retrain_thread.start()

    return {
        "ok": True,
        "message": (
            f"Retrain pipeline started (start_date={start_date}, "
            f"target_mode={target_mode}, notify={notify}). "
            "Check GET /api/admin/ml/retrain-status for results."
        ),
        "start_date": start_date,
        "target_mode": target_mode,
    }


@router.get("/api/admin/ml/retrain-status")
@require_admin_auth
def retrain_status(request: Request) -> dict[str, Any]:
    """Return current retrain pipeline status and latest result."""
    running = bool(_retrain_thread and _retrain_thread.is_alive())
    if _retrain_result is None:
        return {
            "ok": False,
            "running": running,
            "message": "No retrain has been triggered yet",
        }
    result = dict(_retrain_result)
    result["running"] = running
    return result


@router.get("/api/admin/ml/retrain-history")
@require_admin_auth
def retrain_history(
    request: Request,
    limit: int = Query(default=10, ge=1, le=100),
) -> dict[str, Any]:
    """Return historical retrain log entries."""
    try:
        conn = get_conn()
        try:
            rows = conn.execute(
                """
                SELECT id, trained_at, start_date, end_date, target_mode,
                       fold_count, total_samples, avg_auc, best_auc,
                       avg_accuracy, avg_precision, model_path, elapsed_sec
                FROM ml_train_log
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            entries = [dict(r) for r in rows] if rows else []
            return {"ok": True, "count": len(entries), "entries": entries}
        except sqlite3.OperationalError:
            # Table does not exist yet
            return {"ok": True, "count": 0, "entries": []}
        finally:
            conn.close()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


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
