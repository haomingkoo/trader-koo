"""Run full walk-forward ML validation and execution backtest.

This is intentionally a script, not a request-path endpoint, because full
universe validation can take minutes. It writes a compact JSON artifact that
can be reviewed later without trusting console scrollback.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from trader_koo.ml.backtest import run_backtest
from trader_koo.ml.features import ML_CONTEXT_TICKERS
from trader_koo.ml.rule_baseline import run_rule_baseline
from trader_koo.ml.trainer import train_walk_forward

LOG = logging.getLogger("trader_koo.scripts.run_walk_forward_validation")


def _default_db_path() -> Path:
    env_path = os.getenv("TRADER_KOO_DB_PATH")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parents[2] / "data" / "trader_koo.db"


def _latest_spy_date(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT MAX(date) FROM price_daily WHERE ticker='SPY' AND close IS NOT NULL"
    ).fetchone()
    if row and row[0]:
        return str(row[0])
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def _universe_stats(conn: sqlite3.Connection, start_date: str, end_date: str) -> dict[str, Any]:
    excluded = sorted(ML_CONTEXT_TICKERS)
    placeholders = ",".join("?" * len(excluded))
    row = conn.execute(
        f"""
        SELECT COUNT(DISTINCT ticker), COUNT(DISTINCT date), COUNT(*)
        FROM price_daily
        WHERE date >= ? AND date <= ?
          AND close IS NOT NULL
          AND ticker NOT LIKE '^%'
          AND ticker NOT IN ({placeholders})
        """,
        (start_date, end_date, *excluded),
    ).fetchone()
    return {
        "tickers": int(row[0] or 0),
        "dates": int(row[1] or 0),
        "rows": int(row[2] or 0),
    }


def _compact_backtest(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the artifact readable while preserving audit metadata."""
    if not isinstance(result, dict):
        return {"ok": False, "error": "invalid_backtest_result"}
    compact = dict(result)
    trade_log = compact.get("trade_log")
    if isinstance(trade_log, list):
        compact["trade_log_sample"] = trade_log[:25]
        compact["trade_log_count_in_payload"] = len(trade_log)
        compact.pop("trade_log", None)
    curve = compact.get("equity_curve")
    if isinstance(curve, list):
        compact["equity_curve_head"] = curve[:10]
        compact["equity_curve_tail"] = curve[-10:]
        compact["equity_curve_points"] = len(curve)
        compact.pop("equity_curve", None)
    return compact


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full walk-forward validation.")
    parser.add_argument("--db-path", type=Path, default=_default_db_path())
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--backtest-start-date", default="2025-06-01")
    parser.add_argument("--target-mode", default="barrier", choices=["return_sign", "barrier", "rank"])
    parser.add_argument("--train-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--embargo-days", type=int, default=15)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--min-win-prob", type=float, default=0.55)
    parser.add_argument("--short-threshold", type=float, default=0.45)
    parser.add_argument("--rule-min-score", type=float, default=68.0)
    parser.add_argument("--output-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--skip-rule-baseline", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    started = dt.datetime.now(dt.timezone.utc)
    t0 = time.time()
    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row
    try:
        end_date = args.end_date or _latest_spy_date(conn)
        universe = _universe_stats(conn, args.start_date, end_date)
        LOG.info(
            "Validation universe: %s tickers, %s dates, %s rows (%s to %s)",
            universe["tickers"],
            universe["dates"],
            universe["rows"],
            args.start_date,
            end_date,
        )

        training = train_walk_forward(
            conn,
            start_date=args.start_date,
            end_date=end_date,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            embargo_days=args.embargo_days,
            target_mode=args.target_mode,
        )

        backtest: dict[str, Any] | None = None
        if not args.skip_backtest:
            backtest = run_backtest(
                conn,
                start_date=args.backtest_start_date,
                end_date=end_date,
                max_positions=args.max_positions,
                min_win_prob=args.min_win_prob,
                short_threshold=args.short_threshold,
                target_mode=args.target_mode,
            )

        rule_baseline: dict[str, Any] | None = None
        if not args.skip_backtest and not args.skip_rule_baseline:
            rule_baseline = run_rule_baseline(
                conn,
                start_date=args.backtest_start_date,
                end_date=end_date,
                max_positions=args.max_positions,
                min_score=args.rule_min_score,
            )

        components_ok = [bool(training.get("ok"))]
        if backtest is not None:
            components_ok.append(bool(backtest.get("ok")))
        if rule_baseline is not None:
            components_ok.append(bool(rule_baseline.get("ok")))

        elapsed = round(time.time() - t0, 1)
        artifact = {
            "ok": all(components_ok),
            "started_at": started.isoformat(),
            "elapsed_sec": elapsed,
            "db_path": str(args.db_path),
            "config": {
                "start_date": args.start_date,
                "end_date": end_date,
                "backtest_start_date": args.backtest_start_date,
                "target_mode": args.target_mode,
                "train_days": args.train_days,
                "test_days": args.test_days,
                "step_days": args.step_days,
                "embargo_days": args.embargo_days,
                "max_positions": args.max_positions,
                "min_win_prob": args.min_win_prob,
                "short_threshold": args.short_threshold,
                "rule_min_score": args.rule_min_score,
                "skip_rule_baseline": args.skip_rule_baseline,
            },
            "universe": universe,
            "training": training,
            "backtest": _compact_backtest(backtest) if backtest is not None else None,
            "rule_baseline": _compact_backtest(rule_baseline) if rule_baseline is not None else None,
        }

        args.output_dir.mkdir(parents=True, exist_ok=True)
        ts = started.strftime("%Y%m%dT%H%M%SZ")
        out_path = args.output_dir / f"walk_forward_validation_{ts}.json"
        out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

        registry: dict[str, Any] | None = None
        try:
            from trader_koo.ml.validation_registry import record_validation_run

            registry = record_validation_run(
                conn,
                artifact,
                source="walk_forward_validation",
                artifact_path=str(out_path),
                run_id=f"walk-forward-{ts}",
            )
            artifact["validation_registry"] = registry
            out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        except Exception as exc:
            LOG.warning("Validation registry write failed: %s", exc)

        print(json.dumps({
            "ok": artifact["ok"],
            "elapsed_sec": elapsed,
            "artifact": str(out_path),
            "validation_registry": registry,
            "universe": universe,
            "training": {
                "ok": training.get("ok"),
                "fold_count": training.get("fold_count"),
                "total_samples": training.get("total_samples"),
                "aggregate_metrics": training.get("aggregate_metrics"),
                "target_mode": training.get("target_mode"),
            },
            "backtest": (backtest or {}).get("summary") if backtest is not None else None,
            "rule_baseline": (rule_baseline or {}).get("summary") if rule_baseline is not None else None,
        }, indent=2))
        return 0 if artifact["ok"] else 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
