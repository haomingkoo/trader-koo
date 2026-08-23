"""Paper-campaign adapter for the canonical portfolio execution ledger."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

from trader_koo.paper_trade.campaign import canonical_hash, decide_candidate, record_promotion_experiment
from trader_koo.paper_trade.chronology import (
    next_scheduled_session_after,
    publication_precedes_session_open,
)
from trader_koo.paper_trade.config import PaperTradeConfig, config_snapshot
from trader_koo.paper_trade.decision import direction_from_row
from trader_koo.research.next_open_baseline import (
    BaselineConfig, ExecutionDecision, SessionPrice, simulate_portfolio,
)

ENGINE_VERSION = "portfolio-execution-v1.0"


def _publication_block(published_ts: str) -> dict[str, str]:
    if not published_ts:
        return {
            "gate": "execution.next_open",
            "reason_code": "report_publication_timestamp_unavailable",
            "detail": "Replay requires verified publication chronology.",
        }
    return {
        "gate": "execution.next_open",
        "reason_code": "report_published_after_intended_open",
        "detail": "Verified report publication did not precede the intended session open.",
    }


def _max_drawdown(values: list[float]) -> float:
    peak, worst = (values[0] if values else 1.0), 0.0
    for value in values:
        peak = max(peak, value)
        if peak:
            worst = max(worst, (peak - value) / peak * 100)
    return worst


def _confidence_interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    if len(values) == 1:
        return [round(mean, 6), round(mean, 6)]
    margin = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return [round(mean - margin, 6), round(mean + margin, 6)]


def _trade_metrics(trades: list[dict[str, Any]], *, starting_capital: float,
                   equity_curve: list[dict[str, Any]], spy_return_pct: float,
                   matched_spy_return_pct: float, candidate_count: int,
                   admitted_count: int) -> dict[str, Any]:
    pnls = [float(item["net_pnl"]) for item in trades]
    returns = [float(item["net_return_pct"]) for item in trades]
    priced = [float(item["equity"]) for item in equity_curve if item.get("equity") is not None]
    final_equity = priced[-1] if priced else starting_capital
    daily = [after / before - 1 for before, after in zip(priced, priced[1:]) if before]
    mean = statistics.fmean(daily) if daily else 0.0
    volatility = statistics.stdev(daily) if len(daily) > 1 else 0.0
    downside = [min(value, 0.0) for value in daily]
    downside_vol = statistics.pstdev(downside) if len(downside) > 1 else 0.0
    max_dd = _max_drawdown(priced)
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = abs(sum(value for value in pnls if value < 0))
    portfolio_return = (final_equity / starting_capital - 1) * 100
    exposure = [float(item["gross_exposure_pct"]) for item in equity_curve
                if item.get("gross_exposure_pct") is not None]
    return {
        "candidate_count": candidate_count, "admitted_count": admitted_count,
        "closed_trades": len(trades),
        "open_positions": int(equity_curve[-1].get("open_positions", 0)) if equity_curve else 0,
        "conversion_rate_pct": round(admitted_count / candidate_count * 100, 6) if candidate_count else 0.0,
        "average_exposure_pct": round(statistics.fmean(exposure), 6) if exposure else 0.0,
        "turnover_pct": round(sum(float(item["notional"]) for item in trades) / starting_capital * 100, 6),
        "portfolio_return_pct": round(portfolio_return, 6),
        "spy_return_pct": round(spy_return_pct, 6),
        "matched_spy_return_pct": round(matched_spy_return_pct, 6),
        "matched_spy_active_return_pct": round(portfolio_return - matched_spy_return_pct, 6),
        "max_drawdown_pct": round(max_dd, 6),
        "sharpe_ratio": round(mean / volatility * math.sqrt(252), 6) if volatility else None,
        "sortino_ratio": round(mean / downside_vol * math.sqrt(252), 6) if downside_vol else None,
        "calmar_ratio": round(mean * 252 * 100 / max_dd, 6) if max_dd else None,
        "profit_factor": round(gross_profit / gross_loss, 6) if gross_loss else None,
        "win_rate_pct": round(sum(value > 0 for value in pnls) / len(pnls) * 100, 6) if pnls else 0.0,
        "mean_trade_return_pct_ci95": _confidence_interval(returns),
    }


def _campaign_inputs(candidate_runs: list[dict[str, Any]], price_rows: list[dict[str, Any]],
                     config: PaperTradeConfig) -> tuple[list[dict[str, Any]], list[ExecutionDecision], list[SessionPrice], list[str]]:
    by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prices: list[SessionPrice] = []
    for raw in price_rows:
        row = dict(raw)
        ticker, date = str(row.get("ticker") or "").upper(), str(row.get("date") or "")
        row.update(ticker=ticker, date=date)
        by_ticker[ticker].append(row)
        prices.append(SessionPrice(ticker, date, row.get("open"), row.get("close"),
                                   row.get("high"), row.get("low"), row.get("volume")))
    for rows in by_ticker.values():
        rows.sort(key=lambda item: item["date"])
    sessions = sorted({
        row.date for row in prices
        if row.ticker == "SPY" and row.open is not None
    })
    session_index = {date: index for index, date in enumerate(sessions)}
    decisions: list[dict[str, Any]] = []
    executable: list[ExecutionDecision] = []
    for run in sorted(candidate_runs, key=lambda item: (str(item["report_date"]), str(item["report_run_id"]))):
        report_date = str(run["report_date"])
        published_ts = str(run.get("published_ts") or "")
        for rank, raw_candidate in enumerate(run.get("candidates") or [], start=1):
            sealed = raw_candidate.get("__sealed_context") if isinstance(raw_candidate, dict) else None
            candidate = (
                dict(raw_candidate.get("__sealed_candidate") or {})
                if isinstance(raw_candidate, dict) and "__sealed_candidate" in raw_candidate
                else raw_candidate
            )
            ticker = str(candidate.get("ticker") or "").upper()
            intended_session = next_scheduled_session_after(report_date)
            spy_ready = intended_session in session_index
            next_bar = (
                next((bar for bar in by_ticker.get(ticker, [])
                      if bar["date"] == intended_session and bar.get("open") is not None), None)
                if spy_ready else None
            )
            publication_ready = bool(
                intended_session
                and publication_precedes_session_open(published_ts, intended_session)
            )
            execution_key = f"{run['report_run_id']}:{rank}"
            raw_entry = float(next_bar["open"]) if next_bar else None
            direction = direction_from_row(candidate) if isinstance(candidate, dict) else "long"
            slip = config.entry_slippage_bps / 10_000
            entry = (
                round(raw_entry * (1 + slip if direction == "long" else 1 - slip), 4)
                if raw_entry is not None else None
            )
            volumes = [float(bar["volume"]) for bar in by_ticker.get(ticker, [])
                       if bar["date"] <= report_date and bar.get("volume") not in (None, 0)][-20:]
            causal_avg_volume = statistics.fmean(volumes) if volumes else None
            context = {
                "entry_price": entry,
                "avg_daily_volume": causal_avg_volume,
                "portfolio_block": (
                    None
                    if publication_ready
                    else _publication_block(published_ts)
                ),
                "critic_outcome": candidate.get("critic_outcome") or {"approved": False, "error": "missing_replay_critic_evidence"},
                "campaign_active": True, "duplicate": False,
                "execution_ready": next_bar is not None,
                "market_context": candidate.get("market_context") or {},
                "portfolio_context": {},
                "source_context": {"report_run_id": run["report_run_id"],
                                   "intended_session": intended_session,
                                   "price_date": next_bar["date"] if next_bar else None},
            }
            if isinstance(sealed, dict):
                context.update(sealed)
                context.update(
                    entry_price=entry,
                    execution_ready=next_bar is not None,
                )
                if not publication_ready:
                    context["portfolio_block"] = _publication_block(published_ts)
                    context["duplicate"] = False
                    context["portfolio_context"] = {
                        "open_count": int(
                            (context.get("portfolio_context") or {}).get("open_count") or 0
                        )
                    }
                    context["source_context"] = {
                        "report_run_id": run["report_run_id"],
                        "intended_session": intended_session,
                        "price_date": next_bar["date"] if next_bar else None,
                    }
            decision = decide_candidate(row=candidate, rank=rank, config=config, context=context)
            decisions.append({"execution_key": execution_key, **decision})
            if decision["disposition"] != "admitted" or next_bar is None:
                continue
            entry_index = session_index[next_bar["date"]]
            exit_index = min(len(sessions) - 1, entry_index + config.expiry_days)
            position_pct = float(decision["sizing"].get("position_size_pct") or 0)
            locked_notional = config.starting_capital * position_pct / 100
            avg_volume = context.get("avg_daily_volume")
            capacity = float(avg_volume) * raw_entry * config.max_adv_pct / 100 if avg_volume and raw_entry else 0.0
            executable.append(ExecutionDecision(
                execution_key, ticker, direction, report_date,
                next_bar["date"], sessions[exit_index], float(candidate.get("score") or 0),
                capacity, locked_notional=locked_notional,
                metadata=(("execution_key", execution_key),),
                stop_loss=float(decision["stop_loss"]), target_price=float(decision["target_price"]),
                max_holding_sessions=config.expiry_days,
            ))
    return decisions, executable, prices, sessions


def _matched_spy_return(curve: list[dict[str, Any]], spy_rows: list[dict[str, Any]]) -> tuple[float, float]:
    spy = sorted(spy_rows, key=lambda item: str(item["date"]))
    buy_hold = 0.0
    if len(spy) >= 2 and spy[0].get("open") and spy[-1].get("close") is not None:
        buy_hold = (float(spy[-1]["close"]) / float(spy[0]["open"]) - 1) * 100
    by_date = {str(row["date"]): row for row in spy}
    matched, previous_close = 0.0, None
    for point in curve:
        bar = by_date.get(str(point["date"]))
        if not bar or bar.get("close") is None:
            continue
        close = float(bar["close"])
        base = float(bar["open"]) if bar.get("open") is not None else previous_close
        if base:
            matched += (close / base - 1) * float(point.get("net_exposure_pct") or 0)
        previous_close = close
    return buy_hold, matched


def replay_campaign(*, candidate_runs: list[dict[str, Any]], price_rows: list[dict[str, Any]],
                    spy_rows: list[dict[str, Any]], config: PaperTradeConfig,
                    expected_execution: dict[str, dict[str, Any]] | None = None,
                    _include_splits: bool = True) -> dict[str, Any]:
    """Adapt campaign candidates to the one canonical execution ledger."""
    decisions, executable, prices, sessions = _campaign_inputs(candidate_runs, price_rows, config)
    execution = simulate_portfolio(executable, prices, sessions, BaselineConfig(
        initial_capital=config.starting_capital, max_positions=config.max_open,
        position_pct=config.max_position_pct, max_name_pct=config.max_position_pct,
        max_adv_pct=config.max_adv_pct, entry_slippage_bps=config.entry_slippage_bps,
        exit_slippage_bps=config.exit_slippage_bps, commission_bps_per_side=0,
        minimum_commission_per_side=config.commission_per_trade,
        short_borrow_bps_annual=config.short_borrow_annual_pct * 100,
        cash_rate_bps_annual=0, holding_sessions=max(2, config.expiry_days),
    ))
    trades: list[dict[str, Any]] = []
    for raw in execution.trades:
        item = dict(raw)
        item["notional"] = float(item["entry_notional"])
        item["borrow_cost"] = float(item.get("borrow") or 0)
        item["net_return_pct"] = float(item["net_pnl"]) / item["notional"] * 100 if item["notional"] else 0
        trades.append(item)
    curve = [dict(item) for item in execution.equity_curve]
    spy_return, matched_spy = _matched_spy_return(curve, spy_rows)
    actual = {item["execution_key"]: {"disposition": item["disposition"], "inputs_hash": item["inputs_hash"]}
              for item in decisions}
    mismatches = [] if expected_execution is None else [key for key in sorted(set(actual) | set(expected_execution))
                                                        if actual.get(key) != expected_execution.get(key)]
    parity = "not_measured" if expected_execution is None else "matched" if not mismatches else "diverged"
    dates = sorted({str(run["report_date"]) for run in candidate_runs})
    split = min(len(dates) - 1, max(1, int(len(dates) * .8))) if len(dates) > 1 else len(dates)
    training_dates, held_out_dates = dates[:split], dates[split:]
    walk_forward: dict[str, Any] = {"training_dates": training_dates, "held_out_dates": held_out_dates}
    held_out: dict[str, Any] = {"dates": held_out_dates, "metrics": None, "trades": []}
    if _include_splits and training_dates:
        training = replay_campaign(candidate_runs=[run for run in candidate_runs if str(run["report_date"]) in training_dates],
                                   price_rows=price_rows, spy_rows=spy_rows, config=config, _include_splits=False)
        walk_forward.update(training_metrics=training["metrics"], training_trades=training["trades"])
        folds, fold_count = [], min(5, max(0, len(dates) - 1))
        for fold_index in range(fold_count):
            index = max(1, round((fold_index + 1) * (len(dates) - 1) / fold_count))
            validation_date = dates[index]
            fold = replay_campaign(candidate_runs=[run for run in candidate_runs if str(run["report_date"]) == validation_date],
                                   price_rows=price_rows, spy_rows=spy_rows, config=config, _include_splits=False)
            folds.append({"training_dates": dates[:index], "validation_date": validation_date,
                          "metrics": fold["metrics"], "trades": fold["trades"]})
        walk_forward["folds"] = folds
    if _include_splits and held_out_dates:
        held = replay_campaign(candidate_runs=[run for run in candidate_runs if str(run["report_date"]) in held_out_dates],
                               price_rows=price_rows, spy_rows=spy_rows, config=config, _include_splits=False)
        held_out = {"dates": held_out_dates, "metrics": held["metrics"], "trades": held["trades"]}
    metrics = _trade_metrics(
        trades, starting_capital=config.starting_capital, equity_curve=curve,
        spy_return_pct=spy_return, matched_spy_return_pct=matched_spy,
        candidate_count=sum(len(run.get("candidates") or []) for run in candidate_runs),
        admitted_count=sum(item["disposition"] == "admitted" for item in decisions),
    )
    return {
        "engine_version": ENGINE_VERSION, "policy_hash": canonical_hash(config_snapshot(config)),
        "dataset_hash": canonical_hash({"candidate_runs": candidate_runs, "price_rows": price_rows, "spy_rows": spy_rows}),
        "metrics": metrics, "trades": trades,
        "execution_ledger": execution.ledger,
        "execution_ledger_hash": execution.ledger["provenance"]["ledger_sha256"],
        "open_positions": [dict(item) for item in execution.open_positions],
        "exclusions": [dict(item) for item in execution.exclusions],
        "decisions": decisions, "equity_curve": curve, "replay_live_parity": parity,
        "parity_mismatches": mismatches, "walk_forward": walk_forward, "held_out": held_out,
    }


def replay_and_seal_promotion(conn, *, experiment_id: str, preregistration_id: str,
                              campaign_id: str, candidate_runs: list[dict[str, Any]],
                              price_rows: list[dict[str, Any]], spy_rows: list[dict[str, Any]],
                              config: PaperTradeConfig) -> dict[str, Any]:
    """Compare replay with sealed live facts, then persist promotion evidence."""
    run_ids = sorted({str(run["report_run_id"]) for run in candidate_runs})
    if not run_ids:
        raise ValueError("promotion parity requires sealed live report runs")
    placeholders = ",".join("?" for _ in run_ids)
    live_rows = conn.execute(
        f"""SELECT d.report_run_id,d.candidate_rank,
                   CASE WHEN d.disposition='pending' AND po.status='filled'
                        THEN json_extract(ev.payload_json,'$.decision.disposition')
                        WHEN d.disposition='pending' AND po.status='rejected'
                        THEN 'rejected' ELSE d.disposition END,
                   CASE WHEN d.disposition='pending' AND ev.payload_json IS NOT NULL
                        THEN json_extract(ev.payload_json,'$.decision.inputs_hash')
                        ELSE d.inputs_hash END,
                   d.inputs_json
            FROM paper_candidate_decisions d
            LEFT JOIN paper_pending_orders po
              ON po.report_run_id=d.report_run_id AND po.campaign_id=d.campaign_id
             AND po.candidate_rank=d.candidate_rank
            LEFT JOIN paper_order_events ev
              ON ev.order_id=po.order_id AND ev.event_type IN ('filled','rejected')
            WHERE d.campaign_id=? AND d.report_run_id IN ({placeholders})
            ORDER BY d.report_run_id,d.candidate_rank""",
        (campaign_id, *run_ids),
    ).fetchall()
    expected_execution = {
        f"{row[0]}:{row[1]}": {"disposition": row[2], "inputs_hash": row[3]}
        for row in live_rows
    }
    if len(expected_execution) != sum(len(run.get("candidates") or []) for run in candidate_runs):
        raise ValueError("promotion parity requires one sealed live fact per candidate")
    live_inputs = {
        (str(row[0]), int(row[1])): json.loads(str(row[4])) for row in live_rows
    }
    sealed_candidate_runs = []
    publication_rows = conn.execute(
        f"SELECT run_id,published_ts FROM report_runs WHERE run_id IN ({placeholders})",
        tuple(run_ids),
    ).fetchall()
    publication_by_run = {str(row[0]): str(row[1] or "") for row in publication_rows}
    if set(publication_by_run) != set(run_ids) or any(
        not value for value in publication_by_run.values()
    ):
        raise ValueError("promotion parity requires verified publication timestamps")
    for run in candidate_runs:
        run_id = str(run["report_run_id"])
        sealed_candidates = []
        for rank, _candidate in enumerate(run.get("candidates") or [], start=1):
            evidence = live_inputs[(run_id, rank)]
            sealed_candidates.append({
                "__sealed_candidate": evidence["candidate"],
                "__sealed_context": evidence["context"],
            })
        sealed_candidate_runs.append({
            **run,
            "published_ts": publication_by_run[run_id],
            "candidates": sealed_candidates,
        })
    result = replay_campaign(candidate_runs=sealed_candidate_runs, price_rows=price_rows, spy_rows=spy_rows,
                             config=config, expected_execution=expected_execution)
    evidence = record_promotion_experiment(
        conn, experiment_id=experiment_id, preregistration_id=preregistration_id,
        campaign_id=campaign_id, policy_version=config.decision_version,
        policy_hash=result["policy_hash"], dataset_hash=result["dataset_hash"],
        metrics={**result["metrics"], "engine_version": result["engine_version"],
                 "walk_forward": result["walk_forward"], "held_out": result["held_out"]},
        parity_status=result["replay_live_parity"],
    )
    return {**result, "promotion_evidence": evidence}
