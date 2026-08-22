"""Chronological paper-campaign replay using the production decision policy."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

from trader_koo.paper_trade.campaign import (
    canonical_hash,
    decide_candidate,
    record_promotion_experiment,
)
from trader_koo.paper_trade.config import PaperTradeConfig, config_snapshot
from trader_koo.paper_trade.decision import direction_from_row


def _max_drawdown(values: list[float]) -> float:
    peak = values[0] if values else 1.0
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak:
            worst = max(worst, (peak - value) / peak * 100)
    return worst


def _ratio(mean: float, downside: float) -> float | None:
    return mean / downside * math.sqrt(252) if downside > 0 else None


def _confidence_interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    if len(values) == 1:
        return [round(mean, 6), round(mean, 6)]
    margin = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return [round(mean - margin, 6), round(mean + margin, 6)]


def _trade_metrics(
    trades: list[dict[str, Any]],
    *,
    starting_capital: float,
    equity_curve: list[dict[str, Any]],
    spy_buy_hold_return_pct: float,
    matched_spy_return_pct: float,
    candidate_count: int,
    admitted_count: int,
    exposure_samples: list[float],
) -> dict[str, Any]:
    pnls = [float(item["net_pnl"]) for item in trades]
    returns = [float(item["net_return_pct"]) for item in trades]
    gross_profit = sum(value for value in pnls if value > 0)
    gross_loss = abs(sum(value for value in pnls if value < 0))
    final_equity = float(equity_curve[-1]["equity"]) if equity_curve else starting_capital
    portfolio_return = (final_equity / starting_capital - 1) * 100
    daily_returns: list[float] = []
    for before, after in zip(equity_curve, equity_curve[1:]):
        base = float(before["equity"])
        if base:
            daily_returns.append(float(after["equity"]) / base - 1)
    mean_daily = statistics.fmean(daily_returns) if daily_returns else 0.0
    volatility = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0.0
    downside_values = [min(value, 0.0) for value in daily_returns]
    downside = statistics.pstdev(downside_values) if len(downside_values) > 1 else 0.0
    max_dd = _max_drawdown([float(item["equity"]) for item in equity_curve])
    annual_return = mean_daily * 252 * 100
    return {
        "candidate_count": candidate_count,
        "admitted_count": admitted_count,
        "closed_trades": len(trades),
        "conversion_rate_pct": round(admitted_count / candidate_count * 100, 6) if candidate_count else 0.0,
        "average_exposure_pct": round(statistics.fmean(exposure_samples), 6) if exposure_samples else 0.0,
        "turnover_pct": round(sum(float(item["notional"]) for item in trades) / starting_capital * 100, 6),
        "portfolio_return_pct": round(portfolio_return, 6),
        "spy_return_pct": round(spy_buy_hold_return_pct, 6),
        "matched_spy_return_pct": round(matched_spy_return_pct, 6),
        "matched_spy_active_return_pct": round(portfolio_return - matched_spy_return_pct, 6),
        "max_drawdown_pct": round(max_dd, 6),
        "sharpe_ratio": round(mean_daily / volatility * math.sqrt(252), 6) if volatility > 0 else None,
        "sortino_ratio": round(_ratio(mean_daily, downside), 6) if downside > 0 else None,
        "calmar_ratio": round(annual_return / max_dd, 6) if max_dd > 0 else None,
        "profit_factor": round(gross_profit / gross_loss, 6) if gross_loss > 0 else None,
        "win_rate_pct": round(sum(value > 0 for value in pnls) / len(pnls) * 100, 6) if pnls else 0.0,
        "mean_trade_return_pct_ci95": _confidence_interval(returns),
    }


def replay_campaign(
    *,
    candidate_runs: list[dict[str, Any]],
    price_rows: list[dict[str, Any]],
    spy_rows: list[dict[str, Any]],
    config: PaperTradeConfig,
    expected_execution: dict[str, dict[str, Any]] | None = None,
    _include_splits: bool = True,
) -> dict[str, Any]:
    """Replay signal cohorts and portfolio state in strict event-time order.

    Candidate critic outcomes are part of each historical input. Missing critic
    evidence rejects by the same fail-closed policy used live.
    """
    by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in price_rows:
        row = dict(raw)
        row["ticker"] = str(row.get("ticker") or "").upper()
        row["date"] = str(row.get("date") or "")
        by_ticker[row["ticker"]].append(row)
    for rows in by_ticker.values():
        rows.sort(key=lambda item: item["date"])

    fill_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_count = 0
    for run in sorted(candidate_runs, key=lambda item: (str(item["report_date"]), str(item["report_run_id"]))):
        report_date = str(run["report_date"])
        for rank, candidate in enumerate(run.get("candidates") or [], start=1):
            candidate_count += 1
            ticker = str(candidate.get("ticker") or "").upper()
            next_bar = next((bar for bar in by_ticker.get(ticker, []) if bar["date"] > report_date and bar.get("open") is not None), None)
            if not next_bar:
                continue
            fill_events[next_bar["date"]].append({
                "run": run, "rank": rank, "candidate": candidate,
                "bar": next_bar,
            })

    all_dates = sorted({str(row["date"]) for row in price_rows})
    capital = float(config.starting_capital)
    positions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    exposure_samples: list[float] = []
    net_exposure_samples: list[float] = []
    admitted_count = 0
    policy = config_snapshot(config)

    for date in all_dates:
        # Exits precede new entries. If stop and target both touch inside one
        # daily bar, the conservative stop-first assumption is deterministic.
        survivors: list[dict[str, Any]] = []
        for position in positions:
            bar = next((item for item in by_ticker[position["ticker"]] if item["date"] == date), None)
            if not bar:
                survivors.append(position)
                continue
            position["bars_held"] += 1
            direction = position["direction"]
            stop = float(position["stop_loss"])
            target = float(position["target_price"])
            open_price = float(bar["open"])
            low = float(bar["low"])
            high = float(bar["high"])
            reason = None
            raw_exit = None
            if direction == "long":
                if open_price <= stop:
                    reason, raw_exit = "stopped_out", open_price
                elif open_price >= target:
                    reason, raw_exit = "target_hit", open_price
                elif low <= stop:
                    reason, raw_exit = "stopped_out", stop
                elif high >= target:
                    reason, raw_exit = "target_hit", target
            else:
                if open_price >= stop:
                    reason, raw_exit = "stopped_out", open_price
                elif open_price <= target:
                    reason, raw_exit = "target_hit", open_price
                elif high >= stop:
                    reason, raw_exit = "stopped_out", stop
                elif low <= target:
                    reason, raw_exit = "target_hit", target
            if reason is None and position["bars_held"] >= config.expiry_days:
                reason, raw_exit = "expired", float(bar["close"])
            if reason is None:
                survivors.append(position)
                continue
            exit_slip = config.exit_slippage_bps / 10_000
            exit_price = float(raw_exit) * (1 - exit_slip if direction == "long" else 1 + exit_slip)
            signed = (exit_price / position["entry_price"] - 1) * (1 if direction == "long" else -1)
            borrow = position["notional"] * config.short_borrow_annual_pct / 100 / 252 * position["bars_held"] if direction == "short" else 0.0
            pnl = position["notional"] * signed - 2 * config.commission_per_trade - borrow
            capital += pnl
            trades.append({
                **position, "exit_date": date, "exit_price": round(exit_price, 6),
                "exit_reason": reason, "borrow_cost": round(borrow, 6),
                "net_pnl": round(pnl, 6),
                "net_return_pct": round(pnl / position["notional"] * 100, 6),
            })
        positions = survivors

        for event in sorted(fill_events.get(date, []), key=lambda item: (int(item["rank"]), str(item["candidate"].get("ticker") or ""))):
            candidate = event["candidate"]
            direction = direction_from_row(candidate)
            raw_open = float(event["bar"]["open"])
            slip = config.entry_slippage_bps / 10_000
            entry = raw_open * (1 + slip if direction == "long" else 1 - slip)
            historical_volumes = [
                float(bar["volume"])
                for bar in by_ticker.get(str(candidate.get("ticker") or "").upper(), [])
                if bar["date"] <= date and bar.get("volume") not in (None, 0)
            ][-20:]
            avg_daily_volume = (
                statistics.fmean(historical_volumes) if historical_volumes else None
            )
            portfolio_block = None
            if len(positions) >= config.max_open:
                portfolio_block = {"gate": "portfolio_capacity", "reason_code": "max_open_positions", "detail": "Replay portfolio is at capacity."}
            context = {
                "entry_price": entry,
                "avg_daily_volume": avg_daily_volume,
                "portfolio_block": portfolio_block,
                "critic_outcome": candidate.get("critic_outcome") or {"approved": False, "error": "missing_replay_critic_evidence"},
                "campaign_active": True, "duplicate": False,
                "execution_ready": True,
                "market_context": candidate.get("market_context") or {},
                "portfolio_context": {"open_count": len(positions)},
                "source_context": {"report_run_id": event["run"]["report_run_id"], "price_date": date},
            }
            decision = decide_candidate(row=candidate, rank=int(event["rank"]), config=config, context=context)
            execution_key = f"{event['run']['report_run_id']}:{event['rank']}"
            decisions.append({"execution_key": execution_key, **decision})
            if decision["disposition"] != "admitted":
                continue
            admitted_count += 1
            position_pct = float(decision["sizing"].get("position_size_pct") or 0.0)
            notional = min(capital, config.starting_capital * position_pct / 100)
            positions.append({
                "execution_key": execution_key, "ticker": decision["ticker"],
                "direction": direction, "entry_date": date,
                "entry_price": round(entry, 6), "notional": notional,
                "stop_loss": float(decision["stop_loss"]),
                "target_price": float(decision["target_price"]), "bars_held": 0,
            })
        exposure_samples.append(sum(float(item["notional"]) for item in positions) / config.starting_capital * 100)
        net_exposure_samples.append(sum(
            float(item["notional"]) * (1 if item["direction"] == "long" else -1)
            for item in positions
        ) / config.starting_capital)
        equity_curve.append({"date": date, "equity": round(capital, 6), "open_positions": len(positions)})

    # Close residual positions at their final available close so held-out return
    # and turnover are defined without silently discarding open risk.
    for position in positions:
        last_bar = by_ticker[position["ticker"]][-1]
        exit_price = float(last_bar["close"])
        signed = (exit_price / position["entry_price"] - 1) * (1 if position["direction"] == "long" else -1)
        borrow = position["notional"] * config.short_borrow_annual_pct / 100 / 252 * position["bars_held"] if position["direction"] == "short" else 0.0
        pnl = position["notional"] * signed - 2 * config.commission_per_trade - borrow
        capital += pnl
        trades.append({**position, "exit_date": last_bar["date"], "exit_price": exit_price, "exit_reason": "end_of_replay", "borrow_cost": borrow, "net_pnl": pnl, "net_return_pct": pnl / position["notional"] * 100})
    if equity_curve:
        equity_curve[-1]["equity"] = round(capital, 6)

    spy = sorted(spy_rows, key=lambda item: str(item["date"]))
    spy_return = 0.0
    if len(spy) >= 2 and float(spy[0]["open"]):
        spy_return = (float(spy[-1]["close"]) / float(spy[0]["open"]) - 1) * 100
    spy_by_date = {str(row["date"]): row for row in spy}
    matched_spy_return = 0.0
    matched_observations = 0
    previous_close: float | None = None
    for index, curve_point in enumerate(equity_curve):
        bar = spy_by_date.get(str(curve_point["date"]))
        if not bar or bar.get("close") is None:
            continue
        close = float(bar["close"])
        if bar.get("open") is not None:
            base = float(bar["open"])
            exposure = net_exposure_samples[index] if index < len(net_exposure_samples) else 0.0
        elif previous_close is not None:
            base = previous_close
            exposure = net_exposure_samples[index - 1] if index else 0.0
        else:
            previous_close = close
            continue
        if base:
            matched_spy_return += (close / base - 1) * exposure * 100
            matched_observations += 1
        previous_close = close
    if matched_observations == 0 and exposure_samples:
        matched_spy_return = spy_return * statistics.fmean(exposure_samples) / 100
    metrics = _trade_metrics(
        trades, starting_capital=config.starting_capital, equity_curve=equity_curve,
        spy_buy_hold_return_pct=spy_return,
        matched_spy_return_pct=matched_spy_return, candidate_count=candidate_count,
        admitted_count=admitted_count, exposure_samples=exposure_samples,
    )
    mismatches: list[str] = []
    if expected_execution is not None:
        actual = {item["execution_key"]: {"disposition": item["disposition"], "inputs_hash": item["inputs_hash"]} for item in decisions}
        for key in sorted(set(actual) | set(expected_execution)):
            if actual.get(key) != expected_execution.get(key):
                mismatches.append(key)
    parity = "matched" if expected_execution is not None and not mismatches else "diverged" if expected_execution is not None else "not_measured"
    dates = sorted({str(run["report_date"]) for run in candidate_runs})
    split = min(len(dates) - 1, max(1, int(len(dates) * 0.8))) if len(dates) > 1 else len(dates)
    training_dates = dates[:split]
    held_out_dates = dates[split:]
    walk_forward: dict[str, Any] = {
        "training_dates": training_dates,
        "held_out_dates": held_out_dates,
    }
    held_out: dict[str, Any] = {
        "dates": held_out_dates, "metrics": None, "trades": [],
    }
    if _include_splits and training_dates:
        training_result = replay_campaign(
            candidate_runs=[run for run in candidate_runs if str(run["report_date"]) in training_dates],
            price_rows=price_rows, spy_rows=spy_rows, config=config,
            expected_execution=None, _include_splits=False,
        )
        walk_forward["training_metrics"] = training_result["metrics"]
        walk_forward["training_trades"] = training_result["trades"]
        fold_count = min(5, max(0, len(dates) - 1))
        folds: list[dict[str, Any]] = []
        for fold_index in range(fold_count):
            validation_index = max(
                1, round((fold_index + 1) * (len(dates) - 1) / fold_count)
            )
            validation_date = dates[validation_index]
            fold_result = replay_campaign(
                candidate_runs=[
                    run for run in candidate_runs
                    if str(run["report_date"]) == validation_date
                ],
                price_rows=price_rows, spy_rows=spy_rows, config=config,
                expected_execution=None, _include_splits=False,
            )
            folds.append({
                "training_dates": dates[:validation_index],
                "validation_date": validation_date,
                "metrics": fold_result["metrics"],
                "trades": fold_result["trades"],
            })
        walk_forward["folds"] = folds
    if _include_splits and held_out_dates:
        held_result = replay_campaign(
            candidate_runs=[run for run in candidate_runs if str(run["report_date"]) in held_out_dates],
            price_rows=price_rows, spy_rows=spy_rows, config=config,
            expected_execution=None, _include_splits=False,
        )
        held_out = {
            "dates": held_out_dates, "metrics": held_result["metrics"],
            "trades": held_result["trades"],
        }
    return {
        "engine_version": "paper-replay-v2.0",
        "policy_hash": canonical_hash(policy),
        "dataset_hash": canonical_hash({"candidate_runs": candidate_runs, "price_rows": price_rows, "spy_rows": spy_rows}),
        "metrics": metrics,
        "trades": trades,
        "decisions": decisions,
        "equity_curve": equity_curve,
        "replay_live_parity": parity,
        "parity_mismatches": mismatches,
        "walk_forward": walk_forward,
        "held_out": held_out,
    }


def replay_and_seal_promotion(
    conn,
    *,
    experiment_id: str,
    preregistration_id: str,
    campaign_id: str,
    candidate_runs: list[dict[str, Any]],
    price_rows: list[dict[str, Any]],
    spy_rows: list[dict[str, Any]],
    config: PaperTradeConfig,
    expected_execution: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Run the real engine, measure parity, then seal promotion evidence."""
    result = replay_campaign(
        candidate_runs=candidate_runs, price_rows=price_rows, spy_rows=spy_rows,
        config=config, expected_execution=expected_execution,
    )
    promotion_metrics = {
        **result["metrics"],
        "engine_version": result["engine_version"],
        "walk_forward": result["walk_forward"],
        "held_out": result["held_out"],
    }
    evidence = record_promotion_experiment(
        conn, experiment_id=experiment_id, preregistration_id=preregistration_id,
        campaign_id=campaign_id,
        policy_version=config.decision_version, policy_hash=result["policy_hash"],
        dataset_hash=result["dataset_hash"],
        metrics=promotion_metrics, parity_status=result["replay_live_parity"],
    )
    return {**result, "promotion_evidence": evidence}
