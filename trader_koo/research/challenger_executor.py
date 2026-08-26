"""Execute the three frozen challengers through the canonical portfolio ledger.

The module has one interface, :func:`execute_validation_tournament`. It owns
signal scheduling, next-open order construction, portfolio execution, cost
stress, fold scoring, and multiple-testing evidence. Callers only supply the
verified SQLite snapshot and frozen preregistration.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import random
import sqlite3
import statistics
from collections import defaultdict
from typing import Any

from trader_koo.ml.features import ML_CONTEXT_TICKERS
from trader_koo.research.challenger_preregistration import frozen_preregistration
from trader_koo.research.next_open_baseline import (
    BaselineConfig,
    ExecutionDecision,
    SessionPrice,
    canonical_json_bytes,
    simulate_portfolio,
)
from trader_koo.research.universe_membership import (
    VerifiedMembership,
    load_verified_membership,
)

def daily_returns(values: list[float]) -> list[float]:
    return [after / before - 1 for before, after in zip(values, values[1:]) if before > 0]


def _capped_inverse_volatility(volatility: dict[str, float]) -> dict[str, float]:
    remaining = {name: 1 / value for name, value in volatility.items() if value > 0}
    weights: dict[str, float] = {}
    budget = 1.0
    while remaining and budget > 1e-12:
        total = sum(remaining.values())
        proposed = {name: budget * value / total for name, value in remaining.items()}
        capped = {name for name, value in proposed.items() if value > .10}
        if not capped:
            weights.update(proposed)
            break
        for name in capped:
            weights[name] = .10
            budget -= .10
            remaining.pop(name)
    return {name: round(value * 100, 10) for name, value in sorted(weights.items())}


def c1_signal(history: dict[str, list[float]]) -> dict[str, Any]:
    scored: list[tuple[str, float, float]] = []
    for ticker, closes in history.items():
        if len(closes) < 253 or min(closes[-253:]) <= 0:
            continue
        score = closes[-22] / closes[-253] - 1
        daily = daily_returns(closes[-64:])
        volatility = statistics.stdev(daily) * math.sqrt(252) if len(daily) > 1 else 0
        if volatility > 0:
            scored.append((ticker, score, volatility))
    selected_count = min(20, math.ceil(len(scored) * .2))
    selected = sorted(scored, key=lambda row: (-row[1], row[0]))[:selected_count]
    return {
        "scores": {ticker: score for ticker, score, _vol in sorted(scored)},
        "weights_pct": _capped_inverse_volatility({ticker: vol for ticker, _score, vol in selected}),
    }


def c2_signal(
    history: dict[str, list[SessionPrice]],
    spy_rows: list[SessionPrice],
) -> dict[str, Any]:
    spy_by_date = {row.date: row for row in spy_rows}
    if len(spy_by_date) < 127:
        return {"residuals": {}, "weights_pct": {}}
    eligible: list[tuple[str, float]] = []
    for ticker, rows in history.items():
        asset_by_date = {row.date: row for row in rows}
        aligned_dates = sorted(set(asset_by_date) & set(spy_by_date))[-127:]
        if len(aligned_dates) < 127:
            continue
        asset = [asset_by_date[date] for date in aligned_dates]
        market = [spy_by_date[date] for date in aligned_dates]
        closes = [float(row.close) for row in asset]
        spy_closes = [float(row.close) for row in market]
        if min(closes) <= 0 or min(spy_closes) <= 0:
            continue
        dollar_volume = [float(row.close) * float(row.volume or 0) for row in asset[-20:]]
        if statistics.median(dollar_volume) < 50_000_000:
            continue
        daily = daily_returns(closes)
        spy_daily = daily_returns(spy_closes)
        mean = statistics.fmean(daily)
        spy_mean = statistics.fmean(spy_daily)
        spy_variance = sum((value - spy_mean) ** 2 for value in spy_daily)
        covariance = sum(
            (asset - mean) * (market - spy_mean)
            for asset, market in zip(daily, spy_daily)
        )
        beta = covariance / spy_variance if spy_variance > 0 else 0
        spy_five = spy_closes[-1] / spy_closes[-6] - 1
        residual = closes[-1] / closes[-6] - 1 - beta * spy_five
        eligible.append((ticker, residual))
    selected_count = min(20, math.ceil(len(eligible) * .1))
    selected = sorted(eligible, key=lambda row: (row[1], row[0]))[:selected_count]
    weight = min(10.0, 100 / len(selected)) if selected else 0
    return {
        "residuals": {ticker: value for ticker, value in sorted(eligible)},
        "weights_pct": {ticker: weight for ticker, _value in selected},
    }


def c3_exposure(spy_closes: list[float]) -> float:
    if len(spy_closes) < 127 or min(spy_closes[-127:]) <= 0:
        return 0.0
    if spy_closes[-1] / spy_closes[-127] - 1 <= 0:
        return 0.0
    daily = daily_returns(spy_closes[-21:])
    realized = statistics.stdev(daily) * math.sqrt(252) if len(daily) > 1 else 0
    return min(1.0, .10 / realized) if realized > 0 else 0.0


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    if set(p_values) != {"C1", "C2", "C3"}:
        raise ValueError("Holm correction requires exactly C1, C2, and C3")
    ordered = sorted(p_values.items(), key=lambda row: (row[1], row[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * float(value)))
        adjusted[name] = running
    return {name: adjusted[name] for name in sorted(adjusted)}


def _validate_preregistration(preregistration: dict[str, Any]) -> tuple[str, str]:
    dataset_hash = str(preregistration.get("dataset_hash") or "")
    membership_sha256 = str(preregistration.get("membership_sha256") or "")
    canonical = frozen_preregistration(dataset_hash, membership_sha256)
    if len(dataset_hash) != 64 or len(membership_sha256) != 64 or preregistration != canonical:
        raise ValueError("challenger preregistration contract invalid")
    return str(canonical["universe_id"]), membership_sha256


def _market(
    conn: Any,
    universe_id: str,
) -> tuple[list[str], dict[str, list[SessionPrice]], VerifiedMembership]:
    by_ticker: dict[str, list[SessionPrice]] = defaultdict(list)
    for ticker, date, open_, close, volume in conn.execute(
        """SELECT ticker,date,open,close,volume FROM price_daily
           ORDER BY ticker,date,open,close,volume"""
    ):
        name = str(ticker or "").strip()
        if name != "SPY" and (name in ML_CONTEXT_TICKERS or name.startswith("^")):
            continue
        by_ticker[name].append(SessionPrice(
            name, str(date), float(open_), float(close), volume=float(volume)
        ))
    sessions = [row.date for row in by_ticker.get("SPY", [])]
    membership = load_verified_membership(
        conn, universe_id, required_membership_dates(sessions),
    )
    return sessions, dict(by_ticker), membership


def _period_ends(sessions: list[str], period: str) -> list[str]:
    groups: dict[str, str] = {}
    for date in sessions:
        key = (
            date[:7]
            if period == "month"
            else f"{date[:4]}-W{dt.date.fromisoformat(date).isocalendar().week:02d}"
        )
        groups[key] = date
    return sorted(groups.values())


def required_membership_dates(sessions: list[str]) -> list[str]:
    """Return every date where a constituent-aware challenger forms a signal."""
    return sorted(set(_period_ends(sessions, "month")) | set(_period_ends(sessions, "week")))


def _capacity(rows: list[SessionPrice], signal_date: str) -> float:
    prior = [row.close * float(row.volume or 0) for row in rows if row.date <= signal_date][-20:]
    return statistics.median(prior) * .01 if prior else 0.0


def _decisions(
    challenger: str,
    sessions: list[str],
    by_ticker: dict[str, list[SessionPrice]],
    allowed_dates: set[str],
    membership: VerifiedMembership,
) -> list[ExecutionDecision]:
    session_index = {date: index for index, date in enumerate(sessions)}
    monthly = _period_ends(sessions, "month")
    signals = monthly if challenger in {"C1", "C3"} else _period_ends(sessions, "week")
    decisions: list[ExecutionDecision] = []
    for signal_date in signals:
        if signal_date not in allowed_dates or session_index[signal_date] + 1 >= len(sessions):
            continue
        entry_index = session_index[signal_date] + 1
        entry_date = sessions[entry_index]
        if challenger == "C2":
            exit_index = entry_index + 4
        else:
            next_signals = [date for date in monthly if date > signal_date]
            if not next_signals:
                continue
            exit_index = session_index[next_signals[0]] + 1
        if exit_index >= len(sessions) or sessions[exit_index] not in allowed_dates:
            continue
        exit_date = sessions[exit_index]
        active_members = membership.members_on(signal_date)
        tradable = {
            name: rows for name, rows in by_ticker.items()
            if name != "SPY" and name in active_members
        }
        history = {
            ticker: [row for row in rows if row.date <= signal_date]
            for ticker, rows in tradable.items()
        }
        if challenger == "C1":
            signal = c1_signal({ticker: [row.close for row in rows] for ticker, rows in history.items()})
            scores = signal["scores"]
            weights = signal["weights_pct"]
        elif challenger == "C2":
            spy = [row for row in by_ticker["SPY"] if row.date <= signal_date]
            signal = c2_signal(history, spy)
            scores = {ticker: -value for ticker, value in signal["residuals"].items()}
            weights = signal["weights_pct"]
        else:
            spy = [row.close for row in by_ticker["SPY"] if row.date <= signal_date]
            exposure = c3_exposure(spy) * 100
            scores, weights = ({"SPY": exposure}, {"SPY": exposure}) if exposure > 0 else ({}, {})
        for ticker, weight in sorted(weights.items()):
            rows = by_ticker[ticker]
            decisions.append(ExecutionDecision(
                decision_id=f"{challenger}:{signal_date}:{ticker}",
                ticker=ticker,
                direction="long",
                signal_date=signal_date,
                entry_date=entry_date,
                exit_date=exit_date,
                score=float(scores[ticker]),
                capacity_notional=_capacity(rows, signal_date),
                evidence_partition="validation",
                locked_weight_pct=float(weight),
                exit_at="close" if challenger == "C2" else "open",
            ))
    return decisions


def _drawdown(values: list[float]) -> float:
    peak, worst = (values[0] if values else 1.0), 0.0
    for value in values:
        peak = max(peak, value)
        worst = max(worst, (peak - value) / peak * 100 if peak else 0.0)
    return worst


def _bootstrap(active: list[float], seed: int) -> tuple[list[float] | None, float | None]:
    if len(active) < 42:
        return None, None
    rng, block, samples = random.Random(seed), 21, []
    starts = list(range(max(1, len(active) - block + 1)))
    for _ in range(1000):
        draw: list[float] = []
        while len(draw) < len(active):
            start = rng.choice(starts)
            draw.extend(active[start:start + block])
        samples.append(statistics.fmean(draw[:len(active)]))
    samples.sort()
    observed = statistics.fmean(active)
    p_value = sum(value <= 0 for value in samples) / len(samples)
    return [samples[24], samples[974]], max(.001, p_value) if observed > 0 else 1.0


def _execute(
    challenger: str,
    sessions: list[str],
    by_ticker: dict[str, list[SessionPrice]],
    membership: VerifiedMembership,
    allowed_dates: list[str],
    cost_bps: float,
) -> dict[str, Any]:
    allowed = set(allowed_dates)
    decisions = _decisions(challenger, sessions, by_ticker, allowed, membership)
    prices = [row for rows in by_ticker.values() for row in rows if row.date in allowed]
    config = BaselineConfig(
        initial_capital=1_000_000,
        max_positions=1 if challenger == "C3" else 20,
        position_pct=100 if challenger == "C3" else 10,
        max_name_pct=100 if challenger == "C3" else 10,
        max_adv_pct=1, entry_slippage_bps=cost_bps,
        exit_slippage_bps=cost_bps, commission_bps_per_side=0,
        minimum_commission_per_side=0, short_borrow_bps_annual=0,
        cash_rate_bps_annual=0, holding_sessions=21,
    )
    result = simulate_portfolio(decisions, prices, allowed_dates, config)
    curve = [dict(row) for row in result.equity_curve]
    equity_by_date = {
        str(row["date"]): float(row["equity"])
        for row in curve
        if row.get("date") is not None and row.get("equity") is not None
    }
    spy_map = {row.date: row.close for row in by_ticker["SPY"]}
    required_dates = list(allowed_dates)
    missing_equity = [date for date in required_dates if date not in equity_by_date]
    missing_spy = [date for date in required_dates if date not in spy_map]
    if missing_equity or missing_spy:
        raise ValueError(
            "challenger metrics require date-aligned strategy and SPY marks: "
            f"missing_equity={missing_equity[:5]}, missing_spy={missing_spy[:5]}"
        )
    equities = [equity_by_date[date] for date in required_dates]
    daily = daily_returns(equities)
    spy_values = [spy_map[date] for date in required_dates]
    spy_daily = daily_returns(spy_values)
    if len(daily) != len(spy_daily):
        raise ValueError("challenger strategy and SPY return counts are not aligned")
    active = [left - right for left, right in zip(daily, spy_daily)]
    ci, p_value = _bootstrap(active, int.from_bytes(challenger.encode(), "big"))
    net_return = (equities[-1] / equities[0] - 1) * 100 if len(equities) > 1 else None
    spy_return = (spy_values[-1] / spy_values[0] - 1) * 100 if len(spy_values) > 1 else None
    cagr = (
        ((equities[-1] / equities[0]) ** (252 / (len(equities) - 1)) - 1) * 100
        if len(equities) > 1 and equities[0] > 0 else None
    )
    volatility = statistics.stdev(daily) * math.sqrt(252) * 100 if len(daily) > 1 else None
    downside = [min(value, 0) for value in daily]
    downside_vol = statistics.pstdev(downside) if len(downside) > 1 else 0
    pnls = [float(row["net_pnl"]) for row in result.trades]
    positive = sum(value for value in pnls if value > 0)
    negative = abs(sum(value for value in pnls if value < 0))
    years: dict[str, float] = defaultdict(float)
    for row in result.trades:
        years[str(row["exit_date"])[:4]] += max(0.0, float(row["net_pnl"]))
    concentration = max(years.values(), default=0) / positive * 100 if positive else None
    exposure = [float(row["gross_exposure_pct"]) for row in curve if row.get("gross_exposure_pct") is not None]
    max_drawdown = _drawdown(equities)
    return {
        "metrics": {
            "net_total_return_pct": net_return,
            "cagr_pct": cagr,
            "spy_total_return_pct": spy_return,
            "net_active_return_pct": net_return - spy_return if net_return is not None and spy_return is not None else None,
            "volatility_pct": volatility,
            "sharpe": statistics.fmean(daily) / statistics.stdev(daily) * math.sqrt(252) if len(daily) > 1 and statistics.stdev(daily) else None,
            "sortino": statistics.fmean(daily) / downside_vol * math.sqrt(252) if downside_vol else None,
            "max_drawdown_pct": max_drawdown,
            "calmar": cagr / max_drawdown if cagr is not None and max_drawdown else None,
            "profit_factor": positive / negative if negative else None,
            "win_rate_pct": sum(value > 0 for value in pnls) / len(pnls) * 100 if pnls else None,
            "average_exposure_pct": statistics.fmean(exposure) if exposure else 0.0,
            "turnover_pct": sum(float(row.get("entry_notional") or 0) for row in result.trades) / 1_000_000 * 100,
            "trade_count": len(result.trades),
            "capacity_min_notional": min(
                (decision.capacity_notional for decision in decisions), default=None
            ),
            "maximum_adv_pct": 1.0,
            "one_way_cost_bps": cost_bps,
            "profit_concentration_pct": concentration,
            "active_daily_mean_block_ci95": ci,
            "active_return_p_value": p_value,
        },
        "equity_curve": curve,
        "ledger": result.ledger,
        "decision_count": len(decisions),
        "rejections": [dict(row) for row in result.exclusions],
    }


def _holdout_identity(
    preregistration: dict[str, Any],
    selected: str,
    heldout: list[str],
) -> str:
    payload = {
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "dataset_hash": preregistration["dataset_hash"],
        "challenger": selected,
        "config_sha256": preregistration["config_hashes"][selected],
        "window_start": heldout[0],
        "window_end": heldout[-1],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _ensure_holdout_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS challenger_holdout_access (
            access_id TEXT PRIMARY KEY,
            preregistration_sha256 TEXT NOT NULL,
            dataset_hash TEXT NOT NULL,
            challenger TEXT NOT NULL,
            config_sha256 TEXT NOT NULL,
            window_start TEXT NOT NULL,
            window_end TEXT NOT NULL,
            accessed_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS challenger_holdout_results (
            access_id TEXT PRIMARY KEY REFERENCES challenger_holdout_access(access_id),
            result_json TEXT NOT NULL,
            result_sha256 TEXT NOT NULL,
            completed_ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TRIGGER IF NOT EXISTS challenger_holdout_access_no_update
        BEFORE UPDATE ON challenger_holdout_access
        BEGIN SELECT RAISE(ABORT,'challenger holdout access is immutable'); END;
        CREATE TRIGGER IF NOT EXISTS challenger_holdout_access_no_delete
        BEFORE DELETE ON challenger_holdout_access
        BEGIN SELECT RAISE(ABORT,'challenger holdout access is immutable'); END;
        CREATE TRIGGER IF NOT EXISTS challenger_holdout_access_single_insert
        BEFORE INSERT ON challenger_holdout_access
        WHEN EXISTS (SELECT 1 FROM challenger_holdout_access)
        BEGIN SELECT RAISE(ABORT,'challenger holdout access is sealed'); END;
        CREATE TRIGGER IF NOT EXISTS challenger_holdout_results_no_update
        BEFORE UPDATE ON challenger_holdout_results
        BEGIN SELECT RAISE(ABORT,'challenger holdout result is immutable'); END;
        CREATE TRIGGER IF NOT EXISTS challenger_holdout_results_no_delete
        BEFORE DELETE ON challenger_holdout_results
        BEGIN SELECT RAISE(ABORT,'challenger holdout result is immutable'); END;
    """)


def _consume_holdout(
    conn: sqlite3.Connection,
    preregistration: dict[str, Any],
    selected: str,
    sessions: list[str],
    by_ticker: dict[str, list[SessionPrice]],
    membership: VerifiedMembership,
    heldout: list[str],
) -> dict[str, Any]:
    """Durably log access before reading, and return the sealed result on retry."""
    if conn.in_transaction:
        raise ValueError("sealed heldout access requires a caller-committed snapshot")
    access_id = _holdout_identity(preregistration, selected, heldout)
    _ensure_holdout_schema(conn)
    prior_access = conn.execute(
        """SELECT access_id,preregistration_sha256,dataset_hash,challenger,
                  config_sha256,window_start,window_end,accessed_ts
           FROM challenger_holdout_access ORDER BY accessed_ts,access_id"""
    ).fetchall()
    if prior_access and not any(str(row[0]) == access_id for row in prior_access):
        raise ValueError("sealed heldout window was already consumed by different evidence")
    stored = conn.execute(
        "SELECT result_json,result_sha256 FROM challenger_holdout_results WHERE access_id=?",
        (access_id,),
    ).fetchone()
    if stored:
        payload = json.loads(str(stored[0]))
        if hashlib.sha256(canonical_json_bytes(payload)).hexdigest() != str(stored[1]):
            raise ValueError("sealed heldout result hash mismatch")
        return payload
    if prior_access:
        raise ValueError("sealed heldout access is incomplete and cannot be repeated")
    conn.execute(
        """INSERT INTO challenger_holdout_access
           (access_id,preregistration_sha256,dataset_hash,challenger,config_sha256,
            window_start,window_end) VALUES (?,?,?,?,?,?,?)""",
        (
            access_id, preregistration["preregistration_sha256"],
            preregistration["dataset_hash"], selected,
            preregistration["config_hashes"][selected], heldout[0], heldout[-1],
        ),
    )
    conn.commit()
    accessed_ts = str(conn.execute(
        "SELECT accessed_ts FROM challenger_holdout_access WHERE access_id=?",
        (access_id,),
    ).fetchone()[0])

    spec = preregistration["challengers"][selected]
    cost = float(spec.get("selection_cost_bps", spec.get("one_way_cost_bps", 25)))
    stress_cost = max(float(value) for value in (
        spec.get("one_way_cost_scenarios_bps")
        or [cost, spec.get("stress_cost_bps", cost * 2)]
    ))
    selection_run = _execute(
        selected, sessions, by_ticker, membership, heldout, cost,
    )
    stress_run = _execute(
        selected, sessions, by_ticker, membership, heldout, stress_cost,
    )
    metrics = selection_run["metrics"]
    stress_metrics = stress_run["metrics"]
    gate_reasons = []
    active_return = metrics.get("net_active_return_pct")
    if active_return is None or float(active_return) <= 0:
        gate_reasons.append("sealed_holdout_active_return_not_positive")
    stress_return = stress_metrics.get("net_total_return_pct")
    if stress_return is None or float(stress_return) < 0:
        gate_reasons.append("sealed_holdout_double_cost_net_return_negative")
    concentration = metrics.get("profit_concentration_pct")
    if concentration is None or float(concentration) > float(
        preregistration["historical_shadow_gate"]["maximum_profit_concentration_pct"]
    ):
        gate_reasons.append("sealed_holdout_profit_concentration_exceeded")
    drawdown = metrics.get("max_drawdown_pct")
    if drawdown is None or float(drawdown) > float(
        preregistration["historical_shadow_gate"]["maximum_drawdown_pct"]
    ):
        gate_reasons.append("sealed_holdout_risk_rule_failed")
    result = {
        "access_id": access_id,
        "accessed_ts": accessed_ts,
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "dataset_hash": preregistration["dataset_hash"],
        "challenger": selected,
        "config_sha256": preregistration["config_hashes"][selected],
        "window_start": heldout[0],
        "window_end": heldout[-1],
        "selection_cost": selection_run,
        "stress_cost": stress_run,
        "gate_reasons": gate_reasons,
        "eligible_for_prospective_shadow": not gate_reasons,
        "reusable_for_policy_selection": False,
        "automatic_promotion": False,
    }
    result_sha = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    conn.execute(
        "INSERT INTO challenger_holdout_results (access_id,result_json,result_sha256) VALUES (?,?,?)",
        (access_id, canonical_json_bytes(result).decode(), result_sha),
    )
    conn.commit()
    return result


def execute_validation_tournament(
    conn: Any,
    preregistration: dict[str, Any],
    *,
    consume_heldout: bool = False,
) -> dict[str, Any]:
    """Run development-informed chronological validation without held-out access."""
    universe_id, expected_membership_sha256 = _validate_preregistration(preregistration)
    sessions, by_ticker, membership = _market(conn, universe_id)
    if membership.contract["membership_sha256"] != expected_membership_sha256:
        raise ValueError("point-in-time membership changed after preregistration")
    first, second = int(len(sessions) * .6), int(len(sessions) * .8)
    purge = int(preregistration["selection"]["purge_sessions"])
    development = sessions[:max(0, first - purge)]
    validation = sessions[min(len(sessions), first + purge):max(first + purge, second - purge)]
    heldout = sessions[min(len(sessions), second + purge):]
    if not development or not validation or not heldout:
        raise ValueError("chronological partitions are too short after purge and embargo")
    results: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    for name in ("C1", "C2", "C3"):
        spec = preregistration["challengers"][name]
        cost = float(spec.get("selection_cost_bps", spec.get("one_way_cost_bps", 25)))
        scenario_costs = spec.get("one_way_cost_scenarios_bps") or [
            cost, float(spec.get("stress_cost_bps", cost * 2)),
        ]
        scenario_runs = {
            float(value): _execute(
                name, sessions, by_ticker, membership, validation, float(value),
            )
            for value in sorted(set(scenario_costs))
        }
        base = scenario_runs[cost]
        cost_scenarios = {
            str(value): run["metrics"] for value, run in scenario_runs.items()
        }
        stress = scenario_runs[max(scenario_runs)]["metrics"]
        fold_ranges = [
            validation[index * len(validation) // 5:(index + 1) * len(validation) // 5]
            for index in range(5)
        ]
        fold_ranges = [part for part in fold_ranges if len(part) >= 42]
        folds = [{
            "training_start": development[0],
            "training_end": part[0],
            "validation_start": part[0],
            "validation_end": part[-1],
            "metrics": _execute(
                name, sessions, by_ticker, membership, part, cost,
            )["metrics"],
        } for part in fold_ranges]
        positive_pct = (
            sum(
                float(row["metrics"].get("net_active_return_pct") or 0) > 0
                for row in folds
            ) / len(folds) * 100
            if folds else 0.0
        )
        metrics = base["metrics"]
        reasons = []
        if positive_pct < 70:
            reasons.append("positive_active_return_in_fewer_than_70_pct_of_folds")
        stress_return = stress.get("net_total_return_pct")
        if stress_return is None or float(stress_return) < 0:
            reasons.append("double_cost_net_return_negative")
        concentration = metrics.get("profit_concentration_pct")
        if concentration is None or float(concentration) > 50:
            reasons.append("profit_concentration_exceeds_50_pct")
        drawdown = metrics.get("max_drawdown_pct")
        maximum_drawdown = float(
            preregistration["historical_shadow_gate"]["maximum_drawdown_pct"]
        )
        if drawdown is None or float(drawdown) > maximum_drawdown:
            reasons.append("maximum_drawdown_exceeds_25_pct")
        p_values[name] = float(metrics.get("active_return_p_value") or 1.0)
        results[name] = {
            "status": "validation_complete",
            "config_sha256": preregistration["config_hashes"][name],
            **base,
            "cost_scenarios": cost_scenarios,
            "walk_forward_folds": folds,
            "positive_net_active_return_fold_pct": positive_pct,
            "gate_reasons": reasons,
        }
    adjusted = holm_adjust(p_values)
    for name, value in adjusted.items():
        significance = float(
            preregistration["historical_shadow_gate"]["holm_adjusted_p_value_max"]
        )
        if value > significance:
            results[name]["gate_reasons"].append("holm_adjusted_active_return_not_significant")
        results[name]["holm_adjusted_p_value"] = value
        results[name]["eligible_for_selection"] = not results[name]["gate_reasons"]
    qualified = [name for name in results if results[name]["eligible_for_selection"]]
    selected = max(
        qualified,
        key=lambda name: (float(results[name]["metrics"]["net_active_return_pct"]), name),
        default=None,
    )
    heldout_result = (
        _consume_holdout(
            conn, preregistration, selected, sessions, by_ticker, membership, heldout
        )
        if selected and consume_heldout else None
    )
    access_log = []
    if heldout_result is not None:
        access_log = [{
            key: heldout_result[key]
            for key in (
                "access_id", "accessed_ts", "preregistration_sha256", "dataset_hash",
                "challenger", "config_sha256", "window_start", "window_end",
            )
        }]
    return {
        "split": {
            "development": {"start": development[0], "end": development[-1], "session_count": len(development)},
            "validation": {"start": validation[0], "end": validation[-1], "session_count": len(validation)},
            "heldout": {"start": heldout[0], "end": heldout[-1], "session_count": len(heldout)},
            "purge_sessions": purge, "embargo_sessions": purge,
        },
        "challenger_results": results,
        "holm_adjusted_p_values": adjusted,
        "selected_challenger": selected,
        "sealed_heldout": {
            "accessed": heldout_result is not None,
            "access_log": access_log,
            "result": heldout_result,
            "reusable_for_policy_selection": False,
        },
        "prospective_shadow_candidate": (
            selected
            if heldout_result is not None
            and heldout_result["eligible_for_prospective_shadow"]
            else None
        ),
    }
