"""Locked next-open research baseline built on one deterministic execution seam.

``simulate_portfolio`` knows nothing about SQLite, reports, partitions, or UI
artifacts. Campaign replay and challenger research can therefore use the same
accounting path without growing another simulator.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from trader_koo.report.runs import current_code_version

from trader_koo.ml.features import ML_CONTEXT_TICKERS

SCHEMA_VERSION = "2.0"
METHOD = "setup_calls_next_open_to_tenth_close"
IMPLEMENTATION_PATH = Path(__file__)
PACKAGED_ARTIFACT_PATH = Path(__file__).with_name("next_open_baseline_artifact_20260823.json")
RUNTIME_ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "data/research/next_open_baseline_latest.json"


@dataclasses.dataclass(frozen=True)
class BaselineConfig:
    initial_capital: float = 1_000_000.0
    max_positions: int = 20
    position_pct: float = 5.0
    max_name_pct: float = 10.0
    max_adv_pct: float = 1.0
    entry_slippage_bps: float = 10.0
    exit_slippage_bps: float = 10.0
    commission_bps_per_side: float = 1.0
    minimum_commission_per_side: float = 1.0
    short_borrow_bps_annual: float | None = 50.0
    cash_rate_bps_annual: float | None = 0.0
    holding_sessions: int = 10
    minimum_score: float = 0.0

    def validate(self) -> None:
        if self.initial_capital <= 0 or self.max_positions < 1:
            raise ValueError("capital and max_positions must be positive")
        if not 0 < self.position_pct <= self.max_name_pct <= 100:
            raise ValueError("position_pct must be positive and no larger than max_name_pct")
        if not 0 < self.max_adv_pct <= 100:
            raise ValueError("max_adv_pct must be in (0, 100]")
        if not 2 <= self.holding_sessions <= 60:
            raise ValueError("holding_sessions must be between 2 and 60")


@dataclasses.dataclass(frozen=True)
class ExecutionDecision:
    decision_id: str
    ticker: str
    direction: str
    signal_date: str
    entry_date: str
    exit_date: str
    score: float
    capacity_notional: float
    evidence_partition: str = "development"
    locked_notional: float | None = None
    metadata: tuple[tuple[str, Any], ...] = ()
    stop_loss: float | None = None
    target_price: float | None = None
    max_holding_sessions: int | None = None
    locked_weight_pct: float | None = None
    exit_at: str = "close"


@dataclasses.dataclass(frozen=True)
class SessionPrice:
    ticker: str
    date: str
    open: float | None
    close: float | None
    high: float | None = None
    low: float | None = None
    volume: float | None = None


@dataclasses.dataclass(frozen=True)
class ExecutionResult:
    trades: tuple[dict[str, Any], ...]
    exclusions: tuple[dict[str, Any], ...]
    equity_curve: tuple[dict[str, Any], ...]
    financing_priced: bool
    initial_equity: float
    final_equity: float | None
    max_name_weight_pct: float
    max_gross_exposure_pct: float
    open_positions: tuple[dict[str, Any], ...]
    ledger: dict[str, Any]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _canonical(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("research artifacts cannot contain NaN or Infinity")
        return round(value, 10)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported artifact value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _implementation_hash() -> str:
    return hashlib.sha256(IMPLEMENTATION_PATH.read_bytes()).hexdigest()


def _commission(notional: float, config: BaselineConfig) -> float:
    return max(config.minimum_commission_per_side, notional * config.commission_bps_per_side / 10_000)


def _adverse(price: float, direction: str, bps: float, *, entry: bool) -> float:
    sign = 1 if (direction == "long") == entry else -1
    return price * (1 + sign * bps / 10_000)


def _pnl(direction: str, entry: float, exit_: float, shares: int) -> float:
    value = (exit_ - entry) * shares
    return value if direction == "long" else -value


def simulate_portfolio(
    decisions: Iterable[ExecutionDecision],
    prices: Iterable[SessionPrice],
    sessions: Iterable[str],
    config: BaselineConfig,
) -> ExecutionResult:
    """Execute immutable decisions through the canonical portfolio ledger."""
    config.validate()
    session_list = tuple(sorted(set(sessions)))
    price_rows = tuple(sorted(prices, key=lambda row: (row.date, row.ticker)))
    price_map = {(row.ticker, row.date): row for row in price_rows}
    ordered = tuple(sorted(
        decisions,
        key=lambda row: (row.entry_date, -row.score, row.ticker, row.decision_id),
    ))
    if len({row.decision_id for row in ordered}) != len(ordered):
        raise ValueError("decision_id must be unique")
    if any(not (row.signal_date < row.entry_date <= row.exit_date) for row in ordered):
        raise ValueError("decisions must satisfy signal_date < entry_date <= exit_date")
    if any(row.exit_at not in {"open", "close"} for row in ordered):
        raise ValueError("exit_at must be open or close")
    if any(row.exit_at == "open" and row.entry_date == row.exit_date for row in ordered):
        raise ValueError("next-open exits require entry_date < exit_date")
    if any(
        row.locked_weight_pct is not None
        and not (0 < row.locked_weight_pct <= 100)
        for row in ordered
    ):
        raise ValueError("locked_weight_pct must be in (0, 100]")
    by_entry: dict[str, list[ExecutionDecision]] = defaultdict(list)
    exclusions: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    cash_events: list[dict[str, Any]] = [{
        "date": session_list[0] if session_list else None,
        "event_type": "initial_capital",
        "amount": config.initial_capital,
    }]
    position_snapshots: list[dict[str, Any]] = []

    def reject(decision: ExecutionDecision, reason: str, date: str | None = None) -> None:
        exclusions.append({"decision_id": decision.decision_id, "reason": reason})
        orders.append({
            "order_id": f"{decision.decision_id}:entry",
            "decision_id": decision.decision_id,
            "date": date or decision.entry_date,
            "ticker": decision.ticker,
            "direction": decision.direction,
            "order_type": "entry",
            "status": "rejected",
            "reason": reason,
        })

    for decision in ordered:
        if decision.entry_date not in session_list or decision.exit_date not in session_list:
            reject(decision, "execution_session_missing")
            continue
        if not math.isfinite(decision.capacity_notional) or decision.capacity_notional <= 0:
            reject(decision, "capacity_unavailable")
            continue
        by_entry[decision.entry_date].append(decision)

    cash = config.initial_capital
    positions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    financing_priced = config.cash_rate_bps_annual is not None
    max_name = max_gross = 0.0

    def mark(date: str, field: str) -> tuple[float | None, dict[str, float]]:
        total = cash
        exposure: dict[str, float] = defaultdict(float)
        for position in positions:
            row = price_map.get((position["ticker"], date))
            price = _finite(getattr(row, field, None))
            if price is None or price <= 0:
                return None, {}
            exposure[position["ticker"]] += position["shares"] * price
            total += position["shares"] * price if position["direction"] == "long" else (
                position["reserve"] + (position["entry_price"] - price) * position["shares"]
            )
        return total, dict(exposure)

    def close_position(
        position: dict[str, Any], raw_exit: float, date: str, reason: str
    ) -> None:
        nonlocal cash
        exit_price = _adverse(
            raw_exit, position["direction"], config.exit_slippage_bps, entry=False
        )
        gross = _pnl(
            position["direction"], position["entry_price"], exit_price,
            position["shares"],
        )
        exit_commission = _commission(exit_price * position["shares"], config)
        cash_delta = position["reserve"] + gross - exit_commission
        cash += cash_delta
        orders.append({
            "order_id": f"{position['decision_id']}:exit",
            "decision_id": position["decision_id"],
            "date": date,
            "ticker": position["ticker"],
            "direction": position["direction"],
            "order_type": "exit",
            "status": "filled",
            "reason": reason,
        })
        fills.append({
            "fill_id": f"{position['decision_id']}:exit",
            "order_id": f"{position['decision_id']}:exit",
            "decision_id": position["decision_id"],
            "date": date,
            "side": "sell" if position["direction"] == "long" else "buy_to_cover",
            "price": exit_price,
            "shares": position["shares"],
            "commission": exit_commission,
            "reason": reason,
        })
        cash_events.append({
            "date": date, "event_type": "exit_settlement",
            "decision_id": position["decision_id"], "amount": cash_delta,
        })
        trades.append({
            **{key: value for key, value in position.items() if key != "reserve"},
            "exit_date": date,
            "exit_price": exit_price,
            "exit_reason": reason,
            "entry_notional": position["reserve"],
            "gross_pnl": gross,
            "commission": position["entry_commission"] + exit_commission,
            "net_pnl": (
                gross - position["entry_commission"] - exit_commission
                - position["borrow"]
            ),
        })

    previous: str | None = None
    for date in session_list:
        # Accrue the interval ending at today's open. Borrow changes both daily
        # equity and subsequent sizing; it is not deferred until trade close.
        if previous is not None:
            if config.cash_rate_bps_annual is None:
                financing_priced = False
            else:
                interest = max(cash, 0) * config.cash_rate_bps_annual / 10_000 / 252
                cash += interest
                if interest:
                    cash_events.append({
                        "date": date, "event_type": "cash_interest", "amount": interest,
                    })
            for position in positions:
                if position["direction"] != "short":
                    continue
                if config.short_borrow_bps_annual is None:
                    financing_priced = False
                    continue
                prior_row = price_map.get((position["ticker"], previous))
                prior_close = _finite(prior_row.close if prior_row else None)
                if prior_close is None or prior_close <= 0:
                    financing_priced = False
                    continue
                marked_short_value = position["shares"] * prior_close
                charge = marked_short_value * config.short_borrow_bps_annual / 10_000 / 252
                cash -= charge
                position["borrow"] += charge
                cash_events.append({
                    "date": date, "event_type": "short_borrow",
                    "decision_id": position["decision_id"], "amount": -charge,
                })

        # Rebalances and barrier-managed campaign positions release capital
        # before today's open-order admissions. Gap exits use the open;
        # intraday collisions resolve stop-first, the conservative choice.
        survivors: list[dict[str, Any]] = []
        for position in positions:
            if position["exit_date"] != date or position["exit_at"] != "open":
                survivors.append(position)
                continue
            row = price_map.get((position["ticker"], date))
            raw_open = _finite(row.open if row else None)
            if raw_open is None or raw_open <= 0:
                exclusions.append({
                    "decision_id": position["decision_id"],
                    "reason": "exact_exit_open_missing",
                })
                orders.append({
                    "order_id": f"{position['decision_id']}:exit",
                    "decision_id": position["decision_id"],
                    "date": date,
                    "ticker": position["ticker"],
                    "direction": position["direction"],
                    "order_type": "exit",
                    "status": "rejected",
                    "reason": "exact_exit_open_missing",
                })
                survivors.append(position)
                continue
            close_position(position, raw_open, date, "scheduled_open")
        positions = survivors

        survivors = []
        for position in positions:
            if position.get("max_holding_sessions") is None:
                survivors.append(position)
                continue
            if date <= position["entry_date"]:
                survivors.append(position)
                continue
            position["bars_held"] += 1
            row = price_map.get((position["ticker"], date))
            open_ = _finite(row.open if row else None)
            close = _finite(row.close if row else None)
            high = _finite(row.high if row else None)
            low = _finite(row.low if row else None)
            if open_ is None or close is None or high is None or low is None:
                survivors.append(position)
                continue
            direction = position["direction"]
            stop = _finite(position.get("stop_loss"))
            target = _finite(position.get("target_price"))
            reason: str | None = None
            raw_exit: float | None = None
            if direction == "long" and stop is not None and target is not None:
                if open_ <= stop:
                    reason, raw_exit = "stopped_out", open_
                elif open_ >= target:
                    reason, raw_exit = "target_hit", open_
                elif low <= stop:
                    reason, raw_exit = "stopped_out", stop
                elif high >= target:
                    reason, raw_exit = "target_hit", target
            elif direction == "short" and stop is not None and target is not None:
                if open_ >= stop:
                    reason, raw_exit = "stopped_out", open_
                elif open_ <= target:
                    reason, raw_exit = "target_hit", open_
                elif high >= stop:
                    reason, raw_exit = "stopped_out", stop
                elif low <= target:
                    reason, raw_exit = "target_hit", target
            if (
                reason is None
                and position["bars_held"] >= position["max_holding_sessions"]
            ):
                reason, raw_exit = "expired", close
            if reason is None or raw_exit is None:
                survivors.append(position)
            else:
                close_position(position, raw_exit, date, reason)
        positions = survivors

        # Today's close proceeds do not exist while open orders are admitted.
        adv_used: dict[str, float] = defaultdict(float)
        for decision in by_entry.get(date, []):
            if decision.direction not in {"long", "short"}:
                reject(decision, "unsupported_direction", date)
                continue
            row = price_map.get((decision.ticker, date))
            raw_open = _finite(row.open if row else None)
            if raw_open is None or raw_open <= 0:
                reject(decision, "immediate_next_session_open_missing", date)
                continue
            if len(positions) >= config.max_positions:
                reject(decision, "position_capacity_full", date)
                continue
            equity, exposure = mark(date, "open")
            if equity is None or equity <= 0:
                reject(decision, "current_equity_unpriceable", date)
                continue
            entry_price = _adverse(raw_open, decision.direction, config.entry_slippage_bps, entry=True)
            name_room = max(0.0, equity * config.max_name_pct / 100 - exposure.get(decision.ticker, 0))
            adv_room = max(0.0, decision.capacity_notional - adv_used[decision.ticker])
            target = min(
                decision.locked_notional
                if decision.locked_notional is not None
                else equity * (
                    decision.locked_weight_pct
                    if decision.locked_weight_pct is not None
                    else config.position_pct
                ) / 100,
                name_room,
                adv_room,
            )
            # The name cap is measured on marked gross exposure. For a short,
            # adverse entry slippage lowers proceeds, so sizing only from the
            # fill price can exceed the cap at the contemporaneous open mark.
            shares = int(target / max(raw_open, entry_price))
            if shares < 1:
                reject(decision, "capacity_below_one_share", date)
                continue
            reserve = shares * entry_price
            commission = _commission(reserve, config)
            while shares > 0:
                slippage_loss = abs(entry_price - raw_open) * shares
                post_fill_equity = equity - commission - slippage_loss
                post_fill_name = exposure.get(decision.ticker, 0) + shares * raw_open
                within_name_cap = (
                    post_fill_equity > 0
                    and post_fill_name / post_fill_equity <= config.max_name_pct / 100
                )
                if reserve + commission <= cash and within_name_cap:
                    break
                shares -= 1
                reserve = shares * entry_price
                commission = _commission(reserve, config) if shares else 0.0
            if shares < 1:
                reject(decision, "insufficient_cash", date)
                continue
            cash -= reserve + commission
            orders.append({
                "order_id": f"{decision.decision_id}:entry",
                "decision_id": decision.decision_id,
                "date": date,
                "ticker": decision.ticker,
                "direction": decision.direction,
                "order_type": "entry",
                "status": "filled",
                "target_notional": target,
                "capacity_notional": decision.capacity_notional,
            })
            fills.append({
                "fill_id": f"{decision.decision_id}:entry",
                "order_id": f"{decision.decision_id}:entry",
                "decision_id": decision.decision_id,
                "date": date,
                "side": "buy" if decision.direction == "long" else "sell_short",
                "price": entry_price,
                "shares": shares,
                "commission": commission,
                "reason": "next_open",
            })
            cash_events.append({
                "date": date, "event_type": "entry_reserve",
                "decision_id": decision.decision_id,
                "amount": -(reserve + commission),
            })
            adv_used[decision.ticker] += reserve
            positions.append({
                **dict(decision.metadata),
                "decision_id": decision.decision_id,
                "ticker": decision.ticker,
                "direction": decision.direction,
                "signal_date": decision.signal_date,
                "entry_date": decision.entry_date,
                "exit_date": decision.exit_date,
                "evidence_partition": decision.evidence_partition,
                "entry_price": entry_price,
                "shares": shares,
                "reserve": reserve,
                "entry_commission": commission,
                "borrow": 0.0,
                "stop_loss": decision.stop_loss,
                "target_price": decision.target_price,
                "max_holding_sessions": decision.max_holding_sessions,
                "locked_weight_pct": decision.locked_weight_pct,
                "exit_at": decision.exit_at,
                "bars_held": 0,
            })
            opening_equity, opening_exposure = mark(date, "open")
            if opening_equity is not None and opening_equity > 0:
                opening_gross = sum(opening_exposure.values())
                opening_name = max(opening_exposure.values(), default=0)
                max_gross = max(max_gross, opening_gross / opening_equity)
                max_name = max(max_name, opening_name / opening_equity)

        remaining: list[dict[str, Any]] = []
        for position in positions:
            if position["exit_date"] != date or position["exit_at"] != "close":
                remaining.append(position)
                continue
            row = price_map.get((position["ticker"], date))
            raw_close = _finite(row.close if row else None)
            if raw_close is None or raw_close <= 0:
                exclusions.append({
                    "decision_id": position["decision_id"],
                    "reason": "exact_exit_close_missing",
                })
                orders.append({
                    "order_id": f"{position['decision_id']}:exit",
                    "decision_id": position["decision_id"],
                    "date": date,
                    "ticker": position["ticker"],
                    "direction": position["direction"],
                    "order_type": "exit",
                    "status": "rejected",
                    "reason": "exact_exit_close_missing",
                })
                remaining.append(position)
                continue
            close_position(position, raw_close, date, "scheduled_close")
        positions = remaining

        equity, exposure = mark(date, "close")
        if equity is None:
            curve.append({"date": date, "equity": None, "status": "unpriceable_mark"})
        else:
            gross = sum(exposure.values())
            name = max(exposure.values(), default=0)
            net = sum(
                (1 if position["direction"] == "long" else -1)
                * position["shares"]
                * float(exposure.get(position["ticker"], 0))
                / max(
                    1,
                    sum(
                        item["shares"] for item in positions
                        if item["ticker"] == position["ticker"]
                    ),
                )
                for position in positions
            )
            if equity > 0:
                max_gross, max_name = max(max_gross, gross / equity), max(max_name, name / equity)
            curve.append({
                "date": date, "equity": equity, "cash": cash, "open_positions": len(positions),
                "gross_exposure_pct": gross / equity * 100 if equity > 0 else None,
                "net_exposure_pct": net / equity * 100 if equity > 0 else None,
            })
        position_snapshots.append({
            "date": date,
            "positions": [{
                **{key: value for key, value in position.items() if key != "reserve"},
                "reserved_cash": position["reserve"],
                "mark_price": (
                    _finite(price_map[(position["ticker"], date)].close)
                    if (position["ticker"], date) in price_map else None
                ),
            } for position in positions],
        })
        previous = date

    valid = [float(row["equity"]) for row in curve if row.get("equity") is not None]
    decisions_payload = [dataclasses.asdict(row) for row in ordered]
    config_payload = dataclasses.asdict(config)
    input_payload = {
        "decisions": decisions_payload,
        "prices": [dataclasses.asdict(row) for row in price_rows],
        "sessions": session_list,
        "config": config_payload,
    }
    open_positions = [
        {key: value for key, value in row.items() if key != "reserve"}
        for row in positions
    ]
    components = {
        "config": config_payload,
        "market_data": {
            "prices": [dataclasses.asdict(row) for row in price_rows],
            "sessions": session_list,
        },
        "decisions": decisions_payload,
        "orders": orders,
        "fills": fills,
        "cash": {
            "initial": config.initial_capital,
            "final": cash,
            "events": cash_events,
        },
        "positions": position_snapshots,
        "equity": curve,
        "costs": {
            "commissions": sum(float(row["commission"]) for row in fills),
            "short_borrow": -sum(
                float(row["amount"]) for row in cash_events
                if row["event_type"] == "short_borrow"
            ),
            "cash_interest": sum(
                float(row["amount"]) for row in cash_events
                if row["event_type"] == "cash_interest"
            ),
        },
        "rejections": exclusions,
        "open_positions": open_positions,
    }
    component_hashes = {key: _sha256(value) for key, value in components.items()}
    ledger_body = _canonical({
        "schema_version": "portfolio-ledger-v1",
        "engine_version": "portfolio-execution-v1.0",
        **components,
        "provenance": {
            "config_sha256": _sha256(config_payload),
            "input_sha256": _sha256(input_payload),
            "component_sha256": component_hashes,
        },
    })
    ledger = {
        **ledger_body,
        "provenance": {
            **ledger_body["provenance"],
            "ledger_sha256": _sha256(ledger_body),
        },
    }
    return ExecutionResult(
        tuple(trades), tuple(exclusions), tuple(curve), financing_priced,
        config.initial_capital, valid[-1] if valid else None, max_name * 100,
        max_gross * 100,
        tuple(open_positions), ledger,
    )

def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _price_rows(
    conn: sqlite3.Connection, tickers: list[str]
) -> tuple[list[SessionPrice], dict[tuple[str, str], float | None]]:
    required = {"ticker", "date", "open", "close", "volume"}
    if not _table_exists(conn, "price_daily") or not required.issubset(_columns(conn, "price_daily")):
        return [], {}
    placeholders = ",".join("?" for _ in tickers)
    rows = conn.execute(
        "SELECT ticker,date,CAST(open AS REAL),CAST(close AS REAL),CAST(volume AS REAL) "
        f"FROM price_daily WHERE ticker IN ({placeholders}) AND date IS NOT NULL ORDER BY ticker,date",
        tuple(tickers),
    ).fetchall()
    prices, volumes = [], {}
    for ticker, date, open_, close, volume in rows:
        key = str(ticker).upper(), str(date)
        prices.append(SessionPrice(*key, _finite(open_), _finite(close)))
        volumes[key] = _finite(volume)
    return prices, volumes


def _verified_calls(
    conn: sqlite3.Connection, report_dir: Path | None
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    eval_required = {"id", "asof_date", "ticker", "call_direction", "score", "report_run_id"}
    if (
        not _table_exists(conn, "setup_call_evaluations")
        or not eval_required.issubset(_columns(conn, "setup_call_evaluations"))
    ):
        return [], ["report_publication_contract_unavailable"], {}
    if report_dir is None:
        return [], ["report_artifact_directory_unconfigured"], {}
    columns = _columns(conn, "setup_call_evaluations")
    optional = [f"e.{name}" if name in columns else f"NULL AS {name}" for name in ("setup_family", "setup_tier")]
    rows = conn.execute(f"""
        SELECT e.id,e.asof_date,e.ticker,e.call_direction,e.score,{','.join(optional)},e.report_run_id
        FROM setup_call_evaluations e
        ORDER BY e.asof_date,e.score DESC,e.ticker,e.id
    """).fetchall()
    if any(row[7] is None or not str(row[7]).strip() for row in rows):
        return [], ["setup_call_report_run_id_missing"], {}
    try:
        from trader_koo.report.runs import resolve_published_report
    except ImportError:
        return [], ["report_publication_contract_unavailable"], {}
    run_ids = sorted({str(row[7]).strip() for row in rows})
    lineage: dict[str, Any] = {}
    for run_id in run_ids:
        resolved = resolve_published_report(
            conn, report_dir=report_dir, run_id=run_id, require_current=False
        )
        if resolved is None:
            return [], ["setup_call_report_publication_unresolved"], {}
        report = resolved[1].get("report_run", {})
        lineage[run_id] = {
            key: report.get(key) for key in (
                "run_id", "content_hash", "markdown_hash", "config_hash",
                "code_version", "generated_ts", "generation_key",
            )
        }
    return [{
        "call_id": int(row[0]), "signal_date": str(row[1]), "ticker": str(row[2]).upper(),
        "direction": str(row[3]).lower(), "score": _finite(row[4]), "setup_family": row[5],
        "setup_tier": row[6], "report_run_id": row[7],
    } for row in rows], [], lineage


def _price_contract(conn: sqlite3.Connection, tickers: list[str]) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        from trader_koo.db.price_contract import research_price_contract
    except ImportError:
        return None, ["research_price_basis_contract_unavailable"]
    contract = research_price_contract(conn, tickers)
    return (contract, []) if contract.get("eligible") else (
        contract, [str(contract.get("reason") or "research_price_basis_unverified")]
    )


def _split_dates(dates: list[str], sessions: list[str], holding: int) -> dict[str, Any]:
    if not dates:
        return {"development": [], "validation": [], "heldout": [], "purge_sessions": holding, "embargo_sessions": holding}
    dev_end, val_end = max(1, int(len(dates) * .6)), max(1, int(len(dates) * .8))
    development, validation, heldout = dates[:dev_end], dates[dev_end:val_end], dates[val_end:]
    indices = {date: index for index, date in enumerate(sessions)}

    def purge(values: list[str], prior: list[str]) -> list[str]:
        if not values or not prior:
            return values
        if prior[-1] not in indices:
            return []
        boundary = indices[prior[-1]]
        return [date for date in values if date in indices and indices[date] - boundary > holding]

    validation = purge(validation, development)
    heldout = purge(heldout, validation or development)
    return {"development": development, "validation": validation, "heldout": heldout, "purge_sessions": holding, "embargo_sessions": holding}


def _ensure_consumption_schema(conn: sqlite3.Connection) -> None:
    statements = (
        """CREATE TABLE IF NOT EXISTS research_holdout_consumptions (
            consumption_id TEXT PRIMARY KEY, method TEXT NOT NULL, window_start TEXT NOT NULL,
            window_end TEXT NOT NULL, config_hash TEXT NOT NULL, input_hash TEXT NOT NULL,
            reusable_for_policy_selection INTEGER NOT NULL DEFAULT 0)""",
        """CREATE TABLE IF NOT EXISTS research_holdout_dates (
            signal_date TEXT PRIMARY KEY,
            consumption_id TEXT NOT NULL REFERENCES research_holdout_consumptions(consumption_id))""",
        """CREATE TRIGGER IF NOT EXISTS research_holdout_consumptions_no_update BEFORE UPDATE
        ON research_holdout_consumptions BEGIN SELECT RAISE(ABORT,'holdout consumption is immutable'); END""",
        """CREATE TRIGGER IF NOT EXISTS research_holdout_consumptions_no_delete BEFORE DELETE
        ON research_holdout_consumptions BEGIN SELECT RAISE(ABORT,'holdout consumption is immutable'); END""",
        """CREATE TRIGGER IF NOT EXISTS research_holdout_dates_no_update BEFORE UPDATE
        ON research_holdout_dates BEGIN SELECT RAISE(ABORT,'holdout dates are immutable'); END""",
        """CREATE TRIGGER IF NOT EXISTS research_holdout_dates_no_delete BEFORE DELETE
        ON research_holdout_dates BEGIN SELECT RAISE(ABORT,'holdout dates are immutable'); END""",
    )
    for statement in statements:
        conn.execute(statement)


def _seal_holdout_inserts(conn: sqlite3.Connection) -> None:
    """Make the first complete heldout seal append-immutable."""
    statements = (
        """CREATE TRIGGER IF NOT EXISTS research_holdout_consumptions_no_insert BEFORE INSERT
        ON research_holdout_consumptions
        WHEN EXISTS (SELECT 1 FROM research_holdout_consumptions)
        BEGIN SELECT RAISE(ABORT,'holdout consumption is sealed'); END""",
        """CREATE TRIGGER IF NOT EXISTS research_holdout_dates_no_insert BEFORE INSERT
        ON research_holdout_dates
        BEGIN SELECT RAISE(ABORT,'holdout dates are sealed'); END""",
    )
    for statement in statements:
        conn.execute(statement)


def _record_holdout(conn: sqlite3.Connection, dates: list[str], config_hash: str, input_hash: str) -> dict[str, Any]:
    if not dates:
        return {"consumed": False, "reusable_for_policy_selection": False, "status": "heldout_window_unavailable"}
    _ensure_consumption_schema(conn)
    identity = _sha256({"method": METHOD, "dates": dates, "config_hash": config_hash, "input_hash": input_hash})
    existing = conn.execute("SELECT consumption_id,method,config_hash,input_hash FROM research_holdout_consumptions").fetchall()
    if existing:
        if not any(tuple(map(str, row)) == (identity, METHOD, config_hash, input_hash) for row in existing):
            raise ValueError("heldout observations were already consumed by different immutable inputs or method")
        stored = [str(row[0]) for row in conn.execute(
            "SELECT signal_date FROM research_holdout_dates WHERE consumption_id=? ORDER BY signal_date", (identity,)
        )]
        if stored != dates:
            raise ValueError("heldout partition changed after consumption")
    else:
        conn.execute("INSERT INTO research_holdout_consumptions VALUES (?,?,?,?,?,?,0)",
                     (identity, METHOD, dates[0], dates[-1], config_hash, input_hash))
        conn.executemany("INSERT INTO research_holdout_dates VALUES (?,?)", ((date, identity) for date in dates))
    _seal_holdout_inserts(conn)
    return {"consumed": True, "reusable_for_policy_selection": False,
            "status": "sealed_once_not_reusable_for_policy_selection", "consumption_id": identity,
            "window_start": dates[0], "window_end": dates[-1]}


def _guard_consumed_holdout(
    conn: sqlite3.Connection, dates: list[str], config_hash: str, input_hash: str
) -> None:
    """Prevent history growth from relabeling any previously viewed holdout."""
    if not _table_exists(conn, "research_holdout_consumptions"):
        return
    existing = conn.execute(
        "SELECT consumption_id,method,config_hash,input_hash FROM research_holdout_consumptions"
    ).fetchall()
    if not existing:
        return
    identity = _sha256({"method": METHOD, "dates": dates, "config_hash": config_hash, "input_hash": input_hash})
    if not any(tuple(map(str, row)) == (identity, METHOD, config_hash, input_hash) for row in existing):
        raise ValueError("heldout observations were already consumed by different immutable inputs or method")
    stored = [str(row[0]) for row in conn.execute(
        "SELECT signal_date FROM research_holdout_dates WHERE consumption_id=? ORDER BY signal_date", (identity,)
    )]
    if stored != dates:
        raise ValueError("heldout partition changed after consumption")


def _prior_capacity(ticker: str, signal_date: str, sessions: list[str], price_map: dict[tuple[str, str], SessionPrice], volumes: dict[tuple[str, str], float | None], pct: float) -> float | None:
    prior = [date for date in sessions if date <= signal_date]
    if not prior:
        return None
    date = prior[-1]
    close = _finite(getattr(price_map.get((ticker, date)), "close", None))
    volume = volumes.get((ticker, date))
    return close * volume * pct / 100 if close and volume and volume > 0 else None


def _matched_control(trade: dict[str, Any], prices: list[SessionPrice], sessions: list[str], cfg: BaselineConfig) -> ExecutionResult:
    notional = float(trade["entry_notional"])
    control_cfg = dataclasses.replace(cfg, initial_capital=max(cfg.initial_capital, notional * 2), max_positions=1, position_pct=100, max_name_pct=100, max_adv_pct=100)
    return simulate_portfolio(
        [ExecutionDecision(str(trade["decision_id"]), "SPY", str(trade["direction"]), str(trade["signal_date"]), str(trade["entry_date"]), str(trade["exit_date"]), 0, notional, locked_notional=notional)],
        prices, [date for date in sessions if trade["signal_date"] <= date <= trade["exit_date"]], control_cfg,
    )


def _run_next_open_baseline(
    conn: sqlite3.Connection,
    *,
    config: BaselineConfig | None = None,
    consume_heldout: bool = True,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    """Select authoritative inputs, execute them, and build a sealed artifact."""
    cfg = config or BaselineConfig()
    cfg.validate()
    diagnostic = cfg.holding_sessions != 10
    if diagnostic and consume_heldout:
        raise ValueError("diagnostic horizons cannot consume the sealed heldout window")
    calls, reasons, report_lineage = _verified_calls(conn, report_dir)
    lineage_contract_ready = not reasons
    selected_tickers = sorted({call["ticker"] for call in calls} | {"SPY"})
    price_contract, price_reasons = _price_contract(conn, selected_tickers)
    prices, volumes = _price_rows(conn, selected_tickers)
    reasons.extend(price_reasons)
    price_map = {(row.ticker, row.date): row for row in prices}
    sessions = sorted(row.date for row in prices if row.ticker == "SPY" and row.close is not None and row.close > 0)
    indices = {date: index for index, date in enumerate(sessions)}
    signal_dates = sorted({call["signal_date"] for call in calls})
    split = _split_dates(signal_dates, sessions, cfg.holding_sessions)
    partition = {date: name for name in ("development", "validation", "heldout") for date in split[name]}
    observed = bool(split["heldout"]) and all(date in indices and indices[date] + cfg.holding_sessions < len(sessions) for date in split["heldout"])
    reveal = consume_heldout and observed and not diagnostic
    exclusions: list[dict[str, Any]] = []
    decisions: list[ExecutionDecision] = []
    evaluation_windows: list[tuple[str, str, str]] = []
    if not reasons:
        for call in calls:
            call_id, ticker = int(call["call_id"]), str(call["ticker"])
            if ticker in ML_CONTEXT_TICKERS or ticker.startswith("^"):
                exclusions.append({"call_id": call_id, "reason": "non_tradable_context_ticker"}); continue
            if call["signal_date"] not in indices:
                exclusions.append({"call_id": call_id, "reason": "signal_date_not_spy_session"}); continue
            part = partition.get(call["signal_date"])
            if part is None:
                exclusions.append({"call_id": call_id, "reason": "purged_boundary_overlap"}); continue
            if part == "heldout" and not reveal:
                exclusions.append({"call_id": call_id, "reason": "heldout_not_revealed"}); continue
            if call["direction"] not in {"long", "short"}:
                exclusions.append({"call_id": call_id, "reason": "unsupported_direction"}); continue
            entry_i = indices[call["signal_date"]] + 1
            exit_i = entry_i + cfg.holding_sessions - 1
            if exit_i >= len(sessions):
                exclusions.append({"call_id": call_id, "reason": "endpoint_not_observed"}); continue
            entry, exit_ = sessions[entry_i], sessions[exit_i]
            if _finite(getattr(price_map.get((ticker, entry)), "open", None)) is None:
                exclusions.append({"call_id": call_id, "reason": "immediate_next_session_open_missing", "required_entry_date": entry}); continue
            if _finite(getattr(price_map.get((ticker, exit_)), "close", None)) is None:
                exclusions.append({"call_id": call_id, "reason": "exact_tenth_session_close_missing", "required_exit_date": exit_}); continue
            evaluation_windows.append((call["signal_date"], entry, exit_))
            if call["score"] is None or call["score"] < cfg.minimum_score:
                exclusions.append({"call_id": call_id, "reason": "below_minimum_score"}); continue
            capacity = _prior_capacity(ticker, call["signal_date"], sessions, price_map, volumes, cfg.max_adv_pct)
            if capacity is None:
                exclusions.append({"call_id": call_id, "reason": "causal_capacity_unknown"}); continue
            metadata = tuple(sorted({"call_id": call_id, "report_run_id": call["report_run_id"], "setup_family": call["setup_family"], "setup_tier": call["setup_tier"]}.items()))
            decisions.append(ExecutionDecision(str(call_id), ticker, call["direction"], call["signal_date"], entry, exit_, float(call["score"]), capacity, part, metadata=metadata))

    simulation_dates = [
        date for date in sessions
        if evaluation_windows
        and min(row[0] for row in evaluation_windows) <= date <= max(row[2] for row in evaluation_windows)
    ]
    result = simulate_portfolio(decisions, prices, simulation_dates, cfg)
    exclusions.extend({"call_id": int(row["decision_id"]), "reason": row["reason"]} for row in result.exclusions)
    enriched = []
    controls_priced = True
    for trade in result.trades:
        control = _matched_control(trade, prices, sessions, cfg)
        control_complete = (
            control.financing_priced
            and bool(control.trades)
            and all(row.get("equity") is not None for row in control.equity_curve)
        )
        controls_priced = controls_priced and control_complete
        matched = float(control.trades[0]["net_pnl"]) if control_complete else None
        matched_fill = float(control.trades[0]["entry_notional"]) if control.trades else None
        enriched.append({**trade, "spy_matched_target_notional": float(trade["entry_notional"]),
                         "spy_matched_filled_notional": matched_fill,
                         "spy_matched_net_pnl": matched,
                         "active_net_pnl": float(trade["net_pnl"]) - matched if matched is not None else None})

    config_payload = dataclasses.asdict(cfg)
    input_payload = {
        "calls": calls,
        "report_lineage": report_lineage,
        "price_contract": price_contract,
        "prices": [dataclasses.asdict(row) for row in prices],
    }
    config_hash, input_hash = _sha256(config_payload), _sha256(input_payload)
    _guard_consumed_holdout(conn, split["heldout"], config_hash, input_hash)
    consumption = _record_holdout(conn, split["heldout"], config_hash, input_hash) if reveal else {
        "consumed": False, "reusable_for_policy_selection": False,
        "status": "heldout_endpoint_unobserved" if consume_heldout and split["heldout"] else "not_consumed",
    }
    reasons.append("fixed_current_universe_has_survivorship_bias")
    if price_contract and not price_contract.get("distributions_included"):
        reasons.append("price_history_omits_dividend_total_returns")
    if not result.financing_priced or not controls_priced:
        reasons.append("financing_inputs_unpriced")
    valid_curve = [row for row in result.equity_curve if row.get("equity") is not None]
    null_marks = len(result.equity_curve) - len(valid_curve)
    if null_marks:
        reasons.append("unpriceable_daily_marks")
    traded_dates = {trade["entry_date"] for trade in enriched}
    return_intervals = sum(
        left.get("equity") is not None and right.get("equity") is not None
        for left, right in zip(result.equity_curve, result.equity_curve[1:])
    )
    effective_blocks = return_intervals / cfg.holding_sessions
    if return_intervals < 120: reasons.append("fewer_than_120_valid_daily_observations")
    if len(traded_dates) < 20: reasons.append("fewer_than_20_traded_signal_dates")
    if effective_blocks < 12: reasons.append("fewer_than_12_effective_non_overlapping_blocks")
    final_equity = result.final_equity
    matched_values = [trade["spy_matched_net_pnl"] for trade in enriched]
    matched_net = sum(float(v) for v in matched_values) if matched_values and all(v is not None for v in matched_values) else None
    matched_targets = [float(trade["spy_matched_target_notional"]) for trade in enriched]
    matched_fills = [trade["spy_matched_filled_notional"] for trade in enriched]
    matched_target_notional = sum(matched_targets) if matched_targets else None
    matched_filled_notional = (
        sum(float(value) for value in matched_fills)
        if matched_fills and all(value is not None for value in matched_fills)
        else None
    )
    realized = sum(float(trade["net_pnl"]) for trade in enriched)

    full_spy = None
    if evaluation_windows:
        start = min(row[1] for row in evaluation_windows)
        end = max(row[2] for row in evaluation_windows)
        full_cfg = dataclasses.replace(cfg, max_positions=1, position_pct=100, max_name_pct=100, max_adv_pct=100)
        prior_sessions = [date for date in sessions if date < start]
        if prior_sessions:
            full = simulate_portfolio(
                [ExecutionDecision(
                    "full_spy", "SPY", "long", prior_sessions[-1], start, end, 0,
                    cfg.initial_capital, locked_notional=cfg.initial_capital,
                )],
                prices, [date for date in sessions if prior_sessions[-1] <= date <= end], full_cfg,
            )
        else:
            full = None
        full_complete = (
            full is not None
            and len(full.trades) == 1
            and not full.exclusions
            and full.financing_priced
            and full.final_equity is not None
            and bool(full.equity_curve)
            and all(row.get("equity") is not None for row in full.equity_curve)
        )
        if full_complete:
            assert full is not None and full.final_equity is not None
            full_spy = (full.final_equity / full.initial_equity - 1) * 100
        else:
            reasons.append("full_investment_spy_unpriced")
    else:
        reasons.append("evaluation_window_unavailable")
    reasons = sorted(set(reasons))

    artifact = {
        "schema_version": SCHEMA_VERSION, "method": METHOD,
        "snapshot_asof": sessions[-1] if sessions else None, "artifact_scope": "copied_database_research_run",
        "evidence_state": "diagnostic_invalid" if diagnostic else "descriptive_invalid",
        "readiness_status": "insufficient_or_noncausal_evidence", "readiness_reasons": reasons,
        "causal_valid": False, "decision_eligible": False, "causal_limitations": reasons,
        "return_basis": str((price_contract or {}).get("basis") or "unavailable"),
        "benchmark_basis": "same_direction_target_notional_spy_whole_shares_via_canonical_execution_ledger",
        "data_window": {"price_start": sessions[0] if sessions else None, "price_end": sessions[-1] if sessions else None,
                        "signal_start": signal_dates[0] if signal_dates else None, "signal_end": signal_dates[-1] if signal_dates else None},
        "execution_contract": {"signal": "report close", "entry": "immediate next SPY session open before same-day close exits",
                               "exit": f"exact session {cfg.holding_sessions} SPY close", "capacity": "prior-session close times prior-session volume", "delayed_fill_allowed": False},
        "config": config_payload, "splits": split, "consumed_window": consumption,
        "summary": {"selected_calls": len(calls), "closed_trades": len(enriched), "excluded_calls": len(exclusions),
                    "daily_observation_count": return_intervals, "equity_point_count": len(valid_curve),
                    "null_mark_count": null_marks,
                    "signal_date_count": len(signal_dates), "traded_signal_date_count": len(traded_dates),
                    "effective_non_overlapping_block_count": effective_blocks, "initial_capital": cfg.initial_capital,
                    "final_equity": final_equity if result.financing_priced and not null_marks else None,
                    "net_return_pct": (final_equity / cfg.initial_capital - 1) * 100 if final_equity is not None and result.financing_priced and not null_marks else None,
                    "realized_net_pnl": realized if result.financing_priced and not null_marks else None,
                    "matched_spy_net_pnl": matched_net if not null_marks else None,
                    "matched_spy_target_notional": matched_target_notional,
                    "matched_spy_filled_notional": matched_filled_notional,
                    "active_net_pnl": realized - matched_net if matched_net is not None and result.financing_priced and not null_marks else None,
                    "full_investment_spy_net_return_pct": full_spy if not null_marks else None,
                    "opportunity_cost_vs_full_spy_pct": ((final_equity / cfg.initial_capital - 1) * 100 - full_spy)
                    if final_equity is not None and full_spy is not None and result.financing_priced and not null_marks else None,
                    "active_metrics_available": matched_net is not None and result.financing_priced and controls_priced and not null_marks,
                    "max_name_weight_pct": result.max_name_weight_pct, "max_gross_exposure_pct": result.max_gross_exposure_pct},
        "exclusions": sorted(exclusions, key=lambda row: (int(row.get("call_id", 0)), str(row.get("reason", "")))),
        "trades": enriched, "equity_curve": list(result.equity_curve),
        "execution_ledger": result.ledger,
        "provenance": {"config_sha256": config_hash, "input_sha256": input_hash,
                       "code_sha": current_code_version(),
                       "implementation_sha256": _implementation_hash(),
                       "canonical_report_lineage_enforced": lineage_contract_ready,
                       "research_price_basis_enforced": price_contract is not None and bool(price_contract.get("eligible"))},
        "confidence_intervals": None, "confidence_interval_reason": "paired_block_method_not_qualified_or_history_insufficient",
    }
    artifact = _canonical(artifact)
    artifact["provenance"]["artifact_sha256"] = _sha256(artifact)
    return artifact


def run_next_open_baseline(
    conn: sqlite3.Connection,
    *,
    config: BaselineConfig | None = None,
    consume_heldout: bool = True,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one caller-owned snapshot; commit only transactions opened here."""
    opened_snapshot = not conn.in_transaction
    if opened_snapshot:
        conn.execute("BEGIN")
    try:
        artifact = _run_next_open_baseline(
            conn, config=config, consume_heldout=consume_heldout, report_dir=report_dir
        )
        if opened_snapshot:
            conn.commit()
        return artifact
    except Exception:
        if opened_snapshot:
            conn.rollback()
        raise


def write_artifact(path: Path, artifact: dict[str, Any]) -> str:
    payload = canonical_json_bytes(artifact) + b"\n"
    if path.exists() and path.read_bytes() != payload:
        raise FileExistsError(f"refusing to overwrite different artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _valid_artifact_shape(payload: dict[str, Any]) -> bool:
    """Validate the small, current evidence contract without a schema library."""
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("method") != METHOD:
        return False
    if payload.get("causal_valid") is not False or payload.get("decision_eligible") is not False:
        return False
    if payload.get("evidence_state") not in {"descriptive_invalid", "diagnostic_invalid"}:
        return False
    if payload.get("readiness_status") != "insufficient_or_noncausal_evidence":
        return False
    if payload.get("artifact_scope") != "copied_database_research_run":
        return False
    scalar_strings = ("return_basis", "benchmark_basis")
    if any(not isinstance(payload.get(key), str) or not payload[key] for key in scalar_strings):
        return False
    dict_fields = ("data_window", "execution_contract", "config", "splits", "consumed_window", "summary", "provenance")
    list_fields = ("readiness_reasons", "causal_limitations", "exclusions", "trades", "equity_curve")
    if any(not isinstance(payload.get(key), dict) for key in dict_fields):
        return False
    if any(not isinstance(payload.get(key), list) for key in list_fields):
        return False
    if any(not all(isinstance(item, str) and item for item in payload[key])
           for key in ("readiness_reasons", "causal_limitations")):
        return False
    execution = payload["execution_contract"]
    if not all(key in execution for key in ("signal", "entry", "exit", "capacity", "delayed_fill_allowed")):
        return False
    if execution.get("delayed_fill_allowed") is not False:
        return False
    config = payload["config"]
    if set(config) != {field.name for field in dataclasses.fields(BaselineConfig)}:
        return False
    integer_config = ("max_positions", "holding_sessions")
    if any(not isinstance(config[key], int) or isinstance(config[key], bool) for key in integer_config):
        return False
    numeric_config = set(config) - set(integer_config) - {"short_borrow_bps_annual", "cash_rate_bps_annual"}
    if any(_finite(config[key]) is None for key in numeric_config):
        return False
    if any(config[key] is not None and _finite(config[key]) is None
           for key in ("short_borrow_bps_annual", "cash_rate_bps_annual")):
        return False
    try:
        BaselineConfig(**config).validate()
    except (TypeError, ValueError):
        return False
    splits = payload["splits"]
    if not all(
        isinstance(splits.get(key), list)
        and all(isinstance(item, str) and item for item in splits[key])
        for key in ("development", "validation", "heldout")
    ):
        return False
    if any(not isinstance(splits.get(key), int) or isinstance(splits[key], bool) or splits[key] < 0
           for key in ("purge_sessions", "embargo_sessions")):
        return False
    consumed = payload["consumed_window"]
    if not isinstance(consumed.get("consumed"), bool) or not isinstance(
        consumed.get("reusable_for_policy_selection"), bool
    ) or not isinstance(consumed.get("status"), str):
        return False
    required_summary = {
        "selected_calls", "closed_trades", "excluded_calls", "daily_observation_count",
        "null_mark_count", "initial_capital", "net_return_pct", "active_metrics_available",
    }
    required_summary |= {
        "equity_point_count", "signal_date_count", "traded_signal_date_count",
        "effective_non_overlapping_block_count", "final_equity", "realized_net_pnl",
        "matched_spy_net_pnl", "active_net_pnl", "full_investment_spy_net_return_pct",
        "matched_spy_target_notional", "matched_spy_filled_notional",
        "opportunity_cost_vs_full_spy_pct", "max_name_weight_pct", "max_gross_exposure_pct",
    }
    summary = payload["summary"]
    if not required_summary.issubset(summary):
        return False
    count_fields = {
        "selected_calls", "closed_trades", "excluded_calls", "daily_observation_count",
        "equity_point_count", "null_mark_count", "signal_date_count", "traded_signal_date_count",
    }
    if any(not isinstance(summary.get(key), int) or isinstance(summary[key], bool) or summary[key] < 0
           for key in count_fields):
        return False
    numeric_fields = required_summary - count_fields - {"active_metrics_available"}
    if any(value is not None and _finite(value) is None for key in numeric_fields
           if (value := summary.get(key)) is not None):
        return False
    if _finite(summary.get("initial_capital")) is None or float(summary["initial_capital"]) <= 0:
        return False
    if not isinstance(summary.get("active_metrics_available"), bool):
        return False
    if summary["closed_trades"] != len(payload["trades"]) or summary["excluded_calls"] != len(payload["exclusions"]):
        return False
    if any(
        not isinstance(row, dict)
        or not isinstance(row.get("reason"), str)
        or ("call_id" in row and (not isinstance(row["call_id"], int) or isinstance(row["call_id"], bool)))
        for row in payload["exclusions"]
    ):
        return False
    if any(
        not isinstance(row, dict)
        or not all(key in row for key in ("decision_id", "ticker", "entry_date", "exit_date", "net_pnl"))
        or _finite(row.get("net_pnl")) is None
        for row in payload["trades"]
    ):
        return False
    if any(
        not isinstance(row, dict)
        or not isinstance(row.get("date"), str)
        or (row.get("equity") is not None and _finite(row["equity"]) is None)
        for row in payload["equity_curve"]
    ):
        return False
    valid_points = sum(row.get("equity") is not None for row in payload["equity_curve"])
    valid_intervals = sum(
        left.get("equity") is not None and right.get("equity") is not None
        for left, right in zip(payload["equity_curve"], payload["equity_curve"][1:])
    )
    if summary["equity_point_count"] != valid_points:
        return False
    if summary["null_mark_count"] != len(payload["equity_curve"]) - valid_points:
        return False
    if summary["daily_observation_count"] != valid_intervals:
        return False
    provenance = payload["provenance"]
    hashes = ("config_sha256", "input_sha256", "implementation_sha256", "artifact_sha256")
    if any(
        not isinstance(provenance.get(key), str)
        or len(provenance[key]) != 64
        or any(char not in "0123456789abcdef" for char in provenance[key])
        for key in hashes
    ):
        return False
    if not isinstance(provenance.get("canonical_report_lineage_enforced"), bool):
        return False
    if not isinstance(provenance.get("research_price_basis_enforced"), bool):
        return False
    return True


def artifact_state(path: Path | None = None) -> dict[str, Any]:
    configured = os.getenv("TRADER_KOO_NEXT_OPEN_BASELINE_ARTIFACT")
    selected = path or (Path(configured) if configured else RUNTIME_ARTIFACT_PATH if RUNTIME_ARTIFACT_PATH.exists() else PACKAGED_ARTIFACT_PATH)
    unavailable = {"available": False, "evidence_state": "evidence_unavailable", "causal_valid": False,
                   "decision_eligible": False, "artifact_path": selected.name}
    try:
        payload = json.loads(selected.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return unavailable
    if not isinstance(payload, dict) or not _valid_artifact_shape(payload):
        return {**unavailable, "error": "artifact_schema_invalid"}
    check = dict(payload)
    check["provenance"] = dict(payload["provenance"])
    expected = check["provenance"].pop("artifact_sha256", None)
    if not isinstance(expected, str) or expected != _sha256(check):
        return {**unavailable, "error": "artifact_hash_mismatch"}
    if payload["provenance"].get("implementation_sha256") != _implementation_hash():
        return {**unavailable, "error": "implementation_hash_mismatch"}
    return {**payload, "available": True, "artifact_path": selected.name}
