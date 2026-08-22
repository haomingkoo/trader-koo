"""Locked next-open research baseline built on one deterministic execution seam.

``execute_portfolio`` knows nothing about SQLite, reports, partitions, or UI
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


@dataclasses.dataclass(frozen=True)
class SessionPrice:
    ticker: str
    date: str
    open: float | None
    close: float | None


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


def execute_portfolio(
    decisions: Iterable[ExecutionDecision],
    prices: Iterable[SessionPrice],
    sessions: Iterable[str],
    config: BaselineConfig,
) -> ExecutionResult:
    """Execute immutable decisions with opens before same-session close exits."""
    config.validate()
    session_list = tuple(sorted(set(sessions)))
    price_map = {(row.ticker, row.date): row for row in prices}
    ordered = sorted(decisions, key=lambda row: (row.entry_date, -row.score, row.ticker, row.decision_id))
    if len({row.decision_id for row in ordered}) != len(ordered):
        raise ValueError("decision_id must be unique")
    by_entry: dict[str, list[ExecutionDecision]] = defaultdict(list)
    exclusions: list[dict[str, Any]] = []
    for decision in ordered:
        if decision.entry_date not in session_list or decision.exit_date not in session_list:
            exclusions.append({"decision_id": decision.decision_id, "reason": "execution_session_missing"})
            continue
        if not math.isfinite(decision.capacity_notional) or decision.capacity_notional <= 0:
            exclusions.append({"decision_id": decision.decision_id, "reason": "capacity_unavailable"})
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

    previous: str | None = None
    for date in session_list:
        # Accrue the interval ending at today's open. Borrow changes both daily
        # equity and subsequent sizing; it is not deferred until trade close.
        if previous is not None:
            if config.cash_rate_bps_annual is None:
                financing_priced = False
            else:
                cash += max(cash, 0) * config.cash_rate_bps_annual / 10_000 / 252
            for position in positions:
                if position["direction"] != "short":
                    continue
                if config.short_borrow_bps_annual is None:
                    financing_priced = False
                    continue
                charge = position["reserve"] * config.short_borrow_bps_annual / 10_000 / 252
                cash -= charge
                position["borrow"] += charge

        # Today's close proceeds do not exist while open orders are admitted.
        adv_used: dict[str, float] = defaultdict(float)
        for decision in by_entry.get(date, []):
            if decision.direction not in {"long", "short"}:
                exclusions.append({"decision_id": decision.decision_id, "reason": "unsupported_direction"})
                continue
            row = price_map.get((decision.ticker, date))
            raw_open = _finite(row.open if row else None)
            if raw_open is None or raw_open <= 0:
                exclusions.append({"decision_id": decision.decision_id, "reason": "immediate_next_session_open_missing"})
                continue
            if len(positions) >= config.max_positions:
                exclusions.append({"decision_id": decision.decision_id, "reason": "position_capacity_full"})
                continue
            equity, exposure = mark(date, "open")
            if equity is None or equity <= 0:
                exclusions.append({"decision_id": decision.decision_id, "reason": "current_equity_unpriceable"})
                continue
            entry_price = _adverse(raw_open, decision.direction, config.entry_slippage_bps, entry=True)
            name_room = max(0.0, equity * config.max_name_pct / 100 - exposure.get(decision.ticker, 0))
            adv_room = max(0.0, decision.capacity_notional - adv_used[decision.ticker])
            target = min(
                decision.locked_notional
                if decision.locked_notional is not None
                else equity * config.position_pct / 100,
                name_room,
                adv_room,
            )
            shares = int(target / entry_price)
            if shares < 1:
                exclusions.append({"decision_id": decision.decision_id, "reason": "capacity_below_one_share"})
                continue
            reserve = shares * entry_price
            commission = _commission(reserve, config)
            if reserve + commission > cash:
                exclusions.append({"decision_id": decision.decision_id, "reason": "insufficient_cash"})
                continue
            cash -= reserve + commission
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
            })

        remaining: list[dict[str, Any]] = []
        for position in positions:
            if position["exit_date"] != date:
                remaining.append(position)
                continue
            row = price_map.get((position["ticker"], date))
            raw_close = _finite(row.close if row else None)
            if raw_close is None or raw_close <= 0:
                exclusions.append({"decision_id": position["decision_id"], "reason": "exact_exit_close_missing"})
                remaining.append(position)
                continue
            exit_price = _adverse(raw_close, position["direction"], config.exit_slippage_bps, entry=False)
            gross = _pnl(position["direction"], position["entry_price"], exit_price, position["shares"])
            exit_commission = _commission(exit_price * position["shares"], config)
            cash += position["reserve"] + gross - exit_commission
            trades.append({
                **{key: value for key, value in position.items() if key != "reserve"},
                "entry_notional": position["reserve"],
                "exit_price": exit_price,
                "gross_pnl": gross,
                "commission": position["entry_commission"] + exit_commission,
                "net_pnl": gross - position["entry_commission"] - exit_commission - position["borrow"],
            })
        positions = remaining

        equity, exposure = mark(date, "close")
        if equity is None:
            curve.append({"date": date, "equity": None, "status": "unpriceable_mark"})
        else:
            gross = sum(exposure.values())
            name = max(exposure.values(), default=0)
            if equity > 0:
                max_gross, max_name = max(max_gross, gross / equity), max(max_name, name / equity)
            curve.append({
                "date": date, "equity": equity, "cash": cash, "open_positions": len(positions),
                "gross_exposure_pct": gross / equity * 100 if equity > 0 else None,
            })
        previous = date

    valid = [float(row["equity"]) for row in curve if row.get("equity") is not None]
    return ExecutionResult(
        tuple(trades), tuple(exclusions), tuple(curve), financing_priced,
        config.initial_capital, valid[-1] if valid else None, max_name * 100, max_gross * 100,
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
    try:
        from trader_koo.report.runs import resolve_published_report
    except ImportError:
        return [], ["report_publication_contract_unavailable"], {}
    run_ids = [str(row[0]) for row in conn.execute(
        "SELECT DISTINCT report_run_id FROM setup_call_evaluations "
        "WHERE report_run_id IS NOT NULL AND TRIM(report_run_id)!='' ORDER BY report_run_id"
    )]
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
    columns = _columns(conn, "setup_call_evaluations")
    optional = [f"e.{name}" if name in columns else f"NULL AS {name}" for name in ("setup_family", "setup_tier")]
    rows = conn.execute(f"""
        SELECT e.id,e.asof_date,e.ticker,e.call_direction,e.score,{','.join(optional)},e.report_run_id
        FROM setup_call_evaluations e
        WHERE e.report_run_id IS NOT NULL AND TRIM(e.report_run_id)!=''
        ORDER BY e.asof_date,e.score DESC,e.ticker,e.id
    """).fetchall()
    return [{
        "call_id": int(row[0]), "signal_date": str(row[1]), "ticker": str(row[2]).upper(),
        "direction": str(row[3]).lower(), "score": _finite(row[4]), "setup_family": row[5],
        "setup_tier": row[6], "report_run_id": row[7],
    } for row in rows if str(row[7]) in lineage], [], lineage


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
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS research_holdout_consumptions (
            consumption_id TEXT PRIMARY KEY, method TEXT NOT NULL, window_start TEXT NOT NULL,
            window_end TEXT NOT NULL, config_hash TEXT NOT NULL, input_hash TEXT NOT NULL,
            reusable_for_policy_selection INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS research_holdout_dates (
            signal_date TEXT PRIMARY KEY,
            consumption_id TEXT NOT NULL REFERENCES research_holdout_consumptions(consumption_id));
        CREATE TRIGGER IF NOT EXISTS research_holdout_consumptions_no_update BEFORE UPDATE
        ON research_holdout_consumptions BEGIN SELECT RAISE(ABORT,'holdout consumption is immutable'); END;
        CREATE TRIGGER IF NOT EXISTS research_holdout_consumptions_no_delete BEFORE DELETE
        ON research_holdout_consumptions BEGIN SELECT RAISE(ABORT,'holdout consumption is immutable'); END;
        CREATE TRIGGER IF NOT EXISTS research_holdout_dates_no_update BEFORE UPDATE
        ON research_holdout_dates BEGIN SELECT RAISE(ABORT,'holdout dates are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS research_holdout_dates_no_delete BEFORE DELETE
        ON research_holdout_dates BEGIN SELECT RAISE(ABORT,'holdout dates are immutable'); END;
    """)


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
        conn.commit()
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
    return execute_portfolio(
        [ExecutionDecision(str(trade["decision_id"]), "SPY", str(trade["direction"]), str(trade["signal_date"]), str(trade["entry_date"]), str(trade["exit_date"]), 0, notional, locked_notional=notional)],
        prices, [date for date in sessions if trade["signal_date"] <= date <= trade["exit_date"]], control_cfg,
    )


def run_next_open_baseline(
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
    opened_snapshot = not conn.in_transaction
    if opened_snapshot:
        conn.execute("BEGIN")
    try:
        calls, reasons, report_lineage = _verified_calls(conn, report_dir)
        lineage_contract_ready = not reasons
        selected_tickers = sorted({call["ticker"] for call in calls} | {"SPY"})
        price_contract, price_reasons = _price_contract(
            conn, selected_tickers
        )
        prices, volumes = _price_rows(conn, selected_tickers)
        if opened_snapshot:
            conn.commit()
    except Exception:
        if opened_snapshot:
            conn.rollback()
        raise
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
            if call["score"] is None or call["score"] < cfg.minimum_score:
                exclusions.append({"call_id": call_id, "reason": "below_minimum_score"}); continue
            entry_i = indices[call["signal_date"]] + 1
            exit_i = entry_i + cfg.holding_sessions - 1
            if exit_i >= len(sessions):
                exclusions.append({"call_id": call_id, "reason": "endpoint_not_observed"}); continue
            entry, exit_ = sessions[entry_i], sessions[exit_i]
            if _finite(getattr(price_map.get((ticker, entry)), "open", None)) is None:
                exclusions.append({"call_id": call_id, "reason": "immediate_next_session_open_missing", "required_entry_date": entry}); continue
            if _finite(getattr(price_map.get((ticker, exit_)), "close", None)) is None:
                exclusions.append({"call_id": call_id, "reason": "exact_tenth_session_close_missing", "required_exit_date": exit_}); continue
            capacity = _prior_capacity(ticker, call["signal_date"], sessions, price_map, volumes, cfg.max_adv_pct)
            if capacity is None:
                exclusions.append({"call_id": call_id, "reason": "causal_capacity_unknown"}); continue
            metadata = tuple(sorted({"call_id": call_id, "report_run_id": call["report_run_id"], "setup_family": call["setup_family"], "setup_tier": call["setup_tier"]}.items()))
            decisions.append(ExecutionDecision(str(call_id), ticker, call["direction"], call["signal_date"], entry, exit_, float(call["score"]), capacity, part, metadata=metadata))

    simulation_dates = sorted({date for decision in decisions for date in sessions if decision.signal_date <= date <= decision.exit_date})
    result = execute_portfolio(decisions, prices, simulation_dates, cfg)
    exclusions.extend({"call_id": int(row["decision_id"]), "reason": row["reason"]} for row in result.exclusions)
    enriched = []
    controls_priced = True
    for trade in result.trades:
        control = _matched_control(trade, prices, sessions, cfg)
        controls_priced = controls_priced and control.financing_priced and bool(control.trades)
        matched = float(control.trades[0]["net_pnl"]) if control.financing_priced and control.trades else None
        enriched.append({**trade, "spy_matched_net_pnl": matched,
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
    effective_blocks = len(valid_curve) / cfg.holding_sessions
    if len(valid_curve) < 120: reasons.append("fewer_than_120_valid_daily_observations")
    if len(traded_dates) < 20: reasons.append("fewer_than_20_traded_signal_dates")
    if effective_blocks < 12: reasons.append("fewer_than_12_effective_non_overlapping_blocks")
    reasons = sorted(set(reasons))
    final_equity = result.final_equity
    matched_values = [trade["spy_matched_net_pnl"] for trade in enriched]
    matched_net = sum(float(v) for v in matched_values) if matched_values and all(v is not None for v in matched_values) else None
    realized = sum(float(trade["net_pnl"]) for trade in enriched)

    full_spy = None
    if enriched:
        start, end = min(str(t["entry_date"]) for t in enriched), max(str(t["exit_date"]) for t in enriched)
        full_cfg = dataclasses.replace(cfg, max_positions=1, position_pct=100, max_name_pct=100, max_adv_pct=100)
        full = execute_portfolio([ExecutionDecision("full_spy", "SPY", "long", start, start, end, 0, cfg.initial_capital, locked_notional=cfg.initial_capital * .99)], prices, [date for date in sessions if start <= date <= end], full_cfg)
        if full.final_equity is not None and full.financing_priced:
            full_spy = (full.final_equity / full.initial_equity - 1) * 100

    artifact = {
        "schema_version": SCHEMA_VERSION, "method": METHOD,
        "snapshot_asof": sessions[-1] if sessions else None, "artifact_scope": "copied_database_research_run",
        "evidence_state": "diagnostic_invalid" if diagnostic else "descriptive_invalid",
        "readiness_status": "insufficient_or_noncausal_evidence", "readiness_reasons": reasons,
        "causal_valid": False, "decision_eligible": False, "causal_limitations": reasons,
        "return_basis": str((price_contract or {}).get("basis") or "unavailable"),
        "benchmark_basis": "same_direction_same_notional_spy_via_canonical_execution_ledger",
        "data_window": {"price_start": sessions[0] if sessions else None, "price_end": sessions[-1] if sessions else None,
                        "signal_start": signal_dates[0] if signal_dates else None, "signal_end": signal_dates[-1] if signal_dates else None},
        "execution_contract": {"signal": "report close", "entry": "immediate next SPY session open before same-day close exits",
                               "exit": f"exact session {cfg.holding_sessions} SPY close", "capacity": "prior-session close times prior-session volume", "delayed_fill_allowed": False},
        "config": config_payload, "splits": split, "consumed_window": consumption,
        "summary": {"selected_calls": len(calls), "closed_trades": len(enriched), "excluded_calls": len(exclusions),
                    "daily_observation_count": len(valid_curve), "null_mark_count": null_marks,
                    "signal_date_count": len(signal_dates), "traded_signal_date_count": len(traded_dates),
                    "effective_non_overlapping_block_count": effective_blocks, "initial_capital": cfg.initial_capital,
                    "final_equity": final_equity,
                    "net_return_pct": (final_equity / cfg.initial_capital - 1) * 100 if final_equity is not None and result.financing_priced else None,
                    "realized_net_pnl": realized if result.financing_priced else None, "matched_spy_net_pnl": matched_net,
                    "active_net_pnl": realized - matched_net if matched_net is not None and result.financing_priced else None,
                    "full_investment_spy_net_return_pct": full_spy,
                    "active_metrics_available": matched_net is not None and result.financing_priced and controls_priced and not null_marks,
                    "max_name_weight_pct": result.max_name_weight_pct, "max_gross_exposure_pct": result.max_gross_exposure_pct},
        "exclusions": sorted(exclusions, key=lambda row: (int(row.get("call_id", 0)), str(row.get("reason", "")))),
        "trades": enriched, "equity_curve": list(result.equity_curve),
        "provenance": {"config_sha256": config_hash, "input_sha256": input_hash,
                       "implementation_sha256": _implementation_hash(),
                       "canonical_report_lineage_enforced": lineage_contract_ready,
                       "research_price_basis_enforced": price_contract is not None and bool(price_contract.get("eligible"))},
        "confidence_intervals": None, "confidence_interval_reason": "paired_block_method_not_qualified_or_history_insufficient",
    }
    artifact = _canonical(artifact)
    artifact["provenance"]["artifact_sha256"] = _sha256(artifact)
    return artifact


def write_artifact(path: Path, artifact: dict[str, Any]) -> str:
    payload = canonical_json_bytes(artifact) + b"\n"
    if path.exists() and path.read_bytes() != payload:
        raise FileExistsError(f"refusing to overwrite different artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def artifact_state(path: Path | None = None) -> dict[str, Any]:
    configured = os.getenv("TRADER_KOO_NEXT_OPEN_BASELINE_ARTIFACT")
    selected = path or (Path(configured) if configured else RUNTIME_ARTIFACT_PATH if RUNTIME_ARTIFACT_PATH.exists() else PACKAGED_ARTIFACT_PATH)
    unavailable = {"available": False, "evidence_state": "evidence_unavailable", "causal_valid": False,
                   "decision_eligible": False, "artifact_path": selected.name}
    try:
        payload = json.loads(selected.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return unavailable
    if not isinstance(payload, dict) or not isinstance(payload.get("provenance"), dict):
        return unavailable
    check = dict(payload)
    check["provenance"] = dict(payload["provenance"])
    expected = check["provenance"].pop("artifact_sha256", None)
    if not isinstance(expected, str) or expected != _sha256(check):
        return {**unavailable, "error": "artifact_hash_mismatch"}
    if payload["provenance"].get("implementation_sha256") != _implementation_hash():
        return {**unavailable, "error": "implementation_hash_mismatch"}
    return {**payload, "available": True, "artifact_path": selected.name}
