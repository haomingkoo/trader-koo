"""Deterministic, fail-closed next-open research baseline.

The engine evaluates persisted report calls without pretending the historical
rows are causal evidence.  A signal can enter only at the immediate next SPY
session open and exits at that position's tenth session close.  Missing prices
are exclusions, never invitations to slide an order to a convenient later bar.
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
from typing import Any

from trader_koo.ml.features import ML_CONTEXT_TICKERS


SCHEMA_VERSION = "1.0"
METHOD = "setup_calls_next_open_to_tenth_close"
IMPLEMENTATION_PATH = Path(__file__)
PACKAGED_ARTIFACT_PATH = Path(__file__).with_name("next_open_baseline_artifact_20260823.json")
RUNTIME_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "research" / "next_open_baseline_latest.json"
)


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
    spy_short_borrow_bps_annual: float | None = 25.0
    cash_rate_bps_annual: float | None = 0.0
    holding_sessions: int = 10
    minimum_score: float = 0.0

    def validate(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.max_positions < 1:
            raise ValueError("max_positions must be positive")
        if not 0 < self.position_pct <= self.max_name_pct <= 100:
            raise ValueError("position_pct must be positive and no larger than max_name_pct")
        if not 0 < self.max_adv_pct <= 100:
            raise ValueError("max_adv_pct must be in (0, 100]")
        if not 2 <= self.holding_sessions <= 60:
            raise ValueError("holding_sessions must be between 2 and 60")


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


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
    return json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _implementation_hash() -> str:
    return hashlib.sha256(IMPLEMENTATION_PATH.read_bytes()).hexdigest()


def _price_rows(conn: sqlite3.Connection) -> dict[str, dict[str, dict[str, float | None]]]:
    required = {"ticker", "date", "open", "close", "volume"}
    if not _table_exists(conn, "price_daily") or not required.issubset(
        _columns(conn, "price_daily")
    ):
        return {}
    rows = conn.execute(
        """
        SELECT ticker, date, CAST(open AS REAL), CAST(close AS REAL), CAST(volume AS REAL)
        FROM price_daily
        WHERE ticker IS NOT NULL AND date IS NOT NULL
        ORDER BY ticker, date
        """
    ).fetchall()
    prices: dict[str, dict[str, dict[str, float | None]]] = defaultdict(dict)
    for ticker, date, open_price, close, volume in rows:
        prices[str(ticker).upper()][str(date)] = {
            "open": _finite(open_price),
            "close": _finite(close),
            "volume": _finite(volume),
        }
    return dict(prices)


def _report_calls(
    conn: sqlite3.Connection,
) -> tuple[list[dict[str, Any]], list[str], bool]:
    if not _table_exists(conn, "setup_call_evaluations"):
        return [], ["setup_call_evaluations_missing"], False
    columns = _columns(conn, "setup_call_evaluations")
    needed = {"id", "asof_date", "ticker", "call_direction", "score"}
    if not needed.issubset(columns):
        return [], ["setup_call_evaluations_schema_incomplete"], False

    lineage_ready = "report_run_id" in columns and _table_exists(conn, "report_runs")
    reasons: list[str] = []
    where = ""
    join = ""
    if lineage_ready:
        run_columns = _columns(conn, "report_runs")
        canonical_col = (
            "is_generation_canonical"
            if "is_generation_canonical" in run_columns
            else "is_canonical" if "is_canonical" in run_columns else None
        )
        if canonical_col is None or "status" not in run_columns:
            lineage_ready = False
        else:
            join = " JOIN report_runs r ON r.run_id = e.report_run_id "
            where = f" WHERE r.status='published' AND r.{canonical_col}=1 "
    if not lineage_ready:
        reasons.append("persisted_calls_are_not_linked_to_a_canonical_published_report")

    selected = [
        "e.id", "e.asof_date", "e.ticker", "e.call_direction", "e.score"
    ]
    for optional in ("setup_family", "setup_tier", "report_run_id"):
        selected.append(f"e.{optional}" if optional in columns else f"NULL AS {optional}")
    rows = conn.execute(
        f"SELECT {', '.join(selected)} FROM setup_call_evaluations e {join} {where} "
        "ORDER BY e.asof_date, e.score DESC, e.ticker, e.id"
    ).fetchall()
    calls = [
        {
            "call_id": int(row[0]),
            "signal_date": str(row[1]),
            "ticker": str(row[2]).upper(),
            "direction": str(row[3]).lower(),
            "score": _finite(row[4]),
            "setup_family": row[5],
            "setup_tier": row[6],
            "report_run_id": row[7],
        }
        for row in rows
    ]
    return calls, reasons, lineage_ready


def _split_dates(
    dates: list[str], sessions: list[str], holding_sessions: int
) -> dict[str, Any]:
    if not dates:
        return {
            "development": [], "validation": [], "heldout": [],
            "purge_sessions": holding_sessions, "embargo_sessions": holding_sessions,
        }
    n = len(dates)
    dev_end = max(1, int(n * 0.60))
    val_end = max(dev_end, int(n * 0.80))
    development = dates[:dev_end]
    validation = dates[dev_end:val_end]
    heldout = dates[val_end:]
    session_index = {date: index for index, date in enumerate(sessions)}

    def purged(values: list[str], prior: list[str]) -> list[str]:
        if not values or not prior:
            return values
        boundary = session_index.get(prior[-1])
        if boundary is None:
            return []
        return [
            value for value in values
            if session_index.get(value, boundary) - boundary > holding_sessions
        ]

    validation = purged(validation, development)
    heldout = purged(heldout, validation or development)
    return {
        "development": development,
        "validation": validation,
        "heldout": heldout,
        "purge_sessions": holding_sessions,
        "embargo_sessions": holding_sessions,
    }


def _ensure_consumption_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS research_holdout_consumptions (
            consumption_id TEXT PRIMARY KEY,
            method TEXT NOT NULL,
            window_start TEXT NOT NULL,
            window_end TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            reusable_for_policy_selection INTEGER NOT NULL DEFAULT 0,
            UNIQUE(method, window_start, window_end)
        );
        CREATE TRIGGER IF NOT EXISTS research_holdout_consumptions_no_update
        BEFORE UPDATE ON research_holdout_consumptions
        BEGIN SELECT RAISE(ABORT, 'holdout consumption is immutable'); END;
        CREATE TRIGGER IF NOT EXISTS research_holdout_consumptions_no_delete
        BEFORE DELETE ON research_holdout_consumptions
        BEGIN SELECT RAISE(ABORT, 'holdout consumption is immutable'); END;
        """
    )


def _record_holdout_consumption(
    conn: sqlite3.Connection,
    *,
    heldout_dates: list[str],
    config_hash: str,
    input_hash: str,
) -> dict[str, Any]:
    if not heldout_dates:
        return {
            "consumed": False,
            "reusable_for_policy_selection": False,
            "status": "heldout_window_unavailable",
            "window_start": None,
            "window_end": None,
        }
    _ensure_consumption_schema(conn)
    start, end = heldout_dates[0], heldout_dates[-1]
    identity = _sha256(
        {"method": METHOD, "window_start": start, "window_end": end,
         "config_hash": config_hash, "input_hash": input_hash}
    )
    existing = conn.execute(
        """SELECT consumption_id, config_hash, input_hash
           FROM research_holdout_consumptions
           WHERE method=? AND window_start=? AND window_end=?""",
        (METHOD, start, end),
    ).fetchone()
    if existing and (str(existing[1]) != config_hash or str(existing[2]) != input_hash):
        raise ValueError("heldout window was already consumed by different immutable inputs")
    conn.execute(
        """INSERT OR IGNORE INTO research_holdout_consumptions
           (consumption_id, method, window_start, window_end, config_hash, input_hash)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (identity, METHOD, start, end, config_hash, input_hash),
    )
    conn.commit()
    return {
        "consumed": True,
        "reusable_for_policy_selection": False,
        "status": "sealed_once_not_reusable_for_policy_selection",
        "consumption_id": identity,
        "window_start": start,
        "window_end": end,
    }


def _commission(notional: float, config: BaselineConfig) -> float:
    return max(
        config.minimum_commission_per_side,
        notional * config.commission_bps_per_side / 10_000.0,
    )


def _adverse(price: float, direction: str, bps: float, *, entry: bool) -> float:
    sign = 1.0 if (direction == "long") == entry else -1.0
    return price * (1.0 + sign * bps / 10_000.0)


def _signed_pnl(direction: str, entry_price: float, exit_price: float, shares: int) -> float:
    raw = (exit_price - entry_price) * shares
    return raw if direction == "long" else -raw


def _trade_control(
    *,
    direction: str,
    notional: float,
    entry_spy: float,
    exit_spy: float,
    sessions: int,
    config: BaselineConfig,
) -> dict[str, Any]:
    adjusted_entry = _adverse(
        entry_spy, direction, config.entry_slippage_bps, entry=True
    )
    adjusted_exit = _adverse(
        exit_spy, direction, config.exit_slippage_bps, entry=False
    )
    shares = notional / adjusted_entry
    gross = _signed_pnl(direction, adjusted_entry, adjusted_exit, shares)
    costs = _commission(notional, config) + _commission(shares * adjusted_exit, config)
    financing_priced = config.cash_rate_bps_annual is not None
    borrow = 0.0
    if direction == "short":
        financing_priced = financing_priced and config.spy_short_borrow_bps_annual is not None
        if config.spy_short_borrow_bps_annual is not None:
            borrow = notional * config.spy_short_borrow_bps_annual / 10_000.0 * sessions / 252.0
    return {
        "gross_pnl": gross,
        "costs": costs,
        "borrow": borrow,
        "net_pnl": gross - costs - borrow,
        "financing_priced": financing_priced,
    }


def run_next_open_baseline(
    conn: sqlite3.Connection,
    *,
    config: BaselineConfig | None = None,
    consume_heldout: bool = True,
) -> dict[str, Any]:
    """Run the locked descriptive baseline and return an immutable payload."""
    cfg = config or BaselineConfig()
    cfg.validate()
    diagnostic = cfg.holding_sessions != 10
    if diagnostic and consume_heldout:
        raise ValueError("diagnostic horizons cannot consume the sealed heldout window")
    calls, causal_reasons, lineage_ready = _report_calls(conn)
    prices = _price_rows(conn)
    spy = prices.get("SPY", {})
    # The signal is formed at the close, so a missing signal-day open must not
    # erase a genuine market session.  Entry-day SPY open is checked separately.
    sessions = sorted(
        date for date, row in spy.items()
        if row.get("close") and row["close"] > 0
    )
    session_index = {date: idx for idx, date in enumerate(sessions)}
    signal_dates = sorted({call["signal_date"] for call in calls})
    split = _split_dates(signal_dates, sessions, cfg.holding_sessions)
    heldout_dates = list(split["heldout"])
    heldout_observed = bool(heldout_dates) and all(
        (session_index.get(date) is not None)
        and (int(session_index[date]) + cfg.holding_sessions < len(sessions))
        for date in heldout_dates
    )
    reveal_heldout = bool(consume_heldout and heldout_observed and not diagnostic)
    partition_by_date = {
        date: partition
        for partition in ("development", "validation", "heldout")
        for date in split[partition]
    }
    config_payload = dataclasses.asdict(cfg)
    config_hash = _sha256(config_payload)
    input_payload = {
        "calls": calls,
        "price_rows": [
            [ticker, date, row.get("open"), row.get("close"), row.get("volume")]
            for ticker in sorted(prices)
            for date, row in sorted(prices[ticker].items())
            if ticker == "SPY" or any(
                call["ticker"] == ticker and call["signal_date"] <= date
                for call in calls
            )
        ],
    }
    input_hash = _sha256(input_payload)

    exclusions: list[dict[str, Any]] = []
    scheduled: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for call in calls:
        if call["ticker"] in ML_CONTEXT_TICKERS or call["ticker"].startswith("^"):
            exclusions.append({"call_id": call["call_id"], "reason": "non_tradable_context_ticker"})
            continue
        if call["signal_date"] not in session_index:
            exclusions.append({"call_id": call["call_id"], "reason": "signal_date_not_spy_session"})
            continue
        partition = partition_by_date.get(call["signal_date"])
        if partition is None:
            exclusions.append({"call_id": call["call_id"], "reason": "purged_boundary_overlap"})
            continue
        if partition == "heldout" and not reveal_heldout:
            exclusions.append({
                "call_id": call["call_id"],
                "reason": (
                    "heldout_endpoint_unobserved" if consume_heldout
                    else "heldout_not_consumed"
                ),
            })
            continue
        direction = call["direction"]
        if direction not in {"long", "short"}:
            exclusions.append({"call_id": call["call_id"], "reason": "unsupported_direction"})
            continue
        score = call["score"]
        if score is None or score < cfg.minimum_score:
            exclusions.append({"call_id": call["call_id"], "reason": "below_minimum_score"})
            continue
        idx = session_index[call["signal_date"]]
        entry_idx = idx + 1
        exit_idx = entry_idx + cfg.holding_sessions - 1
        if entry_idx >= len(sessions) or exit_idx >= len(sessions):
            exclusions.append({"call_id": call["call_id"], "reason": "endpoint_not_observed"})
            continue
        entry_date, exit_date = sessions[entry_idx], sessions[exit_idx]
        if _finite(spy.get(entry_date, {}).get("open")) is None:
            exclusions.append({
                "call_id": call["call_id"], "reason": "matched_spy_entry_open_missing",
                "required_entry_date": entry_date,
            })
            continue
        ticker_prices = prices.get(call["ticker"], {})
        entry = ticker_prices.get(entry_date)
        exit_row = ticker_prices.get(exit_date)
        if not entry or not entry.get("open"):
            exclusions.append({
                "call_id": call["call_id"], "reason": "immediate_next_session_open_missing",
                "required_entry_date": entry_date,
            })
            continue
        if not exit_row or not exit_row.get("close"):
            exclusions.append({
                "call_id": call["call_id"], "reason": "exact_tenth_session_close_missing",
                "required_exit_date": exit_date,
            })
            continue
        volume = _finite(entry.get("volume"))
        if volume is None or volume <= 0:
            exclusions.append({"call_id": call["call_id"], "reason": "entry_capacity_unknown"})
            continue
        scheduled[entry_date].append({
            **call,
            "evidence_partition": partition,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "raw_entry_open": float(entry["open"]),
            "raw_exit_close": float(exit_row["close"]),
            "entry_volume": volume,
        })

    cash = cfg.initial_capital
    open_positions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    max_name_weight = 0.0
    max_gross_exposure = 0.0
    financing_unpriced = False
    scheduled_exits = [
        candidate["exit_date"]
        for candidates in scheduled.values()
        for candidate in candidates
    ]
    simulation_start = min(signal_dates) if signal_dates else None
    simulation_end = max(scheduled_exits) if scheduled_exits else simulation_start
    simulation_sessions = [
        date for date in sessions
        if simulation_start is not None
        and simulation_end is not None
        and simulation_start <= date <= simulation_end
    ]

    def mark(date: str, field: str) -> float | None:
        total = cash
        for pos in open_positions:
            row = prices.get(pos["ticker"], {}).get(date)
            price = _finite((row or {}).get(field))
            if price is None or price <= 0:
                return None
            if pos["direction"] == "long":
                total += pos["shares"] * price
            else:
                total += pos["reserve"] + (pos["entry_price"] - price) * pos["shares"]
        return total

    for date in simulation_sessions:
        # Exit at the exact endpoint close; no delayed completion.
        remaining: list[dict[str, Any]] = []
        for pos in open_positions:
            if pos["exit_date"] != date:
                remaining.append(pos)
                continue
            raw_exit = float(prices[pos["ticker"]][date]["close"])
            exit_price = _adverse(raw_exit, pos["direction"], cfg.exit_slippage_bps, entry=False)
            exit_notional = exit_price * pos["shares"]
            exit_commission = _commission(exit_notional, cfg)
            borrow = 0.0
            if pos["direction"] == "short":
                if cfg.short_borrow_bps_annual is None:
                    financing_unpriced = True
                else:
                    borrow = (
                        pos["reserve"] * cfg.short_borrow_bps_annual / 10_000.0
                        * cfg.holding_sessions / 252.0
                    )
            pnl = _signed_pnl(
                pos["direction"], pos["entry_price"], exit_price, pos["shares"]
            ) - pos["entry_commission"] - exit_commission - borrow
            cash += pos["reserve"] + pnl + pos["entry_commission"]
            spy_entry = float(spy[pos["entry_date"]]["open"])
            spy_exit = float(spy[date]["close"])
            control = _trade_control(
                direction=pos["direction"], notional=pos["reserve"],
                entry_spy=spy_entry, exit_spy=spy_exit,
                sessions=cfg.holding_sessions, config=cfg,
            )
            financing_unpriced = financing_unpriced or not control["financing_priced"]
            trades.append({
                **{key: value for key, value in pos.items() if key != "reserve"},
                "exit_price": exit_price,
                "gross_pnl": _signed_pnl(
                    pos["direction"], pos["entry_price"], exit_price, pos["shares"]
                ),
                "commission": pos["entry_commission"] + exit_commission,
                "borrow": borrow,
                "net_pnl": pnl,
                "spy_matched_net_pnl": control["net_pnl"] if control["financing_priced"] else None,
                "active_net_pnl": pnl - control["net_pnl"] if control["financing_priced"] else None,
            })
        open_positions = remaining

        candidates = sorted(
            scheduled.get(date, []), key=lambda row: (-float(row["score"]), row["ticker"], row["call_id"])
        )
        for candidate in candidates:
            if len(open_positions) >= cfg.max_positions:
                exclusions.append({"call_id": candidate["call_id"], "reason": "position_capacity_full"})
                continue
            current_equity = mark(date, "open")
            if current_equity is None or current_equity <= 0:
                exclusions.append({"call_id": candidate["call_id"], "reason": "current_equity_unpriceable"})
                continue
            raw_entry = candidate["raw_entry_open"]
            entry_price = _adverse(raw_entry, candidate["direction"], cfg.entry_slippage_bps, entry=True)
            desired = min(
                current_equity * cfg.position_pct / 100.0,
                current_equity * cfg.max_name_pct / 100.0,
                raw_entry * candidate["entry_volume"] * cfg.max_adv_pct / 100.0,
            )
            shares = int(desired / entry_price)
            if shares < 1:
                exclusions.append({"call_id": candidate["call_id"], "reason": "capacity_below_one_share"})
                continue
            reserve = shares * entry_price
            entry_commission = _commission(reserve, cfg)
            if reserve + entry_commission > cash:
                exclusions.append({"call_id": candidate["call_id"], "reason": "insufficient_cash"})
                continue
            cash -= reserve + entry_commission
            open_positions.append({
                **candidate,
                "entry_price": entry_price,
                "shares": shares,
                "reserve": reserve,
                "entry_commission": entry_commission,
                "position_weight_at_entry": reserve / current_equity,
            })
            max_name_weight = max(max_name_weight, reserve / current_equity)

        close_equity = mark(date, "close")
        if close_equity is None:
            equity_curve.append({"date": date, "equity": None, "status": "unpriceable_mark"})
            continue
        gross = sum(pos["reserve"] for pos in open_positions)
        max_gross_exposure = max(max_gross_exposure, gross / close_equity if close_equity > 0 else 0.0)
        equity_curve.append({
            "date": date,
            "equity": close_equity,
            "cash": cash,
            "open_positions": len(open_positions),
            "gross_exposure_pct": gross / close_equity * 100.0 if close_equity > 0 else None,
        })

    config_hash = _sha256(config_payload)
    consumption = (
        _record_holdout_consumption(
            conn, heldout_dates=heldout_dates, config_hash=config_hash, input_hash=input_hash
        )
        if reveal_heldout else {
            "consumed": False, "reusable_for_policy_selection": False,
            "status": (
                "heldout_endpoint_unobserved"
                if consume_heldout and heldout_dates and not heldout_observed
                else "not_consumed"
            ),
        }
    )
    causal_reasons.extend([
        "fixed_current_universe_has_survivorship_bias",
        "price_history_omits_dividend_total_returns",
    ])
    causal_reasons = sorted(set(causal_reasons))
    closed_net = sum(float(trade["net_pnl"]) for trade in trades)
    matched_net = (
        sum(float(trade["spy_matched_net_pnl"]) for trade in trades)
        if trades and not financing_unpriced else None
    )
    start_equity = cfg.initial_capital
    final_equity = next(
        (float(row["equity"]) for row in reversed(equity_curve) if row["equity"] is not None),
        start_equity,
    )
    spy_full_return = None
    observed_trade_dates = sorted(
        {trade["entry_date"] for trade in trades} | {trade["exit_date"] for trade in trades}
    )
    if observed_trade_dates:
        first, last = observed_trade_dates[0], observed_trade_dates[-1]
        spy_full_return = (float(spy[last]["close"]) / float(spy[first]["open"]) - 1.0) * 100.0
    segments = {}
    for partition in ("development", "validation", "heldout"):
        segment_trades = [
            trade for trade in trades if trade.get("evidence_partition") == partition
        ]
        segments[partition] = {
            "signal_dates": len(split[partition]),
            "closed_trades": len(segment_trades),
            "net_pnl": sum(float(trade["net_pnl"]) for trade in segment_trades),
            "matched_spy_net_pnl": (
                sum(float(trade["spy_matched_net_pnl"]) for trade in segment_trades)
                if segment_trades
                and all(trade["spy_matched_net_pnl"] is not None for trade in segment_trades)
                else None
            ),
            "revealed": partition != "heldout" or reveal_heldout,
        }
    effective_blocks = len(equity_curve) / cfg.holding_sessions
    readiness_reasons = list(causal_reasons)
    if len(equity_curve) < 120:
        readiness_reasons.append("fewer_than_120_daily_observations")
    if len({trade["entry_date"] for trade in trades}) < 20:
        readiness_reasons.append("fewer_than_20_traded_signal_dates")
    if effective_blocks < 12:
        readiness_reasons.append("fewer_than_12_effective_non_overlapping_blocks")
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "snapshot_asof": sessions[-1] if sessions else None,
        "artifact_scope": "copied_database_research_run",
        "data_window": {
            "price_start": sessions[0] if sessions else None,
            "price_end": sessions[-1] if sessions else None,
            "signal_start": signal_dates[0] if signal_dates else None,
            "signal_end": signal_dates[-1] if signal_dates else None,
        },
        "evidence_state": "diagnostic_invalid" if diagnostic else "descriptive_invalid",
        "readiness_status": "insufficient_or_noncausal_evidence",
        "readiness_reasons": sorted(set(readiness_reasons)),
        "causal_valid": False,
        "decision_eligible": False,
        "causal_limitations": causal_reasons,
        "return_basis": "split_adjusted_price_return_only_dividends_omitted",
        "benchmark_basis": "same_direction_same_notional_spy_price_return_net_of_costs",
        "execution_contract": {
            "signal": "report close",
            "entry": "immediate next SPY session open only",
            "exit": f"exact session {cfg.holding_sessions} SPY close",
            "daily_marks": True,
            "delayed_fill_allowed": False,
        },
        "config": config_payload,
        "splits": split,
        "segments": segments,
        "consumed_window": consumption,
        "summary": {
            "selected_calls": len(calls),
            "closed_trades": len(trades),
            "excluded_calls": len(exclusions),
            "daily_observation_count": len(equity_curve),
            "signal_date_count": len(signal_dates),
            "traded_signal_date_count": len({trade["entry_date"] for trade in trades}),
            "effective_non_overlapping_block_count": effective_blocks,
            "initial_capital": start_equity,
            "final_equity": final_equity,
            "net_return_pct": (final_equity / start_equity - 1.0) * 100.0,
            "realized_net_pnl": closed_net,
            "matched_spy_net_pnl": matched_net,
            "active_net_pnl": closed_net - matched_net if matched_net is not None else None,
            "full_investment_spy_price_return_pct": spy_full_return,
            "active_metrics_available": matched_net is not None,
            "max_name_weight_pct": max_name_weight * 100.0,
            "max_gross_exposure_pct": max_gross_exposure * 100.0,
        },
        "exclusions": sorted(
            exclusions, key=lambda row: (int(row.get("call_id", 0)), str(row.get("reason", "")))
        ),
        "trades": trades,
        "equity_curve": equity_curve,
        "provenance": {
            "config_sha256": config_hash,
            "input_sha256": input_hash,
            "implementation_sha256": _implementation_hash(),
            "canonical_report_lineage_enforced": lineage_ready,
        },
        "confidence_intervals": None,
        "confidence_interval_reason": "paired_block_method_not_qualified_or_history_insufficient",
    }
    artifact = _canonical(artifact)
    artifact["provenance"]["artifact_sha256"] = _sha256(artifact)
    return artifact


def write_artifact(path: Path, artifact: dict[str, Any]) -> str:
    """Write canonical JSON without overwriting a different immutable artifact."""
    payload = canonical_json_bytes(artifact) + b"\n"
    if path.exists() and path.read_bytes() != payload:
        raise FileExistsError(f"refusing to overwrite different artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def artifact_state(path: Path | None = None) -> dict[str, Any]:
    """Load and verify the latest configured artifact, failing closed."""
    configured = os.getenv("TRADER_KOO_NEXT_OPEN_BASELINE_ARTIFACT")
    selected = path or (
        Path(configured)
        if configured
        else RUNTIME_ARTIFACT_PATH
        if RUNTIME_ARTIFACT_PATH.exists()
        else PACKAGED_ARTIFACT_PATH
    )
    unavailable = {
        "available": False,
        "evidence_state": "evidence_unavailable",
        "causal_valid": False,
        "decision_eligible": False,
        "artifact_path": selected.name,
    }
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return unavailable
    if not isinstance(payload, dict):
        return unavailable
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return unavailable
    expected = provenance.get("artifact_sha256")
    check = dict(payload)
    check_provenance = dict(provenance)
    check_provenance.pop("artifact_sha256", None)
    check["provenance"] = check_provenance
    if not isinstance(expected, str) or expected != _sha256(check):
        return {**unavailable, "error": "artifact_hash_mismatch"}
    return {**payload, "available": True, "artifact_path": selected.name}
