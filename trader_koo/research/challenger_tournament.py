"""Frozen, fail-closed preregistration for the non-TA challenger tournament.

The sealed holdout is deliberately unreachable until the copied research
database proves a consistent total-return basis. Signal helpers are pure so
their formulas can be verified without consuming any research window.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from trader_koo.db.price_contract import research_price_contract
from trader_koo.ml.features import ML_CONTEXT_TICKERS
from trader_koo.report.runs import current_code_version
from trader_koo.research.challenger_preregistration import (
    CHALLENGERS,
    MAX_HOLDING_SESSIONS,
    SCHEMA_VERSION,
    UNIVERSE_ID,
    frozen_preregistration,
)
from trader_koo.research.challenger_executor import (
    c1_signal,
    c2_signal,
    c3_exposure,
    daily_returns,
    execute_validation_tournament,
    holm_adjust,
    required_membership_dates,
)
from trader_koo.research.next_open_baseline import canonical_json_bytes
from trader_koo.research.universe_membership import (
    MembershipContractError,
    load_verified_membership,
)

IMPLEMENTATION_PATH = Path(__file__)
REPOSITORY_ROOT = IMPLEMENTATION_PATH.parents[2]
IMPLEMENTATION_PATHS = (
    "trader_koo/research/challenger_tournament.py",
    "trader_koo/research/challenger_preregistration.py",
    "trader_koo/research/challenger_executor.py",
    "trader_koo/research/universe_membership.py",
    "trader_koo/research/next_open_baseline.py",
    "trader_koo/db/price_contract.py",
    "trader_koo/ml/features.py",
)
IMPLEMENTATION_ENVIRONMENT_PATHS = (
    "pyproject.toml",
    "requirements.txt",
)

def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _implementation_manifest() -> dict[str, Any]:
    """Describe the maintained source, data-gate, and runtime closure."""
    def hashes(paths: tuple[str, ...]) -> dict[str, str]:
        return {
            relative: hashlib.sha256((REPOSITORY_ROOT / relative).read_bytes()).hexdigest()
            for relative in sorted(paths)
        }

    return {
        "schema_version": "challenger-implementation-v2",
        "python_requires": ">=3.11",
        "source_files": hashes(IMPLEMENTATION_PATHS),
        "environment_files": hashes(IMPLEMENTATION_ENVIRONMENT_PATHS),
        "context_tickers_sha256": _sha256(sorted(ML_CONTEXT_TICKERS)),
    }


def _implementation_hash() -> str:
    """Hash the declared signal, execution, data-gate, and runtime closure."""
    return _sha256(_implementation_manifest())


def chronological_split(sessions: Iterable[str]) -> dict[str, Any]:
    """Create a 60/20/20 split with overlap purged and embargoed."""
    dates = tuple(sorted(set(sessions)))
    first, second = int(len(dates) * .6), int(len(dates) * .8)
    purge = MAX_HOLDING_SESSIONS
    windows = {
        "development": dates[:max(0, first - purge)],
        "validation": dates[min(len(dates), first + purge):max(first + purge, second - purge)],
        "heldout": dates[min(len(dates), second + purge):],
    }
    return {
        name: {
            "start": values[0] if values else None,
            "end": values[-1] if values else None,
            "session_count": len(values),
            "session_sha256": _sha256(values),
        }
        for name, values in windows.items()
    }


def _table_exists(conn: Any, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _columns(conn: Any, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def dataset_audit(conn: Any) -> dict[str, Any]:
    """Audit the exact copied database before any validation or holdout read."""
    if not _table_exists(conn, "price_daily"):
        return {"eligible": False, "reasons": ["price_daily_unavailable"]}
    columns = _columns(conn, "price_daily")
    required = {"ticker", "date", "open", "close", "volume"}
    if not required.issubset(columns):
        return {"eligible": False, "reasons": ["required_price_columns_unavailable"]}
    tickers: set[str] = set()
    row_count = 0
    reasons: list[str] = []
    start_date: dt.date | None = None
    end_date: dt.date | None = None
    spy_prices: list[tuple[dt.date, float]] = []
    invalid_spy_prices = 0
    market_rows_hasher = hashlib.sha256()
    excluded_context_rows = 0
    for ticker, date_value, open_value, close_value, volume_value in conn.execute(
        """SELECT ticker,date,open,close,volume FROM price_daily
           ORDER BY ticker,date,open,close,volume"""
    ):
        ticker_name = str(ticker).strip() if ticker is not None else ""
        if ticker_name != "SPY" and (
            ticker_name in ML_CONTEXT_TICKERS or ticker_name.startswith("^")
        ):
            excluded_context_rows += 1
            continue
        market_rows_hasher.update(canonical_json_bytes([
            ticker, date_value, open_value, close_value, volume_value,
        ]))
        market_rows_hasher.update(b"\n")
        row_count += 1
        if ticker_name:
            tickers.add(ticker_name)
        else:
            reasons.append("invalid_ticker")
        try:
            parsed_date = dt.date.fromisoformat(str(date_value))
            start_date = parsed_date if start_date is None else min(start_date, parsed_date)
            end_date = parsed_date if end_date is None else max(end_date, parsed_date)
        except (TypeError, ValueError):
            reasons.append("invalid_price_date")
            parsed_date = None
        try:
            open_price, close_price, volume = (
                float(open_value), float(close_value), float(volume_value)
            )
            valid_values = (
                math.isfinite(open_price) and open_price > 0
                and math.isfinite(close_price) and close_price > 0
                and math.isfinite(volume) and volume >= 0
            )
        except (TypeError, ValueError):
            valid_values = False
        if not valid_values:
            reasons.append("invalid_price_value")
            if ticker_name == "SPY":
                invalid_spy_prices += 1
        elif ticker_name == "SPY" and parsed_date is not None:
            spy_prices.append((parsed_date, close_price))
    start = start_date.isoformat() if start_date else None
    end = end_date.isoformat() if end_date else None
    ticker_names = sorted(tickers)
    ticker_count = len(ticker_names)
    membership_contract: dict[str, Any]
    research_tickers = ticker_names
    try:
        membership = load_verified_membership(
            conn,
            UNIVERSE_ID,
            required_membership_dates([
                date.isoformat() for date, _close in sorted(spy_prices)
            ]),
        )
        membership_contract = {"eligible": True, **membership.contract}
        research_tickers = sorted(set(membership.tickers) | {"SPY"})
    except MembershipContractError as exc:
        reasons.append(exc.code)
        membership_contract = {
            "eligible": False,
            "universe_id": UNIVERSE_ID,
            "reason": exc.code,
        }
    try:
        contract = research_price_contract(conn, research_tickers)
    except Exception as exc:
        contract = {
            "eligible": False, "basis": "unknown", "status": "unresolved",
            "reason": f"price_contract_error:{type(exc).__name__}",
        }
    if not contract.get("eligible"):
        reasons.append(str(contract.get("reason") or "price_contract_unverified"))
    if contract.get("basis") != "total_return" or not contract.get("distributions_included"):
        reasons.append("consistent_total_return_basis_required")
    years = 0.0
    if start and end:
        try:
            years = (
                dt.date.fromisoformat(str(end)) - dt.date.fromisoformat(str(start))
            ).days / 365.2425
        except ValueError:
            reasons.append("invalid_price_date")
    if years < 5:
        reasons.append("fewer_than_five_years")
    spy_closes = [close for _, close in sorted(spy_prices)]
    if invalid_spy_prices:
        reasons.append("invalid_spy_price")
    rolling_volatility = [
        statistics.stdev(daily_returns(spy_closes[index - 20:index + 1])) * math.sqrt(252)
        for index in range(20, len(spy_closes))
        if len(daily_returns(spy_closes[index - 20:index + 1])) > 1
    ]
    regime_count = 0
    if rolling_volatility:
        ordered_volatility = sorted(rolling_volatility)
        low = ordered_volatility[len(ordered_volatility) // 3]
        high = ordered_volatility[len(ordered_volatility) * 2 // 3]
        regime_count = len({
            "low" if value <= low else "high" if value >= high else "middle"
            for value in rolling_volatility
        })
    if regime_count < 3:
        reasons.append("fewer_than_three_volatility_regimes")
    snapshot = {
        "price_start": start, "price_end": end, "years": years,
        "ticker_count": ticker_count, "row_count": row_count,
        "market_rows_sha256": market_rows_hasher.hexdigest(),
        "excluded_context_row_count": excluded_context_rows,
        "volatility_regime_count": regime_count,
        "price_contract": contract,
        "universe_treatment": "point_in_time_membership",
        "membership_contract": membership_contract,
        "research_ticker_count": len(research_tickers),
    }
    return {
        **snapshot,
        "dataset_sha256": _sha256(snapshot),
        "eligible": not reasons,
        "reasons": sorted(set(reasons)),
    }


def run_challenger_tournament(
    conn: Any,
    *,
    consume_heldout: bool = False,
) -> dict[str, Any]:
    """Attempt exactly C1-C3, leaving sealed validation untouched on bad data."""
    audit = dataset_audit(conn)
    preregistration = frozen_preregistration(
        str(audit.get("dataset_sha256") or _sha256(audit)),
        (audit.get("membership_contract") or {}).get("membership_sha256"),
    )
    if not audit["eligible"]:
        results = {
            name: {
                "status": "failed_data_gate", "reasons": audit["reasons"],
                "config_sha256": preregistration["config_hashes"][name],
                "metrics": None,
            }
            for name in CHALLENGERS
        }
        body = {
            "schema_version": SCHEMA_VERSION,
            "code_sha": current_code_version(),
            "implementation_sha256": _implementation_hash(),
            "implementation_manifest": _implementation_manifest(),
            "status": "blocked_before_validation",
            "preregistration": preregistration,
            "dataset_audit": audit,
            "split": None,
            "challenger_results": results,
            "holm_adjusted_p_values": None,
            "selected_challenger": None,
            "sealed_heldout": {"accessed": False, "access_log": []},
            "prospective_shadow_candidate": None,
            "automatic_promotion": False,
        }
        return {**body, "artifact_sha256": _sha256(body)}
    execution = execute_validation_tournament(
        conn, preregistration, consume_heldout=consume_heldout
    )
    selected = execution["selected_challenger"]
    body = {
        "schema_version": SCHEMA_VERSION,
        "code_sha": current_code_version(),
        "implementation_sha256": _implementation_hash(),
        "implementation_manifest": _implementation_manifest(),
        "status": (
            "sealed_heldout_complete"
            if execution["sealed_heldout"]["accessed"]
            else "validation_complete_winner_pending_sealed_holdout"
            if selected
            else "validation_complete_no_eligible_challenger"
        ),
        "preregistration": preregistration,
        "dataset_audit": audit,
        **execution,
        "automatic_promotion": False,
    }
    return {**body, "artifact_sha256": _sha256(body)}
