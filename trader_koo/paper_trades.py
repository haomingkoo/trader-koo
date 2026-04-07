"""Public facade for paper-trade tracking in trader-koo."""

from __future__ import annotations

import os
import sqlite3
from typing import Any

from trader_koo.paper_trade.config import PaperTradeConfig
from trader_koo.paper_trade.decision import (
    compute_stop_and_target as _compute_stop_and_target_impl,
    direction_from_row as _direction_from_row_impl,
    evaluate_setup_for_paper_trade as _evaluate_setup_for_paper_trade_impl,
    qualify_setup_for_paper_trade as _qualify_setup_for_paper_trade_impl,
)
from trader_koo.paper_trade.schema import ensure_paper_trade_schema
from trader_koo.paper_trade.summary import (
    list_paper_trades as _list_paper_trades_impl,
    paper_trade_summary as _paper_trade_summary_impl,
)
from trader_koo.paper_trade.trading import (
    compute_pnl as _compute_pnl_impl,
    compute_r_multiple as _compute_r_multiple_impl,
    compute_trailing_stop,
    create_paper_trades_from_report as _create_paper_trades_from_report_impl,
    manually_close_trade as _manually_close_trade_impl,
    mark_to_market as _mark_to_market_impl,
)

# ── Configuration (env vars) ────────────────────────────────────────
PAPER_TRADE_ENABLED = os.getenv("TRADER_KOO_PAPER_TRADE_ENABLED", "1") == "1"
PAPER_TRADE_BOT_VERSION = os.getenv("TRADER_KOO_PAPER_TRADE_BOT_VERSION", "v1.0.0")
PAPER_TRADE_MIN_TIER = os.getenv("TRADER_KOO_PAPER_TRADE_MIN_TIER", "B")
PAPER_TRADE_MIN_SCORE = float(os.getenv("TRADER_KOO_PAPER_TRADE_MIN_SCORE", "60.0"))
PAPER_TRADE_MAX_OPEN = int(os.getenv("TRADER_KOO_PAPER_TRADE_MAX_OPEN", "20"))
PAPER_TRADE_EXPIRY_DAYS = int(os.getenv("TRADER_KOO_PAPER_TRADE_EXPIRY_DAYS", "10"))
PAPER_TRADE_STOP_ATR_MULT = float(os.getenv("TRADER_KOO_PAPER_TRADE_STOP_ATR_MULT", "1.5"))
PAPER_TRADE_DEFAULT_STOP_PCT = float(os.getenv("TRADER_KOO_PAPER_TRADE_DEFAULT_STOP_PCT", "3.0"))
PAPER_TRADE_MIN_REWARD_R = float(os.getenv("TRADER_KOO_PAPER_TRADE_MIN_REWARD_R", "1.5"))
PAPER_TRADE_MIN_POSITION_PCT = float(os.getenv("TRADER_KOO_PAPER_TRADE_MIN_POSITION_PCT", "2.0"))
PAPER_TRADE_MAX_POSITION_PCT = float(os.getenv("TRADER_KOO_PAPER_TRADE_MAX_POSITION_PCT", "14.0"))
PAPER_TRADE_TIER_A_POSITION_PCT = float(os.getenv("TRADER_KOO_PAPER_TRADE_TIER_A_POSITION_PCT", "12.0"))
PAPER_TRADE_TIER_B_POSITION_PCT = float(os.getenv("TRADER_KOO_PAPER_TRADE_TIER_B_POSITION_PCT", "8.0"))
PAPER_TRADE_TIER_C_POSITION_PCT = float(os.getenv("TRADER_KOO_PAPER_TRADE_TIER_C_POSITION_PCT", "5.0"))
PAPER_TRADE_CAUTION_POSITION_SCALE = float(os.getenv("TRADER_KOO_PAPER_TRADE_CAUTION_POSITION_SCALE", "0.65"))
PAPER_TRADE_HIGH_VOL_POSITION_SCALE = float(os.getenv("TRADER_KOO_PAPER_TRADE_HIGH_VOL_POSITION_SCALE", "0.75"))
PAPER_TRADE_EARNINGS_POSITION_SCALE = float(os.getenv("TRADER_KOO_PAPER_TRADE_EARNINGS_POSITION_SCALE", "0.60"))
PAPER_TRADE_STARTING_CAPITAL = float(os.getenv("TRADER_KOO_PAPER_TRADE_STARTING_CAPITAL", "1000000.0"))
# Graduated trailing stop config
PAPER_TRADE_TRAIL_BREAKEVEN_R = float(os.getenv("TRADER_KOO_PAPER_TRADE_TRAIL_BREAKEVEN_R", "1.25"))
PAPER_TRADE_TRAIL_MID_R = float(os.getenv("TRADER_KOO_PAPER_TRADE_TRAIL_MID_R", "1.5"))
PAPER_TRADE_TRAIL_MID_CUSHION_R = float(os.getenv("TRADER_KOO_PAPER_TRADE_TRAIL_MID_CUSHION_R", "1.0"))
PAPER_TRADE_TRAIL_TIGHT_R = float(os.getenv("TRADER_KOO_PAPER_TRADE_TRAIL_TIGHT_R", "2.0"))
PAPER_TRADE_TRAIL_TIGHT_CUSHION_R = float(os.getenv("TRADER_KOO_PAPER_TRADE_TRAIL_TIGHT_CUSHION_R", "0.5"))
PAPER_TRADE_EXPIRY_USE_TRADING_DAYS = os.getenv("TRADER_KOO_PAPER_TRADE_EXPIRY_USE_TRADING_DAYS", "1") == "1"

_QUALIFYING_TIERS = frozenset({"A", "B"})
_QUALIFYING_ACTIONABILITY = frozenset({"higher-probability", "conditional"})
_QUALIFYING_DIRECTIONS = frozenset({"long", "short"})

_TIER_RANK = {"A": 1, "B": 2, "C": 3, "D": 4, "F": 5}
_PAPER_DECISION_VERSION = "paper-trade-eval-v1"
_DEBATE_CAUTION_AGREEMENT = 60.0
_HIGH_VOL_ATR_PCT = 6.0


def _build_config() -> PaperTradeConfig:
    return PaperTradeConfig(
        bot_version=PAPER_TRADE_BOT_VERSION,
        min_tier=PAPER_TRADE_MIN_TIER,
        min_score=PAPER_TRADE_MIN_SCORE,
        max_open=PAPER_TRADE_MAX_OPEN,
        expiry_days=PAPER_TRADE_EXPIRY_DAYS,
        stop_atr_mult=PAPER_TRADE_STOP_ATR_MULT,
        default_stop_pct=PAPER_TRADE_DEFAULT_STOP_PCT,
        qualifying_tiers=_QUALIFYING_TIERS,
        qualifying_actionability=_QUALIFYING_ACTIONABILITY,
        qualifying_directions=_QUALIFYING_DIRECTIONS,
        tier_rank=dict(_TIER_RANK),
        decision_version=_PAPER_DECISION_VERSION,
        debate_caution_agreement=_DEBATE_CAUTION_AGREEMENT,
        high_vol_atr_pct=_HIGH_VOL_ATR_PCT,
        min_reward_r_multiple=PAPER_TRADE_MIN_REWARD_R,
        min_position_pct=PAPER_TRADE_MIN_POSITION_PCT,
        max_position_pct=PAPER_TRADE_MAX_POSITION_PCT,
        tier_a_position_pct=PAPER_TRADE_TIER_A_POSITION_PCT,
        tier_b_position_pct=PAPER_TRADE_TIER_B_POSITION_PCT,
        tier_c_position_pct=PAPER_TRADE_TIER_C_POSITION_PCT,
        caution_position_scale=PAPER_TRADE_CAUTION_POSITION_SCALE,
        high_vol_position_scale=PAPER_TRADE_HIGH_VOL_POSITION_SCALE,
        earnings_position_scale=PAPER_TRADE_EARNINGS_POSITION_SCALE,
        starting_capital=PAPER_TRADE_STARTING_CAPITAL,
        trail_breakeven_r=PAPER_TRADE_TRAIL_BREAKEVEN_R,
        trail_mid_r=PAPER_TRADE_TRAIL_MID_R,
        trail_mid_cushion_r=PAPER_TRADE_TRAIL_MID_CUSHION_R,
        trail_tight_r=PAPER_TRADE_TRAIL_TIGHT_R,
        trail_tight_cushion_r=PAPER_TRADE_TRAIL_TIGHT_CUSHION_R,
        expiry_use_trading_days=PAPER_TRADE_EXPIRY_USE_TRADING_DAYS,
    )


def _direction_from_row(row: dict[str, Any]) -> str:
    return _direction_from_row_impl(row)


def qualify_setup_for_paper_trade(row: dict[str, Any]) -> bool:
    return _qualify_setup_for_paper_trade_impl(row, config=_build_config())


def evaluate_setup_for_paper_trade(row: dict[str, Any]) -> dict[str, Any]:
    return _evaluate_setup_for_paper_trade_impl(row, config=_build_config())


def compute_stop_and_target(
    row: dict[str, Any],
    direction: str,
) -> dict[str, float | None]:
    return _compute_stop_and_target_impl(row, direction, config=_build_config())


def create_paper_trades_from_report(
    conn: sqlite3.Connection,
    *,
    setup_rows: list[dict[str, Any]],
    report_date: str,
    generated_ts: str,
) -> int:
    return _create_paper_trades_from_report_impl(
        conn,
        setup_rows=setup_rows,
        report_date=report_date,
        generated_ts=generated_ts,
        config=_build_config(),
    )


def _compute_pnl(
    direction: str,
    entry_price: float,
    current_price: float,
) -> float:
    return _compute_pnl_impl(direction, entry_price, current_price)


def _compute_r_multiple(
    direction: str,
    entry_price: float,
    exit_price: float,
    stop_loss: float | None,
) -> float | None:
    return _compute_r_multiple_impl(
        direction,
        entry_price,
        exit_price,
        stop_loss,
        config=_build_config(),
    )


def mark_to_market(conn: sqlite3.Connection) -> dict[str, Any]:
    return _mark_to_market_impl(conn, config=_build_config())


def paper_trade_summary(
    conn: sqlite3.Connection,
    *,
    window_days: int = 180,
) -> dict[str, Any]:
    return _paper_trade_summary_impl(conn, window_days=window_days, config=_build_config())


def manually_close_trade(
    conn: sqlite3.Connection,
    *,
    trade_id: int,
    exit_price: float | None = None,
    exit_reason: str = "manual_close",
) -> dict[str, Any]:
    return _manually_close_trade_impl(
        conn,
        trade_id=trade_id,
        exit_price=exit_price,
        exit_reason=exit_reason,
        config=_build_config(),
    )


def list_paper_trades(
    conn: sqlite3.Connection,
    *,
    status: str = "all",
    ticker: str | None = None,
    direction: str | None = None,
    family: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    return _list_paper_trades_impl(
        conn,
        status=status,
        ticker=ticker,
        direction=direction,
        family=family,
        from_date=from_date,
        to_date=to_date,
        limit=limit,
    )
