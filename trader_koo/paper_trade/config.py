"""Configuration objects for paper-trade logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PaperTradeConfig:
    """Runtime configuration snapshot used by paper-trade helpers."""

    bot_version: str
    min_tier: str
    min_score: float
    max_open: int
    expiry_days: int
    stop_atr_mult: float
    default_stop_pct: float
    qualifying_tiers: frozenset[str]
    qualifying_actionability: frozenset[str]
    qualifying_directions: frozenset[str]
    tier_rank: dict[str, int]
    decision_version: str
    debate_caution_agreement: float
    high_vol_atr_pct: float
    min_reward_r_multiple: float
    min_position_pct: float
    max_position_pct: float
    tier_a_position_pct: float
    tier_b_position_pct: float
    tier_c_position_pct: float
    caution_position_scale: float
    high_vol_position_scale: float
    earnings_position_scale: float
    ml_enabled: bool = False  # Disabled — AUC 0.5051 (random). Rule-based pipeline decides.
    ml_min_win_prob: float = 0.55
    max_drawdown_pct: float = 15.0  # halt new entries if portfolio draws down this much
    max_daily_loss_pct: float = 5.0  # halt new entries if daily portfolio loss exceeds this
    starting_capital: float = 1_000_000.0  # initial paper portfolio value
    # Realism costs (conservative defaults — better to overestimate than underestimate)
    entry_slippage_bps: float = 10.0  # 10 bps entry slippage (0.10%)
    exit_slippage_bps: float = 10.0  # 10 bps exit slippage on stop-market orders
    commission_per_trade: float = 5.0  # $5 per side (IBKR-like for typical position)
    short_borrow_annual_pct: float = 3.0  # 3% annualized (conservative avg across S&P 500)
    max_adv_pct: float = 15.0  # max 15% of average daily volume per position
    # Graduated trailing stop levels (R-multiples)
    trail_breakeven_r: float = 1.25  # move stop to breakeven (was hardcoded 1.0)
    trail_mid_r: float = 1.5  # threshold for mid-width trail
    trail_mid_cushion_r: float = 1.0  # cushion from HWM in R units (was 0.5)
    trail_tight_r: float = 2.0  # threshold for tight trail near target
    trail_tight_cushion_r: float = 0.5  # cushion from HWM at tightest level
    expiry_use_trading_days: bool = True  # count trading days, not calendar


def config_snapshot(config: PaperTradeConfig) -> dict[str, Any]:
    """Return a JSON-safe snapshot of the current paper-trading policy."""
    return {
        "bot_version": config.bot_version,
        "decision_version": config.decision_version,
        "min_tier": config.min_tier,
        "min_score": config.min_score,
        "max_open": config.max_open,
        "expiry_days": config.expiry_days,
        "min_reward_r_multiple": config.min_reward_r_multiple,
        "high_vol_atr_pct": config.high_vol_atr_pct,
        "qualifying_tiers": sorted(config.qualifying_tiers),
        "qualifying_actionability": sorted(config.qualifying_actionability),
        "position_size_pct": {
            "A": config.tier_a_position_pct,
            "B": config.tier_b_position_pct,
            "C": config.tier_c_position_pct,
        },
        "caution_position_scale": config.caution_position_scale,
        "high_vol_position_scale": config.high_vol_position_scale,
        "earnings_position_scale": config.earnings_position_scale,
        "trailing_stop": {
            "breakeven_r": config.trail_breakeven_r,
            "mid_r": config.trail_mid_r,
            "mid_cushion_r": config.trail_mid_cushion_r,
            "tight_r": config.trail_tight_r,
            "tight_cushion_r": config.trail_tight_cushion_r,
        },
        "expiry_use_trading_days": config.expiry_use_trading_days,
    }
