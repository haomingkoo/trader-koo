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
    # Realism costs
    entry_slippage_bps: float = 5.0  # 5 bps entry slippage (0.05%)
    exit_slippage_bps: float = 5.0  # 5 bps exit slippage (0.05%)
    commission_per_trade: float = 1.0  # $1 per side (entry + exit = $2 round trip)
    short_borrow_annual_pct: float = 1.5  # 1.5% annualized borrow cost for shorts
    max_adv_pct: float = 15.0  # max 15% of average daily volume per position


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
    }
