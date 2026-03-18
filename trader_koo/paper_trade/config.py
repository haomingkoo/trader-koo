"""Configuration objects for paper-trade logic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperTradeConfig:
    """Runtime configuration snapshot used by paper-trade helpers."""

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
