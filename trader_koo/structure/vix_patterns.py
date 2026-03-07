"""VIX trap and reclaim pattern detection.

This module implements detection logic for VIX-specific patterns:
- Bull traps (failed breakouts above resistance)
- Bear traps (failed breakdowns below support)
- Support reclaims (recovery after breakdown)
- Resistance reclaims (recovery after breakout)

Requirements: 8.1, 8.2, 8.3, 8.4, 14.1, 14.2, 14.3, 14.4, 14.5, 14.7
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class VIXTrapReclaimConfig:
    """Configuration for VIX trap/reclaim detection."""

    trap_lookback_bars: int = 3  # Bars to check for reversal after break
    reclaim_confirmation_bars: int = 2  # Consecutive bars needed for reclaim
    break_threshold_pct: float = 0.2  # % above/below level to count as break
    reversal_threshold_pct: float = 0.1  # % back through level to count as reversal


@dataclass
class TrapReclaimPattern:
    """Represents a detected trap or reclaim pattern."""

    pattern_type: str  # "bull_trap", "bear_trap", "support_reclaim", "resistance_reclaim"
    date: str  # Date of pattern completion
    price: float  # VIX price at pattern
    level: float  # Support/resistance level involved
    confidence: float  # 0.0 to 1.0
    explanation: str  # Human-readable description
    bars_to_reversal: int | None = None  # For traps: how many bars until reversal
    volume_factor: float | None = None  # Volume contribution to confidence
    reversal_speed_factor: float | None = None  # Speed contribution to confidence


def detect_vix_trap_reclaim_patterns(
    vix_data: pd.DataFrame,
    levels: list[dict[str, Any]],
    cfg: VIXTrapReclaimConfig | None = None,
) -> list[TrapReclaimPattern]:
    """Detect trap and reclaim patterns in VIX data.

    Args:
        vix_data: DataFrame with columns: date, open, high, low, close
        levels: List of support/resistance levels with keys: type, level, zone_low, zone_high
        cfg: Configuration for detection parameters

    Returns:
        List of detected TrapReclaimPattern objects
    """
    if cfg is None:
        cfg = VIXTrapReclaimConfig()

    if vix_data is None or vix_data.empty or len(vix_data) < cfg.trap_lookback_bars + 2:
        return []

    patterns: list[TrapReclaimPattern] = []

    # Sort by date and reset index
    df = vix_data.sort_values("date").reset_index(drop=True)

    # Extract resistance and support levels
    resistance_levels = [lv for lv in levels if lv.get("type") == "resistance"]
    support_levels = [lv for lv in levels if lv.get("type") == "support"]

    # Detect bull traps (failed breakout above resistance)
    patterns.extend(_detect_bull_traps(df, resistance_levels, cfg))

    # Detect bear traps (failed breakdown below support)
    patterns.extend(_detect_bear_traps(df, support_levels, cfg))

    # Detect support reclaims
    patterns.extend(_detect_support_reclaims(df, support_levels, cfg))

    # Detect resistance reclaims
    patterns.extend(_detect_resistance_reclaims(df, resistance_levels, cfg))

    return patterns


def _detect_bull_traps(
    df: pd.DataFrame,
    resistance_levels: list[dict[str, Any]],
    cfg: VIXTrapReclaimConfig,
) -> list[TrapReclaimPattern]:
    """Detect bull traps: VIX breaks above resistance then reverses down within N bars.

    Confidence calculation includes:
    - Reversal magnitude (how far price fell from breakout high)
    - Reversal speed (faster reversals = higher confidence)
    - Volume profile (higher volume on reversal = higher confidence)

    Requirements: 8.1, 14.1, 14.3
    """
    patterns: list[TrapReclaimPattern] = []

    if not resistance_levels or len(df) < cfg.trap_lookback_bars + 1:
        return patterns

    # Check if volume data is available
    has_volume = "volume" in df.columns

    for level_info in resistance_levels:
        resistance = float(level_info.get("level", 0))
        zone_high = float(level_info.get("zone_high", resistance))

        if resistance <= 0:
            continue

        # Look for breakout followed by reversal
        for i in range(len(df) - cfg.trap_lookback_bars):
            # Check if we broke above resistance
            high_i = float(df.iloc[i]["high"])
            close_i = float(df.iloc[i]["close"])

            if high_i <= zone_high:
                continue  # No breakout

            # Check if we reversed back below within lookback window
            for j in range(i + 1, min(i + cfg.trap_lookback_bars + 1, len(df))):
                close_j = float(df.iloc[j]["close"])

                if close_j < resistance:
                    # Bull trap detected!
                    bars_to_reversal = j - i
                    date_str = str(df.iloc[j]["date"])

                    # Calculate reversal magnitude (normalized)
                    reversal_magnitude = (high_i - close_j) / resistance
                    magnitude_score = min(0.4, reversal_magnitude * 2.0)

                    # Calculate reversal speed factor
                    speed_factor = 1.0 - (bars_to_reversal / cfg.trap_lookback_bars)
                    speed_score = speed_factor * 0.3

                    # Calculate volume factor if available
                    volume_score = 0.0
                    volume_factor = None
                    if has_volume:
                        try:
                            # Compare reversal bar volume to breakout bar volume
                            volume_breakout = float(df.iloc[i]["volume"])
                            volume_reversal = float(df.iloc[j]["volume"])
                            
                            if volume_breakout > 0:
                                volume_ratio = volume_reversal / volume_breakout
                                # Higher volume on reversal increases confidence
                                volume_factor = min(2.0, volume_ratio)
                                volume_score = min(0.3, volume_factor * 0.15)
                        except (ValueError, KeyError):
                            pass

                    # Base confidence + magnitude + speed + volume
                    confidence = min(0.95, 0.5 + magnitude_score + speed_score + volume_score)

                    patterns.append(
                        TrapReclaimPattern(
                            pattern_type="bull_trap",
                            date=date_str,
                            price=close_j,
                            level=resistance,
                            confidence=confidence,
                            explanation=f"VIX broke above resistance at {resistance:.2f} then reversed back below within {bars_to_reversal} bars (confidence: {confidence:.0%})",
                            bars_to_reversal=bars_to_reversal,
                            volume_factor=volume_factor,
                            reversal_speed_factor=speed_factor,
                        )
                    )
                    break  # Found trap for this level, move to next level

    return patterns


def _detect_bear_traps(
    df: pd.DataFrame,
    support_levels: list[dict[str, Any]],
    cfg: VIXTrapReclaimConfig,
) -> list[TrapReclaimPattern]:
    """Detect bear traps: VIX breaks below support then reverses up within N bars.

    Confidence calculation includes:
    - Reversal magnitude (how far price rose from breakdown low)
    - Reversal speed (faster reversals = higher confidence)
    - Volume profile (higher volume on reversal = higher confidence)

    Requirements: 8.2, 14.2, 14.3
    """
    patterns: list[TrapReclaimPattern] = []

    if not support_levels or len(df) < cfg.trap_lookback_bars + 1:
        return patterns

    # Check if volume data is available
    has_volume = "volume" in df.columns

    for level_info in support_levels:
        support = float(level_info.get("level", 0))
        zone_low = float(level_info.get("zone_low", support))

        if support <= 0:
            continue

        # Look for breakdown followed by reversal
        for i in range(len(df) - cfg.trap_lookback_bars):
            # Check if we broke below support
            low_i = float(df.iloc[i]["low"])
            close_i = float(df.iloc[i]["close"])

            if low_i >= zone_low:
                continue  # No breakdown

            # Check if we reversed back above within lookback window
            for j in range(i + 1, min(i + cfg.trap_lookback_bars + 1, len(df))):
                close_j = float(df.iloc[j]["close"])

                if close_j > support:
                    # Bear trap detected!
                    bars_to_reversal = j - i
                    date_str = str(df.iloc[j]["date"])

                    # Calculate reversal magnitude (normalized)
                    reversal_magnitude = (close_j - low_i) / support
                    magnitude_score = min(0.4, reversal_magnitude * 2.0)

                    # Calculate reversal speed factor
                    speed_factor = 1.0 - (bars_to_reversal / cfg.trap_lookback_bars)
                    speed_score = speed_factor * 0.3

                    # Calculate volume factor if available
                    volume_score = 0.0
                    volume_factor = None
                    if has_volume:
                        try:
                            # Compare reversal bar volume to breakdown bar volume
                            volume_breakdown = float(df.iloc[i]["volume"])
                            volume_reversal = float(df.iloc[j]["volume"])
                            
                            if volume_breakdown > 0:
                                volume_ratio = volume_reversal / volume_breakdown
                                # Higher volume on reversal increases confidence
                                volume_factor = min(2.0, volume_ratio)
                                volume_score = min(0.3, volume_factor * 0.15)
                        except (ValueError, KeyError):
                            pass

                    # Base confidence + magnitude + speed + volume
                    confidence = min(0.95, 0.5 + magnitude_score + speed_score + volume_score)

                    patterns.append(
                        TrapReclaimPattern(
                            pattern_type="bear_trap",
                            date=date_str,
                            price=close_j,
                            level=support,
                            confidence=confidence,
                            explanation=f"VIX broke below support at {support:.2f} then reversed back above within {bars_to_reversal} bars (confidence: {confidence:.0%})",
                            bars_to_reversal=bars_to_reversal,
                            volume_factor=volume_factor,
                            reversal_speed_factor=speed_factor,
                        )
                    )
                    break  # Found trap for this level, move to next level

    return patterns


def _detect_support_reclaims(
    df: pd.DataFrame,
    support_levels: list[dict[str, Any]],
    cfg: VIXTrapReclaimConfig,
) -> list[TrapReclaimPattern]:
    """Detect support reclaims: VIX closes above broken support for 2+ consecutive bars.

    Requirements: 8.3, 14.4
    """
    patterns: list[TrapReclaimPattern] = []

    if not support_levels or len(df) < cfg.reclaim_confirmation_bars + 2:
        return patterns

    for level_info in support_levels:
        support = float(level_info.get("level", 0))
        zone_low = float(level_info.get("zone_low", support))

        if support <= 0:
            continue

        # Look for breakdown followed by reclaim
        breakdown_found = False
        breakdown_idx = -1

        for i in range(len(df) - cfg.reclaim_confirmation_bars):
            close_i = float(df.iloc[i]["close"])
            low_i = float(df.iloc[i]["low"])

            # Check for breakdown
            if not breakdown_found and low_i < zone_low:
                breakdown_found = True
                breakdown_idx = i
                continue

            # If we had a breakdown, check for reclaim
            if breakdown_found and i > breakdown_idx:
                # Check if we have N consecutive closes above support
                consecutive_above = 0
                for j in range(i, min(i + cfg.reclaim_confirmation_bars, len(df))):
                    close_j = float(df.iloc[j]["close"])
                    if close_j > support:
                        consecutive_above += 1
                    else:
                        break

                if consecutive_above >= cfg.reclaim_confirmation_bars:
                    # Support reclaim detected!
                    date_str = str(df.iloc[i + cfg.reclaim_confirmation_bars - 1]["date"])
                    price = float(df.iloc[i + cfg.reclaim_confirmation_bars - 1]["close"])

                    # Calculate confidence based on strength of reclaim
                    reclaim_strength = (price - support) / support
                    confidence = min(0.95, 0.6 + reclaim_strength * 3.0)

                    patterns.append(
                        TrapReclaimPattern(
                            pattern_type="support_reclaim",
                            date=date_str,
                            price=price,
                            level=support,
                            confidence=confidence,
                            explanation=f"VIX reclaimed support at {support:.2f} after breakdown, closing above for {consecutive_above} consecutive bars",
                        )
                    )
                    breakdown_found = False  # Reset to look for next pattern
                    break

    return patterns


def _detect_resistance_reclaims(
    df: pd.DataFrame,
    resistance_levels: list[dict[str, Any]],
    cfg: VIXTrapReclaimConfig,
) -> list[TrapReclaimPattern]:
    """Detect resistance reclaims: VIX closes below broken resistance for 2+ consecutive bars.

    Requirements: 8.4, 14.5
    """
    patterns: list[TrapReclaimPattern] = []

    if not resistance_levels or len(df) < cfg.reclaim_confirmation_bars + 2:
        return patterns

    for level_info in resistance_levels:
        resistance = float(level_info.get("level", 0))
        zone_high = float(level_info.get("zone_high", resistance))

        if resistance <= 0:
            continue

        # Look for breakout followed by reclaim
        breakout_found = False
        breakout_idx = -1

        for i in range(len(df) - cfg.reclaim_confirmation_bars):
            close_i = float(df.iloc[i]["close"])
            high_i = float(df.iloc[i]["high"])

            # Check for breakout
            if not breakout_found and high_i > zone_high:
                breakout_found = True
                breakout_idx = i
                continue

            # If we had a breakout, check for reclaim
            if breakout_found and i > breakout_idx:
                # Check if we have N consecutive closes below resistance
                consecutive_below = 0
                for j in range(i, min(i + cfg.reclaim_confirmation_bars, len(df))):
                    close_j = float(df.iloc[j]["close"])
                    if close_j < resistance:
                        consecutive_below += 1
                    else:
                        break

                if consecutive_below >= cfg.reclaim_confirmation_bars:
                    # Resistance reclaim detected!
                    date_str = str(df.iloc[i + cfg.reclaim_confirmation_bars - 1]["date"])
                    price = float(df.iloc[i + cfg.reclaim_confirmation_bars - 1]["close"])

                    # Calculate confidence based on strength of reclaim
                    reclaim_strength = (resistance - price) / resistance
                    confidence = min(0.95, 0.6 + reclaim_strength * 3.0)

                    patterns.append(
                        TrapReclaimPattern(
                            pattern_type="resistance_reclaim",
                            date=date_str,
                            price=price,
                            level=resistance,
                            confidence=confidence,
                            explanation=f"VIX reclaimed resistance at {resistance:.2f} after breakout, closing below for {consecutive_below} consecutive bars",
                        )
                    )
                    breakout_found = False  # Reset to look for next pattern
                    break

    return patterns


def get_pattern_glossary() -> dict[str, str]:
    """Return glossary definitions for trap/reclaim patterns.

    Requirements: 8.5
    """
    return {
        "bull_trap": "A false breakout where VIX breaks above resistance but quickly reverses back below, trapping traders who bought the breakout. Also called a 'failed breakout'.",
        "failed_breakout": "Same as bull trap - VIX breaks above resistance then reverses back below within a few bars.",
        "bear_trap": "A false breakdown where VIX breaks below support but quickly reverses back above, trapping traders who sold the breakdown. Also called a 'failed breakdown'.",
        "failed_breakdown": "Same as bear trap - VIX breaks below support then reverses back above within a few bars.",
        "support_reclaim": "VIX recovers and closes back above a support level after previously breaking below it, suggesting the support is holding.",
        "resistance_reclaim": "VIX falls back and closes below a resistance level after previously breaking above it, suggesting the resistance is holding.",
    }


def get_pattern_visual_markers() -> dict[str, dict[str, Any]]:
    """Return visual marker specifications for trap/reclaim patterns.
    
    This provides styling information for frontend chart rendering.
    
    Requirements: 14.7
    
    Returns:
        Dictionary mapping pattern types to visual marker specifications with:
        - color: Hex color code for the marker
        - symbol: Recommended chart symbol (triangle, circle, etc.)
        - label: Short label for the marker
        - position: Where to place marker relative to price (above/below)
    """
    return {
        "bull_trap": {
            "color": "#FF6B6B",  # Red
            "symbol": "triangle-down",
            "label": "Bull Trap",
            "position": "above",
            "description": "Failed breakout above resistance",
        },
        "bear_trap": {
            "color": "#51CF66",  # Green
            "symbol": "triangle-up",
            "label": "Bear Trap",
            "position": "below",
            "description": "Failed breakdown below support",
        },
        "support_reclaim": {
            "color": "#4DABF7",  # Blue
            "symbol": "circle",
            "label": "Support Reclaim",
            "position": "below",
            "description": "VIX reclaimed support after breakdown",
        },
        "resistance_reclaim": {
            "color": "#FAB005",  # Yellow/Orange
            "symbol": "circle",
            "label": "Resistance Reclaim",
            "position": "above",
            "description": "VIX reclaimed resistance after breakout",
        },
    }
