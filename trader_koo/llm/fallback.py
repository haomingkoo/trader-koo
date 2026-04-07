"""
Deterministic Fallback Logic for LLM Failures

Provides template-based narrative generation when LLM service fails.
Implements Requirements 2.6, 2.7.
"""

from __future__ import annotations

import logging
from typing import Any

LOG = logging.getLogger("trader_koo.llm.fallback")


def generate_template_narrative(context: dict[str, Any]) -> dict[str, str]:
    """
    Generate deterministic template-based narrative.

    This is used as a fallback when LLM service fails or returns invalid output.
    Implements Requirements 2.6, 2.7.

    Args:
        context: Context data including ticker, setup info, technical indicators

    Returns:
        Dictionary with observation, action, and risk_note fields

    Example:
        >>> context = {
        ...     "ticker": "AAPL",
        ...     "setup_tier": "A",
        ...     "signal_bias": "bullish",
        ...     "trend_state": "uptrend"
        ... }
        >>> narrative = generate_template_narrative(context)
        >>> print(narrative["observation"])
        AAPL shows A setup in uptrend with bullish bias.
    """
    ticker = context.get("ticker", "UNKNOWN")
    setup_tier = context.get("setup_tier", "unknown")
    signal_bias = context.get("signal_bias", "neutral")
    trend_state = context.get("trend_state", "unknown")
    level_context = context.get("level_context", "")
    yolo_pattern = context.get("yolo_pattern", "")

    # Build observation from context
    observation_parts = [f"{ticker} shows {setup_tier} setup"]

    if trend_state and trend_state != "unknown":
        observation_parts.append(f"in {trend_state}")

    if signal_bias and signal_bias != "neutral":
        observation_parts.append(f"with {signal_bias} bias")

    if yolo_pattern:
        observation_parts.append(f"Pattern: {yolo_pattern}")

    if level_context:
        observation_parts.append(f"Level: {level_context}")

    observation = " ".join(observation_parts) + "."

    # Generate action based on bias
    if signal_bias == "bullish":
        action = (
            f"Watch for entry above key resistance. "
            f"Monitor volume confirmation and price action. "
            f"Consider scaling in on strength."
        )
    elif signal_bias == "bearish":
        action = (
            f"Watch for breakdown below support. "
            f"Consider protective stops. "
            f"Monitor for reversal signals."
        )
    else:
        action = (
            f"Monitor price action at key levels. "
            f"Wait for directional confirmation. "
            f"Avoid premature entries."
        )

    # Generate risk note
    risk_note = (
        "Manage position size appropriately. "
        "Use stop losses. "
        "Monitor market conditions."
    )

    LOG.info(
        "Generated template narrative fallback",
        extra={
            "ticker": ticker,
            "setup_tier": setup_tier,
            "signal_bias": signal_bias
        }
    )

    return {
        "observation": observation,
        "action": action,
        "risk_note": risk_note,
    }


def generate_rule_based_pattern_explanation(pattern_name: str) -> dict[str, Any]:
    """
    Generate rule-based pattern explanation.

    Implements Requirements 2.6, 2.7.

    Args:
        pattern_name: Name of the detected pattern

    Returns:
        Dictionary with pattern explanation fields
    """
    # Pattern explanation templates
    pattern_templates = {
        "bull_flag": {
            "explanation": (
                "A continuation pattern showing brief consolidation after strong uptrend. "
                "Characterized by parallel trendlines forming a flag shape. "
                "Typically resolves with breakout to the upside."
            ),
            "characteristics": [
                "Strong prior uptrend",
                "Consolidation with lower volume",
                "Parallel trendlines",
                "Breakout on increased volume"
            ]
        },
        "bear_flag": {
            "explanation": (
                "A continuation pattern showing brief consolidation after strong downtrend. "
                "Characterized by parallel trendlines forming a flag shape. "
                "Typically resolves with breakdown to the downside."
            ),
            "characteristics": [
                "Strong prior downtrend",
                "Consolidation with lower volume",
                "Parallel trendlines",
                "Breakdown on increased volume"
            ]
        },
        "head_and_shoulders": {
            "explanation": (
                "A reversal pattern indicating potential trend change from bullish to bearish. "
                "Consists of three peaks with the middle peak (head) higher than the others (shoulders). "
                "Confirmed on break below neckline."
            ),
            "characteristics": [
                "Three distinct peaks",
                "Middle peak highest",
                "Neckline support",
                "Volume decreases through pattern"
            ]
        },
        "double_top": {
            "explanation": (
                "A bearish reversal pattern showing resistance at similar price levels. "
                "Two peaks at approximately the same level indicate strong resistance. "
                "Confirmed on break below support between peaks."
            ),
            "characteristics": [
                "Two peaks at similar levels",
                "Strong resistance zone",
                "Support between peaks",
                "Volume typically lower on second peak"
            ]
        },
        "double_bottom": {
            "explanation": (
                "A bullish reversal pattern showing support at similar price levels. "
                "Two troughs at approximately the same level indicate strong support. "
                "Confirmed on break above resistance between troughs."
            ),
            "characteristics": [
                "Two troughs at similar levels",
                "Strong support zone",
                "Resistance between troughs",
                "Volume typically higher on second trough"
            ]
        },
    }

    # Get template or use generic
    template = pattern_templates.get(
        pattern_name.lower().replace(" ", "_"),
        {
            "explanation": f"Technical pattern detected: {pattern_name}. Monitor for confirmation and volume.",
            "characteristics": [
                "Pattern detected by technical analysis",
                "Requires confirmation",
                "Monitor volume and price action"
            ]
        }
    )

    return {
        "pattern_name": pattern_name,
        "explanation": template["explanation"],
        "confidence": 0.7,
        "key_characteristics": template.get("characteristics", [])
    }
