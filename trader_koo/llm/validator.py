"""
LLM Output Validator

Validates LLM responses against expected schemas with fallback logic.
Implements Requirements 2.1, 2.6, 2.7, 7.2, 7.3, 7.6, 7.7.
"""

from __future__ import annotations

import logging
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

from trader_koo.llm.schemas import (
    NarrativeGeneration,
    PatternExplanation,
    RegimeAnalysis,
    SetupRewrite,
)

LOG = logging.getLogger("trader_koo.llm.validator")

T = TypeVar("T", bound=BaseModel)


class ValidationResult:
    """Result of LLM output validation.

    Attributes:
        is_valid: Whether validation succeeded
        data: Validated data (None if validation failed)
        errors: List of validation error messages
        fallback_used: Whether fallback was triggered
    """

    def __init__(
        self,
        is_valid: bool,
        data: BaseModel | None = None,
        errors: list[str] | None = None,
        fallback_used: bool = False,
    ):
        self.is_valid = is_valid
        self.data = data
        self.errors = errors or []
        self.fallback_used = fallback_used


def validate_llm_output(
    output: dict[str, Any],
    schema: Type[T],
    *,
    context: dict[str, Any] | None = None,
) -> ValidationResult:
    """
    Validate LLM output against expected JSON schema.

    Implements Requirements 2.1, 2.6, 2.7, 7.2, 7.3, 7.6, 7.7.

    Args:
        output: Raw LLM output dictionary
        schema: Pydantic model class to validate against
        context: Optional context for logging (ticker, source, etc.)

    Returns:
        ValidationResult with validation status and data or errors

    Example:
        >>> result = validate_llm_output(
        ...     {"observation": "Market is bullish", "action": "Buy"},
        ...     NarrativeGeneration
        ... )
        >>> if result.is_valid:
        ...     print(result.data.observation)
    """
    context = context or {}

    if not isinstance(output, dict):
        error_msg = f"LLM output is not a dictionary: {type(output).__name__}"
        LOG.warning(
            "LLM validation failed: %s",
            error_msg,
            extra={"context": context, "output_type": type(output).__name__}
        )
        return ValidationResult(
            is_valid=False,
            errors=[error_msg]
        )

    try:
        # Validate against schema
        validated_data = schema(**output)

        LOG.debug(
            "LLM validation succeeded",
            extra={
                "context": context,
                "schema": schema.__name__,
                "field_count": len(output)
            }
        )

        return ValidationResult(
            is_valid=True,
            data=validated_data
        )

    except ValidationError as e:
        # Extract validation errors
        errors = []
        for error in e.errors():
            field_path = ".".join(str(loc) for loc in error["loc"])
            error_type = error["type"]
            error_msg = error["msg"]
            errors.append(f"{field_path}: {error_msg} (type: {error_type})")

        LOG.warning(
            "LLM validation failed: schema validation errors",
            extra={
                "context": context,
                "schema": schema.__name__,
                "error_count": len(errors),
                "errors": errors,
                "output_keys": list(output.keys()) if isinstance(output, dict) else None
            }
        )

        return ValidationResult(
            is_valid=False,
            errors=errors
        )

    except Exception as e:
        error_msg = f"Unexpected validation error: {type(e).__name__}: {str(e)}"
        LOG.error(
            "LLM validation failed: unexpected error",
            extra={
                "context": context,
                "schema": schema.__name__,
                "error": error_msg
            },
            exc_info=True
        )

        return ValidationResult(
            is_valid=False,
            errors=[error_msg]
        )


def generate_fallback_narrative(context: dict[str, Any]) -> dict[str, str]:
    """
    Generate deterministic rule-based narrative as fallback.

    Implements Requirements 2.6, 2.7.

    Args:
        context: Context data for narrative generation

    Returns:
        Dictionary with observation, action, and risk_note fields
    """
    ticker = context.get("ticker", "UNKNOWN")
    setup_tier = context.get("setup_tier", "unknown")
    signal_bias = context.get("signal_bias", "neutral")
    trend_state = context.get("trend_state", "unknown")

    # Generate observation
    observation_parts = [f"{ticker} shows {setup_tier} setup"]
    if trend_state and trend_state != "unknown":
        observation_parts.append(f"in {trend_state} trend")
    if signal_bias and signal_bias != "neutral":
        observation_parts.append(f"with {signal_bias} bias")
    observation = " ".join(observation_parts) + "."

    # Generate action
    if signal_bias == "bullish":
        action = f"Watch for entry above key resistance. Monitor volume confirmation."
    elif signal_bias == "bearish":
        action = f"Watch for breakdown below support. Consider protective stops."
    else:
        action = f"Monitor price action at key levels. Wait for directional confirmation."

    # Generate risk note
    risk_note = "Manage position size. Use stop losses. Monitor market conditions."

    return {
        "observation": observation,
        "action": action,
        "risk_note": risk_note,
    }


def generate_fallback_pattern_explanation(pattern_name: str) -> dict[str, Any]:
    """
    Generate rule-based pattern explanation as fallback.

    Implements Requirements 2.6, 2.7.

    Args:
        pattern_name: Name of the pattern

    Returns:
        Dictionary with pattern explanation fields
    """
    # Generic pattern explanations
    explanations = {
        "bull_flag": "A continuation pattern showing brief consolidation after strong uptrend.",
        "bear_flag": "A continuation pattern showing brief consolidation after strong downtrend.",
        "head_and_shoulders": "A reversal pattern indicating potential trend change.",
        "double_top": "A bearish reversal pattern showing resistance at similar price levels.",
        "double_bottom": "A bullish reversal pattern showing support at similar price levels.",
    }

    explanation = explanations.get(
        pattern_name.lower(),
        f"Technical pattern detected: {pattern_name}. Monitor for confirmation."
    )

    return {
        "pattern_name": pattern_name,
        "explanation": explanation,
        "confidence": 0.7,
        "key_characteristics": [
            "Pattern detected by technical analysis",
            "Requires confirmation",
            "Monitor volume and price action"
        ]
    }


def generate_fallback_regime_analysis(vix_level: float | None = None) -> dict[str, Any]:
    """
    Generate rule-based regime analysis as fallback.

    Implements Requirements 2.6, 2.7.

    Args:
        vix_level: Current VIX level (optional)

    Returns:
        Dictionary with regime analysis fields
    """
    if vix_level is None:
        regime_type = "unknown"
        summary = "Market regime analysis unavailable."
        analysis = "Unable to determine current market regime. Monitor VIX and market conditions."
    elif vix_level < 15:
        regime_type = "low_volatility"
        summary = "Low volatility environment with VIX below 15."
        analysis = "Market showing low volatility. Favorable for trend-following strategies. Monitor for volatility expansion."
    elif vix_level < 20:
        regime_type = "normal_volatility"
        summary = "Normal volatility environment with VIX in typical range."
        analysis = "Market volatility in normal range. Standard risk management applies. Watch for regime shifts."
    elif vix_level < 30:
        regime_type = "elevated_volatility"
        summary = "Elevated volatility with VIX above 20."
        analysis = "Market showing elevated volatility. Increase caution. Reduce position sizes. Monitor for stabilization."
    else:
        regime_type = "high_volatility"
        summary = "High volatility environment with VIX above 30."
        analysis = "Market in high volatility regime. Extreme caution advised. Consider defensive positioning. Wait for stabilization."

    return {
        "regime_type": regime_type,
        "summary": summary,
        "analysis": analysis,
        "vix_context": {"vix": vix_level} if vix_level is not None else {},
        "key_levels": [],
    }
