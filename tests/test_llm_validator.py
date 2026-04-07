"""
Unit tests for LLM output validator.

Tests Requirements 2.1, 2.6, 2.7, 7.2, 7.3, 7.6, 7.7.
"""

import pytest

from trader_koo.llm.schemas import (
    NarrativeGeneration,
    PatternExplanation,
    RegimeAnalysis,
    SetupRewrite,
)
from trader_koo.llm.validator import (
    ValidationResult,
    generate_fallback_narrative,
    generate_fallback_pattern_explanation,
    generate_fallback_regime_analysis,
    validate_llm_output,
)


class TestValidateLLMOutput:
    """Test LLM output validation against schemas."""

    def test_valid_setup_rewrite(self):
        """Test validation succeeds for valid SetupRewrite output."""
        output = {
            "observation": "SPY shows bullish setup in uptrend",
            "action": "Watch for entry above resistance",
            "risk_note": "Use stop losses",
        }
        result = validate_llm_output(output, SetupRewrite)

        assert result.is_valid
        assert result.data is not None
        assert result.data.observation == "SPY shows bullish setup in uptrend"
        assert result.data.action == "Watch for entry above resistance"
        assert result.data.risk_note == "Use stop losses"
        assert not result.errors
        assert not result.fallback_used

    def test_valid_setup_rewrite_without_risk_note(self):
        """Test validation succeeds when optional risk_note is missing."""
        output = {
            "observation": "SPY shows bullish setup",
            "action": "Watch for entry",
        }
        result = validate_llm_output(output, SetupRewrite)

        assert result.is_valid
        assert result.data is not None
        assert result.data.observation == "SPY shows bullish setup"
        assert result.data.action == "Watch for entry"
        assert result.data.risk_note == ""

    def test_invalid_missing_required_field(self):
        """Test validation fails when required field is missing."""
        output = {
            "observation": "SPY shows bullish setup",
            # Missing required "action" field
        }
        result = validate_llm_output(output, SetupRewrite)

        assert not result.is_valid
        assert result.data is None
        assert len(result.errors) > 0
        assert any("action" in error.lower() for error in result.errors)

    def test_invalid_field_too_long(self):
        """Test validation fails when field exceeds max length."""
        output = {
            "observation": "x" * 300,  # Max is 260
            "action": "Watch for entry",
        }
        result = validate_llm_output(output, SetupRewrite)

        assert not result.is_valid
        assert result.data is None
        assert len(result.errors) > 0

    def test_invalid_empty_required_field(self):
        """Test validation fails when required field is empty."""
        output = {
            "observation": "",
            "action": "Watch for entry",
        }
        result = validate_llm_output(output, SetupRewrite)

        assert not result.is_valid
        assert result.data is None

    def test_invalid_wrong_type(self):
        """Test validation fails when field has wrong type."""
        output = {
            "observation": 123,  # Should be string
            "action": "Watch for entry",
        }
        result = validate_llm_output(output, SetupRewrite)

        assert not result.is_valid
        assert result.data is None

    def test_invalid_not_dict(self):
        """Test validation fails when output is not a dictionary."""
        output = "not a dict"
        result = validate_llm_output(output, SetupRewrite)

        assert not result.is_valid
        assert result.data is None
        assert any("not a dictionary" in error for error in result.errors)

    def test_valid_narrative_generation(self):
        """Test validation succeeds for valid NarrativeGeneration output."""
        output = {
            "observation": "Market showing strong bullish momentum",
            "action": "Consider long positions with tight stops",
            "risk_note": "Monitor for reversal signals",
        }
        result = validate_llm_output(output, NarrativeGeneration)

        assert result.is_valid
        assert result.data is not None

    def test_valid_pattern_explanation(self):
        """Test validation succeeds for valid PatternExplanation output."""
        output = {
            "pattern_name": "bull_flag",
            "explanation": "A continuation pattern showing brief consolidation",
            "confidence": 0.85,
            "key_characteristics": ["Strong uptrend", "Consolidation", "Volume decline"],
        }
        result = validate_llm_output(output, PatternExplanation)

        assert result.is_valid
        assert result.data is not None
        assert result.data.confidence == 0.85

    def test_invalid_pattern_confidence_out_of_range(self):
        """Test validation fails when confidence is out of range."""
        output = {
            "pattern_name": "bull_flag",
            "explanation": "A continuation pattern",
            "confidence": 1.5,  # Must be 0-1
        }
        result = validate_llm_output(output, PatternExplanation)

        assert not result.is_valid

    def test_valid_regime_analysis(self):
        """Test validation succeeds for valid RegimeAnalysis output."""
        output = {
            "regime_type": "low_volatility",
            "summary": "Low volatility environment",
            "analysis": "Market showing low volatility with VIX below 15",
            "vix_context": {"vix": 12.5},
            "key_levels": [4200.0, 4150.0],
        }
        result = validate_llm_output(output, RegimeAnalysis)

        assert result.is_valid
        assert result.data is not None

    def test_validation_with_context(self):
        """Test validation includes context in logging."""
        output = {
            "observation": "SPY bullish",
            "action": "Buy",
        }
        context = {"ticker": "SPY", "source": "report"}
        result = validate_llm_output(output, SetupRewrite, context=context)

        assert result.is_valid


class TestGenerateFallbackNarrative:
    """Test fallback narrative generation."""

    def test_fallback_with_full_context(self):
        """Test fallback generates valid narrative with full context."""
        context = {
            "ticker": "SPY",
            "setup_tier": "premium",
            "signal_bias": "bullish",
            "trend_state": "uptrend",
        }
        result = generate_fallback_narrative(context)

        assert "observation" in result
        assert "action" in result
        assert "risk_note" in result
        assert "SPY" in result["observation"]
        assert "premium" in result["observation"]
        assert len(result["observation"]) > 0
        assert len(result["action"]) > 0
        assert len(result["risk_note"]) > 0

    def test_fallback_with_minimal_context(self):
        """Test fallback generates valid narrative with minimal context."""
        context = {"ticker": "AAPL"}
        result = generate_fallback_narrative(context)

        assert "observation" in result
        assert "action" in result
        assert "risk_note" in result
        assert "AAPL" in result["observation"]

    def test_fallback_with_empty_context(self):
        """Test fallback generates valid narrative with empty context."""
        context = {}
        result = generate_fallback_narrative(context)

        assert "observation" in result
        assert "action" in result
        assert "risk_note" in result

    def test_fallback_bullish_bias(self):
        """Test fallback generates bullish action for bullish bias."""
        context = {"ticker": "SPY", "signal_bias": "bullish"}
        result = generate_fallback_narrative(context)

        assert "entry above" in result["action"].lower() or "resistance" in result["action"].lower()

    def test_fallback_bearish_bias(self):
        """Test fallback generates bearish action for bearish bias."""
        context = {"ticker": "SPY", "signal_bias": "bearish"}
        result = generate_fallback_narrative(context)

        assert "breakdown" in result["action"].lower() or "support" in result["action"].lower()

    def test_fallback_neutral_bias(self):
        """Test fallback generates neutral action for neutral bias."""
        context = {"ticker": "SPY", "signal_bias": "neutral"}
        result = generate_fallback_narrative(context)

        assert "monitor" in result["action"].lower() or "wait" in result["action"].lower()


class TestGenerateFallbackPatternExplanation:
    """Test fallback pattern explanation generation."""

    def test_fallback_known_pattern(self):
        """Test fallback generates explanation for known pattern."""
        result = generate_fallback_pattern_explanation("bull_flag")

        assert result["pattern_name"] == "bull_flag"
        assert len(result["explanation"]) > 0
        assert "continuation" in result["explanation"].lower()
        assert result["confidence"] == 0.7
        assert len(result["key_characteristics"]) > 0

    def test_fallback_unknown_pattern(self):
        """Test fallback generates generic explanation for unknown pattern."""
        result = generate_fallback_pattern_explanation("unknown_pattern")

        assert result["pattern_name"] == "unknown_pattern"
        assert len(result["explanation"]) > 0
        assert "unknown_pattern" in result["explanation"]


class TestGenerateFallbackRegimeAnalysis:
    """Test fallback regime analysis generation."""

    def test_fallback_low_volatility(self):
        """Test fallback generates low volatility analysis."""
        result = generate_fallback_regime_analysis(vix_level=12.0)

        assert result["regime_type"] == "low_volatility"
        assert "low volatility" in result["summary"].lower()
        assert len(result["analysis"]) > 0
        assert result["vix_context"]["vix"] == 12.0

    def test_fallback_normal_volatility(self):
        """Test fallback generates normal volatility analysis."""
        result = generate_fallback_regime_analysis(vix_level=17.0)

        assert result["regime_type"] == "normal_volatility"
        assert "normal" in result["summary"].lower()

    def test_fallback_elevated_volatility(self):
        """Test fallback generates elevated volatility analysis."""
        result = generate_fallback_regime_analysis(vix_level=25.0)

        assert result["regime_type"] == "elevated_volatility"
        assert "elevated" in result["summary"].lower()

    def test_fallback_high_volatility(self):
        """Test fallback generates high volatility analysis."""
        result = generate_fallback_regime_analysis(vix_level=35.0)

        assert result["regime_type"] == "high_volatility"
        assert "high volatility" in result["summary"].lower()

    def test_fallback_no_vix_level(self):
        """Test fallback generates unknown regime when VIX unavailable."""
        result = generate_fallback_regime_analysis(vix_level=None)

        assert result["regime_type"] == "unknown"
        assert "unavailable" in result["summary"].lower()
        assert result["vix_context"] == {}


class TestValidationResult:
    """Test ValidationResult class."""

    def test_validation_result_success(self):
        """Test ValidationResult for successful validation."""
        data = SetupRewrite(observation="Test", action="Test action")
        result = ValidationResult(is_valid=True, data=data)

        assert result.is_valid
        assert result.data == data
        assert result.errors == []
        assert not result.fallback_used

    def test_validation_result_failure(self):
        """Test ValidationResult for failed validation."""
        errors = ["Field missing", "Invalid type"]
        result = ValidationResult(is_valid=False, errors=errors)

        assert not result.is_valid
        assert result.data is None
        assert result.errors == errors
        assert not result.fallback_used

    def test_validation_result_with_fallback(self):
        """Test ValidationResult with fallback flag."""
        result = ValidationResult(is_valid=True, fallback_used=True)

        assert result.is_valid
        assert result.fallback_used
