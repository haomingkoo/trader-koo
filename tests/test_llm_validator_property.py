"""
Property-based tests for LLM output validation.

Tests universal properties of LLM validation across all inputs.
Validates Requirements 2.1, 2.2, 2.6, 2.7, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7.
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from trader_koo.llm.schemas import (
    NarrativeGeneration,
    PatternExplanation,
    RegimeAnalysis,
    SetupRewrite,
)
from trader_koo.llm.validator import (
    ValidationResult,
    validate_llm_output,
    generate_fallback_narrative,
    generate_fallback_pattern_explanation,
    generate_fallback_regime_analysis,
)


# Strategy for generating text with various characteristics
text_strategy = st.text(min_size=0, max_size=10000)
short_text_strategy = st.text(min_size=0, max_size=100)
html_text_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "P")),
    min_size=0,
    max_size=1000
).map(lambda s: f"<p>{s}</p>" if s else s)


class TestProperty2LlmOutputValidation:
    """
    Feature: enterprise-platform-upgrade, Property 2: LLM Output Schema Validation with Fallback

    **Validates: Requirements 2.1, 2.2, 2.6, 2.7, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7**

    For any LLM response, the platform should validate it against the expected JSON schema,
    and if validation fails (missing required fields, wrong types, or length violations),
    then the platform should log the validation error with context and fall back to
    deterministic rule-based output.
    """

    @given(
        observation=text_strategy,
        action=text_strategy,
        risk_note=text_strategy,
    )
    def test_narrative_validation_accepts_valid_input(
        self,
        observation: str,
        action: str,
        risk_note: str,
    ):
        """Valid narrative input should pass validation."""
        # Only test with valid lengths
        if not observation or len(observation) > 5000:
            return
        if not action or len(action) > 5000:
            return
        if len(risk_note) > 1000:
            return

        output = {
            "observation": observation,
            "action": action,
            "risk_note": risk_note,
        }

        result = validate_llm_output(output, NarrativeGeneration)

        # Should succeed for valid input
        assert result.is_valid
        assert result.data is not None
        assert isinstance(result.data, NarrativeGeneration)
        assert result.data.observation.strip() == observation.strip()
        assert result.data.action.strip() == action.strip()

    @settings(
        suppress_health_check=[
            HealthCheck.large_base_example,
            HealthCheck.data_too_large,
            HealthCheck.too_slow,
        ]
    )
    @given(
        observation=st.text(min_size=5001, max_size=6000),
        action=st.text(min_size=1, max_size=100),
    )
    def test_narrative_validation_rejects_too_long_observation(
        self,
        observation: str,
        action: str,
    ):
        """Observation exceeding 5000 chars should fail validation."""
        output = {
            "observation": observation,
            "action": action,
        }

        result = validate_llm_output(output, NarrativeGeneration)

        # Should fail due to length violation
        assert not result.is_valid
        assert result.data is None
        assert len(result.errors) > 0

    @settings(
        suppress_health_check=[
            HealthCheck.large_base_example,
            HealthCheck.data_too_large,
            HealthCheck.too_slow,
        ]
    )
    @given(
        action=st.text(min_size=5001, max_size=6000),
    )
    def test_narrative_validation_rejects_too_long_action(self, action: str):
        """Action exceeding 5000 chars should fail validation."""
        output = {
            "observation": "Valid observation",
            "action": action,
        }

        result = validate_llm_output(output, NarrativeGeneration)

        # Should fail due to length violation
        assert not result.is_valid
        assert result.data is None
        assert len(result.errors) > 0

    @given(
        risk_note=st.text(min_size=1001, max_size=2000),
    )
    def test_narrative_validation_rejects_too_long_risk_note(self, risk_note: str):
        """Risk note exceeding 1000 chars should fail validation."""
        output = {
            "observation": "Valid observation",
            "action": "Valid action",
            "risk_note": risk_note,
        }

        result = validate_llm_output(output, NarrativeGeneration)

        # Should fail due to length violation
        assert not result.is_valid
        assert result.data is None
        assert len(result.errors) > 0

    @given(
        missing_field=st.sampled_from(["observation", "action"]),
    )
    def test_narrative_validation_rejects_missing_required_fields(
        self,
        missing_field: str,
    ):
        """Missing required fields should fail validation."""
        output = {
            "observation": "Valid observation",
            "action": "Valid action",
        }
        del output[missing_field]

        result = validate_llm_output(output, NarrativeGeneration)

        # Should fail due to missing required field
        assert not result.is_valid
        assert result.data is None
        assert len(result.errors) > 0
        assert any(missing_field in error for error in result.errors)

    @given(
        pattern_name=short_text_strategy,
        explanation=text_strategy,
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_pattern_explanation_validation_accepts_valid_input(
        self,
        pattern_name: str,
        explanation: str,
        confidence: float,
    ):
        """Valid pattern explanation input should pass validation."""
        # Only test with valid lengths
        if not pattern_name or len(pattern_name) > 100:
            return
        if not explanation or len(explanation) > 5000:
            return

        output = {
            "pattern_name": pattern_name,
            "explanation": explanation,
            "confidence": confidence,
        }

        result = validate_llm_output(output, PatternExplanation)

        # Should succeed for valid input
        assert result.is_valid
        assert result.data is not None
        assert isinstance(result.data, PatternExplanation)
        assert result.data.confidence == confidence

    @given(
        confidence=st.floats(min_value=-10.0, max_value=-0.01) | st.floats(min_value=1.01, max_value=10.0),
    )
    def test_pattern_explanation_validation_rejects_invalid_confidence(
        self,
        confidence: float,
    ):
        """Confidence outside [0, 1] range should fail validation."""
        output = {
            "pattern_name": "test_pattern",
            "explanation": "Test explanation",
            "confidence": confidence,
        }

        result = validate_llm_output(output, PatternExplanation)

        # Should fail due to confidence out of range
        assert not result.is_valid
        assert result.data is None
        assert len(result.errors) > 0

    @given(
        regime_type=short_text_strategy,
        summary=st.text(min_size=1, max_size=1000),
        analysis=text_strategy,
    )
    def test_regime_analysis_validation_accepts_valid_input(
        self,
        regime_type: str,
        summary: str,
        analysis: str,
    ):
        """Valid regime analysis input should pass validation."""
        # Only test with valid lengths
        if not regime_type or len(regime_type) > 50:
            return
        if not analysis or len(analysis) > 5000:
            return

        output = {
            "regime_type": regime_type,
            "summary": summary,
            "analysis": analysis,
        }

        result = validate_llm_output(output, RegimeAnalysis)

        # Should succeed for valid input
        assert result.is_valid
        assert result.data is not None
        assert isinstance(result.data, RegimeAnalysis)

    @given(
        summary=st.text(min_size=1001, max_size=2000),
    )
    def test_regime_analysis_validation_rejects_too_long_summary(self, summary: str):
        """Summary exceeding 1000 chars should fail validation."""
        output = {
            "regime_type": "test_regime",
            "summary": summary,
            "analysis": "Valid analysis",
        }

        result = validate_llm_output(output, RegimeAnalysis)

        # Should fail due to length violation
        assert not result.is_valid
        assert result.data is None
        assert len(result.errors) > 0

    @given(
        observation=st.text(min_size=1, max_size=260),
        action=st.text(min_size=1, max_size=180),
        risk_note=st.text(min_size=0, max_size=80),
    )
    def test_setup_rewrite_validation_accepts_valid_input(
        self,
        observation: str,
        action: str,
        risk_note: str,
    ):
        """Valid setup rewrite input should pass validation."""
        output = {
            "observation": observation,
            "action": action,
            "risk_note": risk_note,
        }

        result = validate_llm_output(output, SetupRewrite)

        # Should succeed for valid input
        assert result.is_valid
        assert result.data is not None
        assert isinstance(result.data, SetupRewrite)

    @given(
        observation=st.text(min_size=261, max_size=500),
    )
    def test_setup_rewrite_validation_rejects_too_long_observation(
        self,
        observation: str,
    ):
        """Setup observation exceeding 260 chars should fail validation."""
        output = {
            "observation": observation,
            "action": "Valid action",
        }

        result = validate_llm_output(output, SetupRewrite)

        # Should fail due to length violation
        assert not result.is_valid
        assert result.data is None
        assert len(result.errors) > 0

    @given(
        invalid_type=st.sampled_from([123, True, None, ["list"]]),
    )
    def test_validation_rejects_non_dict_input(self, invalid_type):
        """Non-dictionary input should fail validation."""
        result = validate_llm_output(invalid_type, NarrativeGeneration)

        # Should fail for non-dict input
        assert not result.is_valid
        assert result.data is None
        assert len(result.errors) > 0

    @given(
        ticker=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Lu",))),
        setup_tier=st.sampled_from(["A", "B", "C", "unknown"]),
        signal_bias=st.sampled_from(["bullish", "bearish", "neutral"]),
    )
    def test_fallback_narrative_always_returns_valid_structure(
        self,
        ticker: str,
        setup_tier: str,
        signal_bias: str,
    ):
        """Fallback narrative should always return valid structure."""
        context = {
            "ticker": ticker,
            "setup_tier": setup_tier,
            "signal_bias": signal_bias,
        }

        fallback = generate_fallback_narrative(context)

        # Should always return valid structure
        assert isinstance(fallback, dict)
        assert "observation" in fallback
        assert "action" in fallback
        assert "risk_note" in fallback
        assert len(fallback["observation"]) > 0
        assert len(fallback["action"]) > 0
        assert len(fallback["risk_note"]) > 0

        # Should validate against schema
        result = validate_llm_output(fallback, NarrativeGeneration)
        assert result.is_valid

    @given(
        pattern_name=st.text(min_size=1, max_size=50),
    )
    def test_fallback_pattern_explanation_always_returns_valid_structure(
        self,
        pattern_name: str,
    ):
        """Fallback pattern explanation should always return valid structure."""
        fallback = generate_fallback_pattern_explanation(pattern_name)

        # Should always return valid structure
        assert isinstance(fallback, dict)
        assert "pattern_name" in fallback
        assert "explanation" in fallback
        assert "confidence" in fallback
        assert "key_characteristics" in fallback
        assert 0.0 <= fallback["confidence"] <= 1.0

        # Should validate against schema
        result = validate_llm_output(fallback, PatternExplanation)
        assert result.is_valid

    @given(
        vix_level=st.one_of(
            st.none(),
            st.floats(min_value=5.0, max_value=80.0, allow_nan=False, allow_infinity=False)
        ),
    )
    def test_fallback_regime_analysis_always_returns_valid_structure(
        self,
        vix_level: float | None,
    ):
        """Fallback regime analysis should always return valid structure."""
        fallback = generate_fallback_regime_analysis(vix_level)

        # Should always return valid structure
        assert isinstance(fallback, dict)
        assert "regime_type" in fallback
        assert "summary" in fallback
        assert "analysis" in fallback
        assert "vix_context" in fallback
        assert "key_levels" in fallback
        assert len(fallback["summary"]) <= 1000
        assert len(fallback["analysis"]) <= 5000

        # Should validate against schema
        result = validate_llm_output(fallback, RegimeAnalysis)
        assert result.is_valid
