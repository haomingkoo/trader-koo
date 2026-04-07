"""
LLM Response Schemas

Pydantic models for validating LLM output formats.
Implements Requirements 2.1, 7.1, 7.2, 7.4, 7.5.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class NarrativeGeneration(BaseModel):
    """Schema for narrative generation responses.

    Validates Requirements 2.1, 2.2, 7.4, 7.5.
    """

    observation: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Market observation narrative"
    )
    action: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Recommended action narrative"
    )
    risk_note: str = Field(
        default="",
        max_length=1000,
        description="Risk assessment summary"
    )

    @field_validator("observation", "action", "risk_note")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Strip leading/trailing whitespace from text fields."""
        return v.strip() if v else ""


class PatternExplanation(BaseModel):
    """Schema for pattern explanation responses.

    Validates Requirements 2.1, 7.4, 7.5.
    """

    pattern_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the detected pattern"
    )
    explanation: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Detailed pattern explanation"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )
    key_characteristics: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="List of key pattern characteristics"
    )

    @field_validator("pattern_name", "explanation")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Strip leading/trailing whitespace from text fields."""
        return v.strip() if v else ""

    @field_validator("key_characteristics")
    @classmethod
    def validate_characteristics(cls, v: list[str]) -> list[str]:
        """Validate and clean characteristic strings."""
        return [s.strip() for s in v if s and s.strip()][:10]


class RegimeAnalysis(BaseModel):
    """Schema for regime analysis responses.

    Validates Requirements 2.1, 7.4, 7.5.
    """

    regime_type: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Current market regime type"
    )
    summary: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Regime summary (max 1000 chars)"
    )
    analysis: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Detailed regime analysis"
    )
    vix_context: dict[str, Any] = Field(
        default_factory=dict,
        description="VIX context data"
    )
    key_levels: list[float] = Field(
        default_factory=list,
        max_length=20,
        description="Key price levels"
    )

    @field_validator("regime_type", "summary", "analysis")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Strip leading/trailing whitespace from text fields."""
        return v.strip() if v else ""

    @field_validator("key_levels")
    @classmethod
    def validate_levels(cls, v: list[float]) -> list[float]:
        """Validate key levels are positive numbers."""
        return [float(level) for level in v if level > 0][:20]


class SetupRewrite(BaseModel):
    """Schema for setup copy rewrite responses.

    Validates Requirements 2.1, 2.2, 7.4, 7.5.
    Used by existing maybe_rewrite_setup_copy function.
    """

    observation: str = Field(
        ...,
        min_length=1,
        max_length=260,
        description="Rewritten observation (max 260 chars)"
    )
    action: str = Field(
        ...,
        min_length=1,
        max_length=180,
        description="Rewritten action (max 180 chars)"
    )
    risk_note: str = Field(
        default="",
        max_length=80,
        description="Rewritten risk note (max 80 chars)"
    )

    @field_validator("observation", "action", "risk_note")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Strip leading/trailing whitespace from text fields."""
        return v.strip() if v else ""
