"""
Unit tests for adaptive compression thresholds.

Tests Requirements:
- 13.1: Calculate dynamic thresholds based on 90-day VIX percentile
- 13.2: Use tight thresholds in low vol (20th/80th percentile when VIX percentile < 30)
- 13.3: Use moderate thresholds in normal vol (25th/75th percentile when VIX percentile 30-70)
- 13.4: Use wide thresholds in high vol (30th/70th percentile when VIX percentile > 70)
- 13.5: Display current thresholds in VIX analysis tab
- 13.6: Label compression signals with threshold regime (tight/moderate/wide)
"""

import sqlite3
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from trader_koo.structure.vix_analysis import (
    CompressionThresholds,
    calculate_compression_thresholds,
    detect_compression_signal,
    format_compression_thresholds_display,
)


@pytest.fixture
def mock_conn():
    """Create a mock database connection."""
    return MagicMock(spec=sqlite3.Connection)


def test_compression_thresholds_tight_regime(mock_conn):
    """
    Test tight thresholds in low volatility regime.

    Requirement 13.2: When VIX 90-day percentile < 30, use tight thresholds (20/80).
    """
    # Mock VIX percentile calculation to return low volatility (< 30)
    with patch("trader_koo.structure.vix_analysis.calculate_vix_percentile") as mock_calc:
        mock_calc.return_value = 25.0  # Low volatility

        thresholds = calculate_compression_thresholds(mock_conn)

        assert thresholds.lower_percentile == 20.0
        assert thresholds.upper_percentile == 80.0
        assert thresholds.regime == "tight"
        assert thresholds.vix_90d_percentile == 25.0

        # Verify it called with 90-day window
        mock_calc.assert_called_once_with(mock_conn, window_days=90)


def test_compression_thresholds_moderate_regime_lower_bound(mock_conn):
    """
    Test moderate thresholds at lower bound of normal volatility.

    Requirement 13.3: When VIX 90-day percentile is 30-70, use moderate thresholds (25/75).
    """
    with patch("trader_koo.structure.vix_analysis.calculate_vix_percentile") as mock_calc:
        mock_calc.return_value = 30.0  # Exactly at lower bound

        thresholds = calculate_compression_thresholds(mock_conn)

        assert thresholds.lower_percentile == 25.0
        assert thresholds.upper_percentile == 75.0
        assert thresholds.regime == "moderate"
        assert thresholds.vix_90d_percentile == 30.0


def test_compression_thresholds_moderate_regime_mid(mock_conn):
    """
    Test moderate thresholds in middle of normal volatility range.

    Requirement 13.3: When VIX 90-day percentile is 30-70, use moderate thresholds (25/75).
    """
    with patch("trader_koo.structure.vix_analysis.calculate_vix_percentile") as mock_calc:
        mock_calc.return_value = 50.0  # Middle of range

        thresholds = calculate_compression_thresholds(mock_conn)

        assert thresholds.lower_percentile == 25.0
        assert thresholds.upper_percentile == 75.0
        assert thresholds.regime == "moderate"
        assert thresholds.vix_90d_percentile == 50.0


def test_compression_thresholds_moderate_regime_upper_bound(mock_conn):
    """
    Test moderate thresholds at upper bound of normal volatility.

    Requirement 13.3: When VIX 90-day percentile is 30-70, use moderate thresholds (25/75).
    """
    with patch("trader_koo.structure.vix_analysis.calculate_vix_percentile") as mock_calc:
        mock_calc.return_value = 70.0  # Exactly at upper bound

        thresholds = calculate_compression_thresholds(mock_conn)

        assert thresholds.lower_percentile == 25.0
        assert thresholds.upper_percentile == 75.0
        assert thresholds.regime == "moderate"
        assert thresholds.vix_90d_percentile == 70.0


def test_compression_thresholds_wide_regime(mock_conn):
    """
    Test wide thresholds in high volatility regime.

    Requirement 13.4: When VIX 90-day percentile > 70, use wide thresholds (30/70).
    """
    with patch("trader_koo.structure.vix_analysis.calculate_vix_percentile") as mock_calc:
        mock_calc.return_value = 85.0  # High volatility

        thresholds = calculate_compression_thresholds(mock_conn)

        assert thresholds.lower_percentile == 30.0
        assert thresholds.upper_percentile == 70.0
        assert thresholds.regime == "wide"
        assert thresholds.vix_90d_percentile == 85.0


def test_compression_thresholds_boundary_29_percent(mock_conn):
    """
    Test threshold selection at 29% (should be tight).

    Requirement 13.2: VIX percentile < 30 should use tight thresholds.
    """
    with patch("trader_koo.structure.vix_analysis.calculate_vix_percentile") as mock_calc:
        mock_calc.return_value = 29.0

        thresholds = calculate_compression_thresholds(mock_conn)

        assert thresholds.regime == "tight"
        assert thresholds.lower_percentile == 20.0
        assert thresholds.upper_percentile == 80.0


def test_compression_thresholds_boundary_71_percent(mock_conn):
    """
    Test threshold selection at 71% (should be wide).

    Requirement 13.4: VIX percentile > 70 should use wide thresholds.
    """
    with patch("trader_koo.structure.vix_analysis.calculate_vix_percentile") as mock_calc:
        mock_calc.return_value = 71.0

        thresholds = calculate_compression_thresholds(mock_conn)

        assert thresholds.regime == "wide"
        assert thresholds.lower_percentile == 30.0
        assert thresholds.upper_percentile == 70.0


def test_compression_thresholds_fallback_on_none(mock_conn):
    """
    Test fallback to moderate thresholds when percentile calculation fails.

    Requirement 13.1: System should handle missing data gracefully.
    """
    with patch("trader_koo.structure.vix_analysis.calculate_vix_percentile") as mock_calc:
        mock_calc.return_value = None  # Calculation failed

        thresholds = calculate_compression_thresholds(mock_conn)

        # Should default to moderate thresholds
        assert thresholds.lower_percentile == 25.0
        assert thresholds.upper_percentile == 75.0
        assert thresholds.regime == "moderate"
        assert thresholds.vix_90d_percentile == 50.0  # Default value


def test_detect_compression_signal_tight_compression(mock_conn):
    """
    Test compression detection with tight thresholds.

    Requirement 13.6: Label compression signals with threshold regime.
    """
    thresholds = CompressionThresholds(
        lower_percentile=20.0,
        upper_percentile=80.0,
        regime="tight",
        vix_90d_percentile=25.0,
    )

    # BB width at 15th percentile - should be compression
    state, labeled = detect_compression_signal(15.0, thresholds)

    assert state == "compression"
    assert labeled == "compression (tight)"


def test_detect_compression_signal_tight_expansion(mock_conn):
    """
    Test expansion detection with tight thresholds.

    Requirement 13.6: Label compression signals with threshold regime.
    """
    thresholds = CompressionThresholds(
        lower_percentile=20.0,
        upper_percentile=80.0,
        regime="tight",
        vix_90d_percentile=25.0,
    )

    # BB width at 85th percentile - should be expansion
    state, labeled = detect_compression_signal(85.0, thresholds)

    assert state == "expansion"
    assert labeled == "expansion (tight)"


def test_detect_compression_signal_moderate_compression(mock_conn):
    """
    Test compression detection with moderate thresholds.

    Requirement 13.6: Label compression signals with threshold regime.
    """
    thresholds = CompressionThresholds(
        lower_percentile=25.0,
        upper_percentile=75.0,
        regime="moderate",
        vix_90d_percentile=50.0,
    )

    # BB width at 20th percentile - should be compression
    state, labeled = detect_compression_signal(20.0, thresholds)

    assert state == "compression"
    assert labeled == "compression (moderate)"


def test_detect_compression_signal_wide_compression(mock_conn):
    """
    Test compression detection with wide thresholds.

    Requirement 13.6: Label compression signals with threshold regime.
    """
    thresholds = CompressionThresholds(
        lower_percentile=30.0,
        upper_percentile=70.0,
        regime="wide",
        vix_90d_percentile=85.0,
    )

    # BB width at 25th percentile - should be compression
    state, labeled = detect_compression_signal(25.0, thresholds)

    assert state == "compression"
    assert labeled == "compression (wide)"


def test_detect_compression_signal_normal_state(mock_conn):
    """
    Test normal state detection (between thresholds).

    Requirement 13.6: Detect when volatility is in normal range.
    """
    thresholds = CompressionThresholds(
        lower_percentile=25.0,
        upper_percentile=75.0,
        regime="moderate",
        vix_90d_percentile=50.0,
    )

    # BB width at 50th percentile - should be normal
    state, labeled = detect_compression_signal(50.0, thresholds)

    assert state == "normal"
    assert labeled == "normal"


def test_detect_compression_signal_none_input(mock_conn):
    """
    Test compression detection with None input.

    Requirement 13.6: Handle missing BB width data gracefully.
    """
    thresholds = CompressionThresholds(
        lower_percentile=25.0,
        upper_percentile=75.0,
        regime="moderate",
        vix_90d_percentile=50.0,
    )

    state, labeled = detect_compression_signal(None, thresholds)

    assert state == "normal"
    assert labeled == "normal"


def test_detect_compression_signal_boundary_at_lower_threshold(mock_conn):
    """
    Test compression detection exactly at lower threshold.

    Requirement 13.6: Boundary condition testing.
    """
    thresholds = CompressionThresholds(
        lower_percentile=25.0,
        upper_percentile=75.0,
        regime="moderate",
        vix_90d_percentile=50.0,
    )

    # Exactly at lower threshold - should be compression
    state, labeled = detect_compression_signal(25.0, thresholds)

    assert state == "compression"
    assert labeled == "compression (moderate)"


def test_detect_compression_signal_boundary_at_upper_threshold(mock_conn):
    """
    Test expansion detection exactly at upper threshold.

    Requirement 13.6: Boundary condition testing.
    """
    thresholds = CompressionThresholds(
        lower_percentile=25.0,
        upper_percentile=75.0,
        regime="moderate",
        vix_90d_percentile=50.0,
    )

    # Exactly at upper threshold - should be expansion
    state, labeled = detect_compression_signal(75.0, thresholds)

    assert state == "expansion"
    assert labeled == "expansion (moderate)"


def test_format_compression_thresholds_display_tight(mock_conn):
    """
    Test display formatting for tight regime.

    Requirement 13.5: Display current thresholds in VIX analysis tab.
    """
    thresholds = CompressionThresholds(
        lower_percentile=20.0,
        upper_percentile=80.0,
        regime="tight",
        vix_90d_percentile=25.0,
    )

    display = format_compression_thresholds_display(thresholds)

    assert "TIGHT regime" in display
    assert "20th percentile" in display
    assert "80th percentile" in display
    assert "25.0%" in display
    assert "🔒" in display  # Tight regime emoji


def test_format_compression_thresholds_display_moderate(mock_conn):
    """
    Test display formatting for moderate regime.

    Requirement 13.5: Display current thresholds in VIX analysis tab.
    """
    thresholds = CompressionThresholds(
        lower_percentile=25.0,
        upper_percentile=75.0,
        regime="moderate",
        vix_90d_percentile=50.0,
    )

    display = format_compression_thresholds_display(thresholds)

    assert "MODERATE regime" in display
    assert "25th percentile" in display
    assert "75th percentile" in display
    assert "50.0%" in display
    assert "⚖️" in display  # Moderate regime emoji


def test_format_compression_thresholds_display_wide(mock_conn):
    """
    Test display formatting for wide regime.

    Requirement 13.5: Display current thresholds in VIX analysis tab.
    """
    thresholds = CompressionThresholds(
        lower_percentile=30.0,
        upper_percentile=70.0,
        regime="wide",
        vix_90d_percentile=85.0,
    )

    display = format_compression_thresholds_display(thresholds)

    assert "WIDE regime" in display
    assert "30th percentile" in display
    assert "70th percentile" in display
    assert "85.0%" in display
    assert "📏" in display  # Wide regime emoji


def test_compression_thresholds_to_dict(mock_conn):
    """
    Test conversion of CompressionThresholds to dictionary.

    Requirement 13.5: Ensure thresholds can be serialized for API responses.
    """
    thresholds = CompressionThresholds(
        lower_percentile=25.0,
        upper_percentile=75.0,
        regime="moderate",
        vix_90d_percentile=50.5,
    )

    result = thresholds.to_dict()

    assert result["lower_percentile"] == 25.0
    assert result["upper_percentile"] == 75.0
    assert result["regime"] == "moderate"
    assert result["vix_90d_percentile"] == 50.5
    assert isinstance(result, dict)


def test_regime_classification_comprehensive(mock_conn):
    """
    Test comprehensive regime classification across all ranges.

    Requirements 13.2, 13.3, 13.4: Verify all regime boundaries.
    """
    test_cases = [
        (0.0, "tight"),
        (10.0, "tight"),
        (29.9, "tight"),
        (30.0, "moderate"),
        (50.0, "moderate"),
        (70.0, "moderate"),
        (70.1, "wide"),
        (85.0, "wide"),
        (100.0, "wide"),
    ]

    for percentile, expected_regime in test_cases:
        with patch("trader_koo.structure.vix_analysis.calculate_vix_percentile") as mock_calc:
            mock_calc.return_value = percentile

            thresholds = calculate_compression_thresholds(mock_conn)

            assert thresholds.regime == expected_regime, (
                f"Failed for percentile {percentile}: "
                f"expected {expected_regime}, got {thresholds.regime}"
            )


def test_signal_labeling_comprehensive(mock_conn):
    """
    Test comprehensive signal labeling across all regimes and states.

    Requirement 13.6: Verify all combinations of regime and compression state.
    """
    regimes = [
        ("tight", 20.0, 80.0),
        ("moderate", 25.0, 75.0),
        ("wide", 30.0, 70.0),
    ]

    for regime_name, lower, upper in regimes:
        thresholds = CompressionThresholds(
            lower_percentile=lower,
            upper_percentile=upper,
            regime=regime_name,
            vix_90d_percentile=50.0,
        )

        # Test compression
        state, labeled = detect_compression_signal(lower - 5, thresholds)
        assert state == "compression"
        assert labeled == f"compression ({regime_name})"

        # Test expansion
        state, labeled = detect_compression_signal(upper + 5, thresholds)
        assert state == "expansion"
        assert labeled == f"expansion ({regime_name})"

        # Test normal
        state, labeled = detect_compression_signal((lower + upper) / 2, thresholds)
        assert state == "normal"
        assert labeled == "normal"
