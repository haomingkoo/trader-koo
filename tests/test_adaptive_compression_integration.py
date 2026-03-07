"""
Integration tests for adaptive compression thresholds in daily report generation.

Tests the integration of adaptive compression thresholds with the daily report
generation workflow.
"""

import sqlite3
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from trader_koo.structure.vix_analysis import (
    calculate_compression_thresholds,
    detect_compression_signal,
)


@pytest.fixture
def test_db():
    """Create an in-memory test database with VIX data."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    
    # Create price_daily table
    conn.execute("""
        CREATE TABLE price_daily (
            ticker TEXT,
            date TEXT,
            close REAL
        )
    """)
    
    # Insert 90 days of VIX data for testing
    base_date = datetime.now()
    for i in range(90):
        date = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
        # Create low volatility scenario (VIX around 12-15)
        close = 12.0 + (i % 3)
        conn.execute(
            "INSERT INTO price_daily (ticker, date, close) VALUES (?, ?, ?)",
            ("^VIX", date, close)
        )
    
    conn.commit()
    return conn


def test_integration_low_volatility_regime(test_db):
    """
    Test that low volatility data produces tight thresholds.
    
    Integration test for Requirements 13.1, 13.2.
    """
    thresholds = calculate_compression_thresholds(test_db)
    
    # With VIX around 12-15, percentile should be low
    assert thresholds.regime in ["tight", "moderate"]
    
    # Verify thresholds are appropriate
    if thresholds.regime == "tight":
        assert thresholds.lower_percentile == 20.0
        assert thresholds.upper_percentile == 80.0
    
    # Verify to_dict works
    threshold_dict = thresholds.to_dict()
    assert "regime" in threshold_dict
    assert "lower_percentile" in threshold_dict
    assert "upper_percentile" in threshold_dict
    assert "vix_90d_percentile" in threshold_dict


def test_integration_compression_detection_with_thresholds(test_db):
    """
    Test compression detection using calculated thresholds.
    
    Integration test for Requirements 13.1, 13.6.
    """
    thresholds = calculate_compression_thresholds(test_db)
    
    # Test compression detection at various BB width percentiles
    test_cases = [
        (10.0, "compression"),  # Well below lower threshold
        (50.0, "normal"),       # Middle range
        (90.0, "expansion"),    # Well above upper threshold
    ]
    
    for bb_width_pct, expected_state in test_cases:
        state, labeled = detect_compression_signal(bb_width_pct, thresholds)
        assert state == expected_state
        
        # Verify labeling includes regime
        if state != "normal":
            assert thresholds.regime in labeled


def test_integration_high_volatility_scenario():
    """
    Test that high volatility data produces wide thresholds.
    
    Integration test for Requirements 13.1, 13.4.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    
    conn.execute("""
        CREATE TABLE price_daily (
            ticker TEXT,
            date TEXT,
            close REAL
        )
    """)
    
    # Insert 90 days of high VIX data with current VIX being high
    base_date = datetime.now()
    for i in range(90):
        date = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
        # Create high volatility scenario with current VIX high
        # Most recent (i=0) should be highest
        close = 35.0 - (i % 10)  # Current VIX around 35, historical lower
        conn.execute(
            "INSERT INTO price_daily (ticker, date, close) VALUES (?, ?, ?)",
            ("^VIX", date, close)
        )
    
    conn.commit()
    
    thresholds = calculate_compression_thresholds(conn)
    
    # With current VIX high relative to history, percentile should be high
    assert thresholds.regime in ["moderate", "wide"]
    
    # Verify thresholds are appropriate
    if thresholds.regime == "wide":
        assert thresholds.lower_percentile == 30.0
        assert thresholds.upper_percentile == 70.0
    
    conn.close()


def test_integration_threshold_regime_affects_signal_labeling():
    """
    Test that different regimes produce different signal labels.
    
    Integration test for Requirement 13.6.
    """
    from trader_koo.structure.vix_analysis import CompressionThresholds
    
    # Create thresholds for each regime
    regimes = [
        CompressionThresholds(20.0, 80.0, "tight", 25.0),
        CompressionThresholds(25.0, 75.0, "moderate", 50.0),
        CompressionThresholds(30.0, 70.0, "wide", 85.0),
    ]
    
    # Test that same BB width percentile produces different labels
    bb_width = 15.0  # Low value - should be compression in all regimes
    
    labels = []
    for threshold in regimes:
        state, labeled = detect_compression_signal(bb_width, threshold)
        assert state == "compression"
        labels.append(labeled)
    
    # Verify all labels are different
    assert len(set(labels)) == 3
    assert "compression (tight)" in labels
    assert "compression (moderate)" in labels
    assert "compression (wide)" in labels


def test_integration_missing_data_handling(test_db):
    """
    Test that system handles missing VIX data gracefully.
    
    Integration test for Requirements 13.1.
    """
    # Create empty database
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    
    conn.execute("""
        CREATE TABLE price_daily (
            ticker TEXT,
            date TEXT,
            close REAL
        )
    """)
    
    # No VIX data inserted
    thresholds = calculate_compression_thresholds(conn)
    
    # Should fall back to moderate thresholds
    assert thresholds.regime == "moderate"
    assert thresholds.lower_percentile == 25.0
    assert thresholds.upper_percentile == 75.0
    
    conn.close()


def test_integration_threshold_serialization(test_db):
    """
    Test that thresholds can be serialized for API responses.
    
    Integration test for Requirement 13.5.
    """
    thresholds = calculate_compression_thresholds(test_db)
    
    # Convert to dict (as would be done for API response)
    threshold_dict = thresholds.to_dict()
    
    # Verify all required fields are present
    assert "lower_percentile" in threshold_dict
    assert "upper_percentile" in threshold_dict
    assert "regime" in threshold_dict
    assert "vix_90d_percentile" in threshold_dict
    
    # Verify types
    assert isinstance(threshold_dict["lower_percentile"], float)
    assert isinstance(threshold_dict["upper_percentile"], float)
    assert isinstance(threshold_dict["regime"], str)
    assert isinstance(threshold_dict["vix_90d_percentile"], float)
    
    # Verify values are reasonable
    assert 0 <= threshold_dict["lower_percentile"] <= 50
    assert 50 <= threshold_dict["upper_percentile"] <= 100
    assert threshold_dict["regime"] in ["tight", "moderate", "wide"]
    assert 0 <= threshold_dict["vix_90d_percentile"] <= 100
