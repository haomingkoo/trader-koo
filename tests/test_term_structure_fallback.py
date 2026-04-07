"""
Unit tests for VIX term structure fallback logic.

Tests Requirements 9.1-9.6:
- VIX3M primary source
- VIX6M fallback
- Synthetic calculation from VXX/UVXY
- Source labeling
- Logging
- Complete unavailability handling
- Timestamp inclusion
"""

import sqlite3
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from trader_koo.structure.vix_analysis import (
    TermStructure,
    calculate_term_structure,
    format_term_structure_display,
)


@pytest.fixture
def mock_conn():
    """Create a mock database connection."""
    conn = MagicMock(spec=sqlite3.Connection)
    return conn


def test_vix3m_success(mock_conn):
    """Test VIX3M as primary source (Requirement 9.1)."""
    # Mock VIX spot and VIX3M data
    mock_conn.execute.return_value.fetchone.side_effect = [
        (15.5,),  # VIX spot
        (16.8,),  # VIX3M
    ]

    result = calculate_term_structure(mock_conn)

    assert result.vix_spot == 15.5
    assert result.vix_3m == 16.8
    assert result.vix_6m is None
    assert result.source == "VIX3M"
    assert result.contango is True  # (16.8 - 15.5) / 15.5 > 0.03
    assert result.slope is not None
    assert isinstance(result.timestamp, datetime)


def test_vix6m_fallback(mock_conn):
    """Test VIX6M fallback when VIX3M unavailable (Requirement 9.1)."""
    # Mock VIX spot, no VIX3M, but VIX6M available
    mock_conn.execute.return_value.fetchone.side_effect = [
        (14.2,),  # VIX spot
        None,     # ^VIX3M unavailable
        None,     # VIX3M unavailable
        (15.9,),  # ^VIX6M
    ]

    result = calculate_term_structure(mock_conn)

    assert result.vix_spot == 14.2
    assert result.vix_3m is None
    assert result.vix_6m == 15.9
    assert result.source == "VIX6M"
    assert result.contango is True
    assert result.slope is not None


def test_synthetic_calculation(mock_conn):
    """Test synthetic calculation from VXX/UVXY (Requirement 9.2)."""
    # Mock VIX spot, no VIX3M, no VIX6M, but VXX/UVXY available
    mock_conn.execute.return_value.fetchone.side_effect = [
        (13.5,),  # VIX spot
        None,     # ^VIX3M unavailable
        None,     # VIX3M unavailable
        None,     # ^VIX6M unavailable
        None,     # VIX6M unavailable
        (22.5,),  # VXX
        (18.3,),  # UVXY
    ]

    result = calculate_term_structure(mock_conn)

    assert result.vix_spot == 13.5
    assert result.vix_3m is not None  # Synthetic value
    assert result.vix_6m is None
    assert result.source == "synthetic"
    assert result.slope is not None
    assert isinstance(result.timestamp, datetime)


def test_complete_unavailability(mock_conn):
    """Test complete unavailability handling (Requirement 9.5)."""
    # Mock VIX spot, but all term structure sources unavailable
    mock_conn.execute.return_value.fetchone.side_effect = [
        (14.8,),  # VIX spot
        None,     # ^VIX3M unavailable
        None,     # VIX3M unavailable
        None,     # ^VIX6M unavailable
        None,     # VIX6M unavailable
        None,     # VXX unavailable
        None,     # UVXY unavailable
    ]

    result = calculate_term_structure(mock_conn)

    assert result.vix_spot == 14.8
    assert result.vix_3m is None
    assert result.vix_6m is None
    assert result.source == "unavailable"
    assert result.contango is False
    assert result.slope is None


def test_vix_spot_unavailable(mock_conn):
    """Test when VIX spot itself is unavailable."""
    mock_conn.execute.return_value.fetchone.return_value = None

    result = calculate_term_structure(mock_conn)

    assert result.vix_spot == 0.0
    assert result.source == "unavailable"


def test_source_labeling(mock_conn):
    """Test source labeling in all displays (Requirement 9.3)."""
    # Test VIX3M source
    mock_conn.execute.return_value.fetchone.side_effect = [
        (15.0,),  # VIX spot
        (16.5,),  # VIX3M
    ]

    result = calculate_term_structure(mock_conn)
    result_dict = result.to_dict()

    assert result_dict["source"] == "VIX3M"
    assert "source" in result_dict
    assert result_dict["vix_spot"] == 15.0
    assert result_dict["vix_3m"] == 16.5


def test_timestamp_inclusion(mock_conn):
    """Test timestamp inclusion in displays (Requirement 9.6)."""
    mock_conn.execute.return_value.fetchone.side_effect = [
        (15.0,),  # VIX spot
        (16.5,),  # VIX3M
    ]

    result = calculate_term_structure(mock_conn)
    result_dict = result.to_dict()

    assert "timestamp" in result_dict
    assert isinstance(result.timestamp, datetime)
    # Verify ISO format
    assert "T" in result_dict["timestamp"]


def test_format_display_with_vix3m(mock_conn):
    """Test display formatting with VIX3M source."""
    mock_conn.execute.return_value.fetchone.side_effect = [
        (15.0,),  # VIX spot
        (16.5,),  # VIX3M
    ]

    result = calculate_term_structure(mock_conn)
    display = format_term_structure_display(result)

    assert "VIX Spot: 15.00" in display
    assert "VIX 3M: 16.50" in display
    assert "Source: VIX3M" in display
    assert "Timestamp:" in display
    assert "Contango" in display


def test_format_display_unavailable(mock_conn):
    """Test display formatting when unavailable (Requirement 9.5)."""
    mock_conn.execute.return_value.fetchone.side_effect = [
        (14.8,),  # VIX spot
        None, None, None, None, None, None,  # All sources unavailable
    ]

    result = calculate_term_structure(mock_conn)
    display = format_term_structure_display(result)

    assert display == "Term structure unavailable"


def test_contango_detection():
    """Test contango vs backwardation detection."""
    # Contango case
    ts_contango = TermStructure(
        vix_spot=15.0,
        vix_3m=16.5,
        vix_6m=None,
        source="VIX3M",
        contango=True,
        slope=0.10,
        timestamp=datetime.utcnow(),
    )
    assert ts_contango.contango is True

    # Backwardation case
    ts_backwardation = TermStructure(
        vix_spot=18.0,
        vix_3m=16.5,
        vix_6m=None,
        source="VIX3M",
        contango=False,
        slope=-0.083,
        timestamp=datetime.utcnow(),
    )
    assert ts_backwardation.contango is False


@patch("trader_koo.structure.vix_analysis.logger")
def test_logging_vix3m_success(mock_logger, mock_conn):
    """Test logging when VIX3M succeeds (Requirement 9.4)."""
    mock_conn.execute.return_value.fetchone.side_effect = [
        (15.0,),  # VIX spot
        (16.5,),  # VIX3M
    ]

    calculate_term_structure(mock_conn)

    # Verify info log was called with VIX3M success
    mock_logger.info.assert_called()
    call_args = str(mock_logger.info.call_args)
    assert "VIX3M" in call_args or "Term structure from VIX3M" in call_args


@patch("trader_koo.structure.vix_analysis.logger")
def test_logging_vix6m_fallback(mock_logger, mock_conn):
    """Test logging when falling back to VIX6M (Requirement 9.4)."""
    mock_conn.execute.return_value.fetchone.side_effect = [
        (15.0,),  # VIX spot
        None,     # ^VIX3M unavailable
        None,     # VIX3M unavailable
        (16.8,),  # ^VIX6M
    ]

    calculate_term_structure(mock_conn)

    # Verify warning log for VIX3M unavailable
    mock_logger.warning.assert_called()
    warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
    assert any("VIX3M unavailable" in call for call in warning_calls)


@patch("trader_koo.structure.vix_analysis.logger")
def test_logging_synthetic_calculation(mock_logger, mock_conn):
    """Test logging when using synthetic calculation (Requirement 9.4)."""
    mock_conn.execute.return_value.fetchone.side_effect = [
        (15.0,),  # VIX spot
        None, None, None, None,  # VIX3M and VIX6M unavailable
        (22.5,),  # VXX
        (18.3,),  # UVXY
    ]

    calculate_term_structure(mock_conn)

    # Verify warning logs for fallback chain
    warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
    assert any("VIX6M unavailable" in call for call in warning_calls)


@patch("trader_koo.structure.vix_analysis.logger")
def test_logging_complete_unavailability(mock_logger, mock_conn):
    """Test logging when all sources unavailable (Requirement 9.4)."""
    mock_conn.execute.return_value.fetchone.side_effect = [
        (15.0,),  # VIX spot
        None, None, None, None, None, None,  # All sources unavailable
    ]

    calculate_term_structure(mock_conn)

    # Verify error log for complete unavailability
    mock_logger.error.assert_called()
    error_calls = [str(call) for call in mock_logger.error.call_args_list]
    assert any("All term structure sources unavailable" in call for call in error_calls)


def test_vix3m_with_caret_prefix(mock_conn):
    """Test that both ^VIX3M and VIX3M ticker formats work."""
    # Mock trying ^VIX3M first (returns data), VIX3M not tried
    mock_conn.execute.return_value.fetchone.side_effect = [
        (15.0,),  # VIX spot
        (16.5,),  # ^VIX3M
    ]

    result = calculate_term_structure(mock_conn)

    assert result.source == "VIX3M"
    assert result.vix_3m == 16.5


def test_vix3m_fallback_to_no_caret(mock_conn):
    """Test fallback from ^VIX3M to VIX3M ticker format."""
    # Mock ^VIX3M unavailable, but VIX3M available
    mock_conn.execute.return_value.fetchone.side_effect = [
        (15.0,),  # VIX spot
        None,     # ^VIX3M unavailable
        (16.5,),  # VIX3M (no caret)
    ]

    result = calculate_term_structure(mock_conn)

    assert result.source == "VIX3M"
    assert result.vix_3m == 16.5
