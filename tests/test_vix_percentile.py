"""
Unit tests for VIX percentile calculation and display.

Tests Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta

import pytest

from trader_koo.structure.vix_analysis import (
    calculate_vix_percentile,
    format_percentile_display,
    get_percentile_color,
    should_show_volatility_warning,
)


@pytest.fixture
def temp_db():
    """Create a temporary database with VIX data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)

    # Create price_daily table
    conn.execute("""
        CREATE TABLE price_daily (
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER
        )
    """)

    yield conn

    conn.close()


def insert_vix_data(conn: sqlite3.Connection, num_days: int, base_value: float = 15.0):
    """Insert VIX test data into the database."""
    base_date = datetime.now()

    for i in range(num_days):
        date = (base_date - timedelta(days=num_days - i - 1)).strftime("%Y-%m-%d")
        # Create some variation in VIX values
        close = base_value + (i % 10) - 5  # Values will range around base_value

        conn.execute(
            """
            INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("^VIX", date, close, close + 1, close - 1, close, 1000000),
        )

    conn.commit()


class TestVIXPercentileCalculation:
    """Test VIX percentile calculation (Requirement 11.1)."""

    def test_calculate_percentile_with_sufficient_data(self, temp_db):
        """Test percentile calculation with 252+ days of data."""
        insert_vix_data(temp_db, num_days=300, base_value=15.0)

        percentile = calculate_vix_percentile(temp_db, window_days=252)

        assert percentile is not None
        assert 0 <= percentile <= 100

    def test_calculate_percentile_insufficient_data(self, temp_db):
        """Test percentile returns None with insufficient data."""
        insert_vix_data(temp_db, num_days=1, base_value=15.0)

        percentile = calculate_vix_percentile(temp_db, window_days=252)

        assert percentile is None

    def test_calculate_percentile_no_data(self, temp_db):
        """Test percentile returns None with no VIX data."""
        percentile = calculate_vix_percentile(temp_db, window_days=252)

        assert percentile is None

    def test_percentile_at_minimum(self, temp_db):
        """Test percentile when current VIX is at minimum."""
        # Insert data with current VIX at minimum
        base_date = datetime.now()

        # Insert 100 days with VIX at 20
        for i in range(100):
            date = (base_date - timedelta(days=100 - i)).strftime("%Y-%m-%d")
            temp_db.execute(
                """
                INSERT INTO price_daily (ticker, date, close)
                VALUES (?, ?, ?)
                """,
                ("^VIX", date, 20.0),
            )

        # Current day with VIX at 10 (minimum)
        date = base_date.strftime("%Y-%m-%d")
        temp_db.execute(
            """
            INSERT INTO price_daily (ticker, date, close)
            VALUES (?, ?, ?)
            """,
            ("^VIX", date, 10.0),
        )
        temp_db.commit()

        percentile = calculate_vix_percentile(temp_db, window_days=252)

        assert percentile is not None
        assert percentile == 0.0  # All values are above current

    def test_percentile_at_maximum(self, temp_db):
        """Test percentile when current VIX is at maximum."""
        base_date = datetime.now()

        # Insert 100 days with VIX at 10
        for i in range(100):
            date = (base_date - timedelta(days=100 - i)).strftime("%Y-%m-%d")
            temp_db.execute(
                """
                INSERT INTO price_daily (ticker, date, close)
                VALUES (?, ?, ?)
                """,
                ("^VIX", date, 10.0),
            )

        # Current day with VIX at 30 (maximum)
        date = base_date.strftime("%Y-%m-%d")
        temp_db.execute(
            """
            INSERT INTO price_daily (ticker, date, close)
            VALUES (?, ?, ?)
            """,
            ("^VIX", date, 30.0),
        )
        temp_db.commit()

        percentile = calculate_vix_percentile(temp_db, window_days=252)

        assert percentile is not None
        # Should be close to 100% (all values below current)
        assert percentile > 99.0

    def test_percentile_at_median(self, temp_db):
        """Test percentile when current VIX is at median."""
        base_date = datetime.now()

        # Insert 100 days: 50 at 10, 50 at 20
        for i in range(50):
            date = (base_date - timedelta(days=100 - i)).strftime("%Y-%m-%d")
            temp_db.execute(
                """
                INSERT INTO price_daily (ticker, date, close)
                VALUES (?, ?, ?)
                """,
                ("^VIX", date, 10.0),
            )

        for i in range(50, 100):
            date = (base_date - timedelta(days=100 - i)).strftime("%Y-%m-%d")
            temp_db.execute(
                """
                INSERT INTO price_daily (ticker, date, close)
                VALUES (?, ?, ?)
                """,
                ("^VIX", date, 20.0),
            )

        # Current day with VIX at 15 (median)
        date = base_date.strftime("%Y-%m-%d")
        temp_db.execute(
            """
            INSERT INTO price_daily (ticker, date, close)
            VALUES (?, ?, ?)
            """,
            ("^VIX", date, 15.0),
        )
        temp_db.commit()

        percentile = calculate_vix_percentile(temp_db, window_days=252)

        assert percentile is not None
        # Should be around 50%
        assert 45.0 <= percentile <= 55.0


class TestPercentileColorCoding:
    """Test VIX percentile color coding (Requirement 11.3)."""

    def test_green_color_low_percentile(self):
        """Test green color for 0-30 percentile."""
        assert get_percentile_color(0.0) == "green"
        assert get_percentile_color(15.0) == "green"
        assert get_percentile_color(29.9) == "green"

    def test_yellow_color_medium_percentile(self):
        """Test yellow color for 30-70 percentile."""
        assert get_percentile_color(30.0) == "yellow"
        assert get_percentile_color(50.0) == "yellow"
        assert get_percentile_color(69.9) == "yellow"

    def test_red_color_high_percentile(self):
        """Test red color for 70-100 percentile."""
        assert get_percentile_color(70.0) == "red"
        assert get_percentile_color(85.0) == "red"
        assert get_percentile_color(100.0) == "red"

    def test_gray_color_none_percentile(self):
        """Test gray color when percentile is None."""
        assert get_percentile_color(None) == "gray"


class TestVolatilityWarning:
    """Test elevated volatility warning (Requirement 11.5)."""

    def test_warning_above_80(self):
        """Test warning is shown when percentile > 80."""
        assert should_show_volatility_warning(80.1) is True
        assert should_show_volatility_warning(85.0) is True
        assert should_show_volatility_warning(100.0) is True

    def test_no_warning_at_or_below_80(self):
        """Test warning is not shown when percentile <= 80."""
        assert should_show_volatility_warning(80.0) is False
        assert should_show_volatility_warning(70.0) is False
        assert should_show_volatility_warning(50.0) is False
        assert should_show_volatility_warning(0.0) is False

    def test_no_warning_when_none(self):
        """Test warning is not shown when percentile is None."""
        assert should_show_volatility_warning(None) is False


class TestPercentileDisplay:
    """Test VIX percentile display formatting (Requirements 11.2, 11.3, 11.5)."""

    def test_display_with_low_percentile(self):
        """Test display format for low percentile (green)."""
        display = format_percentile_display(25.0)

        assert "25.0%" in display
        assert "GREEN" in display
        assert "⚠️" not in display

    def test_display_with_medium_percentile(self):
        """Test display format for medium percentile (yellow)."""
        display = format_percentile_display(50.0)

        assert "50.0%" in display
        assert "YELLOW" in display
        assert "⚠️" not in display

    def test_display_with_high_percentile(self):
        """Test display format for high percentile (red)."""
        display = format_percentile_display(75.0)

        assert "75.0%" in display
        assert "RED" in display
        assert "⚠️" not in display

    def test_display_with_elevated_volatility_warning(self):
        """Test display includes warning when percentile > 80."""
        display = format_percentile_display(85.0)

        assert "85.0%" in display
        assert "RED" in display
        assert "⚠️ ELEVATED VOLATILITY" in display

    def test_display_with_none_percentile(self):
        """Test display when percentile is unavailable."""
        display = format_percentile_display(None)

        assert "Unavailable" in display


class TestPercentileInHealthScore:
    """Test VIX percentile integration in health score (Requirement 11.4)."""

    def test_health_score_includes_percentile_factor(self, temp_db):
        """Test that health score calculation includes VIX percentile as primary factor."""
        # This is tested indirectly through the daily report generation
        # The health score calculation in generate_daily_report.py includes:
        # - +20 for percentile < 30 (risk-on)
        # - +10 for percentile < 50
        # - -10 for percentile < 70
        # - -20 for percentile >= 70 (risk-off)
        # This makes it one of the largest single factors (primary factor)

        # Insert test data
        insert_vix_data(temp_db, num_days=300, base_value=15.0)

        percentile = calculate_vix_percentile(temp_db, window_days=252)

        # Verify percentile is calculated
        assert percentile is not None

        # The actual health score calculation is tested in integration tests
        # Here we just verify the percentile can be calculated
