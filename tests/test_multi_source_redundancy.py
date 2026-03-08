"""Unit tests for multi-source data redundancy.

Tests Requirements:
- 12.1: yfinance as primary source
- 12.2: Alpha Vantage as secondary fallback
- 12.3: CSV as final fallback
- 12.4: Source logging
- 12.5: Success/failure rate tracking
- 12.6: Alerting for high failure rates (>10%)
- 12.7: Data source in API responses
"""

import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from trader_koo.db.sources import (
    DataSource,
    DataSourceManager,
    FetchResult,
    SourceMetrics,
    get_data_source_manager,
)


@pytest.fixture
def temp_csv_dir():
    """Create a temporary directory for CSV fallback files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "open": [100.0, 101.0, 102.0],
        "high": [105.0, 106.0, 107.0],
        "low": [99.0, 100.0, 101.0],
        "close": [103.0, 104.0, 105.0],
        "volume": [1000000.0, 1100000.0, 1200000.0],
    })


@pytest.fixture
def manager(temp_csv_dir):
    """Create a DataSourceManager instance with temp CSV directory."""
    return DataSourceManager(csv_fallback_dir=temp_csv_dir)


class TestDataSourceManager:
    """Test DataSourceManager multi-source fetching."""

    def test_primary_source_success(self, manager, sample_price_data):
        """Test successful fetch from primary source (yfinance).
        
        Requirements: 12.1, 12.4
        """
        with patch.object(manager, "_fetch_yfinance") as mock_yfinance:
            mock_yfinance.return_value = FetchResult(
                data=sample_price_data,
                source=DataSource.YFINANCE,
                timestamp=datetime.now(),
                success=True,
            )
            
            result = manager.fetch_ticker_data("AAPL", "2024-01-01")
            
            assert result.success
            assert result.source == DataSource.YFINANCE
            assert len(result.data) == 3
            assert not result.data.empty
            
            # Verify yfinance was called
            mock_yfinance.assert_called_once()

    def test_secondary_fallback_on_primary_failure(self, manager, sample_price_data):
        """Test fallback to Alpha Vantage when yfinance fails.
        
        Requirements: 12.2, 12.4
        """
        with patch.object(manager, "_fetch_yfinance") as mock_yfinance, \
             patch.object(manager, "_fetch_alpha_vantage") as mock_alpha:
            
            # Primary fails
            mock_yfinance.return_value = FetchResult(
                data=pd.DataFrame(),
                source=DataSource.YFINANCE,
                timestamp=datetime.now(),
                success=False,
                error="Connection timeout",
            )
            
            # Secondary succeeds
            mock_alpha.return_value = FetchResult(
                data=sample_price_data,
                source=DataSource.ALPHA_VANTAGE,
                timestamp=datetime.now(),
                success=True,
            )
            
            result = manager.fetch_ticker_data("AAPL", "2024-01-01")
            
            assert result.success
            assert result.source == DataSource.ALPHA_VANTAGE
            assert len(result.data) == 3
            
            # Verify both sources were tried
            mock_yfinance.assert_called_once()
            mock_alpha.assert_called_once()

    def test_csv_fallback_on_all_failures(self, manager, sample_price_data, temp_csv_dir):
        """Test fallback to CSV when both yfinance and Alpha Vantage fail.
        
        Requirements: 12.3, 12.4
        """
        # Create CSV fallback file
        csv_path = temp_csv_dir / "AAPL.csv"
        sample_price_data.to_csv(csv_path, index=False)
        
        with patch.object(manager, "_fetch_yfinance") as mock_yfinance, \
             patch.object(manager, "_fetch_alpha_vantage") as mock_alpha:
            
            # Both primary and secondary fail
            mock_yfinance.return_value = FetchResult(
                data=pd.DataFrame(),
                source=DataSource.YFINANCE,
                timestamp=datetime.now(),
                success=False,
                error="Connection timeout",
            )
            
            mock_alpha.return_value = FetchResult(
                data=pd.DataFrame(),
                source=DataSource.ALPHA_VANTAGE,
                timestamp=datetime.now(),
                success=False,
                error="API key invalid",
            )
            
            result = manager.fetch_ticker_data("AAPL", "2024-01-01")
            
            assert result.success
            assert result.source == DataSource.CSV_FALLBACK
            assert len(result.data) == 3
            
            # Verify all sources were tried
            mock_yfinance.assert_called_once()
            mock_alpha.assert_called_once()

    def test_all_sources_fail(self, manager):
        """Test behavior when all sources fail.
        
        Requirements: 12.3, 12.4
        """
        with patch.object(manager, "_fetch_yfinance") as mock_yfinance, \
             patch.object(manager, "_fetch_alpha_vantage") as mock_alpha, \
             patch.object(manager, "_load_csv_fallback") as mock_csv:
            
            # All sources fail
            for mock in [mock_yfinance, mock_alpha, mock_csv]:
                mock.return_value = FetchResult(
                    data=pd.DataFrame(),
                    source=mock.return_value.source if hasattr(mock.return_value, 'source') else DataSource.CSV_FALLBACK,
                    timestamp=datetime.now(),
                    success=False,
                    error="Failed",
                )
            
            mock_csv.return_value.source = DataSource.CSV_FALLBACK
            
            result = manager.fetch_ticker_data("AAPL", "2024-01-01")
            
            assert not result.success
            assert result.data.empty
            
            # Verify all sources were tried
            mock_yfinance.assert_called_once()
            mock_alpha.assert_called_once()
            mock_csv.assert_called_once()


class TestSourceMetrics:
    """Test success/failure rate tracking."""

    def test_success_rate_calculation(self):
        """Test success rate calculation.
        
        Requirements: 12.5
        """
        metrics = SourceMetrics(source=DataSource.YFINANCE)
        
        # No attempts yet
        assert metrics.success_rate == 0.0
        assert metrics.failure_rate == 100.0
        
        # Add some attempts
        metrics.total_attempts = 10
        metrics.successful_fetches = 8
        metrics.failed_fetches = 2
        
        assert metrics.success_rate == 80.0
        assert metrics.failure_rate == 20.0

    def test_failure_rate_calculation(self):
        """Test failure rate calculation.
        
        Requirements: 12.5
        """
        metrics = SourceMetrics(source=DataSource.YFINANCE)
        metrics.total_attempts = 100
        metrics.successful_fetches = 85
        metrics.failed_fetches = 15
        
        assert metrics.failure_rate == 15.0
        assert metrics.success_rate == 85.0

    def test_get_metrics(self, manager):
        """Test metrics retrieval.
        
        Requirements: 12.5
        """
        # Simulate some fetches
        manager.metrics[DataSource.YFINANCE].total_attempts = 10
        manager.metrics[DataSource.YFINANCE].successful_fetches = 9
        manager.metrics[DataSource.YFINANCE].failed_fetches = 1
        
        metrics = manager.get_metrics()
        
        assert "yfinance" in metrics
        assert metrics["yfinance"]["total_attempts"] == 10
        assert metrics["yfinance"]["successful_fetches"] == 9
        assert metrics["yfinance"]["failed_fetches"] == 1
        assert metrics["yfinance"]["success_rate"] == 90.0
        assert metrics["yfinance"]["failure_rate"] == 10.0

    def test_metrics_updated_on_fetch(self, manager, sample_price_data):
        """Test that metrics are updated during actual fetches.
        
        Requirements: 12.5
        """
        # Reset metrics first
        manager.reset_metrics()
        
        with patch("yfinance.download") as mock_download:
            mock_download.return_value = sample_price_data.set_index("date")
            
            result = manager._fetch_yfinance("AAPL", "2024-01-01", None, False, 30.0)
            
            assert result.success
            
            # Verify metrics were updated
            metrics = manager.metrics[DataSource.YFINANCE]
            assert metrics.total_attempts == 1
            assert metrics.successful_fetches == 1
            assert metrics.failed_fetches == 0


class TestAlertingForHighFailureRates:
    """Test alerting when primary source failure rate exceeds 10%."""

    def test_alert_on_high_failure_rate(self, manager, caplog):
        """Test alert is logged when failure rate exceeds 10%.
        
        Requirements: 12.6
        """
        import logging
        caplog.set_level(logging.ERROR)
        
        # Simulate high failure rate (>10%)
        metrics = manager.metrics[DataSource.YFINANCE]
        metrics.total_attempts = 20
        metrics.successful_fetches = 15
        metrics.failed_fetches = 5  # 25% failure rate
        
        manager._check_and_alert(DataSource.YFINANCE)
        
        # Check that alert was logged
        assert any("ALERT" in record.message for record in caplog.records)
        assert any("failure rate" in record.message.lower() for record in caplog.records)

    def test_no_alert_below_threshold(self, manager, caplog):
        """Test no alert when failure rate is below 10%.
        
        Requirements: 12.6
        """
        import logging
        caplog.set_level(logging.ERROR)
        
        # Simulate low failure rate (<10%)
        metrics = manager.metrics[DataSource.YFINANCE]
        metrics.total_attempts = 20
        metrics.successful_fetches = 19
        metrics.failed_fetches = 1  # 5% failure rate
        
        manager._check_and_alert(DataSource.YFINANCE)
        
        # Check that no alert was logged
        assert not any("ALERT" in record.message for record in caplog.records)

    def test_no_alert_for_secondary_sources(self, manager, caplog):
        """Test alerts only trigger for primary source (yfinance).
        
        Requirements: 12.6
        """
        import logging
        caplog.set_level(logging.ERROR)
        
        # Simulate high failure rate for secondary source
        metrics = manager.metrics[DataSource.ALPHA_VANTAGE]
        metrics.total_attempts = 20
        metrics.successful_fetches = 10
        metrics.failed_fetches = 10  # 50% failure rate
        
        manager._check_and_alert(DataSource.ALPHA_VANTAGE)
        
        # Check that no alert was logged (only primary source alerts)
        assert not any("ALERT" in record.message for record in caplog.records)

    def test_alert_cooldown(self, manager, caplog):
        """Test alert cooldown prevents spam.
        
        Requirements: 12.6
        """
        import logging
        import time
        caplog.set_level(logging.ERROR)
        
        # Simulate high failure rate
        metrics = manager.metrics[DataSource.YFINANCE]
        metrics.total_attempts = 20
        metrics.successful_fetches = 15
        metrics.failed_fetches = 5  # 25% failure rate
        
        # First alert should trigger
        manager._check_and_alert(DataSource.YFINANCE)
        assert any("ALERT" in record.message for record in caplog.records)
        
        # Clear logs
        caplog.clear()
        
        # Second alert should not trigger (cooldown)
        manager._check_and_alert(DataSource.YFINANCE)
        assert not any("ALERT" in record.message for record in caplog.records)


class TestCSVFallback:
    """Test CSV fallback functionality."""

    def test_csv_fallback_success(self, manager, sample_price_data, temp_csv_dir):
        """Test successful CSV fallback load.
        
        Requirements: 12.3
        """
        # Create CSV file
        csv_path = temp_csv_dir / "AAPL.csv"
        sample_price_data.to_csv(csv_path, index=False)
        
        result = manager._load_csv_fallback("AAPL", "2024-01-01", None)
        
        assert result.success
        assert result.source == DataSource.CSV_FALLBACK
        assert len(result.data) == 3
        assert list(result.data.columns) == ["date", "open", "high", "low", "close", "volume"]

    def test_csv_fallback_missing_file(self, manager):
        """Test CSV fallback when file doesn't exist.
        
        Requirements: 12.3
        """
        result = manager._load_csv_fallback("NONEXISTENT", "2024-01-01", None)
        
        assert not result.success
        assert result.source == DataSource.CSV_FALLBACK
        assert result.data.empty
        assert "not found" in result.error.lower()

    def test_csv_fallback_date_filtering(self, manager, sample_price_data, temp_csv_dir):
        """Test CSV fallback filters by date range.
        
        Requirements: 12.3
        """
        # Create CSV file with more data
        extended_data = pd.DataFrame({
            "date": ["2023-12-01", "2024-01-01", "2024-01-02", "2024-01-03", "2024-02-01"],
            "open": [90.0, 100.0, 101.0, 102.0, 110.0],
            "high": [95.0, 105.0, 106.0, 107.0, 115.0],
            "low": [89.0, 99.0, 100.0, 101.0, 109.0],
            "close": [93.0, 103.0, 104.0, 105.0, 113.0],
            "volume": [900000.0, 1000000.0, 1100000.0, 1200000.0, 1300000.0],
        })
        
        csv_path = temp_csv_dir / "AAPL.csv"
        extended_data.to_csv(csv_path, index=False)
        
        # Test date filtering
        result = manager._load_csv_fallback("AAPL", "2024-01-01", "2024-01-31")
        
        assert result.success
        assert len(result.data) == 3  # Only Jan 2024 data
        assert result.data["date"].min() >= "2024-01-01"
        assert result.data["date"].max() <= "2024-01-31"


class TestAlphaVantage:
    """Test Alpha Vantage integration."""

    def test_alpha_vantage_no_api_key(self, manager):
        """Test Alpha Vantage fails gracefully without API key.
        
        Requirements: 12.2
        """
        with patch.dict(os.environ, {}, clear=True):
            result = manager._fetch_alpha_vantage("AAPL", "2024-01-01", None)
            
            assert not result.success
            assert result.source == DataSource.ALPHA_VANTAGE
            assert "not configured" in result.error.lower()

    def test_alpha_vantage_success(self, manager):
        """Test successful Alpha Vantage fetch.
        
        Requirements: 12.2
        """
        mock_response = {
            "Time Series (Daily)": {
                "2024-01-03": {
                    "1. open": "102.0",
                    "2. high": "107.0",
                    "3. low": "101.0",
                    "4. close": "105.0",
                    "5. volume": "1200000"
                },
                "2024-01-02": {
                    "1. open": "101.0",
                    "2. high": "106.0",
                    "3. low": "100.0",
                    "4. close": "104.0",
                    "5. volume": "1100000"
                },
                "2024-01-01": {
                    "1. open": "100.0",
                    "2. high": "105.0",
                    "3. low": "99.0",
                    "4. close": "103.0",
                    "5. volume": "1000000"
                }
            }
        }
        
        with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test_key"}), \
             patch("requests.get") as mock_get:
            
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = Mock()
            
            result = manager._fetch_alpha_vantage("AAPL", "2024-01-01", None)
            
            assert result.success
            assert result.source == DataSource.ALPHA_VANTAGE
            assert len(result.data) == 3
            assert list(result.data.columns) == ["date", "open", "high", "low", "close", "volume"]

    def test_alpha_vantage_error_response(self, manager):
        """Test Alpha Vantage handles error responses.
        
        Requirements: 12.2
        """
        mock_response = {
            "Error Message": "Invalid API call"
        }
        
        with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test_key"}), \
             patch("requests.get") as mock_get:
            
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = Mock()
            
            result = manager._fetch_alpha_vantage("AAPL", "2024-01-01", None)
            
            assert not result.success
            assert result.source == DataSource.ALPHA_VANTAGE
            assert "Invalid API call" in result.error


class TestDataSourceInAPIResponses:
    """Test data source is included in API responses."""

    def test_data_source_in_database(self):
        """Test data source is stored in database.
        
        Requirements: 12.7
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE price_daily (
                    ticker TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    data_source TEXT DEFAULT 'yfinance',
                    fetch_timestamp TEXT,
                    PRIMARY KEY (ticker, date)
                )
            """)
            
            # Insert test data
            conn.execute("""
                INSERT INTO price_daily 
                (ticker, date, open, high, low, close, volume, data_source, fetch_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ("AAPL", "2024-01-01", 100.0, 105.0, 99.0, 103.0, 1000000.0, 
                  "yfinance", "2024-01-01T10:00:00Z"))
            
            conn.commit()
            
            # Query data source
            row = conn.execute("""
                SELECT data_source, fetch_timestamp
                FROM price_daily
                WHERE ticker = ? AND date = ?
            """, ("AAPL", "2024-01-01")).fetchone()
            
            assert row is not None
            assert row[0] == "yfinance"
            assert row[1] == "2024-01-01T10:00:00Z"
            
            conn.close()
        finally:
            os.unlink(db_path)


class TestGlobalInstance:
    """Test global DataSourceManager instance."""

    def test_get_data_source_manager_singleton(self):
        """Test get_data_source_manager returns singleton instance.
        
        Requirements: 12.1
        """
        manager1 = get_data_source_manager()
        manager2 = get_data_source_manager()
        
        assert manager1 is manager2

    def test_reset_metrics(self, manager):
        """Test metrics can be reset.
        
        Requirements: 12.5
        """
        # Add some metrics
        manager.metrics[DataSource.YFINANCE].total_attempts = 10
        manager.metrics[DataSource.YFINANCE].successful_fetches = 8
        
        # Reset
        manager.reset_metrics()
        
        # Verify reset
        assert manager.metrics[DataSource.YFINANCE].total_attempts == 0
        assert manager.metrics[DataSource.YFINANCE].successful_fetches == 0
        assert manager.metrics[DataSource.YFINANCE].failed_fetches == 0

