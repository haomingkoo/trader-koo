"""Unit tests for data source fetching.

Tests:
- yfinance as the sole price source
- Explicit failure when yfinance returns no data (no hidden fallbacks)
- Success/failure rate tracking
- Alerting for high failure rates (>10%)
- Data source in API responses
"""

import os
import sqlite3
import tempfile
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from trader_koo.db.sources import (
    DataSourceManager,
    FetchResult,
    PriceFetchError,
    SourceMetrics,
    get_data_source_manager,
)


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
def manager():
    """Create a DataSourceManager instance."""
    return DataSourceManager()


class TestDataSourceManager:
    """Test DataSourceManager fetching."""

    def test_primary_source_success(self, manager, sample_price_data):
        """Test successful fetch from yfinance."""
        with patch.object(manager, "_fetch_yfinance") as mock_yfinance:
            mock_yfinance.return_value = FetchResult(
                data=sample_price_data,
                source="yfinance",
                timestamp=datetime.now(),
                success=True,
            )

            result = manager.fetch_ticker_data("AAPL", "2024-01-01")

            assert result.success
            assert result.source == "yfinance"
            assert len(result.data) == 3
            mock_yfinance.assert_called_once()

    def test_raises_on_yfinance_failure(self, manager):
        """Test that PriceFetchError is raised when yfinance fails.

        No hidden fallbacks — failure is explicit.
        """
        with patch.object(manager, "_fetch_yfinance") as mock_yfinance:
            mock_yfinance.return_value = FetchResult(
                data=pd.DataFrame(),
                source="yfinance",
                timestamp=datetime.now(),
                success=False,
                error="Connection timeout",
            )

            with pytest.raises(PriceFetchError, match="yfinance returned no data"):
                manager.fetch_ticker_data("AAPL", "2024-01-01")

    def test_raises_on_empty_data(self, manager):
        """Test that PriceFetchError is raised when yfinance returns empty data."""
        with patch.object(manager, "_fetch_yfinance") as mock_yfinance:
            mock_yfinance.return_value = FetchResult(
                data=pd.DataFrame(),
                source="yfinance",
                timestamp=datetime.now(),
                success=True,  # success=True but data is empty
            )

            with pytest.raises(PriceFetchError):
                manager.fetch_ticker_data("AAPL", "2024-01-01")


class TestMultiIndexNormalization:
    """Test yfinance >=1.0 MultiIndex column handling."""

    def test_normalize_multiindex_columns(self):
        """Test that MultiIndex columns from yfinance >=1.0 are flattened."""
        mgr = DataSourceManager()

        index = pd.DatetimeIndex(["2024-01-01", "2024-01-02"], name="Date")
        columns = pd.MultiIndex.from_tuples([
            ("Open", "AAPL"), ("High", "AAPL"), ("Low", "AAPL"),
            ("Close", "AAPL"), ("Volume", "AAPL"),
        ])
        data = [[100, 105, 99, 103, 1e6], [101, 106, 100, 104, 1.1e6]]
        df = pd.DataFrame(data, index=index, columns=columns)

        result = mgr._normalize_ohlcv(df)

        assert list(result.columns) == ["date", "open", "high", "low", "close", "volume"]
        assert len(result) == 2

    def test_normalize_flat_columns(self):
        """Test that flat columns (yfinance <1.0 style) still work."""
        mgr = DataSourceManager()

        index = pd.DatetimeIndex(["2024-01-01"], name="Date")
        df = pd.DataFrame(
            {"Open": [100], "High": [105], "Low": [99],
             "Close": [103], "Volume": [1e6]},
            index=index,
        )

        result = mgr._normalize_ohlcv(df)

        assert list(result.columns) == ["date", "open", "high", "low", "close", "volume"]
        assert len(result) == 1

    def test_normalize_applies_declared_split_to_all_prior_ohlcv(self):
        mgr = DataSourceManager()
        index = pd.DatetimeIndex(
            ["2026-08-07", "2026-08-11", "2026-08-12"],
            name="Date",
        )
        df = pd.DataFrame(
            {
                "Open": [94.0, 46.0, 45.5],
                "High": [96.0, 47.0, 46.0],
                "Low": [90.0, 45.0, 45.0],
                "Close": [92.0, 45.5, 46.0],
                "Adj Close": [46.0, 45.5, 46.0],
                "Volume": [5_000_000, 9_000_000, 8_000_000],
                "Stock Splits": [0.0, 2.0, 0.0],
                "Repaired?": [False, False, False],
            },
            index=index,
        )

        result = mgr._normalize_ohlcv(df)

        assert result.loc[0, ["open", "high", "low", "close"]].tolist() == [
            47.0,
            48.0,
            45.0,
            46.0,
        ]
        assert result.loc[0, "volume"] == 10_000_000
        assert result.loc[1, "close"] == 45.5

    def test_normalize_does_not_double_adjust_already_rebased_split(self):
        mgr = DataSourceManager()
        index = pd.DatetimeIndex(
            ["2024-03-06", "2024-03-07", "2024-03-08"],
            name="Date",
        )
        df = pd.DataFrame(
            {
                "Open": [84.0, 85.0, 86.0],
                "High": [85.0, 86.0, 87.0],
                "Low": [83.0, 84.0, 85.0],
                "Close": [84.5, 85.5, 86.5],
                "Adj Close": [84.5, 85.5, 86.5],
                "Volume": [1_000_000, 5_000_000, 4_000_000],
                "Stock Splits": [0.0, 5.0, 0.0],
            },
            index=index,
        )

        result = mgr._normalize_ohlcv(df)

        assert result["close"].tolist() == [84.5, 85.5, 86.5]
        assert result["volume"].tolist() == [1_000_000, 5_000_000, 4_000_000]

    def test_normalize_applies_multiple_splits_cumulatively(self):
        mgr = DataSourceManager()
        index = pd.DatetimeIndex(
            ["2020-01-02", "2022-01-03", "2024-01-02"],
            name="Date",
        )
        df = pd.DataFrame(
            {
                "Open": [600.0, 300.0, 100.0],
                "High": [600.0, 300.0, 100.0],
                "Low": [600.0, 300.0, 100.0],
                "Close": [600.0, 300.0, 100.0],
                "Adj Close": [100.0, 100.0, 100.0],
                "Volume": [100.0, 200.0, 600.0],
                "Stock Splits": [0.0, 2.0, 3.0],
            },
            index=index,
        )

        result = mgr._normalize_ohlcv(df)

        assert result["close"].tolist() == [100.0, 100.0, 100.0]
        assert result["volume"].tolist() == [600.0, 600.0, 600.0]

    def test_normalize_drops_isolated_repaired_outlier(self):
        mgr = DataSourceManager()
        index = pd.DatetimeIndex(
            ["2026-08-07", "2026-08-10", "2026-08-11"],
            name="Date",
        )
        df = pd.DataFrame(
            {
                "Open": [90.0, 60.0, 46.0],
                "High": [92.0, 61.0, 47.0],
                "Low": [89.0, 59.0, 45.0],
                "Close": [90.0, 60.0, 45.5],
                "Adj Close": [45.0, 60.0, 45.5],
                "Volume": [8_000_000, 3_000_000, 9_000_000],
                "Stock Splits": [0.0, 0.0, 2.0],
                "Repaired?": [False, True, False],
            },
            index=index,
        )

        result = mgr._normalize_ohlcv(df)

        assert result["date"].tolist() == ["2026-08-07", "2026-08-11"]
        assert result["close"].tolist() == [45.0, 45.5]


class TestSourceMetrics:
    """Test success/failure rate tracking."""

    def test_success_rate_calculation(self):
        metrics = SourceMetrics(source="yfinance")

        assert metrics.success_rate == 0.0
        assert metrics.failure_rate == 100.0

        metrics.total_attempts = 10
        metrics.successful_fetches = 8
        metrics.failed_fetches = 2

        assert metrics.success_rate == 80.0
        assert metrics.failure_rate == 20.0

    def test_get_metrics(self, manager):
        manager.metrics.total_attempts = 10
        manager.metrics.successful_fetches = 9
        manager.metrics.failed_fetches = 1

        metrics = manager.get_metrics()

        assert "yfinance" in metrics
        assert metrics["yfinance"]["total_attempts"] == 10
        assert metrics["yfinance"]["success_rate"] == 90.0

    def test_metrics_updated_on_fetch(self, manager, sample_price_data):
        manager.reset_metrics()

        with patch("yfinance.download") as mock_download:
            mock_download.return_value = sample_price_data.set_index("date")

            result = manager._fetch_yfinance("AAPL", "2024-01-01", None, False, 30.0)

            assert result.success
            metrics = manager.metrics
            assert metrics.total_attempts == 1
            assert metrics.successful_fetches == 1


class TestAlerting:
    """Test alerting when failure rate exceeds threshold."""

    def test_alert_on_high_failure_rate(self, manager, caplog):
        import logging
        caplog.set_level(logging.CRITICAL)

        metrics = manager.metrics
        metrics.total_attempts = 20
        metrics.successful_fetches = 15
        metrics.failed_fetches = 5  # 25% failure rate

        manager._check_and_alert()

        assert any("PRICE SOURCE DEGRADED" in record.message for record in caplog.records)

    def test_no_alert_below_threshold(self, manager, caplog):
        import logging
        caplog.set_level(logging.CRITICAL)

        metrics = manager.metrics
        metrics.total_attempts = 20
        metrics.successful_fetches = 19
        metrics.failed_fetches = 1  # 5% failure rate

        manager._check_and_alert()

        assert not any("PRICE SOURCE DEGRADED" in record.message for record in caplog.records)

    def test_alert_cooldown(self, manager, caplog):
        import logging
        caplog.set_level(logging.CRITICAL)

        metrics = manager.metrics
        metrics.total_attempts = 20
        metrics.successful_fetches = 15
        metrics.failed_fetches = 5

        manager._check_and_alert()
        assert any("PRICE SOURCE DEGRADED" in record.message for record in caplog.records)

        caplog.clear()

        manager._check_and_alert()
        assert not any("PRICE SOURCE DEGRADED" in record.message for record in caplog.records)


class TestDataSourceInDatabase:
    """Test data source is included in database records."""

    def test_data_source_in_database(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE price_daily (
                    ticker TEXT, date TEXT, open REAL, high REAL,
                    low REAL, close REAL, volume REAL,
                    data_source TEXT DEFAULT 'yfinance',
                    fetch_timestamp TEXT,
                    PRIMARY KEY (ticker, date)
                )
            """)
            conn.execute(
                "INSERT INTO price_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("AAPL", "2024-01-01", 100.0, 105.0, 99.0, 103.0, 1e6,
                 "yfinance", "2024-01-01T10:00:00Z"),
            )
            conn.commit()

            row = conn.execute(
                "SELECT data_source, fetch_timestamp FROM price_daily WHERE ticker = ?",
                ("AAPL",),
            ).fetchone()

            assert row[0] == "yfinance"
            assert row[1] == "2024-01-01T10:00:00Z"
            conn.close()
        finally:
            os.unlink(db_path)


class TestGlobalInstance:
    """Test global DataSourceManager instance."""

    def test_get_data_source_manager_singleton(self):
        manager1 = get_data_source_manager()
        manager2 = get_data_source_manager()
        assert manager1 is manager2

    def test_reset_metrics(self, manager):
        manager.metrics.total_attempts = 10
        manager.metrics.successful_fetches = 8

        manager.reset_metrics()

        assert manager.metrics.total_attempts == 0
        assert manager.metrics.successful_fetches == 0
