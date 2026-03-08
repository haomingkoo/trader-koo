"""Multi-source data fetching with redundancy and fallback support.

This module implements a multi-source data fetching strategy with:
- yfinance as primary source
- Alpha Vantage as secondary fallback
- CSV as final fallback
- Success/failure rate tracking
- Source logging and metrics
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

LOG = logging.getLogger(__name__)


class DataSource(Enum):
    """Data source enumeration."""
    YFINANCE = "yfinance"
    ALPHA_VANTAGE = "alpha_vantage"
    CSV_FALLBACK = "csv_fallback"


@dataclass
class FetchResult:
    """Result of a data fetch operation."""
    data: pd.DataFrame
    source: DataSource
    timestamp: datetime
    success: bool
    error: Optional[str] = None


@dataclass
class SourceMetrics:
    """Metrics for a data source."""
    source: DataSource
    total_attempts: int = 0
    successful_fetches: int = 0
    failed_fetches: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_attempts == 0:
            return 0.0
        return (self.successful_fetches / self.total_attempts) * 100
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        return 100.0 - self.success_rate


class DataSourceManager:
    """Manages multi-source data fetching with fallback and metrics tracking."""
    
    def __init__(self, csv_fallback_dir: Optional[Path] = None):
        """Initialize the data source manager.
        
        Args:
            csv_fallback_dir: Directory containing CSV fallback files
        """
        self.csv_fallback_dir = csv_fallback_dir or Path(__file__).parent / "fallback_data"
        self.metrics: dict[DataSource, SourceMetrics] = {
            source: SourceMetrics(source=source) for source in DataSource
        }
        self._alert_threshold = 10.0  # Alert when failure rate > 10%
        self._last_alert_time: dict[DataSource, float] = {}
        self._alert_cooldown = 3600  # 1 hour cooldown between alerts
    
    def fetch_ticker_data(
        self,
        ticker: str,
        start: str,
        end: Optional[str] = None,
        auto_adjust: bool = False,
        timeout_sec: float = 30.0,
    ) -> FetchResult:
        """Fetch ticker data with multi-source fallback.
        
        Tries sources in order:
        1. yfinance (primary)
        2. Alpha Vantage (secondary)
        3. CSV fallback (final)
        
        Args:
            ticker: Ticker symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD), optional
            auto_adjust: Whether to auto-adjust prices
            timeout_sec: Request timeout in seconds
            
        Returns:
            FetchResult with data and metadata
        """
        # Try primary source: yfinance
        result = self._fetch_yfinance(ticker, start, end, auto_adjust, timeout_sec)
        if result.success:
            self._check_and_alert(DataSource.YFINANCE)
            return result
        
        LOG.warning(f"yfinance failed for {ticker}: {result.error}, trying Alpha Vantage")
        
        # Try secondary source: Alpha Vantage
        result = self._fetch_alpha_vantage(ticker, start, end)
        if result.success:
            self._check_and_alert(DataSource.ALPHA_VANTAGE)
            return result
        
        LOG.warning(f"Alpha Vantage failed for {ticker}: {result.error}, trying CSV fallback")
        
        # Try final fallback: CSV
        result = self._load_csv_fallback(ticker, start, end)
        self._check_and_alert(DataSource.CSV_FALLBACK)
        return result
    
    def _fetch_yfinance(
        self,
        ticker: str,
        start: str,
        end: Optional[str],
        auto_adjust: bool,
        timeout_sec: float,
    ) -> FetchResult:
        """Fetch data from yfinance."""
        metrics = self.metrics[DataSource.YFINANCE]
        metrics.total_attempts += 1
        
        try:
            LOG.info(f"Fetching {ticker} from yfinance (start={start}, end={end})")
            raw = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                auto_adjust=auto_adjust,
                progress=False,
                actions=False,
                group_by="column",
                threads=False,
                timeout=timeout_sec,
            )
            
            if raw is None or raw.empty:
                metrics.failed_fetches += 1
                return FetchResult(
                    data=pd.DataFrame(),
                    source=DataSource.YFINANCE,
                    timestamp=datetime.now(),
                    success=False,
                    error="Empty response from yfinance"
                )
            
            # Normalize schema
            raw_reset = raw.reset_index()
            df = self._normalize_ohlcv(raw_reset)
            
            metrics.successful_fetches += 1
            LOG.info(f"Successfully fetched {ticker} from yfinance ({len(df)} rows)")
            
            return FetchResult(
                data=df,
                source=DataSource.YFINANCE,
                timestamp=datetime.now(),
                success=True
            )
            
        except Exception as e:
            metrics.failed_fetches += 1
            LOG.error(f"yfinance fetch failed for {ticker}: {e}")
            return FetchResult(
                data=pd.DataFrame(),
                source=DataSource.YFINANCE,
                timestamp=datetime.now(),
                success=False,
                error=str(e)
            )
    
    def _fetch_alpha_vantage(
        self,
        ticker: str,
        start: str,
        end: Optional[str],
    ) -> FetchResult:
        """Fetch data from Alpha Vantage."""
        metrics = self.metrics[DataSource.ALPHA_VANTAGE]
        metrics.total_attempts += 1
        
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            metrics.failed_fetches += 1
            return FetchResult(
                data=pd.DataFrame(),
                source=DataSource.ALPHA_VANTAGE,
                timestamp=datetime.now(),
                success=False,
                error="ALPHA_VANTAGE_API_KEY not configured"
            )
        
        try:
            LOG.info(f"Fetching {ticker} from Alpha Vantage")
            import requests
            
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": ticker,
                "apikey": api_key,
                "outputsize": "full",
                "datatype": "json"
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "Error Message" in data:
                metrics.failed_fetches += 1
                return FetchResult(
                    data=pd.DataFrame(),
                    source=DataSource.ALPHA_VANTAGE,
                    timestamp=datetime.now(),
                    success=False,
                    error=data["Error Message"]
                )
            
            if "Time Series (Daily)" not in data:
                metrics.failed_fetches += 1
                return FetchResult(
                    data=pd.DataFrame(),
                    source=DataSource.ALPHA_VANTAGE,
                    timestamp=datetime.now(),
                    success=False,
                    error="Unexpected response format from Alpha Vantage"
                )
            
            # Convert to DataFrame
            time_series = data["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(time_series, orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns to match our schema
            df = df.rename(columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume"
            })
            
            # Convert to numeric
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Filter by date range
            df = df.loc[start:end] if end else df.loc[start:]
            
            # Add date column
            df = df.reset_index()
            df = df.rename(columns={"index": "date"})
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
            
            metrics.successful_fetches += 1
            LOG.info(f"Successfully fetched {ticker} from Alpha Vantage ({len(df)} rows)")
            
            return FetchResult(
                data=df[["date", "open", "high", "low", "close", "volume"]],
                source=DataSource.ALPHA_VANTAGE,
                timestamp=datetime.now(),
                success=True
            )
            
        except Exception as e:
            metrics.failed_fetches += 1
            LOG.error(f"Alpha Vantage fetch failed for {ticker}: {e}")
            return FetchResult(
                data=pd.DataFrame(),
                source=DataSource.ALPHA_VANTAGE,
                timestamp=datetime.now(),
                success=False,
                error=str(e)
            )
    
    def _load_csv_fallback(
        self,
        ticker: str,
        start: str,
        end: Optional[str],
    ) -> FetchResult:
        """Load data from CSV fallback."""
        metrics = self.metrics[DataSource.CSV_FALLBACK]
        metrics.total_attempts += 1
        
        csv_path = self.csv_fallback_dir / f"{ticker}.csv"
        
        if not csv_path.exists():
            metrics.failed_fetches += 1
            LOG.error(f"CSV fallback not found for {ticker} at {csv_path}")
            return FetchResult(
                data=pd.DataFrame(),
                source=DataSource.CSV_FALLBACK,
                timestamp=datetime.now(),
                success=False,
                error=f"CSV file not found: {csv_path}"
            )
        
        try:
            LOG.info(f"Loading {ticker} from CSV fallback: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Ensure required columns exist
            required_cols = ["date", "open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_cols):
                metrics.failed_fetches += 1
                return FetchResult(
                    data=pd.DataFrame(),
                    source=DataSource.CSV_FALLBACK,
                    timestamp=datetime.now(),
                    success=False,
                    error=f"CSV missing required columns. Found: {df.columns.tolist()}"
                )
            
            # Filter by date range
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= start]
            if end:
                df = df[df["date"] <= end]
            
            # Convert date back to string
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")
            
            metrics.successful_fetches += 1
            LOG.info(f"Successfully loaded {ticker} from CSV fallback ({len(df)} rows)")
            
            return FetchResult(
                data=df[required_cols],
                source=DataSource.CSV_FALLBACK,
                timestamp=datetime.now(),
                success=True
            )
            
        except Exception as e:
            metrics.failed_fetches += 1
            LOG.error(f"CSV fallback load failed for {ticker}: {e}")
            return FetchResult(
                data=pd.DataFrame(),
                source=DataSource.CSV_FALLBACK,
                timestamp=datetime.now(),
                success=False,
                error=str(e)
            )
    
    def _normalize_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize OHLCV DataFrame schema."""
        # Handle different column name formats
        df_copy = df.copy()
        
        # Rename columns to lowercase
        df_copy.columns = [str(col).lower() for col in df_copy.columns]
        
        # Ensure date column
        if "date" not in df_copy.columns:
            if df_copy.index.name and "date" in str(df_copy.index.name).lower():
                df_copy = df_copy.reset_index()
                df_copy = df_copy.rename(columns={df_copy.columns[0]: "date"})
        
        # Convert date to datetime if needed
        if "date" in df_copy.columns:
            df_copy["date"] = pd.to_datetime(df_copy["date"])
            df_copy["date"] = df_copy["date"].dt.strftime("%Y-%m-%d")
        
        # Ensure required columns exist
        required = ["date", "open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df_copy.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df_copy[required]
    
    def _check_and_alert(self, source: DataSource) -> None:
        """Check failure rate and alert if threshold exceeded."""
        metrics = self.metrics[source]
        
        # Only check primary source for alerts
        if source != DataSource.YFINANCE:
            return
        
        # Need minimum attempts before alerting
        if metrics.total_attempts < 10:
            return
        
        failure_rate = metrics.failure_rate
        
        if failure_rate > self._alert_threshold:
            # Check cooldown
            last_alert = self._last_alert_time.get(source, 0)
            if time.time() - last_alert < self._alert_cooldown:
                return
            
            LOG.error(
                f"ALERT: {source.value} failure rate ({failure_rate:.1f}%) exceeds threshold "
                f"({self._alert_threshold}%). Attempts: {metrics.total_attempts}, "
                f"Failures: {metrics.failed_fetches}"
            )
            self._last_alert_time[source] = time.time()
    
    def get_metrics(self) -> dict[str, dict]:
        """Get metrics for all data sources.
        
        Returns:
            Dictionary mapping source name to metrics
        """
        return {
            source.value: {
                "total_attempts": metrics.total_attempts,
                "successful_fetches": metrics.successful_fetches,
                "failed_fetches": metrics.failed_fetches,
                "success_rate": round(metrics.success_rate, 2),
                "failure_rate": round(metrics.failure_rate, 2),
            }
            for source, metrics in self.metrics.items()
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics (useful for testing)."""
        self.metrics = {
            source: SourceMetrics(source=source) for source in DataSource
        }
        self._last_alert_time.clear()


# Global instance
_data_source_manager: Optional[DataSourceManager] = None


def get_data_source_manager() -> DataSourceManager:
    """Get or create the global data source manager instance."""
    global _data_source_manager
    if _data_source_manager is None:
        _data_source_manager = DataSourceManager()
    return _data_source_manager
