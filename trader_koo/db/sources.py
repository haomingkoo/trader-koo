"""Data source fetching for price data.

yfinance is the sole data source. When it fails, the failure is
propagated explicitly — no hidden fallbacks that silently degrade.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd
import yfinance as yf

LOG = logging.getLogger(__name__)


class DataSource(Enum):
    """Data source enumeration."""
    YFINANCE = "yfinance"


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
        if self.total_attempts == 0:
            return 0.0
        return (self.successful_fetches / self.total_attempts) * 100

    @property
    def failure_rate(self) -> float:
        return 100.0 - self.success_rate


class PriceFetchError(Exception):
    """Raised when price data cannot be fetched from any source."""


class DataSourceManager:
    """Fetches price data from yfinance with metrics tracking.

    Fails explicitly when yfinance returns no data — there are no
    hidden fallback sources that silently swallow failures.
    """

    def __init__(self) -> None:
        self.metrics: dict[DataSource, SourceMetrics] = {
            source: SourceMetrics(source=source) for source in DataSource
        }
        self._alert_threshold = 10.0
        self._last_alert_time: dict[DataSource, float] = {}
        self._alert_cooldown = 3600

    def fetch_ticker_data(
        self,
        ticker: str,
        start: str,
        end: Optional[str] = None,
        auto_adjust: bool = False,
        timeout_sec: float = 30.0,
    ) -> FetchResult:
        """Fetch ticker data from yfinance.

        Raises PriceFetchError when yfinance returns no data so the
        caller can mark the ticker as failed instead of silently
        recording zero rows.
        """
        result = self._fetch_yfinance(ticker, start, end, auto_adjust, timeout_sec)
        self._check_and_alert(DataSource.YFINANCE)

        if not result.success or result.data.empty:
            raise PriceFetchError(
                f"yfinance returned no data for {ticker}: {result.error}"
            )

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
                    error=f"Empty response from yfinance for {ticker}",
                )

            df = self._normalize_ohlcv(raw)

            metrics.successful_fetches += 1
            LOG.info(f"Successfully fetched {ticker} from yfinance ({len(df)} rows)")

            return FetchResult(
                data=df,
                source=DataSource.YFINANCE,
                timestamp=datetime.now(),
                success=True,
            )

        except Exception as e:
            metrics.failed_fetches += 1
            LOG.error(f"yfinance fetch failed for {ticker}: {e}")
            return FetchResult(
                data=pd.DataFrame(),
                source=DataSource.YFINANCE,
                timestamp=datetime.now(),
                success=False,
                error=str(e),
            )

    @staticmethod
    def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize OHLCV DataFrame schema.

        Handles both old-style flat columns (yfinance <1.0) and
        new MultiIndex columns (yfinance >=1.0).
        """
        df_copy = df.copy()

        # yfinance >=1.0 returns MultiIndex columns like ('Close', 'AAPL').
        # Flatten to just the first level ('Close').
        if isinstance(df_copy.columns, pd.MultiIndex):
            df_copy.columns = df_copy.columns.get_level_values(0)

        # Reset index to turn the Date index into a column
        if "Date" in (df_copy.index.names or []) or (
            df_copy.index.name and "date" in str(df_copy.index.name).lower()
        ):
            df_copy = df_copy.reset_index()

        # Lowercase all column names
        df_copy.columns = [str(col).strip().lower() for col in df_copy.columns]

        # Ensure date column exists
        if "date" not in df_copy.columns:
            raise ValueError(
                f"No 'date' column after normalization. Columns: {list(df_copy.columns)}"
            )

        df_copy["date"] = pd.to_datetime(df_copy["date"])
        df_copy["date"] = df_copy["date"].dt.strftime("%Y-%m-%d")

        required = ["date", "open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df_copy.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df_copy[required]

    def _check_and_alert(self, source: DataSource) -> None:
        """Log a CRITICAL alert when failure rate exceeds threshold."""
        metrics = self.metrics[source]

        if metrics.total_attempts < 10:
            return

        failure_rate = metrics.failure_rate

        if failure_rate > self._alert_threshold:
            last_alert = self._last_alert_time.get(source, 0)
            if time.time() - last_alert < self._alert_cooldown:
                return

            LOG.critical(
                "PRICE SOURCE DEGRADED: %s failure rate %.1f%% exceeds %.1f%% "
                "threshold (attempts=%d, failures=%d). "
                "Check yfinance version and Yahoo Finance API status.",
                source.value,
                failure_rate,
                self._alert_threshold,
                metrics.total_attempts,
                metrics.failed_fetches,
            )
            self._last_alert_time[source] = time.time()

    def get_metrics(self) -> dict[str, dict]:
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
