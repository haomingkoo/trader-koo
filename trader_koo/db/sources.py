"""Data source fetching for price data.

yfinance is the sole data source. When it fails, the failure is
propagated explicitly — no hidden fallbacks that silently degrade.

Includes a thread-based hard timeout around ``yf.download`` because
yfinance's built-in ``timeout`` parameter only sets the HTTP socket
timeout — it does not protect against DNS hangs, SSL negotiation
stalls, or response-streaming freezes that block the calling thread.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

LOG = logging.getLogger(__name__)

# yfinance is the only price source by design (see module docstring).
SOURCE_NAME = "yfinance"
PRICE_ADJUSTMENT_VERSION = "yfinance-actions-v1"
# 3-for-2 is the smallest common split whose unexplained discontinuity should
# fail research closed. Smaller declared fractional actions are still handled
# from vendor evidence below; they are never inferred from price movement.
SCALE_BREAK_RATIO = 1.45

# Hard timeout for any single yf.download call.  If the call does not
# return within this many seconds, the thread is abandoned and the
# ticker is marked as failed with a TimeoutError.
_HARD_TIMEOUT_SEC = 60.0


@dataclass
class FetchResult:
    """Result of a data fetch operation."""
    data: pd.DataFrame
    source: str
    timestamp: datetime
    success: bool
    error: Optional[str] = None
    adjustment_basis: str = "unknown"
    adjustment_version: str = "unknown"
    basis_status: str = "unverified"
    corporate_actions: list[dict] = field(default_factory=list)
    unresolved_discontinuities: list[dict] = field(default_factory=list)


@dataclass
class SourceMetrics:
    """Success/failure tracking for the price source."""
    source: str = SOURCE_NAME
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
        self.metrics = SourceMetrics()
        self._alert_threshold = 10.0
        self._last_alert_time = 0.0
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
        self._check_and_alert()

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
        metrics = self.metrics
        metrics.total_attempts += 1

        try:
            LOG.info(f"Fetching {ticker} from yfinance (start={start}, end={end})")
            hard_timeout = max(timeout_sec + 10, _HARD_TIMEOUT_SEC)
            raw = self._download_with_hard_timeout(
                ticker=ticker, start=start, end=end,
                auto_adjust=auto_adjust, timeout_sec=timeout_sec,
                hard_timeout=hard_timeout,
            )

            # Index tickers (^VIX, ^GSPC, etc.) often return empty data for
            # narrow date-range queries.  Fall back to period="5d" which uses
            # a different Yahoo endpoint that is more reliable for indices.
            if (raw is None or raw.empty) and ticker.startswith("^"):
                LOG.warning(
                    "yfinance date-range fetch returned empty for index ticker %s; "
                    "retrying with period='5d'",
                    ticker,
                )
                raw = self._download_with_hard_timeout(
                    ticker=ticker, period="5d",
                    auto_adjust=auto_adjust, timeout_sec=timeout_sec,
                    hard_timeout=hard_timeout,
                )

            if raw is None or raw.empty:
                metrics.failed_fetches += 1
                return FetchResult(
                    data=pd.DataFrame(),
                    source=SOURCE_NAME,
                    timestamp=datetime.now(),
                    success=False,
                    error=f"Empty response from yfinance for {ticker}",
                )

            df = self._normalize_ohlcv(raw, auto_adjust=auto_adjust)

            metrics.successful_fetches += 1
            LOG.info(f"Successfully fetched {ticker} from yfinance ({len(df)} rows)")

            return FetchResult(
                data=df,
                source=SOURCE_NAME,
                timestamp=datetime.now(),
                success=True,
                adjustment_basis=str(df.attrs["adjustment_basis"]),
                adjustment_version=str(df.attrs["adjustment_version"]),
                basis_status=str(df.attrs["basis_status"]),
                corporate_actions=list(df.attrs["corporate_actions"]),
                unresolved_discontinuities=list(df.attrs["unresolved_discontinuities"]),
            )

        except Exception as e:
            metrics.failed_fetches += 1
            LOG.error(f"yfinance fetch failed for {ticker}: {e}")
            return FetchResult(
                data=pd.DataFrame(),
                source=SOURCE_NAME,
                timestamp=datetime.now(),
                success=False,
                error=str(e),
            )

    def fetch_ticker_actions(
        self,
        ticker: str,
        *,
        hard_timeout: float = _HARD_TIMEOUT_SEC,
    ) -> list[dict]:
        """Fetch Yahoo's full split ledger for late-action reconciliation."""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(yf.Ticker(ticker).get_actions, period="max")
        try:
            raw = future.result(timeout=hard_timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"yfinance action history hung for {ticker} (hard timeout {hard_timeout}s)"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if raw is None or len(raw) == 0:
            return []
        frame = raw.to_frame() if isinstance(raw, pd.Series) else raw.copy()
        if isinstance(frame.index, pd.MultiIndex):
            frame.index = frame.index.get_level_values(0)
        columns = {str(column).strip().lower(): column for column in frame.columns}
        split_column = columns.get("stock splits")
        if split_column is None:
            return []
        actions: list[dict] = []
        splits = pd.to_numeric(frame[split_column], errors="coerce").fillna(0)
        for index in splits[splits > 0].index:
            factor = float(splits.loc[index])
            if factor == 1.0:
                continue
            actions.append(
                {
                    "action_date": pd.Timestamp(index).date().isoformat(),
                    "action_type": "split" if factor > 1 else "reverse_split",
                    "value": factor,
                }
            )
        return actions

    @staticmethod
    def _download_with_hard_timeout(
        *,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
        period: str | None = None,
        auto_adjust: bool = False,
        timeout_sec: float = 30.0,
        hard_timeout: float = _HARD_TIMEOUT_SEC,
    ) -> pd.DataFrame | None:
        """Run ``yf.download`` in a thread with a hard wall-clock timeout.

        yfinance's ``timeout`` parameter only sets the socket-level
        timeout.  If the underlying HTTP request hangs at DNS, SSL, or
        streaming level, the call blocks forever.  This wrapper uses a
        ``ThreadPoolExecutor`` so the caller is never stuck longer than
        *hard_timeout* seconds.
        """
        kwargs: dict = {
            "tickers": ticker,
            "auto_adjust": auto_adjust,
            "progress": False,
            # Keep corporate actions so normalization can put every OHLCV row
            # on the current share scale. repair=True fixes Yahoo's occasional
            # isolated 2x/4x rows before the declared split factors are applied.
            "actions": True,
            "repair": True,
            "group_by": "column",
            "threads": False,
            "timeout": timeout_sec,
        }
        if period:
            kwargs["period"] = period
        else:
            if start:
                kwargs["start"] = start
            if end:
                kwargs["end"] = end

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(yf.download, **kwargs)
            try:
                return future.result(timeout=hard_timeout)
            except concurrent.futures.TimeoutError:
                LOG.error(
                    "yfinance hard timeout (%ss) for %s — download hung, abandoning",
                    hard_timeout,
                    ticker,
                )
                raise TimeoutError(
                    f"yfinance download hung for {ticker} (hard timeout {hard_timeout}s)"
                )

    @staticmethod
    def _normalize_ohlcv(df: pd.DataFrame, *, auto_adjust: bool = False) -> pd.DataFrame:
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

        required = ["date", "open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df_copy.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        actions: list[dict] = []
        if "dividends" in df_copy.columns:
            dividends = pd.to_numeric(df_copy["dividends"], errors="coerce").fillna(0)
            for index in dividends[dividends != 0].index:
                actions.append(
                    {
                        "action_date": df_copy.loc[index, "date"].date().isoformat(),
                        "action_type": "dividend",
                        "value": float(dividends.loc[index]),
                        "applied_to_prices": bool(auto_adjust),
                    }
                )

        unresolved: list[dict] = []

        # Yahoo is inconsistent about split adjustment: some downloads contain
        # raw pre-split prices while others are already rebased. ``Adj Close``
        # provides an independent provider basis signal: its ratio to ``Close``
        # changes by the declared factor only when the raw OHLC needs rebasing.
        # Price continuity is not evidence because a split can coincide with a
        # genuine large move. If Yahoo omits or contradicts that basis signal,
        # preserve the observations and fail research closed.
        if "stock splits" in df_copy.columns:
            split_factors = pd.to_numeric(df_copy["stock splits"], errors="coerce").fillna(0)
            for index in split_factors[split_factors > 0].index:
                factor = float(split_factors.loc[index])
                if factor == 1.0:
                    continue

                split_date = df_copy.loc[index, "date"]
                action = {
                    "action_date": split_date.date().isoformat(),
                    "action_type": "split" if factor > 1 else "reverse_split",
                    "value": factor,
                    "applied_to_prices": False,
                }
                actions.append(action)
                before_split = df_copy["date"] < split_date
                on_or_after_split = df_copy["date"] >= split_date
                comparison_before = before_split
                if "repaired?" in df_copy.columns:
                    repaired_rows = df_copy["repaired?"].fillna(False).astype(bool)
                    comparison_before = before_split & ~repaired_rows
                before_closes = pd.to_numeric(
                    df_copy.loc[comparison_before, "close"], errors="coerce"
                ).dropna()
                after_closes = pd.to_numeric(
                    df_copy.loc[on_or_after_split, "close"], errors="coerce"
                ).dropna()
                if before_closes.empty or after_closes.empty:
                    continue

                before_close = float(before_closes.iloc[-1])
                after_close = float(after_closes.iloc[0])
                if before_close <= 0 or after_close <= 0:
                    continue

                needs_rebase = False
                already_rebased = bool(auto_adjust)
                if not auto_adjust and "adj close" in df_copy.columns:
                    before_adjusted = pd.to_numeric(
                        df_copy.loc[comparison_before, "adj close"], errors="coerce"
                    ).dropna()
                    after_adjusted = pd.to_numeric(
                        df_copy.loc[on_or_after_split, "adj close"], errors="coerce"
                    ).dropna()
                    if not before_adjusted.empty and not after_adjusted.empty:
                        before_adj_close = float(before_adjusted.iloc[-1])
                        after_adj_close = float(after_adjusted.iloc[0])
                        if before_adj_close > 0 and after_adj_close > 0:
                            ratio_change = (
                                (after_adj_close / after_close)
                                / (before_adj_close / before_close)
                            )
                            matches_factor = math.isclose(
                                ratio_change, factor, rel_tol=0.01, abs_tol=1e-6
                            )
                            matches_one = math.isclose(
                                ratio_change, 1.0, rel_tol=0.01, abs_tol=1e-6
                            )
                            needs_rebase = matches_factor and not matches_one
                            already_rebased = matches_one and not matches_factor

                if not needs_rebase and not already_rebased:
                    action["basis_evidence"] = "unresolved"
                    unresolved.append(
                        {
                            "action_date": action["action_date"],
                            "action_type": action["action_type"],
                            "value": factor,
                            "reason": "declared_action_basis_unresolved",
                        }
                    )
                    continue
                action["basis_evidence"] = (
                    "provider_adjusted_close" if needs_rebase else "provider_already_adjusted"
                )
                if already_rebased:
                    continue

                for column in ("open", "high", "low", "close"):
                    df_copy.loc[before_split, column] = (
                        pd.to_numeric(df_copy.loc[before_split, column], errors="coerce")
                        / factor
                    )
                df_copy["volume"] = pd.to_numeric(
                    df_copy["volume"], errors="coerce"
                ).astype(float)
                df_copy.loc[before_split, "volume"] = (
                    df_copy.loc[before_split, "volume"] * factor
                )
                action["applied_to_prices"] = True

        # A repaired row is yfinance's inferred value, not an exchange print.
        # Drop it when it remains an isolated >35% outlier after split
        # normalization; retaining that synthetic point would corrupt technicals.
        if "repaired?" in df_copy.columns:
            repaired = df_copy["repaired?"].fillna(False).astype(bool)
            closes = pd.to_numeric(df_copy["close"], errors="coerce")
            drop_indexes: list[object] = []
            for position in range(1, len(df_copy) - 1):
                index = df_copy.index[position]
                if not repaired.loc[index]:
                    continue
                neighbors = [float(closes.iloc[position - 1]), float(closes.iloc[position + 1])]
                reference = sum(neighbors) / len(neighbors)
                value = float(closes.iloc[position])
                if value > 0 and reference > 0 and max(value / reference, reference / value) > 1.35:
                    drop_indexes.append(index)
            if drop_indexes:
                LOG.warning(
                    "Dropping %d isolated repaired price row(s) after split normalization",
                    len(drop_indexes),
                )
                df_copy = df_copy.drop(index=drop_indexes)

        # A large remaining scale break is not evidence of a split. Preserve the
        # observations, but fail the series closed until declared evidence exists.
        ordered = df_copy.sort_values("date")
        closes = pd.to_numeric(ordered["close"], errors="coerce")
        dates = ordered["date"].tolist()
        for position in range(1, len(ordered)):
            previous, current = closes.iloc[position - 1], closes.iloc[position]
            if pd.isna(previous) or pd.isna(current) or previous <= 0 or current <= 0:
                continue
            ratio = max(float(previous / current), float(current / previous))
            if ratio >= SCALE_BREAK_RATIO:
                unresolved.append(
                    {
                        "previous_date": dates[position - 1].date().isoformat(),
                        "date": dates[position].date().isoformat(),
                        "ratio": round(ratio, 6),
                        "reason": "unexplained_adjacent_price_discontinuity",
                    }
                )

        df_copy["date"] = df_copy["date"].dt.strftime("%Y-%m-%d")
        normalized = df_copy[required]
        normalized.attrs.update(
            {
                "adjustment_basis": "total_return" if auto_adjust else "split_adjusted_price_only",
                "adjustment_version": PRICE_ADJUSTMENT_VERSION,
                "basis_status": "unresolved" if unresolved else "verified",
                "corporate_actions": actions,
                "unresolved_discontinuities": unresolved,
            }
        )
        return normalized

    def _check_and_alert(self) -> None:
        """Log a CRITICAL alert when the failure rate exceeds threshold."""
        metrics = self.metrics

        if metrics.total_attempts < 10:
            return

        failure_rate = metrics.failure_rate

        if failure_rate > self._alert_threshold:
            if time.time() - self._last_alert_time < self._alert_cooldown:
                return

            LOG.critical(
                "PRICE SOURCE DEGRADED: %s failure rate %.1f%% exceeds %.1f%% "
                "threshold (attempts=%d, failures=%d). "
                "Check yfinance version and Yahoo Finance API status.",
                metrics.source,
                failure_rate,
                self._alert_threshold,
                metrics.total_attempts,
                metrics.failed_fetches,
            )
            self._last_alert_time = time.time()

    def get_metrics(self) -> dict[str, dict]:
        metrics = self.metrics
        return {
            metrics.source: {
                "total_attempts": metrics.total_attempts,
                "successful_fetches": metrics.successful_fetches,
                "failed_fetches": metrics.failed_fetches,
                "success_rate": round(metrics.success_rate, 2),
                "failure_rate": round(metrics.failure_rate, 2),
            }
        }

    def reset_metrics(self) -> None:
        self.metrics = SourceMetrics()
        self._last_alert_time = 0.0


# Global instance
_data_source_manager: Optional[DataSourceManager] = None


def get_data_source_manager() -> DataSourceManager:
    """Get or create the global data source manager instance."""
    global _data_source_manager
    if _data_source_manager is None:
        _data_source_manager = DataSourceManager()
    return _data_source_manager
