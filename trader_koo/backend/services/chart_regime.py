"""HMM regime payload for the chart page, served apart from the commentary.

The nightly report no longer stores per-ticker HMM regimes, so the chart has
to fit one live. Serving it from its own endpoint lets the commentary panel
render without waiting on the fit, and caching the finished result per
(database, ticker, last bar, price revision) makes repeat opens instant. A
failed fit is cached too, so it is not retried on every chart open.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from trader_koo.backend.services.chart_builder import _conn_db_path, _run_read_snapshot
from trader_koo.backend.services.chart_compact import compact_hmm_regime
from trader_koo.backend.services.database import get_price_df
from trader_koo.backend.services.market_data import get_data_sources
from trader_koo.backend.services.report_loader import latest_report_hmm_for_ticker
from trader_koo.db.price_contract import research_price_contract
from trader_koo.structure.hmm_regime import predict_regimes as hmm_predict_regimes

LOG = logging.getLogger("trader_koo.services.chart_regime")

_REGIME_CACHE: "OrderedDict[tuple, tuple[float, dict[str, Any] | None]]" = OrderedDict()
_REGIME_CACHE_LOCK = threading.Lock()
_REGIME_CACHE_MAXSIZE = 128
_REGIME_CACHE_TTL_SEC = 3600.0
_MISS = object()


def _cache_get(key: tuple) -> Any:
    with _REGIME_CACHE_LOCK:
        entry = _REGIME_CACHE.get(key)
        if entry is None or time.time() - entry[0] >= _REGIME_CACHE_TTL_SEC:
            _REGIME_CACHE.pop(key, None)
            return _MISS
        _REGIME_CACHE.move_to_end(key)
        return entry[1]


def _cache_put(key: tuple, value: dict[str, Any] | None) -> None:
    with _REGIME_CACHE_LOCK:
        _REGIME_CACHE[key] = (time.time(), value)
        _REGIME_CACHE.move_to_end(key)
        while len(_REGIME_CACHE) > _REGIME_CACHE_MAXSIZE:
            _REGIME_CACHE.popitem(last=False)


def _live_regime(conn: sqlite3.Connection, ticker: str) -> dict[str, Any] | None:
    prices = get_price_df(conn, ticker)
    if prices.empty:
        return None
    db_path = _conn_db_path(conn)
    revision = research_price_contract(conn, [ticker]).get("revision")
    key = (db_path, ticker, str(prices["date"].max()), len(prices), revision) if db_path else None
    if key is not None:
        cached = _cache_get(key)
        if cached is not _MISS:
            return cached
    try:
        regime = compact_hmm_regime(hmm_predict_regimes(prices, ticker=ticker))
    except Exception as exc:
        LOG.warning("HMM regime detection failed for %s: %s", ticker, exc)
        regime = None
    if key is not None:
        _cache_put(key, regime)
    return regime


def _build_regime_payload(
    conn: sqlite3.Connection,
    ticker: str,
    report_dir: Path,
    report_generated_ts: str | None,
) -> dict[str, Any]:
    ticker = ticker.upper().strip()
    price_contract = get_data_sources(conn, ticker)
    if not price_contract["research_eligible"]:
        return {
            "ticker": ticker,
            "hmm_regime": None,
            "excluded_reason": "price_basis_not_research_eligible",
        }
    stored = latest_report_hmm_for_ticker(report_dir, ticker, generated_ts=report_generated_ts)
    regime = compact_hmm_regime(stored) if stored is not None else _live_regime(conn, ticker)
    return {"ticker": ticker, "hmm_regime": regime}


def build_regime_payload(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    report_dir: Path,
    report_generated_ts: str | None = None,
) -> dict[str, Any]:
    return _run_read_snapshot(
        conn, lambda: _build_regime_payload(conn, ticker, report_dir, report_generated_ts)
    )
