"""Pydantic models that mirror the SQLite tables they are read from.

Fields and Literal values follow the table DDL and CHECK constraints; tests
assert the fields stay a subset of the live columns. Dates stay ``str``
because the tables store them as TEXT.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class PriceBar(BaseModel):
    """One ``price_daily`` row."""

    ticker: str
    date: str
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    data_source: str | None = None
    fetch_timestamp: str | None = None
    adjustment_basis: str | None = None
    adjustment_version: str | None = None
    basis_status: str | None = None
    unresolved_reason: str | None = None


class PaperTrade(BaseModel):
    """The lifecycle columns of a ``paper_trades`` row.

    Decision and review columns pass through as extras.
    """

    model_config = ConfigDict(extra="allow")

    id: int
    campaign_id: str
    report_run_id: str | None = None
    report_date: str
    ticker: str
    direction: Literal["long", "short"]
    status: Literal["open", "closed", "stopped_out", "target_hit", "expired"]
    entry_price: float
    entry_date: str
    target_price: float | None = None
    stop_loss: float | None = None
    exit_price: float | None = None
    exit_date: str | None = None
    exit_reason: str | None = None
    current_price: float | None = None
    unrealized_pnl_pct: float | None = None
    pnl_pct: float | None = None
    r_multiple: float | None = None
    setup_family: str | None = None
    setup_tier: str | None = None
    score: float | None = None
