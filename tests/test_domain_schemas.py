"""Pydantic models must mirror the real SQLite tables, not a parallel contract."""
from __future__ import annotations

import dataclasses
import sqlite3

import pytest
from pydantic import ValidationError

from trader_koo.paper_trade.schema import ensure_paper_trade_schema
from trader_koo.schemas import PaperTrade, PriceBar
from trader_koo.scripts.update_market_db import ensure_schema
from trader_koo.signals.types import SignalOutput


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def test_price_bar_fields_match_price_daily_columns():
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    assert set(PriceBar.model_fields) == _columns(conn, "price_daily")


def test_paper_trade_fields_exist_in_paper_trades_table():
    conn = sqlite3.connect(":memory:")
    ensure_paper_trade_schema(conn)
    assert set(PaperTrade.model_fields) <= _columns(conn, "paper_trades")


def test_price_bar_validates_a_real_price_daily_row():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    conn.execute(
        "INSERT INTO price_daily (ticker,date,open,high,low,close,volume) "
        "VALUES ('AAPL','2026-03-13',1,2,0.5,1.5,1000)"
    )
    row = conn.execute("SELECT * FROM price_daily").fetchone()
    bar = PriceBar.model_validate(dict(row))
    assert bar.close == 1.5
    assert bar.basis_status == "unverified"


@pytest.mark.parametrize(
    "overrides",
    [{"direction": "buy"}, {"status": "pending"}, {"entry_price": "n/a"}],
)
def test_paper_trade_rejects_values_the_table_forbids(overrides):
    row = {
        "id": 1, "report_date": "2026-03-13", "ticker": "AAPL",
        "direction": "long", "entry_price": 150.0, "entry_date": "2026-03-13",
        "status": "open", "campaign_id": "paper-v2",
    }
    with pytest.raises(ValidationError):
        PaperTrade.model_validate({**row, **overrides})


def test_signal_output_keeps_positional_construction_and_is_frozen():
    signal = SignalOutput("trend", "bullish", 75.0, "EMA12>EMA26", 0.25)
    assert signal.weight == 0.25
    with pytest.raises(dataclasses.FrozenInstanceError):
        signal.bias = "bearish"  # type: ignore[misc]


@pytest.mark.parametrize(
    "args",
    [
        ("trend", "up", 50.0, "r"),
        ("trend", "bullish", 101.0, "r"),
        ("trend", "bullish", -1.0, "r"),
        ("trend", "bullish", 50.0, "r", -0.5),
    ],
)
def test_signal_output_rejects_out_of_contract_values(args):
    with pytest.raises(ValidationError):
        SignalOutput(*args)
