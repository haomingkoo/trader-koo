"""Triple-barrier label resolution.

These tests exercise `generate_triple_barrier_labels` against the real
`price_daily` schema and the real code path. The OHLC rows are constructed to
put a bar on a specific side of the barriers, because the property under test
is the resolution ORDER of a bar, not the behaviour of any real instrument.

The ambiguous-bar case is the one that matters: a single daily bar whose high
reaches the profit target AND whose low reaches the stop. Daily OHLC cannot say
which was touched first, so the label must resolve stop-first, matching
`research/next_open_baseline.py:_resolve_bar`. Resolving target-first labels
every such bar a win and inflates the positive class precisely in the
high-volatility bars where both barriers are reachable.
"""
from __future__ import annotations

import math
import sqlite3

import pandas as pd

from trader_koo.ml.labels import generate_triple_barrier_labels

PRICE_DAILY_DDL = """
CREATE TABLE IF NOT EXISTS price_daily (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    data_source TEXT DEFAULT 'yfinance',
    fetch_timestamp TEXT,
    adjustment_basis TEXT,
    adjustment_version TEXT,
    basis_status TEXT DEFAULT 'unverified',
    unresolved_reason TEXT,
    PRIMARY KEY (ticker, date)
)
"""

TICKER = "AAPL"
_BASE_CLOSE = 100.0
_LEAD_IN_BARS = 40


def _lead_in_close(idx: int) -> float:
    """A gently oscillating series, so rolling log-return vol is finite and > 0."""
    return _BASE_CLOSE * (1.0 + 0.01 * math.sin(idx / 3.0))


def _build_conn(*, event_high_mult: float, event_low_mult: float) -> tuple[sqlite3.Connection, str]:
    """Seed price_daily with a lead-in series, an entry bar, then one event bar.

    The event bar's high/low are expressed as multiples of the entry close so a
    test can place it above the target, below the stop, or across both.
    Returns the connection and the entry date.
    """
    conn = sqlite3.connect(":memory:")
    conn.execute(PRICE_DAILY_DDL)

    dates = pd.date_range("2026-01-01", periods=_LEAD_IN_BARS + 2, freq="D")
    rows = []
    for idx in range(_LEAD_IN_BARS):
        close = _lead_in_close(idx)
        rows.append((TICKER, dates[idx].strftime("%Y-%m-%d"), close, close * 1.002, close * 0.998, close))

    entry_idx = _LEAD_IN_BARS
    entry_close = _BASE_CLOSE
    rows.append(
        (TICKER, dates[entry_idx].strftime("%Y-%m-%d"), entry_close, entry_close * 1.002, entry_close * 0.998, entry_close)
    )

    event_idx = _LEAD_IN_BARS + 1
    rows.append(
        (
            TICKER,
            dates[event_idx].strftime("%Y-%m-%d"),
            entry_close,
            entry_close * event_high_mult,
            entry_close * event_low_mult,
            entry_close,
        )
    )

    conn.executemany(
        "INSERT INTO price_daily (ticker, date, open, high, low, close) VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return conn, dates[entry_idx].strftime("%Y-%m-%d")


def _label_row(*, event_high_mult: float, event_low_mult: float) -> dict:
    conn, entry_date = _build_conn(event_high_mult=event_high_mult, event_low_mult=event_low_mult)
    try:
        labels = generate_triple_barrier_labels(
            conn, entry_dates=[entry_date], tickers=[TICKER], max_holding_days=10
        )
    finally:
        conn.close()

    assert not labels.empty, "expected one labelled row for the seeded entry date"
    return labels.iloc[0].to_dict()


def test_bar_touching_both_barriers_is_labelled_a_loss():
    """The ambiguous bar must resolve stop-first, not target-first."""
    row = _label_row(event_high_mult=1.10, event_low_mult=0.90)

    assert row["label"] == -1
    assert row["exit_reason"] == "stop_loss"
    assert row["return_pct"] < 0


def test_bar_touching_only_the_target_is_labelled_a_win():
    row = _label_row(event_high_mult=1.10, event_low_mult=0.999)

    assert row["label"] == 1
    assert row["exit_reason"] == "profit_target"
    assert row["return_pct"] > 0


def test_bar_touching_only_the_stop_is_labelled_a_loss():
    row = _label_row(event_high_mult=1.001, event_low_mult=0.90)

    assert row["label"] == -1
    assert row["exit_reason"] == "stop_loss"
    assert row["return_pct"] < 0


def test_bar_touching_neither_barrier_expires_flat():
    row = _label_row(event_high_mult=1.001, event_low_mult=0.999)

    assert row["label"] == 0
    assert row["exit_reason"] == "time_expired"
