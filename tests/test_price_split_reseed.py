"""Split detection and safe price-history reseeding."""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from trader_koo.scripts.update_market_db import (
    stored_closes_disagree,
    stored_prices_have_scale_break,
)

DATES = ["2026-03-16", "2026-03-17", "2026-03-18"]
PRE_SPLIT = [1438.24, 1481.35, 1482.36]


@pytest.fixture
def conn() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE price_daily (ticker TEXT, date TEXT, close REAL)")
    connection.executemany(
        "INSERT INTO price_daily (ticker, date, close) VALUES ('KLAC', ?, ?)",
        list(zip(DATES, PRE_SPLIT)),
    )
    return connection


def fetched(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": DATES, "close": closes})


@pytest.mark.parametrize("factor", [8, 2, 1.5])
def test_detects_rebased_history_after_split(
    conn: sqlite3.Connection,
    factor: float,
) -> None:
    adjusted = [round(close / factor, 2) for close in PRE_SPLIT]
    assert stored_closes_disagree(conn, "KLAC", fetched(adjusted)) is True


def test_unchanged_or_new_history_is_not_rebased(conn: sqlite3.Connection) -> None:
    assert stored_closes_disagree(conn, "KLAC", fetched(PRE_SPLIT)) is False
    assert stored_closes_disagree(
        conn,
        "KLAC",
        pd.DataFrame({"date": ["2026-03-19"], "close": [1495.10]}),
    ) is False


def test_detects_legacy_mixed_scale_after_split(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM price_daily")
    conn.executemany(
        "INSERT INTO price_daily (ticker, date, close) VALUES ('BKNG', ?, ?)",
        [
            ("2026-04-01", 4184.56),
            ("2026-04-02", 4194.31),
            ("2026-04-06", 176.19),
            ("2026-04-07", 173.41),
        ],
    )
    assert stored_prices_have_scale_break(conn, "BKNG") is True


def test_ordinary_price_moves_do_not_look_like_scale_breaks(conn: sqlite3.Connection) -> None:
    assert stored_prices_have_scale_break(conn, "KLAC") is False


def test_real_crash_prompts_check_but_does_not_reseed(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM price_daily")
    crash = pd.DataFrame(
        {
            "date": ["2026-03-16", "2026-03-17"],
            "close": [100.0, 45.0],
        }
    )
    conn.executemany(
        "INSERT INTO price_daily (ticker, date, close) VALUES ('CRASH', ?, ?)",
        list(crash.itertuples(index=False, name=None)),
    )

    assert stored_prices_have_scale_break(conn, "CRASH") is True
    assert stored_closes_disagree(conn, "CRASH", crash) is False


def test_full_history_detects_old_rebase_outside_incremental_overlap(
    conn: sqlite3.Connection,
) -> None:
    full_history = pd.DataFrame(
        {
            "date": [*DATES, "2026-08-21"],
            "close": [*(round(close / 8, 2) for close in PRE_SPLIT), 190.0],
        }
    )
    recent_only = full_history.tail(1)

    assert stored_closes_disagree(conn, "KLAC", recent_only) is False
    assert stored_closes_disagree(conn, "KLAC", full_history) is True
