"""Split detection and safe price-history reseeding."""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from trader_koo.db.sources import DataSourceManager
from trader_koo.scripts.update_market_db import (
    corporate_actions_require_full_history,
    ensure_schema,
    mark_full_history_actions_verified,
    stored_closes_disagree,
    stored_prices_have_scale_break,
    write_price_daily,
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


@pytest.mark.parametrize("closes", [[150.0, 100.0], [100.0, 150.0]])
def test_stored_three_for_two_break_prompts_full_history_check(
    conn: sqlite3.Connection,
    closes: list[float],
) -> None:
    conn.execute("DELETE FROM price_daily")
    conn.executemany(
        "INSERT INTO price_daily (ticker, date, close) VALUES ('THREEFOR2', ?, ?)",
        list(zip(DATES[:2], closes)),
    )

    assert stored_prices_have_scale_break(conn, "THREEFOR2") is True


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


def test_retroactive_action_requires_one_idempotent_full_history_reconciliation() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    continuous = pd.DataFrame(
        {
            "Open": [99.0, 100.0, 101.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [98.0, 99.0, 100.0],
            "Close": [100.0, 101.0, 102.0],
            "Volume": [1000.0, 1000.0, 1000.0],
        },
        index=pd.DatetimeIndex(DATES, name="Date"),
    )
    raw = continuous.copy()
    raw.loc[raw.index < pd.Timestamp(DATES[1]), ["Open", "High", "Low", "Close"]] *= 1.5
    raw.loc[raw.index < pd.Timestamp(DATES[1]), "Volume"] /= 1.5
    raw["Adj Close"] = continuous["Close"]
    raw["Stock Splits"] = [0.0, 1.5, 0.0]
    normalized = DataSourceManager._normalize_ohlcv(raw)
    vendor_action = [
        {
            "action_date": DATES[1],
            "action_type": "split",
            "value": 1.5,
        }
    ]

    assert corporate_actions_require_full_history(
        conn,
        "LATE",
        vendor_action,
        managed_start=DATES[0],
        managed_end="2026-03-19",
    ) is True

    mark_full_history_actions_verified(normalized)
    write_price_daily(conn, "LATE", normalized, fetch_timestamp="2026-08-22T00:00:00Z")
    write_price_daily(conn, "LATE", normalized, fetch_timestamp="2026-08-22T00:00:00Z")
    conn.commit()

    assert corporate_actions_require_full_history(
        conn,
        "LATE",
        vendor_action,
        managed_start=DATES[0],
        managed_end="2026-03-19",
    ) is False
    action = conn.execute(
        """SELECT COUNT(*), evidence_json FROM price_corporate_actions
        WHERE ticker='LATE' AND action_type='split'"""
    ).fetchone()
    assert action[0] == 1
    assert '"full_history_verified": true' in action[1]


def test_pre_start_vendor_action_does_not_repeat_bounded_history_reseed() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    vendor_action = [
        {"action_date": "2017-12-31", "action_type": "split", "value": 2.0}
    ]

    # First run sees Yahoo's lifetime ledger but manages only 2018 onward.
    assert corporate_actions_require_full_history(
        conn,
        "OLD",
        vendor_action,
        managed_start="2018-01-01",
    ) is False
    bounded = DataSourceManager._normalize_ohlcv(
        _already_adjusted_action_frame(action_date=DATES[1], factor=1.0)
    )
    mark_full_history_actions_verified(bounded)
    write_price_daily(conn, "OLD", bounded)
    conn.commit()

    # The next incremental run must make the same decision without reseeding.
    assert corporate_actions_require_full_history(
        conn,
        "OLD",
        vendor_action,
        managed_start="2018-01-01",
    ) is False


def test_managed_action_window_uses_exact_half_open_boundaries() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)

    def requires(action_date: str) -> bool:
        return corporate_actions_require_full_history(
            conn,
            "BOUNDARY",
            [{"action_date": action_date, "action_type": "split", "value": 1.5}],
            managed_start="2018-01-01",
            managed_end="2020-01-01",
        )

    assert requires("2017-12-31") is False
    assert requires("2018-01-01") is True
    assert requires("2019-12-31") is True
    assert requires("2020-01-01") is False


def test_earlier_managed_start_makes_previously_irrelevant_action_relevant() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    vendor_action = [
        {"action_date": "2017-06-01", "action_type": "split", "value": 1.5}
    ]

    assert corporate_actions_require_full_history(
        conn, "CONFIG", vendor_action, managed_start="2018-01-01"
    ) is False
    assert corporate_actions_require_full_history(
        conn, "CONFIG", vendor_action, managed_start="2017-01-01"
    ) is True


def _already_adjusted_action_frame(*, action_date: str, factor: float) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Open": [99.0, 100.0, 101.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [98.0, 99.0, 100.0],
            "Close": [100.0, 101.0, 102.0],
            "Adj Close": [100.0, 101.0, 102.0],
            "Volume": [1000.0, 1000.0, 1000.0],
            "Stock Splits": [0.0, 0.0, 0.0],
        },
        index=pd.DatetimeIndex(DATES, name="Date"),
    )
    frame.loc[pd.Timestamp(action_date), "Stock Splits"] = factor
    return frame


def test_already_adjusted_full_history_verification_survives_compatible_narrow_replay() -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    full = DataSourceManager._normalize_ohlcv(
        _already_adjusted_action_frame(action_date=DATES[1], factor=1.5)
    )
    assert full.attrs["corporate_actions"][0]["applied_to_prices"] is False
    mark_full_history_actions_verified(full)
    write_price_daily(conn, "ADJUSTED", full, fetch_timestamp="2026-08-20T00:00:00Z")

    narrow_raw = _already_adjusted_action_frame(action_date=DATES[1], factor=1.5).iloc[1:]
    narrow = DataSourceManager._normalize_ohlcv(narrow_raw)
    assert narrow.attrs["corporate_actions"][0]["applied_to_prices"] is False
    assert "full_history_verified" not in narrow.attrs["corporate_actions"][0]
    write_price_daily(conn, "ADJUSTED", narrow, fetch_timestamp="2026-08-21T00:00:00Z")
    conn.commit()

    vendor_action = [
        {"action_date": DATES[1], "action_type": "split", "value": 1.5}
    ]
    assert corporate_actions_require_full_history(conn, "ADJUSTED", vendor_action) is False
    evidence_json = conn.execute(
        """SELECT evidence_json FROM price_corporate_actions
        WHERE ticker='ADJUSTED' AND action_date=? AND action_type='split'
          AND provider='yfinance'""",
        (DATES[1],),
    ).fetchone()[0]
    assert json.loads(evidence_json)["full_history_verified"] is True


@pytest.mark.parametrize("mismatch", ["factor", "date", "source"])
def test_full_history_verification_is_not_inherited_by_mismatched_action(
    mismatch: str,
) -> None:
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    full = DataSourceManager._normalize_ohlcv(
        _already_adjusted_action_frame(action_date=DATES[1], factor=1.5)
    )
    mark_full_history_actions_verified(full)
    initial_source = "other" if mismatch == "source" else "yfinance"
    write_price_daily(conn, "MISMATCH", full, data_source=initial_source)

    action_date = DATES[2] if mismatch == "date" else DATES[1]
    factor = 2.0 if mismatch == "factor" else 1.5
    replay = DataSourceManager._normalize_ohlcv(
        _already_adjusted_action_frame(action_date=action_date, factor=factor).loc[
            action_date:
        ]
    )
    write_price_daily(conn, "MISMATCH", replay, data_source="yfinance")
    conn.commit()

    evidence_json = conn.execute(
        """SELECT evidence_json FROM price_corporate_actions
        WHERE ticker='MISMATCH' AND action_date=? AND action_type='split'
          AND provider='yfinance'""",
        (action_date,),
    ).fetchone()[0]
    assert "full_history_verified" not in json.loads(evidence_json)
    assert corporate_actions_require_full_history(
        conn,
        "MISMATCH",
        [{"action_date": action_date, "action_type": "split", "value": factor}],
    ) is True
