from __future__ import annotations

import math
import sqlite3

import pandas as pd
import pytest

from trader_koo.db.total_return import (
    TR_VERSION,
    load_total_return_inputs,
    total_return_quality_report,
    total_return_series,
)


def _prices(rows, ticker="AAA"):
    return pd.DataFrame(
        [{"ticker": ticker, "date": d, "close": c} for d, c in rows]
    )


def _actions(rows, ticker="AAA"):
    return pd.DataFrame(
        [{"ticker": ticker, "action_date": d, "action_type": t, "value": v} for d, t, v in rows],
        columns=["ticker", "action_date", "action_type", "value"],
    )


def test_dividend_on_exdate_adds_to_return():
    p = _prices([("2024-01-02", 100.0), ("2024-01-03", 99.0), ("2024-01-04", 101.0)])
    a = _actions([("2024-01-03", "dividend", 2.0)])
    out = total_return_series(p, a).set_index("date")
    assert out.loc["2024-01-03", "total_return"] == pytest.approx((99 + 2) / 100 - 1)
    assert out.loc["2024-01-04", "total_return"] == pytest.approx(101 / 99 - 1)
    assert out.loc["2024-01-04", "tr_index"] == pytest.approx(1.01 * 101 / 99)
    assert out.loc["2024-01-02", "tr_index"] == 1.0
    assert out["flag"].eq("").all()


def test_split_day_with_adjusted_closes_is_plain_return():
    p = _prices([("2024-01-02", 50.0), ("2024-01-03", 51.0)])
    a = _actions([("2024-01-03", "split", 2.0)])
    out = total_return_series(p, a)
    assert out["total_return"].iloc[1] == pytest.approx(0.02)


def test_unadjusted_split_jump_is_flagged_and_nan():
    p = _prices([("2024-01-02", 100.0), ("2024-01-03", 50.0), ("2024-01-04", 51.0)])
    a = _actions([("2024-01-03", "split", 2.0)])
    out = total_return_series(p, a).set_index("date")
    assert math.isnan(out.loc["2024-01-03", "total_return"])
    assert out.loc["2024-01-03", "flag"] == "move_on_split_date"
    assert out.loc["2024-01-04", "total_return"] == pytest.approx(0.02)
    assert out.loc["2024-01-04", "segment"] == 1


def test_unexplained_large_move_flagged():
    p = _prices([("2024-01-02", 100.0), ("2024-01-03", 40.0)])
    out = total_return_series(p, _actions([]))
    assert out["flag"].iloc[1] == "unexplained_move"
    assert math.isnan(out["total_return"].iloc[1])


def test_special_dividend_flagged_not_used():
    p = _prices([("2024-01-02", 20.0), ("2024-01-03", 18.0), ("2024-01-04", 18.9)])
    a = _actions([("2024-01-03", "dividend", 5.0)])
    out = total_return_series(p, a).set_index("date")
    assert out.loc["2024-01-03", "flag"] == "special_dividend"
    assert math.isnan(out.loc["2024-01-03", "total_return"])
    assert out.loc["2024-01-04", "total_return"] == pytest.approx(0.05)


def test_missing_or_nonpositive_close_fails_closed_around_the_gap():
    p = _prices([("2024-01-02", 10.0), ("2024-01-03", None), ("2024-01-04", 11.0), ("2024-01-05", 0.0), ("2024-01-08", 12.0)])
    out = total_return_series(p, _actions([])).set_index("date")
    assert out.loc["2024-01-03", "flag"] == "bad_close"
    assert out.loc["2024-01-05", "flag"] == "bad_close"
    assert math.isnan(out.loc["2024-01-04", "total_return"])
    assert math.isnan(out.loc["2024-01-08", "total_return"])


def test_dividend_off_calendar_flags_next_trading_day():
    p = _prices([("2024-01-02", 100.0), ("2024-01-05", 101.0), ("2024-01-08", 102.0)])
    a = _actions([("2024-01-06", "dividend", 1.0)])
    out = total_return_series(p, a).set_index("date")
    assert out.loc["2024-01-08", "flag"] == "dividend_off_calendar"
    assert math.isnan(out.loc["2024-01-08", "total_return"])


def test_tickers_independent_and_version_and_hash():
    p = pd.concat([_prices([("2024-01-02", 10.0), ("2024-01-03", 11.0)], "A"),
                   _prices([("2024-01-02", 20.0), ("2024-01-03", 20.0)], "B")])
    out = total_return_series(p, _actions([]))
    assert out.attrs["version"] == TR_VERSION
    h = out.attrs["input_sha256"]
    assert len(h) == 64
    assert total_return_series(p, _actions([])).attrs["input_sha256"] == h
    p2 = p.copy()
    p2.loc[p2.index[0], "close"] = 10.5
    assert total_return_series(p2, _actions([])).attrs["input_sha256"] != h
    assert out.groupby("ticker")["tr_index"].first().eq(1.0).all()


def test_quality_report_lists_flags():
    p = _prices([("2024-01-02", 100.0), ("2024-01-03", 40.0)])
    rep = total_return_quality_report(total_return_series(p, _actions([])))
    assert rep.to_dict("records") == [
        {"ticker": "AAA", "date": "2024-01-03", "reason": "unexplained_move"}
    ]


def test_loader_excludes_total_return_basis_rows_and_is_read_only(tmp_path):
    db = tmp_path / "t.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE price_daily (ticker TEXT, date TEXT, close REAL, adjustment_basis TEXT)")
    c.execute("CREATE TABLE price_corporate_actions (ticker TEXT, action_date TEXT, action_type TEXT, provider TEXT, value REAL)")
    c.executemany("INSERT INTO price_daily VALUES (?,?,?,?)", [
        ("A", "2024-01-02", 10.0, "split_adjusted_price_only"),
        ("A", "2024-01-03", 10.5, "split_adjusted_price_only"),
        ("B", "2024-01-02", 10.0, "total_return"),
        ("^VIX", "2024-01-02", 14.0, "split_adjusted_price_only"),
    ])
    c.execute("INSERT INTO price_corporate_actions VALUES ('A','2024-01-03','dividend','yfinance',0.1)")
    c.commit()
    c.close()
    prices, actions = load_total_return_inputs(db)
    assert set(prices["ticker"]) == {"A"}
    assert len(actions) == 1
    ro = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("DELETE FROM price_daily")


def test_special_dividend_with_price_collapse_reports_special_dividend():
    p = _prices([("2024-01-02", 120.0), ("2024-01-03", 22.0)])
    a = _actions([("2024-01-03", "dividend", 100.0)])
    out = total_return_series(p, a)
    assert out["flag"].iloc[1] == "special_dividend"
    assert math.isnan(out["total_return"].iloc[1])


def test_dividend_after_the_last_price_and_unknown_action_types_are_reported():
    p = _prices([("2024-01-02", 100.0), ("2024-01-03", 101.0)])
    a = _actions([("2024-01-10", "dividend", 1.0), ("2024-01-03", "spinoff", 5.0)])
    out = total_return_series(p, a)
    assert out.attrs["dividends_after_last_price"] == [("AAA", "2024-01-10", 1.0)]
    assert out.attrs["ignored_action_types"] == {"spinoff": 1}
    assert out["total_return"].iloc[1] == pytest.approx(0.01)  # neither changed the return
