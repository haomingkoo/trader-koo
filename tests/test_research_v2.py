"""Hypotheses v2 (S1-S3): look-ahead guard, trades against plain-loop references on real rows, sealed window untouched.

The fixture is real daily bars of eight S&P 500 stocks and SPY, 2017-01-03 to 2019-06-28.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trader_koo.db.membership import ensure_membership_schema
from trader_koo.research.bench import ledger, v2_run, v2_signals, v2_trades
from trader_koo.research.bench.data import load_market
from trader_koo.research.settings import get as settings

FIX = Path(__file__).parent / "fixtures" / "research_bench"
COST = settings().bench.cost_bps / 1e4


def _db(path: Path, scale_after: str | None = None) -> str:
    px = pd.read_csv(FIX / "ohlc_2017_2019.csv.gz")
    if scale_after:  # corrupt every row on or after a date: the sealed window in a test
        later = px["date"] >= scale_after
        px.loc[later, ["open", "high", "low", "close"]] *= 3.0
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE price_daily (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,"
                 " adjustment_basis TEXT)")
    conn.execute("CREATE TABLE price_corporate_actions (ticker TEXT, action_date TEXT, action_type TEXT, value REAL)")
    conn.execute("CREATE TABLE price_series_revisions (ticker TEXT, evidence_json TEXT)")
    px.assign(adjustment_basis=None).to_sql("price_daily", conn, if_exists="append", index=False)
    pd.read_csv(FIX / "ohlc_actions_2017_2019.csv.gz").to_sql("price_corporate_actions", conn, if_exists="append", index=False)
    ensure_membership_schema(conn)
    conn.executemany("INSERT INTO sp500_membership VALUES (?,?,?,?,?)",
                     [(t, "2000-01-01", None, "t", "t") for t in sorted(set(px.ticker) - {"SPY"})])
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture(scope="module")
def db(tmp_path_factory) -> str:
    return _db(tmp_path_factory.mktemp("v2") / "r.db")


@pytest.fixture(scope="module")
def market(db):
    return load_market(db)


@pytest.fixture(scope="module")
def prices(market):
    p, last, counts = v2_run.load_prices(market, "2100-01-01")
    assert last == len(market.sessions) - 1 and counts["high_low_rows_nonpositive_or_crossed_set_to_missing"] == 0
    return p


def test_no_fixture_ticker_is_a_spin_off_parent(market):
    assert not market.exits  # the references below would otherwise need the spin-off exit rule


# ---- look-ahead guard --------------------------------------------------------------------------------------------------

def _corrupt(p: v2_signals.Prices, t: int, seed: int = 0) -> v2_signals.Prices:
    rng = np.random.default_rng(seed)
    out = []
    for name in ("open", "high", "low", "close"):
        df = getattr(p, name).copy()
        df.iloc[t + 1:] = df.iloc[t + 1:].to_numpy() * rng.uniform(0.2, 5.0, size=df.iloc[t + 1:].shape)
        out.append(df)
    return v2_signals.Prices(*out, p.member)


@pytest.mark.parametrize("t", [260, 400, 520])
def test_signals_up_to_t_do_not_change_when_later_prices_are_corrupted(prices, t):
    a, b = v2_signals.s1(prices), v2_signals.s1(_corrupt(prices, t))
    for k in ("signal", "state", "exit_ema"):
        assert np.allclose(a[k].iloc[:t + 1].to_numpy(float), b[k].iloc[:t + 1].to_numpy(float), equal_nan=True), k
    a, b = v2_signals.s2(prices), v2_signals.s2(_corrupt(prices, t))
    assert a["cross"].iloc[:t + 1].equals(b["cross"].iloc[:t + 1])
    assert np.allclose(a["exit_ema"].iloc[:t + 1].to_numpy(), b["exit_ema"].iloc[:t + 1].to_numpy(), equal_nan=True)
    # a retest on or before t is found from rows up to t only; crosses whose window runs past t may differ
    early = np.flatnonzero(np.any(a["cross"].to_numpy()[: t - 10 + 1], axis=1))
    assert np.array_equal(a["retest_day"][:t - 10 + 1][early], b["retest_day"][:t - 10 + 1][early])
    a, b = v2_signals.s3(prices), v2_signals.s3(_corrupt(prices, t))
    for k in ("signal", "state"):
        assert a[k].iloc[:t + 1].equals(b[k].iloc[:t + 1]), k


def test_the_guard_would_catch_a_signal_that_looks_one_row_ahead(prices):
    p = prices

    def peeking(q: v2_signals.Prices):
        return q.close.shift(-1) > q.close

    assert not peeking(p).iloc[:301].equals(peeking(_corrupt(p, 300)).iloc[:301])


# ---- references: plain loops over the same adjusted prices ----------------------------------------------------------------

def _ewm(x, alpha, minp):
    out, prev, seen = [], None, 0
    for v in x:
        if v == v:
            prev = v if prev is None else prev + alpha * (v - prev)
            seen += 1
        out.append(prev if seen >= minp else np.nan)
    return out


def _ema(x, n):
    return _ewm(x, 2.0 / (n + 1), n)


def _trade(p, col, e, x, cost=COST):
    return round(p.open.iloc[x, col] / p.open.iloc[e, col] * (1 - cost) / (1 + cost) - 1.0, 12)


def _first_below(c, ema, e, hold):
    for s in range(e, min(e + hold, len(c))):
        if c[s] < ema[s]:
            return s + 1
    return e + hold


def _ref_s1(p):
    n, trades = len(p.close), []
    for col in range(p.close.shape[1]):
        c, lo = p.close.iloc[:, col].tolist(), p.low.iloc[:, col].tolist()
        ema = {k: _ema(c, k) for k in (8, 21, 48, 200)}
        busy = -1
        for t in range(n - 1):
            if not p.member.iloc[t, col] or any(ema[k][t] != ema[k][t] for k in ema):
                continue
            stack = ema[8][t] > ema[21][t] > ema[48][t] > ema[200][t] and c[t] > ema[200][t]
            if stack and lo[t] <= ema[21][t] and c[t] >= ema[21][t]:
                e = t + 1
                if e < busy:
                    continue
                x = _first_below(c, ema[48], e, 20) if e + 20 <= n - 1 else None
                busy = (_first_below(c, ema[48], e, 20) if x is None else x)
                if x is not None and np.isfinite(p.open.iloc[e, col]) and np.isfinite(p.open.iloc[x, col]):
                    trades.append((e, col, _trade(p, col, e, x)))
    return sorted(trades)


def _ref_s2(p):
    n, trades = len(p.close), []
    for col in range(p.close.shape[1]):
        c, lo = p.close.iloc[:, col].tolist(), p.low.iloc[:, col].tolist()
        e8, e13, e21 = _ema(c, 8), _ema(c, 13), _ema(c, 21)
        for d in range(1, n):
            if not p.member.iloc[d, col] or any(v != v for v in (e8[d], e13[d], e21[d], e8[d - 1], e13[d - 1])):
                continue
            if e8[d] > e13[d] and e8[d - 1] <= e13[d - 1] and c[d] > e21[d]:
                for t in range(d + 1, min(d + 10, n - 1) + 1):
                    if lo[t] <= e13[t] and c[t] > e13[t]:
                        if p.member.iloc[t, col] and t + 1 < n and t + 1 + 20 <= n - 1:
                            e = t + 1
                            x = _first_below(c, e21, e, 20)
                            if np.isfinite(p.open.iloc[e, col]) and np.isfinite(p.open.iloc[x, col]):
                                trades.append((e, col, _trade(p, col, e, x)))
                        break
    return sorted(trades)


def _ref_s3(p):
    n, trades = len(p.close), []
    for col in range(p.close.shape[1]):
        c, h, lo = p.close.iloc[:, col].to_numpy(), p.high.iloc[:, col].to_numpy(), p.low.iloc[:, col].to_numpy()
        tr = [np.nan] + [max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])) for i in range(1, n)]
        atr = _ewm(tr, 1 / 14, 14)
        e21, e48 = _ema(c.tolist(), 21), _ema(c.tolist(), 48)
        comp = [False] * n
        for i in range(20, n):
            w = c[i - 20:i + 1]
            if atr[i] == atr[i]:
                comp[i] = w.mean() + 2 * w.std() <= w.mean() + 2 * atr[i] and w.mean() - 2 * w.std() >= w.mean() - 2 * atr[i]
        for t in range(21, n - 1):
            if p.member.iloc[t, col] and not comp[t] and comp[t - 1] and atr[t] == atr[t] and e21[t] > e48[t]:
                e = t + 1
                if e + 10 <= n - 1 and np.isfinite(p.open.iloc[e, col]) and np.isfinite(p.open.iloc[e + 10, col]):
                    trades.append((e, col, _trade(p, col, e, e + 10)))
    return sorted(trades)


def _from_grid(w, grid, p):
    ok = np.isfinite(grid.gross) & (w > 0)
    net = v2_trades.net(grid.gross, settings().bench.cost_bps)
    rows, cols = np.nonzero(ok)
    out = []
    for r, c in zip(rows, cols):
        out += [(int(r), int(c), round(float(net[r, c]), 12))] * int(w[r, c])
    return sorted(out)


def _weights(market, p):
    shape = p.close.shape
    a1, a2, a3 = v2_signals.s1(p), v2_signals.s2(p), v2_signals.s3(p)
    g1 = v2_trades.build_grid(market, p.open, p.close, a1["exit_ema"], v2_signals.S1_MAX_HOLD)
    g2 = v2_trades.build_grid(market, p.open, p.close, a2["exit_ema"], v2_signals.S2_MAX_HOLD)
    g3 = v2_trades.build_grid(market, p.open, p.close, None, v2_signals.S3_HOLD)
    cells = v2_trades.entry_counts(*np.nonzero(a1["signal"].to_numpy()), shape) > 0
    w1 = v2_trades.one_at_a_time(cells, g1).astype(int)
    cr, cc = np.nonzero(a2["cross"].to_numpy())
    t = a2["retest_day"][cr, cc]
    use = (t >= 0) & p.member.to_numpy()[np.maximum(t, 0), cc]
    w2 = v2_trades.entry_counts(t[use], cc[use], shape)
    w3 = v2_trades.entry_counts(*np.nonzero(a3["signal"].to_numpy()), shape)
    return (w1, g1), (w2, g2), (w3, g3)


@pytest.fixture(scope="module")
def weighted(market, prices):
    return _weights(market, prices)


def test_s1_trades_match_a_plain_loop(prices, weighted):
    got, ref = _from_grid(*weighted[0], prices), _ref_s1(prices)
    assert got == ref and len(ref) > 5


def test_s2_trades_match_a_plain_loop(prices, weighted):
    got, ref = _from_grid(*weighted[1], prices), _ref_s2(prices)
    assert got == ref and len(ref) > 5


def test_s3_trades_match_a_plain_loop(prices, weighted):
    got, ref = _from_grid(*weighted[2], prices), _ref_s3(prices)
    assert got == ref and len(ref) > 5


def test_s1_has_one_open_trade_per_stock(prices, weighted):
    w, grid = weighted[0]
    for col in range(w.shape[1]):
        rows = np.flatnonzero(w[:, col])
        for a, b in zip(rows, rows[1:]):
            assert b >= grid.exit_row[a, col]


def test_s1_first_trade_hand_check(prices, weighted):
    """AAPL's first S1 entry on real rows: signal at the close of the day before, exit rule applied by hand."""
    w, grid = weighted[0]
    rows, cols = np.nonzero(w)
    i = int(np.argmin(rows))
    e, col = int(rows[i]), int(cols[i])
    t = e - 1
    c = prices.close.iloc[:, col]
    e21, e48 = v2_signals.ema(prices.close, 21).iloc[:, col], v2_signals.ema(prices.close, 48).iloc[:, col]
    assert prices.low.iloc[t, col] <= e21.iloc[t] <= c.iloc[t]  # dip to the 21 EMA, close back above it
    x = int(grid.exit_row[e, col])
    assert x == e + 20 or c.iloc[x - 1] < e48.iloc[x - 1] and (c.iloc[e:x - 1] >= e48.iloc[e:x - 1]).all()
    assert grid.gross[e, col] == pytest.approx(prices.open.iloc[x, col] / prices.open.iloc[e, col])


# ---- exits, fallbacks, costs ------------------------------------------------------------------------------------------------

def _mini(open_, close, ema=None, exits=None, hold=3):
    o, c = pd.DataFrame(open_), pd.DataFrame(close)
    e = None if ema is None else pd.DataFrame(ema)
    return v2_trades.build_grid(SimpleNamespace(exits=exits or {}), o, c, e, hold), o


def test_exit_at_first_close_below_the_average_else_at_the_maximum_hold():
    # entry at row 1 (open 10). Closes under the average on row 2 -> exit at the open of row 3 (12)
    g, _ = _mini({0: [9, 10, 10, 12, 12, 12, 12]}, {0: [10, 10, 9, 11, 12, 12, 12]}, {0: [9.5] * 7}, hold=4)
    assert g.exit_row[1, 0] == 3 and g.gross[1, 0] == pytest.approx(12 / 10)
    # no close below: exit at the open of row e + hold = 5
    g, _ = _mini({0: [9, 10, 10, 11, 11, 13, 13]}, {0: [10, 10, 10, 10, 10, 10, 10]}, {0: [5.0] * 7}, hold=4)
    assert g.exit_row[1, 0] == 5 and g.gross[1, 0] == pytest.approx(13 / 10)
    assert g.late[3, 0] and g.late[4, 0] and not g.late[2, 0]  # e + 4 must be inside the window (last row 6)
    assert np.isnan(g.gross[3, 0])


def test_missing_exit_open_uses_the_next_valid_open_then_the_last_close_and_is_flagged():
    nan = np.nan
    g, _ = _mini({0: [9, 10, 10, nan, 12, 12, 12]}, {0: [10] * 7}, None, hold=2)  # exit at row 3, open missing -> row 4
    assert g.fallback_next_open[1, 0] and g.gross[1, 0] == pytest.approx(1.2)
    g, _ = _mini({0: [9, 10, 10, 10, nan, nan, nan]}, {0: [10, 10, 10, 10, 10, 11, 12]}, None, hold=2)
    # rows 0-3 enter and exit inside the window; the exit open at row 3 exists. Move the window so the exit open is missing:
    g, _ = _mini({0: [9, 10, 10, nan, nan, nan, nan]}, {0: [10, 10, 10, 10, 10, 11, 12]}, None, hold=2)
    assert g.fallback_last_close[1, 0] and g.gross[1, 0] == pytest.approx(10.0 / 10)


def test_spinoff_inside_the_hold_exits_at_the_close_before_the_ex_date():
    g, _ = _mini({0: [10, 10, 10, 10, 10, 20, 20, 20]}, {0: [10, 10, 10, 10, 11, 11, 11, 11]}, None, exits={0: [3]}, hold=4)
    assert g.spinoff_exit[1, 0] and g.exit_row[1, 0] == 3 and g.gross[1, 0] == pytest.approx(10 / 10)
    assert not g.spinoff_exit[4, 0]  # entered after the ex-date


def test_cost_is_charged_on_both_sides():
    c = 0.001
    assert v2_trades.net(np.array([1.0]), 10.0)[0] == pytest.approx((1 - c) / (1 + c) - 1)
    assert v2_trades.net(np.array([1.1]), 20.0)[0] == pytest.approx(1.1 * 0.998 / 1.002 - 1)


def test_one_at_a_time_unit():
    exit_row = np.array([[3], [4], [6], [7], [9], [9]])
    grid = SimpleNamespace(exit_row=exit_row, spinoff_exit=np.zeros((6, 1), bool))
    entries = np.array([[1], [1], [0], [1], [1], [0]], dtype=bool)
    assert v2_trades.one_at_a_time(entries, grid)[:, 0].tolist() == [True, False, False, True, False, False]


def test_signal_on_the_last_row_has_no_entry():
    w = v2_trades.entry_counts(np.array([2, 5]), np.array([0, 0]), (6, 1))
    assert w[:, 0].tolist() == [0, 0, 0, 1, 0, 0]


# ---- statistics and the whole run ---------------------------------------------------------------------------------------------

def test_placebo_is_the_mean_of_random_entries_on_the_same_stocks():
    r = np.array([[0.1, -0.2], [0.1, -0.2], [0.1, -0.2]])
    valid = np.ones_like(r, bool)
    got = v2_run._placebo(np.random.default_rng(0), r, valid, np.array([0, 1]), np.array([3, 1]))
    assert got.shape == (settings().event_study.placebo_draws,) and np.allclose(got, (3 * 0.1 - 0.2) / 4)


def test_sealed_window_does_not_change_any_result(tmp_path):
    seal = "2019-01-02"
    ledger_a, ledger_b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for lg in (ledger_a, ledger_b):
        ledger.append(lg, {"event": "seal", "boundary_date": seal})
    a = v2_run.run_v2(_db(tmp_path / "a.db"), ledger_a)
    b = v2_run.run_v2(_db(tmp_path / "b.db", scale_after=seal), ledger_b)
    assert a["summary"] == b["summary"] and a["config"]["last_session"] < seal
    for k in a["results"]:
        assert a["results"][k]["counts"] == b["results"][k]["counts"]
    entries = ledger.read(ledger_a)
    assert [e["event"] for e in entries] == ["seal", "event_study_run"] and entries[1]["sealed_opened"] is False
    assert "Holm over 3 hypotheses" in a["markdown"] and "not a claim of alpha" in a["markdown"]


def test_run_refuses_without_a_seal(tmp_path):
    with pytest.raises(LookupError, match="no seal entry"):
        v2_run.run_v2(_db(tmp_path / "x.db"), tmp_path / "none.jsonl")
