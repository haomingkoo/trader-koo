"""The critic rejection evaluator (docs/research/critic-rejections-v1.md)."""
from __future__ import annotations

import json
import sqlite3

import pytest

import trader_koo.paper_trades as paper_facade
from tests.test_paper_trades import TEST_REPORT_RUN_ID, _make_setup_row, _seed_price, conn  # noqa: F401
from trader_koo.paper_trade import critic as critic_module
from trader_koo.paper_trade.chronology import next_scheduled_session_after
from trader_koo.paper_trades import PAPER_TRADE_CAMPAIGN_ID as CAMPAIGN
from trader_koo.research import critic_rejections as cr

REPORT_DATE = "2026-03-14"


def _bars(rows: list[tuple[float, float, float, float]]) -> list[cr.Bar]:
    return [cr.Bar(f"d{i}", *row) for i, row in enumerate(rows)]


FLAT = (100.0, 100.5, 99.5, 100.0)


def _net(gross: float, entry: float, exit_: float, risk: float) -> float:
    return gross - cr.COST_PER_SIDE * (entry + exit_) / risk


@pytest.mark.parametrize("direction,stop,target", [("long", 95.0, 110.0), ("short", 105.0, 90.0)])
def test_target_touch_exits_at_the_target(direction, stop, target):
    hit = (100.0, max(100.0, target + 1), min(100.0, target - 1), 100.0)
    out = cr.simulate(direction, _bars([FLAT, hit] + [FLAT] * 8), stop, target, risk=5.0)
    assert (out.exit_reason, out.gross_r) == ("target", pytest.approx(2.0))
    assert out.net_r == pytest.approx(_net(2.0, 100.0, target, 5.0))


@pytest.mark.parametrize("direction,stop,target", [("long", 95.0, 110.0), ("short", 105.0, 90.0)])
def test_stop_wins_when_one_bar_reaches_both(direction, stop, target):
    both = (100.0, 111.0, 89.0, 100.0)
    out = cr.simulate(direction, _bars([both] + [FLAT] * 9), stop, target, risk=5.0)
    assert (out.exit_reason, out.gross_r) == ("stop", pytest.approx(-1.0))


def test_a_gap_through_the_stop_exits_at_the_open_not_the_stop():
    out = cr.simulate("long", _bars([FLAT, (90.0, 91.0, 89.0, 90.0)] + [FLAT] * 8), 95.0, 110.0, risk=5.0)
    assert (out.exit_reason, out.gross_r) == ("stop_gap_open", pytest.approx(-2.0))


def test_no_touch_exits_at_the_last_close():
    bars = _bars([FLAT] * 9 + [(100.0, 103.0, 99.0, 102.0)])
    out = cr.simulate("long", bars, 95.0, 110.0, risk=5.0)
    assert (out.exit_reason, out.gross_r) == ("horizon_close", pytest.approx(0.4))


def test_failed_checks_reads_every_fail_in_order():
    reasons = ["PASS [a]: ok", "FAIL [b]: x [y]: z", "FAIL [c]: w"]
    assert cr.failed_checks(reasons) == ("b", "c")


def _sessions(start: str, count: int) -> list[str]:
    out = [next_scheduled_session_after(start)]
    while len(out) < count:
        out.append(next_scheduled_session_after(out[-1]))
    return out


@pytest.fixture()
def ledger_db(conn, monkeypatch, tmp_path):  # noqa: F811
    """Four critic outcomes written by the real decision path, then copied to a file."""
    monkeypatch.setattr(critic_module, "_check_family_edge",
                        lambda _c, row, *_a: (row["ticker"] == "AAPL", "fixture edge"))
    monkeypatch.setattr(critic_module, "_check_debate_strength",
                        lambda row, _e: (row["ticker"] != "GOOG", "fixture debate"))
    for ticker in ("GOOG", "AAA"):
        _seed_price(conn, ticker, 150.0, date=REPORT_DATE)
    paper_facade.create_paper_trades_from_report(  # no next open seeded, so approval is pending
        conn, setup_rows=[_make_setup_row(ticker=t) for t in ("AAPL", "MSFT", "GOOG", "AAA")],
        report_date=REPORT_DATE, generated_ts="2026-03-14T22:00:00Z", report_run_id=TEST_REPORT_RUN_ID,
    )
    levels = {t: (s, g) for t, s, g in conn.execute(
        "SELECT ticker,stop_loss,target_price FROM paper_candidate_decisions")}
    references = {t: json.loads(i)["context"]["entry_price"] for t, i in conn.execute(
        "SELECT ticker,inputs_json FROM paper_candidate_decisions")}
    sessions = _sessions(REPORT_DATE, cr.HORIZON_BARS)
    aapl_stop, aapl_target = levels["AAPL"]
    goog_stop, goog_target = levels["GOOG"]
    for i, day in enumerate(sessions):
        high = aapl_target + 1 if i == 1 else 151.0
        _seed_price(conn, "AAPL", 150.0, date=day, high=high, low=149.0, open_=150.0)
        if i == 0:
            _seed_price(conn, "GOOG", 150.0, date=day, high=goog_target + 1, low=goog_stop - 1, open_=150.0)
        else:
            _seed_price(conn, "GOOG", 150.0, date=day, high=151.0, low=149.0, open_=150.0)
    for day in sessions[:3]:  # touches its target on day one, but the window is incomplete
        _seed_price(conn, "AAA", 150.0, date=day, high=levels["AAA"][1] + 1, low=149.0, open_=150.0)
    for day in sessions:
        _seed_price(conn, "MSFT", 150.0, date=day, high=151.0, low=149.0, open_=150.0)
    path = tmp_path / "copy.db"
    target = sqlite3.connect(path)
    conn.backup(target)
    target.close()
    return path, levels, references


def test_measures_each_group_from_the_real_decision_ledger(ledger_db):
    path, levels, references = ledger_db
    ro = cr.connect_ro(path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            ro.execute("DELETE FROM price_daily")
        result = cr.run(ro, CAMPAIGN)
    finally:
        ro.close()
    groups = result["groups"]
    stop, target = levels["AAPL"]
    risk = references["AAPL"] - stop  # one R is set from the decision's own entry price
    expected_win = _net((target - 150.0) / risk, 150.0, target, risk)
    assert groups["approved"]["n_closed"] == 1
    assert groups["approved"]["mean_net_r"] == pytest.approx(expected_win)
    assert groups["approved"]["reading"] == "insufficient"
    goog_risk = references["GOOG"] - levels["GOOG"][0]
    expected_loss = _net((levels["GOOG"][0] - 150.0) / goog_risk, 150.0, levels["GOOG"][0], goog_risk)
    rejected = groups["rejected (any check)"]
    assert (rejected["n_closed"], rejected["n_open"]) == (1, 1)  # GOOG closed, AAA open, MSFT excluded
    assert rejected["mean_net_r"] == pytest.approx(expected_loss)
    edge = groups["check: family_edge"]
    assert (edge["n_closed"], edge["n_open"], edge["n_sole_failure"]) == (1, 1, 1)
    debate = groups["check: debate_strength"]
    assert (debate["n_closed"], debate["n_open"], debate["n_sole_failure"]) == (1, 0, 0)
    assert debate["diff_vs_approved"] == pytest.approx(expected_loss - expected_win)
    assert result["excluded"]["basis_mismatch"] == 1  # MSFT's report-date close is 300, its stop was set from 150
    assert result["exit_reasons"] == {"stop": 1, "target": 1}


def test_cli_writes_the_same_markdown_it_prints(ledger_db, tmp_path, capsys):
    path, *_ = ledger_db
    out = tmp_path / "result.md"
    assert cr.main(["--db", str(path), "--campaign-id", CAMPAIGN, "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert text == capsys.readouterr().out
    assert "| check: family_edge | 1 | 1 | 1 |" in text
    assert "Nothing here is a conclusion" in text
    assert cr.main(["--db", str(path), "--campaign-id", CAMPAIGN, "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == text  # fixed seed: identical intervals
