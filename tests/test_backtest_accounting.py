from __future__ import annotations

import pytest

from trader_koo.ml import backtest, rule_baseline


@pytest.mark.parametrize("module", [backtest, rule_baseline])
@pytest.mark.parametrize("direction", ["long", "short"])
def test_cash_accounting_charges_one_round_trip_commission(module, direction):
    pos = {
        "ticker": "TEST",
        "direction": direction,
        "entry_price": 100.0,
        "shares": 10,
    }
    entry_cash_out = pos["entry_price"] * pos["shares"] + module.DEFAULT_COMMISSION_PER_TRADE / 2

    trade = module._compute_exit(
        pos,
        exit_price=100.0,
        exit_date="2026-01-02",
        exit_reason="expired",
        trading_days_held=1,
    )
    returned_cash = module._return_cash_on_close(pos, 100.0, trade["pnl"])

    assert trade["pnl"] == pytest.approx(-module.DEFAULT_COMMISSION_PER_TRADE)
    assert entry_cash_out - returned_cash == pytest.approx(module.DEFAULT_COMMISSION_PER_TRADE)
