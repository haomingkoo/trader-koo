---
paths:
  - "trader_koo/paper_trade/**"
  - "trader_koo/paper_trades.py"
  - "trader_koo/config.py"
  - "tests/test_critic.py"
---

# Feature Activation Checklist

Before promoting any experimental system (new regime filter, new sizing model)
from observation → production, all items must be checked.

## New Regime Filter Activation

1. **Backtest**: Run filter on last 30+ closed trades — measure trades blocked vs trades allowed vs SPY
2. **Override audit**: Check for any `score >= X` overrides that bypass the filter — each override is a potential leak
3. **Hard block verification**: VIX above `REGIME_VIX_HIGH_VOL` in `critic.py` = hard block longs, no exceptions. HMM counter-trend = block, no tier override.
4. **Test coverage**: Add parametrized boundary test to `tests/test_critic.py` for each new threshold

## New Sizing Model Activation

1. **Dollar risk consistency**: Verify `risk_budget_pct` <= `risk_per_trade_pct`; lower is expected when the tier ceiling binds
2. **Stop anchor verification**: Confirm `entry_price` (next-day open + slippage) is passed through to `compute_stop_and_target()`
3. **Wide-stop protection**: `position_size_pct` is about `risk_per_trade_pct` divided by stop distance (about 10% for a 5% stop) before the tier cap
4. **Cap validation**: Ensure no position exceeds tier ceiling even on very tight stops

## General Rule

> Do not promote a feature because it looks good on paper. Promote it because it has survived live market conditions without manual overrides.

Checklist failures are not blockers for deployment — they are blockers for **activation**.
The feature can be deployed (code merged) before activation criteria are met.
