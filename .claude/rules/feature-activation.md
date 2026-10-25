---
paths:
  - "trader_koo/paper_trade/**"
  - "trader_koo/ml/**"
  - "trader_koo/paper_trades.py"
  - "trader_koo/config.py"
  - "tests/test_critic.py"
---

# Feature Activation Checklist

Before promoting any experimental system (ML gating, new regime filter, new sizing model)
from observation → production, all items must be checked.

## ML Gating Activation

The promotion gates and their thresholds live in `trader_koo/ml/validation_registry.py`. That file is the authority, so do not restate its numbers here. All registered gates must pass first. Checks the registry does not cover:

1. **Closed-trade evidence**: enough closed paper trades with `ml_predicted_win_prob` populated to evaluate calibration (count trades, not calendar days). This does not replace the registry's trade-count gate.
2. **Calibration**: Plot calibration curve — predicted win prob 0.6 should correspond to ~60% actual win rate
3. **Feature audit**: `macro_sp500_ret_63d` must NOT be the top feature — that predicts market direction, not cross-sectional alpha
4. **No leakage**: Confirm FRED vintage data, OOS meta-model, and imputation do not look forward
5. **Config flag**: Change `ml_enabled: bool = False` → `True` in `PaperTradeConfig` only after the registry gates and all above pass. `ml_observe` (default `True`) scores trades and populates `ml_predicted_win_prob` independently of `ml_enabled`, so observation does not wait for activation.

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
