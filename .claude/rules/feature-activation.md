# Feature Activation Checklist

Before promoting any experimental system (ML gating, new regime filter, new sizing model)
from observation → production, all items must be checked.

## ML Gating Activation

1. **Sample size**: ≥ 20 closed paper trades with `ml_predicted_win_prob` populated (not 60 calendar days — count trades)
2. **AUC threshold**: AUC > 0.55 on ≥ 5 independent walk-forward folds (not just the latest fold)
3. **Calibration**: Plot calibration curve — predicted win prob 0.6 should correspond to ~60% actual win rate
4. **Feature audit**: `macro_sp500_ret_63d` must NOT be the top feature — that predicts market direction, not cross-sectional alpha
5. **No leakage**: Confirm FRED vintage data, OOS meta-model, and imputation do not look forward
6. **Config flag**: Change `ml_enabled: bool = False` → `True` in `PaperTradeConfig` only after all above

## New Regime Filter Activation

1. **Backtest**: Run filter on last 30+ closed trades — measure trades blocked vs trades allowed vs SPY
2. **Override audit**: Check for any `score >= X` overrides that bypass the filter — each override is a potential leak
3. **Hard block verification**: VIX > 25 = hard block longs, no exceptions. HMM counter-trend = block, no tier override.
4. **Test coverage**: Add parametrized boundary test to `tests/test_critic.py` for each new threshold

## New Sizing Model Activation

1. **Dollar risk consistency**: Verify `risk_budget_pct` column is ~0.5% for all new trades (not tier-notional based)
2. **Stop anchor verification**: Confirm `entry_price` (next-day open + slippage) is passed through to `compute_stop_and_target()`
3. **Wide-stop protection**: Test a 5% stop trade — position_size_pct should be ~10% of the 1% stop case
4. **Cap validation**: Ensure no position exceeds tier ceiling even on very tight stops

## General Rule

> Do not promote a feature because it looks good on paper. Promote it because it has survived live market conditions without manual overrides.

Checklist failures are not blockers for deployment — they are blockers for **activation**.
The feature can be deployed (code merged) before activation criteria are met.
