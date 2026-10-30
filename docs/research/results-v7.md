# Results, hypotheses version 7 (market-regime exposure rules)

Recorded 2026-10-30 for the rules frozen in `hypotheses-v7-regime.md`
(amendment 1: cash at FRED DTB3). Ledger trial `5ad8bdd6c413`, code
`9dd4692a`, development window 2011-01-03 to 2023-06-09, 10 bps per change in
exposure (stress 20 bps), 1,000 circular-shift timing controls. This is a
research result on historical data. It is not a claim of alpha.

**Outcome: no rule passes. The sealed window was not opened.**

SPY over the window: CAGR 12.41%, Sharpe 0.71, max drawdown -34.0%.

| rule | CAGR | Sharpe | max drawdown | in market | switches | pct vs controls | score |
|---|---|---|---|---|---|---|---|
| H1 VIX term structure | 9.51% | 0.68 | -30.6% | 92% | 154 | 83.7 | 2.54 |
| H2 VIX breakout (John) | 6.00% | 0.46 | -21.5% | 84% | 244 | 37.8 | 1.66 |
| H3 variance risk premium | 9.34% | 0.61 | -23.1% | 84% | 184 | 56.8 | 1.74 |
| H4 distribution days | -2.22% | -0.38 | -30.7% | 33% | 185 | 0.0 | 0.44 |
| H5 10-month trend | 7.27% | 0.54 | -24.5% | 83% | 20 | 10.1 | 1.27 |
| H6 12-month momentum | 7.43% | 0.49 | -34.0% | 86% | 12 | 0.9 | 0.00 |
| H7 inflation regime | 13.29% | 0.80 | -34.0% | 91% | 2 | 99.6 | 5.00 (raw 5.86) |
| H8 real money growth | 11.77% | 0.71 | -34.0% | 92% | 1 | 63.5 | 3.13 |
| H9 strong dollar | 10.24% | 0.63 | -34.6% | 96% | 14 | 3.2 | 0.71 |
| H10 inverted yield curve | 11.79% | 0.71 | -34.0% | 92% | 5 | 68.5 | 3.29 |
| H11 vote of H1 to H10 | 8.71% | 0.66 | -22.3% | 83% | 681 | 56.7 | 2.36 |

## What the runs show

- The price and volatility rules (H2, H3, H5, H11) cut the worst drawdown by
  10 to 12 points and gave up 3 to 6 points of CAGR. They sold into declines
  that recovered within weeks (2011, 2015-16, late 2018, March 2020) and
  re-entered after the rebound.
- H4 is out of the market 67% of the time: on SPY's own volume, five
  distribution days in 25 sessions is the normal state, not a warning.
- H7 is the only rule above the controls (99.6th percentile) and above SPY on
  Sharpe and CAGR. It made one round trip: out on 2021-07-13, back on
  2022-09-13. One episode is not evidence of a repeatable rule, and it did not
  reduce the worst drawdown (March 2020), so it fails criterion 2.
- H9 had no usable value before 2019-02-04, when FRED first published the
  broad dollar index (2,160 sessions held at exposure 1).
- H8 and H10 hardly acted in the window (1 and 5 switches).
