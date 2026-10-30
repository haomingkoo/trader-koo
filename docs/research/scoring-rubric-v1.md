# Strategy scoring rubric, version 1

Adopted 2026-10-30. It grades a bench run report and is applied by
`trader_koo/research/bench/verdict.py`. The pass rule of each hypotheses file
still decides pass or fail. The score shows how far a strategy is from
passing, so that runs are compared on one scale rather than by judgment. A
change is a new version; earlier scores are not rewritten.

Output order is fixed: reasoning, pros, cons, then gate results, then the
score. Every pro and con quotes the number it rests on.

## Rubric (10 points)

Points scale linearly between the "0 points at" and "full points at" values
and are clipped to that range.

| # | Criterion | Measure (development window, net of costs) | 0 points at | Full points at | Points |
|---|---|---|---|---|---|
| 1 | Return edge | CAGR minus the equal-weight universe CAGR | -2.0%/yr | +2.0%/yr | 3 |
| 2 | Not luck | percentile against the random portfolios | 50th | 95th | 2 |
| 3 | Consistency | share of calendar years beating SPY and equal weight | 40% | 60% | 2 |
| 4 | Cost robustness | stress-run CAGR minus the higher benchmark CAGR | -1.0%/yr | 0.0%/yr | 1 |
| 5 | Drawdown | max drawdown minus SPY's max drawdown | 15 pts worse | equal or better | 1 |
| 6 | Risk-adjusted | Sharpe ratio minus SPY's Sharpe ratio | -0.30 | 0.00 | 1 |

## Gates (from the common pass rule)

1. Beats SPY and equal weight in at least 60% of years.
2. Above the 95th percentile of the random portfolios.
3. Stress-run CAGR above both benchmark CAGRs.
4. Sealed window, when opened.

A strategy that fails any of gates 1 to 3 is capped at 5.0 points and its
verdict is "fail". A pass on 1 to 3 with the sealed window not yet opened is
"candidate".

## Pros and cons

Each criterion yields one line. It is a pro when the criterion earns at least
half its points and a con otherwise. Data coverage is always listed as a con
when the strategy scores fewer than half the universe names on average, because
the result then speaks for a subset.
