# Results, hypotheses version 6 (insider buying, walk-forward)

Recorded 2026-10-30 for the rules frozen in `hypotheses-v6-insider.md`. Numbers
are copied from the bench trial ledger
(`/Users/koohaoming/dev/.research/bench_trial_ledger.jsonl`) and scored with
`scoring-rubric-v1.md`. This is a research result on historical data. It is not
a claim of alpha.

**Outcome: both hypotheses fail. The sealed window was not opened.**

## Run settings

Development window 2011-01-04 to 2023-06-09, `sp500_pit` universe, 10 bps per
side (stress 20 bps), 200 random portfolios, top 20 with a top-40 buffer,
review every 10 sessions, 20-session minimum hold, code `a1c17c3e`, ledger seal
at 2023-06-12. Insider data: 81 quarterly SEC data sets (2006 Q1 to 2026 Q1),
3,441,628 open-market rows, 69,074 of them on Form 4/A.

## Ledger entries

| trial id | candidate | CAGR | excess vs equal weight | random pct | years beating both | stress CAGR | max drawdown | turnover |
|---|---|---|---|---|---|---|---|---|
| 670eb569dff6 | v6 H1 insider clusters | 6.91% | -5.53% | 2.5 | 23% | 5.57% | -55.97% | 6.27 |
| 6e0b5cb784fe | v6 H2 opportunistic only | 6.94% | -5.50% | 4.0 | 23% | 5.58% | -56.67% | 6.34 |

Benchmarks over the same window: SPY 12.38%, equal weight 12.44%.
Rubric score: 0/10 for both (every criterion at zero).

## What the runs show

- About 11 names qualified per review (3,405 name-reviews over 313 reviews),
  so the portfolio held fewer than 20 names most of the time and turned over
  6.3 times a year. 580 held names were sold when their 90-day cluster expired.
- Dropping routine buyers removed 6,520 purchases and changed almost nothing:
  in these large caps the clusters are opportunistic already.
- The strategy beat both benchmarks only in 2013, 2014 and 2016.
- The published effect is measured mostly in small and mid caps. This test
  covers S&P 500 members only, so it does not test the small-cap claim.
