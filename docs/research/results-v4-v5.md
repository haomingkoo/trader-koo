# Results, hypotheses versions 4 and 5 (walk-forward)

Recorded 2026-10-30 for the rules frozen in `hypotheses-v4.md` and
`hypotheses-v5-composite.md`. Numbers are copied from the bench trial ledger
(`/Users/koohaoming/dev/.research/bench_trial_ledger.jsonl`) and scored with
`scoring-rubric-v1.md`. This is a research result on historical data. It is not
a claim of alpha.

**Outcome: no hypothesis survived the walk-forward. The sealed window was not
opened.**

## Run settings

Development window 2011-01-04 to 2023-06-09, `sp500_pit` universe, 10 bps per
side (stress 20 bps), 200 random portfolios, top 20 with a top-40 buffer,
settings hash `5840ccb14511`, ledger seal at 2023-06-12. v4 ran on code
`bc483575`. v5 H1 ran on `47ea4b7f`. v5 H2 ran on `1d63503a`, which adds only
the scoring module; the signal code is the same as `47ea4b7f`.

## Ledger entries

| trial id | candidate | review / min hold | CAGR | excess vs equal weight | random pct | years beating both | stress CAGR |
|---|---|---|---|---|---|---|---|
| 9078d0cb288a | v4 H1 Magic Formula | 10 / 20 | 12.77% | +0.32% | 96.5 | 46% | 12.55% |
| f452ba6e2b00 | v4 H2 Piotroski value | 10 / 20 | 8.20% | -4.24% | 16.5 | 46% | 7.96% |
| d29e9c57bf17 | v4 H3 Trend Template | 10 / 20 | 1.79% | -10.66% | 0.0 | 8% | 0.11% |
| 612ff8c73011 | v5 H1 composite | 10 / 20 | 12.20% | -0.24% | 96.5 | 46% | 11.67% |
| ef59746fa570 | v5 H2 composite, held longer | 21 / 63 | 11.64% | -0.85% | 49.5 | 38% | 11.25% |

Benchmarks over the same window: SPY 12.38%, equal weight 12.44%.

## Rubric scores

| candidate | raw | score | gates failed |
|---|---|---|---|
| v4 H1 Magic Formula | 6.82 | 5.0 | years |
| v5 H1 composite | 5.94 | 5.0 | years, stress |
| v5 H2 composite, held longer | 2.69 | 2.69 | years, random, stress |
| v4 H3 Trend Template | 0.78 | 0.78 | years, random, stress |
| v4 H2 Piotroski value | 0.62 | 0.62 | years, random, stress |

## What the runs show

- The composite led its higher benchmark in 2011, 2013, 2014, 2015 and 2022
  and trailed in 2016 and 2018 to 2021. Its excess return depends on the
  period, so the year-consistency gate fails.
- Holding longer lowered turnover from 2.37 to 1.77 a year and did not keep
  more of the edge: the random-portfolio percentile fell from 96.5 to 49.5.
- Trend Template churned (8.2 turnover a year, 1,454 held names sold on
  dropping out of the template). The rule ran as registered.
- Coverage: Piotroski scored about 46 and Trend Template about 25 of 385
  names a review, so those results speak for a subset.
- Data limit: 190 of 745 members in the window have no price data. These are
  mainly companies that left the index before the price source began.
