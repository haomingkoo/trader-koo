# Results, hypotheses version 1 (walk-forward)

Recorded 2026-10-26 for the rules frozen in `hypotheses-v1.md`. The numbers are
copied from the bench trial ledger (`/Users/koohaoming/dev/.research/bench_trial_ledger.jsonl`).
A frozen copy of all ten ledger lines is `ledger-v1.jsonl` (sha256
`37813c46a4ef69c9a02bfd7b1d8953ecded2afb25086930bbc2dced39844d624`). This is a
research result on historical data. It is not a claim of alpha.

**Outcome: no hypothesis survived the walk-forward. The sealed window was not
opened** (`sealed_opened: false` on every run), because the pre-registered rule
opens it once, for hypotheses that already pass criteria 1 to 3. Criterion 4 was
therefore not evaluated for any hypothesis.

## Run settings (identical on every row unless stated)

| field | value |
|---|---|
| universe | `sp500_pit` (point-in-time S&P 500 membership) |
| costs | 10 bps per side; stress run 20 bps per side |
| schedule | review every 10 sessions; `min_hold` 20; `buffer_n` 40; `top_n` 20 |
| random controls | `n_random` 200 |
| params hash | `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a` (all rows) |
| ledger seal | boundary `2023-06-12`, fraction 0.2, first session 2010-01-04, last session 2026-10-23, 4228 sessions, sealed 2026-10-26T14:33:55Z |
| code version | `a9749658c8e11aca9b0638ebadbb966c88490d48` (all rows except the last, see below) |

H4 rows use `top_n` 5, `buffer_n` 5, `min_hold` 0 (the annual schedule in
`hypotheses-v1.md`). The `equal_weight` reference uses `top_n` 1000, `buffer_n`
1000 and `n_random` 5.

## Ledger entries, verbatim values

Values are the stored `summary` fields. `excess_*` are annualised differences.

| trial id | candidate | date range | seed | cagr | excess_vs_equal_weight | excess_vs_spy | pct_years_beating_both | pct_years_beating_spy | random_percentile | sharpe | max_drawdown | hit_rate | turnover | stress_cagr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0b1a73e37088 | h1_quality_fair_price | 2011-01-04 to 2023-06-09 | 827902939 | 0.1320449229 | 0.0076038328 | 0.0082003533 | 0.4615384615 | 0.6153846154 | 97.5 | 0.6906614459 | -0.3519915713 | 0.5046474359 | 0.7976169277 | 0.1303570993 |
| 17374bdc9585 | h2_accelerating_fundamentals | 2011-01-04 to 2023-06-09 | 839520964 | 0.1071566557 | -0.0172844344 | -0.0166879139 | 0.3076923077 | 0.3846153846 | 74.5 | 0.5778772466 | -0.3997703412 | 0.4998397436 | 2.657959492 | 0.1013916973 |
| cbf2bce94096 | h3_momentum_confirmed | 2011-01-04 to 2023-06-09 | 4226561740 | 0.0900301889 | -0.0344109012 | -0.0338143807 | 0.2307692308 | 0.3076923077 | 34.0 | 0.4961794731 | -0.3692850237 | 0.4939102564 | 2.9261218693 | 0.0837225738 |
| 80ef159d8ce2 | h4_market_cap_leaders (end 2017-12-31, graded) | 2011-01-03 to 2017-12-29 | 3728324174 | 0.1481391887 | 0.0002616034 | 0.0126598701 | 0.5714285714 | 0.7142857143 | 53.0 | 0.9907533024 | -0.1542083617 | 0.4333333333 | 0.3457796165 | 0.1472643225 |
| 6838afd880d7 | h4_market_cap_leaders (start 2018-01-01, in-sample, inspiration period) | 2018-01-02 to 2023-06-09 | 1151124167 | 0.2292132684 | 0.1401945338 | 0.1200062409 | 0.8333333333 | 0.8333333333 | 100.0 | 0.8578959355 | -0.3984786602 | 0.68 | 0.1077164665 | 0.2288378218 |
| f44e57cd823d | momentum_12_1 (reference) | 2011-01-04 to 2023-06-09 | 418631842 | 0.0822950332 | -0.042146057 | -0.0415495365 | 0.2307692308 | 0.3076923077 | 24.5 | 0.4547976449 | -0.3979467355 | 0.4971153846 | 2.9689878713 | 0.075967975 |
| d0c149ccd3c1 | low_volatility (reference) | 2011-01-04 to 2023-06-09 | 3479064182 | 0.1231605878 | -0.0012805023 | -0.0006839818 | 0.3846153846 | 0.4615384615 | 92.0 | 0.8761960358 | -0.3333078763 | 0.5032051282 | 0.8245663917 | 0.1212951852 |
| 868d5a4ca7a3 | equal_weight (reference) | 2011-01-04 to 2023-06-09 | 1051067836 | 0.1244410901 | 0.0 | 0.0005965205 | 0.0 | 0.5384615385 | 100.0 | 0.718411134 | -0.4006785272 | 0.499521308 | 0.4857227346 | 0.1233341827 |
| ff173f47b790 | low_volatility (repeat, code `6bb8517fe1c3231a29f24d4944cd3fb87a8513f4`) | 2011-01-04 to 2023-06-09 | 3479064182 | 0.1231605878 | -0.0012805023 | -0.0006839818 | 0.3846153846 | 0.4615384615 | 92.0 | 0.8761960358 | -0.3333078763 | 0.5032051282 | 0.8245663917 | 0.1212951852 |

The last row repeats `low_volatility` after a later commit that requires a
complete 252-session window. Every stored number is identical to `d0c149ccd3c1`.
Each ledger line also stores the full run `config` and its timestamp
(2026-10-26, 14:33 to 15:10 UTC).

## Pass or fail per pre-registered criterion

Criteria are numbered as in `hypotheses-v1.md`. Criterion 3 compares
`stress_cagr` (20 bps) with each benchmark CAGR, which is not stored directly and
is derived as `cagr - excess_vs_spy` (SPY) and `cagr - excess_vs_equal_weight`
(equal weight, at 10 bps). Derived benchmark CAGRs: 2011 to 2023-06 SPY 0.1238,
equal weight 0.1244; H4 2011 to 2017 SPY 0.1355, equal weight 0.1479; H4 2018 to
2023-06 SPY 0.1092, equal weight 0.0890.

| hypothesis | 1. beats both in >= 60% of years | 2. above 95th percentile of random | 3. ahead of both at 20 bps | 4. sealed window | verdict |
|---|---|---|---|---|---|
| H1 quality at a fair price | fail (46.2%; beats SPY alone in 61.5%) | pass (97.5) | pass (0.1304 vs 0.1238 and 0.1244) | not evaluated | fails criterion 1 |
| H2 accelerating fundamentals | fail (30.8%) | fail (74.5) | fail (0.1014) | not evaluated | fails 1, 2, 3 |
| H3 momentum confirmed by earnings | fail (23.1%) | fail (34.0) | fail (0.0837) | not evaluated | fails 1, 2, 3 |
| H4 market leaders, 2011 to 2017 (graded) | fail (57.1%) | fail (53.0) | fail (0.1473 vs equal weight 0.1479) | not evaluated | fails 1, 2, 3; falsified as registered |
| H4 2018 to 2023-06 | not graded | not graded | not graded | not evaluated | in-sample, inspiration period; not evidence |
| H5 fade the crowd | n/a | n/a | n/a | n/a | prospective only, no backtest was run |

The 2018 onward H4 row would clear the numeric thresholds (83.3%, 100.0, 0.2288)
but the hypothesis text labels that period in-sample and excludes it from the
pass rule. The three reference runs are not judged against the rule.

## Caveats

- **H1 thin coverage.** H1 needs gross profit, liabilities, operating cash flow
  and capex. Banks and insurers lack gross profit and many names lack a full
  trailing twelve months. From the description of PR #345 (not stored in the
  ledger): H1 scored a mean of 66 of 385 universe names per review (41 of 316 in
  2011, 101 of 453 in 2023), so its 20 names come from a small scorable set.
- **H4 graded 2011 to 2017 and the BRK-B share-class issue.** Per the same PR
  description, the stored SEC facts keep one class's cover count per filing.
  BRK-B receives its class A count, so its market cap is about 100 times too
  small and it never ranks in H4, although it would have been a top-5 name in
  several years. The graded H4 result is therefore not a faithful run of the
  rule as written. Fixing it needs the facts refetched with share-class
  dimensions. The build checks (`make research-db`) list multi-class issuers.
- **2018 onward is the inspiration period.** The H4 idea came from watching
  2018 to 2026. Those rows are labelled in-sample and carry no weight.
- **Survivorship.** Per the PR description, 190 of 745 point-in-time members in
  the window have no price history (departed names), and members with prices
  versus members ranged from 325 of 514 in 2011 to 457 of 508 in 2023. Results
  exclude those names and likely flatter long-only results, mostly against SPY
  in early years. The equal-weight universe is the primary benchmark for that
  reason. The membership check files list 80 member tickers with no price on any
  member day since 2018.
- **Data-quality filter.** A rule-based filter (`bench/quality.py`, rules R1 to
  R3) was fixed before any result was looked at and applied to every run and
  control. It removes flagged ticker-days from the universe.
- **Sealed window.** Boundary 2023-06-12 (last 20% of 4228 sessions). It stays
  unopened and is opened once, for all hypotheses together, only if a
  hypothesis passes criteria 1 to 3. None did.
- **What this does not show.** A deterministic backtest on this data shows
  software behaviour on this data, not market performance. These results do not
  support a claim of alpha, and a failed hypothesis is reported as such.
