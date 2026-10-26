# Results, hypotheses versions 2 and 3 (development window only)

Recorded 2026-10-27 for the rules frozen in `hypotheses-v2-saty.md` and `hypotheses-v3-market-timing.md`.
Both runs read only sessions before the pinned seal (2023-06-12) and append to the bench trial ledger
(`/Users/koohaoming/dev/.research/bench_trial_ledger.jsonl`). A frozen copy of the four ledger lines (two superseded, two final) is
`ledger-v2-v3.jsonl` (sha256 `cc68c43c01adb3bff4cab5e4cede38580c4ce3d4fc401da4bca2cccec7e90e4b`). This is a research result on historical data. It is not a claim of alpha.

**Outcome: nothing earned the right to open the sealed window.** The sealed window was not opened
(`sealed_opened: false` on every entry). Criterion 3 of v3 and criterion 5 of v2 were therefore not evaluated for any
hypothesis.

- v2: S1 and S2 had no excess over their own state baselines. S3 had a positive excess over its state baseline
  (+0.269% per trade, Holm p 0.0204) but its net mean sat below the 95th percentile of the placebo (random entries on the
  same stocks), and it was negative at 20 bps. All three failed.
- v3: one test met the development criteria, H11 (VIX falls more often on Fridays than on Monday to Thursday, +14.4 points,
  Holm p 0.0042). It is a generic day-of-week frequency over 2010 to 2023. The author's claim is about 2024 onward, which is
  inside the sealed window, and it is not tested here. H2a (the stated 85%) is falsified: the event rate is 48.4% and the
  upper interval bound is 0.680. H4, H6 (QQQ, bottoming tail) and H10 had fewer than 30 independent events and are
  `insufficient`. H8 was not falsified. The other tests failed.
- Language: these are frequencies and per-trade averages on one development window with a survivor-biased stock
  universe. They do not support claims of alpha, a tradable edge, or generalisation (see `RESEARCH_STATUS.md`).

## Run inputs

| field | value |
|---|---|
| database | a copy of `trader_koo_research.db` (opened read-only; the original was not written) |
| seal | boundary `2023-06-12`, read from the ledger's `seal` entry, never moved. Last session used: 2023-06-09 (3,382 sessions from 2010-01-04) |
| code version | final runs `86ba26e74e5631fdc6c636f58d671fc379968fed` (clean tree). Earlier runs, before a change to the report text only: `92f7811854e44a529ac981865e67d1f8fc0e61d4` |
| settings sha256 | `5840ccb1451187909343e6f593b87568f1885ebbe5de38f0384e9d11ff49059e` (the `[event_study]` section is new in this change) |
| final trial ids | v2 `e7168b67e8f6`, v3 `f0094291be03` |
| earlier trial ids | v2 `43b073599be4`, v3 `bcf2c20b4329`: every number is identical, only the report text differed |
| seeds | derived from each run's config (v2 4203974023, v3 3704069732), one more per test from the test key |
| bootstrap | 10,000 resamples of 5-session moving blocks (v2: blocks of entry dates), seeds recorded |
| FOMC dates | `fomc-decision-dates-2010-2023.csv`: 125 rows compiled on 2026-10-27 from the Federal Reserve's FOMC historical-materials pages (one `source_url` per row, one page per year to 2020) and its meeting-calendar page for 2021 to 2023. 111 scheduled meetings (decision day = last day of the meeting). Conference calls, unscheduled meetings, notation votes and the cancelled March 2020 meeting are listed with their kind and not used. All 14 years were sourced, so H10 was not blocked. |
| instruments | SPY, QQQ, ^VIX from the research database (3,382 aligned sessions in the window). DIA, IWM, ^GSPC, ^TNX, TLT, UUP, USO and ^VIX3M exist only from 2025-06, which lies in the sealed window; no registered hypothesis needs them. |

The run was done by `python -m trader_koo.research.bench events --db <copy> --ledger <ledger>`.

## Readings where the documents are silent

I took the most literal reading each time and did not choose a reading for its result. Definitions were fixed before
the first run; the only edits after the first look at output were two bug fixes (a crash in H12's non-gap mask and a flag
that double counted spin-off exits) and report text.

Common to v3
- Sessions are SPY's. Bars are the stored price-only rows (`split_adjusted_price_only`), so ex-dividend gaps are in the closes.
  Every query is `date < 2023-06-12`.
- Independence: a signal is dropped if any earlier signal, kept or not, is fewer than 5 sessions before it, so a run of
  signals counts once. "Within 5 sessions" read as 5 or fewer would drop every second Friday of H11 (Fridays are exactly 5
  sessions apart); that stricter count is shown in brackets next to `independent`.
- Holm family: 21 primary tests (H1, H2a, H2b, H3, H4, H5 close break, H6 four instrument and side tests, H7, H8 on SPY and QQQ,
  H9, H10, H11, H12, H13 four instrument and outcome tests). The document says 13; it lists more tests than that. A blocked or
  insufficient test stays in the family with p = 1. The p-value is two-sided from the bootstrap (share of resamples on the far
  side of zero, doubled). The reported interval is the percentile interval at the test's Holm step level, which excludes zero
  exactly when Holm rejects. A test passes the development criteria when effect >= +5 points, the interval's lower bound is
  above zero and it has at least 30 independent events.
- Stated claims are checked with the same interval applied to the event rate itself.
- Trailing windows include session t. EMAs are `ewm(span=n, adjust=False)` and undefined until n observations exist; the seed
  value still weights the first sessions of 2010. Percentile and tercile cut-offs use the trailing 252 sessions including t.
- H1 and H2 matched bands: `[-15%, -5%]` and `[+8%, +20%)` excluding signal sessions. H1 secondary needs any down
  close-to-close day in t+1, t+2.
- H3: sample standard deviation, width = 4 sd / SMA, compressed when width <= the 10th percentile (linear interpolation) of the
  trailing 252 widths. H4 terciles use the same trailing-252 cut-offs on 20-session realised volatility of log returns.
- H5: "the band has held at least 10 sessions" is read as the range test also passing on each of the 10 sessions before t. Long
  and short breaks are pooled; the baseline outcome follows the break direction. The wick-versus-close claim is the difference of
  event rates, tested as `H5-AvsB`.
- H6: boundaries inclusive; the 60-session extreme includes t.
- H7: new low means the close equals the 10-session minimum; "VIX lower high" means strictly below its 10-session maximum.
- H8: verdict by the registered rule (down frequency exceeds the unconditional by 3 points or more falsifies); the +5 point
  lift rule does not apply to a stand-aside gate.
- H9: the 20-session average volume includes t; the matched baseline has no volume condition.
- H10: scheduled meetings only.
- H12: "same distance" is a fraction of the open, taken from each event's own gap; the baseline is recomputed on every
  bootstrap resample. Ex-dividend sessions are excluded from events and from the baseline (counts under `H12`).
- H13: the primary signal is causal (quiet on t and t-1, so the second session of a streak is the first signal). The published
  last-session-of-streak trigger needs the next session and is reported as a variant, labelled as such. "Close on session 8 or 9
  below" is read as either; the both-days reading is a secondary check. "Matched by calendar month" is month of year. The lift
  test compares with half the published lift (14.1 points for (a), 6.8 points for (b) at 2.36%).

Specific to v2
- A trade enters at the open of t+1 and exits at the open after the first close below the exit average, evaluated from the
  entry day's close, or at the open of entry + 20 (S3: + 10) sessions at the latest. Entries whose maximum hold runs past
  2023-06-09 are dropped and counted (`dropped_entry_too_late_for_window`).
- Cost: buy 1 including a 10 bps fee, sell proceeds less 10 bps; stress 20 bps. Equal dollar size per trade.
- S1: one open trade per stock, a stock may re-enter at the open its previous trade exits. S2 and S3 do not state it, so
  overlapping trades are allowed. S2's baseline enters after every cross, including crosses without a retest. S1's baseline
  excludes pullback days.
- S3: Bollinger deviation is the population standard deviation (as Pine's `ta.stdev`), so compression is sd <= ATR14.
  Expansion is the first non-compressed day after a compressed day, with EMA21 > EMA48. The baseline needs no compression in
  the previous 20 sessions (t-20 to t-1).
- Placebo: for every stock with k trades, k random entry days among that stock's scored days, same exit rule, 1,000 draws;
  percentile of the strategy's mean in the draws.
- "Positive in at least 60% of years" is judged on the lower of two fractions, years with a positive net mean and years with
  a positive excess over the baseline; both are shown. "Still positive at 20 bps" requires both net mean and excess to be
  positive. Years are entry years from 2011; 2010 only warms up the averages.
- Benchmarks: SPY and the equal-weight universe (daily-rebalanced mean open-to-open return of members) over each trade's own
  holding window.
- Spin-offs in `spinoffs.csv` exit at the close before the ex-date; a missing exit open falls back to the next valid open, then
  the last close. Both are counted.

## Caveats

- The universe has price history for current and some past members only (departed names have none), which flatters long-only
  results (`RESEARCH_STATUS.md`). The v2 benchmark rows share this bias.
- The data-quality rules (seam, erratic year, zero-volume run) use the stored series of the whole database, as in v1, and so can
  look at rows after the seal when judging a ticker-year or the backfill seam. No return, signal or outcome from the sealed
  window enters a result, and a test (`test_sealed_window_does_not_change_any_result`) shows the v2 results do not change when
  every sealed row is replaced. The three rules removed 2,571 index-member days inside the window.
- The development window is one path of history with heavy overlap in 2010 to 2012 and 2020 to 2022; the bootstrap resamples
  blocks of 5 sessions, which does not capture longer dependence.
- A synthetic or deterministic result proves software behaviour, not market performance.

## Full output of the final runs

### Hypotheses v2 event study (v2), development window

Research output on historical data; not a claim of alpha or profitability. Every qualifying entry is a trade of equal dollar size at the next open, 10 bps per side (stress 20), overlapping trades allowed (S1: one open trade per stock). Departed index members without price history are not in the universe, which flatters long-only results.

Entries 2011-01-03 to 2023-06-09; sealed window starts 2023-06-12 and was not opened. Seed 4203974023; 10000 resamples of 5-session blocks of entry dates; Holm over 3 hypotheses.

| hypothesis | trades | mean net | mean gross | state baseline net | excess | Holm p | Holm interval | placebo percentile (p95) | years net > 0 | years excess > 0 | net at stress | excess at stress | SPY | equal weight | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 23280 | +0.168% | +0.368% | +0.171% | -0.003% | 1.0000 | [-0.152%, +0.155%] | 99.9 (+0.132%) | 54% | 46% | -0.033% | -0.003% | +0.343% | +0.375% | **fail**: excess over the state baseline > 0 with Holm-corrected p < 0.05; positive in at least 60% of years (the lower of net and excess); still positive at the stress cost (net mean and excess) |
| S2 | 26385 | +0.110% | +0.311% | +0.172% | -0.062% | 1.0000 | [-0.294%, +0.160%] | 100.0 (+0.017%) | 62% | 46% | -0.090% | -0.062% | +0.257% | +0.306% | **fail**: excess over the state baseline > 0 with Holm-corrected p < 0.05; positive in at least 60% of years (the lower of net and excess); still positive at the stress cost (net mean and excess) |
| S3 | 24424 | +0.168% | +0.368% | -0.101% | +0.269% | 0.0204 | [+0.035%, +0.508%] | 0.0 (+0.409%) | 62% | 77% | -0.032% | +0.269% | +0.392% | +0.394% | **fail**: net mean above the placebo 95th percentile; still positive at the stress cost (net mean and excess) |

#### S1 by calendar year (entry year; net of 10 bps)

| year | trades | mean net | baseline net | excess |
|---|---|---|---|---|
| 2011 | 1617 | -0.361% | -0.576% | +0.215% |
| 2012 | 1738 | +0.413% | +0.589% | -0.176% |
| 2013 | 2134 | +0.958% | +0.985% | -0.027% |
| 2014 | 2100 | -0.115% | -0.055% | -0.059% |
| 2015 | 1729 | -0.788% | -0.906% | +0.119% |
| 2016 | 1594 | +0.409% | +0.415% | -0.006% |
| 2017 | 2208 | +1.025% | +1.134% | -0.109% |
| 2018 | 1761 | -0.822% | -1.398% | +0.576% |
| 2019 | 1879 | +0.476% | +0.517% | -0.041% |
| 2020 | 1785 | +0.747% | +0.675% | +0.072% |
| 2021 | 2771 | +0.613% | +0.469% | +0.144% |
| 2022 | 1293 | -1.002% | -1.268% | +0.266% |
| 2023 | 671 | -1.152% | -0.857% | -0.295% |

Counts for S1:

```
{
 "signals_entries": 23381,
 "baseline_entries": 442879,
 "dropped_entry_too_late_for_window": 101,
 "dropped_no_open_at_entry": 0,
 "dropped_no_exit_price": 0,
 "exit_open_missing_next_open_used": 0,
 "exit_open_missing_last_close_used": 0,
 "spinoff_exits": 2,
 "baseline_dropped_not_scorable": 1895,
 "baseline_exit_open_missing_next_open_used": 0,
 "baseline_exit_open_missing_last_close_used": 29,
 "baseline_spinoff_exits": 39,
 "signals_skipped_a_trade_was_open": 44332
}
```

#### S2 by calendar year (entry year; net of 10 bps)

| year | trades | mean net | baseline net | excess |
|---|---|---|---|---|
| 2011 | 1993 | -0.371% | -0.490% | +0.120% |
| 2012 | 1902 | +0.124% | -0.031% | +0.155% |
| 2013 | 1672 | +0.592% | +0.806% | -0.214% |
| 2014 | 1755 | +0.196% | +0.516% | -0.320% |
| 2015 | 2291 | -0.441% | -0.580% | +0.139% |
| 2016 | 2097 | +0.433% | +0.449% | -0.016% |
| 2017 | 2086 | +0.421% | +0.541% | -0.120% |
| 2018 | 2279 | -0.579% | -0.787% | +0.208% |
| 2019 | 2183 | +0.602% | +1.108% | -0.505% |
| 2020 | 2289 | +0.467% | +0.292% | +0.175% |
| 2021 | 2432 | +0.403% | +0.727% | -0.324% |
| 2022 | 2408 | -0.213% | -0.429% | +0.216% |
| 2023 | 998 | -0.231% | -0.004% | -0.226% |

Counts for S2:

```
{
 "signals_entries": 26563,
 "baseline_entries": 36357,
 "dropped_entry_too_late_for_window": 178,
 "dropped_no_open_at_entry": 0,
 "dropped_no_exit_price": 0,
 "exit_open_missing_next_open_used": 0,
 "exit_open_missing_last_close_used": 4,
 "spinoff_exits": 4,
 "baseline_dropped_not_scorable": 343,
 "baseline_exit_open_missing_next_open_used": 0,
 "baseline_exit_open_missing_last_close_used": 4,
 "baseline_spinoff_exits": 5,
 "crosses": 36362,
 "crosses_without_a_retest_in_10_sessions": 9799,
 "retests_not_an_index_member": 5
}
```

#### S3 by calendar year (entry year; net of 10 bps)

| year | trades | mean net | baseline net | excess |
|---|---|---|---|---|
| 2011 | 1664 | +0.052% | -0.854% | +0.906% |
| 2012 | 1851 | +0.228% | +0.523% | -0.295% |
| 2013 | 2223 | +1.042% | +0.462% | +0.581% |
| 2014 | 1977 | +0.294% | -0.041% | +0.336% |
| 2015 | 1441 | -0.145% | -0.546% | +0.401% |
| 2016 | 1845 | +0.341% | +0.487% | -0.146% |
| 2017 | 2174 | +0.657% | +0.538% | +0.119% |
| 2018 | 1947 | -0.391% | -0.923% | +0.532% |
| 2019 | 2195 | +0.508% | +0.263% | +0.245% |
| 2020 | 2148 | -0.103% | -0.676% | +0.573% |
| 2021 | 2608 | +0.386% | +0.697% | -0.311% |
| 2022 | 1520 | -1.210% | -1.667% | +0.457% |
| 2023 | 831 | -0.546% | -1.394% | +0.848% |

Counts for S3:

```
{
 "signals_entries": 24502,
 "baseline_entries": 103057,
 "dropped_entry_too_late_for_window": 78,
 "dropped_no_open_at_entry": 0,
 "dropped_no_exit_price": 0,
 "exit_open_missing_next_open_used": 0,
 "exit_open_missing_last_close_used": 3,
 "spinoff_exits": 3,
 "baseline_dropped_not_scorable": 202,
 "baseline_exit_open_missing_next_open_used": 0,
 "baseline_exit_open_missing_last_close_used": 5,
 "baseline_spinoff_exits": 17
}
```

#### Data

Substitutions and skips while loading the price matrix:

```
{
 "high_low_rows_nonpositive_or_crossed_set_to_missing": 0,
 "adjusted_close_rows_without_a_high_or_low": 0,
 "off_calendar_dividend_flags_not_listed_in_flagged_rows": 93
}
```

A counter that is zero is omitted. Index-member days removed by the data-quality filter (`quality.py`; the same rules and the same stored series as hypotheses v1) inside the window, by rule:

```
{
 "erratic_year": {
  "tickers": 2,
  "member_days": 1108
 },
 "seam": {
  "tickers": 1,
  "member_days": 1014
 },
 "zero_volume_run": {
  "tickers": 3,
  "member_days": 449
 }
}
```

ledger trial_id (v2): e7168b67e8f6

### Hypotheses v3 event study (v3), development window

Research output on historical data; not a claim of alpha or profitability. A matched-baseline frequency difference is not a trading result: there are no costs, positions or sizing here.

Window 2010-01-04 to 2023-06-09 (3382 SPY sessions). Sealed window starts 2023-06-12 and was not opened. Seed 3704069732; 10000 resamples of 5-session blocks. Holm family m = 21 primary tests (a blocked or insufficient test stays in the family).

#### Primary tests

| test | signals | with outcome | independent (strict) | event rate | all events | matched | unconditional | effect (pts) | Holm p | Holm interval (pts) | stated claim | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | 48 | 48 | 40 (38) | 50.0% | 45.8% | 42.7% | 41.8% | +7.27 | 1.0000 | [-17.97, +32.17] |  | **fail**: Holm-adjusted interval includes zero |
| H2a | 79 | 79 | 62 (60) | 48.4% | 54.4% | 57.5% | 58.0% | -9.15 | 1.0000 | [-30.80, +11.55] | 85%: rate interval [0.290, 0.680] FALSIFIED | **fail (stated claim falsified)**: effect -9.15 points is below +5; Holm-adjusted interval includes zero; stated 85%: interval upper bound 0.680 is below it |
| H2b | 79 | 79 | 62 (60) | 74.2% | 75.9% | 60.0% | 53.7% | +14.19 | 0.4636 | [-4.41, +31.38] |  | **fail**: Holm-adjusted interval includes zero |
| H3 | 365 | 365 | 49 (48) | 26.5% | 36.4% | 30.0% | 29.0% | -3.49 | 1.0000 | [-21.73, +17.77] |  | **fail**: effect -3.49 points is below +5; Holm-adjusted interval includes zero |
| H4 | 71 | 71 | 20 (20) | 25.0% | 21.1% | 22.1% | 40.1% | +2.94 | 1.0000 | n/a |  | **insufficient**: 20 independent events, fewer than 30 |
| H5-A | 431 | 428 | 120 (101) | 52.5% | 52.6% | 53.4% | 53.5% | -0.87 | 1.0000 | [-14.75, +12.11] |  | **fail**: effect -0.87 points is below +5; Holm-adjusted interval includes zero; a wick break is not at least 5 points weaker than a close break |
| H6-SPY-top | 150 | 149 | 115 (107) | 41.7% | 42.3% | 42.3% | 41.8% | -0.59 | 1.0000 | [-13.93, +13.71] |  | **fail**: effect -0.59 points is below +5; Holm-adjusted interval includes zero |
| H6-SPY-bottom | 44 | 44 | 34 (33) | 67.6% | 61.4% | 53.5% | 58.0% | +14.12 | 1.0000 | [-14.04, +38.47] |  | **fail**: Holm-adjusted interval includes zero |
| H6-QQQ-top | 139 | 138 | 107 (99) | 40.2% | 37.7% | 41.4% | 42.0% | -1.19 | 1.0000 | [-16.10, +13.07] |  | **fail**: effect -1.19 points is below +5; Holm-adjusted interval includes zero |
| H6-QQQ-bottom | 19 | 19 | 14 (14) | 78.6% | 73.7% | 57.6% | 57.8% | +20.95 | 1.0000 | n/a |  | **insufficient**: 14 independent events, fewer than 30 |
| H7 | 170 | 170 | 100 (96) | 61.0% | 62.4% | 58.2% | 59.2% | +2.83 | 1.0000 | [-14.17, +18.93] |  | **fail**: effect +2.83 points is below +5; Holm-adjusted interval includes zero |
| H8-SPY | 1485 | 1480 | 132 (95) | 37.9% | 40.1% | 37.8% | 39.7% | +0.05 | 1.0000 | [-13.26, +14.48] |  | **not falsified**: down frequency -1.86 points against the unconditional (falsified at +3 or more) |
| H8-QQQ | 1492 | 1489 | 125 (102) | 38.4% | 38.5% | 39.0% | 39.3% | -0.59 | 1.0000 | [-15.21, +14.23] |  | **not falsified**: down frequency -0.90 points against the unconditional (falsified at +3 or more) |
| H9 | 103 | 103 | 94 (91) | 40.4% | 41.7% | 37.8% | 39.7% | +2.61 | 1.0000 | [-12.82, +18.62] |  | **fail**: effect +2.61 points is below +5; Holm-adjusted interval includes zero |
| H10 | 25 | 25 | 25 (25) | 32.0% | 32.0% | 55.9% | 54.5% | -23.90 | 1.0000 | n/a |  | **insufficient**: 25 independent events, fewer than 30 |
| H11 | 678 | 678 | 579 (22) | 66.0% | 64.3% | 51.5% | 54.1% | +14.44 | 0.0042 | [+7.86, +21.46] |  | **development criteria met (sealed window not opened)**: criteria 1 and 2 met; criterion 3 needs the sealed window |
| H12 | 1307 | 1303 | 175 (106) | 84.0% | 83.7% | 80.9% | 93.9% | +3.13 | 1.0000 | [-5.73, +10.99] |  | **fail**: effect +3.13 points is below +5; Holm-adjusted interval includes zero |
| H13-SPY-a | 420 | 420 | 106 (87) | 36.8% | 40.5% | 41.9% | 42.1% | -5.13 | 1.0000 | [-18.58, +9.46] |  | **fail**: effect -5.13 points is below +5; Holm-adjusted interval includes zero; lift -5.13 points is below half the published 7.05 |
| H13-SPY-b | 420 | 420 | 106 (87) | 38.7% | 30.5% | 48.2% | 46.3% | -9.55 | 0.6407 | [-22.98, +3.79] |  | **fail**: effect -9.55 points is below +5; Holm-adjusted interval includes zero; lift -9.55 points is below half the published 3.40 |
| H13-QQQ-a | 125 | 125 | 59 (57) | 49.2% | 48.0% | 41.6% | 42.3% | +7.55 | 1.0000 | [-11.20, +28.04] |  | **fail**: Holm-adjusted interval includes zero |
| H13-QQQ-b | 125 | 125 | 59 (57) | 40.7% | 30.4% | 55.3% | 55.1% | -14.63 | 0.4440 | [-32.91, +5.10] |  | **fail**: effect -14.63 points is below +5; Holm-adjusted interval includes zero; lift -14.63 points is below half the published 3.40 |

#### Secondary checks and variants (plain 95% interval, no verdict)

| test | independent | event rate | matched | unconditional | effect (pts) | 95% interval (pts) | note |
|---|---|---|---|---|---|---|---|
| H1-sec (secondary) | 40 | 80.0% | 71.2% | 70.8% | +8.79 | [-4.64, +20.74] |  |
| H1-12 (variant) | 77 | 45.5% | 42.4% | 41.8% | +3.02 | [-8.68, +14.67] |  |
| H1-20 (variant) | 10 | 20.0% | 42.7% | 41.8% | -22.73 | [-44.43, +7.12] |  |
| H2a-sec (secondary) | 62 | 54.8% | 55.4% | 54.5% | -0.55 | [-14.16, +13.12] |  |
| H2b-sec (secondary) | 62 | 67.7% | 62.2% | 54.1% | +5.59 | [-7.36, +18.15] |  |
| H2a+25 (variant) | 38 | 55.3% | 57.5% | 58.0% | -2.28 | [-19.57, +14.66] |  |
| H2b+25 (variant) | 38 | 78.9% | 60.0% | 53.7% | +18.95 | [+4.30, +32.38] |  |
| H2a+40 (variant) | 13 | 38.5% | 57.5% | 58.0% | -19.08 | [-46.25, +10.13] |  |
| H2b+40 (variant) | 13 | 76.9% | 60.0% | 53.7% | +16.92 | [-8.99, +39.31] |  |
| H3-sec (secondary) | 49 | 16.3% | 23.4% | 24.1% | -7.08 | [-17.03, +4.16] | matched baseline: not compressed, same VIX tercile of the trailing 252 sessions |
| H3-always-up (secondary) | 49 | 42.9% | 50.5% | 46.6% | -7.66 | [-21.86, +6.52] | matched baseline: not compressed, same VIX tercile of the trailing 252 sessions |
| H3-p5 (variant) | 34 | 35.3% | 30.6% | 29.0% | +4.66 | [-11.10, +21.26] | matched baseline: not compressed, same VIX tercile of the trailing 252 sessions |
| H3-p20 (variant) | 70 | 27.1% | 27.9% | 29.0% | -0.80 | [-10.89, +10.02] | matched baseline: not compressed, same VIX tercile of the trailing 252 sessions |
| H3-h2 (variant) | 49 | 16.3% | 26.4% | 25.9% | -10.04 | [-19.88, +1.15] | matched baseline: not compressed, same VIX tercile of the trailing 252 sessions |
| H3-h5 (variant) | 49 | 34.7% | 32.9% | 30.6% | +1.83 | [-11.15, +15.79] | matched baseline: not compressed, same VIX tercile of the trailing 252 sessions |
| H5-B (secondary) | 110 | 60.9% | 53.9% | 54.0% | +7.06 | [-2.26, +16.14] |  |
| H5-AvsB (secondary) | 120 | 52.5% | 56.9% | 53.5% | -4.38 | [-14.11, +5.21] | effect = rate after a close break minus rate after a wick break |
| H5-SPY-h3-A (variant) | 120 | 55.0% | 53.3% | 53.2% | +1.72 | [-7.27, +10.53] |  |
| H5-SPY-h3-B (secondary) | 111 | 56.8% | 53.8% | 53.7% | +2.94 | [-6.40, +12.02] |  |
| H5-SPY-h3-AvsB (secondary) | 120 | 55.0% | 53.7% | 53.2% | +1.34 | [-8.47, +11.39] | effect = rate after a close break minus rate after a wick break |
| H5-SPY-h10-A (variant) | 119 | 54.6% | 52.6% | 54.5% | +2.07 | [-6.88, +10.79] |  |
| H5-SPY-h10-B (secondary) | 110 | 57.3% | 53.0% | 55.2% | +4.29 | [-4.79, +13.23] |  |
| H5-SPY-h10-AvsB (secondary) | 119 | 54.6% | 57.3% | 54.5% | -2.63 | [-11.71, +6.68] | effect = rate after a close break minus rate after a wick break |
| H5-QQQ-h5-A (variant) | 71 | 56.3% | 54.5% | 53.7% | +1.87 | [-9.38, +13.06] |  |
| H5-QQQ-h5-B (secondary) | 57 | 56.1% | 56.5% | 55.4% | -0.32 | [-12.66, +12.24] |  |
| H5-QQQ-h5-AvsB (secondary) | 71 | 56.3% | 46.1% | 53.7% | +10.22 | [-3.48, +23.76] | effect = rate after a close break minus rate after a wick break |
| H5-QQQ-h3-A (variant) | 71 | 57.7% | 53.9% | 53.1% | +3.82 | [-7.93, +14.89] |  |
| H5-QQQ-h3-B (secondary) | 57 | 56.1% | 55.7% | 54.5% | +0.40 | [-12.86, +13.16] |  |
| H5-QQQ-h3-AvsB (secondary) | 71 | 57.7% | 48.2% | 53.1% | +9.57 | [-4.50, +22.57] | effect = rate after a close break minus rate after a wick break |
| H5-QQQ-h10-A (variant) | 71 | 62.0% | 56.1% | 54.6% | +5.85 | [-5.76, +17.09] |  |
| H5-QQQ-h10-B (secondary) | 57 | 50.9% | 58.9% | 56.7% | -7.97 | [-20.91, +4.93] |  |
| H5-QQQ-h10-AvsB (secondary) | 71 | 62.0% | 46.3% | 54.6% | +15.70 | [+2.54, +28.72] | effect = rate after a close break minus rate after a wick break |
| H6-SPY-h1-x2-top (variant) | 115 | 37.4% | 46.1% | 45.2% | -8.71 | [-17.98, +0.47] |  |
| H6-SPY-h1-x2-bottom (variant) | 34 | 64.7% | 53.9% | 54.5% | +10.80 | [-6.72, +27.75] |  |
| H6-SPY-h3-x2-top (variant) | 115 | 35.7% | 41.1% | 40.6% | -5.46 | [-14.27, +3.67] |  |
| H6-SPY-h3-x2-bottom (variant) | 34 | 61.8% | 56.9% | 59.2% | +4.89 | [-12.54, +21.24] |  |
| H6-SPY-h2-x1.5-top (variant) | 130 | 43.1% | 42.3% | 41.8% | +0.78 | [-7.65, +9.64] |  |
| H6-SPY-h2-x1.5-bottom (variant) | 34 | 67.6% | 53.8% | 58.0% | +13.89 | [-3.12, +29.88] |  |
| H6-SPY-h2-x3-top (variant) | 93 | 40.9% | 42.4% | 41.8% | -1.53 | [-11.48, +8.74] |  |
| H6-SPY-h2-x3-bottom (variant) | 30 | 60.0% | 53.9% | 58.0% | +6.07 | [-12.91, +23.94] |  |
| H6-QQQ-h1-x2-top (variant) | 107 | 43.0% | 43.8% | 44.2% | -0.76 | [-10.20, +8.95] |  |
| H6-QQQ-h1-x2-bottom (variant) | 14 | 57.1% | 54.3% | 55.3% | +2.86 | [-24.95, +29.22] |  |
| H6-QQQ-h3-x2-top (variant) | 107 | 38.3% | 39.5% | 41.0% | -1.19 | [-10.23, +8.16] |  |
| H6-QQQ-h3-x2-bottom (variant) | 14 | 64.3% | 62.4% | 58.9% | +1.90 | [-26.60, +28.02] |  |
| H6-QQQ-h2-x1.5-top (variant) | 115 | 39.1% | 41.4% | 42.0% | -2.31 | [-11.19, +6.61] |  |
| H6-QQQ-h2-x1.5-bottom (variant) | 19 | 57.9% | 58.9% | 57.8% | -1.02 | [-25.27, +21.92] |  |
| H6-QQQ-h2-x3-top (variant) | 87 | 44.8% | 41.0% | 42.0% | +3.80 | [-6.52, +14.52] |  |
| H6-QQQ-h2-x3-bottom (variant) | 11 | 81.8% | 57.9% | 57.8% | +23.95 | [-4.57, +45.37] |  |
| H7-w5 (variant) | 151 | 58.3% | 57.8% | 59.2% | +0.52 | [-8.19, +9.14] |  |
| H7-w20 (variant) | 70 | 60.0% | 54.2% | 59.2% | +5.77 | [-8.18, +19.16] |  |
| H9-sec (secondary) | 94 | 39.4% | 39.5% | 36.9% | -0.17 | [-10.27, +10.02] |  |
| H13-SPY-a-both (secondary) | 106 | 25.5% | 31.4% | 31.6% | -5.93 | [-14.13, +2.76] |  |
| H13-SPY-b-2124 (secondary) | 106 | 41.5% | 51.9% | 50.0% | -10.41 | [-19.22, -1.25] |  |
| H13-SPY-a-lastday (variant) | 134 | 41.0% | 42.2% | 42.1% | -1.14 | [-9.23, +7.07] | published convention: the trigger is the last session of the streak, known only once the next session is quiet or not |
| H13-SPY-b-lastday (variant) | 134 | 38.1% | 48.1% | 46.3% | -10.08 | [-17.87, -1.77] | published convention: the trigger is the last session of the streak, known only once the next session is quiet or not |
| H13-QQQ-a-both (secondary) | 59 | 32.2% | 31.5% | 32.0% | +0.68 | [-10.94, +12.94] |  |
| H13-QQQ-b-2124 (secondary) | 59 | 45.8% | 58.2% | 58.1% | -12.47 | [-24.86, -0.29] |  |
| H13-QQQ-a-lastday (variant) | 62 | 50.0% | 41.7% | 42.3% | +8.30 | [-4.37, +20.90] | published convention: the trigger is the last session of the streak, known only once the next session is quiet or not |
| H13-QQQ-b-lastday (variant) | 62 | 43.5% | 55.3% | 55.1% | -11.78 | [-23.52, +0.42] | published convention: the trigger is the last session of the streak, known only once the next session is quiet or not |

#### Data

Price basis: `{"adjustment_basis": {"split_adjusted_price_only": 10146}, "basis_status": {"verified": 6764, "unresolved": 3382}}`. SPY, QQQ and ^VIX are the stored daily bars of the research database (price only, so ex-dividend gaps are in the closes).

Scheduled FOMC decision days on sessions in the window: 106 (`docs/research/fomc-decision-dates-2010-2023.csv`). A counter that is zero is omitted.

Substitutions, rejections and skips:

```
{}
```

Signals dropped because their outcome window leaves the development window, and test-specific counts:

```
{
 "H5-A": {
  "signals_without_outcome": 3
 },
 "H5-B": {
  "signals_without_outcome": 2
 },
 "H5-AvsB": {
  "signals_without_outcome": 3
 },
 "H5-SPY-h3-A": {
  "signals_without_outcome": 2
 },
 "H5-SPY-h3-B": {
  "signals_without_outcome": 1
 },
 "H5-SPY-h3-AvsB": {
  "signals_without_outcome": 2
 },
 "H5-SPY-h10-A": {
  "signals_without_outcome": 7
 },
 "H5-SPY-h10-B": {
  "signals_without_outcome": 2
 },
 "H5-SPY-h10-AvsB": {
  "signals_without_outcome": 7
 },
 "H6-SPY-top": {
  "signals_without_outcome": 1
 },
 "H6-SPY-h1-x2-top": {
  "signals_without_outcome": 1
 },
 "H6-SPY-h3-x2-top": {
  "signals_without_outcome": 1
 },
 "H6-SPY-h2-x1.5-top": {
  "signals_without_outcome": 1
 },
 "H6-SPY-h2-x3-top": {
  "signals_without_outcome": 1
 },
 "H6-QQQ-top": {
  "signals_without_outcome": 1
 },
 "H6-QQQ-h1-x2-top": {
  "signals_without_outcome": 1
 },
 "H6-QQQ-h3-x2-top": {
  "signals_without_outcome": 1
 },
 "H6-QQQ-h2-x1.5-top": {
  "signals_without_outcome": 1
 },
 "H6-QQQ-h2-x3-top": {
  "signals_without_outcome": 1
 },
 "H8-SPY": {
  "signals_without_outcome": 5
 },
 "H8-QQQ": {
  "signals_without_outcome": 3
 },
 "H12": {
  "signals_without_outcome": 4,
  "excluded_ex_dividend_gap_sessions": 18,
  "ex_dividend_non_gap_sessions_excluded_from_baseline": 32
 }
}
```

ledger trial_id (v3): f0094291be03
