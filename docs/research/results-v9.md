# Results, hypotheses version 9 (stated rules of US Investing Championship traders)

Recorded 2026-10-31 for the rules frozen in `hypotheses-v9-champions.md`
(registered in commit `c56a5b51`, sha256
`39a522c5f25ea116e82e3c47b1b308be7abb712e6050f29daf0b9fd322eac334`, the same
hash the run wrote into its ledger config). Every number below is copied from
the bench trial ledger (`/Users/koohaoming/dev/.research/bench_trial_ledger.jsonl`),
from the report the same run printed, or from the diagnostics named in "Checks".
A frozen copy of the ledger line is `ledger-v9.jsonl` (sha256
`09b2d307ce173c688a5bb48c184a2c61f71afee6aae2f927e79b721748f0dd00`). This is a
research result on historical data. It is not a claim of alpha.

**Outcome: every hypothesis fails, H6 and H7 included. The H1a-vs-H1b
comparison shows no difference. H5 was skipped as registered. The sealed window
was not opened.**

## Run inputs

| field | value |
|---|---|
| database | a copy of `trader_koo_research.db` in the session scratchpad, opened read-only |
| seal | `2023-06-12`, read from the ledger's seal entry; last session used 2023-06-09 |
| code | `1ec196e5` (clean tree) |
| trial | `5a2c0be7aada`, seed 2471914935, params hash `93566db7adc1` |
| window | first entry 2015-01-02, last session 2023-06-09 (variants `_from2011` from 2011-01-03) |
| placebo | 500 draws per hypothesis; head-to-head bootstrap 10,000 resamples of 5-session blocks |
| command | `python -m trader_koo.research.bench.v9_run --db <copy> --ledger ~/dev/.research/bench_trial_ledger.jsonl --json-out <file>` |

The command ran once. No rule or parameter was changed after a result was seen.

## Universe limits

The universe is point-in-time S&P 500 members. These traders mostly trade
liquid non-S&P growth names: small and mid caps, recent IPOs and theme stocks
with daily ranges above 5%. Most S&P 500 members do not move like that, so the
signals are rare here (49 H1a gaps in 8.5 years). Departed members without
price history are missing, which flatters every long-only result. A fail here
does not refute the traders. A pass would not have shown that their skill is
what the rule captures.

## Pass or fail per criterion

Holm family: H1a, H1b, H1a vs H1b, H2, H3, H4, H6, H7 (8 tests). Costs 10 bps per
side unless stated. SPY CAGR 11.12% and equal-weight CAGR 10.65% over the same
sessions.

| criterion | H1a Flanders | H1b Qullamaggie | H2 J Law | H3 Sean Ryan | H4 Stockbee | H6 blend | H7 agreement |
|---|---|---|---|---|---|---|---|
| 1 Holm placebo p < 0.05 | **fail**: raw p 0.0120, Holm 0.0958 | **fail**: 0.3872, 1.0000 | **fail**: 0.0259, 0.1816 | **fail**: 1.0000, 1.0000 | **fail**: 0.9940, 1.0000 | **fail**: 0.9900, 1.0000 | **fail**: 0.8942, 1.0000 |
| placebo percentile (median, p95) | 99.0 (-0.20%, +0.30%) | 61.4 (-0.25%, +0.57%) | 97.6 (+0.22%, +1.25%) | 0.0 (+0.02%, +1.05%) | 0.6 (-0.91%, +3.20%) | 1.0 (-0.06%, +0.81%) | 10.6 (-0.40%, +0.37%) |
| 2 CAGR above SPY and equal weight | **fail**: +0.48% | **fail**: -0.13% | **fail**: +1.48% | **fail**: -2.64% | **fail**: -7.11% | **fail**: -1.43% | **fail**: -0.92% |
| 3 beats both in 60% of years | **fail**: 22% (2 of 9) | **fail**: 22% | **fail**: 22% | **fail**: 22% | **fail**: 0% | **fail**: 11% | **fail**: 22% |
| 4 stress (20 bps) CAGR above both | **fail**: +0.35% | **fail**: -0.58% | **fail**: +1.37% | **fail**: -3.49% | **fail**: -15.53% | **fail**: -3.57% | **fail**: -1.34% |
| 5 mean net R > 0 | pass: +0.27R | **fail**: -0.03R | pass: +1.20R | **fail**: -0.12R | **fail**: -0.05R | not applicable | **fail**: -0.12R |
| rubric score | 3.00, fail | 1.51, fail | 4.00, fail | 1.00, fail | 0.00, fail | 1.00, fail | 1.00, fail |
| sealed window | not opened | not opened | not opened | not opened | not opened | not opened | not opened |

The years "beaten" are mostly 2018 and 2022. In those years the rules sat
mostly in cash while both benchmarks fell.

H1a vs H1b, mean net R per signal over all signals: H1a +0.27R (43 signals),
H1b -0.03R (127), difference +0.30R, raw p 0.2618, Holm 1.0000, interval
[-0.22R, +0.83R]. **No difference shown.** The point estimate favours
Flanders's gap into a high over Qullamaggie's gap after a quiet period.

## Portfolios

| | trades | CAGR | max drawdown | Sharpe | CAGR at 20 bps | sessions invested | mean open positions |
|---|---|---|---|---|---|---|---|
| H1a | 43 | +0.48% | -3.74% | 0.36 | +0.35% | 13% | 0.15 |
| H1b | 127 | -0.13% | -7.05% | -0.04 | -0.58% | 27% | 0.32 |
| H2 | 34 | +1.48% | -3.46% | 0.70 | +1.37% | 27% | 0.32 |
| H3 | 198 | -2.64% | -21.08% | -1.46 | -3.49% | 13% | 0.08 |
| H4 | 3,517 | -7.11% | -62.75% | -0.33 | -15.53% | 98% | 5.05 |
| H6 | (sleeves) | -1.43% | -18.17% | -0.36 | -3.57% | 99% | |
| H7 | 180 | -0.92% | -11.45% | -0.43 | -1.34% | 29% | 0.30 |
| SPY | | +11.12% | -33.70% | 0.66 | | | |
| equal weight | | +10.65% | -40.05% | 0.61 | | | |

Small risk per trade (0.3% to 1%) and rare signals leave H1a, H1b, H2, H3 and
H7 in cash most of the time. Their CAGRs are small for that reason. The
placebo is matched for timing, count, stop distance and exposure, so it tests
stock selection alone.

## Trade-level statistics (trades each portfolio took, net R at 10 bps)

| | trades | win rate | mean win | mean loss | mean R | top 5% share of net profit | top 5% share of gross profit | median hold |
|---|---|---|---|---|---|---|---|---|
| H1a | 43 | 42% | +1.79R | -0.83R | +0.27R | 116% | 40% | 6 |
| H1b | 127 | 37% | +1.61R | -1.00R | -0.03R | (net < 0) | 40% | 6 |
| H2 | 34 | 24% | +8.63R | -1.09R | +1.20R | 98% | 61% | 10 |
| H3 | 198 | 21% | +0.30R | -0.23R | -0.12R | (net < 0) | 55% | 1 |
| H4 | 3,517 | 42% | +1.07R | -0.88R | -0.05R | (net < 0) | 40% | 5 |
| H7 | 180 | 29% | +1.24R | -0.68R | -0.12R | (net < 0) | 49% | 3 |

The top 5% share is the net profit of the best ceil(5% of n) trades over the
total net profit. It is printed as a negative percentage in the run report when
the total is negative. "(net < 0)" marks those rows here.

With every signal traded once and no portfolio limits, the counts and means
are the same for H1a, H1b, H2 and H3: no signal was skipped for room. H4 has
9,867 signals with mean -0.03R (6,153 were skipped in the portfolio for lack
of room). H7 has 193 signals with mean -0.13R. In the run's printed report, the
"median hold" column of the all-signals table shows 1 for every rule. That was
a reporting bug: the field was given a constant. It was fixed after the run,
and the true medians from the same code path are H1a 6, H1b 6, H2 9.5, H3 1,
H4 5 and H7 3. No result depends on it.

The traders describe a low win rate with large winners. H2 shows that shape:
24% wins, mean win +8.63R, and two trades carry the whole profit. H1a's win
rate of 42% is close to J Law's stated 40-50%, not to Flanders's 30%.

## H6: the sleeve blend

Correlation of the five sleeves' daily returns (2015-01-02 to 2023-06-09):

| | H1a | H1b | H2 | H3 | H4 |
|---|---|---|---|---|---|
| H1a | 1.00 | 0.34 | 0.09 | 0.06 | 0.07 |
| H1b | 0.34 | 1.00 | 0.05 | 0.05 | 0.22 |
| H2 | 0.09 | 0.05 | 1.00 | 0.06 | 0.13 |
| H3 | 0.06 | 0.05 | 0.06 | 1.00 | 0.07 |
| H4 | 0.07 | 0.22 | 0.13 | 0.07 | 1.00 |

The sleeves are close to uncorrelated. The two earnings-gap sleeves share the
most (0.34), because some gaps qualify for both. Low correlation did not help
the blend. Its maximum drawdown (-18.2%) is only slightly smaller than the
average sleeve drawdown (-19.6%). Its Sharpe (-0.36) is below the average
sleeve Sharpe (-0.15). H4 is the only sleeve that is almost always invested,
and it lost 7.1% a year, so it dominates the blend's risk and return. Combining
four mostly idle sleeves with one losing sleeve diversifies the risk without
adding return.

## H7: two-family agreement

317 agreements; 240 had a trade from the triggering rule (F1 87, F2 9, F3 77,
F4 67) and 180 were taken. 82 agreements are two families firing on the same
session, and 77 of those are an earnings gap (F1) that is also a 4% burst (F4).
So about a quarter of "agreement" is one event read by two rules, not two
independent views. The agreement trades lost (-0.12R mean, 10.6th placebo
percentile).

## Variants (reported only, no placebo, outside the Holm family)

| variant | trades | CAGR | max drawdown | mean R | win rate |
|---|---|---|---|---|---|
| H1a_since2010 (high since 2010, at least 5 years) | 23 | +0.30% | -2.94% | +0.23R | 43% |
| H1a_code33 | 0 | 0.00% | 0.00% | n/a | n/a |
| H1a_flat (0.5% risk, no tiers or brake) | 43 | +0.46% | -3.79% | +0.27R | 42% |
| H1a_trail20 | 43 | +0.34% | -3.98% | +0.18R | 42% |
| H1a_spy50 | 36 | +0.47% | -2.69% | +0.33R | 42% |
| H1b_spy50 | 77 | -0.50% | -7.89% | -0.11R | 32% |
| H2_spy50 | 26 | +1.63% | -3.32% | +1.76R | 23% |
| H3_from2011 | 262 | -2.17% | -24.73% | -0.11R | 22% |
| H3_spy50 | 156 | -2.25% | -18.56% | -0.13R | 22% |
| H4_day3 | 4,245 | -7.19% | -57.23% | -0.03R | 44% |
| H4_from2011 (SPY 12.49%, EW 12.88%) | 4,879 | -5.46% | -63.13% | -0.05R | 43% |
| H4_spy50 | 2,487 | -8.28% | -54.26% | -0.07R | 41% |

- Code 33 matched none of the 49 H1a signals. On a random sample of 150
  tickers at three dates, it held on 1 of 450 ticker-dates. The rule is very
  strict on quarterly data, and the variant says nothing.
- Flanders's tiers and monthly brake changed almost nothing (H1a +0.48% against
  +0.46% flat). The brake never fired, because H1a rarely holds more than one
  position.
- The 20-session trail did worse than the 10-session trail on H1a (+0.18R
  against +0.27R). That is the opposite of Flanders's own exit study, on far
  fewer trades.

## H5: Luk's IWM claim

Skipped as registered. IWM's first row is 2025-06-02 (352 rows), and the
database starts in 2010, not 2008. Nothing was fetched.

## Checks

- An independent plain-loop recount of the registration text on the stored
  prices gave the same 49 H1a and 142 H1b signals. For H4 it gave 13,855
  against the code's 13,862. All 7 extra signals fall on 2010-01-08 to
  2010-01-13 warm-up rows that the loop skipped, outside the window.
- A loosened-threshold integration test (`tests/test_research_v9.py`) makes
  every rule trade on real bars and checks that corrupting every row after the
  seal changes no result.
- The two best H2 trades (FDX entered 2020-07-27, +28.0R; IT entered
  2021-06-17, +12.9R) and the best H1a trades (STX 2017-01-26, +7.5R; NFLX
  2018-01-24, +4.7R) were inspected. They are real moves with stops 1-3% below
  entry, not data errors.

## What looked like an edge, and why it may not be one

**H1a (Flanders's earnings gap into a 52-week high with top RS)** beat its
placebo: 99.0th percentile, raw p 0.012, mean +0.27R per trade. It does not
survive Holm (0.096), and it is not an edge worth trading:

- 43 trades in 8.5 years. The best two trades make more than the whole net
  profit (sum +11.5R; without STX and NFLX, -0.7R).
- Its CAGR is +0.48% a year, invested 13% of the time. Scaling the risk up to
  match the market's exposure would multiply the drawdowns with the returns,
  on a sample this small.
- Earnings estimates and actuals are yfinance values stored in 2026 (possible
  revisions). H1a does not use the surprise, so this matters for H1b and H2's
  earnings leg only.

**H2 (J Law's first pullback after a catalyst)** has the traders' shape: 24% wins,
mean win +8.63R, mean +1.20R, 97.6th placebo percentile, raw p 0.026. Reasons
it is not evidence:

- 34 trades. Two trades (FDX in 2020, IT in 2021) carry 98% of the net profit;
  without them the sum is -0.09R. Four trades above +10R hold the rest. A
  result that rests on two to four trades is one outcome of a fat-tailed
  draw, not a rate.
- Both big winners came in the 2020-21 rally after the COVID low, one regime.
- The Holm-adjusted p is 0.18, and CAGR is +1.48% against 10.65% for equal
  weight.

The traders say the same thing about themselves (Flanders "probably break
even" without his top five trades). On daily bars of S&P 500 members, those
few big trades are too rare to tell skill from luck.

## Readings and deviations to note

- **H3's registered RSI divergence sells the breakout itself.** On 99 of 198
  H3 trades, the divergence fired on the fill session or the next one. A
  second-move breakout is a new 20-session high whose RSI sits below the RSI
  at the burst's peak, so the rule as written exits on day 2. With the day-one
  red-close exit (103 trades), H3's median hold is 1 session and its mean win
  only +0.30R. H3 as registered does not test Sean Ryan's method. It tests a
  definition of divergence that does not fit his entry. The rule was not
  changed; a fixed definition is a v10 candidate.
- "The third move" was not modelled, as registered.
- H4's portfolio was full most of the time and skipped 6,153 signals for lack
  of room. Its order is by dollar volume, so it took the largest names.
- H1a and H1b had no earnings exits: every trade ended before the next report.
- Prices are the bench's adjusted index levels, so entries print as small
  numbers (FDX 2.159). Only ratios are used.
- The all-signals median-hold column was a constant in the printed report and
  was fixed after the run (above).
- `RESEARCH_STATUS.md` was not edited. It lists v1 to v3 only; later results
  live in their results files.

## Candidates for v10 (not tested here)

- H3 with a divergence measured only against highs made after entry (or after
  the rest), so it cannot fire on the breakout. Also a "third move" defined as
  a new +20% leg after a rest of at least N sessions since entry.
- H2 on a universe where catalyst moves are common (liquid non-S&P growth
  names). That needs delisted small and mid caps, which the database lacks.
- H7 counting two families only on different sessions, so one event read by
  two rules does not count as agreement.
- Luk's IWM claim, if IWM history back to 2008 is loaded into the research
  database by its own loader.

## Caveats

- One development path, 2015 to mid-2023 (9 entry years). Signals are rare, so
  every per-rule result rests on tens of trades, except H4.
- Survivorship flatters every long-only result. The equal-weight benchmark
  shares that bias.
- Daily bars cannot reproduce the traders' intraday entries, 1-3% stops,
  pyramiding or discretion. Their USIC records come from self-selected,
  survivor-only standings.
- A deterministic test or a historical replay proves software behaviour, not
  market performance. Nothing here supports a claim of alpha, a tradable edge,
  or generalisation (`RESEARCH_STATUS.md`).
