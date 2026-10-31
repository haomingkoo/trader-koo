# Results, hypotheses version 8 (swing trades)

Recorded 2026-10-31 for the rules frozen in `hypotheses-v8-swing.md`
(registered in commit `8c41121e`, sha256
`d5a38c962d6799298c41f671b31c2c6f30b0eba4f444952f6c88c907e44419fd`, the same
hash the run wrote into its ledger config). Every number below is copied from
the bench trial ledger (`/Users/koohaoming/dev/.research/bench_trial_ledger.jsonl`)
or from the report the same run printed. A frozen copy of the two ledger lines
is `ledger-v8.jsonl` (sha256
`3b7ad5c6d4dfa273e1727c01b34034697892aa2246a93095ed2ae4e93b8f61f6`). This is a
research result on historical data. It is not a claim of alpha.

**Outcome: all three hypotheses fail. The sealed window was not opened.**

## Run inputs

| field | value |
|---|---|
| database | a copy of `trader_koo_research.db` in the session scratchpad, opened read-only |
| seal | `2023-06-12`, read from the ledger's seal entry; last session used 2023-06-09 |
| code | `a6dc45ad` (clean tree) for both runs |
| settings sha256 | `5840ccb14511` |
| H1 and H2 | trial `5cb082461412`, entries 2015-01-02 to 2023-06-09, seed 1078489200, 10,000 resamples of 5-session blocks, 1,000 placebo draws |
| H3 | trial `997681b13349`, 2011-01-04 to 2023-06-09, 150 names, review every 21 sessions, no buffer, no minimum hold, 200 random portfolios |
| commands | `python -m trader_koo.research.bench run --db <copy> --candidate v8_h3_52_week_high --start 2011-01-01 --top-n 150 --buffer-n 150 --review-every 21 --min-hold 0`, then `python -m trader_koo.research.bench.v8_run --db <copy> --h3-trial 997681b13349` |

Each command ran once. No rule or parameter was changed after a result was seen.

## Pass or fail per criterion

Holm family: H1, H2 and H3 (3 tests). Costs 10 bps per side unless stated.

| criterion | H1 earnings drift | H2 earnings gap | H3 52-week high |
|---|---|---|---|
| primary test, Holm p < 0.05 | **fail**: excess over equal weight +0.451% per trade, raw p 0.3970, Holm 1.0000, interval [-0.791%, +1.774%] | **fail**: -0.246% per trade, raw p 0.6743, Holm 1.0000, interval [-1.603%, +1.404%] | **fail**: 56.5th percentile of 200 random portfolios, p 0.4378, Holm 1.0000 |
| above the placebo or random 95th percentile | pass: 100.0th (p95 +2.663%) | **fail**: 91.6th (p95 +0.735%) | **fail**: 56.5th |
| consistency (60% of years) | **fail**: 44% of entry years positive against equal weight (4 of 9) | **fail**: 44% (4 of 9) | **fail**: beat SPY and equal weight in 31% of years |
| still positive at 20 bps | pass: +0.244% per trade | **fail**: -0.447% per trade | **fail**: stress CAGR 10.30% against 12.50% equal weight |
| beats its state baseline | pass: +1.305% per trade | pass: +0.622% per trade | not applicable |
| rubric score (calendar-time portfolio for H1 and H2) | 5.00 (raw 6.36), fail | 1.99, fail | 2.47, fail |
| sealed window | not opened | not opened | not opened |

## Per-trade numbers (trial `5cb082461412`)

| test | trades | mean net | state baseline net | excess vs baseline | excess vs equal weight | excess at 20 bps | SPY window | equal-weight window |
|---|---|---|---|---|---|---|---|---|
| H1, 60-session hold | 1,327 | +3.487% | +2.181% | +1.305% | +0.451% | +0.244% | +2.854% | +3.035% |
| H2, stop and 10-session trail | 93 | +0.588% | -0.034% | +0.622% | -0.246% | -0.447% | +0.483% | +0.834% |
| variant H1, 20-session hold | 1,381 | +1.877% | +0.754% | +1.123% | +0.493% (raw p 0.0826) | +0.289% | +1.136% | +1.384% |
| variant H1, AMC and unknown session only | 498 | +3.818% | +2.244% | +1.574% | +0.604% (raw p 0.4672) | +0.396% | +2.993% | +3.214% |
| variant H2, 20-session trail | 93 | +0.338% | +0.025% | +0.312% | -0.473% (raw p 0.5055) | -0.674% | +0.396% | +0.811% |

Variants are outside the Holm family and do not change any verdict.

H1 excess over equal weight by entry year: 2015 -1.206%, 2016 +4.786%,
2017 -0.378%, 2018 -1.804%, 2019 -2.232%, 2020 +0.528%, 2021 +3.370%,
2022 +2.159%, 2023 -3.677% (40 trades).

## Calendar-time portfolios (for the score)

| | H1 | H2 | SPY | equal weight |
|---|---|---|---|---|
| CAGR 2015-01-02 to 2023-06-09 | 12.2% | 0.5% | 11.1% | 10.8% |
| max drawdown | -49.62% | -44.94% | -32.03% | n/a |
| sessions invested | 100% | 25% | | |
| open trades on average | 37.5 | 0.4 | | |

H1 by calendar year (strategy against equal weight): 2015 -12.73% / -3.80%,
2016 +41.60% / +19.50%, 2017 +18.66% / +20.72%, 2018 -16.39% / -8.91%,
2019 +25.44% / +33.23%, 2020 +1.55% / +14.92%, 2021 +49.74% / +30.83%,
2022 +4.93% / -10.19%, 2023 +7.75% / +4.07%.

## H3 portfolio run (trial `997681b13349`)

CAGR 11.04% against equal weight 12.50% and SPY 12.38% (excess -1.46%/yr
against equal weight). Sharpe 0.72 against SPY 0.76. Max drawdown -35.3%
against SPY -33.7%. Turnover 3.43 a year. Stress CAGR 10.30%. The random
portfolios have a median of 10.94% and a 95th percentile of 11.77%. Rubric:
2.47 of 10, failing all three gates.

## Counts

- 19,108 earnings rows before the seal. 5,386 were for a ticker that was not
  an index member on the entry session. 148 had no session and were entered
  after the report date.
- H1: 13,661 eligible events, 1,482 in the top decile (1,386 from 2015 on, 59
  dropped because the 60-session hold ran past the window). 29 events lacked
  EPS or the close before entry.
- H1 timing: 111 BMO trades had higher volume on the next session than on the
  entry session, a sign the report may have come after the close.
- H2: 13,690 reaction sessions with a member entry; 100 qualified (95 from 2015
  on, 2 late). An independent plain-loop recount on stored prices gave the same
  95 entries.
- 943 of 13,910 member-quarters (6.8%) have no earnings row entering that
  quarter.

## What looked like an edge, and why it may not be one

H1 is the closest result in v8. Top-decile surprises beat the other earnings
events by +1.305% per 60-session trade. They beat the placebo draws and kept a
positive per-trade excess at 20 bps. The calendar-time portfolio returned
12.2% a year against 10.8% for equal weight. Reasons this is not evidence of
an edge:

- The test that decides, the excess over the equal-weight universe over each
  trade's own window, is +0.451% per trade with raw p 0.397. That is not
  distinguishable from zero.
- The gain sits in four years (2016, 2020, 2021, 2022). The other five lost to
  equal weight. In the calendar portfolio, 2016, 2021 and 2022 carry the whole
  gap; it lost to equal weight in 2015, 2017, 2018, 2019 and 2020.
- The calendar portfolio's drawdown was -49.6%, 17.6 points worse than SPY. The
  portfolio holds about 37 names chosen by one quarter's news, so it carries
  more risk than the index.
- The placebo is a weak control here. It draws random dates, so it does not
  match the calendar timing of the trades. The timing-matched comparison is
  the equal-weight window, and there the excess is small.
- The surprise is computed from yfinance values stored in 2026. Consensus
  estimates may have been revised after the report and actuals restated. That
  can only make the selection look better than it was in real time.
- Departed companies are missing from both price and earnings data.
- The AMC-only subset, which carries no timing look-ahead from a wrong BMO
  label, shows a similar excess (+0.604%, raw p 0.467). So mislabelled timing
  does not explain the result, but it does not rescue it either.

H2 traded 93 times in eight and a half years. A 10% earnings gap on three times
normal volume is rare among S&P 500 stocks. The sources scan all US stocks,
most of which are smaller than index members, and this universe does not cover
those.
Its trades beat the other reaction days by +0.622% but lost to equal weight.

H3 held 150 names that resemble the market and lost to it by 1.5% a year after
monthly turnover costs.

## Readings and deviations to note

- H3 holds 150 names, registered as 30% of about 500 members. On average only
  385 members per review had prices (190 of 745 window members have none), so
  150 is about 39% of the scored universe. The rule was not changed.
- Dual share classes count as two events (GOOG and GOOGL on 2015-07-17, for
  example). The registration is silent; nothing was merged.
- Membership is taken on the entry session for H1 and H2. The placebo uses
  membership on the session before entry, as v2 does.
- H1 events from mid-2014 set the 2015 Q1 cut-off; no 2014 event is a trade.
- The calendar-time portfolio holds cash at 0% and does not include trades
  dropped as late, so 2023 holds fewer trades. It is used only for the score.
- The rubric text calls the placebo draws "random portfolios". For H1 and H2
  the percentile is the placebo percentile of the mean per-trade return.
- `RESEARCH_STATUS.md` was not edited. It lists v1 to v3 only; v4 to v8
  results live in their results files.

## Candidates for v9 (not tested here)

A separate source review (`~/dev/.research/traders/swing_candidates.md`,
ranked shortlist) arrived after v8 was registered, so v8 was not changed.
These are candidates for a v9 registration:

- Qullamaggie episodic pivot with its fuller filter set: gap of at least 10%
  on earnings, volume of at least 3 times the 50-session average, positive EPS
  surprise and year-on-year quarterly EPS growth of at least 25%, no more than
  a 30% rise over the prior 63 sessions, and a green close. Entry on a buy-stop
  at the gap-day high within three sessions, half sold on day 5. Note the
  overlap with v8 H2 and H1.
- Stockbee 4% momentum burst: a 4% up day on volume above the prior day after
  a quiet day, held 3 to 5 sessions, with the 20 bps stress run central
  because costs dominate at that holding period.
- Qullamaggie breakout and a leader pullback to the 10 or 20 EMA.
- The file marks several thresholds (the universe filter, the base-range rule,
  the SMA50 trend filter) as the reviewer's own choice. A v9 registration must
  label them as the registrant's choice.
- The file's universe needs delisted names and small caps. This database has
  S&P 500 members only, with departed members largely missing, so a v9 would
  test these rules on a different universe from the one the traders used.

## Caveats

- One development path, 2015 to mid-2023 for H1 and H2 (9 entry years). The
  5-session block bootstrap does not capture longer dependence such as the
  2020 crash.
- Survivorship flatters every long-only result here, and the equal-weight
  benchmark shares that bias.
- A deterministic test or a historical replay proves software behaviour, not
  market performance. Nothing here supports a claim of alpha, a tradable edge,
  or generalisation (`RESEARCH_STATUS.md`).
