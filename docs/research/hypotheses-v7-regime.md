# Pre-registered hypotheses, version 7 (market-regime exposure rules)

Registered 2026-10-30 (SGT), before any of these rules was computed or run.
The git commit that adds this file is the registration record. Rules are
frozen: a change is a new version, and results are reported regardless.

v1 to v6 chose stocks and stayed fully invested. These rules choose how much of
the portfolio to hold in the market (SPY) and how much in cash. Thresholds are
taken from the cited sources or from the owner's studied traders, not fitted.

## Data (all in `trader_koo_research.db`)

- SPY and ^VIX daily bars (`price_daily`), total-return adjusted by the bench
  loader (dividends reinvested), so holding SPY and holding cash compete fairly.
- ^IRX (13-week T-bill rate, percent) for the return on cash.
- `macro_series`, loaded by `trader_koo/research/macro.py`:
  - ALFRED vintages of M2SL, CPIAUCSL and DTWEXBGS. A value is used only from
    its `known_from` date, the day FRED first showed it.
  - T10Y2Y from the current FRED file, each value known from the next calendar
    day (market data, not revised; ALFRED refuses the full vintage request).
  - CBOE VIX3M, VIX9D, SKEW and VVIX daily closes, known at their own close.

## Common rules

- **Decision**: at the close of session t, using only data known by then.
- **Execution**: the new exposure applies from the open of t+1. The overnight
  move from close t to open t+1 is earned at the old exposure.
- **Exposure**: between 0 and 1. The rest is cash earning ^IRX / 100 / 252 per
  session, using the rate known at the previous close.
- **Costs**: 10 bps on each change in exposure, stress run at 20 bps.
- **Window**: development 2011-01-03 to the pinned seal (2023-06-12); the
  sealed window is opened once, only for rules that pass criteria 1 to 5.
- **Random controls**: 1,000 circular shifts of the rule's own exposure series
  by a random offset of at least 252 sessions (seed recorded). A shift keeps
  the time in the market and the number of switches and moves only the timing.

### Pass rule (all required, net of costs)

1. Sharpe ratio (daily returns over cash, annualised) above SPY's.
2. Maximum drawdown at least 5 percentage points smaller than SPY's.
3. CAGR no more than 1.0 percentage point below SPY's.
4. Sharpe above the 99.55th percentile of the random controls (5% split over
   the 11 hypotheses, Bonferroni).
5. Criteria 1 to 3 still hold at 20 bps.
6. Sealed window: Sharpe above SPY's and drawdown not worse than SPY's.

### Score (10 points; capped at 5 when any of criteria 1 to 5 fails)

| Criterion | Measure | 0 points at | Full points at | Points |
|---|---|---|---|---|
| Risk-adjusted return | Sharpe minus SPY Sharpe | -0.10 | +0.20 | 3 |
| Drawdown protection | SPY max drawdown minus rule max drawdown | 0 pts | 15 pts | 2 |
| Return cost | CAGR minus SPY CAGR | -3.0 pts | 0 | 2 |
| Not luck | Sharpe percentile vs random controls | 50th | 99.55th | 2 |
| Cost robustness | criteria 1 to 3 at 20 bps | fail | pass | 1 |

Output order: reasoning, pros, cons, gates, score.

## Hypotheses (exposure 1 unless the condition holds, then 0)

| # | Rule | Condition for exposure 0 | Source of the threshold |
|---|---|---|---|
| H1 | VIX term structure | VIX close / VIX3M close > 1.0 (backwardation) | practitioner convention; no peer-reviewed study found |
| H2 | VIX breakout (John) | VIX close >= 20 and above its 20-session average | the studied trader's stated "breakout above 20" |
| H3 | Variance risk premium | VIX minus SPY realised volatility (21-session close-to-close, annualised, in VIX points) below 0 | Bollerslev, Tauchen and Zhou (2009): a low premium precedes low returns |
| H4 | Distribution days | 5 or more of the last 25 sessions with SPY down at least 0.2% on higher volume than the session before | O'Neil; 4 to 5 days is the stated warning, 5 is used |
| H5 | 10-month trend | at the last session of each month: SPY close below the average of the last 10 month-end closes; held through the next month | Faber (2007) |
| H6 | 12-month time-series momentum | at the last session of each month: SPY 12-month total return below the 12-month return on cash; held through the next month | Moskowitz, Ooi and Pedersen (2012) |
| H7 | Inflation regime | CPI year-on-year at or above 5% and above its value 3 months earlier, as first published | Neville, Draaisma, Funnell, Harvey and Van Hemert (2021) |
| H8 | Real money growth | M2 year-on-year growth below CPI year-on-year growth, both as first published | money-and-liquidity view; no single study, threshold 0 |
| H9 | Strong dollar | broad dollar index up more than 5% over 126 sessions | exploratory; threshold set here, no study |
| H10 | Inverted yield curve | 10-year minus 2-year Treasury below 0 | recession-signal literature; threshold 0 |
| H11 | Vote | exposure = share of H1 to H10 that are at exposure 1 that day | equal-weight combination, no fitting |

Monthly macro series (H7, H8): year-on-year means the latest observation known
by t against the observation 12 months earlier, both from the same vintage.
A rule that has no value on a day (data not yet started) holds exposure 1 that
day, and the count of such days is reported. For H1 that is before 2009-09-18,
outside the window.
