# Pre-registered hypotheses, version 1

Registered 2026-10-26, before any of these were run on the research bench.
The git commit that adds this file is the registration record. Rules here are
frozen: a change is a new version with a new file, and results of this version
are reported regardless.

Owner's rules: no hindsight trading, no data leakage, a hypothesis with a reason
before any test, fundamentals-driven swing trades held weeks to months.

## Common rules (all hypotheses)

- **Universe**: S&P 500 members on the signal date (`sp500_membership`), with a
  price that day. Members with no price data are counted and reported.
- **Signal date**: every 10th trading session. A signal uses only data
  available at that session's close: prices adjusted only with corporate actions
  known by then, SEC facts usable from the session after their filing date.
- **Execution**: next session's open. Long only. 20 names, equal weight.
  A holding is sold only if it falls out of the top 40, and is held at least
  20 sessions. Cash earns nothing.
- **Costs**: 10 bps per side. Stress run at 20 bps per side.
- **Benchmarks**: SPY total return; the equal-weight return of the same
  universe on the same schedule; 200 random 20-name portfolios under the same
  rules (seeded, recorded).
- **Periods**: walk-forward, reported by calendar year from 2011. The last 20%
  of sessions in the research database at registration is sealed and opened
  once, for all hypotheses together, after the walk-forward results are in.
- **Pass** (all required, net of costs):
  1. beats SPY and the equal-weight universe in at least 60% of walk-forward
     years;
  2. ranks above the 95th percentile of the random portfolios;
  3. stays ahead of both benchmarks at 20 bps per side;
  4. stays ahead of both benchmarks in the sealed window.
- **Fail** is reported with the same detail as a pass.

## H1. Quality at a fair price

- **Claim**: profitable, financially strong companies bought when not expensive
  beat the market over the following 1 to 3 months.
- **Why**: investors overpay for exciting growth stories and underpay for
  steady profit (published as the profitability and quality premium).
- **Score**: equal-weight average of cross-sectional z-scores, winsorised at
  plus or minus 3:
  - gross profit (TTM) / total assets;
  - minus total liabilities / total assets;
  - (operating cash flow TTM minus capital expenditure TTM) / market cap.
  Market cap = close times the latest filed share count.
- **Excluded**: companies missing any input (mainly banks and insurers, which
  report no gross profit). The count is reported.
- **Falsified if**: it fails the common pass rule.

## H2. Accelerating fundamentals

- **Claim**: companies whose growth is accelerating, as filed, keep
  outperforming for about two months.
- **Why**: analysts and investors update slowly on a change in trend
  (published as fundamental momentum and earnings drift).
- **Score**: equal-weight z-score average of:
  - revenue growth acceleration: year-on-year TTM revenue growth now minus the
    same measure one quarter earlier;
  - year-on-year growth in diluted EPS (TTM).
- **Excluded**: companies with fewer than 8 quarters of filed data.
- **Falsified if**: it fails the common pass rule.

## H3. Momentum confirmed by earnings

- **Claim**: stocks with strong 12-month returns whose earnings are also growing
  keep outperforming.
- **Why**: price strength backed by improving earnings is less likely to be a
  sentiment move that reverses.
- **Rule**: among names with positive year-on-year diluted EPS growth (TTM), rank
  by total return from 252 to 21 sessions before the signal date.
- **Falsified if**: it fails the common pass rule.

## H4. Market leaders keep leading

- **Claim**: the largest companies by market cap beat the index the next year.
- **Why**: network effects and index money flowing to the largest names.
- **Rule**: on the last session of each year, hold the 5 largest members by
  market cap, equal weight, for the next calendar year (this hypothesis uses
  its own annual schedule and 5 names instead of the common 20-name rule).
- **Hindsight guard**: the idea came from watching 2018 to 2026. The pass rule
  is applied to 2011 to 2017 only, plus prospective results from registration.
  2018 to 2026 is shown separately and labelled "in-sample, inspiration period".
- **Falsified if**: it fails the pass rule on 2011 to 2017.

## H5. Fade the crowd (prospective only)

- **Claim**: heavily shorted names lag over the following 1 to 3 months.
- **Why**: short sellers are on average the informed side.
- **Rule**: from the 20-name H1 portfolio, drop names in the top decile of
  short float (Finviz) on the signal date; compare with H1 itself.
- **Data limit**: short float history starts 2026-02, so this is tested only
  prospectively from registration. No backtest result will be reported.

## Reference runs (not hypotheses)

Run alongside, for comparison only: equal weight of the universe, plain 12-1
momentum, low volatility (lowest 252-session volatility). They are not judged
against the pass rule.
