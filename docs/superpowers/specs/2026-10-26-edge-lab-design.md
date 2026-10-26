# Edge lab design: finding a real fundamentals swing-trading edge

Status: draft for owner approval. Date: 2026-10-26.

## Why

The owner wants trader-koo to find real edge. The current technical setups show
none: rank IC about 0.01 over 134 signal days, negative after costs and after
removing the market move, and 1 win in 9 paper trades. The research behind this
design (three read-only reviews, 2026-10-26) found three reasons:

- Nothing was ever tested properly. Features were added without measuring them.
- The data under any test was biased: today's index members, no dividends,
  unadjusted corporate actions.
- The signals were the most crowded ones in the market (chart patterns, moving
  averages, RSI), which fast algorithmic traders arbitrage away.

## Goal

Find, with honest evidence, whether any signal or combination produces a
positive net return over the market for **fundamentals-driven swing trades in
S&P 500 stocks, held weeks to months**. Then paper-trade only what survives.

The owner's direction, in their words: "real fundamentals gain", "long swing
trading". Holding period about 20 to 60 trading days; positions reviewed every
1 to 2 weeks.

## Non-goals

- Intraday or short-horizon trading (days). Algorithms own that horizon and our
  execution is next-day open.
- Live broker orders. Paper only.
- Promising a result. A clean "no edge found" is an acceptable outcome.

## Anti-goals

- A backtest that looks good because it was tuned on the data it is scored on.
- Any LLM result measured on dates the model may have seen in training.
- Treating a 9-year mega-cap tech run as a law of nature.
- More machinery than evidence: every component must be measured or removed.

## Data foundation (being built now, separate PRs)

1. Point-in-time S&P 500 membership (`sp500_membership`), from the fja05680
   dataset cross-checked against Wikipedia.
2. Total-return daily series computed from stored splits and dividends, with
   special dividends and unexplained jumps flagged and excluded.
3. Point-in-time fundamentals from SEC EDGAR XBRL (`sec_fundamentals`): each
   value usable only from its filing date, restatements kept separately. About
   2009 onward.
4. Collectors that cannot be backfilled later: full-universe daily options with
   30/60/90-day expiries, and a permanent earnings history table.

Known, stated limit: prices for most companies that left the index since 2018
are missing (20 of about 195 present). Every result must report this bias and
its likely direction (it flatters long-only and momentum results).

## The bench

One harness in `trader_koo/research/`, reusing the existing next-open execution
code (`next_open_baseline.simulate_portfolio`) rather than adding a sixth
simulator.

- **Universe**: index members on each date (point-in-time), with available
  prices.
- **Signal timing**: a signal on date t uses only data knowable at the close of
  t (prices to t, fundamentals with `filed_date <= t`). Entry at the next open.
- **Costs**: 10 bps per side plus 20 bps stress case; short borrow cost for any
  short leg.
- **Returns**: total return, measured against SPY total return and against an
  equal-weight universe return.
- **Validation**: expanding walk-forward by year; purge and embargo equal to the
  holding period; the final 20% of history sealed as held-out and opened once
  per candidate family.
- **Overfitting control**: a trial ledger records every variant run (signal,
  parameters, universe, dates). Report deflated Sharpe and probability of
  backtest overfitting over the ledger.
- **Controls**: random-entry control on the same dates and holding period;
  equal-weight universe; SPY.
- **Report per candidate**: net annual excess return, Sharpe and its confidence
  interval (block bootstrap), max drawdown, turnover, hit rate, performance in
  up and down markets, and the survivorship caveat.

## Candidates, in test order

Grouped by the reasoning behind them. Every candidate is pre-registered (formula,
horizon, rebalance) in the trial ledger before its first run.

**Fundamentals (the owner's focus, testable 2010 onward from SEC data)**
1. Quality / profitability: gross profit to assets, ROIC, accruals.
2. Value: earnings yield, free-cash-flow yield (better than book-to-market in
   large caps).
3. Fundamental momentum: year-over-year growth and acceleration in revenue and
   earnings, as filed.
4. Balance-sheet strength: leverage, net cash.
5. Composite: equal-weight z-score of 1 to 4.

**Price-based (testable 2018 onward now)**
6. 12-1 month momentum and industry momentum, with a volatility-scaled crash
   guard.
7. Low volatility (judged on risk-adjusted return).
8. Size leadership (owner's hypothesis): hold the top 5 by market cap, tested
   pre-2018 and post-2018 separately, with a pre-written stop rule (lags SPY two
   years running).

**Events and flows**
9. Earnings-window premium and earnings-announcement return drift.
10. Index additions and deletions (mechanical flows).

**Options (pilot only until history accumulates)**
11. Change in implied volatility and call-put IV spread, monthly.

**Existing trader-koo signals, as challengers**
12. Current setup score; YOLO patterns. Expected null; must beat the random
    control to stay.

**Combination**: an equal-weight z-score composite of the survivors. Machine
learning only after the composite shows a signal, and only with purged
time-ordered validation.

## Agent debate (built after the bench has a best signal)

The debate is a layer on top of the best quantitative signal and never replaces
it. Design from the research review:

- **Roles**: analysts (fundamentals and earnings, options, price, news) on a
  cheap model; a bull and a bear on fixed sides with different prompts or
  models; a skeptic who attacks the quantitative signal; a judge on the
  strongest model.
- **Tools**: read-only, parameterized queries over the point-in-time data
  (no free-form sandbox at first). Each tool result gets an evidence id.
- **Protocol**: opening arguments plus one rebuttal each. A rebuttal must cite a
  new evidence id or concede. Structured JSON output from the judge, with every
  numeric claim carrying an evidence id.
- **Grounding check**: deterministic code matches every claimed number to its
  tool result. Too many failures rejects the run and falls back to the plain
  signal, labelled as such.
- **Scope and cost**: shortlist only (about 25 to 50 names per review date).
  Estimated 10 to 30 USD per day; verify against current pricing.
- **Evaluation**: prospective only, on dates after each model's training cutoff.
  Decisions are written to an append-only ledger before outcomes exist. Arms on
  identical inputs: plain signal, single agent with matched budget, debate.
  Ablate roles and rounds. Fixed sample-size gate written down before starting.
- **Status**: advisory until it beats the plain signal on the pre-registered
  gate.

## Promotion to paper trading

A candidate becomes a paper strategy only if it beats the random control and
SPY net of costs in walk-forward and in its single held-out test, survives the
deflated-Sharpe check, and the owner approves it as a new campaign policy
version. Expected honest range for a good composite: net Sharpe about 0.3 to
0.7. A backtest above 1.0 is treated as a bug until proven otherwise.

## Decisions from the builder check

Calls marked "builder call" were settled by the author and can be overridden.

- **Market cap (answered).** Shares outstanding come from SEC
  `dei:EntityCommonStockSharesOutstanding` and `CommonStockSharesOutstanding`,
  usable from their filing date. Market cap on date t is the split-adjusted
  close times the latest filed share count, adjusted for splits after filing.
- **Price history depth (builder call).** Add a price backfill to 2010 for
  every ticker that was an index member at any time since 2010 and that
  yfinance still serves, stored with the same basis contract. This gives the
  fundamentals candidates about 16 years and makes the pre-2018 size test
  possible. The survivorship gap is reported per year.
- **Held-out data (builder call).** One shared sealed window: the last 20% of
  sessions. It is opened once, at the end, for all surviving candidates and the
  composite together, with Holm correction across them. Families are the five
  groups above. Walk-forward results before the seal decide which candidates
  reach the final test; nothing is re-tuned after opening it.
- **Portfolio construction (builder call).** Primary test: long-only, top 20
  names by signal among point-in-time members, equal weight, reviewed every 10
  trading days, with a buffer (a holding is sold only when it falls out of the
  top 40) and a 20-session minimum hold. Benchmarks are plain excess return over
  SPY total return and over the equal-weight member universe. A top-minus-bottom
  quintile spread is reported as a diagnostic of signal strength, not as a
  tradable result.
- **Simulator prices (builder call).** The data layer produces back-adjusted
  OHLC (open and close scaled by the same split and dividend factor) so
  `simulate_portfolio` sees one consistent basis. The simulator is unchanged.
- **Thresholds (builder call, owner to confirm).** A candidate passes when, net
  of 10 bps per side: walk-forward excess return over SPY is positive in at
  least 60% of years; deflated Sharpe probability is at least 0.95; probability
  of backtest overfitting is at most 0.2; it beats the 95th percentile of 1,000
  random-entry portfolios built on the same dates and rules; it stays positive
  at 20 bps per side; and it passes the held-out test after Holm correction.
- **Debate (answered).** Not in the first plan. It gets its own plan once the
  bench has a best signal. Models are Claude; the agents' cost cap and the
  prospective sample-size gate are set in that plan.
- **Options and earnings candidates (builder call).** Code is built but runs
  only once there are at least 12 months of full-universe options snapshots,
  and at least 2,000 stored earnings events with dates.

## Out of scope for this design

The UI for showing results and debates (separate design), and changes to the
current campaign v2 (it keeps running unchanged).
