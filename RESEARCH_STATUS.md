# Research status

This file states what the repository's current evidence supports. It is not a
performance claim.

## Backtest status

- The next-open execution and accounting framework is implemented and covered
  by deterministic tests.
- One copied-local-database descriptive study ran with 50 selected setup calls,
  four closed trades, one traded signal date, and 11 daily observations. It
  returned -0.2066% net, but explicitly failed causal-validity and
  decision-eligibility gates. It is not a qualified backtest.
- The three-challenger tournament ran only its data audit. It stopped
  `blocked_before_validation`; all challenger metrics and the split are null,
  no winner was selected, and the sealed held-out data was not opened.
- Release replay evidence uses a synthetic contract fixture. It verifies code
  and accounting behavior, not trading performance.
- Campaign v2 is designed as a prospective, deterministic paper campaign. Read
  its live state from the campaign admin status endpoint rather than this file.
  It may record simulated trades only when a canonical report publishes before the
  intended next-open cutoff and all configured setup and risk gates pass.
- Campaign v2 changed admission rules mid-campaign without a new policy version.
  PR #303 (merged 2026-10-25) blocks a family when its expectancy OR hit rate
  is below threshold, and blocks when regime or VIX data is missing. PR #302
  puts the book and SPY on the same return basis. Trades admitted before the
  first deploy that contains #303 ran under the old rules, so analyze the two
  periods separately. The deploy date is the first dark-deploy run of a commit
  that includes #303.
- Setup score shows no within-day cross-sectional signal: IC near zero, and the long mean is a
  market factor. Eval returns use the signal-date close with no costs, while the
  paper path charges 10bps per side.
- yfinance survivorship measurement: a large share of 2016-2026 S&P 500 names are gone,
  many sampled departed names return zero bars, and tickers get recycled
  (FB, APC). Both findings support the no-alpha language below.
- The LightGBM stack was removed. It never had a qualified model and never
  vetoed, ranked, or sized Campaign v2 trades.
- Hypotheses version 1 (`docs/research/hypotheses-v1.md`) ran walk-forward on a
  separate research database, 2011-01-04 to 2023-06-09, net of 10 bps per side,
  with the last 20% of sessions sealed. H1 failed criterion 1 (46.2% of years
  beating both benchmarks), H2 and H3 failed criteria 1 to 3, and H4 failed
  criteria 1 to 3 on its graded 2011 to 2017 period. No hypothesis passed, so the
  sealed window was not opened. The H4 result used a BRK-B share count of the
  wrong class, and the universe excludes departed members that have no price
  history. Values, trial ids and caveats: `docs/research/results-v1.md`, with the
  frozen ledger lines in `docs/research/ledger-v1.jsonl`. This is a software and
  data result, not evidence of alpha.
- Hypotheses versions 2 and 3 (`hypotheses-v2-saty.md`, `hypotheses-v3-market-timing.md`) ran as event studies on the
  development window only (2010-01-04 or 2011-01-03 to 2023-06-09), on a copy of the research database, with the sealed
  window unopened. v2 S1 to S3 all failed their pass rules (S3 beat its state baseline but not the placebo, and was negative
  at 20 bps). In v3 only H11 (Friday VIX declines against other weekdays) met the development criteria, the stated 85% of H2
  was falsified, H4, H10 and one H6 cell were insufficient, and the rest failed. No result is evidence of alpha; values,
  trial ids and the readings taken where the documents were silent: `docs/research/results-v2-v3.md`.

The checked-in challenger artifact is immutable historical evidence. Its
negative result is useful: the fixed current-universe dataset lacks verified
total-return basis, point-in-time membership, and complete price-revision
lineage, so the runner correctly refused to produce performance metrics. The
runner has since been deleted; the artifact is served read-only and can no
longer be re-verified against its implementation hash.

## Supported language

It is accurate to say that Trader Koo has a fail-closed backtest framework and
that its current evidence gates prevented an invalid tournament from being
reported as a result.

Do not claim alpha, profitability, a challenger winner, held-out performance,
generalization, risk-adjusted improvement, or an LLM contribution to returns.
