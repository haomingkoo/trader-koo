# Trader Koo Super Upgrade Plan

Goal: move Trader Koo toward an Unusual Whales + Rallies-style research product without depending on options flow or expensive net-premium feeds.

## Current Sprint

- [x] Paper trading leakage review and baseline honesty fixes.
- [x] Hyperliquid fake/hardcoded display cleanup and Telegram HTML escaping.
- [x] Faster Telegram command polling with long polling.
- [x] Faster configurable monitoring intervals for Polymarket, macro, Hyperliquid, crypto, and health checks.
- [x] Free official macro RSS feeds added to the existing news layer.
- [x] Elegant web visual foundation pass.
- [x] Paper-trade critic infrastructure now fails closed by default.
- [x] Options IV/OI rank research context added from local Yahoo/yfinance snapshots, with explicit "not sweeps/net premium" labeling.
- [x] Setup RSS/news context added to debate/scoring and guarded so historical reports do not use live RSS.
- [x] YOLO low-confidence detections are context-only and no longer add directional score.
- [x] LightGBM scorer calibration bug fixed: Booster probabilities are no longer sigmoid-compressed.
- [x] Continuous setup probability added from empirical-Bayes setup-call outcomes.
- [x] Top Suggestions layer added: max 3 research ideas with why, risk, and invalidation.
- [x] Suggestion surfaces explicitly say research only, not financial advice.
- [x] Full walk-forward validation on the local equity universe, with market/context symbols excluded from candidate trades.
- [x] ML target-mode retest: barrier long-only is the best current ML overlay; rank shorts underperformed.
- [x] Frontend build verification.
- [x] Telegram morning summary and `/top` command now prefer compact research suggestions over raw setup lists.
- [x] Nightly bounded options IV/OI snapshot job added so local option history can accumulate without crawling the whole S&P 500 in the main ingest.
- [x] RSS/news headline snapshots persisted so historical reports can use stored point-in-time news context instead of live RSS.
- [x] Python runtime metadata pinned to 3.11 so `scikit-learn==1.8.0` installs consistently locally and on Nixpacks/Railway.
- [x] Dependency freshness guardrails tightened: weekly Dependabot covers root Python, Railway runtime Python, frontend npm, and GitHub Actions; CI/local docs target Python 3.11; local `make dependency-status` reports outdated packages.

## Validated This Pass

- Full backend verification: `928 passed, 11 skipped`; frontend production build passes with the known large Plotly chunk warning.
- Targeted regressions covered paper trades, options/news research, ML scorer calibration, debate roles, Telegram summaries/commands, suggestions, and key-level date handling.
- Real local feature/model smoke run on `trader_koo/data/trader_koo.db`:
  - feature extraction on 6 liquid tickers completed;
  - single historical train/test fold completed;
  - result was not tradable: AUC `0.50`, probabilities collapsed to `0.3333`.
- Earlier clean local walk-forward run, now treated as a superseded benchmark because the later target-mode retest fixed sample handling and passed target mode into the execution backtest:
  - artifact: `data/models/walk_forward_validation_20260425T000411Z.json`;
  - universe after excluding context symbols: `512` tickers, `102,705` price rows, `302` dates;
  - 8 folds, `7,674` labeled samples, average AUC `0.522`, average accuracy `0.405`, average precision `0.272`;
  - execution backtest from 2025-06-01: `+42.55%` versus SPY `+11.59%`, max drawdown `-8.61%`, win rate `61.0%`, profit factor `1.99`, `105` trades;
  - decision: execution/risk rules are promising, but the LightGBM classifier remains observation-only because fold AUC is unstable and several folds are below random.
- Target-mode retest after fixing the sample universe:
  - `return_sign` with time-expired samples included became worse: artifact `data/models/walk_forward_validation_20260425T001822Z.json`, average AUC `0.496`.
  - `rank` is cleaner as a "top/bottom setup today" target: artifact `data/models/walk_forward_validation_20260425T002142Z.json`, average AUC `0.529`, but execution backtest was only `+8.84%` versus SPY `+11.59%`; rank shorts were the weak side.
  - `barrier` is the current best ML framing: artifact `data/models/walk_forward_validation_20260425T002727Z.json`, average AUC `0.550`, long-only execution backtest `+22.11%` versus SPY `+11.59%`, profit factor `1.57`, max drawdown `-8.16%`.
  - Low barrier probability is not a short signal. The backtester now disables shorts for `target_mode=barrier`.
  - Admin retraining/backtesting and the validation CLI now default to `target_mode=barrier`.
  - Model metadata and scoring output now expose the target mode and probability label so the app can distinguish target-hit probability from directional win probability.
  - Removing the most correlated default features made barrier AUC worse (`0.518` in `data/models/walk_forward_validation_20260425T003352Z.json`), so that pruning was reverted.
- Report signal builder on the local DB:
  - built 40 setup rows;
  - correctly skipped live RSS because the local price date is historical (`2026-03-19`);
  - options research annotated 0 rows because local `options_iv` is empty;
  - continuous probabilities were populated from setup-call calibration, but sample size is still thin (`10` scored calls).
- Regime context no longer crashes on string-vs-Timestamp level dates.

## Backtest Protocol

No model or new signal can become a trade gate until it passes this process:

- Use point-in-time features only. Any live-only source such as RSS, Polymarket, or current options chains must be skipped for old `as_of_date`s unless historical snapshots exist.
- Use purged walk-forward splits: training labels must end at least `max_holding_days + buffer` before the scoring date.
- Compare against three baselines: SPY buy-and-hold, all qualifying setups without filters, and current rule-only paper-trade policy.
- Report expectancy, profit factor, max drawdown, hit rate, trade count, and calibration curve. AUC alone is not enough.
- Require at least 5 folds and enough closed trades per bucket before enabling any gate.
- Keep LightGBM observation-only until it shows stable OOS lift over repeated runs. The current best candidate is barrier-mode long-only, but it still needs more feature snapshots and live paper calibration before gating trades.

## Continuous Probability Layer

The TradingAgents-style panel still keeps readable buckets (`ready`, `conditional`, tier A/B/C), but each setup now also gets:

- `calibrated_hit_prob`
- `probability_label`
- `probability_sample_size`
- `probability_source=empirical_bayes_setup_eval`

This is not the weak ML model. It blends closed setup-call outcomes with the current evidence score. News/options/debate affect the current score, while historical setup outcomes anchor the probability.

## Telegram Cadence

Defaults after this upgrade:

- Commands: long-poll up to 25s, loop delay 1s. This improves responsiveness without creating outgoing spam.
- Price alerts: 60s during market hours, still gated by ticker/level cooldowns.
- Polymarket snapshots: 5m.
- Spike alerts: 5m, still sent only on threshold/cooldown breaches.
- Macro alerts: 10m, threshold/cooldown gated.
- Hyperliquid wallets: 5m, Telegram pages only for actionable counter signals and material position changes.
- Site health: 10m checks, alert only after repeated failures.
- Crypto derivatives: 15m snapshots.
- Options IV/OI snapshots: bounded nightly yfinance crawl for latest suggestion/setup tickers, default 21:40 UTC Mon-Fri.

All of these can be tuned through env vars:

- `TRADER_KOO_TELEGRAM_COMMAND_POLL_SEC`
- `TRADER_KOO_TELEGRAM_COMMAND_LONG_POLL_SEC`
- `TRADER_KOO_PRICE_ALERT_POLL_SEC`
- `TRADER_KOO_POLYMARKET_SNAPSHOT_MINUTES`
- `TRADER_KOO_SPIKE_ALERT_MINUTES`
- `TRADER_KOO_MACRO_ALERT_MINUTES`
- `TRADER_KOO_HYPERLIQUID_POLL_MINUTES`
- `TRADER_KOO_SITE_HEALTH_MINUTES`
- `TRADER_KOO_CRYPTO_HEALTH_MINUTES`
- `TRADER_KOO_DERIVATIVES_SNAPSHOT_MINUTES`
- `TRADER_KOO_OPTIONS_SNAPSHOT_ENABLED`
- `TRADER_KOO_OPTIONS_SNAPSHOT_HOUR_UTC`
- `TRADER_KOO_OPTIONS_SNAPSHOT_MINUTE_UTC`
- `TRADER_KOO_OPTIONS_SNAPSHOT_MAX_TICKERS`
- `TRADER_KOO_OPTIONS_SNAPSHOT_TICKERS`
- `TRADER_KOO_OPTIONS_MIN_INTERVAL_HOURS`
- `TRADER_KOO_OPTIONS_MAX_EXPIRIES`

## Free-First Data Source Map

Unusual Whales core value is options flow, dark pools, Congress, alerts, and market event speed. Since we do not have options/net premiums, Trader Koo should build an equivalent "event tape" from free or low-friction sources:

- Official macro: FRED, Fed RSS, BLS RSS, BEA RSS, Census economic indicators.
- Official filings: SEC EDGAR submissions/companyfacts for 8-K, 10-Q/K, 13F, Form 4, 13D/G.
- Short activity: FINRA daily short-sale volume, with clear warnings that it is not short interest.
- Prediction markets: Polymarket probabilities and probability changes.
- Crypto derivatives: Binance/Hyperliquid funding, open interest, liquidation distance, whale positioning.
- Social attention: Reddit public JSON/RSS, Hacker News Firebase API, Stocktwits if the API remains viable, Bluesky AT Protocol.
- X/Twitter: optional paid/pay-per-use connector for curated accounts, not a required free dependency.

## Paid-Only Signals We Should Not Fake

Discord examples from Tradytics-style bots show signals such as option sweeps, golden sweeps, total premium, and intraday blocktrade/dark-pool prints. Those require licensed options/market-tape data. Trader Koo should show these as unavailable unless a real provider is configured.

Free-first substitutes:

- Option sweep substitute: unusual equity relative volume + price breakout + catalyst confirmation.
- Net premium substitute: dollar-volume impulse, gap-adjusted volume z-score, and close-to-VWAP pressure.
- Golden sweep substitute: repeated same-direction breakouts across 15m/1h/daily with news/filing confirmation.
- Dark-pool/blocktrade substitute: FINRA daily off-exchange/short-sale aggregates and delayed large-dollar volume anomalies.
- Scalp bot substitute: 1h continuation/breakdown setups with explicit entry, stop, target, and post-trade scorekeeping.

## Outstanding Trading Quality Work

- Let the new bounded `options_iv` nightly job accumulate enough Yahoo/yfinance snapshots for IV rank and OI rank to stabilize.
- Keep accumulating RSS/news snapshots and use them in future historical backtests. Live RSS remains blocked for historical as-of dates when no stored snapshot exists.
- Promote the walk-forward CLI into scheduled validation once feature snapshots and labels are stronger.
- Do not prune correlated ML features blindly. The first correlation-prune attempt worsened barrier AUC; use ablation tests before removing features.
- Add YOLO geometry validation so image detections must match price structure before they influence score.
- Track WSB/X author calls as scored events, not raw sentiment.
- Add crypto spot-flow monitoring from free exchange trade streams/order books before labeling anything as "whale spot buying."

## X Watchlist

User-requested handles to support when an X connector is configured:

- `jukan05`
- `zephyr_z9`
- `Maximus_Holla`
- `DeItaone`
- `financialjuice`
- `Sino_Market`
- `zerohedge`

Design rule: X content is untrusted external text. Store raw posts, quote sparingly, escape all Telegram/HTML output, and never let posts alter prompts or system behavior.

## WallStreetBets Call Tracker

Build this as a scorekeeping system, not a sentiment toy:

- Ingest posts/comments from target subreddits using public JSON/RSS first.
- Extract explicit calls only: ticker, direction, horizon, author, timestamp, URL, confidence language.
- Ignore vague mentions and memes unless a directional call is present.
- Score after 1d, 5d, and 20d using point-in-time prices.
- Track author hit rate, average return, max drawdown, sample size, and ticker concentration.
- Show "early and accurate" separately from "popular and late."
- Telegram only sends a digest or high-confidence leaderboard movement, not every post.

## Web Product Direction

Rallies-style quality means portfolio-aware, context-aware, and calm:

- The first screen should show a few useful research suggestions, not a wall of raw feeds.
- Suggestions must always be labeled as research only, not financial advice.
- More dense research surfaces, fewer explanatory blocks.
- A clean event tape: macro, filings, prediction-market moves, social attention, crypto/whale changes.
- Watchlist and portfolio-aware summaries.
- Every card should answer: what changed, why it matters, what invalidates it, and source freshness.
- Show uncertainty, sample size, and data age by default.
