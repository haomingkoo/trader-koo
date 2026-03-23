# Changelog

All notable changes to Trader Koo are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.1.0] - 2026-03-24

### Major: Paper Trade Realism Overhaul

The paper trade system was audited by a 7-agent expert panel (Quant Trader, ML Engineer, Finance, Code Reviewer, Security, Infrastructure, Critic). Results showed paper P&L was biased optimistic by 2-5% annually. This release makes paper trading honest.

#### Execution Realism (PRs #108-#110)
- **Entry at next-day open**: Signals generate after close (22:00 UTC). Entry now uses next trading day's open price, not same-day close. Eliminates look-ahead bias.
- **Intraday stop/target checks**: Uses daily high/low (already in DB) instead of only close price. Catches intraday breaches.
- **Open-price priority**: When open itself breaches stop or target, fills immediately with no ambiguity. Gap-through stops fill at open (realistic gap loss), not at stop level.
- **Stop-target ambiguity**: When both stop and target are hit intraday, conservatively assumes stop hit first.

#### Trading Costs (PR #110)
- **Entry slippage**: 10 bps (configurable via `entry_slippage_bps`)
- **Exit slippage**: 10 bps on stop-market orders; limit/target orders fill at exact level
- **Commission**: $5 per side ($10 round trip, configurable via `commission_per_trade`)
- **Short borrow**: 3% annualized, pro-rated to trading days (not calendar days), configurable via `short_borrow_annual_pct`
- **ADV liquidity gate**: Rejects positions > 15% of 20-day average daily volume. SQL query fixed to use subquery (was averaging all history).

### Sprint 1: Security + Code Quality (PR #111)

- **CORS hardened**: Removed wildcard `allow_methods=["*"]` and `allow_headers=["*"]` from legacy CORS fallback. Now uses explicit method/header lists.
- **Admin dev mode guard**: Open-admin now requires both `ADMIN_STRICT_API_KEY=0` AND `TRADER_KOO_DEVELOPMENT_MODE=1`. Prevents accidental open-admin in production.
- **datetime.utcnow() migrated**: 8 instances across 5 files replaced with `dt.datetime.now(dt.timezone.utc)`. Fixes Python 3.12+ deprecation.
- **Test import fix**: `MAX_POLL_TICKERS` renamed to `MAX_REPORT_TICKERS` in both source (alert_engine.py, 3 refs) and tests (test_alert_engine.py, 3 refs).

### Sprint 2: Trading Logic (PR #112)

- **VIX-scaled position sizing**: Positions auto-scale based on VIX level. <15=1.1x, 15-20=1.0x, 20-25=0.85x, 25-30=0.65x, >30=0.5x. VIX pre-fetched once before setup loop.
- **Rolling expectancy gate**: New critic check (#8). Rejects new trades if last 20 closed trades have avg P&L < -0.2%. Fails open with < 5 trades.
- **Daily loss circuit breaker**: Halts new entries if today's realized losses exceed `max_daily_loss_pct` (default 5%). Config field existed but was never checked.
- **Sortino ratio**: Downside deviation-based risk-adjusted return metric. Added to portfolio snapshots alongside Sharpe.
- **Calmar ratio**: Annualized return / max drawdown. Added to portfolio snapshots.
- **Directional HMM on equities**: `predict_directional_regimes()` (bullish/chop/bearish) now called for SPY at trade entry. Was crypto-only. Stored as `directional_regime_at_entry` + confidence on each trade.

### Sprint 3a: ML Pipeline Fixes (PR #113)

- **mean_reversion_signal fixed**: Was returning cumulative return (semantically wrong). Now returns distance from SMA as percentage -- the correct mean-reversion indicator. Existing model needs retraining.
- **is_unbalance=True**: Added to primary LightGBM params. Handles class imbalance in training (was only in meta-label model).
- **Noise filter**: Removes label=0 (time-expiry) samples before training. These are trades where neither stop nor target was hit -- mostly random outcomes.
- **Feature correlation audit**: Logs warning for feature pairs with Pearson r > 0.85 after dataset build.

### Sprint 3b: Signal Architecture (PR #114)

- **Standardized signal format** (`trader_koo/signals/types.py`): `SignalOutput` dataclass with bias, confidence (0-100), reasoning, weight. `aggregate_signals()` for weighted bull/bear scoring with agreement percentage.
- **5-strategy technical ensemble** (`trader_koo/analysis/technical_ensemble.py`):
  - Trend (0.25): EMA 12/26 crossover + ADX trend strength
  - Mean reversion (0.20): Bollinger z-score + RSI oversold/overbought
  - Momentum (0.25): Multi-timeframe returns (5d/21d/63d)
  - Volatility (0.15): ATR expansion/contraction + VIX regime
  - Statistical arbitrage (0.15): Hurst exponent (R/S method) + return skewness
- **New indicators**: ADX (Wilder smoothing), Hurst exponent, z-score

### Bug Fixes (PR #107)

- **Critic max_open**: Was hardcoded to 5 in function default; now passes `config.max_open` (20). New trades were blocked since March 18.
- **Telegram equity**: Morning summary now includes unrealized P&L from open trades. Was only summing realized P&L from closed trades.

### Tests

- Paper trade tests: 47 -> 56 (intraday checks, gap fills, cost deductions, ambiguity cases)
- Alert engine tests: fixed import error, all running
- Technical ensemble: 15 new tests (RSI, ADX, Hurst, signal aggregation, ensemble integration)

---

## [1.0.0] - 2026-03-22

Initial versioned release. See `CODEX_REVIEW.md` for the 112-commit session audit.
