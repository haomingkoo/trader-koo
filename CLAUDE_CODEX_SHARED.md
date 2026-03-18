# Claude ↔ Codex Shared Communication File

This file is used by both Claude Code and Codex to communicate about ongoing work,
avoid conflicts, and hand off tasks. Both tools should read this before starting work
and update it after completing tasks.

Last updated: 2026-03-18 by Claude (versioned bot architecture design + pipeline ops)

---

## Pipeline Data Health Investigation (2026-03-18)

### Root Causes Found

1. **`^VIX` ingest failure**
   - **Exact error**: `PriceFetchError("yfinance returned no data for ^VIX: Empty response from yfinance for ^VIX")`
   - **Code path**: `update_market_db.py:902` → `sources.py:fetch_ticker_data()` → `_fetch_yfinance()` → `yf.download(tickers="^VIX", start=<5-day-lookback>, end=None)` returns empty DataFrame → `PriceFetchError` raised
   - **Root cause**: yfinance `start/end` date-range queries are unreliable for Yahoo Finance index tickers (`^VIX`, `^GSPC`, etc.). Short lookback windows (5 days) often return empty data for indices, while `period="5d"` uses a different Yahoo endpoint that works reliably.
   - **Impact**: `^VIX` is in `DEFAULT_SOFT_FAIL_TICKERS` so it doesn't abort the pipeline, but the run is marked `partial_failed` with error_message mentioning `^VIX`. If other S&P 500 tickers also failed AND `REQUIRE_FULL_DATASET=1`, the pipeline aborts at `daily_update.sh:119-121`, skipping YOLO + report.

2. **`options_summary` table does not exist**
   - **Exact error**: `sqlite3.OperationalError: no such table: options_summary`
   - **Code path**: `fear_greed.py:_score_put_call_ratio()` line 403 queries `FROM options_summary` — but no such table is defined anywhere in the schema. The actual options data table is `options_iv`.
   - **Root cause**: The put/call ratio scorer was written to query `options_summary` (a planned aggregate table) that was never created. Additionally, `INCLUDE_OPTIONS` defaults to `0` in `daily_update.sh`, so `options_iv` is also empty in production.
   - **Impact**: The exception is caught by the try/except in `_score_put_call_ratio` and the component returns `None` + error detail. It doesn't crash the fear_greed computation but produces a confusing error message.

3. **Alpha Vantage news: 0 usable articles**
   - **Root cause**: `TRADER_KOO_ALPHA_VANTAGE_KEY` is likely not set on Railway. Without a key, the code returns immediately with a generic "not configured" note.
   - **Impact**: External news overlay is permanently unavailable; internal composite is unaffected.

4. **Reddit social pulse: 403 Blocked**
   - **Exact error**: `403 Client Error: Forbidden` from `https://www.reddit.com/r/*/top.json`
   - **Root cause**: Reddit blocks public JSON API requests from datacenter/cloud IPs. This is permanent without OAuth credentials. Railway runs on GCP/cloud infra.
   - **Impact**: Every configured subreddit fails, making the social sentiment overlay permanently unavailable in production.

### Fixes Applied

| File | Change |
|------|--------|
| `trader_koo/db/sources.py` | Added `period="5d"` fallback for index tickers (`^`-prefixed) when date-range fetch returns empty |
| `trader_koo/structure/fear_greed.py` | Fixed `_score_put_call_ratio()` to query `options_iv` instead of non-existent `options_summary`; added table existence check |
| `trader_koo/social_sentiment.py` | Added `_is_reddit_blocked()` probe that fast-fails with a clear note instead of iterating all 10 subreddits into 403s |
| `trader_koo/news_sentiment.py` | Improved "not configured" note to specify env var names and free tier limits |

### Sentiment Provider Migration (same session)

| Component | Old Provider | New Provider | Status |
|-----------|-------------|-------------|--------|
| News sentiment | Alpha Vantage (no key set → 0 articles) | **Finnhub** (free, key already on Railway) | Implemented + tested |
| Social sentiment | Reddit public JSON (403 blocked) | **StockTwits** (free, no auth, 200 req/hr) | Implemented + tested |

**Finnhub news** (`news_sentiment.py`):
- Uses `/api/v1/news-sentiment` for per-ticker bullish/bearish % and buzz metrics
- Uses `/api/v1/company-news` for recent headlines
- Falls back to Alpha Vantage if `FINNHUB_API_KEY` is not set but `TRADER_KOO_ALPHA_VANTAGE_KEY` is
- Uses existing `FINNHUB_API_KEY` env var — no new config needed

**StockTwits social** (`social_sentiment.py`):
- Uses `https://api.stocktwits.com/api/2/streams/symbol/{SYMBOL}.json` — no auth required
- Aggregates user-tagged bullish/bearish sentiment labels across 10 default tickers
- Works from datacenter IPs (legitimate REST API, not scraping)
- Env var: `TRADER_KOO_SOCIAL_TICKERS` to customize tracked tickers

### Pipeline Hung-Download Fix (same session)

**Problem**: `yf.download()` hangs indefinitely on certain tickers (observed: LLY at ticker 289/510, then again at ticker 51/510). The SIGALRM-based `_ticker_timeout` doesn't fire because signals can't interrupt blocking C-level socket operations.

**Fixes applied**:
1. **Thread-based hard timeout** (`sources.py`): `_download_with_hard_timeout()` wraps every `yf.download()` call in a `ThreadPoolExecutor` with a 60-second wall-clock deadline. If the call doesn't return, the thread is abandoned and the ticker is marked failed with a `TimeoutError`.
2. **Auto-resume** (`update_market_db.py`): `get_succeeded_tickers_from_latest_run()` checks today's latest run and skips tickers that already completed with `price_rows > 0`. Context tickers (^VIX etc.) are always re-fetched.
3. **Force-cancel endpoint** (`admin.py`): `POST /api/admin/force-cancel-run` marks all stuck "running" runs as failed, bypassing the 75-minute stale timeout.

### Production Pipeline Verification (2026-03-18T10:26 UTC)

| Metric | Value |
|--------|-------|
| `/api/status` ok | `true` |
| Warnings | `[]` |
| Ingest | `ok` — 460/460 tickers, 0 failed |
| `^VIX` | Succeeded via `period="5d"` fallback |
| Price date | `2026-03-18` (today) |
| Report `generated_ts` | `2026-03-18T10:25:17Z` |
| YOLO | Fresh — 390 daily, 340 weekly patterns |
| Email | Sent successfully |
| LLM | Healthy (215 successes, 6 failures) |

### What Still Remains

- **Options data**: Enable with `TRADER_KOO_INCLUDE_OPTIONS=1` on Railway if put/call ratio data is desired. This adds ~2-3 min to ingest per run.
- **RSS news enrichment**: Committed to main (`rss_news.py`) — Yahoo Finance + CNBC + MarketWatch RSS feeds with lexicon scoring. Supplements Finnhub headlines.
- **Test baseline**: 549+ passed, 2 pre-existing failures (LLM validator property tests), 1 pre-existing CORS test failure.

## Current State (commit 06bd17a)

### What is now on `main`
- v1 dashboard at `/` — working
- v2 React dashboard at `/v2/` — working with route-level lazy loading restored
- Binance crypto feed — deep history plus server-side multi-interval aggregation
- Finnhub equity streaming — SPY/QQQ always-on + on-demand with live candle support
- HMM regime detection — integrated into chart builder
- VIX metrics — gauge, position sizing, vol premium
- Market sentiment — internal market composite + external news/social layers
- Paper trade system — split into smaller backend modules behind a stable facade

### What was just done (latest Claude pass)
1. **Pipeline Ops Panel** — New `PipelineOpsPanel.tsx` (446 lines) with:
   - Pipeline state visualization (idle/running/completed/warning/error)
   - Partial failures shown as amber warning, not red error
   - Admin action buttons: full update, YOLO+report, report only
   - API key input (localStorage, masked)
   - Recent events/issue log from /api/status warnings
   - Data freshness indicators
   - `PipelineStatusInline` compact variant for ReportPage
2. **Guide page reorganized** — NFA disclaimer moved to top, ops panel added as collapsible section
3. **useTriggerUpdate()** mutation hook added to hooks.ts
4. Did NOT touch Header.tsx (Codex owns pipeline badge)

### What Codex did previously (for context)
1. **Binance scaling architecture** — `crypto/aggregator.py` now builds forming/finalized higher-interval candles from one base 1m stream per symbol.
2. **Live candle rendering** — equity chart now renders backend `live_candle`; crypto chart renders forming candles and patches just-closed bars.
3. **Crypto runtime fixes** — forming candle volume, websocket cleanup, history refresh.
4. **Paper-trade backend split** — `paper_trades.py` is now a facade over `paper_trade/{schema,decision,trading,summary}.py`.
5. **Deploy-robust app shell** — chunk reload on lazy import failure, no-store headers.
6. **Pipeline badge softening** — partial ingest failures shown as warning in header.

### Local WIP not yet committed
1. **Deploy-robust app shell**
   - `backend/main.py` now serves `/v2` and SPA fallback index responses with no-store headers so browsers do not cling to old app shells across deploys.
   - `frontend-v2/src/components/ui/ErrorBoundary.tsx` now auto-reloads once on lazy chunk fetch failures such as `Failed to fetch dynamically imported module`.
2. **Pipeline badge softening**
   - `frontend-v2/src/components/layout/Header.tsx` now treats partial ingest failures as a warning state instead of a hard error and exposes the latest error message in the badge tooltip.

### Current gap to keep in mind
- The candle/streaming path is now in place, so the next work is mostly polish, regression coverage, and file-splitting rather than core market-data plumbing.

### Who owns what next

#### Claude next
1. **Visual/runtime verification pass**
   - Check `/v2/chart` weekly mode and `/v2/crypto` around interval boundaries in a real browser.
   - Verify the pipeline badge and VIX gauge visually after the recent fixes.
2. **Product polish**
   - Earnings calendar visual verification.
   - Mobile responsiveness audit.
3. **Future market-data enhancements**
   - If we keep expanding crypto, continue refining the aggregator/subscription model rather than adding per-client Binance complexity.

#### Codex next
1. **Non-overlapping refactors**
   - Split `ChartPage.tsx` into chart/panel/lib pieces without changing behavior.
   - Split `CryptoPage.tsx` the same way now that forming-candle rendering is stable.
   - The paper-trade backend split is already done.
2. **Regression coverage**
   - Add lean browser/smoke coverage for `/v2/chart`, `/v2/crypto`, interval toggles, and earnings calendar.
3. **Product/paper-trade follow-through**
   - Keep improving decision logging and risk-stage visibility in paper trades.
   - Keep sentiment metadata/methodology readable as the widget evolves.

#### Shared / sequence
1. The live-candle / forming-candle work is complete on `main`.
2. The next clean engineering task is file-splitting and light regression coverage.
3. After that, revisit Plotly replacement and mobile polish.

### Conflict zones
- **Claude-owned for now**
  - `trader_koo/frontend-v2/src/pages/ChartPage.tsx`
  - `trader_koo/frontend-v2/src/pages/CryptoPage.tsx`
  - `trader_koo/backend/routers/dashboard.py`
  - `trader_koo/backend/routers/crypto.py`
  - `trader_koo/streaming/live_candle.py`
  - `trader_koo/streaming/service.py`
  - `trader_koo/crypto/service.py`
- **Codex-owned / safe parallel lane**
  - docs/handoff files
  - browser/regression harness
  - paper-trade workflow files
  - later refactor extractions after Claude finishes the candle work

### What still needs doing (in priority order)

#### P0: File split + regression safety
- Split `ChartPage.tsx` now that the live candle behavior has stabilized
- Split `CryptoPage.tsx` now that the forming-candle path is stable
- Add lean smoke coverage for chart/crypto/earnings

#### P1: UI verification / polish
- Visual-check VIX gauge, weekly chart mode, and crypto interval-boundary behavior
- Visual-check the earnings grid and mobile layouts
- Tidy any remaining header/pipeline rough edges if they still show up in browser use

#### P2: Performance / architecture follow-up
- Plotly is still ~4.6 MB, so a Lightweight Charts migration is still a serious future win
- Add better browser coverage before any large chart-library swap
- Keep social/news sentiment separated and well-labeled as the methodology evolves

### File split plan after live-candle work lands

#### `ChartPage.tsx`
- `components/chart/ChartToolbar.tsx`
- `components/chart/ChartKpis.tsx`
- `components/chart/EquityChart.tsx`
- `components/chart/CommentarySidebar.tsx`
- `components/chart/PatternTables.tsx`
- `lib/chart/buildEquityPlotlyData.ts`

#### `CryptoPage.tsx`
- `components/crypto/CryptoToolbar.tsx`
- `components/crypto/CryptoChart.tsx`
- `components/crypto/CryptoStructureCard.tsx`
- `components/crypto/CryptoCorrelationPanel.tsx`
- `components/crypto/CryptoBreadthPanel.tsx`
- `lib/crypto/buildCryptoChartData.ts`

#### Other likely follow-up splits
- `components/sentiment/MarketSentimentGauge.tsx`
- `components/sentiment/NewsPulseCard.tsx`
- `components/sentiment/SocialPulseCard.tsx`
- Paper-trade backend split already complete in `trader_koo/paper_trade/`

---

## Versioned Paper-Trading Bot Architecture (2026-03-18)

### What "Bot Version" Means

A bot version is a **frozen strategy bundle** — the full snapshot of everything that affects trade selection and sizing. It is NOT just a model or just a config. It is the combination:

| Layer | What it contains | Example change that bumps version |
|-------|-----------------|-----------------------------------|
| Feature set | What inputs the bot reads | Adding VIX-at-entry, adding HMM regime |
| Decision policy | Gating rules (qualifying tiers, min score, min R) | Changing min_tier from B to A-only |
| Scoring model | How setups are ranked (rules today, ML later) | Switching from score-rank to logistic regression |
| Sizing policy | Position allocation + haircuts | Changing caution_scale from 0.65 to 0.50 |
| Risk management | Stop/target computation, expiry, R-gate | Changing stop_atr_mult from 1.5 to 2.0 |

**Version format**: `v{major}.{minor}.{patch}`
- Major: structural change (new model type, new decision stage)
- Minor: parameter change (new thresholds, new haircuts)
- Patch: bug fix, cosmetic

**Current version**: `v1.0.0` (rules-based, the existing `paper-trade-eval-v1`)

### Separation of Concerns

```
┌─────────────────────────────────────────────────────┐
│                  DETERMINISTIC RULES                 │
│  (config-driven, auditable, never auto-tuned)        │
│                                                      │
│  • Qualifying tiers (A/B)                            │
│  • Min score (60), min R-multiple (1.5)              │
│  • Max open positions (20)                           │
│  • ATR-based stop/target computation                 │
│  • Position sizing by tier + haircuts                │
│  • Expiry (10 days)                                  │
└──────────────────┬──────────────────────────────────┘
                   │ filters setups
┌──────────────────▼──────────────────────────────────┐
│              LEARNED RANKING / SCORING               │
│  (data-driven, versioned, offline-trained)           │
│                                                      │
│  Phase 1 (now): Score-based rank, no ML              │
│  Phase 2: Family-edge estimator from closed trades   │
│  Phase 3: Logistic/GBM model on features:            │
│    setup_family, tier, score, ATR, debate_agreement,  │
│    vix_at_entry, market_breadth, hmm_regime           │
│  Target: P(trade wins) → rank by expected edge       │
└──────────────────┬──────────────────────────────────┘
                   │ ranks approved setups
┌──────────────────▼──────────────────────────────────┐
│              EXECUTION POLICY                        │
│  (frozen per version, tuned between versions)        │
│                                                      │
│  • Position sizing (tier-based + score boost)        │
│  • Entry: market-on-close (current)                  │
│  • Exit: stop/target/expiry (current)                │
│  • Future: trailing stops, partial exits              │
└──────────────────┬──────────────────────────────────┘
                   │ trades execute, lifecycle tracked
┌──────────────────▼──────────────────────────────────┐
│         POST-TRADE CRITIQUE / FEEDBACK               │
│  (automated, generates promotion candidates)         │
│                                                      │
│  • Per-family win rate (rolling 60-day)              │
│  • Per-regime performance                            │
│  • Version comparison (A/B)                          │
│  • Degradation detection                             │
│  • Promotion recommendations                         │
└─────────────────────────────────────────────────────┘
```

### Data Model

#### New table: `bot_versions`

```sql
CREATE TABLE bot_versions (
    version_id TEXT PRIMARY KEY,           -- "v1.0.0"
    created_ts TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',  -- active | shadow | retired
    config_json TEXT NOT NULL,             -- full PaperTradeConfig as JSON
    feature_set_version TEXT,              -- "features-v1"
    model_type TEXT DEFAULT 'rules',       -- rules | logistic | gbm
    model_artifact_path TEXT,              -- path to saved weights (Phase 3+)
    training_sample_size INTEGER,
    training_date_range TEXT,
    promotion_metrics_json TEXT,           -- metrics at time of promotion
    promoted_from TEXT,                    -- previous version_id
    promotion_reason TEXT,
    notes TEXT
);
```

#### Additions to `paper_trades` table

```sql
ALTER TABLE paper_trades ADD COLUMN bot_version TEXT;          -- "v1.0.0"
ALTER TABLE paper_trades ADD COLUMN regime_at_entry TEXT;      -- "bull_low_vol"
ALTER TABLE paper_trades ADD COLUMN vix_at_entry REAL;
ALTER TABLE paper_trades ADD COLUMN market_breadth_at_entry REAL;
ALTER TABLE paper_trades ADD COLUMN hmm_regime_at_entry TEXT;
ALTER TABLE paper_trades ADD COLUMN predicted_win_prob REAL;   -- ML score (Phase 2+)
```

#### New table: `bot_family_edge` (rolling performance by family)

```sql
CREATE TABLE bot_family_edge (
    snapshot_date TEXT NOT NULL,
    bot_version TEXT NOT NULL,
    setup_family TEXT NOT NULL,
    direction TEXT NOT NULL,              -- long | short
    regime TEXT,                          -- bull | bear | neutral | NULL (all)
    trade_count INTEGER NOT NULL,
    win_rate_pct REAL,
    avg_r_multiple REAL,
    expectancy_pct REAL,
    sample_window_days INTEGER,
    PRIMARY KEY (snapshot_date, bot_version, setup_family, direction, regime)
);
```

### Minimum Trade Log (what MUST be recorded per trade)

Already stored (good):
- `ticker`, `direction`, `entry_price`, `entry_date`
- `stop_loss`, `target_price`, `atr_at_entry`
- `exit_price`, `exit_date`, `exit_reason`
- `pnl_pct`, `r_multiple`, `expected_r_multiple`
- `setup_family`, `setup_tier`, `score`, `actionability`
- `debate_agreement_score`, `yolo_pattern`, `yolo_recency`
- `decision_state`, `analyst_stage`, `debate_stage`, `risk_stage`
- `position_size_pct`, `risk_budget_pct`
- `high_water_mark`, `low_water_mark`

**Must add** (for learning):
- `bot_version` — which policy created this trade
- `vix_at_entry` — market volatility context
- `regime_at_entry` — bull/bear/neutral (from HMM or breadth)
- `market_breadth_at_entry` — % advancers
- `hmm_regime_at_entry` — HMM state at entry

**Nice to add later** (Phase 2+):
- `predicted_win_prob` — model's confidence
- `feature_vector_json` — full feature snapshot for replay

### Honest Performance Measurement

**Core metrics** (already computed, keep):
- Win rate, expectancy (avg PnL), avg R-multiple, profit factor, max drawdown, Sharpe

**Add for learning**:
- **By family × regime**: which setups work in which market conditions
- **By VIX bucket**: low (<15), normal (15-25), high (>25)
- **By debate confidence**: high agreement (>80%) vs. contested (<60%)
- **Rolling 30-day expectancy**: detect degradation vs. noise
- **Turnover**: trades/week — high turnover amplifies noise
- **Version comparison**: same metrics, side-by-side, for A vs. B policy

**Overfitting detection**:
- Minimum 50 closed trades before trusting any metric
- Compare rolling 30-day to all-time — persistent divergence = regime shift
- New policy must beat current by ≥ 1 standard error (not just point estimate)
- Track if a family edge degrades after we start trading it (adverse selection)

### What Can Auto-Tune vs. Manual Promotion

| Aspect | Auto-tune OK | Manual promotion required |
|--------|-------------|--------------------------|
| Caution haircut scale (within ±20%) | ✅ | |
| Min score threshold (within ±10) | ✅ | |
| Expiry days (within ±3) | ✅ | |
| Family-level position override | ✅ | |
| Qualifying tier change | | ✅ |
| New model type (rules → ML) | | ✅ |
| New feature additions | | ✅ |
| Structural pipeline change | | ✅ |
| R-multiple gate change | | ✅ |

### Phased Roadmap

#### Phase 1: NOW — Versioned Rules (Codex + Claude)
- Add `bot_versions` table + schema migration
- Add `bot_version`, `vix_at_entry`, `regime_at_entry` columns to `paper_trades`
- Snapshot current config as `v1.0.0` in `bot_versions`
- Tag all new trades with `bot_version`
- Add `vix_at_entry` and `regime_at_entry` from existing HMM/VIX data at trade creation time
- **Frontend**: version badge in BotOverview, family edge breakdown table
- **Backend**: `compare_versions(v1, v2)` function

#### Phase 2: NEXT — Family Edge Learning
- Compute rolling family-level win rate (60-day window) nightly
- Store in `bot_family_edge` table
- Use family edge to rank setups: families with proven edge get priority
- Families with negative edge get flagged for demotion
- **Frontend**: family edge heatmap (family × regime → color), rolling performance chart
- **Backend**: `compute_family_edges()` nightly, `suggest_policy_adjustments()` from edges

#### Phase 3: LATER — ML Scoring + Shadow Mode
- Train logistic regression on closed trades (features → win probability)
- Run in shadow mode: ML policy paper-trades alongside rules policy, both recorded
- Compare after 50+ closed trades per policy
- **Frontend**: shadow vs. active comparison panel, promotion candidate display
- **Backend**: `train_scoring_model()`, `shadow_evaluate()`, `promote_version()`

#### Phase 4: EVENTUALLY — Promotion Gates + Auto-Tuning
- Automated A/B comparison with statistical significance testing
- Auto-tune haircuts and thresholds within bounds
- Manual approval gate for major changes
- Graduated rollout: shadow → 25% allocation → 50% → full

### UI Recommendations (for PaperTradePage)

**Add to existing page**:
1. **Bot version badge** — `v1.0.0 (rules)` with status indicator
2. **Version history** — collapsible table showing all versions, when promoted, why
3. **Family edge table** — setup_family × direction → win rate, R, trade count (color-coded)
4. **Rolling performance chart** — 30-day rolling expectancy line, baseline at 0
5. **Regime breakdown** — performance sliced by VIX bucket + HMM regime
6. **Degradation alerts** — if 30-day expectancy drops below historical by >1σ

### Who Builds What Next

#### Codex (backend + frontend, safe parallel lane)
1. `bot_versions` table creation in `schema.py`
2. `ALTER TABLE paper_trades ADD COLUMN` migrations for `bot_version`, `vix_at_entry`, `regime_at_entry`, `hmm_regime_at_entry`
3. Populate `vix_at_entry` and `regime_at_entry` at trade creation time (read from existing `^VIX` data + HMM model)
4. Snapshot current config as `v1.0.0` in startup hook
5. `compare_versions()` in `summary.py`
6. Frontend: version badge, family edge table, rolling performance chart

#### Claude (architecture review, ML pipeline, shadow mode)
1. Design the `compute_family_edges()` nightly job
2. Design the ML feature vector for Phase 2 scoring
3. Design shadow-mode trade tagging
4. Review Codex's schema migrations before they land

---

## Architecture Reference

```
Backend: trader_koo/backend/main.py (~560 lines)
├── 10 routers: system, dashboard, report, opportunities, paper_trades, email, usage, admin, crypto, streaming
├── 6 services: database, market_data, chart_builder, report_loader, scheduler, pipeline
├── Crypto: trader_koo/crypto/ (binance_ws, binance_history, service, storage, indicators, structure, models)
├── Streaming: trader_koo/streaming/ (finnhub_ws, service, live_candle)
├── Structure: trader_koo/structure/ (vix_analysis, vix_metrics, hmm_regime, fear_greed, vix_patterns)
└── DB: SQLite at /data/trader_koo.db (Railway persistent volume)

Frontend: trader_koo/frontend-v2/
├── React 19 + TypeScript strict + Vite 8 + Tailwind v4
├── 9 pages: Guide, Report, VIX, Earnings, Chart, Opportunities, PaperTrades, Crypto, NotFound
├── Plotly via PlotlyWrapper (safe CJS/ESM interop)
├── WebSocket hooks: useCryptoWebSocket, useEquityWebSocket, useLiveEquityPrice
├── API: hooks.ts, types.ts (867+ lines), client.ts
└── Build: npm run build → ../../dist-v2/
```

## Key Files for Each Area

| Area | Backend | Frontend |
|------|---------|----------|
| Crypto streaming | crypto/binance_ws.py, crypto/service.py | pages/CryptoPage.tsx |
| Equity streaming | streaming/finnhub_ws.py, streaming/service.py, streaming/live_candle.py | hooks/useLiveEquityPrice.ts, pages/ChartPage.tsx |
| Pipeline status | services/pipeline.py, routers/system.py | components/layout/Header.tsx |
| VIX/Regime | structure/vix_analysis.py, structure/vix_metrics.py, structure/hmm_regime.py | pages/VixPage.tsx |
| Daily report | scripts/generate_daily_report.py, services/report_loader.py | pages/ReportPage.tsx |

## Environment Variables (Railway)
- `FINNHUB_API_KEY` — real-time equity streaming
- `TRADER_KOO_API_KEY` — admin auth
- `TRADER_KOO_LLM_PROVIDER` — azure_openai
- `TRADER_KOO_LLM_ENABLED` — 1

## Running Locally
```bash
# Backend
.venv/bin/python -m uvicorn trader_koo.backend.main:app --reload --port 8000

# Frontend
cd trader_koo/frontend-v2 && npm run dev
# → http://localhost:3000/v2/

# Tests
.venv/bin/python -m pytest tests/ -x -q
```

## Rules
- NEVER echo API keys in terminal or code output
- No hidden fallbacks — fail explicitly
- No mock financial data
- TypeScript strict, no `any`
- Conventional commits (feat:, fix:, refactor:, etc.)
- Never push directly to main without verification
