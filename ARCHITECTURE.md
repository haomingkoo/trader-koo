# Architecture

This document explains how trader-koo is built — the data flow, module responsibilities, key algorithms, and deployment topology. Intended for anyone who wants to understand the internals or fork/extend the project.

---

## System overview

Full-stack S&P 500 + crypto analysis dashboard. Live at [trader.kooexperience.com](https://trader.kooexperience.com), deployed on Railway (asia-southeast1).

The entire system — API server, WebSocket clients, background scheduler, and ML pipelines — runs in a single process on a single Railway service. No separate worker, no message queue, no external database.

```
Railway Service (single process)
├── FastAPI (uvicorn, port 8080)
│   ├── 11 routers (82 endpoints)
│   ├── Static file serving (React build → /)
│   └── WebSocket endpoints (/ws/crypto, /ws/equities)
│
├── Binance WS Client
│   └── 5 crypto pairs: BTC, ETH, SOL, XRP, DOGE (real-time 1m klines)
│
├── Finnhub WS Client
│   └── SPY/QQQ always subscribed + on-demand symbols from UI
│
├── APScheduler (BackgroundScheduler, in-process)
│   ├── Mon–Fri 22:00 UTC: daily_update.sh (ingest → YOLO → report)
│   └── Saturday 00:30 UTC: YOLO full seed (daily + weekly timeframes)
│
└── /data/ (Railway persistent volume)
    ├── trader_koo.db    (SQLite, 23 tables)
    ├── models/          (LightGBM .pkl files)
    ├── reports/         (daily JSON + MD archives)
    ├── logs/            (structured log files)
    └── .ultralytics/    (YOLO model cache)
```

---

## Deployment

### Railway setup

| File | Purpose |
|------|---------|
| `railway.toml` | Build command + deploy config (start command, health check) |
| `start.sh` | Entrypoint — seeds DB on first run, then starts uvicorn |
| `pyproject.toml` | Python package definition (allows `pip install -e .`) |

**Build command** (in `railway.toml`):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r trader_koo/requirements.txt
pip install -e .
pip install opencv-python-headless --force-reinstall --quiet
```

The torch/torchvision pair must come from the same CPU index — installing them separately risks an ABI mismatch that breaks the `torchvision::nms` operator at runtime. The `opencv-python-headless` force-reinstall runs last because `sahi` (a transitive dependency of `ultralyticsplus`) installs the non-headless `opencv-python`, which requires `libGL.so.1` and fails on headless servers.

**Runtime**: Python 3.10 via nixpacks. 50+ environment variables for configuration (data sources, API keys, SMTP, feature flags).

### First-run DB seed

`start.sh` checks for `/data/trader_koo.db`. If it doesn't exist, it runs `update_market_db.py` with `--price-lookback-days 365` to pull a full year of history for all ~510 S&P 500 + context tickers. This takes ~20–30 min but only happens once — subsequent deploys skip it.

YOLO patterns are seeded separately via `POST /api/admin/run-yolo-seed` after the first deploy.

### YOLO model caching

The YOLOv8 model (`foduucom/stockmarket-pattern-detection-yolov8`) is downloaded from HuggingFace on first use and cached to `/data/.ultralytics/`. Setting `ULTRALYTICS_CONFIG_DIR=/data/.ultralytics` redirects the cache to the persistent volume so it survives redeploys.

---

## Backend routers (11)

All routes live under `trader_koo/backend/routers/`. The app factory in `main.py` (~600 lines) mounts them along with middleware and static file serving.

| Router | Path prefix | Purpose |
|--------|-------------|---------|
| `system.py` | `/api` | Health check, status, config, VIX glossary, Polymarket proxy |
| `dashboard.py` | `/api/dashboard` | Chart payload (OHLCV + levels + YOLO + live candle merge) |
| `report.py` | `/api/report` | Daily report, earnings calendar, sentiment, VIX metrics, macro data |
| `opportunities.py` | `/api/opportunities` | Valuation screens (discount-to-target, PEG, relative strength) |
| `paper_trades.py` | `/api/paper-trades` | Trade list, summary, detail, equity curve |
| `email.py` | `/api/email` | Subscribe, unsubscribe, confirm, chart preview |
| `crypto.py` | `/api/crypto` | Prices, history, indicators, structure analysis, correlation, open interest |
| `streaming.py` | `/api/streaming` | Equity tick prices, subscription management (Finnhub WS bridge) |
| `data_sync.py` | `/api/data-sync` | Polymarket event ingestion |
| `usage.py` | `/api/usage` | Session tracking, setup feedback collection |
| `admin/` | `/api/admin` | Split into 5 modules (see below) |

### Admin sub-routers

The admin router is split for maintainability. All require `X-API-Key` authentication.

| Module | Responsibilities |
|--------|-----------------|
| `pipeline.py` | Trigger updates, pipeline status, YOLO seed/status |
| `ml.py` | Model training, evaluation, feature importance |
| `data.py` | DB inspection, cache management, data quality checks |
| `system.py` | Logs, audit trail, health events, configuration |
| `email_admin.py` | Subscriber management, delivery history, test sends |

---

## Data sources

```
                     ┌─────────────────────────────────────────┐
                     │              trader_koo.db              │
                     └──────────┬──────────────────────────────┘
                                │
    ┌───────────┐  daily OHLCV  │  ┌────────────┐  fundamentals
    │  yfinance  │──────────────┤  │   Finviz    │──────────────┤
    │ (510+ tkr) │              │  │  (scrape)   │              │
    └───────────┘               │  └────────────┘              │
                                │                               │
    ┌───────────┐  real-time WS │  ┌────────────┐  macro data  │
    │  Finnhub   │──────────────┤  │    FRED     │──────────────┤
    │  (equities)│              │  │ (8 series)  │              │
    └───────────┘               │  └────────────┘              │
                                │                               │
    ┌───────────┐  crypto WS +  │  ┌────────────┐  prediction  │
    │  Binance   │──────────────┤  │ Polymarket  │──────────────┤
    │  (5 pairs) │  REST backfill│  │ (finance)   │  market odds │
    └───────────┘               │  └────────────┘              │
                                │                               │
                                │  ┌────────────┐  news        │
                                │  │Alpha Vantage│──────────────┘
                                │  │ (optional)  │  sentiment
                                │  └────────────┘
                                │
                                ▼
```

| Source | Data type | Frequency | Notes |
|--------|-----------|-----------|-------|
| yfinance | Daily OHLCV for 510+ tickers | Daily cron (22:00 UTC Mon–Fri) | S&P 500 + context tickers (VIX, SPY, QQQ, ^DJI, ^TNX, SVIX) |
| Finviz | Fundamentals (P/E, PEG, EPS, target price) | Daily, throttled (20h min interval) | Free scrape with rate-limiting sleep |
| Finnhub | Real-time equity WebSocket + company news + earnings calendar | Continuous WS + on-demand REST | Free tier: 60 calls/min |
| Binance | Crypto WebSocket (BTC/ETH/SOL/XRP/DOGE 1m klines) + REST backfill | Continuous WS | 5 pairs, CandleAggregator for multi-interval bars |
| FRED | Macro indicators (Fed funds rate, yield curve 10Y-2Y, unemployment, CPI, ISM, consumer sentiment, building permits, initial claims) | Weekly/monthly via `external_data.py` | Per-series API calls, cached in `external_data_cache` |
| Polymarket | Finance-filtered prediction market probabilities | On-demand sync | Proxy endpoint avoids CORS issues for frontend |
| Alpha Vantage | News sentiment scores | Optional, blended into sentiment composite | Disabled if API key not set |

---

## Data ingestion (update_market_db.py)

### S&P 500 ticker list

Scraped from Yahoo Finance's S&P 500 Wikipedia page via yfinance's built-in utility. The list is fetched fresh each run — tickers are added/removed from the index over time.

Market context tickers (`VIX, SPY, QQQ, ^DJI, ^TNX, SVIX`) are always appended regardless of `--use-sp500`.

### Rate limiting

yfinance and Finviz are free public APIs with implicit rate limits. The ingestion loop sleeps `random.uniform(sleep_min, sleep_max)` seconds between each ticker (default 0.5–1.2s for daily cron, 0.3–0.8s for first-run seed). Finviz fundamentals are skipped if the last snapshot is less than `fund_min_interval_hours` old (default 20h) to avoid re-scraping on the same day.

### Ingestion tracking

Each run writes to `ingest_runs` (overall status) and `ingest_ticker_status` (per-ticker OK/failed). The `/api/status` endpoint queries these tables to surface the last run timestamp and ticker counts in the dashboard.

---

## Database schema (23 tables)

### Core market data

| Table | Purpose | Key columns |
|-------|---------|-------------|
| `price_daily` | OHLCV price history | `ticker, date, open, high, low, close, volume` |
| `finviz_fundamentals` | Fundamental snapshots | `ticker, snapshot_ts, pe, peg, eps_ttm, target_price, discount_pct` |
| `options_iv` | Options chain aggregates | `ticker, snapshot_ts, option_type, implied_vol, open_interest` |

### Pattern detection

| Table | Purpose |
|-------|---------|
| `yolo_patterns` | Pre-computed YOLOv8 detections with `(ticker, timeframe, pattern, confidence, x0_date, x1_date, y0, y1, as_of_date)` |
| `yolo_run_events` | Per-ticker YOLO outcomes (ok/skipped/timeout/failed) for observability |

### Pipeline operations

| Table | Purpose |
|-------|---------|
| `ingest_runs` | Overall run status (start/finish timestamps, ticker counts) |
| `ingest_ticker_status` | Per-ticker ingestion outcome for debugging partial failures |

### Paper trading

| Table | Purpose |
|-------|---------|
| `paper_trades` | Full trade lifecycle (entry, exit, PnL, R-multiple, sizing rationale) |
| `paper_portfolio_snapshots` | Daily mark-to-market snapshots for equity curve |
| `bot_versions` | Version tracking for the trading bot decision pipeline |

### User interaction

| Table | Purpose |
|-------|---------|
| `ui_usage_sessions` | Frontend session tracking (page views, interactions) |
| `setup_feedback` | User-submitted feedback from the setup flow |
| `setup_call_evaluations` | LLM evaluation scores for setup call quality |

### Email system

| Table | Purpose |
|-------|---------|
| `email_subscribers` | Subscriber list with confirmation status |
| `email_subscriber_events` | Event log (subscribe, confirm, unsubscribe) |
| `email_report_deliveries` | Delivery history (sent, bounced, opened) |

### System / observability

| Table | Purpose |
|-------|---------|
| `audit_logs` | Admin action audit trail with IP tracking |
| `llm_health_events` | LLM provider health check events |
| `llm_health_state` | Current LLM provider status |
| `llm_token_usage` | Token consumption tracking per LLM call |
| `external_data_cache` | Cached FRED/macro data to avoid redundant API calls |
| `crypto_bars` | Binance real-time kline persistence (multi-interval) |

---

## Request lifecycle — `/api/dashboard/{ticker}`

This is the most compute-heavy endpoint. Everything runs synchronously in the request handler (no async DB calls — SQLite doesn't benefit from async I/O).

```
 1. Load price_daily  →  DataFrame (full history)
 2. add_basic_features()
    - ATR-14 (Wilder's smoothing)
    - MA20, MA50, MA100, MA200
    - pivot_high / pivot_low columns (±3 bars)
 3. Slice to requested window (months parameter)
 4. detect_gaps()          →  bull/bear gaps, fill tracking
 5. build_levels_from_pivots()
    - Cluster pivots within ATR × 0.60 tolerance
    - Score clusters: touches × recency_weight (45-day half-life)
    - Tier as primary (top 2 each side) / secondary / fallback
 6. detect_trendlines()    →  regression lines through swing pivots
 7. detect_patterns()      →  flags, wedges (rule-based geometry)
 8. detect_candlestick_patterns()
 9. score_hybrid_patterns()
    - Blend rule confidence (50%) + candle (20%) + volume (15%) + breakout (15%)
10. detect_cv_proxy_patterns()
    - Independent image-geometry scorer for the same pattern classes
11. compare_hybrid_vs_cv()
    - Consensus confidence, agreement flag, confidence gap
12. get_yolo_patterns()   →  SELECT from pre-computed yolo_patterns table
13. Merge live candle from Finnhub WS (if available)
14. Fetch fundamentals    →  latest finviz_fundamentals row
15. Fetch options         →  latest options_iv rows
16. Score ML model        →  LightGBM probability (if model .pkl exists)
17. Merge pattern overlays (top 10 by confidence across all sources)
18. Return JSON payload
```

---

## Feature engineering

### ATR (Average True Range)

Standard Wilder's 14-bar ATR is computed first and used as a price-relative scale for nearly everything else:
- Level clustering tolerance: `ATR × 0.60`
- Zone half-width: `ATR × 0.35`
- Trendline touch tolerance: `1.5%` (not ATR-based, but similar intent)

### Pivot detection

A bar is a `pivot_high` if its high is the highest within ±3 bars (left=3, right=3). Same for `pivot_low`. This is the foundation for level clustering, trendlines, and pattern fitting.

### Level clustering

```
1. Collect all pivot_high and pivot_low prices
2. Sort ascending
3. Greedy cluster: if |price_a - price_b| < ATR × 0.60, merge into one zone
4. Each zone gets a score = sum(1 / (1 + days_since_touch / 45))
   (45-day half-life recency weighting)
5. Select top 2 support + top 2 resistance zones as "primary"
6. Next 2 each side as "secondary"
7. If no primary found within 20% of current price, add fallback (MA-anchored)
```

---

## Pattern detection (5 layers)

Five independent pattern detection systems run per ticker. Each produces its own confidence scores. The dashboard merges the top 10 detections across all layers by confidence.

```
Layer    Source                          Type             Speed
─────    ──────                          ────             ─────
  1      structure/patterns.py           Rule-based       Fast (in-request)
  2      features/candle_patterns.py     Candlestick      Fast (in-request)
  3      structure/hybrid_patterns.py    Blended score    Fast (in-request)
  4      cv/proxy_patterns.py            CV geometry      Fast (in-request)
  5      scripts/run_yolo_patterns.py    YOLOv8 AI        Pre-computed (nightly)
```

### Layer 1: Rule-based (structure/patterns.py)

Purely geometric checks on OHLCV data.

**Bull/bear flag:**
- Detect a "pole" in the last `flag_pole_bars=8` bars: `|return| >= 4%`
- Check the next `flag_lookback_bars=30` bars form a channel parallel to price
- Channel slope must be opposite to pole direction (consolidation, not continuation)
- Pullback must be `< 75%` of pole height

**Rising/falling wedge:**
- Fit linear regression through pivot highs and pivot lows over 45 bars
- Both lines must slope the same direction
- Convergence ratio: `(initial_width - final_width) / initial_width >= 35%`
- R-squared of both fits must be `>= 0.45`

### Layer 2: Candlestick patterns (features/candle_patterns.py)

Classic single- and multi-bar candle signals. Each detection returns:
- `pattern` name (e.g. `hammer`, `morning_star`)
- `bias` (`bullish` / `bearish`)
- `confidence` (0–1, based on how cleanly the geometry fits)
- `explanation` (human-readable rationale)

Used primarily as an input signal to hybrid scoring rather than standalone.

### Layer 3: Hybrid scoring (structure/hybrid_patterns.py)

Merges the rule-based pattern list with candlestick signals to produce a blended confidence score:

```
hybrid_confidence = (
    base_confidence × 0.50
  + candle_score    × 0.20
  + volume_score    × 0.15
  + breakout_score  × 0.15
)
```

- `candle_score`: 1.0 if a candle pattern at the pattern's end date agrees with the expected bias, else 0
- `volume_score`: `min(volume_ratio / 1.5, 1.0)` where `volume_ratio = last_vol / avg_vol`
- `breakout_score`: 1.0 if price has broken out of the pattern in the expected direction

### Layer 4: CV proxy (cv/proxy_patterns.py)

An independent scorer that uses the same geometric logic as layer 1 but is implemented from the perspective of "what would a trained CV model see". It produces the same pattern classes with its own confidence scores. The comparison between hybrid and CV scores (`cv/compare.py`) surfaces cases where both agree (high conviction) or disagree (uncertain signal).

### Layer 5: YOLO AI (scripts/run_yolo_patterns.py)

The only layer that uses a trained ML model. Rather than running inference on live requests (too slow at ~1–2 seconds per ticker), detections are pre-computed and stored in `yolo_patterns`.

**Pipeline per ticker:**
```
1. Load price_daily for ticker
2. Slice to lookback window (180 days for daily, 730 days for weekly)
3. For weekly pass: resample daily OHLCV to weekly bars
   df.resample("W").agg(open=first, high=max, low=min, close=last, volume=sum)
4. Render a 1200×600px white-background Yahoo-style candlestick chart
   (mplfinance, Yahoo style, volume panel included)
5. Run YOLOv8 inference on the rendered image
6. For each detected bounding box:
   a. Map pixel (x0,y0,x1,y1) → bar index → date string
   b. Map pixel y → price via axis coordinate transform
   c. Filter boxes that fall in the volume panel (price < axis_lo × 0.3)
7. Store in yolo_patterns with timeframe + as_of_date
```

**Why two timeframes?**

The YOLOv8 model was trained on charts with ~100–130 bars visible. Detection quality is strongly tied to visual candle density.

| Pass | Lookback | Bars | Candle = | Purpose |
|------|----------|------|----------|---------|
| `daily` | 180 days | ~124 | 1 trading day | Recent short-term patterns (flags, H&S forming in last few months) |
| `weekly` | 730 days | ~104 | 1 week | Longer structural patterns (multi-month double tops, large wedges) |

Both passes render the same 1200×600px chart. The weekly pass achieves 2-year coverage at the same visual density by resampling daily bars into weekly bars. Using 730 days of daily bars directly would make each candle ~3.5x smaller than the training distribution, and detection rates drop significantly.

**Incremental updates (`--only-new`):**

The daily cron runs with `--only-new`, which skips tickers whose `as_of_date` in `yolo_patterns` already matches the latest candle in `price_daily`. Only tickers that received a new price candle since the last run are re-processed. Each timeframe (daily/weekly) is checked independently.

**Coordinate mapping:**

The key engineering challenge is mapping YOLO pixel bounding boxes back to date/price coordinates for rendering on the interactive chart. At render time, matplotlib axis metadata is captured:

```python
pos  = ax.get_position()   # figure-fraction coordinates of the plot area
xlim = ax.get_xlim()       # bar-index range (float)
ylim = ax.get_ylim()       # price range

# Pixel → bar index → date
x_bar  = xlim[0] + (x_px - ax_x0_px) / (ax_x1_px - ax_x0_px) * (xlim[1] - xlim[0])
date   = dates[round(x_bar)]

# Pixel → price  (image y=0 is top; matplotlib y=0 is bottom)
fig_y  = fig_h_px - y_px
price  = ylim[0] + (fig_y - ax_y0_px) / (ax_y1_px - ax_y0_px) * (ylim[1] - ylim[0])
```

**Frontend rendering of YOLO boxes:**

Daily and weekly detections are rendered differently so they can be visually distinguished while coexisting on the same chart:

| Timeframe | Border style | Opacity | Label |
|-----------|-------------|---------|-------|
| `daily` | Dotted, 1.5px | 0.07 fill | `[D]` |
| `weekly` | Dashed, 2.2px, muted | 0.05 fill | `[W]` |

On mobile, labels are shortened using a compact map (`"Head and shoulders bottom"` → `"H&S Bot"`) and truncated at 11 characters. The number of annotations shown is capped (3 daily + 2 weekly on mobile vs 5 on desktop).

---

## ML pipeline

The `trader_koo/ml/` directory implements a LightGBM-based binary classifier for next-day directional prediction.

```
trader_koo/ml/
├── features.py          51 features across 12 categories
├── labels.py            Triple-barrier labeling (profit/stop/time)
├── trainer.py           LightGBM walk-forward with purged validation
├── scorer.py            Real-time scoring for dashboard integration
├── backtest.py          Walk-forward backtest with equity curve
├── macro_features.py    FRED-sourced macro indicators
├── external_data.py     FRED API client with caching
├── sector_rotation.py   Sector-relative momentum features
├── meta_label.py        Meta-labeling for false positive filtering
├── shap_analysis.py     SHAP feature importance and explanations
└── drift_detection.py   Feature distribution drift monitoring
```

### Feature engineering (51 features, 12 categories)

The feature set covers price action, momentum, volatility, volume, structure, and macro:

- **Price**: returns (1d, 5d, 20d), distance from MA (20/50/200), distance from 52-week high/low
- **Momentum**: RSI-14, MACD signal/histogram, Stochastic %K/%D, rate of change
- **Volatility**: ATR-14, Bollinger bandwidth, historical volatility (20d), VIX level
- **Volume**: relative volume (vs 20d avg), OBV trend, volume-price divergence
- **Structure**: support/resistance proximity, level touch count, gap proximity
- **Patterns**: hybrid pattern confidence, YOLO detection count, pattern agreement score
- **Options**: put/call OI ratio, implied volatility percentile
- **Macro**: Fed funds rate, yield curve slope, unemployment delta, ISM PMI
- **Sector**: relative strength vs SPY, sector rotation score
- **Calendar**: day of week, month, days to earnings
- **Regime**: HMM regime state (bull/bear/neutral)
- **Sentiment**: news sentiment score (when Alpha Vantage available)

### Labeling

Triple-barrier method labels each sample as 1 (profitable) or 0 (not):
- **Take-profit barrier**: price rises `2 × ATR` within the holding period
- **Stop-loss barrier**: price falls `1 × ATR` within the holding period
- **Time barrier**: 10 trading days — if neither barrier is hit, label based on final return sign

### Training

Walk-forward validation with purged gaps to prevent look-ahead bias:
- Train on expanding window (min 252 days)
- Purge 5-day gap between train and validation sets
- Validate on next 63 trading days (~1 quarter)
- Final model: LightGBM with class-weight balancing

### Meta-labeling

A second-stage LightGBM model that predicts whether the primary model's "buy" signal is a true positive. This acts as a false-positive filter — only trades where both the primary model and meta-model agree get through.

### Model serving

Trained models are saved as `.pkl` files in `/data/models/`. The scorer loads the latest model at request time and returns a probability. If no model file exists, the ML score is omitted from the dashboard payload.

---

## Crypto subsystem

The crypto subsystem provides real-time and historical analysis for 5 pairs: BTC/USDT, ETH/USDT, SOL/USDT, XRP/USDT, DOGE/USDT.

```
Binance WebSocket (continuous)
    │
    ▼
CandleAggregator
    │  Builds 1m, 5m, 15m, 1h, 4h bars from raw 1m klines
    │
    ├──▶ SQLite (crypto_bars table)
    │
    ├──▶ WebSocket broadcast (/ws/crypto)
    │       └── React frontend receives real-time price updates
    │
    └──▶ REST endpoints (/api/crypto/*)
            ├── /prices         → latest price + 24h change
            ├── /history        → OHLCV bars for charting
            ├── /indicators     → RSI, MACD, Stochastic, Bollinger, MA
            ├── /structure      → order flow analysis, long/short OI
            ├── /correlation    → BTC-SPY rolling correlation
            └── /open-interest  → aggregated OI data
```

### CandleAggregator

The aggregator receives 1-minute klines from Binance and builds higher-timeframe bars in memory. When a bar closes (e.g., a 5-minute boundary is crossed), the completed bar is:
1. Stored in `crypto_bars` (SQLite)
2. Broadcast to all connected WebSocket clients
3. Available via REST for historical queries

### Market insights

The crypto module computes several cross-asset metrics:
- **BTC-SPY correlation**: rolling 30-day Pearson correlation to detect decoupling/recoupling
- **Regime classification**: trending/ranging/volatile based on ADX + ATR percentile
- **Dominance tracking**: BTC market cap share among the 5 tracked pairs

---

## Real-time equity streaming

```
Browser (React)                      FastAPI Server                 Finnhub
     │                                     │                          │
     ├─── WS connect /ws/equities ────────▶│                          │
     │                                     │                          │
     ├─── {"subscribe": "AAPL"} ──────────▶│                          │
     │                                     ├── subscribe("AAPL") ────▶│
     │                                     │                          │
     │                                     │◀── {"s":"AAPL","p":..} ──│
     │◀── {"ticker":"AAPL","price":..} ────│                          │
     │                                     │                          │
     ├─── {"unsubscribe": "AAPL"} ────────▶│                          │
     │                                     ├── unsubscribe("AAPL") ──▶│
```

SPY and QQQ are always subscribed. When a user opens a chart for a specific ticker, the frontend sends a subscribe message. The server manages a reference count — when no clients are watching a symbol, it unsubscribes from Finnhub to stay within the free tier's connection limits.

The live tick price is merged into the dashboard chart payload as a real-time "current candle" extension of the last daily bar.

---

## Paper trading

The paper trading system simulates trades with full lifecycle tracking, used to evaluate the ML model and pattern detection signals in forward-test conditions.

### Decision pipeline

```
Analyst Agent  →  Debate Agent  →  Risk Agent  →  Portfolio Agent
   │                  │                │                │
   │ Generate         │ Challenge      │ Size the       │ Final
   │ trade thesis     │ the thesis     │ position       │ go/no-go
   │ from signals     │ (bull vs bear) │ (ATR-based)    │ decision
```

Each stage is LLM-driven (with structured output validation). The multi-agent debate format reduces single-point-of-failure reasoning.

### Position sizing

ATR-based position sizing ensures consistent risk per trade:
- Risk per trade: fixed dollar amount (configurable)
- Stop distance: `1.5 × ATR-14`
- Position size: `risk_amount / stop_distance`
- Maximum position: capped at percentage of portfolio

### Performance tracking

| Metric | Table |
|--------|-------|
| Individual trades | `paper_trades` (entry/exit price, PnL, R-multiple, hold duration) |
| Daily snapshots | `paper_portfolio_snapshots` (equity, cash, positions, drawdown) |
| Bot versions | `bot_versions` (tracks which model/parameters produced each trade) |

Dashboard metrics: equity curve, win rate, average R-multiple, Sharpe ratio, profit factor, max drawdown.

---

## Daily report pipeline

The nightly cron (`daily_update.sh`) orchestrates three phases:

```
Phase 1: Ingest (update_market_db.py)
    ├── Fetch S&P 500 ticker list
    ├── Download OHLCV for each ticker (yfinance)
    ├── Scrape fundamentals (Finviz)
    └── Record ingest_runs + ingest_ticker_status

Phase 2: Pattern detection (run_yolo_patterns.py --only-new)
    ├── Daily pass: 180-day lookback, ~124 bars
    ├── Weekly pass: 730-day lookback, ~104 weekly bars
    └── Store detections in yolo_patterns

Phase 3: Report generation (generate_daily_report.py)
    ├── Market summary (SPY, QQQ, VIX, breadth)
    ├── Top movers (by return, by volume)
    ├── Pattern highlights (highest confidence across all layers)
    ├── Earnings preview (upcoming week)
    ├── Sentiment composite (Fear & Greed style)
    └── Write daily_report_latest.json + .md + timestamped archive
```

Reports are served via `/api/report/daily` and optionally emailed to subscribers.

---

## Frontend (React 19 + Vite 8 + TypeScript)

The v2 frontend lives in `trader_koo/frontend-v2/`. It is built at deploy time and served as static files by FastAPI at `/` (root).

### Pages (10, lazy-loaded)

| Page | Route | Purpose |
|------|-------|---------|
| `ChartPage` | `/chart/:ticker?` | Main dashboard — OHLCV chart with overlays |
| `CryptoPage` | `/crypto` | Real-time crypto prices + indicators |
| `OpportunitiesPage` | `/opportunities` | Valuation screens + filters |
| `ReportPage` | `/report` | Daily market report viewer |
| `EarningsPage` | `/earnings` | Earnings calendar + estimates |
| `VixPage` | `/vix` | VIX analysis + term structure |
| `PaperTradePage` | `/paper` | Paper trading dashboard |
| `PolymarketPage` | `/polymarket` | Prediction market probabilities |
| `GuidePage` | `/guide` | User guide / documentation |
| `NotFoundPage` | `*` | 404 handler |

### State management

- **Zustand** for chart state (selected ticker, timeframe, overlay toggles, zoom range)
- React Router for navigation with lazy-loaded route splitting
- No global API state library — data is fetched per-page with standard `useEffect` + `useState`

### Component architecture

```
trader_koo/frontend-v2/src/components/
├── layout/               Sidebar, header, responsive shell
├── chart/                ChartWorkspace, ChartPlotPanel, ChartToolbar,
│                         ChartOverlayControls, ChartFundamentals,
│                         ChartCommentarySidebar, PatternTabs, LevelsCard,
│                         GapsCard, YoloAuditSection, GlassCard
├── crypto/               CryptoChartPanel, CryptoPriceCards,
│                         CryptoIndicatorCards, CryptoStructureCards,
│                         CryptoInsightCards, CryptoAnalyticsPanels,
│                         CryptoToolbar, CryptoCardPrimitives
├── vix/                  VIX gauges, term structure, historical charts
├── earnings/             Earnings calendar, estimate tables
├── paper/                Trade list, equity curve, performance metrics
├── report/               Report viewer, section renderers
├── sentiment/            Fear & Greed gauge, sentiment breakdown
├── ui/                   Shared primitives (buttons, modals, tooltips)
├── PlotlyWrapper.tsx     Safe react-plotly.js interop (lifecycle management)
├── FearGreedGauge.tsx    Composite sentiment gauge
├── PipelineOpsPanel.tsx  Admin pipeline controls
└── KeyboardShortcutsModal.tsx
```

### PlotlyWrapper

A wrapper around `react-plotly.js` that handles the lifecycle issues of Plotly in React:
- Uses `Plotly.newPlot()` instead of `Plotly.react()` to avoid chart collapse on resize
- Manages explicit height to prevent autosize issues
- Preserves zoom state across re-renders via `relayoutData` capture
- Handles cleanup on unmount to prevent memory leaks

### Styling

- Tailwind CSS with dark/light mode toggle
- Dark theme aligned with the kooexperience.com design system
- Responsive layout — sidebar collapses on mobile, charts resize fluidly

---

## SMTP email reports

The email system is entirely optional and disabled if `TRADER_KOO_SMTP_HOST` is not set.

### How it works

`_smtp_settings()` reads configuration from env vars at call time (not at startup), so settings can be changed without redeploying. `_send_smtp_email()` supports three security modes:

| Mode | Behaviour |
|------|-----------|
| `ssl` | Opens `SMTP_SSL` connection immediately (port 465) |
| `starttls` | Plain SMTP connection, then `STARTTLS` upgrade (port 587, default) |
| `none` | Plain SMTP, no encryption (local/dev only) |

### Subscriber management

The email router supports a full subscribe/unsubscribe flow:
- `POST /api/email/subscribe` — adds subscriber, sends confirmation email
- `GET /api/email/confirm/{token}` — confirms subscription via one-time token
- `POST /api/email/unsubscribe` — removes subscriber
- `GET /api/email/chart-preview/{ticker}` — generates a chart image for email embedding

### Gmail setup

1. Enable 2FA on the Google account
2. Generate an App Password (Google Account → Security → App Passwords)
3. Set env vars:

```
TRADER_KOO_SMTP_HOST=smtp.gmail.com
TRADER_KOO_SMTP_PORT=587
TRADER_KOO_SMTP_SECURITY=starttls
TRADER_KOO_SMTP_USER=YOUR_EMAIL
TRADER_KOO_SMTP_PASS=YOUR_APP_PASSWORD
TRADER_KOO_SMTP_FROM=YOUR_EMAIL
TRADER_KOO_REPORT_EMAIL_TO=YOUR_EMAIL
```

### Report format

Reports are written to `$TRADER_KOO_REPORT_DIR` (default `/data/reports`) after each daily update run:
- `daily_report_latest.json` — always the most recent (overwritten each run)
- `daily_report_latest.md` — Markdown version
- `daily_report_YYYYMMDDTHHMMSSZ.json` — timestamped archive

---

## Security

### Authentication middleware

A Starlette `@app.middleware("http")` intercepts `/api/admin/*` requests. It extracts `X-API-Key` from headers and compares against `TRADER_KOO_API_KEY` using `secrets.compare_digest()` to prevent timing-based key enumeration.

When `TRADER_KOO_API_KEY` is unset (local dev), admin auth is bypassed.

### Rate limiting

Rate limiting via `slowapi`:
- **Public endpoints**: 100 requests/minute per IP
- **Admin endpoints**: 10 requests/minute per IP
- Applied at the router level, not globally

### Audit logging

When an unauthorized request is blocked, the middleware logs:
- Client IP (extracted from `X-Forwarded-For`, Railway-aware)
- User-Agent
- Referer

All admin actions write to the `audit_logs` table with IP tracking, making it easy to trace activity.

### CORS

Configurable allowed origins via `TRADER_KOO_CORS_ORIGINS` env var. Defaults to the production domain. Wildcard `*` is never used in production.

### Secret handling

- All secrets stored as Railway env vars, never hardcoded
- Secret redaction in log output (API keys, tokens masked)
- `.env.example` with placeholder values for documentation
- `GET /api/config` returns the API key for frontend auth — acceptable for a personal tool, not suitable for multi-tenant use

---

## CV label pipeline

The `cv/` module and `scripts/grow_gold_labels.py` implement a human-in-the-loop label curation workflow for training a custom pattern detector (future work):

```
1. detect   — run proxy_patterns.py over all tickers, render annotated images
2. review   — human inspects images_review/, marks accept/reject in CSV
3. gold     — approved detections merged into gold_labels.csv
4. calibrate — sweep detection thresholds against gold set, optimize F1
5. pseudo   — high-confidence model predictions promoted to training data
```

This pipeline supports Label Studio for the human review step (optional) or a simple CSV-based workflow. The current state uses the rule-based `cv/proxy_patterns.py` as a detection oracle; the intent is to eventually fine-tune a YOLO model on the gold labels.

---

## Configuration

All detection components are configured via dataclasses with sensible defaults. Instances are created once at startup in `main.py` and passed to each function call. To tune any parameter, change the dataclass default or instantiate with overrides:

```python
# Example: tighter level clustering
LEVEL_CFG = LevelConfig(level_tol_atr=0.40, primary_each_side=3)
```

| Config class | Key parameters |
|---|---|
| `FeatureConfig` | `atr_length=14`, `ma_windows=(20,50,100,200)` |
| `LevelConfig` | `level_tol_atr=0.60`, `recency_half_life_days=45`, `primary_each_side=2` |
| `GapConfig` | `gaps_lookback_months=18`, `max_dist_pct=0.12`, `only_open=True` |
| `TrendlineConfig` | `lookback_bars=50`, `min_touches=2`, `touch_tolerance_pct=0.015` |
| `PatternConfig` | `flag_pole_bars=8`, `min_pole_return=0.04`, `wedge_converge_ratio=0.35` |
| `CandlePatternConfig` | `lookback_bars=20` |
| `HybridPatternConfig` | `candle_weight=0.20`, `volume_weight=0.15`, `breakout_weight=0.15` |
| `CVProxyConfig` | `lookback_bars=100`, `min_shape_r2=0.45`, `wedge_converge_ratio=0.65` |

---

## Scheduler

APScheduler `BackgroundScheduler` runs inside the FastAPI process.

| Schedule | Trigger | Job |
|----------|---------|-----|
| Mon–Fri 22:00 UTC | `CronTrigger(hour=22, minute=0, day_of_week="mon-fri", timezone="UTC")` | `daily_update.sh` (ingest + YOLO + report) |
| Saturday 00:30 UTC | `CronTrigger(hour=0, minute=30, day_of_week="sat", timezone="UTC")` | YOLO full seed (all tickers, both timeframes) |

22:00 UTC covers all US market closes (Mon–Fri 4pm ET = 21:00 UTC in EDT / 22:00 UTC in EST — the schedule uses 22:00 as a conservative post-close time).

The `POST /api/admin/trigger-update` endpoint modifies the job's `next_run_time` to `now()` to fire it immediately without redeploying. Supports `?mode=full|yolo|report` to run specific phases.

---

## Testing

Tests live in `tests/` and run via `python -m pytest tests/ -v`.

Current baseline: **578 passed**.

Key conventions:
- No mock data for price/financial data — test against real schema and code paths
- External services (LLM providers, HTTP APIs) are mocked
- AAA pattern (Arrange-Act-Assert)
- Tests must pass locally before pushing

No frontend test coverage yet.

---

## Known limitations

| Area | Limitation | Mitigation / Future plan |
|------|-----------|--------------------------|
| **Database** | SQLite: fine for single-process, would need Postgres for multi-worker | Acceptable for current single-service architecture |
| **YOLO model** | Pre-trained HuggingFace model, not fine-tuned on gold labels yet | CV label pipeline exists; gold-label fine-tuning is planned |
| **ML model** | AUC 0.5235 — useful as a filter, not a standalone signal generator | Meta-labeling reduces false positives; more features and data needed |
| **FRED features** | Per-date API calls are too slow for bulk fetch | Needs bulk-fetch architecture or pre-cached daily snapshots |
| **Frontend tests** | No test coverage for React components | Planned: Vitest + React Testing Library |
| **Daily cron** | In-process APScheduler — if process restarts mid-run, job is lost | For higher reliability, could move to Railway cron or GitHub Actions |
| **Public config** | `GET /api/config` returns API key in browser — acceptable for personal tool | Not suitable for multi-tenant deployment without auth redesign |
