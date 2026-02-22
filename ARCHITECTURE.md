# Architecture

This document explains how trader-koo is built — the data flow, module responsibilities, key algorithms, and deployment topology. Intended for anyone who wants to understand the internals or fork/extend the project.

---

## High-level overview

```
                          ┌────────────────────────────────────────────┐
                          │              Railway Service                │
                          │                                             │
   Browser  ─── HTTPS ──▶│  FastAPI (uvicorn, port 8080)              │
                          │      │                                      │
                          │      ├─ serves index.html (GET /)          │
                          │      ├─ /api/dashboard/{ticker}            │
                          │      ├─ /api/opportunities                 │
                          │      └─ /api/admin/*  (auth required)      │
                          │                                             │
                          │  APScheduler (in-process, Mon–Fri 22 UTC) │
                          │      └─ daily_update.sh                   │
                          │           ├─ update_market_db.py           │
                          │           └─ run_yolo_patterns.py          │
                          │                                             │
                          │  /data/trader_koo.db  (persistent volume)  │
                          │  /data/logs/                               │
                          │  /data/.ultralytics/  (YOLO model cache)   │
                          └────────────────────────────────────────────┘
```

No separate worker process, no message queue, no external database. The entire system — API server, background scheduler, and data pipeline — runs in a single process on a single Railway service.

---

## Deployment

### Railway setup

| File | Purpose |
|---|---|
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

### First-run DB seed

`start.sh` checks for `/data/trader_koo.db`. If it doesn't exist, it runs `update_market_db.py` with `--price-lookback-days 365` to pull a full year of history for all ~510 S&P 500 + context tickers. This takes ~20–30 min but only happens once — subsequent deploys skip it.

YOLO patterns are seeded separately via `POST /api/admin/run-yolo-seed` after the first deploy.

### YOLO model caching

The YOLOv8 model (`foduucom/stockmarket-pattern-detection-yolov8`) is downloaded from HuggingFace on first use and cached to `/data/.ultralytics/`. Setting `ULTRALYTICS_CONFIG_DIR=/data/.ultralytics` redirects the cache to the persistent volume so it survives redeploys.

---

## Data layer

### SQLite on a persistent volume

All data lives in a single SQLite file at `$TRADER_KOO_DB_PATH` (default `/data/trader_koo.db`). SQLite is sufficient because:
- All reads are served by one process (no concurrent writers)
- The entire dataset fits comfortably in a few hundred MB
- No managed database cost or connection pooling overhead

### Schema

#### `price_daily`
Core OHLCV data. The primary workhorse — almost every computation starts here.
```sql
ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL
INDEX: (ticker, date)
```

#### `finviz_fundamentals`
Finviz scrape snapshots (P/E, PEG, EPS, analyst target price). Multiple snapshots per ticker are kept; queries use `MAX(snapshot_ts)` to get the latest.
```sql
ticker TEXT, snapshot_ts TEXT, price REAL, pe REAL, peg REAL,
eps_ttm REAL, eps_growth_5y REAL, target_price REAL, discount_pct REAL, ...
```

#### `options_iv`
Options chain aggregates (call/put open interest, implied volatility).
```sql
ticker TEXT, snapshot_ts TEXT, option_type TEXT, implied_vol REAL, open_interest INTEGER, ...
```

#### `yolo_patterns`
Pre-computed YOLOv8 detections. Stored by `(ticker, timeframe)` pair so daily and weekly rows coexist independently. `as_of_date` tracks which price candle the detection was computed against — used by `--only-new` to skip tickers whose patterns are already current.
```sql
ticker TEXT, timeframe TEXT, pattern TEXT, confidence REAL,
x0_date TEXT, x1_date TEXT, y0 REAL, y1 REAL,
lookback_days INTEGER, as_of_date TEXT, detected_ts TEXT
```

#### `ingest_runs` / `ingest_ticker_status`
Ingestion audit trail. Each `update_market_db.py` run records start/finish timestamps, total/ok/failed ticker counts, and a per-ticker status row for debugging partial failures.

---

## Backend (main.py)

### Request lifecycle — `/api/dashboard/{ticker}`

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
13. Fetch fundamentals    →  latest finviz_fundamentals row
14. Fetch options         →  latest options_iv rows
15. Merge pattern overlays (top 10 by confidence across all sources)
16. Return JSON payload
```

### Authentication middleware

A single Starlette middleware intercepts every `/api/*` request (except `/api/health` and `/api/config`). It extracts `X-API-Key` from headers and compares against `TRADER_KOO_API_KEY` using `secrets.compare_digest()` to prevent timing attacks. When `TRADER_KOO_API_KEY` is unset (local dev), auth is bypassed entirely.

### Scheduler

APScheduler `BackgroundScheduler` runs inside the FastAPI process. A single `CronTrigger(hour=22, minute=0, day_of_week="mon-fri", timezone="UTC")` fires `daily_update.sh` via `subprocess.run`. This covers all US market closes (Mon–Fri 4pm ET = 21:00 UTC in EDT / 22:00 UTC in EST — the schedule uses 22:00 as a conservative post-close time).

The `POST /api/admin/trigger-update` endpoint modifies the job's `next_run_time` to `now()` to fire it immediately without waiting for the cron schedule.

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
4. Each zone gets a score = Σ(1 / (1 + days_since_touch / 45))
   (45-day half-life recency weighting)
5. Select top 2 support + top 2 resistance zones as "primary"
6. Next 2 each side as "secondary"
7. If no primary found within 20% of current price, add fallback (MA-anchored)
```

---

## Pattern detection layers

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
- R² of both fits must be `>= 0.45`

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

The only layer that uses a trained ML model. Rather than running inference on live requests (too slow), detections are pre-computed and stored in `yolo_patterns`.

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
The YOLOv8 model was trained on charts with ~100–130 bars visible. At 180 daily bars, candle density matches the training distribution. At 365 daily bars, candles are visually too small and detection quality drops significantly. The weekly pass achieves longer-term coverage (~104 weekly bars over 2 years) at the same visual density.

**Coordinate mapping:**
The key challenge is mapping YOLO pixel bounding boxes back to date/price space. This requires capturing matplotlib's axis position (in figure-fraction coordinates) and axis limits (bar index range, price range) at render time:

```python
axes_info = {
    "ax_x0": pos.x0 * fig_w_px,   # left edge of price axis in pixels
    "ax_x1": pos.x1 * fig_w_px,   # right edge
    "ax_y0": pos.y0 * fig_h_px,   # bottom edge
    "ax_y1": pos.y1 * fig_h_px,   # top edge
    "xlim": ax.get_xlim(),         # (bar_index_min, bar_index_max)
    "ylim": ax.get_ylim(),         # (price_min, price_max)
    "fig_h_px": fig_h_px,
}

# x pixel → bar index → date
x_bar = xlim[0] + (x_px - ax_x0) / (ax_x1 - ax_x0) * (xlim[1] - xlim[0])
date = dates[round(x_bar)]

# y pixel → price  (image y=0 is top, figure y=0 is bottom)
fig_y = fig_h_px - y_px
price = ylim[0] + (fig_y - ax_y0) / (ay_y1 - ax_y0) * (ylim[1] - ylim[0])
```

---

## Frontend (index.html)

A single ~1400-line HTML file — no build step, no bundler, no framework. Served directly by FastAPI as a static file.

### Why this approach?

- Zero build tooling to maintain or fail in CI
- The entire UI is one file that can be read, debugged, and deployed trivially
- Plotly.js handles all the heavy charting; vanilla JS handles the rest
- Dark-theme CSS variables make it easy to tweak the color palette

### Chart rendering

Plotly traces are rebuilt from scratch on every ticker load (`renderChart()`):
- Candlestick trace (green up, red down, standard OHLC)
- Volume bar trace (same green/red coloring)
- MA line traces (20/50/100/200)
- `layout.shapes`: levels (horizontal lines + zone rectangles), gaps (filled rectangles), trendlines (dashed lines), pattern geometry (dual-line flags/wedges), YOLO bounding boxes (dotted daily, dashed weekly)
- `layout.annotations`: level labels, pattern labels

Zoom state is preserved across re-renders by capturing `relayoutData` and reapplying `xaxis.range`. Double-click resets to the default view window.

### API key flow

The frontend calls `GET /api/config` on load to retrieve the API key. This allows the dashboard to authenticate subsequent requests without requiring the user to input a key. For public deployments, this means the API key is visible in the browser — acceptable for a personal tool, not for multi-tenant production use.

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

## CV label pipeline

The `cv/` module and `scripts/grow_gold_labels.py` implement a human-in-the-loop label curation workflow for training a custom pattern detector (future work):

```
1. detect   — run proxy_patterns.py over all tickers, render annotated images
2. review   — human inspects images_review/, marks accept/reject in CSV
3. gold     — approved detections merged into gold_labels.csv
4. calibrate — sweep detection thresholds against gold set, optimize F1
5. pseudo   — high-confidence model predictions promoted to training data
```

This pipeline uses `Label Studio` for the human review step (optional) or a simple CSV-based workflow. The current state uses the rule-based `cv/proxy_patterns.py` as a detection oracle; the intent is to eventually fine-tune a YOLO model on the gold labels.

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

## Known limitations and future work

- **SQLite concurrency**: fine for a single-process app but would need migration to Postgres for multi-worker or multi-service deployments
- **YOLO model**: the pre-trained HuggingFace model is a starting point; accuracy varies by ticker and market regime. Fine-tuning on the gold label set is the obvious next step
- **No authentication on `/api/config`**: the API key is returned publicly; acceptable for a personal tool
- **Daily cron is in-process**: if the process restarts during a run, the job is lost. For production use, an external scheduler (Railway cron, GitHub Actions) would be more reliable
- **No test suite**: the codebase has no automated tests; the visual test script (`test_yolo_visual.py`) serves as a manual regression check for the YOLO pipeline
