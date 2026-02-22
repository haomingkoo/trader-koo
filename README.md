# trader-koo

A full-stack stock market analysis dashboard covering the entire S&P 500. Built to understand how to instrument a production ML-assisted data pipeline from scratch — market data ingestion, rule-based chart pattern detection, a computer vision layer, and a live interactive dashboard — all deployed on a single Railway service with a persistent SQLite database.

> **NOT FINANCIAL ADVICE.** This is a personal research and learning project.

---

## What it does

### Dashboard
Dark-theme single-page app with three tabs:

- **Guide** — explains every indicator and signal the dashboard surfaces, with NFA disclaimers
- **Opportunities** — screener across all ~510 S&P 500 tickers; filter by PEG ratio, analyst discount %, valuation label (deep value / undervalued / overvalued)
- **Chart + Levels** — interactive candlestick chart for any ticker with overlays for support/resistance levels, gaps, trendlines, rule-based patterns, CV proxy patterns, and YOLOv8 AI detections

### What gets detected per ticker

| Layer | What it finds |
|---|---|
| **Support / Resistance Levels** | Clusters of pivot highs/lows, scored by recency (45-day half-life), tiered as primary / secondary / fallback |
| **Gaps** | Unfilled bull/bear price gaps within 18 months and 12% of current price |
| **Trendlines** | Rising support and falling resistance lines fitted through swing pivots |
| **Rule-based patterns** | Bull/bear flags, rising/falling wedges (geometry + R² threshold) |
| **Candlestick patterns** | Hammer, shooting star, morning/evening star, engulfing and more |
| **Hybrid scoring** | Blends rule confidence (50%) + candle signal (20%) + volume (15%) + breakout state (15%) |
| **CV proxy patterns** | Image-style OHLC geometry scorer — same patterns, different signal source |
| **YOLO AI patterns** | YOLOv8 model (`foduucom/stockmarket-pattern-detection-yolov8`) run on rendered chart images; two passes — 180-day daily and 730-day weekly |

### Data pipeline

- **~510 tickers**: full S&P 500 + VIX, SPY, QQQ, DJI, TNX, SVIX as market context
- **Daily refresh**: price (yfinance), fundamentals (Finviz), options IV — runs automatically at 22:00 UTC Mon–Fri
- **YOLO seed**: runs once on first deploy; daily incremental updates only re-process tickers with new candles
- **Storage**: single SQLite file on a Railway persistent volume — no external DB needed

---

## Screenshots / demo

> Add screenshots here once deployed.

Live at: **https://trader-koo-production.up.railway.app**

---

## Quick start (local)

### Prerequisites
- Python 3.11+
- ~4 GB disk (PyTorch CPU build)

### 1. Install

```bash
git clone https://github.com/haomingkoo/trader-koo.git
cd trader-koo

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# CPU-only PyTorch first (saves ~1 GB vs the default CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r trader_koo/requirements.txt
pip install -e .
```

### 2. Seed market data

```bash
# Full S&P 500 + context tickers (~30 min first run)
python trader_koo/scripts/update_market_db.py \
    --use-sp500 \
    --price-lookback-days 365 \
    --db-path trader_koo/data/trader_koo.db

# Or just a few tickers to try it out
python trader_koo/scripts/update_market_db.py \
    --tickers "SPY,QQQ,AAPL,NVDA,MSFT" \
    --db-path trader_koo/data/trader_koo.db
```

### 3. (Optional) Run YOLO pattern seed

```bash
# Both daily + weekly passes, all tickers (~25 min)
python trader_koo/scripts/run_yolo_patterns.py \
    --db-path trader_koo/data/trader_koo.db \
    --timeframe both

# Visual test — saves annotated PNGs to /tmp/yolo_test/
python trader_koo/scripts/test_yolo_visual.py \
    --tickers SPY AAPL NVDA \
    --db-path trader_koo/data/trader_koo.db \
    --out-dir /tmp/yolo_test
open /tmp/yolo_test
```

### 4. Start the API

```bash
MPLBACKEND=Agg python -m uvicorn trader_koo.backend.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## API reference

All `/api/*` routes require an `X-API-Key` header when `TRADER_KOO_API_KEY` is set. Locally, leave it unset — auth is disabled automatically.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | DB liveness check |
| GET | `/api/status` | Data freshness, last ingestion run |
| GET | `/api/config` | Returns API key for the frontend client |
| GET | `/api/tickers` | All tickers in DB |
| GET | `/api/dashboard/{ticker}` | Full chart payload (OHLCV + all layers) |
| GET | `/api/opportunities` | Valuation screening across all tickers |
| GET | `/api/yolo/{ticker}` | YOLO pattern detections for one ticker |
| POST | `/api/admin/trigger-update` | Manually trigger the daily data refresh |
| POST | `/api/admin/run-yolo-seed` | Trigger full YOLO seed (background thread) |

---

## Deployment (Railway)

The app is designed for a single Railway service with a persistent `/data` volume.

### Environment variables

| Variable | Description |
|---|---|
| `TRADER_KOO_API_KEY` | Random secret — enforces `X-API-Key` auth on all `/api/*` routes |
| `TRADER_KOO_DB_PATH` | SQLite path, e.g. `/data/trader_koo.db` |
| `TRADER_KOO_LOG_DIR` | Log directory, e.g. `/data/logs` |
| `TRADER_KOO_LOG_LEVEL` | `INFO` (default) or `DEBUG` |
| `TRADER_KOO_ALLOWED_ORIGIN` | Your Railway app URL (CORS) |

### First deploy

1. Push to GitHub, connect repo in Railway, add a volume mounted at `/data`
2. Set the env vars above
3. Deploy — `start.sh` seeds the DB automatically if `/data/trader_koo.db` doesn't exist
4. Once running, trigger the YOLO seed (takes ~25 min, runs in background):

```bash
curl -X POST "https://your-app.up.railway.app/api/admin/run-yolo-seed" \
     -H "X-API-Key: YOUR_KEY"
```

### Daily cron

APScheduler runs `daily_update.sh` inside the process at 22:00 UTC Mon–Fri. No external cron or worker needed.

---

## Project structure

```
trader-koo/
├── start.sh                        # Railway entrypoint — seed DB then start uvicorn
├── railway.toml                    # Railway build + deploy config
├── pyproject.toml
│
└── trader_koo/
    ├── backend/
    │   └── main.py                 # FastAPI app — all endpoints + scheduler
    ├── frontend/
    │   └── index.html              # Single-page dashboard (Plotly, dark theme)
    ├── data/
    │   └── schema.py               # SQLite schema helpers
    ├── features/
    │   ├── technical.py            # ATR, moving averages, pivot detection
    │   └── candle_patterns.py      # Candlestick signal detection
    ├── structure/
    │   ├── levels.py               # Support/resistance clustering + tiering
    │   ├── gaps.py                 # Price gap detection + fill tracking
    │   ├── trendlines.py           # Trendline fitting through swing pivots
    │   ├── patterns.py             # Rule-based flag/wedge detection
    │   └── hybrid_patterns.py      # Multi-source confidence blending
    ├── cv/
    │   ├── proxy_patterns.py       # CV-style geometry pattern scorer
    │   ├── compare.py              # Hybrid vs CV consensus comparison
    │   └── README.md               # CV label pipeline walkthrough
    └── scripts/
        ├── daily_update.sh         # Cron: prices + fundamentals + YOLO (Mon–Fri)
        ├── update_market_db.py     # Market data ingestion (yfinance + Finviz)
        ├── run_yolo_patterns.py    # YOLOv8 batch pattern detection
        ├── test_yolo_visual.py     # Local visual test — saves annotated PNGs
        └── grow_gold_labels.py     # CV label pipeline (detect → review → gold)
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Scheduling | APScheduler (in-process cron) |
| Database | SQLite (Railway persistent volume) |
| Market data | yfinance, Finviz |
| Chart rendering | mplfinance + matplotlib (headless) |
| AI pattern detection | YOLOv8 via `ultralyticsplus` |
| Frontend | Vanilla JS + Plotly.js (no build step) |
| Deployment | Railway (nixpacks, single service) |

---

## License

MIT
