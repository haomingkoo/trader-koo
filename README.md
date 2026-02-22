# trader_koo

Chart pattern scanner for S&P 500 equities. Detects head & shoulders, wedges, double tops/bottoms, flags and triangles using rule-based + CV-assisted analysis. FastAPI backend with live visualization dashboard and automated daily data pipeline.

## Features

- **Pattern detection** — head & shoulders, inverse H&S, double top/bottom, rising/falling wedge, bull/bear flag, symmetrical triangle
- **Market context** — VIX, SPY, QQQ, DJI, TNX fetched daily alongside S&P 500 tickers
- **Fundamentals** — Finviz snapshots (P/E, PEG, EPS, analyst targets) for valuation screening
- **Live dashboard** — dark-theme frontend served directly by the API; chart overlays for levels, gaps, trendlines and patterns
- **CV pipeline** — weak-label → human review → gold label loop for training a YOLO chart pattern detector
- **Daily cron** — automated data refresh script, production-ready for Railway deployment

## Project structure

```
trader_koo/          # Python package
├── backend/         # FastAPI app
├── cv/              # CV pipeline (proxy patterns, label management)
├── data/            # SQLite DB, label CSVs, cache (DB not in git)
├── features/        # ATR, MAs, pivot detection
├── frontend/        # Single-page dashboard (served by FastAPI)
├── scripts/         # Ingestion, pattern detection, label scripts
└── structure/       # Levels, gaps, trendlines, pattern scoring
```

## Quick start (local)

### 1. Create venv and install deps
```bash
cd ~/dev/trader-koo
python -m venv trader_koo/.venv
source trader_koo/.venv/bin/activate
pip install -r requirements.txt
pip install matplotlib  # for chart image rendering
```

Optional — larger candlestick pattern set:
```bash
pip install TA-Lib
```

### 2. Ingest market data
```bash
# Full S&P 500 + market context tickers (VIX, SPY, QQQ, DJI, TNX, SVIX)
python trader_koo/scripts/update_market_db.py \
    --use-sp500 \
    --price-lookback-days 10 \
    --fund-min-interval-hours 24

# Or a small set for testing
python trader_koo/scripts/update_market_db.py \
    --tickers "SPY,QQQ,AAPL,NVDA,MSFT"
```

### 3. Start the API
```bash
# From repo root (~/dev/trader-koo)
python -m uvicorn trader_koo.backend.main:app --reload
```

### 4. Open dashboard
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) — type a ticker, hit Load.

### 5. Health check
```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/status
```

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /api/health` | DB liveness check |
| `GET /api/status` | Data freshness + ingestion run status |
| `GET /api/tickers` | All tickers in DB |
| `GET /api/dashboard/{ticker}?months=3` | Full chart payload |
| `GET /api/opportunities` | Valuation screening (undervalued/overvalued) |

## CV label pipeline

Detect patterns and render chart images:
```bash
python trader_koo/scripts/grow_gold_labels.py detect \
    --render-images \
    --batch-csv trader_koo/data/cv/batch_samples.csv
```

Review annotated images in `trader_koo/data/cv/images_review/`, then update `gold_labels.csv`.

## Deployment (Railway)

See `railway.toml`. Set these environment variables in Railway:

| Variable | Value |
|---|---|
| `TRADER_KOO_API_KEY` | random secret (enforces auth on all `/api/*` routes) |
| `TRADER_KOO_DB_PATH` | `/app/trader_koo/data/trader_koo.db` |
| `TRADER_KOO_ALLOWED_ORIGIN` | your Railway app URL |

Local dev: leave `TRADER_KOO_API_KEY` unset — auth is disabled automatically.

## License

MIT
