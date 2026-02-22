# trader_koo

Workspace for the swing-trading dashboard build (data + API + UI).

## Structure
- `data/` local datasets and cache
- `backend/` API/services
- `frontend/` web app
- `scripts/` ingestion / feature jobs
- `notebooks/` experiments
- `features/` feature engineering layer (ATR, MA, pivots)
- `structure/` structure layer (levels, gaps, trendlines)
- `cv/` CV staging area (dataset + weak-label roadmap)

## Suggested next build order
1. Build local data store (SQLite/DuckDB) in `data/`.
2. Add ingestion script in `scripts/` for OHLCV + fundamentals.
3. Add API in `backend/` to serve dashboard-ready payloads.
4. Add dark-theme frontend in `frontend/`.

## Local DB Ingestion (ready)
`scripts/update_market_db.py` now supports:
- Finviz fundamentals snapshots
- Daily OHLCV history
- Optional options IV snapshots

Canonical DB path (default): `data/trader_koo.db`
Use the **Runbook** section below for the exact commands.

### Production-friendly behavior (new)
- Price data is now incremental by default:
  - script reads latest stored date per ticker
  - fetches from a small lookback window (default `--price-lookback-days 10`)
  - appends/updates only recent rows via upsert
- Fundamentals/options can be rate-limited by freshness:
  - `--fund-min-interval-hours 12`
  - `--options-min-interval-hours 4`
- Each run writes operational metadata:
  - `ingest_runs` (job-level status)
  - `ingest_ticker_status` (per ticker diagnostics)
- Structured logs are written to:
  - `trader_koo/data/logs/update_market_db.log`

### Common ingestion commands
```bash
# normal incremental refresh
python scripts/update_market_db.py --tickers "SPY,QQQ,IWM,DIA,NVDA,AAPL,MSFT,TSLA" --include-options

# faster/smaller refresh cadence (e.g., every 15-30m)
python scripts/update_market_db.py --tickers "SPY,QQQ,IWM,DIA,NVDA,AAPL,MSFT,TSLA" --include-options --fund-min-interval-hours 24 --options-min-interval-hours 2

# full backfill refresh (rare)
python scripts/update_market_db.py --tickers "SPY,QQQ,IWM,DIA,NVDA,AAPL,MSFT,TSLA" --full-price-refresh --price-start 2018-01-01
```

## API + Frontend Starter (ready)

### Start API
```bash
# Option A (recommended, from repo root)
python -m uvicorn trader_koo.backend.main:app --reload

# Option B (if your cwd is trader_koo/)
python -m uvicorn backend.main:app --reload
```

### Open frontend
Open `http://127.0.0.1:8000` in browser (served by FastAPI).

In the page:
- API base is auto-detected
- choose ticker + window
- click `Load`

### API endpoints
- `GET /api/health`
- `GET /api/status`
- `GET /api/tickers`
- `GET /api/dashboard/{ticker}?months=3`

`/api/status` is intended for uptime/debug pages and returns:
- current API/DB availability
- latest ingestion run status
- data freshness (prices/fundamentals/options)
- row/ticker counts

## Persona-aligned refactor (implemented)

The dashboard logic now follows the 4-layer flow from `persona.txt`:

1. `data` layer:
- canonical schema enforcement (`date, open, high, low, close, volume`)
- protects against yfinance MultiIndex and malformed columns

2. `features` layer:
- ATR, ATR%, returns, MA20/50/100/200
- pivot highs/lows

3. `structure` layer:
- level zones with touches, last touch date, recency score
- gap engine with explicit fill logic
- trendline candidates with slope sanity + touch scoring

4. `viz` layer:
- chart overlays for levels, gaps, trendlines, and volume

## Runbook

### 1) Install deps
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Optional (advanced candlestick engine):
```bash
pip install TA-Lib
```

If `TA-Lib` is installed, the candlestick detector will automatically use a much larger pattern set (`CDL*` family) in addition to built-in heuristics.

### 2) Ingest/update DB (incremental, production-safe)
```bash
python scripts/update_market_db.py --tickers "SPY,QQQ,IWM,DIA,NVDA,AAPL,MSFT,TSLA" --include-options
```

### 3) Start API
```bash
python -m uvicorn backend.main:app --reload
```

### 4) Open UI
- Open `http://127.0.0.1:8000` in browser
- API base auto-detects (advanced override available)
- Load opportunities and chart tabs

### 5) Health checks
```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/status
```

## CV Weak-Label Loop (new)

Start the CV dataset workflow:

```bash
cd trader_koo
source .venv/bin/activate

python scripts/build_cv_weak_labels.py --tickers "SPY,QQQ,IWM,DIA,NVDA,AAPL,MSFT,TSLA" --render-images
```

Then:

1. Review `data/cv/review_queue.csv` and fill `data/cv/review_decisions.csv`.
2. Build gold labels:
```bash
python scripts/apply_cv_review.py --weak-labels-csv data/cv/weak_labels.csv --review-decisions-csv data/cv/review_decisions.csv
```
3. Promote high-confidence model predictions later:
```bash
python scripts/promote_cv_pseudo_labels.py --model-predictions-csv data/cv/model_predictions.csv --gold-labels-csv data/cv/gold_labels.csv
```

Details: `trader_koo/cv/README.md`

## Common local issues

### `db_exists: false` in `/api/health`
- This usually means your ingestion wrote DB to a different folder than the API is reading.
- Canonical DB path is:
  - `trader_koo/data/trader_koo.db`
- You can force path explicitly:
```bash
export TRADER_KOO_DB_PATH="/Users/koohaoming/aiap/all-assignments/trader_koo/data/trader_koo.db"
python -m uvicorn backend.main:app --reload
```

### `.venv/bin/activate` not found
- If current dir is `trader_koo/`, activate with:
```bash
source .venv/bin/activate
```
