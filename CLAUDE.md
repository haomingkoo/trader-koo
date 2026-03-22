# trader-koo — Claude Code Instructions

## Project Overview
Full-stack S&P 500 stock market analysis dashboard deployed on Railway.
Personal tool — not for public trading signals (NFA).
Live at: trader.kooexperience.com

## Stack
- **Backend**: FastAPI + APScheduler, SQLite at `/data/trader_koo.db` (Railway persistent volume)
- **Frontend**: `trader_koo/frontend-v2/` — React 19 + Vite 8 + TypeScript served at `/` (root)
- **Frontend (legacy)**: `trader_koo/frontend/index.html` — vanilla JS, no longer served
- **ML**: LightGBM walk-forward classifier (51 features, AUC 0.5235)
- **AI**: YOLOv8 (`foduucom/stockmarket-pattern-detection-yolov8`) batch-run nightly
- **Data**: yfinance + Finviz + Finnhub WS + Binance WS + FRED + Polymarket + optional Alpha Vantage
- **Deploy**: Railway asia-southeast1, nixpacks build, persistent `/data` volume

## Key Files
| File | Purpose |
|------|---------|
| `trader_koo/backend/main.py` | Slim app factory + middleware + static mounts (~600 lines) |
| `trader_koo/backend/routers/` | 11 API routers (82 endpoints total) |
| `trader_koo/frontend-v2/src/App.tsx` | Route definitions + safe lazy loading (10 pages) |
| `trader_koo/frontend-v2/src/components/PlotlyWrapper.tsx` | Safe `react-plotly.js` interop wrapper |
| `trader_koo/ml/features.py` | ML feature engineering (51 features) |
| `trader_koo/ml/trainer.py` | LightGBM walk-forward trainer |
| `trader_koo/ml/backtest.py` | Walk-forward backtester vs SPY |
| `trader_koo/crypto/service.py` | Binance WebSocket + candle aggregation |
| `trader_koo/scripts/daily_update.sh` | Nightly orchestrator: ingest → YOLO → report |
| `trader_koo/scripts/run_yolo_patterns.py` | YOLOv8 batch detection |
| `trader_koo/scripts/update_market_db.py` | yfinance + Finviz ingestion |
| `trader_koo/scripts/generate_daily_report.py` | Daily report generation |
| `trader_koo/structure/fear_greed.py` | Internal market sentiment composite + optional external news blend |
| `trader_koo/paper_trades.py` | Paper trade lifecycle + MTM + equity curve |

## Coding Rules

### NEVER use mock data
Do not use fake, placeholder, or mock stock/price/ticker data anywhere in this project.
This includes tests — test logic against the real schema and real code paths.
For unit tests that require isolation, mock external services (LLM, HTTP APIs) only,
never the underlying financial data itself.

### Import conventions
- `import datetime as dt` → always use `dt.datetime.now(...)`, never `datetime.now()` or `dt.now()`
- All datetime in backend should be UTC: `dt.datetime.now(dt.timezone.utc)`

### SQLite migrations
When adding columns to existing tables, always use ALTER TABLE with try/except in `ensure_schema()`:
```python
for col_ddl in ("ALTER TABLE t ADD COLUMN new_col TYPE DEFAULT val",):
    try:
        conn.execute(col_ddl)
    except sqlite3.OperationalError:
        pass  # column already exists
conn.commit()
```

### LLM output handling
Always sanitize LLM output (truncate field lengths) BEFORE schema validation.
Order: raw LLM response → `sanitize_llm_output(field_limits=...)` → `validate_llm_output(schema)`.

## Scheduling (Production Railway)
- **Daily Mon–Fri 22:00 UTC**: `daily_update.sh` (market ingest → YOLO → report), ~14–15 min
- **Saturday 00:30 UTC**: YOLO-only full seed (daily + weekly timeframes)
- On first deploy: seeds 365d price history if DB missing

## Admin API (X-API-Key required)
- `GET /api/admin/logs?name=yolo&lines=200` — tail YOLO logs on Railway
- `POST /api/admin/trigger-update?mode=full|yolo|report` — manual trigger
- `GET /api/admin/pipeline-status` — current stage
- `GET /api/admin/yolo-events` — per-ticker YOLO outcomes (ok/skipped/timeout/failed)

## Known Issues (track here)
- `data_source` column: old DB rows (pre-migration) have NULL — fixed via ALTER TABLE migration in `ensure_schema()`
- `audit_logs` table: initialised at startup via `ensure_audit_schema(conn)` — if missing on prod, redeploy to trigger startup hook
- LLM validation failures: fixed — sanitize before validate in `llm_narrative.py`
- HMM regime fitting is now clipped/stabilized locally and the previous sklearn warnings no longer reproduce in the current pytest baseline
- Market Sentiment widget now supports optional Alpha Vantage news sentiment, but it still does not use Twitter/Reddit social scraping
- FRED features need bulk-fetch architecture (per-date API calls too slow for ML training)
- Polymarket page shows aggregated YES/NO only — needs timeline sub-markets with individual date milestones
- ML model AUC 0.5235 pre-fix. Bug fixes applied: early stopping, barrier alignment, intraday barrier touches, cross-sectional ranking. Retraining pending.
- Frontend has zero test coverage
- `test_v2_shell_routes_disable_caching` references deleted `/v2` routes (broken test)
- VIX metrics caching: `regime_context` shape differs from `compute_vix_metrics()` return. Need to align fields or compute vix_metrics in nightly report.
- Sentiment NLP: scores are too coarse (0/50/100). StockTwits rule-based scoring misclassifies sarcasm/neutral posts.
- Pre-existing test failures: `test_alert_engine.py` (MAX_POLL_TICKERS import), `test_admin_auth.py` + `test_app.py` (/data read-only on macOS)

## Recent Changes (2026-03-22)
- Earnings markers: cross-validated against Finviz (no more false E BMO)
- Security: API token redaction in logs, HTTP logger suppression
- Performance: HMM cached in report, commentary fast-path, 24hr cache TTL
- Ops: storage cleanup job, crypto resilience (retry + health check + admin endpoints)
- Calendar: renamed to Market Calendar, added economic events (CPI, PPI, FOMC, etc.)
- Crypto: candlestick pattern detection (TA-Lib, 15m+ intervals)
- Chart: removed HMM bg rectangles, reduced YOLO opacity, legend at top-right

## Testing
Run tests with: `python -m pytest tests/ -v`
Current baseline: `603 passed` locally (up from 578, new cross-validation tests added).
All tests must pass locally before pushing. Test files live in `tests/`.
