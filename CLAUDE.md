# trader-koo — Claude Code Instructions

## Project Overview
Full-stack S&P 500 stock market analysis dashboard deployed on Railway.
Personal tool — not for public trading signals (NFA).
Live at: trader.kooexperience.com

## Stack
- **Backend**: FastAPI + APScheduler, SQLite at `/data/trader_koo.db` (Railway persistent volume)
- **Frontend**: Single `trader_koo/frontend/index.html` — vanilla JS + Plotly.js, no build step
- **AI**: YOLOv8 (`foduucom/stockmarket-pattern-detection-yolov8`) batch-run nightly
- **Data**: yfinance (primary) + Finviz (fundamentals) + optional Alpha Vantage
- **Deploy**: Railway asia-southeast1, nixpacks build, persistent `/data` volume

## Key Files
| File | Purpose |
|------|---------|
| `trader_koo/backend/main.py` | All FastAPI endpoints + APScheduler (~6000 lines) |
| `trader_koo/frontend/index.html` | Entire frontend (~1644 lines) |
| `trader_koo/scripts/daily_update.sh` | Nightly orchestrator: ingest → YOLO → report |
| `trader_koo/scripts/run_yolo_patterns.py` | YOLOv8 batch detection |
| `trader_koo/scripts/update_market_db.py` | yfinance + Finviz ingestion |
| `trader_koo/scripts/generate_daily_report.py` | Daily report generation |
| `trader_koo/data/schema.py` | SQLite schema helpers |

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

## Testing
Run tests with: `python -m pytest tests/ -v`
All tests must pass locally before pushing. Test files live in `tests/`.
