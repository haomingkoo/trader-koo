# Claude ↔ Codex Shared Communication File

This file is used by both Claude Code and Codex to communicate about ongoing work,
avoid conflicts, and hand off tasks. Both tools should read this before starting work
and update it after completing tasks.

Last updated: 2026-03-18 by Claude (pipeline ops panel + guide reorg)

---

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
