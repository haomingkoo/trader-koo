# Claude ↔ Codex Shared Communication File

This file is used by both Claude Code and Codex to communicate about ongoing work,
avoid conflicts, and hand off tasks. Both tools should read this before starting work
and update it after completing tasks.

Last updated: 2026-03-17 by Claude Code

---

## Current State (commit 67c8e44)

### What's deployed on Railway
- v1 dashboard at `/` — working
- v2 React dashboard at `/v2/` — mostly working
- Binance crypto WebSocket — 5 pairs (BTC, ETH, SOL, XRP, DOGE)
- Finnhub equity streaming — SPY/QQQ always-on + on-demand
- HMM regime detection — integrated into chart builder
- VIX metrics — gauge, position sizing, vol premium
- Crypto with deep history — 1m through 1W intervals from Binance REST

### What was just done (this session)
1. **Pipeline badge fix** — `derivePipelineStates` restructured with early returns. Idle = gray, active = amber, done = green, failed only when both log + DB confirm.
2. **Equity live candle** — New `streaming/live_candle.py` aggregates Finnhub ticks into forming 1-min candles. Dashboard endpoint includes `live_candle` field.
3. **Crypto forming candle** — `get_forming_candle(symbol, interval)` in crypto/service.py aggregates 1m bars from interval boundary. History endpoint includes `forming_candle` for 5m+.

### What still needs doing (in priority order)

#### P0: Binance Scaling Architecture
- Keep one base 1m stream per symbol (current behavior)
- Aggregate higher intervals (5m, 15m, 1h, etc.) server-side from 1m data
- Fan out from our backend WebSocket, not directly from Binance per client/timeframe
- Currently each client connects to `/ws/crypto` and gets raw 1m ticks
- Need: backend aggregation service that maintains candles for each interval

#### P1: Chart/Crypto Frontend Polish
- Equity chart: use the new `live_candle` field from dashboard API to show forming bar
- Crypto chart: use the new `forming_candle` field to show incomplete bar with different styling
- VIX gauge needle direction may still be wrong (needs visual check)

#### P2: Remaining UI Work
- Unusual Whales 5-day earnings grid (code exists, needs visual verification)
- Mobile responsiveness audit
- Performance: Plotly still ~4.6MB, consider Lightweight Charts

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
