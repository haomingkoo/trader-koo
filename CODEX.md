# CODEX.md — Current Tracker

## Resolved: React #306 on Chart & Crypto

### Actual Root Cause
React 19 minified error `#306` was **not** an "object as React child" error in this app. The real issue was an invalid element type caused by `lazy(() => import("react-plotly.js"))` resolving the wrong CJS/ESM export shape when Chart/Crypto/PaperTrade mounted Plotly.

### Fix Applied
1. `PlotlyWrapper` now unwraps the real Plotly component safely
2. Chart, Crypto, and Paper Trades render Plotly through that wrapper
3. Route-level lazy loading/code splitting is restored safely
4. SPA routing explicitly serves `index.html` for `/v2`, `/v2/`, and nested routes
5. VIX + sentiment gauge needle math corrected
6. Report page sentiment widget restored as clearly labeled **Market Sentiment**
7. Optional Alpha Vantage news sentiment source added as a separate external input with a clearly labeled blended view

### Verification
- `npm run build` succeeds
- `.venv/bin/pytest tests -q` → `550 passed`
- FastAPI `TestClient` verifies `/v2`, `/v2/`, `/v2/chart`, `/v2/crypto` all serve the React app

---

## Current Loose Ends

### Priority 1: HMM Stability
- HMM regime detection is implemented and now includes feature clipping plus a more stable covariance setup
- Current local pytest baseline no longer emits the earlier sklearn HMM warnings
- Remaining work is broader model experimentation, not basic numerical stabilization

### Priority 2: Market Sentiment Direction
- Current widget now supports an optional Alpha Vantage external news pulse in addition to the internal composite
- It still does **not** use Twitter or Reddit social scraping
- Any future social/news sources should remain clearly separated from the internal market-data score

### Priority 3: Performance + UX
- Plotly is still a heavy chunk; profile whether Lightweight Charts or a mixed charting stack would improve UX
- Crypto now has backend structure analysis plus auto-drawn support/resistance overlays on the v2 chart
- Add browser-level regression coverage for Chart, Crypto, and Earnings calendar toggle
- Consider widget linking and pipeline WebSocket after core stability work

### How to Debug
1. Run locally: `cd trader_koo/frontend-v2 && npm run dev`
2. Start backend: `.venv/bin/python -m uvicorn trader_koo.backend.main:app --reload --port 8000`
3. Open `http://localhost:3000/v2/crypto` in Chrome
4. The dev server shows UNMINIFIED errors with exact file:line
5. If Plotly regresses again, inspect the lazy import/export shape first

### Files to Check
- `trader_koo/frontend-v2/src/pages/ChartPage.tsx` (~1400 lines) — massive file, renders Plotly chart with overlays, commentary sidebar, pattern tables, YOLO audit
- `trader_koo/frontend-v2/src/pages/CryptoPage.tsx` (~665 lines) — Plotly chart, indicator cards, price tiles
- `trader_koo/frontend-v2/src/components/layout/Header.tsx` — WebSocket hooks for crypto/equity prices

### Dangerous Patterns to Search For
```
{someVariable}  // where someVariable could be an object
{data?.field}   // where field is typed as string but API returns object
{error}         // Error objects are not valid React children
{v as string}   // TypeScript cast erased at runtime, object passes through
```

### The API Responses to Verify
```bash
# Check what the crypto API actually returns
curl https://trader.kooexperience.com/api/crypto/summary | python3 -m json.tool
curl https://trader.kooexperience.com/api/crypto/indicators/BTC-USD | python3 -m json.tool
curl https://trader.kooexperience.com/api/crypto/history/BTC-USD?interval=1m&limit=60 | python3 -m json.tool

# Check dashboard API
curl "https://trader.kooexperience.com/api/dashboard/SPY?months=0" | python3 -m json.tool | head -50
```

Compare each field against the TypeScript interfaces in `src/api/types.ts`. If any field returns `{"key": "value"}` where the type says `string`, that's the bug.

---

## Other Tasks

### Earnings Grid
- EarningsPage already has the 5-day horizontal calendar grid
- Keep browser coverage on the Calendar/Table toggle because this is a UI-heavy path

---

## Architecture Quick Reference

```
Backend: trader_koo/backend/main.py (~550 lines)
├── 9 routers in trader_koo/backend/routers/
├── 6 services in trader_koo/backend/services/
├── Crypto WS: trader_koo/crypto/ (Binance, 5 pairs)
├── Equity WS: trader_koo/streaming/ (Finnhub, SPY/QQQ + on-demand)
└── DB: SQLite at /data/trader_koo.db

Frontend: trader_koo/frontend-v2/
├── React 19 + TypeScript strict + Vite 8 + Tailwind CSS v4
├── 9 pages in src/pages/
├── API hooks in src/api/hooks.ts
├── Types in src/api/types.ts (867 lines)
└── Build: npm run build → ../../dist-v2/

Deployment: Railway (asia-southeast1)
├── URL: trader.kooexperience.com
├── v1: / (vanilla JS)
├── v2: /v2/ (React)
└── Auto-deploy on push to main
```

## Environment Variables (Railway)
- `TRADER_KOO_API_KEY` — admin auth
- `FINNHUB_API_KEY` — real-time equity streaming
- `TRADER_KOO_LLM_PROVIDER` — azure_openai
- `TRADER_KOO_LLM_ENABLED` — 1
- `TRADER_KOO_ALLOWED_ORIGIN` — https://trader.kooexperience.com

## Running Locally
```bash
# Backend
cd /Users/koohaoming/dev/trader-koo
.venv/bin/python -m uvicorn trader_koo.backend.main:app --reload --port 8000

# Frontend (separate terminal)
cd trader_koo/frontend-v2
npm run dev
# Open http://localhost:3000/v2/
```
