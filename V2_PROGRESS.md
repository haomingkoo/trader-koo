# trader-koo v2 Development Progress

## Branch: merged to `main`

---

## Completed

### Sprint 1: Backend Decomposition
- [x] Extract 6 service modules from `main.py` monolith (database, market_data, chart_builder, report_loader, scheduler, pipeline)
- [x] Create 9 FastAPI routers (51 endpoints — 8 original + crypto router)
- [x] Slim `main.py`: 6140 → 518 lines (app factory + middleware only)
- [x] All 16 modules import cleanly verified

### Sprint 2: React Frontend Scaffold
- [x] Vite 8 + React 19 + TypeScript strict + Tailwind CSS v4
- [x] Full API client layer (TanStack Query hooks, TypeScript interfaces)
- [x] Zustand stores: configStore, chartStore
- [x] Shared UI components: Card (with glass prop), Badge, Table, Spinner, ErrorBoundary
- [x] Layout: collapsible Sidebar (mobile drawer), Header, ClockStrip
- [x] React Router v7 with `/v2` basename

### Sprint 3: All 8 Pages Fully Built
- [x] **ReportPage** — Bloomberg-style KPI cards, setup table with clickable ticker links, debate visualization with confidence bars, risk filters, key changes, VIX regime widget, setup evaluation
- [x] **ChartPage** — Plotly candlestick with S/R levels, gap zones, trendlines, YOLO bounding boxes, MA overlays (20/50/100/200), earnings markers, collapsible commentary sidebar, pattern tabs (rule/hybrid/candlestick), YOLO audit table, weekly resampling
- [x] **VixPage** — regime context, MA matrix, health panel with drivers/warnings, LLM commentary
- [x] **EarningsPage** — calendar/table toggle, session lanes, expandable cards, sortable table
- [x] **OpportunitiesPage** — view presets (All/Undervalued/Deep Value/Overvalued), funnel cards, full PEG table
- [x] **PaperTradePage** — 7 KPI cards, equity curve (Plotly), direction/exit breakdowns, status/direction filters, full trade log
- [x] **CryptoPage** — BTC/ETH candlestick chart, price cards, symbol/interval selectors
- [x] **GuidePage** — feature cards linking to each section, disclaimer, NFA banner

### Sprint 3.5: Crypto Integration (Binance WebSocket)
- [x] `trader_koo/crypto/` package: binance_ws.py, models.py, service.py
- [x] Real-time BTC/ETH streaming via `wss://stream.binance.com` (no API key needed)
- [x] FastAPI WebSocket at `/ws/crypto` pushes ticks to browser (sub-second latency)
- [x] Browser WebSocket in Header (zero polling, auto-reconnect with backoff)
- [x] CryptoPage with Plotly candlestick chart
- [x] Sidebar nav entry for Crypto

### Quality: Bug Fixes & Security (merged to main)
- [x] 6 critical bugs fixed (operator precedence, risk role, paper trades ORDER BY, etc.)
- [x] 6 security fixes (CORS, DB path leak, CSP, XSS escaping, accessibility)
- [x] Timezone clock with NY time + market status badge (v1)
- [x] 30 silent `except Exception: pass` → proper logging + `generation_warnings` field
- [x] Narrowed exception types where possible (ValueError, TypeError)

### Quality: Frontend Polish
- [x] Glassmorphism consistency (Card `glass` prop across all pages)
- [x] Accessibility: focus-visible outlines, aria-labels, text alongside colors
- [x] Responsive: mobile sidebar drawer, overflow-x-auto tables, breakpoints
- [x] All ticker references are clickable `<Link>` to `/chart?t={ticker}`
- [x] Number formatting: +/- signs on P&L, consistent $XX.XX / XX.XX%

### Quality: Testing
- [x] 87 new tests (500 total passing)
- [x] Service unit tests: database, market_data, pipeline, scheduler
- [x] Router integration tests: system, dashboard, report, paper_trades
- [ ] 25 router test failures need fixture updates for refactored code

### Infrastructure
- [x] v2 mount point: `/v2` → `StaticFiles(dist-v2/)`
- [x] `railway.toml` updated: Node.js 22 installed via nodesource in build
- [x] Build verified on Railway (npm ci + npm run build succeeds)
- [x] `.gitignore` for dist-v2/ and node_modules/

---

## TODO — Next Up

### Priority 1: HMM Regime Detection
- [ ] Install `hmmlearn` library
- [ ] Build `trader_koo/structure/hmm_regime.py` — train 3-state Gaussian HMM on VIX features
- [ ] Features: VIX returns, percentile, BB width, MA state, compression, participation bias
- [ ] States: Low Vol/Risk-On, Moderate/Neutral, High Vol/Risk-Off
- [ ] Add regime state + probabilities to daily report payload
- [ ] Chart overlay: colored background shading (green=bull, red=bear, gray=sideways)
- [ ] Regime probability sub-pane below candlestick chart
- [ ] Viterbi sequence visualization (last 20 days state path)

### Priority 2: More Crypto Features
- [ ] Add more pairs: SOL, XRP, DOGE (just add to KLINE_STREAMS in binance_ws.py)
- [ ] Technical indicators on crypto charts (MAs already exist, add RSI, MACD)
- [ ] Crypto correlation panel (BTC vs SPY)
- [ ] 5-min and 1-hour bar aggregation from 1-min stream
- [ ] Persist crypto bars to SQLite (currently in-memory only, lost on restart)

### Priority 3: IBKR Integration (waiting for Lite → Pro upgrade)
- [ ] Install `ibind` library (OAuth headless, no gateway needed)
- [ ] Replace yfinance with IBKR historical data
- [ ] Add real-time streaming for top 50 tickers (WebSocket)
- [ ] Replace custom paper trading with IBKR paper account fills
- [ ] Add live trading UI in v2 (limit/stop/bracket orders)
- [ ] Portfolio positions + P&L from IBKR account

### Priority 4: Enhanced Features
- [ ] Calibration accuracy chart (setup tier predicted vs actual outcomes)
- [ ] Keyboard shortcuts (1-7 page nav, `/` ticker search, `R` refresh)
- [ ] Widget linking (change ticker in chart → report highlights row)
- [ ] News/sentiment feed with scrolling cards
- [ ] Pipeline WebSocket (push status changes instead of polling)

### Priority 5: Polish + Cutover
- [ ] Fix remaining 25 test failures (router test fixtures)
- [ ] Performance profiling (Plotly 4.6MB bundle — consider Lightweight Charts)
- [ ] Code splitting for route-level lazy loading
- [ ] Full mobile responsiveness audit (375px/768px/1024px)
- [ ] Final cutover: `/` = React, `/v1` = old HTML

---

## Architecture

```
trader-koo/
├── trader_koo/
│   ├── backend/
│   │   ├── main.py              # ~520 lines — app factory + middleware
│   │   ├── routers/             # 9 routers (system, dashboard, report, opportunities,
│   │   │                        #   paper_trades, email, usage, admin, crypto)
│   │   └── services/            # 6 services (database, market_data, chart_builder,
│   │                            #   report_loader, scheduler, pipeline)
│   ├── crypto/                  # Binance WebSocket integration
│   │   ├── binance_ws.py        # WS client with auto-reconnect
│   │   ├── models.py            # CryptoTick, CryptoBar dataclasses
│   │   └── service.py           # start/stop feed, subscriber broadcast
│   ├── frontend/                # v1 vanilla JS (preserved at /)
│   ├── frontend-v2/             # v2 React app (served at /v2)
│   │   └── src/
│   │       ├── api/             # client, hooks, types
│   │       ├── components/      # ui/ (Card, Badge, Table, Spinner, ErrorBoundary)
│   │       │                    # layout/ (Sidebar, Header, ClockStrip)
│   │       ├── pages/           # 8 pages
│   │       ├── routes/          # DashboardLayout
│   │       ├── stores/          # Zustand (config, chart)
│   │       └── styles/          # globals.css with design tokens
│   ├── structure/               # gaps, levels, patterns, VIX analysis
│   ├── auth/, audit/, cv/, db/, features/, llm/, middleware/, ratelimit/, security/
│   └── scripts/                 # daily_update.sh, generate_daily_report.py, run_yolo_patterns.py
├── tests/                       # pytest suite (500 passing)
├── dist-v2/                     # Vite build output (gitignored)
├── railway.toml                 # Python + Node.js build
├── CODEX.md                     # 12 delegation tasks
└── V2_PROGRESS.md               # This file
```

## Design System

```css
--bg: #0b0f16        /* Deep dark background */
--panel: #121927     /* Card/panel background */
--muted: #8ea0bd     /* Secondary text */
--text: #e9eef8      /* Primary text */
--line: #25334f      /* Borders */
--green: #38d39f     /* Profit / bullish */
--red: #ff6b6b       /* Loss / bearish */
--amber: #f8c24e     /* Warnings / tier B */
--blue: #6aa9ff      /* Neutral / trending */
--accent: #4a9eff    /* Focus / interactive */
```

Glassmorphism: `<Card glass>` → `backdrop-blur-sm bg-[var(--panel)]/80 border border-[var(--line)]`
