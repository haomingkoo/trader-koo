# trader-koo v2 Development Progress

## Branch: `feat/v2-react-frontend`

## Completed

### Sprint 1: Backend Decomposition
- [x] Extract 6 service modules from `main.py` monolith
  - `services/database.py` — DB connection, SQL query helpers
  - `services/market_data.py` — data source status, YOLO status
  - `services/chart_builder.py` — dashboard payload construction
  - `services/report_loader.py` — report file loading/parsing
  - `services/scheduler.py` — APScheduler factory, job runners
  - `services/pipeline.py` — pipeline status tracking, cache
- [x] Create 8 FastAPI routers (50 endpoints preserved)
  - `routers/system.py` — health, config, status, vix-glossary
  - `routers/dashboard.py` — dashboard/{ticker}, yolo/{ticker}, tickers
  - `routers/report.py` — daily-report, earnings-calendar, market-summary
  - `routers/opportunities.py` — opportunities (PEG screening)
  - `routers/paper_trades.py` — paper-trades, summary, detail
  - `routers/email.py` — subscribe, confirm, unsubscribe, chart-preview
  - `routers/usage.py` — session tracking, feedback
  - `routers/admin.py` — all 27 admin endpoints
- [x] Slim `main.py`: 6140 → 518 lines (app factory + middleware only)

### Sprint 2: React Frontend Scaffold
- [x] Vite 8 + React 19 + TypeScript strict + Tailwind CSS v4
- [x] Full API client layer (`api/client.ts`, `api/hooks.ts`, `api/types.ts`)
- [x] TanStack Query hooks for all 8 endpoints
- [x] Zustand stores: `configStore`, `chartStore`
- [x] Shared UI components: Card, Badge, Table, Spinner, ErrorBoundary
- [x] Layout: Sidebar (collapsible), Header, ClockStrip (local + NY time)
- [x] React Router v7 with `/v2` basename
- [x] Build verified — `npm run build` passes

### Infrastructure
- [x] v2 mount point in main.py (`/v2` → `StaticFiles(dist-v2/)`)
- [x] `railway.toml` updated with Node.js build step
- [x] `.gitignore` updated for `dist-v2/` and `node_modules/`

### Bug Fixes (merged to main)
- [x] Operator precedence in debate_engine.py
- [x] Risk role never-bullish fix
- [x] Paper trades ORDER BY for equity/drawdown
- [x] XSS escaping in frontend innerHTML
- [x] CORS default changed from `*` to production origin
- [x] DB path removed from API responses
- [x] CSP meta tag added
- [x] outline:none → :focus-visible for accessibility
- [x] Deprecated utcnow() replaced
- [x] Timezone clock added to v1 frontend

## In Progress (agents running)

### Sprint 3: Page Implementation
- [ ] ReportPage — Bloomberg-style KPI cards, setup table, debate viz, risk filters
- [ ] ChartPage — Plotly candlestick with overlays, commentary sidebar
- [ ] VixPage — regime context, MA matrix, health panel
- [ ] EarningsPage — calendar/table views, summary cards
- [ ] OpportunitiesPage — PEG table with view presets
- [ ] PaperTradePage — equity curve, trade log, filters (partially done)
- [ ] GuidePage — feature cards, disclaimer

### Quality Assurance
- [ ] Backend audit — import correctness, no hidden fallbacks, endpoint parity
- [ ] Backend test suite — pytest for all routers/services
- [ ] Frontend audit — type accuracy, no fallbacks, build verification

### Research
- [ ] Free market data API comparison (equity + crypto)
- [ ] Cost analysis for higher refresh rates
- [ ] BTC/ETH integration feasibility

## Planned

### Sprint 4: Enhanced Features
- [ ] HMM regime overlay on candlestick chart (colored background shading)
- [ ] Regime probability sub-pane below chart
- [ ] Glassmorphism polish on all KPI cards
- [ ] Widget linking (change ticker in chart → report updates)
- [ ] Keyboard shortcuts for power users
- [ ] Calibration accuracy chart for setup scoring

### Sprint 5: Data Pipeline Expansion
- [ ] Higher-frequency data ingestion (15-min or 5-min candles)
- [ ] Crypto integration (BTC, ETH via Binance/CoinGecko)
- [ ] WebSocket for real-time pipeline status
- [ ] Separate crypto ingest pipeline (24/7 vs market hours)

### Sprint 6: Polish + Cutover
- [ ] Mobile responsiveness audit
- [ ] Performance profiling (lazy loading, code splitting)
- [ ] Dual-serve testing (`/` = v1, `/v2` = React)
- [ ] Final cutover: `/` = React, `/v1` = old HTML

## Architecture

```
trader-koo/
├── trader_koo/
│   ├── backend/
│   │   ├── main.py              # 518 lines — app factory
│   │   ├── routers/             # 8 router modules
│   │   └── services/            # 6 service modules
│   ├── frontend/                # v1 vanilla JS (preserved)
│   │   └── index.html           # 9490 lines
│   ├── frontend-v2/             # v2 React app
│   │   ├── src/
│   │   │   ├── api/             # client, hooks, types
│   │   │   ├── components/      # ui/ + layout/
│   │   │   ├── pages/           # 8 page components
│   │   │   ├── routes/          # DashboardLayout
│   │   │   ├── stores/          # Zustand stores
│   │   │   └── styles/          # globals.css
│   │   └── package.json
│   ├── auth/                    # RBAC, user management
│   ├── audit/                   # logging, export
│   ├── cv/                      # YOLO pattern detection
│   ├── db/                      # schema, sources
│   ├── features/                # technical indicators
│   ├── llm/                     # LLM schemas, validation
│   ├── middleware/               # CORS, auth
│   ├── ratelimit/               # rate limiting
│   ├── security/                # redaction, error sanitization
│   └── structure/               # gaps, levels, patterns
├── dist-v2/                     # Vite build output (gitignored)
├── railway.toml                 # Build: Python + Node.js
└── tests/                       # pytest suite
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

Glassmorphism: `backdrop-blur-sm bg-[var(--panel)]/80 border border-[var(--line)]`
