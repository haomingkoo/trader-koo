# CODEX.md — Help Wanted / Delegation Tasks

Tasks that could benefit from focused, parallel work sessions.
Ordered by priority. Each task is self-contained with clear inputs/outputs.

---

## High Priority

### 1. Fix silent `except Exception: pass` in generate_daily_report.py
**File**: `trader_koo/scripts/generate_daily_report.py` (6248 lines)
**Problem**: 13+ `except Exception: pass` blocks silently discard errors in core report sections (YOLO deltas, volatility, breadth, candlestick patterns, fundamentals, sector heatmap). If any fail, the report ships with missing data and no alert.
**Goal**: Add `LOG.warning(...)` to every silent except block. For critical sections (fundamentals, YOLO patterns), raise the error instead of swallowing it. Add a `warnings` list to the report payload that tracks which sections failed.
**Constraint**: Do NOT change the report output schema — only add logging and the warnings field.

### 2. Remove dead code in LLM modules
**Files**: `trader_koo/llm/fallback.py`, `trader_koo/llm/validator.py`
**Problem**: `fallback.py` has two functions (`generate_template_narrative`, `generate_rule_based_pattern_explanation`) that are never imported by any production code. `validator.py` has two functions (`generate_fallback_pattern_explanation`, `generate_fallback_regime_analysis`) only used in tests, never in production.
**Goal**: Either wire them up properly or remove them. If removing, update any test files that reference them.

### 3. Backend router import verification
**Files**: All files in `trader_koo/backend/routers/` and `trader_koo/backend/services/`
**Problem**: The routers were extracted from a 6140-line monolith by an AI agent. Some imports may reference functions that don't exist in the extracted services, or import from the wrong module.
**Goal**: Run `python -c "from trader_koo.backend.routers import admin, dashboard, email, opportunities, paper_trades, report, system, usage"` and fix every ImportError. Then run `python -m pytest tests/ -v` to verify.
**Constraint**: Do not change endpoint behavior — only fix imports.

### 4. Frontend type accuracy audit
**File**: `trader_koo/frontend-v2/src/api/types.ts`
**Problem**: TypeScript interfaces were inferred from the v1 frontend code. Some fields may be typed as `Record<string, unknown>` when they could be more specific. Some optional fields may be missing.
**Goal**: Compare each interface against the actual backend router response shapes. Make types exactly match reality. Run `npm run build` to verify.

---

## Medium Priority

### 5. HMM regime overlay on chart
**File**: `trader_koo/frontend-v2/src/pages/ChartPage.tsx`
**Problem**: The chart shows price data but doesn't visualize market regime (bull/bear/sideways).
**Goal**: Add colored background shading on the candlestick chart based on VIX regime data. Green = bullish, Red = bearish, Gray = sideways. Data source: `regime_context` from the daily report API.
**Depends on**: VIX regime data being available in the dashboard payload (may need a backend change to include it).

### 6. Calibration accuracy chart
**File**: New component in `trader_koo/frontend-v2/src/pages/ReportPage.tsx`
**Problem**: Setup scoring (tier A/B/C) has no calibration visualization — users can't see if Tier A setups actually outperform.
**Goal**: Add a chart showing predicted probability (setup score) vs actual outcome (paper trade P&L). Inspired by the Polymarket bot's calibration chart.
**Depends on**: Enough paper trade history to compute actual outcomes per tier.

### 7. Crypto data integration (BTC/ETH)
**Files**: New service + router
**Problem**: Dashboard only supports S&P 500 equities. User wants BTC/ETH.
**Goal**: Add a crypto data pipeline using Binance API (free, no key needed for public data). Create `services/crypto_data.py` with functions to fetch OHLCV for BTC/ETH. Add to the dashboard endpoint with a `CRYPTO:BTC` ticker prefix.
**Constraint**: Crypto trades 24/7 — no market hours concept. Schema needs to handle this.

### 8. WebSocket for real-time pipeline status
**File**: New router in `trader_koo/backend/routers/`
**Problem**: Pipeline status (Ingest → YOLO → Report) currently polls every 120 seconds.
**Goal**: Add a WebSocket endpoint at `/ws/pipeline` that pushes status changes to connected React clients. Frontend `Header.tsx` subscribes on mount.
**Constraint**: Must gracefully handle connection drops and reconnect.

---

## Low Priority

### 9. Keyboard shortcuts
**File**: `trader_koo/frontend-v2/src/` (new hook)
**Problem**: Bloomberg-terminal users expect keyboard navigation.
**Goal**: Add keyboard shortcuts: `1-7` for page navigation, `/` for ticker search, `Esc` to close modals, `R` to refresh data. Create a `useKeyboardShortcuts` hook.

### 10. Widget linking (OpenBB pattern)
**Problem**: Changing ticker in the chart doesn't update the report page's highlighted row.
**Goal**: When user navigates from report setup table to chart, the chart loads that ticker. When user changes ticker in chart, navigating back to report highlights that ticker's row.
**Implementation**: Use Zustand `chartStore.ticker` as the linking mechanism.

### 11. Mobile responsiveness audit
**Files**: All page components in `trader_koo/frontend-v2/src/pages/`
**Problem**: Pages are designed for desktop. Sidebar collapses but tables may overflow on mobile.
**Goal**: Test every page at 375px, 768px, 1024px widths. Fix overflow issues, stack grids vertically, ensure touch targets are 44px minimum.

### 12. Performance profiling
**Files**: All pages, especially ChartPage (Plotly is heavy)
**Problem**: Plotly.js bundle is 4.6MB gzipped to 1.4MB. First load may be slow.
**Goal**: Profile with Lighthouse. Implement code splitting for Plotly (already lazy-loaded). Consider using Lightweight Charts for the main candlestick and Plotly only for analytics charts.

---

## Notes

- All work should be on branch `feat/v2-react-frontend`
- Follow conventions in `.claude/rules/` (Python: snake_case, type hints; Frontend: functional components, strict TS)
- NO hidden fallbacks — fail explicitly
- NO arbitrary data limits — show all data or paginate with user control
- Conventional commits: `feat:`, `fix:`, `refactor:`, etc.
