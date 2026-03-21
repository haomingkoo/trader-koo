# Codex Review Brief — 2026-03-20/21 Session

> 112 commits pushed to main in one session. This document briefs a second reviewer on what changed and what to verify.

## What Was Built

### Telegram Integration
- **Files**: `trader_koo/notifications/` (telegram.py, alert_engine.py, bot_commands.py, morning_summary.py, market_monitor.py, macro_monitor.py)
- **What it does**: Price alerts (REST polling Finnhub every 2min), bot commands (/status /top /price /vix /alerts /help), morning briefing (8am SGT daily), Polymarket spike detection (every 15min), crypto spike detection, macro risk-off alerts (yields, oil, gold, VIX)
- **CHECK**: Spike alerts now compile into ONE message (not individual spam). Verify `send_spike_alerts()` in market_monitor.py sends one message, not N messages.
- **CHECK**: Cooldown logic in `spike_alert_cooldown` table — only re-alerts if direction reverses or prob moves 5+ pts.
- **CHECK**: HTML parse_mode is used for clickable links in Telegram.

### Progressive Chart Loading
- **Files**: `trader_koo/backend/routers/dashboard.py` (new /quick and /commentary endpoints), `trader_koo/backend/services/chart_builder.py` (split into build_dashboard_quick_payload + build_commentary_payload)
- **Frontend**: ChartPage.tsx now uses useChartQuick() + useChartCommentary() in parallel
- **CHECK**: The original `/api/dashboard/{ticker}` endpoint still works (backward compatible)
- **CHECK**: ChartCommentarySidebar shows "Loading analysis..." skeleton while commentary loads

### Chart Single Source of Truth Fix
- **File**: `trader_koo/backend/services/chart_builder.py` ~line 594
- **What**: When a report snapshot override exists, skip `_report_score_setup_from_confluence()` and `_report_describe_setup()` entirely. Don't recompute tier from live data — use the report's tier.
- **CHECK**: If `setup_override` is present, the narrative text should match the report's tier. Previously NDAQ showed Tier A in report but "C setup" in chart narrative.

### Earnings Calendar
- **File**: `trader_koo/catalyst_data.py` ~line 939 — ETF/index skip list added
- **File**: `trader_koo/scripts/cache_logos.py` — new logo precaching script
- **File**: `trader_koo/frontend-v2/src/components/earnings/TickerLogo.tsx` — logo component
- **CHECK**: SPY, QQQ, and other ETFs should NOT show "E BMO" markers on charts
- **CHECK**: S&P 500 filter on earnings page — verify it actually filters (uses price_daily tickers)
- **CHECK**: Finnhub earnings data quality — are dates accurate? AAPL was showing false BMO markers.

### ML Pipeline (DISABLED)
- **File**: `trader_koo/paper_trade/config.py` — `ml_enabled: bool = False`
- ML scoring was tested rigorously: AUC 0.5051 (random). Disabled in paper trades.
- **CHECK**: Paper trade pipeline works WITHOUT ML scoring. The `if config.ml_enabled:` guard in trading.py should skip all ML imports.

### Polymarket Smart Cards
- **File**: `trader_koo/frontend-v2/src/pages/PolymarketPage.tsx`
- **File**: `trader_koo/ml/external_data.py` — `_classify_event_type()`, `_parse_event_markets()`
- **CHECK**: Events with 1 active + N resolved markets should show as "simple" YES/NO (not empty)
- **CHECK**: Grid uses `items-start` (cards don't stretch to match tallest neighbor)
- **CHECK**: SubMarketRow shows full question text, not truncated

### Frontend UX
- **File**: Sidebar.tsx — reordered, mobile labels fix, sidebar auto-collapse on landscape
- **File**: ChartPlotPanel.tsx — modebar hidden on mobile
- **File**: buildEquityChartData.ts — volume profile opacity 0.3→0.08, hidden from legend
- **File**: NotificationBell.tsx — fixed positioning on mobile, pulsing badge
- **File**: MethodologyPage.tsx — Lucide icons (not emojis), pipeline animation state machine fix
- **CHECK**: Sidebar shows labels when opened on portrait mobile (not icon-only)
- **CHECK**: Chart at 3M zoom has left/right padding (candles not clipped)
- **CHECK**: Level labels shortened on mobile (PRIMARY SUP → P SUP)

### Infrastructure
- **File**: `build.sh` — replaces railway.toml mega-command
- **File**: `trader_koo/backend/services/scheduler.py` — 7 scheduled jobs, all unique IDs
- **File**: `trader_koo/scripts/backup_db.py` — weekly SQLite backup
- **CHECK**: `build.sh` runs all steps with proper error handling
- **CHECK**: No duplicate job IDs in scheduler
- **CHECK**: Macro alerts run 24/7 (IntervalTrigger, not CronTrigger with hour restriction)

### Report Generation Split
- **Files**: `trader_koo/report/` package (7 modules split from 6,283-line monolith)
- **CHECK**: `scripts/generate_daily_report.py` still works as CLI entry point (thin wrapper)
- **CHECK**: All imports from report submodules resolve correctly

## Known Issues (do NOT try to fix, just verify)
1. Feature extraction is slow (50 min for 156 dates × 500 tickers) — needs pickle caching
2. Finnhub earnings data may return incorrect dates — cross-validate needed
3. 112 commits to main caused Railway to rebuild constantly — use staging branch next time

## How to Verify
```bash
# Run tests
python -m pytest tests/ -v

# Check TypeScript compiles
cd trader_koo/frontend-v2 && npx tsc --noEmit

# Check key imports
python -c "from trader_koo.notifications.market_monitor import send_spike_alerts; print('OK')"
python -c "from trader_koo.notifications.macro_monitor import send_macro_alert; print('OK')"
python -c "from trader_koo.report.generator import fetch_report_payload; print('OK')"
```
