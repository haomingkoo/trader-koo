# Claude Passdown — 2026-03-17

## Repo State

- Repo: `/Users/koohaoming/dev/trader-koo`
- Current committed `HEAD`: `7f66160` (`feat: add crypto structure overlays and stabilize hmm regime`)
- Worktree is intentionally dirty. Do not assume all current edits belong to a single task.

## What Changed In This Working Tree

There are several ongoing, uncommitted buckets of work in the tree:

- React/v2 stabilization and UX fixes
  - Chart page no longer uses the old 3-second equity poll loop.
  - Added shared websocket hook for live equity prices on chart pages.
  - Crypto chart got indicator toggles, better defaults, and longer history windows.
  - VIX gauge and pipeline/header status have in-progress fixes.
- Market sentiment work
  - Internal sentiment model still exists.
  - News/social sentiment work is present in `trader_koo/social_sentiment.py`, `trader_koo/structure/fear_greed.py`, `FearGreedGauge.tsx`, and related API types/hooks.
- Crypto market structure / BTC-vs-SPY work
  - Additional crypto market structure and correlation code exists in `trader_koo/crypto/market_insights.py`.

## This Pass: Crypto History + Timeframes + Chart Streaming

### 1. Equity chart page is now websocket-live

- New hook: `trader_koo/frontend-v2/src/hooks/useLiveEquityPrice.ts`
- `ChartPage.tsx` now uses `/ws/equities` via that hook instead of the old subscribe + poll loop.
- The chart header live badge and the main `Price` KPI card now both follow live ticks.
- Cleanup guard was added so the hook does not reconnect after intentional unmount.

Relevant files:

- `trader_koo/frontend-v2/src/hooks/useLiveEquityPrice.ts`
- `trader_koo/frontend-v2/src/pages/ChartPage.tsx`

### 2. Crypto history is materially deeper

- Binance REST history support now includes:
  - `1m`, `5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `12h`, `1d`, `1w`
- Longer intervals are now treated as native Binance history, stored in SQLite, and patched with recent `1m` bars where useful so the latest bucket is not stale.
- Startup warm-backfill now covers:
  - `1h`, `4h`, `12h`, `1d`, `1w`
- On-demand requests backfill and cache the selected interval if the local DB is shallow.

Relevant files:

- `trader_koo/crypto/binance_history.py`
- `trader_koo/crypto/service.py`
- `trader_koo/crypto/storage.py`
- `trader_koo/crypto/structure.py`
- `trader_koo/backend/routers/crypto.py`

### 3. Crypto page now exposes longer-view timeframes

- Current frontend interval set:
  - `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `12h`, `1D`, `1W`
- Default crypto view is now `1h` instead of `15m`.
- Current target windows:
  - `1m`: ~1d
  - `5m`: ~1w
  - `15m`: ~30d
  - `30m`: ~45d
  - `1h`: ~90d
  - `4h`: ~240d
  - `12h`: ~1.5y
  - `1D`: ~5y
  - `1W`: ~5y
- The explanatory copy now says data comes from native Binance history with live `1m` patching, not only from persisted `1m` aggregation.

Relevant file:

- `trader_koo/frontend-v2/src/pages/CryptoPage.tsx`

## Verification Already Run

- Frontend build:
  - `npm run build`
  - Passed
- Focused backend tests:
  - `./.venv/bin/pytest tests/test_crypto_service.py tests/test_crypto_binance_history.py tests/test_routers/test_crypto_router.py -q`
  - Passed: `9 passed`

## Local Crypto Cache Was Warmed Successfully

The local SQLite DB was warmed from Binance after the code changes.

- DB path:
  - `/Users/koohaoming/dev/trader-koo/trader_koo/data/trader_koo.db`
- Warmed symbols:
  - `BTC-USD`, `ETH-USD`, `SOL-USD`, `XRP-USD`, `DOGE-USD`
- Warmed intervals and counts per symbol:
  - `5m`: `2016`
  - `15m`: `2880`
  - `30m`: `2160`
  - `1h`: `2160`
  - `4h`: `1440`
  - `12h`: `1095`
  - `1d`: `1825`
  - `1w`: `260`

Notes:

- I intentionally did not run a giant multi-symbol `1m` backfill here.
- `1m` still backfills on demand and the UI only asks for a shorter visible window there.

## Important Current Reality

- The worktree contains more than this crypto/streaming pass.
- `git status --short` currently shows edits in:
  - `tests/test_crypto_service.py`
  - `tests/test_routers/test_crypto_router.py`
  - `tests/test_routers/test_report_router.py`
  - `trader_koo/backend/routers/crypto.py`
  - `trader_koo/crypto/service.py`
  - `trader_koo/crypto/storage.py`
  - `trader_koo/crypto/structure.py`
  - `trader_koo/frontend-v2/src/api/hooks.ts`
  - `trader_koo/frontend-v2/src/api/types.ts`
  - `trader_koo/frontend-v2/src/components/FearGreedGauge.tsx`
  - `trader_koo/frontend-v2/src/components/layout/Header.tsx`
  - `trader_koo/frontend-v2/src/pages/ChartPage.tsx`
  - `trader_koo/frontend-v2/src/pages/CryptoPage.tsx`
  - `trader_koo/frontend-v2/src/pages/VixPage.tsx`
  - `trader_koo/structure/fear_greed.py`
  - plus untracked files:
    - `tests/test_crypto_binance_history.py`
    - `tests/test_crypto_market_insights.py`
    - `tests/test_social_sentiment.py`
    - `trader_koo/crypto/binance_history.py`
    - `trader_koo/crypto/market_insights.py`
    - `trader_koo/frontend-v2/src/hooks/useLiveEquityPrice.ts`
    - `trader_koo/social_sentiment.py`

## Recommended Next Steps

1. Open `/v2/crypto` and visually verify the longer timeframe buttons and cached history feel correct.
2. Decide whether to commit the whole current worktree as one batch or split it into:
   - crypto/history + chart streaming
   - sentiment/news/social
   - VIX/header/UI stabilization
3. If further crypto depth is wanted:
   - add a manual admin/backfill trigger endpoint
   - optionally warm `30m` and `15m` at startup too
   - consider a targeted `1m` deep backfill job if intraday replay matters
4. If performance becomes the next focus:
   - `react-plotly` is still the heaviest chunk by far
   - crypto is a good candidate for a future Lightweight Charts migration

## Commands Worth Reusing

```bash
# Focused crypto verification
cd /Users/koohaoming/dev/trader-koo
./.venv/bin/pytest tests/test_crypto_service.py tests/test_crypto_binance_history.py tests/test_routers/test_crypto_router.py -q

# Frontend build
cd /Users/koohaoming/dev/trader-koo/trader_koo/frontend-v2
npm run build

# Example local warm-backfill pattern
cd /Users/koohaoming/dev/trader-koo
./.venv/bin/python - <<'PY'
from trader_koo.backend.services.database import DB_PATH
import trader_koo.crypto.service as svc
from trader_koo.crypto.binance_ws import SYMBOL_MAP

svc._db_path_str = str(DB_PATH)
intervals = ["5m", "15m", "30m", "1h", "4h", "12h", "1d", "1w"]
for symbol in SYMBOL_MAP.values():
    for interval in intervals:
        target = svc._backfill_target_limit(interval, 1)
        bars = svc._backfill_history(symbol, interval, target)
        print(symbol, interval, len(bars), target)
PY
```
