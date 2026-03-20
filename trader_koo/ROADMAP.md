# Trader Koo — Roadmap

> Last updated: 2026-03-20

---

## Phase 1: Data Foundation

- [x] SQLite DB on Railway persistent volume (`/data/trader_koo.db`, 23 tables)
- [x] yfinance daily OHLCV for S&P 500 (510+ tickers)
- [x] Macro indices (VIX, TNX, GSPC, DJI, SPY, QQQ, SVIX, etc.)
- [x] Finviz fundamentals (PE, PEG, EPS, target price, discount)
- [x] Finnhub WebSocket real-time equity streaming (SPY/QQQ always-on + on-demand)
- [x] Binance WebSocket for crypto (BTC, ETH, SOL, XRP, DOGE) with 1m kline aggregation
- [x] FRED macro data (Fed funds rate, yield curve, unemployment, CPI)
- [x] Polymarket API (finance-filtered prediction market probabilities)
- [x] Alpha Vantage news sentiment (optional)

## Phase 2: Pattern Detection (5 layers)

- [x] Rule-based geometric patterns (flags, wedges)
- [x] Candlestick patterns (hammer, morning star, etc.)
- [x] Hybrid scoring (50% rule + 20% candle + 15% volume + 15% breakout)
- [x] CV proxy (independent geometry scorer)
- [x] YOLOv8 AI detection (pre-computed, dual timeframe: daily 180d + weekly 730d)

## Phase 3: ML Pipeline

- [x] LightGBM walk-forward trainer with purged validation
- [x] 51 features (momentum, vol, volume, trend, VIX regime, cross-sectional ranks, macro, commodities, sector rotation, news sentiment, earnings proximity)
- [x] Triple-barrier labeling (2.0x ATR profit/stop, 10-day time barrier)
- [x] Meta-labeling for false positive filtering
- [x] SHAP analysis and drift detection
- [x] Walk-forward backtest vs SPY with slippage modeling
- [x] HMM regime detection (3-state Gaussian)
- [x] Early stopping with purged validation set (50 rounds, lr=0.01, leaves=15, depth=3)
- [x] 3 target modes: return_sign, barrier, cross-sectional rank
- [x] Cross-sectional rank normalization for all per-ticker features
- [x] Volume-confirmed momentum + ATR expansion + gap features
- [x] High/low barrier touches (not just close)

## Phase 4: Frontend (React 19 + Vite + Tailwind)

- [x] 10 pages: Guide, Report, VIX, Earnings, Chart, Opportunities, Paper Trades, Crypto, Polymarket, 404
- [x] Live streaming tickers in header (SPY/QQQ/VIX)
- [x] Plotly interactive charts with overlays (MA, Bollinger, ATR, levels, YOLO boxes)
- [x] Dark/light mode toggle
- [x] Zustand state management
- [x] Lazy-loaded routes with Suspense
- [x] Fear/Greed composite gauge

## Phase 5: Paper Trading & Decision Engine

- [x] Automated trade lifecycle (open → closed/target_hit/stopped_out/expired)
- [x] ATR-based position sizing, mark-to-market, equity curve
- [x] Decision tracking (analyst → debate → risk → portfolio stages)
- [x] Debate engine (5 role analysts + deterministic arbiter)
- [x] $1M simulated portfolio tracking

## Phase 6: Daily Pipeline & Ops

- [x] APScheduler: Mon–Fri 22:00 UTC ingest → YOLO → report → email
- [x] Saturday 00:30 UTC YOLO full seed
- [x] Reconciliation for stale/crashed runs
- [x] 23 admin endpoints (API key auth)
- [x] Pipeline trigger/status/logs
- [x] ML training/backtest/SHAP/drift endpoints
- [x] Data export (CSV/SQLite)
- [x] LLM health monitoring
- [x] Audit logging
- [x] Email subscription system (Resend/SMTP)
- [x] 578 tests passing

---

## Phase 7: Near-Term (Next Up)

- [ ] FRED bulk fetch (1 call per series instead of per-date) — speeds up ML training
- [ ] Polymarket timeline sub-markets (individual date milestones, not just YES/NO)
- [ ] News sentiment DB caching for ML training
- [ ] Retrain model with FRED features
- [ ] Earnings calendar redesign (Unusual Whales-style 5-day grid with Premarket/TBD/Afterhours)
- [ ] IBKR API integration (account upgraded to Pro, API tested with `ib_async`, not yet coded)
- [ ] Local retraining with rank features + early stopping — target AUC 0.53+
- [ ] Optuna hyperparameter optimization within walk-forward
- [ ] Increase max_positions from 5 to 20-30 for small-edge exploitation

## Phase 8: Future

- [ ] IBKR paper trading bridge (live paper orders via IB Gateway)
- [ ] LGBMRanker for cross-sectional stock ranking
- [ ] Feature ablation study (top 15 vs all 51)
- [ ] Frontend + ML test coverage expansion
- [ ] Congressional trading disclosures (Unusual Whales inspiration)
- [ ] Market map (bubble chart visualization)
- [ ] Sector flow analysis
- [ ] Newsletter-quality narrative generation
