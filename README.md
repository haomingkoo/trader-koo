# Trader Koo

Full-stack S&P 500 market analysis platform with ML-powered swing trade forecasting.

**Live**: [trader.kooexperience.com](https://trader.kooexperience.com)

> For informational and educational purposes only. Nothing here constitutes investment advice.
> Past performance is not indicative of future results.

---

## Overview

| Layer | Stack |
|-------|-------|
| **Backend** | FastAPI + SQLite (Railway persistent volume) |
| **Frontend** | React 19 + TypeScript + Vite + Tailwind v4 |
| **ML** | LightGBM + walk-forward validation + SHAP |
| **Data** | yfinance + Finnhub + StockTwits + RSS + FRED + Polymarket |
| **Pattern Detection** | YOLOv8 (chart pattern recognition) |
| **Regime** | HMM (3-state), VIX analysis, sector rotation |
| **Deploy** | Railway (nixpacks), GitHub auto-deploy |

## Features

### Market Analysis
- **Daily Report** — nightly pipeline: S&P 500 ingest, YOLO pattern detection, LLM-powered setup analysis
- **Chart Workspace** — interactive candlestick with support/resistance levels, YOLO overlays, technical indicators
- **VIX Analysis** — regime detection, term structure, compression signals, trap/reclaim patterns
- **Market Sentiment** — internal composite + Finnhub news + StockTwits social + RSS headlines
- **Earnings Calendar** — upcoming catalysts with sector rotation context

### Paper Trading Bot
- **Critic Bot** — devil's advocate that kills low-conviction trades (A+ setups only, max 5 positions)
- **3-Stage Decision Pipeline** — analyst → debate → risk evaluation
- **Context Capture** — VIX, HMM regime, market breadth recorded at entry
- **Family Edge Tracking** — rolling win rate per setup family for learning
- **Bot Versioning** — frozen strategy snapshots, champion/challenger framework

### ML Forecasting
- **48 Features** — momentum, volatility, volume, trend, VIX regime, FRED macro, sector rotation, YOLO patterns
- **Walk-Forward Training** — purged validation with embargo gap, no data leakage
- **Triple-Barrier Labels** — profit target / stop loss / time expiry
- **Meta-Labeling** — secondary model to filter false positives
- **SHAP Analysis** — per-prediction feature attribution
- **Drift Detection** — weekly model accuracy monitoring

### Crypto
- **Live Binance WebSocket** — BTC, ETH, SOL, XRP, DOGE with 1-minute patching
- **Multi-Interval Aggregation** — server-side candle building from 1m base stream
- **Structure Engine** — support/resistance levels, market insights
- **Gap Detection** — auto-backfill on WS reconnect

### Prediction Markets
- **Polymarket Integration** — live prediction market odds as macro regime signals

## Architecture

```
Railway (24/7):
├── FastAPI backend (~560 lines, 10 routers)
├── React v2 dashboard (10 pages, lazy-loaded)
├── Binance WS → 5 crypto pairs (real-time)
├── Finnhub WS → SPY/QQQ always-on + on-demand tickers
├── Daily batch: yfinance → YOLO → LLM report (22:00 UTC Mon-Fri)
├── ML pipeline: 48-feature LightGBM with walk-forward validation
├── Paper trading: critic bot + context capture + family edge
└── SQLite at /data/trader_koo.db (persistent volume)
```

## Development

```bash
# Backend
.venv/bin/python -m uvicorn trader_koo.backend.main:app --reload --port 8000

# Frontend
cd trader_koo/frontend-v2 && npm run dev

# Tests
python -m pytest tests/ -v
```

## Admin API

All admin endpoints require `X-API-Key` header.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/admin/trigger-update?mode=full` | POST | Run full pipeline |
| `/api/admin/train-ml-model` | POST | Train LightGBM model |
| `/api/admin/ml-model-status` | GET | Training progress + results |
| `/api/admin/ml-shap-analysis` | GET | SHAP feature importance |
| `/api/admin/ml-drift-check` | GET | Model accuracy monitoring |
| `/api/admin/run-backtest` | POST | Walk-forward backtest vs SPY |
| `/api/admin/force-cancel-run` | POST | Kill stuck pipeline runs |
| `/api/admin/seed-ticker-history` | POST | Backfill historical data |

## License

Personal project. Not for redistribution.
