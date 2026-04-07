# Trader Koo

Full-stack S&P 500 + crypto market analysis platform with ML-powered swing trade forecasting.

**Live**: [trader.kooexperience.com](https://trader.kooexperience.com)

![CI](https://github.com/haomingkoo/trader-koo/actions/workflows/ci.yml/badge.svg)
![Python >=3.10](https://img.shields.io/badge/Python-%3E%3D3.10-blue)
![React 19](https://img.shields.io/badge/React-19-61DAFB)
![Railway](https://img.shields.io/badge/Deploy-Railway-0B0D0E)

> For informational and educational purposes only. Not investment advice.

<!-- TODO: Add screenshot of dashboard -->

## Features

**Data**
- 7 sources (yfinance, Finnhub, Binance, Finviz, FRED, Polymarket, Alpha Vantage)
- 510+ tickers (S&P 500 + macro indices + 5 crypto pairs)
- Real-time streaming: Binance WebSocket (crypto), Finnhub WebSocket (equities)

**Analysis**
- 5-layer pattern detection: rule-based geometry, candlestick patterns, hybrid scoring, CV proxy, YOLOv8 AI
- ML forecasting: LightGBM with 51 features, walk-forward validation, SHAP explainability
- HMM regime detection (3-state), VIX term structure analysis, sector rotation tracking

**Trading**
- Paper trading with full lifecycle ($1M simulated portfolio)
- Hyperliquid whale tracker with configurable wallets, reload detection, and free Binance crowd context
- Debate engine: 5-role analyst panel + deterministic arbiter
- ATR-aware risk controls, critic review, graduated trailing stops, equity curve, family edge tracking

**Frontend**
- 12 app pages plus methodology, alerts, Hyperliquid, and 404 fallbacks
- Dark/light mode, interactive Plotly charts with overlays, live streaming tickers in header
- Lazy-loaded routes, Zustand state management

**Ops**
- Daily pipeline (Mon-Fri 22:00 UTC): ingest, YOLO detection, LLM report generation, email delivery
- 23 admin endpoints with API key auth
- Pipeline status monitoring, audit logging, LLM health checks

## Quick Start

```bash
git clone https://github.com/haomingkoo/trader-koo.git
cd trader-koo

# Configure environment — see .env.example for all variables
cp .env.example .env

make backend-install
make frontend-install
make hooks-install

# Backend
uvicorn trader_koo.backend.main:app --reload --port 8000

# Frontend (separate terminal)
cd trader_koo/frontend-v2 && npm run dev

# Local CI parity
make ci
```

## Quality

- GitHub Actions runs repo hygiene, backend tests, and frontend production builds on every push and pull request.
- Dependabot is configured for Python, npm, and GitHub Actions dependency updates.
- Pre-commit hooks enforce basic hygiene and secret scanning before code lands.
- See [CONTRIBUTING.md](CONTRIBUTING.md) for the expected local workflow.

## Architecture

Single-process deployment on Railway — API server, WebSocket clients, background scheduler, and ML pipelines all run together. No separate worker, no message queue, no external database.

See [ARCHITECTURE.md](ARCHITECTURE.md) for full details.

```
Railway Service (single process)
├── FastAPI (uvicorn, port 8080)
│   ├── API routers + admin modules
│   ├── Static file serving (React build → /)
│   └── WebSocket endpoints (/ws/crypto, /ws/equities)
│
├── Binance WS Client
│   └── 5 crypto pairs: BTC, ETH, SOL, XRP, DOGE (real-time 1m klines)
│
├── Finnhub WS Client
│   └── SPY/QQQ always subscribed + on-demand symbols from UI
│
├── APScheduler (BackgroundScheduler, in-process)
│   ├── Mon–Fri 22:00 UTC: daily_update.sh (ingest → YOLO → report)
│   └── Saturday 00:30 UTC: YOLO full seed (daily + weekly timeframes)
│
└── /data/ (Railway persistent volume)
    ├── trader_koo.db    (SQLite app data)
    ├── models/          (LightGBM .pkl files)
    ├── reports/         (daily JSON + MD archives)
    └── logs/            (structured log files)
```

## Stack

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI + APScheduler + SQLite (Railway persistent volume) |
| **Frontend** | React 19 + Vite + TypeScript + Tailwind v4 |
| **ML** | LightGBM + SHAP + walk-forward validation |
| **AI** | YOLOv8 chart pattern detection (foduucom/stockmarket-pattern-detection-yolov8) |
| **Data** | yfinance, Finnhub WS, Binance WS, Finviz, FRED, Polymarket |
| **Deploy** | Railway (nixpacks, asia-southeast1) |

## License

MIT
