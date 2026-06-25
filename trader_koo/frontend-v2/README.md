# Trader Koo Frontend

React + TypeScript dashboard for the Trader Koo research workflow.

## What This Frontend Shows

- Guide: a first-run explanation of the research workflow and guardrails.
- Report: daily market report with evidence freshness and stale-data warnings.
- Chart: ticker workspace for price action, levels, patterns, and commentary.
- Opportunities: screened research candidates.
- Paper Trades: simulated trade journal, decision flow, and outcome review.
- Alerts: Telegram price alerts, market spikes, and system events.
- Markets: VIX, options, crypto, Hyperliquid, and prediction-market views.
- Methodology: architecture and model pipeline notes.

## Local Development

From `trader_koo/frontend-v2`:

```bash
npm install
npm run dev
```

The frontend expects the backend API at `http://127.0.0.1:8000` unless Vite
environment settings override it.

## Verification

```bash
npm run lint
npm run build
```

The production build is static and can be served behind the FastAPI backend or
from a static host. Public crawler files live in `public/robots.txt`,
`public/sitemap.xml`, and `public/llms.txt`.

## Product Principles

- Show evidence before claims.
- Make missing data visible.
- Keep paper trading visibly separate from real execution.
- Use Telegram for actionable triage, then link users back to the correct web
  workspace for deeper review.
