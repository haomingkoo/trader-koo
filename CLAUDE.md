# Trader Koo repository instructions

Trader Koo is a research-only FastAPI, React, and SQLite application deployed
as one Railway service. Paper trades never send broker orders.

## Source layout

- `trader_koo/backend/`: FastAPI app, routers, and scheduler modules.
- `trader_koo/frontend-v2/`: the served React/Vite frontend.
- `trader_koo/report/`: daily report generation and scoring.
- `trader_koo/paper_trade/`: paper lifecycle, replay, and campaign internals.
- `trader_koo/paper_trades.py`: stable paper-trading facade for callers.
- `trader_koo/research/`: fail-closed research runners and evidence artifacts.
- `trader_koo/scripts/`: operator and scheduled command entry points.
- `tests/`: backend, contract, and research tests.

Read `ARCHITECTURE.md`, `RESEARCH_STATUS.md`, and `DEPLOYMENT.md` before making
architecture, performance, or release claims.

## Invariants

- Do not invent market data. Unit tests may isolate external HTTP or LLM
  providers, but financial records must use the real schema and code paths.
- Use timezone-aware UTC datetimes in backend code.
- Sanitize LLM output before validating it against a schema.
- Preserve immutable report, policy, dataset, and artifact lineage. Missing or
  stale evidence must fail closed; never replace it with a hidden fallback.
- A deterministic test or synthetic replay proves software behavior, not market
  performance. Use the language in `RESEARCH_STATUS.md`.
- Database migrations must be additive and idempotent during the supported
  rollback window. Validate them on a copied production database.
- Campaign activation is a separate authenticated human action. Code changes,
  CI, and deployment must not activate or reset it.
- Keep secrets out of source, logs, fixtures, and command output.

## Verification

```bash
make ci
npm run lint --prefix trader_koo/frontend-v2
npm run build --prefix trader_koo/frontend-v2
```

Use focused tests while developing, then run verification in proportion to the
change. A local pass, CI pass, merge, deployment, and production-browser
acceptance are distinct states.
