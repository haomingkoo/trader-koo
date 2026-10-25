# Trader Koo repository instructions

Trader Koo is a research-only FastAPI, React, and SQLite application deployed
as one Railway service. Paper trades never send broker orders.

Callers use `trader_koo/paper_trades.py` (the stable facade) rather than
importing `trader_koo/paper_trade/` internals.

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
- Changing `_PAPER_DECISION_VERSION` or adding a column to a `paper_*` or
  `report_*` table is a contract change (`ops/PAPER_SCHEMA_V5_CONTRACT.md`).
  `PAPER_TRADE_SCHEMA_VERSION` in `schema.py` is a floor, not production state.
  `ensure_paper_trade_schema` runs no migration on a v5 database. A bump
  needs its own runbook.
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
