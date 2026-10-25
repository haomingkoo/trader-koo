# Architecture restructure design

Status: draft for owner approval. Date: 2026-10-26.

## Why

The owner's verdict on the current codebase: too many files, too many moving
parts, no layering, no reusable components, hard-coded values and magic numbers
everywhere, and "fake machinery" that presents itself as more than it is.

A read-only inventory (2026-10-25/26) backs this up:

- About 79,000 lines of Python in 210 files, 38,000 lines of tests and
  23,000 lines of frontend, for a research tool whose signals show no measured
  edge.
- `trader_koo/` has 34 top-level folders and loose modules. One pipeline is
  spread across `report/`, `structure/`, `analysis/`, `cv/`, `signals/` and
  `features/`.
- About 35 jobs are implemented more than once: 7 pattern detectors, 5
  backtest simulators, 4 price loaders plus inline SQL in about 25 files, 5
  schedulers, 4 alert-cooldown systems, 5 backup paths, 4 copies of RSI/EMA, 5
  VIX readers, 3 Telegram senders, 3 Finnhub clients.

The real, proven parts are the foundation: ingest, price-basis checks, report
lineage, the paper ledger and campaign governance, admin security, deploys.
They get kept and moved, not rewritten for its own sake.

## Goal

One codebase where each module has one job, dependencies point one way, every
tunable number lives in config with a name and a reason, nothing exists twice,
and nothing claims more than it is. Target size: about 24,000 lines of backend
and about 9,000 lines of frontend. This is an estimate to steer by, not a gate.

## Non-goals

- Changing trading behaviour. Every restructure step is behaviour-identical.
  Strategy changes come later through the research process and a new campaign
  policy version approved by the owner.
- The debate agents and the analysis sandbox. They get their own design once
  this layout exists, and will live in `agents/`.
- Changing the database engine (SQLite stays) or the Railway topology (one
  service stays). Both are open questions for later, not part of this work.

## Anti-goals (what would make this a failure even if it works)

- A big-bang rewrite that breaks the nightly report or the live campaign.
- Renaming and moving files without removing duplication: same mess, new paths.
- New abstraction layers (base classes, registries, plugin systems) that exist
  for symmetry rather than to remove a duplicate.
- Any loss of report or paper lineage, or any change to the campaign policy
  fingerprint.

## Target layout

The package name stays `trader_koo` (no `src/` move; it would churn packaging
and deploys for no gain).

```
trader_koo/
  config/          # settings loader + config/settings.toml (all tunables)
  market_data/     # 1. the only module that talks to vendors and owns price/fundamental/options tables
    sources/       #    one client per vendor: yahoo, finnhub, finviz
    store.py       #    load_prices(), upsert_prices(), the one loader
    basis.py       #    price-basis contract, corporate actions
    ingest.py
  signals/         # 2. pure functions over OHLCV, no I/O
    indicators.py  #    RSI, EMA, ATR, Williams %R (one copy)
    levels.py      #    support/resistance, trendlines, gaps
    patterns.py    #    rule + candlestick patterns
    yolo.py        #    YOLO detector wrapper
    regime.py      #    MA regime, HMM, VIX (one VIX reader)
  report/          # 3. score, assemble, publish one daily report
    scoring.py
    calibration.py
    build.py
    lineage.py     #    canonical runs, hashes
    render/        #    web, email, telegram templates over one renderer
  paper/           # 4. admission, ledger, campaign (behind the paper_trades facade)
  research/        # 5. one backtest engine (next_open_baseline) + evidence catalogue
  notify/          # 6. one send(channel, dedup_key, body), persisted dedup
  web/             # 7. FastAPI app, auth, audit, thin routers
  ops/             # 8. scheduler, pipeline runs, backups, maintenance
  addons/          # side products, isolated: crypto/, hyperliquid/, polymarket/
  agents/          # reserved for the debate design (empty until then)
frontend-v2/src/
  components/      # shared UI building blocks
  pages/
  api/             # types generated from the backend OpenAPI schema
```

The owner has decided to keep crypto, Hyperliquid, Polymarket and the Telegram
alerts. Telegram goes into `notify/`; the other three go under `addons/`.

## Dependency rules

Allowed direction, top imports bottom only:

```
web, ops
  └─ paper, research, addons
       └─ report
            └─ signals
                 └─ market_data, notify
                      └─ config
```

- `signals/` has no I/O: it takes data frames and returns values.
- Only `market_data/` reads or writes market-data tables. Other modules call
  `market_data.store` functions, never inline SQL against those tables.
- `addons/*` may use `data`, `notify` and `config`. No core module imports
  `addons/`.
- Paper callers go through `trader_koo/paper_trades.py` (existing invariant).

## Config and magic numbers

- All tunables (thresholds, windows, weights, timeouts, limits) move to
  `config/settings.toml`, loaded once into frozen, typed settings objects.
  Each entry has a name, a unit and a one-line reason or source.
- Exception: `PaperTradeConfig` and anything inside `config_snapshot()` stay
  exactly as they are. The live campaign is registered against
  `canonical_hash(config_snapshot(config))`, and changing it stops the campaign.
  Those values may only be read from the new settings if the snapshot and its
  hash stay byte-identical.
- A constant moves only with its current value. Changing a value is a separate,
  justified change.

## Duplicate resolution

The survivor for each duplicated job (from the inventory):

| Job | Survivor |
|---|---|
| Price loading | `market_data.store.load_prices()` |
| Price schema, scale-break detection | `market_data/` |
| Indicators (RSI, EMA, ATR) | `signals/indicators.py` |
| Line fitting, weekly resample, chart PNG render | one helper each in `signals/` |
| VIX readers | `signals/regime.py` |
| Pattern detectors | rule + candle + YOLO; cv/ and hybrid already deleted |
| Backtest simulators | `research/` next_open_baseline |
| SPY-matched return, Sharpe, drawdown | one helper each in `research/` |
| Family/edge gating | calibration (paper-side copies only if outputs are identical; otherwise flagged) |
| Telegram senders, Finnhub clients, email transport | one each |
| Alert cooldowns | one persisted dedup table in `notify/` |
| Schedulers | APScheduler in `ops/` plus the external GitHub monitor |
| Overlap guards | the DB pipeline-run row |
| Backups, storage cleanup | one each in `ops/` |
| CORS, throttles | one each in `web/` |
| Migration mechanisms | keep all three tables until a separate runbook merges them (paper schema v5 contract) |

Where two copies differ in behaviour, the survivor keeps the behaviour that
production uses today, and the PR states the difference.

## How it moves (strangler, not rewrite)

Each step is one or a few PRs, merged and deployed before the next starts.

0. Cleanup PRs already in flight: delete ML, the rule-based debate, cv/,
   hybrid patterns, dead research machinery, subscribe flow, usage analytics,
   social sentiment, the Gemini report rewrite.
1. Guardrails first: add import-linter contracts, a file-size check (400
   lines), a duplicate-code check and a bare-number check. They run in CI in
   report-only mode against a recorded baseline, and become blocking once each
   module is migrated.
2. `config/` loader and settings file (excluding the paper policy snapshot).
3. `market_data/`: one loader, one vendor client each; replace inline SQL.
4. `signals/`: consolidate indicators, levels, patterns, YOLO, regime/VIX.
5. `notify/`: one send path with persisted dedup.
6. `ops/`: one scheduler, one overlap guard, one backup path.
7. `report/`: move scoring, calibration, lineage, build, renderers; shrink the
   generator.
8. `paper/` and `research/`: move behind the facade; split trading.py.
9. `web/`: thin routers on shared loaders.
10. `addons/`: move crypto, Hyperliquid, Polymarket.
11. Frontend: shared components, generated API types, split oversized pages.

## Proof each step changes nothing

Every PR must show, in its description:

- Policy fingerprint identical on main and on the branch.
- Golden report: the report generator run on a fixed copied database produces
  identical output to main, apart from fields the PR names and explains.
- Paper decisions identical on the existing paper test fixtures.
- No change to `paper_*` / `report_*` schema or `_PAPER_DECISION_VERSION`.
- `make ci`, frontend lint and build pass; production deploy health check passes
  after merge; the next nightly run publishes.

## Done means

- The tree matches the target layout; no top-level loose modules remain except
  the paper facade.
- Import-linter, file-size, duplicate and bare-number checks are blocking in CI
  and pass.
- Each job in the duplicate table has exactly one implementation.
- Backend and frontend line counts are reported against the targets.
- `ARCHITECTURE.md` describes the new layout and nothing else.

## Decisions from the builder check

Calls marked "builder call" were settled by the author and can be overridden.

- **Golden-report harness comes first (builder call).** No snapshot harness
  exists today. Before step 1, build one: frozen clock, LLM and vendor calls
  stubbed, a pinned copy of the price database kept outside git (local
  `trader_koo/data/trader_koo.db` or a read-only export of production tables;
  never committed), and a normaliser that drops volatile fields such as
  timestamps. "The report" means the report JSON, the rendered email body and
  the paper decision set for that report.
- **Step 0 is a precondition.** The cleanup PRs (ML, fake signals, research
  machinery, subscribers and misc, Gemini rewrite) must merge before any file
  moves.
- **Name collisions (builder call).** `trader_koo/data/` already holds local
  runtime files (dev DB, cache, CV label CSVs). Production uses the `/data`
  volume through `TRADER_KOO_DB_PATH`, so nothing live depends on the package
  folder. The new module is named `market_data/` to avoid the collision, and
  side products go under `addons/` instead of `markets/`. Existing
  `config.py`, `signals/`, `research/` and `notifications/` are merged into
  the new modules of the same purpose in the step that owns them.
- **Entrypoints and old import paths (builder call).** The Railway start
  command, `start.sh`, `railway.toml`, `nixpacks.toml`, GitHub workflows and
  scripts may change, but only in a PR that deploys through the dark-deploy
  gate. Each move leaves a re-export shim at the old path for one step; all
  shims are deleted in the last step. The SwiftBar plugins only call public
  HTTP endpoints and are unaffected.
- **Duplicates that differ (builder call).** Report output and paper decisions
  must stay identical. A duplicate whose output differs from the survivor may
  be merged only if its callers are off the production report and paper path,
  and the PR lists the changed outputs. Paper-side duplicates that would change
  decisions stay, and are listed as exceptions in `ARCHITECTURE.md`. "Done
  means" counts those listed exceptions as allowed.
- **Pace (builder call).** One restructure PR deploys per trading day, and the
  next one waits until the nightly run after that deploy has published. No
  deploys between 21:30 and 23:30 UTC on trading days (nightly run window).
- **Guardrail definitions (builder call).** File limit is 400 lines, excluding
  tests, generated frontend types, `settings.toml` and listed legacy files
  that shrink as steps land. A bare number is a numeric literal other than 0,
  1, -1, 2 and 100 in non-test backend code outside `config/` and `paper/`.
  The frontend gets the duplicate and file-size checks only.
- **Paper config (builder call).** `paper/` keeps its constants, hashed or
  not, where they are. It moves as code only, and is exempt from the
  bare-number check, because moving decision constants is where behaviour
  drift would come from.
- **Frontend types (builder call).** Routers gain `response_model`
  annotations in step 9, before the frontend switches to generated types in
  step 11. Frontend page splits are verified with Playwright screenshots at
  1440, 768 and 390 pixels against main, plus the existing browser contract
  tests.

## Left to the builder

File names inside each module, the exact config file schema, which
import-linter and duplicate-check tools to use, and how to split each step into
PRs, as long as the rules above hold.
