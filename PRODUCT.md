# Product

## Register

product

## Users

Hands-on market researchers, engineers, and reviewers who need to inspect data quality, model evidence, paper-trading decisions, and operational health in one working dashboard. They use dense screens on desktop and need to distinguish observed facts from unavailable or unqualified evidence quickly.

## Product Purpose

Trader Koo is evidence-first market research software. It connects provider-backed market views, reproducible research, immutable decision records, and simulated paper trading while keeping research separate from real execution. Success means a user can trace every displayed claim to its data and lineage, understand why a result failed closed, and never mistake an incomplete experiment for an actionable recommendation.

## Brand Personality

Sober, exact, and quietly confident. The product should feel like a well-maintained analytical instrument: direct enough for daily use, rigorous under review, and calm when evidence is missing.

## Anti-references

- Trading dashboards that use neon spectacle, urgency, or gamified profit language.
- Generic AI dashboards that hide provenance behind polished summaries.
- Dense terminal cosplay that sacrifices hierarchy and legibility.
- Research pages that replace missing metrics with zeros or imply qualification through color alone.

## Design Principles

- Evidence before claims: lineage, basis, freshness, and gates stay near the result they qualify.
- Failure is a first-class state: unavailable and invalid evidence remains inspectable and useful.
- Familiar controls, expert density: use standard product patterns and reserve visual emphasis for decisions and risk.
- One truth across surfaces: API, artifact, table, chart, and explanatory copy must reconcile.
- Research never masquerades as execution: paper-only and inactive states remain explicit.

## Accessibility & Inclusion

Target WCAG 2.1 AA contrast and keyboard operation. Never rely on color alone for state, retain visible focus treatment, respect reduced-motion preferences, and keep data warnings readable in both supported themes.

## Release acceptance

- A published report resolves through its immutable run record and verified
  artifact hashes; a missing or mismatched lineage is displayed as unavailable.
  The only exception is a pre-migration report directory whose registry exists
  but contains zero runs and has no manifest. It may be displayed as visibly
  `unlinked legacy` and is never eligible for paper admission or research proof.
- Every experiment result exposes its implementation hash, data basis, split,
  gate results, and artifact identity. Validation cannot read the sealed held-out
  partition, and the seal is only trusted inside the controlled evidence store.
- Historical universe claims fail closed until point-in-time membership is
  enforced; the current index membership is never presented as historical proof.
- Paper entries target exactly the first scheduled NYSE session after the
  report date. Verified publication must be strictly earlier than 09:30 ET on
  that session; equality or later publication expires the order rather than
  rolling it to another day. The session's high, low, close, and final volume
  cannot influence admission, sizing, or the entry fill. The NYSE calendar
  defines the intended date; an observed SPY open must exist on that exact date
  or execution fails closed without consulting a later row.
- Live and replay execution produce the same canonical ledger for the same
  campaign policy, report lineage, and market data.
- The first Campaign v2 dark deployment verifies the exact commit and introduces
  `paper-v2` inactive. Later software releases must preserve the pre-deploy
  campaign status, including `active`; the sole bootstrap transition is
  `absent` to inactive `draft`. CI/CD never activates, resets, or rolls
  back the campaign. Activation is a separate authenticated, audited human action.
- The Report, Chart, Paper Trades, Agent Traces, and Experiment Results journeys
  pass in Chromium with keyboard focus, non-color status labels, and no clipped
  decision text.

### Chronology acceptance matrix

| Input at the immediate next SPY session | Live result | Replay result |
| --- | --- | --- |
| Valid publication strictly before 09:30 ET and ticker open present | Same admitted fill and canonical fields | Same admitted fill and canonical fields |
| Publication exactly at 09:30 ET or later | `report_published_after_intended_open` | Same rejection, intended session, and sealed inputs |
| Publication missing or malformed | Fail closed with the matching publication reason | Same rejection and sealed inputs |
| Scheduled-session SPY observation missing | `pending` with `scheduled_spy_open_missing` | Same pending disposition and reason; no later row is consulted |
| Scheduled-session ticker open missing | `pending` with `scheduled_ticker_open_missing` | Same pending disposition and reason; no later row is consulted |

Canonical parity compares campaign, report/run lineage, ticker, direction,
decision and reason, intended session, entry date/price, sizing inputs, costs,
and the sealed-input hash. Later-session OHLCV is never an allowed substitute.
Pending orders automatically retry only the exact intended date when that
observation is backfilled; every fill or rejection is appended to the immutable
order-event ledger, and they never advance to another date. Campaign rollback
does not silently cancel unresolved orders: they remain visibly pending until a
separately audited terminal-resolution operation is implemented or the exact
observation arrives. Operators must include unresolved pending orders in every
rollback review.

### Release-state acceptance matrix

| Pre-deploy Campaign v2 state | Required post-deploy state |
| --- | --- |
| Absent on the first schema bootstrap | `draft` |
| `draft` | `draft` |
| `frozen` | `frozen` |
| `active` | `active` |

Empty, unknown, or different states fail deployment verification and trigger
image rollback. Operators must not overlap a human campaign transition with a
dark deployment; a detected state race is a failed release, never an inferred
transition.
