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
- For verified candidate runs, live and replay execution produce the same
  canonical ledger for the same campaign policy, report lineage, and market
  data. Report-level lineage refusal is a separate pre-admission contract and
  produces no candidate ledger on either path.
- The first Campaign v2 dark deployment verifies the exact commit and introduces
  `paper-v2` inactive. Later software releases must preserve the pre-deploy
  campaign status, including `active`; the sole bootstrap transition is
  `absent` to inactive `draft`. CI/CD never activates, resets, or rolls
  back the campaign. Activation is a separate authenticated, audited human action.
- The Report, Chart, Paper Trades, Agent Traces, and Experiment Results journeys
  pass in Chromium with keyboard focus, non-color status labels, and no clipped
  decision text.

### Chronology acceptance matrix

| Input at the immediate next SPY session | Live result | Promotion-parity replay result |
| --- | --- | --- |
| Valid publication strictly before 09:30 ET and ticker open present | Same admitted fill and canonical fields | Same admitted fill and canonical fields |
| Publication exactly at 09:30 ET or later | `report_published_after_intended_open` | Same rejection, intended session, and sealed inputs |
| Publication missing or malformed | Reject the report-level admission request before any candidate decision is sealed | Refuse the unverified run before parity comparison |
| Scheduled-session SPY observation missing | `pending` with `scheduled_spy_open_missing` | Same pending disposition and reason; no later row is consulted |
| Scheduled-session ticker open missing | `pending` with `scheduled_ticker_open_missing` | Same pending disposition and reason; no later row is consulted |

Missing or malformed publication metadata is a report-lineage error, not a
candidate rejection; database publication guards normally make this state
unrepresentable. The promotion-parity replay refuses any run that is not stored
as verified and published before comparing candidates. For a verified report,
late publication has precedence over
observation availability and is rejected before SPY or ticker gaps can create a
pending order. SPY precedence applies next, then ticker availability. Verified
candidate runs use the same late/SPY/ticker ordering as live admission.

A lineage refusal aborts the admission transaction before decision sets, orders,
or trades are written. Missing/unpublished lineage raises
`ReportLineageError(code="report_not_verified_published")`; structurally invalid
lineage raises `ReportLineageError(code="report_publication_lineage_invalid")`;
a valid superseded run presented to live admission raises
`ReportLineageError(code="report_not_current_publication")`. The first two are
retryable only after the same run has valid verified-publication evidence. A
superseded run is terminal for live admission, whose callers must use the
current canonical run. Historical promotion replay may use a superseded run:
it verifies immutable artifact lineage and compares the sealed live facts
produced when that run was current, without admitting new work.

After schema initialization, a known-run outer admission appends a success fact
or attempts a separate durable failure fact in the immutable
`report_admission_attempts` ledger. A process crash before the failure insert, an
unknown run ID, or an audit-storage failure can leave no failure fact; that
condition is logged and the original exception is preserved. Stored failures
contain the stable code and exception class in the compatibility-named
`error_message` column, never raw error text or file paths.
The API rejects caller-owned transactions, and the ledger does not rewrite an
immutable published run. There is no paper-admission
HTTP endpoint that maps it to a transport status in v4. The lower-level
`replay_campaign()` simulation accepts caller-supplied research fixtures; only
`replay_and_seal_promotion()` is the lineage-verified promotion boundary.
Caller-owned transaction rejection and schema-initialization failures are
precondition failures outside the admission-attempt ledger. Non-lineage failure
codes are a closed phase set: `admission_setup_persistence_failed`,
`admission_paper_trade_persistence_failed`, and `admission_finalize_failed`;
the exception class is stored separately as diagnostic metadata. Setup includes
artifact-derived preparation after lineage verification. Existing malformed
legacy ledger rows stop schema initialization with an operator-facing error;
because the ledger is audit evidence, recovery is an explicit reviewed database
migration or backup restore, never an automatic rewrite or quarantine. The
legacy scan validates migration state; the insert trigger enforces all later
rows.

The one pre-contract code `admission_lineage_failed` is grandfathered only for
immutable historical rows so an upgrade does not destroy or block older audit
evidence. New inserts cannot create it. Before rollout, operators back up the
database and run the release-evidence copied-database verifier; a refusal reports
the admission-ledger contract, and the backup remains the rollback source. No
automatic mapping is attempted because the old code does not contain enough
information to infer a truthful replacement phase.
Release database evidence uses the versioned `release-database-copy-v2`
manifest. Failed contracts report the total count and an explicitly truncated,
ascending sample of at most 20 numeric attempt IDs with invariant categories.

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
