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
- Every experiment result exposes its implementation hash, data basis, split,
  gate results, and artifact identity. Validation cannot read the sealed held-out
  partition, and the seal is only trusted inside the controlled evidence store.
- Historical universe claims fail closed until point-in-time membership is
  enforced; the current index membership is never presented as historical proof.
- Paper entries use the first eligible open after verified publication. That
  session's high, low, close, and final volume cannot influence admission,
  sizing, or the entry fill.
- Live and replay execution produce the same canonical ledger for the same
  campaign policy, report lineage, and market data.
- A dark deployment verifies the exact commit and an inactive `paper-v2`
  campaign through the public API. Activation is a separate authenticated,
  audited human action.
- The Report, Chart, Paper Trades, Agent Traces, and Experiment Results journeys
  pass in Chromium with keyboard focus, non-color status labels, and no clipped
  decision text.
