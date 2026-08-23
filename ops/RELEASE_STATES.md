# Release states

Trader Koo deliberately separates these states:

1. **Local pass**: checks passed in one developer worktree only.
2. **CI pass**: the exact commit passed repository hygiene, full backend tests,
   frontend lint/unit/build across three timezones, security contracts, canonical
   replay evidence, and real Chromium journeys.
3. **Merged**: the commit is present on `main`; this is not a deployment.
4. **Dark deployed**: Railway built the exact commit after CI and the production
   database backup passed copy-only migrations. The first Campaign v2 release is
   inactive; later releases preserve the campaign's pre-deploy lifecycle state.
5. **Production verified**: exact SHA, API contracts, auth behavior, and the real
   Report, Chart, Paper Trades, Agent Observability, and Experiment Results pages
   passed against the public service.
6. **Campaign activated**: a human separately used the authenticated, audited
   campaign transition. CI/CD never calls activation or reset endpoints.

## Evidence and rollback

Each CI run stores hash-bound database migration, replay, and execution-ledger
JSON artifacts. Dark deployment first requests a fresh consistent online SQLite
backup, downloads that exact named and hash-verified regular file, runs
expand-compatible migrations against a separate copy, and retains only manifests
in GitHub Actions—not the database itself. A missing named backup is HTTP 404;
path traversal and symlinks are rejected.

Before upload, the workflow records the active Railway deployment and commit
hash from Railway's deployment metadata. Any failed
deploy or post-deploy check invokes Railway's `deploymentRollback` mutation for
that exact previous deployment, then verifies both health and the previous SHA.
The public `/api/release` contract is checked too when the restored image
supports it; this keeps the first migration from an older image bootstrappable.
Railway rollback restores its image and custom variables. The pre-deploy database
backup remains available for a separately approved data recovery; CD never
overwrites the live SQLite volume. Dark-deploy verification also fails unless
the `paper-v2` status exactly matches its pre-deploy state. When Campaign v2 is
first introduced, the only exception is `absent` to inactive `draft`. Activation requires a
different, explicitly approved release transition; later code releases can
preserve an active campaign without invoking activation.

Schema v4 is the initial expand phase. It retains legacy paper-trade and
portfolio-snapshot key shapes alongside the new campaign-aware keys. It
preserves legacy reads. The new image centrally blocks automatic, API, and admin
paper-lifecycle writes during the rollback window. If the previous image is
restored, the disabled environment flag blocks its automatic lifecycle and
operators must not call its authenticated paper-mutation endpoints. Destructive
contract migration is a separate operation after
the old-image rollback window is retired and before multi-campaign activation;
it requires a fresh named backup, integrity and foreign-key checks, free-space
review, and explicit data-recovery approval.

| Image and schema combination | Support |
| --- | --- |
| Previous image + pre-expand schema | Supported before deployment |
| New image + expand-compatible schema | Supported and migration-tested |
| Previous image + expand-compatible schema | Read-compatible rollback target; automatic writes disabled and mutation endpoints outside the supported rollback surface |
| Any retired image + contracted schema | Unsupported; image rollback must be retired first |

The copied-database gate verifies the v4 migration ID, required index uniqueness
and ordered columns, normalized trigger definitions, required foreign-key
declarations, compatible campaign defaults, foreign-key data, legacy read shapes,
integrity, and accounting. It does not claim to execute the previous container;
Campaign v2 and paper writes remain disabled throughout the image-rollback
window.

The public schema initializer owns its migration phases and rejects a connection
that already has a transaction. Private table rebuilds use an immediate
transaction, restore foreign-key settings, run `foreign_key_check` before
commit, and roll back on interruption. The initial dark release does not run a
destructive parent-table rebuild on the live volume.

## Ordered rollout

1. Land shared live/replay SPY chronology and prove the full chronology matrix.
2. Land named-backup HTTP, hash, freshness, regular-file, and authorization evidence.
3. Land schema transaction ownership, expand compatibility, copied-production
   upgrade checks, and previous-image rollback compatibility.
4. Land exact campaign-state verification, scoped CORS preflight behavior, and
   the Report, Chart, Paper Trades, Agent Observability, and Experiment SPA routes.
5. Refresh research evidence in a separate artifact-only commit bound to the
   declared source and dependency-manifest closure. This is source-level
   provenance, not a claim that two unresolved numerical environments are identical.
6. Let the exact commit pass CI, then approve the protected dark-deploy job.
7. Verify the public SHA, preserved campaign state, API contracts, and Chromium
   journeys. Activation remains a later, separately audited human decision.
8. In a separate change, implement and review the versioned contract migration,
   phase-aware initializer, contracted-schema verifier, copied-production test,
   and rollback-incompatibility evidence. The current v4 image rejects Campaign
   v2 activation because that migration does not exist yet.
9. After explicit rollback-retirement approval, take a named backup, run and
   verify the contract migration, deploy the contract-aware image, and perform a
   named non-production write smoke test with recovery criteria at each boundary.
10. In another audited configuration transition, enable paper writes and verify
    the API reports `write_state=enabled`.
11. Only then accept a separate human Campaign v2 activation request and verify
    its first production admission write and audit event.

Every later release must preserve the approved write-gate state instead of
blindly resetting it. An active campaign with `write_state=paused` is visibly
non-operational and fails campaign health until writes are deliberately restored.

Configure the `production-dark` GitHub environment with required reviewers,
the non-secret target variables `TRADER_KOO_PRODUCTION_URL`,
`RAILWAY_PROJECT_ID`, `RAILWAY_SERVICE`, and `RAILWAY_ENVIRONMENT`, plus the
`RAILWAY_TOKEN` and `TRADER_KOO_API_KEY` secrets referenced by
`.github/workflows/dark-deploy.yml`. Required branch checks should be the CI
jobs only after one green pull-request run proves their exact names.

The production Railway service must not have a GitHub repository source
attached. A connected source can deploy `main` outside the reviewed GitHub
environment, so the dark-deploy workflow treats it as a release-blocking
configuration error. CI uploads the exact tested commit with `railway up`.
