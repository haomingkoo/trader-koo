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
window. The legacy `paper_trades.report_run_id` foreign key is intentionally
deferred to the contract migration because SQLite requires a table rebuild to
add it; the expand-only v4 gate does not pretend that constraint already exists.

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
8. Specify and review the v5 contract matrix and fixtures as its own change. It
   must name every retained, removed, and changed table, index, trigger, foreign
   key, default, legacy read shape, collision rule, integrity check, and schema
   fingerprint. The current v4 image rejects Campaign v2 activation because
   the verifier does not exist yet. The frozen specification is
   `paper-schema-contract-v5.json`; `ops/PAPER_SCHEMA_V5_CONTRACT.md` records its
   scope and deliberately deferred implementation steps.
9. Implement and review the migration against the frozen contract fixtures.
   The dedicated maintenance seam is `migrate_paper_schema_v4_to_v5`; it has no
   startup or production-command caller and does not verify or activate v5.
10. Implement and review the exact verifier and schema fingerprint separately.
11. Implement the phase-aware initializer as another change, proving it operates
    safely on both v4 expand and v5 contract states.
12. Implement the writer-quiescence control and recovery drills separately,
    including active-transaction detection, timeouts, abort criteria, backup
    retention, restore evidence, and the restore-versus-complete decision point.
13. Before rollback retirement or any production migration, pass the migration,
    rollback-incompatibility checks, and full v5 journey against a copied
    production database in a named non-production environment: verifier,
    audited activation idempotency, write-state restart, and first admission
    write without test bypasses.
14. Before enabling writes, change the deployment workflow to capture and audit
    the pre-deploy write state, preserve it through deploy and rollback, and
    verify both success and rollback paths. Include the absent/first-release rule
    and the required process restart for configuration changes.
15. Deploy the phase-aware image while production is still on v4 and verify its
    API, background workers, schema no-op behavior, and rollback drill. Only
    after that checkpoint and explicit rollback-retirement approval may the
    operator quiesce every v4 writer, take a named backup, migrate under a
    maintenance boundary, then restart and verify that same image on v5.
16. In another audited configuration transition, enable paper writes and verify
    the API reports `write_state=enabled`.
17. Only then accept the separate human Campaign v2 activation request and
    verify its first production admission write and audit event.

The current dark-deploy workflow deliberately resets writes to paused on every
release. Before the first activation, the contract-aware workflow must be changed
to capture, audit, preserve, and verify the approved write-gate state across
deploy and rollback. An active campaign with `write_state=paused` is visibly
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
