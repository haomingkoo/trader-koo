# Release states

Trader Koo deliberately separates these states:

1. **Local pass**: checks passed in one developer worktree only.
2. **CI pass**: the exact commit passed repository hygiene, full backend tests,
   frontend lint/unit/build across three timezones, security contracts, canonical
   replay evidence, and real Chromium journeys.
3. **Merged**: the commit is present on `main`; this is not a deployment.
4. **Dark deployed**: Railway built the exact commit after CI and the production
   database backup passed copy-only migrations. Campaign v2 remains inactive.
5. **Production verified**: exact SHA, API contracts, auth behavior, and the real
   Report, Chart, Paper Trades, Agent Observability, and Experiment Results pages
   passed against the public service.
6. **Campaign activated**: a human separately used the authenticated, audited
   campaign transition. CI/CD never calls activation or reset endpoints.

## Evidence and rollback

Each CI run stores hash-bound database migration, replay, and execution-ledger
JSON artifacts. Dark deployment first requests a fresh consistent online SQLite
backup, downloads that exact latest backup, runs additive migrations against a
separate copy, and retains only manifests in GitHub Actions—not the database
itself.

Before upload, the workflow records the active Railway deployment and commit
hash from Railway's deployment metadata. Any failed
deploy or post-deploy check invokes Railway's `deploymentRollback` mutation for
that exact previous deployment, then verifies both health and the previous SHA.
The public `/api/release` contract is checked too when the restored image
supports it; this keeps the first migration from an older image bootstrappable.
Railway rollback restores its image and custom variables. The pre-deploy database
backup remains available for a separately approved data recovery; CD never
overwrites the live SQLite volume. Dark-deploy verification also fails unless
`paper-v2` remains inactive; activation requires a different, explicitly approved
release transition.

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
