# Deployment

Production releases are performed by
`.github/workflows/dark-deploy.yml` after the exact `main` commit passes CI.
Do not deploy by attaching the Railway service to GitHub or by manually pushing
a branch to production.

## Release contract

The protected `production-dark` environment must provide the Railway project,
service, environment, and production URL variables plus the Railway project
token and Trader Koo admin API key secrets used by the workflow. Secret values
must not be committed or printed.

The job runs only for a successful CI `push` run on this repository's `main`
(never for a fork pull request), or for the rollback drill below. For each
release, the workflow:

1. checks out the exact CI-tested commit, confirms a successful CI push run
   exists for it, and refuses a commit that is no longer the `main` head (a
   re-run of an old workflow);
2. reads the production SHA from the public `/api/release` endpoint and
   refuses a commit that does not descend from it;
3. refuses deployment if Railway GitHub autodeploy is connected;
4. refuses to restart production while `/api/status` reports
   `pipeline_active`, waiting up to 15 minutes, and inside the weekday nightly
   window of 21:15-23:15 UTC around the 22:00 UTC `daily_update` job. These
   checks run before the backup, so a refused run writes nothing to the
   production volume;
5. creates and verifies a named production database backup;
6. tests migrations and replay against a downloaded copy;
7. records the previous deployment, the campaign state and the
   `TRADER_KOO_PAPER_TRADE_ENABLED` write gate;
8. uploads the exact commit with the pre-deploy write gate preserved, then
   waits for the deployment that `railway up` printed to pass Railway's health
   check;
9. verifies the public release SHA, API contracts, and Chromium journeys; and
10. rolls back to the recorded deployment if any later step fails or the run is
    cancelled.

`railway up --ci` can exit 0 when the build logs end, before the health check
finishes, so the workflow polls the deployment itself. Railway waits up to
600 seconds for `/api/health` (`railway.toml`). The poll and step timeouts
cover that plus about 2 seconds of CLI time per poll, and the 135-minute job
timeout also counts the 15-minute pipeline wait. It leaves room for the
rollback because a job timeout skips cleanup steps.

Rollback (`ops/dark_deploy_rollback.sh`) calls Railway's `deploymentRollback`
mutation for the recorded deployment, which restores its image and custom
variables. It first finds every dark deployment, from the id in the
`railway up` log plus any upload of this release missing from the recorded
deployment list, and cancels each one still in flight. The restored deployment
is the one new entry in the deployment list, or the recorded deployment if
Railway reactivates it instead. The rollback succeeds only once every dark
deployment has stopped, because one still building could pass its health check
later and take traffic back, and then only if health, the previous SHA through
`/api/release`, the campaign state and the write gate all match. Railway
records no commit hash for `railway up` uploads, so the public release SHA is
the code check. GraphQL error bodies are printed; they never contain the
token.

The release evidence manifest (`dark-deploy-manifest-v3`) is kept for 90 days.

Campaign activation and reset endpoints are intentionally absent from the
deployment workflow. They require a separate authenticated human decision.

## Rollback drill

The drill proves that the rollback path works end to end. It runs the normal
release steps under the same `production-dark` approval, then fails on purpose
after every verification passed, so the rollback step runs against production.
Run it only when production already serves the `main` head; otherwise the
workflow refuses, so a drill never ships code. Avoid the nightly window.

```text
gh workflow run dark-deploy.yml --ref main -f rollback_drill=true
gh run list --workflow dark-deploy.yml --event workflow_dispatch --limit 3 --json databaseId,createdAt,status
gh run watch RUN_ID
```

The new run can take a few seconds to appear. Pick the `RUN_ID` whose
`createdAt` is after the dispatch, not an older drill.

Approve the run in the `production-dark` environment. The drill passes when the
job ends failed at "Fail on purpose to exercise the rollback drill", the
rollback step succeeded, and the evidence manifest shows `rollback_drill: true`
with a `rollback_deployment_id` and the expected campaign state and write gate.
Then confirm `/api/release` still returns the `main` SHA. Because the drill
redeploys the running SHA, the SHA check cannot tell the two images apart; the
deployment ids in the run log and manifest show which one serves traffic.

## Verification

Use the GitHub Actions run as the release record. A merge is not a deployment,
and a healthy process is not production acceptance. Confirm all of the
following before calling a release complete:

- the CI workflow passed for the released SHA;
- the dark-deploy workflow passed for that same SHA;
- `/api/health` returns `ok: true`;
- `/api/release` returns the expected Git SHA;
- production browser journeys passed; and
- the paper campaign state matches the pre-deploy state.

The `production-dark` environment requires reviewer approval, so a merge stays
undeployed until a dark-deploy run is approved. Unapproved runs end cancelled.
Check `gh run list --workflow dark-deploy.yml` and compare the `/api/release`
git_sha with `origin/main`.

Run deployment verification only against a release you are authorized to
inspect. Never place the admin API key in shell history, logs, or command output.

The detailed state model, rollback boundary, and database compatibility matrix
are in `ops/RELEASE_STATES.md`.
