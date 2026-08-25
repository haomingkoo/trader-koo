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

For each release, the workflow:

1. checks out the exact CI-tested commit;
2. refuses deployment if Railway GitHub autodeploy is connected;
3. creates and verifies a named production database backup;
4. tests migrations and replay against a downloaded copy;
5. records the previous deployment and campaign state;
6. uploads the exact commit with paper writes disabled;
7. verifies the public release SHA, API contracts, and Chromium journeys; and
8. rolls back to the recorded deployment if a release check fails.

Campaign activation and reset endpoints are intentionally absent from the
deployment workflow. They require a separate authenticated human decision.

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

The detailed state model, rollback boundary, and database compatibility matrix
are in `ops/RELEASE_STATES.md`.

## Local verification

```bash
make ci
npm run lint --prefix trader_koo/frontend-v2
npm run build --prefix trader_koo/frontend-v2
```

Run deployment verification only against a release you are authorized to
inspect. Never place the admin API key in shell history, logs, or command output.
