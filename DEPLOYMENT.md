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

## Build inputs

Railway builds with nixpacks (`railway.toml`, `nixpacks.toml`, `build.sh`).
Every input is pinned so the image matches what CI tested:

- Python 3.11.16 and Node 22.23.4 come from one fixed nixpkgs revision in
  `nixpacks.toml`. `build.sh` fails when `node --version` differs from
  `trader_koo/frontend-v2/.nvmrc`, which CI's `setup-node` also reads. Moving
  the revision changes both, so update `.nvmrc` with it.
- The nixpacks install phase runs
  `pip install --no-deps --require-hashes -r requirements.txt`. `build.sh`
  adds only the package itself (`pip install --no-deps --no-build-isolation
  -e .`, built with the locked `setuptools`), swaps
  `opencv-python` for the locked `opencv-python-headless` wheel, refuses an
  image with CUDA torch or a broken `cv2` import, and runs `npm ci` from the npm lock.

### Python dependency lock

`requirements.in` and `requirements-dev.in` are the inputs; `requirements.txt`
and `requirements-dev.txt` are hashed pip-tools locks for Linux x86_64 and
Python 3.11. Railway installs `requirements.txt`. `requirements-dev.in`
includes `requirements.in` (`-r`) and is constrained by `requirements.txt`
(`-c`), so `requirements-dev.txt` holds every runtime pin at the same version
plus the dev tools, and CI installs it alone. The `-r` line matters for
Dependabot: it recompiles only the locks that already name the package it
bumps, so a runtime bump regenerates both locks. Edit an input, then
regenerate the locks on Linux x86_64:

```bash
docker run --rm --platform linux/amd64 -v "$PWD":/src -w /src python:3.11 \
  sh -c 'make lock-tools lock PYTHON=python'
```

`make lock` refuses anything but Linux x86_64 and Python 3.11.5 or later 3.11:
some locked wheels, such as `soupsieve`, declare `Requires-Python >=3.11.5`.
`make lock-tools` installs the pip and pip-tools versions Dependabot runs, so
Dependabot's pip-compile updates produce the same file shape. The locks are
compiled with `--allow-unsafe` because torch requires `setuptools`; Dependabot
adds that flag whenever a lock pins `setuptools`, so both must agree.

The CI `dependency-lock` job runs `make lock-check` and fails when a lock's
pins or layout no longer match its input. It re-resolves against the live
indexes while keeping the existing pins, so it needs network access. It does
not verify hashes: pip-compile copies the committed hashes for unchanged pins,
so a hand-edited `--hash` line passes the check. A wrong hash still fails
closed when `--require-hashes` installs the lock.

**torch.** `requirements.in` pins torch and torchvision to matching `+cpu`
builds from the PyTorch CPU index on Linux and to the plain (already CPU-only)
wheels elsewhere. The PyPI Linux builds pull about 2.5GB of CUDA libraries, and
a mismatched pair breaks the `torchvision::nms` operator at runtime.
Dependabot ignores both; bump the pair by hand in `requirements.in`, keeping
the Linux and non-Linux pins on the same version, then run `make lock`.

**opencv.** `ultralytics` declares GUI `opencv-python`, so the runtime lock
pins it. It needs `libGL.so.1`, which headless servers lack, and both it and
`opencv-python-headless` own `cv2/`. `scripts/use-headless-opencv.sh`, run by
`build.sh` and the Linux `make backend-install`, uninstalls both and
reinstalls `opencv-python-headless` from its locked hashes.

On macOS, `make backend-install` installs the `.in` inputs with the lock
versions as hash-free constraints, because the `+cpu` wheels exist only for
Linux and Windows. That install is unhashed and keeps GUI opencv; CI is the
reference.

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
