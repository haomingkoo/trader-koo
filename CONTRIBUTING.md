# Contributing

## Local Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
make backend-install
make frontend-install
make hooks-install
cp .env.example .env
```

## Branching

- Branch from `main`.
- Keep PRs focused and small enough to review in one sitting.
- Use clear commit messages such as `fix(report): keep stale report visible`.

## Required Checks

Run the same baseline that GitHub Actions enforces before opening a PR:

```bash
make ci
```

Run it inside the activated `.venv`, or pass `PYTHON=.venv/bin/python`. System Python lacks hmmlearn. Install the hooks with `make hooks-install`, otherwise the repo-hygiene job catches what you cannot run locally. Unset `FINNHUB_API_KEY` before running pytest, and check production llm-health before concluding the LLM is broken from a stale local `.env` key.

That currently covers:

- Python bytecode sanity checks for `trader_koo/` and `tests/`
- Full backend test suite
- Frontend production build
- Pre-commit hygiene and secret scanning
- `make guardrails`: file size, bare-number and duplicate-code counts may only fall from `ops/guardrails_baseline.json`, plus the import-linter layer contract

For changes that move or restructure backend modules, run `make golden BASE=origin/main`. It snapshots the report, email and paper decisions on a copy of the local price database at a frozen clock with the network blocked, and prints `golden: identical` when the branch matches main. Rename a moved file's entry in `ops/guardrails_baseline.json` in the same PR, and lower the baseline with `python -m ops.guardrails baseline` after shrinking a file.

To inspect dependency freshness locally:

```bash
PYTHON=.venv/bin/python make dependency-status
```

That runs `pip check`, lists outdated Python packages, and reports outdated frontend npm packages. Dependabot also opens weekly PRs for root Python dependencies, Railway runtime dependencies, frontend npm packages, and GitHub Actions.

## Pull Requests

- Describe the user-facing or operator-facing impact.
- Call out migrations, env var changes, or deployment sequencing when relevant.
- Include test evidence in the PR description.
- If a check is intentionally skipped, explain why.

## Security

- Do not open public issues for undisclosed vulnerabilities.
- Follow [SECURITY.md](SECURITY.md) for private reporting.
