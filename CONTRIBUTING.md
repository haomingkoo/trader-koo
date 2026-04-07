# Contributing

## Local Setup

```bash
python -m venv .venv
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

That currently covers:

- Python bytecode sanity checks for `trader_koo/` and `tests/`
- Full backend test suite
- Frontend production build
- Pre-commit hygiene and secret scanning

## Pull Requests

- Describe the user-facing or operator-facing impact.
- Call out migrations, env var changes, or deployment sequencing when relevant.
- Include test evidence in the PR description.
- If a check is intentionally skipped, explain why.

## Security

- Do not open public issues for undisclosed vulnerabilities.
- Follow [SECURITY.md](SECURITY.md) for private reporting.
