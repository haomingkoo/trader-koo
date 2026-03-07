# Agent Working Rules (trader_koo)

These rules are mandatory for any coding/debugging task in this repo.

## 1) Requirements-First Workflow
- Before making changes, read:
  - `planning/requirements.md`
  - `planning/design.md`
  - `planning/task-list.md`
  - `planning/test-cases.md`
- At the start of each task, discover project docs and review relevant markdown context:
  - Run a markdown inventory (`rg --files -g '*.md'`) for this repo.
  - Prioritize `planning/*.md`, then any feature-specific docs (for example: `README.md`, `SECURITY.md`, module docs).
  - If the request touches an area with a local doc, read that doc before implementing.
- Confirm the requested change maps to at least one requirement/test case.
- If no requirement exists, add/update it in `planning/*.md` first.

## 2) Single-Source Consistency
- Do not introduce parallel scoring/narrative logic across surfaces.
- Report, chart commentary, and email must use one canonical recommendation payload where possible.
- Any intentional divergence must be documented in `planning/requirements.md`.

## 3) Security Guardrails
- Never commit secrets, tokens, API keys, passwords, or `.env` values.
- Never print secret values in logs, API responses, screenshots, or debug output.
- Keep `/api/admin/*` protected by `X-API-Key`; public endpoints stay read-only.
- Prefer fail-safe behavior: on LLM/provider failure, fall back to deterministic rule output.

## 4) Change Discipline
- For UI changes, verify desktop + mobile behavior.
- For report/email changes, verify readability and clipping risk.
- Update `planning/*.md` pass criteria/test cases when behavior changes.
- Run lightweight validation before commit (syntax/build checks).
