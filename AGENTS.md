# Agent Working Rules (trader_koo)

These rules are mandatory for any coding/debugging task in this repo.

## 1) Requirements-First, Low-Token Workflow
- Default doc load order (minimum required):
  - `planning/INDEX.md`
  - `planning/contracts/workflow.md`
  - one relevant requirement module (`planning/modules/requirements/*.md`)
  - one relevant task module (`planning/modules/tasks/*.md`)
  - one relevant test module (`planning/modules/tests/*.md`)
- Do **not** load all rollup docs (`planning/requirements.md`, `planning/design.md`, `planning/task-list.md`, `planning/test-cases.md`) by default.
- Use rollup docs only when:
  - scope is cross-cutting or ambiguous
  - closing/sign-off requires full consistency checks
  - module docs are missing required detail
- Do **not** run markdown-wide inventory by default.
  - Only run `rg --files -g '*.md'` if scope is unclear or docs are missing.
- Confirm each change maps to at least one requirement ID and one test case ID.
- If missing, update module docs first, then implement code.
- Keep active context in `memory/current-task.md`:
  - scope
  - requirement IDs
  - test IDs
  - files touched
  - open risks/blockers

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
- Update relevant module docs and `planning/CHANGELOG.md` when behavior changes.
- Run lightweight validation before commit (syntax/build checks).
