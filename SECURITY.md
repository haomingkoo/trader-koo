# Security Policy

## Supported Deployments
- `main` branch (production path) is supported.
- Feature branches are best-effort only until merged.

## Report a Vulnerability
- Do **not** open public issues for security findings.
- Email: `security@kooexperience.com` (or your designated operator inbox).
- Include:
  - Affected endpoint/path and environment (local/staging/prod)
  - Reproduction steps
  - Impact assessment
  - Suggested fix (if available)

Response targets:
- Initial acknowledgement: within 48 hours
- Triage decision: within 5 business days
- Patch target: based on severity and exploitability

## Secret Handling
- Never commit secrets to git (`.env`, API keys, SMTP passwords, admin keys).
- Use environment variables in Railway (or equivalent secret manager) only.
- Required rotations after any accidental exposure:
  - `TRADER_KOO_API_KEY`
  - `AZURE_OPENAI_API_KEY`
  - `TRADER_KOO_RESEND_API_KEY`
  - SMTP app password
  - Any CI/CD or deploy tokens

## Access Boundaries
- Public endpoints should remain read-only (`/api/health`, `/api/status`).
- All `/api/admin/*` endpoints must require `X-API-Key`.
- Never expose admin keys in frontend JavaScript, HTML, or browser storage.

## Logging and Data Safety
- Do not log raw credentials, auth headers, or full secrets.
- Sanitize error payloads before returning to clients.
- Keep subscriber data minimal and operationally necessary only.

## LLM Safety Guardrails
- LLM output is advisory and must be bounded + schema validated.
- On LLM failure, fallback to deterministic rule-based output.
- Degraded LLM state should trigger operator alerting (throttled).

## Secure Development Checklist
- Install and run pre-commit hooks before pushing.
- Run secret scanning on staged changes.
- Review `git diff` for accidental key/config leakage.
- Prefer private repo for full implementation; publish sanitized open-core only.

## Hardening Backlog (Must Keep Tracking)
- Admin auth boundary verification for every `/api/admin/*` route
- Secret exposure regression tests for status/report/log endpoints
- Rate limits for sensitive endpoints (subscription/admin)
- Optional IP allow-listing for admin APIs
