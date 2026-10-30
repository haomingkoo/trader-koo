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
  - `TRADER_KOO_RESEND_API_KEY`
  - SMTP app password
  - Any CI/CD or deploy tokens

## Access Boundaries
- Public endpoints should remain read-only (`/api/health`, `/api/status`).
- All `/api/admin/*` endpoints must require `X-API-Key`.
- Never expose admin keys in frontend JavaScript, HTML, or browser storage.
  The Guide page's admin panel keeps the key in React state until reload.
  Every SPA load deletes the `trader_koo_admin_key` value that older builds
  left in `localStorage` (`trader_koo/frontend-v2/src/main.tsx`).
- Public `/api/status` omits the admin auth header, the key-configured flag
  and the admin action paths. They return only with
  `TRADER_KOO_EXPOSE_STATUS_INTERNAL=1`.
- `/openapi.json`, `/docs` and `/redoc` exist only with
  `TRADER_KOO_DOCS_ENABLED=1`. Leave it unset in production.

## Browser Hardening
- Every HTTP response carries the headers in
  `trader_koo/middleware/security_headers.py`. A header set by a route wins.
- The SPA's main policy is the `<meta>` CSP in
  `trader_koo/frontend-v2/index.html`. Its `script-src` has no
  `'unsafe-inline'` because the Vite build emits only external module
  scripts. Keep `static.cloudflareinsights.com` in it for the Cloudflare
  beacon.

## Client IP
Rate limits, the admin lockout and the audit log key on
`trader_koo/backend/utils.py:client_ip`. Forwarded headers are read only when
the direct peer is a trusted proxy (loopback, private, link-local, or inside
`TRADER_KOO_TRUSTED_PROXY_CIDRS`). The client is then the rightmost
`X-Forwarded-For`/`X-Real-IP` hop that is not a trusted proxy.
`CF-Connecting-IP` is never read.

Leave `TRADER_KOO_TRUSTED_PROXY_CIDRS` unset in production. The
`*.up.railway.app` origin answers without Cloudflare, and a Cloudflare Worker
can call it from a Cloudflare address with a forged `X-Forwarded-For`, so
trusting Cloudflare's ranges lets anyone rotate past the lockout and rate
limits. Unset, each Cloudflare edge IP is one coarse key that cannot be
spoofed. Trust those ranges only after direct origin access is closed.

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
- CI fails on known Python dependency vulnerabilities (`pip-audit` in the
  `security-contracts` job) and on high-severity npm advisories. Each ignored
  advisory carries its reason in `.github/workflows/ci.yml`.
- CodeQL scans Python, TypeScript and the workflows on pushes and pull
  requests to `main`, and weekly (`.github/workflows/codeql.yml`).
