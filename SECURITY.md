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
  The Guide page's admin panel keeps the key in React state until reload. It
  also deletes the `trader_koo_admin_key` value that older builds left in
  `localStorage`.
- Public `/api/status` omits the admin auth header, the key-configured flag
  and the admin action paths. They return only with
  `TRADER_KOO_EXPOSE_STATUS_INTERNAL=1`.
- `/openapi.json`, `/docs` and `/redoc` exist only with
  `TRADER_KOO_DOCS_ENABLED=1`. Leave it unset in production.

## Browser Hardening
- Every HTTP response carries `X-Content-Type-Options: nosniff`,
  `X-Frame-Options: DENY` and `Content-Security-Policy: frame-ancestors 'none'`.
  It also sends `Referrer-Policy: strict-origin-when-cross-origin` and a
  `Permissions-Policy` that disables camera, microphone, geolocation, payment
  and USB. A header set by a route wins
  (`trader_koo/middleware/security_headers.py`).
- The SPA's main policy is the `<meta>` CSP in
  `trader_koo/frontend-v2/index.html`. Its `script-src` has no
  `'unsafe-inline'` because the Vite build emits only external module
  scripts. Keep `static.cloudflareinsights.com` in it for the Cloudflare
  beacon.

## Client IP and Proxy Headers
Rate limits, the admin lockout and the audit log key on the client IP from
`trader_koo/backend/utils.py:client_ip`. Production traffic runs
client -> Cloudflare -> Railway edge -> app. The `*.up.railway.app` origin
also answers directly without Cloudflare.

- With `TRADER_KOO_TRUST_PROXY_HEADERS=auto` (the default), forwarded headers
  are read only when the direct peer is a trusted proxy. Trusted means a
  loopback, private or link-local address, or one inside
  `TRADER_KOO_TRUSTED_PROXY_CIDRS`. `1` always reads them and `0` never does.
- The client is the rightmost `X-Forwarded-For` hop that is not a trusted
  proxy. The Railway edge's `X-Real-IP` closes the chain. Entries further left
  are client-supplied and are ignored.
- `CF-Connecting-IP` is never read: anyone who reaches the Railway origin
  directly can set it.
- Leave `TRADER_KOO_TRUSTED_PROXY_CIDRS` unset in production. Every request
  through Cloudflare then keys on a Cloudflare edge IP. That cannot be
  spoofed, but it is coarse: users of one edge share a rate-limit bucket and
  the admin lockout.
- Do not put Cloudflare's ranges (<https://www.cloudflare.com/ips/>) in
  `TRADER_KOO_TRUSTED_PROXY_CIDRS` while the `*.up.railway.app` origin answers
  directly. A Cloudflare Worker can fetch that origin with any
  `X-Forwarded-For` it likes, and it egresses from a Cloudflare address
  (`2a06:98c0:3600::103`, inside `2a06:98c0::/29`). With those ranges trusted,
  the forged entry becomes the client IP and rotating it bypasses the admin
  lockout and the rate limits
  (`tests/test_backend_utils.py::test_cloudflare_cidrs_let_a_worker_forge_the_client_while_origin_is_open`).
- Per-client keys behind Cloudflare need an owner change first: close direct
  origin access (remove the Railway domain, or accept only traffic that
  proves it came through this zone, such as authenticated origin pulls or a
  tunnel). Then re-check that a Worker request cannot reach the app with a
  forged header before trusting any Cloudflare range.

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
