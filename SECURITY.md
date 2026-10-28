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
- Without Cloudflare's ranges in `TRADER_KOO_TRUSTED_PROXY_CIDRS`, every
  request through Cloudflare keys on a Cloudflare edge IP. That cannot be
  spoofed, but it is coarse: users of one edge share a rate-limit bucket and the
  admin lockout. Set the variable in Railway to Cloudflare's published list
  (<https://www.cloudflare.com/ips/>) and refresh it when that list changes:

  ```
  TRADER_KOO_TRUSTED_PROXY_CIDRS=173.245.48.0/20,103.21.244.0/22,103.22.200.0/22,103.31.4.0/22,141.101.64.0/18,108.162.192.0/18,190.93.240.0/20,188.114.96.0/20,197.234.240.0/22,198.41.128.0/17,162.158.0.0/15,104.16.0.0/13,104.24.0.0/14,172.64.0.0/13,131.0.72.0/22,2400:cb00::/32,2606:4700::/32,2803:f800::/32,2405:b500::/32,2405:8100::/32,2a06:98c0::/29,2c0f:f248::/32
  ```

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
