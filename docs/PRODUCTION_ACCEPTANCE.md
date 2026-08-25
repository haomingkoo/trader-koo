# Production acceptance

A release is not accepted because it builds, returns HTTP 200, or deploys. It is accepted only when the public journey is useful and truthful.

Release-blocking checks:

- Every primary navigation page renders meaningful content or a precise failure state.
- Partial data remains useful; one missing artifact cannot hide independent, valid data.
- Draft, paused, frozen, historical, invalid, and active states are explicit and cannot be confused.
- Historical or unreconciled metrics are never presented as current campaign performance.
- Missing data is not relabelled as a market state, successful run, or completed model analysis.
- Admin-only tools do not appear in public navigation.
- Failed experiments remain visible, but the default view prefers usable evidence.
- The header and page body cannot make contradictory claims about the same state.
- Browser acceptance runs locally and again against the exact deployed SHA.

The executable contract lives in `trader_koo/frontend-v2/e2e/public-journey-acceptance.spec.ts` and `trader_koo/frontend-v2/production-e2e/release.spec.ts`.
