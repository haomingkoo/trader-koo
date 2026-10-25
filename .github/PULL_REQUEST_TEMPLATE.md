## Summary

- What changed?
- Why does it matter?

## Testing

- [ ] `make ci`
- [ ] Additional manual verification described below

## Risk

- Any migrations, env var changes, or rollout concerns?

## Checklist

- [ ] Scope is focused and reviewable
- [ ] Docs or examples updated when behavior changed
- [ ] Security, privacy, and operator impact considered

## Restructure proof (delete for PRs that do not move or change backend modules)

- [ ] Policy fingerprint `canonical_hash(config_snapshot(_build_config()))`: main `...` / branch `...`
- [ ] `make golden BASE=origin/main` prints `golden: identical`, or every listed diff is explained below
- [ ] Paper decisions identical (`paper.json` has no diff)
- [ ] Schema files untouched: `git diff --stat origin/main -- trader_koo/paper_trade/schema.py ops/PAPER_SCHEMA_V5_CONTRACT.md` is empty
- [ ] `make guardrails` reports 0 new violations (baseline entries renamed with moved files, only ever lowered)
- [ ] `make ci`
- [ ] `npm run lint --prefix trader_koo/frontend-v2` and `npm run build --prefix trader_koo/frontend-v2`
