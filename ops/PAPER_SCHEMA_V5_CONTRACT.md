# Paper schema v5 contract

`paper-schema-contract-v5.json` is the single source for the contract phase in
steps 8–17 of `ops/RELEASE_STATES.md`. This change specifies the target; it does
not implement a migration or verifier and does not permit writes or activation.

## Contract decisions

- Retain every current paper, shadow, campaign, audit, and report-lineage table.
  Rebuild only `paper_trades` and `paper_portfolio_snapshots`; extend
  `paper_trade_schema_meta` with the contract identity and fingerprint.
- Remove the two old global compatibility keys. Campaign-aware uniqueness is
  authoritative in v5.
- Remove implicit campaign selection: v5 trades and snapshots require callers
  to provide `campaign_id`.
- Add campaign and report-run FKs to `paper_trades`. Keep the other logical
  relations listed as intentionally undeclared to avoid unrelated immutable
  ledger rebuilds; the verifier must still reject their orphan rows.
- Preserve frozen v1 rows, nullable v1 report lineage, existing primary keys,
  annotations, audit facts, and the two legacy snapshot columns.
- Keep the report-admission contract as a required dependency. Its migration
  named `admission-ledger-contract-v5` is not the paper-schema v5 migration.
- Tighten the v1 trade update trigger after contraction: frozen v1 trades may
  no longer use the temporary lineage-backfill exception.

## Semantic fingerprint

The fingerprint is the SHA-256 of one canonical JSON object containing the
contract's table, index, trigger, FK, default, legacy-read, collision, and
integrity sections. It identifies the accepted semantics rather than SQLite
storage details, so root pages and generated autoindex names cannot create
false drift. A later verifier must check every item before reporting the frozen
fingerprint; returning the constant without those checks is not verification.
The exact-v5 SQL fixture hash is replaced by the named canonical placeholder
while computing the fingerprint, avoiding a hash cycle when that fixture stores
the fingerprint in its metadata row. The real fixture hash remains independently
frozen and tested in the contract.

## Frozen fixtures

The fixture manifest names the clean, legacy-production-like, collision,
malformed-object, interruption, and exact-v5 cases required of the later
migration and verifier PRs. Three hash-bound SQL fixtures construct the fresh
v4, 42-trade production-like legacy v4, and exact empty v5 schemas. Collision
cases contain executable minimal SQL, while interruption cases name the required
fault-injection points. Exact normalized table and trigger SQL hashes are part
of the semantic fingerprint. Production database files and sealed research
artifacts are not fixtures and must never be committed.

Every current unscoped reader is also named. Before activation, morning summary
and drift reads must select paper-v2 explicitly, and critic risk/edge reads must
require a campaign identifier. Removing global uniqueness must never mix frozen
v1 history into live v2 decisions.

## Deliberately deferred

`trader_koo.paper_trade.schema_v5_migration.migrate_paper_schema_v4_to_v5`
implements the first step as an explicit offline maintenance seam. It owns one
transaction, consumes the frozen hash-bound fixtures, and has no startup or
production-command caller. Its `already_v5_identity_only` result is deliberately
not schema verification; the activation interlock remains closed. Its v2
accounting gate rejects unreconciled rows and uses exact decimal arithmetic over
persisted values, with no hidden tolerance or rounding threshold.

Later, separate changes must implement, in order:

1. the exact verifier and semantic fingerprint checks;
2. a v4/v5 phase-aware initializer;
3. writer quiescence and restore drills;
4. copied-production rehearsal and deployment-state preservation;
5. separately audited write enablement and human activation.

Until all gates pass, `require_contracted_paper_schema` remains an unconditional
activation interlock and the dark-deploy workflow continues to pause writes.
