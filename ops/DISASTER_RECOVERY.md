# Disaster recovery

This file covers losing or corrupting the production SQLite database. Image
rollback for a bad release is in `ops/RELEASE_STATES.md`; first-response triage
is in `ops/INCIDENT_RUNBOOK.md`.

> **PLACEHOLDER: off-site backup storage is not decided.**
> Every backup lives in `/data/backups` on the same Railway volume as the live
> database `/data/trader_koo.db`. Losing the volume loses the database and all
> of its backups together. Dark-deploy downloads a backup to a GitHub runner
> but keeps only its manifest, and the runner's copy is gone when the job ends.
> Do not use a GitHub release or artifact as off-site storage: the repository
> is public, and backups hold emails, tokens and IP addresses. Until the owner
> picks a private, encrypted destination, the only copies off the volume are
> the ones an operator downloaded by hand (see "Take an off-volume copy").

## What is protected

| Item | Value |
| --- | --- |
| Live database | `/data/trader_koo.db` (WAL mode) |
| Backup directory | `/data/backups`, same volume |
| Volume | Railway Hobby, 5 GB cap, cannot be resized from the CLI |
| Scheduled backup | daily at 03:00 UTC (`daily_backup` in `trader_koo/backend/services/scheduler.py`), after the 22:00 UTC pipeline and paper admission and before the 04:00 storage cleanup |
| Other backups | every dark-deploy run (`POST /api/admin/backups`) and every maintenance `quiesce-backup` |
| Integrity | each snapshot must pass `PRAGMA quick_check` before compression, or the backup fails and nothing is written. Compression writes `<name>.partial` and renames it only after fsync and hashing, so a failed write (for example a full disk) never shows up as a backup |
| Retention | `trader_koo/scripts/backup_db.py`: the 2 newest copies, the newest copy of each of the 3 most recent UTC days, and the newest copy of each of the 4 most recent ISO weeks, never more than `MAX_BACKUPS = 7` files |
| Visibility | public `/api/status` field `backup.latest_backup_utc` and `backup.latest_backup_size_bytes` |
| Alerting | Telegram on a failed scheduled backup; the hourly `prod-freshness.yml` run fails when the newest backup is older than 30 hours |

All three kinds of backup share the one budget of 7. The tiers make sure a burst
of same-day dark-deploy runs cannot evict every older restore point. Files that
do not match `trader_koo_backup_<UTC stamp>.db.gz` (for example
`pre_split_rebase*`) are never pruned and do not count toward the budget.
A `*.db.gz.partial` file left by a killed process is not pruned either; delete
it by hand once no backup is running.

History depth: at a steady daily cadence the oldest kept restore point is about
3 ISO weeks old. The weekly schedule this replaced kept 7 weekly copies when no
dark-deploy bursts happened, so the history is now shorter in exchange for a
restore point at most a day old.

Disk budget: on 2026-10-28 the database was 1.43 GB and each compressed backup
about 0.2 GB, so seven backups take about 1.4 GB. The nightly VACUUM needs about
the database size free. Before you raise `MAX_BACKUPS` or the tier sizes, check
`df -h /data` against database size plus backups plus VACUUM headroom. A
maintenance run's named backup is safe from pruning: while maintenance blocks
writers the scheduled backup skips and `POST /api/admin/backups` answers 503.

## Proposed targets (owner decision)

These are proposals, not commitments. Nothing has measured them.

| Scenario | Proposed RPO | Proposed RTO |
| --- | --- | --- |
| Corrupt or bad data, volume intact | 24 hours (one daily backup) | 2 hours |
| Volume lost, off-site copy exists | age of the newest off-site copy | 4 hours |
| Volume lost, no off-site copy (today) | everything not re-derivable is lost | market data reseeds in hours; paper campaign, report and decision lineage cannot be recovered |

Market prices, fundamentals and options can be fetched again. Paper trades,
campaign audit rows, published report runs and their fingerprints cannot be
fetched again, and they are what the research evidence depends on.

## Take an off-volume copy

This is the only off-volume protection that exists today. Run it from a trusted
machine whose disk is encrypted. Never put the key in shell history or logs;
`read -s` keeps it out of both.

```bash
BASE_URL=https://trader.kooexperience.com
read -rs ADMIN_API_KEY && export ADMIN_API_KEY
curl --fail --silent --show-error --header "X-API-Key: ${ADMIN_API_KEY}" \
  "${BASE_URL}/api/admin/backups" | jq '.backups[] | {name, size_bytes, modified_ts}'
NAME=trader_koo_backup_YYYYMMDDTHHMMSSffffffZ.db.gz   # pick one from the list
curl --fail --silent --show-error --header "X-API-Key: ${ADMIN_API_KEY}" \
  "${BASE_URL}/api/admin/backups/${NAME}" --output "${NAME}"
shasum -a 256 "${NAME}"
```

`POST /api/admin/backups` makes a fresh copy first. It uses the shared budget of
7, so do not call it repeatedly.

## Restore drill on a copy

Run this monthly, and after any change to backup or migration code. It never
touches production.

1. Download one named backup as shown above. Record its name, size and sha256.
2. Check the bytes and open the copy read-only:

   ```bash
   test "$(wc -c < "${NAME}" | tr -d ' ')" = "EXPECTED_SIZE_BYTES"
   gunzip --keep --stdout "${NAME}" > drill.db
   sqlite3 "file:drill.db?mode=ro" "PRAGMA integrity_check;"   # must print ok
   sqlite3 "file:drill.db?mode=ro" \
     "SELECT MAX(date) FROM price_daily; SELECT COUNT(*) FROM paper_trades;"
   ```

3. Run the migration and replay checks that dark-deploy runs, against the copy:

   ```bash
   python -m trader_koo.scripts.release_evidence \
     --source-db "${NAME}" --output-dir drill-evidence
   ```

4. Optionally boot the app on the copy and open the Report and Paper Trades
   pages. Keep paper writes disabled:

   ```bash
   TRADER_KOO_DB_PATH="$PWD/drill.db" TRADER_KOO_PAPER_TRADE_ENABLED=0 \
     uvicorn trader_koo.backend.main:app --port 8010
   ```

5. Record the date, backup name, sha256, how long each step took, and the
   result. Delete `drill.db` and the downloaded copy when you are done, or move
   them to the approved off-site store.

## Corrupt database, volume intact

The maintenance CLI's `restore-backup` installs only the backup that
`quiesce-backup` took for the same maintenance run, after it preserves the
failed live file as `trader_koo.db.pre_restore_<run_id>`. It is meant for
recovery during a schema maintenance run, not for picking an older scheduled
backup. The commands and the request body are in `ops/RELEASE_STATES.md`
(steps 12 to 15). The restore path is:

```text
/opt/venv/bin/python -m trader_koo.scripts.paper_schema_maintenance status --run-id RUN_ID
/opt/venv/bin/python -m trader_koo.scripts.paper_schema_maintenance decide --run-id RUN_ID --decision restore --reason "REASON"
/opt/venv/bin/python -m trader_koo.scripts.paper_schema_maintenance restore-backup --run-id RUN_ID
/opt/venv/bin/python -m trader_koo.scripts.paper_schema_maintenance verify-resolution --run-id RUN_ID
```

No tool installs an arbitrary older scheduled backup. If the live file is
corrupt and the maintenance snapshot is not usable, the owner must decide on a
manual install. That procedure has not been rehearsed. It needs paper writes
paused (`TRADER_KOO_PAPER_TRADE_ENABLED=0`, a human configuration change), the
failed file and its `-wal` and `-shm` sidecars preserved under new names, and
the chosen backup decompressed, integrity-checked and moved into place while no
process holds the database open.

## Full volume loss

1. **Confirm.** `/api/health` shows `db_exists: false`, or `railway ssh "ls -la /data"`
   shows an empty or missing volume.
2. **Stop new writes.** When `start.sh` finds no database it creates an empty
   schema and starts a 365-day price seed, so the service comes back up with no
   paper or report history. The owner pauses paper writes
   (`TRADER_KOO_PAPER_TRADE_ENABLED=0`) and records the incident. Do not
   activate or reset a campaign as part of recovery. That is a separate,
   authenticated human decision.
3. **Find the newest off-volume copy.** [PLACEHOLDER: off-site store not
   decided.] Today the only candidates are copies an operator downloaded by
   hand. Check each one's sha256 against its dark-deploy manifest
   (`production-backup-manifest.json` in the `dark-deploy-evidence-<sha>`
   artifact, kept 30 days) or your own record.
4. **Prove the copy** with steps 2 and 3 of the restore drill.
5. **Put it on the new volume.** No upload endpoint exists, and streaming a
   file over `railway ssh` has not been verified. This step needs an owner
   decision together with the off-site store.
6. **Verify.** `/api/health`, `/api/status` (latest price date, report
   freshness, backup block) and `/api/paper-trades/summary`
   (`campaign_health.status` and `write_state`) must match the copy's state.
   The next hourly freshness run must pass.
7. **If no copy exists,** let the seed rebuild the market data. Record which
   paper campaign, report run and decision lineage was lost. Leave the
   campaign inactive until the owner decides what to do next.
