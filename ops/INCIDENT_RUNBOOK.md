# Incident runbook

First response for production. Data recovery is in `ops/DISASTER_RECOVERY.md`
and the release state model is in `ops/RELEASE_STATES.md`. Never print the
admin API key or the Railway token. Read it with `read -rs`.

## Where alerts arrive

| Source | Channel | Fires on |
| --- | --- | --- |
| `prod-freshness.yml` (hourly, outside the container) | GitHub Actions failure email to the workflow owner | service unreachable or a non-2xx response, stale price date, report not fresh, newest backup older than 30 hours, deployed `git_sha` more than 3 days behind `main`, last attempted dark-deploy failed |
| In-app scheduler | Telegram (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) | nightly pipeline failed, scheduled backup failed, storage pressure, crypto data stale, a kooexperience.com site down |
| `dark-deploy.yml` | GitHub Actions failure email | a release check failed; the workflow has already rolled back |

Pipeline and backup alerts repeat at most once per 6 hours per kind. The
cooldown is in memory, so a restart can repeat an alert. If Telegram is not
configured the scheduler logs `Telegram not configured` and only the hourly
GitHub check remains. In-app alerts stop when the container stops.

## First triage

```bash
BASE_URL=https://trader.kooexperience.com
UA=trader-koo-freshness-monitor/1.0
curl -sS -A "$UA" -w '\nHTTP %{http_code}\n' "$BASE_URL/api/health"
curl -sS -A "$UA" "$BASE_URL/api/status" | jq '{ok, operational_warnings, latest_data, freshness, backup, pipeline: .pipeline | {active, stage, last_completed_stage, last_completed_status, last_completed_ts}}'
curl -sS -A "$UA" "$BASE_URL/api/release" | jq .
curl -sS -A "$UA" "$BASE_URL/api/paper-trades/summary" | jq '.campaign_health | {campaign_id, status, write_state, healthy, health_reasons}'
python ops/check_prod_freshness.py --base-url "$BASE_URL" --repo haomingkoo/trader-koo
gh run list --workflow prod-freshness.yml --limit 5
gh run list --workflow dark-deploy.yml --limit 5
railway logs --service SERVICE_NAME | tail -n 200
railway ssh "df -h /data; ls -la /data/backups; tail -n 60 /data/logs/cron_daily.log"
```

The site firewall answers 403 to the default Python-urllib user agent. The
`-A` flag sends the same agent the monitor uses. Every `railway` command above
only reads.

## Rollback a bad release

Dark-deploy rolls back by itself when a release check fails. To roll back by
hand, pick the last `SUCCESS` deployment from
`railway deployment list --json --service SERVICE_NAME` and use **Rollback** on
it in the Railway dashboard. That is the same `deploymentRollback` mutation as
`ops/railway-rollback.graphql`. Afterwards `/api/release` must show the
previous `git_sha` and `/api/paper-trades/summary` must show the pre-deploy
campaign status and `write_state`. Image rollback never touches the volume.
Check the database compatibility matrix in `ops/RELEASE_STATES.md` before
rolling back across a schema change.

## Disk full or storage pressure

1. `railway ssh "df -h /data; ls -la /data/backups; ls -la /data/*.db*"`.
2. VACUUM needs free space about equal to the database size. When free space
   is below that, the 04:00 cleanup cannot reclaim space.
3. The safest space to free is extra same-day backups. Scheduled retention
   already keeps at most 7 copies across days and weeks. Keep the newest copy
   and any backup a maintenance run names. Download a copy first
   (`ops/DISASTER_RECOVERY.md`).
4. Do not empty `finviz_fundamentals.raw_json`, `report_runs.decisions_json`,
   `report_run_decisions`, `paper_shadow_decisions` or
   `paper_candidate_decisions`. They look reclaimable, but readers and
   lineage fingerprints depend on them.
5. The 5 GB Hobby cap cannot be raised from the CLI.

## Stale data or report

1. `/api/status`: check `latest_data.price_date`, `freshness.report_fresh` and
   `pipeline`. A pipeline that is still running is not stale.
2. Read the cron log tail (see triage) for `[INGEST]`, `[YOLO]` and `[REPORT]`
   return codes.
3. Re-run with the admin key. A provider outage needs `full`. A failed report
   step alone needs `report`:

   ```bash
   read -rs ADMIN_API_KEY && export ADMIN_API_KEY
   curl -sS -X POST -H "X-API-Key: ${ADMIN_API_KEY}" \
     "$BASE_URL/api/admin/trigger-update?mode=report"
   ```

4. The 12:00 UTC pre-open watchdog also recovers a missing report once newer
   prices are stored.

## Failed backup

1. Read the Telegram text or `grep BACKUP /data/logs/cron_daily.log`. A
   `quick_check failed` message means the live database has corruption. Treat
   it as a data incident and stop making new backups, because the older
   copies are the good ones.
2. `Skipped — maintenance intent blocks writers` is expected during a
   maintenance run. Check `GET /api/admin/maintenance/status`.
3. For a disk or permission error, free space first, then make one backup with
   `POST /api/admin/backups` and confirm `/api/status` `backup.latest_backup_utc`
   moved forward.
