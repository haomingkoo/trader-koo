# Research price database (2010+ daily history)

Production keeps daily prices from 2018 only, and its volume has no room for
more. Longer history lives in a separate SQLite file on a workstation. The
production database is never written by this tool.

Default location: `~/dev/.research/trader_koo_research.db`. Keep it outside the
repository and never commit it.

## Build or refresh

1. Export a read-only copy of the production database to the workstation (for
   example with `trader_koo/scripts/sync_prod_data.py`, or use the golden copy).
2. Run the backfill. `--db-path` is required and has no default.

```bash
python -m trader_koo.scripts.backfill_price_history \
  --db-path ~/dev/.research/trader_koo_research.db \
  --source-db /path/to/prod_export.db \
  --report-json ~/dev/.research/backfill_report.json
```

`--source-db` is opened read-only and its `price_daily`,
`price_corporate_actions` and `price_series_revisions` rows are copied in with
`INSERT OR IGNORE`, so rows already in the research file win. The run then adds
earlier dates for every ticker. A full run takes about 25 minutes at the
default 1 second pacing.

To refresh later, repeat the command with a newer export. The state file
`<db-path>.backfill_state.json` lets an interrupted run resume. Use
`--retry-terminal` to retry tickers that were settled earlier, `--tickers` for a
subset and `--coverage-only` to probe yfinance without writing.

Benchmark and context symbols are not index members, so the membership file never
lists them. `--context-symbols SPY,QQQ,^VIX,^IRX` adds them as members from
`--start`. They use the same fetch path and the same no-modification checks as every
other ticker. To backfill only these, also pass `--tickers SPY,QQQ,^VIX,^IRX`.

## One-command build

```bash
make research-db RESEARCH_DB=~/dev/.research/trader_koo_research.db SOURCE_DB=/path/to/export.db
```

Runs, in order and one process at a time: price backfill (members plus
`SPY,QQQ,^VIX,^IRX`), membership load, SEC fundamentals load (with
`--predecessor XOM=34088`), earnings backfill. It stops at the first stage that
exits non-zero, logs `database is locked`, a traceback or an SQLite error, or
reports price tickers with status `error`. Then it runs the data checks
(`ops/research_db_checks.py`) and prints the checklist. A failing check fails
the build. Report-only items (off-scale share counts, vendor splits missing from
`price_corporate_actions`, flagged rows by reason, multi-class issuers) are
listed with real counts. Stage output and the report files go to
`<db stem>_build/` beside the database.

Earnings use Finnhub and yfinance. Without `FINNHUB_API_KEY` the build refuses
to start unless you pass `EARNINGS_SOURCE=yfinance`. For a test build on a
subset into a temporary file, add `TICKERS=AAPL,XOM,BRK-B`. To re-run the checks
alone: `python -m ops.research_db_checks --db-path <file>`.

## What it does and refuses to do

- Universe: S&P 500 members at any time since 2010-01-01 from
  `fja05680/sp500` `sp500_ticker_start_end.csv`, pinned to commit
  `a2430f2af0c79ddf0748e91de11bdeb1616ab5a7` (sha256 checked). The point-in-time
  `sp500_membership` work reads the same file. Symbols map to `price_daily`
  format (`BRK.B` becomes `BRK-B`). `FB` is merged into `META`.
- Fetches only dates before each ticker's earliest stored row, through the same
  functions as the daily ingest, so rows are raw split-adjusted with the same
  basis fields. Splits and dividends are written to `price_corporate_actions`.
- Existing rows are never changed. A write that would change any existing price
  or action row (for example a genuine one-day move above 45 percent that the
  price contract flags) is rolled back and the ticker is reported as
  `skipped_would_modify_existing_rows`.
- Departed tickers are fetched only through their last membership spell plus
  10 days.
- A departed symbol that trades today but has no bars in its spell is skipped as
  `skipped_recycled_symbol`. A series that starts more than 45 days after the
  member joined is skipped as `skipped_history_starts_after_spell`. A membership
  gap above a year drops the earlier spells (the symbol may now belong to
  another company), except for the re-added companies in
  `trader_koo/price_backfill/config.py`.
- Chart patterns (`yolo_patterns`) are kept when the price revision is resealed.

## Known limits

- Yahoo serves no bars for about a quarter of the symbols in the list
  (`no_data`). Many are renames (for example `ANTM` to `ELV`) that need a
  reviewed mapping before their history can be joined.
- Backfilled rows can be `unresolved` when a day moves by 45 percent or more
  without a declared split. They stay out of research until reviewed.
- Prices are not dividend-adjusted. Dividends are recorded in
  `price_corporate_actions`, not applied.
