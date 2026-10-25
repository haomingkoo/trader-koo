"""Build a trimmed, read-only copy of the production database for `make golden`.

Run inside the Railway container (stdlib only):
    python3 ops/golden_db_export.py /tmp/golden.db.gz
It opens /data/trader_koo.db read-only, copies every table's schema, copies the
rows the report and paper paths read (recent prices, fundamentals, options,
news, patterns, report and campaign state), leaves logs, delivery records,
LLM traces and side-product time series empty, and writes a gzip file. Copy it
out, gunzip it outside the repo (never commit it) and pass it as GOLDEN_DB.
"""
from __future__ import annotations

import gzip
import os
import shutil
import sqlite3
import sys

SOURCE = "file:/data/trader_koo.db?mode=ro"
SCHEMA_ONLY = {
    "audit_logs", "admin_audit_log", "admin_managed_services", "email_report_deliveries",
    "email_subscriber_events", "email_subscribers", "telegram_alerts", "llm_call_traces",
    "llm_contributions", "llm_health_events", "llm_health_state", "llm_outcome_links",
    "llm_run_graphs", "llm_token_usage", "hyperliquid_alert_state", "hyperliquid_counter_signals",
    "hyperliquid_fills", "hyperliquid_reload_events", "hyperliquid_snapshots", "hyperliquid_wallets",
    "crypto_bars", "crypto_correlation_snapshots", "crypto_fear_greed", "crypto_funding_rates",
    "crypto_long_short_ratio", "polymarket_snapshots", "spike_alert_cooldown", "macro_alert_cooldowns",
    "options_digest_delivery", "bot_versions", "ui_usage_sessions", "yolo_run_events",
    "setup_feedback", "external_data_cache",
}
PRICE_HISTORY_START = "2025-06-01"  # enough history for 200-day indicators on the latest session
RECENT_DAYS = {"options_iv": 14, "rss_news_snapshots": 14, "news_sentiment_cache": 14, "finviz_fundamentals": 30}
DATE_COLUMNS = ("snapshot_date", "snapshot_ts", "as_of_date", "date", "created_ts", "fetched_ts", "updated_ts")


def _row_filter(src: sqlite3.Connection, table: str) -> str:
    if table == "price_daily":
        return f" WHERE date >= '{PRICE_HISTORY_START}'"
    if table not in RECENT_DAYS:
        return ""
    cols = [r[1] for r in src.execute(f'PRAGMA table_info("{table}")')]
    col = next(c for c in DATE_COLUMNS if c in cols)
    latest = src.execute(f'SELECT max("{col}") FROM "{table}"').fetchone()[0]
    cutoff = src.execute("SELECT date(?, ?)", (str(latest)[:10], f"-{RECENT_DAYS[table]} days")).fetchone()[0]
    return f" WHERE \"{col}\" >= '{cutoff}'"


def export(out_gz: str) -> None:
    work = out_gz.removesuffix(".gz")
    if os.path.exists(work):
        os.remove(work)
    src = sqlite3.connect(SOURCE, uri=True)
    objects = src.execute(
        "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    dst = sqlite3.connect(work)
    for kind, _name, sql in objects:
        if kind == "table":
            dst.execute(sql)
    dst.commit()
    dst.close()
    src.execute("ATTACH ? AS golden", (work,))
    for kind, name, _sql in objects:
        if kind == "table" and name not in SCHEMA_ONLY:
            src.execute(f'INSERT INTO golden."{name}" SELECT * FROM main."{name}"{_row_filter(src, name)}')
    src.commit()
    src.execute("DETACH golden")
    src.close()
    dst = sqlite3.connect(work)
    for kind, _name, sql in objects:
        if kind in ("index", "trigger", "view"):
            dst.execute(sql)
    dst.commit()
    dst.execute("VACUUM")
    dst.close()
    with open(work, "rb") as raw, gzip.open(out_gz, "wb", 6) as packed:
        shutil.copyfileobj(raw, packed)
    os.remove(work)


if __name__ == "__main__":
    export(sys.argv[1] if len(sys.argv) > 1 else "/tmp/golden.db.gz")
