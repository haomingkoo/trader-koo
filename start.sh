#!/usr/bin/env bash
# Entrypoint for Railway deployment.
# On first run (empty volume), seeds the DB before starting the server.
set -euo pipefail

# Make trader_koo importable without installing it as a package
export PYTHONPATH="/app:${PYTHONPATH:-}"

DB_PATH="${TRADER_KOO_DB_PATH:-trader_koo/data/trader_koo.db}"

if [ ! -f "$DB_PATH" ]; then
    echo "[start.sh] DB not found at $DB_PATH — seeding initial data..."
    mkdir -p "$(dirname "$DB_PATH")"
    python trader_koo/scripts/update_market_db.py \
        --tickers "SPY,QQQ,AAPL,NVDA,MSFT,TSLA,AMZN,META,GOOGL,BRK-B" \
        --price-lookback-days 365 \
        --sleep-min 0.3 \
        --sleep-max 0.8
    echo "[start.sh] Seeding complete."
else
    echo "[start.sh] DB found at $DB_PATH — skipping seed."
fi

exec uvicorn trader_koo.backend.main:app --host 0.0.0.0 --port "${PORT:-8000}"
