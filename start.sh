#!/usr/bin/env bash
# Entrypoint for Railway deployment.
# On first run (empty volume), seeds the DB before starting the server.
set -euo pipefail

# Use the nixpacks venv explicitly — bash (non-login) doesn't source /root/.profile
PYTHON=/opt/venv/bin/python
UVICORN=/opt/venv/bin/uvicorn

DB_PATH="${TRADER_KOO_DB_PATH:-/data/trader_koo.db}"

if [ ! -f "$DB_PATH" ]; then
    echo "[start.sh] DB not found at $DB_PATH — seeding initial data..."
    mkdir -p "$(dirname "$DB_PATH")"
    "$PYTHON" trader_koo/scripts/update_market_db.py \
        --use-sp500 \
        --price-lookback-days 365 \
        --sleep-min 0.3 \
        --sleep-max 0.8 \
        --db-path "$DB_PATH"
    echo "[start.sh] Seeding complete."
else
    echo "[start.sh] DB found at $DB_PATH — skipping seed."
fi

exec "$UVICORN" trader_koo.backend.main:app --host 0.0.0.0 --port "${PORT:-8000}"
