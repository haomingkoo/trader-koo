#!/usr/bin/env bash
# Entrypoint for Railway deployment.
# Creates empty DB if needed, starts server immediately, then seeds in background.
set -euo pipefail

# Use the nixpacks venv explicitly — bash (non-login) doesn't source /root/.profile
PYTHON=/opt/venv/bin/python
UVICORN=/opt/venv/bin/uvicorn

DB_PATH="${TRADER_KOO_DB_PATH:-/data/trader_koo.db}"

if [ ! -f "$DB_PATH" ]; then
    echo "[start.sh] DB not found at $DB_PATH — creating empty DB and starting server..."
    mkdir -p "$(dirname "$DB_PATH")"
    mkdir -p /data/logs
    
    # Create empty database with schema (fast, <1 second)
    "$PYTHON" -c "
import sqlite3
from pathlib import Path
from trader_koo.db.schema import ensure_ohlcv_schema

db_path = Path('$DB_PATH')
conn = sqlite3.connect(str(db_path))
ensure_ohlcv_schema(conn)
conn.close()
print('[start.sh] Empty database created with schema')
"
    
    # Start background seeding (non-blocking)
    echo "[start.sh] Starting background data seed..."
    nohup "$PYTHON" trader_koo/scripts/update_market_db.py \
        --use-sp500 \
        --price-lookback-days 365 \
        --sleep-min 0.3 \
        --sleep-max 0.8 \
        --db-path "$DB_PATH" \
        > /data/logs/seed.log 2>&1 &
    
    echo "[start.sh] Background seed started (PID $!). Check /data/logs/seed.log for progress."
else
    # Database exists - check if it has data
    RECORD_COUNT=$("$PYTHON" -c "
import sqlite3
try:
    conn = sqlite3.connect('$DB_PATH')
    cursor = conn.execute('SELECT COUNT(*) FROM price_daily')
    count = cursor.fetchone()[0]
    conn.close()
    print(count)
except:
    print(0)
" 2>/dev/null || echo "0")
    
    echo "[start.sh] DB found at $DB_PATH with $RECORD_COUNT price records — skipping seed."
fi

echo "[start.sh] Starting uvicorn server..."
exec "$UVICORN" trader_koo.backend.main:app --host 0.0.0.0 --port "${PORT:-8000}"
