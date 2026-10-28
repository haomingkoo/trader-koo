#!/usr/bin/env bash
# Entrypoint for Railway deployment.
# Runs a read-only boot preflight, bootstraps an empty DB only when allowed,
# then execs uvicorn. In production (RAILWAY_ENVIRONMENT or
# RAILWAY_VOLUME_MOUNT_PATH set) a missing DB is fatal unless
# TRADER_KOO_ALLOW_EMPTY_BOOTSTRAP=1, and the DB must live on the mounted volume.
set -euo pipefail

# Use the nixpacks venv explicitly — bash (non-login) doesn't source /root/.profile.
# The overrides exist for tests and local runs.
PYTHON="${TRADER_KOO_START_PYTHON:-/opt/venv/bin/python}"
UVICORN="${TRADER_KOO_START_UVICORN:-/opt/venv/bin/uvicorn}"

# Nixpacks exposes some compiled Python dependencies, but the runtime shell can
# start with LD_LIBRARY_PATH=/usr/lib only. Add the Nix runtime libs NumPy needs.
RUNTIME_LIB_DIRS=()
for dir in /nix/store/*-gcc-*-lib/lib /nix/store/*-zlib-*/lib; do
    if [ -d "$dir" ]; then
        RUNTIME_LIB_DIRS+=("$dir")
    fi
done
if [ "${#RUNTIME_LIB_DIRS[@]}" -gt 0 ]; then
    RUNTIME_LIB_PATH="$(IFS=:; echo "${RUNTIME_LIB_DIRS[*]}")"
    export LD_LIBRARY_PATH="${RUNTIME_LIB_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

DB_PATH="${TRADER_KOO_DB_PATH:-/data/trader_koo.db}"
LOG_DIR="${TRADER_KOO_LOG_DIR:-/data/logs}"

# Fails closed (non-zero) on a missing volume, a DB outside it, an unreadable
# DB, or a failed PRAGMA quick_check. Nothing is written before this passes.
# An active maintenance run skips the live DB checks ("maintenance") so a
# corrupt or mid-restore DB can still be recovered over Railway SSH.
if ! BOOT_STATE="$("$PYTHON" -m trader_koo.scripts.start_preflight --db-path "$DB_PATH")"; then
    echo "[start.sh] Boot preflight failed; refusing to start." >&2
    exit 1
fi

if [ "$BOOT_STATE" = "bootstrap" ]; then
    echo "[start.sh] DB not found at $DB_PATH — creating empty DB and starting server..."
    mkdir -p "$(dirname "$DB_PATH")"
    mkdir -p "$LOG_DIR"

    # Create empty database with schema (fast, <1 second)
    "$PYTHON" -c "
import sqlite3
import sys
from trader_koo.scripts.update_market_db import ensure_schema

conn = sqlite3.connect(sys.argv[1])
ensure_schema(conn)
conn.close()
print('[start.sh] Empty database created with schema')
" "$DB_PATH"

    # The seed is unsupervised: nothing restarts it, it holds no writer lease,
    # and it can overlap scheduler writes. A crashed seed is not retried on the
    # next boot because the DB then exists; check the log and rerun by hand.
    echo "[start.sh] Starting background data seed..."
    nohup "$PYTHON" trader_koo/scripts/update_market_db.py \
        --use-sp500 \
        --price-lookback-days 365 \
        --sleep-min 0.3 \
        --sleep-max 0.8 \
        --db-path "$DB_PATH" \
        > "$LOG_DIR/seed.log" 2>&1 &

    echo "[start.sh] Background seed started (PID $!). Check $LOG_DIR/seed.log for progress."
elif [ "$BOOT_STATE" = "maintenance" ]; then
    echo "[start.sh] Maintenance run active; starting maintenance-only server without touching the DB."
elif [ "$BOOT_STATE" != "ready" ]; then
    echo "[start.sh] Unexpected preflight result '$BOOT_STATE'; refusing to start." >&2
    exit 1
fi

echo "[start.sh] Starting uvicorn server..."
# Railway drains for 120s before SIGKILL. Capping connection drain at 90s
# leaves time for lifespan shutdown to stop writer children.
exec "$UVICORN" trader_koo.backend.main:app --host 0.0.0.0 --port "${PORT:-8000}" \
    --timeout-graceful-shutdown 90
