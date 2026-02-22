#!/usr/bin/env bash
# Daily market data scrape — runs after US market close (5am SGT = 9pm UTC)
# Cron schedule: 0 6 * * 1-6  (6am SGT, Mon–Sat, covers Mon–Fri closes)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
# Local dev: use venv. Railway: use nixpacks venv.
if [ -f "/opt/venv/bin/python" ]; then
    PYTHON="/opt/venv/bin/python"
elif [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
else
    PYTHON="$(command -v python3 || command -v python)"
fi
DB_PATH="${TRADER_KOO_DB_PATH:-/data/trader_koo.db}"
LOG_DIR="${TRADER_KOO_LOG_DIR:-/data/logs}"
RUN_LOG="$LOG_DIR/cron_daily.log"

mkdir -p "$LOG_DIR"

echo "========================================" >> "$RUN_LOG"
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [START] daily_update.sh" >> "$RUN_LOG"

# ── 1. Fetch prices + fundamentals for all S&P 500 tickers ──────────────────
#      Market context tickers (VIX, SPY, QQQ, ^DJI, ^TNX, SVIX) are always
#      appended automatically inside update_market_db.py regardless of --use-sp500
# On Railway the CWD is the repo root (all-assignments/), which is correct for
# "from trader_koo.X import Y" style imports inside update_market_db.py.
# On local dev the script is run from any directory; cd to repo root first.
REPO_ROOT="$(dirname "$PROJECT_DIR")"
if [ -d "$REPO_ROOT/trader_koo" ]; then
    cd "$REPO_ROOT"
fi

"$PYTHON" "$SCRIPT_DIR/update_market_db.py" \
    --use-sp500 \
    --price-lookback-days 5 \
    --fund-min-interval-hours 20 \
    --sleep-min 0.5 \
    --sleep-max 1.2 \
    --db-path "$DB_PATH" \
    --log-file "$LOG_DIR/update_market_db.log" \
    >> "$RUN_LOG" 2>&1

echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [DONE]  daily_update.sh" >> "$RUN_LOG"
