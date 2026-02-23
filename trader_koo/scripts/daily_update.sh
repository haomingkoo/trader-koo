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
REPORT_DIR="${TRADER_KOO_REPORT_DIR:-/data/reports}"

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

# ── 2. YOLO pattern detection — daily pass only (Mon–Fri) ────────────────────
#      Weekly pass runs separately on Saturday via the scheduler in main.py.
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [YOLO]  Starting daily pattern detection (180d)..." >> "$RUN_LOG"
"$PYTHON" "$SCRIPT_DIR/run_yolo_patterns.py" \
    --db-path "$DB_PATH" \
    --timeframe daily \
    --lookback-days 180 \
    --only-new \
    --sleep 0.05 \
    >> "$RUN_LOG" 2>&1 || echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [YOLO]  Pattern detection failed (non-fatal)" >> "$RUN_LOG"
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [YOLO]  Daily pattern detection done." >> "$RUN_LOG"

# ── 3. Generate daily report (+ optional email) ───────────────────────────────
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [REPORT] Generating daily report..." >> "$RUN_LOG"
SEND_EMAIL_FLAG=""
if [ "${TRADER_KOO_AUTO_EMAIL:-}" = "1" ]; then
    SEND_EMAIL_FLAG="--send-email"
fi
"$PYTHON" "$SCRIPT_DIR/generate_daily_report.py" \
    --db-path "$DB_PATH" \
    --out-dir "$REPORT_DIR" \
    --run-log "$RUN_LOG" \
    --tail-lines 120 \
    $SEND_EMAIL_FLAG \
    >> "$RUN_LOG" 2>&1 || echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [REPORT] Failed to generate report (non-fatal)" >> "$RUN_LOG"
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [REPORT] Done." >> "$RUN_LOG"

echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [DONE]  daily_update.sh" >> "$RUN_LOG"

# ── 4. Housekeeping — keep last 30 report archives, cap log size ──────────────
# Keep only the 30 most recent timestamped report files (latest.* are always kept)
ls -t "$REPORT_DIR"/daily_report_2*.json 2>/dev/null | tail -n +31 | xargs rm -f 2>/dev/null || true
ls -t "$REPORT_DIR"/daily_report_2*.md   2>/dev/null | tail -n +31 | xargs rm -f 2>/dev/null || true
# Truncate cron log to last 5 MB if it exceeds 10 MB
for f in "$LOG_DIR"/*.log; do
    [ -f "$f" ] || continue
    size=$(wc -c < "$f" 2>/dev/null || echo 0)
    if [ "$size" -gt 10485760 ]; then
        tail -c 5242880 "$f" > "$f.tmp" && mv "$f.tmp" "$f"
    fi
done
