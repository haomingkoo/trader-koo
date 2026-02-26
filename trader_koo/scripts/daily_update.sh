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
INGEST_MAX_SECS_PER_TICKER="${TRADER_KOO_INGEST_MAX_SECS_PER_TICKER:-120}"
PRICE_TIMEOUT_SEC="${TRADER_KOO_PRICE_TIMEOUT_SEC:-25}"
PRICE_RETRY_ATTEMPTS="${TRADER_KOO_PRICE_RETRY_ATTEMPTS:-3}"

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

INGEST_T0=$(date +%s)
if "$PYTHON" "$SCRIPT_DIR/update_market_db.py" \
    --use-sp500 \
    --price-lookback-days 5 \
    --fund-min-interval-hours 20 \
    --max-seconds-per-ticker "$INGEST_MAX_SECS_PER_TICKER" \
    --price-timeout-sec "$PRICE_TIMEOUT_SEC" \
    --price-retry-attempts "$PRICE_RETRY_ATTEMPTS" \
    --sleep-min 0.5 \
    --sleep-max 1.2 \
    --db-path "$DB_PATH" \
    --log-file "$LOG_DIR/update_market_db.log" \
    >> "$RUN_LOG" 2>&1; then
    INGEST_RC=0
else
    INGEST_RC=$?
fi
INGEST_T1=$(date +%s)
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [INGEST] done rc=${INGEST_RC} sec=$((INGEST_T1-INGEST_T0))" >> "$RUN_LOG"
if [ "$INGEST_RC" -ne 0 ]; then
    echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [ERROR] ingest failed rc=${INGEST_RC}; aborting daily_update" >> "$RUN_LOG"
    exit "$INGEST_RC"
fi

# ── 2. YOLO pattern detection — daily pass only (Mon–Fri) ────────────────────
#      Weekly pass runs separately on Saturday via the scheduler in main.py.
YOLO_LOOKBACK_DAYS="${TRADER_KOO_YOLO_LOOKBACK_DAYS:-180}"
YOLO_SLEEP="${TRADER_KOO_YOLO_SLEEP:-0.05}"
YOLO_DPI="${TRADER_KOO_YOLO_DPI:-80}"
YOLO_FIG_W="${TRADER_KOO_YOLO_FIG_W:-10}"
YOLO_FIG_H="${TRADER_KOO_YOLO_FIG_H:-5}"
YOLO_IMGSZ="${TRADER_KOO_YOLO_IMGSZ:-640}"
YOLO_CONF="${TRADER_KOO_YOLO_CONF:-0.25}"
YOLO_IOU="${TRADER_KOO_YOLO_IOU:-0.45}"
YOLO_MAX_SECS_PER_TICKER="${TRADER_KOO_YOLO_MAX_SECS_PER_TICKER:-180}"
YOLO_PREFLIGHT_RC=0
if "$PYTHON" - <<'PY' >> "$RUN_LOG" 2>&1; then
import cv2, torch, ultralyticsplus
print(f"[YOLO] preflight ok cv2={cv2.__version__} torch={torch.__version__}")
PY
    echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [YOLO]  Preflight dependencies OK" >> "$RUN_LOG"
else
    YOLO_PREFLIGHT_RC=$?
    echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [YOLO]  Preflight failed rc=${YOLO_PREFLIGHT_RC} (non-fatal)" >> "$RUN_LOG"
fi

echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [YOLO]  Starting daily pattern detection (${YOLO_LOOKBACK_DAYS}d)..." >> "$RUN_LOG"
YOLO_T0=$(date +%s)
if [ "$YOLO_PREFLIGHT_RC" -eq 0 ] && "$PYTHON" "$SCRIPT_DIR/run_yolo_patterns.py" \
    --db-path "$DB_PATH" \
    --timeframe daily \
    --lookback-days "$YOLO_LOOKBACK_DAYS" \
    --only-new \
    --sleep "$YOLO_SLEEP" \
    --dpi "$YOLO_DPI" \
    --fig-w "$YOLO_FIG_W" \
    --fig-h "$YOLO_FIG_H" \
    --imgsz "$YOLO_IMGSZ" \
    --conf "$YOLO_CONF" \
    --iou "$YOLO_IOU" \
    --max-seconds-per-ticker "$YOLO_MAX_SECS_PER_TICKER" \
    >> "$RUN_LOG" 2>&1; then
    YOLO_RC=0
elif [ "$YOLO_PREFLIGHT_RC" -eq 0 ]; then
    YOLO_RC=$?
    echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [YOLO]  Pattern detection failed rc=${YOLO_RC} (non-fatal)" >> "$RUN_LOG"
else
    YOLO_RC="$YOLO_PREFLIGHT_RC"
    echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [YOLO]  Skipped due to preflight failure rc=${YOLO_RC} (non-fatal)" >> "$RUN_LOG"
fi
YOLO_T1=$(date +%s)
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [YOLO]  Daily pattern detection done. rc=${YOLO_RC} sec=$((YOLO_T1-YOLO_T0))" >> "$RUN_LOG"
"$PYTHON" - <<PY >> "$RUN_LOG" 2>&1 || true
import sqlite3
from pathlib import Path
db = Path("${DB_PATH}")
if not db.exists():
    print("[YOLO] events_table_check db_missing")
else:
    conn = sqlite3.connect(str(db))
    try:
        c = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='yolo_run_events' LIMIT 1"
        ).fetchone()
        if c:
            n = conn.execute("SELECT COUNT(*) FROM yolo_run_events").fetchone()[0]
            print(f"[YOLO] events_table_exists=1 rows={n}")
        else:
            print("[YOLO] events_table_exists=0 rows=0")
    finally:
        conn.close()
PY

# ── 3. Generate daily report (+ optional email) ───────────────────────────────
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [REPORT] Generating daily report..." >> "$RUN_LOG"
SEND_EMAIL_FLAG=""
if [ "${TRADER_KOO_AUTO_EMAIL:-}" = "1" ]; then
    SEND_EMAIL_FLAG="--send-email"
fi
REPORT_T0=$(date +%s)
if "$PYTHON" "$SCRIPT_DIR/generate_daily_report.py" \
    --db-path "$DB_PATH" \
    --out-dir "$REPORT_DIR" \
    --run-log "$RUN_LOG" \
    --tail-lines 120 \
    $SEND_EMAIL_FLAG \
    >> "$RUN_LOG" 2>&1; then
    REPORT_RC=0
else
    REPORT_RC=$?
    echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [REPORT] Failed to generate report rc=${REPORT_RC} (non-fatal)" >> "$RUN_LOG"
fi
REPORT_T1=$(date +%s)
echo "$(date '+%Y-%m-%dT%H:%M:%S%z') [REPORT] Done. rc=${REPORT_RC} sec=$((REPORT_T1-REPORT_T0))" >> "$RUN_LOG"

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
