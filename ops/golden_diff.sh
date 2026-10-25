#!/usr/bin/env bash
# Snapshot BASE (default origin/main) and the current tree on one DB copy and one
# clock, then compare. Usage: ops/golden_diff.sh [BASE]
# Env: PYTHON, GOLDEN_DB (price database to copy), GOLDEN_NOW (ISO8601 UTC).
set -euo pipefail

BASE="${1:-origin/main}"
PYTHON="${PYTHON:-python3}"
case "$PYTHON" in */*) PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")" ;; esac
HEAD_DIR="$(git rev-parse --show-toplevel)"
COMMON_ROOT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"

has_prices() { sqlite3 "file:$1?mode=ro" "SELECT 1 FROM price_daily LIMIT 1" 2>/dev/null | grep -q 1; }
if [ -z "${GOLDEN_DB:-}" ]; then
  for candidate in $(ls -t "$HEAD_DIR/trader_koo/data/trader_koo.db" "$HEAD_DIR/data/prod_paper_price.db" \
      "$COMMON_ROOT/trader_koo/data/trader_koo.db" "$COMMON_ROOT/data/prod_paper_price.db" 2>/dev/null); do
    if has_prices "$candidate"; then GOLDEN_DB="$candidate"; break; fi
  done
fi
[ -f "${GOLDEN_DB:-}" ] || { echo "golden: no local price database; set GOLDEN_DB" >&2; exit 2; }

TMP="$(mktemp -d)"
trap 'git worktree remove --force "$TMP/base" >/dev/null 2>&1 || true; rm -rf "$TMP"' EXIT
cp "$GOLDEN_DB" "$TMP/golden.db"

if [ -z "${GOLDEN_NOW:-}" ]; then
  GOLDEN_NOW="$("$PYTHON" -I -c "import sqlite3,sys; d=sqlite3.connect(sys.argv[1]).execute('select max(date) from price_daily').fetchone()[0]; print(d+'T22:00:00+00:00')" "$TMP/golden.db")"
fi

git worktree add -q --detach "$TMP/base" "$BASE"
# The harness may not exist on BASE yet, so run the head copy inside the base tree.
mkdir -p "$TMP/base/ops"
cp "$HEAD_DIR/ops/golden_report.py" "$TMP/base/ops/golden_report.py"

(cd "$TMP/base" && "$PYTHON" -m ops.golden_report snapshot --db "$TMP/golden.db" --out "$TMP/base_out" --now "$GOLDEN_NOW" >/dev/null)
(cd "$HEAD_DIR" && "$PYTHON" -m ops.golden_report snapshot --db "$TMP/golden.db" --out "$TMP/head_out" --now "$GOLDEN_NOW" >/dev/null)
(cd "$HEAD_DIR" && "$PYTHON" -m ops.golden_report compare "$TMP/base_out" "$TMP/head_out")
