#!/usr/bin/env bash
# Swap GUI opencv-python for the locked headless wheel; see DEPLOYMENT.md, "Python dependency lock".
set -euo pipefail

python="${1:-python}"
lock="$(mktemp)"
trap 'rm -f "$lock"' EXIT

awk '/^opencv-python-headless==/ {found = 1} found {print} found && !/\\$/ {exit}' \
    requirements.txt > "$lock"
"$python" -m pip uninstall -y opencv-python opencv-python-headless
"$python" -m pip install --no-deps --require-hashes -r "$lock"
