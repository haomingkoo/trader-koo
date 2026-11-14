#!/usr/bin/env bash
# Python half of the Railway build, run by build.sh after Nixpacks installs
# requirements.txt. CI runs this same file in a clean venv (production-runtime
# job), so the image and the smoke test cannot drift apart.
set -euo pipefail

log() { echo "==> [build] $*"; }

# ── 1. Install PyTorch (CPU-only) ──────────────────────────────────────
log "Installing PyTorch CPU..."
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu

# ── 2. Install package in editable mode ─────────────────────────────────
log "Installing trader_koo package..."
pip install -e .

# ── 3. Force opencv-headless (remove full opencv if present) ────────────
log "Switching to opencv-python-headless..."
pip uninstall -y opencv-python 2>/dev/null || true
pip install --force-reinstall opencv-python-headless==4.10.0.84

# ── 4. Preflight import check ──────────────────────────────────────────
# Imports every trader_koo module and the YOLO stack in this exact environment,
# so a missing runtime dependency fails the build instead of every nightly run.
log "Running preflight import check..."
python ops/runtime_smoke.py --imports-only
