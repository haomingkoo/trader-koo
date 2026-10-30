#!/usr/bin/env bash
# Railway build script — replaces the inline mega-command in railway.toml.
# Each step is isolated with clear log output for easier debugging.
set -euo pipefail

log() { echo "==> [build] $*"; }

# ── 1. Install package in editable mode ─────────────────────────────────
# nixpacks already installed the hashed lock; see DEPLOYMENT.md, "Python dependency lock".
log "Installing trader_koo package..."
pip install --no-deps --no-build-isolation -e .

# ── 2. Keep only opencv-python-headless ────────────────────────────────
# ultralytics pulls in GUI opencv-python, which needs libGL; see DEPLOYMENT.md, "Python dependency lock".
log "Switching to opencv-python-headless..."
bash scripts/use-headless-opencv.sh python

# ── 3. Preflight import check ──────────────────────────────────────────
log "Running preflight import check..."
python -c "import cv2, torch, ultralytics, huggingface_hub; assert torch.version.cuda is None, 'CUDA torch installed'; print('preflight cv2=', cv2.__version__, 'torch=', torch.__version__)"

# ── 4. Check Node.js against the pinned version ─────────────────────────
# Node comes from the nixpacks setup phase (nixpacks.toml). Fail rather than
# build the frontend with a different Node than CI tested.
expected_node="v$(tr -d '[:space:]' < trader_koo/frontend-v2/.nvmrc)"
actual_node="$(node --version)"
if [ "$actual_node" != "$expected_node" ]; then
    log "Node $actual_node does not match trader_koo/frontend-v2/.nvmrc ($expected_node)"
    exit 1
fi
log "Node $actual_node / npm $(npm --version)"

# ── 5. Build React frontend ────────────────────────────────────────────
log "Building React frontend..."
cd trader_koo/frontend-v2
npm ci
npm run build
cd ../..

log "Build complete."
