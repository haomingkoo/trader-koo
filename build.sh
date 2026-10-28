#!/usr/bin/env bash
# Railway build script — replaces the inline mega-command in railway.toml.
# Each step is isolated with clear log output for easier debugging.
set -euo pipefail

log() { echo "==> [build] $*"; }

# ── 1. Install package in editable mode ─────────────────────────────────
# The nixpacks install phase (nixpacks.toml) already installed the hashed
# requirements.txt lock, including CPU-only torch. --no-deps keeps this step
# from resolving anything outside that lock, and --no-build-isolation builds
# with the locked setuptools instead of fetching the latest one from PyPI.
log "Installing trader_koo package..."
pip install --no-deps --no-build-isolation -e .

# ── 2. Keep only opencv-python-headless ────────────────────────────────
# ultralytics pulls in GUI opencv-python, which needs libGL.so.1. Both wheels
# own cv2/, so remove both and reinstall the headless one from its locked hashes.
log "Switching to opencv-python-headless..."
pip uninstall -y opencv-python opencv-python-headless
awk '/^opencv-python-headless==/ {found = 1} found {print} found && !/\\$/ {exit}' \
    requirements.txt > /tmp/opencv-headless.txt
pip install --no-deps --require-hashes -r /tmp/opencv-headless.txt

# ── 3. Preflight import check ──────────────────────────────────────────
log "Running preflight import check..."
if pip show opencv-python >/dev/null 2>&1; then
    log "opencv-python is still installed; it needs libGL and breaks cv2 here"
    exit 1
fi
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
