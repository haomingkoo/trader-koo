#!/usr/bin/env bash
# Railway build script — replaces the inline mega-command in railway.toml.
# Each step is isolated with clear log output for easier debugging.
set -euo pipefail

log() { echo "==> [build] $*"; }

# ── 1. Install PyTorch (CPU-only) ──────────────────────────────────────
log "Installing PyTorch CPU..."
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu

# ── 2. Install package in editable mode ─────────────────────────────────
# Nixpacks installs the canonical root requirements.txt before this script.
log "Installing trader_koo package..."
pip install -e .

# ── 3. Force opencv-headless (remove full opencv if present) ────────────
log "Switching to opencv-python-headless..."
pip uninstall -y opencv-python 2>/dev/null || true
pip install --force-reinstall opencv-python-headless==4.10.0.84

# ── 4. Preflight import check ──────────────────────────────────────────
log "Running preflight import check..."
python -c "import cv2, torch, ultralyticsplus; print('preflight cv2=', cv2.__version__, 'torch=', torch.__version__)"

# ── 5. Install Node.js via nvm ──────────────────────────────────────────
log "Installing Node.js 22 via NodeSource..."
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [ -s "$NVM_DIR/nvm.sh" ]; then
    # nvm already available (nixpacks may provide it)
    . "$NVM_DIR/nvm.sh"
    nvm install 22
    nvm use 22
elif command -v node >/dev/null 2>&1; then
    log "Node.js already available: $(node --version)"
else
    # Fallback: NodeSource setup script (requires root, typical in Railway)
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt-get install -y nodejs
fi
log "Node $(node --version) / npm $(npm --version)"

# ── 6. Build React frontend ────────────────────────────────────────────
log "Building React frontend..."
cd trader_koo/frontend-v2
npm ci
npm run build
cd ../..

log "Build complete."
