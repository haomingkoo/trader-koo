#!/usr/bin/env bash
# Railway build script — replaces the inline mega-command in railway.toml.
# Each step is isolated with clear log output for easier debugging.
set -euo pipefail

log() { echo "==> [build] $*"; }

# ── 1-4. Python runtime: torch CPU, package, opencv-headless, preflight ─
# Nixpacks installs the canonical root requirements.txt before this script.
bash ops/install_python_runtime.sh

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
