#!/usr/bin/env bash
# =============================================================================
# AITradew.AI - Deploy to VPS
# =============================================================================
# Run from your LOCAL machine to upload project files to VPS.
#
# Usage:
#   bash deploy/deploy.sh user@your-vps-ip
#   bash deploy/deploy.sh root@123.456.789.0
#   bash deploy/deploy.sh root@123.456.789.0 2222    # Custom SSH port
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[x]${NC} $*"; exit 1; }

# --- Args ---
VPS_HOST="${1:-}"
SSH_PORT="${2:-22}"

[[ -z "$VPS_HOST" ]] && err "Kullanim: bash deploy/deploy.sh user@vps-ip [port]"

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REMOTE_DIR="/home/aitradew/app"

log "Proje dizini: ${PROJECT_DIR}"
log "Hedef: ${VPS_HOST}:${REMOTE_DIR}"

# --- Test SSH connection ---
log "SSH baglantisi test ediliyor..."
ssh -p "$SSH_PORT" -o ConnectTimeout=10 "$VPS_HOST" "echo 'Baglanti basarili'" || err "SSH baglantisi basarisiz!"

# --- Ensure remote directory exists ---
log "Uzak dizin hazirlaniyor..."
ssh -p "$SSH_PORT" "$VPS_HOST" "mkdir -p ${REMOTE_DIR}"

# --- Sync project files (exclude data, models, outputs, venv) ---
log "Dosyalar senkronize ediliyor..."
rsync -avz --progress \
    -e "ssh -p ${SSH_PORT}" \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='data/' \
    --exclude='models/' \
    --exclude='outputs/' \
    --exclude='.idea/' \
    --exclude='.vscode/' \
    --exclude='*.egg-info/' \
    --exclude='.pytest_cache/' \
    "${PROJECT_DIR}/" \
    "${VPS_HOST}:${REMOTE_DIR}/"

# --- Fix ownership ---
log "Dosya izinleri ayarlaniyor..."
ssh -p "$SSH_PORT" "$VPS_HOST" "
    chown -R aitradew:aitradew ${REMOTE_DIR} 2>/dev/null || true
    chmod +x ${REMOTE_DIR}/deploy/*.sh
"

# --- Done ---
echo ""
echo "============================================================"
echo -e "${GREEN} DEPLOY TAMAMLANDI${NC}"
echo "============================================================"
echo ""
echo "  Sonraki adimlar (VPS uzerinde):"
echo ""
echo "  1. Ilk kurulum (bir kere):"
echo "     ssh -p ${SSH_PORT} ${VPS_HOST}"
echo "     sudo bash ${REMOTE_DIR}/deploy/setup_vps.sh"
echo "     sudo -u aitradew bash ${REMOTE_DIR}/deploy/install.sh"
echo ""
echo "  2. Servisleri aktifle (bir kere):"
echo "     sudo bash ${REMOTE_DIR}/deploy/enable_services.sh"
echo ""
echo "  3. Durum kontrolu:"
echo "     sudo bash ${REMOTE_DIR}/deploy/health_check.sh"
echo ""
echo "  4. Guncelleme sonrasi (tekrar deploy ettikten sonra):"
echo "     sudo systemctl restart aitradew-paper"
echo "============================================================"
