#!/usr/bin/env bash
# =============================================================================
# AITradew.AI - Project Installation
# =============================================================================
# Run as root after setup_vps.sh: bash /root/app/deploy/install.sh
# Copies files to aitradew user, creates venv, installs deps, runs pipeline.
# =============================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[x]${NC} $*"; exit 1; }

# --- Paths ---
SRC_DIR="/root/app"
APP_DIR="/home/aitradew/app"
LOG_DIR="/home/aitradew/logs"

# --- Pre-checks ---
[[ ! -d "$SRC_DIR" ]] && err "Kaynak dizin bulunamadi: ${SRC_DIR}"
[[ ! -f "${SRC_DIR}/requirements.txt" ]] && err "requirements.txt bulunamadi"
id aitradew &>/dev/null || err "aitradew kullanicisi bulunamadi. Once setup_vps.sh calistirin."

# =============================================================================
# 1. Copy project files to aitradew user
# =============================================================================
log "Proje dosyalari kopyalaniyor..."
mkdir -p "$APP_DIR"
mkdir -p "$LOG_DIR"
cp -r "${SRC_DIR}/"* "$APP_DIR/"
chown -R aitradew:aitradew /home/aitradew
chmod +x "${APP_DIR}/deploy/"*.sh
log "Dosyalar kopyalandi: ${APP_DIR}"

# =============================================================================
# 2. Python Virtual Environment (as aitradew user)
# =============================================================================
log "Virtual environment olusturuluyor..."

# Find Python
if command -v python3.12 &>/dev/null; then
    PYTHON=python3.12
elif command -v python3 &>/dev/null; then
    PYTHON=python3
else
    err "Python3 bulunamadi!"
fi

log "Python: $($PYTHON --version)"

VENV_DIR="${APP_DIR}/.venv"

sudo -u aitradew $PYTHON -m venv "$VENV_DIR"
log "Venv olusturuldu: ${VENV_DIR}"

# =============================================================================
# 3. Install dependencies
# =============================================================================
log "Bagimliliklar yukleniyor..."
sudo -u aitradew bash -c "source ${VENV_DIR}/bin/activate && pip install --upgrade pip setuptools wheel -q && pip install -r ${APP_DIR}/requirements.txt -q"
log "Tum bagimliliklar yuklendi"

# =============================================================================
# 4. Create data directories
# =============================================================================
log "Veri dizinleri olusturuluyor..."
sudo -u aitradew mkdir -p "${APP_DIR}/data"
sudo -u aitradew mkdir -p "${APP_DIR}/models"
sudo -u aitradew mkdir -p "${APP_DIR}/outputs"

# =============================================================================
# 5. Verify installation
# =============================================================================
log "Kurulum dogrulanıyor..."

sudo -u aitradew bash -c "source ${VENV_DIR}/bin/activate && python -c \"import pandas, numpy, lightgbm, sklearn, requests, websockets, tqdm, pyarrow; print('Tum paketler basariyla import edildi')\""

# Run tests
log "Testler calistiriliyor..."
sudo -u aitradew bash -c "cd ${APP_DIR} && source ${VENV_DIR}/bin/activate && python -m pytest tests/ -v --tb=short" || warn "Bazi testler basarisiz oldu - devam ediliyor"

# =============================================================================
# 6. Initial data download + pipeline
# =============================================================================
echo ""
read -p "Verileri indirip tam pipeline calistirmak ister misiniz? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Veri indirme basliyor (bu biraz zaman alabilir)..."
    sudo -u aitradew bash -c "cd ${APP_DIR} && source ${VENV_DIR}/bin/activate && python -m src.cli download" 2>&1 | tee "${LOG_DIR}/initial_download.log"

    log "Build basliyor..."
    sudo -u aitradew bash -c "cd ${APP_DIR} && source ${VENV_DIR}/bin/activate && python -m src.cli build" 2>&1 | tee "${LOG_DIR}/initial_build.log"

    log "Train basliyor..."
    sudo -u aitradew bash -c "cd ${APP_DIR} && source ${VENV_DIR}/bin/activate && python -m src.cli train" 2>&1 | tee "${LOG_DIR}/initial_train.log"

    log "Backtest basliyor..."
    sudo -u aitradew bash -c "cd ${APP_DIR} && source ${VENV_DIR}/bin/activate && python -m src.cli backtest" 2>&1 | tee "${LOG_DIR}/initial_backtest.log"

    log "Tum pipeline tamamlandi!"
else
    warn "Pipeline atlandi. Daha sonra manuel calistirin."
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo -e "${GREEN} KURULUM TAMAMLANDI${NC}"
echo "============================================================"
echo ""
echo "  Uygulama:  ${APP_DIR}"
echo "  Venv:      ${VENV_DIR}"
echo "  Python:    $($PYTHON --version)"
echo "  Loglar:    ${LOG_DIR}"
echo ""
echo "  Sonraki adim:"
echo "    bash ${APP_DIR}/deploy/enable_services.sh"
echo "    cp ${APP_DIR}/deploy/aitradew-sudoers /etc/sudoers.d/aitradew"
echo "============================================================"
