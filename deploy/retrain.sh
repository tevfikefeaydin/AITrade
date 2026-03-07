#!/usr/bin/env bash
# =============================================================================
# AITradew.AI - Weekly Model Retrain
# =============================================================================
# Called by systemd timer (aitradew-retrain.timer) weekly on Sunday 02:00 UTC
# Re-trains model on latest data and runs backtest
# =============================================================================
set -euo pipefail

APP_DIR="/home/aitradew/app"
VENV="${APP_DIR}/.venv/bin/python"
LOG_PREFIX="[$(date -u '+%Y-%m-%d %H:%M:%S UTC')]"

cd "$APP_DIR"

trap 'echo "${LOG_PREFIX} ERROR: Command failed at line $LINENO" >&2; exit 1' ERR

echo "${LOG_PREFIX} Haftalik model egitimi basliyor..."

# 1. Ensure data is fresh (download + build)
echo "${LOG_PREFIX} Veri kontrol ediliyor..."
$VENV -m src.cli download || { echo "${LOG_PREFIX} ERROR: Data download failed" >&2; exit 1; }
$VENV -m src.cli build || { echo "${LOG_PREFIX} ERROR: Feature build failed" >&2; exit 1; }

# 2. Backup current models (timestamp includes hour to avoid same-day collision)
BACKUP_DIR="${APP_DIR}/models/backup_$(date -u '+%Y%m%d_%H%M%S')"
if [[ -d "${APP_DIR}/models" ]] && ls "${APP_DIR}/models"/*.pkl &>/dev/null 2>&1; then
    echo "${LOG_PREFIX} Mevcut modeller yedekleniyor -> ${BACKUP_DIR}"
    mkdir -p "$BACKUP_DIR"
    cp "${APP_DIR}/models"/*.pkl "$BACKUP_DIR/" 2>/dev/null || true
fi

# 3. Train new models
echo "${LOG_PREFIX} Model egitimi basliyor..."
$VENV -m src.cli train || { echo "${LOG_PREFIX} ERROR: Model training failed" >&2; exit 1; }

# 4. Verify model files were created
for symbol in BTCUSDT ETHUSDT; do
    MODEL_FILE="${APP_DIR}/models/${symbol}_model.pkl"
    if [[ ! -f "$MODEL_FILE" ]]; then
        echo "${LOG_PREFIX} ERROR: Model not found: ${MODEL_FILE}" >&2
        if [[ -d "$BACKUP_DIR" ]]; then
            echo "${LOG_PREFIX} Rolling back to previous models..."
            cp "$BACKUP_DIR"/*.pkl "${APP_DIR}/models/" 2>/dev/null || true
        fi
        exit 1
    fi
done
echo "${LOG_PREFIX} Model dosyalari dogrulandi."

# 5. Run backtest on new models
echo "${LOG_PREFIX} Backtest basliyor..."
$VENV -m src.cli backtest || { echo "${LOG_PREFIX} ERROR: Backtest failed" >&2; exit 1; }

# 6. Restart paper trader to load new models
echo "${LOG_PREFIX} Paper trader yeniden baslatiliyor..."
sudo systemctl restart aitradew-paper.service 2>/dev/null || true

# 7. Cleanup old backups (keep last 4 weeks)
echo "${LOG_PREFIX} Eski yedekler temizleniyor..."
find "${APP_DIR}/models/" -maxdepth 1 -name "backup_*" -type d -mtime +28 -exec rm -rf {} + 2>/dev/null || true

echo "${LOG_PREFIX} Haftalik egitim tamamlandi."
