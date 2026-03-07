#!/usr/bin/env bash
# =============================================================================
# AITradew.AI - Daily Data Update
# =============================================================================
# Called by systemd timer (aitradew-update.timer) daily at 00:05 UTC
# Downloads new klines and rebuilds features
# =============================================================================
set -euo pipefail

APP_DIR="/home/aitradew/app"
VENV="${APP_DIR}/.venv/bin/python"
LOG_PREFIX="[$(date -u '+%Y-%m-%d %H:%M:%S UTC')]"

cd "$APP_DIR"

trap 'echo "${LOG_PREFIX} ERROR: Command failed at line $LINENO" >&2; exit 1' ERR

echo "${LOG_PREFIX} Gunluk veri guncelleme basliyor..."

# 1. Download new data (incremental - only missing bars)
echo "${LOG_PREFIX} Veri indiriliyor..."
$VENV -m src.cli download || { echo "${LOG_PREFIX} ERROR: Data download failed" >&2; exit 1; }

# 2. Rebuild features with new data
echo "${LOG_PREFIX} Feature build basliyor..."
$VENV -m src.cli build || { echo "${LOG_PREFIX} ERROR: Feature build failed" >&2; exit 1; }

echo "${LOG_PREFIX} Gunluk guncelleme tamamlandi."
