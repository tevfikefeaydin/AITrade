#!/usr/bin/env bash
# =============================================================================
# AITradew.AI - Enable Systemd Services
# =============================================================================
# Run as root: sudo bash enable_services.sh
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'
NC='\033[0m'
log() { echo -e "${GREEN}[+]${NC} $*"; }

[[ $EUID -ne 0 ]] && { echo "Root olarak calistirin: sudo bash enable_services.sh"; exit 1; }

DEPLOY_DIR="/home/aitradew/app/deploy"

# Copy service files
log "Servis dosyalari kopyalaniyor..."
cp "${DEPLOY_DIR}/aitradew-paper.service"   /etc/systemd/system/
cp "${DEPLOY_DIR}/aitradew-update.service"  /etc/systemd/system/
cp "${DEPLOY_DIR}/aitradew-update.timer"    /etc/systemd/system/
cp "${DEPLOY_DIR}/aitradew-retrain.service" /etc/systemd/system/
cp "${DEPLOY_DIR}/aitradew-retrain.timer"   /etc/systemd/system/

# Reload systemd
log "Systemd yeniden yukleniyor..."
systemctl daemon-reload

# Enable and start paper trader
log "Paper trader servisi baslatiliyor..."
systemctl enable aitradew-paper.service
systemctl start aitradew-paper.service

# Enable timers
log "Zamanlanmis gorevler aktiflestirilyor..."
systemctl enable aitradew-update.timer
systemctl start aitradew-update.timer

systemctl enable aitradew-retrain.timer
systemctl start aitradew-retrain.timer

# Status
echo ""
echo "============================================================"
echo -e "${GREEN} SERVISLER AKTIF${NC}"
echo "============================================================"
echo ""
echo "Paper Trader:"
systemctl status aitradew-paper.service --no-pager -l | head -5
echo ""
echo "Timers:"
systemctl list-timers aitradew-* --no-pager
echo ""
echo "Yararli komutlar:"
echo "  sudo systemctl status aitradew-paper     # Durum"
echo "  sudo systemctl restart aitradew-paper     # Yeniden baslat"
echo "  sudo systemctl stop aitradew-paper        # Durdur"
echo "  sudo journalctl -u aitradew-paper -f      # Canli log"
echo "  tail -f /home/aitradew/logs/paper_trader.log  # Log dosyasi"
echo "  sudo systemctl list-timers aitradew-*     # Timer durumu"
echo "============================================================"
