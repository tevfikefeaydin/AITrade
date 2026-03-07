#!/usr/bin/env bash
# =============================================================================
# AITradew.AI - VPS System Setup (Ubuntu 24.04 LTS)
# =============================================================================
# Run as root: bash setup_vps.sh
# This script prepares a fresh Hostinger VPS for the trading pipeline.
# =============================================================================
set -euo pipefail

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[x]${NC} $*"; exit 1; }

# --- Pre-checks ---
[[ $EUID -ne 0 ]] && err "Bu scripti root olarak calistirin: sudo bash setup_vps.sh"

log "AITradew.AI VPS kurulumu basliyor..."
log "OS: $(lsb_release -ds 2>/dev/null || cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2)"

# =============================================================================
# 1. System update
# =============================================================================
log "Sistem guncelleniyor..."
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
    software-properties-common \
    curl wget git unzip htop tmux \
    build-essential libffi-dev libssl-dev \
    ca-certificates gnupg

# =============================================================================
# 2. Python 3.12 (Ubuntu 24.04 has 3.12 by default)
# =============================================================================
log "Python kontrol ediliyor..."
if command -v python3.12 &>/dev/null; then
    PYTHON=python3.12
    log "Python 3.12 zaten yuklu: $(python3.12 --version)"
elif command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    if [[ "$PY_VER" == "3.12" || "$PY_VER" == "3.11" || "$PY_VER" == "3.13" ]]; then
        PYTHON=python3
        log "Python $PY_VER kullanilacak"
    else
        log "Python 3.12 kuruluyor..."
        add-apt-repository -y ppa:deadsnakes/ppa
        apt-get update -qq
        apt-get install -y -qq python3.12 python3.12-venv python3.12-dev
        PYTHON=python3.12
    fi
else
    log "Python 3.12 kuruluyor..."
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y -qq python3.12 python3.12-venv python3.12-dev
    PYTHON=python3.12
fi

# Ensure pip and venv
apt-get install -y -qq python3-pip python3-venv 2>/dev/null || true

# =============================================================================
# 3. Create dedicated user: aitradew
# =============================================================================
APP_USER="aitradew"
APP_HOME="/home/${APP_USER}"
APP_DIR="${APP_HOME}/app"

if id "$APP_USER" &>/dev/null; then
    log "Kullanici '${APP_USER}' zaten mevcut"
else
    log "Kullanici '${APP_USER}' olusturuluyor..."
    useradd -m -s /bin/bash "$APP_USER"
    log "Kullanici olusturuldu: ${APP_USER}"
fi

# =============================================================================
# 4. Firewall (UFW)
# =============================================================================
log "Firewall ayarlaniyor..."
apt-get install -y -qq ufw

ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
# Binance WebSocket (outbound only - already allowed by default outgoing)
# No inbound ports needed for paper trading

ufw --force enable
log "Firewall aktif: SSH izinli, diger girisler engelli"

# =============================================================================
# 5. Fail2Ban
# =============================================================================
log "Fail2Ban kuruluyor..."
apt-get install -y -qq fail2ban

cat > /etc/fail2ban/jail.local << 'JAIL'
[DEFAULT]
bantime  = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true
port    = ssh
filter  = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime  = 7200
JAIL

systemctl enable fail2ban
systemctl restart fail2ban
log "Fail2Ban aktif: SSH brute-force korumasi"

# =============================================================================
# 6. Swap (Hostinger VPS genelde az RAM ile gelir)
# =============================================================================
SWAP_SIZE="2G"

if swapon --show | grep -q '/swapfile'; then
    log "Swap zaten mevcut"
else
    log "${SWAP_SIZE} swap olusturuluyor..."
    fallocate -l $SWAP_SIZE /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile

    # Persist across reboots
    if ! grep -q '/swapfile' /etc/fstab; then
        echo '/swapfile none swap sw 0 0' >> /etc/fstab
    fi

    # Optimize swap usage
    sysctl vm.swappiness=10
    echo 'vm.swappiness=10' >> /etc/sysctl.conf

    log "Swap aktif: ${SWAP_SIZE}"
fi

# =============================================================================
# 7. Timezone (UTC for trading consistency)
# =============================================================================
log "Timezone UTC olarak ayarlaniyor..."
timedatectl set-timezone UTC

# =============================================================================
# 8. Create application directories
# =============================================================================
log "Uygulama dizinleri olusturuluyor..."
mkdir -p "$APP_DIR"
mkdir -p "${APP_HOME}/logs"
chown -R "${APP_USER}:${APP_USER}" "$APP_HOME"

# =============================================================================
# 9. Logrotate for application logs
# =============================================================================
log "Logrotate ayarlaniyor..."
cat > /etc/logrotate.d/aitradew << LOGROTATE
${APP_HOME}/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 ${APP_USER} ${APP_USER}
    sharedscripts
    postrotate
        systemctl try-restart aitradew-paper.service 2>/dev/null || true
    endscript
}
LOGROTATE

# =============================================================================
# 10. Security hardening (SSH root login remains enabled for management)
# =============================================================================
log "SSH ayarlari kontrol ediliyor..."

# Keep root login enabled (password) for VPS management
# If you set up SSH keys later, you can manually change to:
#   PermitRootLogin prohibit-password
if grep -q "^PermitEmptyPasswords" /etc/ssh/sshd_config; then
    sed -i 's/^PermitEmptyPasswords.*/PermitEmptyPasswords no/' /etc/ssh/sshd_config
else
    echo "PermitEmptyPasswords no" >> /etc/ssh/sshd_config
fi

systemctl reload sshd 2>/dev/null || systemctl reload ssh 2>/dev/null || true

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo -e "${GREEN} VPS KURULUMU TAMAMLANDI${NC}"
echo "============================================================"
echo ""
echo "  Python:       $($PYTHON --version)"
echo "  Kullanici:    ${APP_USER}"
echo "  Uygulama:     ${APP_DIR}"
echo "  Loglar:       ${APP_HOME}/logs/"
echo "  Timezone:     $(timedatectl show --property=Timezone --value)"
echo "  Swap:         $(swapon --show --noheadings | awk '{print $3}' || echo 'N/A')"
echo "  Firewall:     $(ufw status | head -1)"
echo "  Fail2Ban:     $(systemctl is-active fail2ban)"
echo ""
echo "  Sonraki adim: Projeyi ${APP_DIR} dizinine kopyalayin"
echo "  ve 'sudo -u ${APP_USER} bash install.sh' calistirin."
echo "============================================================"
