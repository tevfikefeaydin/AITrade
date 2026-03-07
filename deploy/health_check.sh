#!/usr/bin/env bash
# =============================================================================
# AITradew.AI - Health Check
# =============================================================================
# Quick status check for all services. Run anytime:
#   bash deploy/health_check.sh
# =============================================================================

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}OK${NC}    $*"; }
fail() { echo -e "  ${RED}FAIL${NC}  $*"; }
warn() { echo -e "  ${YELLOW}WARN${NC}  $*"; }

APP_DIR="/home/aitradew/app"

echo ""
echo "============================================================"
echo " AITradew.AI Health Check - $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

# 1. Paper Trader Service
echo ""
echo "[ Servisler ]"
if systemctl is-active --quiet aitradew-paper.service 2>/dev/null; then
    ok "Paper trader calisiyor"
else
    fail "Paper trader CALISMIYIOR"
fi

# 2. Timers
if systemctl is-active --quiet aitradew-update.timer 2>/dev/null; then
    ok "Data update timer aktif"
else
    warn "Data update timer pasif"
fi

if systemctl is-active --quiet aitradew-retrain.timer 2>/dev/null; then
    ok "Retrain timer aktif"
else
    warn "Retrain timer pasif"
fi

# 3. Disk usage
echo ""
echo "[ Disk ]"
DISK_USAGE=$(df -h /home 2>/dev/null | tail -1 | awk '{print $5}' | tr -d '%')
if [[ -n "$DISK_USAGE" ]] && [[ "$DISK_USAGE" -lt 80 ]]; then
    ok "Disk kullanimi: ${DISK_USAGE}%"
elif [[ -n "$DISK_USAGE" ]]; then
    warn "Disk kullanimi yuksek: ${DISK_USAGE}%"
else
    warn "Disk bilgisi alinamadi"
fi

# 4. Memory
echo ""
echo "[ Bellek ]"
MEM_AVAILABLE=$(free -m | awk '/^Mem:/ {print $7}')
if [[ "$MEM_AVAILABLE" -gt 200 ]]; then
    ok "Kullanilabilir RAM: ${MEM_AVAILABLE}MB"
else
    warn "Dusuk RAM: ${MEM_AVAILABLE}MB"
fi

# 5. Data freshness
echo ""
echo "[ Veri ]"
for SYMBOL in BTCUSDT ETHUSDT; do
    PARQUET="${APP_DIR}/data/${SYMBOL}_1m.parquet"
    if [[ -f "$PARQUET" ]]; then
        AGE_HOURS=$(( ( $(date +%s) - $(stat -c %Y "$PARQUET" 2>/dev/null || stat -f %m "$PARQUET" 2>/dev/null) ) / 3600 ))
        if [[ "$AGE_HOURS" -lt 26 ]]; then
            ok "${SYMBOL} verisi guncel (${AGE_HOURS} saat once)"
        else
            warn "${SYMBOL} verisi eski (${AGE_HOURS} saat once)"
        fi
    else
        fail "${SYMBOL} veri dosyasi bulunamadi"
    fi
done

# 6. Models
echo ""
echo "[ Modeller ]"
for SYMBOL in BTCUSDT ETHUSDT; do
    MODEL="${APP_DIR}/models/${SYMBOL}_model.pkl"
    if [[ -f "$MODEL" ]]; then
        AGE_DAYS=$(( ( $(date +%s) - $(stat -c %Y "$MODEL" 2>/dev/null || stat -f %m "$MODEL" 2>/dev/null) ) / 86400 ))
        if [[ "$AGE_DAYS" -lt 8 ]]; then
            ok "${SYMBOL} model guncel (${AGE_DAYS} gun once)"
        else
            warn "${SYMBOL} model eski (${AGE_DAYS} gun once)"
        fi
    else
        fail "${SYMBOL} model bulunamadi"
    fi
done

# 7. Recent logs
echo ""
echo "[ Son Loglar (paper trader) ]"
LOG_FILE="${APP_DIR}/../logs/paper_trader.log"
if [[ -f "$LOG_FILE" ]]; then
    tail -5 "$LOG_FILE" | while read -r line; do
        echo "  $line"
    done
else
    warn "Log dosyasi bulunamadi"
fi

# 8. Paper trade stats
echo ""
echo "[ Paper Trade Sonuclari ]"
for SYMBOL in BTCUSDT ETHUSDT; do
    TRADES="${APP_DIR}/outputs/${SYMBOL}_paper_trades.json"
    if [[ -f "$TRADES" ]]; then
        COUNT=$(python3 -c "import json; d=json.load(open('$TRADES')); print(len([t for t in d if t.get('status')=='CLOSED']))" 2>/dev/null || echo "?")
        ok "${SYMBOL}: ${COUNT} kapanmis trade"
    else
        echo "  --    ${SYMBOL}: Henuz trade yok"
    fi
done

echo ""
echo "============================================================"
