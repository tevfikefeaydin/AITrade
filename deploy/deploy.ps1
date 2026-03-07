# =============================================================================
# AITradew.AI - Deploy to VPS (PowerShell)
# =============================================================================
# Usage: powershell -ExecutionPolicy Bypass -File deploy\deploy.ps1 root@IP
# =============================================================================
param(
    [Parameter(Mandatory=$true)]
    [string]$Target
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "[+] Proje dizini: $ProjectDir" -ForegroundColor Green
Write-Host "[+] Hedef: ${Target}:/root/app" -ForegroundColor Green

# Ensure remote directory
Write-Host "[+] Uzak dizin hazirlaniyor..." -ForegroundColor Green
ssh $Target "mkdir -p /root/app"

# Upload each item separately (Windows scp compat)
$items = @("src", "tests", "deploy")
foreach ($item in $items) {
    $fullPath = Join-Path $ProjectDir $item
    if (Test-Path $fullPath) {
        Write-Host "[+] Gonderiliyor: $item" -ForegroundColor Green
        scp -r $fullPath "${Target}:/root/app/"
    }
}

$files = @("requirements.txt", "CLAUDE.md", "README.md", ".gitignore")
foreach ($f in $files) {
    $fullPath = Join-Path $ProjectDir $f
    if (Test-Path $fullPath) {
        Write-Host "[+] Gonderiliyor: $f" -ForegroundColor Green
        scp $fullPath "${Target}:/root/app/"
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " DOSYALAR YUKLENDI" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Simdi VPS'e baglan:" -ForegroundColor Yellow
Write-Host "    ssh $Target" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Ve sirayla calistir:" -ForegroundColor Yellow
Write-Host "    bash /root/app/deploy/setup_vps.sh" -ForegroundColor Yellow
Write-Host "    bash /root/app/deploy/install.sh" -ForegroundColor Yellow
Write-Host "    bash /root/app/deploy/enable_services.sh" -ForegroundColor Yellow
Write-Host "    cp /root/app/deploy/aitradew-sudoers /etc/sudoers.d/aitradew" -ForegroundColor Yellow
Write-Host ""
