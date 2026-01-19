# /verify - Uygulama Doğrulama Ajanı

Bu komut bir subagent olarak çalışır. Uçtan uca doğrulama yapar.

## Adımlar

### 1. Kod Kalitesi
```bash
# Lint kontrolü
ruff check src/
black --check src/

# Type checking
mypy src/
```

### 2. Testler
```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# E2E tests (varsa)
pytest tests/e2e/ -v
```

### 3. Build Kontrolü
```bash
# Dependencies kontrol
pip check

# Import kontrolü
python -c "from src import *"
```

### 4. API Kontrolü (eğer API varsa)
```bash
# API'yi başlat
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
sleep 3

# Health check
curl http://localhost:8000/health

# API'yi durdur
pkill -f uvicorn
```

## Sonuç Raporu
Her adımın sonucunu özetle:
- ✅ Geçti
- ❌ Başarısız (neden?)
- ⚠️ Uyarı var

## Eğer Hata Varsa
1. Hatanın kaynağını bul
2. Düzeltme önerisi sun
3. Kullanıcıya sor: "Düzeltmemi ister misin?"
