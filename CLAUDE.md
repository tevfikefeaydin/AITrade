# AITrade - Claude Memory & Context

## 🎯 Proje Hakkında
AITrade, yapay zeka destekli bir trading sinyal ve analiz platformudur.
- **Amaç**: Çoklu analiz metodolojileri ile yüksek olasılıklı trade sinyalleri üretmek
- **Hedef**: Kripto, Forex ve hisse senedi piyasalarında çalışan otomatik analiz sistemi

## 🛠 Tech Stack
- **Backend**: Python 3.11+ (FastAPI)
- **ML/AI**: XGBoost, scikit-learn, TensorFlow
- **Data**: pandas, numpy, ta-lib
- **Database**: PostgreSQL + Redis (cache)
- **Frontend**: Next.js 14 + TailwindCSS
- **API**: REST + WebSocket (real-time data)

## 📊 Trading Metodolojileri
Bu projede kullanılan analiz yöntemleri:
1. **ICT (Inner Circle Trader)**: Order blocks, FVG, liquidity sweeps
2. **Smart Money Concepts**: BOS, CHoCH, supply/demand zones
3. **Price Action**: Support/resistance, trend analysis
4. **Volume Analysis**: Volume profile, VWAP
5. **Technical Indicators**: RSI, MACD, EMA, Bollinger Bands

## 📁 Proje Yapısı
```
AITrade/
├── src/
│   ├── api/              # FastAPI endpoints
│   ├── analysis/         # Trading analiz modülleri
│   │   ├── ict/          # ICT metodolojisi
│   │   ├── smc/          # Smart Money Concepts
│   │   ├── indicators/   # Teknik indikatörler
│   │   └── ml/           # Machine Learning modelleri
│   ├── data/             # Data fetching & processing
│   ├── signals/          # Sinyal üretimi
│   └── utils/            # Yardımcı fonksiyonlar
├── frontend/             # Next.js dashboard
├── tests/                # Test dosyaları
├── models/               # Trained ML models
└── config/               # Konfigürasyon dosyaları
```

## ✅ Kodlama Kuralları

### Python
- Type hints ZORUNLU: `def analyze(data: pd.DataFrame) -> dict:`
- Docstring her fonksiyonda olmalı (Google style)
- async/await kullan (FastAPI için)
- Exception handling her API call'da olmalı

### Naming Conventions
- Fonksiyonlar: `snake_case` → `calculate_rsi()`
- Sınıflar: `PascalCase` → `SignalGenerator`
- Sabitler: `UPPER_SNAKE` → `DEFAULT_TIMEFRAME`
- Dosyalar: `snake_case.py`

### Git
- Commit mesajları Türkçe
- Format: `[tip]: açıklama`
- Tipler: `feat`, `fix`, `refactor`, `docs`, `test`
- Örnek: `feat: ICT order block tespiti eklendi`

## ❌ YAPMA - Öğrenilen Hatalar
<!-- Claude hata yaptığında buraya ekle -->
- ❌ API key'leri ASLA hardcode etme → .env kullan
- ❌ Print yerine logging kullan
- ❌ Senkron requests kullanma → aiohttp veya httpx async
- ❌ Global değişken kullanma → dependency injection
- ❌ Test yazmadan PR açma
- ❌ Magic number kullanma → constants.py'de tanımla

## ✅ YAP - Best Practices
- ✅ Her yeni özellik için branch aç
- ✅ Rate limiting uygula (API calls)
- ✅ Retry logic ekle (network errors)
- ✅ Cache kullan (Redis)
- ✅ Logging her önemli işlemde
- ✅ Unit test coverage min %80

## 🔧 Sık Kullanılan Komutlar
```bash
# Virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# Dependencies
pip install -r requirements.txt

# Run API
uvicorn src.api.main:app --reload

# Run tests
pytest tests/ -v

# Lint
ruff check src/
black src/
```

## 📝 Notlar
<!-- Önemli kararlar ve notlar -->
- Timeframe default: 1h
- Risk/Reward minimum: 1:2
- Sinyal confidence threshold: 0.7

## 🔄 Changelog
<!-- Her önemli değişikliği kaydet -->
- [2025-01-19] Proje başlatıldı
