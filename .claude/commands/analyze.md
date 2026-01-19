# /analyze - Trading Analiz Komutu

Bu komut AITrade projesine özel trading analiz ajanıdır.

## Kullanım
```
/analyze [sembol] [timeframe]
```

Örnek:
```
/analyze BTCUSDT 1h
/analyze EURUSD 4h
```

## Analiz Adımları

### 1. 📊 Veri Çek
```python
# Binance/TradingView'den veri çek
data = fetch_ohlcv(symbol, timeframe, limit=500)
```

### 2. 📈 Teknik Analiz
Şu indikatörleri hesapla:
- RSI (14)
- MACD (12, 26, 9)
- EMA (20, 50, 200)
- Bollinger Bands (20, 2)
- ATR (14)
- Volume Profile

### 3. 🎯 ICT Analiz
- Order Block tespiti
- Fair Value Gap (FVG) tespiti
- Liquidity sweep noktaları
- Premium/Discount zone

### 4. 💰 Smart Money Concepts
- Break of Structure (BOS)
- Change of Character (CHoCH)
- Supply/Demand zones
- Imbalance bölgeleri

### 5. 🤖 ML Prediction (opsiyonel)
- XGBoost model ile yön tahmini
- Confidence score

## Çıktı Formatı

```
═══════════════════════════════════════
📊 BTCUSDT 1H ANALİZ RAPORU
═══════════════════════════════════════

🎯 TREND: BULLISH / BEARISH / NEUTRAL

📈 TEKNİK İNDİKATÖRLER:
• RSI: 65 (Neutral)
• MACD: Bullish crossover
• EMA: Fiyat 20 EMA üzerinde

🏦 SMART MONEY:
• Son BOS: Bullish @ 42,500
• Order Block: 41,800 - 42,000 (Demand)
• FVG: 43,200 - 43,500

⚡ SİNYAL: LONG / SHORT / BEKLE
• Entry: 42,500
• Stop Loss: 41,500 (-2.3%)
• Take Profit 1: 44,000 (+3.5%)
• Take Profit 2: 45,500 (+7.0%)
• Risk/Reward: 1:3

📊 CONFIDENCE: 78%
═══════════════════════════════════════
```

## Notlar
- Bu komut henüz implementasyon aşamasında
- Önce temel modüllerin oluşturulması gerekli
- /new-feature ile adım adım eklenecek
