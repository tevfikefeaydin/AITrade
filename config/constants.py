"""
Sabitler - Uygulama genelinde kullanılan sabit değerler.

Magic number kullanmak yerine burada tanımla.
"""

# Timeframe ayarları
DEFAULT_TIMEFRAME = "1h"
SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

# Risk yönetimi
MIN_RISK_REWARD_RATIO = 2.0  # Minimum 1:2 R/R
MAX_RISK_PER_TRADE = 0.02  # Maksimum %2 risk
DEFAULT_STOP_LOSS_PERCENT = 0.02  # %2 stop loss

# Sinyal ayarları
SIGNAL_CONFIDENCE_THRESHOLD = 0.7  # Minimum güven skoru
MIN_CONFLUENCE_SCORE = 3  # Minimum konfluens sayısı

# API ayarları
API_RATE_LIMIT = 1200  # requests per minute
API_RETRY_COUNT = 3
API_RETRY_DELAY = 1.0  # seconds

# Cache ayarları
CACHE_TTL_SECONDS = 60  # 1 dakika
CACHE_TTL_OHLCV = 300  # 5 dakika (OHLCV verisi için)

# Teknik analiz parametreleri
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

EMA_SHORT = 9
EMA_MEDIUM = 21
EMA_LONG = 50
EMA_TREND = 200

BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0

ATR_PERIOD = 14

# Volume analizi
VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_THRESHOLD = 2.0  # Ortalama hacmin 2 katı

# ICT parametreleri
ORDER_BLOCK_LOOKBACK = 20
FVG_MIN_SIZE_PERCENT = 0.001  # %0.1 minimum FVG boyutu
LIQUIDITY_SWEEP_THRESHOLD = 0.002  # %0.2

# SMC parametreleri
STRUCTURE_LOOKBACK = 50
BOS_MIN_MOVE_PERCENT = 0.005  # %0.5 minimum BOS hareketi

# Logging
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
LOG_ROTATION = "10 MB"
LOG_RETENTION = "7 days"
