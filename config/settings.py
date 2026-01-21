"""
Settings - Pydantic ayar yönetimi.

.env dosyasından ayarları okur.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Uygulama ayarları."""

    # API Keys - Binance
    binance_api_key: str = ""
    binance_secret_key: str = ""

    # API Keys - Bybit (opsiyonel)
    bybit_api_key: str = ""
    bybit_secret_key: str = ""

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Database
    database_url: str = "postgresql+asyncpg://localhost:5432/aitrade"
    redis_url: str = "redis://localhost:6379/0"

    # API Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # Trading
    default_symbol: str = "BTCUSDT"
    default_timeframe: str = "1h"
    paper_trading: bool = True
    risk_per_trade: float = 0.02
    min_risk_reward: float = 2.0
    signal_confidence_threshold: float = 0.7

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/aitrade.log"

    # ML Model
    model_path: str = "models/xgboost_v1.pkl"
    retrain_interval_days: int = 7

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance döndürür.

    Returns:
        Settings: Uygulama ayarları
    """
    return Settings()
