"""
Pytest fixtures ve konfigürasyon.
"""

from datetime import datetime, timedelta
from typing import List

import pytest
import pandas as pd
import numpy as np

from src.data.models import OHLCV, MarketData


@pytest.fixture
def sample_ohlcv_data() -> List[OHLCV]:
    """Örnek OHLCV verisi."""
    data = []
    base_price = 50000
    base_time = datetime(2024, 1, 1, 0, 0, 0)

    np.random.seed(42)

    for i in range(100):
        change = np.random.normal(0, 0.02)
        open_price = base_price * (1 + change)
        close_price = open_price * (1 + np.random.normal(0, 0.01))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
        volume = np.random.uniform(100, 1000)

        candle = OHLCV(
            timestamp=base_time + timedelta(hours=i),
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=round(volume, 2),
        )
        data.append(candle)
        base_price = close_price

    return data


@pytest.fixture
def sample_dataframe(sample_ohlcv_data: List[OHLCV]) -> pd.DataFrame:
    """Örnek DataFrame."""
    data = [
        {
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in sample_ohlcv_data
    ]
    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df


@pytest.fixture
def sample_market_data(sample_ohlcv_data: List[OHLCV]) -> MarketData:
    """Örnek MarketData."""
    return MarketData(
        symbol="BTCUSDT",
        timeframe="1h",
        candles=sample_ohlcv_data,
    )


@pytest.fixture
def bullish_dataframe() -> pd.DataFrame:
    """Yükseliş trendi DataFrame."""
    np.random.seed(123)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="h")

    base = 50000
    prices = [base]
    for _ in range(99):
        # Yukarı trend
        change = np.random.normal(0.002, 0.01)
        prices.append(prices[-1] * (1 + change))

    df = pd.DataFrame({
        "open": prices,
        "close": [p * (1 + np.random.normal(0.001, 0.005)) for p in prices],
        "high": [p * 1.005 for p in prices],
        "low": [p * 0.995 for p in prices],
        "volume": np.random.uniform(100, 500, 100),
    }, index=dates)

    return df


@pytest.fixture
def bearish_dataframe() -> pd.DataFrame:
    """Düşüş trendi DataFrame."""
    np.random.seed(456)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="h")

    base = 50000
    prices = [base]
    for _ in range(99):
        # Aşağı trend
        change = np.random.normal(-0.002, 0.01)
        prices.append(prices[-1] * (1 + change))

    df = pd.DataFrame({
        "open": prices,
        "close": [p * (1 + np.random.normal(-0.001, 0.005)) for p in prices],
        "high": [p * 1.005 for p in prices],
        "low": [p * 0.995 for p in prices],
        "volume": np.random.uniform(100, 500, 100),
    }, index=dates)

    return df
