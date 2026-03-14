"""
Pytest fixtures and configuration.
"""

from datetime import datetime, timedelta
from typing import List

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    base_price = 50000
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    dates = [base_time + timedelta(hours=i) for i in range(100)]

    prices = [base_price]
    for _ in range(99):
        change = np.random.normal(0, 0.02)
        prices.append(prices[-1] * (1 + change))

    df = pd.DataFrame({
        "open": prices,
        "close": [p * (1 + np.random.normal(0, 0.01)) for p in prices],
        "high": [max(p, p * (1 + np.random.normal(0, 0.01))) * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        "low": [min(p, p * (1 + np.random.normal(0, 0.01))) * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        "volume": np.random.uniform(100, 1000, 100),
    }, index=pd.DatetimeIndex(dates, name="timestamp"))

    return df


@pytest.fixture
def bullish_dataframe() -> pd.DataFrame:
    """Uptrend DataFrame for testing."""
    np.random.seed(123)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="h")

    base = 50000
    prices = [base]
    for _ in range(99):
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
    """Downtrend DataFrame for testing."""
    np.random.seed(456)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="h")

    base = 50000
    prices = [base]
    for _ in range(99):
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
