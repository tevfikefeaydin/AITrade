"""
Basit teknik indikatörler.

TA-Lib gerektirmeyen temel hesaplamalar.
"""

from typing import Optional

import pandas as pd
import numpy as np

from config.constants import (
    RSI_PERIOD,
    EMA_SHORT,
    EMA_MEDIUM,
    EMA_LONG,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    ATR_PERIOD,
    BOLLINGER_PERIOD,
    BOLLINGER_STD,
)


class SimpleIndicators:
    """Temel teknik indikatörler (TA-Lib gerektirmez)."""

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average hesapla.

        Args:
            series: Fiyat serisi
            period: Periyot

        Returns:
            SMA serisi
        """
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average hesapla.

        Args:
            series: Fiyat serisi
            period: Periyot

        Returns:
            EMA serisi
        """
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
        """
        Relative Strength Index hesapla.

        Args:
            series: Fiyat serisi
            period: RSI periyodu

        Returns:
            RSI serisi (0-100)
        """
        delta = series.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = MACD_FAST,
        slow: int = MACD_SLOW,
        signal: int = MACD_SIGNAL,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD hesapla.

        Args:
            series: Fiyat serisi
            fast: Hızlı EMA periyodu
            slow: Yavaş EMA periyodu
            signal: Sinyal periyodu

        Returns:
            (macd_line, signal_line, histogram) tuple
        """
        ema_fast = SimpleIndicators.ema(series, fast)
        ema_slow = SimpleIndicators.ema(series, slow)

        macd_line = ema_fast - ema_slow
        signal_line = SimpleIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int = BOLLINGER_PERIOD,
        std_dev: float = BOLLINGER_STD,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands hesapla.

        Args:
            series: Fiyat serisi
            period: SMA periyodu
            std_dev: Standart sapma çarpanı

        Returns:
            (upper_band, middle_band, lower_band) tuple
        """
        middle = SimpleIndicators.sma(series, period)
        std = series.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = ATR_PERIOD,
    ) -> pd.Series:
        """
        Average True Range hesapla.

        Args:
            high: High serisi
            low: Low serisi
            close: Close serisi
            period: ATR periyodu

        Returns:
            ATR serisi
        """
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        return atr

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator hesapla.

        Args:
            high: High serisi
            low: Low serisi
            close: Close serisi
            k_period: %K periyodu
            d_period: %D periyodu

        Returns:
            (%K, %D) tuple
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()

        return k, d

    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        Volume Weighted Average Price hesapla.

        Args:
            high: High serisi
            low: Low serisi
            close: Close serisi
            volume: Volume serisi

        Returns:
            VWAP serisi
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()

        return vwap

    @staticmethod
    def support_resistance(
        df: pd.DataFrame,
        window: int = 20,
        num_levels: int = 3,
    ) -> tuple[list[float], list[float]]:
        """
        Destek ve direnç seviyelerini tespit et.

        Args:
            df: OHLCV DataFrame
            window: Pivot pencere boyutu
            num_levels: Döndürülecek seviye sayısı

        Returns:
            (support_levels, resistance_levels) tuple
        """
        highs = df["high"].values
        lows = df["low"].values

        # Pivot high/low tespiti
        pivot_highs = []
        pivot_lows = []

        for i in range(window, len(df) - window):
            if highs[i] == max(highs[i - window : i + window + 1]):
                pivot_highs.append(highs[i])
            if lows[i] == min(lows[i - window : i + window + 1]):
                pivot_lows.append(lows[i])

        # En yakın seviyeleri seç
        current_price = df["close"].iloc[-1]

        resistances = sorted([p for p in pivot_highs if p > current_price])[:num_levels]
        supports = sorted([p for p in pivot_lows if p < current_price], reverse=True)[:num_levels]

        return supports, resistances
