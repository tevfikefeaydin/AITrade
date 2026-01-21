"""
Gelişmiş teknik indikatörler.

pandas-ta veya TA-Lib kullanan hesaplamalar.
"""

from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
from loguru import logger

from config.constants import (
    RSI_PERIOD,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    EMA_SHORT,
    EMA_MEDIUM,
    EMA_LONG,
    EMA_TREND,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    ATR_PERIOD,
    VOLUME_MA_PERIOD,
    VOLUME_SPIKE_THRESHOLD,
)
from src.analysis.indicators.simple import SimpleIndicators


class TechnicalIndicators:
    """Gelişmiş teknik analiz indikatörleri."""

    def __init__(self, df: pd.DataFrame):
        """
        TechnicalIndicators başlat.

        Args:
            df: OHLCV DataFrame (timestamp index, open, high, low, close, volume kolonları)
        """
        self.df = df.copy()
        self._validate_dataframe()

    def _validate_dataframe(self) -> None:
        """DataFrame'in gerekli kolonları içerdiğini doğrula."""
        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Eksik kolonlar: {missing}")

    def add_all_indicators(self) -> pd.DataFrame:
        """
        Tüm temel indikatörleri ekle.

        Returns:
            İndikatörler eklenmiş DataFrame
        """
        self.add_moving_averages()
        self.add_rsi()
        self.add_macd()
        self.add_bollinger_bands()
        self.add_atr()
        self.add_volume_indicators()

        return self.df

    def add_moving_averages(self) -> pd.DataFrame:
        """EMA'ları ekle."""
        close = self.df["close"]

        self.df["ema_9"] = SimpleIndicators.ema(close, EMA_SHORT)
        self.df["ema_21"] = SimpleIndicators.ema(close, EMA_MEDIUM)
        self.df["ema_50"] = SimpleIndicators.ema(close, EMA_LONG)
        self.df["ema_200"] = SimpleIndicators.ema(close, EMA_TREND)

        # SMA
        self.df["sma_20"] = SimpleIndicators.sma(close, 20)
        self.df["sma_50"] = SimpleIndicators.sma(close, 50)

        return self.df

    def add_rsi(self, period: int = RSI_PERIOD) -> pd.DataFrame:
        """RSI ekle."""
        self.df["rsi"] = SimpleIndicators.rsi(self.df["close"], period)

        # RSI durumu
        self.df["rsi_overbought"] = self.df["rsi"] > RSI_OVERBOUGHT
        self.df["rsi_oversold"] = self.df["rsi"] < RSI_OVERSOLD

        return self.df

    def add_macd(self) -> pd.DataFrame:
        """MACD ekle."""
        macd, signal, hist = SimpleIndicators.macd(self.df["close"])

        self.df["macd"] = macd
        self.df["macd_signal"] = signal
        self.df["macd_hist"] = hist

        # MACD crossover
        self.df["macd_bullish"] = (macd > signal) & (macd.shift(1) <= signal.shift(1))
        self.df["macd_bearish"] = (macd < signal) & (macd.shift(1) >= signal.shift(1))

        return self.df

    def add_bollinger_bands(self) -> pd.DataFrame:
        """Bollinger Bands ekle."""
        upper, middle, lower = SimpleIndicators.bollinger_bands(self.df["close"])

        self.df["bb_upper"] = upper
        self.df["bb_middle"] = middle
        self.df["bb_lower"] = lower

        # BB pozisyonu (0-1 arası)
        self.df["bb_position"] = (self.df["close"] - lower) / (upper - lower)

        return self.df

    def add_atr(self, period: int = ATR_PERIOD) -> pd.DataFrame:
        """ATR ekle."""
        self.df["atr"] = SimpleIndicators.atr(
            self.df["high"],
            self.df["low"],
            self.df["close"],
            period,
        )

        # ATR yüzdesi
        self.df["atr_percent"] = (self.df["atr"] / self.df["close"]) * 100

        return self.df

    def add_volume_indicators(self) -> pd.DataFrame:
        """Volume indikatörlerini ekle."""
        volume = self.df["volume"]

        # Volume MA
        self.df["volume_ma"] = SimpleIndicators.sma(volume, VOLUME_MA_PERIOD)

        # Volume spike
        self.df["volume_spike"] = volume > (self.df["volume_ma"] * VOLUME_SPIKE_THRESHOLD)

        # VWAP
        self.df["vwap"] = SimpleIndicators.vwap(
            self.df["high"],
            self.df["low"],
            self.df["close"],
            volume,
        )

        # Price vs VWAP
        self.df["above_vwap"] = self.df["close"] > self.df["vwap"]

        return self.df

    def add_stochastic(self, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Stochastic ekle."""
        k, d = SimpleIndicators.stochastic(
            self.df["high"],
            self.df["low"],
            self.df["close"],
            k_period,
            d_period,
        )

        self.df["stoch_k"] = k
        self.df["stoch_d"] = d

        return self.df

    def get_trend(self) -> str:
        """
        Genel trend yönünü belirle.

        Returns:
            "bullish", "bearish" veya "neutral"
        """
        if "ema_50" not in self.df.columns:
            self.add_moving_averages()

        last = self.df.iloc[-1]

        # EMA sıralaması
        ema_bullish = last["ema_9"] > last["ema_21"] > last["ema_50"]
        ema_bearish = last["ema_9"] < last["ema_21"] < last["ema_50"]

        # Fiyat 200 EMA'ya göre
        above_200 = last["close"] > last["ema_200"]

        if ema_bullish and above_200:
            return "bullish"
        elif ema_bearish and not above_200:
            return "bearish"
        else:
            return "neutral"

    def get_momentum(self) -> Dict[str, Any]:
        """
        Momentum göstergelerini özetle.

        Returns:
            Momentum özeti dict
        """
        if "rsi" not in self.df.columns:
            self.add_rsi()
        if "macd" not in self.df.columns:
            self.add_macd()

        last = self.df.iloc[-1]

        return {
            "rsi": round(last["rsi"], 2),
            "rsi_signal": "overbought" if last["rsi"] > RSI_OVERBOUGHT else (
                "oversold" if last["rsi"] < RSI_OVERSOLD else "neutral"
            ),
            "macd": round(last["macd"], 6),
            "macd_signal": "bullish" if last["macd"] > last["macd_signal"] else "bearish",
            "macd_histogram": round(last["macd_hist"], 6),
        }

    def get_volatility(self) -> Dict[str, Any]:
        """
        Volatilite göstergelerini özetle.

        Returns:
            Volatilite özeti dict
        """
        if "atr" not in self.df.columns:
            self.add_atr()
        if "bb_upper" not in self.df.columns:
            self.add_bollinger_bands()

        last = self.df.iloc[-1]

        return {
            "atr": round(last["atr"], 4),
            "atr_percent": round(last["atr_percent"], 2),
            "bb_width": round((last["bb_upper"] - last["bb_lower"]) / last["bb_middle"] * 100, 2),
            "bb_position": round(last["bb_position"], 2),
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Tüm indikatörlerin özetini al.

        Returns:
            Tam analiz özeti
        """
        self.add_all_indicators()

        return {
            "trend": self.get_trend(),
            "momentum": self.get_momentum(),
            "volatility": self.get_volatility(),
            "last_price": round(self.df["close"].iloc[-1], 4),
            "timestamp": str(self.df.index[-1]),
        }
