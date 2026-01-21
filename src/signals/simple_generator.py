"""
SimpleSignalGenerator - Basit sinyal üretici.

Hızlı analiz için basitleştirilmiş sinyal üretici.
"""

from typing import Optional
from datetime import datetime

import pandas as pd
from loguru import logger

from src.analysis.indicators.simple import SimpleIndicators
from src.signals.models import Signal, SignalType, SignalStrength
from config.constants import (
    RSI_PERIOD,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    EMA_SHORT,
    EMA_MEDIUM,
    EMA_LONG,
)


class SimpleSignalGenerator:
    """Basit sinyal üretici."""

    def generate(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "1h",
    ) -> Signal:
        """
        DataFrame'den sinyal üret.

        Args:
            df: OHLCV DataFrame
            symbol: Trading pair
            timeframe: Timeframe

        Returns:
            Trading sinyali
        """
        # İndikatörleri hesapla
        close = df["close"]

        rsi = SimpleIndicators.rsi(close, RSI_PERIOD)
        ema_short = SimpleIndicators.ema(close, EMA_SHORT)
        ema_medium = SimpleIndicators.ema(close, EMA_MEDIUM)
        ema_long = SimpleIndicators.ema(close, EMA_LONG)
        atr = SimpleIndicators.atr(df["high"], df["low"], close)

        last_close = close.iloc[-1]
        last_rsi = rsi.iloc[-1]
        last_ema_short = ema_short.iloc[-1]
        last_ema_medium = ema_medium.iloc[-1]
        last_ema_long = ema_long.iloc[-1]
        last_atr = atr.iloc[-1]

        # Sinyal mantığı
        reasons = []
        signal_type = SignalType.WAIT
        strength = SignalStrength.WEAK
        confidence = 0.0

        # EMA sıralaması
        ema_bullish = last_ema_short > last_ema_medium > last_ema_long
        ema_bearish = last_ema_short < last_ema_medium < last_ema_long

        # Long koşulları
        if ema_bullish and last_rsi < 70:
            signal_type = SignalType.LONG
            reasons.append("EMA bullish alignment")
            confidence += 0.4

            if last_rsi < RSI_OVERSOLD:
                reasons.append("RSI oversold")
                strength = SignalStrength.STRONG
                confidence += 0.3
            elif last_rsi < 50:
                reasons.append("RSI nötr-düşük")
                confidence += 0.1

            if last_close > last_ema_short:
                reasons.append("Fiyat EMA9 üzerinde")
                confidence += 0.2

        # Short koşulları
        elif ema_bearish and last_rsi > 30:
            signal_type = SignalType.SHORT
            reasons.append("EMA bearish alignment")
            confidence += 0.4

            if last_rsi > RSI_OVERBOUGHT:
                reasons.append("RSI overbought")
                strength = SignalStrength.STRONG
                confidence += 0.3
            elif last_rsi > 50:
                reasons.append("RSI nötr-yüksek")
                confidence += 0.1

            if last_close < last_ema_short:
                reasons.append("Fiyat EMA9 altında")
                confidence += 0.2

        else:
            reasons.append("Net sinyal yok")

        # Entry, SL, TP
        entry_price = last_close

        if signal_type == SignalType.LONG:
            stop_loss = entry_price - (last_atr * 1.5)
            take_profit = entry_price + (last_atr * 3)
        elif signal_type == SignalType.SHORT:
            stop_loss = entry_price + (last_atr * 1.5)
            take_profit = entry_price - (last_atr * 3)
        else:
            stop_loss = entry_price
            take_profit = entry_price

        # Confidence düşükse WAIT
        if confidence < 0.5:
            signal_type = SignalType.WAIT
            strength = SignalStrength.WEAK

        signal = Signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=min(confidence, 1.0),
            entry_price=round(entry_price, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            timeframe=timeframe,
            reasons=reasons,
            indicators={
                "rsi": round(last_rsi, 2),
                "ema_short": round(last_ema_short, 4),
                "ema_medium": round(last_ema_medium, 4),
                "ema_long": round(last_ema_long, 4),
                "atr": round(last_atr, 4),
            },
        )

        signal.calculate_risk_reward()
        return signal
