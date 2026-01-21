"""
SignalGenerator - Ana sinyal üretici.

Multi-timeframe analiz ve ML tahminlerini birleştirerek sinyal üretir.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime

import pandas as pd
from loguru import logger

from src.data.fetcher import DataFetcher
from src.data.models import MarketData
from src.analysis.mtf_analyzer import MTFAnalyzer, MTFResult
from src.analysis.indicators.technical import TechnicalIndicators
from src.signals.models import Signal, SignalType, SignalStrength
from config.constants import (
    SIGNAL_CONFIDENCE_THRESHOLD,
    MIN_RISK_REWARD_RATIO,
    MIN_CONFLUENCE_SCORE,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
)


class SignalGenerator:
    """Gelişmiş sinyal üretici."""

    def __init__(
        self,
        timeframes: Optional[List[str]] = None,
        use_ml: bool = False,
    ):
        """
        SignalGenerator başlat.

        Args:
            timeframes: Analiz edilecek timeframe listesi
            use_ml: ML tahminlerini kullan
        """
        self.timeframes = timeframes or ["15m", "1h", "4h"]
        self.use_ml = use_ml
        self.fetcher = DataFetcher()
        self.mtf_analyzer = MTFAnalyzer(self.timeframes)
        self._ml_predictor = None

    @property
    def ml_predictor(self):
        """Lazy-loaded ML predictor."""
        if self._ml_predictor is None and self.use_ml:
            from src.ml.predictor import MLPredictor
            self._ml_predictor = MLPredictor()
        return self._ml_predictor

    async def generate_signal(
        self,
        symbol: str,
        primary_tf: str = "1h",
    ) -> Signal:
        """
        Sembol için sinyal üret.

        Args:
            symbol: Trading pair (örn: BTCUSDT)
            primary_tf: Ana timeframe

        Returns:
            Trading sinyali
        """
        logger.info(f"Sinyal üretiliyor: {symbol}")

        # Multi-timeframe veri çek
        market_data = await self.fetcher.fetch_multi_timeframe(
            symbol, self.timeframes
        )

        if not market_data:
            return self._create_wait_signal(symbol, primary_tf, "Veri alınamadı")

        # MTF analiz
        mtf_result = self.mtf_analyzer.analyze(market_data)

        # Primary TF için detaylı analiz
        primary_data = market_data.get(primary_tf)
        if not primary_data:
            return self._create_wait_signal(symbol, primary_tf, "Primary TF verisi yok")

        df = self._to_dataframe(primary_data)
        ta = TechnicalIndicators(df)
        ta.add_all_indicators()

        # Sinyal oluştur
        signal = self._build_signal(
            symbol=symbol,
            timeframe=primary_tf,
            mtf_result=mtf_result,
            ta=ta,
            df=ta.df,
        )

        # ML tahmin ekle
        if self.use_ml and self.ml_predictor:
            try:
                ml_confidence = await self.ml_predictor.predict(df)
                signal.confidence = (signal.confidence + ml_confidence) / 2
                signal.indicators["ml_confidence"] = ml_confidence
            except Exception as e:
                logger.warning(f"ML tahmin hatası: {e}")

        # Risk/Reward hesapla
        signal.calculate_risk_reward()

        logger.info(
            f"Sinyal: {signal.signal_type.value} | "
            f"Güven: {signal.confidence:.2f} | "
            f"R/R: {signal.risk_reward_ratio}"
        )

        return signal

    def _build_signal(
        self,
        symbol: str,
        timeframe: str,
        mtf_result: MTFResult,
        ta: TechnicalIndicators,
        df: pd.DataFrame,
    ) -> Signal:
        """
        Analizlerden sinyal oluştur.

        Args:
            symbol: Trading pair
            timeframe: Timeframe
            mtf_result: MTF analiz sonucu
            ta: Technical indicators
            df: DataFrame

        Returns:
            Signal objesi
        """
        last = df.iloc[-1]
        reasons = []

        # Signal type belirle
        signal_type = SignalType.WAIT
        strength = SignalStrength.WEAK

        if mtf_result.overall_bias == "bullish":
            signal_type = SignalType.LONG
            reasons.append(f"MTF Bias: Bullish")
        elif mtf_result.overall_bias == "bearish":
            signal_type = SignalType.SHORT
            reasons.append(f"MTF Bias: Bearish")

        # Ek koşullar
        if last["rsi"] < RSI_OVERSOLD and signal_type == SignalType.LONG:
            reasons.append("RSI oversold bölgesinde")
            strength = SignalStrength.STRONG
        elif last["rsi"] > RSI_OVERBOUGHT and signal_type == SignalType.SHORT:
            reasons.append("RSI overbought bölgesinde")
            strength = SignalStrength.STRONG

        if last.get("macd_bullish", False) and signal_type == SignalType.LONG:
            reasons.append("MACD bullish crossover")
        elif last.get("macd_bearish", False) and signal_type == SignalType.SHORT:
            reasons.append("MACD bearish crossover")

        if last.get("volume_spike", False):
            reasons.append("Volume spike tespit edildi")

        # Confidence hesapla
        confidence = self._calculate_confidence(mtf_result, ta, df)

        # Confidence düşükse WAIT
        if confidence < SIGNAL_CONFIDENCE_THRESHOLD:
            signal_type = SignalType.WAIT
            reasons = ["Düşük güven skoru"]

        # Entry, SL, TP hesapla
        entry_price = last["close"]
        atr = last.get("atr", entry_price * 0.02)

        if signal_type == SignalType.LONG:
            stop_loss = entry_price - (atr * 1.5)
            take_profit = entry_price + (atr * 3)
        elif signal_type == SignalType.SHORT:
            stop_loss = entry_price + (atr * 1.5)
            take_profit = entry_price - (atr * 3)
        else:
            stop_loss = entry_price
            take_profit = entry_price

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            entry_price=round(entry_price, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            timeframe=timeframe,
            reasons=reasons,
            confluence_score=mtf_result.confluence_score,
            indicators={
                "rsi": round(last["rsi"], 2),
                "macd": round(last["macd"], 6),
                "atr": round(atr, 4),
                "trend": ta.get_trend(),
            },
        )

    def _calculate_confidence(
        self,
        mtf_result: MTFResult,
        ta: TechnicalIndicators,
        df: pd.DataFrame,
    ) -> float:
        """
        Güven skoru hesapla.

        Args:
            mtf_result: MTF sonucu
            ta: Technical indicators
            df: DataFrame

        Returns:
            0-1 arası güven skoru
        """
        score = 0.0

        # Confluence score katkısı (0-0.4)
        score += (mtf_result.confluence_score / 10) * 0.4

        # Trend alignment katkısı (0-0.3)
        trend = ta.get_trend()
        if trend != "neutral":
            score += 0.3

        # Momentum katkısı (0-0.2)
        momentum = ta.get_momentum()
        last = df.iloc[-1]
        if (trend == "bullish" and last["rsi"] < 70) or (
            trend == "bearish" and last["rsi"] > 30
        ):
            score += 0.2

        # Volume katkısı (0-0.1)
        if last.get("volume_spike", False):
            score += 0.1

        return min(score, 1.0)

    def _create_wait_signal(
        self,
        symbol: str,
        timeframe: str,
        reason: str,
    ) -> Signal:
        """WAIT sinyali oluştur."""
        return Signal(
            symbol=symbol,
            signal_type=SignalType.WAIT,
            strength=SignalStrength.WEAK,
            confidence=0,
            entry_price=0,
            stop_loss=0,
            take_profit=0,
            timeframe=timeframe,
            reasons=[reason],
        )

    def _to_dataframe(self, market_data: MarketData) -> pd.DataFrame:
        """MarketData'yı DataFrame'e çevir."""
        data = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in market_data.candles
        ]
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    async def close(self) -> None:
        """Kaynakları temizle."""
        await self.fetcher.close()
