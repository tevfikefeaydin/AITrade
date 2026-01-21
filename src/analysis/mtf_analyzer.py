"""
Multi-Timeframe Analyzer.

Çoklu zaman dilimlerinde analiz yaparak konfluens arar.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import pandas as pd
from loguru import logger

from src.data.models import MarketData
from src.analysis.indicators.technical import TechnicalIndicators
from config.constants import MIN_CONFLUENCE_SCORE


@dataclass
class TimeframeAnalysis:
    """Tek bir timeframe analiz sonucu."""

    timeframe: str
    trend: str
    rsi: float
    macd_signal: str
    above_vwap: bool
    ema_aligned: bool
    volume_spike: bool


@dataclass
class MTFResult:
    """Multi-timeframe analiz sonucu."""

    symbol: str
    analyses: Dict[str, TimeframeAnalysis]
    overall_bias: str
    confluence_score: int
    trade_recommendation: str
    key_levels: Dict[str, float]


class MTFAnalyzer:
    """Multi-Timeframe analiz sınıfı."""

    # Timeframe ağırlıkları (yüksek TF daha önemli)
    TF_WEIGHTS = {
        "1m": 1,
        "5m": 2,
        "15m": 3,
        "30m": 4,
        "1h": 5,
        "4h": 6,
        "1d": 7,
        "1w": 8,
    }

    def __init__(self, timeframes: Optional[List[str]] = None):
        """
        MTFAnalyzer başlat.

        Args:
            timeframes: Analiz edilecek timeframe listesi
        """
        self.timeframes = timeframes or ["15m", "1h", "4h"]

    def analyze_single_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str,
    ) -> TimeframeAnalysis:
        """
        Tek timeframe analiz et.

        Args:
            df: OHLCV DataFrame
            timeframe: Timeframe adı

        Returns:
            TimeframeAnalysis sonucu
        """
        ta = TechnicalIndicators(df)
        ta.add_all_indicators()

        last = ta.df.iloc[-1]

        # EMA alignment kontrolü
        ema_aligned = (
            last.get("ema_9", 0) > last.get("ema_21", 0) > last.get("ema_50", 0)
            if last.get("close", 0) > last.get("ema_200", 0)
            else last.get("ema_9", 0) < last.get("ema_21", 0) < last.get("ema_50", 0)
        )

        return TimeframeAnalysis(
            timeframe=timeframe,
            trend=ta.get_trend(),
            rsi=round(last.get("rsi", 50), 2),
            macd_signal="bullish" if last.get("macd", 0) > last.get("macd_signal", 0) else "bearish",
            above_vwap=bool(last.get("above_vwap", False)),
            ema_aligned=ema_aligned,
            volume_spike=bool(last.get("volume_spike", False)),
        )

    def analyze(
        self,
        market_data: Dict[str, MarketData],
    ) -> MTFResult:
        """
        Çoklu timeframe analiz yap.

        Args:
            market_data: Timeframe -> MarketData mapping

        Returns:
            MTFResult sonucu
        """
        if not market_data:
            raise ValueError("Market data boş olamaz")

        symbol = list(market_data.values())[0].symbol
        analyses: Dict[str, TimeframeAnalysis] = {}

        # Her timeframe için analiz
        for tf, data in market_data.items():
            try:
                df = self._market_data_to_df(data)
                analysis = self.analyze_single_timeframe(df, tf)
                analyses[tf] = analysis
            except Exception as e:
                logger.error(f"{tf} analiz hatası: {e}")

        # Genel bias hesapla
        overall_bias = self._calculate_overall_bias(analyses)

        # Konfluens skoru
        confluence_score = self._calculate_confluence(analyses)

        # Trade önerisi
        recommendation = self._get_recommendation(overall_bias, confluence_score)

        # Key levels (en yüksek TF'den al)
        key_levels = self._get_key_levels(market_data)

        return MTFResult(
            symbol=symbol,
            analyses=analyses,
            overall_bias=overall_bias,
            confluence_score=confluence_score,
            trade_recommendation=recommendation,
            key_levels=key_levels,
        )

    def _market_data_to_df(self, market_data: MarketData) -> pd.DataFrame:
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

    def _calculate_overall_bias(
        self,
        analyses: Dict[str, TimeframeAnalysis],
    ) -> str:
        """
        Ağırlıklı genel bias hesapla.

        Args:
            analyses: Timeframe analizleri

        Returns:
            "bullish", "bearish" veya "neutral"
        """
        if not analyses:
            return "neutral"

        bullish_score = 0
        bearish_score = 0
        total_weight = 0

        for tf, analysis in analyses.items():
            weight = self.TF_WEIGHTS.get(tf, 1)
            total_weight += weight

            if analysis.trend == "bullish":
                bullish_score += weight
            elif analysis.trend == "bearish":
                bearish_score += weight

        if total_weight == 0:
            return "neutral"

        bullish_pct = bullish_score / total_weight
        bearish_pct = bearish_score / total_weight

        if bullish_pct > 0.6:
            return "bullish"
        elif bearish_pct > 0.6:
            return "bearish"
        else:
            return "neutral"

    def _calculate_confluence(
        self,
        analyses: Dict[str, TimeframeAnalysis],
    ) -> int:
        """
        Konfluens skoru hesapla (0-10).

        Args:
            analyses: Timeframe analizleri

        Returns:
            Konfluens skoru
        """
        if not analyses:
            return 0

        score = 0

        # Trend alignment
        trends = [a.trend for a in analyses.values()]
        if len(set(trends)) == 1 and trends[0] != "neutral":
            score += 3

        # MACD alignment
        macd_signals = [a.macd_signal for a in analyses.values()]
        if len(set(macd_signals)) == 1:
            score += 2

        # VWAP alignment
        vwap_positions = [a.above_vwap for a in analyses.values()]
        if len(set(vwap_positions)) == 1:
            score += 2

        # EMA alignment
        ema_aligned_count = sum(1 for a in analyses.values() if a.ema_aligned)
        if ema_aligned_count == len(analyses):
            score += 2

        # Volume confirmation
        volume_spikes = sum(1 for a in analyses.values() if a.volume_spike)
        if volume_spikes > 0:
            score += 1

        return min(score, 10)

    def _get_recommendation(
        self,
        bias: str,
        confluence: int,
    ) -> str:
        """
        Trade önerisi oluştur.

        Args:
            bias: Genel bias
            confluence: Konfluens skoru

        Returns:
            Trade önerisi
        """
        if confluence < MIN_CONFLUENCE_SCORE:
            return "WAIT - Düşük konfluens"

        if bias == "bullish" and confluence >= 6:
            return "LONG - Güçlü konfluens"
        elif bias == "bullish" and confluence >= MIN_CONFLUENCE_SCORE:
            return "LONG - Orta konfluens"
        elif bias == "bearish" and confluence >= 6:
            return "SHORT - Güçlü konfluens"
        elif bias == "bearish" and confluence >= MIN_CONFLUENCE_SCORE:
            return "SHORT - Orta konfluens"
        else:
            return "WAIT - Nötr piyasa"

    def _get_key_levels(
        self,
        market_data: Dict[str, MarketData],
    ) -> Dict[str, float]:
        """
        Önemli fiyat seviyelerini bul.

        Args:
            market_data: Timeframe -> MarketData mapping

        Returns:
            Key levels dict
        """
        # En yüksek timeframe'i kullan
        sorted_tfs = sorted(
            market_data.keys(),
            key=lambda x: self.TF_WEIGHTS.get(x, 0),
            reverse=True,
        )

        if not sorted_tfs:
            return {}

        highest_tf = sorted_tfs[0]
        data = market_data[highest_tf]
        df = self._market_data_to_df(data)

        # Son mumdan seviyeler
        last = df.iloc[-1]

        # ATR hesapla
        from src.analysis.indicators.simple import SimpleIndicators
        atr = SimpleIndicators.atr(df["high"], df["low"], df["close"]).iloc[-1]

        return {
            "current_price": round(last["close"], 4),
            "daily_high": round(df["high"].tail(24).max(), 4),  # Son 24 mum high
            "daily_low": round(df["low"].tail(24).min(), 4),  # Son 24 mum low
            "atr": round(atr, 4),
            "suggested_sl": round(last["close"] - (atr * 1.5), 4),
            "suggested_tp": round(last["close"] + (atr * 3), 4),
        }
