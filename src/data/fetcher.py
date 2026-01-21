"""
DataFetcher - Çoklu kaynaklardan veri çekme.

Binance, Yahoo Finance ve diğer kaynakları destekler.
"""

from datetime import datetime
from typing import List, Optional

import pandas as pd
from loguru import logger

from src.data.binance import BinanceClient
from src.data.models import OHLCV, MarketData
from src.data.constants import DEFAULT_CANDLE_LIMIT, OHLCV_COLUMNS
from config.constants import DEFAULT_TIMEFRAME


class DataFetcher:
    """Çoklu kaynaktan veri çekici."""

    def __init__(self):
        """DataFetcher başlat."""
        self._binance: Optional[BinanceClient] = None

    @property
    def binance(self) -> BinanceClient:
        """Lazy-loaded Binance client."""
        if self._binance is None:
            self._binance = BinanceClient()
        return self._binance

    async def close(self) -> None:
        """Tüm bağlantıları kapat."""
        if self._binance:
            await self._binance.close()

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
        limit: int = DEFAULT_CANDLE_LIMIT,
        source: str = "binance",
    ) -> List[OHLCV]:
        """
        OHLCV verisi çek.

        Args:
            symbol: Trading pair (örn: BTCUSDT)
            timeframe: Zaman dilimi (örn: 1h, 4h)
            limit: Mum sayısı
            source: Veri kaynağı (binance, yahoo)

        Returns:
            OHLCV listesi
        """
        logger.info(f"Veri çekiliyor: {symbol} {timeframe} ({source})")

        if source == "binance":
            return await self.binance.get_klines(symbol, timeframe, limit)
        else:
            raise ValueError(f"Desteklenmeyen kaynak: {source}")

    async def fetch_market_data(
        self,
        symbol: str,
        timeframe: str = DEFAULT_TIMEFRAME,
        limit: int = DEFAULT_CANDLE_LIMIT,
    ) -> MarketData:
        """
        MarketData objesi döndür.

        Args:
            symbol: Trading pair
            timeframe: Zaman dilimi
            limit: Mum sayısı

        Returns:
            MarketData container
        """
        candles = await self.fetch_ohlcv(symbol, timeframe, limit)

        return MarketData(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            last_update=datetime.utcnow(),
        )

    async def fetch_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str],
        limit: int = DEFAULT_CANDLE_LIMIT,
    ) -> dict[str, MarketData]:
        """
        Çoklu timeframe verisi çek.

        Args:
            symbol: Trading pair
            timeframes: Timeframe listesi
            limit: Her timeframe için mum sayısı

        Returns:
            Timeframe -> MarketData mapping
        """
        import asyncio

        tasks = [
            self.fetch_market_data(symbol, tf, limit)
            for tf in timeframes
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        data = {}
        for tf, result in zip(timeframes, results):
            if isinstance(result, Exception):
                logger.error(f"{symbol} {tf} veri çekme hatası: {result}")
            else:
                data[tf] = result

        return data

    def ohlcv_to_dataframe(self, candles: List[OHLCV]) -> pd.DataFrame:
        """
        OHLCV listesini DataFrame'e çevir.

        Args:
            candles: OHLCV listesi

        Returns:
            pandas DataFrame
        """
        data = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df
