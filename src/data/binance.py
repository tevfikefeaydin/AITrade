"""
Binance API istemcisi.

Async HTTP client ile Binance API'ye bağlanır.
"""

from datetime import datetime
from typing import List, Optional

import httpx
from loguru import logger

from config.constants import API_RATE_LIMIT, API_RETRY_COUNT, API_RETRY_DELAY
from config.settings import get_settings
from src.data.models import OHLCV, OrderBook, OrderBookLevel, Ticker
from src.data.constants import DEFAULT_CANDLE_LIMIT, MAX_CANDLE_LIMIT


class BinanceClient:
    """Binance API async istemcisi."""

    BASE_URL = "https://api.binance.com"
    FUTURES_URL = "https://fapi.binance.com"

    def __init__(self, use_futures: bool = False):
        """
        Binance client başlat.

        Args:
            use_futures: Futures API kullan (default: False)
        """
        self.settings = get_settings()
        self.base_url = self.FUTURES_URL if use_futures else self.BASE_URL
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """HTTP client instance döndür."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=30.0,
                headers={"X-MBX-APIKEY": self.settings.binance_api_key or ""},
            )
        return self._client

    async def close(self) -> None:
        """Client bağlantısını kapat."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
    ) -> dict:
        """
        API isteği yap (retry logic ile).

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parametreleri

        Returns:
            API yanıtı

        Raises:
            httpx.HTTPError: API hatası
        """
        client = await self._get_client()

        for attempt in range(API_RETRY_COUNT):
            try:
                response = await client.request(method, endpoint, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.warning(
                    f"Binance API hatası (deneme {attempt + 1}/{API_RETRY_COUNT}): {e}"
                )
                if attempt == API_RETRY_COUNT - 1:
                    raise
                import asyncio
                await asyncio.sleep(API_RETRY_DELAY * (attempt + 1))

        return {}

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = DEFAULT_CANDLE_LIMIT,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[OHLCV]:
        """
        OHLCV (kline) verisi çek.

        Args:
            symbol: Trading pair (örn: BTCUSDT)
            interval: Timeframe (örn: 1h, 4h, 1d)
            limit: Mum sayısı (max 1000)
            start_time: Başlangıç timestamp (ms)
            end_time: Bitiş timestamp (ms)

        Returns:
            OHLCV listesi
        """
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, MAX_CANDLE_LIMIT),
        }

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = await self._request("GET", "/api/v3/klines", params)

        candles = []
        for item in data:
            candle = OHLCV(
                timestamp=datetime.fromtimestamp(item[0] / 1000),
                open=float(item[1]),
                high=float(item[2]),
                low=float(item[3]),
                close=float(item[4]),
                volume=float(item[5]),
            )
            candles.append(candle)

        logger.debug(f"{symbol} için {len(candles)} mum çekildi")
        return candles

    async def get_ticker(self, symbol: str) -> Ticker:
        """
        24 saatlik ticker verisi çek.

        Args:
            symbol: Trading pair

        Returns:
            Ticker verisi
        """
        data = await self._request(
            "GET", "/api/v3/ticker/24hr", {"symbol": symbol.upper()}
        )

        return Ticker(
            symbol=data["symbol"],
            price=float(data["lastPrice"]),
            price_change=float(data["priceChange"]),
            price_change_percent=float(data["priceChangePercent"]),
            high_24h=float(data["highPrice"]),
            low_24h=float(data["lowPrice"]),
            volume_24h=float(data["volume"]),
            quote_volume_24h=float(data["quoteVolume"]),
        )

    async def get_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        """
        Order book verisi çek.

        Args:
            symbol: Trading pair
            limit: Derinlik (5, 10, 20, 50, 100, 500, 1000)

        Returns:
            OrderBook verisi
        """
        data = await self._request(
            "GET", "/api/v3/depth", {"symbol": symbol.upper(), "limit": limit}
        )

        bids = [OrderBookLevel(price=float(b[0]), quantity=float(b[1])) for b in data["bids"]]
        asks = [OrderBookLevel(price=float(a[0]), quantity=float(a[1])) for a in data["asks"]]

        return OrderBook(symbol=symbol, bids=bids, asks=asks)

    async def get_price(self, symbol: str) -> float:
        """
        Anlık fiyat çek.

        Args:
            symbol: Trading pair

        Returns:
            Güncel fiyat
        """
        data = await self._request(
            "GET", "/api/v3/ticker/price", {"symbol": symbol.upper()}
        )
        return float(data["price"])

    async def get_all_tickers(self) -> List[Ticker]:
        """
        Tüm sembollerin ticker verilerini çek.

        Returns:
            Ticker listesi
        """
        data = await self._request("GET", "/api/v3/ticker/24hr")

        tickers = []
        for item in data:
            if item["symbol"].endswith("USDT"):
                ticker = Ticker(
                    symbol=item["symbol"],
                    price=float(item["lastPrice"]),
                    price_change=float(item["priceChange"]),
                    price_change_percent=float(item["priceChangePercent"]),
                    high_24h=float(item["highPrice"]),
                    low_24h=float(item["lowPrice"]),
                    volume_24h=float(item["volume"]),
                    quote_volume_24h=float(item["quoteVolume"]),
                )
                tickers.append(ticker)

        return tickers
