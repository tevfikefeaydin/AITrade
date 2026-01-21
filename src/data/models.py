"""
Data modelleri - Pydantic veri modelleri.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class OHLCV(BaseModel):
    """Tek bir mum verisi."""

    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @property
    def is_bullish(self) -> bool:
        """Mum yeşil mi?"""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Mum kırmızı mı?"""
        return self.close < self.open

    @property
    def body_size(self) -> float:
        """Mum gövde boyutu."""
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        """Üst fitil boyutu."""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        """Alt fitil boyutu."""
        return min(self.open, self.close) - self.low

    @property
    def total_range(self) -> float:
        """Toplam mum aralığı (high - low)."""
        return self.high - self.low


class MarketData(BaseModel):
    """Piyasa verisi container."""

    symbol: str
    timeframe: str
    candles: List[OHLCV]
    last_update: datetime = Field(default_factory=datetime.utcnow)

    @property
    def latest_price(self) -> Optional[float]:
        """Son kapanış fiyatı."""
        if self.candles:
            return self.candles[-1].close
        return None

    @property
    def candle_count(self) -> int:
        """Mum sayısı."""
        return len(self.candles)


class OrderBookLevel(BaseModel):
    """Order book seviyesi."""

    price: float
    quantity: float


class OrderBook(BaseModel):
    """Order book verisi."""

    symbol: str
    bids: List[OrderBookLevel]  # Alış emirleri
    asks: List[OrderBookLevel]  # Satış emirleri
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def best_bid(self) -> Optional[float]:
        """En iyi alış fiyatı."""
        if self.bids:
            return self.bids[0].price
        return None

    @property
    def best_ask(self) -> Optional[float]:
        """En iyi satış fiyatı."""
        if self.asks:
            return self.asks[0].price
        return None

    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


class Ticker(BaseModel):
    """24 saatlik ticker verisi."""

    symbol: str
    price: float
    price_change: float
    price_change_percent: float
    high_24h: float
    low_24h: float
    volume_24h: float
    quote_volume_24h: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
