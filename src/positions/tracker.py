"""
PositionTracker - Açık pozisyon takibi.
"""

from datetime import datetime
from typing import Optional, List, Dict
from enum import Enum

from pydantic import BaseModel, Field
from loguru import logger


class PositionStatus(str, Enum):
    """Pozisyon durumu."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PENDING = "PENDING"


class PositionSide(str, Enum):
    """Pozisyon yönü."""

    LONG = "LONG"
    SHORT = "SHORT"


class Position(BaseModel):
    """Pozisyon modeli."""

    id: str
    symbol: str
    side: PositionSide
    status: PositionStatus = PositionStatus.OPEN

    entry_price: float
    current_price: float = 0
    quantity: float
    leverage: int = 1

    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    opened_at: datetime = Field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None
    close_price: Optional[float] = None

    pnl: float = 0
    pnl_percent: float = 0
    fees: float = 0

    signal_id: Optional[str] = None
    notes: str = ""

    @property
    def is_profitable(self) -> bool:
        """Pozisyon karda mı?"""
        return self.pnl > 0

    def update_pnl(self, current_price: float) -> None:
        """
        PnL güncelle.

        Args:
            current_price: Güncel fiyat
        """
        self.current_price = current_price

        if self.side == PositionSide.LONG:
            self.pnl = (current_price - self.entry_price) * self.quantity * self.leverage
            self.pnl_percent = ((current_price - self.entry_price) / self.entry_price) * 100 * self.leverage
        else:
            self.pnl = (self.entry_price - current_price) * self.quantity * self.leverage
            self.pnl_percent = ((self.entry_price - current_price) / self.entry_price) * 100 * self.leverage

        self.pnl -= self.fees

    def should_close(self, current_price: float) -> tuple[bool, str]:
        """
        Pozisyon kapatılmalı mı kontrol et.

        Args:
            current_price: Güncel fiyat

        Returns:
            (kapatılmalı_mı, neden)
        """
        if self.side == PositionSide.LONG:
            if self.stop_loss and current_price <= self.stop_loss:
                return True, "STOP_LOSS"
            if self.take_profit and current_price >= self.take_profit:
                return True, "TAKE_PROFIT"
        else:
            if self.stop_loss and current_price >= self.stop_loss:
                return True, "STOP_LOSS"
            if self.take_profit and current_price <= self.take_profit:
                return True, "TAKE_PROFIT"

        return False, ""

    def close(self, close_price: float, reason: str = "") -> None:
        """
        Pozisyonu kapat.

        Args:
            close_price: Kapanış fiyatı
            reason: Kapanış nedeni
        """
        self.close_price = close_price
        self.closed_at = datetime.utcnow()
        self.status = PositionStatus.CLOSED
        self.update_pnl(close_price)
        self.notes = f"Kapatıldı: {reason}" if reason else "Manuel kapatıldı"


class PositionTracker:
    """Pozisyon takip sınıfı."""

    def __init__(self):
        """PositionTracker başlat."""
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        entry_price: float,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: int = 1,
        signal_id: Optional[str] = None,
    ) -> Position:
        """
        Yeni pozisyon aç.

        Args:
            symbol: Trading pair
            side: LONG veya SHORT
            entry_price: Giriş fiyatı
            quantity: Miktar
            stop_loss: Stop loss fiyatı
            take_profit: Take profit fiyatı
            leverage: Kaldıraç
            signal_id: İlişkili sinyal ID

        Returns:
            Açılan pozisyon
        """
        position_id = f"{symbol}_{side.value}_{datetime.utcnow().timestamp()}"

        position = Position(
            id=position_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage,
            signal_id=signal_id,
        )

        self.positions[position_id] = position
        logger.info(f"Pozisyon açıldı: {position_id} | {side.value} {symbol} @ {entry_price}")

        return position

    def close_position(
        self,
        position_id: str,
        close_price: float,
        reason: str = "",
    ) -> Optional[Position]:
        """
        Pozisyon kapat.

        Args:
            position_id: Pozisyon ID
            close_price: Kapanış fiyatı
            reason: Kapanış nedeni

        Returns:
            Kapatılan pozisyon
        """
        position = self.positions.get(position_id)
        if not position:
            logger.warning(f"Pozisyon bulunamadı: {position_id}")
            return None

        position.close(close_price, reason)
        self.closed_positions.append(position)
        del self.positions[position_id]

        logger.info(
            f"Pozisyon kapatıldı: {position_id} | "
            f"PnL: {position.pnl:.2f} ({position.pnl_percent:.2f}%)"
        )

        return position

    def update_prices(self, prices: Dict[str, float]) -> List[tuple[str, str]]:
        """
        Fiyatları güncelle ve SL/TP kontrol et.

        Args:
            prices: Symbol -> fiyat mapping

        Returns:
            Kapatılması gereken pozisyonlar [(position_id, reason), ...]
        """
        to_close = []

        for position_id, position in self.positions.items():
            price = prices.get(position.symbol)
            if price is None:
                continue

            position.update_pnl(price)

            should_close, reason = position.should_close(price)
            if should_close:
                to_close.append((position_id, reason))

        return to_close

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Açık pozisyonları getir.

        Args:
            symbol: Filtrelenecek sembol (opsiyonel)

        Returns:
            Pozisyon listesi
        """
        positions = list(self.positions.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions

    def get_total_pnl(self) -> float:
        """Toplam açık PnL."""
        return sum(p.pnl for p in self.positions.values())

    def get_statistics(self) -> Dict:
        """
        İstatistikleri hesapla.

        Returns:
            İstatistik dict'i
        """
        all_closed = self.closed_positions
        if not all_closed:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_pnl": 0,
            }

        wins = [p for p in all_closed if p.pnl > 0]
        losses = [p for p in all_closed if p.pnl <= 0]

        return {
            "total_trades": len(all_closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(all_closed)) * 100 if all_closed else 0,
            "total_pnl": sum(p.pnl for p in all_closed),
            "avg_pnl": sum(p.pnl_percent for p in all_closed) / len(all_closed),
            "best_trade": max(p.pnl_percent for p in all_closed) if all_closed else 0,
            "worst_trade": min(p.pnl_percent for p in all_closed) if all_closed else 0,
        }
