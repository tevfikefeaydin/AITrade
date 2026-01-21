"""
Sinyal modelleri.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    """Sinyal tipi."""

    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    WAIT = "WAIT"


class SignalStrength(str, Enum):
    """Sinyal gücü."""

    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


class Signal(BaseModel):
    """Trading sinyali."""

    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: float = Field(ge=0, le=1)

    entry_price: float
    stop_loss: float
    take_profit: float

    timeframe: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Analiz detayları
    reasons: List[str] = Field(default_factory=list)
    confluence_score: int = Field(ge=0, le=10, default=0)
    indicators: dict = Field(default_factory=dict)

    # Risk metrikleri
    risk_reward_ratio: float = Field(default=0)
    risk_percent: float = Field(default=0)

    @property
    def is_valid(self) -> bool:
        """Sinyal geçerli mi?"""
        return (
            self.confidence >= 0.7
            and self.risk_reward_ratio >= 2.0
            and self.signal_type != SignalType.WAIT
        )

    def calculate_risk_reward(self) -> float:
        """Risk/Reward oranını hesapla."""
        if self.signal_type == SignalType.LONG:
            risk = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        elif self.signal_type == SignalType.SHORT:
            risk = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit
        else:
            return 0

        if risk <= 0:
            return 0

        self.risk_reward_ratio = round(reward / risk, 2)
        return self.risk_reward_ratio

    def to_message(self) -> str:
        """Telegram mesaj formatı."""
        emoji = "" if self.signal_type == SignalType.LONG else ""
        strength_emoji = "" if self.strength == SignalStrength.STRONG else ""

        msg = f"""
{emoji} **{self.signal_type.value} SİNYALİ** {strength_emoji}

**Symbol**: {self.symbol}
**Timeframe**: {self.timeframe}
**Güven**: {self.confidence * 100:.1f}%
**Konfluens**: {self.confluence_score}/10

**Entry**: {self.entry_price:.4f}
**Stop Loss**: {self.stop_loss:.4f}
**Take Profit**: {self.take_profit:.4f}
**R/R**: 1:{self.risk_reward_ratio:.1f}

**Nedenler**:
{chr(10).join(f'- {r}' for r in self.reasons)}
"""
        return msg.strip()


class SignalHistory(BaseModel):
    """Sinyal geçmişi."""

    signal: Signal
    result: Optional[str] = None  # WIN, LOSS, BREAKEVEN
    pnl_percent: Optional[float] = None
    closed_at: Optional[datetime] = None
    notes: str = ""
