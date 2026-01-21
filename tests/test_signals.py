"""
Sinyal testleri.
"""

import pytest
import pandas as pd

from src.signals.models import Signal, SignalType, SignalStrength
from src.signals.simple_generator import SimpleSignalGenerator


class TestSignalModel:
    """Signal model testleri."""

    def test_signal_creation(self):
        """Sinyal oluşturma testi."""
        signal = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.LONG,
            strength=SignalStrength.STRONG,
            confidence=0.85,
            entry_price=50000,
            stop_loss=49000,
            take_profit=53000,
            timeframe="1h",
            reasons=["Test reason"],
        )

        assert signal.symbol == "BTCUSDT"
        assert signal.signal_type == SignalType.LONG
        assert signal.confidence == 0.85

    def test_risk_reward_calculation(self):
        """R/R hesaplama testi."""
        signal = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.LONG,
            strength=SignalStrength.MODERATE,
            confidence=0.75,
            entry_price=50000,
            stop_loss=49000,  # 1000 risk
            take_profit=52000,  # 2000 reward
            timeframe="1h",
        )

        rr = signal.calculate_risk_reward()
        assert rr == 2.0  # 1:2 R/R

    def test_short_risk_reward(self):
        """Short R/R hesaplama testi."""
        signal = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.SHORT,
            strength=SignalStrength.MODERATE,
            confidence=0.75,
            entry_price=50000,
            stop_loss=51000,  # 1000 risk
            take_profit=47000,  # 3000 reward
            timeframe="1h",
        )

        rr = signal.calculate_risk_reward()
        assert rr == 3.0  # 1:3 R/R

    def test_signal_validity(self):
        """Sinyal geçerlilik kontrolü."""
        valid_signal = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.LONG,
            strength=SignalStrength.STRONG,
            confidence=0.8,
            entry_price=50000,
            stop_loss=49000,
            take_profit=52000,
            timeframe="1h",
        )
        valid_signal.calculate_risk_reward()
        assert valid_signal.is_valid

        invalid_signal = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.WAIT,
            strength=SignalStrength.WEAK,
            confidence=0.3,
            entry_price=50000,
            stop_loss=50000,
            take_profit=50000,
            timeframe="1h",
        )
        assert not invalid_signal.is_valid

    def test_signal_to_message(self):
        """Telegram mesaj formatı testi."""
        signal = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.LONG,
            strength=SignalStrength.STRONG,
            confidence=0.85,
            entry_price=50000,
            stop_loss=49000,
            take_profit=53000,
            timeframe="1h",
            reasons=["EMA bullish alignment", "RSI oversold"],
            confluence_score=7,
        )
        signal.calculate_risk_reward()

        message = signal.to_message()

        assert "BTCUSDT" in message
        assert "LONG" in message
        assert "50000" in message
        assert "EMA bullish alignment" in message


class TestSimpleSignalGenerator:
    """SimpleSignalGenerator testleri."""

    def test_generate_with_bullish_data(self, bullish_dataframe: pd.DataFrame):
        """Bullish veride sinyal üretimi."""
        generator = SimpleSignalGenerator()
        signal = generator.generate(bullish_dataframe, "BTCUSDT")

        assert signal.symbol == "BTCUSDT"
        assert signal.signal_type in [SignalType.LONG, SignalType.WAIT]

    def test_generate_with_bearish_data(self, bearish_dataframe: pd.DataFrame):
        """Bearish veride sinyal üretimi."""
        generator = SimpleSignalGenerator()
        signal = generator.generate(bearish_dataframe, "ETHUSDT")

        assert signal.symbol == "ETHUSDT"
        assert signal.signal_type in [SignalType.SHORT, SignalType.WAIT]

    def test_signal_has_indicators(self, sample_dataframe: pd.DataFrame):
        """Sinyal indikatör verisi içermeli."""
        generator = SimpleSignalGenerator()
        signal = generator.generate(sample_dataframe, "BTCUSDT")

        assert "rsi" in signal.indicators
        assert "atr" in signal.indicators
        assert "ema_short" in signal.indicators
