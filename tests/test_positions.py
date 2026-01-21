"""
Position tracker testleri.
"""

import pytest

from src.positions.tracker import PositionTracker, Position, PositionSide, PositionStatus


class TestPosition:
    """Position model testleri."""

    def test_position_creation(self):
        """Pozisyon oluşturma testi."""
        position = Position(
            id="test_1",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000,
            quantity=0.1,
            stop_loss=49000,
            take_profit=52000,
        )

        assert position.symbol == "BTCUSDT"
        assert position.side == PositionSide.LONG
        assert position.status == PositionStatus.OPEN

    def test_long_pnl_calculation(self):
        """Long pozisyon PnL hesaplama."""
        position = Position(
            id="test_long",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000,
            quantity=0.1,
        )

        # Fiyat yükseldi
        position.update_pnl(51000)
        assert position.pnl > 0
        assert position.pnl_percent == 2.0  # %2 kar

        # Fiyat düştü
        position.update_pnl(49000)
        assert position.pnl < 0
        assert position.pnl_percent == -2.0  # %2 zarar

    def test_short_pnl_calculation(self):
        """Short pozisyon PnL hesaplama."""
        position = Position(
            id="test_short",
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            entry_price=50000,
            quantity=0.1,
        )

        # Fiyat düştü (short için kar)
        position.update_pnl(49000)
        assert position.pnl > 0
        assert position.pnl_percent == 2.0

        # Fiyat yükseldi (short için zarar)
        position.update_pnl(51000)
        assert position.pnl < 0

    def test_leverage_effect(self):
        """Kaldıraç etkisi testi."""
        position = Position(
            id="test_lev",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000,
            quantity=0.1,
            leverage=10,
        )

        position.update_pnl(50500)  # %1 hareket
        assert position.pnl_percent == 10.0  # 10x kaldıraç = %10

    def test_stop_loss_trigger(self):
        """Stop loss tetikleme testi."""
        position = Position(
            id="test_sl",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000,
            quantity=0.1,
            stop_loss=49000,
        )

        # SL üzerinde - kapatılmamalı
        should_close, reason = position.should_close(49500)
        assert not should_close

        # SL'de - kapatılmalı
        should_close, reason = position.should_close(49000)
        assert should_close
        assert reason == "STOP_LOSS"

    def test_take_profit_trigger(self):
        """Take profit tetikleme testi."""
        position = Position(
            id="test_tp",
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000,
            quantity=0.1,
            take_profit=52000,
        )

        # TP altında - kapatılmamalı
        should_close, reason = position.should_close(51500)
        assert not should_close

        # TP'de - kapatılmalı
        should_close, reason = position.should_close(52000)
        assert should_close
        assert reason == "TAKE_PROFIT"


class TestPositionTracker:
    """PositionTracker testleri."""

    def test_open_position(self):
        """Pozisyon açma testi."""
        tracker = PositionTracker()

        position = tracker.open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000,
            quantity=0.1,
            stop_loss=49000,
            take_profit=52000,
        )

        assert position.id in tracker.positions
        assert len(tracker.get_open_positions()) == 1

    def test_close_position(self):
        """Pozisyon kapatma testi."""
        tracker = PositionTracker()

        position = tracker.open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000,
            quantity=0.1,
        )

        closed = tracker.close_position(position.id, 51000, "Test close")

        assert closed is not None
        assert closed.status == PositionStatus.CLOSED
        assert len(tracker.positions) == 0
        assert len(tracker.closed_positions) == 1

    def test_update_prices(self):
        """Fiyat güncelleme testi."""
        tracker = PositionTracker()

        tracker.open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000,
            quantity=0.1,
            stop_loss=49000,
        )

        # SL tetiklenmeli
        to_close = tracker.update_prices({"BTCUSDT": 48000})

        assert len(to_close) == 1
        assert to_close[0][1] == "STOP_LOSS"

    def test_get_statistics(self):
        """İstatistik testi."""
        tracker = PositionTracker()

        # Karlı pozisyon
        pos1 = tracker.open_position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=50000,
            quantity=0.1,
        )
        tracker.close_position(pos1.id, 51000)

        # Zararlı pozisyon
        pos2 = tracker.open_position(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            entry_price=3000,
            quantity=1,
        )
        tracker.close_position(pos2.id, 2900)

        stats = tracker.get_statistics()

        assert stats["total_trades"] == 2
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["win_rate"] == 50.0
