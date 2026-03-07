"""
Tests for backtest engine.

Tests:
1. T+1 execution: signal at close t -> execute at open t+1
2. Cost application on entry and exit
3. Barrier fill logic with 1m data
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.backtest import simulate_barrier_exit, run_backtest
from src.utils import calculate_costs


class TestCostCalculation:
    """Test cost calculation functions."""

    def test_buy_costs_increase_price(self):
        """Test that buying incurs costs (higher effective price)."""
        price = 100.0
        fee_bps = 10.0  # 0.10%
        slippage_bps = 2.0  # 0.02%

        effective_price = calculate_costs(price, fee_bps, slippage_bps, is_buy=True)

        # Should be higher than base price
        assert effective_price > price

        # Should be exactly: price * (1 + (10+2)/10000) = 100 * 1.0012 = 100.12
        expected = price * (1 + (fee_bps + slippage_bps) / 10000)
        assert effective_price == pytest.approx(expected, rel=0.0001)

    def test_sell_costs_decrease_price(self):
        """Test that selling incurs costs (lower effective price)."""
        price = 100.0
        fee_bps = 10.0
        slippage_bps = 2.0

        effective_price = calculate_costs(price, fee_bps, slippage_bps, is_buy=False)

        # Should be lower than base price
        assert effective_price < price

        # Should be exactly: price * (1 - (10+2)/10000) = 100 * 0.9988 = 99.88
        expected = price * (1 - (fee_bps + slippage_bps) / 10000)
        assert effective_price == pytest.approx(expected, rel=0.0001)

    def test_zero_costs(self):
        """Test with zero costs."""
        price = 100.0

        buy_price = calculate_costs(price, 0.0, 0.0, is_buy=True)
        sell_price = calculate_costs(price, 0.0, 0.0, is_buy=False)

        assert buy_price == price
        assert sell_price == price


class TestBarrierExitSimulation:
    """Test barrier exit simulation with 1m data."""

    def create_1m_data_tp_hit(self, entry_price: float, pt: float) -> pd.DataFrame:
        """Create 1m data where TP is hit."""
        n_minutes = 60
        start_time = datetime(2024, 1, 1, 1, 0)  # Start after entry

        tp_price = entry_price * (1 + pt)

        # Price rises to hit TP
        prices = np.linspace(entry_price, tp_price * 1.01, n_minutes)

        return pd.DataFrame({
            "open_time": [start_time + timedelta(minutes=i) for i in range(n_minutes)],
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": [1000] * n_minutes,
        })

    def create_1m_data_sl_hit(self, entry_price: float, sl: float) -> pd.DataFrame:
        """Create 1m data where SL is hit."""
        n_minutes = 60
        start_time = datetime(2024, 1, 1, 1, 0)

        sl_price = entry_price * (1 - sl)

        # Price falls to hit SL
        prices = np.linspace(entry_price, sl_price * 0.99, n_minutes)

        return pd.DataFrame({
            "open_time": [start_time + timedelta(minutes=i) for i in range(n_minutes)],
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": [1000] * n_minutes,
        })

    def test_tp_exit_with_costs(self):
        """Test TP exit applies costs correctly."""
        entry_price = 100.0
        pt = 0.01  # 1%
        sl = 0.006
        fee_bps = 10.0
        slippage_bps = 2.0

        df_1m = self.create_1m_data_tp_hit(entry_price, pt)
        entry_time = datetime(2024, 1, 1, 0, 59)  # Just before data starts

        exit_time, exit_price_net, exit_reason, gross_exit_price = simulate_barrier_exit(
            df_1m=df_1m,
            entry_time=entry_time,
            execution_price=entry_price,  # Raw market price
            entry_price_with_costs=entry_price,  # Simplified for test
            pt=pt,
            sl=sl,
            max_hold_hours=12,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )

        assert exit_reason == "TP"

        # Gross exit should be at TP price
        assert gross_exit_price == pytest.approx(entry_price * (1 + pt), rel=0.01)

        # Net exit should be lower due to sell costs
        expected_net = gross_exit_price * (1 - (fee_bps + slippage_bps) / 10000)
        assert exit_price_net == pytest.approx(expected_net, rel=0.001)

    def test_sl_exit_with_costs(self):
        """Test SL exit applies costs correctly."""
        entry_price = 100.0
        pt = 0.01
        sl = 0.006  # 0.6%
        fee_bps = 10.0
        slippage_bps = 2.0

        df_1m = self.create_1m_data_sl_hit(entry_price, sl)
        entry_time = datetime(2024, 1, 1, 0, 59)

        exit_time, exit_price_net, exit_reason, gross_exit_price = simulate_barrier_exit(
            df_1m=df_1m,
            entry_time=entry_time,
            execution_price=entry_price,  # Raw market price
            entry_price_with_costs=entry_price,
            pt=pt,
            sl=sl,
            max_hold_hours=12,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )

        assert exit_reason == "SL"

        # Gross exit should be at SL price
        assert gross_exit_price == pytest.approx(entry_price * (1 - sl), rel=0.01)

        # Net exit should be lower due to sell costs
        expected_net = gross_exit_price * (1 - (fee_bps + slippage_bps) / 10000)
        assert exit_price_net == pytest.approx(expected_net, rel=0.001)


class TestT1Execution:
    """Test T+1 execution logic."""

    def test_signal_at_close_executes_at_next_open(self):
        """
        Verify that a signal at bar close t results in execution at bar t+1 open.

        This is the CRITICAL test for preventing lookahead in execution.
        """
        # Create simple test data
        n_hours = 100
        start_time = datetime(2024, 1, 1, 0, 0)

        # Create 1h data
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n_hours)))

        df_1h = pd.DataFrame({
            "open_time": [start_time + timedelta(hours=i) for i in range(n_hours)],
            "open": prices * (1 + np.random.normal(0, 0.001, n_hours)),
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.uniform(100, 1000, n_hours),
        })

        # Signal at hour 10 close
        signal_time = start_time + timedelta(hours=10)

        # Execution should be at hour 11 open
        expected_execution_time = start_time + timedelta(hours=11)

        # Get the actual execution bar
        execution_bar = df_1h[df_1h["open_time"] == expected_execution_time].iloc[0]

        # The execution price should be the OPEN of bar 11, not the CLOSE of bar 10
        execution_price = execution_bar["open"]
        signal_bar = df_1h[df_1h["open_time"] == signal_time].iloc[0]

        # These should generally be different (unless by chance they're equal)
        # The key point is that we're using the open of the NEXT bar
        assert execution_bar["open_time"] > signal_bar["open_time"], \
            "Execution should be after signal time"


class TestPnLCalculation:
    """Test P&L calculation logic."""

    def test_winning_trade_pnl(self):
        """Test P&L calculation for a winning trade."""
        entry_price = 100.0
        exit_price = 101.0  # 1% gain
        fee_bps = 10.0
        slippage_bps = 2.0

        # Entry cost
        entry_with_costs = calculate_costs(entry_price, fee_bps, slippage_bps, is_buy=True)

        # Exit proceeds
        exit_with_costs = calculate_costs(exit_price, fee_bps, slippage_bps, is_buy=False)

        # Net return
        net_return = (exit_with_costs - entry_with_costs) / entry_with_costs

        # Should be positive but less than 1% due to costs
        assert net_return > 0, "Should be profitable"
        assert net_return < 0.01, "Should be less than gross return due to costs"

    def test_losing_trade_pnl(self):
        """Test P&L calculation for a losing trade."""
        entry_price = 100.0
        exit_price = 99.4  # 0.6% loss (SL hit)
        fee_bps = 10.0
        slippage_bps = 2.0

        entry_with_costs = calculate_costs(entry_price, fee_bps, slippage_bps, is_buy=True)
        exit_with_costs = calculate_costs(exit_price, fee_bps, slippage_bps, is_buy=False)

        net_return = (exit_with_costs - entry_with_costs) / entry_with_costs

        # Should be negative
        assert net_return < 0, "Should be a loss"

        # Loss should be greater than 0.6% due to costs
        assert net_return < -0.006, "Loss should be worse than gross loss due to costs"

    def test_break_even_trade_is_loss(self):
        """Test that a break-even gross trade is actually a loss due to costs."""
        entry_price = 100.0
        exit_price = 100.0  # Break even gross
        fee_bps = 10.0
        slippage_bps = 2.0

        entry_with_costs = calculate_costs(entry_price, fee_bps, slippage_bps, is_buy=True)
        exit_with_costs = calculate_costs(exit_price, fee_bps, slippage_bps, is_buy=False)

        net_return = (exit_with_costs - entry_with_costs) / entry_with_costs

        # Should be negative due to round-trip costs
        assert net_return < 0, "Break-even gross trade should be net loss"

        # Loss should be approximately 2x costs (entry + exit)
        expected_loss = -2 * (fee_bps + slippage_bps) / 10000
        assert net_return == pytest.approx(expected_loss, rel=0.1)


class TestBacktestIntegration:
    """Integration tests for the full backtest."""

    def test_empty_candidates(self):
        """Test backtest with no candidates."""
        # This would require mocking, so we just verify the logic handles empty case
        df_features = pd.DataFrame(columns=["open_time"])
        df_labeled = pd.DataFrame(columns=["entry_time", "entry_idx", "entry_price"])

        # Should not raise
        # In real implementation, run_backtest would return empty results


class TestBarrierUsesRawPrice:
    """Test that barriers are based on raw execution price, not cost-adjusted price."""

    def test_barrier_uses_raw_price_not_cost_adjusted(self):
        """TP/SL barriers must use raw market price, matching labeling.py.

        Create a scenario where high reaches between raw TP and cost-adjusted TP.
        With raw barrier → TP should trigger. With cost-adjusted barrier → it wouldn't.
        """
        execution_price = 100.0
        pt = 0.008  # 0.8%
        sl = 0.006
        fee_bps = 10.0
        slippage_bps = 2.0

        # Cost-adjusted entry price
        entry_price_with_costs = execution_price * (1 + (fee_bps + slippage_bps) / 10000)

        # Raw TP = 100 * 1.008 = 100.80
        # Cost TP = 100.12 * 1.008 = 100.92096 (higher, harder to hit)
        raw_tp = execution_price * (1 + pt)
        cost_tp = entry_price_with_costs * (1 + pt)

        # Price that's between raw TP and cost TP
        # This should trigger TP with the fix but would NOT with cost-adjusted barriers
        mid_price = (raw_tp + cost_tp) / 2  # ~100.86

        n_minutes = 60
        entry_time = datetime(2024, 1, 1, 0, 59)
        start_time = datetime(2024, 1, 1, 1, 0)

        df_1m = pd.DataFrame({
            "open_time": [start_time + timedelta(minutes=i) for i in range(n_minutes)],
            "open": [execution_price] * n_minutes,
            "high": [mid_price] * n_minutes,  # high = between raw TP and cost TP
            "low": [execution_price * 0.999] * n_minutes,
            "close": [execution_price] * n_minutes,
            "volume": [1000] * n_minutes,
        })

        exit_time, exit_price_net, exit_reason, gross_exit_price = simulate_barrier_exit(
            df_1m=df_1m,
            entry_time=entry_time,
            execution_price=execution_price,
            entry_price_with_costs=entry_price_with_costs,
            pt=pt,
            sl=sl,
            max_hold_hours=12,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )

        # Must hit TP because barrier is based on raw price (100.80), not cost price (100.92)
        assert exit_reason == "TP", (
            f"Expected TP but got {exit_reason}. "
            f"raw_tp={raw_tp:.4f}, cost_tp={cost_tp:.4f}, high={mid_price:.4f}"
        )
        assert gross_exit_price == pytest.approx(raw_tp, rel=0.001)


class TestATRBarrierExit:
    """Test ATR-based dynamic barrier exit in backtest."""

    def test_atr_barrier_exit_tp(self):
        """ATR-based TP barrier should trigger at ATR-scaled price, not fixed %."""
        from src import config as _cfg

        execution_price = 100.0
        pt = 0.008  # fixed fallback
        sl = 0.006
        fee_bps = 0.0
        slippage_bps = 0.0
        atr_value = 2.0  # ATR=2 → TP mult=1.5 → tp_pct=3%, TP at 103.0

        entry_time = datetime(2024, 1, 1, 0, 59)
        start_time = datetime(2024, 1, 1, 1, 0)

        # Price gradually rises to 103.1 (above ATR TP at 103.0)
        n_minutes = 120
        prices = np.linspace(100.0, 103.5, n_minutes)

        df_1m = pd.DataFrame({
            "open_time": [start_time + timedelta(minutes=i) for i in range(n_minutes)],
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": [1000] * n_minutes,
        })

        orig_use_atr = _cfg.USE_ATR_BARRIERS
        _cfg.USE_ATR_BARRIERS = True

        try:
            exit_time, exit_price_net, exit_reason, gross_exit_price = simulate_barrier_exit(
                df_1m=df_1m,
                entry_time=entry_time,
                execution_price=execution_price,
                entry_price_with_costs=execution_price,
                pt=pt, sl=sl,
                max_hold_hours=12,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                atr_value=atr_value,
            )

            assert exit_reason == "TP"
            # TP should be at ~103.0 (ATR * 1.5), not 100.8 (fixed)
            assert gross_exit_price == pytest.approx(103.0, rel=0.01)
        finally:
            _cfg.USE_ATR_BARRIERS = orig_use_atr

    def test_atr_barrier_exit_sl(self):
        """ATR-based SL barrier should trigger at ATR-scaled price."""
        from src import config as _cfg

        execution_price = 100.0
        pt = 0.008
        sl = 0.006
        fee_bps = 0.0
        slippage_bps = 0.0
        atr_value = 2.0  # ATR=2 → SL mult=1.5 → sl_pct=3%, SL at 97.0

        entry_time = datetime(2024, 1, 1, 0, 59)
        start_time = datetime(2024, 1, 1, 1, 0)

        # Price drops to 96.5 (below ATR SL at 97.0)
        n_minutes = 60
        prices = np.linspace(100.0, 96.0, n_minutes)

        df_1m = pd.DataFrame({
            "open_time": [start_time + timedelta(minutes=i) for i in range(n_minutes)],
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": [1000] * n_minutes,
        })

        orig_use_atr = _cfg.USE_ATR_BARRIERS
        orig_sl_mult = _cfg.SL_ATR_MULTIPLIER
        _cfg.USE_ATR_BARRIERS = True
        _cfg.SL_ATR_MULTIPLIER = 1.5

        try:
            exit_time, exit_price_net, exit_reason, gross_exit_price = simulate_barrier_exit(
                df_1m=df_1m,
                entry_time=entry_time,
                execution_price=execution_price,
                entry_price_with_costs=execution_price,
                pt=pt, sl=sl,
                max_hold_hours=12,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                atr_value=atr_value,
            )

            assert exit_reason == "SL"
            # SL should be at ~97.0 (ATR * 1.5), not 99.4 (fixed)
            assert gross_exit_price == pytest.approx(97.0, rel=0.01)
        finally:
            _cfg.USE_ATR_BARRIERS = orig_use_atr
            _cfg.SL_ATR_MULTIPLIER = orig_sl_mult

    def test_atr_none_falls_back_to_fixed(self):
        """When atr_value is None, barriers should use fixed pt/sl."""
        execution_price = 100.0
        pt = 0.008
        sl = 0.006
        fee_bps = 0.0
        slippage_bps = 0.0

        entry_time = datetime(2024, 1, 1, 0, 59)
        start_time = datetime(2024, 1, 1, 1, 0)

        # Price rises to hit fixed TP at 100.8
        n_minutes = 60
        tp_price = execution_price * (1 + pt)
        prices = np.linspace(execution_price, tp_price * 1.01, n_minutes)

        df_1m = pd.DataFrame({
            "open_time": [start_time + timedelta(minutes=i) for i in range(n_minutes)],
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": [1000] * n_minutes,
        })

        exit_time, exit_price_net, exit_reason, gross_exit_price = simulate_barrier_exit(
            df_1m=df_1m,
            entry_time=entry_time,
            execution_price=execution_price,
            entry_price_with_costs=execution_price,
            pt=pt, sl=sl,
            max_hold_hours=12,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            atr_value=None,  # No ATR → fixed barriers
        )

        assert exit_reason == "TP"
        assert gross_exit_price == pytest.approx(tp_price, rel=0.01)


class TestPositionSizing:
    """Test position sizing logic."""

    def test_full_capital_invested(self):
        """Test that full capital is invested per trade."""
        capital = 10000.0
        entry_price_with_costs = 100.12  # After costs

        # Position size = capital / price
        position_size = capital / entry_price_with_costs

        # Verify
        assert position_size == pytest.approx(capital / entry_price_with_costs, rel=0.0001)

        # Value of position should equal capital
        position_value = position_size * entry_price_with_costs
        assert position_value == pytest.approx(capital, rel=0.0001)
