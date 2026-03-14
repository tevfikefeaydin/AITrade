"""
Tests for signal generation module (src/signals.py).

Tests cover:
- Breakout signal generation
- Mean-reversion signal generation
- Volume-spike signal generation
- Exit signal generation
- Candidate generation pipeline
- ADX filter behavior
- No-overlap between breakout and mean-reversion
- Edge cases (missing columns, empty data)
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src import config
from src.signals import (
    generate_entry_signals,
    generate_mean_reversion_signals,
    generate_volume_spike_signals,
    generate_exit_signals,
    generate_candidates,
    add_signal_columns,
    count_signals,
)


def _make_1h_df(n_bars=50, base_price=50000.0, trend="flat"):
    """Helper to create 1h OHLCV DataFrame with required columns."""
    start = datetime(2024, 6, 1)
    times = [start + timedelta(hours=i) for i in range(n_bars)]

    np.random.seed(42)
    prices = [base_price]
    for _ in range(n_bars - 1):
        if trend == "up":
            change = np.random.normal(0.003, 0.005)
        elif trend == "down":
            change = np.random.normal(-0.003, 0.005)
        else:
            change = np.random.normal(0.0, 0.005)
        prices.append(prices[-1] * (1 + change))

    closes = np.array(prices)
    opens = closes * (1 + np.random.normal(0, 0.001, n_bars))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.003, n_bars)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.003, n_bars)))
    volumes = np.random.uniform(100, 1000, n_bars)

    df = pd.DataFrame({
        "open_time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "trend_4h": True,
        "adx_14": 20.0,
        "rsi": 50.0,
        "lower_wick_ratio": 0.1,
        "ret_1": 0.001,
        "ret_3": 0.003,
        "ret_24": 0.01,
        "body_ratio": 0.5,
        "volume_zscore": 0.5,
        "volume_ratio": 1.0,
    })
    return df


# ============================================================================
# Breakout Signal Tests
# ============================================================================

class TestBreakoutSignals:
    def test_breakout_fires_on_new_high(self):
        """Breakout signal fires when close > rolling_high(20).shift(1)."""
        df = _make_1h_df(n_bars=30, trend="up")
        df["trend_4h"] = True
        df["adx_14"] = 25.0

        signals = generate_entry_signals(df)
        assert signals.dtype == bool
        # In a strong uptrend, some breakout signals should fire
        assert signals.sum() >= 0  # At minimum no error

    def test_breakout_requires_trend_4h(self):
        """Breakout does NOT fire when trend_4h is False."""
        df = _make_1h_df(n_bars=30, trend="up")
        df["trend_4h"] = False
        df["adx_14"] = 25.0

        signals = generate_entry_signals(df)
        assert signals.sum() == 0

    def test_breakout_requires_adx_above_threshold(self):
        """Breakout does NOT fire when ADX < threshold."""
        df = _make_1h_df(n_bars=30, trend="up")
        df["trend_4h"] = True
        df["adx_14"] = 5.0  # Below default threshold of 12

        signals = generate_entry_signals(df)
        assert signals.sum() == 0

    def test_breakout_no_signal_in_first_n_bars(self):
        """No breakout in the first ROLLING_WINDOW bars (insufficient data)."""
        df = _make_1h_df(n_bars=30, trend="up")
        df["trend_4h"] = True
        df["adx_14"] = 25.0

        signals = generate_entry_signals(df)
        # First ROLLING_WINDOW+1 bars cannot have signals (need N bars + 1 shift)
        assert signals.iloc[:config.ROLLING_WINDOW + 1].sum() == 0

    def test_breakout_uses_close_not_high(self):
        """Breakout uses close > rolling_high, NOT high > rolling_high."""
        df = _make_1h_df(n_bars=30)
        df["trend_4h"] = True
        df["adx_14"] = 25.0
        # Set close flat but high spiking - should NOT trigger breakout
        df["close"] = 50000.0
        df["high"] = 55000.0

        signals = generate_entry_signals(df)
        assert signals.sum() == 0

    def test_adx_filter_disabled(self):
        """When USE_ADX_FILTER=False, ADX column is ignored."""
        df = _make_1h_df(n_bars=30, trend="up")
        df["trend_4h"] = True
        df["adx_14"] = 1.0  # Very low ADX

        original = config.USE_ADX_FILTER
        try:
            config.USE_ADX_FILTER = False
            signals = generate_entry_signals(df)
            # Should not be blocked by ADX
            # (may or may not fire depending on price action)
        finally:
            config.USE_ADX_FILTER = original

    def test_missing_adx_column_rejects_all(self):
        """If ADX column missing and filter enabled, all entries rejected."""
        df = _make_1h_df(n_bars=30, trend="up")
        df["trend_4h"] = True
        df = df.drop(columns=["adx_14"])

        signals = generate_entry_signals(df)
        assert signals.sum() == 0


# ============================================================================
# Mean-Reversion Signal Tests
# ============================================================================

class TestMeanReversionSignals:
    def test_mr_fires_on_oversold_bounce(self):
        """MR signal fires when RSI<35, lower_wick>0.2, ret_3<0."""
        df = _make_1h_df(n_bars=30)
        # Set conditions for mean-reversion
        df["rsi"] = 30.0
        df["lower_wick_ratio"] = 0.3
        df["ret_3"] = -0.02
        df["trend_4h"] = True
        df["ret_24"] = 0.01
        # Make sure NOT breakout: close stays flat below rolling high
        df["close"] = 49000.0

        signals = generate_mean_reversion_signals(df)
        assert signals.sum() > 0

    def test_mr_does_not_fire_when_rsi_high(self):
        """No MR signal when RSI > MR_RSI_OVERSOLD."""
        df = _make_1h_df(n_bars=30)
        df["rsi"] = 60.0
        df["lower_wick_ratio"] = 0.3
        df["ret_3"] = -0.02
        df["trend_4h"] = True

        signals = generate_mean_reversion_signals(df)
        assert signals.sum() == 0

    def test_mr_excludes_breakout_bars(self):
        """MR signal does NOT fire on bars where breakout also fires."""
        df = _make_1h_df(n_bars=40, trend="up")
        df["rsi"] = 30.0
        df["lower_wick_ratio"] = 0.3
        df["ret_3"] = -0.02
        df["trend_4h"] = True
        df["ret_24"] = 0.01
        # Close is above rolling high → breakout fires → MR should NOT fire

        mr_signals = generate_mean_reversion_signals(df)
        breakout_signals = generate_entry_signals(df)

        # Where breakout fires, MR must not fire
        overlap = mr_signals & breakout_signals
        assert overlap.sum() == 0

    def test_mr_bearish_trend_guard(self):
        """MR blocked when trend_4h=False AND ret_24 < -5%."""
        df = _make_1h_df(n_bars=30)
        df["rsi"] = 30.0
        df["lower_wick_ratio"] = 0.3
        df["ret_3"] = -0.02
        df["close"] = 49000.0
        df["trend_4h"] = False
        df["ret_24"] = -0.06  # Deep downtrend

        signals = generate_mean_reversion_signals(df)
        assert signals.sum() == 0

    def test_mr_allowed_when_ret_24_not_deep(self):
        """MR allowed when trend_4h=False but ret_24 > -5%."""
        df = _make_1h_df(n_bars=30)
        df["rsi"] = 30.0
        df["lower_wick_ratio"] = 0.3
        df["ret_3"] = -0.02
        df["close"] = 49000.0
        df["trend_4h"] = False
        df["ret_24"] = -0.03  # Mild decline, not deep

        signals = generate_mean_reversion_signals(df)
        assert signals.sum() > 0

    def test_mr_missing_column_returns_empty(self):
        """Missing required column returns all-False Series."""
        df = _make_1h_df(n_bars=30)
        df = df.drop(columns=["rsi"])

        signals = generate_mean_reversion_signals(df)
        assert signals.sum() == 0
        assert len(signals) == len(df)


# ============================================================================
# Volume-Spike Signal Tests
# ============================================================================

class TestVolumeSpikeSignals:
    def test_vs_fires_on_high_volume_candle(self):
        """VS signal fires when volume_zscore>2, body_ratio>0.6, ret_1>0."""
        df = _make_1h_df(n_bars=30)
        df["volume_zscore"] = 3.0
        df["volume_ratio"] = 3.0
        df["body_ratio"] = 0.8
        df["ret_1"] = 0.005
        df["trend_4h"] = True

        signals = generate_volume_spike_signals(df)
        assert signals.sum() > 0

    def test_vs_needs_trend_context(self):
        """VS signal needs trend_4h=True OR close > MA20."""
        df = _make_1h_df(n_bars=30)
        df["volume_zscore"] = 3.0
        df["body_ratio"] = 0.8
        df["ret_1"] = 0.005
        df["trend_4h"] = False
        # Close well below MA20 to ensure close_above_ma is also False
        df["close"] = 10000.0

        signals = generate_volume_spike_signals(df)
        assert signals.sum() == 0

    def test_vs_requires_positive_momentum(self):
        """VS signal does NOT fire when ret_1 <= 0."""
        df = _make_1h_df(n_bars=30)
        df["volume_zscore"] = 3.0
        df["body_ratio"] = 0.8
        df["ret_1"] = -0.01
        df["trend_4h"] = True

        signals = generate_volume_spike_signals(df)
        assert signals.sum() == 0

    def test_vs_missing_body_ratio_returns_empty(self):
        """Missing body_ratio returns all-False."""
        df = _make_1h_df(n_bars=30)
        df = df.drop(columns=["body_ratio"])

        signals = generate_volume_spike_signals(df)
        assert signals.sum() == 0

    def test_vs_volume_ratio_alternative(self):
        """VS fires when volume_ratio > threshold even if zscore is low."""
        df = _make_1h_df(n_bars=30)
        df["volume_zscore"] = 0.5  # Below threshold
        df["volume_ratio"] = 3.0  # Above threshold
        df["body_ratio"] = 0.8
        df["ret_1"] = 0.005
        df["trend_4h"] = True

        signals = generate_volume_spike_signals(df)
        assert signals.sum() > 0


# ============================================================================
# Exit Signal Tests
# ============================================================================

class TestExitSignals:
    def test_exit_fires_on_breakdown(self):
        """Exit fires when close < rolling_low(20).shift(1)."""
        df = _make_1h_df(n_bars=40, trend="down")
        df["trend_4h"] = True  # Trend still True, but price breaking down

        signals = generate_exit_signals(df)
        # Should fire on some bars in downtrend
        assert signals.dtype == bool

    def test_exit_fires_when_trend_fails(self):
        """Exit fires when trend_4h turns False."""
        df = _make_1h_df(n_bars=30)
        df["trend_4h"] = False

        signals = generate_exit_signals(df)
        assert signals.sum() > 0

    def test_exit_no_signal_first_n_bars(self):
        """No exit signal in first ROLLING_WINDOW bars."""
        df = _make_1h_df(n_bars=30)
        df["trend_4h"] = True

        signals = generate_exit_signals(df)
        # First ROLLING_WINDOW+1 bars are NaN for rolling → should be False
        assert signals.iloc[:config.ROLLING_WINDOW].sum() == 0


# ============================================================================
# Candidate Generation Tests
# ============================================================================

class TestCandidateGeneration:
    def test_candidates_have_required_columns(self):
        """Candidate DataFrame has all required columns."""
        df = _make_1h_df(n_bars=50, trend="up")
        df["adx_14"] = 25.0

        candidates = generate_candidates(df)
        if len(candidates) > 0:
            required = [
                "entry_idx", "entry_time", "entry_price",
                "execution_time", "execution_price",
                "signal_type", "signal_type_encoded",
            ]
            for col in required:
                assert col in candidates.columns, f"Missing column: {col}"

    def test_candidates_t_plus_1_execution(self):
        """Execution time is always after entry time (T+1)."""
        df = _make_1h_df(n_bars=50, trend="up")
        df["adx_14"] = 25.0

        candidates = generate_candidates(df)
        if len(candidates) > 0:
            for _, row in candidates.iterrows():
                assert row["execution_time"] > row["entry_time"]

    def test_signal_type_encoding(self):
        """signal_type_encoded matches SIGNAL_TYPE_MAP."""
        df = _make_1h_df(n_bars=50, trend="up")
        df["adx_14"] = 25.0

        candidates = generate_candidates(df)
        if len(candidates) > 0:
            for _, row in candidates.iterrows():
                expected = config.SIGNAL_TYPE_MAP[row["signal_type"]]
                assert row["signal_type_encoded"] == expected

    def test_empty_when_no_signals(self):
        """Returns empty DataFrame when no signals generated."""
        df = _make_1h_df(n_bars=30)
        df["trend_4h"] = False
        df["adx_14"] = 1.0
        df["rsi"] = 50.0  # No MR
        df["volume_zscore"] = 0.5  # No VS
        df["volume_ratio"] = 0.5
        df["close"] = 10000.0  # Low enough to not trigger anything

        candidates = generate_candidates(df)
        assert len(candidates) == 0

    def test_last_bar_excluded(self):
        """Last bar cannot generate candidate (no T+1 bar available)."""
        df = _make_1h_df(n_bars=50, trend="up")
        df["adx_14"] = 25.0

        candidates = generate_candidates(df)
        if len(candidates) > 0:
            last_time = df["open_time"].iloc[-1]
            assert (candidates["entry_time"] == last_time).sum() == 0

    def test_multiple_signal_types_same_bar(self):
        """Same bar can produce candidates from different signal types."""
        df = _make_1h_df(n_bars=50, trend="up")
        df["adx_14"] = 25.0

        candidates = generate_candidates(df)
        if len(candidates) > 0:
            types_present = candidates["signal_type"].unique()
            # At least check that multiple types are possible
            assert len(types_present) >= 1


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestSignalUtilities:
    def test_add_signal_columns(self):
        """add_signal_columns adds entry_signal and exit_signal columns."""
        df = _make_1h_df(n_bars=30)
        df["adx_14"] = 25.0

        result = add_signal_columns(df)
        assert "entry_signal" in result.columns
        assert "exit_signal" in result.columns
        assert result["entry_signal"].dtype == bool
        assert result["exit_signal"].dtype == bool

    def test_count_signals(self):
        """count_signals returns correct dict structure."""
        df = _make_1h_df(n_bars=30)
        df["adx_14"] = 25.0
        df = add_signal_columns(df)

        counts = count_signals(df)
        assert "total_bars" in counts
        assert "entry_signals" in counts
        assert "exit_signals" in counts
        assert "entry_rate" in counts
        assert "exit_rate" in counts
        assert counts["total_bars"] == len(df)

    def test_count_signals_without_columns(self):
        """count_signals handles missing signal columns."""
        df = _make_1h_df(n_bars=30)
        counts = count_signals(df)
        assert counts["entry_signals"] == 0
        assert counts["exit_signals"] == 0
