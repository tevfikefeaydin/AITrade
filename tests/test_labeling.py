"""
Tests for triple-barrier labeling.

Tests the labeling logic with synthetic price series where
the outcome is known in advance.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.labeling import find_barrier_touch, label_candidates


def create_1m_data_reaching_tp(entry_price: float = 100.0, pt: float = 0.008) -> pd.DataFrame:
    """Create 1m data that reaches take-profit."""
    # Price gradually moves up and hits TP
    n_minutes = 60  # 1 hour of data
    start_time = datetime(2024, 1, 1, 0, 0)

    tp_price = entry_price * (1 + pt)

    # Price moves from entry to TP linearly
    prices = np.linspace(entry_price, tp_price * 1.01, n_minutes)

    df = pd.DataFrame({
        "open_time": [start_time + timedelta(minutes=i) for i in range(n_minutes)],
        "open": prices,
        "high": prices * 1.001,  # Slightly above
        "low": prices * 0.999,  # Slightly below
        "close": prices,
        "volume": [1000] * n_minutes,
    })

    return df


def create_1m_data_reaching_sl(entry_price: float = 100.0, sl: float = 0.006) -> pd.DataFrame:
    """Create 1m data that reaches stop-loss."""
    n_minutes = 60
    start_time = datetime(2024, 1, 1, 0, 0)

    sl_price = entry_price * (1 - sl)

    # Price moves from entry to SL linearly
    prices = np.linspace(entry_price, sl_price * 0.99, n_minutes)

    df = pd.DataFrame({
        "open_time": [start_time + timedelta(minutes=i) for i in range(n_minutes)],
        "open": prices,
        "high": prices * 1.001,
        "low": prices * 0.999,
        "close": prices,
        "volume": [1000] * n_minutes,
    })

    return df


def create_1m_data_timeout(entry_price: float = 100.0) -> pd.DataFrame:
    """Create 1m data that neither hits TP nor SL (timeout)."""
    n_minutes = 60 * 24  # 24 hours of data
    start_time = datetime(2024, 1, 1, 0, 0)

    # Price oscillates around entry without hitting barriers
    t = np.arange(n_minutes)
    prices = entry_price + 0.001 * entry_price * np.sin(t / 30)  # Small oscillation

    df = pd.DataFrame({
        "open_time": [start_time + timedelta(minutes=i) for i in range(n_minutes)],
        "open": prices,
        "high": prices * 1.0001,  # Very small wicks
        "low": prices * 0.9999,
        "close": prices,
        "volume": [1000] * n_minutes,
    })

    return df


class TestBarrierTouch:
    """Test the barrier touch detection logic."""

    def test_tp_hit(self):
        """Test that TP hit is correctly detected."""
        entry_price = 100.0
        pt = 0.008
        sl = 0.006

        df_1m = create_1m_data_reaching_tp(entry_price, pt)
        entry_time = datetime(2024, 1, 1, 0, 0) - timedelta(minutes=1)  # Before data starts

        label, exit_time, exit_price, exit_reason = find_barrier_touch(
            df_1m=df_1m,
            entry_time=entry_time,
            entry_price=entry_price,
            pt=pt,
            sl=sl,
            max_hold_hours=12,
        )

        assert label == 1, "Should be labeled as successful (1) when TP hit"
        assert exit_reason == "TP", "Exit reason should be TP"
        assert exit_price == pytest.approx(entry_price * (1 + pt), rel=0.001)

    def test_sl_hit(self):
        """Test that SL hit is correctly detected."""
        entry_price = 100.0
        pt = 0.008
        sl = 0.006

        df_1m = create_1m_data_reaching_sl(entry_price, sl)
        entry_time = datetime(2024, 1, 1, 0, 0) - timedelta(minutes=1)

        label, exit_time, exit_price, exit_reason = find_barrier_touch(
            df_1m=df_1m,
            entry_time=entry_time,
            entry_price=entry_price,
            pt=pt,
            sl=sl,
            max_hold_hours=12,
        )

        assert label == 0, "Should be labeled as unsuccessful (0) when SL hit"
        assert exit_reason == "SL", "Exit reason should be SL"
        assert exit_price == pytest.approx(entry_price * (1 - sl), rel=0.001)

    def test_timeout(self):
        """Test that timeout is correctly detected."""
        entry_price = 100.0
        pt = 0.008
        sl = 0.006
        max_hold = 12

        df_1m = create_1m_data_timeout(entry_price)
        entry_time = datetime(2024, 1, 1, 0, 0) - timedelta(minutes=1)

        label, exit_time, exit_price, exit_reason = find_barrier_touch(
            df_1m=df_1m,
            entry_time=entry_time,
            entry_price=entry_price,
            pt=pt,
            sl=sl,
            max_hold_hours=max_hold,
        )

        # Fractional timeout: label is between 0 and 1 based on exit return vs barriers
        assert 0.0 <= label <= 1.0, f"Timeout label should be fractional [0,1], got {label}"
        assert exit_reason == "TIMEOUT", "Exit reason should be TIMEOUT"

    def test_tp_before_sl(self):
        """Test that TP is registered when hit before SL."""
        entry_price = 100.0
        pt = 0.008
        sl = 0.006

        # Create data where TP is hit first
        n_minutes = 100
        start_time = datetime(2024, 1, 1, 0, 0)

        # Price goes up first, then down
        prices_up = np.linspace(entry_price, entry_price * (1 + pt * 1.5), 50)
        prices_down = np.linspace(entry_price * (1 + pt * 1.5), entry_price * (1 - sl * 1.5), 50)
        prices = np.concatenate([prices_up, prices_down])

        df_1m = pd.DataFrame({
            "open_time": [start_time + timedelta(minutes=i) for i in range(n_minutes)],
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": [1000] * n_minutes,
        })

        entry_time = start_time - timedelta(minutes=1)

        label, _, _, exit_reason = find_barrier_touch(
            df_1m=df_1m,
            entry_time=entry_time,
            entry_price=entry_price,
            pt=pt,
            sl=sl,
            max_hold_hours=12,
        )

        assert label == 1, "Should be TP since it was hit first"
        assert exit_reason == "TP"

    def test_sl_before_tp(self):
        """Test that SL is registered when hit before TP."""
        entry_price = 100.0
        pt = 0.008
        sl = 0.006

        # Create data where SL is hit first
        n_minutes = 100
        start_time = datetime(2024, 1, 1, 0, 0)

        # Price goes down first, then up
        prices_down = np.linspace(entry_price, entry_price * (1 - sl * 1.5), 50)
        prices_up = np.linspace(entry_price * (1 - sl * 1.5), entry_price * (1 + pt * 1.5), 50)
        prices = np.concatenate([prices_down, prices_up])

        df_1m = pd.DataFrame({
            "open_time": [start_time + timedelta(minutes=i) for i in range(n_minutes)],
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": [1000] * n_minutes,
        })

        entry_time = start_time - timedelta(minutes=1)

        label, _, _, exit_reason = find_barrier_touch(
            df_1m=df_1m,
            entry_time=entry_time,
            entry_price=entry_price,
            pt=pt,
            sl=sl,
            max_hold_hours=12,
        )

        assert label == 0, "Should be SL since it was hit first"
        assert exit_reason == "SL"


class TestLabelCandidates:
    """Test the full labeling pipeline."""

    def test_label_multiple_candidates(self):
        """Test labeling multiple candidates."""
        entry_price = 100.0
        pt = 0.008
        sl = 0.006

        # Create 1m data with multiple scenarios
        n_minutes = 60 * 24 * 3  # 3 days
        start_time = datetime(2024, 1, 1, 0, 0)

        np.random.seed(42)
        returns = np.random.normal(0, 0.0005, n_minutes)
        prices = entry_price * np.exp(np.cumsum(returns))

        df_1m = pd.DataFrame({
            "open_time": [start_time + timedelta(minutes=i) for i in range(n_minutes)],
            "open": prices,
            "high": prices * (1 + np.abs(np.random.normal(0, 0.001, n_minutes))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.001, n_minutes))),
            "close": prices * (1 + np.random.normal(0, 0.0001, n_minutes)),
            "volume": np.random.uniform(100, 1000, n_minutes),
        })

        # Create candidates at different times
        df_candidates = pd.DataFrame({
            "entry_idx": [0, 1, 2],
            "entry_time": [
                start_time + timedelta(hours=1),
                start_time + timedelta(hours=24),
                start_time + timedelta(hours=48),
            ],
            "entry_price": [
                prices[60],  # Price at hour 1
                prices[60 * 24],  # Price at hour 24
                prices[60 * 48],  # Price at hour 48
            ],
        })

        df_labeled = label_candidates(df_candidates, df_1m, pt, sl, max_hold=12)

        assert len(df_labeled) == 3
        assert "label" in df_labeled.columns
        assert "exit_time" in df_labeled.columns
        assert "exit_reason" in df_labeled.columns

        # All labels should be 0 or 1
        assert all(df_labeled["label"].isin([0, 1]))

        # Exit reasons should be valid
        assert all(df_labeled["exit_reason"].isin(["TP", "SL", "TIMEOUT"]))

    def test_empty_candidates(self):
        """Test handling of empty candidates DataFrame."""
        df_1m = create_1m_data_reaching_tp()
        df_candidates = pd.DataFrame(columns=["entry_idx", "entry_time", "entry_price"])

        df_labeled = label_candidates(df_candidates, df_1m)

        assert len(df_labeled) == 0

    def test_label_statistics(self):
        """Test that label statistics are computed correctly."""
        from src.labeling import compute_label_statistics

        df_labeled = pd.DataFrame({
            "entry_idx": [0, 1, 2, 3, 4],
            "label": [1, 1, 0, 0, 0],
            "exit_reason": ["TP", "TP", "SL", "SL", "TIMEOUT"],
        })

        stats = compute_label_statistics(df_labeled)

        assert stats["total_candidates"] == 5
        assert stats["positive_rate"] == 0.4  # 2/5
        assert stats["tp_count"] == 2
        assert stats["sl_count"] == 2
        assert stats["timeout_count"] == 1


class TestEdgeCases:
    """Test edge cases in labeling."""

    def test_no_data_after_entry(self):
        """Test handling when there's no data after entry time."""
        entry_price = 100.0
        entry_time = datetime(2024, 1, 1, 0, 0)

        # Create data that ends before entry time
        df_1m = pd.DataFrame({
            "open_time": [entry_time - timedelta(minutes=i) for i in range(60, 0, -1)],
            "open": [100.0] * 60,
            "high": [101.0] * 60,
            "low": [99.0] * 60,
            "close": [100.0] * 60,
            "volume": [1000] * 60,
        })

        label, exit_time, exit_price, exit_reason = find_barrier_touch(
            df_1m=df_1m,
            entry_time=entry_time,
            entry_price=entry_price,
            pt=0.008,
            sl=0.006,
            max_hold_hours=12,
        )

        # Should return timeout with entry price
        assert label == 0
        assert exit_reason == "TIMEOUT"

    def test_exact_barrier_touch(self):
        """Test when price exactly touches a barrier."""
        entry_price = 100.0
        pt = 0.01  # 1%
        tp_price = entry_price * (1 + pt)

        # Create data where high exactly equals TP price
        df_1m = pd.DataFrame({
            "open_time": [datetime(2024, 1, 1, 0, i) for i in range(10)],
            "open": [100.0] * 10,
            "high": [100.0] * 5 + [tp_price] * 5,  # High exactly at TP
            "low": [99.5] * 10,
            "close": [100.0] * 10,
            "volume": [1000] * 10,
        })

        entry_time = datetime(2024, 1, 1, 0, 0) - timedelta(minutes=1)

        label, _, _, exit_reason = find_barrier_touch(
            df_1m=df_1m,
            entry_time=entry_time,
            entry_price=entry_price,
            pt=pt,
            sl=0.006,
            max_hold_hours=12,
        )

        assert label == 1, "Exact touch should count as TP hit"
        assert exit_reason == "TP"

    def test_atr_barrier_widens_tp_and_sl(self):
        """Test that ATR-based barriers produce wider TP/SL than fixed %.

        With ATR=2.0 on entry_price=100, TP mult=1.5 → tp_pct = 3.0/100 = 0.03 (3%)
        which is wider than fixed pt=0.008 (0.8%).
        SL mult=1.0 → sl_pct = 2.0/100 = 0.02 (2%) wider than fixed sl=0.006.
        The wider TP means a bar reaching 100.8 (fixed TP) should NOT trigger TP.
        """
        from src import config as _cfg

        entry_price = 100.0
        pt = 0.008  # fixed fallback
        sl = 0.006
        atr_value = 2.0  # ATR = 2.0 → tp_pct=3%, sl_pct=2%

        # Price rises to 100.8 (just at fixed TP), stays above SL
        n_minutes = 60
        start_time = datetime(2024, 1, 1, 0, 0)
        prices = np.linspace(entry_price, entry_price * 1.009, n_minutes)

        df_1m = pd.DataFrame({
            "open_time": [start_time + timedelta(minutes=i) for i in range(n_minutes)],
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": [1000] * n_minutes,
        })

        entry_time = start_time - timedelta(minutes=1)

        # Save and set ATR config
        orig_use_atr = _cfg.USE_ATR_BARRIERS
        _cfg.USE_ATR_BARRIERS = True

        try:
            # With ATR: TP is at 103.0, so price at 100.9 should NOT hit TP → TIMEOUT
            label_atr, _, _, reason_atr = find_barrier_touch(
                df_1m=df_1m,
                entry_time=entry_time,
                entry_price=entry_price,
                pt=pt, sl=sl,
                max_hold_hours=1,  # short hold so we get TIMEOUT
                atr_value=atr_value,
            )
            assert reason_atr == "TIMEOUT", f"ATR barrier should be wider, got {reason_atr}"

            # Without ATR: TP is at 100.8, price should hit TP
            label_fixed, _, _, reason_fixed = find_barrier_touch(
                df_1m=df_1m,
                entry_time=entry_time,
                entry_price=entry_price,
                pt=pt, sl=sl,
                max_hold_hours=1,
                atr_value=None,
            )
            assert reason_fixed == "TP", f"Fixed barrier should hit TP, got {reason_fixed}"
        finally:
            _cfg.USE_ATR_BARRIERS = orig_use_atr
