"""
Tests for wick formula correctness.

Tests candlestick wick calculations with known candle patterns:
- Doji (open == close)
- Hammer (long lower wick)
- Inverted hammer (long upper wick)
- Marubozu (no wicks)
- Standard candles
"""

import pytest
import pandas as pd
import numpy as np

from src.features import compute_wick_features


class TestWickFormulas:
    """Test wick calculation correctness."""

    def test_bullish_candle(self):
        """Test wick calculation for a bullish (green) candle."""
        df = pd.DataFrame({
            "open": [100.0],
            "high": [110.0],
            "low": [95.0],
            "close": [105.0],
        })

        result = compute_wick_features(df)

        # Upper wick: high - max(open, close) = 110 - 105 = 5
        assert result["upper_wick"].iloc[0] == 5.0

        # Lower wick: min(open, close) - low = 100 - 95 = 5
        assert result["lower_wick"].iloc[0] == 5.0

        # Body: |close - open| = |105 - 100| = 5
        assert result["body"].iloc[0] == 5.0

        # Range: high - low = 110 - 95 = 15
        assert result["range"].iloc[0] == 15.0

    def test_bearish_candle(self):
        """Test wick calculation for a bearish (red) candle."""
        df = pd.DataFrame({
            "open": [105.0],
            "high": [110.0],
            "low": [95.0],
            "close": [100.0],
        })

        result = compute_wick_features(df)

        # Upper wick: high - max(open, close) = 110 - 105 = 5
        assert result["upper_wick"].iloc[0] == 5.0

        # Lower wick: min(open, close) - low = 100 - 95 = 5
        assert result["lower_wick"].iloc[0] == 5.0

        # Body: |close - open| = |100 - 105| = 5
        assert result["body"].iloc[0] == 5.0

    def test_doji_candle(self):
        """Test wick calculation for a doji (open == close)."""
        df = pd.DataFrame({
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [100.0],
        })

        result = compute_wick_features(df)

        # Upper wick: high - max(open, close) = 105 - 100 = 5
        assert result["upper_wick"].iloc[0] == 5.0

        # Lower wick: min(open, close) - low = 100 - 95 = 5
        assert result["lower_wick"].iloc[0] == 5.0

        # Body: |close - open| = 0
        assert result["body"].iloc[0] == 0.0

    def test_hammer_candle(self):
        """Test wick calculation for a hammer (long lower wick)."""
        df = pd.DataFrame({
            "open": [100.0],
            "high": [101.0],
            "low": [90.0],
            "close": [100.5],
        })

        result = compute_wick_features(df)

        # Upper wick: 101 - 100.5 = 0.5
        assert result["upper_wick"].iloc[0] == 0.5

        # Lower wick: 100 - 90 = 10
        assert result["lower_wick"].iloc[0] == 10.0

        # Lower wick should be much larger than upper wick
        assert result["lower_wick"].iloc[0] > result["upper_wick"].iloc[0] * 10

    def test_inverted_hammer(self):
        """Test wick calculation for an inverted hammer (long upper wick)."""
        df = pd.DataFrame({
            "open": [100.0],
            "high": [110.0],
            "low": [99.5],
            "close": [100.5],
        })

        result = compute_wick_features(df)

        # Upper wick: 110 - 100.5 = 9.5
        assert result["upper_wick"].iloc[0] == 9.5

        # Lower wick: 100 - 99.5 = 0.5
        assert result["lower_wick"].iloc[0] == 0.5

        # Upper wick should be much larger than lower wick
        assert result["upper_wick"].iloc[0] > result["lower_wick"].iloc[0] * 10

    def test_marubozu_candle(self):
        """Test wick calculation for a marubozu (no wicks)."""
        df = pd.DataFrame({
            "open": [100.0],
            "high": [105.0],
            "low": [100.0],
            "close": [105.0],
        })

        result = compute_wick_features(df)

        # Upper wick: 105 - 105 = 0
        assert result["upper_wick"].iloc[0] == 0.0

        # Lower wick: 100 - 100 = 0
        assert result["lower_wick"].iloc[0] == 0.0

        # Body equals range
        assert result["body"].iloc[0] == result["range"].iloc[0]

    def test_wick_ratios(self):
        """Test that wick ratios are calculated correctly."""
        df = pd.DataFrame({
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [100.0],
        })

        result = compute_wick_features(df)

        # Range is 20
        assert result["range"].iloc[0] == 20.0

        # Upper wick is 10, ratio = 10/20 = 0.5
        assert result["upper_wick_ratio"].iloc[0] == 0.5

        # Lower wick is 10, ratio = 10/20 = 0.5
        assert result["lower_wick_ratio"].iloc[0] == 0.5

        # Body is 0, ratio = 0/20 = 0
        assert result["body_ratio"].iloc[0] == 0.0

        # Ratios should sum to 1
        total_ratio = (
            result["upper_wick_ratio"].iloc[0]
            + result["lower_wick_ratio"].iloc[0]
            + result["body_ratio"].iloc[0]
        )
        assert abs(total_ratio - 1.0) < 0.001

    def test_zero_range_handling(self):
        """Test handling of zero range (flat candle)."""
        df = pd.DataFrame({
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
        })

        result = compute_wick_features(df)

        # All values should be 0
        assert result["upper_wick"].iloc[0] == 0.0
        assert result["lower_wick"].iloc[0] == 0.0
        assert result["body"].iloc[0] == 0.0
        assert result["range"].iloc[0] == 0.0

        # Ratios should be 0 (safe divide)
        assert result["upper_wick_ratio"].iloc[0] == 0.0
        assert result["lower_wick_ratio"].iloc[0] == 0.0
        assert result["body_ratio"].iloc[0] == 0.0

    def test_multiple_candles(self):
        """Test wick calculation for multiple candles."""
        df = pd.DataFrame({
            "open": [100.0, 105.0, 100.0],
            "high": [110.0, 110.0, 105.0],
            "low": [95.0, 100.0, 95.0],
            "close": [105.0, 102.0, 100.0],
        })

        result = compute_wick_features(df)

        assert len(result) == 3
        assert all(result["upper_wick"] >= 0)
        assert all(result["lower_wick"] >= 0)
        assert all(result["body"] >= 0)
        assert all(result["range"] >= 0)

    def test_wicks_never_negative(self):
        """Ensure wicks are never negative regardless of input."""
        # Random candle data
        np.random.seed(42)
        n = 100
        opens = np.random.uniform(95, 105, n)
        closes = np.random.uniform(95, 105, n)
        highs = np.maximum(opens, closes) + np.random.uniform(0, 5, n)
        lows = np.minimum(opens, closes) - np.random.uniform(0, 5, n)

        df = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        })

        result = compute_wick_features(df)

        assert all(result["upper_wick"] >= 0), "Upper wick should never be negative"
        assert all(result["lower_wick"] >= 0), "Lower wick should never be negative"
        assert all(result["body"] >= 0), "Body should never be negative"
        assert all(result["range"] >= 0), "Range should never be negative"
