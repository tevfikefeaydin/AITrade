"""
Tests for resampling module (src/resample.py).

Tests cover:
- 1m -> 1h resampling (OHLCV aggregation correctness)
- 1h -> 4h resampling
- Intrabar feature computation
- 4h-to-1h alignment (no lookahead via available_time)
- Edge cases
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.resample import (
    resample_ohlcv,
    resample_1h,
    resample_4h,
    compute_intrabar_features,
    build_multi_timeframe_data,
    align_4h_to_1h,
)


def _make_1m_df(n_hours=4):
    """Create n_hours of 1m OHLCV data."""
    n_bars = n_hours * 60
    start = pd.Timestamp("2024-01-01", tz="UTC")
    times = [start + pd.Timedelta(minutes=i) for i in range(n_bars)]

    np.random.seed(42)
    base = 50000.0
    closes = [base]
    for _ in range(n_bars - 1):
        closes.append(closes[-1] * (1 + np.random.normal(0, 0.0005)))
    closes = np.array(closes)

    opens = closes * (1 + np.random.normal(0, 0.0001, n_bars))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.0003, n_bars)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.0003, n_bars)))

    return pd.DataFrame({
        "open_time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.random.uniform(10, 100, n_bars),
    })


# ============================================================================
# Basic Resampling
# ============================================================================

class TestResampleOHLCV:
    def test_1m_to_1h_bar_count(self):
        """4 hours of 1m data produces 4 1h bars."""
        df_1m = _make_1m_df(n_hours=4)
        df_1h = resample_1h(df_1m)
        assert len(df_1h) == 4

    def test_1h_open_is_first_1m_open(self):
        """1h open == first 1m open of that hour."""
        df_1m = _make_1m_df(n_hours=2)
        df_1h = resample_1h(df_1m)

        first_hour_1m = df_1m[df_1m["open_time"] < pd.Timestamp("2024-01-01 01:00", tz="UTC")]
        assert abs(df_1h.iloc[0]["open"] - first_hour_1m.iloc[0]["open"]) < 0.01

    def test_1h_close_is_last_1m_close(self):
        """1h close == last 1m close of that hour."""
        df_1m = _make_1m_df(n_hours=2)
        df_1h = resample_1h(df_1m)

        first_hour_1m = df_1m[df_1m["open_time"] < pd.Timestamp("2024-01-01 01:00", tz="UTC")]
        assert abs(df_1h.iloc[0]["close"] - first_hour_1m.iloc[-1]["close"]) < 0.01

    def test_1h_high_is_max_1m_high(self):
        """1h high == max of all 1m highs in that hour."""
        df_1m = _make_1m_df(n_hours=2)
        df_1h = resample_1h(df_1m)

        first_hour_1m = df_1m[df_1m["open_time"] < pd.Timestamp("2024-01-01 01:00", tz="UTC")]
        assert abs(df_1h.iloc[0]["high"] - first_hour_1m["high"].max()) < 0.01

    def test_1h_low_is_min_1m_low(self):
        """1h low == min of all 1m lows in that hour."""
        df_1m = _make_1m_df(n_hours=2)
        df_1h = resample_1h(df_1m)

        first_hour_1m = df_1m[df_1m["open_time"] < pd.Timestamp("2024-01-01 01:00", tz="UTC")]
        assert abs(df_1h.iloc[0]["low"] - first_hour_1m["low"].min()) < 0.01

    def test_1h_volume_is_sum(self):
        """1h volume == sum of all 1m volumes in that hour."""
        df_1m = _make_1m_df(n_hours=2)
        df_1h = resample_1h(df_1m)

        first_hour_1m = df_1m[df_1m["open_time"] < pd.Timestamp("2024-01-01 01:00", tz="UTC")]
        assert abs(df_1h.iloc[0]["volume"] - first_hour_1m["volume"].sum()) < 0.01

    def test_4h_bar_count(self):
        """8 hours of 1m data -> 8 1h bars -> 2 4h bars."""
        df_1m = _make_1m_df(n_hours=8)
        df_1h = resample_1h(df_1m)
        df_4h = resample_4h(df_1h)
        assert len(df_4h) == 2

    def test_missing_open_time_raises(self):
        """Missing open_time column raises ValueError."""
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="open_time"):
            resample_ohlcv(df, "1h")

    def test_columns_preserved(self):
        """Output has same OHLCV columns as input."""
        df_1m = _make_1m_df(n_hours=2)
        df_1h = resample_1h(df_1m)
        for col in ["open_time", "open", "high", "low", "close", "volume"]:
            assert col in df_1h.columns


# ============================================================================
# Intrabar Features
# ============================================================================

class TestIntrabarFeatures:
    def test_feature_columns_present(self):
        """Intrabar features have expected columns."""
        df_1m = _make_1m_df(n_hours=4)
        df_1h = resample_1h(df_1m)
        df_intra = compute_intrabar_features(df_1m, df_1h)

        expected = ["open_time", "max_runup", "max_drawdown",
                    "intrabar_vol", "intrabar_skew", "up_down_ratio"]
        for col in expected:
            assert col in df_intra.columns

    def test_row_count_matches_1h(self):
        """One intrabar row per 1h bar."""
        df_1m = _make_1m_df(n_hours=4)
        df_1h = resample_1h(df_1m)
        df_intra = compute_intrabar_features(df_1m, df_1h)
        assert len(df_intra) == len(df_1h)

    def test_max_runup_non_negative(self):
        """max_runup is always >= 0 (high >= open)."""
        df_1m = _make_1m_df(n_hours=4)
        df_1h = resample_1h(df_1m)
        df_intra = compute_intrabar_features(df_1m, df_1h)
        assert (df_intra["max_runup"] >= -1e-10).all()

    def test_max_drawdown_non_positive(self):
        """max_drawdown is always <= 0 (low <= open)."""
        df_1m = _make_1m_df(n_hours=4)
        df_1h = resample_1h(df_1m)
        df_intra = compute_intrabar_features(df_1m, df_1h)
        assert (df_intra["max_drawdown"] <= 1e-10).all()

    def test_up_down_ratio_bounded(self):
        """up_down_ratio is between 0 and 1."""
        df_1m = _make_1m_df(n_hours=4)
        df_1h = resample_1h(df_1m)
        df_intra = compute_intrabar_features(df_1m, df_1h)
        assert (df_intra["up_down_ratio"] >= 0).all()
        assert (df_intra["up_down_ratio"] <= 1).all()

    def test_intrabar_vol_non_negative(self):
        """intrabar_vol is always >= 0."""
        df_1m = _make_1m_df(n_hours=4)
        df_1h = resample_1h(df_1m)
        df_intra = compute_intrabar_features(df_1m, df_1h)
        assert (df_intra["intrabar_vol"] >= 0).all()

    def test_empty_input(self):
        """Empty input returns empty DataFrame."""
        df_empty = pd.DataFrame()
        df_1h = pd.DataFrame()
        result = compute_intrabar_features(df_empty, df_1h)
        assert len(result) == 0


# ============================================================================
# 4h Alignment (No Lookahead)
# ============================================================================

class TestAlign4hTo1h:
    def test_no_lookahead(self):
        """1h bar at time T can only see 4h bars that COMPLETED before T."""
        df_1m = _make_1m_df(n_hours=12)
        df_1h = resample_1h(df_1m)
        df_4h = resample_4h(df_1h)

        merged = align_4h_to_1h(df_1h, df_4h)

        # The first 4h bar opens at 00:00 and closes at 04:00
        # Its data should only be available_time = open_time + 4h = 04:00
        # So 1h bars at 00:00, 01:00, 02:00, 03:00 should NOT see this 4h bar's close
        # Only 1h bar at 04:00+ should see it

        # The first few 1h bars should have NaN for 4h data (no completed 4h bar yet)
        first_1h = merged[merged["open_time"] < pd.Timestamp("2024-01-01 04:00", tz="UTC")]
        assert first_1h["close_4h"].isna().all()

    def test_4h_columns_present(self):
        """Aligned DataFrame has 4h columns."""
        df_1m = _make_1m_df(n_hours=8)
        df_1h = resample_1h(df_1m)
        df_4h = resample_4h(df_1h)

        merged = align_4h_to_1h(df_1h, df_4h)
        for col in ["open_4h", "high_4h", "low_4h", "close_4h", "volume_4h"]:
            assert col in merged.columns

    def test_row_count_preserved(self):
        """Alignment preserves number of 1h rows."""
        df_1m = _make_1m_df(n_hours=8)
        df_1h = resample_1h(df_1m)
        df_4h = resample_4h(df_1h)

        merged = align_4h_to_1h(df_1h, df_4h)
        assert len(merged) == len(df_1h)


# ============================================================================
# Full Pipeline
# ============================================================================

class TestBuildMultiTimeframe:
    def test_returns_four_dataframes(self):
        """build_multi_timeframe_data returns 4 DataFrames."""
        df_1m = _make_1m_df(n_hours=8)
        result = build_multi_timeframe_data(df_1m)
        assert len(result) == 4

    def test_output_sizes_consistent(self):
        """1h count = n_hours, 4h count = n_hours/4."""
        df_1m = _make_1m_df(n_hours=8)
        _, df_1h, df_4h, df_intra = build_multi_timeframe_data(df_1m)
        assert len(df_1h) == 8
        assert len(df_4h) == 2
        assert len(df_intra) == 8
