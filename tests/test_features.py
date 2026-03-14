"""
Tests for feature engineering - NO LOOKAHEAD verification.

CRITICAL: These tests verify that features at time t only use data
available up to time t. Any lookahead would invalidate the entire pipeline.

Test approach:
1. Compute features on full dataset
2. Truncate data after time t
3. Recompute features
4. Verify features at time <= t are identical
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features import (
    compute_wick_features,
    compute_return_features,
    compute_volatility_features,
    compute_rsi,
    compute_ma_features,
    compute_volume_features,
    compute_regime_features,
    compute_stoch_rsi,
    compute_macd_hist,
    build_features,
)
from src.resample import resample_1h, resample_4h


def create_synthetic_1m_data(n_hours: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create synthetic 1-minute data for testing."""
    np.random.seed(seed)

    n_minutes = n_hours * 60
    start_time = datetime(2024, 1, 1)

    # Generate random walk price
    returns = np.random.normal(0, 0.0001, n_minutes)
    prices = 100 * np.exp(np.cumsum(returns))

    # Generate OHLCV
    opens = prices
    closes = prices * (1 + np.random.normal(0, 0.0001, n_minutes))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.0001, n_minutes)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.0001, n_minutes)))
    volumes = np.random.uniform(100, 1000, n_minutes)

    df = pd.DataFrame({
        "open_time": [start_time + timedelta(minutes=i) for i in range(n_minutes)],
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    return df


def create_synthetic_1h_data(n_hours: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create synthetic 1-hour data for testing."""
    df_1m = create_synthetic_1m_data(n_hours, seed)
    return resample_1h(df_1m)


class TestNoLookahead:
    """Test that features don't have lookahead bias."""

    def test_return_features_no_lookahead(self):
        """Test that return features don't look ahead."""
        df = create_synthetic_1h_data(100)

        # Compute features on full data
        df_full = compute_return_features(df.copy())

        # Truncate data at various points and verify
        for t in [50, 60, 70, 80]:
            df_truncated = compute_return_features(df.iloc[:t].copy())

            # Features at time <= t should be identical
            for col in ["ret_1", "ret_3", "ret_6", "ret_12"]:
                if col in df_full.columns and col in df_truncated.columns:
                    # Compare values (allowing for NaN at start)
                    full_vals = df_full[col].iloc[:t].values
                    trunc_vals = df_truncated[col].values

                    # Mask NaN values for comparison
                    mask = ~np.isnan(full_vals) & ~np.isnan(trunc_vals)

                    if mask.sum() > 0:
                        np.testing.assert_array_almost_equal(
                            full_vals[mask],
                            trunc_vals[mask],
                            decimal=10,
                            err_msg=f"Lookahead detected in {col} at t={t}",
                        )

    def test_volatility_no_lookahead(self):
        """Test that volatility features don't look ahead."""
        df = create_synthetic_1h_data(100)
        df = compute_return_features(df)  # Need ret_1 first

        df_full = compute_volatility_features(df.copy(), window=20)

        for t in [50, 60, 70, 80]:
            df_truncated = compute_volatility_features(df.iloc[:t].copy(), window=20)

            col = "vol_20"
            if col in df_full.columns and col in df_truncated.columns:
                full_vals = df_full[col].iloc[:t].values
                trunc_vals = df_truncated[col].values

                mask = ~np.isnan(full_vals) & ~np.isnan(trunc_vals)

                if mask.sum() > 0:
                    np.testing.assert_array_almost_equal(
                        full_vals[mask],
                        trunc_vals[mask],
                        decimal=10,
                        err_msg=f"Lookahead detected in {col} at t={t}",
                    )

    def test_rsi_no_lookahead(self):
        """Test that RSI doesn't look ahead."""
        df = create_synthetic_1h_data(100)

        df_full = compute_rsi(df.copy())

        for t in [50, 60, 70, 80]:
            df_truncated = compute_rsi(df.iloc[:t].copy())

            col = "rsi"
            full_vals = df_full[col].iloc[:t].values
            trunc_vals = df_truncated[col].values

            mask = ~np.isnan(full_vals) & ~np.isnan(trunc_vals)

            if mask.sum() > 0:
                np.testing.assert_array_almost_equal(
                    full_vals[mask],
                    trunc_vals[mask],
                    decimal=8,  # RSI uses EMA which may have small numerical differences
                    err_msg=f"Lookahead detected in {col} at t={t}",
                )

    def test_ma_features_no_lookahead(self):
        """Test that MA features don't look ahead."""
        df = create_synthetic_1h_data(100)

        df_full = compute_ma_features(df.copy(), window=20)

        for t in [50, 60, 70, 80]:
            df_truncated = compute_ma_features(df.iloc[:t].copy(), window=20)

            for col in ["ma_20", "ma_gap"]:
                if col in df_full.columns and col in df_truncated.columns:
                    full_vals = df_full[col].iloc[:t].values
                    trunc_vals = df_truncated[col].values

                    mask = ~np.isnan(full_vals) & ~np.isnan(trunc_vals)

                    if mask.sum() > 0:
                        np.testing.assert_array_almost_equal(
                            full_vals[mask],
                            trunc_vals[mask],
                            decimal=10,
                            err_msg=f"Lookahead detected in {col} at t={t}",
                        )

    def test_volume_zscore_no_lookahead(self):
        """Test that volume z-score and volume_ratio don't look ahead."""
        df = create_synthetic_1h_data(100)

        df_full = compute_volume_features(df.copy(), window=20)

        for t in [50, 60, 70, 80]:
            df_truncated = compute_volume_features(df.iloc[:t].copy(), window=20)

            for col in ["volume_zscore", "volume_ratio"]:
                full_vals = df_full[col].iloc[:t].values
                trunc_vals = df_truncated[col].values

                mask = ~np.isnan(full_vals) & ~np.isnan(trunc_vals)

                if mask.sum() > 0:
                    np.testing.assert_array_almost_equal(
                        full_vals[mask],
                        trunc_vals[mask],
                        decimal=10,
                        err_msg=f"Lookahead detected in {col} at t={t}",
                    )

    def test_ret_24_no_lookahead(self):
        """Test that ret_24 doesn't look ahead."""
        df = create_synthetic_1h_data(100)

        df_full = compute_return_features(df.copy())

        for t in [50, 60, 70, 80]:
            df_truncated = compute_return_features(df.iloc[:t].copy())

            col = "ret_24"
            full_vals = df_full[col].iloc[:t].values
            trunc_vals = df_truncated[col].values

            mask = ~np.isnan(full_vals) & ~np.isnan(trunc_vals)

            if mask.sum() > 0:
                np.testing.assert_array_almost_equal(
                    full_vals[mask],
                    trunc_vals[mask],
                    decimal=10,
                    err_msg=f"Lookahead detected in {col} at t={t}",
                )

    def test_wick_features_no_lookahead(self):
        """Test that wick features don't look ahead (they use current bar only)."""
        df = create_synthetic_1h_data(100)

        df_full = compute_wick_features(df.copy())

        for t in [50, 60, 70, 80]:
            df_truncated = compute_wick_features(df.iloc[:t].copy())

            for col in ["upper_wick", "lower_wick", "body", "range"]:
                full_vals = df_full[col].iloc[:t].values
                trunc_vals = df_truncated[col].values

                np.testing.assert_array_almost_equal(
                    full_vals,
                    trunc_vals,
                    decimal=10,
                    err_msg=f"Lookahead detected in {col} at t={t}",
                )


    def test_rolling_sharpe_no_lookahead(self):
        """Test that rolling_sharpe_20 doesn't look ahead."""
        df = create_synthetic_1h_data(100)
        df = compute_return_features(df)

        df_full = compute_regime_features(df.copy(), window=20)

        for t in [50, 60, 70, 80]:
            df_truncated = compute_regime_features(df.iloc[:t].copy(), window=20)

            col = "rolling_sharpe_20"
            full_vals = df_full[col].iloc[:t].values
            trunc_vals = df_truncated[col].values

            mask = ~np.isnan(full_vals) & ~np.isnan(trunc_vals)

            if mask.sum() > 0:
                np.testing.assert_array_almost_equal(
                    full_vals[mask],
                    trunc_vals[mask],
                    decimal=10,
                    err_msg=f"Lookahead detected in {col} at t={t}",
                )

    def test_bb_width_no_lookahead(self):
        """Test that bb_width doesn't look ahead."""
        df = create_synthetic_1h_data(100)
        df = compute_return_features(df)

        df_full = compute_regime_features(df.copy(), window=20)

        for t in [50, 60, 70, 80]:
            df_truncated = compute_regime_features(df.iloc[:t].copy(), window=20)

            col = "bb_width"
            full_vals = df_full[col].iloc[:t].values
            trunc_vals = df_truncated[col].values

            mask = ~np.isnan(full_vals) & ~np.isnan(trunc_vals)

            if mask.sum() > 0:
                np.testing.assert_array_almost_equal(
                    full_vals[mask],
                    trunc_vals[mask],
                    decimal=10,
                    err_msg=f"Lookahead detected in {col} at t={t}",
                )

    def test_stoch_rsi_no_lookahead(self):
        """Test that stoch_rsi doesn't look ahead."""
        df = create_synthetic_1h_data(100)
        df = compute_rsi(df)

        df_full = compute_stoch_rsi(df.copy())

        for t in [50, 60, 70, 80]:
            df_truncated = compute_stoch_rsi(df.iloc[:t].copy())

            col = "stoch_rsi"
            full_vals = df_full[col].iloc[:t].values
            trunc_vals = df_truncated[col].values

            mask = ~np.isnan(full_vals) & ~np.isnan(trunc_vals)

            if mask.sum() > 0:
                np.testing.assert_array_almost_equal(
                    full_vals[mask],
                    trunc_vals[mask],
                    decimal=8,
                    err_msg=f"Lookahead detected in {col} at t={t}",
                )

    def test_macd_hist_no_lookahead(self):
        """Test that macd_hist doesn't look ahead."""
        df = create_synthetic_1h_data(100)

        df_full = compute_macd_hist(df.copy())

        for t in [50, 60, 70, 80]:
            df_truncated = compute_macd_hist(df.iloc[:t].copy())

            col = "macd_hist"
            full_vals = df_full[col].iloc[:t].values
            trunc_vals = df_truncated[col].values

            mask = ~np.isnan(full_vals) & ~np.isnan(trunc_vals)

            if mask.sum() > 0:
                np.testing.assert_array_almost_equal(
                    full_vals[mask],
                    trunc_vals[mask],
                    decimal=8,
                    err_msg=f"Lookahead detected in {col} at t={t}",
                )


class TestFeatureConsistency:
    """Test feature computation consistency."""

    def test_returns_formula(self):
        """Test that returns are computed correctly."""
        prices = [100, 101, 102, 100, 98]
        df = pd.DataFrame({
            "open_time": pd.date_range("2024-01-01", periods=5, freq="h"),
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1000] * 5,
        })

        df = compute_return_features(df)

        # ret_1 should be log(close_t / close_{t-1})
        expected_ret_1 = np.log(101 / 100)
        assert abs(df["ret_1"].iloc[1] - expected_ret_1) < 0.0001

    def test_rsi_bounds(self):
        """Test that RSI is bounded between 0 and 100."""
        df = create_synthetic_1h_data(100)
        df = compute_rsi(df)

        valid_rsi = df["rsi"].dropna()
        assert all(valid_rsi >= 0), "RSI should be >= 0"
        assert all(valid_rsi <= 100), "RSI should be <= 100"

    def test_volatility_non_negative(self):
        """Test that volatility is non-negative."""
        df = create_synthetic_1h_data(100)
        df = compute_return_features(df)
        df = compute_volatility_features(df, window=20)

        valid_vol = df["vol_20"].dropna()
        assert all(valid_vol >= 0), "Volatility should be non-negative"

    def test_feature_names_consistency(self):
        """Test that feature names are consistent across runs."""
        df_1m = create_synthetic_1m_data(100)
        df_1h = resample_1h(df_1m)
        df_4h = resample_4h(df_1h)

        features1 = build_features(df_1m, df_1h, df_4h)
        features2 = build_features(df_1m, df_1h, df_4h)

        assert list(features1.columns) == list(features2.columns)


class TestNewFeatures:
    """Test new features: volume_ratio, atr_ratio, ret_24, hour_sin, hour_cos."""

    def test_volume_ratio_positive(self):
        """volume_ratio should be positive when volume > 0."""
        df = create_synthetic_1h_data(100)
        df = compute_volume_features(df, window=20)
        valid = df["volume_ratio"].dropna()
        assert all(valid > 0), "volume_ratio should be positive"

    def test_ret_24_formula(self):
        """ret_24 should equal log(close_t / close_{t-24})."""
        df = create_synthetic_1h_data(100)
        df = compute_return_features(df)
        # Check at index 30 (has 24 bars of history)
        expected = np.log(df["close"].iloc[30] / df["close"].iloc[6])
        assert abs(df["ret_24"].iloc[30] - expected) < 1e-10

    def test_hour_features_range(self):
        """hour_sin and hour_cos should be in [-1, 1]."""
        from src.features import build_features
        from src.resample import resample_1h, resample_4h
        df_1m = create_synthetic_1m_data(300)
        df_1h = resample_1h(df_1m)
        df_4h = resample_4h(df_1h)
        df = build_features(df_1m, df_1h, df_4h)
        assert all(df["hour_sin"].between(-1, 1))
        assert all(df["hour_cos"].between(-1, 1))

    def test_feature_count_is_34(self):
        """get_feature_columns should return 34 features (27 + 7 new: taker_buy_ratio, btc_ret_1, btc_volume_zscore, rolling_sharpe_20, bb_width, stoch_rsi, macd_hist)."""
        from src.features import get_feature_columns
        assert len(get_feature_columns()) == 34

    def test_build_features_has_new_columns(self):
        """build_features output should include all new feature columns."""
        from src.features import build_features
        from src.resample import resample_1h, resample_4h
        df_1m = create_synthetic_1m_data(300)
        df_1h = resample_1h(df_1m)
        df_4h = resample_4h(df_1h)
        df = build_features(df_1m, df_1h, df_4h)
        for col in [
            "volume_ratio", "atr_ratio", "ret_24", "hour_sin", "hour_cos",
            "taker_buy_ratio", "rolling_sharpe_20", "bb_width",
            "stoch_rsi", "macd_hist", "btc_ret_1", "btc_volume_zscore",
        ]:
            assert col in df.columns, f"Missing column: {col}"


class TestEdgeCases:
    """Test edge cases in feature computation."""

    def test_insufficient_data_for_rolling(self):
        """Test handling of insufficient data for rolling windows."""
        df = create_synthetic_1h_data(10)  # Only 10 hours

        # Should not raise, but should have NaN for early values
        df = compute_return_features(df)
        df = compute_volatility_features(df, window=20)
        df = compute_ma_features(df, window=20)

        # vol_20 and ma_20 should be mostly NaN with only 10 bars
        assert df["vol_20"].isna().sum() >= 10, "Should have NaN for insufficient data"

    def test_single_bar(self):
        """Test handling of single bar."""
        df = pd.DataFrame({
            "open_time": [datetime(2024, 1, 1)],
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [102.0],
            "volume": [1000.0],
        })

        # Should not raise
        df = compute_wick_features(df)
        df = compute_return_features(df)

        assert len(df) == 1
        assert not np.isnan(df["upper_wick"].iloc[0])
        assert np.isnan(df["ret_1"].iloc[0])  # No previous bar

    def test_constant_prices(self):
        """Test handling of constant prices (no movement)."""
        n = 50
        df = pd.DataFrame({
            "open_time": pd.date_range("2024-01-01", periods=n, freq="h"),
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
        })

        df = compute_return_features(df)
        df = compute_volatility_features(df, window=20)

        # Returns should be 0
        valid_rets = df["ret_1"].dropna()
        assert all(valid_rets == 0), "Returns should be 0 for constant prices"

        # Volatility should be 0
        valid_vol = df["vol_20"].dropna()
        assert all(valid_vol == 0), "Volatility should be 0 for constant prices"
