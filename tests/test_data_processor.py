"""
DataProcessor testleri.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.processor import DataProcessor


class TestDataProcessor:
    """DataProcessor testleri."""

    def test_clean_ohlcv_removes_duplicates(self):
        """Duplicate satırlar kaldırılmalı."""
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
            "close": [102, 103, 104, 105, 106],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }, index=dates)

        # Duplicate ekle
        df = pd.concat([df, df.iloc[[2]]])

        cleaned = DataProcessor.clean_ohlcv(df)
        assert len(cleaned) == 5  # Duplicate kaldırıldı

    def test_clean_ohlcv_handles_nan(self):
        """NaN değerler doldurulmalı."""
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame({
            "open": [100, np.nan, 102, 103, 104],
            "high": [105, 106, np.nan, 108, 109],
            "low": [95, 96, 97, np.nan, 99],
            "close": [102, 103, 104, 105, np.nan],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }, index=dates)

        cleaned = DataProcessor.clean_ohlcv(df)
        assert not cleaned.isna().any().any()

    def test_calculate_returns(self, sample_dataframe: pd.DataFrame):
        """Getiri hesaplama testi."""
        returns = DataProcessor.calculate_returns(sample_dataframe)

        assert len(returns) == len(sample_dataframe)
        assert returns.iloc[0] != returns.iloc[0]  # İlk değer NaN

    def test_calculate_log_returns(self, sample_dataframe: pd.DataFrame):
        """Log getiri hesaplama testi."""
        log_returns = DataProcessor.calculate_log_returns(sample_dataframe)

        assert len(log_returns) == len(sample_dataframe)
        # Log returns normal returns'e yakın olmalı (küçük değişimler için)

    def test_detect_outliers(self, sample_dataframe: pd.DataFrame):
        """Outlier tespiti testi."""
        # Outlier ekle
        df = sample_dataframe.copy()
        df.iloc[50, df.columns.get_loc("close")] = df["close"].mean() * 10  # Çok yüksek

        outliers = DataProcessor.detect_outliers(df, "close", threshold=3.0)

        assert outliers.iloc[50] == True  # Eklediğimiz outlier bulunmalı

    def test_add_time_features(self, sample_dataframe: pd.DataFrame):
        """Zaman özellikleri ekleme testi."""
        result = DataProcessor.add_time_features(sample_dataframe)

        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "is_weekend" in result.columns

    def test_normalize_minmax(self, sample_dataframe: pd.DataFrame):
        """Min-max normalizasyon testi."""
        normalized = DataProcessor.normalize(
            sample_dataframe, columns=["close"], method="minmax"
        )

        assert normalized["close"].min() >= 0
        assert normalized["close"].max() <= 1

    def test_normalize_zscore(self, sample_dataframe: pd.DataFrame):
        """Z-score normalizasyon testi."""
        normalized = DataProcessor.normalize(
            sample_dataframe, columns=["close"], method="zscore"
        )

        # Z-score: ortalama ~0, std ~1
        assert abs(normalized["close"].mean()) < 0.01
        assert abs(normalized["close"].std() - 1) < 0.01

    def test_create_sequences(self, sample_dataframe: pd.DataFrame):
        """Sequence oluşturma testi."""
        X, y = DataProcessor.create_sequences(
            sample_dataframe,
            sequence_length=10,
            target_column="close",
        )

        assert len(X) == len(sample_dataframe) - 10
        assert len(y) == len(X)
        assert X.shape[1] == 10  # Sequence length
