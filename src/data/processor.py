"""
DataProcessor - Veri işleme ve temizleme.
"""

from typing import List, Optional

import pandas as pd
import numpy as np
from loguru import logger

from src.data.models import OHLCV


class DataProcessor:
    """Veri işleme sınıfı."""

    @staticmethod
    def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """
        OHLCV verisini temizle.

        Args:
            df: Ham OHLCV DataFrame

        Returns:
            Temizlenmiş DataFrame
        """
        # Duplicate'leri kaldır
        df = df[~df.index.duplicated(keep="last")]

        # Sırala
        df = df.sort_index()

        # NaN değerleri doldur
        df = df.ffill().bfill()

        # Sıfır/negatif değerleri düzelt
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: max(x, 0.0001))

        # Volume negatif olamaz
        if "volume" in df.columns:
            df["volume"] = df["volume"].apply(lambda x: max(x, 0))

        return df

    @staticmethod
    def resample_ohlcv(
        df: pd.DataFrame,
        target_timeframe: str,
    ) -> pd.DataFrame:
        """
        OHLCV verisini farklı timeframe'e dönüştür.

        Args:
            df: Kaynak DataFrame
            target_timeframe: Hedef timeframe (örn: '4H', '1D')

        Returns:
            Resample edilmiş DataFrame
        """
        resampled = df.resample(target_timeframe).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })

        return resampled.dropna()

    @staticmethod
    def calculate_returns(df: pd.DataFrame, column: str = "close") -> pd.Series:
        """
        Getiri hesapla.

        Args:
            df: OHLCV DataFrame
            column: Hesaplanacak kolon

        Returns:
            Getiri serisi
        """
        return df[column].pct_change()

    @staticmethod
    def calculate_log_returns(df: pd.DataFrame, column: str = "close") -> pd.Series:
        """
        Log getiri hesapla.

        Args:
            df: OHLCV DataFrame
            column: Hesaplanacak kolon

        Returns:
            Log getiri serisi
        """
        return np.log(df[column] / df[column].shift(1))

    @staticmethod
    def detect_outliers(
        df: pd.DataFrame,
        column: str,
        threshold: float = 3.0,
    ) -> pd.Series:
        """
        Outlier tespit et (z-score yöntemi).

        Args:
            df: DataFrame
            column: Kontrol edilecek kolon
            threshold: Z-score eşiği

        Returns:
            Boolean series (True = outlier)
        """
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return z_scores > threshold

    @staticmethod
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Zaman bazlı özellikler ekle.

        Args:
            df: OHLCV DataFrame (datetime index)

        Returns:
            Özellikler eklenmiş DataFrame
        """
        df = df.copy()

        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["month"] = df.index.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        return df

    @staticmethod
    def normalize(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "minmax",
    ) -> pd.DataFrame:
        """
        Veriyi normalize et.

        Args:
            df: DataFrame
            columns: Normalize edilecek kolonlar (None = hepsi)
            method: Normalizasyon metodu (minmax, zscore)

        Returns:
            Normalize edilmiş DataFrame
        """
        df = df.copy()
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if method == "minmax":
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
            elif method == "zscore":
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std

        return df

    @staticmethod
    def create_sequences(
        df: pd.DataFrame,
        sequence_length: int,
        target_column: str = "close",
        feature_columns: Optional[List[str]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        ML için sequence verisi oluştur.

        Args:
            df: DataFrame
            sequence_length: Sequence uzunluğu
            target_column: Hedef kolon
            feature_columns: Özellik kolonları

        Returns:
            (X, y) tuple - features ve targets
        """
        features = feature_columns or df.columns.tolist()
        data = df[features].values

        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i : i + sequence_length])
            y.append(df[target_column].iloc[i + sequence_length])

        return np.array(X), np.array(y)
