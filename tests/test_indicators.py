"""
İndikatör testleri.
"""

import pytest
import pandas as pd
import numpy as np

from src.analysis.indicators.simple import SimpleIndicators
from src.analysis.indicators.technical import TechnicalIndicators


class TestSimpleIndicators:
    """SimpleIndicators testleri."""

    def test_sma(self, sample_dataframe: pd.DataFrame):
        """SMA hesaplaması testi."""
        sma = SimpleIndicators.sma(sample_dataframe["close"], 20)

        assert len(sma) == len(sample_dataframe)
        assert sma.iloc[:19].isna().all()  # İlk 19 değer NaN
        assert not sma.iloc[19:].isna().any()  # Sonrası hesaplanmış

    def test_ema(self, sample_dataframe: pd.DataFrame):
        """EMA hesaplaması testi."""
        ema = SimpleIndicators.ema(sample_dataframe["close"], 20)

        assert len(ema) == len(sample_dataframe)
        assert not ema.isna().any()  # EMA tüm değerleri hesaplar

    def test_rsi_range(self, sample_dataframe: pd.DataFrame):
        """RSI 0-100 aralığında olmalı."""
        rsi = SimpleIndicators.rsi(sample_dataframe["close"])

        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_macd_components(self, sample_dataframe: pd.DataFrame):
        """MACD bileşenleri testi."""
        macd, signal, histogram = SimpleIndicators.macd(sample_dataframe["close"])

        assert len(macd) == len(sample_dataframe)
        assert len(signal) == len(sample_dataframe)
        assert len(histogram) == len(sample_dataframe)

        # Histogram = MACD - Signal
        valid_idx = ~(histogram.isna() | macd.isna() | signal.isna())
        np.testing.assert_array_almost_equal(
            histogram[valid_idx].values,
            (macd[valid_idx] - signal[valid_idx]).values,
            decimal=10,
        )

    def test_bollinger_bands(self, sample_dataframe: pd.DataFrame):
        """Bollinger Bands testi."""
        upper, middle, lower = SimpleIndicators.bollinger_bands(
            sample_dataframe["close"]
        )

        # Upper > Middle > Lower olmalı
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()

    def test_atr_positive(self, sample_dataframe: pd.DataFrame):
        """ATR pozitif olmalı."""
        atr = SimpleIndicators.atr(
            sample_dataframe["high"],
            sample_dataframe["low"],
            sample_dataframe["close"],
        )

        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()

    def test_stochastic_range(self, sample_dataframe: pd.DataFrame):
        """Stochastic 0-100 aralığında olmalı."""
        k, d = SimpleIndicators.stochastic(
            sample_dataframe["high"],
            sample_dataframe["low"],
            sample_dataframe["close"],
        )

        valid_k = k.dropna()
        valid_d = d.dropna()

        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()


class TestTechnicalIndicators:
    """TechnicalIndicators testleri."""

    def test_add_all_indicators(self, sample_dataframe: pd.DataFrame):
        """Tüm indikatörlerin eklenmesi."""
        ta = TechnicalIndicators(sample_dataframe)
        result = ta.add_all_indicators()

        # Beklenen kolonlar
        expected_cols = [
            "ema_9", "ema_21", "ema_50", "ema_200",
            "rsi", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_middle", "bb_lower",
            "atr", "volume_ma",
        ]

        for col in expected_cols:
            assert col in result.columns, f"{col} eksik"

    def test_get_trend_bullish(self, bullish_dataframe: pd.DataFrame):
        """Bullish trend tespiti."""
        ta = TechnicalIndicators(bullish_dataframe)
        ta.add_moving_averages()
        trend = ta.get_trend()

        # Yükseliş trendinde bullish bekliyoruz
        assert trend in ["bullish", "neutral"]

    def test_get_trend_bearish(self, bearish_dataframe: pd.DataFrame):
        """Bearish trend tespiti."""
        ta = TechnicalIndicators(bearish_dataframe)
        ta.add_moving_averages()
        trend = ta.get_trend()

        # Düşüş trendinde bearish bekliyoruz
        assert trend in ["bearish", "neutral"]

    def test_get_summary(self, sample_dataframe: pd.DataFrame):
        """Özet çıktısı testi."""
        ta = TechnicalIndicators(sample_dataframe)
        summary = ta.get_summary()

        assert "trend" in summary
        assert "momentum" in summary
        assert "volatility" in summary
        assert "last_price" in summary

    def test_invalid_dataframe(self):
        """Eksik kolon hatası."""
        invalid_df = pd.DataFrame({"close": [1, 2, 3]})

        with pytest.raises(ValueError, match="Eksik kolonlar"):
            TechnicalIndicators(invalid_df)
