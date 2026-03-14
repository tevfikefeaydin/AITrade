"""
Tests for Binance data downloader (src/data_binance.py).

Tests cover:
- Interval conversion
- Gap detection
- DataFrame construction and column types
- Incremental download logic (mocked)
- Load/save round-trip
- Error handling
"""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.data_binance import (
    _interval_to_ms,
    check_gaps,
    load_klines,
    KLINE_COLUMNS,
)
from src import config


# ============================================================================
# Interval Conversion
# ============================================================================

class TestIntervalToMs:
    def test_1m(self):
        assert _interval_to_ms("1m") == 60_000

    def test_5m(self):
        assert _interval_to_ms("5m") == 300_000

    def test_1h(self):
        assert _interval_to_ms("1h") == 3_600_000

    def test_4h(self):
        assert _interval_to_ms("4h") == 14_400_000

    def test_1d(self):
        assert _interval_to_ms("1d") == 86_400_000

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown interval unit"):
            _interval_to_ms("1w")

    def test_multi_digit(self):
        assert _interval_to_ms("15m") == 900_000
        assert _interval_to_ms("12h") == 43_200_000


# ============================================================================
# Gap Detection
# ============================================================================

class TestCheckGaps:
    def test_no_gaps_in_continuous_data(self):
        """Continuous 1m data has no gaps."""
        start = pd.Timestamp("2024-01-01", tz="UTC")
        times = [start + pd.Timedelta(minutes=i) for i in range(100)]
        df = pd.DataFrame({"open_time": times})

        gaps = check_gaps(df, "1m")
        assert len(gaps) == 0

    def test_detects_gap(self):
        """Detects a 10-minute gap in 1m data."""
        start = pd.Timestamp("2024-01-01", tz="UTC")
        # 0..49 minutes, then skip to 60..109 (gap from 49 to 60)
        times_before = [start + pd.Timedelta(minutes=i) for i in range(50)]
        times_after = [start + pd.Timedelta(minutes=i) for i in range(60, 110)]
        df = pd.DataFrame({"open_time": times_before + times_after}).reset_index(drop=True)

        gaps = check_gaps(df, "1m")
        assert len(gaps) == 1
        gap_start, gap_end = gaps[0]
        assert gap_start == times_before[-1]
        assert gap_end == times_after[0]

    def test_multiple_gaps(self):
        """Detects multiple gaps."""
        start = pd.Timestamp("2024-01-01", tz="UTC")
        times = (
            [start + pd.Timedelta(minutes=i) for i in range(10)]
            + [start + pd.Timedelta(minutes=i) for i in range(20, 30)]
            + [start + pd.Timedelta(minutes=i) for i in range(40, 50)]
        )
        df = pd.DataFrame({"open_time": times}).reset_index(drop=True)

        gaps = check_gaps(df, "1m")
        assert len(gaps) == 2

    def test_empty_dataframe(self):
        """Empty DataFrame returns no gaps."""
        df = pd.DataFrame({"open_time": []})
        gaps = check_gaps(df, "1m")
        assert len(gaps) == 0

    def test_single_row(self):
        """Single row returns no gaps."""
        df = pd.DataFrame({"open_time": [pd.Timestamp("2024-01-01", tz="UTC")]})
        gaps = check_gaps(df, "1m")
        assert len(gaps) == 0

    def test_no_gap_with_hourly_data(self):
        """Continuous 1h data has no gaps."""
        start = pd.Timestamp("2024-01-01", tz="UTC")
        times = [start + pd.Timedelta(hours=i) for i in range(48)]
        df = pd.DataFrame({"open_time": times})

        gaps = check_gaps(df, "1h")
        assert len(gaps) == 0

    def test_gap_tolerance(self):
        """Gaps within 1.5x tolerance are ignored."""
        start = pd.Timestamp("2024-01-01", tz="UTC")
        # 1m data with a single 89-second gap (within 1.5x of 60s)
        times = [start + pd.Timedelta(minutes=i) for i in range(10)]
        # Shift one bar by 29 seconds - still within tolerance
        times[5] = times[5] + pd.Timedelta(seconds=29)
        df = pd.DataFrame({"open_time": times})

        gaps = check_gaps(df, "1m")
        assert len(gaps) == 0


# ============================================================================
# Load Klines
# ============================================================================

class TestLoadKlines:
    def test_load_nonexistent_returns_none(self, tmp_path):
        """Loading non-existent file returns None."""
        with patch.object(config, "DATA_DIR", tmp_path):
            result = load_klines("BTCUSDT", "1m")
            assert result is None

    def test_load_existing_parquet(self, tmp_path):
        """Loading existing parquet file works."""
        df = pd.DataFrame({
            "open_time": pd.date_range("2024-01-01", periods=10, freq="min", tz="UTC"),
            "open": range(10),
            "high": range(10),
            "low": range(10),
            "close": range(10),
            "volume": range(10),
            "close_time": pd.date_range("2024-01-01 00:00:59", periods=10, freq="min", tz="UTC"),
        })
        path = tmp_path / "BTCUSDT_1m.parquet"
        df.to_parquet(path)

        with patch.object(config, "DATA_DIR", tmp_path):
            result = load_klines("BTCUSDT", "1m")
            assert result is not None
            assert len(result) == 10

    def test_invalid_symbol_raises(self):
        """Invalid symbol raises ValueError."""
        with pytest.raises(ValueError):
            load_klines("INVALID", "1m")


# ============================================================================
# DataFrame Construction (Mocked API)
# ============================================================================

class TestDataFrameConstruction:
    def _make_mock_klines(self, n=5, start_ms=1704067200000, interval_ms=60000):
        """Create mock Binance API kline response."""
        klines = []
        for i in range(n):
            open_time = start_ms + i * interval_ms
            close_time = open_time + interval_ms - 1
            klines.append([
                open_time,
                str(50000 + i),     # open
                str(50100 + i),     # high
                str(49900 + i),     # low
                str(50050 + i),     # close
                str(100 + i),       # volume
                close_time,
                str(5000000),       # quote_volume
                100,                # trades
                str(50),            # taker_buy_base
                str(2500000),       # taker_buy_quote
                "0",                # ignore
            ])
        return klines

    @patch("src.data_binance.fetch_klines_batch")
    def test_download_creates_correct_columns(self, mock_fetch, tmp_path):
        """Downloaded DataFrame has expected columns."""
        from src.data_binance import download_klines

        mock_fetch.side_effect = [self._make_mock_klines(5), []]

        with patch.object(config, "DATA_DIR", tmp_path):
            df = download_klines("BTCUSDT", "1m", "2024-01-01", "2024-01-02", save=False)

        expected_cols = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
        for col in expected_cols:
            assert col in df.columns

    @patch("src.data_binance.fetch_klines_batch")
    def test_download_types_are_correct(self, mock_fetch, tmp_path):
        """Numeric columns are float, timestamps are datetime."""
        from src.data_binance import download_klines

        mock_fetch.side_effect = [self._make_mock_klines(5), []]

        with patch.object(config, "DATA_DIR", tmp_path):
            df = download_klines("BTCUSDT", "1m", "2024-01-01", "2024-01-02", save=False)

        assert pd.api.types.is_datetime64_any_dtype(df["open_time"])
        assert pd.api.types.is_datetime64_any_dtype(df["close_time"])
        assert df["open"].dtype == float
        assert df["close"].dtype == float
        assert df["volume"].dtype == float

    @patch("src.data_binance.fetch_klines_batch")
    def test_download_removes_duplicates(self, mock_fetch, tmp_path):
        """Duplicate rows are removed."""
        from src.data_binance import download_klines

        klines = self._make_mock_klines(5)
        klines_with_dup = klines + [klines[0]]
        mock_fetch.side_effect = [klines_with_dup, []]

        with patch.object(config, "DATA_DIR", tmp_path):
            df = download_klines("BTCUSDT", "1m", "2024-01-01", "2024-01-02", save=False)

        assert len(df) == 5  # Duplicate removed

    @patch("src.data_binance.fetch_klines_batch")
    def test_download_sorted_by_time(self, mock_fetch, tmp_path):
        """Result is sorted by open_time."""
        from src.data_binance import download_klines

        klines = self._make_mock_klines(5)
        mock_fetch.side_effect = [list(reversed(klines)), []]

        with patch.object(config, "DATA_DIR", tmp_path):
            df = download_klines("BTCUSDT", "1m", "2024-01-01", "2024-01-02", save=False)

        assert df["open_time"].is_monotonic_increasing


# ============================================================================
# Incremental Download Logic (Mocked)
# ============================================================================

class TestIncrementalDownload:
    @patch("src.data_binance.fetch_klines_batch")
    def test_no_download_when_data_up_to_date(self, mock_fetch, tmp_path):
        """No API calls when existing data covers requested range."""
        from src.data_binance import download_klines_incremental

        # Create existing data covering full range
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-01-02", tz="UTC")
        times = pd.date_range(start, end, freq="min")[:-1]  # Full day of 1m data

        df_existing = pd.DataFrame({
            "open_time": times,
            "open": 50000.0,
            "high": 50100.0,
            "low": 49900.0,
            "close": 50050.0,
            "volume": 100.0,
            "close_time": times + pd.Timedelta(seconds=59),
        })

        path = tmp_path / "BTCUSDT_1m.parquet"
        df_existing.to_parquet(path)

        with patch.object(config, "DATA_DIR", tmp_path):
            df, stats = download_klines_incremental(
                "BTCUSDT", "1m", "2024-01-01", "2024-01-02"
            )

        # No API calls should be made
        mock_fetch.assert_not_called()
        assert stats["new_rows"] == 0
        assert stats["existing_rows"] == len(df_existing)

    @patch("src.data_binance.download_klines")
    def test_force_redownloads(self, mock_download, tmp_path):
        """Force flag causes full re-download even with existing data."""
        from src.data_binance import download_klines_incremental

        # Create existing data
        df_existing = pd.DataFrame({
            "open_time": pd.date_range("2024-01-01", periods=10, freq="min", tz="UTC"),
            "open": 50000.0, "high": 50100.0, "low": 49900.0,
            "close": 50050.0, "volume": 100.0,
            "close_time": pd.date_range("2024-01-01 00:00:59", periods=10, freq="min", tz="UTC"),
        })
        path = tmp_path / "BTCUSDT_1m.parquet"
        df_existing.to_parquet(path)

        mock_download.return_value = df_existing

        with patch.object(config, "DATA_DIR", tmp_path):
            df, stats = download_klines_incremental(
                "BTCUSDT", "1m", "2024-01-01", "2024-01-02", force=True
            )

        mock_download.assert_called_once()
        assert stats["gaps_filled"] == ["Full: 2024-01-01 to 2024-01-02"]
