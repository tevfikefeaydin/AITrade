"""
Resampling module for ML-Assisted Crypto Trading Research Pipeline.

Aggregates 1-minute klines to higher timeframes (1h, 4h) while preserving
intrabar data needed for feature computation.
"""

import logging
from typing import Dict, Tuple

import pandas as pd
import numpy as np

from . import config

logger = logging.getLogger(__name__)


def resample_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a higher timeframe.

    Args:
        df: DataFrame with columns [open_time, open, high, low, close, volume]
        interval: Target interval (e.g., "1h", "4h")

    Returns:
        Resampled DataFrame with same columns
    """
    if "open_time" not in df.columns:
        raise ValueError("DataFrame must have 'open_time' column")

    # Set index for resampling
    df_indexed = df.set_index("open_time")

    # Resample with proper OHLCV aggregation
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "taker_buy_base": "sum",
    }

    # Only include columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df_indexed.columns}

    resampled = df_indexed.resample(interval).agg(agg_dict)

    # Drop rows with NaN (incomplete bars at edges)
    resampled = resampled.dropna()

    # Reset index to get open_time as column
    resampled = resampled.reset_index()

    logger.info(f"Resampled {len(df):,} rows to {len(resampled):,} {interval} bars")

    return resampled


def resample_1h(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-minute data to 1-hour bars.

    Args:
        df_1m: 1-minute OHLCV DataFrame

    Returns:
        1-hour OHLCV DataFrame
    """
    return resample_ohlcv(df_1m, "1h")


def resample_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-hour data to 4-hour bars.

    Args:
        df_1h: 1-hour OHLCV DataFrame

    Returns:
        4-hour OHLCV DataFrame
    """
    return resample_ohlcv(df_1h, "4h")


def compute_intrabar_features(df_1m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Compute intrabar features from 1-minute data for each 1-hour bar.

    For each 1h bar, we compute statistics from the 1m bars within that hour:
    - max_runup: Maximum high relative to 1h open
    - max_drawdown: Minimum low relative to 1h open
    - intrabar_vol: Standard deviation of 1m log returns
    - intrabar_skew: Skewness of 1m log returns
    - up_down_ratio: Ratio of up moves to down moves

    Args:
        df_1m: 1-minute OHLCV DataFrame
        df_1h: 1-hour OHLCV DataFrame

    Returns:
        DataFrame with 1h index and intrabar features
    """
    if len(df_1m) == 0 or len(df_1h) == 0:
        return pd.DataFrame()

    # Ensure open_time is datetime
    df_1m = df_1m.copy()
    df_1h = df_1h.copy()

    if not pd.api.types.is_datetime64_any_dtype(df_1m["open_time"]):
        df_1m["open_time"] = pd.to_datetime(df_1m["open_time"])
    if not pd.api.types.is_datetime64_any_dtype(df_1h["open_time"]):
        df_1h["open_time"] = pd.to_datetime(df_1h["open_time"])

    # Add hour floor to 1m data for grouping
    df_1m["hour_floor"] = df_1m["open_time"].dt.floor("1h")

    # Compute 1m log returns
    df_1m["log_ret"] = np.log(df_1m["close"] / df_1m["close"].shift(1))

    # Vectorized groupby approach (much faster than iterrows)
    grouped = df_1m.groupby("hour_floor")

    # Max high and min low per hour
    hour_high = grouped["high"].max()
    hour_low = grouped["low"].min()

    # Log return stats per hour
    hour_vol = grouped["log_ret"].std()
    hour_skew = grouped["log_ret"].apply(lambda x: x.dropna().skew() if len(x.dropna()) > 2 else 0.0)
    hour_up = grouped["log_ret"].apply(lambda x: (x > 0).sum())
    hour_down = grouped["log_ret"].apply(lambda x: (x < 0).sum())

    # Build lookup by mapping df_1h open_time to groupby results
    hour_times = df_1h["open_time"].values
    h_high = hour_high.reindex(hour_times).values
    h_low = hour_low.reindex(hour_times).values
    h_open = df_1h["open"].values

    safe_open = np.where(h_open > 0, h_open, 1.0)
    max_runup = (h_high - h_open) / safe_open
    max_drawdown = (h_low - h_open) / safe_open

    intrabar_vol = hour_vol.reindex(hour_times).values
    intrabar_skew = hour_skew.reindex(hour_times).values

    up_vals = hour_up.reindex(hour_times).values
    down_vals = hour_down.reindex(hour_times).values
    total_moves = up_vals + down_vals
    up_down_ratio = np.where(total_moves > 0, up_vals / total_moves, 0.5)

    df_features = pd.DataFrame({
        "open_time": df_1h["open_time"].reset_index(drop=True),
        "max_runup": np.nan_to_num(max_runup, nan=0.0),
        "max_drawdown": np.nan_to_num(max_drawdown, nan=0.0),
        "intrabar_vol": np.nan_to_num(intrabar_vol, nan=0.0),
        "intrabar_skew": np.nan_to_num(intrabar_skew, nan=0.0),
        "up_down_ratio": np.nan_to_num(up_down_ratio, nan=0.5),
    })

    logger.info(f"Computed intrabar features for {len(df_features):,} hourly bars")

    return df_features


def build_multi_timeframe_data(
    df_1m: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build all timeframe data from 1-minute source.

    Args:
        df_1m: 1-minute OHLCV DataFrame

    Returns:
        Tuple of (df_1m, df_1h, df_4h, df_intrabar_features)
    """
    logger.info("Building multi-timeframe data...")

    # Resample to 1h
    df_1h = resample_1h(df_1m)

    # Resample to 4h
    df_4h = resample_4h(df_1h)

    # Compute intrabar features
    df_intrabar = compute_intrabar_features(df_1m, df_1h)

    logger.info(
        f"Built: 1m={len(df_1m):,}, 1h={len(df_1h):,}, "
        f"4h={len(df_4h):,}, intrabar={len(df_intrabar):,}"
    )

    return df_1m, df_1h, df_4h, df_intrabar


def align_4h_to_1h(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill 4h data to align with 1h bars.

    For each 1h bar, find the most recent completed 4h bar (no lookahead).

    Args:
        df_1h: 1-hour DataFrame with open_time
        df_4h: 4-hour DataFrame with open_time

    Returns:
        DataFrame with 4h features aligned to 1h timestamps
    """
    df_1h = df_1h.copy()
    df_4h = df_4h.copy()

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df_1h["open_time"]):
        df_1h["open_time"] = pd.to_datetime(df_1h["open_time"])
    if not pd.api.types.is_datetime64_any_dtype(df_4h["open_time"]):
        df_4h["open_time"] = pd.to_datetime(df_4h["open_time"])

    # For 4h bars, the bar is only "complete" at the close of that 4h period
    # So we shift the 4h open_time forward by 4 hours to represent when the data is available
    # Then use merge_asof to find the most recent completed 4h bar

    df_4h_shifted = df_4h.copy()
    df_4h_shifted["available_time"] = df_4h_shifted["open_time"] + pd.Timedelta(hours=4)

    # Rename 4h columns to avoid conflicts
    df_4h_shifted = df_4h_shifted.rename(columns={
        "open": "open_4h",
        "high": "high_4h",
        "low": "low_4h",
        "close": "close_4h",
        "volume": "volume_4h",
    })

    # Sort both DataFrames
    df_1h = df_1h.sort_values("open_time")
    df_4h_shifted = df_4h_shifted.sort_values("available_time")

    # Ensure same datetime resolution for merge_asof (pandas 2.x compat)
    df_4h_shifted["available_time"] = df_4h_shifted["available_time"].astype(df_1h["open_time"].dtype)

    # Merge: for each 1h bar, get the most recent 4h bar that was available
    # The 1h bar at time T can see 4h data from bars that closed before T
    merged = pd.merge_asof(
        df_1h,
        df_4h_shifted[["available_time", "open_4h", "high_4h", "low_4h", "close_4h", "volume_4h"]],
        left_on="open_time",
        right_on="available_time",
        direction="backward",
    )

    # Drop the temporary column
    merged = merged.drop(columns=["available_time"], errors="ignore")

    return merged
