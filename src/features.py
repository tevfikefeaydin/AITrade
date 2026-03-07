"""
Feature Engineering for ML-Assisted Crypto Trading Research Pipeline.

CRITICAL: All features are computed with STRICT NO-LOOKAHEAD guarantees.
Features at time t can only use data from bars that closed at or before time t.

Feature Categories:
1. Price-based (wicks, body, range, ratios)
2. Returns (log returns at various lookbacks)
3. Volatility (rolling standard deviation)
4. Momentum (RSI, MA gap)
5. Volume (z-score)
6. Intrabar (from 1m data)
7. Higher timeframe context (4h)
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

from . import config
from .utils import safe_divide
from .resample import compute_intrabar_features, align_4h_to_1h

logger = logging.getLogger(__name__)


def compute_wick_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute candlestick wick features.

    Wicks measure rejection from price levels and are computed from the
    relationship between OHLC values within each bar.

    Args:
        df: DataFrame with open, high, low, close columns

    Returns:
        DataFrame with wick features added
    """
    df = df.copy()

    # Body is the absolute difference between open and close
    df["body"] = (df["close"] - df["open"]).abs()

    # Range is high - low (total bar range)
    df["range"] = df["high"] - df["low"]

    # Upper wick: high minus the higher of open/close
    upper_body = df[["open", "close"]].max(axis=1)
    df["upper_wick"] = df["high"] - upper_body

    # Lower wick: the lower of open/close minus low
    lower_body = df[["open", "close"]].min(axis=1)
    df["lower_wick"] = lower_body - df["low"]

    # Ratios (safe divide to handle zero range)
    df["upper_wick_ratio"] = safe_divide(df["upper_wick"], df["range"], 0.0)
    df["lower_wick_ratio"] = safe_divide(df["lower_wick"], df["range"], 0.0)
    df["body_ratio"] = safe_divide(df["body"], df["range"], 0.0)

    return df


def compute_return_features(
    df: pd.DataFrame,
    periods: Optional[list] = None,
) -> pd.DataFrame:
    """
    Compute log return features at various lookback periods.

    NO LOOKAHEAD: Returns are computed as log(close_t / close_{t-n}),
    which only uses past data.

    Args:
        df: DataFrame with close column
        periods: List of lookback periods (default from config)

    Returns:
        DataFrame with return features added
    """
    df = df.copy()
    periods = periods or config.RETURN_PERIODS

    for n in periods:
        # Log return: only uses data at t and t-n (past data only)
        df[f"ret_{n}"] = np.log(df["close"] / df["close"].shift(n))

    # 24-hour return (NO LOOKAHEAD: uses close at t and t-24)
    df["ret_24"] = np.log(df["close"] / df["close"].shift(24))

    return df


def compute_volatility_features(
    df: pd.DataFrame,
    window: int = None,
) -> pd.DataFrame:
    """
    Compute rolling volatility features.

    NO LOOKAHEAD: Rolling window is backward-looking only.

    Args:
        df: DataFrame with ret_1 column (1-bar returns)
        window: Rolling window size (default from config)

    Returns:
        DataFrame with volatility features added
    """
    df = df.copy()
    window = window or config.FEATURE_ROLLING_WINDOW

    if "ret_1" not in df.columns:
        df["ret_1"] = np.log(df["close"] / df["close"].shift(1))

    # Rolling volatility (standard deviation of returns)
    df[f"vol_{window}"] = df["ret_1"].rolling(window=window, min_periods=window).std()

    return df


def compute_rsi(
    df: pd.DataFrame,
    period: int = None,
) -> pd.DataFrame:
    """
    Compute Relative Strength Index (RSI).

    NO LOOKAHEAD: RSI uses only past price changes.

    Args:
        df: DataFrame with close column
        period: RSI lookback period (default from config)

    Returns:
        DataFrame with RSI feature added
    """
    df = df.copy()
    period = period or config.RSI_PERIOD

    # Price changes
    delta = df["close"].diff()

    # Separate gains and losses
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    # Exponential moving average of gains and losses
    avg_gains = gains.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_losses = losses.ewm(span=period, min_periods=period, adjust=False).mean()

    # RSI calculation
    # When avg_losses=0 (all gains), RS->inf and RSI->100
    # When avg_gains=0 (all losses), RS=0 and RSI=0
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    # Handle division edge cases: avg_losses=0 → RSI=100, both=0 → RSI=50
    rsi = rsi.fillna(50.0)
    rsi = rsi.where(avg_losses != 0, 100.0)
    rsi = rsi.where((avg_losses != 0) | (avg_gains != 0), 50.0)
    df["rsi"] = rsi

    return df


def compute_ma_features(
    df: pd.DataFrame,
    window: int = None,
) -> pd.DataFrame:
    """
    Compute moving average related features.

    NO LOOKAHEAD: MA is computed with backward-looking window.

    Args:
        df: DataFrame with close column
        window: MA window size (default from config)

    Returns:
        DataFrame with MA features added
    """
    df = df.copy()
    window = window or config.FEATURE_ROLLING_WINDOW

    # Simple moving average
    df[f"ma_{window}"] = df["close"].rolling(window=window, min_periods=window).mean()

    # MA gap: how far price is from MA (percentage)
    df["ma_gap"] = safe_divide(df["close"], df[f"ma_{window}"], 1.0) - 1

    return df


def compute_volume_features(
    df: pd.DataFrame,
    window: int = None,
) -> pd.DataFrame:
    """
    Compute volume-based features.

    NO LOOKAHEAD: Rolling window is backward-looking.

    Args:
        df: DataFrame with volume column
        window: Rolling window size

    Returns:
        DataFrame with volume features added
    """
    df = df.copy()
    window = window or config.FEATURE_ROLLING_WINDOW

    # Rolling mean and std of volume
    vol_mean = df["volume"].rolling(window=window, min_periods=window).mean()
    vol_std = df["volume"].rolling(window=window, min_periods=window).std()

    # Volume z-score: how unusual is current volume
    df["volume_zscore"] = safe_divide(df["volume"] - vol_mean, vol_std, 0.0)

    # Volume ratio: current volume vs rolling mean (NO LOOKAHEAD: backward-looking)
    df["volume_ratio"] = safe_divide(df["volume"], vol_mean, 1.0)

    return df


def compute_adx_features(
    df: pd.DataFrame,
    period: int = None,
) -> pd.DataFrame:
    """
    Compute Average Directional Index (ADX) from OHLC data.

    NO LOOKAHEAD: Uses only current and prior bars via diff and Wilder smoothing.

    Args:
        df: DataFrame with high, low, close columns
        period: ADX lookback period (default from config)

    Returns:
        DataFrame with adx_{period} column added
    """
    df = df.copy()
    period = period or config.ADX_PERIOD

    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_components = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)

    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_dm_smoothed = pd.Series(plus_dm, index=df.index).ewm(
        alpha=alpha, min_periods=period, adjust=False
    ).mean()
    minus_dm_smoothed = pd.Series(minus_dm, index=df.index).ewm(
        alpha=alpha, min_periods=period, adjust=False
    ).mean()

    plus_di = 100 * safe_divide(plus_dm_smoothed, atr, 0.0)
    minus_di = 100 * safe_divide(minus_dm_smoothed, atr, 0.0)
    dx = 100 * safe_divide((plus_di - minus_di).abs(), plus_di + minus_di, 0.0)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    df[f"atr_{period}"] = atr.fillna(0.0)
    df[f"adx_{period}"] = adx.fillna(0.0)

    # ATR ratio: current ATR vs 20-bar rolling mean (NO LOOKAHEAD: backward-looking)
    atr_rolling_mean = atr.rolling(window=20, min_periods=20).mean()
    df["atr_ratio"] = safe_divide(atr, atr_rolling_mean, 1.0)

    return df


def compute_4h_context_features(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    ma_length: int = None,
) -> pd.DataFrame:
    """
    Compute 4-hour timeframe context features aligned to 1h bars.

    NO LOOKAHEAD: We only use 4h bars that have completed before each 1h bar.

    Args:
        df_1h: 1-hour DataFrame
        df_4h: 4-hour DataFrame
        ma_length: MA length for 4h trend filter

    Returns:
        1h DataFrame with 4h context features added
    """
    ma_length = ma_length or config.MA_LENGTH

    # Align 4h data to 1h (with proper lag to avoid lookahead)
    df_merged = align_4h_to_1h(df_1h, df_4h)

    # Compute 4h MA on the 4h data first
    df_4h = df_4h.copy()
    df_4h[f"ma_{ma_length}_4h"] = (
        df_4h["close"].rolling(window=ma_length, min_periods=ma_length).mean()
    )

    # Now align this MA to the merged 1h data
    # We need to re-align after computing the MA
    df_4h_with_ma = df_4h[["open_time", "close", f"ma_{ma_length}_4h"]].copy()
    df_4h_with_ma["available_time"] = df_4h_with_ma["open_time"] + pd.Timedelta(hours=4)
    df_4h_with_ma = df_4h_with_ma.rename(columns={
        "close": "close_4h_for_ma",
        f"ma_{ma_length}_4h": f"ma_{ma_length}_4h",
    })

    # Sort for merge_asof
    df_merged = df_merged.sort_values("open_time")
    df_4h_with_ma = df_4h_with_ma.sort_values("available_time")

    # Ensure same datetime resolution (pandas 2.x compat)
    df_4h_with_ma["available_time"] = df_4h_with_ma["available_time"].astype(df_merged["open_time"].dtype)

    # Merge the MA
    df_merged = pd.merge_asof(
        df_merged,
        df_4h_with_ma[["available_time", f"ma_{ma_length}_4h"]],
        left_on="open_time",
        right_on="available_time",
        direction="backward",
    )
    df_merged = df_merged.drop(columns=["available_time"], errors="ignore")

    # 4h trend filter: is 4h close above 4h MA?
    df_merged["trend_4h"] = df_merged["close_4h"] > df_merged[f"ma_{ma_length}_4h"]

    # 4h MA slope (change in MA)
    # First compute on 4h data
    ma_col = df_4h[f"ma_{ma_length}_4h"]
    df_4h["ma_slope_4h"] = (ma_col - ma_col.shift(1)) / ma_col.shift(1)

    df_4h_slope = df_4h[["open_time", "ma_slope_4h"]].copy()
    df_4h_slope["available_time"] = df_4h_slope["open_time"] + pd.Timedelta(hours=4)

    df_4h_slope = df_4h_slope.sort_values("available_time")

    # Ensure same datetime resolution (pandas 2.x compat)
    df_4h_slope["available_time"] = df_4h_slope["available_time"].astype(df_merged["open_time"].dtype)

    df_merged = pd.merge_asof(
        df_merged,
        df_4h_slope[["available_time", "ma_slope_4h"]],
        left_on="open_time",
        right_on="available_time",
        direction="backward",
    )
    df_merged = df_merged.drop(columns=["available_time"], errors="ignore")

    return df_merged


def build_features(
    df_1m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    rolling_window: int = None,
    ma_length: int = None,
) -> pd.DataFrame:
    """
    Build complete feature set from multi-timeframe data.

    CRITICAL: This function ensures NO LOOKAHEAD in any feature computation.
    All rolling operations use backward-looking windows only.

    Args:
        df_1m: 1-minute OHLCV DataFrame
        df_1h: 1-hour OHLCV DataFrame
        df_4h: 4-hour OHLCV DataFrame
        rolling_window: Rolling window for various features
        ma_length: MA length for trend filter

    Returns:
        DataFrame with all features (indexed by 1h open_time)
    """
    rolling_window = rolling_window or config.FEATURE_ROLLING_WINDOW
    ma_length = ma_length or config.MA_LENGTH

    logger.info("Building features (with strict no-lookahead guarantees)...")

    # Start with 1h OHLCV
    df = df_1h.copy()

    # 1. Wick features (from current bar OHLC - no lookahead by construction)
    df = compute_wick_features(df)
    logger.info("  - Computed wick features")

    # 2. Return features (backward-looking)
    df = compute_return_features(df)
    logger.info("  - Computed return features")

    # 3. Volatility features (backward-looking rolling)
    df = compute_volatility_features(df, window=rolling_window)
    logger.info("  - Computed volatility features")

    # 4. RSI (backward-looking)
    df = compute_rsi(df)
    logger.info("  - Computed RSI")

    # 5. MA features (backward-looking rolling)
    df = compute_ma_features(df, window=rolling_window)
    logger.info("  - Computed MA features")

    # 6. Volume features (backward-looking rolling)
    df = compute_volume_features(df, window=rolling_window)
    logger.info("  - Computed volume features")

    # 7. ADX trend-strength feature on 1h bars
    df = compute_adx_features(df)
    logger.info("  - Computed ADX features")

    # 8. Intrabar features from 1m data
    df_intrabar = compute_intrabar_features(df_1m, df_1h)
    if len(df_intrabar) > 0:
        df = df.merge(df_intrabar, on="open_time", how="left")
        logger.info("  - Merged intrabar features")

    # 9. 4h context features (with proper lag)
    df = compute_4h_context_features(df, df_4h, ma_length=ma_length)
    logger.info("  - Computed 4h context features")

    # 10. Hour-of-day cyclical encoding (NO LOOKAHEAD: uses bar's own timestamp)
    hour = df["open_time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    logger.info("  - Computed hour-of-day features")

    # 11. Day-of-week cyclical encoding (NO LOOKAHEAD: uses bar's own timestamp)
    dow = df["open_time"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    logger.info("  - Computed day-of-week features")

    # 12. RSI slope: direction of RSI over last 3 bars (NO LOOKAHEAD: backward-looking diff)
    if "rsi" in df.columns:
        df["rsi_slope"] = df["rsi"] - df["rsi"].shift(3)
    logger.info("  - Computed RSI slope feature")

    # Fill any remaining NaN with 0 (for early bars without enough history)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    logger.info(f"Built {len(df):,} rows with {len(df.columns)} columns")

    return df


def get_feature_columns() -> list:
    """
    Get the list of feature column names used for model training.

    Returns:
        List of feature column names
    """
    return [
        # Wick features (ratios only — raw values are price-level dependent)
        "upper_wick_ratio",
        "lower_wick_ratio",
        "body_ratio",
        # Return features
        "ret_1",
        "ret_3",
        "ret_6",
        "ret_12",
        "ret_24",
        # Volatility
        "vol_20",
        # Momentum
        "rsi",
        "rsi_slope",
        "ma_gap",
        # Volume
        "volume_zscore",
        "volume_ratio",
        # Intrabar
        "max_runup",
        "max_drawdown",
        "intrabar_vol",
        "intrabar_skew",
        "up_down_ratio",
        # 4h context
        "trend_4h",
        "ma_slope_4h",
        # Market regime
        "atr_ratio",
        # Time cyclical
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        # Signal type (set by generate_candidates, not build_features)
        "signal_type_encoded",
    ]
