"""
Signal Generation for ML-Assisted Crypto Trading Research Pipeline.

Implements multiple candidate entry/exit signal generators:
1. Breakout: Trend-following breakout with HTF filter
2. Mean-Reversion: RSI oversold + rejection wick + recent decline
3. Volume Spike: Abnormal volume + strong directional candle + momentum

Exit Signal:
- 1h close breaks below N-bar rolling low
- OR 4h trend filter fails
- OR maximum holding period reached

NO LOOKAHEAD: All signals use .shift(1) or current-bar-only data.
"""

import logging
from typing import List, Tuple

import pandas as pd
import numpy as np

from . import config

logger = logging.getLogger(__name__)


def generate_entry_signals(
    df: pd.DataFrame,
    rolling_window: int = None,
    adx_threshold: float = None,
) -> pd.Series:
    """
    Generate candidate entry signals based on breakout with HTF filter.

    Entry occurs when:
    1. 1h close > rolling_high(N).shift(1) [breakout above prior N-bar high]
    2. 4h trend filter is bullish [close_4h > ma_50_4h]

    Args:
        df: DataFrame with close, trend_4h columns
        rolling_window: N-bar lookback for rolling high/low
        adx_threshold: Minimum ADX for entry filter (default from config)

    Returns:
        Boolean Series indicating entry signals
    """
    rolling_window = rolling_window or config.ROLLING_WINDOW

    # Rolling high of the PREVIOUS N bars (shift to avoid lookahead)
    rolling_high = df["close"].rolling(window=rolling_window, min_periods=rolling_window).max()
    rolling_high_prev = rolling_high.shift(1)

    # Breakout: current close > previous N-bar high
    breakout = df["close"] > rolling_high_prev

    # 4h trend filter (already computed in features with proper lag)
    trend_filter = df["trend_4h"].fillna(False).astype(bool)

    # ADX regime filter (optional)
    if config.USE_ADX_FILTER:
        adx_col = f"adx_{config.ADX_PERIOD}"
        adx_threshold = (
            adx_threshold
            if adx_threshold is not None
            else config.ADX_MIN_THRESHOLD
        )
        if adx_col not in df.columns:
            logger.warning("ADX column '%s' missing; rejecting entries by default", adx_col)
            adx_filter = pd.Series(False, index=df.index)
        else:
            adx_filter = df[adx_col] >= adx_threshold
    else:
        adx_filter = pd.Series(True, index=df.index)

    # Entry signal: breakout AND trend filter AND ADX regime filter
    entry_signal = breakout & trend_filter & adx_filter.fillna(False)

    logger.debug(f"Generated {entry_signal.sum()} breakout entry signals")

    return entry_signal


def generate_mean_reversion_signals(df: pd.DataFrame) -> pd.Series:
    """
    Generate mean-reversion entry signals.

    Signal fires when:
    1. RSI < MR_RSI_OVERSOLD (oversold condition)
    2. lower_wick_ratio > MR_LOWER_WICK_RATIO_MIN (rejection wick = buying pressure)
    3. ret_3 < MR_RET_THRESHOLD (recent decline = bounce context)
    4. NOT breakout (avoid overlap with breakout signal)

    Required columns: rsi, lower_wick_ratio, ret_3, close, trend_4h

    Returns:
        Boolean Series indicating mean-reversion entry signals
    """
    required = ["rsi", "lower_wick_ratio", "ret_3"]
    for col in required:
        if col not in df.columns:
            logger.warning("MR signal: missing column '%s', returning no signals", col)
            return pd.Series(False, index=df.index)

    rsi_oversold = df["rsi"] < config.MR_RSI_OVERSOLD
    wick_rejection = df["lower_wick_ratio"] > config.MR_LOWER_WICK_RATIO_MIN
    recent_decline = df["ret_3"] < config.MR_RET_THRESHOLD

    # Exclude bars where breakout also fires (avoid double-counting)
    rolling_window = config.ROLLING_WINDOW
    rolling_high = df["close"].rolling(window=rolling_window, min_periods=rolling_window).max()
    rolling_high_prev = rolling_high.shift(1)
    is_breakout = df["close"] > rolling_high_prev
    not_breakout = ~is_breakout.fillna(True)

    # Bearish trend guard: don't take MR longs in strong downtrends
    # Allow if 4h trend is bullish OR 24h return is not deeply negative
    if "trend_4h" in df.columns and "ret_24" in df.columns:
        trend_4h = df["trend_4h"].fillna(False).astype(bool)
        not_deep_downtrend = df["ret_24"] > -0.05
        trend_guard = trend_4h | not_deep_downtrend
    elif "trend_4h" in df.columns:
        trend_guard = df["trend_4h"].fillna(False).astype(bool) | pd.Series(True, index=df.index)
    else:
        trend_guard = pd.Series(True, index=df.index)

    signal = rsi_oversold & wick_rejection & recent_decline & not_breakout & trend_guard

    # Fill NaN as False
    signal = signal.fillna(False)

    logger.debug(f"Generated {signal.sum()} mean-reversion signals")
    return signal


def generate_volume_spike_signals(df: pd.DataFrame) -> pd.Series:
    """
    Generate volume-spike entry signals.

    Signal fires when:
    1. volume_zscore > VS_VOLUME_ZSCORE_MIN OR volume_ratio > VS_VOLUME_RATIO_MIN
    2. body_ratio > VS_BODY_RATIO_MIN (strong directional candle)
    3. ret_1 > VS_RET_1_MIN (positive momentum)
    4. trend_4h support OR close > MA20 (some trend context)

    Required columns: volume_zscore, body_ratio, ret_1, trend_4h, close

    Returns:
        Boolean Series indicating volume-spike entry signals
    """
    # Check required columns
    required = ["body_ratio", "ret_1"]
    for col in required:
        if col not in df.columns:
            logger.warning("VS signal: missing column '%s', returning no signals", col)
            return pd.Series(False, index=df.index)

    # Volume condition: either zscore or ratio is high
    vol_cond = pd.Series(False, index=df.index)
    if "volume_zscore" in df.columns:
        vol_cond = vol_cond | (df["volume_zscore"] > config.VS_VOLUME_ZSCORE_MIN)
    if "volume_ratio" in df.columns:
        vol_cond = vol_cond | (df["volume_ratio"] > config.VS_VOLUME_RATIO_MIN)

    if not vol_cond.any():
        return pd.Series(False, index=df.index)

    body_strong = df["body_ratio"] > config.VS_BODY_RATIO_MIN
    positive_momentum = df["ret_1"] > config.VS_RET_1_MIN

    # Trend context: 4h trend bullish OR close > 20-bar MA
    trend_4h = df["trend_4h"].fillna(False).astype(bool)
    ma_20 = df["close"].rolling(window=config.ROLLING_WINDOW, min_periods=config.ROLLING_WINDOW).mean()
    close_above_ma = df["close"] > ma_20
    trend_context = trend_4h | close_above_ma.fillna(False)

    signal = vol_cond & body_strong & positive_momentum & trend_context

    # Fill NaN as False
    signal = signal.fillna(False)

    logger.debug(f"Generated {signal.sum()} volume-spike signals")
    return signal


def generate_exit_signals(
    df: pd.DataFrame,
    rolling_window: int = None,
) -> pd.Series:
    """
    Generate exit signals based on breakdown or trend reversal.

    Exit occurs when:
    1. 1h close < rolling_low(N).shift(1) [breakdown below prior N-bar low]
    2. OR 4h trend filter turns bearish

    Args:
        df: DataFrame with close, trend_4h columns
        rolling_window: N-bar lookback for rolling low

    Returns:
        Boolean Series indicating exit signals
    """
    rolling_window = rolling_window or config.ROLLING_WINDOW

    # Rolling low of the PREVIOUS N bars (shift to avoid lookahead)
    rolling_low = df["close"].rolling(window=rolling_window, min_periods=rolling_window).min()
    rolling_low_prev = rolling_low.shift(1)

    # Breakdown: current close < previous N-bar low
    breakdown = df["close"] < rolling_low_prev

    # 4h trend filter failure
    trend_filter_fail = ~df["trend_4h"].fillna(True).astype(bool)

    # Exit signal: breakdown OR trend filter fails
    exit_signal = breakdown | trend_filter_fail

    logger.debug(f"Generated {exit_signal.sum()} exit signals")

    return exit_signal


def _build_candidate_rows(
    df: pd.DataFrame,
    signal_mask: pd.Series,
    signal_type: str,
) -> List[dict]:
    """
    Build candidate trade rows from a boolean signal mask.

    Args:
        df: DataFrame with OHLCV + features
        signal_mask: Boolean Series indicating signals
        signal_type: Signal type string ("breakout", "mean_reversion", "volume_spike")

    Returns:
        List of candidate dicts with entry/execution info + signal_type
    """
    entry_mask = signal_mask.fillna(False)
    entry_indices = df.index[entry_mask].tolist()

    candidates = []
    for idx in entry_indices:
        row = df.loc[idx]

        # T+1 execution: find next bar's open price
        next_rows = df.loc[df["open_time"] > row["open_time"]]
        if len(next_rows) == 0:
            continue  # No next bar available, skip
        next_row = next_rows.iloc[0]

        candidates.append({
            "entry_idx": idx,
            "entry_time": row["open_time"],             # Signal bar time (for merging)
            "entry_price": row["close"],                 # Signal price (reference)
            "execution_time": next_row["open_time"],     # T+1 bar open time
            "execution_price": next_row["open"],         # T+1 bar open price
            "signal_type": signal_type,
        })

    return candidates


def generate_candidates(
    df: pd.DataFrame,
    rolling_window: int = None,
) -> pd.DataFrame:
    """
    Generate candidate trades from all enabled signal types.

    Each signal type produces its own candidates. Same bar can produce
    multiple candidates from different signal types. The ML model will
    evaluate each and the backtest picks the highest-probability one.

    Args:
        df: DataFrame with features including close, trend_4h, rsi, etc.
        rolling_window: N-bar lookback for breakout detection

    Returns:
        DataFrame with candidate trades including signal_type and signal_type_encoded
    """
    rolling_window = rolling_window or config.ROLLING_WINDOW

    df = df.copy()

    # Generate exit signals for reference
    exit_signals = generate_exit_signals(df, rolling_window)
    df["exit_signal"] = exit_signals

    all_candidates = []

    # Breakout signals
    if config.SIGNAL_BREAKOUT_ENABLED:
        breakout_signals = generate_entry_signals(df, rolling_window)
        df["entry_signal"] = breakout_signals
        breakout_cands = _build_candidate_rows(df, breakout_signals, "breakout")
        all_candidates.extend(breakout_cands)
        logger.info(f"Breakout: {len(breakout_cands)} candidates")

    # Mean-reversion signals
    if config.SIGNAL_MEAN_REVERSION_ENABLED:
        mr_signals = generate_mean_reversion_signals(df)
        mr_cands = _build_candidate_rows(df, mr_signals, "mean_reversion")
        all_candidates.extend(mr_cands)
        logger.info(f"Mean-reversion: {len(mr_cands)} candidates")

    # Volume-spike signals
    if config.SIGNAL_VOLUME_SPIKE_ENABLED:
        vs_signals = generate_volume_spike_signals(df)
        vs_cands = _build_candidate_rows(df, vs_signals, "volume_spike")
        all_candidates.extend(vs_cands)
        logger.info(f"Volume-spike: {len(vs_cands)} candidates")

    if not all_candidates:
        logger.warning("No candidate entries generated from any signal type")
        return pd.DataFrame()

    df_candidates = pd.DataFrame(all_candidates)

    # Add signal type encoding
    df_candidates["signal_type_encoded"] = df_candidates["signal_type"].map(
        config.SIGNAL_TYPE_MAP
    )

    # Store entry_signal in df for backward compatibility (breakout)
    if "entry_signal" not in df.columns:
        df["entry_signal"] = False

    logger.info(f"Generated {len(df_candidates)} total candidates across all signal types")

    return df_candidates


def add_signal_columns(
    df: pd.DataFrame,
    rolling_window: int = None,
) -> pd.DataFrame:
    """
    Add entry and exit signal columns to the feature DataFrame.

    This is useful for backtesting and analysis.

    Args:
        df: Feature DataFrame
        rolling_window: N-bar lookback

    Returns:
        DataFrame with entry_signal and exit_signal columns added
    """
    df = df.copy()
    df["entry_signal"] = generate_entry_signals(df, rolling_window)
    df["exit_signal"] = generate_exit_signals(df, rolling_window)
    return df


def count_signals(df: pd.DataFrame) -> dict:
    """
    Count entry and exit signals in the DataFrame.

    Args:
        df: DataFrame with entry_signal and exit_signal columns

    Returns:
        Dict with signal counts and rates
    """
    total_bars = len(df)
    entry_count = df["entry_signal"].sum() if "entry_signal" in df.columns else 0
    exit_count = df["exit_signal"].sum() if "exit_signal" in df.columns else 0

    return {
        "total_bars": total_bars,
        "entry_signals": int(entry_count),
        "exit_signals": int(exit_count),
        "entry_rate": entry_count / total_bars if total_bars > 0 else 0,
        "exit_rate": exit_count / total_bars if total_bars > 0 else 0,
    }
