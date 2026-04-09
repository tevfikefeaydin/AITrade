"""
Triple-Barrier Labeling for ML-Assisted Crypto Trading Research Pipeline.

Implements the triple-barrier labeling method:
- Take-profit (TP) barrier: price reaches +pt% above entry
- Stop-loss (SL) barrier: price reaches -sl% below entry
- Time barrier: maximum holding period (max_hold) reached

Label assignment:
- y = 1.0 if TP barrier hit first (successful trade)
- y = 0.0 if SL barrier hit first (unsuccessful trade)
- y = 0.0..1.0 if timeout: fractional label scaled by exit return
  position between SL and TP (0 at SL level, 1 at TP level)

Uses 1-minute data for precise barrier detection.
"""

import logging
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

from . import config
from .utils import compute_barrier_prices

logger = logging.getLogger(__name__)


def find_barrier_touch(
    df_1m: pd.DataFrame,
    entry_time: pd.Timestamp,
    entry_price: float,
    pt: float,
    sl: float,
    max_hold_hours: int,
    atr_value: float = None,
    cost_adjust: bool = False,
) -> Tuple[int, pd.Timestamp, float, str]:
    """
    Find which barrier is touched first using 1-minute data.

    Args:
        df_1m: 1-minute OHLCV DataFrame with open_time index or column
        entry_time: Entry timestamp (1h bar close time)
        entry_price: Entry price
        pt: Take-profit barrier as decimal (e.g., 0.008 for 0.8%)
        sl: Stop-loss barrier as decimal (e.g., 0.006 for 0.6%)
        max_hold_hours: Maximum holding period in hours
        atr_value: ATR-14 at entry time (None → use fixed pt/sl)
        cost_adjust: If True, add round-trip costs to TP (for labeling only)

    Returns:
        Tuple of (label, exit_time, exit_price, exit_reason)
        - label: 1.0 if TP hit, 0.0 if SL hit, 0.0-1.0 fractional if timeout
        - exit_time: Timestamp of exit
        - exit_price: Price at exit
        - exit_reason: "TP", "SL", or "TIMEOUT"
    """
    # Calculate barrier prices (ATR-based or fixed %)
    tp_price, sl_price = compute_barrier_prices(
        entry_price, pt, sl, atr_value=atr_value,
        cost_adjust=cost_adjust,
    )

    # Calculate time horizon
    max_hold_delta = pd.Timedelta(hours=max_hold_hours)
    timeout_time = entry_time + max_hold_delta

    # Ensure df_1m has open_time as column
    if "open_time" not in df_1m.columns:
        df_1m = df_1m.reset_index()

    # Filter to relevant time window
    # Start from entry_time (the next 1m bar after entry signal)
    mask = (df_1m["open_time"] > entry_time) & (df_1m["open_time"] <= timeout_time)
    window_df = df_1m.loc[mask].copy()

    if len(window_df) == 0:
        # No data in window - treat as timeout at entry price
        return 0, entry_time, entry_price, "TIMEOUT"

    # Check each 1m bar for barrier touch
    for _, row in window_df.iterrows():
        bar_time = row["open_time"]
        bar_high = row["high"]
        bar_low = row["low"]
        bar_close = row["close"]

        # Check TP hit (high reaches TP price)
        tp_hit = bar_high >= tp_price

        # Check SL hit (low reaches SL price)
        sl_hit = bar_low <= sl_price

        if tp_hit and sl_hit:
            # Both barriers hit in same bar - determine which was hit first
            # Heuristic: compare distance from bar_open to each barrier.
            # The closer barrier was likely reached first.
            bar_open = row["open"]
            if bar_open >= tp_price:
                # Already at/above TP at open → TP hit first
                return 1, bar_time, tp_price, "TP"
            elif bar_open <= sl_price:
                # Already at/below SL at open → SL hit first
                return 0, bar_time, sl_price, "SL"
            else:
                dist_to_tp = tp_price - bar_open
                dist_to_sl = bar_open - sl_price
                if dist_to_tp <= dist_to_sl:
                    return 1, bar_time, tp_price, "TP"
                else:
                    return 0, bar_time, sl_price, "SL"

        if tp_hit:
            return 1, bar_time, tp_price, "TP"

        if sl_hit:
            return 0, bar_time, sl_price, "SL"

    # Timeout - no barrier hit within max_hold
    # Exit at the close of the last bar in window
    last_row = window_df.iloc[-1]
    exit_price = last_row["close"]

    # Fractional label for timeout: scale by how close to TP vs SL
    # 0.0 if at SL level, 0.5 if at entry, 1.0 if at TP level
    timeout_return = (exit_price - entry_price) / entry_price
    tp_return = (tp_price - entry_price) / entry_price
    sl_return = (sl_price - entry_price) / entry_price
    barrier_range = tp_return - sl_return
    if barrier_range > 0:
        label = np.clip((timeout_return - sl_return) / barrier_range, 0.0, 1.0)
    else:
        label = 0.0

    return float(label), last_row["open_time"], exit_price, "TIMEOUT"


def label_candidates(
    df_candidates: pd.DataFrame,
    df_1m: pd.DataFrame,
    pt: float = None,
    sl: float = None,
    max_hold: int = None,
    df_features: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Apply triple-barrier labeling to candidate entries.

    For each candidate entry:
    1. Look forward in 1m data
    2. Find which barrier (TP, SL, or time) is touched first
    3. Assign label: 1.0 if TP, 0.0 if SL, fractional 0-1 if timeout

    Args:
        df_candidates: DataFrame with entry_time, entry_price columns
        df_1m: 1-minute OHLCV DataFrame
        pt: Take-profit percentage (default from config)
        sl: Stop-loss percentage (default from config)
        max_hold: Maximum holding period in hours (default from config)

    Returns:
        DataFrame with original columns plus:
        - label: 0 or 1
        - exit_time: When position was closed
        - exit_price: Price at exit
        - exit_reason: "TP", "SL", or "TIMEOUT"
    """
    pt = pt if pt is not None else config.DEFAULT_PT
    sl = sl if sl is not None else config.DEFAULT_SL
    max_hold = max_hold if max_hold is not None else config.DEFAULT_MAX_HOLD

    if len(df_candidates) == 0:
        return df_candidates

    logger.info(
        f"Labeling {len(df_candidates)} candidates with "
        f"PT={pt:.2%}, SL={sl:.2%}, max_hold={max_hold}h"
    )

    # Ensure 1m data is sorted
    df_1m = df_1m.sort_values("open_time").reset_index(drop=True)

    # Use T+1 execution columns if available, fall back to entry columns
    has_execution = "execution_time" in df_candidates.columns and "execution_price" in df_candidates.columns

    # Build ATR lookup if features are provided and ATR barriers enabled
    atr_lookup = {}
    atr_col = f"atr_{config.ADX_PERIOD}"
    if config.USE_ATR_BARRIERS and df_features is not None and atr_col in df_features.columns:
        for _, frow in df_features[["open_time", atr_col]].dropna().iterrows():
            atr_lookup[frow["open_time"]] = frow[atr_col]
        logger.info("ATR barrier mode: %d entries in ATR lookup", len(atr_lookup))

    results = []
    for _, row in tqdm(df_candidates.iterrows(), total=len(df_candidates), desc="Labeling"):
        # Use T+1 execution data for barrier computation (matches backtest)
        if has_execution:
            barrier_time = row["execution_time"]
            barrier_price = row["execution_price"]
        else:
            barrier_time = row["entry_time"]
            barrier_price = row["entry_price"]

        # Lookup ATR at signal time (entry_time = 1h bar open_time)
        atr_value = atr_lookup.get(row["entry_time"])

        label, exit_time, exit_price, exit_reason = find_barrier_touch(
            df_1m=df_1m,
            entry_time=barrier_time,
            entry_price=barrier_price,
            pt=pt,
            sl=sl,
            max_hold_hours=max_hold,
            atr_value=atr_value,
            cost_adjust=config.COST_AWARE_LABELING,
        )

        results.append({
            **row.to_dict(),
            "label": label,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
        })

    df_labeled = pd.DataFrame(results)

    # Log statistics
    n_tp = (df_labeled["label"] == 1).sum()
    n_sl = (df_labeled["exit_reason"] == "SL").sum()
    n_timeout = (df_labeled["exit_reason"] == "TIMEOUT").sum()

    logger.info(
        f"Labeling complete: TP={n_tp} ({n_tp/len(df_labeled):.1%}), "
        f"SL={n_sl} ({n_sl/len(df_labeled):.1%}), "
        f"TIMEOUT={n_timeout} ({n_timeout/len(df_labeled):.1%})"
    )

    return df_labeled


def compute_label_statistics(df_labeled: pd.DataFrame) -> dict:
    """
    Compute statistics on labeled data.

    Args:
        df_labeled: DataFrame with label, exit_reason columns

    Returns:
        Dict with label statistics
    """
    if len(df_labeled) == 0:
        return {
            "total_candidates": 0,
            "positive_rate": 0.0,
            "tp_count": 0,
            "sl_count": 0,
            "timeout_count": 0,
        }

    n_total = len(df_labeled)
    n_positive = (df_labeled["label"] == 1).sum()
    n_tp = (df_labeled["exit_reason"] == "TP").sum()
    n_sl = (df_labeled["exit_reason"] == "SL").sum()
    n_timeout = (df_labeled["exit_reason"] == "TIMEOUT").sum()

    return {
        "total_candidates": n_total,
        "positive_rate": n_positive / n_total,
        "tp_count": int(n_tp),
        "sl_count": int(n_sl),
        "timeout_count": int(n_timeout),
        "tp_rate": n_tp / n_total,
        "sl_rate": n_sl / n_total,
        "timeout_rate": n_timeout / n_total,
    }
