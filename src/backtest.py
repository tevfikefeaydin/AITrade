"""
Realistic Backtest Engine for ML-Assisted Crypto Trading Research Pipeline.

Implements a realistic spot backtest with:
- T+1 execution: signal at bar close t, execute at bar t+1 open
- Cost model: fees + slippage on entry AND exit
- Realistic barrier fill using 1m data
- LONG/FLAT only (spot trading, no shorting)
- Probability threshold for trade filtering

Output:
- trades.csv: Individual trade records
- equity.csv: Equity curve over time
- summary.json: Performance metrics
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from . import config
from .utils import calculate_costs, compute_barrier_prices
from .train import predict_proba

logger = logging.getLogger(__name__)


def simulate_barrier_exit(
    df_1m: pd.DataFrame,
    entry_time: pd.Timestamp,
    execution_price: float,
    entry_price_with_costs: float,
    pt: float,
    sl: float,
    max_hold_hours: int,
    fee_bps: float,
    slippage_bps: float,
    atr_value: float = None,
) -> Tuple[pd.Timestamp, float, str, float]:
    """
    Simulate exit using 1m data for realistic barrier fill.

    Args:
        df_1m: 1-minute OHLCV DataFrame
        entry_time: Entry timestamp
        execution_price: Raw market execution price (barriers based on this)
        entry_price_with_costs: Entry price after costs (what we actually paid)
        pt: Take-profit percentage
        sl: Stop-loss percentage
        max_hold_hours: Maximum holding period
        fee_bps: Fee in basis points
        slippage_bps: Slippage in basis points
        atr_value: ATR-14 at entry time (None → use fixed pt/sl)

    Returns:
        Tuple of (exit_time, exit_price_net, exit_reason, gross_exit_price)
    """
    # Barriers based on raw market price, matching labeling.py logic
    tp_price, sl_price = compute_barrier_prices(
        execution_price, pt, sl, atr_value=atr_value,
    )

    max_hold_delta = pd.Timedelta(hours=max_hold_hours)
    timeout_time = entry_time + max_hold_delta

    # Ensure df_1m has open_time as column
    if "open_time" not in df_1m.columns:
        df_1m = df_1m.reset_index()

    # Filter to relevant time window (start after entry)
    mask = (df_1m["open_time"] > entry_time) & (df_1m["open_time"] <= timeout_time)
    window_df = df_1m.loc[mask].copy()

    if len(window_df) == 0:
        # No data - exit at raw execution price minus costs
        exit_price_net = calculate_costs(execution_price, fee_bps, slippage_bps, is_buy=False)
        return entry_time, exit_price_net, "TIMEOUT", execution_price

    # Check each 1m bar
    for _, row in window_df.iterrows():
        bar_time = row["open_time"]
        bar_high = row["high"]
        bar_low = row["low"]
        bar_close = row["close"]

        tp_hit = bar_high >= tp_price
        sl_hit = bar_low <= sl_price

        if tp_hit and sl_hit:
            # Both barriers hit in same bar - determine which was hit first
            # Heuristic: compare distance from bar_open to each barrier.
            # The closer barrier was likely reached first.
            bar_open = row["open"]
            if bar_open >= tp_price:
                gross_price, exit_reason = tp_price, "TP"
            elif bar_open <= sl_price:
                gross_price, exit_reason = sl_price, "SL"
            else:
                dist_to_tp = tp_price - bar_open
                dist_to_sl = bar_open - sl_price
                if dist_to_tp <= dist_to_sl:
                    gross_price, exit_reason = tp_price, "TP"
                else:
                    gross_price, exit_reason = sl_price, "SL"
            exit_price_net = calculate_costs(gross_price, fee_bps, slippage_bps, is_buy=False)
            return bar_time, exit_price_net, exit_reason, gross_price

        if tp_hit:
            gross_price = tp_price
            exit_price_net = calculate_costs(gross_price, fee_bps, slippage_bps, is_buy=False)
            return bar_time, exit_price_net, "TP", gross_price

        if sl_hit:
            gross_price = sl_price
            exit_price_net = calculate_costs(gross_price, fee_bps, slippage_bps, is_buy=False)
            return bar_time, exit_price_net, "SL", gross_price

    # Timeout - exit at last bar's close
    last_row = window_df.iloc[-1]
    gross_price = last_row["close"]
    exit_price_net = calculate_costs(gross_price, fee_bps, slippage_bps, is_buy=False)
    return last_row["open_time"], exit_price_net, "TIMEOUT", gross_price


def run_backtest(
    symbol: str,
    df_features: pd.DataFrame,
    df_labeled: pd.DataFrame,
    model: object,
    df_1m: pd.DataFrame,
    df_1h: pd.DataFrame,
    prob_threshold: float = None,
    fee_bps: float = None,
    slippage_bps: float = None,
    pt: float = None,
    sl: float = None,
    max_hold: int = None,
    initial_capital: float = None,
    oos_predictions: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run a realistic backtest with ML-filtered trades.

    T+1 EXECUTION: Signal at bar t close -> execute at bar t+1 open.

    Args:
        symbol: Trading symbol
        df_features: Feature DataFrame with open_time
        df_labeled: Labeled candidates DataFrame
        model: Trained ML model
        df_1m: 1-minute OHLCV data
        df_1h: 1-hour OHLCV data
        prob_threshold: Minimum probability to take trade
        fee_bps: Trading fee in basis points
        slippage_bps: Slippage in basis points
        pt: Take-profit percentage
        sl: Stop-loss percentage
        max_hold: Maximum holding period in hours
        initial_capital: Starting capital

    Returns:
        Tuple of (trades_df, equity_df, summary_dict)
    """
    # Set defaults
    prob_threshold = prob_threshold if prob_threshold is not None else config.DEFAULT_PROB_THRESHOLD
    fee_bps = fee_bps if fee_bps is not None else config.DEFAULT_FEE_BPS
    slippage_bps = slippage_bps if slippage_bps is not None else config.DEFAULT_SLIPPAGE_BPS
    pt = pt if pt is not None else config.DEFAULT_PT
    sl = sl if sl is not None else config.DEFAULT_SL
    max_hold = max_hold if max_hold is not None else config.DEFAULT_MAX_HOLD
    initial_capital = initial_capital if initial_capital is not None else config.INITIAL_CAPITAL

    logger.info(f"Running backtest for {symbol}")
    logger.info(
        f"Parameters: prob_threshold={prob_threshold}, fee_bps={fee_bps}, "
        f"slippage_bps={slippage_bps}, pt={pt:.2%}, sl={sl:.2%}, max_hold={max_hold}h"
    )

    # Merge features with labeled candidates (include signal_type_encoded for multi-signal)
    label_merge_cols = ["entry_time", "entry_idx", "entry_price"]
    if "signal_type_encoded" in df_labeled.columns:
        label_merge_cols.append("signal_type_encoded")

    df_candidates = df_features.merge(
        df_labeled[label_merge_cols],
        left_on="open_time",
        right_on="entry_time",
        how="inner",
    )

    # Ensure signal_type_encoded exists (default to 0=breakout for backward compat)
    if "signal_type_encoded" not in df_candidates.columns:
        df_candidates["signal_type_encoded"] = 0

    if len(df_candidates) == 0:
        logger.warning("No candidates to backtest")
        return pd.DataFrame(), pd.DataFrame(), {}

    # Get predictions: prefer OOS fold predictions over final model
    if oos_predictions is not None:
        n_total = len(df_candidates)
        # Ensure consistent types for composite key merge
        if "signal_type_encoded" in df_candidates.columns:
            df_candidates["signal_type_encoded"] = df_candidates["signal_type_encoded"].astype(int)
        if "signal_type_encoded" in oos_predictions.columns:
            oos_predictions = oos_predictions.copy()
            oos_predictions["signal_type_encoded"] = oos_predictions["signal_type_encoded"].astype(int)

        # Use composite key for multi-signal support
        oos_merge_cols = ["open_time", "oos_probability"]
        oos_on_cols = ["open_time"]
        if "signal_type_encoded" in oos_predictions.columns:
            oos_merge_cols.append("signal_type_encoded")
            oos_on_cols.append("signal_type_encoded")

        df_candidates = df_candidates.merge(
            oos_predictions[oos_merge_cols],
            on=oos_on_cols,
            how="inner",
        )
        df_candidates["probability"] = df_candidates["oos_probability"]
        df_candidates = df_candidates.drop(columns=["oos_probability"])
        n_dropped = n_total - len(df_candidates)
        if n_dropped > 0:
            logger.warning(
                "Dropped %d/%d candidates with no matching OOS prediction",
                n_dropped, n_total,
            )
        logger.info(
            f"Using {len(df_candidates)} OOS predictions (out of {n_total} candidates)"
        )
    else:
        proba = predict_proba(model, df_candidates)
        df_candidates["probability"] = proba

    # Filter by probability threshold
    df_filtered = df_candidates[df_candidates["probability"] >= prob_threshold].copy()
    logger.info(
        f"Filtered: {len(df_filtered)} / {len(df_candidates)} candidates "
        f"passed threshold {prob_threshold}"
    )

    if len(df_filtered) == 0:
        logger.warning("No trades after filtering")
        return pd.DataFrame(), pd.DataFrame(), {}

    # Sort by entry time
    df_filtered = df_filtered.sort_values("entry_time").reset_index(drop=True)

    # Ensure 1m and 1h data are sorted
    df_1m = df_1m.sort_values("open_time").reset_index(drop=True)
    df_1h = df_1h.sort_values("open_time").reset_index(drop=True)

    # Build ATR lookup for dynamic barriers
    atr_col = f"atr_{config.ADX_PERIOD}"
    atr_lookup = {}
    if config.USE_ATR_BARRIERS and atr_col in df_features.columns:
        for _, frow in df_features[["open_time", atr_col]].dropna().iterrows():
            atr_lookup[frow["open_time"]] = frow[atr_col]
        logger.info("ATR barrier mode: %d entries in ATR lookup", len(atr_lookup))

    # Run simulation
    trades = []
    equity_curve = []
    capital = initial_capital
    # We simulate exits immediately, so track a cooldown to prevent overlapping trades.
    blocked_until_time = None

    # Track equity at each hour
    for _, hour_row in df_1h.iterrows():
        hour_time = hour_row["open_time"]

        # Check for entry signal at this hour
        entry_rows = df_filtered[df_filtered["entry_time"] == hour_time]

        if len(entry_rows) > 0:
            # We have an entry signal; pick highest probability candidate.
            entry_row = entry_rows.sort_values("probability", ascending=False).iloc[0]

            # T+1 execution: find next hour's open price
            next_hour_mask = df_1h["open_time"] > hour_time
            if next_hour_mask.sum() == 0:
                continue

            next_hour_row = df_1h[next_hour_mask].iloc[0]
            execution_price = next_hour_row["open"]
            execution_time = next_hour_row["open_time"]

            if blocked_until_time is not None and execution_time <= blocked_until_time:
                continue

            # Apply entry costs
            entry_price_with_costs = calculate_costs(execution_price, fee_bps, slippage_bps, is_buy=True)

            # Calculate position size (invest all capital)
            position_size = capital / entry_price_with_costs

            # Simulate exit using 1m data
            trade_atr = atr_lookup.get(hour_time)
            exit_time, exit_price_net, exit_reason, gross_exit_price = simulate_barrier_exit(
                df_1m=df_1m,
                entry_time=execution_time,
                execution_price=execution_price,
                entry_price_with_costs=entry_price_with_costs,
                pt=pt,
                sl=sl,
                max_hold_hours=max_hold,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                atr_value=trade_atr,
            )

            # Calculate P&L
            gross_return = (gross_exit_price - execution_price) / execution_price
            net_return = (exit_price_net - entry_price_with_costs) / entry_price_with_costs
            pnl = position_size * entry_price_with_costs * net_return

            # Update capital
            capital = capital + pnl
            blocked_until_time = pd.Timestamp(exit_time)

            # Record trade
            trades.append({
                "entry_time": execution_time,
                "exit_time": exit_time,
                "entry_price": execution_price,
                "entry_price_with_costs": entry_price_with_costs,
                "exit_price_gross": gross_exit_price,
                "exit_price_net": exit_price_net,
                "position_size": position_size,
                "gross_return": gross_return,
                "net_return": net_return,
                "pnl": pnl,
                "exit_reason": exit_reason,
                "probability": entry_row["probability"],
            })

        # Record equity
        equity_curve.append({
            "timestamp": hour_time,
            "equity": capital,
        })

    # Create DataFrames
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    # Add drawdown column to equity curve
    if len(equity_df) > 0:
        equity_df["cummax"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = (equity_df["equity"] / equity_df["cummax"]) - 1
        equity_df = equity_df.drop(columns=["cummax"])

    if len(trades_df) > 0:
        # Calculate cumulative metrics
        trades_df["cumulative_return"] = (1 + trades_df["net_return"]).cumprod() - 1

        # Log summary
        n_winners = (trades_df["net_return"] > 0).sum()
        n_losers = (trades_df["net_return"] <= 0).sum()
        total_return = (capital - initial_capital) / initial_capital

        logger.info(
            f"Backtest complete: {len(trades_df)} trades, "
            f"{n_winners} wins, {n_losers} losses, "
            f"total return: {total_return:.2%}"
        )

    # Create summary
    from .metrics import compute_all_metrics
    summary = compute_all_metrics(trades_df, equity_df)
    summary["symbol"] = symbol
    summary["n_candidates"] = len(df_candidates)
    summary["n_filtered"] = len(df_filtered)
    summary["prob_threshold"] = prob_threshold

    # Add metadata
    summary["metadata"] = {
        "backtest_start": df_1h["open_time"].min().isoformat() if len(df_1h) > 0 else None,
        "backtest_end": df_1h["open_time"].max().isoformat() if len(df_1h) > 0 else None,
        "fee_bps": fee_bps,
        "slippage_bps": slippage_bps,
        "pt": pt,
        "sl": sl,
        "max_hold": max_hold,
        "initial_capital": initial_capital,
    }

    return trades_df, equity_df, summary


def run_baseline_all_candidates(
    symbol: str,
    df_features: pd.DataFrame,
    df_labeled: pd.DataFrame,
    df_1m: pd.DataFrame,
    df_1h: pd.DataFrame,
    fee_bps: float = None,
    slippage_bps: float = None,
    pt: float = None,
    sl: float = None,
    max_hold: int = None,
    initial_capital: float = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run baseline backtest taking ALL candidates (no ML filter).

    Same as run_backtest but with prob_threshold=0 (take all trades).
    """
    # Create a dummy model that returns 1.0 for all predictions
    class DummyModel:
        def __init__(self):
            self.feature_cols_ = []

        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    dummy_model = DummyModel()

    return run_backtest(
        symbol=symbol,
        df_features=df_features,
        df_labeled=df_labeled,
        model=dummy_model,
        df_1m=df_1m,
        df_1h=df_1h,
        prob_threshold=0.0,  # Take all candidates
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        pt=pt,
        sl=sl,
        max_hold=max_hold,
        initial_capital=initial_capital,
    )


def save_backtest_results(
    symbol: str,
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    summary: Dict,
) -> Dict[str, Path]:
    """
    Save backtest results to files.

    Args:
        symbol: Trading symbol
        trades_df: Trades DataFrame
        equity_df: Equity curve DataFrame
        summary: Summary dictionary

    Returns:
        Dict mapping output type to file path
    """
    output_paths = config.get_symbol_output_paths(symbol)

    # Ensure directory exists
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save trades
    if len(trades_df) > 0:
        trades_df.to_csv(output_paths["trades"], index=False)
        logger.info(f"Saved trades to {output_paths['trades']}")

    # Save equity curve
    if len(equity_df) > 0:
        equity_df.to_csv(output_paths["equity"], index=False)
        logger.info(f"Saved equity curve to {output_paths['equity']}")

    # Save summary
    from .metrics import save_summary
    save_summary(summary, output_paths["summary"])

    return output_paths
