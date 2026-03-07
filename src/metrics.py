"""
Performance Metrics for ML-Assisted Crypto Trading Research Pipeline.

Computes various performance metrics for trading strategies:
- Return metrics (total return, CAGR)
- Risk metrics (Sharpe ratio, max drawdown)
- Trade metrics (win rate, profit factor)
- Comparison with baselines
"""

import logging
from typing import Dict, Optional
import json

import pandas as pd
import numpy as np

from . import config

logger = logging.getLogger(__name__)


def compute_returns(equity_series: pd.Series) -> Dict[str, float]:
    """
    Compute return-based metrics from equity curve.

    Args:
        equity_series: Series of portfolio values over time

    Returns:
        Dict with return metrics
    """
    if len(equity_series) < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
        }

    initial_value = equity_series.iloc[0]
    final_value = equity_series.iloc[-1]

    # Total return
    total_return = (final_value - initial_value) / initial_value

    # CAGR (Compound Annual Growth Rate)
    # Assuming equity_series index is datetime
    if hasattr(equity_series.index, "to_pydatetime"):
        start_date = equity_series.index[0]
        end_date = equity_series.index[-1]
        years = (end_date - start_date).total_seconds() / (365.25 * 24 * 3600)
    else:
        # Fallback: assume hourly bars
        years = len(equity_series) / (365.25 * 24)

    if years > 0 and final_value > 0 and initial_value > 0:
        cagr = (final_value / initial_value) ** (1 / years) - 1
    else:
        cagr = 0.0

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
    }


def compute_risk_metrics(
    equity_series: pd.Series,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Compute risk-adjusted metrics from equity curve.

    Args:
        equity_series: Series of portfolio values over time
        risk_free_rate: Annual risk-free rate (default 0)

    Returns:
        Dict with risk metrics
    """
    if len(equity_series) < 2:
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration_hours": 0,
        }

    # Calculate returns
    returns = equity_series.pct_change().dropna()

    if len(returns) == 0:
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration_hours": 0,
        }

    # Sharpe Ratio (annualized, assuming hourly returns)
    # Annualization factor for hourly data: sqrt(365.25 * 24)
    hours_per_year = 365.25 * 24
    annualization_factor = np.sqrt(hours_per_year)

    mean_return = returns.mean()
    std_return = returns.std()

    if std_return > 0:
        hourly_rf = risk_free_rate / hours_per_year
        sharpe = (mean_return - hourly_rf) / std_return * annualization_factor
    else:
        sharpe = 0.0

    # Maximum Drawdown
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_drawdown = drawdown.min()

    # Max Drawdown Duration (in hours)
    in_drawdown = drawdown < 0
    drawdown_groups = (~in_drawdown).cumsum()
    drawdown_durations = in_drawdown.groupby(drawdown_groups).sum()
    max_dd_duration = drawdown_durations.max() if len(drawdown_durations) > 0 else 0

    return {
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "max_drawdown_duration_hours": int(max_dd_duration),
    }


def compute_trade_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute trade-level metrics.

    Args:
        trades_df: DataFrame with net_return column for each trade

    Returns:
        Dict with trade metrics
    """
    if len(trades_df) == 0 or "net_return" not in trades_df.columns:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "trade_sharpe": 0.0,
        }

    net_returns = trades_df["net_return"]
    n_trades = len(net_returns)

    # Win rate
    winners = net_returns > 0
    n_winners = winners.sum()
    win_rate = n_winners / n_trades if n_trades > 0 else 0.0

    # Profit factor (gross profit / gross loss)
    gross_profit = net_returns[winners].sum()
    gross_loss = abs(net_returns[~winners].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Average win/loss
    avg_win = net_returns[winners].mean() if n_winners > 0 else 0.0
    avg_loss = net_returns[~winners].mean() if n_trades - n_winners > 0 else 0.0

    # Expectancy (average return per trade)
    expectancy = net_returns.mean()

    # Per-trade Sharpe (annualized assuming ~250 trades/year as reference)
    # More meaningful than hourly equity Sharpe for low-frequency strategies
    std_return = net_returns.std()
    if std_return > 0 and n_trades >= 2:
        trade_sharpe = float(expectancy / std_return * np.sqrt(n_trades))
    else:
        trade_sharpe = 0.0

    return {
        "total_trades": n_trades,
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor) if profit_factor != float("inf") else 999.0,
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "expectancy": float(expectancy),
        "trade_sharpe": trade_sharpe,
    }


def compute_all_metrics(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute all performance metrics.

    Args:
        trades_df: DataFrame with trade-level data
        equity_df: DataFrame with equity curve

    Returns:
        Dict with all metrics
    """
    # Get equity series
    if "equity" in equity_df.columns:
        if "timestamp" in equity_df.columns:
            equity_series = equity_df.set_index("timestamp")["equity"]
        else:
            equity_series = equity_df["equity"]
    else:
        equity_series = pd.Series([config.INITIAL_CAPITAL])

    metrics = {}

    # Return metrics
    metrics.update(compute_returns(equity_series))

    # Risk metrics
    metrics.update(compute_risk_metrics(equity_series))

    # Trade metrics
    metrics.update(compute_trade_metrics(trades_df))

    return metrics


def compute_baseline_buy_hold(
    df_1h: pd.DataFrame,
    initial_capital: float = None,
) -> Dict[str, float]:
    """
    Compute buy-and-hold baseline metrics.

    Args:
        df_1h: 1-hour OHLCV DataFrame
        initial_capital: Starting capital

    Returns:
        Dict with buy-and-hold metrics
    """
    initial_capital = initial_capital or config.INITIAL_CAPITAL

    if len(df_1h) < 2:
        return {
            "bh_total_return": 0.0,
            "bh_cagr": 0.0,
            "bh_sharpe_ratio": 0.0,
            "bh_max_drawdown": 0.0,
        }

    # Calculate equity curve for buy-and-hold
    prices = df_1h["close"].values
    shares = initial_capital / prices[0]
    equity = shares * prices

    equity_series = pd.Series(equity, index=df_1h["open_time"] if "open_time" in df_1h.columns else None)

    return_metrics = compute_returns(equity_series)
    risk_metrics = compute_risk_metrics(equity_series)

    return {
        "bh_total_return": return_metrics["total_return"],
        "bh_cagr": return_metrics["cagr"],
        "bh_sharpe_ratio": risk_metrics["sharpe_ratio"],
        "bh_max_drawdown": risk_metrics["max_drawdown"],
    }


def create_summary(
    symbol: str,
    strategy_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    all_candidates_metrics: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Create a summary comparing strategy to baselines.

    Args:
        symbol: Trading symbol
        strategy_metrics: Metrics for ML-filtered strategy
        baseline_metrics: Buy-and-hold baseline metrics
        all_candidates_metrics: Metrics for unfiltered candidates

    Returns:
        Dict with complete summary
    """
    summary = {
        "symbol": symbol,
        "strategy": strategy_metrics,
        "baseline_buy_hold": baseline_metrics,
    }

    if all_candidates_metrics:
        summary["baseline_all_candidates"] = all_candidates_metrics

    # Compute improvements
    if baseline_metrics.get("bh_total_return", 0) != 0:
        summary["return_vs_bh"] = (
            strategy_metrics.get("total_return", 0) - baseline_metrics.get("bh_total_return", 0)
        )

    if all_candidates_metrics and all_candidates_metrics.get("total_return", 0) != 0:
        summary["return_vs_all_candidates"] = (
            strategy_metrics.get("total_return", 0) - all_candidates_metrics.get("total_return", 0)
        )

    return summary


def create_leaderboard(summaries: list, formatted: bool = True) -> pd.DataFrame:
    """
    Create a leaderboard comparing multiple symbols/strategies.

    Args:
        summaries: List of summary dicts from create_summary
        formatted: If True, format numbers for readability (rounded percentages)

    Returns:
        DataFrame with side-by-side comparison
    """
    rows = []
    for summary in summaries:
        symbol = summary.get("symbol", "Unknown")
        strategy = summary.get("strategy", {})
        bh = summary.get("baseline_buy_hold", {})
        all_cand = summary.get("baseline_all_candidates", {})

        if formatted:
            # Format numbers for readability
            row = {
                "symbol": symbol,
                # Strategy metrics (percentages with 2 decimals)
                "total_return_%": round(strategy.get("total_return", 0) * 100, 2),
                "sharpe_ratio": round(strategy.get("sharpe_ratio", 0), 2),
                "max_drawdown_%": round(strategy.get("max_drawdown", 0) * 100, 2),
                "win_rate_%": round(strategy.get("win_rate", 0) * 100, 1),
                "total_trades": int(strategy.get("total_trades", 0)),
                # Buy-and-hold
                "bh_return_%": round(bh.get("bh_total_return", 0) * 100, 2),
                "bh_sharpe": round(bh.get("bh_sharpe_ratio", 0), 2),
                # All candidates (if available)
                "all_cand_return_%": round(all_cand.get("total_return", 0) * 100, 2) if all_cand else None,
                # Comparisons
                "vs_bh_%": round(summary.get("return_vs_bh", 0) * 100, 2),
            }
        else:
            # Raw values (original format)
            row = {
                "symbol": symbol,
                "total_return": strategy.get("total_return", 0),
                "sharpe_ratio": strategy.get("sharpe_ratio", 0),
                "max_drawdown": strategy.get("max_drawdown", 0),
                "win_rate": strategy.get("win_rate", 0),
                "total_trades": strategy.get("total_trades", 0),
                "bh_return": bh.get("bh_total_return", 0),
                "bh_sharpe": bh.get("bh_sharpe_ratio", 0),
                "all_cand_return": all_cand.get("total_return", 0) if all_cand else None,
                "return_vs_bh": summary.get("return_vs_bh", 0),
            }
        rows.append(row)

    return pd.DataFrame(rows)


def save_summary(summary: Dict, path) -> None:
    """Save summary to JSON file."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert any numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj

    summary_converted = json.loads(
        json.dumps(summary, default=convert)
    )

    with open(path, "w") as f:
        json.dump(summary_converted, f, indent=2)

    logger.info(f"Saved summary to {path}")
