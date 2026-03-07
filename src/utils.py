"""
Utility functions for ML-Assisted Crypto Trading Research Pipeline.

Shared utilities used across multiple modules.
"""

import logging
from datetime import datetime, timezone
from typing import Optional
import pandas as pd
import numpy as np

from . import config


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging for the application.

    Args:
        level: Logging level (default INFO)

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("trading_pipeline")


def parse_date(date_str: str) -> datetime:
    """
    Parse a date string to datetime.

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        datetime object
    """
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def date_to_ms(date_str: str) -> int:
    """
    Convert a date string to milliseconds since epoch (for Binance API).

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        Milliseconds since epoch (UTC)
    """
    dt = parse_date(date_str)
    return int(dt.timestamp() * 1000)


def ms_to_datetime(ms: int) -> datetime:
    """
    Convert milliseconds since epoch to datetime.

    Args:
        ms: Milliseconds since epoch

    Returns:
        datetime object (UTC)
    """
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def ensure_directories() -> None:
    """Create necessary directories if they don't exist."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def validate_symbol(symbol: str) -> bool:
    """
    Validate that a symbol is in the allowed list.

    Args:
        symbol: Symbol to validate

    Returns:
        True if valid, raises ValueError otherwise
    """
    if symbol not in config.SYMBOLS:
        raise ValueError(
            f"Symbol {symbol} not in allowed symbols: {config.SYMBOLS}. "
            "This project is designed for BTCUSDT and ETHUSDT only."
        )
    return True


def safe_divide(numerator: pd.Series, denominator: pd.Series, fill: float = 0.0) -> pd.Series:
    """
    Safely divide two series, handling division by zero.

    Args:
        numerator: Numerator series
        denominator: Denominator series
        fill: Value to use when denominator is zero

    Returns:
        Result series with zeros replaced by fill value
    """
    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], fill)
    result = result.fillna(fill)
    return result


def calculate_costs(price: float, fee_bps: float, slippage_bps: float, is_buy: bool) -> float:
    """
    Calculate the effective price after costs.

    Args:
        price: Base price
        fee_bps: Fee in basis points
        slippage_bps: Slippage in basis points
        is_buy: True for buy (costs increase price), False for sell (costs decrease price)

    Returns:
        Effective price after costs
    """
    total_cost_bps = fee_bps + slippage_bps
    cost_multiplier = total_cost_bps / 10000

    if is_buy:
        # Buying: pay more than mid price
        return price * (1 + cost_multiplier)
    else:
        # Selling: receive less than mid price
        return price * (1 - cost_multiplier)


def compute_barrier_prices(
    entry_price: float,
    pt: float,
    sl: float,
    atr_value: float = None,
    tp_atr_mult: float = None,
    sl_atr_mult: float = None,
    min_barrier_pct: float = None,
    max_barrier_pct: float = None,
) -> tuple:
    """
    Compute TP/SL barrier prices, optionally using ATR-based sizing.

    When atr_value is provided and USE_ATR_BARRIERS is True, barriers are
    scaled to current volatility.  Otherwise falls back to fixed percentages.

    Args:
        entry_price: Raw execution price (barriers anchor on this)
        pt: Fixed take-profit pct (fallback)
        sl: Fixed stop-loss pct (fallback)
        atr_value: Current ATR-14 value (None → use fixed %)
        tp_atr_mult: TP multiplier (default from config)
        sl_atr_mult: SL multiplier (default from config)
        min_barrier_pct: Floor for barrier % (default from config)
        max_barrier_pct: Ceiling for barrier % (default from config)

    Returns:
        (tp_price, sl_price)
    """
    use_atr = config.USE_ATR_BARRIERS and atr_value is not None and atr_value > 0

    if use_atr:
        tp_mult = tp_atr_mult if tp_atr_mult is not None else config.TP_ATR_MULTIPLIER
        sl_mult = sl_atr_mult if sl_atr_mult is not None else config.SL_ATR_MULTIPLIER
        floor = min_barrier_pct if min_barrier_pct is not None else config.MIN_BARRIER_PCT
        ceil = max_barrier_pct if max_barrier_pct is not None else config.MAX_BARRIER_PCT

        tp_pct = (atr_value * tp_mult) / entry_price
        sl_pct = (atr_value * sl_mult) / entry_price

        tp_pct = max(floor, min(ceil, tp_pct))
        sl_pct = max(floor, min(ceil, sl_pct))
    else:
        tp_pct = pt
        sl_pct = sl

    tp_price = entry_price * (1 + tp_pct)
    sl_price = entry_price * (1 - sl_pct)
    return tp_price, sl_price


def format_pct(value: float, decimals: int = 2) -> str:
    """Format a decimal as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with specified decimal places."""
    return f"{value:,.{decimals}f}"


def load_parquet_safe(path) -> Optional[pd.DataFrame]:
    """
    Safely load a parquet file, returning None if not found.

    Args:
        path: Path to parquet file

    Returns:
        DataFrame or None if file doesn't exist
    """
    from pathlib import Path
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    return None


def save_parquet(df: pd.DataFrame, path, compression: str = "snappy") -> None:
    """
    Save DataFrame to parquet with compression.

    Args:
        df: DataFrame to save
        path: Output path
        compression: Compression algorithm
    """
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression=compression)


class ProgressTracker:
    """Simple progress tracker for long-running operations."""

    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()

    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        self.current += n

    def get_progress(self) -> float:
        """Get progress as a fraction (0-1)."""
        if self.total == 0:
            return 1.0
        return self.current / self.total

    def get_eta_seconds(self) -> Optional[float]:
        """Estimate remaining time in seconds."""
        if self.current == 0:
            return None
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current / elapsed
        remaining = self.total - self.current
        return remaining / rate if rate > 0 else None
