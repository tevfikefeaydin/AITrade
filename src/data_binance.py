"""
Binance Data Downloader for ML-Assisted Crypto Trading Research Pipeline.

Downloads historical klines (candlestick data) from Binance Spot public REST API.
No API key required for historical data.

Features:
- Pagination (max 1000 klines per request)
- Retry with exponential backoff
- Polite rate limiting
- Duplicate handling
- Gap detection
"""

import time
import logging
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

from . import config
from .utils import date_to_ms, ms_to_datetime, validate_symbol, ensure_directories

logger = logging.getLogger(__name__)


# Binance kline columns
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def fetch_klines_batch(
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int,
    limit: int = 1000,
) -> List[List]:
    """
    Fetch a single batch of klines from Binance.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Kline interval (e.g., "1m", "1h")
        start_time: Start time in milliseconds
        end_time: End time in milliseconds
        limit: Maximum number of klines (max 1000)

    Returns:
        List of kline data
    """
    url = f"{config.BINANCE_BASE_URL}{config.BINANCE_KLINES_ENDPOINT}"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": min(limit, config.BINANCE_MAX_KLINES_PER_REQUEST),
    }

    for attempt in range(config.BINANCE_MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            wait_time = config.BINANCE_RETRY_BACKOFF_FACTOR ** attempt
            logger.warning(
                f"Request failed (attempt {attempt + 1}/{config.BINANCE_MAX_RETRIES}): {e}. "
                f"Retrying in {wait_time:.1f}s..."
            )
            time.sleep(wait_time)

    raise RuntimeError(f"Failed to fetch klines after {config.BINANCE_MAX_RETRIES} attempts")


def download_klines(
    symbol: str,
    interval: str = "1m",
    start: str = None,
    end: str = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Download historical klines from Binance with pagination.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Kline interval (default "1m")
        start: Start date string (YYYY-MM-DD), defaults to config.DEFAULT_START
        end: End date string (YYYY-MM-DD), defaults to config.DEFAULT_END
        save: Whether to save to parquet file

    Returns:
        DataFrame with OHLCV data
    """
    validate_symbol(symbol)
    ensure_directories()

    start = start or config.DEFAULT_START
    end = end or config.DEFAULT_END

    start_ms = date_to_ms(start)
    end_ms = date_to_ms(end)

    # Calculate expected number of requests for progress bar
    interval_ms = _interval_to_ms(interval)
    total_ms = end_ms - start_ms
    expected_klines = total_ms // interval_ms
    expected_requests = (expected_klines // config.BINANCE_MAX_KLINES_PER_REQUEST) + 1

    logger.info(f"Downloading {symbol} {interval} klines from {start} to {end}")
    logger.info(f"Expected ~{expected_klines:,} klines in ~{expected_requests} requests")

    all_klines = []
    current_start = start_ms

    with tqdm(total=expected_requests, desc=f"Downloading {symbol}", unit="batch") as pbar:
        while current_start < end_ms:
            batch = fetch_klines_batch(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_ms,
            )

            if not batch:
                break

            all_klines.extend(batch)

            # Move start time to after the last kline
            last_open_time = batch[-1][0]
            current_start = last_open_time + interval_ms

            pbar.update(1)

            # Polite sleep to respect rate limits
            time.sleep(config.BINANCE_RATE_LIMIT_SLEEP_MS / 1000)

    if not all_klines:
        raise ValueError(f"No data downloaded for {symbol}")

    # Create DataFrame
    df = pd.DataFrame(all_klines, columns=KLINE_COLUMNS)

    # Convert types
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)

    # Sort and remove duplicates
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"])
    df = df.reset_index(drop=True)

    # Keep only essential columns
    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]

    # Check for gaps
    gaps = check_gaps(df, interval)
    if gaps:
        logger.warning(f"Found {len(gaps)} gaps in data. First gap: {gaps[0]}")

    logger.info(f"Downloaded {len(df):,} klines for {symbol}")

    if save:
        output_path = config.get_symbol_data_path(symbol, interval)
        df.to_parquet(output_path, compression="snappy")
        logger.info(f"Saved to {output_path}")

    return df


def _interval_to_ms(interval: str) -> int:
    """Convert interval string to milliseconds."""
    unit = interval[-1]
    value = int(interval[:-1])

    multipliers = {
        "m": 60 * 1000,
        "h": 60 * 60 * 1000,
        "d": 24 * 60 * 60 * 1000,
    }

    if unit not in multipliers:
        raise ValueError(f"Unknown interval unit: {unit}")

    return value * multipliers[unit]


def check_gaps(df: pd.DataFrame, interval: str) -> List[Tuple[datetime, datetime]]:
    """
    Check for gaps in the data.

    Args:
        df: DataFrame with open_time column
        interval: Expected interval between rows

    Returns:
        List of (start, end) tuples for each gap
    """
    if len(df) < 2:
        return []

    expected_delta = pd.Timedelta(_interval_to_ms(interval), unit="ms")
    time_diffs = df["open_time"].diff()

    # Find gaps (where diff is greater than expected)
    gap_mask = time_diffs > expected_delta * 1.5  # Allow 50% tolerance
    gap_indices = df.index[gap_mask].tolist()

    gaps = []
    for idx in gap_indices:
        if idx > 0:
            gap_start = df.loc[idx - 1, "open_time"]
            gap_end = df.loc[idx, "open_time"]
            gaps.append((gap_start, gap_end))

    return gaps


def load_klines(symbol: str, interval: str = "1m") -> Optional[pd.DataFrame]:
    """
    Load previously downloaded klines from parquet.

    Args:
        symbol: Trading pair
        interval: Kline interval

    Returns:
        DataFrame or None if file doesn't exist
    """
    validate_symbol(symbol)
    path = config.get_symbol_data_path(symbol, interval)

    if not path.exists():
        logger.warning(f"Data file not found: {path}")
        return None

    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} klines from {path}")
    return df


def download_klines_incremental(
    symbol: str,
    interval: str = "1m",
    start: str = None,
    end: str = None,
    force: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """
    Smart incremental download - only downloads missing data.

    If data file exists, only downloads gaps (before start or after end).
    If data file doesn't exist or force=True, downloads entire range.

    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        interval: Kline interval (default "1m")
        start: Start date string (YYYY-MM-DD)
        end: End date string (YYYY-MM-DD)
        force: If True, ignore existing data and re-download

    Returns:
        Tuple of (DataFrame, stats_dict) where stats_dict contains:
        - existing_rows: Number of rows before update
        - new_rows: Number of new rows downloaded
        - total_rows: Final total rows
        - gaps_filled: List of gap descriptions
    """
    validate_symbol(symbol)
    ensure_directories()

    start = start or config.DEFAULT_START
    end = end or config.DEFAULT_END

    request_start = pd.Timestamp(start, tz="UTC")
    request_end = pd.Timestamp(end, tz="UTC")

    path = config.get_symbol_data_path(symbol, interval)
    stats = {
        "existing_rows": 0,
        "new_rows": 0,
        "total_rows": 0,
        "gaps_filled": [],
    }

    # Check if file exists and we're not forcing re-download
    if path.exists() and not force:
        existing_df = pd.read_parquet(path)
        stats["existing_rows"] = len(existing_df)

        existing_start = existing_df["open_time"].min()
        existing_end = existing_df["open_time"].max()

        logger.info(f"Found existing data for {symbol}:")
        logger.info(f"  Existing range: {existing_start} to {existing_end} ({len(existing_df):,} rows)")
        logger.info(f"  Requested range: {request_start} to {request_end}")

        # Determine gaps to fill
        gaps_to_download = []

        # Gap before existing data
        if request_start < existing_start:
            gap_end_ms = int(existing_start.timestamp() * 1000) - _interval_to_ms(interval)
            gaps_to_download.append({
                "type": "before",
                "start": start,
                "end_ms": gap_end_ms,
                "description": f"Before: {request_start.date()} to {existing_start.date()}",
            })
            stats["gaps_filled"].append(f"Before: {request_start.date()} to {existing_start.date()}")

        # Gap after existing data
        if request_end > existing_end:
            gap_start_ms = int(existing_end.timestamp() * 1000) + _interval_to_ms(interval)
            gaps_to_download.append({
                "type": "after",
                "start_ms": gap_start_ms,
                "end": end,
                "description": f"After: {existing_end.date()} to {request_end.date()}",
            })
            stats["gaps_filled"].append(f"After: {existing_end.date()} to {request_end.date()}")

        # If no gaps, data is up to date
        if not gaps_to_download:
            logger.info(f"[OK] Data is up to date for {symbol}. No download needed.")
            stats["total_rows"] = len(existing_df)
            return existing_df, stats

        # Download gaps
        new_dfs = [existing_df]

        for gap in gaps_to_download:
            logger.info(f"Downloading gap: {gap['description']}")

            if gap["type"] == "before":
                gap_df = _download_range(
                    symbol=symbol,
                    interval=interval,
                    start_ms=date_to_ms(gap["start"]),
                    end_ms=gap["end_ms"],
                )
            else:  # after
                gap_df = _download_range(
                    symbol=symbol,
                    interval=interval,
                    start_ms=gap["start_ms"],
                    end_ms=date_to_ms(gap["end"]),
                )

            if gap_df is not None and len(gap_df) > 0:
                new_dfs.append(gap_df)
                stats["new_rows"] += len(gap_df)

        # Merge all data
        merged_df = pd.concat(new_dfs, ignore_index=True)
        merged_df = merged_df.sort_values("open_time").drop_duplicates(subset=["open_time"])
        merged_df = merged_df.reset_index(drop=True)

        # Save merged data
        merged_df.to_parquet(path, compression="snappy")
        stats["total_rows"] = len(merged_df)

        logger.info(f"[OK] Updated {symbol}: {stats['existing_rows']:,} to {stats['total_rows']:,} rows (+{stats['new_rows']:,} new)")

        return merged_df, stats

    else:
        # No existing file or force=True - download entire range
        if force and path.exists():
            logger.info(f"Force flag set - re-downloading entire range for {symbol}")
        else:
            logger.info(f"No existing data found for {symbol} - downloading full range")

        df = download_klines(symbol, interval, start, end, save=True)
        stats["new_rows"] = len(df)
        stats["total_rows"] = len(df)
        stats["gaps_filled"].append(f"Full: {start} to {end}")

        return df, stats


def _download_range(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> Optional[pd.DataFrame]:
    """
    Download a specific range of klines (internal helper).

    Args:
        symbol: Trading pair
        interval: Kline interval
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds

    Returns:
        DataFrame with OHLCV data or None if no data
    """
    interval_ms = _interval_to_ms(interval)
    total_ms = end_ms - start_ms
    expected_klines = total_ms // interval_ms
    expected_requests = (expected_klines // config.BINANCE_MAX_KLINES_PER_REQUEST) + 1

    all_klines = []
    current_start = start_ms

    with tqdm(total=expected_requests, desc=f"Downloading {symbol}", unit="batch") as pbar:
        while current_start < end_ms:
            batch = fetch_klines_batch(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_ms,
            )

            if not batch:
                break

            all_klines.extend(batch)

            last_open_time = batch[-1][0]
            current_start = last_open_time + interval_ms

            pbar.update(1)
            time.sleep(config.BINANCE_RATE_LIMIT_SLEEP_MS / 1000)

    if not all_klines:
        return None

    df = pd.DataFrame(all_klines, columns=KLINE_COLUMNS)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)

    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"])
    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]

    return df


def download_all_symbols(
    interval: str = "1m",
    start: str = None,
    end: str = None,
    force: bool = False,
) -> dict:
    """
    Download klines for all configured symbols (with incremental support).

    Args:
        interval: Kline interval
        start: Start date string
        end: End date string
        force: If True, ignore existing data and re-download

    Returns:
        Dict mapping symbol to (DataFrame, stats_dict)
    """
    results = {}
    for symbol in config.SYMBOLS:
        try:
            df, stats = download_klines_incremental(symbol, interval, start, end, force)
            results[symbol] = {"df": df, "stats": stats}
        except Exception as e:
            logger.error(f"Failed to download {symbol}: {e}")
            raise

    return results
