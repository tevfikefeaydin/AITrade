"""
Binance WebSocket Client for Real-Time Data.

Connects to Binance WebSocket streams to receive:
- 1-minute kline data in real-time
- Aggregates to 1-hour bars for signal generation
"""

import asyncio
import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Optional, List, Dict

import pandas as pd
import websockets

from .. import config
from ..data_binance import load_klines, fetch_klines_batch, KLINE_COLUMNS
from ..resample import resample_1h, resample_4h

logger = logging.getLogger(__name__)


def _utc_now() -> pd.Timestamp:
    """Return current UTC time. Separated for testability."""
    return pd.Timestamp.now(tz="UTC")


def _fetch_recent_rest(symbol: str) -> Optional[pd.DataFrame]:
    """
    Fetch the most recent ~25 hours of 1m bars via REST API.

    Paginates in chunks of 1000 (Binance limit) to cover the full
    1500-minute window so the 1440-bar buffer can be completely filled.

    Returns a DataFrame with KLINE_COLUMNS or None on failure.
    """
    try:
        now_ms = int(_utc_now().timestamp() * 1000)
        start_ms = now_ms - 1500 * 60 * 1000  # 1500 minutes ago

        all_raw: list = []
        cursor = start_ms
        while cursor < now_ms:
            raw = fetch_klines_batch(symbol, "1m", cursor, now_ms, limit=1000)
            if not raw:
                break
            all_raw.extend(raw)
            # Advance cursor past last returned bar's open_time + 1ms
            last_open_ms = raw[-1][0]
            cursor = last_open_ms + 60_000  # next minute
            if len(raw) < 1000:
                break  # no more data to fetch

        if not all_raw:
            return None
        df = pd.DataFrame(all_raw, columns=KLINE_COLUMNS)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.drop_duplicates("open_time")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        logger.info(f"REST fetched {len(df)} 1m bars for {symbol}")
        return df
    except Exception as e:
        logger.warning(f"REST backfill failed for {symbol}: {e}")
        return None


class BinanceWebSocket:
    """
    Binance WebSocket client for real-time kline data.

    Connects to 1-minute kline stream and:
    - Buffers 1m bars (24 hours)
    - Aggregates to 1h bars
    - Triggers callback on 1h bar close
    """

    BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"

    def __init__(self, symbol: str = "btcusdt"):
        """
        Initialize WebSocket client.

        Args:
            symbol: Trading pair in lowercase (e.g., "btcusdt")
        """
        self.symbol = symbol.lower()
        self.stream_url = f"{self.BINANCE_WS_URL}/{self.symbol}@kline_1m"

        # Data buffers
        self.buffer_1m: deque = deque(maxlen=1440)  # 24 hours of 1m bars
        self.buffer_1h: deque = deque(maxlen=720)   # 30 days of 1h bars
        self.buffer_4h: deque = deque(maxlen=180)   # 30 days of 4h bars

        # Current incomplete bars
        self._current_1h_bars: List[Dict] = []
        self._current_4h_bars: List[Dict] = []

        # Callbacks
        self.on_1m_bar: Optional[Callable] = None
        self.on_1h_bar: Optional[Callable] = None

        # Connection state
        self._running = False
        self._reconnect_delay = 1
        self._max_reconnect_delay = 60

    def backfill(self, symbol: str) -> None:
        """
        Fill buffers from existing parquet files so the system is
        immediately ready for signal generation after startup/restart.

        Merges parquet data with recent REST API data to cover the gap
        between the last parquet update and now (parquet is refreshed
        daily at 00:05 UTC, so intra-day restarts would otherwise miss
        hours of data).

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
        """
        symbol = symbol.upper()
        now = _utc_now()

        # --- 1m data ---
        df_1m = load_klines(symbol, "1m")
        has_parquet = df_1m is not None and not df_1m.empty

        # --- REST gap fill: fetch recent 1m bars from API ---
        df_rest = _fetch_recent_rest(symbol)

        if not has_parquet and (df_rest is None or df_rest.empty):
            logger.warning(
                f"Backfill: no parquet and no REST data for {symbol}"
            )
            return

        if has_parquet:
            df_1m = df_1m.sort_values("open_time").drop_duplicates("open_time")
        else:
            logger.warning(
                f"Backfill: no parquet for {symbol}, using REST-only warm start"
            )
            df_1m = pd.DataFrame(columns=df_rest.columns).astype(df_rest.dtypes)

        if df_rest is not None and not df_rest.empty:
            # Align datetime resolution (ms vs us) before concat
            if not df_1m.empty:
                df_rest["open_time"] = df_rest["open_time"].astype(
                    df_1m["open_time"].dtype
                )
            df_1m = (
                pd.concat([df_1m, df_rest], ignore_index=True)
                .sort_values("open_time")
                .drop_duplicates("open_time", keep="last")
            )
            logger.info(
                f"Backfill: merged {len(df_rest)} REST bars for {symbol}"
            )

        # Keep only closed bars (open_time before the current minute)
        current_minute_start = now.floor("min")
        df_1m = df_1m[df_1m["open_time"] < current_minute_start]

        # --- 1h data ---
        path_1h = config.get_symbol_data_path(symbol, "1h")
        if path_1h.exists():
            df_1h = pd.read_parquet(path_1h)
            logger.info(f"Backfill: loaded {len(df_1h)} pre-built 1h bars")
        else:
            df_1h = resample_1h(df_1m)

        # If REST data was available, resample only the REST portion
        # and merge into 1h to cover the gap without re-resampling everything
        if df_rest is not None and not df_rest.empty:
            df_rest_closed = df_rest[
                df_rest["open_time"] < current_minute_start
            ]
            if not df_rest_closed.empty:
                df_rest_1h = resample_1h(df_rest_closed)
                if not df_rest_1h.empty:
                    df_rest_1h["open_time"] = df_rest_1h["open_time"].astype(
                        df_1h["open_time"].dtype
                    )
                    df_1h = (
                        pd.concat([df_1h, df_rest_1h], ignore_index=True)
                        .sort_values("open_time")
                        .drop_duplicates("open_time", keep="last")
                    )

        df_1h = df_1h.sort_values("open_time").drop_duplicates("open_time")
        current_hour_start = now.floor("h")
        df_1h = df_1h[df_1h["open_time"] < current_hour_start]

        # --- 4h data ---
        df_4h = resample_4h(df_1h)
        df_4h = df_4h.sort_values("open_time").drop_duplicates("open_time")
        current_4h_start = pd.Timestamp(
            year=now.year, month=now.month, day=now.day,
            hour=(now.hour // 4) * 4, tz="UTC"
        )
        df_4h = df_4h[df_4h["open_time"] < current_4h_start]

        # --- Fill buffers (take tail matching maxlen) ---
        cols = ["open_time", "open", "high", "low", "close", "volume"]

        for _, row in df_1m[cols].tail(self.buffer_1m.maxlen).iterrows():
            self.buffer_1m.append(row.to_dict())

        for _, row in df_1h[cols].tail(self.buffer_1h.maxlen).iterrows():
            self.buffer_1h.append(row.to_dict())

        for _, row in df_4h[cols].tail(self.buffer_4h.maxlen).iterrows():
            self.buffer_4h.append(row.to_dict())

        # Pre-populate incomplete-bar accumulators with data from
        # the current (not-yet-closed) hour/4h period so the first
        # live bar isn't built from partial minutes only.
        self._current_1h_bars = [
            row.to_dict()
            for _, row in df_1m[cols][
                df_1m["open_time"] >= current_hour_start
            ].iterrows()
        ]
        self._current_4h_bars = [
            row.to_dict()
            for _, row in df_1h[cols][
                df_1h["open_time"] >= current_4h_start
            ].iterrows()
        ]

        logger.info(
            f"Backfilled {symbol}: 1m={len(self.buffer_1m)}, "
            f"1h={len(self.buffer_1h)}, 4h={len(self.buffer_4h)} bars, "
            f"partial 1h={len(self._current_1h_bars)}m, "
            f"partial 4h={len(self._current_4h_bars)}h"
        )

    def _rest_gap_fill(self):
        """Fetch missing 1m bars via REST after a WS disconnect/reconnect.

        Compares the last buffered 1m bar time with the current time and
        fills any gap using the REST API. Also rebuilds 1h/4h bars from the
        gap data so higher-timeframe buffers stay consistent.
        """
        if not self.buffer_1m:
            logger.info("Gap-fill: buffer empty, skipping REST fill")
            return

        last_bar_time = pd.Timestamp(self.buffer_1m[-1]["open_time"])
        if last_bar_time.tzinfo is None:
            last_bar_time = last_bar_time.tz_localize("UTC")

        now = _utc_now()
        gap_minutes = (now - last_bar_time).total_seconds() / 60

        if gap_minutes < 2:
            return  # No meaningful gap

        logger.info("Gap-fill: %.0f minutes gap detected, fetching REST data", gap_minutes)

        try:
            symbol_upper = self.symbol.upper()
            start_ms = int(last_bar_time.timestamp() * 1000) + 60_000  # bar after last
            now_ms = int(now.timestamp() * 1000)

            all_raw: list = []
            cursor = start_ms
            while cursor < now_ms:
                raw = fetch_klines_batch(symbol_upper, "1m", cursor, now_ms, limit=1000)
                if not raw:
                    break
                all_raw.extend(raw)
                last_open_ms = raw[-1][0]
                cursor = last_open_ms + 60_000
                if len(raw) < 1000:
                    break

            if not all_raw:
                logger.warning("Gap-fill: REST returned no data")
                return

            df = pd.DataFrame(all_raw, columns=KLINE_COLUMNS)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df = df.drop_duplicates("open_time")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            # Only closed bars (before current minute)
            current_minute_start = now.floor("min")
            df = df[df["open_time"] < current_minute_start]

            if df.empty:
                return

            # Append to 1m buffer
            cols = ["open_time", "open", "high", "low", "close", "volume"]
            added_1m = 0
            for _, row in df[cols].iterrows():
                bar = row.to_dict()
                self.buffer_1m.append(bar)
                added_1m += 1

                # Accumulate for 1h aggregation
                self._current_1h_bars.append(bar)
                bar_time = pd.Timestamp(bar["open_time"])
                if bar_time.minute == 59:
                    # Build 1h bar synchronously (no callback during gap-fill)
                    if self._current_1h_bars:
                        bars = self._current_1h_bars
                        bar_1h = {
                            "open_time": bars[0]["open_time"].replace(minute=0, second=0)
                            if isinstance(bars[0]["open_time"], datetime)
                            else pd.Timestamp(bars[0]["open_time"]).replace(minute=0, second=0),
                            "open": bars[0]["open"],
                            "high": max(b["high"] for b in bars),
                            "low": min(b["low"] for b in bars),
                            "close": bars[-1]["close"],
                            "volume": sum(b["volume"] for b in bars),
                        }
                        self.buffer_1h.append(bar_1h)
                        self._current_4h_bars.append(bar_1h)

                        ot = bar_1h["open_time"]
                        hour = ot.hour if isinstance(ot, datetime) else pd.Timestamp(ot).hour
                        if hour % 4 == 3:
                            if self._current_4h_bars:
                                bars_4h = self._current_4h_bars
                                first_ot = bars_4h[0]["open_time"]
                                if isinstance(first_ot, datetime):
                                    ot_4h = first_ot.replace(
                                        hour=(first_ot.hour // 4) * 4, minute=0, second=0
                                    )
                                else:
                                    ts = pd.Timestamp(first_ot)
                                    ot_4h = ts.replace(
                                        hour=(ts.hour // 4) * 4, minute=0, second=0
                                    )
                                bar_4h = {
                                    "open_time": ot_4h,
                                    "open": bars_4h[0]["open"],
                                    "high": max(b["high"] for b in bars_4h),
                                    "low": min(b["low"] for b in bars_4h),
                                    "close": bars_4h[-1]["close"],
                                    "volume": sum(b["volume"] for b in bars_4h),
                                }
                                self.buffer_4h.append(bar_4h)
                                self._current_4h_bars = []
                        self._current_1h_bars = []

            logger.info(
                "Gap-fill: added %d 1m bars, buffer sizes: 1m=%d 1h=%d 4h=%d",
                added_1m, len(self.buffer_1m), len(self.buffer_1h), len(self.buffer_4h),
            )
        except Exception as e:
            logger.error("Gap-fill REST failed: %s", e, exc_info=True)

    async def connect(self):
        """
        Connect to WebSocket and start receiving data.

        Handles reconnection with exponential backoff.
        """
        self._running = True
        is_reconnect = False

        while self._running:
            try:
                logger.info(f"Connecting to Binance WebSocket: {self.symbol}")

                # Fill any gap from previous disconnect
                if is_reconnect:
                    self._rest_gap_fill()

                async with websockets.connect(
                    self.stream_url,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    logger.info(f"Connected to {self.symbol}@kline_1m stream")
                    self._reconnect_delay = 1  # Reset on successful connection

                    async for message in ws:
                        if not self._running:
                            break
                        await self._handle_message(message)

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

            is_reconnect = True
            if self._running:
                logger.info(f"Reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self._max_reconnect_delay
                )

    async def _handle_message(self, message: str):
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
            kline = data.get("k", {})

            # Only process closed bars
            if not kline.get("x", False):
                return

            # Parse bar data
            bar = {
                "open_time": datetime.fromtimestamp(
                    kline["t"] / 1000, tz=timezone.utc
                ),
                "open": float(kline["o"]),
                "high": float(kline["h"]),
                "low": float(kline["l"]),
                "close": float(kline["c"]),
                "volume": float(kline["v"]),
            }

            # Add to 1m buffer
            self.buffer_1m.append(bar)

            # Trigger 1m callback
            if self.on_1m_bar:
                await self._safe_callback(self.on_1m_bar, bar)

            # Aggregate to 1h bar
            self._current_1h_bars.append(bar)

            # Check if 1h bar is complete (minute == 59)
            if bar["open_time"].minute == 59:
                await self._build_1h_bar()

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def _build_1h_bar(self):
        """Aggregate 1m bars into a 1h bar."""
        if not self._current_1h_bars:
            return

        bars = self._current_1h_bars

        if len(bars) < 60:
            logger.warning("Incomplete 1h bar: %d/60 1m bars", len(bars))

        # OHLC aggregation
        bar_1h = {
            "open_time": bars[0]["open_time"].replace(minute=0, second=0),
            "open": bars[0]["open"],
            "high": max(b["high"] for b in bars),
            "low": min(b["low"] for b in bars),
            "close": bars[-1]["close"],
            "volume": sum(b["volume"] for b in bars),
        }

        # Add to 1h buffer
        self.buffer_1h.append(bar_1h)

        # Add to 4h aggregation
        self._current_4h_bars.append(bar_1h)

        # Check if 4h bar is complete (hour % 4 == 3 means end of 4h period)
        if bar_1h["open_time"].hour % 4 == 3:
            await self._build_4h_bar()

        # Clear 1h buffer
        self._current_1h_bars = []

        # Trigger 1h callback
        if self.on_1h_bar:
            await self._safe_callback(self.on_1h_bar, bar_1h)

        logger.debug(f"1h bar: {bar_1h['open_time']} close={bar_1h['close']}")

    async def _build_4h_bar(self):
        """Aggregate 1h bars into a 4h bar."""
        if not self._current_4h_bars:
            return

        bars = self._current_4h_bars

        # OHLC aggregation
        bar_4h = {
            "open_time": bars[0]["open_time"].replace(
                hour=(bars[0]["open_time"].hour // 4) * 4,
                minute=0, second=0
            ),
            "open": bars[0]["open"],
            "high": max(b["high"] for b in bars),
            "low": min(b["low"] for b in bars),
            "close": bars[-1]["close"],
            "volume": sum(b["volume"] for b in bars),
        }

        # Add to 4h buffer
        self.buffer_4h.append(bar_4h)

        # Clear 4h buffer
        self._current_4h_bars = []

        logger.debug(f"4h bar: {bar_4h['open_time']} close={bar_4h['close']}")

    async def _safe_callback(self, callback: Callable, *args):
        """Execute callback with error handling."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    def stop(self):
        """Stop the WebSocket connection."""
        self._running = False
        logger.info("WebSocket client stopping...")

    def get_latest_price(self) -> Optional[float]:
        """Get the latest close price."""
        if self.buffer_1m:
            return self.buffer_1m[-1]["close"]
        return None

    def get_1m_dataframe(self):
        """Get 1m data as DataFrame."""
        import pandas as pd
        if not self.buffer_1m:
            return pd.DataFrame()
        return pd.DataFrame(list(self.buffer_1m))

    def get_1h_dataframe(self):
        """Get 1h data as DataFrame."""
        import pandas as pd
        if not self.buffer_1h:
            return pd.DataFrame()
        return pd.DataFrame(list(self.buffer_1h))

    def get_4h_dataframe(self):
        """Get 4h data as DataFrame."""
        import pandas as pd
        if not self.buffer_4h:
            return pd.DataFrame()
        return pd.DataFrame(list(self.buffer_4h))


async def test_websocket():
    """Test WebSocket connection."""
    ws = BinanceWebSocket("btcusdt")

    def on_1h(bar):
        print(f"1H BAR: {bar['open_time']} O={bar['open']:.2f} H={bar['high']:.2f} L={bar['low']:.2f} C={bar['close']:.2f}")

    ws.on_1h_bar = on_1h

    try:
        await ws.connect()
    except KeyboardInterrupt:
        ws.stop()


if __name__ == "__main__":
    asyncio.run(test_websocket())
