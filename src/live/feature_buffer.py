"""
Live Feature Buffer for Real-Time Feature Computation.

Computes the same 22 features as the backtest system,
but incrementally from streaming data.
"""

import logging
from collections import deque
from typing import Optional, Dict, List

import pandas as pd
import numpy as np

from .. import config

logger = logging.getLogger(__name__)


class FeatureBuffer:
    """
    Real-time feature computation buffer.

    Maintains rolling windows of data and computes features
    matching the backtest feature engineering.
    """

    def __init__(self):
        """Initialize feature buffer with rolling windows."""
        # Data buffers (references to WebSocket buffers)
        self.bars_1m: Optional[deque] = None
        self.bars_1h: Optional[deque] = None
        self.bars_4h: Optional[deque] = None

        # Feature computation parameters (from config)
        self.rolling_window = config.FEATURE_ROLLING_WINDOW  # 20
        self.rsi_period = config.RSI_PERIOD  # 14
        self.ma_length = config.MA_LENGTH  # 50
        self.adx_period = config.ADX_PERIOD  # 14

        # Cache for computed features
        self._feature_cache: Dict = {}

    def link_websocket(self, ws):
        """Link to WebSocket client buffers."""
        self.bars_1m = ws.buffer_1m
        self.bars_1h = ws.buffer_1h
        self.bars_4h = ws.buffer_4h

    def compute_features(self) -> Optional[pd.DataFrame]:
        """
        Compute all 22 features for the current bar.

        Returns:
            DataFrame with single row of features, or None if insufficient data
        """
        if not self._has_sufficient_data():
            logger.warning("Insufficient data for feature computation")
            return None

        # Convert buffers to DataFrames
        df_1h = pd.DataFrame(list(self.bars_1h))
        df_4h = pd.DataFrame(list(self.bars_4h))
        df_1m = pd.DataFrame(list(self.bars_1m))

        # Get latest bar
        latest = df_1h.iloc[-1]

        features = {}

        # 1. Wick Features (7 features)
        features.update(self._compute_wick_features(latest))

        # 2. Return Features (4 features)
        features.update(self._compute_return_features(df_1h))

        # 3. Volatility (1 feature)
        features.update(self._compute_volatility(df_1h))

        # 4. RSI (1 feature)
        features.update(self._compute_rsi(df_1h))

        # 5. Moving Average Features (2 features)
        features.update(self._compute_ma_features(df_1h))

        # 6. Volume Z-Score + Volume Ratio (2 features)
        features.update(self._compute_volume_features(df_1h))

        # 7. Intrabar Features (5 features)
        features.update(self._compute_intrabar_features(df_1m, latest))

        # 8. 4H Context Features (aligned to training/backtest availability timing)
        features.update(self._compute_4h_features(df_4h, latest["open_time"]))

        # 9. ATR Ratio
        features.update(self._compute_atr_ratio(df_1h))

        # 10. ret_24 (24-hour return)
        features.update(self._compute_ret_24(df_1h))

        # 11. Hour-of-day cyclical encoding
        features.update(self._compute_hour_features(latest["open_time"]))

        # 12. Day-of-week cyclical encoding
        features.update(self._compute_dow_features(latest["open_time"]))

        # 13. RSI slope
        features.update(self._compute_rsi_slope(df_1h))

        return pd.DataFrame([features])

    # Maximum age (in hours) for the most recent 1h bar to be considered fresh.
    MAX_DATA_AGE_HOURS: int = 3
    # Maximum age (in hours) for the most recent 4h bar to be considered fresh.
    MAX_DATA_AGE_HOURS_4H: int = 8

    @staticmethod
    def _empty_intrabar_features() -> Dict:
        """Return neutral intrabar features when the exact 1h window is unavailable."""
        return {
            "max_runup": 0.0,
            "max_drawdown": 0.0,
            "intrabar_vol": 0.0,
            "intrabar_skew": 0.0,
            "up_down_ratio": 0.5,
        }

    def _has_sufficient_data(self) -> bool:
        """Check if we have enough *and fresh enough* data for feature computation."""
        if self.bars_1h is None or len(self.bars_1h) < self.ma_length:
            logger.debug(
                "Insufficient 1h data: %d/%d",
                0 if self.bars_1h is None else len(self.bars_1h),
                self.ma_length,
            )
            return False
        if self.bars_4h is None or len(self.bars_4h) < self.ma_length:
            logger.debug(
                "Insufficient 4h data: %d/%d",
                0 if self.bars_4h is None else len(self.bars_4h),
                self.ma_length,
            )
            return False
        if self.bars_1m is None or len(self.bars_1m) < 60:
            logger.debug(
                "Insufficient 1m data: %d/60",
                0 if self.bars_1m is None else len(self.bars_1m),
            )
            return False

        # Freshness check: most recent 1h bar should not be too old
        latest_1h = self.bars_1h[-1]
        latest_time = pd.Timestamp(latest_1h["open_time"])
        now = pd.Timestamp.now(tz="UTC")
        if latest_time.tzinfo is None:
            latest_time = latest_time.tz_localize("UTC")
        age_hours = (now - latest_time).total_seconds() / 3600
        if age_hours > self.MAX_DATA_AGE_HOURS:
            logger.warning(
                "Data too stale: latest 1h bar is %.1fh old (max %dh)",
                age_hours,
                self.MAX_DATA_AGE_HOURS,
            )
            return False

        # Freshness check: most recent 4h bar should not be too old
        latest_4h = self.bars_4h[-1]
        latest_4h_time = pd.Timestamp(latest_4h["open_time"])
        if latest_4h_time.tzinfo is None:
            latest_4h_time = latest_4h_time.tz_localize("UTC")
        age_4h = (now - latest_4h_time).total_seconds() / 3600
        if age_4h > self.MAX_DATA_AGE_HOURS_4H:
            logger.warning(
                "4h data too stale: latest 4h bar is %.1fh old (max %dh)",
                age_4h, self.MAX_DATA_AGE_HOURS_4H,
            )
            return False

        return True

    def _compute_wick_features(self, row: pd.Series) -> Dict:
        """Compute candle body and wick features."""
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]

        body = abs(c - o)
        range_val = h - l

        if range_val > 0:
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l

            upper_wick_ratio = upper_wick / range_val
            lower_wick_ratio = lower_wick / range_val
            body_ratio = body / range_val
        else:
            upper_wick = 0.0
            lower_wick = 0.0
            upper_wick_ratio = 0.0
            lower_wick_ratio = 0.0
            body_ratio = 0.0

        return {
            "body": body,
            "range": range_val,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "upper_wick_ratio": upper_wick_ratio,
            "lower_wick_ratio": lower_wick_ratio,
            "body_ratio": body_ratio,
        }

    def _compute_return_features(self, df: pd.DataFrame) -> Dict:
        """Compute log return features at different lags."""
        close = df["close"].values

        def safe_log_return(current, past):
            if past > 0 and current > 0:
                return np.log(current / past)
            return 0.0

        n = len(close)
        current = close[-1]

        return {
            "ret_1": safe_log_return(current, close[-2]) if n >= 2 else 0.0,
            "ret_3": safe_log_return(current, close[-4]) if n >= 4 else 0.0,
            "ret_6": safe_log_return(current, close[-7]) if n >= 7 else 0.0,
            "ret_12": safe_log_return(current, close[-13]) if n >= 13 else 0.0,
        }

    def _compute_volatility(self, df: pd.DataFrame) -> Dict:
        """Compute rolling volatility."""
        if len(df) < self.rolling_window + 1:
            return {"vol_20": 0.0}

        returns = np.log(df["close"] / df["close"].shift(1)).dropna()
        vol = returns.iloc[-self.rolling_window:].std()

        return {"vol_20": float(vol) if pd.notna(vol) else 0.0}

    def _compute_rsi(self, df: pd.DataFrame) -> Dict:
        """Compute RSI indicator."""
        if len(df) < self.rsi_period + 1:
            return {"rsi": 50.0}

        delta = df["close"].diff()
        gains = delta.clip(lower=0)
        losses = (-delta).clip(lower=0)
        avg_gain = gains.ewm(span=self.rsi_period, min_periods=self.rsi_period, adjust=False).mean().iloc[-1]
        avg_loss = losses.ewm(span=self.rsi_period, min_periods=self.rsi_period, adjust=False).mean().iloc[-1]

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        return {"rsi": float(rsi)}

    def _compute_ma_features(self, df: pd.DataFrame) -> Dict:
        """Compute moving average features."""
        if len(df) < self.rolling_window:
            return {"ma_gap": 0.0, "ma_slope": 0.0}

        ma = df["close"].iloc[-self.rolling_window:].mean()
        current = df["close"].iloc[-1]

        ma_gap = (current - ma) / ma if ma > 0 else 0.0

        # MA slope (compare current MA to MA 5 bars ago)
        if len(df) >= self.rolling_window + 5:
            ma_prev = df["close"].iloc[-(self.rolling_window + 5):-5].mean()
            ma_slope = (ma - ma_prev) / ma_prev if ma_prev > 0 else 0.0
        else:
            ma_slope = 0.0

        return {"ma_gap": float(ma_gap), "ma_slope": float(ma_slope)}

    def _compute_volume_features(self, df: pd.DataFrame) -> Dict:
        """Compute volume z-score and volume ratio."""
        if len(df) < self.rolling_window:
            return {"volume_zscore": 0.0, "volume_ratio": 1.0}

        vol_series = df["volume"].iloc[-self.rolling_window:]
        mean_vol = vol_series.mean()
        std_vol = vol_series.std()

        current_vol = df["volume"].iloc[-1]

        if std_vol > 0:
            zscore = (current_vol - mean_vol) / std_vol
        else:
            zscore = 0.0

        volume_ratio = current_vol / mean_vol if mean_vol > 0 else 1.0

        return {
            "volume_zscore": float(zscore),
            "volume_ratio": float(volume_ratio),
        }

    def _compute_intrabar_features(self, df_1m: pd.DataFrame, bar_1h: pd.Series) -> Dict:
        """Compute intrabar features from 1m data within the 1h bar."""
        if len(df_1m) < 60:
            return self._empty_intrabar_features()

        df_1m = df_1m.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_1m["open_time"]):
            df_1m["open_time"] = pd.to_datetime(df_1m["open_time"], utc=True)
        if "open_time" in bar_1h:
            hour_start = pd.Timestamp(bar_1h["open_time"])
            if hour_start.tzinfo is None and getattr(df_1m["open_time"].dt, "tz", None) is not None:
                hour_start = hour_start.tz_localize(df_1m["open_time"].dt.tz)
            hour_end = hour_start + pd.Timedelta(hours=1)
            df_hour = df_1m[
                (df_1m["open_time"] >= hour_start) & (df_1m["open_time"] < hour_end)
            ].sort_values("open_time")

            if len(df_hour) != 60:
                logger.warning(
                    "Incomplete intrabar window for %s: %d/60 1m bars",
                    hour_start,
                    len(df_hour),
                )
                return self._empty_intrabar_features()
        else:
            df_hour = df_1m.sort_values("open_time").iloc[-60:].copy()

        entry_price = df_hour["open"].iloc[0]

        # Max runup and drawdown from entry
        if entry_price > 0:
            max_runup = (df_hour["high"].max() - entry_price) / entry_price
            max_drawdown = (df_hour["low"].min() - entry_price) / entry_price
        else:
            max_runup = 0.0
            max_drawdown = 0.0

        # Intrabar volatility (std of 1m returns)
        returns_1m = np.log(df_hour["close"] / df_hour["close"].shift(1)).dropna()
        intrabar_vol = returns_1m.std() if len(returns_1m) > 0 else 0.0

        # Intrabar skewness
        if len(returns_1m) > 2:
            intrabar_skew = returns_1m.skew()
        else:
            intrabar_skew = 0.0

        # Up/down ratio based on 1m log returns (matches offline resample.py)
        log_rets = returns_1m  # already computed above as log returns
        up_count = (log_rets > 0).sum()
        down_count = (log_rets < 0).sum()
        total_moves = up_count + down_count
        up_down_ratio = up_count / total_moves if total_moves > 0 else 0.5

        return {
            "max_runup": float(max_runup),
            "max_drawdown": float(max_drawdown),
            "intrabar_vol": float(intrabar_vol) if pd.notna(intrabar_vol) else 0.0,
            "intrabar_skew": float(intrabar_skew) if pd.notna(intrabar_skew) else 0.0,
            "up_down_ratio": float(up_down_ratio),
        }

    def _compute_atr_ratio(self, df_1h: pd.DataFrame) -> Dict:
        """Compute ATR ratio: current ATR vs 20-bar rolling mean."""
        period = self.adx_period
        if len(df_1h) < period + 20:
            return {"atr_ratio": 1.0}

        try:
            high = df_1h["high"]
            low = df_1h["low"]
            close = df_1h["close"]

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
            atr_rolling_mean = atr.rolling(window=20, min_periods=20).mean()

            current_atr = atr.iloc[-1]
            current_mean = atr_rolling_mean.iloc[-1]

            if pd.notna(current_mean) and current_mean > 0:
                ratio = current_atr / current_mean
            else:
                ratio = 1.0

            return {"atr_ratio": float(ratio)}
        except Exception:
            return {"atr_ratio": 1.0}

    def _compute_ret_24(self, df_1h: pd.DataFrame) -> Dict:
        """Compute 24-hour log return."""
        if len(df_1h) < 25:
            return {"ret_24": 0.0}

        current = df_1h["close"].iloc[-1]
        past = df_1h["close"].iloc[-25]  # shift(24) equivalent

        if past > 0 and current > 0:
            ret_24 = np.log(current / past)
        else:
            ret_24 = 0.0

        return {"ret_24": float(ret_24)}

    def _compute_hour_features(self, open_time) -> Dict:
        """Compute cyclical hour-of-day encoding."""
        ts = pd.Timestamp(open_time)
        hour = ts.hour
        return {
            "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        }

    def _compute_dow_features(self, open_time) -> Dict:
        """Compute cyclical day-of-week encoding."""
        ts = pd.Timestamp(open_time)
        dow = ts.dayofweek  # 0=Monday, 6=Sunday
        return {
            "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
            "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
        }

    def _compute_rsi_slope(self, df_1h: pd.DataFrame) -> Dict:
        """Compute RSI slope: RSI change over last 3 bars."""
        if len(df_1h) < self.rsi_period + 4:
            return {"rsi_slope": 0.0}

        delta = df_1h["close"].diff()
        gains = delta.clip(lower=0)
        losses = (-delta).clip(lower=0)
        avg_gain = gains.ewm(span=self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()
        avg_loss = losses.ewm(span=self.rsi_period, min_periods=self.rsi_period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi_series = rsi_series.fillna(50.0)
        rsi_series = rsi_series.where(avg_loss != 0, 100.0)
        rsi_series = rsi_series.where((avg_loss != 0) | (avg_gain != 0), 50.0)

        current_rsi = rsi_series.iloc[-1]
        past_rsi = rsi_series.iloc[-4] if len(rsi_series) >= 4 else current_rsi
        rsi_slope = current_rsi - past_rsi

        return {"rsi_slope": float(rsi_slope) if pd.notna(rsi_slope) else 0.0}

    def _compute_4h_features(self, df_4h: pd.DataFrame, current_1h_open_time) -> Dict:
        """Compute 4-hour timeframe context features."""
        if len(df_4h) == 0:
            return {
                "trend_4h": 0.0,
                "ma_slope_4h": 0.0,
                "volatility_4h": 0.0,
            }

        df_4h = df_4h.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_4h["open_time"]):
            df_4h["open_time"] = pd.to_datetime(df_4h["open_time"])

        current_1h_open_time = pd.Timestamp(current_1h_open_time)
        df_4h["available_time"] = df_4h["open_time"] + pd.Timedelta(hours=4)
        eligible = df_4h[df_4h["available_time"] <= current_1h_open_time].copy()

        if len(eligible) < self.ma_length:
            return {
                "trend_4h": 0.0,
                "ma_slope_4h": 0.0,
                "volatility_4h": 0.0,
            }

        ma_series = eligible["close"].rolling(window=self.ma_length, min_periods=self.ma_length).mean()
        ma_current = ma_series.iloc[-1]
        current_close = eligible["close"].iloc[-1]
        trend_4h = 1.0 if current_close > ma_current else 0.0

        if len(ma_series.dropna()) >= 2:
            ma_prev = ma_series.iloc[-2]
            ma_slope_4h = (ma_current - ma_prev) / ma_prev if ma_prev > 0 else 0.0
        else:
            ma_slope_4h = 0.0

        # 4h volatility
        returns_4h = np.log(eligible["close"] / eligible["close"].shift(1)).dropna()
        if len(returns_4h) >= 20:
            volatility_4h = returns_4h.iloc[-20:].std()
        else:
            volatility_4h = returns_4h.std() if len(returns_4h) > 0 else 0.0

        return {
            "trend_4h": trend_4h,
            "ma_slope_4h": float(ma_slope_4h),
            "volatility_4h": float(volatility_4h) if pd.notna(volatility_4h) else 0.0,
        }

    def _compute_latest_adx(self, df_1h: pd.DataFrame, period: int = None) -> Optional[float]:
        """Compute latest ADX value from 1h bars using Wilder smoothing.

        Returns the ADX value, or None if computation fails.
        """
        period = period or self.adx_period
        min_required = 2 * period + 1
        if len(df_1h) < min_required:
            logger.warning(
                "ADX: insufficient data (%d bars, need %d for period=%d)",
                len(df_1h), min_required, period,
            )
            return None

        try:
            df = df_1h.copy()
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

            plus_di = 100 * np.divide(
                plus_dm_smoothed,
                atr,
                out=np.zeros(len(df), dtype=float),
                where=(atr != 0),
            )
            minus_di = 100 * np.divide(
                minus_dm_smoothed,
                atr,
                out=np.zeros(len(df), dtype=float),
                where=(atr != 0),
            )
            di_sum = plus_di + minus_di
            dx = 100 * np.divide(
                np.abs(plus_di - minus_di),
                di_sum,
                out=np.zeros(len(df), dtype=float),
                where=(di_sum != 0),
            )
            adx = pd.Series(dx, index=df.index).ewm(
                alpha=alpha, min_periods=period, adjust=False
            ).mean()

            value = float(adx.iloc[-1]) if len(adx) else 0.0
            if np.isnan(value):
                logger.warning("ADX: computed NaN, returning None")
                return None
            return value
        except Exception as e:
            logger.error("ADX computation failed: %s", e, exc_info=True)
            return None

    def get_latest_atr(self, period: int = None) -> Optional[float]:
        """Return latest ATR value for barrier sizing.

        Uses the same Wilder-smoothed True Range as the ADX computation.
        """
        if self.bars_1h is None or len(self.bars_1h) < (period or self.adx_period) + 1:
            return None

        df_1h = pd.DataFrame(list(self.bars_1h))
        period = period or self.adx_period

        try:
            high = df_1h["high"]
            low = df_1h["low"]
            close = df_1h["close"]

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

            value = float(atr.iloc[-1])
            if np.isnan(value) or value <= 0:
                return None
            return value
        except Exception as e:
            logger.error("ATR computation failed: %s", e, exc_info=True)
            return None

    def check_entry_signal(self, adx_min_threshold: float = None) -> bool:
        """
        Check if breakout entry signal fires.

        Signal = (1h close > 20-bar rolling high) AND (4h bullish trend)
        """
        if not self._has_sufficient_data():
            logger.info("No signal: insufficient data")
            return False

        df_1h = pd.DataFrame(list(self.bars_1h))
        df_4h = pd.DataFrame(list(self.bars_4h))

        if len(df_1h) < self.rolling_window + 1:
            logger.info("No signal: not enough 1h bars for rolling window (%d/%d)",
                        len(df_1h), self.rolling_window + 1)
            return False

        # Rolling close-high equivalent to offline signal definition.
        rolling_high = df_1h["close"].iloc[-(self.rolling_window + 1):-1].max()
        current_close = df_1h["close"].iloc[-1]

        # Breakout check
        breakout = current_close > rolling_high

        if not breakout:
            logger.info("No signal: breakout=False (close=%.2f <= rolling_high=%.2f)",
                        current_close, rolling_high)
            return False

        # 4h trend check aligned with offline "available_time" logic.
        current_1h_open_time = pd.Timestamp(df_1h["open_time"].iloc[-1])
        df_4h = df_4h.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_4h["open_time"]):
            df_4h["open_time"] = pd.to_datetime(df_4h["open_time"])
        df_4h["available_time"] = df_4h["open_time"] + pd.Timedelta(hours=4)
        eligible = df_4h[df_4h["available_time"] <= current_1h_open_time]

        if len(eligible) < self.ma_length:
            logger.info("No signal: insufficient eligible 4h bars (%d/%d)",
                        len(eligible), self.ma_length)
            return False

        ma_4h = eligible["close"].iloc[-self.ma_length:].mean()
        close_4h = eligible["close"].iloc[-1]
        trend_bullish = close_4h > ma_4h

        if not trend_bullish:
            logger.info("No signal: trend_4h=bearish (close_4h=%.2f <= ma_4h=%.2f)",
                        close_4h, ma_4h)
            return False

        # Optional ADX regime filter to avoid low-trend conditions.
        if config.USE_ADX_FILTER:
            threshold = (
                adx_min_threshold
                if adx_min_threshold is not None
                else config.ADX_MIN_THRESHOLD
            )
            adx_value = self._compute_latest_adx(df_1h, self.adx_period)
            if adx_value is None:
                logger.warning(
                    "No signal: ADX computation returned None (insufficient data or error)"
                )
                return False
            if adx_value < threshold:
                logger.info(
                    "No signal: ADX too low (adx=%.2f < %.2f)",
                    adx_value, threshold,
                )
                return False

        logger.info(
            "Signal fired! close=%.2f > rolling_high=%.2f, close_4h=%.2f > ma_4h=%.2f",
            current_close, rolling_high, close_4h, ma_4h,
        )
        return True

    def check_mean_reversion_signal(self) -> bool:
        """
        Check if mean-reversion entry signal fires.

        Signal = RSI < MR_RSI_OVERSOLD AND lower_wick_ratio > MR_LOWER_WICK_RATIO_MIN
                 AND ret_3 < MR_RET_THRESHOLD AND NOT breakout
        """
        if not self._has_sufficient_data():
            return False

        df_1h = pd.DataFrame(list(self.bars_1h))

        if len(df_1h) < self.rolling_window + 1:
            return False

        # Compute needed features from latest bar
        latest = df_1h.iloc[-1]

        # RSI check
        rsi_features = self._compute_rsi(df_1h)
        rsi = rsi_features.get("rsi", 50.0)

        # Lower wick ratio check
        wick_features = self._compute_wick_features(latest)
        lwr = wick_features.get("lower_wick_ratio", 0.0)

        # ret_3 check (recent decline)
        ret_features = self._compute_return_features(df_1h)
        ret_3 = ret_features.get("ret_3", 0.0)

        # NOT breakout check (exclude if breakout also fires)
        rolling_high = df_1h["close"].iloc[-(self.rolling_window + 1):-1].max()
        current_close = df_1h["close"].iloc[-1]
        is_breakout = current_close > rolling_high

        # Bearish trend guard: don't take MR longs in strong downtrends
        df_4h = pd.DataFrame(list(self.bars_4h))
        current_1h_open_time = pd.Timestamp(df_1h["open_time"].iloc[-1])
        htf_features = self._compute_4h_features(df_4h, current_1h_open_time)
        trend_4h = htf_features.get("trend_4h", 0.0) > 0.5

        ret_24_features = self._compute_ret_24(df_1h)
        ret_24 = ret_24_features.get("ret_24", 0.0)
        not_deep_downtrend = ret_24 > -0.05
        trend_guard = trend_4h or not_deep_downtrend

        # Evaluate all conditions
        rsi_ok = rsi < config.MR_RSI_OVERSOLD
        lwr_ok = lwr > config.MR_LOWER_WICK_RATIO_MIN
        ret3_ok = ret_3 < config.MR_RET_THRESHOLD
        not_breakout = not is_breakout

        if not (rsi_ok and lwr_ok and ret3_ok and not_breakout and trend_guard):
            logger.info(
                "No MR signal: rsi=%.1f(%s) lwr=%.3f(%s) ret3=%.4f(%s) breakout=%s trend_guard=%s(4h=%s ret24=%.4f)",
                rsi, "OK" if rsi_ok else "FAIL",
                lwr, "OK" if lwr_ok else "FAIL",
                ret_3, "OK" if ret3_ok else "FAIL",
                "no" if not_breakout else "YES-skip",
                "OK" if trend_guard else "FAIL", trend_4h, ret_24,
            )
            return False

        logger.info(
            "MR signal fired! rsi=%.1f, lower_wick_ratio=%.3f, ret_3=%.4f, trend_guard=True",
            rsi, lwr, ret_3,
        )
        return True

    def check_volume_spike_signal(self) -> bool:
        """
        Check if volume-spike entry signal fires.

        Signal = (volume_zscore > VS_VOLUME_ZSCORE_MIN OR volume_ratio > VS_VOLUME_RATIO_MIN)
                 AND body_ratio > VS_BODY_RATIO_MIN AND ret_1 > VS_RET_1_MIN
                 AND (trend_4h OR close > MA20)
        """
        if not self._has_sufficient_data():
            return False

        df_1h = pd.DataFrame(list(self.bars_1h))
        df_4h = pd.DataFrame(list(self.bars_4h))

        if len(df_1h) < self.rolling_window + 1:
            return False

        latest = df_1h.iloc[-1]

        # Volume condition
        vol_features = self._compute_volume_features(df_1h)
        vol_zscore = vol_features.get("volume_zscore", 0.0)
        vol_ratio = vol_features.get("volume_ratio", 1.0)

        vol_spike = (vol_zscore > config.VS_VOLUME_ZSCORE_MIN or
                     vol_ratio > config.VS_VOLUME_RATIO_MIN)

        # Body ratio check
        wick_features = self._compute_wick_features(latest)
        body_ratio = wick_features.get("body_ratio", 0.0)

        # ret_1 check (positive momentum)
        ret_features = self._compute_return_features(df_1h)
        ret_1 = ret_features.get("ret_1", 0.0)

        # Trend context: 4h trend bullish OR close > MA20
        current_1h_open_time = pd.Timestamp(df_1h["open_time"].iloc[-1])
        htf_features = self._compute_4h_features(df_4h, current_1h_open_time)
        trend_4h = htf_features.get("trend_4h", 0.0) > 0.5

        ma_20 = df_1h["close"].iloc[-self.rolling_window:].mean()
        current_close = df_1h["close"].iloc[-1]
        close_above_ma = current_close > ma_20

        # Evaluate all conditions
        vol_ok = vol_spike
        body_ok = body_ratio > config.VS_BODY_RATIO_MIN
        ret1_ok = ret_1 > config.VS_RET_1_MIN
        trend_ok = trend_4h or close_above_ma

        if not (vol_ok and body_ok and ret1_ok and trend_ok):
            logger.info(
                "No VS signal: vol_z=%.2f vol_r=%.2f(%s) body=%.3f(%s) ret1=%.4f(%s) trend4h=%s ma=%s(%s)",
                vol_zscore, vol_ratio, "OK" if vol_ok else "FAIL",
                body_ratio, "OK" if body_ok else "FAIL",
                ret_1, "OK" if ret1_ok else "FAIL",
                trend_4h, close_above_ma,
                "OK" if trend_ok else "FAIL",
            )
            return False

        logger.info(
            "VS signal fired! vol_zscore=%.2f, vol_ratio=%.2f, body_ratio=%.3f, ret_1=%.4f",
            vol_zscore, vol_ratio, body_ratio, ret_1,
        )
        return True

    def check_all_signals(self, adx_min_threshold: float = None) -> List[str]:
        """
        Check all enabled signal types and return list of fired signals.

        Returns:
            List of fired signal type names (e.g., ["breakout", "mean_reversion"])
        """
        fired = []

        if config.SIGNAL_BREAKOUT_ENABLED:
            if self.check_entry_signal(adx_min_threshold=adx_min_threshold):
                fired.append("breakout")

        if config.SIGNAL_MEAN_REVERSION_ENABLED:
            if self.check_mean_reversion_signal():
                fired.append("mean_reversion")

        if config.SIGNAL_VOLUME_SPIKE_ENABLED:
            if self.check_volume_spike_signal():
                fired.append("volume_spike")

        return fired

    def get_rolling_high(self) -> Optional[float]:
        """Get current 20-bar rolling high (debug/display only).

        NOTE: This uses high column and includes the current bar.
        The actual signal logic in check_entry_signal() uses close column
        and excludes the current bar: df["close"].iloc[-(window+1):-1].max().
        Do NOT use this for trading decisions.
        """
        if self.bars_1h is None or len(self.bars_1h) < self.rolling_window:
            return None
        df = pd.DataFrame(list(self.bars_1h))
        return float(df["high"].iloc[-self.rolling_window:].max())

    def get_rolling_low(self) -> Optional[float]:
        """Get current 20-bar rolling low (debug/display only).

        NOTE: This uses low column and includes the current bar.
        Not used in signal logic. For display/monitoring purposes only.
        """
        if self.bars_1h is None or len(self.bars_1h) < self.rolling_window:
            return None
        df = pd.DataFrame(list(self.bars_1h))
        return float(df["low"].iloc[-self.rolling_window:].min())
