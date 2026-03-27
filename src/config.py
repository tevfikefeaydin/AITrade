"""
Configuration for ML-Assisted Crypto Trading Research Pipeline.

This module contains all configurable parameters for the trading research system.
Default values are designed to be "sane" for initial testing - small date ranges
to avoid multi-GB downloads and reasonable hyperparameters.

IMPORTANT: This is an educational project. No trading strategy guarantees profits.
"""

from datetime import datetime
from typing import List
from pathlib import Path

# =============================================================================
# FIXED SYMBOLS - Only these two symbols are supported
# =============================================================================
SYMBOLS: List[str] = ["BTCUSDT", "ETHUSDT"]

# =============================================================================
# DATE RANGE DEFAULTS
# Small default range to avoid huge downloads. Users can expand later.
# 1-minute data for many years is extremely large (several GB per symbol).
# =============================================================================
DEFAULT_START: str = "2024-01-01"
DEFAULT_END: str = datetime.now().strftime("%Y-%m-%d")

# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# =============================================================================
# BINANCE API CONFIGURATION
# =============================================================================
BINANCE_BASE_URL: str = "https://api.binance.com"
BINANCE_KLINES_ENDPOINT: str = "/api/v3/klines"
BINANCE_MAX_KLINES_PER_REQUEST: int = 1000
BINANCE_RATE_LIMIT_SLEEP_MS: int = 100  # Polite sleep between requests
BINANCE_MAX_RETRIES: int = 3
BINANCE_RETRY_BACKOFF_FACTOR: float = 2.0

# =============================================================================
# SIGNAL GENERATION PARAMETERS
# =============================================================================
# Trend-following breakout with HTF filter
ROLLING_WINDOW: int = 20  # N-bar rolling high/low for breakout detection
MA_LENGTH: int = 50  # 4h MA length for trend filter
USE_ADX_FILTER: bool = True  # Gate entries when trend strength is weak
ADX_PERIOD: int = 14  # ADX lookback period on 1h bars
ADX_MIN_THRESHOLD: float = 12.0  # Minimum ADX to allow entries

# Multi-Signal Configuration
SIGNAL_BREAKOUT_ENABLED: bool = True
SIGNAL_MEAN_REVERSION_ENABLED: bool = True
SIGNAL_VOLUME_SPIKE_ENABLED: bool = True

# Mean-Reversion Signal Thresholds
MR_RSI_OVERSOLD: float = 35.0          # RSI < this → oversold
MR_LOWER_WICK_RATIO_MIN: float = 0.2   # Rejection wick threshold
MR_RET_LOOKBACK: int = 3               # ret_3 < 0 → recent decline
MR_RET_THRESHOLD: float = 0.0          # ret_3 must be below this

# Volume Spike Signal Thresholds
VS_VOLUME_ZSCORE_MIN: float = 2.0      # Volume > 2 std above mean
VS_VOLUME_RATIO_MIN: float = 2.0       # Volume > 2x rolling average
VS_BODY_RATIO_MIN: float = 0.6         # Strong directional candle
VS_RET_1_MIN: float = 0.0              # Positive current return

# Signal type encoding for model feature
SIGNAL_TYPE_MAP: dict = {"breakout": 0, "mean_reversion": 1, "volume_spike": 2}

# =============================================================================
# TRIPLE-BARRIER LABELING PARAMETERS
# =============================================================================
DEFAULT_PT: float = 0.008  # Take-profit barrier: +0.8% (fallback when ATR disabled)
DEFAULT_SL: float = 0.006  # Stop-loss barrier: -0.6% (fallback when ATR disabled)
DEFAULT_MAX_HOLD: int = 12  # Maximum holding period in 1h bars

# ATR-Based Dynamic Barriers
USE_ATR_BARRIERS: bool = True
TP_ATR_MULTIPLIER: float = 2.5  # TP = entry + ATR_14 * 2.5
SL_ATR_MULTIPLIER: float = 1.0  # SL = entry - ATR_14 * 1.0  (gross R:R = 2.5:1)
MIN_BARRIER_PCT: float = 0.012  # Floor: 1.2% (barriers never narrower than this — costs eat too much below this)
MAX_BARRIER_PCT: float = 0.030  # Ceiling: 3.0% (barriers never wider than this)

# Cost-Aware Labeling: add round-trip cost to TP target so label=1 means net-profitable
COST_AWARE_LABELING: bool = True

# =============================================================================
# COST MODEL PARAMETERS
# =============================================================================
DEFAULT_FEE_BPS: float = 10.0  # Trading fee in basis points (0.10%)
DEFAULT_SLIPPAGE_BPS: float = 2.0  # Estimated slippage in basis points

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
DEFAULT_TRAIN_WINDOW_DAYS: int = 270  # ~9 months — more folds for reliable CV
DEFAULT_TEST_WINDOW_DAYS: int = 60  # ~2 months of test data per fold
DEFAULT_PROB_THRESHOLD: float = 0.50  # Minimum probability to take a trade

# =============================================================================
# LIVE SOFT GUARDRAIL CONFIGURATION
# =============================================================================
SOFT_GUARD_ENABLED: bool = True
SOFT_GUARD_SL_STREAK_TRIGGER: int = 3
SOFT_GUARD_LOOKBACK_TRADES: int = 7
SOFT_GUARD_MIN_WINRATE: float = 0.30
SOFT_GUARD_THRESHOLD_BONUS: float = 0.10
SOFT_GUARD_MAX_THRESHOLD: float = 0.80
SOFT_GUARD_COOLDOWN_MINUTES: int = 180
SOFT_GUARD_RECOVERY_HOURS: int = 12

# =============================================================================
# MODEL PARAMETERS (LightGBM defaults)
# =============================================================================
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 15,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 500,
    "random_state": 42,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

# XGBoost parameters (used in ensemble mode)
XGBM_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": 0,
}

# Ensemble configuration
USE_ENSEMBLE: bool = True

# Feature pruning: drop features with < this % of total importance
FEATURE_MIN_IMPORTANCE_PCT: float = 1.0

# =============================================================================
# WALK-FORWARD IMPROVEMENTS
# =============================================================================
PURGE_GAP_HOURS: int = 12  # Gap between train end and test start to prevent leakage
USE_EXPANDING_WINDOW: bool = False  # When True, train_start is always data start

# =============================================================================
# OPTUNA HYPERPARAMETER OPTIMIZATION
# =============================================================================
OPTUNA_N_TRIALS: int = 50
OPTUNA_TIMEOUT_SECONDS: int = 3600  # 1 hour max

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================
# RSI period
RSI_PERIOD: int = 14

# Rolling window for volatility and z-scores
FEATURE_ROLLING_WINDOW: int = 20

# Return lookback periods
RETURN_PERIODS: List[int] = [1, 3, 6, 12]

# =============================================================================
# BACKTEST CONFIGURATION
# =============================================================================
INITIAL_CAPITAL: float = 10000.0  # Starting capital for backtest


def get_symbol_data_path(symbol: str, interval: str = "1m") -> Path:
    """Get the path for a symbol's data file."""
    return DATA_DIR / f"{symbol}_{interval}.parquet"


def get_symbol_features_path(symbol: str) -> Path:
    """Get the path for a symbol's features file."""
    return DATA_DIR / f"{symbol}_features.parquet"


def get_symbol_model_path(symbol: str) -> Path:
    """Get the path for a symbol's trained model."""
    return MODELS_DIR / f"{symbol}_model.pkl"


def get_symbol_labeled_path(symbol: str) -> Path:
    """Get the path for a symbol's labeled candidates file."""
    return DATA_DIR / f"{symbol}_labeled.parquet"


def get_symbol_oos_path(symbol: str) -> Path:
    """Get the path for a symbol's out-of-sample predictions."""
    return DATA_DIR / f"{symbol}_oos_predictions.parquet"


def get_symbol_best_params_path(symbol: str) -> Path:
    """Get the path for a symbol's Optuna best hyperparameters."""
    return MODELS_DIR / f"{symbol}_best_params.json"


def get_symbol_output_paths(symbol: str) -> dict:
    """Get all output paths for a symbol."""
    return {
        "trades": OUTPUTS_DIR / f"{symbol}_trades.csv",
        "equity": OUTPUTS_DIR / f"{symbol}_equity.csv",
        "summary": OUTPUTS_DIR / f"{symbol}_summary.json",
    }
