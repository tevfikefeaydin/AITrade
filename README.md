# ML-Assisted Crypto Trading Research Pipeline

An educational project for building and backtesting ML-assisted trading strategies on Binance Spot for BTCUSDT and ETHUSDT.

**DISCLAIMER**: This is an educational project. No trading strategy guarantees profits. Past performance does not indicate future results. Trading cryptocurrencies involves substantial risk of loss.

## Overview

This pipeline implements a meta-labeling approach to trading:

1. **Candidate Generator**: Multi-signal entries (breakout, mean-reversion, volume-spike) with higher timeframe context
2. **Triple-Barrier Labeling**: TP/SL with fractional timeout labels based on realized exit location
3. **ML Meta-Filter**: Predicts probability of trade success using offline/live-aligned features
4. **Realistic Backtesting**: T+1 execution, fees, slippage, and 1m barrier fill logic

## Quick Start

### Installation

```bash
# Clone or download the project
cd AITradew.AI

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# 1. Download data (smart incremental - only downloads missing data)
python -m src.cli download

# 2. Build features and label candidates
python -m src.cli build

# 3. Train models with walk-forward validation
python -m src.cli train --train_window_days 270 --test_window_days 60

# 4. Run backtest with ML filter
python -m src.cli backtest --prob_threshold 0.50 --fee_bps 10 --slippage_bps 2
```

### Smart Download Feature

The download command now supports **incremental downloads** - it only downloads missing data:

```bash
# First run: Downloads full date range
python -m src.cli download

# Subsequent runs: Checks existing data, only downloads if needed
python -m src.cli download
# Output: "Data is up to date. No download needed."

# Expand date range: Only downloads the new portion
python -m src.cli download --start 2023-01-01
# Output: Downloads only 2023-01-01 to existing start date

# Force re-download: Ignore existing data
python -m src.cli download --force
```

### Output Files

After running the pipeline, you'll find:

```
outputs/
  BTCUSDT_trades.csv      # Individual trades
  BTCUSDT_equity.csv      # Equity curve
  BTCUSDT_summary.json    # Performance metrics
  ETHUSDT_trades.csv
  ETHUSDT_equity.csv
  ETHUSDT_summary.json
  leaderboard.csv         # Side-by-side comparison

models/
  BTCUSDT_model.pkl       # Trained model
  ETHUSDT_model.pkl
```

## Pipeline Architecture

### Data Flow

```
1. Download: Binance API -> data/{symbol}_1m.parquet
2. Build: 1m -> 1h/4h resampling -> features -> labels
3. Train: features + labels -> walk-forward CV -> models
4. Backtest: model + features + 1m data -> trades/equity/metrics
```

### Signal Generation

**Entry Signals**:
- **Breakout**: 1h close breaks above rolling high with 4h trend confirmation and ADX filter
- **Mean-Reversion**: Oversold pullback with lower-wick rejection and trend/regime guard
- **Volume-Spike**: Bullish impulse bar with abnormal volume and trend support

**Exit Conditions**:
- Take-profit: +0.8% (configurable)
- Stop-loss: -0.6% (configurable)
- Max hold: 12 hours (configurable)

### Feature Engineering (NO LOOKAHEAD)

All features are computed using only data available at the time of the signal:

**1h OHLCV Features**:
- Wick features (upper_wick, lower_wick, body, range, ratios)
- Log returns (1, 3, 6, 12 bar lookback)
- Rolling volatility (20-bar std of returns)
- RSI(14)
- MA gap (distance from 20-bar MA)
- Volume z-score

**Intrabar Features** (from 1m within each 1h):
- Max runup/drawdown relative to open
- Intrabar volatility and skew
- Up/down move ratio

**4h Context** (properly lagged):
- 4h trend filter (close > MA50)
- 4h MA slope

### Triple-Barrier Labeling

For each candidate entry at time t:
1. Look forward in 1m data for up to `max_hold` hours
2. If TP barrier hit first -> label = 1 (success)
3. If SL barrier hit first -> label = 0 (failure)
4. If timeout occurs, assign a fractional label based on the exit return's position between SL and TP

### Walk-Forward Training

- **No shuffle**: Respects temporal order
- **Rolling window**: Train on 270 days, test on 60 days, step forward
- **Per-fold metrics**: AUC, log loss
- **OOS predictions**: Saved per fold for honest backtests
- **Final model**: Trained on all data for paper trading

### Realistic Backtesting

**T+1 Execution**: Signal at bar close t -> execute at bar t+1 open

**Cost Model**:
- Entry: `price * (1 + fee_bps/10000 + slippage_bps/10000)`
- Exit: `price * (1 - fee_bps/10000 - slippage_bps/10000)`

**Barrier Fill**: Uses 1m data to detect precise barrier touches

## CLI Reference

### Download Data

```bash
python -m src.cli download [--start YYYY-MM-DD] [--end YYYY-MM-DD]

# Examples:
python -m src.cli download  # Uses default range (2024-01-01 to today)
python -m src.cli download --start 2023-01-01 --end 2024-12-31
```

### Build Features

```bash
python -m src.cli build
```

### Train Models

```bash
python -m src.cli train [options]

Options:
  --train_window_days INT   Training window (default: 270)
  --test_window_days INT    Test window per fold (default: 60)
```

### Run Backtest

```bash
python -m src.cli backtest [options]

Options:
  --prob_threshold FLOAT    Min probability to take trade (default: 0.50)
  --fee_bps FLOAT          Trading fee in basis points (default: 10)
  --slippage_bps FLOAT     Slippage in basis points (default: 2)
  --pt FLOAT               Take-profit percentage (default: 0.008)
  --sl FLOAT               Stop-loss percentage (default: 0.006)
  --max_hold INT           Max holding period in hours (default: 12)
```

## Important Considerations

### Data Size Warning

1-minute data is large. The default date range starts at 2024-01-01 and runs to today, which typically means:
- ~1M rows per symbol per year
- ~100MB per symbol in Parquet format

For longer periods, expect multi-GB downloads. Start small and expand.

### No Guarantee of Profits

This pipeline demonstrates trading research techniques but:
- **Edges are small**: Real alpha is hard to find
- **Regimes change**: What worked historically may not work in the future
- **Costs matter**: Fees and slippage erode returns
- **Overfitting risk**: More features/parameters increase overfitting risk

### Wick/Intrabar Limitations

- OHLC data doesn't capture intrabar order flow
- Wicks indicate rejection but not the sequence of price movements
- 1m data helps but still has limitations

### Fees and Slippage

Default assumptions:
- **Fee**: 10 bps (0.10%) - typical for Binance maker/taker
- **Slippage**: 2 bps (0.02%) - conservative estimate for liquid pairs

Actual costs depend on:
- Account tier (VIP level reduces fees)
- BNB holding (fee discount)
- Order size relative to liquidity
- Market conditions

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_features.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Test Coverage

- `test_wicks.py`: Wick formula correctness
- `test_features.py`: NO LOOKAHEAD verification
- `test_labeling.py`: Triple-barrier on synthetic data
- `test_backtest.py`: T+1 execution, cost application

## Project Structure

```
AITradew.AI/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration and defaults
│   ├── data_binance.py    # Binance API downloader
│   ├── resample.py        # 1m -> 1h -> 4h resampling
│   ├── features.py        # Feature engineering
│   ├── signals.py         # Candidate generation
│   ├── labeling.py        # Triple-barrier labeling
│   ├── train.py           # Walk-forward training
│   ├── backtest.py        # Realistic backtest engine
│   ├── metrics.py         # Performance metrics
│   ├── cli.py             # CLI entry point
│   └── utils.py           # Shared utilities
├── tests/
│   ├── test_wicks.py
│   ├── test_features.py
│   ├── test_labeling.py
│   └── test_backtest.py
├── data/                  # Downloaded data (gitignored)
├── models/                # Trained models (gitignored)
├── outputs/               # Backtest results (gitignored)
├── requirements.txt
├── README.md
└── .gitignore
```

## Extending the Pipeline

### Adding New Features

Edit `src/features.py`:
1. Create a new `compute_*_features()` function
2. Add to `build_features()` pipeline
3. Update `get_feature_columns()` list

### Adding New Signals

Edit `src/signals.py`:
1. Modify `generate_entry_signals()` for entry logic
2. Modify `generate_exit_signals()` for exit logic

### Changing the Model

Edit `src/train.py`:
1. Modify `get_model()` to return your preferred model
2. Update `config.LGBM_PARAMS` if using LightGBM

## License

This project is for educational purposes only. Use at your own risk.

## Acknowledgments

Inspired by:
- Advances in Financial Machine Learning (Marcos Lopez de Prado)
- Triple-barrier labeling method
- Walk-forward validation techniques
