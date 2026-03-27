# CLAUDE.md - AITradew.AI Project Guide

## Project Overview
ML-Assisted Crypto Trading Research Pipeline for BTCUSDT/ETHUSDT spot trading.
Educational project - meta-labeling approach with LightGBM+XGBoost ensemble,
triple-barrier labeling, walk-forward validation with purge gap, Optuna HPO,
and realistic backtesting.

## Quick Commands
```bash
pip install -r requirements.txt
python -m src.cli download                      # Incremental 1m kline download
python -m src.cli build                         # Features + signals + labeling
python -m src.cli optimize                      # Optuna hyperparameter search
python -m src.cli train                         # Walk-forward ML training (ensemble)
python -m src.cli backtest --prob_threshold 0.50 # Backtest with ML filter
python -m src.cli paper --all                   # Live paper trading (all symbols)
pytest tests/ -v                                # Run all tests
```

## Architecture
```
DOWNLOAD (Binance REST) -> BUILD (resample + features + label) -> TRAIN (LightGBM) -> BACKTEST -> PAPER TRADE
```

### Pipeline Stages
1. **Download** (`data_binance.py`): Smart incremental 1m kline download from Binance public API
2. **Build** (`resample.py` + `features.py` + `signals.py` + `labeling.py`): 1m->1h->4h resampling, 27 features + ADX signal filter, breakout signals, triple-barrier labeling
3. **Train** (`train.py`): Walk-forward validation with LightGBM binary classifier + OOS predictions saved
4. **Backtest** (`backtest.py` + `metrics.py`): T+1 execution, fee+slippage cost model, 1m barrier fills, OOS predictions for honest evaluation
5. **Paper Trade** (`src/live/`): Real-time WebSocket streaming, live feature computation, position management

## Project Structure
```
src/
  config.py          # All parameters: symbols, dates, API, barriers, LightGBM, costs
  cli.py             # 5 CLI commands: download, build, train, backtest, paper
  data_binance.py    # Binance REST API downloader (incremental, gap detection)
  resample.py        # 1m->1h->4h OHLCV resampling + intrabar features
  features.py        # 22 features + ADX (signal filter, not model feature) with NO-LOOKAHEAD guarantee
  signals.py         # Trend-following breakout signal (close > rolling_high(20) AND 4h_trend AND adx >= 12)
  labeling.py        # Triple-barrier: TP=+0.8%, SL=-0.6%, timeout=12h
  train.py           # Walk-forward LightGBM training (270d train, 60d test, early stopping, feature pruning)
  backtest.py        # Realistic spot backtest (T+1, costs, 1m barrier fill)
  metrics.py         # Sharpe, drawdown, win rate, profit factor, leaderboard
  utils.py           # Logging, date utils, safe_divide, cost calc, parquet I/O
  live/
    websocket_client.py  # Binance WS (wss://stream.binance.com) real-time 1m klines
    feature_buffer.py    # Real-time feature computation (mirrors offline features)
    position_manager.py  # Paper trade tracking, JSON persistence
    paper_trader.py      # Main engine: PaperTrader + MultiPaperTrader + soft guardrail
tests/
  test_wicks.py          # Wick formula correctness
  test_features.py       # NO-LOOKAHEAD verification (critical)
  test_labeling.py       # Triple-barrier on synthetic data
  test_backtest.py       # T+1 execution, cost application, barrier raw price
  test_regressions.py    # Regression tests (overlapping trades, intrabar signs, close-breakout, ADX filter, soft guard, PM history)
  test_signals.py        # Signal generation (breakout, MR, volume-spike, exit, candidates)
  test_data_binance.py   # Data downloader (interval conversion, gap detection, mocked API)
  test_resample.py       # Resampling (OHLCV aggregation, intrabar features, 4h alignment)
data/                    # Parquet files (gitignored)
models/                  # Trained .pkl models (gitignored)
outputs/                 # Backtest CSVs + JSON summaries (gitignored)
```

## Key Design Decisions
- **NO-LOOKAHEAD**: All features use backward-looking windows + `.shift(1)`. 4h data lagged via `available_time` mechanism. Tests verify this.
- **T+1 Execution**: Signal at bar close t -> execute at bar t+1 open price
- **Cost Model**: `entry = price * (1 + (fee+slippage)/10000)`, `exit = price * (1 - (fee+slippage)/10000)`
- **Triple-Barrier**: TP/SL/Timeout checked on 1m data for precise fills
- **Walk-Forward**: No temporal leakage, rolling train/test windows, early stopping (patience=20), two-pass feature pruning (<1% importance dropped)
- **OOS Backtest**: Backtest uses out-of-sample fold predictions (`{symbol}_oos_predictions.parquet`), not final model — prevents in-sample overfitting. Final model is saved only for live paper trading.
- **Multi-Signal System**: Three signal types (breakout, mean-reversion, volume-spike) generate candidates independently. Same bar can produce multiple candidates from different types. ML model evaluates each via `signal_type_encoded` feature, backtest picks highest-probability candidate per bar.
- **Signal Logic**: Breakout: `close > rolling_high(20).shift(1)` AND `trend_4h` AND `adx_14 >= 12`. Mean-Reversion: `rsi < 35` AND `lower_wick_ratio > 0.2` AND `ret_3 < 0` AND NOT breakout AND `(trend_4h OR ret_24 > -5%)`. Volume-Spike: `(vol_zscore > 2 OR vol_ratio > 2)` AND `body_ratio > 0.6` AND `ret_1 > 0` AND trend context.
- **ADX Filter**: 1h ADX (Wilder smoothing, period=14, threshold=12) gates entries in low-trend regimes. ADX is NOT a model feature — it's a pre-signal filter in signals.py and feature_buffer.py
- **Soft Guardrail** (paper only): After 3 consecutive SL or win_rate < 30% in last 7 trades, threshold increases +0.10 and 180min cooldown activates for 12h
- **Barrier Prices**: TP/SL barriers use raw execution price (matches labeling.py), not cost-adjusted price
- **Overlapping trade prevention**: `blocked_until_time` in backtest prevents new entries during open positions

## Feature List (34 features in `get_feature_columns()`)
Wick (3): upper_wick_ratio, lower_wick_ratio, body_ratio
Returns (5): ret_1, ret_3, ret_6, ret_12, ret_24
Volatility (1): vol_20
Momentum (5): rsi, rsi_slope, ma_gap, stoch_rsi, macd_hist
Volume (3): volume_zscore, volume_ratio, taker_buy_ratio
Intrabar (5): max_runup, max_drawdown, intrabar_vol, intrabar_skew, up_down_ratio
4h Context (2): trend_4h, ma_slope_4h
Market Regime (3): atr_ratio, rolling_sharpe_20, bb_width
Cross-Asset (2): btc_ret_1, btc_volume_zscore (BTC features for ETH; 0.0 for BTC itself)
Time (4): hour_sin, hour_cos, dow_sin, dow_cos
Signal (1): signal_type_encoded (0=breakout, 1=mean_reversion, 2=volume_spike)

## Config Defaults (src/config.py)
- Symbols: BTCUSDT, ETHUSDT
- Date range: 2024-01-01 to today (datetime.now())
- Barriers: PT=0.8%, SL=0.6% (fallback), max_hold=12h, ATR dynamic (TP mult=2.5, SL mult=1.0, floor=1.2%, ceil=3.0%)
- Costs: fee=10bps, slippage=2bps
- Training: 270d train (or expanding), 60d test, prob_threshold=0.50, purge_gap=12h
- Ensemble: LightGBM + XGBoost (averaged probabilities), USE_ENSEMBLE=True
- LightGBM: gbdt, 15 leaves, lr=0.05, 500 estimators (early stopping), seed=42, min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0
- XGBoost: max_depth=4, lr=0.05, 500 estimators, subsample=0.8, colsample=0.8
- Feature pruning: FEATURE_MIN_IMPORTANCE_PCT=1.0 (features with <1% avg importance dropped, retrain with pruned set)
- Target: cost-aware labeling (TP shifted up by round-trip costs), sample weights from fractional labels
- Optuna HPO: 50 trials, walk-forward OOS AUC objective
- ADX: period=14, threshold=12, USE_ADX_FILTER=True
- Soft Guard: enabled, SL_streak=3, lookback=7, min_winrate=0.30, bonus=+0.10, cooldown=180min, recovery=12h

## Current State (Mar 2026)

### Completed
- [x] Full pipeline: download -> build -> train -> backtest -> paper
- [x] Incremental data download with gap detection
- [x] 34-feature engineering with strict no-lookahead (incl. taker_buy_ratio, cross-asset, regime, stoch_rsi, macd_hist)
- [x] Walk-forward LightGBM+XGBoost ensemble training with purge gap
- [x] Optuna hyperparameter optimization (walk-forward OOS AUC objective)
- [x] Cost-aware labeling + sample weights from fractional labels
- [x] Realistic backtest (T+1, costs, 1m barrier fills, overlapping trade prevention)
- [x] Paper trading with WebSocket + real-time features
- [x] 177 tests across 9 modules (wicks, features, labeling, backtest, regressions, signals, data_binance, resample, improvements)
- [x] CLI with 6 commands (download, build, optimize, train, backtest, paper)
- [x] 1h ADX regime filter (Wilder smoothing, threshold=12) in signals + live
- [x] Soft guardrail in paper trader (threshold bonus + cooldown on bad streaks)
- [x] Barrier fix: TP/SL use raw execution price (consistent with labeling)
- [x] OOS backtest: walk-forward fold predictions saved & used for honest backtesting (prevents in-sample overfitting)
- [x] ML model improvements: 270d train window (~7-8 folds), LightGBM regularization (num_leaves=15, reg_alpha/lambda), early stopping (patience=20), two-pass feature pruning (<1% importance)
- [x] Multi-signal entry system: breakout + mean-reversion + volume-spike with signal_type_encoded model feature
- [x] ATR dynamic barriers: TP_mult=2.0, SL_mult=1.5, floor=0.4%, ceil=3.0%
- [x] 5 new features: volume_ratio, atr_ratio, ret_24, hour_sin, hour_cos
- [x] Barrier floor raised 0.4%→0.8% (costs were eating floor-level TPs)
- [x] MR bearish trend guard: `trend_4h OR ret_24 > -5%` prevents longs in deep downtrends
- [x] Fractional TIMEOUT labeling: label scaled 0..1 by exit return position between SL and TP
- [x] New features: dow_sin/dow_cos (day-of-week), rsi_slope (RSI direction)
- [x] Removed raw wick features (body, range, upper_wick, lower_wick) — price-level dependent noise
- [x] Per-trade Sharpe added to metrics (trade_sharpe) alongside hourly Sharpe

### Known Discrepancies (Live vs Offline)
- (Resolved) `feature_buffer.py` no longer computes extra `volatility_4h` or `ma_slope` features
- (Resolved) `feature_buffer.py` no longer returns raw wick values (body, range, upper_wick, lower_wick)

### Not Implemented
- [ ] Real live trading (only paper trading)
- [ ] API key management for real orders
- [ ] Advanced risk management (position sizing, account limits)
- [ ] Web dashboard / monitoring UI
- [ ] Hyperparameter optimization
- [ ] Multi-exchange support
- [ ] Database persistence (trades stored as JSON/CSV only)
- [x] VPS deployment scripts (Ubuntu 24.04 / Hostinger) -- see deploy/
- [x] VPS DEPLOYED & RUNNING (2026-02-10, Hostinger 194.31.55.142)

## VPS Deployment (Hostinger Ubuntu 24.04) -- LIVE
```
deploy/
  setup_vps.sh           # System setup (Python, ufw, fail2ban, swap, aitradew user)
  install.sh             # Project setup (venv, pip, tests, optional initial pipeline)
  deploy.sh              # Local->VPS file sync via rsync
  enable_services.sh     # Install & start all systemd services
  health_check.sh        # Quick status check for all components
  aitradew-paper.service # Systemd: paper trader (--all, auto-restart)
  aitradew-update.service/.timer  # Daily data download+build (00:05 UTC)
  aitradew-retrain.service/.timer # Weekly retrain+backtest (Sun 02:00 UTC)
  aitradew-sudoers       # Passwordless restart for retrain script
```

### Deployment Steps (Windows PowerShell -> VPS)
```powershell
# 1. Local: Upload files (elle yaz, kopyalama - PowerShell link olarak algilayabiliyor)
ssh root@VPS_IP "mkdir -p /root/app"
scp -r src root@VPS_IP:/root/app/
scp -r tests root@VPS_IP:/root/app/
scp -r deploy root@VPS_IP:/root/app/
scp requirements.txt root@VPS_IP:/root/app/
scp CLAUDE.md root@VPS_IP:/root/app/
scp README.md root@VPS_IP:/root/app/
```
```bash
# 2. VPS: System setup (run once as root)
bash /root/app/deploy/setup_vps.sh

# 3. VPS: Install (run as root, copies to aitradew user automatically)
tmux                                    # internet kesilirse devam etsin
bash /root/app/deploy/install.sh        # pipeline sorusuna y yaz

# 4. VPS: Enable services
bash /home/aitradew/app/deploy/enable_services.sh
cp /home/aitradew/app/deploy/aitradew-sudoers /etc/sudoers.d/aitradew

# 5. VPS: Health check
bash /home/aitradew/app/deploy/health_check.sh
```

### Service Management
```bash
sudo systemctl status aitradew-paper     # Paper trader durumu
sudo systemctl restart aitradew-paper    # Yeniden baslat
sudo journalctl -u aitradew-paper -f     # Canli log
sudo systemctl list-timers aitradew-*    # Timer durumu
tail -f /home/aitradew/logs/paper_trader.log
```

## Testing
```bash
pytest tests/ -v                    # All tests
pytest tests/test_features.py -v    # Critical: no-lookahead verification
pytest tests/test_regressions.py -v # Regression tests
```

## Bug Fixes Applied
- `labeling.py` + `backtest.py`: Both-barrier heuristic improved — uses bar_open distance to barriers instead of bar_close >= entry_price (more accurate ~5% of ambiguous trades)
- `train.py`: OOS predictions deduplicated on (open_time, signal_type_encoded) composite key with consistent int type
- `backtest.py`: OOS merge enforces int type for signal_type_encoded; logs warning when candidates are dropped due to missing OOS predictions
- `paper_trader.py`: Guard state + cooldown expiry now persisted to disk immediately (was lost on crash between expiry check and next save)
- `position_manager.py`: OPEN positions deduplicated by (symbol, entry_time) during history restore to prevent double-restore on crash
- `resample.py` + `features.py`: datetime64 resolution mismatch fix (ms vs us) for pandas 2.x merge_asof
- `setup_vps.sh`: Removed SSH PermitRootLogin hardening (was locking out password-based root access)
- `install.sh`: Now runs as root and auto-copies files from /root/app to /home/aitradew/app
- `backtest.py`: Barrier prices now use raw execution_price (was cost-adjusted, mismatched labeling)
- `paper_trader.py`: Barrier prices use raw market price; naive datetime fix in _format_hold_time
- `position_manager.py`: fromisoformat() naive datetime → UTC fallback
- `cli.py`: Removed unused --bar_interval args from build/train/backtest
- `deploy/*.sh`: Removed --bar_interval 1h from build commands
- `feature_buffer.py`: ADX computation returns None on failure with proper logging (was silent)
- `paper_trader.py`: compute_features/predict_proba exceptions now increment failure counter
- `paper_trader.py`: Guard state (guard_mode_until, next_entry_allowed_at) persisted to disk across restarts
- `websocket_client.py`: REST gap-fill on WS reconnect (fills missing 1m bars + rebuilds 1h/4h)
- `train.py` + `backtest.py` + `cli.py`: OOS prediction pipeline — backtest was using final model (trained on ALL data) causing in-sample overfitting; now uses walk-forward out-of-sample fold predictions
- `config.py` + `train.py`: ML model improvements — train window 540→270d, LightGBM regularization (num_leaves 31→15, reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20), early stopping (n_estimators 100→500 with patience=20), two-pass feature pruning (drops features with <1% avg importance)
- `metrics.py`: `compute_baseline_buy_hold` early return was missing `bh_cagr` key (inconsistent with normal return path)
- `paper_trader.py`: TP/SL percentage display now shows actual computed barrier percentages (was showing fixed config values even with ATR barriers)
- `feature_buffer.py`: Removed extra features not used by model (volatility_4h, ma_slope, raw wick values)
- `backtest.py`: `simulate_barrier_exit` empty-window edge case now correctly returns raw execution_price as gross_exit (was returning cost-adjusted entry price)
- `signals.py`: Removed unused `max_hold` parameter from `generate_candidates`
- `config.py` + `cli.py`: Added `get_symbol_labeled_path` helper to avoid hardcoded labeled paths

## Important Notes
- **STALE BUILD WARNING**: After ANY code change to `signals.py`, `labeling.py`, `features.py`, or `resample.py`, you MUST re-run `python -m src.cli build` before `train`/`backtest`. Stale parquet files will silently produce wrong labels. Symptom: label distribution mismatch between environments (e.g., 23% vs 40% positive rate was caused by stale T+0 labels after T+1 fix). The labeled parquet must contain `execution_time`/`execution_price` columns — if missing, the build is stale.
- Never modify feature computation without re-running `test_features.py` (lookahead check)
- Backtest uses `blocked_until_time` to prevent overlapping trades (verified by test_regressions)
- Signal uses CLOSE breakout, not HIGH spike (verified by test_regressions)
- 4h data alignment uses `available_time = open_time + 4h` to prevent lookahead
- `run_baseline_all_candidates()` uses DummyModel with prob=1.0 and threshold=0.0
- Backtest loads `{symbol}_oos_predictions.parquet` by default; falls back to final model with IN-SAMPLE warning if missing
- `train` command must be run before `backtest` to generate OOS predictions — final model is for live only
- ADX (`adx_14`) is computed in features.py and used in signals.py but is NOT in `get_feature_columns()` — it's a signal filter, not a model feature
- Soft guardrail is paper-only; backtest does not simulate it (intentional — live risk layer)
- PowerShell'den scp/ssh komutlarini ELLE YAZ, kopyala-yapistir link olarak algilanabiliyor
- Uzun islemler icin VPS'te `tmux` kullan (internet kesilirse devam eder)
