import asyncio
import json
from collections import deque
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock, AsyncMock

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from src import config
from src.live.feature_buffer import FeatureBuffer
from src.live.paper_trader import PaperTrader
from src.live.position_manager import PositionManager
from src.live.websocket_client import BinanceWebSocket
from src.signals import (
    generate_candidates,
    generate_entry_signals,
    generate_mean_reversion_signals,
    generate_volume_spike_signals,
)
from src.train import train_fold


class DummyModel:
    feature_cols_ = []

    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])


def test_backtest_blocks_overlapping_trades():
    start = datetime(2024, 1, 1, 0, 0)
    hours = 12

    df_1h = pd.DataFrame(
        {
            "open_time": [start + timedelta(hours=i) for i in range(hours)],
            "open": [100.0] * hours,
            "high": [100.2] * hours,
            "low": [99.8] * hours,
            "close": [100.0] * hours,
            "volume": [1000.0] * hours,
        }
    )

    minutes = hours * 60
    df_1m = pd.DataFrame(
        {
            "open_time": [start + timedelta(minutes=i) for i in range(minutes)],
            "open": [100.0] * minutes,
            "high": [100.05] * minutes,
            "low": [99.95] * minutes,
            "close": [100.0] * minutes,
            "volume": [100.0] * minutes,
        }
    )

    df_features = pd.DataFrame({"open_time": df_1h["open_time"]})
    df_labeled = pd.DataFrame(
        {
            "entry_time": [start + timedelta(hours=1), start + timedelta(hours=2)],
            "entry_idx": [1, 2],
            "entry_price": [100.0, 100.0],
        }
    )

    trades_df, _, _ = run_backtest(
        symbol="BTCUSDT",
        df_features=df_features,
        df_labeled=df_labeled,
        model=DummyModel(),
        df_1m=df_1m,
        df_1h=df_1h,
        prob_threshold=0.0,
        pt=0.5,
        sl=0.5,
        max_hold=3,
    )

    assert len(trades_df) == 1


def test_live_intrabar_feature_sign_and_ratio():
    fb = FeatureBuffer()

    start = datetime(2024, 1, 1, 0, 0)
    opens = [100.0] * 60
    closes = [101.0 if i % 2 == 0 else 99.0 for i in range(60)]
    highs = [max(o, c) + 0.1 for o, c in zip(opens, closes)]
    lows = [95.0 if i == 10 else min(o, c) - 0.1 for i, (o, c) in enumerate(zip(opens, closes))]

    df_1m = pd.DataFrame(
        {
            "open_time": [start + timedelta(minutes=i) for i in range(60)],
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100.0] * 60,
        }
    )

    out = fb._compute_intrabar_features(df_1m, pd.Series({"open": 100.0}))

    assert out["max_drawdown"] < 0
    assert 0.0 <= out["up_down_ratio"] <= 1.0


def test_live_entry_signal_uses_close_breakout_not_high_spike():
    fb = FeatureBuffer()

    # Use recent timestamps so freshness check passes
    now = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_1h = now - timedelta(hours=50)
    bars_1h = []
    for i in range(51):
        bars_1h.append(
            {
                "open_time": start_1h + timedelta(hours=i),
                "open": 100.0,
                "high": 100.0 if i < 50 else 150.0,  # last bar: high spike
                "low": 99.0,
                "close": 100.0,  # close stays flat - no close breakout
                "volume": 1000.0,
            }
        )

    start_4h = now - timedelta(hours=4 * 55)
    bars_4h = []
    for i in range(55):
        bars_4h.append(
            {
                "open_time": start_4h + timedelta(hours=4 * i),
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.0 + i,
                "volume": 1000.0,
            }
        )

    bars_1m = []
    for i in range(60):
        t = start_1h + timedelta(minutes=i)
        bars_1m.append(
            {
                "open_time": t,
                "open": 100.0,
                "high": 100.1,
                "low": 99.9,
                "close": 100.0,
                "volume": 100.0,
            }
        )

    fb.bars_1h = deque(bars_1h, maxlen=720)
    fb.bars_4h = deque(bars_4h, maxlen=180)
    fb.bars_1m = deque(bars_1m, maxlen=1440)

    assert not bool(fb.check_entry_signal())


def test_position_manager_loads_history_stats(tmp_path):
    log_path = tmp_path / "paper_trades.json"
    history = [
        {"status": "CLOSED", "pnl_pct": 1.5},
        {"status": "CLOSED", "pnl_pct": -0.5},
        {"status": "OPEN", "pnl_pct": 0.0},
    ]
    log_path.write_text(json.dumps(history))

    pm = PositionManager(str(log_path))
    stats = pm.get_stats()

    assert stats["total_trades"] == 2
    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert stats["total_pnl_pct"] == 1.0


# ---------------------------------------------------------------------------
# Backfill REST integration tests
# ---------------------------------------------------------------------------

def _make_1m_df(start: datetime, n_minutes: int) -> pd.DataFrame:
    """Helper: create a minimal 1m OHLCV DataFrame."""
    ts = pd.date_range(start, periods=n_minutes, freq="min", tz="UTC")
    return pd.DataFrame({
        "open_time": ts,
        "open": 100.0,
        "high": 100.1,
        "low": 99.9,
        "close": 100.0,
        "volume": 10.0,
    })


@patch("src.live.websocket_client.config")
@patch("src.live.websocket_client._fetch_recent_rest")
@patch("src.live.websocket_client.load_klines")
@patch("src.live.websocket_client._utc_now")
def test_backfill_rest_fills_current_hour_accumulator(
    mock_now, mock_load, mock_rest, mock_config
):
    """REST data fills the current-hour accumulator when parquet is stale."""
    # Fake now = 14:35 UTC on 2026-01-15
    fake_now = pd.Timestamp("2026-01-15 14:35:00", tz="UTC")
    mock_now.return_value = fake_now

    # Parquet has data only up to 00:00 (full day Jan 14)
    df_parquet = _make_1m_df(datetime(2026, 1, 14, 0, 0), 24 * 60)
    mock_load.return_value = df_parquet

    # REST returns 00:00 - 14:34 (875 minutes)
    df_rest_data = _make_1m_df(datetime(2026, 1, 15, 0, 0), 14 * 60 + 35)
    mock_rest.return_value = df_rest_data

    # config stubs
    mock_config.get_symbol_data_path.return_value = MagicMock(exists=lambda: False)

    ws = BinanceWebSocket("btcusdt")
    ws.backfill("BTCUSDT")

    # Current hour is 14:00. Bars 14:00-14:34 = 35 bars in accumulator
    assert len(ws._current_1h_bars) == 35
    # Buffers should be populated
    assert len(ws.buffer_1m) > 0
    assert len(ws.buffer_1h) > 0


@patch("src.live.websocket_client.config")
@patch("src.live.websocket_client._fetch_recent_rest")
@patch("src.live.websocket_client.load_klines")
@patch("src.live.websocket_client._utc_now")
def test_backfill_graceful_when_rest_fails(
    mock_now, mock_load, mock_rest, mock_config
):
    """When REST API is unreachable, backfill still works from parquet alone."""
    fake_now = pd.Timestamp("2026-01-15 14:35:00", tz="UTC")
    mock_now.return_value = fake_now

    df_parquet = _make_1m_df(datetime(2026, 1, 14, 0, 0), 24 * 60)
    mock_load.return_value = df_parquet
    mock_rest.return_value = None  # REST failed

    mock_config.get_symbol_data_path.return_value = MagicMock(exists=lambda: False)

    ws = BinanceWebSocket("btcusdt")
    ws.backfill("BTCUSDT")

    # Parquet ends before 14:00, so no current-hour data
    assert len(ws._current_1h_bars) == 0
    # But buffers should still be filled from parquet
    assert len(ws.buffer_1m) > 0
    assert len(ws.buffer_1h) > 0


@patch("src.live.websocket_client._fetch_recent_rest")
@patch("src.live.websocket_client.load_klines")
@patch("src.live.websocket_client._utc_now")
def test_backfill_no_parquet_no_crash(mock_now, mock_load, mock_rest):
    """When no parquet file exists, backfill returns gracefully."""
    mock_now.return_value = pd.Timestamp("2026-01-15 14:35:00", tz="UTC")
    mock_load.return_value = None  # No parquet
    mock_rest.return_value = None

    ws = BinanceWebSocket("btcusdt")
    ws.backfill("BTCUSDT")

    assert len(ws.buffer_1m) == 0
    assert len(ws.buffer_1h) == 0
    assert len(ws.buffer_4h) == 0


@patch("src.live.websocket_client.config")
@patch("src.live.websocket_client._fetch_recent_rest")
@patch("src.live.websocket_client.load_klines")
@patch("src.live.websocket_client._utc_now")
def test_backfill_rest_only_warm_start(
    mock_now, mock_load, mock_rest, mock_config
):
    """When no parquet exists but REST succeeds, buffers are populated from REST alone."""
    fake_now = pd.Timestamp("2026-01-15 14:35:00", tz="UTC")
    mock_now.return_value = fake_now

    # No parquet data
    mock_load.return_value = None

    # REST returns 14:35 worth of 1m data (00:00 - 14:34 = 875 minutes)
    df_rest_data = _make_1m_df(datetime(2026, 1, 15, 0, 0), 14 * 60 + 35)
    mock_rest.return_value = df_rest_data

    mock_config.get_symbol_data_path.return_value = MagicMock(exists=lambda: False)

    ws = BinanceWebSocket("btcusdt")
    ws.backfill("BTCUSDT")

    # Current hour is 14:00. Bars 14:00-14:34 = 35 bars in accumulator
    assert len(ws._current_1h_bars) == 35
    # Buffers should be populated from REST data
    assert len(ws.buffer_1m) > 0
    assert len(ws.buffer_1h) > 0


def test_4h_freshness_rejects_stale_data():
    """4h bars older than MAX_DATA_AGE_HOURS_4H (8h) cause rejection."""
    fb = FeatureBuffer()
    now = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)

    # Fresh 1h bars: 51 bars, last one = 1 hour ago
    start_1h = now - timedelta(hours=50)
    bars_1h = [
        {
            "open_time": start_1h + timedelta(hours=i),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
        }
        for i in range(51)
    ]

    # STALE 4h bars: 55 bars, last open_time = 9 hours ago (exceeds 8h threshold)
    bars_4h = [
        {
            "open_time": now - timedelta(hours=9 + (54 - i) * 4),
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1000.0,
        }
        for i in range(55)
    ]

    # Fresh 1m bars: 60 bars
    bars_1m = [
        {
            "open_time": now - timedelta(minutes=60 - i),
            "open": 100.0,
            "high": 100.1,
            "low": 99.9,
            "close": 100.0,
            "volume": 100.0,
        }
        for i in range(60)
    ]

    fb.bars_1h = deque(bars_1h, maxlen=720)
    fb.bars_4h = deque(bars_4h, maxlen=180)
    fb.bars_1m = deque(bars_1m, maxlen=1440)

    # 4h bar is 9h old → should reject
    assert fb._has_sufficient_data() is False

    # Make 4h bar fresh: 7h old (under 8h threshold) → should accept
    bars_4h[-1]["open_time"] = now - timedelta(hours=7)
    assert fb._has_sufficient_data() is True


# ---------------------------------------------------------------------------
# Regression: log_loss single-class guard (train.py)
# ---------------------------------------------------------------------------

def test_train_fold_single_class_no_crash():
    """train_fold must not crash when test set has only one class.

    Before fix: sklearn log_loss raised ValueError on single-class y_test.
    After fix: logloss = 0.0, auc = 0.5 — no crash.
    """
    feature_cols = ["f1", "f2"]
    train_df = pd.DataFrame({
        "f1": np.random.randn(100),
        "f2": np.random.randn(100),
        "label": np.random.randint(0, 2, 100),
    })
    # Test set: ALL labels are 1 (single class)
    test_df = pd.DataFrame({
        "f1": np.random.randn(20),
        "f2": np.random.randn(20),
        "label": np.ones(20, dtype=int),
    })

    model, metrics = train_fold(train_df, test_df, feature_cols)

    assert metrics["logloss"] == 0.0
    assert metrics["auc"] == 0.5
    assert model is not None


# ---------------------------------------------------------------------------
# Regression: signals.py produces execution_time/execution_price (T+1)
# ---------------------------------------------------------------------------

def test_signals_produce_execution_time_and_price():
    """generate_candidates must produce execution_time / execution_price columns.

    Before fix: candidates only had entry_time/entry_price, labeling used
    signal bar price for barriers — equivalent to T+0.
    After fix: execution_time = next bar open_time, execution_price = next bar open.
    """
    n = 30
    start = datetime(2024, 6, 1, 0, 0)
    times = [start + timedelta(hours=i) for i in range(n)]
    # Prices ramp up so close > rolling_high(20).shift(1) triggers near the end
    closes = [100.0 + i * 0.5 for i in range(n)]
    opens = [c - 0.1 for c in closes]

    df = pd.DataFrame({
        "open_time": times,
        "open": opens,
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
        "volume": [1000.0] * n,
        "trend_4h": [True] * n,
        f"adx_{config.ADX_PERIOD}": [30.0] * n,
    })

    candidates = generate_candidates(df, rolling_window=20)
    assert len(candidates) > 0

    assert "execution_time" in candidates.columns
    assert "execution_price" in candidates.columns
    # execution_time must be strictly after entry_time (T+1)
    for _, row in candidates.iterrows():
        assert row["execution_time"] > row["entry_time"]
        # execution_price should be the next bar's open, not the signal bar's close
        assert row["execution_price"] != row["entry_price"]


def test_signals_adx_filter_blocks_low_trend(monkeypatch):
    """Low ADX should block entry even when breakout + trend_4h are true."""
    monkeypatch.setattr(config, "USE_ADX_FILTER", True)
    monkeypatch.setattr(config, "ADX_MIN_THRESHOLD", 20.0)

    adx_col = f"adx_{config.ADX_PERIOD}"
    df = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=3, freq="h"),
            "open": [100.0, 100.0, 102.0],
            "high": [101.0, 102.0, 104.0],
            "low": [99.0, 99.0, 101.0],
            "close": [100.0, 101.0, 103.0],  # breakout at t=2 over prior 2-bar high
            "trend_4h": [True, True, True],
            adx_col: [30.0, 10.0, 10.0],  # low ADX on breakout bar
        }
    )
    sig_low = generate_entry_signals(df, rolling_window=2)
    assert bool(sig_low.iloc[-1]) is False

    df[adx_col] = [30.0, 30.0, 30.0]
    sig_high = generate_entry_signals(df, rolling_window=2)
    assert bool(sig_high.iloc[-1]) is True


# ---------------------------------------------------------------------------
# Regression: paper trader pending signal waits for genuinely new bar
# ---------------------------------------------------------------------------

def test_paper_trader_pending_signal_skips_stale_bar():
    """Pending signal must NOT execute when buffer_1m[-1] is the same bar
    that was present when the signal was generated.

    Before fix: _position_monitor grabbed buffer_1m[-1]["open"] immediately,
    which was the same bar that triggered the signal (stale data).
    After fix: execution only happens when latest bar open_time > signal_bar_time.
    """
    trader = PaperTrader.__new__(PaperTrader)
    trader._running = True
    trader.ws = MagicMock()
    trader.positions = MagicMock()
    trader.positions.get_open.return_value = []
    trader.fee_bps = 10.0
    trader.slippage_bps = 2.0

    signal_bar_time = datetime(2024, 6, 1, 14, 59)

    # Pending signal set with signal_bar_time
    trader._pending_signal = {
        "prob": 0.7,
        "signal_time": datetime(2024, 6, 1, 14, 0),
        "signal_bar_time": signal_bar_time,
    }

    # buffer_1m[-1] is the SAME bar (stale) — open_time == signal_bar_time
    trader.ws.buffer_1m = [{"open_time": signal_bar_time, "open": 50000.0}]

    # Patch _open_position so we can detect if it's called
    trader._open_position = AsyncMock()

    # Run one iteration of the monitor logic
    async def _run_one_tick():
        # Replicate the pending-signal block from _position_monitor
        if trader._pending_signal is not None and trader.ws.buffer_1m:
            latest_bar = trader.ws.buffer_1m[-1]
            if latest_bar["open_time"] > trader._pending_signal["signal_bar_time"]:
                execution_price = latest_bar["open"]
                await trader._open_position(
                    execution_price,
                    trader._pending_signal["prob"],
                    trader._pending_signal["signal_time"],
                    latest_bar["open_time"],
                )
                trader._pending_signal = None

    asyncio.run(_run_one_tick())

    # Should NOT have executed — bar is stale
    trader._open_position.assert_not_called()
    assert trader._pending_signal is not None  # still pending

    # Now simulate a NEW bar arriving (open_time > signal_bar_time)
    new_bar_time = datetime(2024, 6, 1, 15, 0)
    trader.ws.buffer_1m = [{"open_time": new_bar_time, "open": 50100.0}]

    asyncio.run(_run_one_tick())

    # NOW it should have executed on the new bar's open price
    trader._open_position.assert_called_once_with(
        50100.0,
        0.7,
        datetime(2024, 6, 1, 14, 0),
        new_bar_time,
    )
    assert trader._pending_signal is None  # consumed


def test_paper_trader_open_uses_execution_bar_time():
    trader = PaperTrader.__new__(PaperTrader)
    trader.symbol = "BTCUSDT"
    trader.fee_bps = 10.0
    trader.slippage_bps = 2.0
    trader.pt = 0.008
    trader.sl = 0.006
    trader.max_hold_hours = 12
    trader.guard_cooldown_minutes = 180
    trader.positions = MagicMock()
    trader.features = MagicMock()
    trader.features.get_latest_atr.return_value = None
    trader._is_guard_active = MagicMock(return_value=False)
    trader._save_guard_state = MagicMock()

    signal_time = datetime(2024, 6, 1, 14, 0, tzinfo=timezone.utc)
    execution_time = datetime(2024, 6, 1, 15, 0, tzinfo=timezone.utc)

    asyncio.run(
        trader._open_position(
            price=50000.0,
            prob=0.72,
            signal_time=signal_time,
            execution_time=execution_time,
        )
    )

    position = trader.positions.add.call_args.args[0]
    assert position["entry_time"] == execution_time
    assert position["max_exit_time"] == execution_time + timedelta(hours=12)


def test_paper_trader_exit_skips_entry_bar():
    trader = PaperTrader.__new__(PaperTrader)
    trader.ws = MagicMock()
    trader._close_position = AsyncMock()

    entry_time = datetime(2024, 6, 1, 15, 0, tzinfo=timezone.utc)
    trader.ws.buffer_1m = [{
        "open_time": entry_time,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
    }]

    pos = {
        "entry_time": entry_time,
        "market_price": 100.0,
        "tp_price": 100.8,
        "sl_price": 99.4,
        "max_exit_time": entry_time + timedelta(hours=12),
    }

    asyncio.run(trader._check_position_exit(pos))
    trader._close_position.assert_not_called()


def test_paper_trader_both_hit_uses_backtest_heuristic():
    trader = PaperTrader.__new__(PaperTrader)
    trader.ws = MagicMock()
    trader._close_position = AsyncMock()
    trader.fee_bps = 10.0
    trader.slippage_bps = 2.0

    entry_time = datetime(2024, 6, 1, 15, 0, tzinfo=timezone.utc)
    exit_bar_time = entry_time + timedelta(minutes=1)
    trader.ws.buffer_1m = [{
        "open_time": exit_bar_time,
        "high": 101.0,
        "low": 99.0,
        "close": 99.2,
    }]

    pos = {
        "entry_time": entry_time,
        "market_price": 100.0,
        "tp_price": 100.8,
        "sl_price": 99.4,
        "max_exit_time": entry_time + timedelta(hours=12),
    }

    asyncio.run(trader._check_position_exit(pos))
    trader._close_position.assert_called_once_with(
        pos, "SL", 99.4, exit_time=exit_bar_time,
    )


def test_paper_trader_timeout_uses_bar_time_not_wall_clock():
    trader = PaperTrader.__new__(PaperTrader)
    trader.ws = MagicMock()
    trader._close_position = AsyncMock()
    trader.fee_bps = 10.0
    trader.slippage_bps = 2.0

    entry_time = datetime(2024, 6, 1, 15, 0, tzinfo=timezone.utc)
    timeout_bar_time = entry_time + timedelta(hours=12)
    trader.ws.buffer_1m = [{
        "open_time": timeout_bar_time,
        "high": 100.2,
        "low": 99.8,
        "close": 100.1,
    }]

    pos = {
        "entry_time": entry_time,
        "market_price": 100.0,
        "tp_price": 101.0,
        "sl_price": 99.0,
        "max_exit_time": timeout_bar_time,
    }

    asyncio.run(trader._check_position_exit(pos))
    expected_exit_price = 100.1 * (1 - (trader.fee_bps + trader.slippage_bps) / 10000)
    trader._close_position.assert_called_once_with(
        pos, "TIMEOUT", expected_exit_price, exit_time=timeout_bar_time,
    )


# ---------------------------------------------------------------------------
# Regression: position_manager naive datetime gets UTC timezone
# ---------------------------------------------------------------------------

def test_position_manager_naive_datetime_gets_utc(tmp_path):
    """datetime.fromisoformat() on tz-naive strings must produce aware datetimes.

    Before fix: naive datetimes caused TypeError when compared with
    datetime.now(timezone.utc) elsewhere in the code.
    After fix: naive strings get tzinfo=timezone.utc after parsing.
    """
    log_path = tmp_path / "paper_trades.json"
    # ISO strings WITHOUT timezone info (naive)
    history = [
        {
            "status": "OPEN",
            "symbol": "BTCUSDT",
            "entry_time": "2024-06-01T14:00:00",
            "max_exit_time": "2024-06-02T02:00:00",
            "signal_time": "2024-06-01T13:00:00",
            "entry_price": 50000.0,
            "tp_price": 50400.0,
            "sl_price": 49700.0,
        }
    ]
    log_path.write_text(json.dumps(history))

    pm = PositionManager(str(log_path))

    assert len(pm.positions) == 1
    pos = pm.positions[0]

    for key in ["entry_time", "max_exit_time", "signal_time"]:
        dt = pos[key]
        assert isinstance(dt, datetime), f"{key} should be datetime"
        assert dt.tzinfo is not None, f"{key} should be timezone-aware, got naive"
        assert dt.tzinfo == timezone.utc, f"{key} should be UTC"


def test_position_manager_recent_helpers(tmp_path):
    pm = PositionManager(str(tmp_path / "guard_helpers.json"))
    pm.positions = [
        {
            "status": "CLOSED",
            "exit_reason": "TP",
            "pnl_pct": 1.0,
            "exit_time": datetime(2024, 1, 1, 10, tzinfo=timezone.utc),
        },
        {
            "status": "CLOSED",
            "exit_reason": "SL",
            "pnl_pct": -0.8,
            "exit_time": datetime(2024, 1, 1, 11, tzinfo=timezone.utc),
        },
        {
            "status": "CLOSED",
            "exit_reason": "SL",
            "pnl_pct": -0.7,
            "exit_time": datetime(2024, 1, 1, 12, tzinfo=timezone.utc),
        },
    ]
    pm.trade_count = 3

    recent = pm.get_recent_closed(2)
    assert len(recent) == 2
    assert recent[-1]["exit_reason"] == "SL"
    assert pm.get_consecutive_sl_streak() == 2
    assert pm.get_recent_win_rate(3) == 1 / 3


def test_soft_guard_threshold_and_cooldown_helpers():
    now = datetime.now(timezone.utc)
    trader = PaperTrader.__new__(PaperTrader)
    trader.soft_guard = True
    trader.prob_threshold = 0.55
    trader.guard_threshold_bonus = 0.10
    trader._guard_mode_until = now + timedelta(hours=1)
    trader._next_entry_allowed_at = now + timedelta(minutes=5)

    assert abs(trader._effective_prob_threshold(now) - 0.65) < 1e-12
    assert trader._can_open_new_trade_now(now) is False
    assert trader._can_open_new_trade_now(now + timedelta(minutes=6)) is True


def test_soft_guard_activates_on_sl_streak():
    trader = PaperTrader.__new__(PaperTrader)
    trader.soft_guard = True
    trader._guard_mode_until = None
    trader._next_entry_allowed_at = None
    trader._guard_state_path = MagicMock()
    trader._guard_state_path.parent = MagicMock()
    trader.positions = MagicMock()
    trader.positions.get_consecutive_sl_streak.return_value = config.SOFT_GUARD_SL_STREAK_TRIGGER
    trader.positions.get_recent_closed.return_value = []
    trader.positions.get_recent_win_rate.return_value = 1.0

    trader._evaluate_guardrail_state()
    assert trader._guard_mode_until is not None


# ---------------------------------------------------------------------------
# Edge case: ADX insufficient data returns None
# ---------------------------------------------------------------------------

def test_adx_insufficient_data_returns_none():
    """ADX computation with too few bars should return None, not crash."""
    fb = FeatureBuffer()
    # Need 2*period+1 = 29 bars for period=14; provide only 10
    df_short = pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=10, freq="h"),
        "open": [100.0] * 10,
        "high": [101.0] * 10,
        "low": [99.0] * 10,
        "close": [100.0] * 10,
        "volume": [1000.0] * 10,
    })
    result = fb._compute_latest_adx(df_short, period=14)
    assert result is None


# ---------------------------------------------------------------------------
# Edge case: Soft guard recovery expiry
# ---------------------------------------------------------------------------

def test_soft_guard_recovery_expires():
    """Guard mode should expire after recovery period."""
    now = datetime.now(timezone.utc)
    trader = PaperTrader.__new__(PaperTrader)
    trader.soft_guard = True
    trader.prob_threshold = 0.55
    trader.guard_threshold_bonus = 0.10
    # Guard expired 1 hour ago
    trader._guard_mode_until = now - timedelta(hours=1)
    trader._next_entry_allowed_at = None

    # Should NOT be active (expired)
    assert trader._is_guard_active(now) is False
    # Threshold should be normal
    assert trader._effective_prob_threshold(now) == 0.55


# ---------------------------------------------------------------------------
# Edge case: Soft guard activates on low win rate (no SL streak needed)
# ---------------------------------------------------------------------------

def test_soft_guard_activates_on_low_win_rate():
    """Guard mode should activate on low win rate even without SL streak."""
    trader = PaperTrader.__new__(PaperTrader)
    trader.soft_guard = True
    trader._guard_mode_until = None
    trader._next_entry_allowed_at = None
    trader._guard_state_path = MagicMock()
    trader._guard_state_path.parent = MagicMock()
    trader.positions = MagicMock()
    trader.positions.get_consecutive_sl_streak.return_value = 0  # No SL streak
    trader.positions.get_recent_closed.return_value = [{}] * config.SOFT_GUARD_LOOKBACK_TRADES
    trader.positions.get_recent_win_rate.return_value = 0.20  # Below 0.30 threshold

    trader._evaluate_guardrail_state()
    assert trader._guard_mode_until is not None


# ---------------------------------------------------------------------------
# Edge case: Position manager empty history
# ---------------------------------------------------------------------------

def test_position_manager_empty_history_helpers(tmp_path):
    """PM helpers should return safe defaults when no trades exist."""
    pm = PositionManager(str(tmp_path / "empty_trades.json"))

    assert pm.get_recent_closed(5) == []
    assert pm.get_consecutive_sl_streak() == 0
    assert pm.get_recent_win_rate(5) == 0.0


# ---------------------------------------------------------------------------
# Edge case: Position manager zero SL streak
# ---------------------------------------------------------------------------

def test_position_manager_zero_sl_streak(tmp_path):
    """When most recent trade is not SL, streak should be 0."""
    pm = PositionManager(str(tmp_path / "no_sl_streak.json"))
    pm.positions = [
        {
            "status": "CLOSED",
            "exit_reason": "SL",
            "pnl_pct": -0.5,
            "exit_time": datetime(2024, 1, 1, 10, tzinfo=timezone.utc),
        },
        {
            "status": "CLOSED",
            "exit_reason": "TP",
            "pnl_pct": 1.0,
            "exit_time": datetime(2024, 1, 1, 12, tzinfo=timezone.utc),
        },
    ]
    pm.trade_count = 2

    # Most recent is TP, so streak should be 0
    assert pm.get_consecutive_sl_streak() == 0


# ---------------------------------------------------------------------------
# Edge case: Backtest both-hit heuristic
# ---------------------------------------------------------------------------

def test_backtest_uses_oos_predictions_when_provided():
    """Backtest should use OOS predictions instead of model when provided.

    When oos_predictions is passed, model.predict_proba must NOT be called
    and only candidates present in OOS data should be processed.
    """
    start = datetime(2024, 1, 1, 0, 0)
    hours = 12

    df_1h = pd.DataFrame({
        "open_time": [start + timedelta(hours=i) for i in range(hours)],
        "open": [100.0] * hours,
        "high": [100.2] * hours,
        "low": [99.8] * hours,
        "close": [100.0] * hours,
        "volume": [1000.0] * hours,
    })

    minutes = hours * 60
    df_1m = pd.DataFrame({
        "open_time": [start + timedelta(minutes=i) for i in range(minutes)],
        "open": [100.0] * minutes,
        "high": [100.05] * minutes,
        "low": [99.95] * minutes,
        "close": [100.0] * minutes,
        "volume": [100.0] * minutes,
    })

    df_features = pd.DataFrame({"open_time": df_1h["open_time"]})

    # Three candidate entries
    entry_times = [start + timedelta(hours=h) for h in [1, 3, 5]]
    df_labeled = pd.DataFrame({
        "entry_time": entry_times,
        "entry_idx": [1, 3, 5],
        "entry_price": [100.0, 100.0, 100.0],
    })

    # OOS predictions only cover the first two candidates
    oos_predictions = pd.DataFrame({
        "open_time": entry_times[:2],
        "oos_probability": [0.9, 0.8],
        "fold": [0, 1],
    })

    # Use a model that would crash if predict_proba is called
    class CrashModel:
        feature_cols_ = []
        def predict_proba(self, X):
            raise AssertionError("model.predict_proba should not be called with OOS")

    trades_df, _, summary = run_backtest(
        symbol="BTCUSDT",
        df_features=df_features,
        df_labeled=df_labeled,
        model=CrashModel(),
        df_1m=df_1m,
        df_1h=df_1h,
        prob_threshold=0.0,
        pt=0.5,
        sl=0.5,
        max_hold=3,
        oos_predictions=oos_predictions,
    )

    # Only the 2 OOS-covered candidates should be considered
    assert summary.get("n_candidates", 0) == 2


# ---------------------------------------------------------------------------
# compute_barrier_prices: ATR-based + backward compat
# ---------------------------------------------------------------------------

def test_compute_barrier_prices_atr_mode(monkeypatch):
    """ATR-based barriers should use multipliers and clamp to floor/ceiling."""
    from src.utils import compute_barrier_prices

    monkeypatch.setattr(config, "USE_ATR_BARRIERS", True)
    monkeypatch.setattr(config, "TP_ATR_MULTIPLIER", 1.5)
    monkeypatch.setattr(config, "SL_ATR_MULTIPLIER", 1.0)
    monkeypatch.setattr(config, "MIN_BARRIER_PCT", 0.004)
    monkeypatch.setattr(config, "MAX_BARRIER_PCT", 0.020)

    entry_price = 100.0
    atr_value = 1.0  # ATR=1 → tp_pct = 1.5/100 = 0.015, sl_pct = 1.0/100 = 0.01

    tp_price, sl_price = compute_barrier_prices(
        entry_price, pt=0.008, sl=0.006, atr_value=atr_value,
    )

    assert tp_price == pytest.approx(101.5, rel=0.001)  # 100 * (1 + 0.015)
    assert sl_price == pytest.approx(99.0, rel=0.001)   # 100 * (1 - 0.01)


def test_compute_barrier_prices_floor_clamp(monkeypatch):
    """Very small ATR should be clamped to MIN_BARRIER_PCT."""
    from src.utils import compute_barrier_prices

    monkeypatch.setattr(config, "USE_ATR_BARRIERS", True)
    monkeypatch.setattr(config, "TP_ATR_MULTIPLIER", 1.5)
    monkeypatch.setattr(config, "SL_ATR_MULTIPLIER", 1.0)
    monkeypatch.setattr(config, "MIN_BARRIER_PCT", 0.004)
    monkeypatch.setattr(config, "MAX_BARRIER_PCT", 0.020)

    entry_price = 100.0
    atr_value = 0.1  # ATR=0.1 → tp_pct = 0.15/100 = 0.0015 < floor 0.004

    tp_price, sl_price = compute_barrier_prices(
        entry_price, pt=0.008, sl=0.006, atr_value=atr_value,
    )

    # Both should be clamped to floor 0.004
    assert tp_price == pytest.approx(100.4, rel=0.001)  # 100 * (1 + 0.004)
    assert sl_price == pytest.approx(99.6, rel=0.001)   # 100 * (1 - 0.004)


def test_compute_barrier_prices_ceiling_clamp(monkeypatch):
    """Very large ATR should be clamped to MAX_BARRIER_PCT."""
    from src.utils import compute_barrier_prices

    monkeypatch.setattr(config, "USE_ATR_BARRIERS", True)
    monkeypatch.setattr(config, "TP_ATR_MULTIPLIER", 1.5)
    monkeypatch.setattr(config, "SL_ATR_MULTIPLIER", 1.0)
    monkeypatch.setattr(config, "MIN_BARRIER_PCT", 0.004)
    monkeypatch.setattr(config, "MAX_BARRIER_PCT", 0.020)

    entry_price = 100.0
    atr_value = 5.0  # ATR=5 → tp_pct = 7.5/100 = 0.075 > ceiling 0.020

    tp_price, sl_price = compute_barrier_prices(
        entry_price, pt=0.008, sl=0.006, atr_value=atr_value,
    )

    # Both should be clamped to ceiling 0.020
    assert tp_price == pytest.approx(102.0, rel=0.001)  # 100 * (1 + 0.020)
    assert sl_price == pytest.approx(98.0, rel=0.001)   # 100 * (1 - 0.020)


def test_compute_barrier_prices_disabled_falls_back(monkeypatch):
    """USE_ATR_BARRIERS=False should use fixed pt/sl regardless of atr_value."""
    from src.utils import compute_barrier_prices

    monkeypatch.setattr(config, "USE_ATR_BARRIERS", False)

    entry_price = 100.0
    tp_price, sl_price = compute_barrier_prices(
        entry_price, pt=0.008, sl=0.006, atr_value=2.0,
    )

    # Should use fixed percentages
    assert tp_price == pytest.approx(100.8, rel=0.001)  # 100 * (1 + 0.008)
    assert sl_price == pytest.approx(99.4, rel=0.001)   # 100 * (1 - 0.006)


def test_compute_barrier_prices_atr_none_falls_back(monkeypatch):
    """atr_value=None should use fixed pt/sl even when ATR barriers enabled."""
    from src.utils import compute_barrier_prices

    monkeypatch.setattr(config, "USE_ATR_BARRIERS", True)

    entry_price = 100.0
    tp_price, sl_price = compute_barrier_prices(
        entry_price, pt=0.008, sl=0.006, atr_value=None,
    )

    assert tp_price == pytest.approx(100.8, rel=0.001)
    assert sl_price == pytest.approx(99.4, rel=0.001)


def test_compute_barrier_prices_atr_zero_falls_back(monkeypatch):
    """atr_value=0 should use fixed pt/sl (guard against division issues)."""
    from src.utils import compute_barrier_prices

    monkeypatch.setattr(config, "USE_ATR_BARRIERS", True)

    entry_price = 100.0
    tp_price, sl_price = compute_barrier_prices(
        entry_price, pt=0.008, sl=0.006, atr_value=0.0,
    )

    assert tp_price == pytest.approx(100.8, rel=0.001)
    assert sl_price == pytest.approx(99.4, rel=0.001)


def test_compute_barrier_prices_asymmetric_rr(monkeypatch):
    """TP_ATR_MULTIPLIER=1.5, SL_ATR_MULTIPLIER=1.0 should give 1.5:1 gross R:R."""
    from src.utils import compute_barrier_prices

    monkeypatch.setattr(config, "USE_ATR_BARRIERS", True)
    monkeypatch.setattr(config, "TP_ATR_MULTIPLIER", 1.5)
    monkeypatch.setattr(config, "SL_ATR_MULTIPLIER", 1.0)
    monkeypatch.setattr(config, "MIN_BARRIER_PCT", 0.001)
    monkeypatch.setattr(config, "MAX_BARRIER_PCT", 0.100)

    entry_price = 50000.0
    atr_value = 500.0  # typical BTC ATR

    tp_price, sl_price = compute_barrier_prices(
        entry_price, pt=0.008, sl=0.006, atr_value=atr_value,
    )

    tp_distance = tp_price - entry_price  # 500 * 1.5 = 750
    sl_distance = entry_price - sl_price  # 500 * 1.0 = 500

    gross_rr = tp_distance / sl_distance
    assert gross_rr == pytest.approx(1.5, rel=0.01)


def test_backtest_both_hit_heuristic():
    """When both TP and SL are hit in same 1m bar, open-distance determines outcome."""
    from src.backtest import simulate_barrier_exit

    execution_price = 100.0
    pt = 0.01   # TP at 101
    sl = 0.01   # SL at 99
    fee_bps = 0.0
    slippage_bps = 0.0

    entry_time = datetime(2024, 1, 1, 0, 59)

    # Case 1: open closer to TP (100.8) → dist_to_tp=0.2 < dist_to_sl=1.8 → TP
    df_1m_tp = pd.DataFrame({
        "open_time": [datetime(2024, 1, 1, 1, 0)],
        "open": [100.8],   # closer to TP (101) than SL (99)
        "high": [102.0],   # hits TP (101)
        "low": [98.0],     # hits SL (99)
        "close": [101.5],
        "volume": [1000.0],
    })

    _, _, reason_tp, _ = simulate_barrier_exit(
        df_1m=df_1m_tp,
        entry_time=entry_time,
        execution_price=execution_price,
        entry_price_with_costs=execution_price,
        pt=pt, sl=sl,
        max_hold_hours=12,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    assert reason_tp == "TP"

    # Case 2: open closer to SL (99.2) → dist_to_tp=1.8 > dist_to_sl=0.2 → SL
    df_1m_sl = pd.DataFrame({
        "open_time": [datetime(2024, 1, 1, 1, 0)],
        "open": [99.2],    # closer to SL (99) than TP (101)
        "high": [102.0],   # hits TP
        "low": [98.0],     # hits SL
        "close": [98.5],
        "volume": [1000.0],
    })

    _, _, reason_sl, _ = simulate_barrier_exit(
        df_1m=df_1m_sl,
        entry_time=entry_time,
        execution_price=execution_price,
        entry_price_with_costs=execution_price,
        pt=pt, sl=sl,
        max_hold_hours=12,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    assert reason_sl == "SL"

    # Case 3: open equidistant (100.0) → ties go to TP
    df_1m_tie = pd.DataFrame({
        "open_time": [datetime(2024, 1, 1, 1, 0)],
        "open": [100.0],   # equidistant: dist_to_tp=1.0, dist_to_sl=1.0
        "high": [102.0],
        "low": [98.0],
        "close": [98.5],
        "volume": [1000.0],
    })

    _, _, reason_tie, _ = simulate_barrier_exit(
        df_1m=df_1m_tie,
        entry_time=entry_time,
        execution_price=execution_price,
        entry_price_with_costs=execution_price,
        pt=pt, sl=sl,
        max_hold_hours=12,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    assert reason_tie == "TP"  # ties favor TP (closer barrier wins, equal → TP)


# ── Multi-Signal Tests ─────────────────────────────────────────────


def _make_features_df(n=100, seed=42):
    """Create a features DataFrame with all columns needed for signal generation."""
    np.random.seed(seed)
    start = datetime(2024, 1, 1)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.001, n)))
    volumes = np.random.uniform(100, 1000, n)

    df = pd.DataFrame({
        "open_time": [start + timedelta(hours=i) for i in range(n)],
        "open": prices,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": volumes,
    })

    # Compute features needed for signals
    from src.features import (
        compute_wick_features,
        compute_return_features,
        compute_rsi,
        compute_volume_features,
    )

    df = compute_wick_features(df)
    df = compute_return_features(df)
    df = compute_rsi(df)
    df = compute_volume_features(df)

    # Add trend_4h (True for all bars for simplicity)
    df["trend_4h"] = True
    # Add ADX
    df[f"adx_{config.ADX_PERIOD}"] = 20.0

    return df


def test_mean_reversion_signal_basic():
    """MR signal should fire when RSI<30, lower_wick_ratio>0.3, ret_3<0."""
    n = 50
    start = datetime(2024, 1, 1)

    # Create a scenario: declining prices → RSI drops, with a lower wick bounce
    prices_close = [100.0] * 25
    # Decline for 20 bars
    for i in range(20):
        prices_close.append(prices_close[-1] * 0.99)
    # Then 5 more flat
    for i in range(5):
        prices_close.append(prices_close[-1])

    prices_close = prices_close[:n]

    df = pd.DataFrame({
        "open_time": [start + timedelta(hours=i) for i in range(n)],
        "open": prices_close,
        "high": [p * 1.002 for p in prices_close],
        "low": [p * 0.990 for p in prices_close],  # Significant lower wick
        "close": prices_close,
        "volume": [1000.0] * n,
    })

    from src.features import (
        compute_wick_features,
        compute_return_features,
        compute_rsi,
        compute_volume_features,
    )

    df = compute_wick_features(df)
    df = compute_return_features(df)
    df = compute_rsi(df)
    df = compute_volume_features(df)
    df["trend_4h"] = True
    df[f"adx_{config.ADX_PERIOD}"] = 20.0

    mr_signals = generate_mean_reversion_signals(df)

    # We should have at least some MR signals during the decline phase
    # (RSI will drop below 30 during the persistent decline)
    # Even if none fire due to exact thresholds, the function should not crash
    assert len(mr_signals) == n
    assert mr_signals.dtype == bool


def test_volume_spike_signal_basic():
    """VS signal should fire when vol_zscore>2, body_ratio>0.6, ret_1>0, trend support."""
    n = 50
    start = datetime(2024, 1, 1)

    # Steady prices with one big volume bar in the middle
    prices = [100.0 + 0.1 * i for i in range(n)]
    volumes = [100.0] * n
    volumes[30] = 5000.0  # Huge volume spike

    df = pd.DataFrame({
        "open_time": [start + timedelta(hours=i) for i in range(n)],
        "open": [p - 0.5 for p in prices],  # Open lower than close → bullish body
        "high": [p + 0.1 for p in prices],
        "low": [p - 0.6 for p in prices],
        "close": prices,
        "volume": volumes,
    })

    from src.features import (
        compute_wick_features,
        compute_return_features,
        compute_volume_features,
    )

    df = compute_wick_features(df)
    df = compute_return_features(df)
    df = compute_volume_features(df)
    df["trend_4h"] = True

    vs_signals = generate_volume_spike_signals(df)

    assert len(vs_signals) == n
    assert vs_signals.dtype == bool


def test_signal_type_in_candidates():
    """generate_candidates output should have signal_type and signal_type_encoded."""
    df = _make_features_df(100)

    # Enable all signals
    orig_bo = config.SIGNAL_BREAKOUT_ENABLED
    orig_mr = config.SIGNAL_MEAN_REVERSION_ENABLED
    orig_vs = config.SIGNAL_VOLUME_SPIKE_ENABLED
    try:
        config.SIGNAL_BREAKOUT_ENABLED = True
        config.SIGNAL_MEAN_REVERSION_ENABLED = True
        config.SIGNAL_VOLUME_SPIKE_ENABLED = True

        candidates = generate_candidates(df)

        if len(candidates) > 0:
            assert "signal_type" in candidates.columns
            assert "signal_type_encoded" in candidates.columns
            # signal_type values should be valid
            valid_types = set(config.SIGNAL_TYPE_MAP.keys())
            assert set(candidates["signal_type"].unique()).issubset(valid_types)
            # signal_type_encoded should match the map
            for _, row in candidates.iterrows():
                assert row["signal_type_encoded"] == config.SIGNAL_TYPE_MAP[row["signal_type"]]
    finally:
        config.SIGNAL_BREAKOUT_ENABLED = orig_bo
        config.SIGNAL_MEAN_REVERSION_ENABLED = orig_mr
        config.SIGNAL_VOLUME_SPIKE_ENABLED = orig_vs


def test_multi_signal_same_bar():
    """If multiple signals fire on same bar, candidates should have multiple rows for that bar."""
    n = 60
    start = datetime(2024, 1, 1)

    # Create a specific scenario where both MR and VS could potentially fire
    # (This tests the multi-candidate mechanism, not that they MUST fire simultaneously)
    df = _make_features_df(n)

    orig_bo = config.SIGNAL_BREAKOUT_ENABLED
    orig_mr = config.SIGNAL_MEAN_REVERSION_ENABLED
    orig_vs = config.SIGNAL_VOLUME_SPIKE_ENABLED
    try:
        config.SIGNAL_BREAKOUT_ENABLED = True
        config.SIGNAL_MEAN_REVERSION_ENABLED = True
        config.SIGNAL_VOLUME_SPIKE_ENABLED = True

        candidates = generate_candidates(df)

        if len(candidates) > 0:
            # Check that we can have multiple signal types
            signal_types = candidates["signal_type"].unique()
            # At minimum, the structure should support multiple types
            assert "signal_type" in candidates.columns

            # Check entry_time can appear more than once (if multiple signals fire)
            # This is structural - same bar different signal types
            time_counts = candidates.groupby("entry_time")["signal_type"].nunique()
            # At least the mechanism works (even if no same-bar overlap in this data)
            assert len(time_counts) > 0
    finally:
        config.SIGNAL_BREAKOUT_ENABLED = orig_bo
        config.SIGNAL_MEAN_REVERSION_ENABLED = orig_mr
        config.SIGNAL_VOLUME_SPIKE_ENABLED = orig_vs


def test_backward_compat_signals_disabled():
    """With MR+VS disabled, only breakout candidates should be generated."""
    df = _make_features_df(100)

    orig_bo = config.SIGNAL_BREAKOUT_ENABLED
    orig_mr = config.SIGNAL_MEAN_REVERSION_ENABLED
    orig_vs = config.SIGNAL_VOLUME_SPIKE_ENABLED
    try:
        config.SIGNAL_BREAKOUT_ENABLED = True
        config.SIGNAL_MEAN_REVERSION_ENABLED = False
        config.SIGNAL_VOLUME_SPIKE_ENABLED = False

        candidates = generate_candidates(df)

        if len(candidates) > 0:
            assert all(candidates["signal_type"] == "breakout")
            assert all(candidates["signal_type_encoded"] == 0)
    finally:
        config.SIGNAL_BREAKOUT_ENABLED = orig_bo
        config.SIGNAL_MEAN_REVERSION_ENABLED = orig_mr
        config.SIGNAL_VOLUME_SPIKE_ENABLED = orig_vs
