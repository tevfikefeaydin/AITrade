"""
Paper Trading Engine for ML-Assisted Crypto Trading.

Main engine that coordinates:
- WebSocket data streaming
- Feature computation
- ML prediction
- Position management with barrier monitoring
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .. import config
from ..utils import compute_barrier_prices
from ..train import load_model, predict_proba
from .websocket_client import BinanceWebSocket
from .feature_buffer import FeatureBuffer
from .position_manager import PositionManager

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Paper Trading Bot.

    Simulates live trading using real-time data but
    without executing actual orders.

    Flow:
    1. WebSocket receives 1m data
    2. On 1h bar close, check for entry signal
    3. If signal, compute features and get ML probability
    4. If prob >= threshold, open paper position
    5. Monitor positions every minute for TP/SL/Timeout
    6. Log all trades to JSON file
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        prob_threshold: float = None,
        pt: float = None,
        sl: float = None,
        max_hold_hours: int = None,
        fee_bps: float = None,
        slippage_bps: float = None,
        adx_min_threshold: float = None,
        soft_guard: bool = None,
        guard_threshold_bonus: float = None,
        guard_cooldown_minutes: int = None,
    ):
        """
        Initialize paper trader.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            prob_threshold: Min probability to take trade
            pt: Take profit percentage
            sl: Stop loss percentage
            max_hold_hours: Maximum holding period
            fee_bps: Trading fee in basis points
            slippage_bps: Slippage in basis points
            adx_min_threshold: Minimum ADX required by signal filter
            soft_guard: Whether soft guardrail is enabled
            guard_threshold_bonus: Extra probability threshold in guard mode
            guard_cooldown_minutes: Entry cooldown in guard mode
        """
        self.symbol = symbol.upper()

        # Configuration (use defaults from config if not specified)
        self.prob_threshold = prob_threshold if prob_threshold is not None else config.DEFAULT_PROB_THRESHOLD
        self.pt = pt if pt is not None else config.DEFAULT_PT
        self.sl = sl if sl is not None else config.DEFAULT_SL
        self.max_hold_hours = max_hold_hours if max_hold_hours is not None else config.DEFAULT_MAX_HOLD
        self.fee_bps = fee_bps if fee_bps is not None else config.DEFAULT_FEE_BPS
        self.slippage_bps = slippage_bps if slippage_bps is not None else config.DEFAULT_SLIPPAGE_BPS
        self.adx_min_threshold = (
            adx_min_threshold
            if adx_min_threshold is not None
            else config.ADX_MIN_THRESHOLD
        )
        self.soft_guard = (
            soft_guard
            if soft_guard is not None
            else config.SOFT_GUARD_ENABLED
        )
        self.guard_threshold_bonus = (
            guard_threshold_bonus
            if guard_threshold_bonus is not None
            else config.SOFT_GUARD_THRESHOLD_BONUS
        )
        self.guard_cooldown_minutes = (
            guard_cooldown_minutes
            if guard_cooldown_minutes is not None
            else config.SOFT_GUARD_COOLDOWN_MINUTES
        )

        # Components
        self.ws = BinanceWebSocket(symbol.lower())
        self.features = FeatureBuffer()
        self.positions = PositionManager(f"outputs/{symbol}_paper_trades.json")

        # Model (loaded on start)
        self.model = None

        # Guard state persistence file
        self._guard_state_path = Path(f"outputs/{symbol}_guard_state.json")

        # State
        self._running = False
        self._last_signal_time: Optional[datetime] = None
        self._pending_signal: Optional[dict] = None  # T+1 deferred execution
        self._guard_mode_until: Optional[datetime] = None
        self._next_entry_allowed_at: Optional[datetime] = None
        self._consecutive_feature_failures = 0
        self._max_consecutive_failures = 5

        # Restore guard state from disk
        self._load_guard_state()

    def _load_guard_state(self):
        """Restore guard mode timestamps from disk so they survive restarts."""
        if not self._guard_state_path.exists():
            return
        try:
            data = json.loads(self._guard_state_path.read_text())
            now_utc = datetime.now(timezone.utc)
            if data.get("guard_mode_until"):
                dt = datetime.fromisoformat(data["guard_mode_until"])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt > now_utc:
                    self._guard_mode_until = dt
                    logger.info("Restored guard_mode_until: %s", dt.isoformat())
            if data.get("next_entry_allowed_at"):
                dt = datetime.fromisoformat(data["next_entry_allowed_at"])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt > now_utc:
                    self._next_entry_allowed_at = dt
                    logger.info("Restored next_entry_allowed_at: %s", dt.isoformat())
        except Exception as e:
            logger.warning("Could not load guard state: %s", e)

    def _save_guard_state(self):
        """Persist guard mode timestamps to disk."""
        data = {
            "guard_mode_until": (
                self._guard_mode_until.isoformat()
                if self._guard_mode_until else None
            ),
            "next_entry_allowed_at": (
                self._next_entry_allowed_at.isoformat()
                if self._next_entry_allowed_at else None
            ),
        }
        try:
            self._guard_state_path.parent.mkdir(parents=True, exist_ok=True)
            self._guard_state_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Could not save guard state: %s", e)

    async def run(self):
        """
        Start paper trading bot.

        Runs until interrupted (Ctrl+C).
        """
        self._running = True

        # Load ML model
        self.model = load_model(self.symbol)
        if self.model is None:
            logger.error(f"Model not found for {self.symbol}. Run 'train' first.")
            return
        logger.info(f"Loaded model for {self.symbol}")

        # Backfill buffers from parquet before linking
        try:
            self.ws.backfill(self.symbol)
        except Exception as e:
            logger.warning(f"Backfill failed for {self.symbol}: {e}")

        # Link feature buffer to WebSocket
        self.features.link_websocket(self.ws)

        # Set up callbacks
        self.ws.on_1h_bar = self._on_hourly_bar

        # Print startup info
        self._print_startup_info()

        # Run WebSocket and position monitor concurrently
        try:
            await asyncio.gather(
                self.ws.connect(),
                self._position_monitor(),
                self._heartbeat(),
            )
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            self.positions.print_summary()

    def _print_startup_info(self):
        """Print startup configuration."""
        n1m = len(self.ws.buffer_1m)
        n1h = len(self.ws.buffer_1h)
        n4h = len(self.ws.buffer_4h)
        ready = self.features._has_sufficient_data()

        print("\n" + "=" * 60)
        print("PAPER TRADER STARTED")
        print("=" * 60)
        print(f"  Symbol:          {self.symbol}")
        print(f"  Probability:     >= {self.prob_threshold}")
        print(f"  Take Profit:     +{self.pt*100:.1f}%")
        print(f"  Stop Loss:       -{self.sl*100:.1f}%")
        print(f"  Max Hold:        {self.max_hold_hours} hours")
        print(f"  Fee:             {self.fee_bps} bps")
        print(f"  Slippage:        {self.slippage_bps} bps")
        print(f"  ADX Filter:      >= {self.adx_min_threshold:.1f}")
        print(f"  ATR Barriers:    {config.USE_ATR_BARRIERS}")
        if config.USE_ATR_BARRIERS:
            print(f"  TP ATR Mult:     {config.TP_ATR_MULTIPLIER}x")
            print(f"  SL ATR Mult:     {config.SL_ATR_MULTIPLIER}x")
            print(f"  Barrier Clamp:   [{config.MIN_BARRIER_PCT:.1%}, {config.MAX_BARRIER_PCT:.1%}]")
        print(f"  Soft Guard:      {self.soft_guard}")
        if self.soft_guard:
            print(f"  Guard Bonus:     +{self.guard_threshold_bonus:.2f} threshold")
            print(f"  Guard Cooldown:  {self.guard_cooldown_minutes} minutes")
        print(f"  Buffer 1m:       {n1m} bars")
        print(f"  Buffer 1h:       {n1h} bars")
        print(f"  Buffer 4h:       {n4h} bars")
        print(f"  Signal ready:    {ready}")
        print("=" * 60)
        print("Waiting for signals... (Press Ctrl+C to stop)")
        print()

    def _is_guard_active(self, now: Optional[datetime] = None) -> bool:
        """Return True when soft guard mode is currently active."""
        if not self.soft_guard or self._guard_mode_until is None:
            return False
        now_utc = now or datetime.now(timezone.utc)
        if now_utc >= self._guard_mode_until:
            self._guard_mode_until = None
            self._save_guard_state()  # persist expiry so restart doesn't reload stale guard
            return False
        return True

    def _effective_prob_threshold(self, now: Optional[datetime] = None) -> float:
        """Compute active probability threshold with optional guard bonus."""
        if not self._is_guard_active(now):
            return self.prob_threshold
        boosted = self.prob_threshold + self.guard_threshold_bonus
        return min(boosted, config.SOFT_GUARD_MAX_THRESHOLD)

    def _can_open_new_trade_now(self, now: Optional[datetime] = None) -> bool:
        """Check cooldown gate for new entries."""
        if self._next_entry_allowed_at is None:
            return True
        now_utc = now or datetime.now(timezone.utc)
        if now_utc >= self._next_entry_allowed_at:
            self._next_entry_allowed_at = None
            self._save_guard_state()  # persist expiry so restart doesn't reload stale cooldown
            return True
        return False

    def _evaluate_guardrail_state(self):
        """Activate soft guard mode based on recent trade stress signals."""
        if not self.soft_guard:
            return

        now_utc = datetime.now(timezone.utc)
        if self._is_guard_active(now_utc):
            return

        sl_streak = self.positions.get_consecutive_sl_streak()
        recent_trades = self.positions.get_recent_closed(config.SOFT_GUARD_LOOKBACK_TRADES)
        recent_win_rate = self.positions.get_recent_win_rate(config.SOFT_GUARD_LOOKBACK_TRADES)

        streak_triggered = sl_streak >= config.SOFT_GUARD_SL_STREAK_TRIGGER
        winrate_triggered = (
            len(recent_trades) >= config.SOFT_GUARD_LOOKBACK_TRADES
            and recent_win_rate < config.SOFT_GUARD_MIN_WINRATE
        )

        if streak_triggered or winrate_triggered:
            self._guard_mode_until = now_utc + timedelta(hours=config.SOFT_GUARD_RECOVERY_HOURS)
            logger.warning(
                "Soft guard activated until %s (sl_streak=%d, recent_win_rate=%.2f)",
                self._guard_mode_until.isoformat(),
                sl_streak,
                recent_win_rate,
            )
            self._save_guard_state()

    async def _on_hourly_bar(self, bar: dict):
        """
        Handle 1-hour bar close event.

        This is where we check for entry signals.
        """
        timestamp = bar["open_time"]
        close_price = bar["close"]

        logger.info(f"1H bar closed: {timestamp} close=${close_price:.2f}")

        # Skip if we already have an open position
        if self.positions.has_open_position(self.symbol):
            logger.debug("Skipping - already have open position")
            return

        self._evaluate_guardrail_state()

        if not self._can_open_new_trade_now():
            logger.info("Skipping entry - cooldown active until %s", self._next_entry_allowed_at)
            return

        # Check all signal types
        fired_signals = self.features.check_all_signals(adx_min_threshold=self.adx_min_threshold)
        if not fired_signals:
            return

        # Compute features once (shared across all signal evaluations)
        try:
            df_features = self.features.compute_features()
        except Exception as e:
            self._consecutive_feature_failures += 1
            logger.error("Feature computation exception: %s", e, exc_info=True)
            if self._consecutive_feature_failures >= self._max_consecutive_failures:
                logger.error(
                    "ALERT: %d consecutive feature failures for %s!",
                    self._consecutive_feature_failures, self.symbol,
                )
            return

        if df_features is None:
            self._consecutive_feature_failures += 1
            if self._consecutive_feature_failures >= self._max_consecutive_failures:
                logger.error(
                    "ALERT: %d consecutive feature computation failures for %s!",
                    self._consecutive_feature_failures, self.symbol,
                )
            logger.warning("Could not compute features")
            return

        self._consecutive_feature_failures = 0

        # Evaluate each fired signal, pick highest probability
        best_prob = -1.0
        best_signal_type = None

        for signal_type in fired_signals:
            df_eval = df_features.copy()
            df_eval["signal_type_encoded"] = config.SIGNAL_TYPE_MAP[signal_type]

            try:
                prob = predict_proba(self.model, df_eval)[0]
            except Exception as e:
                self._consecutive_feature_failures += 1
                logger.error("Prediction exception for %s: %s", signal_type, e, exc_info=True)
                continue

            logger.info("Signal %s: prob=%.3f", signal_type, prob)

            if prob > best_prob:
                best_prob = prob
                best_signal_type = signal_type

        if best_signal_type is None:
            logger.warning("All signal predictions failed")
            return

        print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] SIGNAL DETECTED!")
        print(f"  Type: {best_signal_type} (from {fired_signals})")
        print(f"  Price: ${close_price:.2f}")
        print(f"  ML Probability: {best_prob:.3f}")

        threshold = self._effective_prob_threshold()

        # Check threshold
        if best_prob < threshold:
            print(f"  -> SKIP (prob {best_prob:.3f} < threshold {threshold:.3f})")
            return

        if not self.ws.buffer_1m:
            logger.warning("No 1m buffer available, cannot stage T+1 execution")
            return

        # Defer execution to next 1m bar (T+1 execution, matches backtest)
        self._pending_signal = {
            "prob": best_prob,
            "signal_time": timestamp,
            "signal_bar_time": self.ws.buffer_1m[-1]["open_time"],
            "threshold": threshold,
            "signal_type": best_signal_type,
        }
        print(f"  -> SIGNAL ACCEPTED ({best_signal_type}) at threshold {threshold:.3f}, waiting for next bar (T+1)...")

    async def _open_position(self, price: float, prob: float, signal_time: datetime):
        """Open a new paper trade position."""
        # Apply entry costs
        cost_mult = 1 + (self.fee_bps + self.slippage_bps) / 10000
        entry_price = price * cost_mult

        # Barrier levels based on raw market price (matches labeling.py)
        atr_value = self.features.get_latest_atr()
        tp_price, sl_price = compute_barrier_prices(
            price, self.pt, self.sl, atr_value=atr_value,
        )
        now_utc = datetime.now(timezone.utc)
        max_exit_time = now_utc + timedelta(hours=self.max_hold_hours)

        position = {
            "symbol": self.symbol,
            "signal_time": signal_time,
            "entry_time": now_utc,
            "entry_price": entry_price,
            "market_price": price,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "max_exit_time": max_exit_time,
            "probability": prob,
            "status": "OPEN",
        }

        self.positions.add(position)

        now_utc = datetime.now(timezone.utc)
        if self._is_guard_active(now_utc):
            self._next_entry_allowed_at = now_utc + timedelta(minutes=self.guard_cooldown_minutes)
            self._save_guard_state()

        print(f"\n{'='*50}")
        print(f"[PAPER TRADE] {self.symbol} LONG OPENED")
        print(f"{'='*50}")
        print(f"  Market Price:  ${price:.2f}")
        print(f"  Entry Price:   ${entry_price:.2f} (with costs)")
        print(f"  Take Profit:   ${tp_price:.2f} (+{self.pt*100:.1f}%)")
        print(f"  Stop Loss:     ${sl_price:.2f} (-{self.sl*100:.1f}%)")
        print(f"  Max Exit:      {max_exit_time.strftime('%H:%M:%S')}")
        print(f"  Probability:   {prob:.3f}")
        print(f"{'='*50}\n")

    async def _position_monitor(self):
        """Monitor open positions for barrier exits and execute pending signals."""
        while self._running:
            await asyncio.sleep(60)  # Check every minute

            # Execute pending signal at next bar's open (T+1 execution)
            if self._pending_signal is not None and self.ws.buffer_1m:
                latest_bar = self.ws.buffer_1m[-1]
                # Only execute when a genuinely new bar has arrived after the signal
                if latest_bar["open_time"] > self._pending_signal["signal_bar_time"]:
                    execution_price = latest_bar["open"]
                    await self._open_position(
                        execution_price,
                        self._pending_signal["prob"],
                        self._pending_signal["signal_time"],
                    )
                    self._pending_signal = None

            for pos in self.positions.get_open():
                await self._check_position_exit(pos)

    async def _check_position_exit(self, pos: dict):
        """Check if position should be closed."""
        # Get current price from latest 1m bar
        if not self.ws.buffer_1m:
            return

        current_bar = self.ws.buffer_1m[-1]
        current_high = current_bar["high"]
        current_low = current_bar["low"]
        current_close = current_bar["close"]

        # Check take profit
        if current_high >= pos["tp_price"]:
            await self._close_position(pos, "TP", pos["tp_price"])
            return

        # Check stop loss
        if current_low <= pos["sl_price"]:
            await self._close_position(pos, "SL", pos["sl_price"])
            return

        # Check timeout
        if datetime.now(timezone.utc) >= pos["max_exit_time"]:
            # Apply exit costs
            cost_mult = 1 - (self.fee_bps + self.slippage_bps) / 10000
            exit_price = current_close * cost_mult
            await self._close_position(pos, "TIMEOUT", exit_price)
            return

    async def _close_position(self, pos: dict, reason: str, exit_price: float):
        """Close a position and log results."""
        # Apply exit costs if TP/SL
        if reason in ["TP", "SL"]:
            cost_mult = 1 - (self.fee_bps + self.slippage_bps) / 10000
            exit_price = exit_price * cost_mult

        # Close position
        self.positions.close_position(pos, reason, exit_price)

        # Calculate P&L
        pnl_pct = pos["pnl_pct"]
        icon = "[WIN]" if pnl_pct > 0 else "[LOSS]"

        print(f"\n{'='*50}")
        print(f"[PAPER TRADE] {icon} {self.symbol} CLOSED - {reason}")
        print(f"{'='*50}")
        print(f"  Entry Price:  ${pos['entry_price']:.2f}")
        print(f"  Exit Price:   ${exit_price:.2f}")
        print(f"  P&L:          {pnl_pct:+.2f}%")
        print(f"  Hold Time:    {self._format_hold_time(pos)}")
        print()

        # Print running stats
        stats = self.positions.get_stats()
        print(f"  Session Stats: {stats['wins']}W / {stats['losses']}L "
              f"({stats['win_rate']:.0f}% WR) | "
              f"Total P&L: {stats['total_pnl_pct']:+.2f}%")
        print(f"{'='*50}\n")

    def _format_hold_time(self, pos: dict) -> str:
        """Format holding time as human-readable string."""
        if "exit_time" not in pos or "entry_time" not in pos:
            return "N/A"

        entry = pos["entry_time"]
        exit_time = pos["exit_time"]

        if isinstance(entry, str):
            entry = datetime.fromisoformat(entry)
            if entry.tzinfo is None:
                entry = entry.replace(tzinfo=timezone.utc)
        if isinstance(exit_time, str):
            exit_time = datetime.fromisoformat(exit_time)
            if exit_time.tzinfo is None:
                exit_time = exit_time.replace(tzinfo=timezone.utc)

        delta = exit_time - entry
        hours = delta.total_seconds() / 3600

        if hours < 1:
            return f"{int(delta.total_seconds() / 60)} minutes"
        else:
            return f"{hours:.1f} hours"

    async def _heartbeat(self):
        """Print periodic status update."""
        while self._running:
            await asyncio.sleep(300)  # Every 5 minutes

            if self.ws.buffer_1m:
                price = self.ws.buffer_1m[-1]["close"]
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M')}] "
                      f"Heartbeat - {self.symbol} ${price:.2f} | "
                      f"1m bars: {len(self.ws.buffer_1m)} | "
                      f"1h bars: {len(self.ws.buffer_1h)} | "
                      f"Open positions: {len(self.positions.get_open())}")

    def stop(self):
        """Stop the paper trader."""
        self._running = False
        self.ws.stop()
        logger.info("Paper trader stopping...")


class MultiPaperTrader:
    """
    Run paper trading for multiple symbols simultaneously.
    """

    def __init__(self, symbols: list = None, **kwargs):
        """
        Initialize multi-symbol paper trader.

        Args:
            symbols: List of trading pairs (default: config.SYMBOLS)
            **kwargs: Additional arguments passed to each PaperTrader
        """
        from .. import config as cfg
        self.symbols = symbols or cfg.SYMBOLS
        self.traders = []
        self.kwargs = kwargs

    async def run(self):
        """Run all traders concurrently."""
        print("\n" + "=" * 60)
        print("MULTI-SYMBOL PAPER TRADER")
        print("=" * 60)
        print(f"  Symbols: {', '.join(self.symbols)}")
        print("=" * 60 + "\n")

        # Create traders for each symbol
        for symbol in self.symbols:
            try:
                trader = PaperTrader(symbol=symbol, **self.kwargs)
                self.traders.append(trader)
                logger.info(f"Initialized trader for {symbol}")
            except Exception as e:
                logger.error(f"Failed to initialize {symbol}: {e}")

        if not self.traders:
            logger.error("No traders initialized!")
            return

        # Run all traders concurrently
        tasks = [trader.run() for trader in self.traders]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self.print_combined_summary()

    def print_combined_summary(self):
        """Print combined summary for all symbols."""
        print("\n" + "=" * 60)
        print("COMBINED PAPER TRADING SUMMARY")
        print("=" * 60)

        total_trades = 0
        total_wins = 0
        total_pnl = 0.0

        for trader in self.traders:
            stats = trader.positions.get_stats()
            print(f"\n{trader.symbol}:")
            print(f"  Trades: {stats['total_trades']} | "
                  f"Win Rate: {stats['win_rate']:.0f}% | "
                  f"P&L: {stats['total_pnl_pct']:+.2f}%")

            total_trades += stats['total_trades']
            total_wins += stats['wins']
            total_pnl += stats['total_pnl_pct']

        if total_trades > 0:
            overall_wr = (total_wins / total_trades) * 100
        else:
            overall_wr = 0

        print(f"\n{'='*60}")
        print(f"TOTAL: {total_trades} trades | "
              f"{overall_wr:.0f}% WR | "
              f"{total_pnl:+.2f}% P&L")
        print("=" * 60)

    def stop(self):
        """Stop all traders."""
        for trader in self.traders:
            trader.stop()


async def run_paper_trading(symbol: str = "BTCUSDT", **kwargs):
    """
    Convenience function to run paper trading for single symbol.

    Args:
        symbol: Trading pair
        **kwargs: Additional arguments for PaperTrader
    """
    trader = PaperTrader(symbol=symbol, **kwargs)

    try:
        await trader.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
        trader.stop()


async def run_multi_paper_trading(symbols: list = None, **kwargs):
    """
    Run paper trading for multiple symbols concurrently.

    Args:
        symbols: List of trading pairs (default: all from config)
        **kwargs: Additional arguments for each PaperTrader
    """
    trader = MultiPaperTrader(symbols=symbols, **kwargs)

    try:
        await trader.run()
    except KeyboardInterrupt:
        print("\nShutting down all traders...")
        trader.stop()


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Check for --all flag
    if "--all" in sys.argv:
        asyncio.run(run_multi_paper_trading())
    else:
        symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
        asyncio.run(run_paper_trading(symbol))
