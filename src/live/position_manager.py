"""
Position Manager for Paper Trading.

Tracks open positions, manages trade lifecycle,
and logs completed trades to JSON file.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages paper trading positions and trade history.

    Features:
    - Track open positions with TP/SL/Timeout barriers
    - Log completed trades to JSON file
    - Compute running P&L statistics
    """

    def __init__(self, log_path: str = "outputs/paper_trades.json"):
        """
        Initialize position manager.

        Args:
            log_path: Path to JSON file for trade history
        """
        self.log_path = Path(log_path)
        self.positions: List[Dict] = []
        self._file_lock = threading.Lock()

        # Statistics
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self._load_history()

    def _load_history(self):
        """Load trade history and restore OPEN positions from file."""
        if self.log_path.exists():
            try:
                history = json.loads(self.log_path.read_text())
                # Count stats from closed trades
                for trade in history:
                    if trade.get("status") == "CLOSED":
                        self.trade_count += 1
                        pnl = trade.get("pnl_pct", 0)
                        self.total_pnl += pnl
                        if pnl > 0:
                            self.win_count += 1
                # Restore OPEN positions so restart doesn't lose them
                open_count = 0
                seen_entry_times = set()
                for trade in history:
                    if trade.get("status") == "OPEN":
                        for key in ["entry_time", "exit_time", "max_exit_time", "signal_time"]:
                            if key in trade and isinstance(trade[key], str):
                                try:
                                    dt = datetime.fromisoformat(trade[key])
                                    if dt.tzinfo is None:
                                        dt = dt.replace(tzinfo=timezone.utc)
                                    trade[key] = dt
                                except (ValueError, TypeError):
                                    pass
                        # Deduplicate by (symbol, entry_time) to prevent double-restore
                        dedup_key = (trade.get("symbol"), str(trade.get("entry_time")))
                        if dedup_key in seen_entry_times:
                            logger.warning(f"Skipping duplicate OPEN position: {dedup_key}")
                            continue
                        seen_entry_times.add(dedup_key)
                        self.positions.append(trade)
                        open_count += 1
                        logger.info(f"Restored OPEN position: {trade.get('symbol')}")
                logger.info(
                    f"Loaded {self.trade_count} closed + {open_count} open trades from {self.log_path}"
                )
            except Exception as e:
                logger.warning(f"Could not load trade history: {e}")

    def add(self, position: Dict):
        """
        Add a new position.

        Args:
            position: Dict with position details:
                - symbol: Trading pair
                - entry_time: Entry timestamp
                - entry_price: Entry price (with costs)
                - tp_price: Take profit price
                - sl_price: Stop loss price
                - max_exit_time: Timeout timestamp
                - probability: ML probability score
                - status: "OPEN"
        """
        self.positions.append(position)
        self.log_trade(position)
        logger.info(f"Position opened: {position['symbol']} at ${position['entry_price']:.2f}")

    def get_open(self) -> List[Dict]:
        """Get list of open positions."""
        return [p for p in self.positions if p.get("status") == "OPEN"]

    def get_closed(self) -> List[Dict]:
        """Get list of closed positions."""
        return [p for p in self.positions if p.get("status") == "CLOSED"]

    @staticmethod
    def _parse_dt(value):
        """Parse datetime-like values for robust sorting."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                return None
        return None

    def get_recent_closed(self, n: int) -> List[Dict]:
        """Return the most recent n closed positions ordered by exit time."""
        if n <= 0:
            return []

        closed = self.get_closed()
        if not closed:
            return []

        def sort_key(pos: Dict):
            exit_time = self._parse_dt(pos.get("exit_time"))
            if exit_time is not None:
                return exit_time
            entry_time = self._parse_dt(pos.get("entry_time"))
            if entry_time is not None:
                return entry_time
            return datetime.min.replace(tzinfo=timezone.utc)

        closed_sorted = sorted(closed, key=sort_key)
        return closed_sorted[-n:]

    def get_consecutive_sl_streak(self) -> int:
        """Count consecutive stop-loss outcomes from most recent closed trades."""
        streak = 0
        for trade in reversed(self.get_recent_closed(self.trade_count)):
            if trade.get("exit_reason") == "SL":
                streak += 1
            else:
                break
        return streak

    def get_recent_win_rate(self, n: int) -> float:
        """Compute win rate over recent closed trades using pnl_pct > 0."""
        recent = self.get_recent_closed(n)
        if not recent:
            return 0.0
        wins = sum(1 for t in recent if t.get("pnl_pct", 0) > 0)
        return wins / len(recent)

    def has_open_position(self, symbol: str) -> bool:
        """Check if there's an open position for a symbol."""
        return any(
            p.get("symbol") == symbol and p.get("status") == "OPEN"
            for p in self.positions
        )

    def close_position(
        self,
        position: Dict,
        reason: str,
        exit_price: float,
        exit_time: Optional[datetime] = None
    ) -> Dict:
        """
        Close a position and calculate P&L.

        Args:
            position: Position dict to close
            reason: Exit reason ("TP", "SL", "TIMEOUT")
            exit_price: Exit price (before costs)
            exit_time: Exit timestamp (default: now)

        Returns:
            Updated position dict with exit details
        """
        if exit_time is None:
            exit_time = datetime.now(timezone.utc)

        # Calculate P&L
        entry_price = position["entry_price"]
        pnl_pct = (exit_price - entry_price) / entry_price * 100

        # Update position
        position["exit_time"] = exit_time
        position["exit_price"] = exit_price
        position["exit_reason"] = reason
        position["pnl_pct"] = pnl_pct
        position["status"] = "CLOSED"

        # Update statistics
        self.trade_count += 1
        self.total_pnl += pnl_pct
        if pnl_pct > 0:
            self.win_count += 1

        # Log to file
        self.log_trade(position)

        return position

    def log_trade(self, position: Dict):
        """
        Log trade to JSON file (both OPEN and CLOSED).

        Updates existing OPEN entry in-place when closing,
        or appends new entry when opening.

        Args:
            position: Position dict (OPEN or CLOSED)
        """
        # Create copy with serializable datetime
        pos_copy = position.copy()
        for key in ["entry_time", "exit_time", "max_exit_time", "signal_time"]:
            if key in pos_copy and pos_copy[key] is not None:
                if isinstance(pos_copy[key], datetime):
                    pos_copy[key] = pos_copy[key].isoformat()

        with self._file_lock:
            # Load existing history
            history = []
            if self.log_path.exists():
                try:
                    history = json.loads(self.log_path.read_text())
                except Exception:
                    pass

            # Update existing OPEN entry or append new one
            updated = False
            for i, trade in enumerate(history):
                if (trade.get("symbol") == pos_copy.get("symbol") and
                        trade.get("entry_time") == pos_copy.get("entry_time") and
                        trade.get("status") == "OPEN"):
                    history[i] = pos_copy
                    updated = True
                    break

            if not updated:
                history.append(pos_copy)

            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text(json.dumps(history, indent=2))

        logger.debug(f"Trade logged to {self.log_path}")

    def get_stats(self) -> Dict:
        """
        Get trading statistics.

        Returns:
            Dict with performance metrics
        """
        win_rate = (self.win_count / self.trade_count * 100) if self.trade_count > 0 else 0

        return {
            "total_trades": self.trade_count,
            "wins": self.win_count,
            "losses": self.trade_count - self.win_count,
            "win_rate": win_rate,
            "total_pnl_pct": self.total_pnl,
            "avg_pnl_pct": self.total_pnl / self.trade_count if self.trade_count > 0 else 0,
        }

    def print_summary(self):
        """Print trading summary to console."""
        stats = self.get_stats()

        print("\n" + "=" * 50)
        print("PAPER TRADING SUMMARY")
        print("=" * 50)
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Wins:         {stats['wins']}")
        print(f"  Losses:       {stats['losses']}")
        print(f"  Win Rate:     {stats['win_rate']:.1f}%")
        print(f"  Total P&L:    {stats['total_pnl_pct']:+.2f}%")
        print(f"  Avg P&L:      {stats['avg_pnl_pct']:+.2f}%")
        print("=" * 50)

        # Recent trades
        closed = self.get_closed()
        if closed:
            print("\nRecent Trades:")
            for trade in closed[-5:]:
                icon = "[WIN]" if trade.get("pnl_pct", 0) > 0 else "[LOSS]"
                print(
                    f"  {icon} {trade.get('symbol')} "
                    f"{trade.get('exit_reason')} "
                    f"P&L: {trade.get('pnl_pct', 0):+.2f}%"
                )

    def clear_history(self):
        """Clear trade history (use with caution)."""
        self.positions = []
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0

        if self.log_path.exists():
            self.log_path.unlink()

        logger.info("Trade history cleared")
