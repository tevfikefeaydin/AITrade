"""
Live Trading Module for ML-Assisted Crypto Trading.

This module provides paper trading capabilities with:
- Real-time data via Binance WebSocket
- Live feature computation matching backtest logic
- ML model inference for trade filtering
- Position management with TP/SL/Timeout barriers
"""

from .paper_trader import PaperTrader, MultiPaperTrader
from .websocket_client import BinanceWebSocket
from .feature_buffer import FeatureBuffer
from .position_manager import PositionManager

__all__ = [
    "PaperTrader",
    "MultiPaperTrader",
    "BinanceWebSocket",
    "FeatureBuffer",
    "PositionManager",
]
