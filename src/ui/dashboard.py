"""
Dashboard - Terminal tabanlı dashboard.

Rich kütüphanesi ile terminal UI.
"""

import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime

from loguru import logger

from src.data.fetcher import DataFetcher
from src.signals.generator import SignalGenerator
from src.positions.tracker import PositionTracker
from config.constants import CACHE_TTL_SECONDS


class Dashboard:
    """Terminal dashboard sınıfı."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        refresh_interval: int = CACHE_TTL_SECONDS,
    ):
        """
        Dashboard başlat.

        Args:
            symbols: İzlenecek semboller
            refresh_interval: Yenileme aralığı (saniye)
        """
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self.refresh_interval = refresh_interval

        self.fetcher = DataFetcher()
        self.signal_generator = SignalGenerator()
        self.position_tracker = PositionTracker()

        self._running = False
        self._data_cache: Dict[str, Any] = {}
        self._last_update: Optional[datetime] = None

    async def start(self) -> None:
        """Dashboard'u başlat."""
        self._running = True
        logger.info(f"Dashboard başlatıldı. Semboller: {self.symbols}")

        try:
            while self._running:
                await self._update()
                self._render()
                await asyncio.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            logger.info("Dashboard durduruluyor...")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Dashboard'u durdur."""
        self._running = False
        await self.fetcher.close()
        await self.signal_generator.close()
        logger.info("Dashboard durduruldu")

    async def _update(self) -> None:
        """Verileri güncelle."""
        for symbol in self.symbols:
            try:
                # Fiyat verisi
                ticker = await self.fetcher.binance.get_ticker(symbol)
                self._data_cache[f"{symbol}_ticker"] = ticker

                # Sinyal
                signal = await self.signal_generator.generate_signal(symbol)
                self._data_cache[f"{symbol}_signal"] = signal

            except Exception as e:
                logger.error(f"{symbol} güncelleme hatası: {e}")

        self._last_update = datetime.utcnow()

    def _render(self) -> None:
        """Terminal'e render et."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            from rich.layout import Layout
        except ImportError:
            self._render_simple()
            return

        console = Console()
        console.clear()

        # Header
        console.print(
            Panel(
                f"[bold cyan]AITrade Dashboard[/bold cyan]\n"
                f"Son Güncelleme: {self._last_update.strftime('%H:%M:%S') if self._last_update else 'N/A'}",
                title="",
            )
        )

        # Fiyat tablosu
        price_table = Table(title="Fiyatlar")
        price_table.add_column("Sembol", style="cyan")
        price_table.add_column("Fiyat", style="green")
        price_table.add_column("24h %", style="yellow")
        price_table.add_column("Hacim", style="blue")

        for symbol in self.symbols:
            ticker = self._data_cache.get(f"{symbol}_ticker")
            if ticker:
                change_color = "green" if ticker.price_change_percent > 0 else "red"
                price_table.add_row(
                    symbol,
                    f"${ticker.price:,.2f}",
                    f"[{change_color}]{ticker.price_change_percent:+.2f}%[/{change_color}]",
                    f"${ticker.quote_volume_24h:,.0f}",
                )

        console.print(price_table)

        # Sinyal tablosu
        signal_table = Table(title="Sinyaller")
        signal_table.add_column("Sembol", style="cyan")
        signal_table.add_column("Sinyal", style="bold")
        signal_table.add_column("Güven", style="yellow")
        signal_table.add_column("R/R", style="green")

        for symbol in self.symbols:
            signal = self._data_cache.get(f"{symbol}_signal")
            if signal:
                signal_color = (
                    "green" if signal.signal_type.value == "LONG"
                    else "red" if signal.signal_type.value == "SHORT"
                    else "white"
                )
                signal_table.add_row(
                    symbol,
                    f"[{signal_color}]{signal.signal_type.value}[/{signal_color}]",
                    f"{signal.confidence * 100:.0f}%",
                    f"1:{signal.risk_reward_ratio:.1f}",
                )

        console.print(signal_table)

        # Pozisyonlar
        open_positions = self.position_tracker.get_open_positions()
        if open_positions:
            pos_table = Table(title="Açık Pozisyonlar")
            pos_table.add_column("Sembol")
            pos_table.add_column("Yön")
            pos_table.add_column("Giriş")
            pos_table.add_column("PnL")

            for pos in open_positions:
                pnl_color = "green" if pos.pnl > 0 else "red"
                pos_table.add_row(
                    pos.symbol,
                    pos.side.value,
                    f"${pos.entry_price:,.2f}",
                    f"[{pnl_color}]{pos.pnl_percent:+.2f}%[/{pnl_color}]",
                )

            console.print(pos_table)

    def _render_simple(self) -> None:
        """Basit terminal render (rich olmadan)."""
        print("\n" + "=" * 50)
        print("AITrade Dashboard")
        print(f"Son Güncelleme: {self._last_update}")
        print("=" * 50)

        for symbol in self.symbols:
            ticker = self._data_cache.get(f"{symbol}_ticker")
            signal = self._data_cache.get(f"{symbol}_signal")

            if ticker:
                print(f"\n{symbol}")
                print(f"  Fiyat: ${ticker.price:,.2f}")
                print(f"  24h: {ticker.price_change_percent:+.2f}%")

            if signal:
                print(f"  Sinyal: {signal.signal_type.value}")
                print(f"  Güven: {signal.confidence * 100:.0f}%")

        print("\n" + "=" * 50)

    async def run_once(self) -> Dict[str, Any]:
        """
        Tek seferlik güncelleme ve sonuç döndür.

        Returns:
            Güncel veriler
        """
        await self._update()
        return self._data_cache
