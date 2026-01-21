"""
Telegram bildirim servisi.
"""

from typing import Optional

import httpx
from loguru import logger

from config.settings import get_settings
from config.constants import API_RETRY_COUNT, API_RETRY_DELAY


class TelegramNotifier:
    """Telegram bot ile bildirim gönderici."""

    BASE_URL = "https://api.telegram.org/bot{token}"

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        """
        TelegramNotifier başlat.

        Args:
            token: Bot token (None ise .env'den alınır)
            chat_id: Chat ID (None ise .env'den alınır)
        """
        settings = get_settings()
        self.token = token or settings.telegram_bot_token
        self.chat_id = chat_id or settings.telegram_chat_id
        self.base_url = self.BASE_URL.format(token=self.token)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def is_configured(self) -> bool:
        """Telegram yapılandırılmış mı?"""
        return bool(self.token and self.chat_id)

    async def _get_client(self) -> httpx.AsyncClient:
        """HTTP client döndür."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        """Client kapat."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def send_message(
        self,
        text: str,
        parse_mode: str = "Markdown",
        disable_notification: bool = False,
    ) -> bool:
        """
        Mesaj gönder.

        Args:
            text: Mesaj metni
            parse_mode: Parse modu (Markdown, HTML)
            disable_notification: Sessiz bildirim

        Returns:
            Başarı durumu
        """
        if not self.is_configured:
            logger.warning("Telegram yapılandırılmamış")
            return False

        client = await self._get_client()
        url = f"{self.base_url}/sendMessage"

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification,
        }

        for attempt in range(API_RETRY_COUNT):
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

                if data.get("ok"):
                    logger.debug("Telegram mesajı gönderildi")
                    return True
                else:
                    logger.error(f"Telegram API hatası: {data}")
                    return False

            except httpx.HTTPError as e:
                logger.warning(
                    f"Telegram gönderim hatası (deneme {attempt + 1}): {e}"
                )
                if attempt == API_RETRY_COUNT - 1:
                    return False
                import asyncio
                await asyncio.sleep(API_RETRY_DELAY)

        return False

    async def send_signal(self, signal) -> bool:
        """
        Trading sinyali gönder.

        Args:
            signal: Signal objesi

        Returns:
            Başarı durumu
        """
        message = signal.to_message()
        return await self.send_message(message)

    async def send_alert(
        self,
        title: str,
        message: str,
        level: str = "INFO",
    ) -> bool:
        """
        Alert gönder.

        Args:
            title: Alert başlığı
            message: Alert mesajı
            level: Seviye (INFO, WARNING, ERROR)

        Returns:
            Başarı durumu
        """
        emoji_map = {
            "INFO": "",
            "WARNING": "",
            "ERROR": "",
            "SUCCESS": "",
        }
        emoji = emoji_map.get(level, "")

        text = f"""
{emoji} **{title}**

{message}
"""
        return await self.send_message(text.strip())

    async def send_daily_report(
        self,
        stats: dict,
    ) -> bool:
        """
        Günlük rapor gönder.

        Args:
            stats: İstatistik dict'i

        Returns:
            Başarı durumu
        """
        text = f"""
 **Günlük Rapor**

**Sinyal Sayısı**: {stats.get('total_signals', 0)}
**Kazanan**: {stats.get('wins', 0)}
**Kaybeden**: {stats.get('losses', 0)}
**Win Rate**: {stats.get('win_rate', 0):.1f}%

**Toplam PnL**: {stats.get('total_pnl', 0):.2f}%
**En İyi Trade**: {stats.get('best_trade', 0):.2f}%
**En Kötü Trade**: {stats.get('worst_trade', 0):.2f}%
"""
        return await self.send_message(text.strip())

    async def test_connection(self) -> bool:
        """
        Bağlantı testi yap.

        Returns:
            Bağlantı başarılı mı
        """
        if not self.is_configured:
            return False

        client = await self._get_client()
        url = f"{self.base_url}/getMe"

        try:
            response = await client.get(url)
            data = response.json()
            if data.get("ok"):
                bot_name = data["result"].get("username", "Unknown")
                logger.info(f"Telegram bağlantısı başarılı: @{bot_name}")
                return True
        except Exception as e:
            logger.error(f"Telegram bağlantı hatası: {e}")

        return False
