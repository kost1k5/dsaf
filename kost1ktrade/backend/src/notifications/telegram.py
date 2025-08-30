import asyncio
import telegram
from src.core.config import settings

class TelegramNotifier:
    """
    A class to handle sending notifications to a Telegram chat.
    """
    def __init__(self):
        """
        Initializes the notifier. Raises ValueError if credentials are not set.
        """
        if not settings.TELEGRAM_TOKEN or not settings.TELEGRAM_CHAT_ID:
            self.bot = None
            self.chat_id = None
            print("Warning: Telegram notifications are not configured. Skipping initialization.")
            return

        self.bot = telegram.Bot(token=settings.TELEGRAM_TOKEN)
        self.chat_id = settings.TELEGRAM_CHAT_ID

    async def send_message(self, message: str):
        """
        Sends a message to the configured Telegram chat.
        """
        if not self.bot:
            print(f"Telegram disabled. Would have sent: {message}")
            return

        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='Markdown')
            print(f"Sent Telegram notification.")
        except Exception as e:
            print(f"Failed to send Telegram notification: {e}")

# Example usage
async def main():
    print("--- Telegram Notifier Demonstration ---")
    print("NOTE: This script requires a valid TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in a .env file to run.")

    notifier = TelegramNotifier()
    if notifier.bot:
        await notifier.send_message("Hello from *Kost1kTrade* bot! This is a test message.")
    else:
        print("Notifier not initialized due to missing configuration.")

if __name__ == '__main__':
    asyncio.run(main())
