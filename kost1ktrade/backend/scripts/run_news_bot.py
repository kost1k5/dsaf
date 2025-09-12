import sys
import os
import time
import ccxt

# Adjust the path to allow imports from the 'src' directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.news_websocket_client import NewsWebsocketClient
from src.core.config import settings

def main():
    """
    Main function to initialize and run the news trading WebSocket bot.
    """
    print("--- Initializing News Trading Bot ---")

    # 1. Load configuration from the central settings object
    try:
        if not settings.OKX_REAL:
            print("FATAL: OKX_REAL credentials not found in your .env file or configuration.")
            sys.exit(1)

        trade_config = {
            "instrument": settings.COMMANDER_SYMBOL, # Use the globally defined symbol
            "target_events": ["CPI", "FOMC", "Non-Farm Payrolls"],
            "risk_percent": settings.RISK.RISK_PER_TRADE_PCT,
            "atr_timeframe": "5m", # Or make this configurable in settings too
            "atr_multiplier": settings.RISK.ATR_MULTIPLIER,
            "time_exit_minutes": 30 # Or make configurable
        }
        print("Bot configuration loaded successfully.")
        print(trade_config)
    except Exception as e:
        print(f"FATAL: Could not load configuration from settings. Error: {e}")
        sys.exit(1)

    # 2. Initialize the authenticated CCXT exchange instance
    try:
        exchange = ccxt.okx({
            'apiKey': settings.OKX_REAL.API_KEY,
            'secret': settings.OKX_REAL.SECRET_KEY,
            'password': settings.OKX_REAL.PASSPHRASE,
            'options': {
                'defaultType': 'swap',
            },
        })
        # IMPORTANT: Set to sandbox mode for testing. Change to False for live trading.
        exchange.set_sandbox_mode(True)
        print("CCXT exchange instance created in SANDBOX/DEMO mode.")
        # Verify connection
        exchange.fetch_balance()
        print("Successfully connected to the exchange and fetched balance.")
    except Exception as e:
        print(f"FATAL: Could not initialize CCXT exchange. Check API keys and connection. Error: {e}")
        sys.exit(1)

    # 3. Create and start the WebSocket client
    client = NewsWebsocketClient(exchange=exchange, config=trade_config)
    client.start()

    print("\n--- News Trading Bot is now running ---")
    print("Listening for economic calendar events. Press Ctrl+C to stop.")

    try:
        # Keep the main thread alive while the client runs in the background
        while client.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutdown signal received.")
    finally:
        print("Stopping bot...")
        client.stop()
        print("Bot has been shut down gracefully.")

if __name__ == '__main__':
    main()
