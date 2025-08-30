import time
import threading
import pandas as pd
from src.core.bot_state import bot_state
from src.data_collector.collector import DataCollector
from src.trading.engine import TradingEngine
from src.strategies.sma_crossover import SmaCrossoverStrategy
from src.core.config import settings

def signal_trading_loop():
    """The main loop for the signal-based trading bot."""

    engine = bot_state.signal_bot_engine
    if not engine:
        print("FATAL in thread: Trading engine not initialized.")
        bot_state.signal_bot_mode = "stopped"
        return

    try:
        collector = DataCollector(exchange_id='okx')
    except Exception as e:
        print(f"FATAL in thread: Failed to initialize DataCollector: {e}")
        bot_state.signal_bot_mode = "stopped"
        return

    strategy = SmaCrossoverStrategy(
        short_window=settings.INDICATORS.SMA_PERIOD,
        long_window=settings.INDICATORS.SMA_LONG_PERIOD
    )
    symbol = settings.SYMBOLS[0]
    timeframe = '1h'
    sleep_duration_seconds = 3600

    print(f"--- Background signal bot loop started in '{bot_state.signal_bot_mode}' mode for {symbol} ---")

    while not bot_state.signal_bot_stop_event.is_set():
        # ... [fetch-analyze-execute logic] ...
        print(f"({time.ctime()}) --- Signal Bot Cycle ---")
        time.sleep(1) # Placeholder for real logic
        bot_state.signal_bot_stop_event.wait(timeout=sleep_duration_seconds)

    print(f"--- Background signal bot loop stopped ---")


def start_bot_loop(mode: str):
    """Initializes and starts the signal-based bot in a background thread."""
    if bot_state.signal_bot_thread and bot_state.signal_bot_thread.is_alive():
        raise ValueError("Signal bot is already running.")

    bot_state.signal_bot_mode = mode
    bot_state.signal_bot_stop_event.clear()

    try:
        bot_state.signal_bot_engine = TradingEngine(mode=mode)
    except Exception as e:
        bot_state.signal_bot_mode = "stopped"
        raise e

    bot_state.signal_bot_thread = threading.Thread(target=signal_trading_loop, daemon=True)
    bot_state.signal_bot_thread.start()

def stop_bot_loop():
    """Stops the background signal-based trading loop."""
    if not (bot_state.signal_bot_thread and bot_state.signal_bot_thread.is_alive()):
        raise ValueError("Signal bot is not running.")

    bot_state.signal_bot_stop_event.set()
    bot_state.signal_bot_thread.join(timeout=10)
    bot_state.signal_bot_mode = "stopped"
    bot_state.signal_bot_engine = None
    bot_state.signal_bot_thread = None
