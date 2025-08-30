import time
import pandas as pd
from src.core.bot_state import bot_state
from src.data_collector.collector import DataCollector
from src.trading.engine import TradingEngine
from src.strategies.sma_crossover import SmaCrossoverStrategy
from src.core.config import settings

def signal_trading_loop():
    """
    The main loop for the signal-based trading bot.
    This function is intended to be run in a background task.
    """
    # This check is important because the background task starts after the response is sent
    if bot_state.signal_bot_mode == "stopped":
        print("Bot was stopped before the trading loop could start.")
        return

    engine = bot_state.signal_bot_engine
    if not engine:
        print("FATAL in thread: Trading engine not available in bot_state.")
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
        # ... [Full fetch-analyze-execute logic will be restored here] ...
        print(f"({time.ctime()}) --- Signal Bot Cycle ---")

        # Placeholder for the full logic
        print("Fetching, analyzing, executing...")

        bot_state.signal_bot_stop_event.wait(timeout=sleep_duration_seconds)

    print(f"--- Background signal bot loop for '{bot_state.signal_bot_mode}' mode has gracefully stopped ---")
    # Reset state after the loop finishes
    bot_state.signal_bot_mode = "stopped"
    bot_state.signal_bot_engine = None


def start_bot_loop(mode: str):
    """
    Prepares the state for the signal-based bot to be started in a background task.
    """
    if bot_state.signal_bot_mode != "stopped":
        raise ValueError("Signal bot is already running or starting.")

    print(f"Preparing to start signal bot in '{mode}' mode.")
    bot_state.signal_bot_mode = mode
    bot_state.signal_bot_stop_event.clear()

    try:
        bot_state.signal_bot_engine = TradingEngine(mode=mode)
    except Exception as e:
        bot_state.signal_bot_mode = "stopped" # Reset on failure
        raise e

def stop_bot_loop():
    """
    Signals the background signal-based trading loop to stop.
    """
    if bot_state.signal_bot_mode == "stopped":
        raise ValueError("Signal bot is not running.")

    print("Signaling signal bot to stop...")
    bot_state.signal_bot_stop_event.set()
    # The background task will see this event and exit its loop.
    # The state will be fully reset by the loop itself upon exit.
    return "Stop signal sent. The bot will shut down after its current cycle."
