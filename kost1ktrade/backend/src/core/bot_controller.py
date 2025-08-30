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

    try:
        while not bot_state.signal_bot_stop_event.is_set():
            try:
                print(f"({time.ctime()}) --- Signal Bot Cycle ---")

                # Full fetch-analyze-execute logic
                print(f"Fetching latest {strategy.long_window} candles for {symbol}...")
                candles_list = collector.fetch_candles(symbol, timeframe, limit=strategy.long_window)
                if not candles_list or len(candles_list) < strategy.long_window:
                    raise ValueError(f"Could not fetch enough candle data")

                candles_df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])

                print("Analyzing data and generating signal...")
                result_df = strategy.generate_signals(candles_df)
                latest_signal = result_df.iloc[-1]['signal']
                print(f"Strategy: {strategy.__class__.__name__} | Signal: {latest_signal}")

                base_currency, quote_currency = symbol.split('-')
                if latest_signal == 'BUY':
                    balance = engine.get_balance(quote_currency)
                    if balance and balance['free'] > 10:
                        amount_to_buy = balance['free'] / candles_df.iloc[-1]['close']
                        engine.create_order(symbol, 'market', 'buy', amount_to_buy)
                elif latest_signal == 'SELL':
                    balance = engine.get_balance(base_currency)
                    if balance and balance['free'] > 0.0001:
                        engine.create_order(symbol, 'market', 'sell', balance['free'])

                print(f"Cycle complete. Sleeping...")
                bot_state.signal_bot_stop_event.wait(timeout=sleep_duration_seconds)

            except Exception as e:
                print(f"An error occurred in the trading loop: {e}")
                bot_state.signal_bot_stop_event.wait(timeout=60)
    finally:
        print(f"--- Background signal bot loop for '{bot_state.signal_bot_mode}' mode has gracefully stopped ---")
        bot_state.signal_bot_mode = "stopped"
        bot_state.signal_bot_engine = None


def start_bot_loop(mode: str):
    """
    Prepares the state for the signal-based bot to be started in a background task.
    """
    if bot_state.signal_bot_mode != "stopped":
        raise ValueError("Signal bot is already running or starting.")

    bot_state.signal_bot_stop_event.clear()

    try:
        # Initialize engine first. If this fails, state is not set to running.
        bot_state.signal_bot_engine = TradingEngine(mode=mode)
        # Only set mode to running after successful initialization
        bot_state.signal_bot_mode = mode
        print(f"Signal bot state prepared for '{mode}' mode.")
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
