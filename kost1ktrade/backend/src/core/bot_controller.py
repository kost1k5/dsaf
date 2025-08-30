import time
import pandas as pd
import importlib
from typing import Dict, Any
from src.core.bot_state import bot_state
from src.data_collector.collector import DataCollector
from src.trading.engine import TradingEngine
from src.core.config import settings

def _get_strategy_class(strategy_name: str):
    """Dynamically imports and returns a strategy class."""
    try:
        module_name = f"src.strategies.{strategy_name.lower()}"
        # Convention: a file like 'macd.py' contains a class named 'MacdStrategy'
        class_name = f"{strategy_name.replace('_', ' ').title().replace(' ', '')}Strategy"
        strategy_module = importlib.import_module(module_name)
        return getattr(strategy_module, class_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not find strategy '{strategy_name}'. "
                         f"Ensure '{module_name}.py' and class '{class_name}' exist.") from e

def signal_trading_loop():
    """
    The main loop for the signal-based trading bot.
    This function is intended to be run in a background task.
    """
    if bot_state.signal_bot_mode == "stopped":
        print("Bot was stopped before the trading loop could start.")
        return

    engine = bot_state.signal_bot_engine
    strategy = bot_state.signal_bot_strategy
    if not engine or not strategy:
        print("FATAL in thread: Trading engine or strategy not available in bot_state.")
        bot_state.signal_bot_mode = "stopped"
        return

    try:
        collector = DataCollector(exchange_id='okx')
    except Exception as e:
        print(f"FATAL in thread: Failed to initialize DataCollector: {e}")
        bot_state.signal_bot_mode = "stopped"
        return

    # TODO: These should be parameterized and passed during bot startup
    symbol = settings.SYMBOLS[0]
    timeframe = '1h'
    sleep_duration_seconds = 3600
    # Use a generic candle limit, as strategies have different requirements
    candle_limit = 200

    print(f"--- Background signal bot loop started for {strategy.__class__.__name__} on {symbol} ---")

    try:
        while not bot_state.signal_bot_stop_event.is_set():
            try:
                print(f"({time.ctime()}) --- Signal Bot Cycle for {strategy.__class__.__name__} ---")

                print(f"Fetching latest {candle_limit} candles for {symbol}...")
                candles_list = collector.fetch_candles(symbol, timeframe, limit=candle_limit)
                if not candles_list or len(candles_list) < 50: # Basic check for some data
                    raise ValueError(f"Could not fetch enough candle data (got {len(candles_list)})")

                candles_df = pd.DataFrame(candles_list, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])

                print("Analyzing data and generating signal...")
                result_df = strategy.generate_signals(candles_df)
                latest_signal = result_df.iloc[-1]['signal']
                print(f"Strategy: {strategy.__class__.__name__} | Signal: {latest_signal}")

                base_currency, quote_currency = symbol.split('-')
                if latest_signal == 'BUY':
                    balance = engine.get_balance()
                    quote_balance = balance.get(quote_currency, 0)
                    if quote_balance > 10: # Min order size check
                        amount_to_buy = quote_balance / candles_df.iloc[-1]['close']
                        engine.create_order(symbol, 'market', 'buy', amount_to_buy)
                elif latest_signal == 'SELL':
                    balance = engine.get_balance()
                    base_balance = balance.get(base_currency, 0)
                    if base_balance > 0.0001:
                        engine.create_order(symbol, 'market', 'sell', base_balance)

                print(f"Cycle complete. Sleeping...")
                bot_state.signal_bot_stop_event.wait(timeout=sleep_duration_seconds)

            except Exception as e:
                print(f"An error occurred in the trading loop: {e}")
                bot_state.signal_bot_stop_event.wait(timeout=60)
    finally:
        print(f"--- Background signal bot loop for '{bot_state.signal_bot_mode}' mode has gracefully stopped ---")
        bot_state.signal_bot_mode = "stopped"
        bot_state.signal_bot_engine = None
        bot_state.signal_bot_strategy = None


def start_bot_loop(mode: str, strategy_name: str, strategy_params: Dict[str, Any]):
    """
    Prepares the state for the signal-based bot to be started in a background task.
    """
    if bot_state.signal_bot_mode != "stopped":
        raise ValueError("Signal bot is already running or starting.")

    bot_state.signal_bot_stop_event.clear()

    try:
        # Initialize strategy first
        StrategyClass = _get_strategy_class(strategy_name)
        bot_state.signal_bot_strategy = StrategyClass(**strategy_params)

        # Initialize engine
        bot_state.signal_bot_engine = TradingEngine(mode=mode)

        # Set mode to running only after successful initialization
        bot_state.signal_bot_mode = mode
        print(f"Signal bot state prepared for '{mode}' mode with strategy '{strategy_name}'.")
    except (ValueError, TypeError, ConnectionError) as e:
        # Reset state on failure
        bot_state.signal_bot_mode = "stopped"
        bot_state.signal_bot_strategy = None
        bot_state.signal_bot_engine = None
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
