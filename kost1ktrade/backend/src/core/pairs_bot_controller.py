import time
import pandas as pd
import threading
from typing import Dict, Any, Tuple

from src.core.bot_state import bot_state
from src.data_collector.collector import DataCollector
from src.trading.engine import TradingEngine
from src.core.config import settings
from src.strategies.pairs_trading_strategy import PairsTradingStrategy

# --- Pairs Bot State (Simplified for this controller) ---
# In a real system, this would be integrated more deeply into bot_state
pairs_bot_active = False
pairs_bot_stop_event = threading.Event()
current_pair: Tuple[str, str] = None
pairs_bot_engine: TradingEngine = None

def pairs_trading_loop():
    """
    The main trading loop for the pairs trading bot.
    Manages two assets simultaneously.
    """
    global pairs_bot_active, current_pair, pairs_bot_engine

    if not all([pairs_bot_active, current_pair, pairs_bot_engine]):
        print("FATAL in pairs thread: Bot not started correctly.")
        pairs_bot_active = False
        return

    symbol1, symbol2 = current_pair
    strategy = PairsTradingStrategy(window=20, z_threshold=2.0)
    collector = DataCollector(exchange_id='okx')

    # --- Configuration ---
    timeframe = settings.TIMEFRAME
    candle_limit = 100
    capital_per_trade = 1000  # Allocate $1000 to each side of the pair trade
    sleep_duration_seconds = 3600

    # --- State Management (FIX) ---
    in_position = False

    print(f"--- Background pairs trading loop started for {symbol1} / {symbol2} ---")

    while not pairs_bot_stop_event.is_set():
        try:
            print(f"({time.ctime()}) --- Pairs Bot Cycle for {symbol1}/{symbol2} ---")

            # 1. Fetch data for both symbols
            candles1 = collector.fetch_candles(symbol1, timeframe, limit=candle_limit)
            candles2 = collector.fetch_candles(symbol2, timeframe, limit=candle_limit)

            if not candles1 or not candles2 or len(candles1) < 50 or len(candles2) < 50:
                raise ValueError("Could not fetch enough candle data for both symbols.")

            df1 = pd.DataFrame(candles1, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])
            df2 = pd.DataFrame(candles2, columns=['open_time', 'open', 'high', 'low', 'close', 'volume'])

            # 2. Prepare data for the strategy
            merged_df = pd.merge(df1[['open_time', 'close']], df2[['open_time', 'close']], on='open_time', suffixes=('_asset1', '_asset2'))

            # 3. Generate stateless signals
            result_df = strategy.generate_signals(merged_df)
            latest_signal = result_df.iloc[-1]['signal']
            print(f"Pairs Strategy Signal: {latest_signal}")

            # 4. Execute trades based on state
            if not in_position:
                if latest_signal in ['BUY_PAIR', 'SELL_PAIR']:
                    print(f"Entering position based on signal: {latest_signal}")
                    price1 = df1.iloc[-1]['close']
                    price2 = df2.iloc[-1]['close']
                    amount1 = capital_per_trade / price1
                    amount2 = capital_per_trade / price2

                    if latest_signal == 'BUY_PAIR': # Buy asset 1, Sell asset 2
                        print(f"Executing BUY_PAIR: Buying {amount1:.4f} of {symbol1}, Selling {amount2:.4f} of {symbol2}")
                        pairs_bot_engine.create_order(symbol1, 'market', 'buy', amount1)
                        pairs_bot_engine.create_order(symbol2, 'market', 'sell', amount2)
                    elif latest_signal == 'SELL_PAIR': # Sell asset 1, Buy asset 2
                        print(f"Executing SELL_PAIR: Selling {amount1:.4f} of {symbol1}, Buying {amount2:.4f} of {symbol2}")
                        pairs_bot_engine.create_order(symbol1, 'market', 'sell', amount1)
                        pairs_bot_engine.create_order(symbol2, 'market', 'buy', amount2)

                    in_position = True # Update state

            elif in_position and latest_signal == 'CLOSE_PAIR':
                print("Executing CLOSE_PAIR: Closing all positions for the pair.")
                pairs_bot_engine.close_all_positions_for_symbol(symbol1)
                pairs_bot_engine.close_all_positions_for_symbol(symbol2)
                in_position = False # Update state

            print("Cycle complete. Sleeping...")
            pairs_bot_stop_event.wait(timeout=sleep_duration_seconds)

        except Exception as e:
            print(f"An error occurred in the pairs trading loop: {e}")
            pairs_bot_stop_event.wait(timeout=60)

    print("--- Background pairs trading loop has gracefully stopped ---")
    pairs_bot_active = False

def start_pairs_bot(mode: str, pair: Tuple[str, str]):
    """Starts the pairs trading bot."""
    global pairs_bot_active, current_pair, pairs_bot_engine, pairs_bot_stop_event
    if pairs_bot_active:
        raise ValueError("Pairs trading bot is already running.")

    print(f"Starting pairs trading bot for {pair[0]} / {pair[1]} in {mode} mode.")
    pairs_bot_stop_event.clear()
    current_pair = pair
    pairs_bot_engine = TradingEngine(mode=mode)
    pairs_bot_active = True

    # This would typically be started in a background thread from an API call
    # For now, this function just sets up the state.
    # e.g., background_tasks.add_task(pairs_trading_loop)

    return "Pairs bot state prepared. The loop needs to be started in a background task."

def stop_pairs_bot():
    """Stops the pairs trading bot."""
    global pairs_bot_active, pairs_bot_stop_event
    if not pairs_bot_active:
        raise ValueError("Pairs trading bot is not running.")

    print("Signaling pairs trading bot to stop...")
    pairs_bot_stop_event.set()
    return "Stop signal sent to pairs trading bot."
