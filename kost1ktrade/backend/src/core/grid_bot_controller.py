import time
import threading
import numpy as np
from src.core.bot_state import bot_state
from src.trading.engine import TradingEngine
from src.strategies.grid import GridStrategy
from src.core.config import settings

def grid_trading_loop(symbol: str, grid_config: dict, amount_per_grid: float):
    """The main loop for the grid trading bot."""

    engine = bot_state.grid_bot_engine
    strategy = GridStrategy(**grid_config)

    print(f"--- Background grid bot loop started for {symbol} ---")

    while not bot_state.grid_bot_stop_event.is_set():
        try:
            print(f"({time.ctime()}) --- Grid Bot Cycle for {symbol} ---")

            # --- Main Grid Logic ---

            # 1. Get ideal grid levels and current market price
            ideal_levels = strategy.generate_grid_levels()
            ticker = engine.fetch_ticker(symbol)
            if not ticker or 'last' not in ticker:
                raise ValueError(f"Could not fetch a valid ticker for {symbol}")
            current_price = ticker['last']
            print(f"Current price for {symbol} is ${current_price:,.2f}")

            # 2. Get currently open orders
            open_orders = engine.fetch_open_orders(symbol)
            open_order_prices = {order['price'] for order in open_orders}
            print(f"Found {len(open_orders)} open orders.")

            # 3. Reconcile orders
            # Place buy orders below current price and sell orders above
            for level in ideal_levels:
                if level in open_order_prices:
                    continue # Order already exists at this level

                if level < current_price:
                    # Place a BUY limit order
                    print(f"Placing BUY limit order at ${level:,.2f}")
                    engine.create_order(symbol, 'limit', 'buy', amount_per_grid, level)
                elif level > current_price:
                    # Place a SELL limit order
                    print(f"Placing SELL limit order at ${level:,.2f}")
                    engine.create_order(symbol, 'limit', 'sell', amount_per_grid, level)

            print("Grid reconciliation complete.")
            bot_state.grid_bot_stop_event.wait(timeout=60) # Run reconciliation every minute

        except Exception as e:
            print(f"An error occurred in the grid trading loop: {e}")
            bot_state.grid_bot_stop_event.wait(timeout=60)

    print(f"--- Background grid bot loop stopped ---")
    # Clean up orders on stop
    if bot_state.grid_bot_engine:
        print("Cleaning up open grid orders...")
        bot_state.grid_bot_engine.cancel_all_orders(symbol)


def start_grid_bot(mode: str, symbol: str, grid_config: dict, amount_per_grid: float):
    """Initializes and starts the grid bot in a background thread."""
    if bot_state.grid_bot_thread and bot_state.grid_bot_thread.is_alive():
        raise ValueError("Grid bot is already running.")

    bot_state.grid_bot_mode = mode
    bot_state.grid_bot_stop_event.clear()

    try:
        bot_state.grid_bot_engine = TradingEngine(mode=mode)
    except Exception as e:
        bot_state.grid_bot_mode = "stopped"
        raise e

    thread_args = (symbol, grid_config, amount_per_grid)
    bot_state.grid_bot_thread = threading.Thread(target=grid_trading_loop, args=thread_args, daemon=True)
    bot_state.grid_bot_thread.start()
    print(f"Grid bot thread started for {symbol} in '{mode}' mode.")

def stop_grid_bot():
    """Stops the background grid trading loop."""
    if not (bot_state.grid_bot_thread and bot_state.grid_bot_thread.is_alive()):
        raise ValueError("Grid bot is not running.")

    print("Stopping grid bot thread...")
    bot_state.grid_bot_stop_event.set()
    bot_state.grid_bot_thread.join(timeout=10)

    bot_state.grid_bot_mode = "stopped"
    bot_state.grid_bot_engine = None
    bot_state.grid_bot_thread = None
    print("Grid bot stopped.")
