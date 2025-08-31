import time
import threading
from typing import Dict, Any
from src.core.bot_state import bot_state
from src.trading.engine import TradingEngine
from src.strategies.grid import GridStrategy

def grid_trading_loop(symbol: str):
    """
    The main loop for an individual grid trading bot, identified by its symbol.
    This function is intended to be run in a background task.
    """
    try:
        engine = bot_state.grid_bot_engines[symbol]
        config = bot_state.grid_bot_configs[symbol]
        stop_event = bot_state.grid_bot_stop_events[symbol]

        amount_per_grid = config['amount_per_grid']
        grid_config = {
            "grid_range_low": config['grid_range_low'],
            "grid_range_high": config['grid_range_high'],
            "num_grids": config['num_grids'],
        }
        strategy = GridStrategy(**grid_config)

        print(f"--- Background grid bot loop started for {symbol} ---")

        while not stop_event.is_set():
            try:
                print(f"({time.ctime()}) --- Grid Reconciliation Cycle for {symbol} ---")

                ideal_levels = strategy.generate_grid_levels()
                ticker = engine.fetch_ticker(symbol)
                if not ticker or 'last' not in ticker:
                    raise ValueError(f"Could not fetch a valid ticker for {symbol}")
                current_price = ticker['last']
                print(f"Current Price for {symbol}: ${current_price:,.2f}")

                open_orders = engine.fetch_open_orders(symbol)
                open_order_prices = {order['price'] for order in open_orders}
                print(f"Found {len(open_orders)} open orders for {symbol}.")

                required_buy_prices = {level for level in ideal_levels if level < current_price}
                required_sell_prices = {level for level in ideal_levels if level > current_price}

                for order in open_orders:
                    price = order['price']
                    side = order['side']
                    required = required_buy_prices if side == 'buy' else required_sell_prices
                    if price not in required:
                        print(f"Cancelling extraneous {side} order for {symbol} at ${price:,.2f}")
                        engine.cancel_order(order['id'], symbol)

                for price in required_buy_prices:
                    if price not in open_order_prices:
                        amount_to_buy = amount_per_grid / price
                        engine.create_order(symbol, 'limit', 'buy', amount_to_buy, price)

                for price in required_sell_prices:
                    if price not in open_order_prices:
                        amount_to_sell = amount_per_grid / price
                        engine.create_order(symbol, 'limit', 'sell', amount_to_sell, price)

                print(f"Grid reconciliation cycle complete for {symbol}.")
                stop_event.wait(timeout=30)

            except Exception as e:
                print(f"An error occurred in the grid trading loop for {symbol}: {e}")
                stop_event.wait(timeout=60)
    finally:
        print(f"--- Background grid bot loop for {symbol} stopping... ---")
        if symbol in bot_state.grid_bot_engines:
            print(f"Cleaning up all open grid orders for {symbol}...")
            bot_state.grid_bot_engines[symbol].cancel_all_orders(symbol)

        # Clean up state for this specific bot
        bot_state.grid_bot_states.pop(symbol, None)
        bot_state.grid_bot_engines.pop(symbol, None)
        bot_state.grid_bot_configs.pop(symbol, None)
        bot_state.grid_bot_threads.pop(symbol, None)
        bot_state.grid_bot_stop_events.pop(symbol, None)
        print(f"State for {symbol} has been cleaned up.")


def start_grid_bot(symbol: str, mode: str, config: Dict[str, Any]):
    """Prepares and starts a grid bot for a specific symbol."""
    if bot_state.grid_bot_states.get(symbol) == "running":
        raise ValueError(f"Grid bot for {symbol} is already running.")

    bot_state.grid_bot_configs[symbol] = config
    bot_state.grid_bot_states[symbol] = "running"
    bot_state.grid_bot_stop_events[symbol] = threading.Event()

    try:
        bot_state.grid_bot_engines[symbol] = TradingEngine(mode=mode)
        thread = threading.Thread(target=grid_trading_loop, args=(symbol,))
        bot_state.grid_bot_threads[symbol] = thread
        thread.start()
        print(f"Grid bot for {symbol} started in '{mode}' mode.")
    except Exception as e:
        bot_state.grid_bot_states[symbol] = "stopped"
        raise e

def stop_grid_bot(symbol: str):
    """Signals a specific grid trading bot to stop."""
    if bot_state.grid_bot_states.get(symbol) != "running":
        raise ValueError(f"Grid bot for {symbol} is not running.")

    print(f"Signaling grid bot for {symbol} to stop...")
    stop_event = bot_state.grid_bot_stop_events.get(symbol)
    if stop_event:
        stop_event.set()

    thread = bot_state.grid_bot_threads.get(symbol)
    if thread:
        thread.join(timeout=60) # Wait for the thread to finish
        if thread.is_alive():
            print(f"Warning: Grid bot thread for {symbol} did not stop in time.")

    return f"Stop signal sent to grid bot for {symbol}."

def stop_all_grid_bots():
    """Stops all currently running grid bots."""
    running_bots = list(bot_state.grid_bot_states.keys())
    if not running_bots:
        return "No grid bots are currently running."

    print(f"Stopping all running grid bots: {running_bots}")
    for symbol in running_bots:
        try:
            stop_grid_bot(symbol)
        except ValueError as e:
            print(f"Could not stop bot {symbol}: {e}") # Already stopped or state inconsistent
    return "Stop signal sent to all running grid bots."
