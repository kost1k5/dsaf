import time
import threading
from typing import Dict, Any

from src.core.bot_state import bot_state
from src.trading.engine import TradingEngine
from src.strategies.grid import GridStrategy

def _run_single_grid_loop(symbol: str, config: Dict[str, Any], stop_event: threading.Event):
    """
    The main trading loop for a single grid bot instance.
    This function runs in a dedicated background thread for one symbol.
    """
    engine = bot_state.grid_bots_engine
    if not engine:
        print(f"FATAL: Grid bot for {symbol} stopping because trading engine is not available.")
        return

    amount_per_grid = config['amount_per_grid']
    grid_config = {
        "grid_range_low": config['grid_range_low'],
        "grid_range_high": config['grid_range_high'],
        "num_grids": config['num_grids'],
    }
    strategy = GridStrategy(**grid_config)

    print(f"--- Background grid bot loop started for {symbol} ---")

    try:
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

                # --- Reconciliation Logic ---
                required_buy_prices = {level for level in ideal_levels if level < current_price}
                required_sell_prices = {level for level in ideal_levels if level > current_price}

                # Cancel extraneous orders
                for order in open_orders:
                    price = order['price']
                    side = order['side']
                    required = required_buy_prices if side == 'buy' else required_sell_prices
                    if price not in required:
                        print(f"Cancelling extraneous {side} order for {symbol} at ${price:,.2f}")
                        engine.cancel_order(order['id'], symbol)

                # Place missing orders
                for price in required_buy_prices:
                    if price not in open_order_prices:
                        amount_to_buy = amount_per_grid / price
                        engine.create_order(symbol, 'limit', 'buy', amount_to_buy, price)

                for price in required_sell_prices:
                    if price not in open_order_prices:
                        amount_to_sell = amount_per_grid / price
                        engine.create_order(symbol, 'limit', 'sell', amount_to_sell, price)

                print(f"Grid reconciliation cycle for {symbol} complete.")
                stop_event.wait(timeout=30)  # Use the symbol-specific event

            except Exception as e:
                print(f"An error occurred in the grid trading loop for {symbol}: {e}")
                stop_event.wait(timeout=60)

    finally:
        print(f"--- Background grid bot loop for {symbol} stopping... ---")
        if bot_state.grid_bots_engine:
            print(f"Cleaning up all open grid orders for {symbol}...")
            bot_state.grid_bots_engine.cancel_all_orders(symbol)

        # Clean up this bot's state from the global dictionaries
        bot_state.grid_bot_threads.pop(symbol, None)
        bot_state.grid_bot_stop_events.pop(symbol, None)
        bot_state.grid_bot_configs.pop(symbol, None)
        print(f"State for {symbol} has been cleaned up.")

        # If this was the last bot, clean up the global engine and mode
        if not bot_state.grid_bot_threads:
            print("All grid bots have stopped. Shutting down the grid trading engine.")
            bot_state.grid_bots_engine = None
            bot_state.grid_bots_mode = "stopped"


def start_grid_bot(mode: str, symbol: str, config: Dict[str, Any]):
    """
    Starts a new grid bot for a specific symbol in a background thread.
    """
    if symbol in bot_state.grid_bot_threads:
        raise ValueError(f"A grid bot for {symbol} is already running.")

    # If this is the first grid bot, initialize the engine and set the mode
    if not bot_state.grid_bots_engine:
        print(f"This is the first grid bot. Initializing trading engine in '{mode}' mode.")
        bot_state.grid_bots_engine = TradingEngine(mode=mode)
        bot_state.grid_bots_mode = mode
    # If an engine already exists, ensure the mode is consistent
    elif bot_state.grid_bots_mode != mode:
        raise ValueError(f"Cannot start a '{mode}' bot. Grid bots are already running in '{bot_state.grid_bots_mode}' mode.")

    # Create and store the state for the new bot
    stop_event = threading.Event()
    bot_state.grid_bot_stop_events[symbol] = stop_event
    bot_state.grid_bot_configs[symbol] = config

    # Create and start the thread
    thread = threading.Thread(target=_run_single_grid_loop, args=(symbol, config, stop_event))
    bot_state.grid_bot_threads[symbol] = thread
    thread.start()

    print(f"Grid bot for {symbol} has been started in '{mode}' mode.")
    return f"Grid bot for {symbol} started successfully."


def stop_grid_bot(symbol: str):
    """
    Signals a specific grid trading bot to stop.
    """
    if symbol not in bot_state.grid_bot_threads:
        raise ValueError(f"No running grid bot found for symbol {symbol}.")

    print(f"Signaling grid bot for {symbol} to stop...")
    stop_event = bot_state.grid_bot_stop_events.get(symbol)
    if stop_event:
        stop_event.set()
        return f"Stop signal sent to grid bot for {symbol}."
    else:
        # This case should ideally not be reached if threads and events are managed correctly
        raise RuntimeError(f"Could not find a stop event for {symbol} despite a thread existing.")

def stop_all_grid_bots():
    """
    Signals all running grid trading bots to stop.
    """
    if not bot_state.grid_bot_threads:
        return "No grid bots are currently running."

    print("Signaling all grid bots to stop...")
    # Create a copy of the keys to avoid issues with dictionary size changing during iteration
    symbols = list(bot_state.grid_bot_threads.keys())
    for symbol in symbols:
        stop_grid_bot(symbol)

    return f"Stop signal sent to {len(symbols)} grid bots."
