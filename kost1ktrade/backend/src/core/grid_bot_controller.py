import time
import numpy as np
from src.core.bot_state import bot_state
from src.trading.engine import TradingEngine
from src.strategies.grid import GridStrategy

def grid_trading_loop():
    """
    The main loop for the grid trading bot.
    This function is intended to be run in a background task.
    """
    if bot_state.grid_bot_mode == "stopped":
        return

    engine = bot_state.grid_bot_engine
    config = bot_state.grid_bot_config

    symbol = config['symbol']
    amount_per_grid = config['amount_per_grid']
    grid_config = {
        "grid_range_low": config['grid_range_low'],
        "grid_range_high": config['grid_range_high'],
        "num_grids": config['num_grids'],
    }
    strategy = GridStrategy(**grid_config)

    print(f"--- Background grid bot loop started for {symbol} ---")

    try:
        while not bot_state.grid_bot_stop_event.is_set():
            try:
                print(f"({time.ctime()}) --- Grid Reconciliation Cycle for {symbol} ---")

                ideal_levels = strategy.generate_grid_levels()
                ticker = engine.fetch_ticker(symbol)
                if not ticker or 'last' not in ticker:
                    raise ValueError(f"Could not fetch a valid ticker for {symbol}")
                current_price = ticker['last']
                print(f"Current Price: ${current_price:,.2f}")

                open_orders = engine.fetch_open_orders(symbol)
                open_order_prices = {order['price'] for order in open_orders}
                print(f"Found {len(open_orders)} open orders.")

                # Determine which orders should exist
                required_buy_prices = {level for level in ideal_levels if level < current_price}
                required_sell_prices = {level for level in ideal_levels if level > current_price}

                # Cancel extraneous orders
                for order in open_orders:
                    price = order['price']
                    side = order['side']
                    required = required_buy_prices if side == 'buy' else required_sell_prices
                    if price not in required:
                        print(f"Cancelling extraneous {side} order at ${price:,.2f}")
                        engine.cancel_order(order['id'], symbol)

                # Place missing orders
                for price in required_buy_prices:
                    if price not in open_order_prices:
                        engine.create_order(symbol, 'limit', 'buy', amount_per_grid, price)

                for price in required_sell_prices:
                    if price not in open_order_prices:
                        engine.create_order(symbol, 'limit', 'sell', amount_per_grid, price)

                print("Grid reconciliation cycle complete.")
                bot_state.grid_bot_stop_event.wait(timeout=30)

            except Exception as e:
                print(f"An error occurred in the grid trading loop: {e}")
                bot_state.grid_bot_stop_event.wait(timeout=60)
    finally:
        print(f"--- Background grid bot loop stopping... ---")
        if bot_state.grid_bot_engine:
            print("Cleaning up all open grid orders...")
            bot_state.grid_bot_engine.cancel_all_orders(symbol)

        bot_state.grid_bot_mode = "stopped"
        bot_state.grid_bot_engine = None


def start_grid_bot(mode: str):
    """Prepares the state for the grid bot to be started in a background task."""
    if bot_state.grid_bot_mode != "stopped":
        raise ValueError("Grid bot is already running.")

    bot_state.grid_bot_stop_event.clear()

    try:
        bot_state.grid_bot_engine = TradingEngine(mode=mode)
        bot_state.grid_bot_mode = mode
        print(f"Grid bot state prepared for '{mode}' mode.")
    except Exception as e:
        bot_state.grid_bot_mode = "stopped"
        raise e

def stop_grid_bot():
    """Signals the background grid trading loop to stop."""
    if bot_state.grid_bot_mode == "stopped":
        raise ValueError("Grid bot is not running.")

    print("Signaling grid bot to stop...")
    bot_state.grid_bot_stop_event.set()
    return "Stop signal sent. The grid bot will shut down after its current cycle."
